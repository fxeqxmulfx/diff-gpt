//! Complete-split scoring. Generated answers read their own mistakes until EOS.
use crate::{
    data::Example,
    engine::{Engine, Result},
};
use serde_json::{Value, json};

#[derive(Default)]
pub struct Packed {
    pub tokens: Vec<u32>,
    pub labels: Vec<u32>,
    pub length: usize,
}
impl Packed {
    pub fn fill(&mut self, examples: &[Example], indices: &[u32]) {
        self.length = indices
            .iter()
            .map(|&i| examples[i as usize].tokens.len())
            .max()
            .unwrap();
        self.tokens.clear();
        self.tokens.resize(indices.len() * self.length, 0);
        self.labels.clear();
        self.labels.resize(self.tokens.len(), u32::MAX);
        for (row, &index) in indices.iter().enumerate() {
            let e = &examples[index as usize];
            let at = row * self.length;
            for (i, &v) in e.tokens.iter().enumerate() {
                self.tokens[at + i] = v as u32;
            }
            for (i, &v) in e.targets.iter().enumerate() {
                self.labels[at + i] = v as u32;
            }
        }
    }
}
#[derive(Default)]
struct Counts {
    loss: f64,
    example_loss: f64,
    targets: usize,
    correct: usize,
    exact: usize,
    transitions: usize,
    transition_correct: usize,
    examples: usize,
    classes: Vec<usize>,
    right: Vec<usize>,
}
impl Counts {
    fn new(vocab: usize) -> Self {
        Self {
            classes: vec![0; vocab],
            right: vec![0; vocab],
            ..Self::default()
        }
    }
    fn report(&self) -> Value {
        let classes:serde_json::Map<String,Value>=self.classes.iter().enumerate().filter(|&(_,n)|*n!=0).map(|(label,&count)|
            (label.to_string(),json!({"name":label.to_string(),"count":count,"correct":self.right[label],"accuracy":self.right[label]as f64/count as f64}))).collect();
        let balanced = self
            .classes
            .iter()
            .zip(&self.right)
            .filter(|(n, _)| **n > 0)
            .map(|(&n, &c)| c as f64 / n as f64)
            .sum::<f64>()
            / classes.len() as f64;
        json!({"loss":self.loss/self.targets as f64,"example_loss":self.example_loss/self.examples as f64,
            "sequence_accuracy":self.exact as f64/self.examples as f64,"token_accuracy":self.correct as f64/self.targets as f64,
            "balanced_accuracy":balanced,"transition_accuracy":if self.transitions>0{Some(self.transition_correct as f64/self.transitions as f64)}else{None},
            "correct":self.correct,"exact":self.exact,"targets":self.targets,"target_count":self.targets,
            "examples":self.examples,"example_count":self.examples,"transition_count":self.transitions,"classes":classes})
    }
}
fn numeric_error(e: &Example, predicted: &[u32]) -> (f64, usize) {
    let mut error = 0.;
    let mut coverage = 0;
    for i in 0..32 {
        let truth = if e.task == "linear-transport" {
            (e.prompt[1 + i] - 52) as f64 / 16.
        } else {
            (e.prompt[1 + i] - 52) as f64 / 4.
                + (e.prompt[33 + i] - 52) as f64 / 16. / (1. + f32::EPSILON as f64).sqrt()
        };
        if let Some(&token) = predicted.get(i).filter(|&&t| (36..69).contains(&t)) {
            error += ((token as f64 - 52.) / 4. - truth).powi(2);
            coverage += 1;
        } else {
            error += 64.;
        }
    }
    (error, coverage)
}
fn field_prediction(logits: &[f32], base: i64) -> usize {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut mass = [0f64; 97];
    for (token, &value) in logits.iter().enumerate() {
        mass[(token as i64 - logits.len() as i64 / 2 + base).rem_euclid(97) as usize] +=
            ((value - max) as f64).exp();
    }
    (0..97)
        .max_by(|&a, &b| mass[a].total_cmp(&mass[b]).then(b.cmp(&a)))
        .unwrap()
}
fn rollout(
    examples: &[Example],
    mut predict: impl FnMut(&[&[u32]]) -> Result<Vec<u32>>,
) -> Result<Vec<Vec<u32>>> {
    let mut contexts: Vec<Vec<u32>> = examples
        .iter()
        .map(|e| e.prompt.iter().map(|&v| v as u32).collect())
        .collect();
    let mut written = vec![Vec::new(); examples.len()];
    let mut active: Vec<usize> = (0..examples.len()).collect();
    while !active.is_empty() {
        let current: Vec<&[u32]> = active.iter().map(|&i| contexts[i].as_slice()).collect();
        let predictions = predict(&current)?;
        if predictions.len() != active.len() {
            return Err("generation batch length mismatch".into());
        }
        let mut remaining = Vec::new();
        for (&index, &token) in active.iter().zip(&predictions) {
            written[index].push(token);
            if token != 17 && written[index].len() < examples[index].limit {
                contexts[index].push(token);
                remaining.push(index);
            }
        }
        active = remaining;
    }
    Ok(written)
}
pub fn evaluate(engine: &mut Engine, examples: &[Example], generate: bool) -> Result<Value> {
    if examples.is_empty() {
        return Err("cannot evaluate empty split".into());
    }
    let vocab = engine.config.vocab;
    let batch = engine.config.max_batch;
    let generative = generate && !examples[0].prompt.is_empty();
    let numeric = matches!(
        examples[0].task.as_str(),
        "linear-transport" | "branch-amplitudes"
    );
    let field = examples[0].field.is_some();
    let mut teacher = Counts::new(vocab);
    let mut generated = Counts::new(vocab);
    let (mut generated_tokens, mut ended, mut final_correct) = (0, 0, 0);
    let (mut mse, mut covered, mut native_correct) = (0., 0, 0);
    let mut packed = Packed::default();
    for start in (0..examples.len()).step_by(batch) {
        let stop = (start + batch).min(examples.len());
        let indices: Vec<u32> = (start as u32..stop as u32).collect();
        packed.fill(examples, &indices);
        let length = packed.length;
        engine.forward(&packed.tokens, &packed.labels, indices.len(), length)?;
        let predictions = engine.predictions.read()?;
        let losses = engine.losses(indices.len() * length)?;
        let logits = if field {
            Some(engine.logits.read()?)
        } else {
            None
        };
        for local in 0..indices.len() {
            let e = &examples[start + local];
            let mut exact = true;
            let mut row_loss = 0.;
            let mut targets = 0;
            let mut written = Vec::new();
            for (pos, &target) in e.targets.iter().enumerate() {
                if target < 0 {
                    continue;
                }
                let label = target as usize;
                let i = local * length + pos;
                let hit = predictions[i] as usize == label;
                exact &= hit;
                row_loss += losses[i] as f64;
                targets += 1;
                written.push(predictions[i]);
                teacher.classes[label] += 1;
                teacher.right[label] += usize::from(hit);
                teacher.correct += usize::from(hit);
                if e.changes[pos] {
                    teacher.transitions += 1;
                    teacher.transition_correct += usize::from(hit);
                }
            }
            teacher.loss += row_loss;
            teacher.example_loss += row_loss / targets as f64;
            teacher.targets += targets;
            teacher.examples += 1;
            if let Some((base, future)) = e.field {
                native_correct += usize::from(exact);
                let z = &logits.as_ref().unwrap()
                    [(local * length + length - 1) * vocab..(local * length + length) * vocab];
                let best = field_prediction(z, base);
                exact = best as i64 == future;
            }
            teacher.exact += usize::from(exact);
            if numeric && !generative {
                let (error, valid) = numeric_error(e, &written);
                mse += error;
                covered += valid;
            }
        }
        if !generative {
            continue;
        }
        let written = rollout(&examples[start..stop], |contexts| {
            let length = contexts.iter().map(|c| c.len()).max().unwrap();
            packed.tokens.clear();
            packed.tokens.resize(contexts.len() * length, 0);
            packed.labels.clear();
            packed.labels.resize(packed.tokens.len(), u32::MAX);
            for (row, context) in contexts.iter().enumerate() {
                packed.tokens[row * length..row * length + context.len()].copy_from_slice(context);
            }
            engine.forward(&packed.tokens, &packed.labels, contexts.len(), length)?;
            let predictions = engine.predictions.read()?;
            Ok(contexts
                .iter()
                .enumerate()
                .map(|(row, c)| predictions[row * length + c.len() - 1])
                .collect())
        })?;
        for (i, tokens) in written.iter().enumerate() {
            let e = &examples[start + i];
            let exact = tokens.len() == e.answer.len()
                && tokens.iter().zip(&e.answer).all(|(&a, &b)| a as i64 == b);
            let eos = tokens.last() == Some(&17);
            ended += usize::from(eos);
            generated_tokens += tokens.len();
            let final_hit = if matches!(e.task.as_str(), "parity" | "boolean_and") {
                eos && tokens.len() >= 2
                    && tokens[tokens.len() - 2] as i64 == e.answer[e.answer.len() - 2]
            } else {
                exact
            };
            final_correct += usize::from(final_hit);
            generated.exact += usize::from(exact);
            generated.examples += 1;
            let mut previous = None;
            for (pos, &label) in e.answer.iter().enumerate() {
                let transition = previous != Some(label);
                previous = Some(label);
                let hit = tokens.get(pos).copied().map(i64::from) == Some(label);
                generated.targets += 1;
                generated.correct += usize::from(hit);
                generated.classes[label as usize] += 1;
                generated.right[label as usize] += usize::from(hit);
                if transition {
                    generated.transitions += 1;
                    generated.transition_correct += usize::from(hit);
                }
            }
            if numeric {
                let (error, valid) = numeric_error(e, tokens);
                mse += error;
                covered += valid;
            }
        }
    }
    if !teacher.loss.is_finite() {
        return Err("nonfinite evaluation loss".into());
    }
    let mut result = if generative {
        generated.loss = teacher.loss;
        generated.example_loss = teacher.example_loss;
        let mut result = generated.report();
        result["teacher_forced"] = teacher.report();
        result["final_answer_accuracy"] = (final_correct as f64 / examples.len() as f64).into();
        result["eos_rate"] = (ended as f64 / examples.len() as f64).into();
        result["generated_tokens"] = generated_tokens.into();
        result["extra_tokens"] = 0.into();
        result
    } else {
        teacher.report()
    };
    if field {
        result["decoded_accuracy"] = result["sequence_accuracy"].clone();
        result["token_accuracy"] = result["sequence_accuracy"].clone();
        result["native_token_accuracy"] = (native_correct as f64 / examples.len() as f64).into();
        result["native_correct"] = native_correct.into();
        result["field_correct"] = teacher.exact.into();
        result["correct"] = teacher.exact.into();
    }
    if numeric {
        result["mse"] = (mse / (32 * examples.len()) as f64).into();
        result["numeric_output_coverage"] = (covered as f64 / (32 * examples.len()) as f64).into();
    }
    Ok(result)
}

#[test]
fn generation_reads_its_own_errors_and_respects_eos_and_limits() -> Result<()> {
    let examples = [
        Example::generation("test", vec![1, 18], vec![4, 17]),
        Example::generation("test", vec![2, 18], vec![5, 17]),
        Example::generation("test", vec![3, 18], vec![6, 17]),
    ];
    let mut calls = 0;
    let generated = rollout(&examples, |contexts| {
        calls += 1;
        if calls == 1 {
            assert_eq!(contexts.len(), 3);
            Ok(vec![3, 17, 9])
        } else {
            assert_eq!(contexts, vec![&[1, 18, 3][..], &[3, 18, 9][..]]);
            Ok(vec![17, 10])
        }
    })?;
    assert_eq!(generated, vec![vec![3, 17], vec![17], vec![9, 10]]);
    assert_eq!(calls, 2);
    Ok(())
}
#[test]
fn field_readout_pools_aliases_and_numeric_eos_is_penalized() {
    let mut logits = vec![-1000f32; 256];
    logits[0] = 0.5f32.ln();
    logits[97] = 0.4f32.ln();
    logits[10] = 0.8f32.ln();
    assert_eq!(
        field_prediction(&logits, 8),
        (-128i64 + 8).rem_euclid(97) as usize
    );
    let mut prompt = vec![1];
    prompt.extend([56; 32]);
    prompt.push(18);
    let mut answer = vec![53; 32];
    answer.push(17);
    let e = Example::generation("linear-transport", prompt, answer);
    assert_eq!(numeric_error(&e, &[17]), (64. * 32., 0));
    assert_eq!(numeric_error(&e, &[53; 32]), (0., 32));
}
