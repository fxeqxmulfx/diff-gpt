//! Read-only loss/gradient oracle on the exact saved-model native FP32 logits.
use crate::*;

// Compute the target derivative from the other-class mass, avoiding p_target-1.
pub(crate) fn reference_softmax(logits: &[f32], gold: usize) -> (f64, Vec<f64>) {
    let max = logits
        .iter()
        .map(|&v| v as f64)
        .fold(f64::NEG_INFINITY, f64::max);
    let mass: Vec<_> = logits.iter().map(|&v| (v as f64 - max).exp()).collect();
    let other: f64 = mass
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != gold)
        .map(|(_, v)| v)
        .sum();
    let total = mass[gold] + other;
    let mut gradient: Vec<_> = mass.iter().map(|v| v / total).collect();
    gradient[gold] = -other / total;
    let loss = if logits[gold] as f64 == max {
        other.ln_1p()
    } else {
        (max - logits[gold] as f64) + total.ln()
    };
    (loss, gradient)
}

pub fn run(cp: &Checkpoint, name: &str) -> Result<Value> {
    let task = data::build(&cp.protocol["case"])?;
    let mut engine = cp.engine()?;
    let mut reports = serde_json::Map::new();
    for split in ["train", "validation"] {
        let examples = &task.splits[split];
        let n = examples.len();
        let vocab = cp.cfg.vocab;
        let mut native_gradient = vec![0.; cp.p.len()];
        let mut reference_gradient = vec![0.; cp.p.len()];
        let mut packed = metrics::Packed::default();
        let mut row_error = Vec::new();
        let mut row_sum = Vec::new();
        let mut zero_target = 0;
        let mut zero_loss = 0;
        let mut exact = 0;
        let mut native_loss = 0.;
        let mut reference_loss = 0.;
        let mut logit_native_energy = 0.;
        let mut logit_ref_energy = 0.;
        let mut logit_error_energy = 0.;
        let mut logit_dot = 0.;
        let mut common_energy = 0.;
        let mut top = Vec::new();
        for start in (0..n).step_by(cp.cfg.max_batch) {
            let end = (start + cp.cfg.max_batch).min(n);
            let batch = end - start;
            let ids: Vec<_> = (start as u32..end as u32).collect();
            packed.fill(examples, &ids);
            engine.forward(&packed.tokens, &packed.labels, batch, 2)?;
            let logits = engine.logits.read()?;
            let native = engine.dlogits.read()?;
            let losses = engine.losses(batch * 2)?;
            let predictions = engine.predictions.read()?;
            let mut reference = native.clone();
            for (local, e) in examples[start..end].iter().enumerate() {
                let row = 2 * local + 1;
                let gold = e.targets[1] as usize;
                let (loss, gradient) =
                    reference_softmax(&logits[row * vocab..(row + 1) * vocab], gold);
                let actual: Vec<_> = native[row * vocab..(row + 1) * vocab]
                    .iter()
                    .map(|&g| g as f64 * batch as f64)
                    .collect();
                let error: Vec<_> = actual.iter().zip(&gradient).map(|(a, b)| a - b).collect();
                let en = stats::dot(&actual, &actual);
                let er = stats::dot(&gradient, &gradient);
                let ee = stats::dot(&error, &error);
                let sum: f64 = actual.iter().sum();
                let relative = (ee / er).sqrt();
                row_error.push(relative);
                row_sum.push(sum);
                logit_native_energy += en;
                logit_ref_energy += er;
                logit_error_energy += ee;
                logit_dot += stats::dot(&actual, &gradient);
                common_energy += sum * sum / vocab as f64;
                zero_target += usize::from(actual[gold] == 0. && gradient[gold] != 0.);
                zero_loss += usize::from(losses[row] == 0. && loss > 0.);
                exact += usize::from(predictions[row] as usize == gold);
                native_loss += losses[row] as f64;
                reference_loss += loss;
                if relative > 0.1 {
                    top.push(json!({"a":e.tokens[0],"b":e.tokens[1],"target":gold,
                        "native_loss":losses[row],"reference_loss":loss,"native_target_gradient":actual[gold],
                        "reference_target_gradient":gradient[gold],"native_gradient_sum":sum,"relative_gradient_l2_error":relative}));
                }
                for (c, &g) in gradient.iter().enumerate() {
                    reference[row * vocab + c] = (g / batch as f64) as f32;
                }
            }
            engine.backward(batch, 2)?;
            let native_g = engine.gradients.read()?;
            engine.dlogits.write(&reference)?;
            engine.backward(batch, 2)?;
            let ref_g = engine.gradients.read()?;
            require(
                native_g.iter().chain(&ref_g).all(|v| v.is_finite()),
                "nonfinite precision gradient",
            )?;
            for i in 0..cp.p.len() {
                native_gradient[i] += native_g[i] as f64 * batch as f64 / n as f64;
                reference_gradient[i] += ref_g[i] as f64 * batch as f64 / n as f64;
            }
        }
        require(
            cp.result["endpoint"][split]["exact"] == exact
                && (cp.result["endpoint"][split]["loss"].as_f64().unwrap()
                    - native_loss / n as f64)
                    .abs()
                    < 1e-7,
            "precision check did not reproduce saved scores",
        )?;
        let ng = projection_gradient(&cp.cfg, &native_gradient);
        let rg = projection_gradient(&cp.cfg, &reference_gradient);
        let gradients: Vec<_> = blocks(&cp.cfg).iter().map(|b| {
            let ids = owner_indices(&cp.cfg,b);
            let a: Vec<_> = ids.iter().map(|&i| ng[i]).collect();
            let r: Vec<_> = ids.iter().map(|&i| rg[i]).collect();
            let d: Vec<_> = a.iter().zip(&r).map(|(a,r)| a-r).collect();
            json!({"block":b.name,"native_gradient_l2":stats::norm(&a),"reference_gradient_l2":stats::norm(&r),
                "native_reference_cosine":stats::cosine(&a,&r),"relative_gradient_error":stats::norm(&d)/stats::norm(&r)})
        }).collect();
        top.sort_by(|a, b| {
            b["relative_gradient_l2_error"]
                .as_f64()
                .unwrap()
                .total_cmp(&a["relative_gradient_l2_error"].as_f64().unwrap())
        });
        reports.insert(split.into(),json!({"examples":n,"exact":exact,"native_mean_loss":native_loss/n as f64,
            "reference_mean_loss":reference_loss/n as f64,"zero_native_loss_but_positive_reference":zero_loss,
            "zero_native_target_gradient_but_nonzero_reference":zero_target,"row_gradient_relative_error":stats::summary(&row_error),
            "native_row_gradient_sum":stats::summary(&row_sum),"logit_gradient_relative_error":(logit_error_energy/logit_ref_energy).sqrt(),
            "logit_gradient_cosine":logit_dot/(logit_native_energy*logit_ref_energy).sqrt(),
            "native_logit_gradient_constant_component_fraction":common_energy/logit_native_energy,
            "parameter_gradients":gradients,"largest_row_errors":top.into_iter().take(10).collect::<Vec<_>>()}));
    }
    require(
        engine.parameters.read()? == cp.p
            && engine.first_moment.read()? == cp.m
            && engine.second_moment.read()? == cp.v,
        "precision probe changed parameters or optimizer",
    )?;
    Ok(json!({"snapshot":name,"splits":reports,
        "reference":"FP64 softmax of unchanged native FP32 logits; target derivative = -sum(non-target probabilities); cast to FP32 for original backward",
        "scope":"diagnostic gradients only; no optimizer updates or saved-model changes"}))
}

#[test]
fn softmax_reference_keeps_high_confidence_target_gradient() {
    let (loss, g) = reference_softmax(&[1000., 980., 979.], 0);
    let (shifted, sg) = reference_softmax(&[0., -20., -21.], 0);
    assert!(loss > 0. && loss < 1e-8);
    assert!(g[0] < 0.);
    assert!(g.iter().sum::<f64>().abs() < 1e-23);
    assert_eq!(loss, shifted);
    assert_eq!(g, sg);
    let rounded = 1f32 / (1f32 + ((-20f32).exp() + (-21f32).exp()));
    assert_eq!(rounded - 1., 0.);
}
