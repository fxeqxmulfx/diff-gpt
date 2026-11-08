//! Native, cycle-disjoint Fibonacci97 windows for the positional study.
//! Codec semantics come from the separately validated version4 native codec.
#![forbid(unsafe_code)]
use diffgpt_mash_native::{Difference, Domain, Matrix, Tokenizer, decode, encode};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

#[path = "../../../complex_geometry_suite/native/src/random.rs"]
#[allow(dead_code)]
mod random;
pub type Result<T> = std::result::Result<T, String>;
pub type State = (i64, i64);
pub const VOCAB: usize = 512;

#[derive(Clone, Debug)]
pub struct Example {
    pub states: Vec<State>,
    pub tokens: Vec<u32>,
    pub targets: Vec<u32>,
    pub prompt: Vec<u32>,
    pub future_tokens: Vec<u32>,
    pub future_values: Vec<i64>,
    pub window_base: Vec<f64>,
    pub forecast_base: Vec<f64>,
    pub scale: Vec<f64>,
    pub zero: Vec<i64>,
}

impl Example {
    pub fn decoded_hit(&self, feature: usize, token: u32, reanchored: bool) -> bool {
        let base = if reanchored {
            self.forecast_base[feature]
        } else {
            self.window_base[feature]
        };
        (base + (i64::from(token) - self.zero[feature]) as f64 / self.scale[feature])
            .round_ties_even()
            == self.future_values[feature] as f64
    }
}

pub struct Dataset {
    pub train: Vec<Example>,
    pub validation: Vec<Example>,
    pub training_cycle_ids: Vec<usize>,
    pub validation_cycle_ids: Vec<usize>,
    pub features: usize,
    pub context: usize,
    pub tokenizer: Tokenizer,
    pub seed: u32,
}

pub fn cycles() -> Vec<Vec<State>> {
    let mut remaining: BTreeSet<State> = (0..97)
        .flat_map(|a| (0..97).map(move |b| (a, b)))
        .filter(|s| *s != (0, 0))
        .collect();
    let mut result = Vec::new();
    while let Some(&first) = remaining.first() {
        let mut cycle = Vec::new();
        let mut state = first;
        while remaining.remove(&state) {
            cycle.push(state);
            state = (state.1, (state.0 + state.1) % 97);
        }
        assert_eq!(state, first);
        result.push(cycle);
    }
    result
}

fn raw_series(state: State, rows: usize) -> Vec<f64> {
    let mut x = vec![state.0 as f64, state.1 as f64];
    while x.len() < rows {
        let n = x.len();
        x.push(((x[n - 2] as i64 + x[n - 1] as i64) % 97) as f64);
    }
    x
}

fn examples(states: &[Vec<State>], context: usize, tokenizer: Tokenizer) -> Result<Vec<Example>> {
    let features = states[0].len();
    let columns = states.len() * features;
    let series: Vec<_> = states
        .iter()
        .flatten()
        .map(|&s| raw_series(s, context + 2))
        .collect();
    let raw: Vec<f64> = (0..context + 2)
        .flat_map(|t| series.iter().map(move |x| x[t]))
        .collect();
    let input = Matrix::new(columns, raw.clone())?;
    let operators = vec![Difference::order(1)?; columns];
    let encoded = encode(
        &input,
        VOCAB as i64,
        &vec![Domain::symmetric(96.); columns],
        &operators,
        tokenizer,
    )?;
    if encoded.clipped_tokens != 0 {
        return Err("declared Fibonacci grid clips tokens".into());
    }
    let mut zero_future = encoded.tokens.values().to_vec();
    zero_future[context * columns..].copy_from_slice(&encoded.grid.zero);
    let zero_future = Matrix::new(columns, zero_future)?;
    let window_zero = decode(
        &encoded.prefix,
        &encoded.grid,
        &zero_future,
        VOCAB as i64,
        &operators,
        tokenizer,
    )?;
    let forecast_prefix = Matrix::new(
        columns,
        raw[context * columns..(context + 1) * columns].to_vec(),
    )?;
    let forecast_zero = decode(
        &forecast_prefix,
        &encoded.grid,
        &Matrix::new(columns, encoded.grid.zero.clone())?,
        VOCAB as i64,
        &operators,
        tokenizer,
    )?;
    let truth = decode(
        &encoded.prefix,
        &encoded.grid,
        &encoded.tokens,
        VOCAB as i64,
        &operators,
        tokenizer,
    )?;
    if truth
        .values()
        .iter()
        .zip(&raw)
        .any(|(a, b)| a.round_ties_even() != *b)
    {
        return Err("native round-trip changed a residue".into());
    }
    let mut result = Vec::new();
    for (sample, state) in states.iter().enumerate() {
        let at = sample * features;
        let mut complete = Vec::new();
        for t in 0..context + 1 {
            for f in 0..features {
                complete.push(encoded.tokens.values()[t * columns + at + f] as u32);
            }
        }
        let length = complete.len() - 1;
        let mut targets = vec![u32::MAX; length];
        targets[context * features - 1..].copy_from_slice(&complete[context * features..]);
        let value = Example {
            states: state.clone(),
            tokens: complete[..length].to_vec(),
            targets,
            prompt: complete[..context * features].to_vec(),
            future_tokens: complete[context * features..].to_vec(),
            future_values: raw
                [(context + 1) * columns + at..(context + 1) * columns + at + features]
                .iter()
                .map(|v| *v as i64)
                .collect(),
            window_base: window_zero.values()
                [(context + 1) * columns + at..(context + 1) * columns + at + features]
                .to_vec(),
            forecast_base: forecast_zero.values()[columns + at..columns + at + features].to_vec(),
            scale: encoded.grid.scale[at..at + features].to_vec(),
            zero: encoded.grid.zero[at..at + features].to_vec(),
        };
        if value
            .future_tokens
            .iter()
            .enumerate()
            .any(|(f, &t)| !value.decoded_hit(f, t, false))
        {
            return Err("decoded target token fails window-frame rounding".into());
        }
        result.push(value);
    }
    Ok(result)
}

pub fn build(seed: u32, features: usize, context: usize, tokenizer: Tokenizer) -> Result<Dataset> {
    if ![11, 17, 23].contains(&seed) || ![1, 2].contains(&features) || ![2, 4, 8].contains(&context)
    {
        return Err("undeclared DiffGPT dataset geometry/seed".into());
    }
    let all = cycles();
    if all.len() != 48 || all.iter().any(|c| c.len() != 196) {
        return Err("unexpected Fibonacci cycle geometry".into());
    }
    let mut ids: Vec<u32> = (0..48).collect();
    random::Random::new(seed).shuffle(&mut ids);
    let make = |selected: &[u32]| -> Result<Vec<Example>> {
        let first: Vec<State> = selected
            .iter()
            .flat_map(|&id| all[id as usize].iter().copied())
            .collect();
        let mut permutation: Vec<u32> = (0..first.len() as u32).collect();
        random::Random::new(seed + 20000).shuffle(&mut permutation);
        let paired: Vec<Vec<State>> = first
            .iter()
            .enumerate()
            .map(|(i, &s)| {
                if features == 1 {
                    vec![s]
                } else {
                    vec![s, first[permutation[i] as usize]]
                }
            })
            .collect();
        examples(&paired, context, tokenizer)
    };
    Ok(Dataset {
        train: make(&ids[..24])?,
        validation: make(&ids[24..])?,
        training_cycle_ids: ids[..24].iter().map(|x| *x as usize).collect(),
        validation_cycle_ids: ids[24..].iter().map(|x| *x as usize).collect(),
        features,
        context,
        tokenizer,
        seed,
    })
}

/// Hash exact row-major tokens, targets, decode constants, labels and state IDs.
pub fn fingerprint(examples: &[Example]) -> String {
    let mut bytes = Vec::new();
    bytes.extend((examples.len() as u64).to_le_bytes());
    for e in examples {
        for v in [&e.tokens, &e.targets, &e.prompt, &e.future_tokens] {
            bytes.extend((v.len() as u64).to_le_bytes());
            bytes.extend(v.iter().flat_map(|x| x.to_le_bytes()));
        }
        for &(a, b) in &e.states {
            bytes.extend(a.to_le_bytes());
            bytes.extend(b.to_le_bytes());
        }
        bytes.extend(e.future_values.iter().flat_map(|x| x.to_le_bytes()));
        for v in [&e.window_base, &e.forecast_base, &e.scale] {
            bytes.extend(v.iter().flat_map(|x| x.to_le_bytes()));
        }
        bytes.extend(e.zero.iter().flat_map(|x| x.to_le_bytes()));
    }
    Sha256::digest(bytes)
        .iter()
        .map(|x| format!("{x:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cycles_cover_every_nonzero_state_once_and_are_closed() {
        let c = cycles();
        assert_eq!(c.len(), 48);
        let mut all = BTreeSet::new();
        for cycle in c {
            assert_eq!(cycle.len(), 196);
            for i in 0..196 {
                assert!(all.insert(cycle[i]));
                let (a, b) = cycle[i];
                assert_eq!(cycle[(i + 1) % 196], (b, (a + b) % 97));
            }
        }
        assert_eq!(all.len(), 9408);
        assert!(!all.contains(&(0, 0)));
    }
    #[test]
    fn every_encoder_and_feature_count_preserves_cycle_holdout_and_decoded_targets() -> Result<()> {
        for tokenizer in [Tokenizer::Mash, Tokenizer::Lgt, Tokenizer::MashHalf] {
            for features in [1, 2] {
                let d = build(11, features, 2, tokenizer)?;
                assert_eq!(d.train.len(), 4704);
                assert_eq!(d.validation.len(), 4704);
                let tr: BTreeSet<_> = d.train.iter().map(|e| e.states[0]).collect();
                let va: BTreeSet<_> = d.validation.iter().map(|e| e.states[0]).collect();
                assert!(tr.is_disjoint(&va));
                for (split, allowed) in [(&d.train, &tr), (&d.validation, &va)] {
                    for e in split {
                        assert!(e.states.iter().all(|s| allowed.contains(s)));
                        assert_eq!(e.tokens.len(), 3 * features - 1);
                        assert_eq!(e.prompt.len(), 2 * features);
                        assert_eq!(
                            e.targets.iter().filter(|&&t| t != u32::MAX).count(),
                            features
                        );
                        for f in 0..features {
                            assert!(e.decoded_hit(f, e.future_tokens[f], false));
                        }
                    }
                }
                assert_ne!(fingerprint(&d.train), fingerprint(&d.validation));
            }
        }
        Ok(())
    }
    #[test]
    fn argmax_decoding_formula_matches_native_decode_in_both_anchor_frames() -> Result<()> {
        let cases: Vec<Vec<State>> = (1..=64)
            .map(|a| vec![(a, (7 * a + 3) % 97), ((11 * a + 2) % 97, a)])
            .collect();
        for tokenizer in [Tokenizer::Mash, Tokenizer::Lgt, Tokenizer::MashHalf] {
            for context in [2, 4, 8] {
                let prepared = examples(&cases, context, tokenizer)?;
                for e in prepared {
                    let series: Vec<_> = e
                        .states
                        .iter()
                        .map(|&s| raw_series(s, context + 2))
                        .collect();
                    let raw: Vec<f64> = (0..context + 2)
                        .flat_map(|r| series.iter().map(move |x| x[r]))
                        .collect();
                    let ops = vec![Difference::order(1)?; 2];
                    let encoded = encode(
                        &Matrix::new(2, raw.clone())?,
                        512,
                        &[Domain::symmetric(96.); 2],
                        &ops,
                        tokenizer,
                    )?;
                    for token in [0, 1, 127, 255, 256, 257, 384, 510, 511] {
                        let mut tokens = encoded.tokens.values().to_vec();
                        tokens[2 * context..].fill(token);
                        let decoded = decode(
                            &encoded.prefix,
                            &encoded.grid,
                            &Matrix::new(2, tokens)?,
                            512,
                            &ops,
                            tokenizer,
                        )?;
                        let fresh = decode(
                            &Matrix::new(2, raw[2 * context..2 * context + 2].to_vec())?,
                            &encoded.grid,
                            &Matrix::new(2, vec![token; 2])?,
                            512,
                            &ops,
                            tokenizer,
                        )?;
                        for f in 0..2 {
                            assert_eq!(
                                e.decoded_hit(f, token as u32, false),
                                decoded.values()[2 * (context + 1) + f].round_ties_even()
                                    == e.future_values[f] as f64
                            );
                            assert_eq!(
                                e.decoded_hit(f, token as u32, true),
                                fresh.values()[2 + f].round_ties_even()
                                    == e.future_values[f] as f64
                            );
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
