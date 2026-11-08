//! Independent primitive references and integration checks before rotary training.
use crate::{
    Config, Engine, Result, attention, hash, layout,
    optimizer::Optimizer,
    oracle::{self, Jet},
    random::Random,
    rotary, runner,
};
use kernite_cuda::{Cuda, Grid};
use serde_json::{Value, json};
use std::{fs, path::Path};

fn bits_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}
fn bytes(a: &[f32]) -> Vec<u8> {
    a.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn grid(n: usize) -> Grid {
    Grid::new(n.div_ceil(256) as u32, 256)
}
fn cfg(width: usize, heads: usize, layers: usize, arm: &str) -> Config {
    Config {
        width,
        heads,
        layers,
        vocab: 11,
        max_batch: 2,
        max_length: 4,
        normalized: true,
        ssmax: true,
        gain: heads as f32 / width as f32,
        lr: 0.001,
        decay: 0.1,
        warmup: 400,
        phase_optimizer: false,
        architecture: arm.into(),
    }
}

fn tables(rows: &mut Vec<Value>) -> Result<()> {
    let cuda = Cuda::new()?;
    let k = rotary::kernels::Kernels::load(&cuda)?;
    for d in [16, 32, 128] {
        let bands = d / 2;
        for length in [1, 2, 17, 67, 128, 257, 1025] {
            for log in [false, true] {
                let (base, rope) = rotary::base_tables(length, d, log);
                let db = cuda.upload(&base)?;
                let dr = cuda.upload(&rope)?;
                for magnitude in [0., 1e-9f32, 1e-4, 0.25, 1., 4.] {
                    let alpha: Vec<f32> = (0..bands)
                        .map(|i| magnitude * ((i % 7) as f32 - 3.) / 3.)
                        .collect();
                    let da = cuda.upload(&alpha)?;
                    let out = cuda.upload(&vec![12345.; rope.len() + 4])?;
                    let theta = cuda.upload(&vec![12345.; base.len() + 4])?;
                    // SAFETY: all table buffers cover length*bands; tail canaries are extra.
                    unsafe {
                        k.table(
                            grid(base.len()),
                            out.ptr(),
                            theta.ptr(),
                            db.ptr(),
                            dr.ptr(),
                            da.ptr(),
                            bands as u32,
                            base.len() as u32,
                        )?;
                    }
                    let actual = out.read()?;
                    let angles = theta.read()?;
                    let (mut error, mut normalized_error, mut norm_error, mut theta_error) =
                        (0f64, 0f64, 0f64, 0f64);
                    let mut passed = true;
                    for i in 0..base.len() {
                        let expected = base[i] as f64 * (alpha[i % bands] as f64).exp();
                        let (s, c) = expected.sin_cos();
                        let err = (actual[i] as f64 - c)
                            .abs()
                            .max((actual[base.len() + i] as f64 - s).abs());
                        // Fixed FP32 phase error budget declared before executing the screen.
                        let tol = 3e-6 + 5e-7 * expected.abs();
                        error = error.max(err);
                        normalized_error = normalized_error.max(err / tol);
                        let nerr = (f64::from(actual[i]).powi(2)
                            + f64::from(actual[base.len() + i]).powi(2)
                            - 1.)
                            .abs();
                        norm_error = norm_error.max(nerr);
                        let aerr = (angles[i] as f64 - expected).abs();
                        theta_error = theta_error.max(aerr / (1e-12 + 3e-7 * expected.abs()));
                        passed &= err.is_finite()
                            && err <= tol
                            && nerr < 3e-6
                            && aerr <= 1e-12 + 3e-7 * expected.abs();
                    }
                    let zero_equal = magnitude != 0. || bits_equal(&actual[..rope.len()], &rope);
                    let canaries = actual[rope.len()..]
                        .iter()
                        .chain(&angles[base.len()..])
                        .all(|x| *x == 12345.);
                    let position0 = (0..bands).all(|i| {
                        actual[i] == 1. && actual[base.len() + i] == 0. && angles[i] == 0.
                    });
                    rows.push(json!({"kind":"table","d":d,"length":length,"log":log,"alpha_magnitude":magnitude,"passed":passed&&zero_equal&&canaries&&position0,"zero_alpha_bit_equal":zero_equal,"canaries":canaries,"position_zero_exact":position0,"max_absolute_error":error,"max_error_over_bound":normalized_error,"max_norm_error":norm_error,"theta_error_over_bound":theta_error}));
                }
            }
        }
    }
    Ok(())
}

fn adjoints(rows: &mut Vec<Value>) -> Result<()> {
    let cuda = Cuda::new()?;
    let k = rotary::kernels::Kernels::load(&cuda)?;
    let rotate = attention::kernels::Kernels::load(&cuda)?;
    for (d, heads) in [(16, 4), (32, 4), (128, 1)] {
        let bands = d / 2;
        let width = d * heads;
        for (batch, length) in [(1, 1), (2, 3), (3, 17), (1, 65)] {
            let r = batch * length;
            let count = r * 3 * width;
            for log in [false, true] {
                let (base, rope) = rotary::base_tables(length, d, log);
                let db = cuda.upload(&base)?;
                let dr = cuda.upload(&rope)?;
                let alpha: Vec<f32> = (0..bands).map(|i| ((i % 5) as f32 - 2.) * 0.125).collect();
                let da = cuda.upload(&alpha)?;
                let table = cuda.alloc(rope.len())?;
                let theta = cuda.alloc(base.len())?;
                unsafe {
                    k.table(
                        grid(base.len()),
                        table.ptr(),
                        theta.ptr(),
                        db.ptr(),
                        dr.ptr(),
                        da.ptr(),
                        bands as u32,
                        base.len() as u32,
                    )?;
                }
                let mut rng = Random::new(17);
                let x: Vec<f32> = (0..count).map(|_| rng.gauss(0., 1.) as f32).collect();
                let dx = cuda.upload(&x)?;
                let y = cuda.upload(&vec![12345.; count + 4])?;
                unsafe {
                    rotate.rotate::<false>(
                        grid(count),
                        y.ptr(),
                        dx.ptr(),
                        table.ptr(),
                        d as u32,
                        heads as u32,
                        length as u32,
                        count as u32,
                    )?;
                }
                for part in ["query", "key", "value", "both"] {
                    let dy: Vec<f32> = (0..count)
                        .map(|i| {
                            let p = (i / width) % 3;
                            let active = match part {
                                "query" => p == 0,
                                "key" => p == 1,
                                "value" => p == 2,
                                _ => p < 2,
                            };
                            if active { rng.gauss(0., 1.) as f32 } else { 0. }
                        })
                        .collect();
                    let ddy = cuda.upload(&dy)?;
                    let dinput = cuda.upload(&vec![12345.; count + 4])?;
                    let gradient = cuda.upload(&vec![12345.; bands + 4])?;
                    unsafe {
                        rotate.rotate::<true>(
                            grid(count),
                            dinput.ptr(),
                            ddy.ptr(),
                            table.ptr(),
                            d as u32,
                            heads as u32,
                            length as u32,
                            count as u32,
                        )?;
                        k.frequency_backward(
                            Grid::new(bands as u32, 32),
                            gradient.ptr(),
                            y.ptr(),
                            ddy.ptr(),
                            theta.ptr(),
                            bands as u32,
                            heads as u32,
                            length as u32,
                            r as u32,
                            1,
                        )?;
                    }
                    let g = gradient.read()?;
                    let input = dinput.read()?;
                    let rotated = y.read()?;
                    let mut expected = vec![0f64; bands];
                    let mut sum_abs = vec![0f64; bands];
                    let mut input_error = 0f64;
                    let mut input_pass = true;
                    for row in 0..r {
                        for p in 0..3 {
                            for head in 0..heads {
                                for band in 0..bands {
                                    let at = row * 3 * width + p * width + head * d + band;
                                    let im = at + bands;
                                    if p == 2 {
                                        input_pass &= input[at] == dy[at]
                                            && input[im] == dy[im]
                                            && rotated[at] == x[at]
                                            && rotated[im] == x[im];
                                        continue;
                                    }
                                    let angle = base[(row % length) * bands + band] as f64
                                        * (alpha[band] as f64).exp();
                                    let (s, c) = angle.sin_cos();
                                    let (yr, yi) = (
                                        x[at] as f64 * c - x[im] as f64 * s,
                                        x[at] as f64 * s + x[im] as f64 * c,
                                    );
                                    let term = angle * (yr * dy[im] as f64 - yi * dy[at] as f64);
                                    expected[band] += term;
                                    sum_abs[band] += term.abs();
                                    let (er, ei) = (
                                        dy[at] as f64 * c + dy[im] as f64 * s,
                                        dy[im] as f64 * c - dy[at] as f64 * s,
                                    );
                                    let err = (input[at] as f64 - er)
                                        .abs()
                                        .max((input[im] as f64 - ei).abs());
                                    let tol = (3e-6 + 5e-7 * angle.abs())
                                        * (1. + (dy[at] as f64).abs() + (dy[im] as f64).abs());
                                    input_error = input_error.max(err / tol);
                                    input_pass &= err.is_finite() && err <= tol;
                                }
                            }
                        }
                    }
                    let mut max_ratio = 0f64;
                    let mut gradient_pass = true;
                    for band in 0..bands {
                        let tol = 2e-5 * sum_abs[band] + 1e-7;
                        let error = (g[band] as f64 - expected[band]).abs();
                        max_ratio = max_ratio.max(error / tol);
                        gradient_pass &= error.is_finite() && error <= tol;
                    }
                    unsafe {
                        k.frequency_backward(
                            Grid::new(bands as u32, 32),
                            gradient.ptr(),
                            y.ptr(),
                            ddy.ptr(),
                            theta.ptr(),
                            bands as u32,
                            heads as u32,
                            length as u32,
                            r as u32,
                            0,
                        )?;
                    }
                    let doubled = gradient.read()?;
                    let accumulated =
                        (0..bands).all(|i| doubled[i].to_bits() == (g[i] + g[i]).to_bits());
                    unsafe {
                        k.frequency_backward(
                            Grid::new(bands as u32, 32),
                            gradient.ptr(),
                            y.ptr(),
                            ddy.ptr(),
                            theta.ptr(),
                            bands as u32,
                            heads as u32,
                            length as u32,
                            r as u32,
                            1,
                        )?;
                    }
                    let reset = gradient.read()?;
                    let reset_equal = bits_equal(&g, &reset);
                    let canaries = g[bands..]
                        .iter()
                        .chain(&reset[bands..])
                        .chain(&input[count..])
                        .chain(&rotated[count..])
                        .all(|x| *x == 12345.);
                    let zero =
                        !(part == "value" || length == 1) || g[..bands].iter().all(|x| *x == 0.);
                    rows.push(json!({"kind":"adjoint","d":d,"heads":heads,"batch":batch,"length":length,"log":log,"path":part,"passed":gradient_pass&&input_pass&&accumulated&&reset_equal&&canaries&&zero,"gradient_error_over_bound":max_ratio,"input_error_over_bound":input_error,"layer_accumulation_exact":accumulated,"backward_reset_exact":reset_equal,"zero_path_exact":zero,"canaries":canaries}));
                }
            }
        }
    }
    Ok(())
}

fn integration(rows: &mut Vec<Value>) -> Result<()> {
    let tokens = [1, 2, 3, 4, 3, 2];
    let labels = [2, 3, 4, 3, 2, 1];
    for (width, heads, layers) in [(64, 4, 1), (64, 4, 2), (128, 1, 1)] {
        let c = cfg(width, heads, layers, "complex-mhc");
        let p = layout::initialize_model(&c, 17);
        let mut e = Engine::new(c.clone(), &p)?;
        e.forward(&tokens, &labels, 2, 3)?;
        e.backward(2, 3)?;
        let baseline_logits = e.logits.read()?;
        let baseline_gradient = e.gradients.read()?;
        let mut opt = Optimizer::new(&e, 20000)?;
        opt.step(&e, 731)?;
        let baseline_state = [
            e.parameters.read()?,
            e.first_moment.read()?,
            e.second_moment.read()?,
        ];
        drop(opt);
        drop(e);
        for arm in ["complex-mhc-lerope", "complex-mhc-log-lerope"] {
            let c = cfg(width, heads, layers, arm);
            let offset = layout::frequency_offset(&c);
            let bands = layout::frequency_count(&c);
            let mut parameters = layout::initialize_model(&c, 17);
            let initial_match = bits_equal(&parameters[..offset], &p)
                && parameters[offset..].iter().all(|x| *x == 0.);
            let mut e = Engine::new(c.clone(), &parameters)?;
            e.forward(&tokens, &labels, 2, 3)?;
            e.backward(2, 3)?;
            let g = e.gradients.read()?;
            let logits = e.logits.read()?;
            e.backward(2, 3)?;
            let reset = bits_equal(&g, &e.gradients.read()?);
            let ordinary = arm == "complex-mhc-lerope";
            let logits_equal = bits_equal(&logits, &baseline_logits);
            let gradients_equal = bits_equal(&g[..offset], &baseline_gradient);
            let mut opt = Optimizer::new(&e, 20000)?;
            opt.step(&e, 731)?;
            let state = [
                e.parameters.read()?,
                e.first_moment.read()?,
                e.second_moment.read()?,
            ];
            let state_equal = state
                .iter()
                .zip(&baseline_state)
                .all(|(a, b)| bits_equal(&a[..offset], b));
            let frequencies_updated = state[0][offset..].iter().any(|x| *x != 0.);
            rows.push(json!({"kind":"initial_integration","arm":arm,"width":width,"heads":heads,"layers":layers,"passed":initial_match&&reset&&frequencies_updated&&(!ordinary||(logits_equal&&gradients_equal&&state_equal)),"original_parameter_bits_equal":initial_match,"backward_reset_exact":reset,"zero_frequency_logits_equal":logits_equal,"original_gradients_equal":gradients_equal,"original_update_state_equal":state_equal,"frequency_parameters_updated":frequencies_updated,"control_initial_sha256":hash(&bytes(&p))}));
            drop(opt);
            // Every learned frequency direction is differentiated by independent CPU Jets.
            for (i, v) in parameters[offset..].iter_mut().enumerate() {
                *v = ((i % 5) as f32 - 2.) * 0.075;
            }
            e.parameters.write(&parameters)?;
            e.forward(&tokens, &labels, 2, 3)?;
            e.backward(2, 3)?;
            let actual = e.gradients.read()?;
            let full_logits = e.logits.read()?;
            let mut expected = Vec::new();
            for band in 0..bands {
                let mut pjet: Vec<Jet> = parameters.iter().map(|&v| Jet::from(v as f64)).collect();
                pjet[offset + band].d = 1.;
                let logits = oracle::main_model(&c, &pjet, &tokens, 2, 3);
                expected.push(oracle::cross_entropy(&logits, &labels, c.vocab).d);
            }
            let norm = expected.iter().map(|x| x * x).sum::<f64>().sqrt();
            let strict_tolerance = 5e-4 * norm + 1e-12;
            let directions:Vec<_>=(0..bands).map(|i|{
                let error=(actual[offset+i]as f64-expected[i]).abs();
                json!({"band":i,"native":actual[offset+i],"reference":expected[i],"error":error,"tolerance":strict_tolerance,"passed":error.is_finite()&&error<=strict_tolerance})
            }).collect();
            let passed = directions.iter().all(|v| v["passed"] == true);
            rows.push(json!({"kind":"full_model_frequency_AD","arm":arm,"width":width,"heads":heads,"layers":layers,"passed":passed,"reference_gradient_norm":norm,"directions":directions}));

            let mut changed_future = tokens;
            changed_future[2] = 9;
            changed_future[5] = 10;
            e.forward(&changed_future, &labels, 2, 3)?;
            let changed = e.logits.read()?;
            let causal = [0, 1, 3, 4].iter().all(|&r| {
                bits_equal(
                    &full_logits[r * c.vocab..(r + 1) * c.vocab],
                    &changed[r * c.vocab..(r + 1) * c.vocab],
                )
            });
            e.forward(&[1, 2, 4, 3], &[2, 3, 3, 2], 2, 2)?;
            let prefix = e.logits.read()?;
            let prefix_equal = [0, 1, 3, 4].iter().enumerate().all(|(r, &original)| {
                bits_equal(
                    &full_logits[original * c.vocab..(original + 1) * c.vocab],
                    &prefix[r * c.vocab..(r + 1) * c.vocab],
                )
            });
            e.forward(&tokens, &labels, 2, 3)?;
            let cache_restored = bits_equal(
                &full_logits[..6 * c.vocab],
                &e.logits.read()?[..6 * c.vocab],
            );
            rows.push(json!({"kind":"causal_prefix","arm":arm,"width":width,"heads":heads,"layers":layers,
                "passed":causal&&prefix_equal&&cache_restored,"causal_future_exact":causal,"prefix_exact":prefix_equal,"length_cache_restore_exact":cache_restored}));
        }
    }
    Ok(())
}

pub fn run(root: &Path) -> Result<()> {
    if root.exists() {
        return Err("rotary validation output already exists".into());
    }
    fs::create_dir_all(root)?;
    for (name, body) in crate::sources::FILES {
        let path = root.join("source").join(name);
        fs::create_dir_all(path.parent().ok_or("missing parent")?)?;
        fs::write(path, body)?;
    }
    let mut rows = Vec::new();
    let outcome = (|| -> Result<()> {
        tables(&mut rows)?;
        adjoints(&mut rows)?;
        integration(&mut rows)
    })();
    let implementation_passed = outcome.is_ok()
        && rows
            .iter()
            .filter(|v| v["kind"] != "full_model_frequency_AD")
            .all(|v| v["passed"] == true);
    let strict_misses = rows
        .iter()
        .filter_map(|v| v["directions"].as_array())
        .flatten()
        .filter(|v| v["passed"] != true)
        .count();
    let passed = implementation_passed && strict_misses == 0;
    fs::write(
        root.join("report.json"),
        serde_json::to_vec_pretty(
            &json!({"passed":passed,"implementation_checks_passed":implementation_passed,"strict_scalar_AD_misses":strict_misses,"cases":rows,"source_sha256":runner::source_hash(),"ptx_sha256":runner::ptx_hash(),"execution_error":outcome.as_ref().err().map(|e|e.to_string()),"training_started":false,"remaining_validation":["frequency_optimizer_reference_and_restore","maximum_shapes","sanitizers","frozen_parent_executable_replay","classify_strict_AD_misses"]}),
        )?,
    )?;
    outcome?;
    if implementation_passed {
        Ok(())
    } else {
        Err("rotary validation contains failures; inspect report.json".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn initialization_and_optimizer_groups_preserve_every_original_coordinate() -> Result<()> {
        for (w, h, l) in [
            (64, 4, 1),
            (128, 4, 2),
            (128, 4, 6),
            (128, 1, 1),
            (256, 2, 2),
        ] {
            let baseline = cfg(w, h, l, "complex-mhc");
            let original = layout::initialize_model(&baseline, 7);
            for arm in ["complex-mhc-lerope", "complex-mhc-log-lerope"] {
                let c = cfg(w, h, l, arm);
                let p = layout::initialize_model(&c, 7);
                assert!(bits_equal(&p[..original.len()], &original));
                assert_eq!(p.len() - original.len(), w / h / 2);
                assert!(p[original.len()..].iter().all(|x| x.to_bits() == 0));
                assert_eq!(Optimizer::owners(&c)?, Optimizer::owners(&baseline)?);
                let mut owners = Optimizer::owners(&c)?;
                owners.extend(Optimizer::frequency_owners(&c));
                let mut seen = vec![0; p.len()];
                for [a, b, nb, aa, kind] in owners {
                    seen[a as usize] += 1;
                    seen[b as usize] += 1;
                    if kind == 0 {
                        seen[nb as usize] += 1;
                        seen[aa as usize] += 1;
                    }
                }
                assert!(seen.iter().all(|x| *x == 1));
                layout::validate(&c, &p)?;
            }
        }
        Ok(())
    }
}
