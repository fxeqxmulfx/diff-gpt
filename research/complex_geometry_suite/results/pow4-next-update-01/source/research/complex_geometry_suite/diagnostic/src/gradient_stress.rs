//! End-to-end derivative stress checks using independent FP64 forward-mode AD.
use crate::gradient_oracle::{self as oracle, Jet};
use crate::*;
use kernite_cuda::{Cuda, Grid};
use std::ops::Range;

struct Case {
    name: String,
    cfg: Config,
    p: Vec<f32>,
    tokens: Vec<u32>,
    labels: Vec<u32>,
    batch: usize,
    length: usize,
}
fn paths(cfg: &Config) -> Vec<(String, Range<usize>)> {
    let w = cfg.width;
    let h = layout::hidden(w, "");
    let mut result = vec![("embedding_readout".into(), 0..cfg.vocab * w)];
    let mut start = cfg.vocab * w;
    for layer in 0..cfg.layers {
        for (name, size) in [
            ("Q", w * w),
            ("K", w * w),
            ("V", w * w),
            ("attention_output", w * w),
            ("ffn_gate", h * w),
            ("ffn_value", h * w),
            ("ffn_output", h * w),
        ] {
            result.push((format!("layer{layer}.{name}"), start..start + size));
            start += size;
        }
    }
    result
}
fn owner_ids(cfg: &Config, r: &Range<usize>) -> Vec<usize> {
    if !layout::is_complex(&cfg.architecture) {
        return r.clone().collect();
    }
    layout::pairs(cfg)
        .into_iter()
        .flat_map(|v| [v[0] as usize, v[1] as usize])
        .filter(|i| r.contains(i))
        .collect()
}
fn synthetic(
    name: &str,
    width: usize,
    heads: usize,
    layers: usize,
    batch: usize,
    length: usize,
) -> Case {
    let cfg = Config {
        width,
        heads,
        layers,
        vocab: 17,
        max_batch: batch,
        max_length: length,
        normalized: true,
        ssmax: true,
        gain: heads as f32 / width as f32,
        lr: 0.002,
        decay: 0.1,
        warmup: 50,
        architecture: "complex-hermitian".into(),
        phase_optimizer: false,
    };
    let p = layout::initialize_model(&cfg, 7);
    let tokens: Vec<_> = (0..batch * length)
        .map(|i| {
            if i % length < 2 {
                (i / length + 1) as u32
            } else {
                ((i * 7 + 3) % 17) as u32
            }
        })
        .collect();
    let labels: Vec<_> = tokens
        .iter()
        .enumerate()
        .map(|(i, &t)| if i % 3 == 0 { u32::MAX } else { (t + 7) % 17 })
        .collect();
    Case {
        name: name.into(),
        cfg,
        p,
        tokens,
        labels,
        batch,
        length,
    }
}
fn scale(case: &mut Case, names: &[&str], factor: f32) {
    for (name, range) in paths(&case.cfg) {
        if names
            .iter()
            .any(|n| name == *n || name.ends_with(&format!(".{n}")))
        {
            for p in &mut case.p[range] {
                *p *= factor;
            }
        }
    }
}
fn synthetic_cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for (name, w, h, l, b, t) in [
        ("nominal", 64, 4, 1, 2, 4),
        ("depth_two_masks_tail", 64, 4, 2, 3, 7),
        ("depth_four", 64, 4, 4, 1, 3),
        ("width_128", 128, 4, 1, 2, 3),
        ("head_width_128", 128, 1, 1, 1, 4),
        ("width_256", 256, 8, 1, 1, 3),
        ("long_prefix_128", 64, 4, 1, 1, 128),
        ("single_token", 64, 4, 1, 3, 1),
    ] {
        cases.push(synthetic(name, w, h, l, b, t));
    }
    for arch in ["real", "real-product", "complex-linear"] {
        let mut c = synthetic(arch, 64, 4, 1, 2, 4);
        c.cfg.architecture = arch.into();
        c.p = layout::initialize_model(&c.cfg, 7);
        cases.push(c);
    }
    for (name, names, factor) in [
        ("embedding_tiny", vec!["embedding_readout"], 1e-6),
        ("qk_tiny", vec!["Q", "K"], 1e-6),
        ("qk_large", vec!["Q", "K"], 1e4),
        ("value_zero", vec!["V"], 0.),
        ("value_clipped", vec!["V"], 1e-13),
        ("value_small_unclipped", vec!["V"], 1e-9),
        ("ffn_product_tiny", vec!["ffn_gate", "ffn_value"], 1e-4),
        ("ffn_product_large", vec!["ffn_gate", "ffn_value"], 1e4),
        ("ffn_output_zero", vec!["ffn_output"], 0.),
        ("attention_output_zero", vec!["attention_output"], 0.),
    ] {
        let mut c = synthetic(name, 64, 4, 1, 2, 4);
        scale(&mut c, &names, factor);
        cases.push(c);
    }
    let mut c = synthetic("constant_prefix", 64, 4, 2, 2, 7);
    for b in 0..c.batch {
        c.tokens[b * c.length..(b + 1) * c.length].fill(b as u32 + 1);
    }
    cases.push(c);
    let mut c = synthetic("all_labels_masked", 64, 4, 2, 2, 4);
    c.labels.fill(u32::MAX);
    cases.push(c);
    let mut c = synthetic("confident_readout", 64, 4, 1, 2, 4);
    scale(&mut c, &["embedding_readout"], 50.);
    let p: Vec<_> = c.p.iter().map(|&v| Jet::from(v as f64)).collect();
    let z = oracle::model(&c.cfg, &p, &c.tokens, c.batch, c.length);
    for (row, values) in z.chunks_exact(c.cfg.vocab).enumerate() {
        if c.labels[row] != u32::MAX {
            c.labels[row] = values
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.v.total_cmp(&b.1.v))
                .unwrap()
                .0 as u32;
        }
    }
    cases.push(c);
    cases
}
fn checkpoint_case(name: &str, cp: &Checkpoint) -> Result<Case> {
    let task = data::build(&cp.protocol["case"])?;
    let train = &task.splits["train"];
    let mut packed = metrics::Packed::default();
    packed.fill(
        train,
        &[0, (train.len() / 3) as u32, (2 * train.len() / 3) as u32],
    );
    let mut cfg = cp.cfg.clone();
    cfg.max_batch = 3;
    Ok(Case {
        name: name.into(),
        cfg,
        p: cp.p.clone(),
        tokens: packed.tokens,
        labels: packed.labels,
        batch: 3,
        length: 2,
    })
}
fn check_case(case: &Case) -> Result<Value> {
    let mut engine = Engine::new(case.cfg.clone(), &case.p)?;
    engine.enable_stable_loss()?;
    engine.enable_stable_attention()?;
    let zero = vec![0.; case.p.len()];
    engine.first_moment.write(&zero)?;
    engine.second_moment.write(&zero)?;
    engine.forward(&case.tokens, &case.labels, case.batch, case.length)?;
    engine.backward(case.batch, case.length)?;
    let z = engine.logits.read()?;
    let g = engine.gradients.read()?;
    require(
        z[..case.batch * case.length * case.cfg.vocab]
            .iter()
            .chain(&g)
            .all(|x| x.is_finite()),
        "nonfinite full-path stress values",
    )?;
    let independent =
        projection_gradient(&case.cfg, &g.iter().map(|&v| v as f64).collect::<Vec<_>>());
    let p: Vec<_> = case.p.iter().map(|&v| Jet::from(v as f64)).collect();
    let reference_z = oracle::model(&case.cfg, &p, &case.tokens, case.batch, case.length);
    let reference_loss = oracle::cross_entropy(&reference_z, &case.labels, case.cfg.vocab);
    let n = case
        .labels
        .iter()
        .filter(|&&v| (v as usize) < case.cfg.vocab)
        .count()
        .max(1);
    let actual_loss = engine
        .losses(case.batch * case.length)?
        .iter()
        .map(|&v| v as f64)
        .sum::<f64>()
        / n as f64;
    let forward_error = z
        .iter()
        .zip(&reference_z)
        .map(|(a, b)| (*a as f64 - b.v).abs())
        .fold(0., f64::max);
    let forward_scale = reference_z.iter().map(|v| v.v.abs()).fold(1., f64::max);
    let mut failures = Vec::new();
    if forward_error > 5e-5 * forward_scale {
        failures.push(format!("forward error {forward_error}"));
    }
    if (actual_loss - reference_loss.v).abs() > 5e-5 * reference_loss.v.abs() + 1e-30 {
        failures.push(format!("loss {actual_loss} != {}", reference_loss.v));
    }
    let mut results = Vec::new();
    for (name, range) in paths(&case.cfg) {
        let ids = owner_ids(&case.cfg, &range);
        let values: Vec<_> = ids.iter().map(|&i| independent[i]).collect();
        let norm = stats::norm(&values);
        for direction in 0..4 {
            if direction == 3 && norm == 0. {
                continue;
            }
            // Unit directions in independent coordinates: two dense probes,
            // largest-magnitude coordinate, and the native-gradient direction.
            let mut d = vec![0.; case.p.len()];
            if direction < 2 {
                for &i in &ids {
                    d[i] = ((i + 1) as f64 * (0.319 + direction as f64 * 0.417)).sin();
                }
                let length = ids.iter().map(|&i| d[i] * d[i]).sum::<f64>().sqrt();
                for &i in &ids {
                    d[i] /= length;
                }
            } else if direction == 2 {
                let &i = ids
                    .iter()
                    .max_by(|&&a, &&b| independent[a].abs().total_cmp(&independent[b].abs()))
                    .unwrap();
                d[i] = 1.;
            } else {
                for &i in &ids {
                    d[i] = independent[i] / norm;
                }
            }
            let analytic: f64 = ids.iter().map(|&i| independent[i] * d[i]).sum();
            if layout::is_complex(&case.cfg.architecture) {
                for [ar, bi, br, ai, kind] in layout::pairs(&case.cfg) {
                    if kind == 0 {
                        d[ai as usize] = d[ar as usize];
                        d[br as usize] = -d[bi as usize];
                    }
                }
            }
            let dual: Vec<_> = case
                .p
                .iter()
                .zip(&d)
                .map(|(&v, &d)| Jet { v: v as f64, d })
                .collect();
            let logits = oracle::model(&case.cfg, &dual, &case.tokens, case.batch, case.length);
            let expected = oracle::cross_entropy(&logits, &case.labels, case.cfg.vocab).d;
            // A norm-scaled bound remains meaningful for almost-orthogonal
            // directions. Gradient-aligned and coordinate probes complement it.
            let tolerance = 5e-4 * norm + 1e-12;
            let error = (analytic - expected).abs();
            let passed = expected.is_finite() && error <= tolerance;
            if !passed {
                failures.push(format!(
                    "{name} direction{direction}: {analytic} vs {expected}, tolerance {tolerance}"
                ));
            }
            results.push(
                json!({"path":name,"direction":direction,"native_derivative":analytic,
                "independent_fp64_derivative":expected,"absolute_error":error,"tolerance":tolerance,
                "native_block_gradient_norm":norm,"passed":passed}),
            );
        }
    }
    require(
        engine.parameters.read()? == case.p
            && engine.first_moment.read()? == zero
            && engine.second_moment.read()? == zero,
        "stress altered model state",
    )?;
    Ok(
        json!({"case":case.name,"shape":{"width":case.cfg.width,"heads":case.cfg.heads,"layers":case.cfg.layers,"vocab":case.cfg.vocab,"batch":case.batch,"length":case.length},
        "architecture":case.cfg.architecture,"native_loss":actual_loss,"reference_loss":reference_loss.v,
        "max_logit_error":forward_error,"relative_logit_error":forward_error/forward_scale,
        "parameter_paths":paths(&case.cfg).len(),"directional_checks":results,"failures":failures,
        "passed":failures.is_empty(),"training_updates":0}),
    )
}

fn primitives() -> Result<Vec<Value>> {
    let cuda = Cuda::new()?;
    let rms = base_ops::kernels::Kernels::load(&cuda)?;
    let product = complex::kernels::Kernels::load(&cuda)?;
    let mut results = Vec::new();
    for w in [64, 128, 256] {
        let scales = [0., 1e-10, 1e-5, 0.00034526698, 1., 1e4, 1048576.];
        let rows = scales.len();
        let mut x = vec![0f32; rows * w];
        let mut dy = x.clone();
        for row in 0..rows {
            for c in 0..w {
                x[row * w + c] = if row == 6 {
                    scales[row]
                } else {
                    scales[row] * ((c as f32 * 0.173).sin() + 0.17)
                };
                dy[row * w + c] = if row == 6 {
                    x[row * w + c]
                } else {
                    0.1 * (c as f32 * 0.217).cos()
                };
            }
        }
        let input = cuda.upload(&x)?;
        let gout = cuda.upload(&dy)?;
        let y = cuda.alloc::<f32>(x.len())?;
        let dx = cuda.alloc::<f32>(x.len())?;
        macro_rules! launch {
            ($w:literal) => {
                unsafe {
                    rms.rms::<$w>(
                        Grid::new(rows.div_ceil(4) as u32, 128),
                        y.ptr(),
                        input.ptr(),
                        rows as u32,
                    )?;
                    rms.rms_bwd::<$w>(
                        Grid::new(rows.div_ceil(4) as u32, 128),
                        dx.ptr(),
                        input.ptr(),
                        gout.ptr(),
                        rows as u32,
                    )?;
                }
            };
        }
        match w {
            64 => launch!(64),
            128 => launch!(128),
            256 => launch!(256),
            _ => unreachable!(),
        }
        let actual = y.read()?;
        let grad = dx.read()?;
        for row in 0..rows {
            let values: Vec<_> = x[row * w..(row + 1) * w]
                .iter()
                .map(|&v| v as f64)
                .collect();
            let dy: Vec<_> = dy[row * w..(row + 1) * w]
                .iter()
                .map(|&v| v as f64)
                .collect();
            let mean = stats::dot(&values, &values) / w as f64;
            let inv = (mean + f32::EPSILON as f64).sqrt().recip();
            let dot = stats::dot(&values, &dy) / w as f64;
            let expected: Vec<_> = (0..w)
                .map(|i| {
                    if row == 6 {
                        values[i] * f32::EPSILON as f64 * inv.powi(3)
                    } else {
                        inv * (dy[i] - values[i] * dot * inv * inv)
                    }
                })
                .collect();
            let error: Vec<_> = expected
                .iter()
                .enumerate()
                .map(|(i, &e)| grad[row * w + i] as f64 - e)
                .collect();
            let rel = stats::norm(&error) / stats::norm(&expected).max(1e-300);
            let fwd = (0..w)
                .map(|i| (actual[row * w + i] as f64 - values[i] * inv).abs())
                .fold(0., f64::max);
            let passed = rel <= 5e-6 && fwd <= 5e-6;
            results.push(json!({"primitive":"RMSNorm","width":w,"scale":scales[row],"parallel_gradient":row==6,"gradient_relative_error":rel,"forward_max_error":fwd,"passed":passed}));
        }
    }
    let pairs = 17;
    let scales = [0., 1e-10, 1e-4, 1., 20., 1e4, 1e8];
    let rows = scales.len();
    let mut x = vec![0f32; rows * 4 * pairs];
    let mut dy = vec![0f32; rows * 2 * pairs];
    for row in 0..rows {
        for c in 0..4 * pairs {
            let scale = if row == 0 && c >= 2 * pairs {
                1.
            } else {
                scales[row]
            };
            x[row * 4 * pairs + c] = scale * ((c as f32 * 0.173).sin() + 0.21);
        }
        for c in 0..2 * pairs {
            dy[row * 2 * pairs + c] = 0.1 * (c as f32 * 0.317).cos();
        }
    }
    let input = cuda.upload(&x)?;
    let gout = cuda.upload(&dy)?;
    let y = cuda.alloc::<f32>(dy.len())?;
    let dx = cuda.alloc::<f32>(x.len())?;
    unsafe {
        product.product::<true>(
            Grid::new((rows * pairs).div_ceil(256) as u32, 256),
            y.ptr(),
            input.ptr(),
            pairs as u32,
            (rows * pairs) as u32,
        )?;
        product.product_bwd::<true>(
            Grid::new((rows * pairs).div_ceil(256) as u32, 256),
            dx.ptr(),
            input.ptr(),
            gout.ptr(),
            pairs as u32,
            (rows * pairs) as u32,
        )?;
    }
    let actual = y.read()?;
    let grad = dx.read()?;
    for row in 0..rows {
        let base: Vec<_> = x[row * 4 * pairs..(row + 1) * 4 * pairs]
            .iter()
            .map(|&v| Jet::from(v as f64))
            .collect();
        let expected = oracle::product(&base);
        let fe: Vec<_> = expected
            .iter()
            .enumerate()
            .map(|(i, e)| actual[row * 2 * pairs + i] as f64 - e.v)
            .collect();
        let fv: Vec<_> = expected.iter().map(|v| v.v).collect();
        let mut reference = Vec::new();
        for c in 0..4 * pairs {
            let mut seeded = base.clone();
            seeded[c].d = 1.;
            reference.push(
                oracle::product(&seeded)
                    .iter()
                    .enumerate()
                    .map(|(i, v)| v.d * dy[row * 2 * pairs + i] as f64)
                    .sum::<f64>(),
            );
        }
        let ge: Vec<_> = reference
            .iter()
            .enumerate()
            .map(|(i, &v)| grad[row * 4 * pairs + i] as f64 - v)
            .collect();
        let forward = stats::norm(&fe) / stats::norm(&fv).max(1e-300);
        let backward = stats::norm(&ge) / stats::norm(&reference).max(1e-300);
        results.push(json!({"primitive":"radial_complex_product","scale":scales[row],"input_gradient_coordinates":4*pairs,
            "forward_relative_error":forward,"gradient_relative_error":backward,"passed":forward<=5e-6&&backward<=5e-6}));
    }
    Ok(results)
}

fn scalar_vjp(x: &[f32], dy: &[f32], f: impl Fn(&[Jet]) -> Vec<Jet>) -> Vec<f64> {
    let mut dual: Vec<_> = x.iter().map(|&v| Jet::from(v as f64)).collect();
    (0..x.len())
        .map(|i| {
            dual[i].d = 1.;
            let y = f(&dual);
            dual[i].d = 0.;
            y.iter().zip(dy).map(|(v, &g)| v.d * g as f64).sum()
        })
        .collect()
}

fn adjoint_error(name: &str, actual: &[f32], expected: &[f64]) -> Value {
    let error = actual
        .iter()
        .zip(expected)
        .map(|(&a, &b)| (a as f64 - b).powi(2))
        .sum::<f64>()
        .sqrt();
    let norm = stats::norm(expected);
    json!({"stage":name,"reference_norm":norm,"absolute_error":error,"relative_error":error/norm.max(1e-300),"passed":error<=5e-6*norm+1e-12})
}

/// Localizes loss of the small radial derivative in the single-head stress case.
/// Run explicitly on CUDA; it performs no optimizer update.
fn trace_head_width_128(stable_attention: bool) -> Result<Value> {
    let c = synthetic("head_width_128", 128, 1, 1, 1, 4);
    let mut e = Engine::new(c.cfg.clone(), &c.p)?;
    e.enable_stable_loss()?;
    if stable_attention {
        e.enable_stable_attention()?;
    }
    e.forward(&c.tokens, &c.labels, c.batch, c.length)?;
    e.backward(c.batch, c.length)?;
    let snapshot = e.diagnostic_snapshot(c.batch * c.length)?;
    let grad = projection_gradient(
        &c.cfg,
        &e.gradients
            .read()?
            .iter()
            .map(|&v| v as f64)
            .collect::<Vec<_>>(),
    );
    let (w, o) = (c.cfg.width, c.cfg.vocab * c.cfg.width);
    let ids = owner_ids(&c.cfg, &(o..o + w * w));
    let norm = ids.iter().map(|&i| grad[i] * grad[i]).sum::<f64>().sqrt();
    let mut direction = vec![0.; c.p.len()];
    for i in ids {
        direction[i] = grad[i] / norm;
    }
    for [a, b, br, ai, kind] in layout::pairs(&c.cfg) {
        if kind == 0 {
            direction[ai as usize] = direction[a as usize];
            direction[br as usize] = -direction[b as usize];
        }
    }
    let p: Vec<_> =
        c.p.iter()
            .zip(&direction)
            .map(|(&v, &d)| Jet { v: v as f64, d })
            .collect();
    let qkv: Vec<_> = c
        .tokens
        .iter()
        .flat_map(|&t| {
            oracle::projection(
                &c.cfg,
                &oracle::rms(&oracle::rms(&p[t as usize * w..(t as usize + 1) * w])),
                &p[o..],
                3 * w,
                3,
                1,
            )
        })
        .collect();
    let att = oracle::attention(&c.cfg, &qkv, c.batch, c.length);
    let ap: Vec<_> = att
        .chunks_exact(w)
        .flat_map(|r| oracle::projection(&c.cfg, r, &p[o + 3 * w * w..], w, 1, 1))
        .collect();
    let an: Vec<_> = ap.chunks_exact(w).flat_map(oracle::rms).collect();
    let z = oracle::model(&c.cfg, &p, &c.tokens, c.batch, c.length);
    let expected = oracle::cross_entropy(&z, &c.labels, c.cfg.vocab).d;
    let mut rows = Vec::new();
    for (name, values, gradient) in [
        ("qkv", &qkv, "grad_qkv"),
        ("attention", &att, "grad_attention"),
        ("attention_projected", &ap, "grad_attention_projected"),
        ("attention_normalized", &an, "grad_attention_normalized"),
    ] {
        let g = &snapshot[&format!("layer0.{gradient}")];
        let actual: f64 = values.iter().zip(g).map(|(v, &g)| v.d * g as f64).sum();
        rows.push(json!({"stage":name,"derivative":actual,"expected":expected,"relative_error":(actual-expected)/expected}));
    }
    // Exact derivative identities for the radial chain at the active third token.
    let mut radial = Vec::new();
    for (value, gradient) in [
        ("attention", "grad_attention"),
        ("attention_projected", "grad_attention_projected"),
        ("qkv_rotated", "grad_qkv"),
    ] {
        if value == "qkv_rotated" {
            continue;
        }
        let x = &snapshot[&format!("layer0.{value}")][2 * w..3 * w];
        let g = &snapshot[&format!("layer0.{gradient}")][2 * w..3 * w];
        let dot = x
            .iter()
            .zip(g)
            .map(|(&x, &g)| x as f64 * g as f64)
            .sum::<f64>();
        radial.push(json!({"stage":value,"radial_derivative":dot}));
    }
    let x = &snapshot["layer0.attention_projected"][2 * w..3 * w];
    let g = &snapshot["layer0.grad_attention_normalized"][2 * w..3 * w];
    let energy = x.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / w as f64;
    let dot = x
        .iter()
        .zip(g)
        .map(|(&x, &g)| x as f64 * g as f64)
        .sum::<f64>();
    let stable_radial = dot * f32::EPSILON as f64 / (energy + f32::EPSILON as f64).powf(1.5);
    let mut local = Vec::new();
    let x = &snapshot["layer0.attention_projected"];
    let dy = &snapshot["layer0.grad_attention_normalized"];
    let reference = scalar_vjp(x, dy, |x| x.chunks_exact(w).flat_map(oracle::rms).collect());
    local.push(adjoint_error(
        "peri_rms",
        &snapshot["layer0.grad_attention_projected"],
        &reference,
    ));
    let x = &snapshot["layer0.attention"];
    let dy = &snapshot["layer0.grad_attention_projected"];
    let reference = scalar_vjp(x, dy, |x| {
        x.chunks_exact(w)
            .flat_map(|r| oracle::projection(&c.cfg, r, &p[o + 3 * w * w..], w, 1, 1))
            .collect()
    });
    local.push(adjoint_error(
        "output_projection",
        &snapshot["layer0.grad_attention"],
        &reference,
    ));
    let x = &snapshot["layer0.qkv_rotated"];
    let dy = &snapshot["layer0.grad_attention"];
    let reference = scalar_vjp(x, dy, |x| {
        oracle::attention_qkv(&c.cfg, x, c.batch, c.length)
    });
    for (part, name) in ["Q", "K", "V"].iter().enumerate() {
        let ids: Vec<_> = (0..c.batch * c.length)
            .flat_map(|r| r * 3 * w + part * w..r * 3 * w + (part + 1) * w)
            .collect();
        local.push(adjoint_error(
            &format!("XSA.{name}"),
            &ids.iter()
                .map(|&i| snapshot["layer0.grad_qkv_rotated"][i])
                .collect::<Vec<_>>(),
            &ids.iter().map(|&i| reference[i]).collect::<Vec<_>>(),
        ));
    }
    Ok(
        json!({"stable_attention":stable_attention,"norm":norm,"expected":expected,"stages":rows,"radial":radial,"rms_stable_radial":stable_radial,"local_adjoints":local}),
    )
}

pub fn run(parent_root: &Path, continuation_root: &Path, output: &Path) -> Result<()> {
    require(
        !output.exists(),
        "refuse overwrite of gradient stress artifacts",
    )?;
    let mut cases = synthetic_cases();
    let synthetic_count = cases.len();
    let mut protected = Vec::new();
    for arm in ARMS {
        for seed in 0..3 {
            let cp = Checkpoint::load(parent_root, arm, seed)?;
            cases.push(checkpoint_case(&format!("{arm}-seed{seed}"), &cp)?);
            protected.push((
                cp.dir.join("checkpoint.bin"),
                cp.checkpoint["state_sha256"].as_str().unwrap().to_owned(),
            ));
            cp.unchanged()?;
        }
    }
    let parent = Checkpoint::load(parent_root, "complex-hermitian", 1)?;
    for arm in ["unchanged", "lower_lr"] {
        for step in [0, 9500, 19000] {
            let s = raw::load(&parent, continuation_root, arm, step)?;
            cases.push(checkpoint_case(&s.name, &s.cp)?);
            protected.push((
                continuation_root
                    .join(arm)
                    .join(format!("checkpoint-{step:06}.bin")),
                s.cp.checkpoint["state_sha256"].as_str().unwrap().to_owned(),
            ));
            s.unchanged()?;
        }
    }
    fs::create_dir_all(output)?;
    let files = [
        ("src/gradient_stress.rs", include_str!("gradient_stress.rs")),
        ("src/gradient_oracle.rs", include_str!("gradient_oracle.rs")),
        ("src/engine.rs", include_str!("engine.rs")),
        ("src/stable_loss.rs", include_str!("stable_loss.rs")),
        (
            "src/stable_attention.rs",
            include_str!("stable_attention.rs"),
        ),
        ("src/main.rs", include_str!("main.rs")),
        ("src/next_update.rs", include_str!("next_update.rs")),
        (
            "NEXT_UPDATE_PLAN.txt",
            include_str!("../NEXT_UPDATE_PLAN.txt"),
        ),
        ("src/retrain.rs", include_str!("retrain.rs")),
        ("RETRAIN_PLAN.txt", include_str!("../RETRAIN_PLAN.txt")),
        ("src/raw.rs", include_str!("raw.rs")),
        ("src/loss_recheck.rs", include_str!("loss_recheck.rs")),
        ("src/loss_precision.rs", include_str!("loss_precision.rs")),
        ("src/continuation.rs", include_str!("continuation.rs")),
        ("src/probe.rs", include_str!("probe.rs")),
        ("src/stats.rs", include_str!("stats.rs")),
        ("Cargo.toml", include_str!("../Cargo.toml")),
        ("Cargo.lock", include_str!("../Cargo.lock")),
        (
            "CONTINUATION_PLAN.txt",
            include_str!("../CONTINUATION_PLAN.txt"),
        ),
    ];
    let mut source = Vec::new();
    for (name, body) in files {
        let path = output.join("source").join(name);
        fs::create_dir_all(path.parent().unwrap())?;
        fs::write(path, body)?;
        source.extend((name.len() as u64).to_le_bytes());
        source.extend(name.as_bytes());
        source.extend((body.len() as u64).to_le_bytes());
        source.extend(body.as_bytes());
    }
    let plan = json!({"source_sha256":hash(&source),"original_ptx_sha256":ptx_hash(),"corrected_loss_ptx_sha256":hash(stable_loss::kernels::PTX.as_bytes()),
        "corrected_attention_ptx_sha256":hash(stable_attention::kernels::PTX.as_bytes()),
        "independent_oracle":"scalar FP64 forward-mode AD; direct complex matrix products and Hermitian projection",
        "synthetic_cases":synthetic_count,"saved_checkpoint_cases":18,"full_model_cases":cases.len(),
        "cases":cases.iter().map(|c|&c.name).collect::<Vec<_>>(),"directions_per_parameter_block":4,
        "directional_tolerance":"5e-4 * native independent-coordinate block-gradient norm + 1e-12",
        "coordinate_sampling_limitation":"Every parameter block and component path is exercised, not every individual model parameter or nondifferentiable clip boundary",
        "checkpoint_examples":"three fixed training examples: indices 0, floor(n/3), floor(2n/3)",
        "primitive_checks":"all RMS and complex-product input coordinates on specified rows; all input adjoints of the head128 Peri-RMS, output projection and XSA",
        "training_updates":0,"python_invoked":false});
    save(&output.join("protocol.json"), &plan)?;
    let primitive = primitives()?;
    save(&output.join("primitives.json"), &json!(primitive))?;
    let trace =
        json!({"legacy":trace_head_width_128(false)?,"candidate":trace_head_width_128(true)?});
    save(&output.join("head128-local-adjoints.json"), &trace)?;
    let mut rows = Vec::new();
    for case in &cases {
        let result = check_case(case)?;
        save(&output.join(format!("{}.json", case.name)), &result)?;
        eprintln!(
            "gradient stress {}: {}",
            case.name,
            if result["passed"] == true {
                "passed"
            } else {
                "FAILED"
            }
        );
        rows.push(result);
    }
    for (path, sha) in &protected {
        require(
            hash(&fs::read(path)?) == *sha,
            "stress changed a checkpoint",
        )?;
    }
    let passed = primitive.iter().all(|r| r["passed"] == true)
        && rows.iter().all(|r| r["passed"] == true)
        && trace["candidate"]["local_adjoints"]
            .as_array()
            .unwrap()
            .iter()
            .all(|r| r["passed"] == true);
    save(
        &output.join("summary.json"),
        &json!({"complete":true,"passed":passed,"protocol":plan,"primitives":primitive,"rows":rows,"local_adjoint_trace":trace,
        "checkpoints_unchanged":true,"training_updates":0}),
    )?;
    require(
        passed,
        "one or more full gradient path stress checks failed; inspect saved summary",
    )
}
