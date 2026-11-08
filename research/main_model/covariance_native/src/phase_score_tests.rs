//! Independent CPU complex arithmetic and complete primitive adjoint checks.
use crate::{
    AttentionMode, Config, Engine, Result, layout,
    optimizer::Optimizer,
    oracle::{self, Jet},
    runner,
};
use kernite_cuda::{Cuda, Grid};
use serde_json::{Value, json};
use std::{fs, io::Write, path::Path};
const GUARD: f32 = 12345.;
fn record(name: &str, value: impl std::borrow::Borrow<Value>) -> Result<()> {
    if let Some(root) = std::env::var_os("SCORE_VALIDATION_OUTPUT") {
        let mut f = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(Path::new(&root).join(format!("{name}.json")))?;
        f.write_all(&serde_json::to_vec_pretty(value.borrow())?)?;
        f.sync_all()?;
    }
    Ok(())
}
fn config(w: usize, h: usize, l: usize, score: bool) -> Config {
    Config {
        width: w,
        heads: h,
        layers: l,
        vocab: 11,
        max_batch: 3,
        max_length: 17,
        normalized: true,
        ssmax: true,
        gain: h as f32 / w as f32,
        lr: 0.001,
        decay: 0.1,
        warmup: 400,
        architecture: if score {
            "complex-phase-score-lerope"
        } else {
            "complex-phase-mhc-lerope"
        }
        .into(),
        phase_optimizer: false,
    }
}
fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}
#[test]
fn score_layout_and_complete_optimizer_ownership() -> Result<()> {
    for (w, h, l) in [
        (64, 4, 2),
        (128, 4, 1),
        (128, 1, 1),
        (64, 2, 3),
        (128, 1, 2),
    ] {
        let c = config(w, h, l, true);
        let p = config(w, h, l, false);
        for seed in 0..3 {
            let a = layout::initialize_model(&c, seed);
            let b = layout::initialize_model(&p, seed);
            let offset = layout::score_offset(&c);
            assert_eq!(offset, layout::frequency_offset(&p));
            assert_eq!(a.len(), b.len() + l * h);
            assert_eq!(
                layout::independent_count(&c),
                layout::independent_count(&p) + l * h
            );
            assert!(same_bits(&a[..offset], &b[..offset]));
            assert!(same_bits(&a[layout::frequency_offset(&c)..], &b[offset..]));
            assert!(
                a[offset..layout::frequency_offset(&c)]
                    .iter()
                    .all(|v| v.to_bits() == 0)
            );
        }
        let mut writes = vec![0usize; layout::total_count(&c)];
        let mut coordinates = 0;
        for [ar, bi, br, ai, kind] in Optimizer::owners(&c)?
            .into_iter()
            .chain(Optimizer::frequency_owners(&c))
        {
            writes[ar as usize] += 1;
            coordinates += 1;
            if kind != 2 {
                writes[bi as usize] += 1;
                coordinates += 1;
            }
            if kind == 0 {
                writes[br as usize] += 1;
                writes[ai as usize] += 1;
            }
        }
        assert!(writes.iter().all(|&n| n == 1));
        assert_eq!(coordinates, layout::independent_count(&c));
    }
    Ok(())
}
struct Reference {
    y: Vec<f64>,
    dx: Vec<f64>,
    da: Vec<f64>,
    abs_da: Vec<f64>,
}
fn reference(
    d: usize,
    heads: usize,
    batch: usize,
    length: usize,
    x: &[f32],
    a: &[f32],
    dy: &[f32],
) -> Reference {
    let w = d * heads;
    let mut out = Reference {
        y: vec![0.; batch * length * w],
        dx: vec![0.; x.len()],
        da: vec![0.; heads],
        abs_da: vec![0.; heads],
    };
    let get = |b: usize, t: usize, part: usize, h: usize, c: usize| {
        f64::from(x[(b * length + t) * 3 * w + part * w + h * d + c])
    };
    for b in 0..batch {
        for t in 0..length {
            for (h, &coordinate) in a.iter().enumerate().take(heads) {
                let parameter = f64::from(coordinate);
                let denominator = parameter.hypot(1.);
                let beta = parameter / denominator;
                let derivative = denominator.powi(-3);
                let mut scores = Vec::new();
                let mut ims = Vec::new();
                for k in 0..=t {
                    let mut real = 0.;
                    let mut imaginary = 0.;
                    for c in 0..d / 2 {
                        let qr = get(b, t, 0, h, c);
                        let qi = get(b, t, 0, h, c + d / 2);
                        let kr = get(b, k, 1, h, c);
                        let ki = get(b, k, 1, h, c + d / 2);
                        real += qr * kr + qi * ki;
                        imaginary += qr * ki - qi * kr;
                    }
                    let re = real / d as f64;
                    let im = imaginary / d as f64;
                    scores.push(((t + 1) as f64).ln() * (re + beta * im * im));
                    ims.push(imaginary);
                }
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let mass: Vec<_> = scores.iter().map(|s| (s - max).exp()).collect();
                let den: f64 = mass.iter().sum();
                let prob: Vec<_> = mass.iter().map(|v| v / den).collect();
                let at = (b * length + t) * w + h * d;
                let mut dp = vec![0.; t + 1];
                for k in 0..=t {
                    for c in 0..d {
                        let v = get(b, k, 2, h, c);
                        out.y[at + c] += prob[k] * v;
                        dp[k] += f64::from(dy[at + c]) * v;
                        out.dx[(b * length + k) * 3 * w + 2 * w + h * d + c] +=
                            prob[k] * f64::from(dy[at + c]);
                    }
                }
                let mean: f64 = dp.iter().zip(&prob).map(|(v, p)| v * p).sum();
                let gain = ((t + 1) as f64).ln() / d as f64;
                for k in 0..=t {
                    let gr = prob[k] * (dp[k] - mean) * gain;
                    let gi = gr * 2. * beta * ims[k] / d as f64;
                    let term = gr * ims[k] * ims[k] / d as f64 * derivative;
                    out.da[h] += term;
                    out.abs_da[h] += term.abs();
                    let q = (b * length + t) * 3 * w + h * d;
                    let kk = (b * length + k) * 3 * w + w + h * d;
                    for c in 0..d / 2 {
                        let qr = get(b, t, 0, h, c);
                        let qi = get(b, t, 0, h, c + d / 2);
                        let kr = get(b, k, 1, h, c);
                        let ki = get(b, k, 1, h, c + d / 2);
                        out.dx[q + c] += gr * kr + gi * ki;
                        out.dx[q + c + d / 2] += gr * ki - gi * kr;
                        out.dx[kk + c] += gr * qr - gi * qi;
                        out.dx[kk + c + d / 2] += gr * qi + gi * qr;
                    }
                }
            }
        }
    }
    out
}
fn inputs(d: usize, h: usize, b: usize, t: usize, pattern: usize) -> (Vec<f32>, Vec<f32>) {
    let w = d * h;
    let mut x = vec![0.; b * t * 3 * w];
    for row in 0..b * t {
        for head in 0..h {
            for part in 0..3 {
                let start = row * 3 * w + part * w + head * d;
                for c in 0..d {
                    let real = c < d / 2;
                    let band = c % (d / 2);
                    let angle = (row * 19 + head * 7 + band * 3 + part * 11 + 1) as f64 * 0.37;
                    let r = angle.sin();
                    let i = (angle * 1.3).cos();
                    x[start + c] = if part == 2 {
                        ((angle * 0.17).sin() * 0.7) as f32
                    } else {
                        (match pattern {
                            0 => 0.,
                            1 => {
                                if real {
                                    r
                                } else {
                                    0.
                                }
                            }
                            2 => {
                                if real {
                                    0.
                                } else {
                                    i
                                }
                            }
                            3 => {
                                if (part == 0) == real {
                                    r
                                } else {
                                    0.
                                }
                            }
                            4 => {
                                ((head * 3 + band + 1) as f64 * 0.7).sin()
                                    * if real { 1. } else { -1. }
                            }
                            5 | 8 | 9 => {
                                if real {
                                    r
                                } else {
                                    i
                                }
                            }
                            6 => ((band + 1) as f64 * 0.3).cos() * if part == 0 { 1. } else { -1. },
                            _ => {
                                if (row + band).is_multiple_of(2) {
                                    1.
                                } else {
                                    -1.
                                }
                            }
                        }) as f32
                    };
                }
                if part < 2 {
                    let norm = x[start..start + d]
                        .iter()
                        .map(|&v| f64::from(v).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    if norm > 0. {
                        for v in &mut x[start..start + d] {
                            *v = (f64::from(*v) * (d as f64).sqrt() / norm) as f32;
                        }
                    }
                }
            }
        }
    }
    if pattern == 8 || pattern == 9 {
        for row in 0..b * t {
            for head in 0..h {
                for part in 0..2 {
                    for c in 0..d / 2 {
                        let at = row * 3 * w + part * w + head * d + c;
                        let re = x[at];
                        let im = x[at + d / 2];
                        if pattern == 8 {
                            x[at] = -im;
                            x[at + d / 2] = re;
                        } else {
                            x[at + d / 2] = -im;
                        }
                    }
                }
            }
        }
    }
    let dy = (0..b * t * w)
        .map(|i| ((i as f64 * 0.39).cos() * 0.4) as f32)
        .collect();
    (x, dy)
}
#[test]
fn common_phase_and_conjugation_preserve_scores_and_transform_gradients() {
    for (d, h) in [(16, 4), (32, 2), (128, 1)] {
        let (batch, length) = (2, 5);
        let w = d * h;
        let (x, dy) = inputs(d, h, batch, length, 5);
        let a = vec![0.7; h];
        let original = reference(d, h, batch, length, &x, &a, &dy);
        for pattern in [8, 9] {
            let (changed, _) = inputs(d, h, batch, length, pattern);
            let rotated = reference(d, h, batch, length, &changed, &a, &dy);
            assert!(
                original
                    .y
                    .iter()
                    .zip(&rotated.y)
                    .all(|(x, y)| (x - y).abs() < 2e-13)
            );
            assert!(
                original
                    .da
                    .iter()
                    .zip(&rotated.da)
                    .all(|(x, y)| (x - y).abs() < 2e-12)
            );
            let mut expected = original.dx.clone();
            for row in 0..batch * length {
                for head in 0..h {
                    for part in 0..2 {
                        for c in 0..d / 2 {
                            let at = row * 3 * w + part * w + head * d + c;
                            let re = original.dx[at];
                            let im = original.dx[at + d / 2];
                            if pattern == 8 {
                                expected[at] = -im;
                                expected[at + d / 2] = re;
                            } else {
                                expected[at + d / 2] = -im;
                            }
                        }
                    }
                }
            }
            assert!(
                expected
                    .iter()
                    .zip(&rotated.dx)
                    .all(|(x, y)| (x - y).abs() < 2e-12)
            );
        }
    }
}
#[test]
fn independent_attention_adjoint_against_scalar_dual_and_finite_differences() {
    for (d, h, t) in [(16, 1, 3), (32, 2, 2)] {
        for value in [0., 0.4, -1.] {
            let c = config(d * h, h, 1, true);
            let (x, dy) = inputs(d, h, 1, t, 5);
            let a = vec![value; h];
            let expected = reference(d, h, 1, t, &x, &a, &dy);
            let objective = |p: &[Jet], s: &[Jet]| {
                oracle::attention_qkv_score(&c, p, 1, t, AttentionMode::Ordinary, s)
                    .iter()
                    .zip(&dy)
                    .fold(Jet::default(), |v, (x, &y)| v + *x * f64::from(y))
            };
            for id in 0..x.len() + h {
                let mut p: Vec<Jet> = x.iter().map(|&v| Jet::from(f64::from(v))).collect();
                let mut s: Vec<Jet> = a.iter().map(|&v| Jet::from(f64::from(v))).collect();
                let expected_grad = if id < x.len() {
                    p[id].d = 1.;
                    expected.dx[id]
                } else {
                    s[id - x.len()].d = 1.;
                    expected.da[id - x.len()]
                };
                let dual = objective(&p, &s).d;
                assert!((dual - expected_grad).abs() < 2e-10 + 1e-8 * expected_grad.abs());
                let eps = 1e-5;
                for z in p.iter_mut().chain(&mut s) {
                    z.v += eps * z.d;
                }
                let plus = objective(&p, &s).v;
                for z in p.iter_mut().chain(&mut s) {
                    z.v -= 2. * eps * z.d;
                }
                let minus = objective(&p, &s).v;
                assert!(((plus - minus) / (2. * eps) - dual).abs() < 2e-9 + 1e-7 * dual.abs());
            }
        }
    }
}
fn compare(a: &[f32], b: &[f64]) -> (usize, f64) {
    let mut misses = 0;
    let mut worst = 0f64;
    for (&a, &b) in a.iter().zip(b) {
        let a = f64::from(a);
        let ratio = (a - b).abs() / (2e-6 + 2e-5 * a.abs().max(b.abs()));
        if !ratio.is_finite() || ratio > 1. {
            misses += 1;
        }
        worst = worst.max(ratio);
    }
    (misses, worst)
}
#[test]
#[ignore = "requires CUDA; independent full primitive gradient screen"]
fn gpu_score_primitives() -> Result<()> {
    let cuda = Cuda::new()?;
    let score = crate::phase_score::kernels::Kernels::load(&cuda)?;
    let forward = crate::phase_score_forward::kernels::Kernels::load(&cuda)?;
    let ordinary = crate::ordinary_attention::kernels::Kernels::load(&cuda)?;
    let base = crate::attention::kernels::Kernels::load(&cuda)?;
    let control = crate::ordinary_forward::kernels::Kernels::load(&cuda)?;
    let mut rows = Vec::new();
    for (d, h) in [(16, 4), (32, 2), (32, 4), (128, 1)] {
        for b in [1, 3] {
            for t in [1, 3, 7, 17] {
                let n = b * t * h;
                let w = d * h;
                let scores = b * h * t * t;
                for value in [
                    0.,
                    1e-8,
                    -1e-8,
                    0.001,
                    -0.001,
                    1.,
                    -1.,
                    16.,
                    -16.,
                    1e10,
                    -1e10,
                    1e30,
                    -1e30,
                    f32::MAX,
                    -f32::MAX,
                ] {
                    for pattern in 0..10 {
                        let (x, dy) = inputs(d, h, b, t, pattern);
                        let a: Vec<f32> = (0..h)
                            .map(|j| if j % 2 == 0 { value } else { -value })
                            .collect();
                        let reference = reference(d, h, b, t, &x, &a, &dy);
                        let input = cuda.upload(&x)?;
                        let upstream = cuda.upload(&dy)?;
                        let parameters = cuda.upload(&a)?;
                        let table = cuda.upload(&vec![GUARD; 2 * h + 8])?;
                        let output = cuda.upload(&vec![GUARD; b * t * w + 8])?;
                        let prob = cuda.upload(&vec![GUARD; scores + 8])?;
                        let im = cuda.upload(&vec![GUARD; scores + 8])?;
                        let ds = cuda.upload(&vec![GUARD; scores + 8])?;
                        let dx = cuda.upload(&vec![GUARD; x.len() + 8])?;
                        let gd = cuda.upload(&vec![GUARD; b * t * w + 8])?;
                        let dv = cuda.upload(&vec![GUARD; b * t * w + 8])?;
                        let parts = cuda.upload(&vec![GUARD; n + 8])?;
                        let da = cuda.upload(&vec![GUARD; h + 8])?;
                        let packed =
                            Grid::new((n / if d == 16 { 2 } else { 1 }).div_ceil(4) as u32, 128);
                        let correction = Grid::new(n.div_ceil(4) as u32, 128);
                        macro_rules! launch {
                            ($d:literal) => {
                                unsafe {
                                    score.coefficients(
                                        Grid::new(1, 256),
                                        table.ptr(),
                                        parameters.ptr(),
                                        h as u32,
                                    )?;
                                    forward.forward::<$d>(
                                        packed,
                                        output.ptr(),
                                        prob.ptr(),
                                        input.ptr(),
                                        im.ptr(),
                                        table.ptr(),
                                        h as u32,
                                        t as u32,
                                        n as u32,
                                    )?;
                                    ordinary.query_backward::<$d>(
                                        packed,
                                        dx.ptr(),
                                        ds.ptr(),
                                        gd.ptr(),
                                        dv.ptr(),
                                        upstream.ptr(),
                                        prob.ptr(),
                                        input.ptr(),
                                        h as u32,
                                        t as u32,
                                        n as u32,
                                    )?;
                                    base.key_backward::<$d>(
                                        packed,
                                        dx.ptr(),
                                        ds.ptr(),
                                        gd.ptr(),
                                        dv.ptr(),
                                        prob.ptr(),
                                        input.ptr(),
                                        h as u32,
                                        t as u32,
                                        n as u32,
                                    )?;
                                    score.query_correction::<$d>(
                                        correction,
                                        dx.ptr(),
                                        parts.ptr(),
                                        ds.ptr(),
                                        im.ptr(),
                                        input.ptr(),
                                        table.ptr(),
                                        h as u32,
                                        t as u32,
                                        n as u32,
                                    )?;
                                    score.key_correction::<$d>(
                                        correction,
                                        dx.ptr(),
                                        ds.ptr(),
                                        im.ptr(),
                                        input.ptr(),
                                        table.ptr(),
                                        h as u32,
                                        t as u32,
                                        n as u32,
                                    )?;
                                    score.coordinate_gradient(
                                        Grid::new(1, 256),
                                        da.ptr(),
                                        parts.ptr(),
                                        h as u32,
                                        (b * t) as u32,
                                    )?;
                                }
                            };
                        }
                        // SAFETY: each buffer covers the full declared shape plus checked guards.
                        match d {
                            16 => launch!(16),
                            32 => launch!(32),
                            128 => launch!(128),
                            _ => unreachable!(),
                        }
                        let y = output.read()?;
                        let grad = dx.read()?;
                        let parameter_grad = da.read()?;
                        let (ym, yw) = compare(&y[..b * t * w], &reference.y);
                        let (gm, gw) = compare(&grad[..x.len()], &reference.dx);
                        let mut am = 0;
                        let mut aw = 0f64;
                        for (j, &got) in parameter_grad.iter().enumerate().take(h) {
                            let error = (f64::from(got) - reference.da[j]).abs()
                                / (3e-6 + 2e-5 * reference.abs_da[j]);
                            if !error.is_finite() || error > 1. {
                                am += 1;
                            }
                            aw = aw.max(error);
                        }
                        let tv = table.read()?;
                        let pv = prob.read()?;
                        let iv = im.read()?;
                        let sv = ds.read()?;
                        let gv = gd.read()?;
                        let vv = dv.read()?;
                        let pp = parts.read()?;
                        let guards = [
                            &y[b * t * w..],
                            &grad[x.len()..],
                            &parameter_grad[h..],
                            &tv[2 * h..],
                            &pv[scores..],
                            &iv[scores..],
                            &sv[scores..],
                            &gv[b * t * w..],
                            &vv[b * t * w..],
                            &pp[n..],
                        ]
                        .iter()
                        .all(|v| v.iter().all(|&x| x == GUARD));
                        let table_ok = (0..h).all(|j| {
                            let a = f64::from(a[j]);
                            let den = a.hypot(1.);
                            (f64::from(tv[2 * j]) - a / den).abs() < 2e-6
                                && (f64::from(tv[2 * j + 1]) - den.powi(-3)).abs() < 2e-6
                        });
                        let first_exact = t != 1
                            || ((0..b).all(|bb| {
                                (0..w).all(|c| y[bb * w + c] == x[bb * 3 * w + 2 * w + c])
                            }) && parameter_grad[..h].iter().all(|&v| v == 0.)
                                && (0..b).all(|bb| {
                                    grad[bb * 3 * w..bb * 3 * w + 2 * w]
                                        .iter()
                                        .all(|&v| v == 0.)
                                }));
                        let mut zero_exact = true;
                        if value == 0. {
                            macro_rules! baseline {
                                ($d:literal) => {
                                    unsafe {
                                        control.forward::<$d>(
                                            packed,
                                            output.ptr(),
                                            prob.ptr(),
                                            input.ptr(),
                                            h as u32,
                                            t as u32,
                                            n as u32,
                                        )?;
                                    }
                                };
                            }
                            // SAFETY: same checked shape, writing only the comparison outputs.
                            match d {
                                16 => baseline!(16),
                                32 => baseline!(32),
                                128 => baseline!(128),
                                _ => unreachable!(),
                            }
                            zero_exact =
                                same_bits(&y, &output.read()?) && same_bits(&pv, &prob.read()?);
                        }
                        rows.push(json!({"d":d,"heads":h,"batch":b,"length":t,"a":value,"pattern":pattern,"passed":ym+gm+am==0&&guards&&table_ok&&first_exact&&zero_exact,"forward_misses":ym,"qkv_gradient_misses":gm,"coordinate_gradient_misses":am,"forward_error_over_bound":yw,"qkv_error_over_bound":gw,"coordinate_error_over_bound":aw,"guards":guards,"coefficient_checks":table_ok,"one_token_exact":first_exact,"zero_coefficient_forward_exact":zero_exact}));
                    }
                }
            }
        }
    }
    let failed = rows.iter().filter(|r| r["passed"] != true).count();
    record(
        "score-primitives",
        json!({"cases":rows.len(),"passed":rows.len()-failed,"failed":failed,"rows":rows,
          "source_sha256":runner::source_hash(),"ptx_sha256":runner::ptx_hash()}),
    )?;
    assert_eq!(
        failed, 0,
        "retain every failed case in score-primitives.json"
    );
    Ok(())
}

type Direction = (String, Vec<(usize, f64)>);
fn pair_direction(name: String, pair: [u32; 5], imaginary: bool) -> Direction {
    let [ar, bi, br, ai, kind] = pair;
    let terms = if imaginary {
        if kind == 0 {
            vec![(bi as usize, 1.), (br as usize, -1.)]
        } else {
            vec![(bi as usize, 1.)]
        }
    } else if kind == 0 {
        vec![(ar as usize, 1.), (ai as usize, 1.)]
    } else {
        vec![(ar as usize, 1.)]
    };
    (name, terms)
}
#[test]
#[ignore = "requires CUDA; complete score integration and independent full-model AD"]
fn score_model_integration_and_all_changed_gradient_paths() -> Result<()> {
    let tokens = [1, 2, 3, 4, 3, 2];
    let labels = [2, 3, 4, 3, 2, 1];
    let mut records = Vec::new();
    for (w, h, l) in [(64, 4, 1), (64, 4, 2), (128, 1, 1)] {
        let mut parent = config(w, h, l, false);
        parent.max_batch = 2;
        parent.max_length = 3;
        let old = layout::initialize_model(&parent, 17);
        let mut control = Engine::new(parent.clone(), &old)?;
        control.forward(&tokens, &labels, 2, 3)?;
        control.backward(2, 3)?;
        let old_logits = control.logits.read()?;
        let old_gradient = control.gradients.read()?;
        let old_input = control.embedding_gradient_probe(6)?;
        drop(control);
        let mut cfg = config(w, h, l, true);
        cfg.max_batch = 2;
        cfg.max_length = 3;
        let offset = layout::score_offset(&cfg);
        let frequencies = layout::frequency_offset(&cfg);
        let mut parameters = layout::initialize_model(&cfg, 17);
        let mut engine = Engine::new(cfg.clone(), &parameters)?;
        engine.forward(&tokens, &labels, 2, 3)?;
        engine.backward(2, 3)?;
        let y = engine.logits.read()?;
        let g = engine.gradients.read()?;
        let logits = same_bits(&old_logits[..6 * cfg.vocab], &y[..6 * cfg.vocab]);
        let gradients = same_bits(&old_gradient[..offset], &g[..offset])
            && same_bits(&old_gradient[offset..], &g[frequencies..]);
        let input = same_bits(&old_input, &engine.embedding_gradient_probe(6)?);
        engine.backward(2, 3)?;
        let reset = same_bits(&g, &engine.gradients.read()?);
        let nonzero = g[offset..frequencies].iter().any(|&v| v != 0.);
        records.push(
            json!({"kind":"initial_integration","width":w,"heads":h,"layers":l,
            "passed":logits&&gradients&&input&&reset&&nonzero,"logits_equal":logits,
            "existing_gradients_equal":gradients,"embedding_input_equal":input,
            "backward_reset_equal":reset,"new_gradient_nonzero":nonzero}),
        );
        for (i, p) in parameters[layout::phase_offset(&cfg)..offset]
            .iter_mut()
            .enumerate()
        {
            *p = [0.2, -0.4, 0.8][i % 3];
        }
        for (i, p) in parameters[frequencies..].iter_mut().enumerate() {
            *p = 0.005 * ((i % 3) as f32 - 1.);
        }
        let pairs = layout::pairs(&cfg);
        let d = w / h;
        let hidden = layout::hidden(w, &cfg.architecture);
        let mut directions: Vec<Direction> = (offset..frequencies)
            .map(|i| ("score".into(), vec![(i, 1.)]))
            .collect();
        for i in layout::phase_offset(&cfg)..offset {
            directions.push(("residual-phase".into(), vec![(i, 1.)]));
        }
        for i in frequencies..parameters.len() {
            directions.push(("frequency".into(), vec![(i, 1.)]));
        }
        for layer in 0..l {
            let start = cfg.vocab * w + layer * (4 * w * w + 3 * w * hidden);
            for (group, block) in ["Q", "K", "V", "O"].into_iter().zip(0..4) {
                for head in 0..h {
                    let at = start + block * w * w + (head * d) * w + ((head + 1) % h) * d + 1;
                    let pair = *pairs
                        .iter()
                        .find(|p| p[0] as usize == at)
                        .expect("complex owner");
                    directions.push(pair_direction(format!("{group}-real"), pair, false));
                    directions.push(pair_direction(format!("{group}-imaginary"), pair, true));
                }
            }
            for (group, at) in [
                ("FFN-input", start + 4 * w * w),
                ("FFN-output", start + 4 * w * w + 2 * hidden * w),
            ] {
                let pair = *pairs
                    .iter()
                    .find(|p| p[0] as usize == at)
                    .expect("FFN owner");
                directions.push(pair_direction(format!("{group}-real"), pair, false));
                directions.push(pair_direction(format!("{group}-imaginary"), pair, true));
            }
        }
        for head in 0..h {
            let pair = *pairs
                .iter()
                .find(|p| p[0] as usize == w + head * d)
                .expect("embedding owner");
            directions.push(pair_direction("tied-embedding-real".into(), pair, false));
            directions.push(pair_direction(
                "tied-embedding-imaginary".into(),
                pair,
                true,
            ));
        }
        for sublayer in 0..2 * l {
            let router = layout::router_offset(&cfg, sublayer);
            for gate in 0..24 {
                let group = if gate < 4 {
                    "read"
                } else if gate < 8 {
                    "write"
                } else {
                    "skip"
                };
                directions.push((
                    format!("router{sublayer}-{group}"),
                    vec![(router + 96 * w + gate, 1.)],
                ));
            }
            for group in 0..3 {
                directions.push((
                    format!("router{sublayer}-scale"),
                    vec![(router + 96 * w + 24 + group, 1.)],
                ));
                directions.push((
                    format!("router{sublayer}-projection"),
                    vec![(router + 4 * group * 4 * w + group, 1.)],
                ));
            }
        }
        let mut zero_logits = Vec::new();
        for state in 0..2 {
            for (i, a) in parameters[offset..frequencies].iter_mut().enumerate() {
                *a = if state == 0 {
                    0.
                } else {
                    [0.4, -0.7, 1.3][i % 3]
                };
            }
            engine.parameters.write(&parameters)?;
            engine.forward(&tokens, &labels, 2, 3)?;
            engine.backward(2, 3)?;
            let native = engine.gradients.read()?;
            let logits = engine.logits.read()?;
            if state == 0 {
                zero_logits = logits.clone();
            }
            let effect = state == 0 || !same_bits(&zero_logits, &logits);
            let mut expected = Vec::new();
            let mut differences = Vec::new();
            for (group, terms) in &directions {
                let mut p: Vec<Jet> = parameters
                    .iter()
                    .map(|&v| Jet::from(f64::from(v)))
                    .collect();
                for &(i, d) in terms {
                    p[i].d = d;
                }
                let objective = |p: &[Jet]| {
                    oracle::cross_entropy(
                        &oracle::main_model(&cfg, p, &tokens, 2, 3),
                        &labels,
                        cfg.vocab,
                    )
                };
                let want = objective(&p).d;
                let eps = 1e-5;
                for &(i, d) in terms {
                    p[i].v += eps * d;
                }
                let plus = objective(&p).v;
                for &(i, d) in terms {
                    p[i].v -= 2. * eps * d;
                }
                let minus = objective(&p).v;
                let fd = (plus - minus) / (2. * eps);
                let bound = 2e-8 + 2e-5 * want.abs();
                expected.push(want);
                differences.push(json!({"path":group,"terms":terms,"AD":want,"finite_difference":fd,"bound":bound,"passed":(fd-want).abs()<=bound}));
            }
            let mut norms = std::collections::BTreeMap::<String, f64>::new();
            for ((group, _), &want) in directions.iter().zip(&expected) {
                *norms.entry(group.clone()).or_default() += want * want;
            }
            let comparisons:Vec<_>=directions.iter().zip(&expected).map(|((group,terms),&want)| {
                let have: f64=terms.iter().map(|&(i,d)|f64::from(native[i])*d).sum();
                let bound=5e-4*norms[group].sqrt()+1e-12; let error=(have-want).abs();
                json!({"path":group,"terms":terms,"native":have,"reference":want,"bound":bound,"error":error,"passed":error.is_finite()&&error<=bound})
            }).collect();
            let mut changed = tokens;
            changed[2] = 9;
            changed[5] = 10;
            engine.forward(&changed, &labels, 2, 3)?;
            let future = engine.logits.read()?;
            let causal = [0, 1, 3, 4].iter().all(|&i| {
                same_bits(
                    &logits[i * cfg.vocab..(i + 1) * cfg.vocab],
                    &future[i * cfg.vocab..(i + 1) * cfg.vocab],
                )
            });
            let fd = differences.iter().all(|r| r["passed"] == true);
            records.push(json!({"kind":"full_model_AD","width":w,"heads":h,"layers":l,"score_state":state,
                "passed":effect&&causal&&fd,"strict_AD_passed":comparisons.iter().all(|r|r["passed"]==true),
                "directions":comparisons,"finite_differences":differences,"causal":causal,"nonzero_score_effect":effect}));
        }
    }
    let passed = records.iter().all(|r| r["passed"] == true);
    let misses: usize = records
        .iter()
        .filter_map(|r| r["directions"].as_array())
        .map(|v| v.iter().filter(|r| r["passed"] != true).count())
        .sum();
    record(
        "score-full-model",
        json!({"implementation_passed":passed,"strict_AD_misses":misses,
        "cases":records,"source_sha256":runner::source_hash(),"ptx_sha256":runner::ptx_hash(),"dtype":crate::precision::TAG} ),
    )?;
    assert!(
        passed,
        "retain all implementation and strict AD failures separately"
    );
    Ok(())
}
#[test]
#[ignore = "requires CUDA"]
fn score_optimizer_and_odd_coordinate_match_independent_fp64() -> Result<()> {
    use crate::schedule::Schedule;
    let mut records = Vec::new();
    for (width, heads, layers, budget) in [(64, 4, 2, 20000), (128, 1, 1, 100000)] {
        let mut cfg = config(width, heads, layers, true);
        let schedule = Schedule::new(budget)?;
        cfg.warmup = schedule.warmup;
        let base = layout::base_count(&cfg);
        let phase = layout::phase_offset(&cfg);
        let score = layout::score_offset(&cfg);
        let frequency = layout::frequency_offset(&cfg);
        let mut initial = layout::initialize_model(&cfg, 19);
        for (i, p) in initial[phase..].iter_mut().enumerate() {
            *p = 0.04 * (i as f32 - 2.);
        }
        let n = initial.len();
        let mut engine = Engine::new(cfg.clone(), &initial)?;
        let mut optimizer = Optimizer::new(&engine, budget)?;
        let before_guard = optimizer.save(&engine)?;
        assert!(engine.update(0).is_err());
        assert!(same_bits(&before_guard, &optimizer.save(&engine)?));
        for (base_scale, phase_scale, score_scale, frequency_scale) in [
            (0., 0., 0., 0.),
            (1e-8, 1e-8, 1e-8, 1e-8),
            (1e8, 1e-8, 1e-8, 1e-8),
            (1e-8, 1e8, 1e-8, 1e-8),
            (1e-8, 1e-8, 1e8, 1e-8),
            (1e-8, 1e-8, 1e-8, 1e8),
            (1e8, 1e8, 1e8, 1e8),
        ] {
            for step in [
                0,
                1,
                schedule.warmup - 1,
                schedule.warmup,
                schedule.cooldown_start - 1,
                schedule.cooldown_start,
                (schedule.cooldown_start + budget) / 2,
                budget - 1,
            ] {
                let gradient: Vec<f32> = (0..n)
                    .map(|i| {
                        let scale = if i < phase {
                            base_scale
                        } else if i < score {
                            phase_scale
                        } else if i < frequency {
                            score_scale
                        } else {
                            frequency_scale
                        };
                        ((i * 7 % 29) as f32 - 14.) * scale / 17.
                    })
                    .collect();
                let mut m = vec![0.; n];
                let mut v = vec![0.; n];
                if step > 1 {
                    for i in base..n {
                        m[i] = ((i - base) % 3) as f32 * 0.0003 - 0.0003;
                        v[i] = 0.00002 + ((i - base) % 2) as f32 * 0.00001;
                    }
                }
                // Independent-coordinate contraction from matrix geometry,
                // without the optimizer's packed ownership map or GPU reductions.
                let mut coordinates = Vec::new();
                for [a, b, nb, aa, kind] in layout::pairs(&cfg) {
                    let mut ga = f64::from(gradient[a as usize]);
                    let mut gb = f64::from(gradient[b as usize]);
                    if kind == 0 {
                        ga += f64::from(gradient[aa as usize]);
                        gb -= f64::from(gradient[nb as usize]);
                    }
                    coordinates.push((a as usize, ga));
                    coordinates.push((b as usize, gb));
                }
                coordinates.extend((base..frequency).map(|i| (i, f64::from(gradient[i]))));
                let base_norm = coordinates
                    .iter()
                    .map(|(_, g)| g * g)
                    .sum::<f64>()
                    .sqrt()
                    .max(1.);
                let frequency_norm = gradient[frequency..]
                    .iter()
                    .map(|&g| f64::from(g).powi(2))
                    .sum::<f64>()
                    .sqrt()
                    .max(1.);
                coordinates.extend((frequency..n).map(|i| (i, f64::from(gradient[i]))));
                engine.parameters.write(&initial)?;
                engine.first_moment.write(&m)?;
                engine.second_moment.write(&v)?;
                engine.gradients.write(&gradient)?;
                optimizer.step(&engine, step)?;
                let actual = optimizer.save(&engine)?; // verifies all expanded ties/state
                let rate = f64::from(schedule.rate(step));
                let t = (step + 1) as f64;
                let bc1 = 1. - 0.9f64.powf(t);
                let bc2 = 1. - 0.95f64.powf(t);
                let mut misses = 0;
                let mut max_ratio = 0f64;
                let mut worst = Value::Null;
                for &(i, unscaled) in &coordinates {
                    let is_frequency = i >= frequency;
                    let g = unscaled
                        / if is_frequency {
                            frequency_norm
                        } else {
                            base_norm
                        };
                    let first = 0.9 * f64::from(m[i]) + 0.1 * g;
                    let second = 0.95 * f64::from(v[i]) + 0.05 * g * g;
                    let decay = if is_frequency { 0. } else { 0.1 };
                    let parameter = f64::from(initial[i]) * (1. - rate * decay)
                        - rate * (first / bc1) / ((second / bc2).sqrt() + 1e-8);
                    let cancellation =
                        4. * f64::from(f32::EPSILON) * (f64::from(m[i]).abs() + 0.1 * g.abs());
                    for (part, expected, floor) in [
                        (0, parameter, 2e-8),
                        (1, first, 1e-12 + cancellation),
                        (2, second, 1e-18),
                    ] {
                        let have = f64::from(actual[part * n + i]);
                        let error = (have - expected).abs();
                        let bound = floor + 4e-6 * expected.abs();
                        if !error.is_finite() || error > bound {
                            misses += 1;
                        }
                        if error / bound > max_ratio || !error.is_finite() {
                            max_ratio = error / bound;
                            worst = json!({"parameter":i,"state_part":part,"native":have,
                                "reference":expected,"error":error,"bound":bound});
                        }
                    }
                    if is_frequency && frequency_scale == 0. && step <= 1 {
                        assert_eq!(actual[i].to_bits(), initial[i].to_bits(), "frequency decay");
                    }
                }
                records.push(
                    json!({"width":width,"heads":heads,"layers":layers,"budget":budget,
                    "step":step,"base_scale":base_scale,"phase_scale":phase_scale,
                    "score_scale":score_scale,"frequency_scale":frequency_scale,"base_clip_denominator":base_norm,
                    "frequency_clip_denominator":frequency_norm,"comparisons":3*coordinates.len(),
                    "phase_parameters":score-phase,"score_parameters":frequency-score,
                    "singleton_owner":(frequency-base)%2==1,"passed":misses==0,"misses":misses,
                    "max_error_over_bound":max_ratio,"worst":worst}),
                );
            }
        }
    }
    let misses: usize = records
        .iter()
        .map(|r| r["misses"].as_u64().unwrap() as usize)
        .sum();
    assert_eq!(records.len(), 112);
    record(
        "score-optimizer",
        json!({"cases":records,"misses":misses,
        "source_sha256":runner::source_hash(),"ptx_sha256":runner::ptx_hash()}),
    )?;
    assert_eq!(
        misses, 0,
        "phase/group optimizer misses retained in score-optimizer.json"
    );
    eprintln!("112 independent base/phase/score/frequency optimizer cases passed");
    Ok(())
}
