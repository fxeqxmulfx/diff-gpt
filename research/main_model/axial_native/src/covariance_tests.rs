//! Independent FP64 automatic differentiation; no manual GPU adjoint reused.
use crate::{
    Config, Engine, Result, layout,
    oracle::{self, Jet},
    runner,
};
use kernite_cuda::{Cuda, Grid};
use serde_json::{Value, json};
use std::{fs, io::Write, path::Path};
const GUARD: f32 = 12345.;
fn record(name: &str, value: &Value) -> Result<()> {
    if let Some(root) = std::env::var_os("COVARIANCE_VALIDATION_OUTPUT") {
        let mut f = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(Path::new(&root).join(format!("{name}.json")))?;
        f.write_all(&serde_json::to_vec_pretty(value)?)?;
        f.sync_all()?;
    }
    Ok(())
}
fn config(w: usize, h: usize, l: usize) -> Config {
    Config {
        width: w,
        heads: h,
        layers: l,
        vocab: 11,
        max_batch: 2,
        max_length: 3,
        normalized: true,
        ssmax: true,
        gain: h as f32 / w as f32,
        lr: 0.001,
        decay: 0.1,
        warmup: 400,
        architecture: "complex-covariance-lerope".into(),
        phase_optimizer: false,
    }
}
fn bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}
fn pattern(w: usize, h: usize, kind: usize, scale: f32, row: usize) -> Vec<f32> {
    let d = w / h;
    let mut out = vec![0.; w];
    for j in 0..w / 2 {
        let r = (j / (d / 2)) * d + j % (d / 2);
        let z = ((j * 17 + row * 3) % 29) as f64 / 14. - 1.;
        let q = ((j * 11 + row * 7 + 3) % 31) as f64 / 15. - 1.;
        let (a, b) = match kind {
            0 => {
                if j % 2 == 0 {
                    (1., 0.)
                } else {
                    (0., 1.)
                }
            }
            1 => (z, 0.75 * z),
            2 => (z, 0.75 * z + 1e-5 * q),
            3 => (z, 0.),
            4 => (z, 1e-5 * q),
            5 => {
                if j == row % (w / 2) {
                    (1., -0.5)
                } else {
                    (0., 0.)
                }
            }
            6 => (z, q),
            _ => (0.7, -0.3),
        };
        out[r] = (a * f64::from(scale)) as f32;
        out[r + d / 2] = (b * f64::from(scale)) as f32;
    }
    out
}
fn reference(x: &[f32], g: &[f32], h: usize) -> (Vec<f64>, Vec<f64>) {
    let mut a: Vec<_> = x.iter().map(|&v| Jet::from(f64::from(v))).collect();
    let y = oracle::covariance(&a, h).iter().map(|v| v.v).collect();
    let mut dx = Vec::with_capacity(x.len());
    for i in 0..x.len() {
        a[i].d = 1.;
        let v = oracle::covariance(&a, h)
            .iter()
            .zip(g)
            .map(|(v, &g)| v.d * f64::from(g))
            .sum();
        dx.push(v);
        a[i].d = 0.;
    }
    (y, dx)
}
#[test]
fn covariance_layout_preserves_all_initial_parameters_and_owners() -> Result<()> {
    for (w, h, l) in [(64, 4, 1), (128, 4, 6), (128, 1, 1), (256, 2, 3)] {
        let c = config(w, h, l);
        let mut p = c.clone();
        p.architecture = "complex-phase-score-lerope".into();
        for seed in 0..3 {
            assert!(bits(
                &layout::initialize_model(&c, seed),
                &layout::initialize_model(&p, seed)
            ));
        }
        assert_eq!(layout::total_count(&c), layout::total_count(&p));
        assert_eq!(
            crate::optimizer::Optimizer::owners(&c)?,
            crate::optimizer::Optimizer::owners(&p)?
        );
        assert_eq!(
            crate::optimizer::Optimizer::frequency_owners(&c),
            crate::optimizer::Optimizer::frequency_owners(&p)
        );
    }
    Ok(())
}
#[test]
fn covariance_cpu_ad_matches_finite_differences() -> Result<()> {
    let mut rows = Vec::new();
    for kind in 0..8 {
        for scale in [1e-5f32, 0.01, 1., 100.] {
            let x = pattern(16, 2, kind, scale, 0);
            let mut a: Vec<_> = x
                .iter()
                .enumerate()
                .map(|(i, &v)| Jet {
                    v: f64::from(v),
                    d: ((i * 11 % 17) as f64 - 8.) / 8.,
                })
                .collect();
            let f = |a: &[Jet]| -> Jet {
                oracle::covariance(a, 2)
                    .iter()
                    .enumerate()
                    .fold(Jet::default(), |s, (i, &v)| {
                        s + v * ((i * 7 % 13) as f64 - 6.) / 6.
                    })
            };
            let want = f(&a).d;
            let step = 1e-6 * f64::from(scale).max(1e-4);
            for v in &mut a {
                v.v += step * v.d;
            }
            let plus = f(&a).v;
            for v in &mut a {
                v.v -= 2. * step * v.d;
            }
            let minus = f(&a).v;
            let fd = (plus - minus) / (2. * step);
            let bound = 2e-5 * (1. / f64::from(scale).max(1e-4) + want.abs());
            rows.push(json!({"kind":kind,"scale":scale,"ad":want,"fd":fd,"bound":bound,"passed":(fd-want).abs()<=bound}));
        }
    }
    record(
        "cpu-finite-differences",
        &json!({"cases":rows.len(),"rows":rows}),
    )?;
    assert!(rows.iter().all(|r| r["passed"] == true));
    Ok(())
}
#[test]
#[ignore = "requires CUDA; complete normalization forward/adjoint stress"]
fn covariance_primitives_against_independent_ad() -> Result<()> {
    let cuda = Cuda::new()?;
    let kernel = crate::covariance::kernels::Kernels::load(&cuda)?;
    let mut records = Vec::new();
    for (w, h) in [
        (16, 1),
        (32, 1),
        (64, 2),
        (64, 4),
        (128, 1),
        (128, 4),
        (256, 2),
        (256, 8),
    ] {
        for kind in 0..8 {
            for scale in [
                0f32,
                1e-35,
                1e-8,
                0.00034526698,
                0.01,
                1.,
                1e6,
                1e18,
                1e30,
                f32::MAX * 0.5,
            ] {
                let rows = 5;
                let x: Vec<_> = (0..rows)
                    .flat_map(|r| pattern(w, h, kind, scale, r))
                    .collect();
                let g: Vec<_> = (0..x.len())
                    .map(|i| ((i * 13 % 23) as f32 - 11.) / 11.)
                    .collect();
                let input = cuda.upload(&x)?;
                let grad = cuda.upload(&g)?;
                let out = cuda.upload(&vec![GUARD; x.len() + 32])?;
                let dx = cuda.upload(&vec![GUARD; x.len() + 32])?;
                macro_rules! launch {
                    ($w:literal) => {
                        unsafe {
                            kernel.normalize::<$w, false>(
                                Grid::new(2, 128),
                                out.ptr(),
                                input.ptr(),
                                grad.ptr(),
                                rows as u32,
                                h as u32,
                                0,
                            )?;
                            kernel.normalize::<$w, true>(
                                Grid::new(2, 128),
                                dx.ptr(),
                                input.ptr(),
                                grad.ptr(),
                                rows as u32,
                                h as u32,
                                0,
                            )?;
                        }
                    };
                }
                match w {
                    16 => launch!(16),
                    32 => launch!(32),
                    64 => launch!(64),
                    128 => launch!(128),
                    256 => launch!(256),
                    _ => unreachable!(),
                }
                cuda.sync()?;
                let y = out.read()?;
                let actual = dx.read()?;
                let mut fm = 0;
                let mut gm = 0;
                let mut fw = 0f64;
                let mut gw = 0f64;
                let mut bounds = true;
                for row in 0..rows {
                    let at = row * w;
                    let (fy, fg) = reference(&x[at..at + w], &g[at..at + w], h);
                    let norm = fg.iter().map(|v| v * v).sum::<f64>().sqrt();
                    let gb = 2e-5 / f64::from(scale).abs().max(f64::from(f32::EPSILON).sqrt())
                        + 2e-5 * norm;
                    for j in 0..w {
                        let fe = (f64::from(y[at + j]) - fy[j]).abs();
                        let ge = (f64::from(actual[at + j]) - fg[j]).abs();
                        let fb = 3e-5 + 5e-5 * fy[j].abs();
                        fm += usize::from(!fe.is_finite() || fe > fb);
                        gm += usize::from(!ge.is_finite() || ge > gb);
                        fw = fw.max(fe / fb);
                        gw = gw.max(ge / gb);
                    }
                    let energy = y[at..at + w]
                        .iter()
                        .map(|&v| f64::from(v).powi(2))
                        .sum::<f64>();
                    bounds &= energy.is_finite() && energy <= w as f64 * (1. + 5e-5);
                }
                let guards = y[x.len()..]
                    .iter()
                    .chain(&actual[x.len()..])
                    .all(|&v| v == GUARD);
                records.push(json!({"width":w,"heads":h,"kind":kind,"scale":scale,"rows":rows,"forward_misses":fm,"gradient_misses":gm,"forward_error_over_bound":fw,"gradient_error_over_bound":gw,"energy_bound":bounds,"guards":guards,"passed":fm+gm==0&&bounds&&guards}));
            }
        }
    }
    // Q/K are independent rows; V bypasses normalization in both directions.
    for w in [16, 32, 128] {
        let h = 3;
        let rows = 18;
        let x: Vec<_> = (0..rows).flat_map(|r| pattern(w, 1, 6, 0.1, r)).collect();
        let g: Vec<_> = x.iter().map(|v| v * 0.7 + 0.1).collect();
        let input = cuda.upload(&x)?;
        let grad = cuda.upload(&g)?;
        let out = cuda.upload(&vec![GUARD; x.len() + 32])?;
        let dx = cuda.upload(&vec![GUARD; x.len() + 32])?;
        macro_rules! launch {
            ($w:literal) => {
                unsafe {
                    kernel.normalize::<$w, false>(
                        Grid::new(5, 128),
                        out.ptr(),
                        input.ptr(),
                        grad.ptr(),
                        rows as u32,
                        1,
                        h as u32,
                    )?;
                    kernel.normalize::<$w, true>(
                        Grid::new(5, 128),
                        dx.ptr(),
                        input.ptr(),
                        grad.ptr(),
                        rows as u32,
                        1,
                        h as u32,
                    )?;
                }
            };
        }
        match w {
            16 => launch!(16),
            32 => launch!(32),
            128 => launch!(128),
            _ => unreachable!(),
        }
        cuda.sync()?;
        let y = out.read()?;
        let dg = dx.read()?;
        let mut pass = true;
        for row in 0..rows {
            let at = row * w;
            if row / h % 3 == 2 {
                pass &=
                    bits(&y[at..at + w], &x[at..at + w]) && bits(&dg[at..at + w], &g[at..at + w]);
            } else {
                let (a, b) = reference(&x[at..at + w], &g[at..at + w], 1);
                pass &= (0..w).all(|j| {
                    (f64::from(y[at + j]) - a[j]).abs() < 1e-4
                        && (f64::from(dg[at + j]) - b[j]).abs() < 1e-3
                });
            }
        }
        pass &= y[x.len()..]
            .iter()
            .chain(&dg[x.len()..])
            .all(|&v| v == GUARD);
        records.push(json!({"qkv_width":w,"passed":pass}));
    }
    let failed = records.iter().filter(|r| r["passed"] != true).count();
    record(
        "primitives",
        &json!({"cases":records.len(),"failed":failed,"passed":records.len()-failed,"rows":records,"source_sha256":runner::source_hash(),"ptx_sha256":runner::ptx_hash()}),
    )?;
    assert_eq!(failed, 0, "retain every failure in primitives.json");
    Ok(())
}

type Direction = (String, Vec<(usize, f64)>);

#[test]
#[ignore = "requires CUDA; phase equivariance of forward and adjoint"]
fn covariance_common_phase_equivariance() -> Result<()> {
    let cuda = Cuda::new()?;
    let kernel = crate::covariance::kernels::Kernels::load(&cuda)?;
    let mut records = Vec::new();
    for (w, h) in [
        (16, 1),
        (32, 1),
        (64, 2),
        (64, 4),
        (128, 1),
        (128, 4),
        (256, 2),
        (256, 8),
    ] {
        for scale in [1e-8f32, 1., 1e18] {
            for angle in [
                0.2f64,
                0.7,
                std::f64::consts::FRAC_PI_2,
                std::f64::consts::PI,
            ] {
                let (si, co) = angle.sin_cos();
                let x = pattern(w, h, 2, scale, 0);
                let g = pattern(w, h, 6, 1., 2);
                let rotate = |v: &[f32]| {
                    let mut y = v.to_vec();
                    let d = w / h;
                    for head in 0..h {
                        for j in 0..d / 2 {
                            let r = head * d + j;
                            let i = r + d / 2;
                            y[r] = (co * f64::from(v[r]) - si * f64::from(v[i])) as f32;
                            y[i] = (si * f64::from(v[r]) + co * f64::from(v[i])) as f32;
                        }
                    }
                    y
                };
                let input = cuda.upload(&[x.clone(), rotate(&x)].concat())?;
                let grad = cuda.upload(&[g.clone(), rotate(&g)].concat())?;
                let out = cuda.alloc::<f32>(2 * w)?;
                let dx = cuda.alloc::<f32>(2 * w)?;
                macro_rules! launch {
                    ($w:literal) => {
                        unsafe {
                            kernel.normalize::<$w, false>(
                                Grid::new(1, 128),
                                out.ptr(),
                                input.ptr(),
                                grad.ptr(),
                                2,
                                h as u32,
                                0,
                            )?;
                            kernel.normalize::<$w, true>(
                                Grid::new(1, 128),
                                dx.ptr(),
                                input.ptr(),
                                grad.ptr(),
                                2,
                                h as u32,
                                0,
                            )?;
                        }
                    };
                }
                match w {
                    16 => launch!(16),
                    32 => launch!(32),
                    64 => launch!(64),
                    128 => launch!(128),
                    256 => launch!(256),
                    _ => unreachable!(),
                }
                cuda.sync()?;
                let y = out.read()?;
                let dg = dx.read()?;
                let ey = rotate(&y[..w]);
                let eg = rotate(&dg[..w]);
                let gn = eg.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>().sqrt();
                let gb = 2e-5 / f64::from(scale).max(f64::from(f32::EPSILON).sqrt()) + 2e-5 * gn;
                let yf = y[w..]
                    .iter()
                    .zip(&ey)
                    .map(|(&a, &b)| (f64::from(a) - f64::from(b)).abs())
                    .fold(0., f64::max);
                let gf = dg[w..]
                    .iter()
                    .zip(&eg)
                    .map(|(&a, &b)| (f64::from(a) - f64::from(b)).abs())
                    .fold(0., f64::max);
                records.push(json!({"width":w,"heads":h,"scale":scale,"angle":angle,
                    "forward_max_error":yf,"gradient_max_error":gf,"gradient_bound":gb,
                    "passed":yf.is_finite()&&gf.is_finite()&&yf<1e-4&&gf<gb}));
            }
        }
    }
    let failed = records.iter().filter(|r| r["passed"] != true).count();
    record(
        "phase-equivariance",
        &json!({"cases":records.len(),"failed":failed,"rows":records}),
    )?;
    assert_eq!(failed, 0);
    Ok(())
}

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
fn covariance_model_integration_and_all_gradient_paths() -> Result<()> {
    let tokens = [1, 2, 3, 4, 3, 2];
    let labels = [2, 3, 4, 3, 2, 1];
    let mut records = Vec::new();
    for (w, h, l) in [(64, 4, 1), (64, 4, 2), (128, 1, 1)] {
        let cfg = config(w, h, l);
        let offset = layout::score_offset(&cfg);
        let frequencies = layout::frequency_offset(&cfg);
        let mut parameters = layout::initialize_model(&cfg, 17);
        let mut engine = Engine::new(cfg.clone(), &parameters)?;
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
            let finite = native.iter().chain(&logits).all(|v| v.is_finite());
            engine.backward(2, 3)?;
            let reset = bits(&native, &engine.gradients.read()?);
            let scalar_parameters: Vec<_> = parameters
                .iter()
                .map(|&v| Jet::from(f64::from(v)))
                .collect();
            let scalar_logits = oracle::main_model(&cfg, &scalar_parameters, &tokens, 2, 3);
            let logit_norm = scalar_logits.iter().map(|v| v.v * v.v).sum::<f64>().sqrt();
            let logit_error = logits[..6 * cfg.vocab]
                .iter()
                .zip(&scalar_logits)
                .map(|(&a, b)| (f64::from(a) - b.v).powi(2))
                .sum::<f64>()
                .sqrt();
            let logit_bound = 1e-3 * logit_norm + 1e-6;
            let forward_ok = logit_error <= logit_bound;
            if state == 0 {
                zero_logits = logits.clone();
            }
            let effect = state == 0 || !bits(&zero_logits, &logits);
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
                bits(
                    &logits[i * cfg.vocab..(i + 1) * cfg.vocab],
                    &future[i * cfg.vocab..(i + 1) * cfg.vocab],
                )
            });
            let fd = differences.iter().all(|r| r["passed"] == true);
            records.push(json!({"kind":"full_model_AD","width":w,"heads":h,"layers":l,"score_state":state,
                "passed":effect&&causal&&fd&&finite&&reset&&forward_ok,"strict_AD_passed":comparisons.iter().all(|r|r["passed"]==true),
                "all_finite":finite,"backward_reset_equal":reset,"forward_passed":forward_ok,"forward_l2_error":logit_error,"forward_l2_bound":logit_bound,
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
        "full-model",
        &json!({"implementation_passed":passed,"strict_AD_misses":misses,
        "cases":records,"source_sha256":runner::source_hash(),"ptx_sha256":runner::ptx_hash(),"dtype":crate::precision::TAG} ),
    )?;
    assert!(
        passed,
        "retain all implementation and strict AD failures separately"
    );
    Ok(())
}
