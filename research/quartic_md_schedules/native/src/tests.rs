//! Independent complex FP64 reference and native GPU optimizer validation.
use crate::schedule::{Family, Schedule};
use crate::{
    Result, arithmetic_data,
    engine::Engine,
    layout,
    optimizer::{self, Matrix, Ops, Optimizer},
    runner,
};
use kernite_cuda::Cuda;
use serde_json::json;
use std::fs;

#[derive(Clone, Copy, Debug, Default)]
struct Z(f64, f64);
impl Z {
    fn add(self, b: Self) -> Self {
        Self(self.0 + b.0, self.1 + b.1)
    }
    fn sub(self, b: Self) -> Self {
        Self(self.0 - b.0, self.1 - b.1)
    }
    fn mul(self, b: Self) -> Self {
        Self(self.0 * b.0 - self.1 * b.1, self.0 * b.1 + self.1 * b.0)
    }
    fn scale(self, b: f64) -> Self {
        Self(self.0 * b, self.1 * b)
    }
    fn conj(self) -> Self {
        Self(self.0, -self.1)
    }
    fn sq(self) -> f64 {
        self.0 * self.0 + self.1 * self.1
    }
}
fn adjoint(a: &[Z], m: usize, n: usize) -> Vec<Z> {
    (0..m * n).map(|i| a[(i % m) * n + i / m].conj()).collect()
}
fn mm(a: &[Z], b: &[Z], m: usize, n: usize, k: usize) -> Vec<Z> {
    let mut out = vec![Z::default(); m * n];
    for i in 0..m {
        for l in 0..k {
            for j in 0..n {
                out[i * n + j] = out[i * n + j].add(a[i * k + l].mul(b[l * n + j]));
            }
        }
    }
    out
}
fn polar(input: &[Z], m: usize, n: usize) -> Vec<Z> {
    let (r, c) = (m.min(n), m.max(n));
    let mut x = if m > n {
        adjoint(input, m, n)
    } else {
        input.to_vec()
    };
    let norm = x.iter().map(|z| z.sq()).sum::<f64>().sqrt();
    if norm == 0. {
        return input.to_vec();
    }
    for z in &mut x {
        *z = z.scale(1. / norm);
    }
    for _ in 0..5 {
        let a = mm(&x, &adjoint(&x, r, c), r, r, c);
        let a2 = mm(&a, &a, r, r, r);
        let b: Vec<_> = a
            .iter()
            .zip(a2)
            .map(|(a, b)| a.scale(-4.775).add(b.scale(2.0315)))
            .collect();
        let bx = mm(&b, &x, r, c, r);
        for (x, b) in x.iter_mut().zip(bx) {
            *x = x.scale(3.4445).add(b);
        }
    }
    if m > n { adjoint(&x, r, c) } else { x }
}
fn close(gpu: &[f32], reference: &[f64], abs: f64, rel: f64) {
    assert_eq!(gpu.len(), reference.len());
    for (i, (&a, &b)) in gpu.iter().zip(reference).enumerate() {
        assert!(
            (a as f64 - b).abs() <= abs + rel * b.abs(),
            "index{i}: gpu{a} reference{b}, abs error{}",
            (a as f64 - b).abs()
        );
    }
}
fn components(z: &[Z]) -> Vec<f64> {
    z.iter().flat_map(|z| [z.0, z.1]).collect()
}
fn softplus(x: f64) -> f64 {
    x.max(0.) + (-x.abs()).exp().ln_1p()
}
fn sigmoid(x: f64) -> f64 {
    1. / (1. + (-x).exp())
}

#[test]
fn gain_chain_rule_matches_real_directional_finite_differences() {
    let (m, n) = (3, 2);
    let d: Vec<_> = (0..m * n)
        .map(|i| Z((i + 1) as f64 / 7., (i as f64 - 2.) / 9.))
        .collect();
    let g: Vec<_> = (0..m * n)
        .map(|i| Z((i as f64).sin(), (i as f64).cos()))
        .collect();
    let raw = vec![-0.7, 0.3, 1.2, -0.2, 0.9];
    let objective = |raw: &[f64]| -> f64 {
        (0..m * n)
            .map(|i| g[i].conj().mul(d[i]).0 * softplus(raw[i / n]) * softplus(raw[m + i % n]))
            .sum()
    };
    for q in 0..m + n {
        let analytic = if q < m {
            (0..n)
                .map(|j| g[q * n + j].conj().mul(d[q * n + j]).0 * softplus(raw[m + j]))
                .sum::<f64>()
        } else {
            (0..m)
                .map(|i| g[i * n + q - m].conj().mul(d[i * n + q - m]).0 * softplus(raw[i]))
                .sum::<f64>()
        } * sigmoid(raw[q]);
        let mut plus = raw.clone();
        let mut minus = raw.clone();
        plus[q] += 1e-6;
        minus[q] -= 1e-6;
        assert!((analytic - (objective(&plus) - objective(&minus)) / 2e-6).abs() < 1e-8);
    }
}
#[test]
fn schedule_data_and_epoch_tail_are_frozen() -> Result<()> {
    let linear = Schedule::parse("muonmd-linear")?;
    assert_eq!(linear.rate(0.01, 0), 0.01f32);
    assert_eq!(linear.rate(0.01, 99999), 1e-8f32);
    let d = arithmetic_data::arithmetic(&runner::row())?;
    assert_eq!((d.train.len(), d.validation.len()), (4704, 4705));
    let (o, edges) = runner::order(4704, 39, 1);
    assert_eq!(edges[19], 4704);
    assert_eq!(edges[19] - edges[18], 96);
    for e in 0..2 {
        let mut epoch = o[edges[e * 19]..edges[(e + 1) * 19]].to_vec();
        epoch.sort();
        assert_eq!(epoch, (0..4704).collect::<Vec<_>>());
    }
    assert_eq!((o, edges), runner::order(4704, 39, 1));
    let mut all = d.train;
    all.extend(d.validation);
    all.sort();
    all.dedup();
    assert_eq!(all.len(), 97 * 97);
    for [a, b, y] in all {
        assert_eq!(y, ((a + b).pow(4)) % 97);
    }
    Ok(())
}
#[test]
fn complex_polar_handles_zero_rank_deficiency_and_phase() -> Result<()> {
    let cuda = Cuda::new()?;
    let ops = Ops::new(&cuda, 2 * 192 * 64)?;
    for (m, n, kind) in [
        (3, 2, 0),
        (2, 3, 0),
        (3, 3, 1),
        (2, 5, 2),
        (64, 171, 0),
        (192, 64, 0),
    ] {
        let z: Vec<_> = (0..m * n)
            .map(|i| match kind {
                1 => Z((i / n + 1) as f64 * (i % n + 1) as f64, 0.),
                2 => Z::default(),
                _ => Z(((i * 17 + 3) as f64).sin(), ((i * 11 + 5) as f64).cos()),
            })
            .collect();
        let mut mat = Matrix::new(&cuda, m, n, 0, 1.)?;
        let float: Vec<_> = components(&z).iter().map(|v| *v as f32).collect();
        mat.nu.write(&float)?;
        mat.polar(&ops)?;
        let z: Vec<_> = float
            .as_chunks::<2>()
            .0
            .iter()
            .map(|v| Z(v[0] as f64, v[1] as f64))
            .collect();
        let expected = polar(&z, m, n);
        let x = mat.x.read()?;
        let (r, c) = (m.min(n), m.max(n));
        let mut actual = Vec::new();
        for i in 0..m {
            for j in 0..n {
                let (a, b) = if m > n { (j, i) } else { (i, j) };
                actual.extend([
                    x[a * 2 * c + b],
                    x[(a + r) * 2 * c + b] * if m > n { -1. } else { 1. },
                ]);
            }
        }
        close(&actual, &components(&expected), 2e-5, 3e-4);
        let phase = Z(0.71f64.cos(), 0.71f64.sin());
        let rotated: Vec<_> = z.iter().map(|z| z.mul(phase)).collect();
        let reference = polar(&rotated, m, n);
        for (a, b) in reference.iter().zip(&expected) {
            assert!(a.sub(b.mul(phase)).sq() < 1e-23);
        }
    }
    Ok(())
}

#[test]
fn positive_gain_materialization_is_stable_at_small_and_large_raw_values() -> Result<()> {
    use kernite_cuda::Grid;
    let cuda = Cuda::new()?;
    let k = crate::md_ops::kernels::Kernels::load(&cuda)?;
    let values = [-30f32, -15., -2., 0., 2., 30.];
    let raw = cuda.upload(&values)?;
    let m = cuda.alloc::<f32>(6)?;
    let v = cuda.alloc::<f32>(6)?;
    let next = cuda.alloc::<f32>(6)?;
    let old = cuda.upload(&[1f32; 6])?;
    let signals = cuda.alloc::<f32>(9)?;
    // Zero gain gradient and LR preserve raw gains, exposing only materialization.
    unsafe {
        k.gains(
            Grid::new(6, 32),
            raw.ptr(),
            m.ptr(),
            v.ptr(),
            next.ptr(),
            old.ptr(),
            signals.ptr(),
            3,
            3,
            0.,
            1.,
            1.,
        )?;
    }
    let result = next.read()?;
    for (&a, &b) in result.iter().zip(&values) {
        assert!(a > 0.);
        assert!(
            (a as f64 / softplus(b as f64) - 1.).abs() < 2e-6,
            "{a} at {b}"
        );
    }
    Ok(())
}
struct RefMatrix {
    m: usize,
    n: usize,
    offset: usize,
    radius: f64,
    momentum: Vec<Z>,
    variance: Vec<Z>,
    raw: Vec<f64>,
    gains: Vec<f64>,
    gm: Vec<f64>,
    gv: Vec<f64>,
}
fn adam(p: &mut f64, m: &mut f64, v: &mut f64, g: f64, step: usize, lr: f64, decay: f64) {
    *m = 0.9 * *m + 0.1 * g;
    *v = 0.99 * *v + 0.01 * g * g;
    *p = *p * (1. - lr * decay)
        - lr * (*m / (1. - 0.9f64.powi((step + 1) as i32)))
            / ((*v / (1. - 0.99f64.powi((step + 1) as i32))).sqrt() + 1e-8);
}
#[test]
fn full_optimizer_matches_fp64_complex_reference_and_preserves_sphere() -> Result<()> {
    let mut cfg = runner::config();
    cfg.width = 64;
    cfg.gain = 4. / 64.;
    let owners = layout::pairs(&cfg);
    let initial = layout::initialize_model(&cfg, 2);
    for arm in crate::schedule::BASELINE_ARMS
        .into_iter()
        .chain(runner::ARMS)
    {
        let schedule = Schedule::parse(arm)?;
        let mut e = Engine::new(cfg.clone(), &initial)?;
        let mut opt = Optimizer::new(&e, &initial, arm)?;
        let mut p: Vec<_> = initial.iter().map(|x| *x as f64).collect();
        let mut m = vec![0.; p.len()];
        let mut v = m.clone();
        let mut mats: Vec<_> = opt
            .matrices
            .iter()
            .map(|a| RefMatrix {
                m: a.rows,
                n: a.cols,
                offset: a.offset,
                radius: a.radius as f64,
                momentum: vec![Z::default(); a.rows * a.cols],
                variance: vec![Z::default(); a.rows * a.cols],
                raw: vec![1f32.exp_m1().ln() as f64; a.rows + a.cols],
                gains: vec![1.; a.rows + a.cols],
                gm: vec![0.; a.rows + a.cols],
                gv: vec![0.; a.rows + a.cols],
            })
            .collect();
        // Synthetic clocks exercise every boundary without claiming a long training run.
        for step in [0, 1, 999, 1000, 79999, 80000, 99999] {
            let scale = if step == 1 { 0.0001 } else { 0.15 };
            let expanded: Vec<f32> = (0..p.len())
                .map(|i| (((i * 13 + step * 37) as f64).sin() * scale) as f32)
                .collect();
            e.gradients.write(&expanded)?;
            let mut g: Vec<_> = owners
                .iter()
                .map(|&[ar, bi, br, ai, kind]| {
                    Z(
                        expanded[ar as usize] as f64
                            + if kind == 0 {
                                expanded[ai as usize] as f64
                            } else {
                                0.
                            },
                        expanded[bi as usize] as f64
                            - if kind == 0 {
                                expanded[br as usize] as f64
                            } else {
                                0.
                            },
                    )
                })
                .collect();
            let norm = g.iter().map(|z| z.sq()).sum::<f64>().sqrt().max(1.);
            for z in &mut g {
                *z = z.scale(1. / norm);
            }
            let adam_count = if schedule.family == Family::AdamW {
                owners.len()
            } else {
                cfg.vocab * cfg.width / 2
            };
            for i in 0..adam_count {
                let [ar, bi, br, ai, kind] = owners[i];
                for (at, grad) in [(ar as usize, g[i].0), (bi as usize, g[i].1)] {
                    adam(
                        &mut p[at],
                        &mut m[at],
                        &mut v[at],
                        grad,
                        step,
                        schedule.rate(0.002, step) as f64,
                        schedule.family.decay() as f64,
                    );
                }
                if kind == 0 {
                    for (to, from, sgn) in [(ai, ar, 1.), (br, bi, -1.)] {
                        p[to as usize] = sgn * p[from as usize];
                        m[to as usize] = sgn * m[from as usize];
                        v[to as usize] = v[from as usize];
                    }
                }
            }
            for mat in &mut mats {
                let lr = schedule.rate(schedule.family.peak(), step) as f64;
                let mut d = Vec::new();
                let mut nu = Vec::new();
                let mut gain_grad = vec![0.; mat.m + mat.n];
                for i in 0..mat.m * mat.n {
                    let [ar, bi, _, _, _] = owners[mat.offset + i];
                    let (r, c) = (i / mat.n, mat.m + i % mat.n);
                    let z =
                        Z(p[ar as usize], p[bi as usize]).scale(1. / (mat.gains[r] * mat.gains[c]));
                    d.push(z);
                    let gi = g[mat.offset + i];
                    let signal = gi.conj().mul(z).0;
                    gain_grad[r] += signal * mat.gains[c] * sigmoid(mat.raw[r]);
                    gain_grad[c] += signal * mat.gains[r] * sigmoid(mat.raw[c]);
                    let gd = gi.scale(mat.gains[r] * mat.gains[c]);
                    if schedule.family == Family::MuonMD {
                        mat.momentum[i] = mat.momentum[i].scale(0.95).add(gd);
                        nu.push(gd.add(mat.momentum[i].scale(0.95)));
                    } else {
                        adam(
                            &mut d[i].0,
                            &mut mat.momentum[i].0,
                            &mut mat.variance[i].0,
                            gd.0,
                            step,
                            lr,
                            0.,
                        );
                        adam(
                            &mut d[i].1,
                            &mut mat.momentum[i].1,
                            &mut mat.variance[i].1,
                            gd.1,
                            step,
                            lr,
                            0.,
                        );
                    }
                }
                if schedule.family == Family::MuonMD {
                    let u = polar(&nu, mat.m, mat.n);
                    for (d, u) in d.iter_mut().zip(u) {
                        *d = d.sub(u.scale(lr * mat.radius / (mat.m.min(mat.n) as f64).sqrt()));
                    }
                }
                let norm = d.iter().map(|z| z.sq()).sum::<f64>().sqrt();
                for (q, &gq) in gain_grad.iter().enumerate() {
                    adam(
                        &mut mat.raw[q],
                        &mut mat.gm[q],
                        &mut mat.gv[q],
                        gq,
                        step,
                        lr,
                        0.,
                    );
                    if lr != 0. {
                        mat.gains[q] = softplus(mat.raw[q]);
                    }
                }
                if lr == 0. {
                    continue;
                }
                for (i, z) in d.iter().enumerate() {
                    let mut z = z.scale(
                        mat.radius / norm * mat.gains[i / mat.n] * mat.gains[mat.m + i % mat.n],
                    );
                    let [ar, bi, br, ai, _] = owners[mat.offset + i];
                    if schedule.family == Family::AdamMDW {
                        z = z.sub(Z(p[ar as usize], p[bi as usize]).scale(lr * 0.1));
                    }
                    p[ar as usize] = z.0;
                    p[ai as usize] = z.0;
                    p[bi as usize] = z.1;
                    p[br as usize] = -z.1;
                }
                if schedule.family == Family::AdamMDW {
                    let norm = (0..mat.m * mat.n)
                        .map(|i| {
                            let [ar, bi, _, _, _] = owners[mat.offset + i];
                            Z(p[ar as usize], p[bi as usize])
                                .scale(1. / (mat.gains[i / mat.n] * mat.gains[mat.m + i % mat.n]))
                                .sq()
                        })
                        .sum::<f64>()
                        .sqrt();
                    if norm > 0. {
                        for r in 0..mat.m {
                            mat.gains[r] *= norm / mat.radius;
                            let g = mat.gains[r];
                            mat.raw[r] = g + (-(-g).exp_m1()).ln();
                        }
                    }
                }
            }
            opt.step(&e, step)?;
            opt.save(&e)?;
            close(&e.parameters.read()?, &p, 4e-6, 4e-4);
            close(&e.first_moment.read()?, &m, 2e-7, 1e-4);
            close(&e.second_moment.read()?, &v, 2e-9, 2e-4);
            for (actual, reference) in opt.matrices.iter().zip(&mats) {
                close(&actual.raw.read()?, &reference.raw, 3e-5, 1e-4);
                close(
                    &actual.momentum.read()?,
                    &components(&reference.momentum),
                    2e-6,
                    3e-4,
                );
                close(&actual.gm.read()?, &reference.gm, 2e-7, 3e-4);
                close(&actual.gv.read()?, &reference.gv, 2e-9, 4e-4);
                close(
                    &actual.variance.read()?,
                    &components(&reference.variance),
                    2e-9,
                    4e-4,
                );
            }
        }
    }
    Ok(())
}
#[test]
fn checkpoint_resumes_identical_tail_updates_and_rejects_unclean_or_corrupt_state() -> Result<()> {
    let dir = std::env::temp_dir().join(format!("quartic-md-continuation-{}", std::process::id()));
    fs::create_dir_all(&dir)?;
    let data = arithmetic_data::arithmetic(&runner::row())?;
    let mut cfg = runner::config();
    cfg.width = 64;
    cfg.gain = 4. / 64.;
    let p = layout::initialize_model(&cfg, 1);
    for arm in runner::ARMS {
        let mut e = Engine::new(cfg.clone(), &p)?;
        let mut opt = Optimizer::new(&e, &p, arm)?;
        let (order, edges) = runner::order(4704, 20, 1);
        for step in 0..18 {
            let indices = &order[edges[step]..edges[step + 1]];
            runner::batch(&mut e, &data.train, indices)?;
            e.backward(indices.len(), 2)?;
            opt.step(&e, step)?;
        }
        runner::checkpoint(&e, &opt, &dir, 18, edges[18], &[json!({"step":18})])?;
        let mut resumed = Engine::new(cfg.clone(), &p)?;
        let mut resumed_opt = Optimizer::new(&resumed, &p, arm)?;
        assert_eq!(runner::restore(&mut resumed, &mut resumed_opt, &dir)?.0, 18);
        for step in 18..20 {
            let indices = &order[edges[step]..edges[step + 1]];
            for (e, o) in [(&mut e, &mut opt), (&mut resumed, &mut resumed_opt)] {
                runner::batch(e, &data.train, indices)?;
                e.backward(indices.len(), 2)?;
                o.step(e, step)?;
            }
            assert_eq!(
                runner::float_bytes(&opt.save(&e)?),
                runner::float_bytes(&resumed_opt.save(&resumed)?)
            );
        }
        runner::record(&dir.join("execution.json"), &json!({"clean":false}))?;
        assert!(runner::restore(&mut resumed, &mut resumed_opt, &dir).is_err());
        runner::record(&dir.join("execution.json"), &json!({"clean":true}))?;
        let mut bytes = fs::read(dir.join("state.f32"))?;
        bytes[11] ^= 1;
        fs::write(dir.join("state.f32"), bytes)?;
        assert!(runner::restore(&mut resumed, &mut resumed_opt, &dir).is_err());
    }
    fs::remove_dir_all(dir)?;
    Ok(())
}

#[test]
fn offline_audit_checks_all_metrics_and_rejects_rehashed_nonfinite_state() -> Result<()> {
    // Synthetic audit fixture only: never stored in scientific result directories.
    let root = std::env::temp_dir().join(format!("quartic-md-audit-test-{}", std::process::id()));
    let dir = root.join("adamw-linear-warmup1000/seed0");
    fs::create_dir_all(&dir)?;
    let initial = layout::initialize_model(&runner::config(), 0);
    let mut values = initial.clone();
    values.resize(3 * initial.len(), 0.);
    let bytes = runner::float_bytes(&values);
    fs::write(dir.join("state.f32"), &bytes)?;
    let initial = runner::float_bytes(&initial);
    fs::write(dir.join("initial.f32"), &initial)?;
    let history:Vec<_>=(0..=optimizer::BUDGET.div_ceil(19)).map(|i| {
        let step=(i*19).min(optimizer::BUDGET);let examples=(step/19)*4704+(step%19)*256;
        json!({"step":step,"examples":examples,"epochs":examples as f64/4704.,"seconds":i as f64,
            "train":{"exact":1093,"total":4704,"accuracy":1093./4704.,"loss":3.1},
            "validation":{"exact":1091,"total":4705,"accuracy":1091./4705.,"loss":3.2}})
    }).collect();
    let mut text = Vec::new();
    for point in &history {
        text.extend(serde_json::to_vec(point)?);
        text.push(b'\n');
    }
    fs::write(dir.join("history.jsonl"), &text)?;
    let last = history.last().unwrap();
    let mut cp = json!({"completed":last["step"],"examples":last["examples"],"source_sha256":runner::source_hash(),
        "ptx_sha256":runner::ptx_hash(),"state_sha256":crate::hash(&bytes),"history_sha256":crate::hash(&text)});
    runner::record(&dir.join("checkpoint.json"), &cp)?;
    runner::record(
        &dir.join("protocol.json"),
        &json!({"source_sha256":runner::source_hash(),"initial_sha256":crate::hash(&initial),"arm":"adamw-linear-warmup1000","seed":0}),
    )?;
    runner::record(&dir.join("execution.json"), &json!({"clean":true}))?;
    let mut result = runner::summary(&history);
    result["status"] = json!("complete");
    runner::record(&dir.join("result.json"), &result)?;
    let valid = runner::audit(&root)?;
    assert_eq!(valid["completed"], 1);
    assert_eq!(valid["integrity_failures"], json!([]));
    result["late_min_accuracy"] = json!(1.);
    runner::record(&dir.join("result.json"), &result)?;
    assert_eq!(
        runner::audit(&root)?["integrity_failures"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    result = runner::summary(&history);
    result["status"] = json!("complete");
    runner::record(&dir.join("result.json"), &result)?;
    values[0] = f32::NAN;
    let bytes = runner::float_bytes(&values);
    fs::write(dir.join("state.f32"), &bytes)?;
    cp["state_sha256"] = json!(crate::hash(&bytes));
    runner::record(&dir.join("checkpoint.json"), &cp)?;
    assert_eq!(
        runner::audit(&root)?["integrity_failures"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn schedules_cover_the_factorial_and_match_all_legacy_linear_rates() -> Result<()> {
    use crate::schedule::{ARMS, BASELINE_ARMS, BUDGET, Shape};
    let mut cells = std::collections::BTreeSet::new();
    for arm in BASELINE_ARMS.into_iter().chain(ARMS) {
        let s = Schedule::parse(arm)?;
        assert!(cells.insert((s.family.name(), s.name(), s.warmup)));
        for peak in [0.002, 0.01] {
            assert_eq!(s.rate(peak, 99999), 1e-8f32);
            if s.warmup > 0 {
                assert_eq!(s.rate(peak, 0), 0.);
                assert_eq!(s.rate(peak, 1000), peak as f32);
            }
            for t in 0..BUDGET {
                let actual = s.rate(peak, t);
                assert!(actual.is_finite() && actual >= 0. && actual <= peak as f32);
                if s.shape == Shape::Linear && s.warmup == 0 {
                    assert_eq!(
                        actual.to_bits(),
                        crate::legacy_optimizer::rate(peak, t).to_bits()
                    );
                }
                if t > 0 && t > s.warmup {
                    assert!(actual <= s.rate(peak, t - 1));
                }
                if s.shape != Shape::Linear && t >= s.warmup && t <= 80000 {
                    assert_eq!(actual, peak as f32);
                }
            }
            if s.shape != Shape::Linear {
                let u = (85000. - 80000.) / (99999. - 80000.);
                let exponent = match s.shape {
                    Shape::WsdCuberoot20 => 1. / 3.,
                    Shape::WsdFourthroot20 => 0.25,
                    _ => 0.5,
                };
                let reference = (1e-8 + (peak - 1e-8) * (1. - f64::powf(u, exponent))) as f32;
                assert!((s.rate(peak, 85000) - reference).abs() < 1e-9);
            }
        }
    }
    assert_eq!(cells.len(), 18);
    assert_eq!(ARMS.len() * 3 + BASELINE_ARMS.len() * 3, 54);
    Ok(())
}

#[test]
fn unchanged_linear_controls_match_frozen_optimizer_bit_for_bit() -> Result<()> {
    let cfg = runner::config();
    let p = layout::initialize_model(&cfg, 0);
    let data = arithmetic_data::arithmetic(&runner::row())?;
    for arm in crate::schedule::BASELINE_ARMS {
        let mut new = Engine::new(cfg.clone(), &p)?;
        let mut old = Engine::new(cfg.clone(), &p)?;
        let mut opt = Optimizer::new(&new, &p, arm)?;
        let mut legacy = crate::legacy_optimizer::Optimizer::new(&old, &p, arm)?;
        for step in 0..4 {
            let indices: Vec<_> = (0..if step == 3 { 96 } else { 256 }).collect();
            runner::batch(&mut new, &data.train, &indices)?;
            runner::batch(&mut old, &data.train, &indices)?;
            new.backward(indices.len(), 2)?;
            old.backward(indices.len(), 2)?;
            opt.step(&new, step)?;
            legacy.step(&old, step)?;
            for (a, b) in [
                (&new.parameters, &old.parameters),
                (&new.first_moment, &old.first_moment),
                (&new.second_moment, &old.second_moment),
            ] {
                assert_eq!(
                    runner::float_bytes(&a.read()?),
                    runner::float_bytes(&b.read()?)
                );
            }
            for (a, b) in opt.matrices.iter().zip(&legacy.matrices) {
                for (a, b) in [
                    (&a.momentum, &b.momentum),
                    (&a.raw, &b.raw),
                    (&a.gains, &b.gains),
                    (&a.gm, &b.gm),
                    (&a.gv, &b.gv),
                ] {
                    assert_eq!(
                        runner::float_bytes(&a.read()?),
                        runner::float_bytes(&b.read()?)
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn zero_warmup_step_advances_moments_without_moving_weights_or_gains() -> Result<()> {
    let cfg = runner::config();
    let p = layout::initialize_model(&cfg, 1);
    for arm in runner::ARMS
        .into_iter()
        .filter(|s| s.ends_with("warmup1000"))
    {
        let mut e = Engine::new(cfg.clone(), &p)?;
        let mut o = Optimizer::new(&e, &p, arm)?;
        e.gradients.write(&vec![0.01; p.len()])?;
        o.step(&e, 0)?;
        assert_eq!(
            runner::float_bytes(&e.parameters.read()?),
            runner::float_bytes(&p)
        );
        assert!(e.first_moment.read()?.iter().any(|x| *x != 0.));
        for m in &o.matrices {
            assert!(m.gains.read()?.iter().all(|x| *x == 1.));
            assert!(m.momentum.read()?.iter().any(|x| *x != 0.));
        }
        o.save(&e)?;
    }
    Ok(())
}

#[test]
fn physical_mdw_correction_and_zero_branch_preserve_exact_weights() -> Result<()> {
    use kernite_cuda::Grid;
    let cuda = Cuda::new()?;
    let ops = Ops::new(&cuda, 8)?;
    let map = cuda.upload(&[
        0u32, 8, 2, 10, 0, 1, 9, 3, 11, 0, 4, 12, 6, 14, 0, 5, 13, 7, 15, 0,
    ])?;
    for decay_rate in [0., 0.2, 1.] {
        let mut physical = vec![0.; 16];
        physical[0] = 3.;
        physical[10] = 3.;
        physical[8] = 4.;
        physical[2] = -4.;
        let candidate = cuda.upload(&[3f32, 4., 0., 0., 0., 0., 0., 0.])?;
        let norm = cuda.upload(&[5f32])?;
        let p = cuda.upload(&physical)?;
        let raw = cuda.upload(&[1f32.exp_m1().ln(); 4])?;
        let gains = cuda.upload(&[1f32; 4])?;
        unsafe {
            ops.adam.fuse_decay(
                Grid::new(1, 256),
                p.ptr(),
                candidate.ptr(),
                norm.ptr(),
                gains.ptr(),
                map.ptr(),
                0,
                2,
                2,
                5.,
                decay_rate,
            )?;
        }
        let corrected = p.read()?;
        for (a, b) in corrected.iter().zip(&physical) {
            assert!((*a - (*b - decay_rate * *b)).abs() < 5e-7);
        }
        ops.norm(&candidate, &norm)?;
        unsafe {
            ops.adam
                .repair_rows(Grid::new(1, 256), raw.ptr(), gains.ptr(), norm.ptr(), 2, 5.)?;
        }
        assert_eq!(
            runner::float_bytes(&p.read()?),
            runner::float_bytes(&corrected)
        );
        let g = gains.read()?;
        let r = raw.read()?;
        assert_eq!(&g[2..], &[1., 1.]);
        if decay_rate == 1. {
            assert!(corrected.iter().all(|x| *x == 0.));
            assert_eq!(g, vec![1.; 4]);
        } else {
            assert!((g[0] as f64 - (1. - decay_rate as f64)).abs() < 3e-7);
        }
        for i in 0..4 {
            assert!(r[i].is_finite());
            assert!((softplus(r[i] as f64) - g[i] as f64).abs() < 2e-6);
        }
    }
    let values = [1e-20f32, 1e-8, 0.001, 0.1, 1., 30.];
    let gains = cuda.upload(&values)?;
    let raw = cuda.alloc::<f32>(6)?;
    let norm = cuda.upload(&[1f32])?;
    unsafe {
        ops.adam
            .repair_rows(Grid::new(1, 256), raw.ptr(), gains.ptr(), norm.ptr(), 6, 1.)?;
    }
    for (a, g) in raw.read()?.into_iter().zip(values) {
        assert!(
            (softplus(a as f64) / g as f64 - 1.).abs() < 4e-6,
            "inverse-softplus at {g}: {a}"
        );
    }
    Ok(())
}

#[test]
fn combined_report_checks_frozen_baseline_identity_and_pairing() -> Result<()> {
    const BASE: &str = "2744d9d51b4f5436c3b1d6b94bcb4ebb09f9466f426fbd9fb861e0746cca4799";
    // Structural report fixture, not scientific observations.
    let root = std::env::temp_dir().join(format!("quartic-combined-test-{}", std::process::id()));
    let prior = root.join("prior");
    let new = root.join("new");
    fs::create_dir_all(&prior)?;
    fs::create_dir_all(&new)?;
    let protocol = json!({"initial_sha256":"fixture-initial","sampler_sha256":"fixture-sampler","case":{"fixture":true},"source_sha256":BASE});
    let mut rows = Vec::new();
    for seed in 0..3 {
        for arm in crate::schedule::BASELINE_ARMS {
            let dir = prior.join(arm).join(format!("seed{seed}"));
            fs::create_dir_all(&dir)?;
            runner::record(&dir.join("protocol.json"), &protocol)?;
            rows.push(json!({"arm":arm,"seed":seed,"status":"complete"}));
        }
    }
    let mut old = json!({"expected":6,"completed":6,"source_sha256":BASE,"integrity_failures":[],"results":rows});
    runner::record(&prior.join("audit.json"), &old)?;
    let valid = runner::combined(&new, &prior)?;
    assert_eq!(valid["expected"], 54);
    assert_eq!(valid["completed"], 6);
    assert_eq!(valid["results"].as_array().unwrap().len(), 54);
    assert_eq!(valid["integrity_failures"], json!([]));
    let dir = new.join("adammd-linear/seed0");
    fs::create_dir_all(&dir)?;
    let mut wrong = protocol;
    wrong["sampler_sha256"] = json!("changed");
    runner::record(&dir.join("protocol.json"), &wrong)?;
    assert_eq!(
        runner::combined(&new, &prior)?["integrity_failures"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    old["source_sha256"] = json!("untrusted");
    runner::record(&prior.join("audit.json"), &old)?;
    assert!(runner::combined(&new, &prior).is_err());
    fs::remove_dir_all(root)?;
    Ok(())
}
