//! Native CPU and CUDA checks, independent of Python and of scientific outcomes.
use crate::{
    complex,
    engine::{Config, Engine},
    layout, prepare,
    random::Random,
    reference,
};
use kernite_cuda::{Cuda, Grid};

fn config(architecture: &str, phase: bool) -> Config {
    Config {
        width: 128,
        layers: 1,
        normalized: true,
        ssmax: true,
        gain: 1.0 / 32.0,
        lr: 0.002,
        decay: 0.1,
        warmup: 50,
        architecture: architecture.into(),
        phase_optimizer: phase,
    }
}

#[test]
fn complex_layout_partitions_storage_and_matches_budget() {
    let (w, l, h) = (128, 1, layout::hidden(128, "complex-product"));
    assert_eq!(h, 852);
    let map = layout::pairs(w, l, h);
    assert_eq!(2 * map.len(), 208768);
    assert_eq!(layout::count(w, l, "real"), 208896);
    let mut seen = vec![0; layout::count(w, l, "complex-product")];
    for &[a, b, c, d, kind] in &map {
        let indices = if kind == 1 {
            vec![a, b]
        } else {
            vec![a, b, c, d]
        };
        for index in indices {
            seen[index as usize] += 1;
        }
    }
    assert!(seen.iter().all(|&n| n == 1));
    for seed in 0..3 {
        let p = layout::initialize(w, l, "complex-product", seed);
        assert!(layout::validate(&p, w, l, "complex-product"));
        assert_eq!(p, layout::initialize(w, l, "complex-diagonal", seed));
        assert_eq!(
            &p[..97 * w],
            &layout::initialize(w, l, "real", seed)[..97 * w]
        );
    }
}

#[test]
fn arithmetic_split_is_exhaustive_and_phase_product_composes() {
    let rows = prepare::dataset();
    let mut seen = vec![false; 97 * 97];
    for &[a, b, c] in &rows {
        assert_eq!(c, (a + b).pow(3) % 97);
        assert!(!seen[(97 * a + b) as usize]);
        seen[(97 * a + b) as usize] = true;
    }
    assert!(seen.iter().all(|&b| b));
    for a in 0..97 {
        for b in 0..97 {
            let alpha = std::f64::consts::TAU * a as f64 / 97.0;
            let beta = std::f64::consts::TAU * b as f64 / 97.0;
            let y = reference::product([alpha.cos(), alpha.sin(), beta.cos(), beta.sin()], true);
            let scale = 1.0 / (1.0 + (-(1.0 + f32::EPSILON as f64).sqrt()).exp()) / 2.0f64.sqrt();
            assert!((y[0] - scale * (alpha + beta).cos()).abs() < 1e-14);
            assert!((y[1] - scale * (alpha + beta).sin()).abs() < 1e-14);
        }
    }
}

#[test]
#[ignore = "requires CUDA"]
fn product_kernel_forward_and_independent_finite_differences() -> crate::Result<()> {
    let cuda = Cuda::new()?;
    let kernels = complex::kernels::Kernels::load(&cuda)?;
    let (pairs, rows) = (7usize, 3usize);
    let mut x: Vec<f32> = (0..rows * pairs * 4)
        .map(|i| (i as f32 * 0.17).sin())
        .collect();
    x[0] = 0.;
    x[pairs] = 0.;
    let dy: Vec<f32> = (0..rows * pairs * 2)
        .map(|i| (i as f32 * 0.23).cos())
        .collect();
    let bx = cuda.upload(&x)?;
    let bg = cuda.upload(&dy)?;
    let y = cuda.alloc::<f32>(rows * pairs * 2)?;
    let dx = cuda.alloc::<f32>(x.len())?;
    for cross in [false, true] {
        // SAFETY: all four component planes and both output planes are allocated;
        // the kernel checks its flattened complex-element bound before access.
        unsafe {
            if cross {
                kernels.product::<true>(
                    Grid::new(1, 256),
                    y.ptr(),
                    bx.ptr(),
                    pairs as u32,
                    (rows * pairs) as u32,
                )?;
                kernels.product_bwd::<true>(
                    Grid::new(1, 256),
                    dx.ptr(),
                    bx.ptr(),
                    bg.ptr(),
                    pairs as u32,
                    (rows * pairs) as u32,
                )?;
            } else {
                kernels.product::<false>(
                    Grid::new(1, 256),
                    y.ptr(),
                    bx.ptr(),
                    pairs as u32,
                    (rows * pairs) as u32,
                )?;
                kernels.product_bwd::<false>(
                    Grid::new(1, 256),
                    dx.ptr(),
                    bx.ptr(),
                    bg.ptr(),
                    pairs as u32,
                    (rows * pairs) as u32,
                )?;
            }
        }
        let y = y.read()?;
        let dx = dx.read()?;
        for row in 0..rows {
            for c in 0..pairs {
                let at = row * 4 * pairs + c;
                let out = row * 2 * pairs + c;
                let z = std::array::from_fn(|j| x[at + j * pairs] as f64);
                let expected = reference::product(z, cross);
                for j in 0..2 {
                    assert!((y[out + j * pairs] as f64 - expected[j]).abs() < 2e-6);
                }
                for j in 0..4 {
                    let mut plus = z;
                    let mut minus = z;
                    let eps = 1e-6;
                    plus[j] += eps;
                    minus[j] -= eps;
                    let a = reference::product(plus, cross);
                    let b = reference::product(minus, cross);
                    let fd = ((a[0] - b[0]) * dy[out] as f64
                        + (a[1] - b[1]) * dy[out + pairs] as f64)
                        / (2. * eps);
                    assert!(
                        (dx[at + j * pairs] as f64 - fd).abs() < 3e-6,
                        "cross={cross} component={j} derivative={fd} gpu={}",
                        dx[at + j * pairs]
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn complete_model_matches_scalar_oracle_and_directional_derivatives() -> crate::Result<()> {
    let rows = [[3, 7, 30], [0, 0, 0], [19, 80, 8]];
    let data: Vec<_> = rows.iter().flatten().copied().collect();
    let rope = prepare::rope();
    for architecture in ["real", "complex-diagonal", "complex-product"] {
        let cfg = config(architecture, false);
        let initial = layout::initialize(128, 1, architecture, 0);
        let engine = Engine::new(cfg.clone(), &initial, &rope, &data, &data, &[0, 1, 2])?;
        engine.forward(false, 0, 3, true)?;
        engine.backward(3)?;
        let p: Vec<f64> = initial.iter().map(|&x| x as f64).collect();
        let expected = reference::logits(&cfg, &p, &rows, &rope);
        let actual = engine.logits.read()?;
        let error = expected
            .iter()
            .zip(&actual)
            .map(|(a, b)| (a - *b as f64).abs())
            .fold(0., f64::max);
        assert!(error < 5e-5, "{architecture} max logit error={error}");
        let (loss, _) = engine.metrics(3)?;
        assert!((loss / 3. - reference::loss(&cfg, &p, &rows, &rope)).abs() < 1e-5);
        let gradient = engine.gradients.read()?;
        assert!(gradient.iter().all(|v| v.is_finite()));
        let h = layout::hidden(128, architecture);
        let w = 128;
        let ranges = [
            (0, 97 * w),
            (97 * w, 97 * w + 3 * w * w),
            (97 * w + 3 * w * w, 97 * w + 4 * w * w),
            (97 * w + 4 * w * w, 97 * w + 4 * w * w + 2 * h * w),
            (97 * w + 4 * w * w + 2 * h * w, p.len()),
        ];
        let mut rng = Random::new(91);
        for (start, end) in ranges {
            let mut direction = vec![0.; p.len()];
            for value in &mut direction[start..end] {
                *value = rng.gauss(0., 1.) / ((end - start) as f64).sqrt();
            }
            let eps = 1e-5;
            let plus: Vec<_> = p.iter().zip(&direction).map(|(p, d)| p + eps * d).collect();
            let minus: Vec<_> = p.iter().zip(&direction).map(|(p, d)| p - eps * d).collect();
            let fd = (reference::loss(&cfg, &plus, &rows, &rope)
                - reference::loss(&cfg, &minus, &rows, &rope))
                / (2. * eps);
            let derivative: f64 = gradient
                .iter()
                .zip(&direction)
                .map(|(g, d)| *g as f64 * d)
                .sum();
            assert!(
                (derivative - fd).abs() < 5e-5 + 0.002 * fd.abs(),
                "{architecture} range={start}..{end} gpu={derivative} fd={fd}"
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn phase_adamw_is_rotation_equivariant_and_preserves_expansion() -> crate::Result<()> {
    let cuda = Cuda::new()?;
    let kernels = complex::kernels::Kernels::load(&cuda)?;
    let theta = 0.73f64;
    let rotate = |a: f64, b: f64| {
        [
            a * theta.cos() - b * theta.sin(),
            a * theta.sin() + b * theta.cos(),
        ]
    };
    let initial = rotate(0.2, -0.3);
    let p = cuda.upload(&[0.2f32, -0.3, initial[0] as f32, initial[1] as f32])?;
    let m = cuda.alloc::<f32>(4)?;
    let v = cuda.alloc::<f32>(4)?;
    let map = cuda.upload(&[0u32, 1, 0, 0, 1, 2, 3, 0, 0, 1])?;
    let mut g = cuda.alloc::<f32>(4)?;
    let mut oracle = [0.2f64, -0.3];
    let mut mean = [0.; 2];
    let mut square = 0.;
    for step in 1..=9 {
        let ga = (step as f64 * 0.7).sin();
        let gb = (step as f64 * 0.2).cos();
        let rg = rotate(ga, gb);
        g.write(&[ga as f32, gb as f32, rg[0] as f32, rg[1] as f32])?;
        let lr = if step == 1 { 0. } else { 0.002 };
        // SAFETY: two disjoint complex pairs own all four initialized coordinates.
        unsafe {
            kernels.adamw::<true>(
                Grid::new(1, 256),
                p.ptr(),
                m.ptr(),
                v.ptr(),
                g.ptr(),
                map.ptr(),
                2,
                lr,
                0.1,
                (1. - 0.9f64.powi(step)) as f32,
                (1. - 0.98f64.powi(step)).sqrt() as f32,
            )?;
        }
        let grads = [ga as f32 as f64, gb as f32 as f64];
        square = 0.98 * square + 0.02 * (grads[0].powi(2) + grads[1].powi(2)) / 2.;
        for j in 0..2 {
            mean[j] = 0.9 * mean[j] + 0.1 * grads[j];
            oracle[j] = oracle[j] * (1. - lr as f64 * 0.1)
                - lr as f64 * mean[j]
                    / (1. - 0.9f64.powi(step))
                    / ((square / (1. - 0.98f64.powi(step))).sqrt() + 1e-8);
        }
        let values = p.read()?;
        let rotated = rotate(values[0] as f64, values[1] as f64);
        for j in 0..2 {
            assert!((values[j] as f64 - oracle[j]).abs() < 2e-7);
            assert!((values[j + 2] as f64 - rotated[j]).abs() < 2e-7);
        }
        let moments = v.read()?;
        assert_eq!(moments[0], moments[1]);
        assert_eq!(moments[2], moments[3]);
    }
    // Full engine updates contract the four stored matrix derivatives before AdamW.
    for phase in [false, true] {
        let cfg = config("complex-product", phase);
        let params = layout::initialize(128, 1, "complex-product", 1);
        let data = [2, 3, 28, 9, 9, 12]; // Labels are immaterial to this tying/state test.
        let engine = Engine::new(cfg, &params, &prepare::rope(), &data, &data, &[0, 1])?;
        for step in 0..3 {
            engine.forward(false, 0, 2, true)?;
            engine.backward(2)?;
            engine.update(step)?;
        }
        assert!(layout::validate(
            &engine.parameters.read()?,
            128,
            1,
            "complex-product"
        ));
        if phase {
            let v = engine.second_moment.read()?;
            for [a, b, _, _, _] in layout::pairs(128, 1, 852) {
                assert_eq!(v[a as usize], v[b as usize]);
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn checkpoint_restores_nonempty_state_before_tail_batch() -> crate::Result<()> {
    use crate::{Input, checkpoint, record, restore};
    let root = std::env::temp_dir().join(format!("complex-kernite-resume-{}", std::process::id()));
    std::fs::create_dir(&root)?;
    for optimizer in ["adamw", "phase"] {
        let input_dir = root.join(format!("input-{optimizer}"));
        prepare::write(&input_dir, "complex-product", optimizer, 0, 38)?;
        let first = Input::load(&input_dir)?;
        for step in 0..18 {
            first.step(step)?;
            first.engine.update(step)?;
        }
        let out = root.join(optimizer);
        std::fs::create_dir(&out)?;
        std::fs::write(out.join("history.jsonl"), "{\"step\":18}\n")?;
        checkpoint(&first, &out, 18)?;
        record(
            &out.join("execution.json"),
            &serde_json::json!({"completed":18,"possible_end":18}),
        )?;
        let mut second = Input::load(&input_dir)?;
        assert_eq!(restore(&mut second, &out)?, 18);
        for step in 18..22 {
            assert_eq!(first.step(step)?, if step == 18 { 96 } else { 256 });
            second.step(step)?;
            first.engine.update(step)?;
            second.engine.update(step)?;
        }
        assert_eq!(
            first.engine.parameters.read()?,
            second.engine.parameters.read()?
        );
        assert_eq!(
            first.engine.first_moment.read()?,
            second.engine.first_moment.read()?
        );
        assert_eq!(
            first.engine.second_moment.read()?,
            second.engine.second_moment.read()?
        );
        record(
            &out.join("execution.json"),
            &serde_json::json!({"completed":18,"possible_end":38}),
        )?;
        assert!(restore(&mut second, &out).is_err());
    }
    std::fs::remove_dir_all(root)?;
    Ok(())
}
