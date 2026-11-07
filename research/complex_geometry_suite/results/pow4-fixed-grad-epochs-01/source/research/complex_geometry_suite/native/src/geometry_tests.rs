//! Independent complex arithmetic, parameter ownership and optimizer checks.
use super::*;

fn config(width: usize, heads: usize) -> Config {
    Config {
        width,
        heads,
        vocab: 11,
        layers: 2,
        max_batch: 3,
        max_length: 4,
        normalized: true,
        ssmax: true,
        gain: heads as f32 / width as f32,
        lr: 0.002,
        decay: 0.1,
        warmup: 3,
        architecture: "complex-hermitian".into(),
        phase_optimizer: false,
    }
}

fn phase(x: &[f64], groups: usize, angle: f64) -> Vec<f64> {
    let m = x.len() / (2 * groups);
    let (s, c) = angle.sin_cos();
    let mut out = x.to_vec();
    for group in 0..groups {
        for j in 0..m {
            let at = group * 2 * m + j;
            out[at] = c * x[at] - s * x[at + m];
            out[at + m] = s * x[at] + c * x[at + m];
        }
    }
    out
}

fn close(a: &[f64], b: &[f64], tolerance: f64) {
    assert_eq!(a.len(), b.len());
    for (i, (a, b)) in a.iter().zip(b).enumerate() {
        assert!((a - b).abs() <= tolerance, "coordinate{i}: {a} != {b}");
    }
}

#[test]
fn complex_layout_covers_parameters_once_and_matches_complex_multiplication() -> Result<()> {
    for (w, heads) in [(64, 4), (128, 4), (128, 1), (256, 8)] {
        let cfg = config(w, heads);
        let p = layout::initialize_model(&cfg, 7);
        layout::validate(&cfg, &p)?;
        let mut ownership = vec![0; p.len()];
        for [ar, bi, br, ai, kind] in layout::pairs(&cfg) {
            for at in if kind == 0 {
                vec![ar, bi, br, ai]
            } else {
                vec![ar, bi]
            } {
                ownership[at as usize] += 1;
            }
        }
        assert!(ownership.iter().all(|&n| n == 1));
        assert_eq!(
            layout::independent_count(&cfg),
            cfg.vocab * w + (p.len() - cfg.vocab * w) / 2
        );
        let mut ablation = cfg.clone();
        ablation.architecture = "complex-linear".into();
        assert_eq!(layout::initialize_model(&ablation, 7), p);
        let h = layout::hidden(w, "");
        let mut offset = cfg.vocab * w;
        for _ in 0..cfg.layers {
            for (n, k, og, ig) in [
                (3 * w, w, 3 * heads, heads),
                (w, w, heads, heads),
                (2 * h, w, 2, heads),
                (w, h, heads, 1),
            ] {
                let weights: Vec<f64> = p[offset..offset + n * k]
                    .iter()
                    .map(|&v| v as f64)
                    .collect();
                let x: Vec<f64> = (0..k).map(|i| (i as f64 * 0.37).sin()).collect();
                let direct = complex_linear(&x, &weights, n, og, ig);
                close(&linear(&x, &weights, n), &direct, 1e-12);
                close(
                    &linear(&phase(&x, ig, 0.73), &weights, n),
                    &phase(&direct, og, 0.73),
                    1e-12,
                );
                offset += n * k;
            }
        }
        let mut corrupted = p.clone();
        let [_, _, _, ai, _] = layout::pairs(&cfg).into_iter().find(|p| p[4] == 0).unwrap();
        corrupted[ai as usize] += 1.;
        assert!(layout::validate(&cfg, &corrupted).is_err());
    }
    Ok(())
}

#[test]
fn hermitian_geometry_and_magnitude_norm_have_expected_invariances() {
    let v = vec![0.3, -0.7, 0.1, 0.6];
    let a = vec![0.8, 0.2, -0.5, 0.9];
    let y = hermitian_projection(&v, &a);
    let jv = phase(&v, 1, std::f64::consts::FRAC_PI_2);
    for direction in [&v, &jv] {
        assert!(
            direction
                .iter()
                .zip(&y)
                .map(|(a, b)| a * b)
                .sum::<f64>()
                .abs()
                < 1e-14
        );
        close(&hermitian_projection(&v, direction), &[0.; 4], 1e-14);
    }
    close(&hermitian_projection(&v, &y), &y, 1e-14);
    close(&hermitian_projection(&phase(&v, 1, 0.8), &a), &y, 1e-14);
    close(&hermitian_projection(&[0.; 4], &a), &a, 0.);
    close(&rms(&phase(&a, 1, 0.8)), &phase(&rms(&a), 1, 0.8), 1e-14);
    // A clipped tiny direction is deliberately not an exact projector.
    let tiny = vec![1e-13, 0., 0., 0.];
    close(
        &hermitian_projection(&tiny, &[1., 0., 1., 0.]),
        &[0.99, 0., 0.99, 0.],
        1e-14,
    );
}

#[test]
#[ignore = "requires CUDA"]
fn adamw_contracts_gradients_before_moments_and_preserves_complex_constraints() -> Result<()> {
    for phase_optimizer in [false, true] {
        let mut cfg = config(64, 4);
        cfg.phase_optimizer = phase_optimizer;
        let initial = layout::initialize_model(&cfg, 3);
        let mut engine = Engine::new(cfg.clone(), &initial)?;
        let map = layout::pairs(&cfg);
        let mut expected: Vec<f64> = initial.iter().map(|&v| v as f64).collect();
        let mut m = vec![0f64; initial.len()];
        let mut v = m.clone();
        for step in 0..5 {
            let g: Vec<f32> = (0..initial.len())
                .map(|i| 0.05 * (i as f32 * 0.37 + step as f32).sin())
                .collect();
            engine.gradients.write(&g)?;
            engine.update(step)?;
            let lr = cfg.lr * (step as f64 / cfg.warmup as f64).min(1.);
            for &[ar, bi, br, ai, kind] in &map {
                let (ar, bi, br, ai) = (ar as usize, bi as usize, br as usize, ai as usize);
                // Deliberately independent gradients in all four expanded blocks.
                let ga = g[ar] as f64 + if kind == 0 { g[ai] as f64 } else { 0. };
                let gb = g[bi] as f64 - if kind == 0 { g[br] as f64 } else { 0. };
                for (i, gradient) in [(ar, ga), (bi, gb)] {
                    m[i] = 0.9 * m[i] + 0.1 * gradient;
                    let square = if phase_optimizer {
                        (ga * ga + gb * gb) / 2.
                    } else {
                        gradient * gradient
                    };
                    v[i] = 0.98 * v[i] + 0.02 * square;
                    let mh = m[i] / (1. - 0.9f64.powi(step as i32 + 1));
                    let vh = v[i] / (1. - 0.98f64.powi(step as i32 + 1));
                    expected[i] =
                        expected[i] * (1. - lr * cfg.decay as f64) - lr * mh / (vh.sqrt() + 1e-8);
                }
                if kind == 0 {
                    expected[ai] = expected[ar];
                    expected[br] = -expected[bi];
                    m[ai] = m[ar];
                    m[br] = -m[bi];
                    v[ai] = v[ar];
                    v[br] = v[bi];
                }
            }
            let actual = engine.parameters.read()?;
            layout::validate(&cfg, &actual)?;
            layout::validate_state(
                &cfg,
                &actual,
                &engine.first_moment.read()?,
                &engine.second_moment.read()?,
            )?;
            close(
                &actual.iter().map(|&x| x as f64).collect::<Vec<_>>(),
                &expected,
                2e-7,
            );
            close(
                &engine
                    .first_moment
                    .read()?
                    .iter()
                    .map(|&x| x as f64)
                    .collect::<Vec<_>>(),
                &m,
                2e-8,
            );
            close(
                &engine
                    .second_moment
                    .read()?
                    .iter()
                    .map(|&x| x as f64)
                    .collect::<Vec<_>>(),
                &v,
                2e-9,
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn paired_variance_adamw_commutes_with_complex_coordinate_rotations() -> Result<()> {
    let mut cfg = config(64, 4);
    cfg.phase_optimizer = true;
    cfg.warmup = 0;
    let p = layout::initialize_model(&cfg, 5);
    let map = layout::pairs(&cfg);
    let mut rotated = p.clone();
    let mut gradients = vec![0.; p.len()];
    let mut rotated_gradients = gradients.clone();
    let angle = 0.71f32;
    let (sin, cos) = angle.sin_cos();
    for &[ar, bi, br, ai, kind] in &map {
        let (ar, bi, br, ai) = (ar as usize, bi as usize, br as usize, ai as usize);
        rotated[ar] = cos * p[ar] - sin * p[bi];
        rotated[bi] = sin * p[ar] + cos * p[bi];
        if kind == 0 {
            rotated[ai] = rotated[ar];
            rotated[br] = -rotated[bi];
        }
        let a = 0.017 * (ar as f32 * 0.193).cos();
        let b = 0.043 * (bi as f32 * 0.271).sin();
        gradients[ar] = a;
        gradients[bi] = b;
        rotated_gradients[ar] = cos * a - sin * b;
        rotated_gradients[bi] = sin * a + cos * b;
    }
    let mut first = Engine::new(cfg.clone(), &p)?;
    let mut second = Engine::new(cfg.clone(), &rotated)?;
    first.gradients.write(&gradients)?;
    second.gradients.write(&rotated_gradients)?;
    for step in 0..4 {
        first.update(step)?;
        second.update(step)?;
    }
    let original = first.parameters.read()?;
    let actual = second.parameters.read()?;
    layout::validate_state(
        &cfg,
        &actual,
        &second.first_moment.read()?,
        &second.second_moment.read()?,
    )?;
    for [ar, bi, _, _, _] in map {
        let (ar, bi) = (ar as usize, bi as usize);
        assert!((actual[ar] - (cos * original[ar] - sin * original[bi])).abs() < 4e-8);
        assert!((actual[bi] - (sin * original[ar] + cos * original[bi])).abs() < 4e-8);
    }
    Ok(())
}
