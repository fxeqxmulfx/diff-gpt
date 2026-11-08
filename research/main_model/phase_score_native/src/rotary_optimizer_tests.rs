//! Independent frequency-group AdamW oracle and cross-group isolation checks.
use crate::{Config, Engine, Result, layout, optimizer::Optimizer, schedule::Schedule};

fn cfg(width: usize, heads: usize, layers: usize, arm: &str, budget: usize) -> Config {
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
        warmup: budget * 2 / 100,
        phase_optimizer: false,
        architecture: arm.into(),
    }
}
fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

#[test]
#[ignore = "requires CUDA"]
fn frequency_group_matches_fp64_and_does_not_change_original_clipping() -> Result<()> {
    let mut checks = 0;
    for (width, heads, layers, budget) in [(64, 4, 2, 20000), (128, 1, 1, 100000)] {
        let base_cfg = cfg(width, heads, layers, "complex-mhc", budget);
        let base_parameters = layout::initialize_model(&base_cfg, 19);
        let base_n = base_parameters.len();
        let mut control = Engine::new(base_cfg, &base_parameters)?;
        let mut control_optimizer = Optimizer::new(&control, budget)?;
        let zero_base = vec![0.; base_n];
        for arm in ["complex-mhc-lerope", "complex-mhc-log-lerope"] {
            let candidate_cfg = cfg(width, heads, layers, arm, budget);
            let mut initial = layout::initialize_model(&candidate_cfg, 19);
            for (band, value) in initial[base_n..].iter_mut().enumerate() {
                *value = 0.04 * (band as f32 - 2.);
            }
            let n = initial.len();
            let mut e = Engine::new(candidate_cfg, &initial)?;
            let mut optimizer = Optimizer::new(&e, budget)?;
            let before_guard = optimizer.save(&e)?;
            assert!(e.update(0).is_err());
            assert!(same_bits(&before_guard, &optimizer.save(&e)?));
            let schedule = Schedule::new(budget)?;
            for (base_scale, frequency_scale) in [
                (0., 0.),
                (1e-8, 1e-8),
                (1e8, 1e-8),
                (1e-8, 1e8),
                (1e8, 1e8),
                (0., 1e8),
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
                            if i < base_n {
                                ((i % 29) as f32 - 14.) * base_scale / 17.
                            } else {
                                ((i - base_n) % 7) as f32 * frequency_scale / 5.
                                    - frequency_scale * 0.6
                            }
                        })
                        .collect();
                    let mut m = vec![0.; n];
                    let mut v = vec![0.; n];
                    if step > 1 {
                        for i in base_n..n {
                            m[i] = ((i - base_n) % 3) as f32 * 0.0003 - 0.0003;
                            v[i] = 0.00002 + ((i - base_n) % 2) as f32 * 0.00001;
                        }
                    }
                    e.parameters.write(&initial)?;
                    e.first_moment.write(&m)?;
                    e.second_moment.write(&v)?;
                    e.gradients.write(&gradient)?;
                    control.parameters.write(&base_parameters)?;
                    control.first_moment.write(&zero_base)?;
                    control.second_moment.write(&zero_base)?;
                    control.gradients.write(&gradient[..base_n])?;
                    optimizer.step(&e, step)?;
                    control_optimizer.step(&control, step)?;
                    let actual = optimizer.save(&e)?;
                    let baseline = control_optimizer.save(&control)?;
                    for part in 0..3 {
                        assert!(
                            same_bits(
                                &actual[part * n..part * n + base_n],
                                &baseline[part * base_n..(part + 1) * base_n]
                            ),
                            "{arm} step{step}: frequency group changed original weights/moments"
                        );
                    }
                    let denominator = gradient[base_n..]
                        .iter()
                        .map(|&x| f64::from(x).powi(2))
                        .sum::<f64>()
                        .sqrt()
                        .max(1.);
                    let rate = f64::from(schedule.rate(step));
                    let t = (step + 1) as f64;
                    for i in base_n..n {
                        let g = f64::from(gradient[i]) / denominator;
                        let first = 0.9 * f64::from(m[i]) + 0.1 * g;
                        let second = 0.95 * f64::from(v[i]) + 0.05 * g * g;
                        let parameter = f64::from(initial[i])
                            - rate * (first / (1. - 0.9f64.powf(t)))
                                / ((second / (1. - 0.95f64.powf(t))).sqrt() + 1e-8);
                        let cancellation =
                            4. * f64::from(f32::EPSILON) * (f64::from(m[i]).abs() + 0.1 * g.abs());
                        for (offset, expected, floor) in [
                            (0, parameter, 2e-8),
                            (n, first, 1e-12 + cancellation),
                            (2 * n, second, 1e-18),
                        ] {
                            let have = f64::from(actual[offset + i]);
                            let tolerance = floor + 4e-6 * expected.abs();
                            assert!(
                                (have - expected).abs() <= tolerance,
                                "{arm} step{step} base{base_scale} frequency{frequency_scale} offset{offset} band{}: {have} vs {expected}, bound {tolerance}",
                                i - base_n
                            );
                        }
                        if frequency_scale == 0. && step <= 1 {
                            assert_eq!(
                                actual[i].to_bits(),
                                initial[i].to_bits(),
                                "unexpected frequency weight decay"
                            );
                        }
                    }
                    checks += 1;
                }
            }
        }
    }
    assert_eq!(checks, 192);
    eprintln!(
        "192 separate-frequency optimizer cases passed; original group states bit-exact to ungated control"
    );
    Ok(())
}
