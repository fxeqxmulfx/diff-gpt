use crate::{
    Config, Engine, Result, data, hash, layout, metrics, optimizer::Optimizer, reference_optimizer,
    reference_schedule, runner, schedule::Schedule,
};
use serde_json::{Value, json};
use std::{fs, path::Path};

fn repository() -> &'static Path {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../.."))
}
fn cfg(width: usize, heads: usize, layers: usize, mhc: bool, budget: usize) -> Config {
    Config {
        width,
        heads,
        layers,
        vocab: 11,
        max_batch: 3,
        max_length: 4,
        normalized: true,
        ssmax: true,
        gain: heads as f32 / width as f32,
        lr: 0.001,
        decay: 0.1,
        warmup: budget * 2 / 100,
        architecture: if mhc { "complex-mhc" } else { "complex-linear" }.into(),
        phase_optimizer: false,
    }
}
fn reference_rate(budget: usize, step: usize) -> f64 {
    let warmup = budget / 50;
    let start = budget * 4 / 5;
    if step < warmup {
        0.001 * step as f64 / warmup as f64
    } else if step <= start {
        0.001
    } else {
        let angle =
            std::f64::consts::FRAC_PI_2 * (step - start) as f64 / (budget - 1 - start) as f64;
        1e-8 + (0.001 - 1e-8) * angle.cos().powi(2)
    }
}
#[test]
fn schedules_cover_all_frozen_budgets_and_exact_json_roundtrips() -> Result<()> {
    assert_eq!(
        reference_schedule::Schedule::parse(reference_schedule::ARM)?.name(),
        "wsd-cosine20"
    );
    for budget in [20000, 30000, 100000] {
        let schedule = Schedule::new(budget)?;
        for t in 0..budget {
            let actual = schedule.rate(t);
            assert!((actual as f64 - reference_rate(budget, t)).abs() < 1e-10);
            let encoded = serde_json::to_string(&json!(actual))?;
            let decoded: Value = serde_json::from_str(&encoded)?;
            assert_eq!(decoded.as_f64(), Some(actual as f64));
            if budget == 100000 {
                let old = reference_schedule::Schedule::parse(reference_schedule::ARM)?;
                assert_eq!(actual.to_bits(), old.rate(0.001, t).to_bits());
            }
        }
        assert_eq!(schedule.rate(0), 0.);
        assert_eq!(schedule.rate(schedule.warmup), 0.001);
        assert_eq!(schedule.rate(budget - 1), 1e-8);
    }
    assert!(Schedule::new(100001).is_err());
    Ok(())
}
#[test]
fn optimizer_owns_every_parameter_and_all_router_coordinates_once() -> Result<()> {
    for (width, heads, layers) in [
        (64, 4, 2),
        (128, 4, 1),
        (128, 4, 2),
        (128, 4, 6),
        (128, 1, 1),
    ] {
        for mhc in [false, true] {
            let c = cfg(width, heads, layers, mhc, 100000);
            let owners = Optimizer::owners(&c)?;
            let mut seen = vec![0; layout::total_count(&c)];
            for &[a, b, nb, aa, kind] in &owners {
                seen[a as usize] += 1;
                seen[b as usize] += 1;
                if kind == 0 {
                    seen[nb as usize] += 1;
                    seen[aa as usize] += 1;
                }
            }
            assert!(seen.iter().all(|x| *x == 1));
            assert_eq!(owners.len() * 2, layout::independent_count(&c));
            let base = layout::pairs(&c);
            assert_eq!(&owners[..base.len()], base);
            let extras = &owners[base.len()..];
            assert_eq!(
                extras.len() * 2,
                if mhc {
                    2 * layers * (96 * width + 27)
                } else {
                    0
                }
            );
            for (i, &[a, b, nb, aa, kind]) in extras.iter().enumerate() {
                assert_eq!(
                    [a, b, nb, aa, kind],
                    [
                        (layout::base_count(&c) + 2 * i) as u32,
                        (layout::base_count(&c) + 2 * i + 1) as u32,
                        0,
                        0,
                        1
                    ]
                );
            }
        }
    }
    Ok(())
}
#[test]
fn quartic_initialization_sampler_and_optimized_kernel_provenance() -> Result<()> {
    let manifest = runner::suite();
    let row = manifest["tasks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["name"] == "arithmetic-sum-pow4")
        .unwrap();
    let task = data::build(row)?;
    let c = runner::config(row, &task, "main")?;
    assert_eq!(task.splits["train"].len(), 4704);
    let protocol: Value = serde_json::from_slice(&fs::read(
        repository().join("runs/main-model-quartic-wsd-lr001-01/protocol.json"),
    )?)?;
    // Fixed exact integer products deliberately change PTX. Pin this candidate
    // separately from the selected packed-kernel optimizer/data baseline.
    assert_eq!(
        protocol["ptx_sha256"],
        "9b3a413469e02a32037b76d404ec5df8c59aa63f012b74020ef5b2404a4227c9"
    );
    assert_eq!(
        runner::ptx_hash(),
        "0bcd8d2b9660bc70b462cd64d6bc173ee8e3b5bfb2dc428815ed32cd8c77a29e"
    );
    for seed in 0..3 {
        let initial: Vec<u8> = layout::initialize_model(&c, seed)
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        let old = fs::read(repository().join(format!(
            "runs/main-model-quartic-wsd-lr001-01/adamw-wsd-cosine20-warmup2000/seed{seed}/initial.f32")))?;
        assert_eq!(initial, old);
    }
    let (order, edges) = runner::order(4704, 256, 100000, 0);
    let old: Value = serde_json::from_slice(&fs::read(
        repository().join("runs/main-model-mhc-02/main/arithmetic-sum-pow4/seed0/protocol.json"),
    )?)?;
    let bytes: Vec<_> = order.iter().flat_map(|x| x.to_le_bytes()).collect();
    assert_eq!(hash(&bytes), old["sampler_sha256"]);
    assert_eq!(edges[19], 4704);
    assert_eq!(edges[19] - edges[18], 96);
    Ok(())
}
#[test]
#[ignore = "requires CUDA"]
fn clipped_adamw_including_router_only_gradients_matches_fp64() -> Result<()> {
    for (width, heads, layers, budget) in [(64, 4, 2, 20000), (128, 1, 1, 100000)] {
        for mhc in [false, true] {
            let c = cfg(width, heads, layers, mhc, budget);
            let initial = layout::initialize_model(&c, 0);
            let n = initial.len();
            let base = layout::base_count(&c);
            let owners = Optimizer::owners(&c)?;
            let mut e = Engine::new(c, &initial)?;
            let mut o = Optimizer::new(&e, budget)?;
            let schedule = Schedule::new(budget)?;
            for (scale, router_only) in [(0., false), (1e-8, false), (1e8, false), (1e8, true)] {
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
                            if router_only && i < base {
                                0.
                            } else {
                                ((i % 29) as f32 - 14.) * scale / 17.
                            }
                        })
                        .collect();
                    let mut m = vec![0f32; n];
                    let mut v = vec![0f32; n];
                    if step > 1 {
                        for &[a, b, nb, aa, kind] in &owners {
                            m[a as usize] = 0.0003;
                            m[b as usize] = -0.0007;
                            v[a as usize] = 0.00002;
                            v[b as usize] = 0.00003;
                            if kind == 0 {
                                m[aa as usize] = m[a as usize];
                                m[nb as usize] = -m[b as usize];
                                v[aa as usize] = v[a as usize];
                                v[nb as usize] = v[b as usize];
                            }
                        }
                    }
                    e.parameters.write(&initial)?;
                    e.first_moment.write(&m)?;
                    e.second_moment.write(&v)?;
                    e.gradients.write(&gradient)?;
                    let g: Vec<[f64; 2]> = owners
                        .iter()
                        .map(|&[a, b, nb, aa, kind]| {
                            if kind == 0 {
                                [
                                    (gradient[a as usize] + gradient[aa as usize]) as f64,
                                    (gradient[b as usize] - gradient[nb as usize]) as f64,
                                ]
                            } else {
                                [gradient[a as usize] as f64, gradient[b as usize] as f64]
                            }
                        })
                        .collect();
                    let denominator = g
                        .iter()
                        .flatten()
                        .map(|x| x * x)
                        .sum::<f64>()
                        .sqrt()
                        .max(1.);
                    let rate = reference_rate(budget, step);
                    let t = (step + 1) as f64;
                    o.step(&e, step)?;
                    let got = o.save(&e)?;
                    for (owner, gradient) in owners.iter().zip(&g) {
                        for component in 0..2 {
                            let index = owner[component] as usize;
                            let gg = gradient[component] / denominator;
                            let mm = 0.9 * m[index] as f64 + 0.1 * gg;
                            let vv = 0.95 * v[index] as f64 + 0.05 * gg * gg;
                            let pp = initial[index] as f64 * (1. - rate * 0.1)
                                - rate * (mm / (1. - 0.9f64.powf(t)))
                                    / ((vv / (1. - 0.95f64.powf(t))).sqrt() + 1e-8);
                            // Momentum can cancel almost exactly. Its rounding
                            // envelope depends on the input terms, not only on
                            // the tiny result. Eight FP32 unit roundoffs cover
                            // norm conversion, division, coefficient rounding,
                            // subtraction and FMA without hiding scale errors.
                            let momentum_roundoff = 4.
                                * f32::EPSILON as f64
                                * ((m[index] as f64).abs() + 0.1 * gg.abs());
                            for (offset, want, absolute) in [
                                (0, pp, 2e-8),
                                (n, mm, 1e-12 + momentum_roundoff),
                                (2 * n, vv, 1e-18),
                            ] {
                                let have = got[offset + index] as f64;
                                assert!(
                                    (have - want).abs() <= absolute + 4e-6 * want.abs(),
                                    "mhc{mhc} step{step} router_only{router_only} offset{offset} index{index}: {have} vs {want}"
                                );
                            }
                        }
                    }
                    if step == 0 {
                        assert_eq!(&got[..n], initial);
                    }
                }
            }
        }
    }
    Ok(())
}
#[test]
#[ignore = "requires CUDA"]
fn full_runner_matches_frozen_quartic_optimizer_trajectory_and_checkpoint() -> Result<()> {
    let manifest = runner::suite();
    let row = manifest["tasks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["name"] == "arithmetic-sum-pow4")
        .unwrap();
    let task = data::build(row)?;
    let cfg = runner::config(row, &task, "main")?;
    let initial = layout::initialize_model(&cfg, 0);
    let mut current = Engine::new(cfg.clone(), &initial)?;
    let mut reference = Engine::new(cfg, &initial)?;
    let mut opt = Optimizer::new(&current, reference_optimizer::BUDGET)?;
    let mut old =
        reference_optimizer::Optimizer::new(&reference, &initial, reference_schedule::ARM)?;
    let (order, edges) = runner::order(4704, 256, 19, 0);
    let mut packed = metrics::Packed::default();
    for step in 0..19 {
        let rows = &order[edges[step]..edges[step + 1]];
        packed.fill(&task.splits["train"], rows);
        for e in [&mut current, &mut reference] {
            e.forward(&packed.tokens, &packed.labels, rows.len(), packed.length)?;
            e.backward(rows.len(), packed.length)?;
        }
        opt.step(&current, step)?;
        old.step(&reference, step)?;
    }
    assert_eq!(opt.save(&current)?, old.save(&reference)?);
    let dir = repository()
        .join("runs/main-model-quartic-wsd-lr001-01/adamw-wsd-cosine20-warmup2000/seed0");
    let bytes = fs::read(dir.join("state.f32"))?;
    let state: Vec<_> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|x| f32::from_le_bytes(*x))
        .collect();
    opt.restore(&mut current, &state)?;
    assert_eq!(opt.save(&current)?, state);
    let result: Value = serde_json::from_slice(&fs::read(dir.join("result.json"))?)?;
    for split in ["train", "validation"] {
        let got = metrics::evaluate(&mut current, &task.splits[split], split != "train")?;
        assert_eq!(got["exact"], result["endpoint"][split]["exact"]);
        assert!(
            (got["loss"].as_f64().unwrap() - result["endpoint"][split]["loss"].as_f64().unwrap())
                .abs()
                < 1e-7
        );
    }
    Ok(())
}
