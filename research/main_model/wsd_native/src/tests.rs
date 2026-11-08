use crate::{
    Result, arithmetic_data, engine::Engine, hash, layout, optimizer::Optimizer, runner,
    schedule::Schedule,
};
use serde_json::{Value, json};
use std::{fs, path::Path};

fn repository() -> &'static Path {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../.."))
}
fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|x| f32::from_le_bytes(*x))
        .collect()
}
fn reference_rate(t: usize) -> f64 {
    if t < 2000 {
        0.002 * t as f64 / 2000.
    } else if t <= 80000 {
        0.002
    } else {
        let angle = std::f64::consts::FRAC_PI_2 * (t - 80000) as f64 / 19999.;
        1e-8 + (0.002 - 1e-8) * angle.cos().powi(2)
    }
}
#[test]
fn schedule_and_frozen_data_initialization_and_samples() -> Result<()> {
    assert_eq!(runner::ARMS, ["adamw-wsd-cosine20-warmup2000"]);
    let schedule = Schedule::parse(runner::ARMS[0])?;
    for step in 0..100000 {
        let actual = schedule.rate(0.002, step) as f64;
        let expected = reference_rate(step);
        assert!((actual - expected).abs() <= 2e-10);
    }
    assert_eq!(schedule.rate(0.002, 0), 0.);
    assert_eq!(schedule.warmup, 2000);
    assert_eq!(runner::config().warmup, 2000);
    assert_eq!(schedule.rate(0.002, 1000), 0.001);
    assert_eq!(schedule.rate(0.002, 2000), 0.002);
    assert_eq!(schedule.rate(0.002, 80000), 0.002);
    assert_eq!(schedule.rate(0.002, 99999), 1e-8);
    let data = arithmetic_data::arithmetic(&runner::row())?;
    assert_eq!((data.train.len(), data.validation.len()), (4704, 4705));
    for seed in 0..3 {
        let initial = layout::initialize_model(&runner::config(), seed);
        let old = fs::read(repository().join(format!(
            "runs/quartic-muonmd/adamw-linear/seed{seed}/initial.f32"
        )))?;
        assert_eq!(runner::float_bytes(&initial), old);
        let (order, edges) = runner::order(4704, 100000, seed);
        assert_eq!(edges[19] - edges[18], 96);
        assert_eq!(edges[19], 4704);
        assert_eq!(edges[100000], 5263 * 4704 + 3 * 256);
        if seed == 0 {
            let main: Value = runner::read(
                &repository()
                    .join("runs/main-model-mhc-02/main/arithmetic-sum-pow4/seed0/protocol.json"),
            )?;
            assert_eq!(main["initial_parameters_sha256"], hash(&old));
            let bytes: Vec<_> = order.iter().flat_map(|x| x.to_le_bytes()).collect();
            assert_eq!(main["sampler_sha256"], hash(&bytes));
        }
    }
    Ok(())
}
#[test]
fn archive_tampering_is_rejected_without_gpu() -> Result<()> {
    let root = std::env::temp_dir().join(format!("main-wsd-audit-{}", std::process::id()));
    fs::create_dir(&root)?;
    runner::archive(&root)?;
    let audit = runner::audit(&root)?;
    assert_eq!(audit["completed"], 0);
    assert_eq!(audit["expected"], 3);
    assert!(audit["integrity_failures"].as_array().unwrap().is_empty());
    fs::write(
        root.join("source/research/main_model/WSD_PLAN.txt"),
        b"changed",
    )?;
    assert!(runner::audit(&root).is_err());
    fs::remove_dir_all(root)?;
    Ok(())
}
#[test]
fn first_crossing_and_preceding_observation_are_recorded() {
    let history: Vec<_>=[(0,0),(19,4657),(38,4658),(57,4000)].into_iter().map(|(step,hits)|
        json!({"step":step,"epochs":step as f64/19.,"validation":{"exact":hits,"accuracy":hits as f64/4705.}})).collect();
    let result = runner::summary(&history);
    assert_eq!(result["first99"]["step"], 38);
    assert_eq!(result["previous_observed_epochs"], 1.);
    assert_eq!(result["endpoint"]["step"], 57);
}
#[test]
#[ignore = "requires CUDA"]
fn corrected_main_checkpoint_replays_without_model_changes() -> Result<()> {
    let root = repository().join("runs/main-model-mhc-02/main/arithmetic-sum-pow4/seed0");
    let bytes = fs::read(root.join("checkpoint.bin"))?;
    let cp = runner::read(&root.join("checkpoint.json"))?;
    assert_eq!(cp["state_sha256"], hash(&bytes));
    let state = floats(&bytes);
    let cfg = runner::config();
    let n = layout::total_count(&cfg);
    let mut e = Engine::new(cfg, &state[..n])?;
    e.first_moment.write(&state[n..2 * n])?;
    e.second_moment.write(&state[2 * n..])?;
    let data = arithmetic_data::arithmetic(&runner::row())?;
    let expected = runner::read(&root.join("result.json"))?;
    for (split, rows) in [("train", &data.train), ("validation", &data.validation)] {
        let got = runner::evaluate(&mut e, rows)?;
        let reference = &expected["endpoint"][split];
        assert_eq!(got["exact"], reference["exact"]);
        assert!((got["loss"].as_f64().unwrap() - reference["loss"].as_f64().unwrap()).abs() < 1e-7);
    }
    let o = Optimizer::new(&e, &state[..n], runner::ARMS[0])?;
    assert_eq!(o.save(&e)?, state);
    Ok(())
}
#[test]
#[ignore = "requires CUDA"]
fn clipped_adamw_matches_independent_fp64_at_schedule_boundaries() -> Result<()> {
    let cfg = runner::config();
    let initial = layout::initialize_model(&cfg, 0);
    let pairs = layout::pairs(&cfg);
    let n = initial.len();
    let mut e = Engine::new(cfg, &initial)?;
    let mut o = Optimizer::new(&e, &initial, runner::ARMS[0])?;
    for scale in [0., 1e-8, 1., 1e8] {
        for step in [0, 1, 1999, 2000, 79999, 80000, 80001, 89999, 90000, 99999] {
            let gradient: Vec<f32> = (0..n)
                .map(|i| ((i % 29) as f32 - 14.) * scale / 17.)
                .collect();
            let mut m = vec![0f32; n];
            let mut v = vec![0f32; n];
            if step > 1 {
                for &[ar, bi, br, ai, kind] in &pairs {
                    m[ar as usize] = 0.0003;
                    m[bi as usize] = -0.0007;
                    v[ar as usize] = 0.00002;
                    v[bi as usize] = 0.00003;
                    if kind == 0 {
                        m[ai as usize] = m[ar as usize];
                        m[br as usize] = -m[bi as usize];
                        v[ai as usize] = v[ar as usize];
                        v[br as usize] = v[bi as usize];
                    }
                }
            }
            e.parameters.write(&initial)?;
            e.first_moment.write(&m)?;
            e.second_moment.write(&v)?;
            e.gradients.write(&gradient)?;
            let contracted: Vec<[f64; 2]> = pairs
                .iter()
                .map(|&[ar, bi, br, ai, kind]| {
                    if kind == 0 {
                        [
                            (gradient[ar as usize] + gradient[ai as usize]) as f64,
                            (gradient[bi as usize] - gradient[br as usize]) as f64,
                        ]
                    } else {
                        [gradient[ar as usize] as f64, gradient[bi as usize] as f64]
                    }
                })
                .collect();
            let denominator = contracted
                .iter()
                .flatten()
                .map(|g| g * g)
                .sum::<f64>()
                .sqrt()
                .max(1.);
            let rate = reference_rate(step);
            let t = (step + 1) as f64;
            o.step(&e, step)?;
            let got = o.save(&e)?;
            for (pair, g) in pairs.iter().zip(&contracted) {
                for component in 0..2 {
                    let index = pair[component] as usize;
                    let g = g[component] / denominator;
                    let mm = 0.9 * m[index] as f64 + 0.1 * g;
                    let vv = 0.95 * v[index] as f64 + 0.05 * g * g;
                    let pp = initial[index] as f64 * (1. - rate * 0.1)
                        - rate * (mm / (1. - 0.9f64.powf(t)))
                            / ((vv / (1. - 0.95f64.powf(t))).sqrt() + 1e-8);
                    for (offset, want, absolute) in
                        [(0, pp, 2e-8), (n, mm, 1e-12), (2 * n, vv, 1e-18)]
                    {
                        let have = got[offset + index] as f64;
                        assert!(
                            (have - want).abs() <= absolute + 4e-6 * want.abs(),
                            "step{step} scale{scale} index{index} offset{offset}: {have} vs {want}"
                        );
                    }
                }
            }
            if step == 0 {
                assert_eq!(&got[..n], initial);
            }
        }
    }
    Ok(())
}
#[test]
#[ignore = "requires CUDA"]
fn checkpoint_resume_matches_the_real_short_tail_and_rejects_unclean_state() -> Result<()> {
    let initial = layout::initialize_model(&runner::config(), 1);
    let mut a = Engine::new(runner::config(), &initial)?;
    let mut oa = Optimizer::new(&a, &initial, runner::ARMS[0])?;
    let data = arithmetic_data::arithmetic(&runner::row())?;
    let (order, edges) = runner::order(4704, 19, 1);
    for step in 0..18 {
        runner::batch(&mut a, &data.train, &order[edges[step]..edges[step + 1]])?;
        a.backward(256, 2)?;
        oa.step(&a, step)?;
    }
    let root = std::env::temp_dir().join(format!("main-wsd-resume-{}", std::process::id()));
    fs::create_dir(&root)?;
    runner::checkpoint(&a, &oa, &root, 18, edges[18], &[json!({"step":18})])?;
    let mut b = Engine::new(runner::config(), &initial)?;
    let mut ob = Optimizer::new(&b, &initial, runner::ARMS[0])?;
    assert_eq!(runner::restore(&mut b, &mut ob, &root)?.0, 18);
    let tail = &order[edges[18]..edges[19]];
    assert_eq!(tail.len(), 96);
    for (e, o) in [(&mut a, &mut oa), (&mut b, &mut ob)] {
        runner::batch(e, &data.train, tail)?;
        e.backward(tail.len(), 2)?;
        o.step(e, 18)?;
    }
    assert_eq!(oa.save(&a)?, ob.save(&b)?);
    runner::record(
        &root.join("execution.json"),
        &json!({"clean":false,"durable_completed":18,"possible_completed":9500}),
    )?;
    assert!(runner::restore(&mut b, &mut ob, &root).is_err());
    fs::remove_dir_all(root)?;
    Ok(())
}
