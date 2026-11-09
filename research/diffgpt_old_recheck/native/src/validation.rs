use crate::{
    Engine, Result, config, initialize,
    oracle::{self, Jet},
    random::Random,
    runner,
    schedule_free::Optimizer,
};
use serde_json::json;
use std::{fs, path::Path};

pub fn run(dir: &Path) -> Result<()> {
    if dir.exists() {
        return Err("validation directory exists".into());
    }
    fs::create_dir_all(dir)?;
    let cfg = config(4);
    let p = initialize(31415);
    let mut e = Engine::new(cfg.clone(), &p)?;
    let tokens = [31, 42, 15, 15, 17, 208, 5, 300];
    let labels = [u32::MAX, 215, u32::MAX, 93, u32::MAX, 511, u32::MAX, 9];
    e.forward(&tokens, &labels, 4, 2)?;
    e.backward(4, 2)?;
    let gradient = e.gradients.read()?;
    let logits = e.logits.read()?;
    let loss = e.losses(8)?.iter().map(|&x| f64::from(x)).sum::<f64>() / 4.;
    let mut params: Vec<Jet> = p.iter().map(|&v| Jet::from(f64::from(v))).collect();
    let reference = oracle::model(&cfg, &params, &tokens, 4, 2);
    let max_logit_error = logits
        .iter()
        .zip(&reference)
        .map(|(&a, b)| (f64::from(a) - b.v).abs())
        .fold(0., f64::max);
    let reference_loss = oracle::cross_entropy(&reference, &labels, 512).v;
    if max_logit_error > 1e-4 || (loss - reference_loss).abs() > 1e-5 {
        return Err(format!(
            "forward reference mismatch logits={max_logit_error}, loss={}",
            (loss - reference_loss).abs()
        )
        .into());
    }
    let w = 128;
    let h = 341;
    let base = 512 * w;
    let blocks = [
        ("embedding", 0, base),
        ("q", base, w * w),
        ("k", base + w * w, w * w),
        ("v", base + 2 * w * w, w * w),
        ("attention_out", base + 3 * w * w, w * w),
        ("swiglu_gate", base + 4 * w * w, h * w),
        ("swiglu_value", base + 4 * w * w + h * w, h * w),
        ("ffn_out", base + 4 * w * w + 2 * h * w, w * h),
    ];
    let mut directions = Vec::new();
    let mut rng = Random::new(781);
    for (name, offset, count) in blocks {
        for _ in 0..3 {
            for v in &mut params {
                v.d = 0.;
            }
            let mut actual = 0.;
            for i in offset..offset + count {
                let d = if rng.below(2) == 0 { -1. } else { 1. } / (count as f64).sqrt();
                params[i].d = d;
                actual += f64::from(gradient[i]) * d;
            }
            let output = oracle::model(&cfg, &params, &tokens, 4, 2);
            let expected = oracle::cross_entropy(&output, &labels, 512).d;
            let error = (actual - expected).abs();
            let tolerance = 2e-6 + 0.003 * expected.abs();
            directions.push(json!({"block":name,"gpu":actual,"cpu_ad":expected,"error":error,"tolerance":tolerance,"passed":error<=tolerance}));
        }
    }
    runner::write(
        &dir.join("model.json"),
        &json!({"max_logit_error":max_logit_error,"loss_error":(loss-reference_loss).abs(),"directions":directions}),
    )?;
    if directions.iter().any(|v| v["passed"] != true) {
        return Err("model AD validation failed; inspect model.json".into());
    }

    // Independent reference holds x,z,v explicitly; GPU uses packed y,z,v.
    let mut opt = Optimizer::new(&mut e)?;
    let mut x: Vec<f64> = p.iter().map(|&v| f64::from(v)).collect();
    let mut z = x.clone();
    let mut v = vec![0.; p.len()];
    let mut total = 0.;
    let mut max_state_error = 0_f64;
    for t in 1..=24 {
        let g: Vec<f32> = (0..p.len())
            .map(|i| {
                if i % 19 == 0 {
                    0.
                } else {
                    ((i % 31) as f64 - 15.) as f32 * 0.0001 + t as f32 * 0.00001
                }
            })
            .collect();
        e.gradients.write(&g)?;
        let rate = 0.03 * (f64::from(t) / 10.).min(1.);
        total += rate * rate;
        let c = rate * rate / total;
        for i in 0..p.len() {
            let query = 0.1 * z[i] + 0.9 * x[i];
            let gi = f64::from(g[i]);
            v[i] = 0.98 * v[i] + 0.02 * gi * gi;
            let update = gi / ((v[i] / (1. - 0.98_f64.powi(t))).sqrt() + 1e-8) + query;
            z[i] -= rate * update;
            x[i] += c * (z[i] - x[i]);
        }
        opt.step(&e)?;
        if t % 3 == 0 {
            opt.mode(&e, false)?;
            assert!(opt.step(&e).is_err());
            let actual = runner::state(&e)?;
            for i in 0..p.len() {
                max_state_error = max_state_error
                    .max((f64::from(actual[i]) - x[i]).abs())
                    .max((f64::from(actual[p.len() + i]) - z[i]).abs())
                    .max((f64::from(actual[2 * p.len() + i]) - v[i]).abs());
            }
            opt.mode(&e, true)?;
        }
    }
    if max_state_error > 1e-5 {
        return Err(format!("ScheduleFree reference error {max_state_error}").into());
    }
    runner::write(
        &dir.join("optimizer.json"),
        &json!({"passed":true,"updates":24,"coordinates":p.len(),"explicit_x_z_v_reference":true,"max_absolute_error":max_state_error}),
    )?;

    // Actual model updates, serialized average-state restart, and exact continuation.
    e.parameters.write(&p)?;
    let mut opt = Optimizer::new(&mut e)?;
    for _ in 0..4 {
        e.forward(&tokens, &labels, 4, 2)?;
        e.backward(4, 2)?;
        opt.step(&e)?;
    }
    opt.mode(&e, false)?;
    let info = runner::save(&e, &opt, dir, b"")?;
    let values = runner::load_state(dir, &info)?;
    opt.mode(&e, true)?;
    e.forward(&tokens, &labels, 4, 2)?;
    e.backward(4, 2)?;
    opt.step(&e)?;
    let expected = runner::bytes(&runner::state(&e)?);
    let mut restored = Engine::new(cfg, &values[..p.len()])?;
    runner::restore(&mut restored, &values)?;
    let mut opt = Optimizer::restore(&restored, &info["optimizer"])?;
    opt.mode(&restored, true)?;
    restored.forward(&tokens, &labels, 4, 2)?;
    restored.backward(4, 2)?;
    opt.step(&restored)?;
    if runner::bytes(&runner::state(&restored)?) != expected {
        return Err("serialized continuation differs".into());
    }
    let ptx = crate::ptx();
    if ptx
        .as_array()
        .unwrap()
        .iter()
        .any(|m| m["gpu_fp64"] != false)
    {
        return Err("GPU FP64 instruction present".into());
    }
    runner::write(
        &dir.join("validation.json"),
        &json!({"passed":true,"model_ad_directions":directions.len(),"max_logit_error":max_logit_error,
        "schedule_free_reference_updates":24,"schedule_free_max_absolute_error":max_state_error,"serialized_continuation_bit_exact":true,"gpu_fp64":false,"ptx":ptx}),
    )?;
    println!(
        "validated24model AD directions,24ScheduleFree updates,exact serialized continuation,no GPU FP64"
    );
    Ok(())
}
