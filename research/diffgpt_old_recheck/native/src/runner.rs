use crate::{Engine, Result, config, hash, initialize, random::Random, schedule_free::Optimizer};
use diffgpt_axial_data_native::{self as data, Example};
use diffgpt_mash_native::Tokenizer;
use serde_json::{Value, json};
use std::{
    fs,
    io::{BufWriter, Write},
    path::Path,
    time::Instant,
};

pub const BATCH: usize = 512;
pub const BUDGET: usize = 5000;
pub fn read(path: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}
pub fn write(path: &Path, value: &Value) -> Result<()> {
    let mut out = BufWriter::with_capacity(1 << 20, fs::File::create_new(path)?);
    serde_json::to_writer_pretty(&mut out, value)?;
    out.write_all(b"\n")?;
    out.flush()?;
    out.get_ref().sync_all()?;
    Ok(())
}
pub fn state(e: &Engine) -> Result<Vec<f32>> {
    let mut values = e.parameters.read()?;
    values.extend(e.first_moment.read()?);
    let v = e.second_moment.read()?;
    if v.iter().any(|&x| x < 0.) {
        return Err("negative second moment".into());
    }
    values.extend(v);
    if values.iter().any(|x| !x.is_finite()) {
        return Err("nonfinite saved state".into());
    }
    Ok(values)
}
pub fn restore(e: &mut Engine, values: &[f32]) -> Result<()> {
    let n = e.parameters.len();
    if values.len() != 3 * n
        || values.iter().any(|x| !x.is_finite())
        || values[2 * n..].iter().any(|&x| x < 0.)
    {
        return Err("invalid restored state".into());
    }
    e.parameters.write(&values[..n])?;
    e.first_moment.write(&values[n..2 * n])?;
    e.second_moment.write(&values[2 * n..])?;
    Ok(())
}
pub fn bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
pub fn save(e: &Engine, opt: &Optimizer, dir: &Path, history: &[u8]) -> Result<Value> {
    let values = bytes(&state(e)?);
    let basename = format!("checkpoint-{:05}", opt.completed);
    let path = dir.join(format!("{basename}.bin"));
    let mut out = BufWriter::with_capacity(1 << 20, fs::File::create_new(&path)?);
    out.write_all(&values)?;
    out.flush()?;
    out.get_ref().sync_all()?;
    let info = json!({"state_sha256":hash(&values),"optimizer":opt.state(),"file":path.file_name().unwrap().to_string_lossy()});
    write(&dir.join(format!("{basename}.json")), &info)?;
    let mut out = BufWriter::with_capacity(
        1 << 20,
        fs::File::create_new(dir.join(format!("history-{:05}.jsonl", opt.completed)))?,
    );
    out.write_all(history)?;
    out.flush()?;
    out.get_ref().sync_all()?;
    Ok(info)
}
pub fn load_state(dir: &Path, info: &Value) -> Result<Vec<f32>> {
    let bytes = fs::read(dir.join(info["file"].as_str().ok_or("missing checkpoint filename")?))?;
    if bytes.len() % 4 != 0 || info["state_sha256"] != hash(&bytes) {
        return Err("checkpoint checksum mismatch".into());
    }
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect())
}
pub fn evaluate(e: &mut Engine, examples: &[Example]) -> Result<Value> {
    let mut correct = 0;
    let mut token_correct = 0;
    let mut forecast_correct = 0;
    let mut loss = 0.;
    for chunk in examples.chunks(e.config.max_batch) {
        let x: Vec<_> = chunk
            .iter()
            .flat_map(|r| r.tokens.iter().copied())
            .collect();
        let y: Vec<_> = chunk
            .iter()
            .flat_map(|r| r.targets.iter().copied())
            .collect();
        e.forward(&x, &y, chunk.len(), 2)?;
        let predicted = e.predictions.read()?;
        let losses = e.losses(chunk.len() * 2)?;
        if losses.iter().any(|x| !x.is_finite()) {
            return Err("nonfinite evaluation loss".into());
        }
        for (i, row) in chunk.iter().enumerate() {
            let token = predicted[2 * i + 1];
            correct += usize::from(row.decoded_hit(0, token, false));
            forecast_correct += usize::from(row.decoded_hit(0, token, true));
            token_correct += usize::from(token == row.future_tokens[0]);
            loss += f64::from(losses[2 * i + 1]);
        }
    }
    Ok(
        json!({"decoded_correct":correct,"decoded_accuracy":correct as f64/examples.len() as f64,
        "token_correct":token_correct,"token_accuracy":token_correct as f64/examples.len() as f64,
        "forecast_reanchored_correct":forecast_correct,"examples":examples.len(),"loss":loss/examples.len() as f64}),
    )
}
pub fn run(root: &Path, seed: u32) -> Result<()> {
    if ![11, 17, 23].contains(&seed) {
        return Err("undeclared seed".into());
    }
    let registration = read(&root.join("registration.json"))?;
    if registration["budget_updates"] != BUDGET
        || registration["binary_sha256"] != hash(&fs::read(std::env::current_exe()?)?)
        || registration["ptx"] != crate::ptx()
        || registration["validation_passed"] != true
    {
        return Err("unvalidated or mismatched registered executable".into());
    }
    for (path, expected) in registration["source_files"]
        .as_object()
        .ok_or("missing source pins")?
    {
        if *expected != hash(&fs::read(path)?) {
            return Err(format!("source changed: {path}").into());
        }
    }
    let dir = root.join(format!("seed{seed}"));
    if dir.exists() || root.join("STOP").exists() {
        return Err("refuse existing output or STOP".into());
    }
    let data = data::build(seed, 1, 2, Tokenizer::MashHalf)?;
    let prior_dir = Path::new("runs/main-model-diffgpt-axial-01/sequence/mash-half/features1")
        .join(format!("seed{seed}"));
    let prior = read(&prior_dir.join("metadata.json"))?;
    let mut rng = Random::new(seed + 10000);
    let ids: Vec<u32> = (0..BATCH * BUDGET)
        .map(|_| rng.below(data.train.len() as u32))
        .collect();
    let sampler_sha256 = hash(&ids.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
    if prior["train_sha256"] != data::fingerprint(&data.train)
        || prior["heldout_sha256"] != data::fingerprint(&data.validation)
        || prior["sampler_sha256"] != sampler_sha256
    {
        return Err("native data/sampler differs from failed DiffGPT cohort".into());
    }
    let initial = initialize(seed);
    let mut e = Engine::new(config(BATCH), &initial)?;
    let mut opt = Optimizer::new(&mut e)?;
    fs::create_dir(&dir)?;
    let metadata = json!({"seed":seed,"tokenizer":"mash-half","features":1,"context":2,
        "budget_updates":BUDGET,"batch":BATCH,"observe":50,"width":128,"heads":1,"layers":1,"hidden":341,"parameters":initial.len(),
        "architecture":"historical_real_peri_rms_xsa_swiglu_qkrms_rope","historical_commit":"90544cdc4207f48cabc1a2acf28913b897caab83",
        "attention":"1/sqrt(128)","residual_gain":1,"optimizer":"ScheduleFreeAdamW","lr":0.03,"betas":[0.9,0.98],"weight_decay":1,"warmup":10,
        "gradient_clipping":false,"precision":"fixed_FP32_GPU","torch_bit_exact_replay":false,
        "train_sha256":data::fingerprint(&data.train),"heldout_sha256":data::fingerprint(&data.validation),"sampler_sha256":sampler_sha256,
        "initial_sha256":hash(&bytes(&initial)),"registration_sha256":hash(&fs::read(root.join("registration.json"))?)});
    write(&dir.join("metadata.json"), &metadata)?;
    let mut history = Vec::new();
    let mut first90 = None;
    let mut next_epoch = 500.;
    let start = Instant::now();
    println!("start historical MASH-half seed{seed}; fixed5000updates; no retuning");
    loop {
        opt.mode(&e, false)?;
        let train = evaluate(&mut e, &data.train)?;
        let validation = evaluate(&mut e, &data.validation)?;
        let newly_passed = first90.is_none()
            && validation["decoded_correct"].as_u64().unwrap() * 10
                >= 9 * data.validation.len() as u64;
        if newly_passed {
            first90 = Some(opt.completed);
        }
        let epoch = (opt.completed * BATCH) as f64 / data.train.len() as f64;
        let point = json!({"step":opt.completed,"epochs":epoch,"train":train,"validation":validation,"seconds":start.elapsed().as_secs_f64()});
        history.extend(serde_json::to_vec(&point)?);
        history.push(b'\n');
        let stopped = root.join("STOP").exists();
        let mut saved = None;
        if opt.completed == 0
            || newly_passed
            || epoch >= next_epoch
            || opt.completed == BUDGET
            || stopped
        {
            saved = Some(save(&e, &opt, &dir, &history)?);
            if epoch >= next_epoch {
                next_epoch += 500.;
            }
        }
        if opt.completed == BUDGET || stopped {
            let result = json!({"status":if stopped{"stopped"}else if first90.is_some(){"reached90"}else{"budget_exhausted"},
                "first90_updates":first90,"completed":opt.completed,"endpoint":point,"checkpoint":saved,"metadata":metadata,
                "final_reached90":validation["decoded_correct"].as_u64().unwrap()*10>=9*data.validation.len() as u64});
            write(&dir.join("result.json"), &result)?;
            println!(
                "finished seed{seed}: first90={first90:?}, final={:.6}%, {}updates",
                100. * validation["decoded_accuracy"].as_f64().unwrap(),
                opt.completed
            );
            return Ok(());
        }
        opt.mode(&e, true)?;
        let end = (opt.completed + 50).min(BUDGET);
        while opt.completed < end {
            let mut x = Vec::with_capacity(2 * BATCH);
            let mut y = Vec::with_capacity(2 * BATCH);
            for &id in &ids[opt.completed * BATCH..(opt.completed + 1) * BATCH] {
                x.extend(&data.train[id as usize].tokens);
                y.extend(&data.train[id as usize].targets);
            }
            e.forward(&x, &y, BATCH, 2)?;
            e.backward(BATCH, 2)?;
            opt.step(&e)?;
        }
    }
}
pub fn replay(dir: &Path) -> Result<()> {
    let result = read(&dir.join("result.json"))?;
    let seed = result["metadata"]["seed"].as_u64().ok_or("missing seed")? as u32;
    let data = data::build(seed, 1, 2, Tokenizer::MashHalf)?;
    let values = load_state(dir, &result["checkpoint"])?;
    let mut e = Engine::new(config(BATCH), &values[..262016])?;
    restore(&mut e, &values)?;
    let validation = evaluate(&mut e, &data.validation)?;
    let train = evaluate(&mut e, &data.train)?;
    if validation != result["endpoint"]["validation"] || train != result["endpoint"]["train"] {
        return Err("checkpoint metrics did not replay exactly".into());
    }
    let history = fs::read_to_string(dir.join(format!(
        "history-{:05}.jsonl",
        result["completed"].as_u64().ok_or("missing completed")?
    )))?;
    let rows: Vec<Value> = history
        .lines()
        .map(serde_json::from_str)
        .collect::<std::result::Result<_, _>>()?;
    let first = rows
        .iter()
        .find(|r| r["validation"]["decoded_correct"].as_u64().unwrap() * 10 >= 9 * 4704)
        .map(|r| r["step"].clone())
        .unwrap_or(Value::Null);
    if first != result["first90_updates"]
        || rows.len() != 101
        || rows.iter().enumerate().any(|(i, r)| r["step"] != i * 50)
    {
        return Err("history cadence or first90 mismatch".into());
    }
    write(
        &dir.join("replay.json"),
        &json!({"passed":true,"seed":seed,"exact_training_and_validation_metrics":true,"history_observations":rows.len(),
        "first90_updates":first,"final_decoded_accuracy":validation["decoded_accuracy"],"state_sha256":hash(&bytes(&values))}),
    )
}
