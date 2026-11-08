//! Matched native DiffGPT training with separately reported encoder/axis cells.
use crate::{
    Config, Engine, Result, hash, layout,
    optimizer::Optimizer,
    position::{Axes, Extension},
    random::Random,
    runner,
};
use diffgpt_axial_data_native::{self as data, Dataset, Example};
use diffgpt_mash_native::Tokenizer;
use serde_json::{Value, json};
use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::Path,
    time::Instant,
};
const BUDGET: usize = 5000;
const BATCH: usize = 512;
const OBSERVE: usize = 50;

fn read(path: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}
fn durable(path: &Path, bytes: &[u8]) -> Result<()> {
    let temporary = path.with_extension("pending");
    let mut writer = BufWriter::new(File::create(&temporary)?);
    writer.write_all(bytes)?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    fs::rename(temporary, path)?;
    File::open(path.parent().ok_or("missing parent")?)?.sync_all()?;
    Ok(())
}
fn record(path: &Path, value: &Value) -> Result<()> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    durable(path, &bytes)
}
fn checkpoint(
    engine: &Engine,
    dir: &Path,
    step: usize,
    history: &[u8],
    first90: bool,
) -> Result<()> {
    engine.cuda.sync()?;
    let p = engine.parameters.read()?;
    let m = engine.first_moment.read()?;
    let v = engine.second_moment.read()?;
    layout::validate_state(&engine.config, &p, &m, &v)?;
    let bytes: Vec<u8> = p
        .iter()
        .chain(&m)
        .chain(&v)
        .flat_map(|x| x.to_le_bytes())
        .collect();
    let metadata = json!({"completed":step,"examples":step*BATCH,"epochs":step as f64*BATCH as f64/4704.,
        "state_sha256":hash(&bytes),"history_sha256":hash(history),"source_sha256":runner::source_hash(),
        "ptx_sha256":runner::ptx_hash(),"layout":"FP32 parameters,first moment,second moment"});
    durable(&dir.join("history.jsonl"), history)?;
    durable(&dir.join("checkpoint.bin"), &bytes)?;
    record(&dir.join("checkpoint.json"), &metadata)?;
    for (tag, enabled) in [("initial", step == 0), ("first90", first90)] {
        if enabled {
            durable(&dir.join(format!("{tag}.bin")), &bytes)?;
            record(&dir.join(format!("{tag}.json")), &metadata)?;
        }
    }
    record(
        &dir.join("execution.json"),
        &json!({"clean":true,"completed":step}),
    )
}

fn evaluate(
    engine: &mut Engine,
    examples: &[Example],
    training_extent: Option<usize>,
) -> Result<Value> {
    engine.set_inference_extension(None)?;
    let features = examples[0].future_tokens.len();
    let mut joint = 0;
    let mut reanchored_joint = 0;
    let mut exact = 0;
    let mut feature_hits = vec![0usize; features];
    let mut forecast_hits = vec![0usize; features];
    let mut all_tokens = 0;
    for chunk in examples.chunks(128) {
        let mut contexts: Vec<Vec<u32>> = chunk.iter().map(|e| e.prompt.clone()).collect();
        let mut output = vec![Vec::new(); chunk.len()];
        if let Some(train) = training_extent {
            let total = contexts[0].len() + features - 1;
            engine.set_inference_extension(Some(Extension::new(
                train,
                engine.positional_axes().time_extent(total),
            )?))?;
        }
        for _ in 0..features {
            let length = contexts[0].len();
            let tokens: Vec<u32> = contexts.iter().flatten().copied().collect();
            engine.forward(&tokens, &vec![u32::MAX; tokens.len()], chunk.len(), length)?;
            let predictions = engine.predictions.read()?;
            for (i, context) in contexts.iter_mut().enumerate() {
                let token = predictions[i * length + length - 1];
                context.push(token);
                output[i].push(token);
            }
        }
        for (e, predicted) in chunk.iter().zip(output) {
            let mut hit = true;
            let mut fresh = true;
            let mut token_hit = true;
            for (f, &token) in predicted.iter().enumerate() {
                let a = e.decoded_hit(f, token, false);
                let b = e.decoded_hit(f, token, true);
                hit &= a;
                fresh &= b;
                feature_hits[f] += usize::from(a);
                forecast_hits[f] += usize::from(b);
                token_hit &= token == e.future_tokens[f];
                all_tokens += usize::from(token == e.future_tokens[f]);
            }
            joint += usize::from(hit);
            reanchored_joint += usize::from(fresh);
            exact += usize::from(token_hit);
        }
    }
    engine.set_inference_extension(None)?;
    Ok(
        json!({"examples":examples.len(),"features":features,"decoded_correct":joint,
        "decoded_accuracy":joint as f64/examples.len() as f64,
        "forecast_reanchored_correct":reanchored_joint,"forecast_reanchored_accuracy":reanchored_joint as f64/examples.len() as f64,
        "exact_token_rows":exact,"exact_token_row_accuracy":exact as f64/examples.len() as f64,
        "token_correct":all_tokens,"token_accuracy":all_tokens as f64/(examples.len()*features) as f64,
        "feature_decoded_correct":feature_hits,"feature_reanchored_correct":forecast_hits,
        "inference_yarn":training_extent.is_some()}),
    )
}

fn verified_data(
    reg: &Value,
    seed: u32,
    features: usize,
    context: usize,
    name: &str,
    tokenizer: Tokenizer,
) -> Result<Dataset> {
    let rows: Vec<_> = reg["rows"]
        .as_array()
        .ok_or("missing registered data rows")?
        .iter()
        .filter(|r| {
            r["seed"] == seed
                && r["features"] == features
                && r["context"] == context
                && r["tokenizer"] == name
        })
        .collect();
    if rows.len() != 1 {
        return Err("unregistered or ambiguous DiffGPT dataset".into());
    }
    let data = data::build(seed, features, context, tokenizer)?;
    if rows[0]["train_sha256"] != data::fingerprint(&data.train)
        || rows[0]["heldout_sha256"] != data::fingerprint(&data.validation)
    {
        return Err("DiffGPT token/target/decoder/state fingerprint mismatch".into());
    }
    Ok(data)
}

pub fn run(args: &[String]) -> Result<()> {
    if args.len() != 7 {
        return Err(
            "usage: diffgpt-cell REGISTERED_ROOT mash|lgt|mash-half 1|2 SEED sequence|axial".into(),
        );
    }
    if crate::precision::FP32_AFFINE_REFERENCE {
        return Err("FP32 affine build is diagnostic only".into());
    }
    let root = Path::new(&args[2]);
    let name = &args[3];
    let features: usize = args[4].parse()?;
    let seed: u32 = args[5].parse()?;
    let arm = &args[6];
    let tokenizer = match name.as_str() {
        "mash" => Tokenizer::Mash,
        "lgt" => Tokenizer::Lgt,
        "mash-half" => Tokenizer::MashHalf,
        _ => return Err("unknown tokenizer".into()),
    };
    if ![1, 2].contains(&features)
        || ![11, 17, 23].contains(&seed)
        || !["sequence", "axial"].contains(&arm.as_str())
    {
        return Err("undeclared DiffGPT cell".into());
    }
    let registration = read(&root.join("experiment.json"))?;
    if registration["kind"] != "diffgpt_axial_lerope_yarn"
        || registration["source_sha256"] != runner::source_hash()
        || registration["ptx_sha256"] != runner::ptx_hash()
        || registration["binary_sha256"] != hash(&fs::read(std::env::current_exe()?)?)
        || registration["budget_updates"] != BUDGET
        || registration["batch_size"] != BATCH
        || registration["observation_updates"] != OBSERVE
    {
        return Err("DiffGPT executable/protocol registration mismatch".into());
    }
    let data_path = registration["data_registration"]
        .as_str()
        .ok_or("missing data registration")?;
    let raw = fs::read(data_path)?;
    if registration["data_registration_sha256"] != hash(&raw) {
        return Err("data registration hash mismatch".into());
    }
    let data_reg: Value = serde_json::from_slice(&raw)?;
    let data = verified_data(&data_reg, seed, features, 2, name, tokenizer)?;
    let length = data.train[0].tokens.len();
    let dir = root.join(format!("{arm}/{name}/features{features}/seed{seed}"));
    if dir.exists() || root.join("STOP").exists() {
        return Err("refuse replay or launch under STOP".into());
    }
    let axes = if arm == "sequence" {
        Axes::Sequence
    } else {
        Axes::Axial { features }
    };
    let cfg = Config {
        width: 128,
        heads: 1,
        layers: 1,
        vocab: 512,
        max_batch: BATCH,
        max_length: 9 * features - 1,
        normalized: true,
        ssmax: true,
        gain: 1. / 128.,
        lr: 0.001,
        decay: 0.1,
        warmup: 100,
        architecture: "complex-phase-score-lerope".into(),
        phase_optimizer: false,
    };
    let initial = layout::initialize_model(&cfg, seed);
    let mut engine = Engine::new(cfg.clone(), &initial)?;
    engine.set_positional_axes(axes)?;
    let mut optimizer = Optimizer::new(&engine, BUDGET)?;
    let mut rng = Random::new(seed + 10000);
    let indices: Vec<u32> = (0..BATCH * BUDGET)
        .map(|_| rng.below(data.train.len() as u32))
        .collect();
    let sampler_bytes: Vec<u8> = indices.iter().flat_map(|i| i.to_le_bytes()).collect();
    let initial_bytes: Vec<u8> = initial.iter().flat_map(|i| i.to_le_bytes()).collect();
    fs::create_dir_all(&dir)?;
    let metadata = json!({"arm":arm,"tokenizer":name,"features":features,"seed":seed,"context":2,
        "axes":format!("{axes:?}"),"training_extent":axes.time_extent(length),"vocab":512,"width":128,"heads":1,"layers":1,
        "independent_parameters":layout::independent_count(&cfg),"expanded_parameters":initial.len(),
        "initial_sha256":hash(&initial_bytes),"sampler_sha256":hash(&sampler_bytes),
        "train_sha256":data::fingerprint(&data.train),"heldout_sha256":data::fingerprint(&data.validation),
        "source_sha256":runner::source_hash(),"ptx_sha256":runner::ptx_hash(),
        "budget_updates":BUDGET,"batch_size":BATCH,"observe":OBSERVE,"optimizer":"AdamW/WSD",
        "lr":0.001,"betas":[0.9,0.95],"weight_decay":0.1,"warmup":100,"cosine_start":4000,
        "separate_frequency_clip":1,"frequency_weight_decay":0,"checkpoint_epochs":500,
        "training_has_yarn":false,"native_split_not_torch_randperm":true});
    record(&dir.join("metadata.json"), &metadata)?;
    let mut history = Vec::new();
    let mut step = 0;
    let mut first90 = None;
    let mut next_checkpoint_epoch = 500.;
    let started = Instant::now();
    loop {
        let validation = evaluate(&mut engine, &data.validation, None)?;
        let epochs = step as f64 * BATCH as f64 / data.train.len() as f64;
        let newly_passed = first90.is_none()
            && validation["decoded_correct"]
                .as_u64()
                .ok_or("missing decoded count")?
                * 10
                >= 9 * data.validation.len() as u64;
        if newly_passed {
            first90 = Some(step);
        }
        let point = json!({"step":step,"examples":step*BATCH,"epochs":epochs,"validation":validation,"seconds":started.elapsed().as_secs_f64()});
        history.extend(serde_json::to_vec(&point)?);
        history.push(b'\n');
        let stopped = root.join("STOP").exists() || dir.join("STOP").exists();
        if step == 0 || newly_passed || step == BUDGET || stopped || epochs >= next_checkpoint_epoch
        {
            checkpoint(&engine, &dir, step, &history, newly_passed)?;
            if epochs >= next_checkpoint_epoch {
                next_checkpoint_epoch += 500.;
            }
            if step < BUDGET && !stopped {
                record(
                    &dir.join("execution.json"),
                    &json!({"clean":false,"durable_completed":step,"possible_completed":BUDGET}),
                )?;
            }
        }
        if step == BUDGET || stopped {
            let mut extrapolation = Vec::new();
            if !stopped {
                for context in [4, 8] {
                    let longer =
                        verified_data(&data_reg, seed, features, context, name, tokenizer)?;
                    let native = evaluate(&mut engine, &longer.validation, None)?;
                    let yarn = evaluate(
                        &mut engine,
                        &longer.validation,
                        Some(axes.time_extent(length)),
                    )?;
                    extrapolation.push(json!({"context":context,"native":native,"yarn":yarn,
                        "heldout_sha256":data::fingerprint(&longer.validation)}));
                }
                let expected = fs::read(dir.join("checkpoint.bin"))?;
                let current: Vec<u8> = engine
                    .parameters
                    .read()?
                    .iter()
                    .chain(&engine.first_moment.read()?)
                    .chain(&engine.second_moment.read()?)
                    .flat_map(|v| v.to_le_bytes())
                    .collect();
                if current != expected {
                    return Err("extrapolation changed weights or optimizer state".into());
                }
            }
            record(
                &dir.join("result.json"),
                &json!({"status":if stopped{"stopped"}else if first90.is_some(){"reached90"}else{"budget_exhausted"},
                "metadata":metadata,"completed":step,"first90_updates":first90,
                "first90_epochs":first90.map(|s|s as f64*BATCH as f64/data.train.len() as f64),
                "previous_observation_updates":first90.map(|s|s.saturating_sub(OBSERVE)),
                "endpoint":point,"extrapolation":extrapolation,"first90_is_not_sustained90":true}),
            )?;
            return Ok(());
        }
        let next = (step + OBSERVE).min(BUDGET);
        while step < next {
            let mut tokens = Vec::with_capacity(BATCH * length);
            let mut targets = Vec::with_capacity(BATCH * length);
            for &i in &indices[step * BATCH..(step + 1) * BATCH] {
                tokens.extend(&data.train[i as usize].tokens);
                targets.extend(&data.train[i as usize].targets);
            }
            engine.forward(&tokens, &targets, BATCH, length)?;
            engine.backward(BATCH, length)?;
            optimizer.step(&engine, step)?;
            step += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn declared_diffgpt_schedule_has_the_requested_relative_phases() -> Result<()> {
        let s = crate::schedule::Schedule::new(BUDGET)?;
        assert_eq!((s.warmup, s.cooldown_start), (100, 4000));
        assert_eq!(s.rate(0), 0.);
        assert_eq!(s.rate(50), 0.0005);
        assert_eq!(s.rate(100), 0.001);
        assert_eq!(s.rate(4000), 0.001);
        assert!((s.rate(4999) - 1e-8).abs() < 1e-14);
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA; engineering only, no scientific model-learning result"]
    fn diffgpt_training_and_checkpoint_contract_all_encoders_and_axes() -> Result<()> {
        let mut records = Vec::new();
        let root =
            std::env::var_os("AXIAL_VALIDATION_OUTPUT").ok_or("set AXIAL_VALIDATION_OUTPUT")?;
        let root = Path::new(&root);
        if !root.is_absolute() {
            return Err("validation path must be absolute".into());
        }
        fs::create_dir_all(root)?;
        for (name, tokenizer) in [
            ("mash", Tokenizer::Mash),
            ("lgt", Tokenizer::Lgt),
            ("mash-half", Tokenizer::MashHalf),
        ] {
            for features in [1, 2] {
                let data = data::build(11, features, 2, tokenizer)?;
                for axes in [Axes::Sequence, Axes::Axial { features }] {
                    let cfg = Config {
                        width: 128,
                        heads: 1,
                        layers: 1,
                        vocab: 512,
                        max_batch: BATCH,
                        max_length: 9 * features - 1,
                        normalized: true,
                        ssmax: true,
                        gain: 1. / 128.,
                        lr: 0.001,
                        decay: 0.1,
                        warmup: 100,
                        architecture: "complex-phase-score-lerope".into(),
                        phase_optimizer: false,
                    };
                    let p = layout::initialize_model(&cfg, 11);
                    let mut engine = Engine::new(cfg.clone(), &p)?;
                    engine.set_positional_axes(axes)?;
                    let mut optimizer = Optimizer::new(&engine, BUDGET)?;
                    let small = &data.train[..3];
                    let tokens: Vec<u32> = small
                        .iter()
                        .flat_map(|e| e.tokens.iter().copied())
                        .collect();
                    let targets: Vec<u32> = small
                        .iter()
                        .flat_map(|e| e.targets.iter().copied())
                        .collect();
                    let length = small[0].tokens.len();
                    engine.forward(&tokens, &targets, 3, length)?;
                    engine.backward(3, length)?;
                    optimizer.step(&engine, 0)?;
                    assert_eq!(
                        engine
                            .parameters
                            .read()?
                            .iter()
                            .map(|x| x.to_bits())
                            .collect::<Vec<_>>(),
                        p.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                    );
                    let params = engine.parameters.read()?;
                    let first = engine.first_moment.read()?;
                    let second = engine.second_moment.read()?;
                    let mut restored = Engine::new(cfg.clone(), &params)?;
                    restored.set_positional_axes(axes)?;
                    restored.first_moment.write(&first)?;
                    restored.second_moment.write(&second)?;
                    for e in [&mut engine, &mut restored] {
                        e.forward(&tokens, &targets, 3, length)?;
                        e.backward(3, length)?;
                        Optimizer::new(e, BUDGET)?.step(e, 1)?;
                    }
                    assert_eq!(engine.parameters.read()?, restored.parameters.read()?);
                    assert_eq!(engine.first_moment.read()?, restored.first_moment.read()?);
                    assert_eq!(engine.second_moment.read()?, restored.second_moment.read()?);
                    layout::validate_state(
                        &cfg,
                        &engine.parameters.read()?,
                        &engine.first_moment.read()?,
                        &engine.second_moment.read()?,
                    )?;
                    let before = engine.parameters.read()?;
                    let mut plain = evaluate(&mut engine, &data.validation[..3], None)?;
                    let mut identity = evaluate(
                        &mut engine,
                        &data.validation[..3],
                        Some(axes.time_extent(length)),
                    )?;
                    plain.as_object_mut().unwrap().remove("inference_yarn");
                    identity.as_object_mut().unwrap().remove("inference_yarn");
                    assert_eq!(plain, identity);
                    assert_eq!(before, engine.parameters.read()?);
                    records.push(
                        json!({"tokenizer":name,"features":features,"axes":format!("{axes:?}"),
                        "zero_lr_weights_exact":true,"restored_next_update_exact":true,
                        "in_range_evaluation_identity":true,"parameters_finite":true,
                        "scientific_learning_result":false,"passed":true}),
                    );
                }
            }
        }
        let path = root.join("diffgpt-engineering.json");
        if path.exists() {
            return Err("engineering result already exists".into());
        }
        record(
            &path,
            &json!({"cases":records.len(),"rows":records,"scientific_training":false}),
        )
    }
}
