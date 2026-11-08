//! JSON CLI wiring for the native model and checkpoint API.
use diff_gpt_kernite::{
    Checkpoint, CheckpointCadence, DiffGpt, EpochSampler, Model, ModelConfig, Result,
    codec::{Difference, Domain, Matrix, Tokenizer},
};
use serde_json::{Value, json};
use std::{fs, path::Path};
fn read(path: &str) -> Result<Value> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}
fn tokens(value: &Value) -> Result<Vec<u32>> {
    value
        .as_array()
        .ok_or("expected token array")?
        .iter()
        .map(|v| {
            if v.as_i64() == Some(-100) {
                return Ok(u32::MAX);
            }
            Ok(v.as_u64().ok_or("expected integer token")?.try_into()?)
        })
        .collect()
}
fn output(value: &Value) -> Result<()> {
    serde_json::to_writer_pretty(std::io::stdout().lock(), value)?;
    println!();
    Ok(())
}
fn fresh(path: &str) -> Result<()> {
    if Path::new(path).exists() {
        Err("output checkpoint already exists".into())
    } else {
        Ok(())
    }
}
fn run(args: &[String]) -> Result<()> {
    match args.get(1).map(String::as_str) {
        Some("build-info") if args.len() == 2 => output(&diff_gpt_kernite::build_info()),
        Some("inspect") if args.len()==3 => {
            let cp = Checkpoint::load(&args[2])?;
            output(&json!({"config":cp.config().to_json(),"completed":cp.completed(),"examples":cp.examples(),
                "state_sha256":cp.state_sha256(),"precision":diff_gpt_kernite::PRECISION}))
        }
        Some("init") if args.len()==5 => {
            fresh(&args[4])?;
            let config = ModelConfig::from_json(&read(&args[2])?)?;
            let model = Model::new(config,args[3].parse()?)?;
            model.checkpoint()?.save(&args[4])?;
            output(&json!({"checkpoint":args[4],"completed":0,"device":model.device_name()}))
        }
        Some("init-diffgpt") if args.len()==7 => {
            fresh(&args[6])?;
            let mut value=read(&args[2])?;
            let features:usize=args[3].parse()?;
            let length:usize=args[4].parse()?;
            let positions=json!({"kind":"axial_lerope","features":features});
            if !value["positions"].is_null() && value["positions"]!=positions { return Err("DiffGPT config must use matching2Daxial positions".into()); }
            value["positions"]=positions;
            value["training_input_length"]=json!(length);
            let model=DiffGpt::with_config(ModelConfig::from_json(&value)?,args[5].parse()?)?;
            model.checkpoint()?.save(&args[6])?;
            output(&json!({"checkpoint":args[6],"completed":0,"positions":"axial_time_feature","training_extent":model.config().training_extent()}))
        }
        Some("generate") if args.len()==5 => {
            let cp = Checkpoint::load(&args[2])?;
            let prompt = tokens(&read(&args[3])?)?;
            let mut model = Model::from_checkpoint(&cp)?;
            output(&json!({"generated":model.generate(&prompt,args[4].parse()?,None)?}))
        }
        Some("forecast") if args.len()==5 => {
            let cp = Checkpoint::load(&args[2])?;
            let spec = read(&args[3])?;
            let features: usize = spec["features"].as_u64().ok_or("missing features")?.try_into()?;
            let values = spec["values"].as_array().ok_or("missing values")?.iter()
                .map(|v|v.as_f64().ok_or("invalid raw value")).collect::<std::result::Result<Vec<_>,_>>()?;
            let series = Matrix::new(features,values)?;
            let operators = spec["orders"].as_array().ok_or("missing difference orders")?.iter()
                .map(|v| -> Result<Difference> { Ok(Difference::order(v.as_u64().ok_or("invalid order")?.try_into()?)?) }).collect::<Result<Vec<_>>>()?;
            let domains = spec["domains"].as_array().ok_or("missing domains")?.iter()
                .map(|v| -> Result<Domain> { Ok(Domain { low:v[0].as_f64().ok_or("invalid low")?,high:v[1].as_f64().ok_or("invalid high")? }) }).collect::<Result<Vec<_>>>()?;
            let tokenizer = match spec["tokenizer"].as_str() { Some("mash")=>Tokenizer::Mash,Some("lgt")=>Tokenizer::Lgt,
                Some("mash-half")=>Tokenizer::MashHalf,_=>return Err("unknown tokenizer".into()) };
            let mut model = DiffGpt::from_checkpoint(&cp)?;
            let forecast = model.forecast(&series,&domains,&operators,tokenizer,args[4].parse()?)?;
            output(&json!({"features":features,"rows":forecast.values.rows(),"values":forecast.values.values(),
                "tokens":forecast.tokens.values(),"clipped_context_tokens":forecast.clipped_context_tokens}))
        }
        Some("train") if args.len()==4 => train(&args[2],&args[3],false),
        Some("train-diffgpt") if args.len()==4 => train(&args[2],&args[3],true),
        _ => Err("usage: diff-gpt-kernite build-info | inspect CHECKPOINT | init CONFIG_JSON SEED FRESH_CHECKPOINT | init-diffgpt CONFIG_JSON FEATURES TRAIN_INPUT_LENGTH SEED FRESH_CHECKPOINT | generate CHECKPOINT PROMPT_JSON COUNT | forecast CHECKPOINT SERIES_JSON ROWS | train|train-diffgpt SPEC_JSON FRESH_CHECKPOINT".into()),
    }
}
fn train(spec_path: &str, checkpoint: &str, diffgpt: bool) -> Result<()> {
    fresh(checkpoint)?;
    let spec = read(spec_path)?;
    let seed = spec["seed"].as_u64().ok_or("missing seed")?.try_into()?;
    let samples: Vec<_> = spec["examples"]
        .as_array()
        .ok_or("missing examples")?
        .iter()
        .map(|e| -> Result<_> { Ok((tokens(&e["tokens"])?, tokens(&e["targets"])?)) })
        .collect::<Result<_>>()?;
    let length = samples.first().ok_or("empty train set")?.0.len();
    let mut value = spec["config"].clone();
    if diffgpt {
        let features = spec["features"]
            .as_u64()
            .ok_or("DiffGPT training needs its feature count")?;
        let positions = json!({"kind":"axial_lerope","features":features});
        if !value["positions"].is_null() && value["positions"] != positions {
            return Err("DiffGPT training accepts2Daxial positions only".into());
        }
        value["positions"] = positions;
    }
    value["training_input_length"] = json!(length);
    let config = ModelConfig::from_json(&value)?;
    if length == 0
        || length > config.max_length
        || samples.iter().any(|(x, y)| {
            x.len() != length
                || y.len() != length
                || x.iter().any(|&t| t as usize >= config.vocab)
                || y.iter()
                    .any(|&t| t != u32::MAX && t as usize >= config.vocab)
                || y.iter().all(|&t| t == u32::MAX)
        })
    {
        return Err(
            "training examples must have valid, uniform lengths and supervised targets".into(),
        );
    }
    let mut cadence = CheckpointCadence::new(samples.len(), 500)?;
    let mut sampler = EpochSampler::new(seed)?;
    let mut model = Model::new(config.clone(), seed)?;
    model.checkpoint()?.save(checkpoint)?;
    let stop = format!("{checkpoint}.stop");
    loop {
        for ids in sampler.next_epoch(samples.len())?.chunks(config.max_batch) {
            let mut x = Vec::with_capacity(ids.len() * length);
            let mut y = Vec::with_capacity(ids.len() * length);
            for &i in ids {
                x.extend(&samples[i].0);
                y.extend(&samples[i].1);
            }
            model.train_step(&x, &y, ids.len(), length)?;
            let finished = model.completed() == config.budget;
            // A read-only stop check at observation cadence, never a per-update write.
            let stopped = model.completed().is_multiple_of(50) && Path::new(&stop).exists();
            if cadence.due(model.examples()) || finished || stopped {
                model.checkpoint()?.save(checkpoint)?;
                cadence.saved(model.examples());
                output(
                    &json!({"checkpoint":checkpoint,"completed":model.completed(),"examples":model.examples(),
                    "epochs":model.examples() as f64/samples.len() as f64,"stopped":stopped}),
                )?;
            }
            if finished || stopped {
                return Ok(());
            }
        }
    }
}
fn main() -> std::process::ExitCode {
    match run(&std::env::args().collect::<Vec<_>>()) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            std::process::ExitCode::FAILURE
        }
    }
}
