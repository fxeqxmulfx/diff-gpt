//! Immutable native cohorts with buffered observations and durable work windows.
use crate::{
    Result, data,
    engine::{Config, Engine},
    hash, layout, metrics, random,
};
use serde_json::{Value, json};
use std::{
    fs::{self, File},
    io::Write,
    path::Path,
    process::Command,
    time::Instant,
};
const SUITE: &str = include_str!("../../suite.json");
const PLAN: &str = include_str!("../../PLAN.txt");
const FLUSH_EPOCHS: usize = 500;
const ARMS: [&str; 4] = [
    "real-product",
    "complex-linear",
    "complex-hermitian",
    "complex-hermitian-phase",
];
fn source_hash() -> String {
    let mut bytes = Vec::new();
    for (name, body) in crate::sources::FILES {
        bytes.extend((name.len() as u64).to_le_bytes());
        bytes.extend(name.as_bytes());
        bytes.extend((body.len() as u64).to_le_bytes());
        bytes.extend(body.as_bytes());
    }
    hash(&bytes)
}
fn ptx_hash() -> String {
    hash(
        [
            crate::attention::kernels::PTX,
            crate::base_ops::kernels::PTX,
            crate::sequence_ops::kernels::PTX,
            crate::complex::kernels::PTX,
            crate::gemm::kernels::PTX,
        ]
        .concat()
        .as_bytes(),
    )
}
fn durable(path: &Path, bytes: &[u8]) -> Result<()> {
    let temporary = path.with_extension("pending");
    let mut file = File::create(&temporary)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    fs::rename(temporary, path)?;
    File::open(path.parent().ok_or("no parent")?)?.sync_all()?;
    Ok(())
}
fn record(path: &Path, value: &Value) -> Result<()> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    durable(path, &bytes)
}
fn read(path: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}
fn suite() -> Value {
    serde_json::from_str(SUITE).unwrap()
}
fn number(v: &Value, k: &str) -> Result<usize> {
    Ok(v[k].as_u64().ok_or_else(|| format!("missing {k}"))? as usize)
}
fn config(row: &Value, task: &data::Task, arm: &str) -> Result<Config> {
    if !ARMS.contains(&arm) {
        return Err("unknown arm".into());
    }
    let c = &row["case"];
    let width = number(c, "width")?;
    let heads = number(c, "heads")?;
    let layers = number(c, "layers")?;
    // Six width-128 layers at T=128 reserve over 3GB with 128 evaluation
    // rows. Use 64-row chunks on the 4GB GPU; every held-out row is scored.
    let evaluation_batch = if width >= 128 && layers >= 6 { 64 } else { 128 };
    let max_length = task
        .splits
        .values()
        .flatten()
        .map(|e| {
            e.tokens
                .len()
                .max(e.prompt.len() + e.limit.saturating_sub(1))
        })
        .max()
        .ok_or("empty task")?;
    Ok(Config {
        width,
        heads,
        vocab: task.vocab,
        layers,
        max_batch: number(c, "batch")?.max(evaluation_batch),
        max_length,
        normalized: true,
        ssmax: true,
        gain: heads as f32 / width as f32,
        lr: c["learning_rate"].as_f64().ok_or("missing LR")? * 2.,
        decay: c["weight_decay"].as_f64().ok_or("missing decay")? as f32,
        warmup: number(c, "warmup")?,
        architecture: arm.trim_end_matches("-phase").into(),
        phase_optimizer: arm.ends_with("-phase"),
    })
}
fn order(n: usize, batch: usize, updates: usize, seed: u32) -> (Vec<u32>, Vec<usize>) {
    let mut rng = random::Random::new(seed + 10000);
    let mut values = Vec::with_capacity(updates * batch);
    let mut edges = vec![0];
    while edges.len() <= updates {
        let mut epoch: Vec<u32> = (0..n as u32).collect();
        rng.shuffle(&mut epoch);
        for chunk in epoch.chunks(batch) {
            if edges.len() > updates {
                break;
            }
            values.extend_from_slice(chunk);
            edges.push(values.len());
        }
    }
    (values, edges)
}
fn checkpoint(
    engine: &Engine,
    dir: &Path,
    completed: usize,
    examples: usize,
    history: &[u8],
) -> Result<()> {
    engine.cuda.sync()?;
    let mut bytes = Vec::new();
    for buffer in [
        &engine.parameters,
        &engine.first_moment,
        &engine.second_moment,
    ] {
        let data = buffer.read()?;
        if data.iter().any(|x| !x.is_finite()) {
            return Err("nonfinite checkpoint".into());
        }
        bytes.extend(data.iter().flat_map(|x| x.to_le_bytes()));
    }
    layout::validate_state(
        &engine.config,
        &engine.parameters.read()?,
        &engine.first_moment.read()?,
        &engine.second_moment.read()?,
    )?;
    durable(&dir.join("history.jsonl"), history)?;
    durable(&dir.join("checkpoint.bin"), &bytes)?;
    record(
        &dir.join("checkpoint.json"),
        &json!({"completed":completed,"examples":examples,"state_sha256":hash(&bytes),
        "history_sha256":hash(history),"source_sha256":source_hash(),"ptx_sha256":ptx_hash(),
        "state_layout":"parameters, first moment, second moment; f32 little endian",
        "sampler":"regenerate immutable seed+10000 order; examples is exact next offset"}),
    )?;
    record(
        &dir.join("execution.json"),
        &json!({"durable_completed":completed,"possible_completed":completed,"clean":true}),
    )
}
fn restore(engine: &mut Engine, dir: &Path) -> Result<(usize, Vec<u8>, f64)> {
    let execution = read(&dir.join("execution.json"))?;
    if execution["clean"] != true {
        return Err("unclean work window: refuse silent update replay".into());
    }
    let cp = read(&dir.join("checkpoint.json"))?;
    let bytes = fs::read(dir.join("checkpoint.bin"))?;
    let history = fs::read(dir.join("history.jsonl"))?;
    if cp["source_sha256"] != source_hash()
        || cp["ptx_sha256"] != ptx_hash()
        || cp["state_sha256"] != hash(&bytes)
        || cp["history_sha256"] != hash(&history)
    {
        return Err("checkpoint provenance/hash mismatch".into());
    }
    let n = engine.parameters.len();
    if bytes.len() != 12 * n {
        return Err("checkpoint layout mismatch".into());
    }
    let values: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    layout::validate_state(
        &engine.config,
        &values[..n],
        &values[n..2 * n],
        &values[2 * n..],
    )?;
    engine.parameters.write(&values[..n])?;
    engine.first_moment.write(&values[n..2 * n])?;
    engine.second_moment.write(&values[2 * n..])?;
    let last: Value = serde_json::from_slice(
        history
            .split(|&c| c == b'\n')
            .rfind(|line| !line.is_empty())
            .ok_or("missing observations")?,
    )?;
    Ok((
        number(&cp, "completed")?,
        history,
        last["seconds"].as_f64().ok_or("missing elapsed time")?,
    ))
}
fn run_cell(root: &Path, row: &Value, seed: u32, arm: &str, resume: bool) -> Result<()> {
    if root.join("STOP").exists() {
        return Err("remove STOP before starting or resuming a cell".into());
    }
    let name = row["name"].as_str().ok_or("missing name")?;
    let dir = root.join(arm).join(name).join(format!("seed{seed}"));
    if dir.join("STOP").exists() {
        return Err("remove the cell STOP marker before resuming".into());
    }
    if dir.exists() && !resume {
        return Err(format!("refuse overwrite/replay: {}", dir.display()).into());
    }
    let started = Instant::now();
    let task = data::build(row)?;
    let cfg = config(row, &task, arm)?;
    let train = &task.splits["train"];
    let batch = number(&row["case"], "batch")?;
    let budget = number(&row["case"], "max_updates")?;
    let observe = number(&row["case"], "observe")?;
    let flush_updates = FLUSH_EPOCHS * train.len().div_ceil(batch);
    let (order, edges) = order(train.len(), batch, budget, seed);
    let parameters = layout::initialize_model(&cfg, seed);
    let initial: Vec<_> = parameters.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut sampler = sha2::Sha256::new();
    use sha2::Digest;
    for v in &order {
        sampler.update(v.to_le_bytes());
    }
    let sampler_hash: String = sampler
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    let metadata = json!({"case":row,"seed":seed,"arm":arm,"source_sha256":source_hash(),"ptx_sha256":ptx_hash(),
        "backend":"Rust/Kernite complex geometry","initial_parameters_sha256":hash(&initial),"sampler_sha256":sampler_hash,
        "expanded_parameters":parameters.len(),"independent_parameters":layout::independent_count(&cfg),"hidden":layout::hidden(cfg.width,""),"learning_rate":cfg.lr,"weight_decay":cfg.decay,
        "warmup":cfg.warmup,"optimizer":if cfg.phase_optimizer{"AdamW shared complex-pair variance"}else{"AdamW separate real/imaginary variances"},"betas":[0.9,0.98],"epsilon":1e-8,"gain":cfg.gain,
        "checkpoint_epochs":FLUSH_EPOCHS,"matrix_constraints":if layout::is_complex(&cfg.architecture){"complex-linear: [[A,-B],[B,A]]"}else{"none"},"hermitian_xsa":cfg.architecture=="complex-hermitian",
        "dtype":"FP32 with repaired FP64 RMS backward","native_rng":"Kernite Random",
        "max_batch":cfg.max_batch,"max_length":cfg.max_length,"primary":task.selection,
        "generation":"greedy autoregression from prompts; own mistakes retained; EOS/limit stopping",
        "recurrence_readout":"known-field residue probability pooling; observed prefix only"});
    if resume {
        if read(&dir.join("protocol.json"))? != metadata {
            return Err("resume protocol changed".into());
        }
        let previous = read(&dir.join("result.json"))?;
        if previous["status"] != "stopped" {
            return Err("only explicitly stopped cells can resume".into());
        }
    } else {
        fs::create_dir_all(&dir)?;
        record(&dir.join("protocol.json"), &metadata)?;
        durable(&dir.join("initial.f32"), &initial)?;
    }
    let mut engine = Engine::new(cfg, &parameters)?;
    let (mut completed, mut history, carried) = if resume {
        restore(&mut engine, &dir)?
    } else {
        (0, Vec::new(), 0.)
    };
    let resumed_at = completed;
    if resume {
        let prior = read(&dir.join("result.json"))?;
        record(&dir.join(format!("stop-{completed}.json")), &prior)?;
        fs::remove_file(dir.join("result.json"))?;
    }
    let mut previous = completed;
    let mut window_end = completed;
    let mut packed = metrics::Packed::default();
    loop {
        let train_metrics = metrics::evaluate(&mut engine, train, false)?;
        let validation = metrics::evaluate(&mut engine, &task.splits[&task.selection], true)?;
        let mut observed = serde_json::Map::new();
        observed.insert("train".into(), train_metrics.clone());
        observed.insert(task.selection.clone(), validation.clone());
        if let Some(base) = task.selection.strip_suffix("/novel") {
            observed.insert(
                base.into(),
                metrics::evaluate(&mut engine, &task.splits[base], true)?,
            );
        }
        let epochs = edges[completed] as f64 / train.len() as f64;
        let mut point = json!({"step":completed,"examples":edges[completed],"epochs":epochs,"seconds":carried+started.elapsed().as_secs_f64(),
            "train":train_metrics,"validation":validation,"splits":observed});
        // A resumed checkpoint already includes its observation.
        if !resume || completed != resumed_at {
            history.extend(serde_json::to_vec(&point)?);
            history.push(b'\n');
        } else {
            // Keep the durable endpoint identical even if STOP arrives during
            // the re-evaluation at the restored step.
            point = serde_json::from_slice(
                history
                    .split(|&c| c == b'\n')
                    .rfind(|s| !s.is_empty())
                    .ok_or("missing last observation")?,
            )?;
        }
        let passed = validation["exact"].as_u64().ok_or("missing exact")? * 100
            >= 99 * task.splits[&task.selection].len() as u64;
        let stopped = root.join("STOP").exists() || dir.join("STOP").exists();
        let finished = passed || completed == budget || stopped;
        if completed == 0 || completed >= window_end || finished {
            checkpoint(&engine, &dir, completed, edges[completed], &history)?;
            record(&dir.join("status.json"), &point)?;
            println!(
                "{}",
                json!({"case":name,"seed":seed,"arm":arm,"updates":completed,"epochs":epochs,"validation":validation["sequence_accuracy"],"seconds":point["seconds"]})
            );
            if !finished {
                window_end = ((completed / flush_updates + 1) * flush_updates).min(budget);
                record(
                    &dir.join("execution.json"),
                    &json!({"durable_completed":completed,"possible_completed":window_end,"clean":false}),
                )?;
            }
        }
        if finished {
            let mut final_splits = observed;
            if !stopped {
                for (split, examples) in &task.splits {
                    if !final_splits.contains_key(split) {
                        final_splits.insert(
                            split.clone(),
                            metrics::evaluate(&mut engine, examples, !split.starts_with("train"))?,
                        );
                    }
                }
            }
            let result = json!({"case":name,"seed":seed,"arm":arm,"status":if passed{"passed"}else if stopped{"stopped"}else{"budget_exhausted"},
                "completed":completed,"examples":edges[completed],"epochs":epochs,"first99_epochs":if passed{Some(epochs)}else{None},
                "previous_observed_epochs":if passed{Some(edges[previous]as f64/train.len()as f64)}else{None},
                "seconds":carried+started.elapsed().as_secs_f64(),"endpoint":point,"final_splits":final_splits,"metadata":metadata});
            record(&dir.join("result.json"), &result)?;
            return Ok(());
        }
        previous = completed;
        let next = (completed + observe).min(budget).min(window_end);
        while completed < next {
            let indices = &order[edges[completed]..edges[completed + 1]];
            packed.fill(train, indices);
            engine.forward(&packed.tokens, &packed.labels, indices.len(), packed.length)?;
            engine.backward(indices.len(), packed.length)?;
            engine.update(completed)?;
            completed += 1;
        }
    }
}
fn preflight() -> Result<Value> {
    let manifest = suite();
    let mut checked = Vec::new();
    for row in manifest["tasks"].as_array().unwrap() {
        {
            let d = data::build(row)?;
            let cfg = config(row, &d, "complex-hermitian")?;
            checked.push(json!({"case":row["name"],"tier":row["tier"],"train":d.splits["train"].len(),
            "primary":d.selection,"heldout":d.splits[&d.selection].len(),"vocab":d.vocab,"splits":d.splits.len(),
            "max_length":cfg.max_length,"all_frozen_fingerprints_match":true}));
        }
    }
    Ok(json!({"suite_sha256":hash(SUITE.as_bytes()),"checked_tasks":checked}))
}
pub fn report(root: &Path) -> Result<()> {
    let manifest = suite();
    let mut rows = Vec::new();
    for row in manifest["tasks"].as_array().unwrap() {
        for seed in 0..3 {
            for arm in ARMS {
                let dir = root
                    .join(arm)
                    .join(row["name"].as_str().unwrap())
                    .join(format!("seed{seed}"));
                let result = if dir.join("result.json").exists() {
                    read(&dir.join("result.json"))?
                } else {
                    json!({"case":row["name"],"seed":seed,"arm":arm,"status":if dir.join("error.json").exists(){"execution_error"}else{"pending"},"first99_epochs":null})
                };
                rows.push(json!({"tier":row["tier"],"result":result}));
            }
        }
    }
    record(
        &root.join("summary.json"),
        &json!({"expected_cells":420,"rows":rows,"all_failures_retained":true}),
    )
}
fn archive(root: &Path) -> Result<()> {
    fs::create_dir_all(root)?;
    let path = root.join("protocol.json");
    let protocol = json!({"suite_sha256":hash(SUITE.as_bytes()),"plan_sha256":hash(PLAN.as_bytes()),"source_sha256":source_hash(),
        "ptx_sha256":ptx_hash(),"seeds":[0,1,2],"arms":ARMS,"expected_cells":420});
    if path.exists() {
        if read(&path)? != protocol {
            return Err("immutable complex-geometry cohort source changed".into());
        }
        return Ok(());
    }
    let source = root.join("source/research/complex_geometry_suite");
    for (name, body) in crate::sources::FILES {
        let dest = source.join(name);
        fs::create_dir_all(dest.parent().ok_or("source parent missing")?)?;
        durable(&dest, body.as_bytes())?;
    }
    record(&root.join("data-validation.json"), &preflight()?)?;
    record(&path, &protocol)
}

/// Eight updates on the real cubic task, kept outside scientific cell paths.
/// This checks the executable/data/scorer/checkpoint pipeline, not generalization.
fn smoke(root: &Path) -> Result<()> {
    if root.exists() {
        return Err("smoke output already exists; choose a fresh directory".into());
    }
    archive(root)?;
    let manifest = suite();
    let row = manifest["tasks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["name"] == "arithmetic-sum-cube")
        .ok_or("missing cubic task")?;
    let task = data::build(row)?;
    let train = &task.splits["train"];
    let batch = number(&row["case"], "batch")?;
    let updates = 8;
    let (order, edges) = order(train.len(), batch, updates, 0);
    let mut results = Vec::new();
    for arm in ARMS {
        let cfg = config(row, &task, arm)?;
        let initial = layout::initialize_model(&cfg, 0);
        let independent = layout::independent_count(&cfg);
        let mut engine = Engine::new(cfg, &initial)?;
        let mut packed = metrics::Packed::default();
        let before = metrics::evaluate(&mut engine, train, false)?;
        let started = Instant::now();
        for step in 0..updates {
            let indices = &order[edges[step]..edges[step + 1]];
            packed.fill(train, indices);
            engine.forward(&packed.tokens, &packed.labels, indices.len(), packed.length)?;
            engine.backward(indices.len(), packed.length)?;
            engine.update(step)?;
        }
        let after = metrics::evaluate(&mut engine, train, false)?;
        let validation = metrics::evaluate(&mut engine, &task.splits[&task.selection], true)?;
        if before["loss"].as_f64().is_none_or(|v| !v.is_finite())
            || after["loss"].as_f64().is_none_or(|v| !v.is_finite())
        {
            return Err("nonfinite smoke loss".into());
        }
        let point = json!({"step":updates,"examples":edges[updates],"seconds":started.elapsed().as_secs_f64(),
            "train":after,"validation":validation});
        let history = format!("{}\n", serde_json::to_string(&point)?);
        let dir = root.join("engineering").join(arm);
        fs::create_dir_all(&dir)?;
        checkpoint(&engine, &dir, updates, edges[updates], history.as_bytes())?;
        let row = json!({"arm":arm,"before":before,"after":point,
            "independent_parameters":independent,"expanded_parameters":initial.len(),
            "weights_changed":engine.parameters.read()?!=initial});
        println!("{row}");
        results.push(row);
    }
    record(
        &root.join("smoke.json"),
        &json!({"purpose":"engineering_smoke_only","case":"arithmetic-sum-cube",
        "updates":updates,"rows":results,"source_sha256":source_hash(),"ptx_sha256":ptx_hash()}),
    )
}

pub fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str){
        Some("preflight")=>{println!("{}",serde_json::to_string_pretty(&preflight()?)?);Ok(())},
        Some("report")if args.len()==3=>report(Path::new(&args[2])),
        Some("smoke")if args.len()==3=>smoke(Path::new(&args[2])),
        Some("cell"|"resume")if args.len()==7=>{
            let root=Path::new(&args[2]);let manifest=suite();let row=manifest["tasks"].as_array().unwrap().iter().find(|r|r["name"]==args[3]).ok_or("unknown case")?;
            if row["tier"]!=args[6]{return Err("case/tier mismatch".into());}
            let seed:u32=args[4].parse()?;if seed>2{return Err("undeclared seed".into());}
            archive(root)?;run_cell(root,row,seed,&args[5],args[1]=="resume")
        },
        Some("suite")if args.len()==4=>{
            let root=Path::new(&args[2]);let tier=&args[3];if !["short","long"].contains(&tier.as_str()){return Err("unknown tier".into());}archive(root)?;
            let manifest=suite();let executable=std::env::current_exe()?;
            for row in manifest["tasks"].as_array().unwrap(){
                if row["tier"]!=*tier{continue;}let name=row["name"].as_str().unwrap();
                for seed in 0..3{let mut arms=ARMS;arms.rotate_left(seed as usize);
                    for arm in arms{
                        if root.join("STOP").exists(){return Ok(());}
                        let dir=root.join(arm).join(name).join(format!("seed{seed}"));if dir.join("result.json").exists(){continue;}
                        let log=File::create(root.join(format!("{name}-{arm}-seed{seed}.log")))?;
                        let code=Command::new(&executable).arg("cell").arg(root).arg(name).arg(seed.to_string()).arg(arm).arg(tier).stdout(log.try_clone()?).stderr(log).status()?;
                        if !code.success(){fs::create_dir_all(&dir)?;record(&dir.join("error.json"),&json!({"status":"execution_error","exit_code":code.code()}))?;report(root)?;return Err(format!("cell failed; evidence retained: {name} {arm} {seed}").into());}
                        report(root)?;println!("{}",json!({"complete":name,"seed":seed,"arm":arm}));
                    }
                }
            }Ok(())
        },
        _=>Err("usage: complex-geometry preflight | smoke ROOT | suite ROOT short|long | report ROOT | cell|resume ROOT CASE SEED ARM TIER".into())
    }
}

#[test]
#[ignore = "requires CUDA"]
fn checkpoint_resumes_identical_updates_and_rejects_unclean_windows() -> Result<()> {
    for architecture in ARMS {
        let cfg = Config {
            width: 64,
            heads: 4,
            vocab: 11,
            layers: 2,
            max_batch: 3,
            max_length: 4,
            normalized: true,
            ssmax: true,
            gain: 1. / 16.,
            lr: 0.002,
            decay: 0.1,
            warmup: 50,
            architecture: architecture.trim_end_matches("-phase").into(),
            phase_optimizer: architecture.ends_with("-phase"),
        };
        let initial = layout::initialize_model(&cfg, 2);
        let mut first = Engine::new(cfg.clone(), &initial)?;
        let inputs = [1, 2, 3, 4, 5, 6];
        let labels = [u32::MAX, 3, u32::MAX, 5, u32::MAX, 7];
        for step in 0..2 {
            first.forward(&inputs, &labels, 3, 2)?;
            first.backward(3, 2)?;
            first.update(step)?;
        }
        let dir = std::env::temp_dir().join(format!("complex-ffn-resume-{}", std::process::id()));
        fs::create_dir(&dir)?;
        let history = b"{\"step\":2,\"seconds\":0.5}\n";
        checkpoint(&first, &dir, 2, 6, history)?;
        let mut second = Engine::new(cfg, &initial)?;
        let (step, restored, seconds) = restore(&mut second, &dir)?;
        assert_eq!((step, restored, seconds), (2, history.to_vec(), 0.5));
        // The next tail batch has different batch and sequence lengths.
        for engine in [&mut first, &mut second] {
            engine.forward(&[1, 2, 3], &[u32::MAX, u32::MAX, 4], 1, 3)?;
            engine.backward(1, 3)?;
            engine.update(step)?;
        }
        assert_eq!(first.parameters.read()?, second.parameters.read()?);
        layout::validate(&first.config, &first.parameters.read()?)?;
        assert_eq!(first.first_moment.read()?, second.first_moment.read()?);
        assert_eq!(first.second_moment.read()?, second.second_moment.read()?);
        record(
            &dir.join("execution.json"),
            &json!({"clean":false,"durable_completed":2,"possible_completed":500}),
        )?;
        assert!(restore(&mut second, &dir).is_err());
        fs::remove_dir_all(dir)?;
    }
    Ok(())
}

#[test]
fn archives_compiled_sources_and_rejects_stop_before_mutating_a_cell() -> Result<()> {
    let root =
        std::env::temp_dir().join(format!("complex-geometry-archive-{}", std::process::id()));
    fs::create_dir(&root)?;
    archive(&root)?;
    for (name, body) in crate::sources::FILES {
        assert_eq!(
            fs::read(
                root.join("source/research/complex_geometry_suite")
                    .join(name)
            )?,
            body.as_bytes()
        );
    }
    archive(&root)?;
    durable(&root.join("STOP"), b"test")?;
    assert!(run_cell(&root, &json!({}), 0, "complex-hermitian", true).is_err());
    assert!(!root.join("complex-hermitian").exists());
    let mut protocol = read(&root.join("protocol.json"))?;
    protocol["source_sha256"] = "mismatch".into();
    record(&root.join("protocol.json"), &protocol)?;
    assert!(archive(&root).is_err());
    fs::remove_dir_all(root)?;
    Ok(())
}
