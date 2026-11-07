//! Matched, post-budget LR continuation. Original experiment files are read-only.
use crate::*;
use std::io::{BufWriter, Write};
use std::time::Instant;

const EXTRA_EPOCHS: usize = 1000;
const FLUSH_EPOCHS: usize = 500;
const PLAN: &str = include_str!("../CONTINUATION_PLAN.txt");
const FILES: &[(&str, &str)] = &[
    ("src/main.rs", include_str!("main.rs")),
    ("src/continuation.rs", include_str!("continuation.rs")),
    ("src/engine.rs", include_str!("engine.rs")),
    ("src/stats.rs", include_str!("stats.rs")),
    ("src/probe.rs", include_str!("probe.rs")),
    ("src/raw.rs", include_str!("raw.rs")),
    ("src/loss_precision.rs", include_str!("loss_precision.rs")),
    ("src/loss_recheck.rs", include_str!("loss_recheck.rs")),
    ("src/stable_loss.rs", include_str!("stable_loss.rs")),
    (
        "src/stable_attention.rs",
        include_str!("stable_attention.rs"),
    ),
    ("src/gradient_oracle.rs", include_str!("gradient_oracle.rs")),
    ("src/gradient_stress.rs", include_str!("gradient_stress.rs")),
    ("Cargo.toml", include_str!("../Cargo.toml")),
    ("Cargo.lock", include_str!("../Cargo.lock")),
    ("CONTINUATION_PLAN.txt", PLAN),
];
const ARMS: [(&str, f64); 2] = [("unchanged", 1.), ("lower_lr", 0.1)];
fn file_hash(files: &[(&str, &str)]) -> String {
    let mut bytes = Vec::new();
    for &(name, body) in files {
        bytes.extend((name.len() as u64).to_le_bytes());
        bytes.extend(name.as_bytes());
        bytes.extend((body.len() as u64).to_le_bytes());
        bytes.extend(body.as_bytes());
    }
    hash(&bytes)
}
fn durable(path: &Path, bytes: &[u8]) -> Result<()> {
    let temporary = path.with_extension("pending");
    let mut file = BufWriter::new(fs::File::create(&temporary)?);
    file.write_all(bytes)?;
    file.flush()?;
    file.get_ref().sync_all()?;
    fs::rename(temporary, path)?;
    fs::File::open(path.parent().ok_or("missing parent")?)?.sync_all()?;
    Ok(())
}
fn record(path: &Path, value: &Value) -> Result<()> {
    durable(path, &serde_json::to_vec_pretty(value)?)
}
fn sha_text(hasher: Sha256) -> String {
    hasher
        .finalize()
        .iter()
        .map(|v| format!("{v:02x}"))
        .collect()
}
fn floats_hash(values: &[f32]) -> String {
    hash(
        &values
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}
struct Stream {
    order: Vec<u32>,
    edges: Vec<usize>,
    prefix_examples: usize,
    prefix_hash: String,
    future_hash: String,
}
fn stream(n: usize, batch: usize, skip: usize, updates: usize, seed: u32) -> Stream {
    let mut rng = random::Random::new(seed + 10000);
    let mut order = Vec::with_capacity(updates * batch);
    let mut edges = vec![0];
    let mut prefix = Sha256::new();
    let mut future = Sha256::new();
    let mut done = 0;
    let mut prefix_examples = 0;
    while done < skip + updates {
        let mut epoch: Vec<u32> = (0..n as u32).collect();
        rng.shuffle(&mut epoch);
        for chunk in epoch.chunks(batch) {
            if done == skip + updates {
                break;
            }
            if done < skip {
                prefix_examples += chunk.len();
                for value in chunk {
                    prefix.update(value.to_le_bytes());
                }
            } else {
                order.extend_from_slice(chunk);
                edges.push(order.len());
                for value in chunk {
                    future.update(value.to_le_bytes());
                }
            }
            done += 1;
        }
    }
    Stream {
        order,
        edges,
        prefix_examples,
        prefix_hash: sha_text(prefix),
        future_hash: sha_text(future),
    }
}
fn observation(
    engine: &mut Engine,
    task: &data::Task,
    cp: &Checkpoint,
    s: &Stream,
    completed: usize,
    started: Instant,
) -> Result<Value> {
    let train = metrics::evaluate(engine, &task.splits["train"], false)?;
    let heldout = metrics::evaluate(engine, &task.splits["validation"], false)?;
    let previous = cp.checkpoint["completed"].as_u64().unwrap() as usize;
    let n = task.splits["train"].len();
    let examples = s.prefix_examples + s.edges[completed];
    Ok(
        json!({"additional_updates":completed,"global_updates":previous+completed,
        "additional_examples":s.edges[completed],"global_examples":examples,
        "additional_epochs":s.edges[completed] as f64/n as f64,"global_epochs":examples as f64/n as f64,
        "seconds":started.elapsed().as_secs_f64(),"train":train,"validation":heldout}),
    )
}
fn state_bytes(engine: &Engine) -> Result<Vec<u8>> {
    let p = engine.parameters.read()?;
    let m = engine.first_moment.read()?;
    let v = engine.second_moment.read()?;
    layout::validate_state(&engine.config, &p, &m, &v)?;
    Ok(p.iter()
        .chain(&m)
        .chain(&v)
        .flat_map(|x| x.to_le_bytes())
        .collect())
}
fn checkpoint(
    dir: &Path,
    engine: &Engine,
    history: &[Value],
    source_hash: &str,
    parent_hash: &Value,
) -> Result<Value> {
    let point = history.last().ok_or("missing observation")?;
    let completed = point["additional_updates"].as_u64().unwrap() as usize;
    let state = state_bytes(engine)?;
    let mut bytes = Vec::new();
    for p in history {
        serde_json::to_writer(&mut bytes, p)?;
        bytes.push(b'\n');
    }
    let state_file = format!("checkpoint-{completed:06}.bin");
    let history_file = format!("history-{completed:06}.jsonl");
    durable(&dir.join(&state_file), &state)?;
    durable(&dir.join(&history_file), &bytes)?;
    let cp = json!({"additional_updates":completed,"global_updates":point["global_updates"],
        "additional_examples":point["additional_examples"],"global_examples":point["global_examples"],
        "state_file":state_file,"state_sha256":hash(&state),"history_file":history_file,"history_sha256":hash(&bytes),
        "source_sha256":source_hash,"parent_checkpoint_sha256":parent_hash,"point":point});
    record(&dir.join(format!("checkpoint-{completed:06}.json")), &cp)?;
    record(
        &dir.join("execution.json"),
        &json!({"clean":true,"durable_completed":completed,"possible_completed":completed}),
    )?;
    record(&dir.join("status.json"), point)?;
    Ok(cp)
}
struct FirstStep {
    parameters: Vec<f32>,
    gradient_hash: String,
    first_hash: String,
    second_hash: String,
}
fn first_step(engine: &Engine, gradient_hash: String) -> Result<FirstStep> {
    Ok(FirstStep {
        parameters: engine.parameters.read()?,
        gradient_hash,
        first_hash: floats_hash(&engine.first_moment.read()?),
        second_hash: floats_hash(&engine.second_moment.read()?),
    })
}
fn compare_first(cp: &Checkpoint, a: &FirstStep, b: &FirstStep, factor: f64) -> Result<Value> {
    require(
        a.gradient_hash == b.gradient_hash
            && a.first_hash == b.first_hash
            && a.second_hash == b.second_hash,
        "paired first-step gradients or moments differ",
    )?;
    let mut max_error: f64 = 0.;
    let mut large = 0.;
    let mut small = 0.;
    for ((&start, &high), &low) in cp.p.iter().zip(&a.parameters).zip(&b.parameters) {
        let d = high as f64 - start as f64;
        let e = low as f64 - start as f64;
        let error = (e - factor * d).abs();
        max_error = max_error.max(error);
        require(
            error <= 2e-7 * (1. + (start as f64).abs()),
            "first-step parameter changes do not match prescribed LR ratio",
        )?;
        large += d * d;
        small += e * e;
    }
    require(large > 0. && small > 0., "zero first-step update")?;
    Ok(
        json!({"identical_gradients":true,"identical_updated_moments":true,
        "first_step_max_scaling_error":max_error,"measured_displacement_norm_ratio":(small/large).sqrt(),
        "expected_lr_ratio":factor,"gradient_sha256":a.gradient_hash,"first_moment_sha256":a.first_hash,"second_moment_sha256":a.second_hash}),
    )
}
fn brief(point: &Value) -> Value {
    json!({"additional_updates":point["additional_updates"],"additional_epochs":point["additional_epochs"],
        "train_accuracy":point["train"]["sequence_accuracy"],"train_loss":point["train"]["loss"],
        "heldout_accuracy":point["validation"]["sequence_accuracy"],"heldout_loss":point["validation"]["loss"]})
}
fn summarize(history: &[Value]) -> Value {
    let first = history.iter().position(|p| {
        p["validation"]["exact"].as_u64().unwrap() * 100
            >= 99 * p["validation"]["examples"].as_u64().unwrap()
    });
    let tail = &history[history.len().saturating_sub(200)..];
    let values: Vec<f64> = tail
        .iter()
        .map(|p| p["validation"]["sequence_accuracy"].as_f64().unwrap())
        .collect();
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let stddev =
        (values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64).sqrt();
    let largest_drop = history
        .windows(2)
        .map(|w| {
            w[0]["validation"]["sequence_accuracy"].as_f64().unwrap()
                - w[1]["validation"]["sequence_accuracy"].as_f64().unwrap()
        })
        .fold(0., f64::max);
    let peak = history
        .iter()
        .max_by_key(|p| p["validation"]["exact"].as_u64().unwrap())
        .unwrap();
    json!({"initial":brief(&history[0]),"final":brief(history.last().unwrap()),"peak":brief(peak),
        "first99":first.map(|i|brief(&history[i])),"previous_first99_observation":first.and_then(|i|i.checked_sub(1)).map(|i|brief(&history[i])),
        "last200_mean":mean,"last200_stddev":stddev,"last200_min":values.iter().copied().fold(f64::INFINITY,f64::min),
        "last200_max":values.iter().copied().fold(f64::NEG_INFINITY,f64::max),
        "last200_fraction_above99":values.iter().filter(|&&v|v>=0.99).count() as f64/values.len() as f64,
        "largest_one_observation_drop":largest_drop,"observations":history.len()})
}
struct RunInputs<'a> {
    cp: &'a Checkpoint,
    task: &'a data::Task,
    stream: &'a Stream,
    root: &'a Path,
    source_hash: &'a str,
}
fn arm(
    input: &RunInputs<'_>,
    name: &str,
    factor: f64,
    control: Option<&FirstStep>,
) -> Result<(Value, FirstStep)> {
    let RunInputs {
        cp,
        task,
        stream: s,
        root,
        source_hash,
    } = input;
    let dir = root.join(name);
    fs::create_dir(&dir)?;
    let base = cp.checkpoint["completed"].as_u64().unwrap() as usize;
    let observe = cp.protocol["case"]["case"]["observe"].as_u64().unwrap() as usize;
    let per_epoch = task.splits["train"].len().div_ceil(256);
    let budget = EXTRA_EPOCHS * per_epoch;
    let flush = FLUSH_EPOCHS * per_epoch;
    let mut engine = cp.engine()?;
    engine.config.lr = cp.cfg.lr * factor;
    let protocol = json!({"name":name,"learning_rate":engine.config.lr,"decay_coefficient":engine.config.decay,
        "lr_factor":factor,"starting_global_update":base,"starting_examples":s.prefix_examples,
        "parent_checkpoint_sha256":cp.checkpoint["state_sha256"],"source_sha256":source_hash,
        "sampler_sha256":s.future_hash,"additional_updates":budget,"additional_epochs":EXTRA_EPOCHS,
        "checkpoint_epochs":FLUSH_EPOCHS,"observe_updates":observe,"optimizer_moments_reset":false,"warmup_restarted":false});
    record(&dir.join("protocol.json"), &protocol)?;
    let started = Instant::now();
    let initial = observation(&mut engine, task, cp, s, 0, started)?;
    for key in ["train", "validation"] {
        require(
            initial[key]["exact"] == cp.result["endpoint"][key]["exact"],
            "starting checkpoint scores changed",
        )?;
        require(
            (initial[key]["loss"].as_f64().unwrap()
                - cp.result["endpoint"][key]["loss"].as_f64().unwrap())
            .abs()
                < 1e-7,
            "starting checkpoint loss changed",
        )?;
    }
    let mut history = vec![initial];
    let mut snapshots = vec![checkpoint(
        &dir,
        &engine,
        &history,
        source_hash,
        &cp.checkpoint["state_sha256"],
    )?];
    require(
        snapshots[0]["state_sha256"] == cp.checkpoint["state_sha256"],
        "loaded state differs from complete parent checkpoint",
    )?;
    let mut completed = 0;
    let mut window_end = flush.min(budget);
    let mut first99 = false;
    let mut first = None;
    let mut first_check = Value::Null;
    let mut consumed = Sha256::new();
    let mut packed = metrics::Packed::default();
    record(
        &dir.join("execution.json"),
        &json!({"clean":false,"durable_completed":0,"possible_completed":window_end}),
    )?;
    while completed < budget {
        let next = (completed + observe).min(budget).min(window_end);
        while completed < next {
            let indices = &s.order[s.edges[completed]..s.edges[completed + 1]];
            packed.fill(&task.splits["train"], indices);
            engine.forward(&packed.tokens, &packed.labels, indices.len(), packed.length)?;
            engine.backward(indices.len(), packed.length)?;
            let gradient_hash = if completed == 0 {
                Some(floats_hash(&engine.gradients.read()?))
            } else {
                None
            };
            engine.update(base + completed)?;
            if let Some(gradient_hash) = gradient_hash {
                let point = first_step(&engine, gradient_hash)?;
                if let Some(control) = control {
                    first_check = compare_first(cp, control, &point, factor)?;
                }
                first = Some(point);
            }
            for index in indices {
                consumed.update(index.to_le_bytes());
            }
            completed += 1;
        }
        let point = observation(&mut engine, task, cp, s, completed, started)?;
        let crossing = !first99
            && point["validation"]["exact"].as_u64().unwrap() * 100
                >= 99 * task.splits["validation"].len() as u64;
        first99 |= crossing;
        history.push(point);
        let stopped = root.join("STOP").exists();
        let done = completed == budget || stopped;
        if completed == window_end || crossing || done {
            snapshots.push(checkpoint(
                &dir,
                &engine,
                &history,
                source_hash,
                &cp.checkpoint["state_sha256"],
            )?);
            println!(
                "{}",
                json!({"arm":name,"checkpoint":brief(history.last().unwrap()),"first99":crossing,"finished":done})
            );
            if !done {
                window_end = ((completed / flush + 1) * flush).min(budget);
                record(
                    &dir.join("execution.json"),
                    &json!({"clean":false,"durable_completed":completed,"possible_completed":window_end}),
                )?;
            }
        }
        if done {
            break;
        }
    }
    let consumed_hash = sha_text(consumed);
    if completed == budget {
        require(
            consumed_hash == s.future_hash,
            "training did not consume the declared continuation sample stream",
        )?;
    }
    cp.unchanged()?;
    let result = json!({"arm":name,"status":if completed==budget{"complete"}else{"stopped"},"protocol":protocol,
        "consumed_sampler_sha256":consumed_hash,"snapshots":snapshots,"summary":summarize(&history),"first_step_comparison":first_check,
        "parent_checkpoint_unchanged":true,"seconds":started.elapsed().as_secs_f64()});
    record(&dir.join("result.json"), &result)?;
    Ok((result, first.ok_or("no training step completed")?))
}

// Re-read artifacts, rather than trusting in-memory counters or the runner's summary.
fn audit(input: &RunInputs<'_>, name: &str) -> Result<Value> {
    let RunInputs {
        cp,
        task,
        stream: s,
        root,
        source_hash,
    } = input;
    let dir = root.join(name);
    let result = read(&dir.join("result.json"))?;
    let execution = read(&dir.join("execution.json"))?;
    require(
        execution["clean"] == true
            && execution["durable_completed"] == execution["possible_completed"],
        "uncertain continuation exposure",
    )?;
    let budget = EXTRA_EPOCHS * task.splits["train"].len().div_ceil(256);
    let observe = cp.protocol["case"]["case"]["observe"].as_u64().unwrap() as usize;
    let base = cp.checkpoint["completed"].as_u64().unwrap() as usize;
    let mut final_history = Vec::new();
    let mut snapshot_steps = Vec::new();
    for snapshot in result["snapshots"].as_array().ok_or("snapshots missing")? {
        require(
            snapshot["source_sha256"] == *source_hash
                && snapshot["parent_checkpoint_sha256"] == cp.checkpoint["state_sha256"],
            "snapshot provenance changed",
        )?;
        let bytes = fs::read(dir.join(snapshot["state_file"].as_str().unwrap()))?;
        let raw = fs::read(dir.join(snapshot["history_file"].as_str().unwrap()))?;
        require(
            hash(&bytes) == snapshot["state_sha256"] && hash(&raw) == snapshot["history_sha256"],
            "snapshot integrity failure",
        )?;
        let n = cp.p.len();
        require(bytes.len() == 12 * n, "invalid state length")?;
        let state: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        layout::validate_state(&cp.cfg, &state[..n], &state[n..2 * n], &state[2 * n..])?;
        let records: Vec<Value> = raw
            .split(|&b| b == b'\n')
            .filter(|v| !v.is_empty())
            .map(serde_json::from_slice)
            .collect::<std::result::Result<_, _>>()?;
        for (i, p) in records.iter().enumerate() {
            let step = i * observe;
            require(
                step <= budget
                    && p["additional_updates"] == step
                    && p["global_updates"] == base + step,
                "observation cadence or global counter changed",
            )?;
            require(
                p["additional_examples"] == s.edges[step]
                    && p["global_examples"] == s.prefix_examples + s.edges[step],
                "sample exposure changed",
            )?;
            for (split, n) in [
                ("train", task.splits["train"].len()),
                ("validation", task.splits["validation"].len()),
            ] {
                let m = &p[split];
                let exact = m["exact"].as_u64().ok_or("exact missing")? as usize;
                require(
                    m["examples"] == n
                        && exact <= n
                        && m["loss"].as_f64().is_some_and(f64::is_finite),
                    "incomplete/nonfinite split metrics",
                )?;
                require(
                    (m["sequence_accuracy"].as_f64().ok_or("accuracy missing")?
                        - exact as f64 / n as f64)
                        .abs()
                        < 1e-12,
                    "wrong exact accuracy denominator",
                )?;
            }
        }
        require(
            records.last() == Some(&snapshot["point"]),
            "snapshot/history endpoint mismatch",
        )?;
        snapshot_steps.push(snapshot["additional_updates"].as_u64().unwrap() as usize);
        final_history = records;
    }
    let mut summary_roundoff = Vec::new();
    compare_summary(
        "summary",
        &result["summary"],
        &summarize(&final_history),
        &mut summary_roundoff,
    )?;
    if result["status"] == "complete" {
        require(
            snapshot_steps.contains(&0)
                && snapshot_steps.contains(&(budget / 2))
                && snapshot_steps.contains(&budget),
            "missing declared500-epoch save",
        )?;
        require(
            result["consumed_sampler_sha256"] == s.future_hash,
            "different consumed sample stream",
        )?;
        require(
            final_history.len() == budget / observe + 1,
            "wrong fixed continuation budget",
        )?;
    }
    cp.unchanged()?;
    Ok(
        json!({"arm":name,"status":result["status"],"audit":"passed","snapshots":snapshot_steps,"observations":final_history.len(),
        "summary":result["summary"],"first_step_comparison":result["first_step_comparison"],"summary_roundoff":summary_roundoff}),
    )
}
pub fn run(parent: &Path, root: &Path) -> Result<()> {
    require(
        !root.exists(),
        "refuse overwrite or silent replay of existing continuation directory",
    )?;
    let cp = Checkpoint::load(parent, "complex-hermitian", 1)?;
    require(
        cp.checkpoint["completed"] == 100000 && cp.result["status"] == "budget_exhausted",
        "wrong diagnostic parent",
    )?;
    require(
        file_hash(training_sources::FILES) == cp.protocol["source_sha256"],
        "training sources differ from the checkpoint archive",
    )?;
    let task = data::build(&cp.protocol["case"])?;
    require(
        task.selection == "validation"
            && task.splits["train"].len() == 4704
            && task.splits["validation"].len() == 4705,
        "wrong fixed task",
    )?;
    let per_epoch = 4704usize.div_ceil(256);
    let s = stream(4704, 256, 100000, EXTRA_EPOCHS * per_epoch, 1);
    require(
        s.prefix_hash == cp.protocol["sampler_sha256"]
            && cp.checkpoint["examples"] == s.prefix_examples,
        "parent sample order/exposure mismatch",
    )?;
    require(
        s.order.len() == EXTRA_EPOCHS * 4704,
        "additional exposure differs from declared1000epochs",
    )?;
    fs::create_dir(root)?;
    for (group, files) in [("diagnostic", FILES), ("training", training_sources::FILES)] {
        for &(name, body) in files {
            let target = root.join("source").join(group).join(name);
            fs::create_dir_all(target.parent().unwrap())?;
            durable(&target, body.as_bytes())?;
        }
    }
    let source_hash = file_hash(FILES);
    let protocol = json!({"case":CASE,"seed":1,"model":"complex-hermitian","parent_root":parent,
        "parent_checkpoint_sha256":cp.checkpoint["state_sha256"],"parent_source_sha256":cp.protocol["source_sha256"],
        "source_sha256":source_hash,"canonical_ptx_sha256":ptx_hash(),"plan_sha256":hash(PLAN.as_bytes()),
        "parent_updates":100000,"parent_examples":s.prefix_examples,"parent_sampler_sha256":s.prefix_hash,
        "continuation_sampler_sha256":s.future_hash,"additional_updates":EXTRA_EPOCHS*per_epoch,"additional_examples":s.order.len(),
        "additional_epochs":EXTRA_EPOCHS,"arms":[{"name":"unchanged","lr":cp.cfg.lr},{"name":"lower_lr","lr":cp.cfg.lr*0.1}],
        "checkpoint_epochs":FLUSH_EPOCHS,"frozen_benchmark_modified":false,"python_invoked":false});
    record(&root.join("protocol.json"), &protocol)?;
    let input = RunInputs {
        cp: &cp,
        task: &task,
        stream: &s,
        root,
        source_hash: &source_hash,
    };
    let mut first = None;
    let mut rows = Vec::new();
    for (name, factor) in ARMS {
        if root.join("STOP").exists() {
            break;
        }
        let (result, step) = arm(&input, name, factor, first.as_ref())?;
        if first.is_none() {
            first = Some(step);
        }
        rows.push(audit(&input, name)?);
        record(
            &root.join("summary.json"),
            &json!({"complete":rows.len()==2&&rows.iter().all(|r|r["status"]=="complete"),"expected_arms":2,"rows":rows,"protocol":protocol}),
        )?;
        if result["status"] != "complete" {
            break;
        }
    }
    cp.unchanged()?;
    Ok(())
}

// Integer counts, strings, nulls, keys and shapes remain exact. Only derived
// floating-point summaries allow roundoff from decimal parsing and recomputation.
fn compare_summary(path: &str, a: &Value, b: &Value, roundoff: &mut Vec<Value>) -> Result<()> {
    match (a, b) {
        (Value::Object(a), Value::Object(b)) => {
            require(a.len() == b.len(), "summary keys changed")?;
            for (key, value) in a {
                compare_summary(
                    &format!("{path}.{key}"),
                    value,
                    b.get(key).ok_or("summary key missing")?,
                    roundoff,
                )?;
            }
        }
        (Value::Array(a), Value::Array(b)) => {
            require(a.len() == b.len(), "summary array length changed")?;
            for (i, (a, b)) in a.iter().zip(b).enumerate() {
                compare_summary(&format!("{path}[{i}]"), a, b, roundoff)?;
            }
        }
        (Value::Number(a), Value::Number(b)) if a.is_f64() && b.is_f64() => {
            let (x, y) = (a.as_f64().unwrap(), b.as_f64().unwrap());
            let error = (x - y).abs();
            require(
                error <= 1e-12 * (1. + x.abs().max(y.abs())),
                &format!("summary differs at {path}: {x} vs {y}"),
            )?;
            if error > 0. {
                roundoff
                    .push(json!({"path":path,"reported":x,"recomputed":y,"absolute_error":error}));
            }
        }
        _ => require(a == b, &format!("summary differs at {path}"))?,
    }
    Ok(())
}
fn existing(parent: &Path, root: &Path) -> Result<(Checkpoint, data::Task, Stream, Value)> {
    let cp = Checkpoint::load(parent, "complex-hermitian", 1)?;
    let protocol = read(&root.join("protocol.json"))?;
    require(
        protocol["parent_checkpoint_sha256"] == cp.checkpoint["state_sha256"]
            && protocol["plan_sha256"] == hash(PLAN.as_bytes())
            && protocol["additional_updates"] == 19000
            && protocol["canonical_ptx_sha256"] == ptx_hash(),
        "existing continuation protocol changed",
    )?;
    require(
        file_hash(training_sources::FILES) == cp.protocol["source_sha256"],
        "model source changed",
    )?;
    let task = data::build(&cp.protocol["case"])?;
    let s = stream(4704, 256, 100000, 19000, 1);
    require(
        s.prefix_hash == protocol["parent_sampler_sha256"]
            && s.future_hash == protocol["continuation_sampler_sha256"],
        "sample stream changed",
    )?;
    let archived: Vec<(String, String)> = FILES
        .iter()
        .map(|(name, _)| {
            Ok((
                name.to_string(),
                fs::read_to_string(root.join("source/diagnostic").join(name))?,
            ))
        })
        .collect::<Result<_>>()?;
    let refs: Vec<_> = archived
        .iter()
        .map(|(name, body)| (name.as_str(), body.as_str()))
        .collect();
    require(
        file_hash(&refs) == protocol["source_sha256"],
        "original continuation archive changed",
    )?;
    Ok((cp, task, s, protocol))
}
pub fn audit_saved(parent: &Path, root: &Path) -> Result<()> {
    let (cp, task, s, protocol) = existing(parent, root)?;
    let source_hash = protocol["source_sha256"].as_str().unwrap();
    let row = audit(
        &RunInputs {
            cp: &cp,
            task: &task,
            stream: &s,
            root,
            source_hash,
        },
        "unchanged",
    )?;
    record(&root.join("audit-correction.json"), &row)?;
    println!(
        "{}",
        json!({"audit":row["audit"],"summary_roundoff":row["summary_roundoff"]})
    );
    Ok(())
}
pub fn finish(parent: &Path, root: &Path) -> Result<()> {
    require(
        !root.join("lower_lr").exists(),
        "lower-LR arm already exists; refuse replay",
    )?;
    require(!root.join("STOP").exists(), "STOP requested")?;
    let (cp, task, s, protocol) = existing(parent, root)?;
    let old_hash = protocol["source_sha256"].as_str().unwrap();
    let old_input = RunInputs {
        cp: &cp,
        task: &task,
        stream: &s,
        root,
        source_hash: old_hash,
    };
    let control = audit(&old_input, "unchanged")?;
    require(
        control["status"] == "complete",
        "only a completed, audited control can be retained",
    )?;
    // Protect the numerical experiment: only dispatch/audit/recovery code may
    // differ from the archived control executable. Everything used to train
    // and measure either arm is text-identical.
    for &(name, body) in FILES {
        let prior = fs::read_to_string(root.join("source/diagnostic").join(name))?;
        let marker = match name {
            "src/continuation.rs" => Some("// Re-read artifacts, rather than trusting"),
            "src/main.rs" => Some("fn run() -> Result<()>"),
            _ => None,
        };
        if let Some(marker) = marker {
            require(
                prior
                    .split_once(marker)
                    .ok_or("missing old source boundary")?
                    .0
                    == body.split_once(marker).ok_or("missing source boundary")?.0,
                "scientific continuation code changed",
            )?;
        } else {
            require(prior == body, "non-audit continuation dependency changed")?;
        }
    }
    let new_hash = file_hash(FILES);
    for &(name, body) in FILES {
        let path = root.join("source-after-audit-fix/diagnostic").join(name);
        fs::create_dir_all(path.parent().unwrap())?;
        durable(&path, body.as_bytes())?;
    }
    // The original process's first-step reference lived only in RAM. Recompute
    // one discarded validation step, never restarting the completed control.
    let mut reference = cp.engine()?;
    let mut packed = metrics::Packed::default();
    let indices = &s.order[..s.edges[1]];
    packed.fill(&task.splits["train"], indices);
    reference.forward(&packed.tokens, &packed.labels, indices.len(), packed.length)?;
    reference.backward(indices.len(), packed.length)?;
    let gradient_hash = floats_hash(&reference.gradients.read()?);
    reference.update(100000)?;
    let first = first_step(&reference, gradient_hash)?;
    drop(reference);
    let recovery = json!({"reason":"Post-run summary equality rejected decimal float roundoff; no training failure",
        "control_source_sha256":old_hash,"remaining_source_sha256":new_hash,"scientific_code_text_identical":true,
        "completed_control_updates_replayed":0,"discarded_engineering_reference_updates":1,
        "control_summary_roundoff":control["summary_roundoff"],"first_step_reference":"recomputed from identical parent and first continuation batch; discarded"});
    record(&root.join("audit-recovery.json"), &recovery)?;
    record(
        &root.join("summary.json"),
        &json!({"complete":false,"expected_arms":2,"rows":[control],"protocol":protocol,"audit_recovery":recovery}),
    )?;
    let input = RunInputs {
        cp: &cp,
        task: &task,
        stream: &s,
        root,
        source_hash: &new_hash,
    };
    arm(&input, "lower_lr", 0.1, Some(&first))?;
    let lower = audit(&input, "lower_lr")?;
    record(
        &root.join("summary.json"),
        &json!({"complete":lower["status"]=="complete","expected_arms":2,"rows":[control,lower],"protocol":protocol,"audit_recovery":recovery}),
    )?;
    cp.unchanged()?;
    Ok(())
}

#[test]
fn summary_audit_accepts_roundoff_and_rejects_meaningful_changes() {
    let mut differences = Vec::new();
    compare_summary(
        "test",
        &json!({"accuracy":0.97,"count":19000}),
        &json!({"accuracy":0.9700000000000001,"count":19000}),
        &mut differences,
    )
    .unwrap();
    assert_eq!(differences.len(), 1);
    assert!(
        compare_summary(
            "test",
            &json!({"accuracy":0.97}),
            &json!({"accuracy":0.9701}),
            &mut differences
        )
        .is_err()
    );
    assert!(compare_summary("test", &json!(19000), &json!(19001), &mut differences).is_err());
}

#[test]
fn continuation_preserves_partial_epoch_and_tail_batch() {
    let s = stream(10, 4, 5, 7, 1);
    assert_eq!(s.prefix_examples, 18);
    assert_eq!(
        s.edges.windows(2).map(|w| w[1] - w[0]).collect::<Vec<_>>(),
        vec![2, 4, 4, 2, 4, 4, 2]
    );
    for range in [2..12, 12..22] {
        let mut epoch = s.order[range].to_vec();
        epoch.sort_unstable();
        assert_eq!(epoch, (0..10).collect::<Vec<_>>());
    }
    let all = stream(10, 4, 0, 12, 1);
    assert_eq!(s.order, all.order[18..]);
    assert_eq!(
        s.prefix_hash,
        hash(
            &all.order[..18]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>()
        )
    );
}
