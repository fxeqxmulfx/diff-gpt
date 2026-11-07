//! One fresh seed1 rerun with the tested numerical corrections; no baseline mutation.
use crate::continuation::{Stream, durable, record, stream};
use crate::*;
use std::time::Instant;

const BUDGET: usize = 100000;
const OBSERVE: usize = 19;
const FLUSH: usize = 500 * OBSERVE;
const PLAN: &str = include_str!("../RETRAIN_PLAN.txt");

pub(crate) fn setup(parent_root: &Path) -> Result<(Checkpoint, data::Task, Stream)> {
    let parent = Checkpoint::load(parent_root, "complex-hermitian", 1)?;
    require(
        parent.protocol["case"]["case"]["data"] == json!({"operation":"sum-pow4","prime":97})
            && parent.protocol["seed"] == 1
            && parent.result["completed"] == BUDGET
            && parent.protocol["case"]["case"]["observe"] == OBSERVE,
        "unexpected baseline task or budget",
    )?;
    let task = data::build(&parent.protocol["case"])?;
    let samples = stream(task.splits["train"].len(), 256, 0, BUDGET, 1);
    require(
        samples.future_hash == parent.protocol["sampler_sha256"]
            && samples.edges[BUDGET] == parent.result["examples"].as_u64().unwrap() as usize,
        "rerun sampler differs from the original100000 updates",
    )?;
    Ok((parent, task, samples))
}
pub(crate) fn corrected(cfg: Config, p: &[f32], m: &[f32], v: &[f32]) -> Result<Engine> {
    let mut e = Engine::new(cfg, p)?;
    e.first_moment.write(m)?;
    e.second_moment.write(v)?;
    e.enable_stable_loss()?;
    e.enable_stable_attention()?;
    Ok(e)
}
pub(crate) fn protected(parent: &Checkpoint) -> Result<Value> {
    let mut files = serde_json::Map::new();
    for name in [
        "checkpoint.bin",
        "checkpoint.json",
        "initial.f32",
        "history.jsonl",
        "protocol.json",
        "result.json",
        "execution.json",
    ] {
        files.insert(name.into(), hash(&fs::read(parent.dir.join(name))?).into());
    }
    Ok(Value::Object(files))
}
pub(crate) fn archive(root: &Path) -> Result<Value> {
    let mut files = Vec::new();
    let mut bytes = Vec::new();
    for (prefix, entries) in [("", training_sources::FILES), ("diagnostic/", raw::FILES)] {
        for &(name, body) in entries {
            let relative = format!("{prefix}{name}");
            let dest = root
                .join("source/research/complex_geometry_suite")
                .join(&relative);
            fs::create_dir_all(dest.parent().unwrap())?;
            durable(&dest, body.as_bytes())?;
            bytes.extend((relative.len() as u64).to_le_bytes());
            bytes.extend(relative.as_bytes());
            bytes.extend((body.len() as u64).to_le_bytes());
            bytes.extend(body.as_bytes());
            files.push(json!({"path":relative,"sha256":hash(body.as_bytes())}));
        }
    }
    Ok(json!({"sha256":hash(&bytes),"files":files}))
}
pub(crate) fn observe(
    e: &mut Engine,
    task: &data::Task,
    samples: &Stream,
    step: usize,
    started: &Instant,
) -> Result<Value> {
    if e.uses_plain_product() {
        return Ok(
            json!({"step":step,"examples":samples.edges[step],"epochs":samples.edges[step] as f64/task.splits["train"].len() as f64,
            "seconds":started.elapsed().as_secs_f64(),"train":momentum_continuation::evaluate(e,&task.splits["train"])?,
            "validation":momentum_continuation::evaluate(e,&task.splits["validation"])?}),
        );
    }
    Ok(
        json!({"step":step,"examples":samples.edges[step],"epochs":samples.edges[step] as f64/task.splits["train"].len() as f64,
        "seconds":started.elapsed().as_secs_f64(),"train":metrics::evaluate(e,&task.splits["train"],false)?,
        "validation":metrics::evaluate(e,&task.splits["validation"],true)?}),
    )
}
pub(crate) fn checkpoint(
    e: &Engine,
    root: &Path,
    history: &[u8],
    point: &Value,
    protocol: &Value,
) -> Result<Value> {
    let p = e.parameters.read()?;
    let m = e.first_moment.read()?;
    let v = e.second_moment.read()?;
    layout::validate_state(&e.config, &p, &m, &v)?;
    let bytes: Vec<_> = p
        .iter()
        .chain(&m)
        .chain(&v)
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let step = point["step"].as_u64().unwrap();
    let meta = json!({"step":step,"examples":point["examples"],"epochs":point["epochs"],"state_sha256":hash(&bytes),
        "history_sha256":hash(history),"history_bytes":history.len(),"source_sha256":protocol["source"]["sha256"],
        "corrected_loss_ptx_sha256":protocol["corrected_loss_ptx_sha256"],"corrected_attention_ptx_sha256":protocol["corrected_attention_ptx_sha256"],
        "layout":"parameters, first moment, second moment; f32 little endian"});
    durable(&root.join(format!("checkpoint-{step:06}.bin")), &bytes)?;
    durable(&root.join("history.jsonl"), history)?;
    record(&root.join(format!("checkpoint-{step:06}.json")), &meta)?;
    record(&root.join("status.json"), point)?;
    record(
        &root.join("execution.json"),
        &json!({"durable_completed":step,"possible_completed":step,"clean":true}),
    )?;
    Ok(meta)
}
pub fn run(parent_root: &Path, root: &Path) -> Result<()> {
    require(
        !root.exists(),
        "refuse overwrite or replay of corrected-gradient rerun",
    )?;
    let (parent, task, samples) = setup(parent_root)?;
    let before = protected(&parent)?;
    fs::create_dir(root)?;
    let source = archive(root)?;
    let initial: Vec<_> = parent
        .initial
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let protocol = json!({"purpose":"fresh corrected-gradient rerun of the existing sum-pow4 seed1; original benchmark preserved",
        "case":parent.protocol["case"],"seed":1,"arm":"complex-hermitian","baseline_protocol":parent.protocol,
        "original_files":before,"source":source,"plan_sha256":hash(PLAN.as_bytes()),"budget":BUDGET,"observe":OBSERVE,
        "checkpoint_epochs":500,"initial_parameters_sha256":hash(&initial),"sampler_sha256":samples.future_hash,
        "original_ptx_sha256":ptx_hash(),"corrected_loss_ptx_sha256":hash(stable_loss::kernels::PTX.as_bytes()),
        "corrected_attention_ptx_sha256":hash(stable_attention::kernels::PTX.as_bytes()),"optimizer_moments":"zero at original initialization",
        "stopping_rule":"first observed heldout exact >=99%,100000updates,or explicit STOP","python_invoked":false});
    require(
        protocol["initial_parameters_sha256"] == parent.protocol["initial_parameters_sha256"],
        "initial bytes changed",
    )?;
    record(&root.join("protocol.json"), &protocol)?;
    durable(&root.join("initial.f32"), &initial)?;
    durable(&root.join("PLAN.txt"), PLAN.as_bytes())?;
    let zeros = vec![0.; parent.initial.len()];
    let mut e = corrected(parent.cfg.clone(), &parent.initial, &zeros, &zeros)?;
    let started = Instant::now();
    let mut history = Vec::new();
    let mut snapshots = Vec::new();
    let mut completed = 0;
    let mut window_end = 0;
    let mut previous = 0;
    let mut packed = metrics::Packed::default();
    loop {
        let point = observe(&mut e, &task, &samples, completed, &started)?;
        if completed == 0 {
            for split in ["train", "validation"] {
                require(
                    point[split]["exact"] == parent.history[0][split]["exact"],
                    "initial classifications changed",
                )?;
            }
        }
        history.extend(serde_json::to_vec(&point)?);
        history.push(b'\n');
        let passed = point["validation"]["exact"].as_u64().unwrap() * 100
            >= 99 * task.splits["validation"].len() as u64;
        let stopped = root.join("STOP").exists();
        let done = passed || stopped || completed == BUDGET;
        if completed == 0 || completed == window_end || done {
            snapshots.push(checkpoint(&e, root, &history, &point, &protocol)?);
            println!(
                "{}",
                json!({"updates":completed,"epochs":point["epochs"],"train":point["train"]["sequence_accuracy"],
                "heldout":point["validation"]["sequence_accuracy"],"seconds":point["seconds"]})
            );
            if !done {
                window_end = ((completed / FLUSH + 1) * FLUSH).min(BUDGET);
                record(
                    &root.join("execution.json"),
                    &json!({"clean":false,"durable_completed":completed,"possible_completed":window_end}),
                )?;
            }
        }
        if done {
            let result = json!({"case":CASE,"seed":1,"arm":"complex-hermitian","status":if passed{"passed"}else if stopped{"stopped"}else{"budget_exhausted"},
                "completed":completed,"examples":samples.edges[completed],"epochs":point["epochs"],"first99_epochs":if passed{Some(point["epochs"].clone())}else{None},
                "previous_observed_epochs":if passed{Some(samples.edges[previous] as f64/task.splits["train"].len() as f64)}else{None},
                "endpoint":point,"snapshots":snapshots,"source_sha256":protocol["source"]["sha256"],"zero_rate_first_update_checked":true});
            record(&root.join("result.json"), &result)?;
            require(
                protected(&parent)? == before,
                "rerun changed baseline files",
            )?;
            return audit(parent_root, root);
        }
        previous = completed;
        let next = (completed + OBSERVE).min(window_end).min(BUDGET);
        while completed < next {
            let batch = &samples.order[samples.edges[completed]..samples.edges[completed + 1]];
            packed.fill(&task.splits["train"], batch);
            e.forward(&packed.tokens, &packed.labels, batch.len(), packed.length)?;
            e.backward(batch.len(), packed.length)?;
            e.update(completed)?;
            if completed == 0 {
                require(
                    e.parameters.read()? == parent.initial,
                    "zero-rate first update changed weights",
                )?;
                require(
                    e.first_moment.read()?.iter().any(|&v| v != 0.)
                        && e.second_moment.read()?.iter().any(|&v| v > 0.),
                    "first update did not insert moments",
                )?;
            }
            completed += 1;
        }
    }
}

pub(crate) fn decode(path: &Path, n: usize) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let bytes = fs::read(path)?;
    require(bytes.len() == 12 * n, "wrong checkpoint size")?;
    let values: Vec<_> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&x| f32::from_le_bytes(x))
        .collect();
    Ok((
        values[..n].to_vec(),
        values[n..2 * n].to_vec(),
        values[2 * n..].to_vec(),
    ))
}
pub(crate) fn audit_readonly(parent_root: &Path, root: &Path) -> Result<Value> {
    let (parent, task, samples) = setup(parent_root)?;
    let protocol = read(&root.join("protocol.json"))?;
    let plain = match protocol["ffn"].as_str() {
        None | Some("radial_complex_product") => false,
        Some("plain_complex_product") => {
            require(
                protocol["plain_product_ptx_sha256"]
                    == hash(plain_product::kernels::PTX.as_bytes()),
                "plain product kernel mismatch",
            )?;
            true
        }
        _ => return Err("unknown checkpoint FFN".into()),
    };
    require(
        protocol["original_files"] == protected(&parent)?
            && protocol["baseline_protocol"] == parent.protocol,
        "baseline changed",
    )?;
    require(
        protocol["sampler_sha256"] == samples.future_hash
            && protocol["budget"] == BUDGET
            && protocol["observe"] == OBSERVE,
        "rerun protocol changed",
    )?;
    require(
        protocol["corrected_loss_ptx_sha256"] == hash(stable_loss::kernels::PTX.as_bytes())
            && protocol["corrected_attention_ptx_sha256"]
                == hash(stable_attention::kernels::PTX.as_bytes()),
        "corrected kernel mismatch",
    )?;
    let mut source = Vec::new();
    for entry in protocol["source"]["files"]
        .as_array()
        .ok_or("missing source manifest")?
    {
        let name = entry["path"].as_str().unwrap();
        let body = fs::read(
            root.join("source/research/complex_geometry_suite")
                .join(name),
        )?;
        require(hash(&body) == entry["sha256"], "archived source modified")?;
        source.extend((name.len() as u64).to_le_bytes());
        source.extend(name.as_bytes());
        source.extend((body.len() as u64).to_le_bytes());
        source.extend(body);
    }
    require(
        hash(&source) == protocol["source"]["sha256"]
            && hash(&fs::read(root.join("PLAN.txt"))?) == protocol["plan_sha256"],
        "source or plan hash mismatch",
    )?;
    let initial = fs::read(root.join("initial.f32"))?;
    require(
        initial == fs::read(parent.dir.join("initial.f32"))?
            && hash(&initial) == protocol["initial_parameters_sha256"],
        "initialization is not paired",
    )?;
    let result = read(&root.join("result.json"))?;
    let execution = read(&root.join("execution.json"))?;
    let completed = result["completed"].as_u64().ok_or("missing completed")? as usize;
    require(
        completed <= BUDGET
            && execution["clean"] == true
            && execution["durable_completed"] == completed
            && execution["possible_completed"] == completed,
        "unclean rerun",
    )?;
    let bytes = fs::read(root.join("history.jsonl"))?;
    let history: Vec<Value> = bytes
        .split(|&b| b == b'\n')
        .filter(|v| !v.is_empty())
        .map(serde_json::from_slice)
        .collect::<std::result::Result<_, _>>()?;
    let mut first = None;
    let mut prior = 0;
    for (i, point) in history.iter().enumerate() {
        let step = point["step"].as_u64().unwrap() as usize;
        require(
            step == (if i == 0 {
                0
            } else {
                (prior + OBSERVE).min(BUDGET)
            }) && point["examples"] == samples.edges[step],
            "observation/exposure gap",
        )?;
        require(
            (point["epochs"].as_f64().unwrap() - samples.edges[step] as f64 / 4704.).abs() < 1e-10,
            "epoch mismatch",
        )?;
        for split in ["train", "validation"] {
            let metric = &point[split];
            let n = task.splits[split].len();
            let exact = metric["exact"].as_u64().ok_or("missing exact")?;
            require(
                exact <= n as u64
                    && metric["examples"] == n
                    && (metric["sequence_accuracy"].as_f64().unwrap() - exact as f64 / n as f64)
                        .abs()
                        < 1e-14
                    && metric["loss"]
                        .as_f64()
                        .is_some_and(|v| v.is_finite() && v >= 0.),
                "invalid stored score",
            )?;
        }
        if point["validation"]["exact"].as_u64().unwrap() * 100 >= 99 * 4705 && first.is_none() {
            first = Some(step);
        }
        prior = step;
    }
    require(
        prior == completed && history.last() == Some(&result["endpoint"]),
        "endpoint/history mismatch",
    )?;
    match result["status"].as_str() {
        Some("passed") => require(
            first == Some(completed) && result["first99_epochs"] == result["epochs"],
            "not the first99 crossing",
        )?,
        Some("budget_exhausted") => require(
            completed == BUDGET && first.is_none() && result["first99_epochs"].is_null(),
            "invalid censored endpoint",
        )?,
        Some("stopped") => require(
            first.is_none() && result["first99_epochs"].is_null(),
            "invalid stop",
        )?,
        _ => return Err("unknown status".into()),
    }
    let mut steps = Vec::new();
    for cp in result["snapshots"].as_array().ok_or("missing snapshots")? {
        let step = cp["step"].as_u64().unwrap() as usize;
        let path = root.join(format!("checkpoint-{step:06}.bin"));
        require(
            read(&root.join(format!("checkpoint-{step:06}.json")))? == *cp
                && hash(&fs::read(&path)?) == cp["state_sha256"],
            "snapshot changed",
        )?;
        let history_bytes = cp["history_bytes"].as_u64().unwrap() as usize;
        require(
            history_bytes <= bytes.len() && hash(&bytes[..history_bytes]) == cp["history_sha256"],
            "snapshot history prefix changed",
        )?;
        let (p, m, v) = decode(&path, parent.p.len())?;
        layout::validate_state(&parent.cfg, &p, &m, &v)?;
        if step == 0 {
            require(
                p == parent.initial && m.iter().chain(&v).all(|&x| x == 0.),
                "initial snapshot not fresh",
            )?;
        }
        steps.push(step);
    }
    let mut expected_steps: Vec<_> = (0..completed).step_by(FLUSH).collect();
    expected_steps.push(completed);
    require(steps == expected_steps, "missing retained checkpoint")?;
    let (p, m, v) = decode(
        &root.join(format!("checkpoint-{completed:06}.bin")),
        parent.p.len(),
    )?;
    let mut e = corrected(parent.cfg.clone(), &p, &m, &v)?;
    if plain {
        e.enable_plain_product()?;
    }
    let replay = observe(&mut e, &task, &samples, completed, &Instant::now())?;
    for split in ["train", "validation"] {
        require(
            replay[split]["exact"] == result["endpoint"][split]["exact"]
                && (replay[split]["loss"].as_f64().unwrap()
                    - result["endpoint"][split]["loss"].as_f64().unwrap())
                .abs()
                    < 1e-10,
            "final GPU replay failed",
        )?;
    }
    require(
        e.parameters.read()? == p && e.first_moment.read()? == m && e.second_moment.read()? == v,
        "audit altered state",
    )?;
    let best = history
        .iter()
        .max_by_key(|r| r["validation"]["exact"].as_u64().unwrap())
        .unwrap();
    let report = json!({"passed":true,"case":CASE,"seed":1,"status":result["status"],"updates":completed,"epochs":result["epochs"],"first99_epochs":result["first99_epochs"],
        "heldout_exact":replay["validation"]["exact"],"heldout_examples":4705,"heldout_accuracy":replay["validation"]["sequence_accuracy"],"train_accuracy":replay["train"]["sequence_accuracy"],
        "best_observed_heldout":best["validation"]["sequence_accuracy"],"best_observed_epochs":best["epochs"],"observations":history.len(),"retained_checkpoint_steps":steps,
        "initialization_paired":true,"sampler_paired":true,"original_files_unchanged":true,"full_split_replay":"passed","source_hashes":"passed","complex_constraints":"passed",
        "baseline":{"status":parent.result["status"],"updates":parent.result["completed"],"first99_epochs":parent.result["first99_epochs"],"final_accuracy":parent.result["endpoint"]["validation"]["sequence_accuracy"]},
        "ffn":if plain{"plain_complex_product"}else{"radial_complex_product"},
        "python_invoked":false,"benchmark_replaced":false,"limitation":"one-seed corrected-gradient experiment; no universal convergence claim"});
    require(
        protected(&parent)? == protocol["original_files"],
        "audit changed baseline files",
    )?;
    Ok(report)
}
pub fn audit(parent_root: &Path, root: &Path) -> Result<()> {
    let report = audit_readonly(parent_root, root)?;
    record(&root.join("audit.json"), &report)?;
    println!(
        "{}",
        json!({"audit":"passed","status":report["status"],"updates":report["updates"],"heldout":report["heldout_accuracy"]})
    );
    Ok(())
}

/// Inspect every retained state, without selecting only favorable epochs.
pub fn inspect_epochs(parent_root: &Path, root: &Path, output: &Path) -> Result<()> {
    require(
        read(&root.join("protocol.json"))?["ffn"] != "plain_complex_product",
        "legacy weight probe uses radial FFN; use plain-ffn-audit for plain checkpoints",
    )?;
    require(
        !output.exists(),
        "refuse overwrite of epoch weight inspection",
    )?;
    audit(parent_root, root)?;
    let parent = Checkpoint::load(parent_root, "complex-hermitian", 1)?;
    let protocol = read(&root.join("protocol.json"))?;
    let result = read(&root.join("result.json"))?;
    let audit_before = fs::read(root.join("audit.json"))?;
    let history: Vec<Value> = fs::read_to_string(root.join("history.jsonl"))?
        .lines()
        .map(serde_json::from_str)
        .collect::<std::result::Result<_, _>>()?;
    fs::create_dir(output)?;
    let source = archive(output)?;
    record(
        &output.join("protocol.json"),
        &json!({"purpose":"read-only raw weights, moments and complete-split corrected gradients at every retained epoch",
        "training_protocol":protocol,"source":source,"selection":"all retained snapshots, including initialization and final state",
        "historical_step_limitation":"saved moments evaluated at current weights, including bias correction and decay; not the next minibatch update",
        "training_updates":0,"python_invoked":false}),
    )?;
    let mut previous: Option<Checkpoint> = None;
    let mut rows = Vec::new();
    for snapshot in result["snapshots"].as_array().unwrap() {
        let step = snapshot["step"].as_u64().unwrap() as usize;
        let path = root.join(format!("checkpoint-{step:06}.bin"));
        require(
            hash(&fs::read(&path)?) == snapshot["state_sha256"],
            "input checkpoint hash mismatch",
        )?;
        let (p, m, v) = decode(&path, parent.p.len())?;
        layout::validate_state(&parent.cfg, &p, &m, &v)?;
        let point = history
            .iter()
            .find(|r| r["step"] == step)
            .ok_or("snapshot lacks an observation")?;
        let mut metadata = snapshot.clone();
        metadata["completed"] = step.into();
        let cp = Checkpoint {
            dir: root.to_path_buf(),
            protocol: parent.protocol.clone(),
            result: json!({"endpoint":point,"epochs":point["epochs"]}),
            checkpoint: metadata,
            cfg: parent.cfg.clone(),
            p,
            m,
            v,
            initial: parent.initial.clone(),
            history: vec![point.clone()],
        };
        let mut weights = Vec::new();
        for block in blocks(&cp.cfg) {
            let ids = owner_indices(&cp.cfg, &block);
            let w: Vec<_> = ids.iter().map(|&i| cp.p[i] as f64).collect();
            let m: Vec<_> = ids.iter().map(|&i| cp.m[i] as f64).collect();
            let sv: Vec<_> = ids.iter().map(|&i| (cp.v[i] as f64).sqrt()).collect();
            let expanded: Vec<_> = cp.p[block.start..block.end()]
                .iter()
                .map(|&v| v as f64)
                .collect();
            let last_lr =
                cp.cfg.lr * ((step.saturating_sub(1) as f64 / cp.cfg.warmup as f64).min(1.));
            let historical = if step == 0 {
                Value::Null
            } else {
                let b1 = 1. - 0.9f64.powf(step as f64);
                let b2 = (1. - 0.98f64.powf(step as f64)).sqrt();
                let direction: Vec<_> = m
                    .iter()
                    .zip(&sv)
                    .zip(&w)
                    .map(|((&m, &v), &w)| {
                        last_lr * (m / b1 / (v / b2 + 1e-8) + cp.cfg.decay as f64 * w)
                    })
                    .collect();
                let ww = stats::dot(&w, &w);
                let dd = stats::dot(&direction, &direction);
                let wd = stats::dot(&w, &direction);
                json!({"relative_l2":(dd/ww).sqrt(),"tangent_energy_fraction":if dd>0.{Some((1.-wd*wd/(ww*dd)).clamp(0.,1.))}else{None},"last_update_lr":last_lr})
            };
            weights.push(json!({"block":block.name,"independent_weight":stats::summary(&w),"expanded_matrix":stats::matrix(&expanded,block.rows,block.cols),
                "first_moment":stats::summary(&m),"sqrt_second_moment":stats::summary(&sv),"epsilon_dominated_fraction":sv.iter().filter(|&&x|x<=1e-8).count() as f64/sv.len() as f64,
                "historical_step":historical}));
        }
        let dir = output.join(format!("step-{step:06}"));
        let mut probe = probe::run_with_corrections(&cp, &dir, true)?;
        if step == 0 {
            for block in probe["parameter_gradients"].as_array_mut().unwrap() {
                block["momentum_direction_train_cosine"] = Value::Null;
                block["momentum_direction_heldout_cosine"] = Value::Null;
            }
        }
        let report = json!({"step":step,"epochs":point["epochs"],"state_sha256":snapshot["state_sha256"],"weights":weights,"probe":probe,
            "change_from_previous_retained_epoch":previous.as_ref().map(|old|raw::difference(old,&cp)),"training_updates":0});
        record(&dir.join("report.json"), &report)?;
        require(
            hash(&fs::read(&path)?) == snapshot["state_sha256"],
            "probe changed saved state",
        )?;
        println!(
            "{}",
            json!({"inspected_epoch":point["epochs"],"step":step,"heldout":report["probe"]["heldout"]["accuracy"]})
        );
        rows.push(report);
        previous = Some(cp);
    }
    require(
        protected(&parent)? == protocol["original_files"]
            && fs::read(root.join("audit.json"))? == audit_before,
        "inspection changed input files",
    )?;
    record(
        &output.join("summary.json"),
        &json!({"complete":true,"checkpoints":rows.len(),"all_retained_epochs_included":true,"rows":rows,
        "training_updates":0,"input_checkpoints_unchanged":true,"limitations":"500-epoch retained states cannot localize every one-epoch accuracy crash; correlations are not causal interventions"}),
    )?;
    Ok(())
}
