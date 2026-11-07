//! Native quartic data, full-budget training, infrequent durable saves and strict continuation.
pub use crate::schedule::ARMS;
use crate::{
    Result, arithmetic_data,
    engine::{Config, Engine},
    hash, layout,
    optimizer::{BUDGET, Optimizer},
    random::Random,
    schedule::{Family, Schedule},
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::Write,
    path::Path,
    process::Command,
    time::Instant,
};
const SUITE: &str = include_str!("../../../complex_geometry_suite/suite.json");
const FLUSH: usize = 500 * 19;
pub fn row() -> Value {
    serde_json::from_str::<Value>(SUITE).unwrap()["tasks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|v| v["name"] == "arithmetic-sum-pow4")
        .unwrap()
        .clone()
}
pub fn config() -> Config {
    Config {
        width: 128,
        vocab: 97,
        heads: 4,
        max_batch: 256,
        max_length: 2,
        layers: 1,
        normalized: true,
        ssmax: true,
        gain: 4. / 128.,
        lr: 0.002,
        decay: 0.1,
        warmup: 0,
        architecture: "complex-linear".into(),
        phase_optimizer: false,
    }
}
pub fn order(n: usize, updates: usize, seed: u32) -> (Vec<u32>, Vec<usize>) {
    let mut rng = Random::new(seed + 10000);
    let mut values = Vec::with_capacity(updates * 256);
    let mut edges = vec![0];
    while edges.len() <= updates {
        let mut epoch: Vec<u32> = (0..n as u32).collect();
        rng.shuffle(&mut epoch);
        for chunk in epoch.chunks(256) {
            if edges.len() > updates {
                break;
            }
            values.extend_from_slice(chunk);
            edges.push(values.len());
        }
    }
    (values, edges)
}
pub fn source_hash() -> String {
    let mut bytes = Vec::new();
    for (name, body) in crate::sources::FILES {
        bytes.extend((name.len() as u64).to_le_bytes());
        bytes.extend(name.as_bytes());
        bytes.extend((body.len() as u64).to_le_bytes());
        bytes.extend(body.as_bytes());
    }
    hash(&bytes)
}
pub fn ptx_hash() -> String {
    hash(
        [
            crate::attention::kernels::PTX,
            crate::base_ops::kernels::PTX,
            crate::sequence_ops::kernels::PTX,
            crate::complex::kernels::PTX,
            crate::gemm::kernels::PTX,
            crate::md_ops::kernels::PTX,
            crate::adam_md_ops::kernels::PTX,
        ]
        .concat()
        .as_bytes(),
    )
}
pub fn durable(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = path.with_extension("pending");
    let mut f = File::create(&tmp)?;
    f.write_all(bytes)?;
    f.sync_all()?;
    fs::rename(tmp, path)?;
    File::open(path.parent().ok_or("missing parent")?)?.sync_all()?;
    Ok(())
}
pub fn record(path: &Path, v: &Value) -> Result<()> {
    let mut b = serde_json::to_vec_pretty(v)?;
    b.push(b'\n');
    durable(path, &b)
}
pub fn read(path: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}
pub fn float_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
pub fn archive(root: &Path) -> Result<()> {
    let data = arithmetic_data::arithmetic(&row())?;
    let metadata = json!({"study":"quartic complex-linear AdamW/AdamMD/AdamMDW/MuonMD schedules","source_sha256":source_hash(),"ptx_sha256":ptx_hash(),
        "seeds":[0,1,2],"arms":ARMS,"expected_cells":48,"combined_expected_cells":54,"reused_cells":6,"budget":BUDGET,"checkpoint_epochs":500,
        "baseline_source_sha256":"2744d9d51b4f5436c3b1d6b94bcb4ebb09f9466f426fbd9fb861e0746cca4799",
        "model_commit":"910f206a1e9650c20d47a0c16c94270a692804d5","kernite_commit":"a1b6bc2c956e1a5f35960e26b209e1b6c213096c",
        "train_sha256":arithmetic_data::fingerprint(&data.train),"validation_sha256":arithmetic_data::fingerprint(&data.validation),
        "python_invoked":false});
    let dest = root.join("protocol.json");
    if dest.exists() {
        if read(&dest)? != metadata {
            return Err("immutable cohort identity changed".into());
        }
        return Ok(());
    }
    fs::create_dir_all(root)?;
    for (name, body) in crate::sources::FILES {
        let p = root.join("source").join(name);
        fs::create_dir_all(p.parent().ok_or("source path")?)?;
        durable(&p, body.as_bytes())?;
    }
    record(&dest, &metadata)
}
pub fn checkpoint(
    e: &Engine,
    o: &Optimizer,
    dir: &Path,
    completed: usize,
    examples: usize,
    history: &[Value],
) -> Result<()> {
    let bytes = float_bytes(&o.save(e)?);
    let mut text = Vec::new();
    for h in history {
        text.extend(serde_json::to_vec(h)?);
        text.push(b'\n');
    }
    durable(&dir.join("state.f32"), &bytes)?;
    durable(&dir.join("history.jsonl"), &text)?;
    record(
        &dir.join("checkpoint.json"),
        &json!({"completed":completed,"examples":examples,"state_sha256":hash(&bytes),
        "history_sha256":hash(&text),"source_sha256":source_hash(),"ptx_sha256":ptx_hash(),"arm":o.arm,
        "layout":"parameters,m,v; per matrix: complex direction m,v,raw gains,positive gains,gain m,gain v; f32 LE"}),
    )?;
    record(
        &dir.join("execution.json"),
        &json!({"clean":true,"durable_completed":completed,"possible_completed":completed}),
    )
}
pub fn restore(e: &mut Engine, o: &mut Optimizer, dir: &Path) -> Result<(usize, Vec<Value>)> {
    if read(&dir.join("execution.json"))?["clean"] != true {
        return Err("unclean window: refuse replay".into());
    }
    let cp = read(&dir.join("checkpoint.json"))?;
    let bytes = fs::read(dir.join("state.f32"))?;
    let text = fs::read(dir.join("history.jsonl"))?;
    if cp["source_sha256"] != source_hash()
        || cp["ptx_sha256"] != ptx_hash()
        || cp["arm"] != o.arm
        || cp["state_sha256"] != hash(&bytes)
        || cp["history_sha256"] != hash(&text)
        || !bytes.len().is_multiple_of(4)
    {
        return Err("checkpoint provenance/hash mismatch".into());
    }
    let data: Vec<_> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    o.restore(e, &data)?;
    let history = text
        .split(|b| *b == b'\n')
        .filter(|x| !x.is_empty())
        .map(serde_json::from_slice)
        .collect::<std::result::Result<Vec<Value>, _>>()?;
    let completed = cp["completed"].as_u64().ok_or("missing step")? as usize;
    if history.last().ok_or("empty history")?["step"] != completed {
        return Err("checkpoint/history step mismatch".into());
    }
    Ok((completed, history))
}
pub fn batch(e: &mut Engine, rows: &[[u32; 3]], indices: &[u32]) -> Result<()> {
    let mut tokens = Vec::with_capacity(2 * indices.len());
    let mut labels = Vec::with_capacity(2 * indices.len());
    for &i in indices {
        let [a, b, y] = rows[i as usize];
        tokens.extend([a, b]);
        labels.extend([u32::MAX, y]);
    }
    e.forward(&tokens, &labels, indices.len(), 2)
}
pub fn evaluate(e: &mut Engine, rows: &[[u32; 3]]) -> Result<Value> {
    let mut loss = 0.;
    let mut hits = 0usize;
    for start in (0..rows.len()).step_by(256) {
        let end = (start + 256).min(rows.len());
        let indices: Vec<_> = (start as u32..end as u32).collect();
        batch(e, rows, &indices)?;
        let predictions = e.predictions.read()?;
        let losses = e.losses(indices.len() * 2)?;
        for (local, &i) in indices.iter().enumerate() {
            let l = losses[2 * local + 1] as f64;
            if !l.is_finite() {
                return Err("nonfinite classification loss".into());
            }
            loss += l;
            hits += usize::from(predictions[2 * local + 1] == rows[i as usize][2]);
        }
    }
    Ok(
        json!({"exact":hits,"total":rows.len(),"accuracy":hits as f64/rows.len() as f64,"loss":loss/rows.len() as f64}),
    )
}
fn sampler_hash(order: &[u32], edges: &[usize]) -> String {
    let mut h = Sha256::new();
    for x in order {
        h.update(x.to_le_bytes());
    }
    for x in edges {
        h.update((*x as u64).to_le_bytes());
    }
    h.finalize().iter().map(|x| format!("{x:02x}")).collect()
}
pub fn summary(history: &[Value]) -> Value {
    let final_point = history.last().unwrap();
    let first = history
        .iter()
        .find(|p| p["validation"]["exact"].as_u64().unwrap() * 100 >= 99 * 4705);
    let acc = |p: &Value| p["validation"]["accuracy"].as_f64().unwrap();
    let best = history
        .iter()
        .max_by(|a, b| acc(a).total_cmp(&acc(b)))
        .unwrap();
    let end = final_point["epochs"].as_f64().unwrap();
    let late: Vec<_> = history
        .iter()
        .filter(|p| p["epochs"].as_f64().unwrap() >= end - 1000.)
        .map(acc)
        .collect();
    json!({"endpoint":final_point,"first99":first,"best":best,"late_epochs":1000,
        "late_min_accuracy":late.iter().copied().reduce(f64::min),"late_max_accuracy":late.iter().copied().reduce(f64::max)})
}
fn cell(root: &Path, arm: &str, seed: u32, resume: bool) -> Result<()> {
    if !ARMS.contains(&arm) || seed > 2 {
        return Err("unknown cell".into());
    }
    if root.join("STOP").exists() {
        return Err("STOP prevents new cell".into());
    }
    archive(root)?;
    let dir = root.join(arm).join(format!("seed{seed}"));
    if dir.exists() && !resume {
        return Err("refuse overwrite/replay".into());
    }
    if dir.join("STOP").exists() {
        return Err("cell STOP remains".into());
    }
    let data = arithmetic_data::arithmetic(&row())?;
    let cfg = config();
    assert_eq!(data.vocab, cfg.vocab);
    let (order, edges) = order(data.train.len(), BUDGET, seed);
    let initial = layout::initialize_model(&cfg, seed);
    let initial_bytes = float_bytes(&initial);
    let schedule = Schedule::parse(arm)?;
    let protocol = json!({"arm":arm,"seed":seed,"source_sha256":source_hash(),"ptx_sha256":ptx_hash(),"case":row(),
        "initial_sha256":hash(&initial_bytes),"sampler_sha256":sampler_hash(&order,&edges),"max_updates":BUDGET,
        "model":"complex-linear; real XSA; radial product; tied embedding","width":128,"heads":4,"layers":1,"hidden":342,
        "independent_parameters":layout::independent_count(&cfg),"expanded_parameters":initial.len(),
        "optimizer":schedule.family.name(),"warmup":schedule.warmup,"clip_global_independent_gradient_norm":1,"adam_betas":[0.9,0.99],"adam_epsilon":1e-8,
        "schedule":schedule.name(),"schedule_floor":1e-8,"final_scheduled_update":99999,"cooldown_start":if schedule.name().starts_with("wsd-"){Some(80000)}else{None},
        "embedding_peak":0.002,"matrix_peak":schedule.family.peak(),"gain_peak":if schedule.family==Family::AdamW{None}else{Some(schedule.family.peak())},
        "weight_decay":schedule.family.decay(),"momentum":if schedule.family==Family::MuonMD{Some(0.95)}else{None},"nesterov":schedule.family==Family::MuonMD,"ns_steps":if schedule.family==Family::MuonMD{5}else{0},
        "gains":"softplus row and column; initial1","direction_radius":"actual initial complex Frobenius norm",
        "muon_multiplier":"fixed radius/sqrt(min(complex_rows,complex_columns))","adam_direction_multiplier":1.,
        "mdw_rule":"subtract direction_lr*decay*old_physical_weight; repair row gains without changing weight or moments","checkpoint_epochs":500});
    if resume {
        if read(&dir.join("protocol.json"))? != protocol
            || read(&dir.join("result.json"))?["status"] != "stopped"
        {
            return Err("only identical explicitly stopped cells resume".into());
        }
    } else {
        fs::create_dir_all(&dir)?;
        record(&dir.join("protocol.json"), &protocol)?;
        durable(&dir.join("initial.f32"), &initial_bytes)?;
    }
    let mut e = Engine::new(cfg, &initial)?;
    let mut o = Optimizer::new(&e, &initial, arm)?;
    let (mut completed, mut history) = if resume {
        restore(&mut e, &mut o, &dir)?
    } else {
        (0, Vec::new())
    };
    let resumed_at = completed;
    let carried = history
        .last()
        .map_or(0., |p| p["seconds"].as_f64().unwrap());
    let started = Instant::now();
    if resume {
        fs::rename(
            dir.join("result.json"),
            dir.join(format!("stop-{completed}.json")),
        )?;
    }
    let mut window_end = completed;
    let mut had99 = history
        .iter()
        .any(|p| p["validation"]["exact"].as_u64().unwrap() * 100 >= 99 * 4705);
    loop {
        if !resume || completed != resumed_at {
            let train = evaluate(&mut e, &data.train)?;
            let validation = evaluate(&mut e, &data.validation)?;
            history.push(json!({"step":completed,"examples":edges[completed],"epochs":edges[completed] as f64/4704.,
                "seconds":carried+started.elapsed().as_secs_f64(),"train":train,"validation":validation,
                "next_embedding_lr":schedule.rate(0.002,completed),"next_matrix_lr":schedule.rate(schedule.family.peak(),completed)}));
        }
        let point = history.last().ok_or("no observation")?;
        let passed = point["validation"]["exact"].as_u64().unwrap() * 100 >= 99 * 4705;
        let first99 = passed && !had99;
        had99 |= passed;
        let stop = root.join("STOP").exists() || dir.join("STOP").exists();
        let done = completed == BUDGET || stop;
        if completed >= window_end || first99 || done {
            checkpoint(&e, &o, &dir, completed, edges[completed], &history)?;
            record(&dir.join("status.json"), point)?;
            println!(
                "{}",
                json!({"arm":arm,"seed":seed,"step":completed,"epochs":point["epochs"],"train":point["train"]["accuracy"],
                "validation":point["validation"]["accuracy"],"loss":point["validation"]["loss"],"seconds":point["seconds"]})
            );
            if done {
                let mut result = summary(&history);
                result["status"] = json!(if stop { "stopped" } else { "complete" });
                result["arm"] = json!(arm);
                result["seed"] = json!(seed);
                result["source_sha256"] = json!(source_hash());
                result["ptx_sha256"] = json!(ptx_hash());
                record(&dir.join("result.json"), &result)?;
                return Ok(());
            }
            window_end = ((completed / FLUSH + 1) * FLUSH).min(BUDGET);
            record(
                &dir.join("execution.json"),
                &json!({"clean":false,"durable_completed":completed,"possible_completed":window_end}),
            )?;
        }
        let next = ((completed / 19 + 1) * 19).min(window_end).min(BUDGET);
        while completed < next {
            let indices = &order[edges[completed]..edges[completed + 1]];
            batch(&mut e, &data.train, indices)?;
            e.backward(indices.len(), 2)?;
            o.step(&e, completed)?;
            completed += 1;
        }
    }
}
pub fn audit(root: &Path) -> Result<Value> {
    let mut results = Vec::new();
    let mut failures = Vec::new();
    let mut completed = 0;
    for seed in 0..3 {
        for arm in ARMS {
            let dir = root.join(arm).join(format!("seed{seed}"));
            if !dir.join("result.json").exists() {
                results.push(json!({"arm":arm,"seed":seed,"status":"pending_or_running"}));
                continue;
            }
            let result = read(&dir.join("result.json"))?;
            if result["status"] == "execution_error" {
                failures.push(json!({"arm":arm,"seed":seed,"error":result["error"]}));
                results.push(result);
                continue;
            }
            let check = (|| -> Result<()> {
                let cp = read(&dir.join("checkpoint.json"))?;
                let p = read(&dir.join("protocol.json"))?;
                if cp["source_sha256"] != source_hash()
                    || cp["ptx_sha256"] != ptx_hash()
                    || p["source_sha256"] != source_hash()
                    || cp["state_sha256"] != hash(&fs::read(dir.join("state.f32"))?)
                    || cp["history_sha256"] != hash(&fs::read(dir.join("history.jsonl"))?)
                    || p["initial_sha256"] != hash(&fs::read(dir.join("initial.f32"))?)
                    || read(&dir.join("execution.json"))?["clean"] != true
                {
                    return Err("checkpoint/source integrity failure".into());
                }
                let history: Vec<Value> = fs::read_to_string(dir.join("history.jsonl"))?
                    .lines()
                    .map(serde_json::from_str)
                    .collect::<std::result::Result<_, _>>()?;
                if history.len() > BUDGET.div_ceil(19) + 1 {
                    return Err("duplicate observations after budget end".into());
                }
                if history.first().ok_or("empty history")?["step"] != 0 {
                    return Err("summary/history mismatch".into());
                }
                let recomputed = summary(&history);
                for (key, value) in recomputed.as_object().unwrap() {
                    if result[key] != *value {
                        return Err(format!("summary mismatch: {key}").into());
                    }
                }
                let expected_initial = layout::initialize_model(&config(), seed);
                if p["initial_sha256"] != hash(&float_bytes(&expected_initial))
                    || p["seed"] != seed
                    || p["arm"] != arm
                {
                    return Err("initialization/identity mismatch".into());
                }
                for (i, point) in history.iter().enumerate() {
                    let step = (i * 19).min(BUDGET);
                    let examples = (step / 19) * 4704 + (step % 19) * 256;
                    if point["step"] != json!(step)
                        || point["examples"] != examples
                        || !point["epochs"]
                            .as_f64()
                            .is_some_and(|x| (x - examples as f64 / 4704.).abs() < 1e-10)
                    {
                        return Err("missing or repeated observation".into());
                    }
                    for (split, total) in [("train", 4704), ("validation", 4705)] {
                        let exact = point[split]["exact"].as_u64().ok_or("missing hit count")?;
                        if exact > total
                            || point[split]["total"] != total
                            || !point[split]["accuracy"]
                                .as_f64()
                                .is_some_and(|x| (x - exact as f64 / total as f64).abs() < 1e-14)
                            || !point[split]["loss"]
                                .as_f64()
                                .is_some_and(|x| x.is_finite() && x >= -1e-5)
                        {
                            return Err("invalid classification metrics".into());
                        }
                    }
                }
                let last = history.last().unwrap();
                if cp["completed"] != last["step"] || cp["examples"] != last["examples"] {
                    return Err("checkpoint endpoint mismatch".into());
                }
                let bytes = fs::read(dir.join("state.f32"))?;
                if !bytes.len().is_multiple_of(4) {
                    return Err("partial float in state".into());
                }
                let values: Vec<_> = bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b))
                    .collect();
                let n = expected_initial.len();
                let extra = if Schedule::parse(arm)?.family != Family::AdamW {
                    [(192, 64), (64, 64), (342, 64), (64, 171)]
                        .iter()
                        .map(|&(r, c)| 4 * r * c + 4 * (r + c))
                        .sum()
                } else {
                    0
                };
                if values.len() != 3 * n + extra || !values.iter().all(|v| v.is_finite()) {
                    return Err("invalid optimizer state size/values".into());
                }
                layout::validate_state(
                    &config(),
                    &values[..n],
                    &values[n..2 * n],
                    &values[2 * n..3 * n],
                )?;
                if Schedule::parse(arm)?.family != Family::AdamW {
                    let mut at = 3 * n;
                    for (r, c) in [(192, 64), (64, 64), (342, 64), (64, 171)] {
                        at += 2 * r * c;
                        if values[at..at + 2 * r * c].iter().any(|x| *x < 0.) {
                            return Err("negative direction variance".into());
                        }
                        at += 2 * r * c + r + c;
                        if values[at..at + r + c].iter().any(|x| *x <= 0.) {
                            return Err("nonpositive gain".into());
                        }
                        at += 2 * (r + c);
                        if values[at..at + r + c].iter().any(|x| *x < 0.) {
                            return Err("negative gain variance".into());
                        }
                        at += r + c;
                    }
                }
                if result["status"] == "complete" && cp["completed"] != BUDGET {
                    return Err("truncated full-budget run".into());
                }
                Ok(())
            })();
            if let Err(e) = check {
                failures.push(json!({"arm":arm,"seed":seed,"error":e.to_string()}));
            }
            if result["status"] == "complete" {
                completed += 1;
            }
            results.push(result);
        }
    }
    for seed in 0..3 {
        for arm in &ARMS[1..] {
            let a = root.join(ARMS[0]).join(format!("seed{seed}/protocol.json"));
            let b = root.join(arm).join(format!("seed{seed}/protocol.json"));
            if a.exists() && b.exists() {
                let (a, b) = (read(&a)?, read(&b)?);
                for key in ["initial_sha256", "sampler_sha256"] {
                    if a[key] != b[key] {
                        failures.push(json!({"seed":seed,"error":format!("pair mismatch {key}")}));
                    }
                }
            }
        }
    }
    let result = json!({"expected":48,"completed":completed,"integrity_failures":failures,"results":results,"source_sha256":source_hash()});
    record(&root.join("audit.json"), &result)?;
    Ok(result)
}
fn smoke(root: &Path) -> Result<()> {
    if root.exists() {
        return Err("smoke output exists".into());
    }
    archive(root)?;
    let data = arithmetic_data::arithmetic(&row())?;
    for arm in ARMS {
        let initial = layout::initialize_model(&config(), 0);
        let mut e = Engine::new(config(), &initial)?;
        let mut o = Optimizer::new(&e, &initial, arm)?;
        let before = evaluate(&mut e, &data.train)?;
        let (order, edges) = order(4704, 19, 0);
        let started = Instant::now();
        for step in 0..19 {
            let indices = &order[edges[step]..edges[step + 1]];
            batch(&mut e, &data.train, indices)?;
            e.backward(indices.len(), 2)?;
            o.step(&e, step)?;
        }
        e.cuda.sync()?;
        let elapsed = started.elapsed().as_secs_f64();
        let after = evaluate(&mut e, &data.train)?;
        o.save(&e)?;
        let result = json!({"arm":arm,"updates":19,"seconds":elapsed,"before":before,"after":after,"scientific_result":false});
        record(&root.join(format!("{arm}.json")), &result)?;
        println!("{result}");
    }
    Ok(())
}

pub fn combined(root: &Path, baseline: &Path) -> Result<Value> {
    const BASE_SOURCE: &str = "2744d9d51b4f5436c3b1d6b94bcb4ebb09f9466f426fbd9fb861e0746cca4799";
    let old = read(&baseline.join("audit.json"))?;
    if old["expected"] != 6
        || old["completed"] != 6
        || old["source_sha256"] != BASE_SOURCE
        || !old["integrity_failures"]
            .as_array()
            .is_some_and(|x| x.is_empty())
    {
        return Err("baseline cohort is incomplete or has the wrong frozen identity".into());
    }
    let current = audit(root)?;
    let mut failures = current["integrity_failures"].as_array().unwrap().clone();
    let mut results = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    for (cohort, base, reused) in [(&old, baseline, true), (&current, root, false)] {
        for entry in cohort["results"].as_array().ok_or("missing results")? {
            let mut r = entry.clone();
            let arm = r["arm"].as_str().ok_or("missing arm")?.to_owned();
            let seed = r["seed"].as_u64().ok_or("missing seed")?;
            if seed > 2 || !seen.insert((arm.clone(), seed)) {
                return Err("duplicate/invalid combined cell".into());
            }
            let s = Schedule::parse(&arm)?;
            let dir = base.join(&arm).join(format!("seed{seed}"));
            let protocol = dir.join("protocol.json");
            let pair = baseline
                .join("adamw-linear")
                .join(format!("seed{seed}/protocol.json"));
            if protocol.exists() {
                let (p, q) = (read(&protocol)?, read(&pair)?);
                for key in ["initial_sha256", "sampler_sha256", "case"] {
                    if p[key] != q[key] {
                        failures.push(json!({"arm":arm,"seed":seed,"error":format!("baseline pairing mismatch: {key}")}));
                    }
                }
                if reused && p["source_sha256"] != BASE_SOURCE {
                    return Err("baseline cell source changed".into());
                }
                r["cell_source_sha256"] = p["source_sha256"].clone();
            }
            r["reused_baseline"] = json!(reused);
            r["optimizer"] = json!(s.family.name());
            r["schedule"] = json!(s.name());
            r["warmup"] = json!(s.warmup);
            results.push(r);
        }
    }
    if results.len() != 54 {
        return Err("combined cohort must contain54 declared cells".into());
    }
    let completed = results.iter().filter(|r| r["status"] == "complete").count();
    let result = json!({"expected":54,"completed":completed,"integrity_failures":failures,"results":results,
        "baseline_source_sha256":BASE_SOURCE,"additional_source_sha256":source_hash(),
        "interpretation":"fixed-peak schedule and warmup comparison; no best-tuned optimizer claim"});
    record(&root.join("combined-audit.json"), &result)?;
    Ok(result)
}
pub fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let command = args.get(1).ok_or(
        "commands: archive,smoke,suite,cell,resume,audit ROOT; combined ROOT BASELINE_ROOT",
    )?;
    let root = Path::new(args.get(2).ok_or("missing ROOT")?);
    match command.as_str() {
        "archive" => archive(root),
        "smoke" => smoke(root),
        "audit" => {
            println!("{}", audit(root)?);
            Ok(())
        }
        "combined" => {
            println!(
                "{}",
                combined(root, Path::new(args.get(3).ok_or("missing baseline root")?))?
            );
            Ok(())
        }
        "cell" | "resume" => cell(
            root,
            args.get(3).ok_or("missing arm")?,
            args.get(4).ok_or("missing seed")?.parse()?,
            command == "resume",
        ),
        "suite" => {
            archive(root)?;
            for seed in 0..3 {
                for j in 0..ARMS.len() {
                    let arm = ARMS[(j + seed as usize) % ARMS.len()];
                    let dir = root.join(arm).join(format!("seed{seed}"));
                    if root.join("STOP").exists() {
                        return Ok(());
                    }
                    if dir.exists() {
                        return Err(format!(
                            "existing cell {}; use explicit resume if stopped",
                            dir.display()
                        )
                        .into());
                    }
                    let status = Command::new(std::env::current_exe()?)
                        .args(["cell"])
                        .arg(root)
                        .arg(arm)
                        .arg(seed.to_string())
                        .status()?;
                    if !status.success() {
                        fs::create_dir_all(&dir)?;
                        record(
                            &dir.join("result.json"),
                            &json!({"arm":arm,"seed":seed,
                    "status":"execution_error","error":format!("child exited {status}; see cohort.log; no replay")}),
                        )?;
                    }
                    audit(root)?;
                }
            }
            let result = audit(root)?;
            println!("{result}");
            if result["completed"] != 48
                || !result["integrity_failures"].as_array().unwrap().is_empty()
            {
                return Err("cohort incomplete or failed".into());
            }
            Ok(())
        }
        _ => Err("unknown command".into()),
    }
}
