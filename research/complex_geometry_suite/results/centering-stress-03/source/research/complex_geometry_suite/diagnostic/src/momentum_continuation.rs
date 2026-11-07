//! Fixed-window, one-time first-moment reset; unchanged architecture and samples.
use crate::continuation::{Stream, durable, record};
use crate::*;
use std::time::Instant;

const START: usize = 76000;
const END: usize = 85500;
const OBSERVE: usize = 19;
const PLAN: &str = include_str!("../MOMENTUM_CONTINUATION_PLAN.txt");
const INITIAL_SHA: &str = "ec4acff217e8b267a4947aa45a2bd409455ec9bb2c67b5264de8bdb4e7cda3d5";

#[derive(Default)]
struct Counts {
    n: usize,
    exact: usize,
    loss: f64,
}
impl Counts {
    fn add(&mut self, hit: bool, loss: f64) {
        self.n += 1;
        self.exact += usize::from(hit);
        self.loss += loss;
    }
    fn value(&self) -> Value {
        json!({"examples":self.n,"exact":self.exact,"loss":self.loss/self.n as f64,
            "sequence_accuracy":self.exact as f64/self.n as f64})
    }
}
pub(crate) fn evaluate(e: &mut Engine, examples: &[data::Example]) -> Result<Value> {
    let mut all = Counts::default();
    let mut diagonal = Counts::default();
    let mut non_diagonal = Counts::default();
    let mut packed = metrics::Packed::default();
    for start in (0..examples.len()).step_by(e.config.max_batch) {
        let stop = (start + e.config.max_batch).min(examples.len());
        let ids: Vec<_> = (start as u32..stop as u32).collect();
        packed.fill(examples, &ids);
        require(packed.length == 2, "not the two-token arithmetic task")?;
        e.forward(&packed.tokens, &packed.labels, ids.len(), 2)?;
        let predictions = e.predictions.read()?;
        let losses = e.losses(2 * ids.len())?;
        for (local, example) in examples[start..stop].iter().enumerate() {
            require(
                example.targets.len() == 2 && example.targets[0] < 0 && example.targets[1] >= 0,
                "unexpected arithmetic labels",
            )?;
            let at = 2 * local + 1;
            let hit = predictions[at] == packed.labels[at];
            let loss = losses[at] as f64;
            require(loss.is_finite() && loss >= 0., "invalid loss")?;
            all.add(hit, loss);
            if example.tokens[0] == example.tokens[1] {
                diagonal.add(hit, loss);
            } else {
                non_diagonal.add(hit, loss);
            }
        }
    }
    require(
        diagonal.n > 0 && non_diagonal.n > 0,
        "missing arithmetic subgroup",
    )?;
    let mut result = all.value();
    result["diagonal"] = diagonal.value();
    result["non_diagonal"] = non_diagonal.value();
    Ok(result)
}
fn observe(
    e: &mut Engine,
    task: &data::Task,
    samples: &Stream,
    step: usize,
    started: &Instant,
) -> Result<Value> {
    Ok(
        json!({"step":step,"additional_updates":step-START,"examples":samples.edges[step],
        "additional_examples":samples.edges[step]-samples.edges[START],"epochs":samples.edges[step] as f64/4704.,
        "additional_epochs":(samples.edges[step]-samples.edges[START]) as f64/4704.,"seconds":started.elapsed().as_secs_f64(),
        "train":evaluate(e,&task.splits["train"])?,"validation":evaluate(e,&task.splits["validation"])?}),
    )
}
fn same_score(a: &Value, b: &Value) -> Result<()> {
    require(
        a["examples"] == b["examples"]
            && a["exact"] == b["exact"]
            && (a["loss"].as_f64().ok_or("no loss")? - b["loss"].as_f64().ok_or("no loss")?).abs()
                <= 1e-10,
        "score replay differs",
    )
}
fn same_point(a: &Value, b: &Value, groups: bool) -> Result<()> {
    for split in ["train", "validation"] {
        same_score(&a[split], &b[split])?;
        if groups {
            for group in ["diagonal", "non_diagonal"] {
                same_score(&a[split][group], &b[split][group])?;
            }
        }
    }
    Ok(())
}
fn state_bytes(e: &Engine) -> Result<Vec<u8>> {
    let (p, m, v) = (
        e.parameters.read()?,
        e.first_moment.read()?,
        e.second_moment.read()?,
    );
    layout::validate_state(&e.config, &p, &m, &v)?;
    Ok(p.iter()
        .chain(&m)
        .chain(&v)
        .flat_map(|x| x.to_le_bytes())
        .collect())
}
fn checkpoint(
    e: &Engine,
    out: &Path,
    point: &Value,
    history: &[u8],
    source: &Value,
) -> Result<Value> {
    let state = state_bytes(e)?;
    let step = point["step"].as_u64().unwrap();
    let name = format!("checkpoint-{step:06}.bin");
    let meta = json!({"step":step,"state_file":name,"state_sha256":hash(&state),"history_bytes":history.len(),
        "history_sha256":hash(history),"source_sha256":source["sha256"],"layout":"parameters, m, v; f32 little endian"});
    durable(&out.join(name), &state)?;
    durable(&out.join("history.jsonl"), history)?;
    record(&out.join(format!("checkpoint-{step:06}.json")), &meta)?;
    record(
        &out.join("execution.json"),
        &json!({"clean":true,"durable_completed":step,"possible_completed":step}),
    )?;
    Ok(meta)
}
fn history(root: &Path) -> Result<Vec<Value>> {
    fs::read(root.join("history.jsonl"))?
        .split(|&b| b == b'\n')
        .filter(|x| !x.is_empty())
        .map(|x| Ok(serde_json::from_slice(x)?))
        .collect()
}
fn brief(p: &Value) -> Value {
    json!({"step":p["step"],"epochs":p["epochs"],"additional_epochs":p["additional_epochs"],"train":p["train"],"validation":p["validation"]})
}
fn summarize(points: &[Value]) -> Value {
    let first = points
        .iter()
        .position(|p| p["validation"]["exact"].as_u64().unwrap() * 100 >= 99 * 4705);
    let tail = &points[points.len().saturating_sub(200)..];
    let values: Vec<_> = tail
        .iter()
        .map(|p| p["validation"]["sequence_accuracy"].as_f64().unwrap())
        .collect();
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let stddev =
        (values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64).sqrt();
    let peak = points.iter().fold(&points[0], |best, p| {
        if p["validation"]["exact"].as_u64() > best["validation"]["exact"].as_u64() {
            p
        } else {
            best
        }
    });
    let max_drop = points
        .windows(2)
        .map(|w| {
            w[0]["validation"]["sequence_accuracy"].as_f64().unwrap()
                - w[1]["validation"]["sequence_accuracy"].as_f64().unwrap()
        })
        .fold(0., f64::max);
    json!({"initial":brief(&points[0]),"final":brief(points.last().unwrap()),"peak":brief(peak),"first99":first.map(|i|brief(&points[i])),
        "previous_first99_observation":first.and_then(|i|i.checked_sub(1)).map(|i|brief(&points[i])),
        "observations":points.len(),"last200_observations":tail.len(),"last200_mean":mean,"last200_stddev":stddev,
        "last200_min":values.iter().copied().fold(f64::INFINITY,f64::min),"last200_max":values.iter().copied().fold(f64::NEG_INFINITY,f64::max),
        "last200_fraction_above99":tail.iter().filter(|p|p["validation"]["exact"].as_u64().unwrap()*100>=99*4705).count() as f64/tail.len() as f64,
        "largest_one_epoch_drop":max_drop,"last200_train_loss_mean":tail.iter().map(|p|p["train"]["loss"].as_f64().unwrap()).sum::<f64>()/tail.len() as f64})
}
struct Inputs {
    parent: Checkpoint,
    task: data::Task,
    samples: Stream,
    baseline: BTreeMap<usize, Value>,
    original_files: Value,
    input_files: Value,
    one_step_files: Value,
    input_audit: Value,
}
fn inputs(parent_root: &Path, corrected: &Path, one_step: &Path) -> Result<Inputs> {
    let input_files = next_update::files(corrected)?;
    let one_step_files = next_update::files(one_step)?;
    let input_audit = retrain::audit_readonly(parent_root, corrected)?;
    let (parent, task, samples) = retrain::setup(parent_root)?;
    let original_files = retrain::protected(&parent)?;
    require(
        hash(&fs::read(
            corrected.join(format!("checkpoint-{START:06}.bin")),
        )?) == INITIAL_SHA,
        "wrong initial checkpoint",
    )?;
    require(
        samples.edges[START] == 4000 * 4704 && samples.edges[END] == 4500 * 4704,
        "wrong continuation exposure",
    )?;
    let baseline = history(corrected)?
        .into_iter()
        .filter_map(|p| {
            let step = p["step"].as_u64()? as usize;
            (START..=END).contains(&step).then_some((step, p))
        })
        .collect::<BTreeMap<_, _>>();
    require(
        baseline.len() == 501,
        "missing original comparison interval",
    )?;
    let previous = read(&one_step.join("summary.json"))?;
    require(
        previous["passed"] == true && previous["global_completed"] == START,
        "wrong prior one-step diagnosis",
    )?;
    Ok(Inputs {
        parent,
        task,
        samples,
        baseline,
        original_files,
        input_files,
        one_step_files,
        input_audit,
    })
}
fn window_hash(samples: &Stream) -> String {
    hash(
        &samples.order[samples.edges[START]..samples.edges[END]]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}
fn unchanged(i: &Inputs, corrected: &Path, one_step: &Path) -> Result<()> {
    require(
        retrain::protected(&i.parent)? == i.original_files
            && next_update::files(corrected)? == i.input_files
            && next_update::files(one_step)? == i.one_step_files,
        "protected input artifact changed",
    )
}
fn run_arm(
    i: &Inputs,
    corrected: &Path,
    one_step: &Path,
    output: &Path,
    protocol: &Value,
    reset: bool,
) -> Result<Value> {
    let arm = if reset { "reset_first" } else { "keep" };
    let out = output.join(arm);
    fs::create_dir(&out)?;
    let (p, mut m, v) = retrain::decode(
        &corrected.join(format!("checkpoint-{START:06}.bin")),
        i.parent.p.len(),
    )?;
    if reset {
        m.fill(0.);
    }
    let mut e = retrain::corrected(i.parent.cfg.clone(), &p, &m, &v)?;
    let expected: Vec<_> = p
        .iter()
        .chain(&m)
        .chain(&v)
        .flat_map(|x| x.to_le_bytes())
        .collect();
    require(
        state_bytes(&e)? == expected,
        "initial state not restored exactly",
    )?;
    let first_name = if reset { "reset_first_lr1" } else { "keep_lr1" };
    let first_reference = read(&one_step.join(format!("{first_name}.json")))?;
    let first_bytes = fs::read(one_step.join(format!("{first_name}.bin")))?;
    require(
        hash(&first_bytes) == first_reference["state_sha256"],
        "prior one-step state changed",
    )?;
    let started = Instant::now();
    let mut points = Vec::new();
    let mut history_bytes = Vec::new();
    let mut snapshots = Vec::new();
    let mut packed = metrics::Packed::default();
    let mut step = START;
    let mut consumed = Sha256::new();
    let mut first_update_matched = false;
    loop {
        let point = observe(&mut e, &i.task, &i.samples, step, &started)?;
        if !reset || step == START {
            same_point(&point, &i.baseline[&step], false)?;
        }
        history_bytes.extend(serde_json::to_vec(&point)?);
        history_bytes.push(b'\n');
        points.push(point.clone());
        let stopped = output.join("STOP").exists();
        let done = step == END || stopped;
        if step == START || done {
            snapshots.push(checkpoint(
                &e,
                &out,
                &point,
                &history_bytes,
                &protocol["source"],
            )?);
            println!(
                "{}",
                json!({"arm":arm,"epoch":point["epochs"],"heldout":point["validation"]["sequence_accuracy"],
                "diagonal_correct":point["validation"]["diagonal"]["exact"],"seconds":started.elapsed().as_secs_f64()})
            );
            if !done {
                record(
                    &out.join("execution.json"),
                    &json!({"clean":false,"durable_completed":step,"possible_completed":END}),
                )?;
            }
        }
        if done {
            let consumed_hash: String = consumed
                .finalize()
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect();
            let paired_end = !reset && step == END;
            if step == END {
                require(
                    consumed_hash == window_hash(&i.samples),
                    "actual sample stream differs",
                )?;
            }
            if paired_end {
                require(
                    state_bytes(&e)?
                        == fs::read(corrected.join(format!("checkpoint-{END:06}.bin")))?,
                    "control endpoint differs in parameters or moments",
                )?;
            }
            let result = json!({"arm":arm,"status":if step==END{"complete"}else{"stopped"},"completed":step,"additional_updates":step-START,
                "first_moment_reset_count":usize::from(reset),"first_update_matches_previous_artifact":first_update_matched,
                "actual_sampler_sha256":consumed_hash,"snapshots":snapshots,"summary":summarize(&points),
                "control_original_observations_matched":if reset{0}else{points.len()},"control_endpoint_bitwise_identical":paired_end});
            record(&out.join("result.json"), &result)?;
            return Ok(result);
        }
        while step < points.last().unwrap()["step"].as_u64().unwrap() as usize + OBSERVE {
            let ids = &i.samples.order[i.samples.edges[step]..i.samples.edges[step + 1]];
            packed.fill(&i.task.splits["train"], ids);
            e.forward(&packed.tokens, &packed.labels, ids.len(), packed.length)?;
            e.backward(ids.len(), packed.length)?;
            e.update(step)?;
            for id in ids {
                consumed.update(id.to_le_bytes());
            }
            if step == START {
                require(
                    state_bytes(&e)? == first_bytes,
                    "first update does not reproduce the prior one-step intervention",
                )?;
                first_update_matched = true;
            }
            step += 1;
        }
    }
}
pub fn run(parent: &Path, corrected: &Path, one_step: &Path, output: &Path) -> Result<()> {
    require(
        !output.exists() && !output.starts_with(corrected) && !output.starts_with(one_step),
        "choose a fresh output outside inputs",
    )?;
    let i = inputs(parent, corrected, one_step)?;
    fs::create_dir(output)?;
    let source = retrain::archive(output)?;
    let protocol = json!({"purpose":"paired 500-epoch single first-moment reset diagnostic","source":source,"plan_sha256":hash(PLAN.as_bytes()),
        "input_audit":i.input_audit,"original_files":i.original_files,"input_files":i.input_files,"one_step_files":i.one_step_files,
        "start":START,"end":END,"updates_per_arm":END-START,"total_updates":2*(END-START),"observe":OBSERVE,
        "arms":["keep","reset_first"],"lr":i.parent.cfg.lr,"first_moment_reset":"once, retain second moment and global Adam clock",
        "initial_state_sha256":INITIAL_SHA,"full_sampler_sha256":i.samples.future_hash,"window_sampler_sha256":window_hash(&i.samples),
        "input_protocol":read(&corrected.join("protocol.json"))?,"python_invoked":false,"benchmark_modified":false});
    record(&output.join("protocol.json"), &protocol)?;
    durable(&output.join("PLAN.txt"), PLAN.as_bytes())?;
    for reset in [false, true] {
        let result = run_arm(&i, corrected, one_step, output, &protocol, reset)?;
        unchanged(&i, corrected, one_step)?;
        if result["status"] == "stopped" {
            return Ok(());
        }
    }
    drop(i);
    audit(parent, corrected, one_step, output)
}
fn valid_counts(score: &Value) -> Result<()> {
    let n = score["examples"].as_u64().ok_or("missing count")?;
    let exact = score["exact"].as_u64().ok_or("missing correct count")?;
    require(
        n > 0
            && exact <= n
            && score["loss"]
                .as_f64()
                .is_some_and(|v| v.is_finite() && v >= 0.)
            && (score["sequence_accuracy"]
                .as_f64()
                .ok_or("missing accuracy")?
                - exact as f64 / n as f64)
                .abs()
                < 1e-14,
        "invalid score",
    )
}
pub fn audit(parent: &Path, corrected: &Path, one_step: &Path, output: &Path) -> Result<()> {
    let i = inputs(parent, corrected, one_step)?;
    let protocol = read(&output.join("protocol.json"))?;
    require(
        protocol["input_files"] == i.input_files
            && protocol["one_step_files"] == i.one_step_files
            && protocol["original_files"] == i.original_files
            && protocol["window_sampler_sha256"] == window_hash(&i.samples)
            && protocol["initial_state_sha256"] == INITIAL_SHA,
        "input provenance differs",
    )?;
    let mut source_bytes = Vec::new();
    for row in protocol["source"]["files"]
        .as_array()
        .ok_or("missing source manifest")?
    {
        let name = row["path"].as_str().unwrap();
        let bytes = fs::read(
            output
                .join("source/research/complex_geometry_suite")
                .join(name),
        )?;
        require(hash(&bytes) == row["sha256"], "archived source changed")?;
        source_bytes.extend((name.len() as u64).to_le_bytes());
        source_bytes.extend(name.as_bytes());
        source_bytes.extend((bytes.len() as u64).to_le_bytes());
        source_bytes.extend(bytes);
    }
    require(
        hash(&source_bytes) == protocol["source"]["sha256"]
            && hash(&fs::read(output.join("PLAN.txt"))?) == protocol["plan_sha256"],
        "source/plan changed",
    )?;
    let original_state = fs::read(corrected.join(format!("checkpoint-{START:06}.bin")))?;
    let mut results = Vec::new();
    for arm in ["keep", "reset_first"] {
        let out = output.join(arm);
        let result = read(&out.join("result.json"))?;
        let execution = read(&out.join("execution.json"))?;
        require(
            result["status"] == "complete"
                && result["completed"] == END
                && result["additional_updates"] == END - START
                && execution["clean"] == true
                && execution["durable_completed"] == END
                && execution["possible_completed"] == END
                && result["actual_sampler_sha256"] == protocol["window_sampler_sha256"]
                && result["first_update_matches_previous_artifact"] == true,
            "incomplete or unpaired branch",
        )?;
        let points = history(&out)?;
        require(points.len() == 501, "missing observation")?;
        for (at, p) in points.iter().enumerate() {
            let step = START + at * OBSERVE;
            require(
                p["step"] == step
                    && p["additional_updates"] == step - START
                    && p["examples"] == i.samples.edges[step]
                    && p["additional_examples"] == i.samples.edges[step] - i.samples.edges[START]
                    && p["epochs"].as_f64() == Some((4000 + at) as f64)
                    && p["additional_epochs"].as_f64() == Some(at as f64),
                "wrong exposure",
            )?;
            for (split, n, diag) in [("train", 4704, 55), ("validation", 4705, 42)] {
                let s = &p[split];
                valid_counts(s)?;
                valid_counts(&s["diagonal"])?;
                valid_counts(&s["non_diagonal"])?;
                require(
                    s["examples"] == n
                        && s["diagonal"]["examples"] == diag
                        && s["non_diagonal"]["examples"] == n - diag
                        && s["exact"].as_u64().unwrap()
                            == s["diagonal"]["exact"].as_u64().unwrap()
                                + s["non_diagonal"]["exact"].as_u64().unwrap()
                        && (s["loss"].as_f64().unwrap() * n as f64
                            - s["diagonal"]["loss"].as_f64().unwrap() * diag as f64
                            - s["non_diagonal"]["loss"].as_f64().unwrap() * (n - diag) as f64)
                            .abs()
                            < 1e-8,
                    "subgroup accounting failed",
                )?;
            }
            if arm == "keep" {
                same_point(p, &i.baseline[&step], false)?;
            }
        }
        let mut recomputed = summarize(&points);
        // JSON roundtrips may change the final bit of computed floating-point summaries.
        continuation::compare_summary("summary", &result["summary"], &recomputed, &mut Vec::new())?;
        let bytes = fs::read(out.join("history.jsonl"))?;
        let snapshots = result["snapshots"].as_array().ok_or("missing snapshots")?;
        require(snapshots.len() == 2, "unexpected checkpoint cadence")?;
        for (meta, step) in snapshots.iter().zip([START, END]) {
            let cp = out.join(format!("checkpoint-{step:06}.bin"));
            let state = fs::read(&cp)?;
            let prefix = meta["history_bytes"].as_u64().unwrap() as usize;
            require(
                meta["step"] == step
                    && meta["source_sha256"] == protocol["source"]["sha256"]
                    && read(&out.join(format!("checkpoint-{step:06}.json")))? == *meta
                    && hash(&state) == meta["state_sha256"]
                    && prefix <= bytes.len()
                    && hash(&bytes[..prefix]) == meta["history_sha256"],
                "snapshot integrity failed",
            )?;
            let (p, m, v) = retrain::decode(&cp, i.parent.p.len())?;
            layout::validate_state(&i.parent.cfg, &p, &m, &v)?;
            if step == START {
                let chunk = 4 * i.parent.p.len();
                require(
                    state[..chunk] == original_state[..chunk]
                        && state[2 * chunk..] == original_state[2 * chunk..]
                        && if arm == "keep" {
                            state[chunk..2 * chunk] == original_state[chunk..2 * chunk]
                        } else {
                            state[chunk..2 * chunk].iter().all(|&b| b == 0)
                        },
                    "initial moment intervention differs",
                )?;
            }
            let mut e = retrain::corrected(i.parent.cfg.clone(), &p, &m, &v)?;
            let replay = observe(&mut e, &i.task, &i.samples, step, &Instant::now())?;
            same_point(&replay, &points[(step - START) / OBSERVE], true)?;
            require(state_bytes(&e)? == state, "score replay mutated state")?;
            if arm == "keep" && step == END {
                require(
                    state == fs::read(corrected.join(format!("checkpoint-{END:06}.bin")))?,
                    "control endpoint not bitwise identical",
                )?;
            }
        }
        recomputed["arm"] = arm.into();
        recomputed["passed"] = true.into();
        results.push(recomputed);
    }
    unchanged(&i, corrected, one_step)?;
    let report = json!({"passed":true,"complete":true,"case":CASE,"seed":1,"start_epoch":4000,"end_epoch":4500,
        "updates_per_arm":END-START,"actual_training_updates":2*(END-START),"control_original_observations_matched":501,
        "control_endpoint_bitwise_identical":true,"first_updates_match_prior_diagnostic":true,"source_and_checkpoint_hashes":true,
        "actual_sampler_hashes_matched":true,"subgroup_accounting":true,"input_artifacts_unchanged":true,"endpoint_reloads_passed":true,
        "arms":results,"limits":"one selected checkpoint and seed; one-time m reset only; no universal optimizer or architecture conclusion","benchmark_modified":false,"python_invoked":false});
    record(&output.join("audit.json"), &report)?;
    println!(
        "{}",
        json!({"audit":"passed","updates":2*(END-START),"control_replay":"bitwise identical"})
    );
    Ok(())
}

#[test]
fn subgroup_weighting_and_first99_are_count_based() {
    let mut total = Counts::default();
    let mut diagonal = Counts::default();
    let mut other = Counts::default();
    for (is_diag, hit, loss) in [
        (true, false, 2.),
        (false, true, 0.1),
        (false, true, 0.2),
        (false, true, 0.3),
    ] {
        total.add(hit, loss);
        if is_diag {
            diagonal.add(hit, loss);
        } else {
            other.add(hit, loss);
        }
    }
    assert_eq!(total.exact, diagonal.exact + other.exact);
    assert!((total.loss / total.n as f64 - 0.65).abs() < 1e-12);
    let points:Vec<_>=[4657,4658,4656].iter().enumerate().map(|(j,&exact)|json!({"step":START+j*OBSERVE,"epochs":4000+j,
        "train":{"loss":0.},"validation":{"examples":4705,"exact":exact,"sequence_accuracy":exact as f64/4705.}})).collect();
    let s = summarize(&points);
    assert_eq!(s["first99"]["step"], START + OBSERVE);
    assert_eq!(s["previous_first99_observation"]["step"], START);
    assert_eq!(s["final"]["validation"]["exact"], 4656);
    assert!((s["last200_fraction_above99"].as_f64().unwrap() - 1. / 3.).abs() < 1e-12);
}
