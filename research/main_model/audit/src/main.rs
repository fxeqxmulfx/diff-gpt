//! Independent CPU audit of native suite records, with no Python runtime.
#[path = "../../../complex_geometry_suite/native/src/random.rs"]
#[expect(
    dead_code,
    reason = "Frozen RNG also exposes methods used only by data generation"
)]
mod random;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const ARMS: [&str; 2] = ["main", "mhc"];
const VALIDATION: &str = include_str!("../../results/mhc-stress-05/validation.json");
#[path = "../../native/src/sources.rs"]
mod candidate_sources;
fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|x| format!("{x:02x}"))
        .collect()
}
fn read(path: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}
fn number(v: &Value, k: &str) -> Result<usize> {
    Ok(v[k].as_u64().ok_or_else(|| format!("missing {k}"))? as usize)
}
fn require(ok: bool, message: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(message.to_owned().into())
    }
}
fn archive_hash(root: &Path) -> Result<String> {
    let mut source = Vec::new();
    let base = root.join("source");
    for &(name, _) in candidate_sources::FILES {
        let body = fs::read(base.join(name))?;
        source.extend((name.len() as u64).to_le_bytes());
        source.extend(name.as_bytes());
        source.extend((body.len() as u64).to_le_bytes());
        source.extend(body);
    }
    Ok(hash(&source))
}

// CPU-only geometry oracle. Do not depend on the GPU engine or its layout code.
type ComplexPair = (usize, usize, Option<(usize, usize)>);
struct Geometry {
    base_parameters: usize,
    router_shape: Option<(usize, usize)>,
    parameters: usize,
    independent: usize,
    hidden: usize,
    // real, imaginary, optional (-imaginary, real) matrix duplicates
    pairs: Vec<ComplexPair>,
}
impl Geometry {
    fn new(width: usize, heads: usize, layers: usize, vocab: usize, mhc: bool) -> Self {
        let hidden = ((8 * width + 1) / 3).div_ceil(2) * 2;
        let per_layer = 4 * width * width + 3 * width * hidden;
        let parameters = vocab * width + layers * per_layer;
        let mut pairs = Vec::new();
        let d = width / heads;
        for token in 0..vocab {
            for head in 0..heads {
                for i in 0..d / 2 {
                    let re = token * width + head * d + i;
                    pairs.push((re, re + d / 2, None));
                }
            }
        }
        for layer in 0..layers {
            let offset = vocab * width + layer * per_layer;
            for (base, rows, cols, row_groups, col_groups) in [
                (offset, 3 * width, width, 3 * heads, heads),
                (offset + 3 * width * width, width, width, heads, heads),
                (offset + 4 * width * width, 2 * hidden, width, 2, heads),
                (
                    offset + 4 * width * width + 2 * hidden * width,
                    width,
                    hidden,
                    heads,
                    1,
                ),
            ] {
                let rh = rows / row_groups / 2;
                let ch = cols / col_groups / 2;
                for rg in 0..row_groups {
                    for r in 0..rh {
                        for cg in 0..col_groups {
                            for c in 0..ch {
                                let re = base + (rg * 2 * rh + r) * cols + cg * 2 * ch + c;
                                let im = re + rh * cols;
                                pairs.push((re, im, Some((re + ch, im + ch))));
                            }
                        }
                    }
                }
            }
        }
        let extra = if mhc {
            2 * layers * (96 * width + 27)
        } else {
            0
        };
        Self {
            base_parameters: parameters,
            router_shape: if mhc { Some((width, layers)) } else { None },
            parameters: parameters + extra,
            independent: 2 * pairs.len() + extra,
            hidden,
            pairs,
        }
    }
    fn initial(&self, seed: u32, complex: bool) -> Vec<f32> {
        let mut rng = random::Random::new(seed);
        let mut result: Vec<f32> = (0..self.base_parameters)
            .map(|_| rng.gauss(0., 0.02) as f32)
            .collect();
        if complex {
            for &(re, im, duplicate) in &self.pairs {
                if let Some((neg_im, copy_re)) = duplicate {
                    result[neg_im] = -result[im];
                    result[copy_re] = result[re];
                }
            }
        }
        if let Some((width, layers)) = self.router_shape {
            let mut rng = random::Random::new(seed + 20000);
            for _ in 0..2 * layers {
                result.extend((0..24 * 4 * width).map(|_| rng.gauss(0., 0.02) as f32));
                result.extend([0.; 8]);
                for row in 0..4 {
                    for col in 0..4 {
                        result.push(if row == col { 2. } else { 0. });
                    }
                }
                result.extend([0.01; 3]);
            }
        }
        result
    }
    fn check_state(&self, state: &[f32], arm: &str) -> Result<()> {
        require(state.len() == 3 * self.parameters, "state size mismatch")?;
        require(
            state.iter().all(|x| x.is_finite()),
            "nonfinite optimizer state",
        )?;
        let (p, rest) = state.split_at(self.parameters);
        let (m, v) = rest.split_at(self.parameters);
        require(v.iter().all(|x| *x >= 0.), "negative AdamW variance")?;
        if arm != "real-product" {
            for &(re, im, duplicate) in &self.pairs {
                if let Some((neg_im, copy_re)) = duplicate {
                    for values in [p, m] {
                        require(
                            values[re] == values[copy_re] && values[im] == -values[neg_im],
                            "broken complex-linear weights or first moments",
                        )?;
                    }
                    require(
                        v[re] == v[copy_re] && v[im] == v[neg_im],
                        "broken complex matrix variance ties",
                    )?;
                }
                if arm == "complex-hermitian-phase" {
                    require(v[re] == v[im], "broken shared complex-pair variance")?;
                }
            }
        }
        Ok(())
    }
}
fn sampler_hash(n: usize, batch: usize, updates: usize, seed: u32) -> String {
    let mut rng = random::Random::new(seed + 10000);
    let mut sha = Sha256::new();
    let mut done = 0;
    while done < updates {
        let mut order: Vec<u32> = (0..n as u32).collect();
        rng.shuffle(&mut order);
        for chunk in order.chunks(batch) {
            if done == updates {
                break;
            }
            for value in chunk {
                sha.update(value.to_le_bytes());
            }
            done += 1;
        }
    }
    sha.finalize().iter().map(|x| format!("{x:02x}")).collect()
}
fn examples(step: usize, n: usize, batch: usize) -> usize {
    let per = n.div_ceil(batch);
    (step / per) * n + ((step % per) * batch).min(n)
}
fn audit_cell(
    root: &Path,
    row: &Value,
    seed: u32,
    arm: &str,
    source: &Value,
    data: &Value,
    sampler: &str,
) -> Result<Value> {
    let name = row["name"].as_str().unwrap();
    let dir = root.join(arm).join(name).join(format!("seed{seed}"));
    let result = read(&dir.join("result.json"))?;
    let protocol = read(&dir.join("protocol.json"))?;
    let checkpoint = read(&dir.join("checkpoint.json"))?;
    let execution = read(&dir.join("execution.json"))?;
    require(
        protocol["case"] == *row && protocol["seed"] == seed && protocol["arm"] == arm,
        "declared case/seed/arm changed",
    )?;
    require(
        result["metadata"] == protocol,
        "result metadata disagrees with protocol",
    )?;
    require(
        protocol["source_sha256"] == source["source_sha256"]
            && protocol["ptx_sha256"] == source["ptx_sha256"],
        "source/PTX provenance mismatch",
    )?;
    require(
        checkpoint["source_sha256"] == source["source_sha256"]
            && checkpoint["ptx_sha256"] == source["ptx_sha256"],
        "checkpoint provenance mismatch",
    )?;
    require(
        protocol["sampler_sha256"] == sampler,
        "native sampler fingerprint changed",
    )?;
    let initial = fs::read(dir.join("initial.f32"))?;
    require(
        protocol["initial_parameters_sha256"] == hash(&initial),
        "initial parameter hash mismatch",
    )?;
    let c = &row["case"];
    let geometry = Geometry::new(
        number(c, "width")?,
        number(c, "heads")?,
        number(c, "layers")?,
        number(data, "vocab")?,
        arm == "mhc",
    );
    let params = number(&protocol, "expanded_parameters")?;
    require(
        params == geometry.parameters && number(&protocol, "hidden")? == geometry.hidden,
        "expanded model dimensions changed",
    )?;
    let independent = if arm == "real-product" {
        params
    } else {
        geometry.independent
    };
    require(
        number(&protocol, "independent_parameters")? == independent,
        "independent parameter count changed",
    )?;
    require(
        protocol["learning_rate"].as_f64() == c["learning_rate"].as_f64().map(|x| 2. * x)
            && protocol["warmup"] == c["warmup"]
            && protocol["betas"] == json!([0.9, 0.98])
            && protocol["epsilon"] == json!(1e-8)
            && (protocol["weight_decay"].as_f64().ok_or("missing decay")?
                - c["weight_decay"].as_f64().ok_or("missing task decay")?)
            .abs()
                < 1e-7
            && protocol["checkpoint_epochs"] == 500
            && protocol["gain"].as_f64()
                == Some(number(c, "heads")? as f64 / number(c, "width")? as f64),
        "optimizer, gain or save cadence changed",
    )?;
    require(
        protocol["hermitian_xsa"] == false
            && protocol["ordinary_attention"] == true
            && protocol["plain_complex_product"] == true
            && protocol["mhc"] == (arm == "mhc")
            && protocol["primary"] == data["primary"]
            && protocol["optimizer"]
                == if arm.ends_with("-phase") {
                    "AdamW shared complex-pair variance"
                } else {
                    "AdamW separate real/imaginary variances"
                },
        "declared architecture, optimizer or primary metric changed",
    )?;
    require(initial.len() == 4 * params, "initial state layout mismatch")?;
    let expected: Vec<u8> = geometry
        .initial(seed, arm != "real-product")
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    require(
        initial == expected,
        "initial weights differ from declared native RNG",
    )?;
    require(
        protocol["base_initial_sha256"] == hash(&initial[..4 * geometry.base_parameters]),
        "base initialization hash mismatch",
    )?;
    let state = fs::read(dir.join("checkpoint.bin"))?;
    let history = fs::read(dir.join("history.jsonl"))?;
    require(state.len() == 12 * params, "checkpoint size mismatch")?;
    require(
        checkpoint["state_sha256"] == hash(&state)
            && checkpoint["history_sha256"] == hash(&history),
        "checkpoint/history integrity failure",
    )?;
    geometry.check_state(
        &state
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| f32::from_le_bytes(*v))
            .collect::<Vec<_>>(),
        arm,
    )?;
    require(
        execution["clean"] == true
            && execution["durable_completed"] == execution["possible_completed"],
        "uncertain execution exposure",
    )?;
    let n = number(row, "train_examples")?;
    let heldout = number(row, "heldout_examples")?;
    let batch = number(c, "batch")?;
    let observe = number(c, "observe")?;
    let budget = number(c, "max_updates")?;
    let flush = 500 * n.div_ceil(batch);
    let records: Vec<Value> = history
        .split(|&v| v == b'\n')
        .filter(|v| !v.is_empty())
        .map(serde_json::from_slice)
        .collect::<std::result::Result<_, _>>()?;
    require(!records.is_empty(), "empty history")?;
    let mut previous = 0;
    let mut previous_seconds = 0.;
    let mut first = None;
    for (i, point) in records.iter().enumerate() {
        let step = number(point, "step")?;
        let count = examples(step, n, batch);
        let expected_step = if i == 0 {
            0
        } else {
            (previous + observe)
                .min(budget)
                .min((previous / flush + 1) * flush)
        };
        require(
            step == expected_step,
            "observation cadence/update history changed",
        )?;
        require(
            step <= budget && number(point, "examples")? == count,
            "budget or example count mismatch",
        )?;
        require(
            (point["epochs"].as_f64().ok_or("missing epochs")? - count as f64 / n as f64).abs()
                < 1e-9,
            "epoch denominator changed",
        )?;
        let seconds = point["seconds"].as_f64().ok_or("missing time")?;
        require(seconds >= previous_seconds, "elapsed time went backwards")?;
        for (which, size) in [("train", n), ("validation", heldout)] {
            let m = &point[which];
            require(
                number(m, "examples")? == size,
                "incomplete split measurement",
            )?;
            let exact = number(m, "exact")?;
            require(exact <= size, "impossible exact count")?;
            require(
                (m["sequence_accuracy"].as_f64().ok_or("missing accuracy")?
                    - exact as f64 / size as f64)
                    .abs()
                    < 1e-12,
                "accuracy denominator changed",
            )?;
            require(
                m["loss"].as_f64().is_some_and(f64::is_finite),
                "nonfinite loss",
            )?;
        }
        if first.is_none() && number(&point["validation"], "exact")? * 100 >= 99 * heldout {
            first = Some((step, point["epochs"].clone()));
        }
        previous = step;
        previous_seconds = seconds;
    }
    let last = records.last().unwrap();
    let completed = number(&result, "completed")?;
    require(
        completed == previous
            && completed == number(&checkpoint, "completed")?
            && completed == number(&execution, "durable_completed")?,
        "endpoint/update mismatch",
    )?;
    require(
        result["endpoint"] == *last,
        "endpoint differs from last observation",
    )?;
    require(
        result["examples"] == checkpoint["examples"] && result["examples"] == last["examples"],
        "checkpoint sample count mismatch",
    )?;
    match result["status"].as_str() {
        Some("passed") => {
            let (step, epoch) = first.as_ref().ok_or("pass without threshold crossing")?;
            require(
                *step == completed && result["first99_epochs"] == *epoch,
                "wrong first crossing",
            )?;
        }
        Some("budget_exhausted") => {
            require(
                completed == budget && first.is_none() && result["first99_epochs"].is_null(),
                "incorrect censored budget result",
            )?;
        }
        Some("stopped") => require(
            first.is_none() && result["first99_epochs"].is_null(),
            "incorrect stop result",
        )?,
        _ => return Err("unknown endpoint status".into()),
    }
    let first_prior = if first.is_some() && records.len() > 1 {
        records[records.len() - 2]["epochs"].clone()
    } else {
        json!(0.)
    };
    if first.is_some() {
        require(
            result["previous_observed_epochs"] == first_prior,
            "first crossing interval changed",
        )?;
    }
    Ok(
        json!({"case":name,"tier":row["tier"],"seed":seed,"arm":arm,"status":result["status"],"audit":"passed",
        "completed":completed,"examples":result["examples"],"epochs":result["epochs"],
        "first99_epochs":result["first99_epochs"],"previous_observed_epochs":result["previous_observed_epochs"],
        "sequence_accuracy":last["validation"]["sequence_accuracy"],"train_accuracy":last["train"]["sequence_accuracy"],
        "seconds":result["seconds"],"observations":records.len(),"initial_parameters_sha256":protocol["initial_parameters_sha256"],
        "expanded_parameters":params,"independent_parameters":independent,"base_initial_sha256":protocol["base_initial_sha256"],
        "sampler_sha256":protocol["sampler_sha256"],"checkpoint_sha256":checkpoint["state_sha256"],
        "history_sha256":checkpoint["history_sha256"],"source_sha256":protocol["source_sha256"]}),
    )
}
fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: main-model-audit RUN_ROOT OUTPUT.json".into());
    }
    let root = PathBuf::from(&args[1]);
    let manifest: Value =
        serde_json::from_str(include_str!("../../../complex_geometry_suite/suite.json"))?;
    let validated: Value = serde_json::from_str(VALIDATION)?;
    let mut datasets = BTreeMap::new();
    let protocol = if root.join("protocol.json").exists() {
        let p = read(&root.join("protocol.json"))?;
        require(
            p["suite_sha256"] == hash(include_bytes!("../../../complex_geometry_suite/suite.json"))
                && p["plan_sha256"] == hash(include_bytes!("../../MHC_PLAN.txt"))
                && p["arms"] == json!(ARMS)
                && p["seeds"] == json!([0, 1, 2])
                && p["expected_cells"] == 210,
            "frozen suite or declared cohort changed",
        )?;
        require(
            p["source_sha256"] == archive_hash(&root)?
                && p["source_sha256"] == validated["source_sha256"]
                && p["ptx_sha256"] == validated["ptx_sha256"],
            "archived sources/PTX differ from validated commit",
        )?;
        let data = read(&root.join("data-validation.json"))?;
        require(
            data["suite_sha256"] == p["suite_sha256"],
            "dataset suite changed",
        )?;
        for row in manifest["tasks"].as_array().ok_or("missing tasks")? {
            let name = row["name"].as_str().ok_or("missing case name")?;
            let item = data["checked_tasks"]
                .as_array()
                .ok_or("missing data validation")?
                .iter()
                .find(|x| x["case"] == name)
                .ok_or("task data validation missing")?;
            require(
                item["train"] == row["train_examples"]
                    && item["heldout"] == row["heldout_examples"]
                    && item["primary"] == row["selection"]
                    && item["all_frozen_fingerprints_match"] == true,
                "data sizes, primary split or fingerprints changed",
            )?;
            datasets.insert(name, item.clone());
        }
        Some(p)
    } else {
        None
    };
    let mut rows = Vec::new();
    let mut cache = BTreeMap::new();
    let mut failures = Vec::new();
    for row in manifest["tasks"].as_array().unwrap() {
        let name = row["name"].as_str().unwrap();
        for seed in 0..3u32 {
            for arm in ARMS {
                let dir = root.join(arm).join(name).join(format!("seed{seed}"));
                if !dir.join("result.json").exists() {
                    let status = if dir.join("error.json").exists() {
                        "execution_error"
                    } else if dir.exists() {
                        "in_progress"
                    } else {
                        "pending"
                    };
                    rows.push(json!({"case":name,"tier":row["tier"],"seed":seed,"arm":arm,"status":status,"audit":"pending","first99_epochs":null}));
                    continue;
                }
                let n = number(row, "train_examples")?;
                let batch = number(&row["case"], "batch")?;
                let budget = number(&row["case"], "max_updates")?;
                let sampler = cache
                    .entry((n, batch, budget, seed))
                    .or_insert_with(|| sampler_hash(n, batch, budget, seed));
                let source = protocol.as_ref().ok_or("missing cohort protocol")?;
                let data = datasets.get(name).ok_or("missing task data validation")?;
                match audit_cell(&root, row, seed, arm, source, data, sampler) {
                    Ok(result) => rows.push(result),
                    Err(e) => {
                        failures.push(format!("{name} {arm} seed{seed}: {e}"));
                        rows.push(json!({"case":name,"tier":row["tier"],"seed":seed,"arm":arm,"status":"audit_failure","audit":e.to_string()}));
                    }
                }
            }
        }
    }
    let mut totals = Vec::new();
    for tier in ["short", "long"] {
        for arm in ARMS {
            let cells: Vec<_> = rows
                .iter()
                .filter(|r| r["tier"] == tier && r["arm"] == arm)
                .collect();
            let mut statuses: BTreeMap<String, usize> = BTreeMap::new();
            for cell in &cells {
                *statuses
                    .entry(cell["status"].as_str().unwrap().into())
                    .or_default() += 1;
            }
            totals.push(json!({"tier":tier,"arm":arm,"expected":cells.len(),"statuses":statuses}));
        }
    }
    let mut pairs = Vec::new();
    for row in manifest["tasks"].as_array().unwrap() {
        for seed in 0..3 {
            {
                let (candidate, control) = ("mhc", "main");
                let a = rows
                    .iter()
                    .find(|r| {
                        r["case"] == row["name"] && r["seed"] == seed && r["arm"] == candidate
                    })
                    .unwrap();
                let b = rows
                    .iter()
                    .find(|r| r["case"] == row["name"] && r["seed"] == seed && r["arm"] == control)
                    .unwrap();
                if a["audit"] == "passed" && b["audit"] == "passed" {
                    require(
                        (a["base_initial_sha256"] == b["base_initial_sha256"])
                            && a["sampler_sha256"] == b["sampler_sha256"],
                        "complex-arm initialization or paired samples differ",
                    )?;
                    let ratio = match (a["first99_epochs"].as_f64(), b["first99_epochs"].as_f64()) {
                        (Some(x), Some(y)) if x > 0. && y > 0. => Some(x / y),
                        _ => None,
                    };
                    pairs.push(json!({"case":row["name"],"tier":row["tier"],"seed":seed,
                    "candidate":candidate,"control":control,"candidate_status":a["status"],"control_status":b["status"],"candidate_over_control_epochs":ratio}));
                }
            }
        }
    }
    let output = json!({"expected_cells":210,"complete":rows.iter().all(|r|r["audit"]=="passed"&&r["status"]!="stopped"),
        "source_verified":protocol.is_some(),"source_sha256":protocol.as_ref().map(|p|p["source_sha256"].clone()),
        "comparison":"same base parameters and samples; mHC adds 2L*(96D+27) trainable routing parameters",
        "all_failures_retained":true,"python_invoked":false,"audit_failures":failures,"totals":totals,"pairs":pairs,"rows":rows});
    let path = Path::new(&args[2]);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temp = path.with_extension("pending");
    fs::write(&temp, serde_json::to_vec_pretty(&output)?)?;
    fs::rename(temp, path)?;
    use std::fmt::Write as _;
    let mut table =
        String::from("case\ttier\tseed\tarm\tstatus\tfirst99_epochs\tfinal_accuracy\taudit\n");
    for row in &rows {
        writeln!(
            &mut table,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            row["case"].as_str().unwrap(),
            row["tier"].as_str().unwrap(),
            row["seed"],
            row["arm"].as_str().unwrap(),
            row["status"].as_str().unwrap(),
            row["first99_epochs"]
                .as_f64()
                .map(|v| v.to_string())
                .unwrap_or_default(),
            row["sequence_accuracy"]
                .as_f64()
                .map(|v| v.to_string())
                .unwrap_or_default(),
            row["audit"].as_str().unwrap()
        )?;
    }
    let table_path = path.with_extension("tsv");
    let temp = table_path.with_extension("pending");
    fs::write(&temp, table)?;
    fs::rename(temp, table_path)?;
    println!(
        "{}",
        json!({"complete":output["complete"],"totals":output["totals"],"audit_failures":output["audit_failures"]})
    );
    require(failures.is_empty(), "audit failures retained in report")
}

#[test]
fn geometry_audit_rejects_broken_weights_moments_and_pair_variances() -> Result<()> {
    for (width, heads, layers, vocab) in [(64, 4, 2, 11), (128, 4, 1, 97), (128, 1, 2, 256)] {
        let geometry = Geometry::new(width, heads, layers, vocab, false);
        let mut seen = vec![0usize; geometry.parameters];
        for &(re, im, duplicate) in &geometry.pairs {
            seen[re] += 1;
            seen[im] += 1;
            if let Some((neg_im, copy_re)) = duplicate {
                seen[neg_im] += 1;
                seen[copy_re] += 1;
            }
        }
        assert!(seen.iter().all(|n| *n == 1));
        if width == 128 && layers == 1 && vocab == 97 {
            assert_eq!(
                (geometry.parameters, geometry.independent),
                (209280, 110848)
            );
        }
        let mut state = geometry.initial(1, true);
        state.resize(3 * geometry.parameters, 0.);
        geometry.check_state(&state, "complex-hermitian-phase")?;
        let &(re, im, Some((neg_im, copy_re))) =
            geometry.pairs.iter().find(|p| p.2.is_some()).unwrap()
        else {
            unreachable!()
        };
        for index in [
            copy_re,
            geometry.parameters + neg_im,
            2 * geometry.parameters + copy_re,
        ] {
            let mut corrupt = state.clone();
            corrupt[index] += 1.;
            assert!(
                geometry
                    .check_state(&corrupt, "complex-hermitian-phase")
                    .is_err()
            );
        }
        state[2 * geometry.parameters + re] = 1.;
        state[2 * geometry.parameters + copy_re] = 1.;
        geometry.check_state(&state, "complex-hermitian")?;
        assert!(
            geometry
                .check_state(&state, "complex-hermitian-phase")
                .is_err()
        );
        state[2 * geometry.parameters + im] = 1.;
        state[2 * geometry.parameters + neg_im] = 1.;
        geometry.check_state(&state, "complex-hermitian-phase")?;
        state[2 * geometry.parameters] = -1.;
        assert!(geometry.check_state(&state, "real-product").is_err());
    }
    Ok(())
}

#[test]
fn exposure_counts_short_tail_batches_exactly() {
    assert_eq!(examples(0, 10, 4), 0);
    assert_eq!(examples(2, 10, 4), 8);
    assert_eq!(examples(3, 10, 4), 10);
    assert_eq!(examples(5, 10, 4), 18);
    assert_eq!(examples(6, 10, 4), 20);
}
fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[test]
fn mhc_audit_checks_extra_parameters_without_complex_ties() -> Result<()> {
    let base = Geometry::new(128, 4, 1, 97, false);
    let mhc = Geometry::new(128, 4, 1, 97, true);
    assert_eq!((mhc.parameters, mhc.independent), (233910, 135478));
    for seed in 0..3 {
        let main = base.initial(seed, true);
        let candidate = mhc.initial(seed, true);
        assert_eq!(&candidate[..main.len()], main);
        let mut state = candidate.clone();
        state.resize(3 * mhc.parameters, 0.);
        mhc.check_state(&state, "mhc")?;
        for location in [
            base.parameters,
            mhc.parameters + base.parameters,
            3 * mhc.parameters - 1,
        ] {
            let mut bad = state.clone();
            bad[location] = f32::NAN;
            assert!(mhc.check_state(&bad, "mhc").is_err());
        }
        state[2 * mhc.parameters + base.parameters] = -1.;
        assert!(mhc.check_state(&state, "mhc").is_err());
    }
    Ok(())
}
