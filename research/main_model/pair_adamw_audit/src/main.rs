//! Independent audit of the 105-cell complex-pair AdamW candidate.
//! Geometry reconstructs shared frequencies independently; sampler/history/state
//! checks do not depend on the GPU engine or its layout module.
#[path = "../../../complex_geometry_suite/native/src/random.rs"]
#[expect(dead_code, reason = "Frozen RNG also exposes data-generation methods")]
mod random;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
// Frozen only after native numerical, state, sanitizer and control-replay checks.
const SOURCE: &str = "260444853b45837b19f1b2adedaac957a3f82b2c595487b00602e510ea8386cb";
const PTX: &str = "ba07795eeb08a13691f7f15848f6bea3dc239315fe9211e74c3e21a598ac030f";
const BINARY: &str = "e56f2c138900d7024059fa942157cd8b96391ccd0bec272c06e1ce9f4ae5eaf0";
const CONTROL: &str = "research/main_model/results/lerope-suite-01/final-audit.json";
const CONTROL_HASH: &str = "c6e33bb035d04d7d2e8e6c8ef8315c36f6453d675f0d06eafc4243a034ff541b";
const ARTIFACTS: &str = "research/main_model/results/pair-adamw-validation-final";
const CANDIDATE_ARM: &str = "pair-lerope";
const CONTROL_ARM: &str = "lerope";
const PLAN: &[u8] = include_bytes!("../../COMPLEX_PAIR_ADAMW_PLAN.txt");
const COMPILED_PLAN: &[u8] = include_bytes!("../../COMPLEX_PAIR_ADAMW_PLAN.txt");
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
include!("oracle.rs");

fn declared(manifest: &Value) -> Result<Vec<Value>> {
    let mut cells = Vec::new();
    for tier in ["short", "long"] {
        for task in manifest["tasks"].as_array().ok_or("missing tasks")? {
            if task["tier"] == tier {
                for seed in 0..3 {
                    cells.push(json!({"task":task["name"],"seed":seed,"tier":tier}));
                }
            }
        }
    }
    require(
        cells.len() == 105 && cells.iter().filter(|v| v["tier"] == "short").count() == 75,
        "frozen suite scope changed",
    )?;
    Ok(cells)
}

fn archive_hash(root: &Path) -> Result<String> {
    let base = root.join("source");
    let manifest =
        fs::read_to_string(base.join("research/main_model/pair_adamw_native/src/sources.rs"))?;
    let mut bytes = Vec::new();
    let mut seen = BTreeSet::new();
    for name in manifest
        .lines()
        .filter_map(|line| line.split('"').find(|s| s.starts_with("research/")))
    {
        require(seen.insert(name), "duplicate source entry")?;
        let body = fs::read(base.join(name))?;
        bytes.extend((name.len() as u64).to_le_bytes());
        bytes.extend(name.as_bytes());
        bytes.extend((body.len() as u64).to_le_bytes());
        bytes.extend(body);
    }
    require(!seen.is_empty(), "empty source manifest")?;
    Ok(hash(&bytes))
}

fn phase_check(rows: &[Value], terminal: &Value) -> Result<()> {
    require(rows.len() == 105, "wrong cell count")?;
    let passed = rows.iter().take_while(|r| r["status"] == "passed").count();
    let remaining_unrun = |start: usize| rows[start..].iter().all(|r| r["status"] == "unrun");
    match terminal["status"].as_str() {
        Some("prepared") => require(
            remaining_unrun(0),
            "prepared scope already contains training",
        ),
        Some("short_complete") => require(
            passed == 75 && remaining_unrun(75),
            "short audit needs exactly75 completed short passes",
        ),
        Some("complete") => require(passed == 105, "complete phase requires all105 passes"),
        Some("stopped_between_cells") => require(
            passed < 105 && remaining_unrun(passed),
            "between-cell stop must preserve a contiguous passed prefix",
        ),
        Some("rejected_lost_main_pass" | "stopped") => {
            require(passed < 105, "terminal failure missing")?;
            let failed = &rows[passed];
            let expected = if terminal["status"] == "stopped" {
                "stopped"
            } else {
                "budget_exhausted"
            };
            require(
                failed["status"] == expected
                    && remaining_unrun(passed + 1)
                    && terminal["task"] == failed["case"]
                    && terminal["seed"] == failed["seed"],
                "failure/stop is not the last exposed cell in declared order",
            )
        }
        _ => Err("unknown/nonterminal audit phase".into()),
    }
}

fn tier_stats(rows: &[Value], tier: &str, expected: usize) -> Value {
    let cells: Vec<_> = rows.iter().filter(|r| r["tier"] == tier).collect();
    let solved: Vec<_> = cells
        .iter()
        .copied()
        .filter(|r| r["status"] == "passed" && r["audit"] == "passed")
        .collect();
    let complete = cells.len() == expected && solved.len() == expected;
    let ratio = if complete {
        if solved
            .iter()
            .any(|r| r["candidate_over_main_epochs"].as_f64() == Some(0.))
        {
            Some(0.)
        } else {
            Some(
                (solved
                    .iter()
                    .map(|r| r["candidate_over_main_epochs"].as_f64().unwrap().ln())
                    .sum::<f64>()
                    / expected as f64)
                    .exp(),
            )
        }
    } else {
        None
    };
    json!({"tier":tier,"expected":expected,"solved":solved.len(),"complete":complete,
        "geometric_mean_epoch_ratio":ratio,
        "earlier":solved.iter().filter(|r|r["candidate_over_main_epochs"].as_f64().unwrap()<1.).count(),
        "tied":solved.iter().filter(|r|r["candidate_over_main_epochs"].as_f64().unwrap()==1.).count(),
        "later":solved.iter().filter(|r|r["candidate_over_main_epochs"].as_f64().unwrap()>1.).count()})
}

fn promotion(rows: &[Value], failures: &[String]) -> Value {
    let short = tier_stats(rows, "short", 75);
    let long = tier_stats(rows, "long", 30);
    let lost: Vec<_> = rows
        .iter()
        .filter(|r| r["audit"] == "passed" && r["status"] == "budget_exhausted")
        .map(|r| json!({"case":r["case"],"seed":r["seed"],"tier":r["tier"]}))
        .collect();
    let coverage = failures.is_empty() && short["complete"] == true && long["complete"] == true;
    let improves = short["geometric_mean_epoch_ratio"]
        .as_f64()
        .is_some_and(|v| v < 1.);
    let eligible = coverage && improves;
    json!({"short":short,"long":long,"lost_main_passes":lost,"coverage_requirement_satisfied":coverage,
        "short_epoch_improvement":improves,"eligible_for_promotion":eligible,
        "status":if !failures.is_empty(){"invalid_evidence"}else if !lost.is_empty(){"rejected_lost_main_pass"}
            else if eligible{"eligible_full_coverage_and_fewer_short_epochs"}else if coverage{"rejected_no_short_epoch_gain"}else{"pending"},
        "rule":"All105main passes and fewer geometric-mean short epochs; wall time alone never qualifies."})
}

fn matched_initialization(row: &Value, baseline: &Value) -> Result<()> {
    for key in [
        "initial_parameters_sha256",
        "common_initial_sha256",
        "base_initial_sha256",
        "sampler_sha256",
    ] {
        require(
            row[key].as_str().is_some_and(|s| s.len() == 64) && row[key] == baseline[key],
            &format!("control {key} mismatch"),
        )?;
    }
    for key in [
        "expanded_parameters",
        "independent_parameters",
        "frequency_parameters",
    ] {
        require(
            row[key].as_u64().is_some_and(|v| v > 0) && row[key] == baseline[key],
            &format!("control {key} mismatch"),
        )?;
    }
    Ok(())
}

fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    require(
        args.len() == 3,
        "usage: main-model-pair-adamw-audit RUN_ROOT FRESH_OUTPUT.json",
    )?;
    let root = Path::new(&args[1]);
    let output = Path::new(&args[2]);
    require(!output.exists(), "refusing to overwrite audit")?;
    let manifest: Value =
        serde_json::from_str(include_str!("../../../complex_geometry_suite/suite.json"))?;
    let cells = declared(&manifest)?;
    let plan = read(&root.join("experiment.json"))?;
    require(
        plan["cells"] == json!(cells)
            && plan["expected_cells"] == 105
            && plan["candidate_arm"] == CANDIDATE_ARM
            && plan["control_arm"] == CONTROL_ARM
            && plan["control_root"] == "runs/main-model-lerope-01"
            && plan["source_sha256"] == SOURCE
            && plan["ptx_sha256"] == PTX
            && plan["binary_sha256"] == BINARY
            && plan["control_audit_sha256"] == CONTROL_HASH
            && plan["plan_sha256"] == hash(PLAN)
            && plan["compiled_plan_sha256"] == hash(COMPILED_PLAN)
            && plan["fresh_initialization"] == true
            && plan["parameter_count_matched"] == true,
        "experiment scope or pins changed",
    )?;
    require(
        fs::read(root.join("PLAN.txt"))? == PLAN,
        "archived experiment plan changed",
    )?;
    require(
        hash(&fs::read(root.join("pair-adamw-transformer"))?) == BINARY
            && hash(&fs::read(Path::new(ARTIFACTS).join("candidate"))?) == BINARY,
        "training executable changed",
    )?;
    let source = read(&root.join("protocol.json"))?;
    let validated = read(&Path::new(ARTIFACTS).join("ptx/report.json"))?;
    require(
        source["source_sha256"] == SOURCE
            && source["ptx_sha256"] == PTX
            && archive_hash(root)? == SOURCE
            && validated["source_sha256"] == SOURCE
            && validated["ptx_sha256"] == PTX
            && validated["passed"] == true,
        "source/PTX provenance mismatch",
    )?;
    require(
        source["suite_sha256"]
            == hash(include_bytes!("../../../complex_geometry_suite/suite.json"))
            && source["plan_sha256"] == hash(COMPILED_PLAN)
            && source["peak_lr"] == json!(0.001)
            && source["betas"] == json!([0.9, 0.95])
            && source["warmup_fraction"] == json!(0.02)
            && source["cosine_decay_fraction"] == json!(0.2)
            && source["weight_decay"] == json!(0.1)
            && source["global_independent_gradient_clip"] == 1,
        "frozen data/recipe changed",
    )?;
    require(
        source["arms"] == json!(["main", "mhc", "lerope", "log-lerope", "pair-lerope"])
            && source["expected_cells"] == 525,
        "generic runner scope changed",
    )?;
    require(!root.join("main").exists(), "undeclared main arm present")?;
    require(!root.join("mhc").exists(), "undeclared control arm present")?;
    require(
        !root.join(CONTROL_ARM).exists(),
        "undeclared lerope control arm present",
    )?;
    require(!root.join("gated").exists(), "undeclared gated arm present")?;
    require(
        !root.join("log-lerope").exists(),
        "undeclared log-lerope arm present",
    )?;
    if root.join(CANDIDATE_ARM).exists() {
        for task in fs::read_dir(root.join(CANDIDATE_ARM))? {
            let task = task?;
            let name = task.file_name();
            require(
                cells
                    .iter()
                    .any(|c| Some(c["task"].as_str().unwrap()) == name.to_str()),
                "undeclared task directory",
            )?;
            for seed in fs::read_dir(task.path())? {
                require(
                    matches!(
                        seed?.file_name().to_str(),
                        Some("seed0" | "seed1" | "seed2")
                    ),
                    "undeclared seed directory",
                )?;
            }
        }
    }
    let control_bytes = fs::read(CONTROL)?;
    require(
        hash(&control_bytes) == CONTROL_HASH,
        "control audit changed",
    )?;
    let control: Value = serde_json::from_slice(&control_bytes)?;
    require(
        control["complete"] == true
            && control["audit_passed"] == true
            && control["source_verified"] == true
            && control["audit_failures"]
                .as_array()
                .is_some_and(Vec::is_empty),
        "main control is not completely audited",
    )?;
    let controls: Vec<_> = control["rows"]
        .as_array()
        .ok_or("no control rows")?
        .iter()
        .filter(|r| r["arm"] == CONTROL_ARM)
        .collect();
    require(
        controls.len() == 105
            && controls
                .iter()
                .all(|r| r["status"] == "passed" && r["audit"] == "passed"),
        "main105passes missing",
    )?;
    let data = read(&root.join("data-validation.json"))?;
    require(
        data["suite_sha256"] == source["suite_sha256"],
        "dataset suite changed",
    )?;
    let precision = fs::read_to_string(
        root.join("source/research/main_model/pair_adamw_native/build-precision.txt"),
    )?;
    let placement = fs::read_to_string(
        root.join("source/research/main_model/pair_adamw_native/build-placement.txt"),
    )?;
    let mut rows = Vec::new();
    let mut failures = Vec::new();
    let mut cache = BTreeMap::new();
    for cell in &cells {
        let name = cell["task"].as_str().unwrap();
        let seed = number(cell, "seed")? as u32;
        let dir = root
            .join(CANDIDATE_ARM)
            .join(name)
            .join(format!("seed{seed}"));
        if !dir.join("result.json").exists() {
            let status = if dir.exists() {
                "incomplete_execution"
            } else {
                "unrun"
            };
            rows.push(json!({"case":name,"seed":seed,"tier":cell["tier"],"status":status,"audit":"not_applicable"}));
            continue;
        }
        let task = manifest["tasks"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["name"] == name)
            .ok_or("unknown task")?;
        let checked = data["checked_tasks"]
            .as_array()
            .ok_or("missing data preflight")?
            .iter()
            .find(|r| r["case"] == name)
            .ok_or("missing task preflight")?;
        require(
            checked["train"] == task["train_examples"]
                && checked["heldout"] == task["heldout_examples"]
                && checked["primary"] == task["selection"]
                && checked["all_frozen_fingerprints_match"] == true,
            "data size/split/fingerprint changed",
        )?;
        let n = number(task, "train_examples")?;
        let batch = number(&task["case"], "batch")?;
        let budget = number(&task["case"], "max_updates")?;
        let sampler = cache
            .entry((n, batch, budget, seed))
            .or_insert_with(|| sampler_hash(n, batch, budget, seed));
        let audit = (|| -> Result<Value> {
            let mut row = audit_cell(root, task, seed, CANDIDATE_ARM, &source, checked, sampler)?;
            let baseline = controls
                .iter()
                .find(|r| r["case"] == name && r["seed"] == seed)
                .ok_or("missing control cell")?;
            matched_initialization(&row, baseline)?;
            let epoch = baseline["first99_epochs"]
                .as_f64()
                .ok_or("missing main epochs")?;
            require(epoch > 0., "nonpositive main epoch denominator")?;
            row["main_first99_epochs"] = json!(epoch);
            row["main_completed"] = baseline["completed"].clone();
            row["main_previous_observed_epochs"] = baseline["previous_observed_epochs"].clone();
            row["candidate_over_main_epochs"] =
                json!(row["first99_epochs"].as_f64().map(|v| v / epoch));
            let protocol = read(&dir.join("protocol.json"))?;
            require(
                protocol["dtype"] == precision && protocol["placement"] == placement,
                "numerical/placement tag changed",
            )?;
            Ok(row)
        })();
        match audit {
            Ok(row) => rows.push(row),
            Err(e) => {
                failures.push(format!("{name} seed{seed}: {e}"));
                rows.push(json!({"case":name,"seed":seed,"tier":cell["tier"],"status":"audit_failure","audit":"failed"}));
            }
        }
    }
    let terminal = read(&root.join("experiment-status.json"))?;
    if let Err(e) = phase_check(&rows, &terminal) {
        failures.push(e.to_string());
    }
    let gate = promotion(&rows, &failures);
    let report = json!({"audit_passed":failures.is_empty(),"source_verified":true,"complete":gate["coverage_requirement_satisfied"],
        "scope":"all75short_and30long_frozen_pair_adamw_task_seed_cells","expected_cells":105,
        "phase":terminal["status"],"completed_cells":rows.iter().filter(|r|r["audit"]=="passed").count(),
        "solved_cells":rows.iter().filter(|r|r["audit"]=="passed"&&r["status"]=="passed").count(),
        "unrun_cells":rows.iter().filter(|r|r["status"]=="unrun").count(),
        "source_sha256":SOURCE,"ptx_sha256":PTX,"binary_sha256":BINARY,"control_audit_sha256":CONTROL_HASH,
        "strict_numerical_misses_retained":true,"promotion":gate,"rows":rows,"audit_failures":failures});
    fs::write(output, serde_json::to_vec_pretty(&report)?)?;
    println!(
        "{}",
        json!({"phase":report["phase"],"audited":report["completed_cells"],"solved":report["solved_cells"],"unrun":report["unrun_cells"],"promotion":report["promotion"]["status"]})
    );
    require(
        report["audit_passed"] == true,
        "independent suite audit failed",
    )
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shared_variance_is_required_for_embedding_and_projection_pairs_only() {
        let g = Geometry::new(64, 4, 2, 11, true, true);
        let n = g.parameters;
        let mut state = g.initial(0, true);
        state.resize(3 * n, 0.);
        for i in g.base_parameters..n {
            state[2 * n + i] = 0.01 * (1 + i % 7) as f32;
        }
        assert!(g.check_state(&state, CANDIDATE_ARM).is_ok());
        for (re, im, duplicate) in [
            g.pairs[0],
            g.pairs.iter().copied().find(|p| p.2.is_some()).unwrap(),
        ] {
            let mut changed = state.clone();
            changed[2 * n + re] = 0.1;
            changed[2 * n + im] = 0.2;
            if let Some((neg, copy)) = duplicate {
                changed[2 * n + neg] = 0.2;
                changed[2 * n + copy] = 0.1;
            }
            assert!(g.check_state(&changed, "lerope").is_ok());
            assert!(g.check_state(&changed, CANDIDATE_ARM).is_err());
        }
    }
    fn paired_initialization() -> Value {
        json!({"initial_parameters_sha256":"a".repeat(64),
            "common_initial_sha256":"b".repeat(64),"base_initial_sha256":"c".repeat(64),
            "sampler_sha256":"d".repeat(64),"expanded_parameters":233926,
            "independent_parameters":135494,"frequency_parameters":16})
    }
    #[test]
    fn both_rotary_variants_require_full_initial_weights_and_equal_capacity() {
        let base = paired_initialization();
        assert!(matched_initialization(&base, &base).is_ok());
        for key in [
            "initial_parameters_sha256",
            "common_initial_sha256",
            "base_initial_sha256",
            "sampler_sha256",
        ] {
            let mut changed = base.clone();
            changed[key] = json!("e".repeat(64));
            assert!(matched_initialization(&changed, &base).is_err());
        }
        for key in [
            "expanded_parameters",
            "independent_parameters",
            "frequency_parameters",
        ] {
            let mut changed = base.clone();
            changed[key] = json!(base[key].as_u64().unwrap() + 1);
            assert!(matched_initialization(&changed, &base).is_err());
        }
    }
    #[test]
    fn missing_full_initial_hash_cannot_match_as_null() {
        let mut missing = paired_initialization();
        missing
            .as_object_mut()
            .unwrap()
            .remove("initial_parameters_sha256");
        assert!(matched_initialization(&missing, &missing).is_err());
    }
    #[test]
    fn frequency_capacity_and_initial_prefix_are_independently_reconstructed() {
        for (w, h, l, v) in [(64, 4, 2, 11), (128, 4, 1, 97), (128, 1, 6, 101)] {
            let old = Geometry::new(w, h, l, v, true, false);
            let new = Geometry::new(w, h, l, v, true, true);
            assert_eq!(new.parameters - old.parameters, w / h / 2);
            assert_eq!(new.independent - old.independent, w / h / 2);
            assert_eq!(new.frequency_parameters, w / h / 2);
            assert_eq!(new.pairs, old.pairs);
            for seed in 0..3 {
                let p = new.initial(seed, true);
                let baseline = old.initial(seed, true);
                assert!(
                    p[..baseline.len()]
                        .iter()
                        .zip(&baseline)
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                );
                assert!(p[baseline.len()..].iter().all(|v| v.to_bits() == 0));
            }
        }
    }
    #[test]
    fn frequency_state_must_be_finite_with_nonnegative_variance() {
        let geometry = Geometry::new(64, 4, 2, 11, true, true);
        let n = geometry.parameters;
        let mut state = geometry.initial(0, true);
        state.resize(3 * n, 0.);
        let offset = n - geometry.frequency_parameters;
        state[offset] = 0.3;
        state[n + offset] = -0.2;
        state[2 * n + offset] = 0.1;
        assert!(geometry.check_state(&state, "lerope").is_ok());
        state[2 * n + offset] = -0.1;
        assert!(geometry.check_state(&state, "lerope").is_err());
        state[2 * n + offset] = 0.1;
        state[offset] = f32::NAN;
        assert!(geometry.check_state(&state, "lerope").is_err());
    }
    fn fixture() -> Vec<Value> {
        (0..105)
            .map(|i| {
                json!({"case":format!("task{}",i/3),"seed":i%3,"tier":if i<75{"short"}else{"long"},
        "audit":"passed","status":"passed","candidate_over_main_epochs":0.9})
            })
            .collect()
    }
    #[test]
    fn exact_frozen_scope() {
        let v: Value =
            serde_json::from_str(include_str!("../../../complex_geometry_suite/suite.json"))
                .unwrap();
        let c = declared(&v).unwrap();
        assert_eq!(c.len(), 105);
        assert_eq!(c[74]["tier"], "short");
        assert_eq!(c[75]["tier"], "long");
    }
    #[test]
    fn complete_gain_qualifies() {
        let r = fixture();
        assert!(phase_check(&r, &json!({"status":"complete"})).is_ok());
        assert_eq!(promotion(&r, &[])["eligible_for_promotion"], true);
    }
    #[test]
    fn one_lost_pass_defeats_large_gain() {
        let mut r = fixture();
        r[104]["status"] = json!("budget_exhausted");
        r[104]["candidate_over_main_epochs"] = Value::Null;
        assert_eq!(promotion(&r, &[])["status"], "rejected_lost_main_pass");
        assert_eq!(promotion(&r, &[])["eligible_for_promotion"], false);
    }
    #[test]
    fn short_only_cannot_promote() {
        let mut r = fixture();
        for row in &mut r[75..] {
            row["status"] = json!("unrun");
            row["audit"] = json!("not_applicable");
        }
        assert!(phase_check(&r, &json!({"status":"short_complete"})).is_ok());
        let g = promotion(&r, &[]);
        assert_eq!(g["short"]["complete"], true);
        assert_eq!(g["long"]["geometric_mean_epoch_ratio"], Value::Null);
        assert_eq!(g["eligible_for_promotion"], false);
    }
    #[test]
    fn coverage_without_epoch_gain_does_not_promote() {
        let mut r = fixture();
        for row in &mut r {
            row["candidate_over_main_epochs"] = json!(1.1);
        }
        assert_eq!(promotion(&r, &[])["status"], "rejected_no_short_epoch_gain");
    }
    #[test]
    fn invalid_evidence_never_promotes() {
        assert_eq!(
            promotion(&fixture(), &["corrupt checkpoint".into()])["status"],
            "invalid_evidence"
        );
    }
    #[test]
    fn reject_must_be_last_exposed_cell() {
        let mut r = fixture();
        r[7]["status"] = json!("budget_exhausted");
        for row in &mut r[8..] {
            row["status"] = json!("unrun");
        }
        let s = json!({"status":"rejected_lost_main_pass","task":r[7]["case"],"seed":r[7]["seed"]});
        assert!(phase_check(&r, &s).is_ok());
        r[8]["status"] = json!("passed");
        assert!(phase_check(&r, &s).is_err());
    }
    #[test]
    fn explicit_user_stop_between_cells_preserves_unrun_scope() {
        let mut rows = fixture();
        for row in &mut rows[81..] {
            row["status"] = json!("unrun");
            row["audit"] = json!("not_applicable");
        }
        let phase = json!({"status":"stopped_between_cells"});
        assert!(phase_check(&rows, &phase).is_ok());
        assert_eq!(promotion(&rows, &[])["eligible_for_promotion"], false);
        rows[82]["status"] = json!("passed");
        assert!(phase_check(&rows, &phase).is_err());
    }
    #[test]
    fn epoch_and_schedule_oracles() {
        assert_eq!(examples(19, 4704, 256), 4704);
        assert_eq!(expected_rate(100000, 0), 0.);
        assert!(expected_rate(100000, 2000) > 0.000999);
        assert!(expected_rate(100000, 100000) < 0.000001);
    }
}
