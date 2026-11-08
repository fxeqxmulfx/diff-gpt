//! Frozen-checkpoint YaRN evaluation. No optimizer step or checkpoint write.
use crate::{Engine, Result, data, hash, layout, metrics, runner};
use serde_json::{Value, json};
use std::{collections::BTreeMap, fs, io::Write, path::Path};

fn read(path: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

fn save_new(path: &Path, value: &Value) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value)?;
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    file.write_all(&bytes)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    Ok(())
}

fn str_field<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    value[key]
        .as_str()
        .ok_or_else(|| format!("missing string {key}").into())
}

fn usize_field(value: &Value, key: &str) -> Result<usize> {
    Ok(value[key]
        .as_u64()
        .ok_or_else(|| format!("missing integer {key}"))?
        .try_into()?)
}

fn input_lengths(examples: &[data::Example]) -> BTreeMap<usize, usize> {
    let mut out = BTreeMap::new();
    for example in examples {
        *out.entry(example.tokens.len()).or_default() += 1;
    }
    out
}

pub fn archive(root: &Path) -> Result<()> {
    if root.exists() {
        return Err("context source archive already exists".into());
    }
    fs::create_dir(root)?;
    let mut index = Vec::new();
    for (name, body) in crate::sources::FILES {
        let path = root.join("source").join(name);
        fs::create_dir_all(path.parent().ok_or("missing source parent")?)?;
        if path.exists() {
            if fs::read(&path)? != body.as_bytes() {
                return Err("conflicting duplicate source entry".into());
            }
        } else {
            let mut file = fs::OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(path)?;
            file.write_all(body.as_bytes())?;
        }
        index.push(json!({"name":name,"sha256":hash(body.as_bytes()),"bytes":body.len()}));
    }
    save_new(
        &root.join("source-index.json"),
        &json!({"files":index,
        "source_sha256":runner::source_hash(),"ptx_sha256":runner::ptx_hash(),
        "precision":crate::precision::TAG,"placement":crate::placement::TAG,
        "plan_sha256":hash(include_bytes!("../../LEROPE_YARN_PLAN.txt"))}),
    )?;
    crate::ptx_audit::run(&root.join("ptx"))
}

fn document_length(example: &data::Example) -> usize {
    if example.prompt.is_empty() {
        example.tokens.len()
    } else {
        example.prompt.len() + example.limit.saturating_sub(1)
    }
}

pub fn lengths(output: &Path) -> Result<()> {
    let suite = runner::suite();
    let mut tasks = Vec::new();
    for row in suite["tasks"].as_array().ok_or("missing tasks")? {
        let task = data::build(row)?;
        let train_lengths = input_lengths(&task.splits["train"]);
        let training_extent = *train_lengths
            .last_key_value()
            .ok_or("empty training data")?
            .0;
        let mut splits = serde_json::Map::new();
        for (name, examples) in &task.splits {
            let mut documents = BTreeMap::<usize, usize>::new();
            for example in examples {
                *documents.entry(document_length(example)).or_default() += 1;
            }
            splits.insert(name.clone(), json!({"examples":examples.len(),
                "input_lengths":input_lengths(examples),"document_capacities":documents,
                "extended_examples":examples.iter().filter(|e|document_length(e)>training_extent).count()}));
        }
        tasks.push(
            json!({"task":row["name"],"tier":row["tier"],"primary":task.selection,
            "training_extent":training_extent,"training_input_lengths":train_lengths,
            "frozen_fingerprints_verified":true,"splits":splits}),
        );
    }
    save_new(
        output,
        &json!({"kind":"dataset_length_preflight","tasks":tasks,
        "suite_sha256":hash(include_bytes!("../../../complex_geometry_suite/suite.json")),
        "source_sha256":runner::source_hash(),"ptx_sha256":runner::ptx_hash(),
        "scientific_training":false,"gpu_used":false}),
    )
}

fn state(engine: &Engine) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    for buffer in [
        &engine.parameters,
        &engine.first_moment,
        &engine.second_moment,
    ] {
        out.extend(buffer.read()?.into_iter().flat_map(f32::to_le_bytes));
    }
    Ok(out)
}

fn retain_pass(metrics: &Value) -> Result<bool> {
    let exact = metrics["exact"].as_u64().ok_or("missing exact")?;
    let examples = metrics["examples"].as_u64().ok_or("missing examples")?;
    if examples == 0 || exact > examples {
        return Err("invalid exact count".into());
    }
    Ok(100 * exact >= 99 * examples)
}

fn verify_file(path: &Path, expected: &Value, key: &str) -> Result<Vec<u8>> {
    let bytes = fs::read(path)?;
    if expected[key] != hash(&bytes) {
        return Err(format!("frozen file hash mismatch: {} ({key})", path.display()).into());
    }
    Ok(bytes)
}

pub fn evaluate(registration_path: &Path, output: &Path) -> Result<()> {
    if crate::precision::FP32_AFFINE_REFERENCE {
        return Err(
            "all-affine-FP32 build is diagnostic only, not a scientific evaluation arm".into(),
        );
    }
    if output.exists() {
        return Err("refuse to overwrite or resume checkpoint study".into());
    }
    let registration_bytes = fs::read(registration_path)?;
    let registration: Value = serde_json::from_slice(&registration_bytes)?;
    if registration["source_sha256"] != runner::source_hash()
        || registration["ptx_sha256"] != runner::ptx_hash()
        || registration["binary_sha256"] != hash(&fs::read(std::env::current_exe()?)?)
        || registration["kind"] != "lerope_yarn_frozen_checkpoint_study"
        || registration["training_updates"] != 0
    {
        return Err("checkpoint study executable/protocol mismatch".into());
    }
    let control = &registration["control"];
    let arm = str_field(control, "arm")?;
    if !matches!(arm, "phase-score-lerope" | "covariance-lerope") {
        return Err("unsupported selected control".into());
    }
    let root = Path::new(str_field(control, "root")?);
    let audit_path = Path::new(str_field(control, "audit")?);
    let audit_bytes = verify_file(audit_path, control, "audit_sha256")?;
    let audit: Value = serde_json::from_slice(&audit_bytes)?;
    if audit["audit_passed"] != true || audit["solved_cells"] != 105 {
        return Err("control does not have all105 independently verified passes".into());
    }
    let cells = registration["cells"].as_array().ok_or("missing cells")?;
    if cells.len() != 105 {
        return Err("expected105 registered checkpoint cells".into());
    }
    let suite = runner::suite();
    let mut expected = Vec::new();
    for tier in ["short", "long"] {
        for row in suite["tasks"].as_array().ok_or("missing suite tasks")? {
            if row["tier"] == tier {
                for seed in 0..3 {
                    expected.push((str_field(row, "name")?, seed));
                }
            }
        }
    }
    for (cell, &(name, seed)) in cells.iter().zip(&expected) {
        if cell["task"] != name || cell["seed"] != seed {
            return Err("checkpoint cell order, identity or seed differs from frozen suite".into());
        }
    }
    fs::create_dir(output)?;
    save_new(&output.join("registration.json"), &registration)?;
    let mut rows = Vec::new();
    let mut lost = false;
    for cell in cells {
        let name = str_field(cell, "task")?;
        let seed = usize_field(cell, "seed")?;
        if lost {
            rows.push(json!({"task":name,"seed":seed,"status":"unrun_after_lost_main_pass"}));
            continue;
        }
        let row = suite["tasks"]
            .as_array()
            .unwrap()
            .iter()
            .find(|row| row["name"] == name)
            .ok_or("unknown registered task")?;
        let task = data::build(row)?;
        let train_extent = *input_lengths(&task.splits["train"])
            .last_key_value()
            .unwrap()
            .0;
        if cell["training_extent"] != train_extent {
            return Err("training extent mismatch".into());
        }
        let dir = root.join(arm).join(name).join(format!("seed{seed}"));
        let checkpoint_bytes = verify_file(&dir.join("checkpoint.bin"), cell, "checkpoint_sha256")?;
        let cp: Value = serde_json::from_slice(&verify_file(
            &dir.join("checkpoint.json"),
            cell,
            "checkpoint_metadata_sha256",
        )?)?;
        let saved: Value = serde_json::from_slice(&verify_file(
            &dir.join("result.json"),
            cell,
            "result_sha256",
        )?)?;
        let history = verify_file(&dir.join("history.jsonl"), cell, "history_sha256")?;
        if cp["state_sha256"] != hash(&checkpoint_bytes)
            || cp["source_sha256"] != control["source_sha256"]
            || cp["ptx_sha256"] != control["ptx_sha256"]
            || cp["history_sha256"] != hash(&history)
            || read(&dir.join("execution.json"))?["clean"] != true
            || saved["status"] != "passed"
            || saved["completed"] != cp["completed"]
            || saved["metadata"]["primary"] != task.selection
        {
            return Err("selected checkpoint provenance/state is inconsistent".into());
        }
        let cfg = runner::config(row, &task, arm)?;
        let n = layout::total_count(&cfg);
        if checkpoint_bytes.len() != 12 * n {
            return Err("checkpoint geometry mismatch".into());
        }
        let values: Vec<f32> = checkpoint_bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        layout::validate_state(&cfg, &values[..n], &values[n..2 * n], &values[2 * n..])?;
        let mut engine = Engine::new(cfg, &values[..n])?;
        engine.first_moment.write(&values[n..2 * n])?;
        engine.second_moment.write(&values[2 * n..])?;
        let primary_examples = &task.splits[&task.selection];
        let original = metrics::evaluate(&mut engine, primary_examples, true)?;
        if original != saved["endpoint"]["validation"] {
            save_new(
                &output.join(format!("{name}-seed{seed}-control-mismatch.json")),
                &json!({"expected":saved["endpoint"]["validation"],"actual":original}),
            )?;
            return Err("unchanged selected-main evaluation failed exact replay".into());
        }
        let control_metrics =
            metrics::evaluate_documents(&mut engine, primary_examples, true, None)?;
        if !retain_pass(&control_metrics)? {
            return Err("document grouping loses a control pass".into());
        }
        let candidate_metrics =
            metrics::evaluate_documents(&mut engine, primary_examples, true, Some(train_extent))?;
        let retained = retain_pass(&candidate_metrics)?;
        let mut auxiliary = serde_json::Map::new();
        // The extension candidate's first lost primary pass terminates scoring.
        // Existing longer held-out splits are a separate, non-tuning report.
        if retained {
            for (split, examples) in &task.splits {
                if split == &task.selection
                    || split.starts_with("train")
                    || !examples.iter().any(|e| document_length(e) > train_extent)
                {
                    continue;
                }
                let c = metrics::evaluate_documents(&mut engine, examples, true, None)?;
                let y =
                    metrics::evaluate_documents(&mut engine, examples, true, Some(train_extent))?;
                auxiliary.insert(split.clone(), json!({"control":c,"candidate":y}));
            }
        }
        let unchanged = state(&engine)? == checkpoint_bytes;
        if !unchanged || fs::read(dir.join("checkpoint.bin"))? != checkpoint_bytes {
            return Err("inference changed model/optimizer/checkpoint state".into());
        }
        let result = json!({"task":name,"seed":seed,"tier":row["tier"],
            "status":if retained{"retained_main_pass"}else{"lost_main_pass"},
            "primary":task.selection,"training_extent":train_extent,
            "training_updates":0,"original_first99_epochs":saved["first99_epochs"],
            "new_first99_epochs":null,"original_control_exact_replay":true,
            "original_control":original,"grouped_control":control_metrics,"candidate":candidate_metrics,
            "checkpoint_sha256":hash(&checkpoint_bytes),"state_bitwise_unchanged":unchanged,
            "auxiliary_extrapolation_splits":auxiliary,
            "primary_extended_examples":primary_examples.iter().filter(|e|document_length(e)>train_extent).count()});
        save_new(&output.join(format!("{name}-seed{seed}.json")), &result)?;
        rows.push(result);
        lost = !retained;
    }
    save_new(
        &output.join("result.json"),
        &json!({"kind":"lerope_yarn_frozen_checkpoint_study",
        "registration_sha256":hash(&registration_bytes),"source_sha256":runner::source_hash(),
        "ptx_sha256":runner::ptx_hash(),"expected_cells":105,"rows":rows,
        "training_updates":0,"stopped_on_lost_main_pass":lost,"eligible_epoch_promotion":false,
        "limitations":"Inference-only checkpoint evidence; no fresh training and no new first99 epoch measurement."}),
    )
}
