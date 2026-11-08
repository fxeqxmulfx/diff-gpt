//! Independent checkpoint-study audit. No model, CUDA, or training dependency.
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs, io::Write, path::Path};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
fn read(path: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}
fn string<'a>(v: &'a Value, key: &str) -> Result<&'a str> {
    v[key]
        .as_str()
        .ok_or_else(|| format!("missing {key}").into())
}
fn number(v: &Value, key: &str) -> Result<u64> {
    v[key]
        .as_u64()
        .ok_or_else(|| format!("missing {key}").into())
}
fn require(check: bool, message: &str) -> Result<()> {
    if check { Ok(()) } else { Err(message.into()) }
}
fn pinned(path: &Path, expected: &Value) -> Result<Vec<u8>> {
    let bytes = fs::read(path)?;
    require(
        *expected == hash(&bytes),
        &format!("file hash mismatch:{}", path.display()),
    )?;
    Ok(bytes)
}
fn pass(metrics: &Value) -> Result<bool> {
    let n = number(metrics, "examples")?;
    let hits = number(metrics, "exact")?;
    require(n > 0 && hits <= n, "invalid exact/example count")?;
    let accuracy = metrics["sequence_accuracy"]
        .as_f64()
        .ok_or("missing accuracy")?;
    require(
        accuracy.is_finite() && (accuracy - hits as f64 / n as f64).abs() < 2e-15,
        "accuracy/count mismatch",
    )?;
    require(
        metrics["loss"].as_f64().is_some_and(f64::is_finite),
        "nonfinite loss",
    )?;
    Ok(100 * hits >= 99 * n)
}
fn audit(reg_path: &Path, result_root: &Path) -> Result<Value> {
    let reg_bytes = fs::read(reg_path)?;
    let reg: Value = serde_json::from_slice(&reg_bytes)?;
    require(
        reg["kind"] == "lerope_yarn_frozen_checkpoint_study" && reg["training_updates"] == 0,
        "wrong study kind",
    )?;
    pinned(Path::new(string(&reg, "binary")?), &reg["binary_sha256"])?;
    let validation = pinned(
        Path::new(string(&reg, "validation")?),
        &reg["validation_sha256"],
    )?;
    let validation: Value = serde_json::from_slice(&validation)?;
    require(
        validation["implementation_passed"] == true,
        "validation not passed",
    )?;
    require(
        validation["source_sha256"] == reg["source_sha256"]
            && validation["ptx_sha256"] == reg["ptx_sha256"],
        "validation build mismatch",
    )?;
    let archive = Path::new(string(&reg, "source_archive")?);
    let index = read(&archive.join("source-index.json"))?;
    let mut captured = Vec::new();
    for row in index["files"].as_array().ok_or("missing source list")? {
        let name = string(row, "name")?;
        let bytes = pinned(&archive.join("source").join(name), &row["sha256"])?;
        require(
            bytes.len() as u64 == number(row, "bytes")?,
            "source length mismatch",
        )?;
        captured.extend((name.len() as u64).to_le_bytes());
        captured.extend(name.as_bytes());
        captured.extend((bytes.len() as u64).to_le_bytes());
        captured.extend(&bytes);
    }
    require(
        reg["source_sha256"] == hash(&captured),
        "compiled source digest mismatch",
    )?;
    let ptx = read(&archive.join("ptx/report.json"))?;
    require(
        ptx["passed"] == true && ptx["ptx_sha256"] == reg["ptx_sha256"],
        "PTX audit mismatch",
    )?;
    for module in ptx["modules"].as_array().ok_or("missing PTX modules")? {
        let path = archive
            .join("ptx")
            .join(format!("{}.ptx", string(module, "module")?));
        let bytes = pinned(&path, &module["sha256"])?;
        require(
            !String::from_utf8(bytes)?.contains(".f64"),
            "GPU FP64 found",
        )?;
    }
    let control = &reg["control"];
    let control_audit: Value = serde_json::from_slice(&pinned(
        Path::new(string(control, "audit")?),
        &control["audit_sha256"],
    )?)?;
    require(
        control_audit["audit_passed"] == true && control_audit["solved_cells"] == 105,
        "control coverage not verified",
    )?;
    let lengths: Value = serde_json::from_slice(&pinned(
        Path::new(string(&reg, "lengths")?),
        &reg["lengths_sha256"],
    )?)?;
    let suite = read(Path::new(string(&reg, "suite")?))?;
    pinned(Path::new(string(&reg, "suite")?), &lengths["suite_sha256"])?;
    let cells = reg["cells"].as_array().ok_or("missing cells")?;
    require(cells.len() == 105, "wrong number of cells")?;
    let mut expected = Vec::new();
    for tier in ["short", "long"] {
        for task in suite["tasks"].as_array().ok_or("missing suite tasks")? {
            if task["tier"] == tier {
                for seed in 0..3 {
                    expected.push((string(task, "name")?, seed, tier));
                }
            }
        }
    }
    let mut stop = false;
    let mut counts = [0usize; 2];
    let mut unrun = 0;
    let mut lost = Vec::new();
    let mut audited = Vec::new();
    let result_path = result_root.join("result.json");
    let terminal = if result_path.exists() {
        Some(read(&result_path)?)
    } else {
        None
    };
    if let Some(result) = &terminal {
        require(
            result["registration_sha256"] == hash(&reg_bytes)
                && result["source_sha256"] == reg["source_sha256"]
                && result["ptx_sha256"] == reg["ptx_sha256"],
            "result provenance mismatch",
        )?;
        require(
            result["training_updates"] == 0 && result["eligible_epoch_promotion"] == false,
            "inference result claims training/promotion",
        )?;
        require(
            result["rows"].as_array().is_some_and(|r| r.len() == 105),
            "wrong terminal row count",
        )?;
    }
    for (at, (cell, (name, seed, tier))) in cells.iter().zip(expected).enumerate() {
        require(
            cell["task"] == name && cell["seed"] == seed && cell["tier"] == tier,
            "cell order/seed/tier mismatch",
        )?;
        let length = lengths["tasks"]
            .as_array()
            .ok_or("missing length tasks")?
            .iter()
            .find(|r| r["task"] == name)
            .ok_or("missing task length")?;
        require(
            cell["training_extent"] == length["training_extent"]
                && length["frozen_fingerprints_verified"] == true,
            "training extent/fingerprint mismatch",
        )?;
        let dir = Path::new(string(control, "root")?)
            .join(string(control, "arm")?)
            .join(name)
            .join(format!("seed{seed}"));
        let checkpoint = pinned(&dir.join("checkpoint.bin"), &cell["checkpoint_sha256"])?;
        let cp: Value = serde_json::from_slice(&pinned(
            &dir.join("checkpoint.json"),
            &cell["checkpoint_metadata_sha256"],
        )?)?;
        let saved: Value =
            serde_json::from_slice(&pinned(&dir.join("result.json"), &cell["result_sha256"])?)?;
        let parent_matches: Vec<_> = control_audit["rows"]
            .as_array()
            .ok_or("missing control audit rows")?
            .iter()
            .filter(|r| r["case"] == name && r["seed"] == seed && r["arm"] == control["arm"])
            .collect();
        require(
            parent_matches.len() == 1,
            "ambiguous/missing independently audited control cell",
        )?;
        let parent = parent_matches[0];
        require(
            parent["audit"] == "passed"
                && parent["status"] == "passed"
                && parent["checkpoint_sha256"] == cell["checkpoint_sha256"]
                && parent["first99_epochs"] == saved["first99_epochs"],
            "checkpoint differs from independently audited control",
        )?;
        require(
            saved["case"] == name && saved["seed"] == seed,
            "saved control identity differs from task/seed",
        )?;
        pinned(&dir.join("history.jsonl"), &cell["history_sha256"])?;
        require(
            cp["state_sha256"] == hash(&checkpoint)
                && cp["source_sha256"] == control["source_sha256"]
                && cp["ptx_sha256"] == control["ptx_sha256"],
            "checkpoint provenance mismatch",
        )?;
        require(
            read(&dir.join("execution.json"))?["clean"] == true
                && saved["status"] == "passed"
                && pass(&saved["endpoint"]["validation"])?,
            "control checkpoint is not a clean pass",
        )?;
        let output = result_root.join(format!("{name}-seed{seed}.json"));
        if !output.exists() {
            require(terminal.is_none() || stop, "missing required terminal cell")?;
            if let Some(result) = &terminal {
                require(
                    result["rows"][at]
                        == json!({"task":name,"seed":seed,"status":"unrun_after_lost_main_pass"}),
                    "unrun cell incorrectly represented",
                )?;
            }
            unrun += 1;
            continue;
        }
        require(!stop, "scored another primary cell after a lost pass")?;
        let row = read(&output)?;
        if let Some(result) = &terminal {
            require(
                result["rows"][at] == row,
                "terminal row differs from durable cell",
            )?;
        }
        require(
            row["task"] == name && row["seed"] == seed && row["tier"] == tier,
            "result identity mismatch",
        )?;
        require(
            row["training_updates"] == 0 && row["new_first99_epochs"].is_null(),
            "inference result fabricates new epochs",
        )?;
        require(
            row["original_first99_epochs"] == saved["first99_epochs"],
            "inherited epochs mislabeled",
        )?;
        require(
            row["primary"] == length["primary"]
                && row["training_extent"] == length["training_extent"],
            "wrong primary/reference extent",
        )?;
        require(
            row["original_control_exact_replay"] == true
                && row["original_control"] == saved["endpoint"]["validation"],
            "original control did not replay",
        )?;
        require(
            row["state_bitwise_unchanged"] == true
                && row["checkpoint_sha256"] == cell["checkpoint_sha256"],
            "state change in inference",
        )?;
        require(
            pass(&row["grouped_control"])?,
            "grouped control loses a pass",
        )?;
        let retained = pass(&row["candidate"])?;
        require(
            row["status"]
                == if retained {
                    "retained_main_pass"
                } else {
                    "lost_main_pass"
                },
            "wrong pass classification",
        )?;
        if retained {
            counts[usize::from(tier == "long")] += 1;
        } else {
            stop = true;
            lost.push(json!({"task":name,"seed":seed}));
        }
        audited.push(json!({"task":name,"seed":seed,"tier":tier,"status":row["status"],
            "control_accuracy":row["grouped_control"]["sequence_accuracy"],"candidate_accuracy":row["candidate"]["sequence_accuracy"],
            "training_extent":row["training_extent"],"primary_extended_examples":row["primary_extended_examples"]}));
    }
    if let Some(result) = &terminal {
        require(
            result["stopped_on_lost_main_pass"] == stop,
            "wrong stop status",
        )?;
    }
    Ok(
        json!({"audit_passed":true,"source_verified":true,"phase":if terminal.is_none(){"prepared_or_partial"}else if stop{"rejected_lost_main_pass"}else{"complete"},
        "short_retained":counts[0],"long_retained":counts[1],"unrun_cells":unrun,"lost_main_passes":lost,"rows":audited,
        "training_updates":0,"epoch_promotion_eligible":false,"registration_sha256":hash(&reg_bytes)}),
    )
}
fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: yarn-audit REGISTRATION RESULT_ROOT FRESH_OUTPUT".into());
    }
    let result = audit(Path::new(&args[1]), Path::new(&args[2]));
    let mut file = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&args[3])?;
    match result {
        Ok(value) => {
            serde_json::to_writer_pretty(&mut file, &value)?;
            file.write_all(b"\n")?;
            Ok(())
        }
        Err(e) => {
            serde_json::to_writer_pretty(
                &mut file,
                &json!({"audit_passed":false,"error":e.to_string()}),
            )?;
            Err(e)
        }
    }
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
    fn exact_count_threshold_and_metric_integrity() {
        let m =
            |h, n| json!({"exact":h,"examples":n,"sequence_accuracy":h as f64/n as f64,"loss":0.1});
        assert!(pass(&m(99, 100)).unwrap());
        assert!(!pass(&m(98, 100)).unwrap());
        assert!(!pass(&m(98, 99)).unwrap());
        assert!(pass(&m(99, 99)).unwrap());
        assert!(pass(&m(101, 100)).is_err());
        assert!(pass(&m(0, 0)).is_err());
        let mut wrong = m(98, 100);
        wrong["sequence_accuracy"] = json!(0.99);
        assert!(pass(&wrong).is_err());
    }
    #[test]
    fn sha256_known_vector() {
        assert_eq!(
            hash(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
