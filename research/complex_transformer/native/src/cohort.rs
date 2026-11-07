//! A bounded, predeclared native pilot; every arm/seed is retained and audited.
use crate::{Input, Result, fs, hash, integer, json, prepare, record, source_hash, train};
use std::{path::Path, time::Instant};

const ARMS: [(&str, &str); 5] = [
    ("real", "adamw"),
    ("complex-diagonal", "adamw"),
    ("complex-product", "adamw"),
    ("complex-diagonal", "phase"),
    ("complex-product", "phase"),
];
const UPDATES: usize = 4750;

pub fn run(root: &Path) -> Result<()> {
    fs::create_dir(root)?;
    fs::create_dir(root.join("inputs"))?;
    let protocol = json!({"source_sha256":source_hash(),"seeds":[0,1,2],"arms":ARMS,"updates":UPDATES,"epochs":250,"seconds_per_cell":120,"cohort_seconds":1800,"flush_epochs":500,"selection":"first heldout >=99% or cap","status":"exploratory pilot, no eventual-convergence conclusion"});
    record(&root.join("protocol.json"), &protocol)?;
    let start = Instant::now();
    for seed in 0..3usize {
        for shift in 0..5 {
            let (architecture, optimizer) = ARMS[(seed + shift) % 5];
            let name = format!("{architecture}-{optimizer}-seed{seed}");
            let out = root.join(&name);
            fs::create_dir(&out)?;
            if start.elapsed().as_secs_f64() >= 1800. {
                record(
                    &out.join("error.json"),
                    &json!({"status":"not_started_cohort_time_cap","name":name}),
                )?;
                continue;
            }
            let input = root.join("inputs").join(&name);
            let begin = Instant::now();
            let result = (|| -> Result<()> {
                prepare::write(&input, architecture, optimizer, seed as u32, UPDATES)?;
                let mut bundle = Input::load(&input)?;
                let setup = begin.elapsed().as_secs_f64();
                record(
                    &out.join("setup.json"),
                    &json!({"seconds":setup,"source_sha256":source_hash()}),
                )?;
                if setup > 300. {
                    return Err("setup exceeded 300-second allowance".into());
                }
                train(&mut bundle, &out, UPDATES, false)
            })();
            if let Err(error) = result {
                record(
                    &out.join("error.json"),
                    &json!({"status":"error","name":name,"message":error.to_string()}),
                )?;
            }
        }
    }
    audit(root)
}

pub fn audit(root: &Path) -> Result<()> {
    let mut summary = Vec::new();
    let mut control_inputs = std::collections::BTreeMap::new();
    for seed in 0..3 {
        for (architecture, optimizer) in ARMS {
            let name = format!("{architecture}-{optimizer}-seed{seed}");
            let dir = root.join(&name);
            if dir.join("error.json").exists() {
                summary.push(serde_json::from_slice::<serde_json::Value>(&fs::read(
                    dir.join("error.json"),
                )?)?);
                continue;
            }
            let mut input = Input::load(&root.join("inputs").join(&name))?;
            let completed = crate::restore(&mut input, &dir)?;
            let history = fs::read_to_string(dir.join("history.jsonl"))?;
            let observations: Vec<serde_json::Value> = history
                .lines()
                .map(serde_json::from_str)
                .collect::<std::result::Result<_, _>>()?;
            if observations.first().ok_or("empty history")?["step"] != 0
                || observations.last().unwrap()["step"] != completed
                || observations
                    .windows(2)
                    .any(|w| w[1]["step"].as_u64() != w[0]["step"].as_u64().map(|s| s + 19))
            {
                return Err(format!("invalid observation sequence: {name}").into());
            }
            let final_row = observations.last().unwrap();
            for (is_validation, key) in [(false, "train"), (true, "validation")] {
                let measured = input.engine.evaluate(is_validation)?;
                if measured["exact"] != final_row[key]["exact"]
                    || (measured["loss"].as_f64().unwrap()
                        - final_row[key]["loss"].as_f64().unwrap())
                    .abs()
                        > 1e-6
                {
                    return Err(format!("endpoint replay mismatch: {name}/{key}").into());
                }
            }
            let initial = input.manifest["files"]["parameters.f32"]["sha256"].clone();
            if architecture != "real" {
                if let Some(expected) = control_inputs.get(&seed) {
                    if expected != &initial {
                        return Err("unpaired complex initialization".into());
                    }
                } else {
                    control_inputs.insert(seed, initial);
                }
            }
            let first99 = observations
                .iter()
                .find(|row| row["validation"]["exact"].as_u64().unwrap() >= 4658)
                .map(|row| row["step"].clone());
            let result_path = if dir.join("result.json").exists() {
                dir.join("result.json")
            } else {
                dir.join("segment.json")
            };
            let result: serde_json::Value = serde_json::from_slice(&fs::read(&result_path)?)?;
            if result["completed"] != completed || result["first99_updates"] != json!(first99) {
                return Err("result/history disagreement".into());
            }
            let item = json!({"name":name,"seed":seed,"architecture":architecture,"optimizer":optimizer,
                "completed":completed,"epochs":completed as f64/19.0,"status":result["status"],"first99_updates":first99,
                "train":final_row["train"],"validation":final_row["validation"],"seconds":result["segment_seconds"],
                "checkpoint_sha256":hash(&fs::read(dir.join("checkpoint.bin"))?),"history_sha256":hash(history.as_bytes()),
                "independent_parameters":integer(&input.manifest,"independent_parameters")?,"replayed":true});
            println!("{item}");
            summary.push(item);
        }
    }
    record(
        &root.join("summary.json"),
        &json!({"cells":summary,"expected_cells":15,"source_sha256":source_hash(),"scope":"short native pilot; no architecture promotion or historical wall-time comparison"}),
    )
}
