//! Replay every long endpoint and retain late generalization, censoring and parent lineage.
use crate::{
    Input, Result, Value,
    continuation::{ARMS, PARENT_SOURCE, UPDATES, read},
    fs, hash, integer, json, number, record, restore, source_hash,
};
use std::{collections::BTreeMap, path::Path};

pub fn audit(root: &Path) -> Result<()> {
    let mut cells = Vec::new();
    let mut paired = BTreeMap::new();
    let mut orders = BTreeMap::new();
    for seed in 0..3 {
        for (recipe, optimizer) in ARMS {
            let name = format!("{recipe}-{optimizer}-seed{seed}");
            let dir = root.join(&name);
            if dir.join("error.json").exists() {
                cells.push(read(&dir.join("error.json"))?);
                continue;
            }
            let mut input = Input::load(&root.join("inputs").join(&name))?;
            let completed = restore(&mut input, &dir)?;
            let history = fs::read(dir.join("history.jsonl"))?;
            let observations: Vec<Value> = std::str::from_utf8(&history)?
                .lines()
                .map(serde_json::from_str)
                .collect::<std::result::Result<_, _>>()?;
            if observations.first().ok_or("empty history")?["step"] != 0
                || observations.last().unwrap()["step"] != completed
            {
                return Err(format!("invalid history endpoints: {name}").into());
            }
            for pair in observations.windows(2) {
                let first = integer(&pair[0], "step")?;
                let second = integer(&pair[1], "step")?;
                if second != (first + 19).min(UPDATES)
                    || number(&pair[1], "seconds")? < number(&pair[0], "seconds")?
                {
                    return Err(format!("invalid observation sequence/time: {name}").into());
                }
            }
            let last = observations.last().unwrap();
            for (validation, key) in [(false, "train"), (true, "validation")] {
                let replay = input.engine.evaluate(validation)?;
                if replay["exact"] != last[key]["exact"]
                    || (number(&replay, "loss")? - number(&last[key], "loss")?).abs() > 1e-6
                {
                    return Err(format!("endpoint replay differs: {name}/{key}").into());
                }
            }
            let manifest = &input.manifest;
            if manifest["updates"] != UPDATES || manifest["recipe"] != recipe {
                return Err("long protocol mismatch".into());
            }
            let order = manifest["files"]["order.u32"].clone();
            if let Some(expected) = orders.get(&seed) {
                if expected != &order {
                    return Err("unpaired batch orders".into());
                }
            } else {
                orders.insert(seed, order);
            }
            if recipe.starts_with("complex-") {
                let weights = manifest["files"]["parameters.f32"].clone();
                if let Some(expected) = paired.get(&seed) {
                    if expected != &weights {
                        return Err("unpaired complex initialization".into());
                    }
                } else {
                    paired.insert(seed, weights);
                }
            }
            if recipe == "original" {
                if manifest["normalized"] != false
                    || manifest["ssmax"] != false
                    || manifest["learning_rate"] != 0.001
                    || manifest["phase_optimizer"] != false
                {
                    return Err("original GPT control differs from its recipe".into());
                }
            } else {
                let lineage = read(&dir.join("lineage.json"))?;
                let parent_checkpoint = read(&dir.join("parent-checkpoint.json"))?;
                let count = integer(&lineage, "parent_history_bytes")?;
                if lineage["parent_source_sha256"] != PARENT_SOURCE
                    || parent_checkpoint["source_sha256"] != PARENT_SOURCE
                    || lineage["parent_checkpoint_sha256"] != parent_checkpoint["sha256"]
                    || count > history.len()
                    || hash(&history[..count]) != lineage["parent_history_sha256"]
                    || lineage["completed"] != 4750
                    || lineage["new_input_sha256"] != input.digest
                    || manifest["normalized"] != true
                    || manifest["ssmax"] != true
                    || manifest["learning_rate"] != 0.002
                {
                    return Err(format!("invalid pilot lineage: {name}").into());
                }
            }
            let first99 = observations
                .iter()
                .find(|r| r["validation"]["exact"].as_u64().unwrap() >= 4658);
            let crossing = first99.map(|r| integer(r, "step")).transpose()?;
            if crossing.is_some_and(|step| step != completed) {
                return Err("run continued after its first99 crossing".into());
            }
            let result = read(&dir.join(if dir.join("result.json").exists() {
                "result.json"
            } else {
                "segment.json"
            }))?;
            if result["completed"] != completed || result["first99_updates"] != json!(crossing) {
                return Err("result/history mismatch".into());
            }
            let peak = observations
                .iter()
                .map(|r| r["validation"]["exact"].as_u64().unwrap())
                .max()
                .unwrap();
            let cell = json!({"name":name,"recipe":recipe,"optimizer":optimizer,"seed":seed,"status":result["status"],
            "completed":completed,"epochs":input.boundaries[completed] as f64/4704.0,"first99_updates":crossing,
            "first99_epochs":crossing.map(|step|input.boundaries[step] as f64/4704.0),
            "train":last["train"],"validation":last["validation"],"peak_heldout_accuracy":peak as f64/4705.0,
            "new_segment_seconds":result["segment_seconds"],"prior_seconds":result["prior_seconds"],"total_seconds":result["total_seconds"],
            "continued_from_updates":if recipe=="original"{0}else{4750},"independent_parameters":manifest["independent_parameters"],
            "checkpoint_sha256":hash(&fs::read(dir.join("checkpoint.bin"))?),"history_sha256":hash(&history),"replayed":true});
            println!("{cell}");
            cells.push(cell);
        }
    }
    record(
        &root.join("summary.json"),
        &json!({"source_sha256":source_hash(),"expected_cells":18,"cells":cells,
        "scope":"All pilot arms continued with unchanged arithmetic; fresh original-GPT LR.001 controls; fixed rate, task and native RNG; no historical wall-time ratio"}),
    )
}
