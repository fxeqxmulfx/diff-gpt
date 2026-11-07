//! Explicit continuation of the frozen 250-epoch pilot; no optimizer or sample reset.
use crate::{
    Input, Result, Value, checkpoint, fs, hash, integer, json, number, prepare, ptx_hash, record,
    restore_from_source, source_hash, train,
};
use std::{path::Path, time::Instant};

pub const PARENT_SOURCE: &str = "ba65c51de909bb54cc7544317717c4bb57dadc66fb4c448a1c64355a28f9e29b";
pub const UPDATES: usize = 100_000;
pub const SECONDS: f64 = 300.0;
pub const ARMS: [(&str, &str); 6] = [
    ("original", "adamw"),
    ("complex-product", "adamw"),
    ("real", "adamw"),
    ("complex-diagonal", "adamw"),
    ("complex-product", "phase"),
    ("complex-diagonal", "phase"),
];

pub fn check_parent_math() -> Result<()> {
    for (bytes, expected) in [
        (
            include_bytes!("engine.rs").as_slice(),
            "7e28ba6e4d398604db1991e0ed9836e2d8f6dde6699a9e90d358f639cdd1d37c",
        ),
        (
            include_bytes!("layout.rs").as_slice(),
            "a9f7f2281291cf212fe22dbf37cc3e2e1f26d40cbb60023eeca1927cacf939a6",
        ),
        (
            include_bytes!("random.rs").as_slice(),
            "64ff394f87a0421b81fbaa32b8d2f5eaffe5d75ef201cb73e782110470a088d7",
        ),
        (
            include_bytes!("prepare.rs").as_slice(),
            "1351c4462bddfbe113b9c56ad02c4a346e3549226afc8022789d137ca8732048",
        ),
    ] {
        if hash(bytes) != expected {
            return Err("pilot host math or sampler changed".into());
        }
    }
    if ptx_hash() != "6015dddc9ba08752de33a954e052e78ddef0750e594b5720fee75a513438ef6b" {
        return Err("pilot GPU arithmetic changed".into());
    }
    Ok(())
}

pub fn read(path: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

pub fn prepare_long(
    dir: &Path,
    recipe: &str,
    optimizer: &str,
    seed: u32,
    updates: usize,
    prior_seconds: f64,
) -> Result<()> {
    let architecture = if recipe == "original" { "real" } else { recipe };
    prepare::write(dir, architecture, optimizer, seed, updates)?;
    let mut manifest = read(&dir.join("manifest.json"))?;
    manifest["recipe"] = json!(recipe);
    manifest["prior_seconds"] = json!(prior_seconds);
    manifest["seconds_cap"] = json!(SECONDS - prior_seconds);
    manifest["total_seconds_cap"] = json!(SECONDS);
    manifest["scope"] = json!(
        "User-requested continuation to 100000 absolute updates with fresh original-GPT controls"
    );
    if recipe == "original" {
        manifest["normalized"] = json!(false);
        manifest["ssmax"] = json!(false);
        manifest["attention_gain"] = json!(((0.5 * 32.0f64.ln()) as f32).exp() / 32.0);
        manifest["learning_rate"] = json!(0.001);
    }
    record(&dir.join("manifest.json"), &manifest)
}

/// Check compatibility before transferring any state, including the entire issued batch prefix.
pub fn compatible(old_dir: &Path, new_dir: &Path) -> Result<()> {
    let old = read(&old_dir.join("manifest.json"))?;
    let new = read(&new_dir.join("manifest.json"))?;
    for key in [
        "architecture",
        "phase_optimizer",
        "seed",
        "width",
        "layers",
        "heads",
        "vocab",
        "batch",
        "observe",
        "target",
        "normalized",
        "ssmax",
        "attention_gain",
        "learning_rate",
        "weight_decay",
        "warmup",
        "train_examples",
        "validation_examples",
        "parameter_count",
        "independent_parameters",
        "parameters",
        "initialization",
        "sampler",
        "split",
        "task",
        "flush_epochs",
    ] {
        if old[key] != new[key] {
            return Err(format!("continuation changes {key}").into());
        }
    }
    if integer(&new, "updates")? <= integer(&old, "updates")? {
        return Err("continuation does not extend the horizon".into());
    }
    for name in ["parameters.f32", "rope.f32", "train.u32", "validation.u32"] {
        if old["files"][name] != new["files"][name]
            || fs::read(old_dir.join(name))? != fs::read(new_dir.join(name))?
        {
            return Err(format!("continuation changes {name}").into());
        }
    }
    for name in ["order.u32", "boundaries.u32"] {
        if !fs::read(new_dir.join(name))?.starts_with(&fs::read(old_dir.join(name))?) {
            return Err(format!("continuation changes the issued {name} prefix").into());
        }
    }
    Ok(())
}

/// The caller supplies either the checked pilot pin or the current source in a regression test.
pub fn transfer(
    old_input: &Path,
    old_run: &Path,
    new_input: &Path,
    new_run: &Path,
    expected_source: &str,
) -> Result<usize> {
    compatible(old_input, new_input)?;
    let mut parent = Input::load(old_input)?;
    let completed = restore_from_source(&mut parent, old_run, expected_source)?;
    let parent_history = fs::read(old_run.join("history.jsonl"))?;
    let last: Value = serde_json::from_str(
        std::str::from_utf8(&parent_history)?
            .lines()
            .last()
            .ok_or("empty parent history")?,
    )?;
    if integer(&last, "step")? != completed {
        return Err("parent history does not end at its checkpoint".into());
    }
    let mut child = Input::load(new_input)?;
    child
        .engine
        .parameters
        .write(&parent.engine.parameters.read()?)?;
    child
        .engine
        .first_moment
        .write(&parent.engine.first_moment.read()?)?;
    child
        .engine
        .second_moment
        .write(&parent.engine.second_moment.read()?)?;
    fs::write(new_run.join("history.jsonl"), &parent_history)?;
    checkpoint(&child, new_run, completed)?;
    if fs::read(old_run.join("checkpoint.bin"))? != fs::read(new_run.join("checkpoint.bin"))? {
        return Err("continuation altered parameter, moment or update bytes".into());
    }
    record(
        &new_run.join("execution.json"),
        &json!({"version":2,"completed":completed,"possible_end":completed,"input_sha256":child.digest,"source_sha256":source_hash()}),
    )?;
    for name in ["checkpoint.json", "execution.json", "result.json"] {
        if old_run.join(name).exists() {
            fs::copy(old_run.join(name), new_run.join(format!("parent-{name}")))?;
        }
    }
    fs::copy(
        old_input.join("manifest.json"),
        new_run.join("parent-manifest.json"),
    )?;
    record(
        &new_run.join("lineage.json"),
        &json!({
            "parent_run":old_run,"parent_input":old_input,"parent_source_sha256":expected_source,
            "parent_input_sha256":parent.digest,"parent_checkpoint_sha256":hash(&fs::read(old_run.join("checkpoint.bin"))?),
            "parent_history_sha256":hash(&parent_history),"parent_history_bytes":parent_history.len(),
            "completed":completed,"new_input_sha256":child.digest,"source_sha256":source_hash(),
            "ptx_sha256":ptx_hash(),"weights_moments_counter_byte_identical":true,"issued_batch_prefix_identical":true
        }),
    )?;
    Ok(completed)
}

pub fn run(parent_root: &Path, root: &Path) -> Result<()> {
    check_parent_math()?;
    let pilot = read(&parent_root.join("protocol.json"))?;
    if pilot["source_sha256"] != PARENT_SOURCE || pilot["updates"] != 4750 {
        return Err("not the frozen pilot".into());
    }
    fs::create_dir(root)?;
    fs::create_dir(root.join("inputs"))?;
    record(
        &root.join("protocol.json"),
        &json!({
            "source_sha256":source_hash(),"parent_source_sha256":PARENT_SOURCE,"parent_root":parent_root,
            "updates":UPDATES,"approximate_epochs":5263.163265306122,"arms":ARMS,"seeds":[0,1,2],
            "cells":18,"continued_cells":15,"fresh_original_controls":3,"flush_epochs":500,
            "total_seconds_per_cell":SECONDS,"cohort_seconds":5400,"setup_seconds_cap":300,
            "stop":"first >=4658/4705 validation correct, absolute update cap, or time cap",
            "old_torch_reference_epochs":[3531,3197,4315],"old_torch_reference_lr":0.001,
            "historical_runtime_comparison":false,"retune":false,"all_failures_retained":true
        }),
    )?;
    let start = Instant::now();
    let mut setup_seconds = 0.;
    for seed in 0..3usize {
        for shift in 0..ARMS.len() {
            let (recipe, optimizer) = ARMS[(seed + shift) % ARMS.len()];
            let name = format!("{recipe}-{optimizer}-seed{seed}");
            let out = root.join(&name);
            fs::create_dir(&out)?;
            if start.elapsed().as_secs_f64() >= 5400. || setup_seconds >= 300. {
                record(
                    &out.join("error.json"),
                    &json!({"name":name,"status":"not_started_cohort_or_setup_time_cap"}),
                )?;
                continue;
            }
            let begin = Instant::now();
            let input = root.join("inputs").join(&name);
            let result = (|| -> Result<()> {
                let original = recipe == "original";
                let old_run = parent_root.join(&name);
                let prior = if original {
                    0.
                } else {
                    number(&read(&old_run.join("result.json"))?, "segment_seconds")?
                };
                prepare_long(&input, recipe, optimizer, seed as u32, UPDATES, prior)?;
                if !original {
                    let completed = transfer(
                        &parent_root.join("inputs").join(&name),
                        &old_run,
                        &input,
                        &out,
                        PARENT_SOURCE,
                    )?;
                    if completed != 4750 {
                        return Err("unexpected pilot checkpoint position".into());
                    }
                }
                let mut bundle = Input::load(&input)?;
                let setup = begin.elapsed().as_secs_f64();
                setup_seconds += setup;
                record(
                    &out.join("setup.json"),
                    &json!({"seconds":setup,"source_sha256":source_hash()}),
                )?;
                train(&mut bundle, &out, UPDATES, !original)
            })();
            if let Err(error) = result {
                record(
                    &out.join("error.json"),
                    &json!({"name":name,"status":"error","message":error.to_string()}),
                )?;
            }
        }
    }
    audit(root)
}

pub fn audit(root: &Path) -> Result<()> {
    crate::long_audit::audit(root)
}
