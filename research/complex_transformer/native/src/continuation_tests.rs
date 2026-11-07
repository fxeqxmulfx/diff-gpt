//! Horizon extension must preserve issued samples, optimizer state and absolute warmup.
use crate::{Input, Result, checkpoint, continuation, fs, json, record, restore, source_hash};

#[test]
fn pilot_math_is_pinned_and_changed_recipes_or_samples_are_rejected() -> Result<()> {
    continuation::check_parent_math()?;
    let root = std::env::temp_dir().join(format!("complex-long-contract-{}", std::process::id()));
    fs::create_dir(&root)?;
    let old = root.join("old");
    let new = root.join("new");
    crate::prepare::write(&old, "complex-product", "adamw", 0, 19)?;
    continuation::prepare_long(&new, "complex-product", "adamw", 0, 57, 0.1)?;
    continuation::compatible(&old, &new)?;
    let path = new.join("manifest.json");
    let manifest = continuation::read(&path)?;
    let mut changed = manifest.clone();
    changed["learning_rate"] = json!(0.001);
    record(&path, &changed)?;
    assert!(continuation::compatible(&old, &new).is_err());
    record(&path, &manifest)?;
    let path = new.join("order.u32");
    let mut bytes = fs::read(&path)?;
    bytes[0] ^= 1;
    fs::write(&path, bytes)?;
    assert!(continuation::compatible(&old, &new).is_err());
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn extended_horizon_matches_uninterrupted_weights_moments_and_warmup() -> Result<()> {
    let root = std::env::temp_dir().join(format!("complex-long-state-{}", std::process::id()));
    fs::create_dir(&root)?;
    for optimizer in ["adamw", "phase"] {
        let old = root.join(format!("old-{optimizer}"));
        let new = root.join(format!("new-{optimizer}"));
        let parent = root.join(format!("parent-{optimizer}"));
        let child = root.join(format!("child-{optimizer}"));
        fs::create_dir(&parent)?;
        fs::create_dir(&child)?;
        crate::prepare::write(&old, "complex-product", optimizer, 1, 19)?;
        continuation::prepare_long(&new, "complex-product", optimizer, 1, 57, 0.1)?;
        let original = Input::load(&old)?;
        let uninterrupted = Input::load(&new)?;
        for step in 0..19 {
            original.step(step)?;
            original.engine.update(step)?;
            uninterrupted.step(step)?;
            uninterrupted.engine.update(step)?;
        }
        fs::write(parent.join("history.jsonl"), b"{\"step\":19}\n")?;
        checkpoint(&original, &parent, 19)?;
        record(
            &parent.join("execution.json"),
            &json!({"completed":19,"possible_end":19}),
        )?;
        let parent_bytes = fs::read(parent.join("checkpoint.bin"))?;
        let parent_metadata = fs::read(parent.join("checkpoint.json"))?;
        let parent_history = fs::read(parent.join("history.jsonl"))?;
        assert_eq!(
            continuation::transfer(&old, &parent, &new, &child, &source_hash())?,
            19
        );
        assert_eq!(parent_bytes, fs::read(child.join("checkpoint.bin"))?);
        assert_eq!(parent_bytes, fs::read(parent.join("checkpoint.bin"))?);
        assert_eq!(parent_metadata, fs::read(parent.join("checkpoint.json"))?);
        assert_eq!(parent_history, fs::read(parent.join("history.jsonl"))?);
        let mut resumed = Input::load(&new)?;
        assert_eq!(restore(&mut resumed, &child)?, 19);
        for step in 19..57 {
            assert_eq!(resumed.step(step)?, if step % 19 == 18 { 96 } else { 256 });
            uninterrupted.step(step)?;
            resumed.engine.update(step)?;
            uninterrupted.engine.update(step)?;
        }
        for (left, right) in [
            (&resumed.engine.parameters, &uninterrupted.engine.parameters),
            (
                &resumed.engine.first_moment,
                &uninterrupted.engine.first_moment,
            ),
            (
                &resumed.engine.second_moment,
                &uninterrupted.engine.second_moment,
            ),
        ] {
            assert_eq!(left.read()?, right.read()?);
        }
    }
    fs::remove_dir_all(root)?;
    Ok(())
}
