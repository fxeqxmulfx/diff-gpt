//! Integration checks use the public production API, never its private engine.
#[path = "../../../complex_geometry_suite/native/src/arithmetic_data.rs"]
#[expect(
    dead_code,
    reason = "Frozen arithmetic oracle retains an unrelated fixed-RoPE helper"
)]
mod arithmetic_data;
use diff_gpt_kernite::{
    Checkpoint, DiffGpt, EpochSampler, IGNORE_TARGET, Model, ModelConfig, Positions, Result,
    codec::{Difference, Domain, Matrix, Tokenizer},
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs, io::Write, path::Path};
fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
fn read(path: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}
fn save(path: &Path, value: &Value) -> Result<()> {
    let mut f = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)?;
    serde_json::to_writer_pretty(&mut f, value)?;
    f.write_all(b"\n")?;
    Ok(())
}
fn replay(repo: &Path, output: &Path) -> Result<Value> {
    let audited =
        fs::read(repo.join("research/main_model/results/phase-score-suite-01/final-audit.json"))?;
    assert_eq!(
        hash(&audited),
        "a9d297f7bea99ab9e294ab80e7a3c022a8476432278553b744a260954aba26e7"
    );
    let suite = read(&repo.join("research/complex_geometry_suite/suite.json"))?;
    let row = suite["tasks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["name"] == "arithmetic-sum-pow4")
        .unwrap();
    let data = arithmetic_data::arithmetic(row)?;
    assert_eq!(data.vocab, 97);
    let dir =
        repo.join("runs/main-model-phase-score-01/phase-score-lerope/arithmetic-sum-pow4/seed1");
    let meta = read(&dir.join("checkpoint.json"))?;
    let raw = fs::read(dir.join("checkpoint.bin"))?;
    assert_eq!(
        hash(&raw),
        "9244865af260605da7c0e32c1eb48d3a44d33c37c41cacbdeafa94a79b032612"
    );
    assert_eq!(meta["state_sha256"], hash(&raw));
    let completed = meta["completed"].as_u64().unwrap() as usize;
    let examples = meta["examples"].as_u64().unwrap();
    let mut config = ModelConfig::main(97, 256, 2, 100000);
    config.positions = Positions::LeRoPE;
    config.training_input_length = Some(2);
    let cp = Checkpoint::from_raw(config, completed, examples, raw.clone())?;
    let mut model = Model::from_checkpoint(&cp)?;
    assert_eq!(model.checkpoint()?.raw_state(), raw);
    let before = model.checkpoint()?.state_sha256();
    let mut hits = 0usize;
    for rows in data.validation.chunks(128) {
        let x: Vec<_> = rows.iter().flat_map(|r| [r[0], r[1]]).collect();
        let y: Vec<_> = rows.iter().flat_map(|r| [IGNORE_TARGET, r[2]]).collect();
        let e = model.evaluate(&x, &y, rows.len(), 2)?;
        hits += e.correct;
        assert_eq!(e.targets, rows.len());
        assert!(e.mean_loss.is_some_and(|v| v.is_finite()));
    }
    let saved = read(&dir.join("result.json"))?;
    assert_eq!(
        hits as u64,
        saved["endpoint"]["validation"]["exact"].as_u64().unwrap()
    );
    assert_eq!(before, model.checkpoint()?.state_sha256());
    let mut sampler = EpochSampler::new(1)?;
    let mut at = 0usize;
    let mut seen = 0u64;
    let mut restored = None;
    while at < completed + 100 {
        for ids in sampler.next_epoch(data.train.len())?.chunks(256) {
            if at == completed {
                assert_eq!(seen, examples);
            }
            if at >= completed && at < completed + 100 {
                let x: Vec<_> = ids
                    .iter()
                    .flat_map(|&i| [data.train[i][0], data.train[i][1]])
                    .collect();
                let y: Vec<_> = ids
                    .iter()
                    .flat_map(|&i| [IGNORE_TARGET, data.train[i][2]])
                    .collect();
                model.train_step(&x, &y, ids.len(), 2)?;
                if let Some(resumed) = &mut restored {
                    Model::train_step(resumed, &x, &y, ids.len(), 2)?;
                }
                if at + 1 == completed + 50 {
                    model.checkpoint()?.save(output.join("middle.ckpt"))?;
                    restored = Some(Model::from_checkpoint(&Checkpoint::load(
                        output.join("middle.ckpt"),
                    )?)?);
                }
            }
            seen += ids.len() as u64;
            at += 1;
            if at == completed + 100 {
                break;
            }
        }
    }
    let final_state = model.checkpoint()?;
    assert_eq!(
        final_state.state_sha256(),
        "6e719c31b89f1d01c6ec5c919b063c79d947a84a073335590c9f1784292886dc"
    );
    assert_eq!(
        final_state.raw_state(),
        restored.unwrap().checkpoint()?.raw_state()
    );
    assert_eq!(model.completed(), completed + 100);
    assert_eq!(model.examples(), seen);
    assert_eq!(fs::read(dir.join("checkpoint.bin"))?, raw);
    Ok(
        json!({"passed":true,"case":"arithmetic-sum-pow4","seed":1,"updates":100,
        "restored_after_updates":50,"restored_continuation_bit_exact":true,"initial_control_accuracy_exact":true,
        "final_state_sha256":final_state.state_sha256(),"original_checkpoint_unchanged":true,
        "scientific_training":false}),
    )
}
fn api_smoke(output: &Path) -> Result<Value> {
    let mut rows = Vec::new();
    for positions in [
        Positions::LeRoPE,
        Positions::AxialLeRoPE { features: 1 },
        Positions::AxialLeRoPE { features: 2 },
    ] {
        let mut config = ModelConfig::main(512, 3, 17, 5000);
        config.heads = 1;
        config.positions = positions;
        config.training_input_length = Some(2);
        let mut model = Model::new(config, 11)?;
        let initial = model.checkpoint()?;
        let count = initial.config().parameter_count()?;
        assert!(
            model
                .train_step(&[1, 2], &[IGNORE_TARGET, 512], 1, 2)
                .is_err()
        );
        let progress = model.train_step(
            &[1, 2, 3, 4, 5, 6],
            &[IGNORE_TARGET, 3, IGNORE_TARGET, 5, IGNORE_TARGET, 7],
            3,
            2,
        )?;
        assert_eq!(progress.learning_rate, 0.);
        assert_eq!(progress.completed, 1);
        assert_eq!(
            &model.checkpoint()?.raw_state()[..4 * count],
            &initial.raw_state()[..4 * count]
        );
        let saved = model.checkpoint()?;
        let file = output.join(format!("state-{}.ckpt", rows.len()));
        saved.save(&file)?;
        let mut second = Model::from_checkpoint(&Checkpoint::load(file)?)?;
        let prefix = [2, 3, 5, 8];
        let generated = model.generate(&prefix, 2, None)?;
        let identity = model.generate(&prefix, 2, Some(128))?;
        assert_eq!(generated, identity);
        assert_eq!(model.checkpoint()?.raw_state(), saved.raw_state());
        for active in [&mut model, &mut second] {
            active.train_step(
                &[1, 2, 3, 4, 5, 6],
                &[IGNORE_TARGET, 3, IGNORE_TARGET, 5, IGNORE_TARGET, 7],
                3,
                2,
            )?;
        }
        assert_eq!(
            model.checkpoint()?.raw_state(),
            second.checkpoint()?.raw_state()
        );
        let features = if matches!(positions, Positions::AxialLeRoPE { features: 2 }) {
            2
        } else {
            1
        };
        let raw = if features == 2 {
            vec![1., 2., 2., 3., 3., 5., 5., 8.]
        } else {
            vec![1., 2., 3., 5.]
        };
        let observed = Matrix::new(features, raw)?;
        let domains = vec![Domain::symmetric(96.); features];
        let operators = vec![Difference::order(1)?; features];
        let before = model.checkpoint()?.state_sha256();
        let mut diffgpt = if matches!(positions, Positions::AxialLeRoPE { .. }) {
            Some(DiffGpt::from_checkpoint(&model.checkpoint()?)?)
        } else {
            assert!(DiffGpt::from_checkpoint(&model.checkpoint()?).is_err());
            None
        };
        for tokenizer in [Tokenizer::Mash, Tokenizer::Lgt, Tokenizer::MashHalf] {
            let forecast = model.forecast(&observed, &domains, &operators, tokenizer, 1, None)?;
            assert_eq!(
                (forecast.values.rows(), forecast.values.columns()),
                (1, features)
            );
            assert_eq!(
                (forecast.tokens.rows(), forecast.tokens.columns()),
                (1, features)
            );
            assert!(forecast.values.values().iter().all(|v| v.is_finite()));
            assert!(forecast.tokens.values().iter().all(|v| *v >= 0 && *v < 512));
            assert_eq!(forecast.clipped_context_tokens, 0);
            if let Some(d) = &mut diffgpt {
                let f = d.forecast(&observed, &domains, &operators, tokenizer, 1)?;
                assert_eq!((f.values.rows(), f.values.columns()), (1, features));
                assert!(f.values.values().iter().all(|v| v.is_finite()));
                assert_eq!(d.checkpoint()?.state_sha256(), before);
                assert_eq!(d.config().training_input_length, Some(2));
            }
        }
        assert_eq!(model.checkpoint()?.state_sha256(), before);
        rows.push(
            json!({"positions":format!("{positions:?}"),"passed":true,"encoders":3,
            "zero_lr_preserves_weights":true,"resume_exact":true,"generation_identity_exact":true,
            "forecast_state_unchanged":true,"invalid_target_rejected":true,
            "diffgpt_2d_only_enforced":true,"diffgpt_default_yarn_tested":diffgpt.is_some()}),
        );
    }
    Ok(json!({"passed":true,"rows":rows,"scientific_training":false}))
}
fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 || !matches!(args[1].as_str(), "all" | "smoke") {
        return Err("usage: production-check all|smoke REPOSITORY FRESH_OUTPUT".into());
    }
    let repo = Path::new(&args[2]);
    let output = Path::new(&args[3]);
    fs::create_dir(output)?;
    let build = diff_gpt_kernite::build_info();
    assert_eq!(build["gpu_fp64"], false);
    save(&output.join("build.json"), &build)?;
    save(&output.join("api.json"), &api_smoke(output)?)?;
    if args[1] == "all" {
        save(&output.join("replay.json"), &replay(repo, output)?)?;
    }
    Ok(())
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
