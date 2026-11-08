//! One-time native dataset registration; emits lengths and exact fingerprints.
use diffgpt_axial_data_native::{build, fingerprint};
use diffgpt_mash_native::Tokenizer;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{fs, io::Write, path::Path};

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 2 {
        return Err("usage: diffgpt-axial-data-native FRESH_JSON".into());
    }
    let path = Path::new(&args[1]);
    if path.exists() {
        return Err("refuse overwrite of data registration".into());
    }
    let mut rows = Vec::new();
    for seed in [11, 17, 23] {
        for (name, tokenizer) in [
            ("mash", Tokenizer::Mash),
            ("lgt", Tokenizer::Lgt),
            ("mash-half", Tokenizer::MashHalf),
        ] {
            for features in [1, 2] {
                for context in [2, 4, 8] {
                    let d = build(seed, features, context, tokenizer)?;
                    rows.push(json!({"seed":seed,"tokenizer":name,"features":features,"context":context,
                "train_examples":d.train.len(),"heldout_examples":d.validation.len(),
                "train_sha256":fingerprint(&d.train),"heldout_sha256":fingerprint(&d.validation),
                "training_cycle_ids":d.training_cycle_ids,"validation_cycle_ids":d.validation_cycle_ids,
                "input_length":(context+1)*features-1,"prompt_length":context*features,
                "time_extent":((context+1)*features-1).div_ceil(features),
                "scale":d.train[0].scale,"zero":d.train[0].zero,
                "clipped_tokens":0,"all_truth_tokens_decode_correctly":true}));
                }
            }
        }
    }
    let mut sources = Vec::new();
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    for path in [
        "Cargo.toml",
        "Cargo.lock",
        "src/lib.rs",
        "src/main.rs",
        "../mash_native/src/lib.rs",
        "../mash_native/src/codec.rs",
        "../../complex_geometry_suite/native/src/random.rs",
        "../DIFFGPT_AXIAL_PLAN.txt",
    ] {
        let digest = Sha256::digest(fs::read(root.join(path))?);
        sources.push(json!({"path":path,"sha256":digest.iter().map(|b|format!("{b:02x}")).collect::<String>()}));
    }
    let value = json!({"kind":"native_diffgpt_axial_dataset_registration","rows":rows,"sources":sources,
        "vocab":512,"domain":96,"difference_order":1,"raw_states":9408,"cycles":48,"cycle_length":196,
        "split_rng":"native Random(seed),shuffle48cycle IDs; native split is not Torch randperm",
        "feature_pairing_rng":"native Random(seed+20000),permutation within each partition",
        "no_python_execution":true,"scientific_training_started":false});
    let mut file = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)?;
    serde_json::to_writer_pretty(&mut file, &value)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
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
