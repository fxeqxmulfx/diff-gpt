//! CPU-only independent provenance, history, state and metric audit.
//! Shares the separately validated codec/data, never the GPU engine or layout.
#![forbid(unsafe_code)]
#[path = "../../../complex_geometry_suite/native/src/random.rs"]
#[allow(dead_code)]
mod random;
use diffgpt_axial_data_native as data;
use diffgpt_mash_native::Tokenizer;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs, io::Write, path::Path};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const WIDTH: usize = 128;
const HIDDEN: usize = 342;
const BASE: usize = 512 * WIDTH + 4 * WIDTH * WIDTH + 3 * WIDTH * HIDDEN;
const COUNT: usize = BASE + 2 * (96 * WIDTH + 27) + 6 + 1 + 64;
fn require(value: bool, error: &str) -> Result<()> {
    if value { Ok(()) } else { Err(error.into()) }
}
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
fn number(v: &Value, key: &str) -> Result<usize> {
    Ok(v[key]
        .as_u64()
        .ok_or_else(|| format!("missing {key}"))?
        .try_into()?)
}
fn pinned(path: &Path, expected: &Value) -> Result<Vec<u8>> {
    let bytes = fs::read(path)?;
    require(
        *expected == hash(&bytes),
        &format!("hash mismatch:{}", path.display()),
    )?;
    Ok(bytes)
}
// Enumerate the independent complex coordinates from matrix geometry.
fn pairs() -> Vec<(usize, usize, usize, usize)> {
    let mut out = Vec::new();
    let embedding = 512 * WIDTH;
    for (offset, rows, cols, groups) in [
        (embedding, 3 * WIDTH, WIDTH, 3),
        (embedding + 3 * WIDTH * WIDTH, WIDTH, WIDTH, 1),
        (embedding + 4 * WIDTH * WIDTH, 2 * HIDDEN, WIDTH, 2),
        (
            embedding + 4 * WIDTH * WIDTH + 2 * HIDDEN * WIDTH,
            WIDTH,
            HIDDEN,
            1,
        ),
    ] {
        let half_rows = rows / groups / 2;
        for group in 0..groups {
            for r in 0..half_rows {
                for c in 0..cols / 2 {
                    let re = offset + (group * 2 * half_rows + r) * cols + c;
                    let im = re + half_rows * cols;
                    out.push((re, im, re + cols / 2, im + cols / 2));
                }
            }
        }
    }
    out
}
fn floats(bytes: &[u8]) -> Result<Vec<f32>> {
    require(bytes.len().is_multiple_of(4), "unaligned FP32 state")?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect())
}
fn state(bytes: &[u8]) -> Result<()> {
    let all = floats(bytes)?;
    require(all.len() == 3 * COUNT, "wrong state geometry")?;
    require(all.iter().all(|x| x.is_finite()), "nonfinite state")?;
    let (p, rest) = all.split_at(COUNT);
    let (m, v) = rest.split_at(COUNT);
    require(v.iter().all(|x| *x >= 0.), "negative variance")?;
    for (re, im, ni, copy) in pairs() {
        require(
            p[re] == p[copy] && p[im] == -p[ni],
            "broken complex weight ties",
        )?;
        require(
            m[re] == m[copy] && m[im] == -m[ni],
            "broken first moment ties",
        )?;
        require(v[re] == v[copy] && v[im] == v[ni], "broken variance ties")?;
    }
    Ok(())
}
fn initial(seed: u32) -> Vec<u8> {
    let mut rng = random::Random::new(seed);
    let mut p: Vec<f32> = (0..BASE).map(|_| rng.gauss(0., 0.02) as f32).collect();
    for (re, im, ni, copy) in pairs() {
        p[copy] = p[re];
        p[ni] = -p[im];
    }
    let mut rng = random::Random::new(seed + 20000);
    for _ in 0..2 {
        p.extend((0..96 * WIDTH).map(|_| rng.gauss(0., 0.02) as f32));
        p.extend([0.; 8]);
        for r in 0..4 {
            for c in 0..4 {
                p.push(if r == c { 2. } else { 0. });
            }
        }
        p.extend([0.01; 3]);
    }
    p.resize(COUNT, 0.);
    p.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn sampler(seed: u32) -> String {
    let mut rng = random::Random::new(seed + 10000);
    let mut digest = Sha256::new();
    for _ in 0..5000 * 512 {
        digest.update(rng.below(4704).to_le_bytes());
    }
    digest
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
fn ratio(metrics: &Value, count: &str, field: &str, n: usize) -> Result<usize> {
    let hits = number(metrics, count)?;
    let accuracy = metrics[field].as_f64().ok_or("missing accuracy")?;
    require(
        n > 0
            && hits <= n
            && accuracy.is_finite()
            && (accuracy - hits as f64 / n as f64).abs() < 2e-15,
        "metric/count mismatch",
    )?;
    Ok(hits)
}
fn metrics(v: &Value, features: usize, yarn: bool) -> Result<bool> {
    require(
        v["examples"] == 4704 && v["features"] == features && v["inference_yarn"] == yarn,
        "wrong evaluation geometry/mode",
    )?;
    let decoded = ratio(v, "decoded_correct", "decoded_accuracy", 4704)?;
    let forecast = ratio(
        v,
        "forecast_reanchored_correct",
        "forecast_reanchored_accuracy",
        4704,
    )?;
    let exact = ratio(v, "exact_token_rows", "exact_token_row_accuracy", 4704)?;
    let tokens = ratio(v, "token_correct", "token_accuracy", 4704 * features)?;
    require(
        tokens >= exact * features,
        "joint token hits exceed marginal hits",
    )?;
    for (field, joint) in [
        ("feature_decoded_correct", decoded),
        ("feature_reanchored_correct", forecast),
    ] {
        let counts = v[field].as_array().ok_or("missing feature counts")?;
        require(counts.len() == features, "wrong feature count length")?;
        let counts: Vec<_> = counts
            .iter()
            .map(|x| x.as_u64().ok_or("invalid feature count"))
            .collect::<std::result::Result<_, _>>()?;
        require(
            counts.iter().all(|x| *x <= 4704 && *x >= joint as u64),
            "invalid joint/marginal counts",
        )?;
        require(
            counts.iter().sum::<u64>() <= ((features - 1) * 4704 + joint) as u64,
            "inconsistent feature intersections",
        )?;
    }
    Ok(10 * decoded >= 9 * 4704)
}
fn checkpoint(dir: &Path, tag: &str, history: &[u8], step: usize, reg: &Value) -> Result<Vec<u8>> {
    let meta = read(&dir.join(format!("{tag}.json")))?;
    let bytes = pinned(&dir.join(format!("{tag}.bin")), &meta["state_sha256"])?;
    state(&bytes)?;
    require(
        meta["completed"] == step
            && meta["examples"] == step * 512
            && meta["epochs"] == step as f64 * 512. / 4704.
            && meta["history_sha256"] == hash(history)
            && meta["source_sha256"] == reg["source_sha256"]
            && meta["ptx_sha256"] == reg["ptx_sha256"],
        "checkpoint provenance/history/step mismatch",
    )?;
    Ok(bytes)
}
fn archive(reg: &Value) -> Result<()> {
    pinned(Path::new(string(reg, "binary")?), &reg["binary_sha256"])?;
    let root = Path::new(string(reg, "source_archive")?);
    let index = read(&root.join("source-index.json"))?;
    let mut captured = Vec::new();
    for row in index["files"].as_array().ok_or("missing source files")? {
        let name = string(row, "name")?;
        let bytes = pinned(&root.join("source").join(name), &row["sha256"])?;
        require(
            bytes.len() == number(row, "bytes")?,
            "source length mismatch",
        )?;
        captured.extend((name.len() as u64).to_le_bytes());
        captured.extend(name.as_bytes());
        captured.extend((bytes.len() as u64).to_le_bytes());
        captured.extend(bytes);
    }
    require(
        hash(&captured) == reg["source_sha256"],
        "compiled source mismatch",
    )?;
    let report = read(&root.join("ptx/report.json"))?;
    require(
        report["passed"] == true && report["ptx_sha256"] == reg["ptx_sha256"],
        "PTX report mismatch",
    )?;
    for row in report["modules"].as_array().ok_or("missing PTX list")? {
        let bytes = pinned(
            &root
                .join("ptx")
                .join(format!("{}.ptx", string(row, "module")?)),
            &row["sha256"],
        )?;
        require(
            !String::from_utf8(bytes)?.contains(".f64"),
            "GPU FP64 forbidden",
        )?;
    }
    let validation: Value = serde_json::from_slice(&pinned(
        Path::new(string(reg, "validation")?),
        &reg["validation_sha256"],
    )?)?;
    require(
        validation["implementation_passed"] == true
            && validation["source_sha256"] == reg["source_sha256"]
            && validation["ptx_sha256"] == reg["ptx_sha256"],
        "validation mismatch",
    )
}
fn audit(root: &Path) -> Result<Value> {
    let raw = fs::read(root.join("experiment.json"))?;
    let reg: Value = serde_json::from_slice(&raw)?;
    require(
        reg["kind"] == "diffgpt_axial_lerope_yarn"
            && reg["budget_updates"] == 5000
            && reg["batch_size"] == 512
            && reg["observation_updates"] == 50,
        "protocol mismatch",
    )?;
    archive(&reg)?;
    let data_reg: Value = serde_json::from_slice(&pinned(
        Path::new(string(&reg, "data_registration")?),
        &reg["data_registration_sha256"],
    )?)?;
    let mut expected = Vec::new();
    for (name, _) in [
        ("mash", Tokenizer::Mash),
        ("lgt", Tokenizer::Lgt),
        ("mash-half", Tokenizer::MashHalf),
    ] {
        for features in [1, 2] {
            for seed in [11, 17, 23] {
                for arm in ["sequence", "axial"] {
                    expected
                        .push(json!({"tokenizer":name,"features":features,"seed":seed,"arm":arm}));
                }
            }
        }
    }
    require(
        reg["cells"] == json!(expected),
        "undeclared/misordered DiffGPT cells",
    )?;
    let mut results = Vec::new();
    let mut unrun = false;
    for cell in expected {
        let name = string(&cell, "tokenizer")?;
        let seed = number(&cell, "seed")? as u32;
        let features = number(&cell, "features")?;
        let arm = string(&cell, "arm")?;
        let dir = root.join(format!("{arm}/{name}/features{features}/seed{seed}"));
        if !dir.exists() {
            unrun = true;
            results.push(json!({"cell":cell,"status":"unrun"}));
            continue;
        }
        require(!unrun, "completed cell after an unrun hole")?;
        let meta = read(&dir.join("metadata.json"))?;
        let result = read(&dir.join("result.json"))?;
        let extent = if arm == "sequence" {
            3 * features - 1
        } else {
            (3 * features - 1).div_ceil(features)
        };
        require(
            meta["arm"] == arm
                && meta["tokenizer"] == name
                && meta["features"] == features
                && meta["seed"] == seed
                && meta["context"] == 2
                && meta["training_extent"] == extent
                && meta["vocab"] == 512
                && meta["width"] == 128
                && meta["heads"] == 1
                && meta["layers"] == 1
                && meta["expanded_parameters"] == COUNT
                && meta["independent_parameters"] == COUNT - 2 * pairs().len()
                && meta["budget_updates"] == 5000
                && meta["batch_size"] == 512
                && meta["observe"] == 50
                && meta["optimizer"] == "AdamW/WSD"
                && meta["lr"] == 0.001
                && meta["betas"] == json!([0.9, 0.95])
                && meta["weight_decay"] == 0.1
                && meta["warmup"] == 100
                && meta["cosine_start"] == 4000
                && meta["separate_frequency_clip"] == 1
                && meta["frequency_weight_decay"] == 0
                && meta["checkpoint_epochs"] == 500
                && meta["training_has_yarn"] == false
                && meta["native_split_not_torch_randperm"] == true
                && meta["source_sha256"] == reg["source_sha256"]
                && meta["ptx_sha256"] == reg["ptx_sha256"],
            "model/data/optimizer metadata mismatch",
        )?;
        require(
            meta["axes"]
                == if arm == "sequence" {
                    "Sequence".to_owned()
                } else {
                    format!("Axial {{ features: {features} }}")
                },
            "axes mismatch",
        )?;
        let init = initial(seed);
        require(
            meta["initial_sha256"] == hash(&init) && meta["sampler_sha256"] == sampler(seed),
            "initialization/sampler mismatch",
        )?;
        let tokenizer = match name {
            "mash" => Tokenizer::Mash,
            "lgt" => Tokenizer::Lgt,
            _ => Tokenizer::MashHalf,
        };
        let mut heldout = Vec::new();
        for context in [2, 4, 8] {
            let d = data::build(seed, features, context, tokenizer)?;
            let rows: Vec<_> = data_reg["rows"]
                .as_array()
                .ok_or("missing data rows")?
                .iter()
                .filter(|r| {
                    r["tokenizer"] == name
                        && r["features"] == features
                        && r["seed"] == seed
                        && r["context"] == context
                })
                .collect();
            require(rows.len() == 1, "missing/ambiguous registered dataset")?;
            let th = data::fingerprint(&d.train);
            let vh = data::fingerprint(&d.validation);
            require(
                rows[0]["train_sha256"] == th && rows[0]["heldout_sha256"] == vh,
                "dataset rebuild mismatch",
            )?;
            if context == 2 {
                require(
                    meta["train_sha256"] == th && meta["heldout_sha256"] == vh,
                    "training data mismatch",
                )?;
            }
            heldout.push(vh);
        }
        let history = fs::read(dir.join("history.jsonl"))?;
        let mut points = Vec::new();
        let mut prefix = Vec::new();
        let mut first = None;
        let mut first_prefix = Vec::new();
        for (i, line) in history.split_inclusive(|b| *b == b'\n').enumerate() {
            require(line.last() == Some(&b'\n'), "incomplete history line")?;
            prefix.extend(line);
            let point: Value = serde_json::from_slice(line)?;
            let step = i * 50;
            require(
                point["step"] == step
                    && point["examples"] == step * 512
                    && point["epochs"] == step as f64 * 512. / 4704.
                    && point["seconds"]
                        .as_f64()
                        .is_some_and(|v| v.is_finite() && v >= 0.),
                "observation schedule mismatch",
            )?;
            if metrics(&point["validation"], features, false)? && first.is_none() {
                first = Some(step);
                first_prefix = prefix.clone();
            }
            points.push(point);
        }
        require(
            points.len() == 101
                && result["completed"] == 5000
                && result["endpoint"] == *points.last().unwrap()
                && result["metadata"] == meta
                && result["first90_updates"] == json!(first)
                && result["first90_epochs"] == json!(first.map(|s| s as f64 * 512. / 4704.))
                && result["previous_observation_updates"]
                    == json!(first.map(|s| s.saturating_sub(50)))
                && result["first90_is_not_sustained90"] == true,
            "terminal/first90 mismatch",
        )?;
        require(
            result["status"]
                == if first.is_some() {
                    "reached90"
                } else {
                    "budget_exhausted"
                },
            "wrong terminal classification",
        )?;
        require(
            read(&dir.join("execution.json"))? == json!({"clean":true,"completed":5000}),
            "unclean terminal state",
        )?;
        checkpoint(&dir, "checkpoint", &history, 5000, &reg)?;
        let initial_state = checkpoint(
            &dir,
            "initial",
            history
                .split_inclusive(|b| *b == b'\n')
                .next()
                .ok_or("empty history")?,
            0,
            &reg,
        )?;
        require(
            initial_state[..4 * COUNT] == init
                && initial_state[4 * COUNT..].iter().all(|b| *b == 0),
            "initial checkpoint mismatch",
        )?;
        if let Some(step) = first {
            checkpoint(&dir, "first90", &first_prefix, step, &reg)?;
        } else {
            require(
                !dir.join("first90.bin").exists() && !dir.join("first90.json").exists(),
                "fabricated first90 checkpoint",
            )?;
        }
        let extra = result["extrapolation"]
            .as_array()
            .ok_or("missing extrapolation")?;
        require(extra.len() == 2, "missing context")?;
        for (i, context) in [4, 8].iter().enumerate() {
            require(
                extra[i]["context"] == *context && extra[i]["heldout_sha256"] == heldout[i + 1],
                "extrapolation data mismatch",
            )?;
            metrics(&extra[i]["native"], features, false)?;
            metrics(&extra[i]["yarn"], features, true)?;
        }
        results.push(json!({"cell":cell,"status":result["status"],"first90_updates":first,
            "first90_epochs":result["first90_epochs"],"endpoint":result["endpoint"]["validation"],
            "extrapolation":extra,"checkpoint_sha256":hash(&fs::read(dir.join("checkpoint.bin"))?)}));
    }
    Ok(
        json!({"audit_passed":true,"registration_sha256":hash(&raw),"expected_cells":36,
        "completed_cells":results.iter().filter(|r|r["status"]!="unrun").count(),
        "unrun_cells":results.iter().filter(|r|r["status"]=="unrun").count(),
        "rows":results,"eligible_fixed_suite_promotion":false,
        "limitations":"Independent provenance/history/state/count audit; prediction correctness uses separately validated native inference and decoder. Additional DiffGPT evidence cannot replace the fixed105 coverage gate."}),
    )
}
fn main() -> std::process::ExitCode {
    let args: Vec<_> = std::env::args().collect();
    let run = || -> Result<()> {
        require(args.len() == 3, "usage: diffgpt-audit ROOT FRESH_OUTPUT")?;
        let result = audit(Path::new(&args[1]));
        let mut file = fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&args[2])?;
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
    };
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
    fn state_rejects_broken_ties_and_invalid_variance() {
        let mut bytes = initial(11);
        bytes.resize(12 * COUNT, 0);
        assert!(state(&bytes).is_ok());
        let (_, _, _, copy) = pairs()[0];
        bytes[4 * copy..4 * copy + 4].copy_from_slice(&1f32.to_le_bytes());
        assert!(state(&bytes).is_err());
        let mut bytes = initial(11);
        bytes.resize(12 * COUNT, 0);
        bytes[8 * COUNT..8 * COUNT + 4].copy_from_slice(&(-1f32).to_le_bytes());
        assert!(state(&bytes).is_err());
    }
    #[test]
    fn metric_counts_cannot_be_replaced_with_rounded_accuracy() {
        assert!(
            ratio(
                &json!({"correct":4233,"accuracy":0.9}),
                "correct",
                "accuracy",
                4704
            )
            .is_err()
        );
        assert!(
            ratio(
                &json!({"correct":4705,"accuracy":1.}),
                "correct",
                "accuracy",
                4704
            )
            .is_err()
        );
        assert_eq!(
            ratio(
                &json!({"correct":4234,"accuracy":4234f64/4704.}),
                "correct",
                "accuracy",
                4704
            )
            .unwrap(),
            4234
        );
    }
}
