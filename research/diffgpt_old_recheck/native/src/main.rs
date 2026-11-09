//! Narrow historical architecture/recipe recheck; not a production model choice.
#![recursion_limit = "256"]
mod attention;
#[path = "../../../complex_geometry_suite/diagnostic/src/attention_mode.rs"]
#[allow(
    dead_code,
    reason = "Only historical real XSA is selected in this executable."
)]
mod attention_mode;
#[path = "../../../main_model/fp32_native/src/base_ops.rs"]
mod base_ops;
#[path = "../../../main_model/fp32_native/src/centered_attention.rs"]
mod centered_attention;
#[path = "../../../complex_geometry_suite/native/src/complex.rs"]
mod complex;
#[allow(
    dead_code,
    reason = "Validated shared engine retains unreachable research modes."
)]
mod engine;
#[path = "../../../complex_geometry_suite/native/src/gemm.rs"]
mod gemm;
#[allow(
    dead_code,
    reason = "Real historical layout does not use complex ownership helpers."
)]
mod layout;
#[path = "../../../main_model/fp32_native/src/mhc.rs"]
mod mhc;
#[allow(
    dead_code,
    reason = "Independent reference includes other archived architecture modes."
)]
mod oracle;
#[path = "../../../main_model/fp32_native/src/ordinary_attention.rs"]
mod ordinary_attention;
#[path = "../../../complex_geometry_suite/diagnostic/src/plain_product.rs"]
mod plain_product;
#[path = "../../../complex_geometry_suite/native/src/random.rs"]
#[allow(
    dead_code,
    reason = "Shared deterministic RNG also supports unused task samplers."
)]
mod random;
mod runner;
mod schedule_free;
#[path = "../../../complex_geometry_suite/native/src/sequence_ops.rs"]
mod sequence_ops;
mod stable_attention;
mod stable_chain;
#[path = "../../../main_model/fp32_native/src/stable_loss.rs"]
mod stable_loss;
mod validation;
use attention_mode::AttentionMode;
use engine::{Config, Engine, Result};
use serde_json::Value;
use sha2::{Digest, Sha256};

fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|v| format!("{v:02x}"))
        .collect()
}
fn config(batch: usize) -> Config {
    Config {
        width: 128,
        vocab: 512,
        heads: 1,
        max_batch: batch,
        max_length: 2,
        layers: 1,
        normalized: true,
        ssmax: false,
        gain: 128_f32.sqrt().recip(),
        lr: 0.03,
        decay: 1.,
        warmup: 10,
        architecture: "real".into(),
        phase_optimizer: false,
    }
}
fn initialize(seed: u32) -> Vec<f32> {
    let cfg = config(512);
    let mut p = layout::initialize(cfg.width, cfg.layers, cfg.vocab, seed);
    let start = cfg.vocab * cfg.width;
    let h = layout::hidden(cfg.width, "real");
    for (offset, count) in [
        (start + 3 * 128 * 128, 128 * 128),
        (start + 4 * 128 * 128 + 2 * h * 128, 128 * h),
    ] {
        for value in &mut p[offset..offset + count] {
            *value *= 2_f32.sqrt().recip();
        }
    }
    assert_eq!(p.len(), 262016);
    p
}
fn ptx() -> serde_json::Value {
    let modules = [
        ("attention", attention::kernels::PTX),
        ("base_ops", base_ops::kernels::PTX),
        ("centered_attention", centered_attention::kernels::PTX),
        ("complex", complex::kernels::PTX),
        ("gemm", gemm::kernels::PTX),
        ("mhc", mhc::kernels::PTX),
        ("ordinary_attention", ordinary_attention::kernels::PTX),
        ("plain_product", plain_product::kernels::PTX),
        ("sequence_ops", sequence_ops::kernels::PTX),
        ("stable_attention", stable_attention::kernels::PTX),
        ("stable_chain", stable_chain::kernels::PTX),
        ("stable_loss", stable_loss::kernels::PTX),
        ("schedule_free", schedule_free::kernels::PTX),
    ];
    serde_json::json!(
        modules
            .iter()
            .map(|(name, source)| serde_json::json!({
                "name":name,"sha256":hash(source.as_bytes()),"gpu_fp64":source.contains(".f64")
            }))
            .collect::<Vec<_>>()
    )
}
fn main() -> std::process::ExitCode {
    let args: Vec<_> = std::env::args().collect();
    let result = match args.get(1).map(String::as_str) {
        Some("validate") if args.len()==3 => validation::run(std::path::Path::new(&args[2])),
        Some("run") if args.len()==4 => runner::run(std::path::Path::new(&args[2]), args[3].parse().unwrap_or(0)),
        Some("replay") if args.len()==3 => runner::replay(std::path::Path::new(&args[2])),
        Some("build-info") => { println!("{}",ptx()); Ok(()) },
        _ => Err("usage: diffgpt-old-recheck validate FRESH_DIR | run REGISTERED_ROOT SEED | replay CELL_DIR | build-info".into()),
    };
    match result {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            std::process::ExitCode::FAILURE
        }
    }
}
