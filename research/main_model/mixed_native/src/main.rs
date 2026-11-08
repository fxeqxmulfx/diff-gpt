//! Current model with validated warp packing, training, replay and diagnostics.
#![recursion_limit = "256"]
#[path = "../../../complex_geometry_suite/native/src/arithmetic_data.rs"]
#[allow(dead_code)]
mod arithmetic_data;
#[path = "../../optimized_native/src/attention.rs"]
mod attention;
#[path = "../../../complex_geometry_suite/diagnostic/src/attention_mode.rs"]
#[allow(dead_code)]
mod attention_mode;
#[path = "../../../complex_geometry_suite/native/src/base_ops.rs"]
mod base_ops;
#[path = "../../../complex_geometry_suite/native/src/blake.rs"]
mod blake;
#[path = "../../../complex_geometry_suite/diagnostic/src/centered_attention.rs"]
mod centered_attention;
#[path = "../../../complex_geometry_suite/native/src/complex.rs"]
mod complex;
#[path = "../../../complex_geometry_suite/native/src/data.rs"]
mod data;
#[allow(dead_code)]
#[path = "../../optimized_native/src/engine.rs"]
mod engine;
#[path = "../../../complex_geometry_suite/native/src/gemm.rs"]
mod gemm;
#[path = "../../native/src/layout.rs"]
mod layout;
#[path = "../../../complex_geometry_suite/native/src/metrics.rs"]
mod metrics;
mod mhc;
#[path = "../../wsd_suite_native/src/optimizer.rs"]
mod optimizer;
#[path = "../../wsd_native/src/optimizer_kernels.rs"]
mod optimizer_kernels;
#[allow(dead_code)]
#[path = "../../native/src/oracle.rs"]
mod oracle;
#[path = "../../optimized_native/src/ordinary_attention.rs"]
mod ordinary_attention;
#[path = "../../../complex_geometry_suite/native/src/pcg.rs"]
mod pcg;
#[path = "../../../complex_geometry_suite/diagnostic/src/plain_product.rs"]
mod plain_product;
#[path = "../../../complex_geometry_suite/native/src/random.rs"]
mod random;
#[path = "../../wsd_suite_native/src/runner.rs"]
#[allow(dead_code)]
mod runner;
#[path = "../../wsd_suite_native/src/schedule.rs"]
mod schedule;
#[path = "../../../complex_geometry_suite/native/src/sequence_ops.rs"]
mod sequence_ops;
mod sources;
#[path = "../../../complex_geometry_suite/diagnostic/src/stable_attention.rs"]
mod stable_attention;
#[path = "../../../complex_geometry_suite/diagnostic/src/stable_chain.rs"]
mod stable_chain;
#[path = "../../../complex_geometry_suite/diagnostic/src/stable_loss.rs"]
mod stable_loss;
#[path = "../../native/src/stress.rs"]
#[allow(dead_code)]
mod stress;
use attention_mode::AttentionMode;
use engine::{Config, Engine, Result};
use serde_json::Value;
use sha2::{Digest, Sha256};
fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|x| format!("{x:02x}"))
        .collect()
}
mod numerical;
mod profile;
#[path = "../../../complex_geometry_suite/native/src/attention.rs"]
mod reference_attention;
#[path = "../../native/src/mhc.rs"]
mod reference_mhc;
#[path = "../../../complex_geometry_suite/diagnostic/src/ordinary_attention.rs"]
mod reference_ordinary;
fn main() -> std::process::ExitCode {
    let args: Vec<_> = std::env::args().collect();
    let result = match args.get(1).map(String::as_str) {
        Some("stress") if args.len() == 3 => stress::run(std::path::Path::new(&args[2])),
        Some("numerical") if args.len() == 3 => numerical::run(std::path::Path::new(&args[2])),
        Some("preflight" | "report" | "archive" | "smoke" | "cell" | "resume" | "suite") => {
            runner::run()
        }
        _ => profile::run(),
    };
    match result {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
#[allow(dead_code)]
#[path = "../../wsd_suite_native/src/reference_optimizer.rs"]
mod reference_optimizer;
#[cfg(test)]
#[path = "../../wsd_lr001_native/src/schedule.rs"]
mod reference_schedule;
#[cfg(test)]
mod tests;

#[cfg(test)]
mod loss_precision {
    // Independent scalar confidence oracle, unchanged from the frozen runner.
    pub fn reference_softmax(logits: &[f32], gold: usize) -> (f64, Vec<f64>) {
        let max = logits
            .iter()
            .map(|&v| v as f64)
            .fold(f64::NEG_INFINITY, f64::max);
        let mass: Vec<_> = logits.iter().map(|&v| (v as f64 - max).exp()).collect();
        let other: f64 = mass
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != gold)
            .map(|(_, v)| v)
            .sum();
        let total = mass[gold] + other;
        let mut grad: Vec<_> = mass.iter().map(|v| v / total).collect();
        grad[gold] = -other / total;
        let loss = if logits[gold] as f64 == max {
            other.ln_1p()
        } else {
            max - logits[gold] as f64 + total.ln()
        };
        (loss, grad)
    }
}
