//! Executable current main model and its preregistered four-stream mHC candidate.
#[path = "../../../complex_geometry_suite/native/src/arithmetic_data.rs"]
#[allow(dead_code)]
mod arithmetic_data;
#[path = "../../../complex_geometry_suite/native/src/attention.rs"]
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
mod engine;
#[path = "../../../complex_geometry_suite/native/src/gemm.rs"]
mod gemm;
mod layout;
#[path = "../../../complex_geometry_suite/native/src/metrics.rs"]
mod metrics;
mod mhc;
#[allow(dead_code)]
mod oracle;
#[path = "../../../complex_geometry_suite/diagnostic/src/ordinary_attention.rs"]
mod ordinary_attention;
#[path = "../../../complex_geometry_suite/native/src/pcg.rs"]
mod pcg;
#[path = "../../../complex_geometry_suite/diagnostic/src/plain_product.rs"]
mod plain_product;
#[path = "../../../complex_geometry_suite/native/src/random.rs"]
mod random;
mod runner;
#[path = "../../../complex_geometry_suite/native/src/sequence_ops.rs"]
mod sequence_ops;
mod sources;
#[path = "../../../complex_geometry_suite/diagnostic/src/stable_attention.rs"]
mod stable_attention;
#[path = "../../../complex_geometry_suite/diagnostic/src/stable_chain.rs"]
mod stable_chain;
#[path = "../../../complex_geometry_suite/diagnostic/src/stable_loss.rs"]
mod stable_loss;
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
fn main() -> std::process::ExitCode {
    let args: Vec<_> = std::env::args().collect();
    let result = if args.get(1).map(String::as_str) == Some("stress") && args.len() == 3 {
        stress::run(std::path::Path::new(&args[2]))
    } else {
        runner::run()
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
mod loss_precision {
    // Ported scalar confidence oracle from diagnostic/src/loss_precision.rs.
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
