//! Current model with validated warp packing, training, replay and diagnostics.
#![recursion_limit = "256"]
#[path = "../../../complex_geometry_suite/native/src/arithmetic_data.rs"]
#[allow(dead_code)]
mod arithmetic_data;
mod attention;
#[path = "../../../complex_geometry_suite/diagnostic/src/attention_mode.rs"]
#[allow(dead_code)]
mod attention_mode;
mod base_ops;
#[path = "../../../complex_geometry_suite/native/src/blake.rs"]
mod blake;
mod centered_attention;
#[path = "../../../complex_geometry_suite/native/src/complex.rs"]
mod complex;
#[path = "../../../complex_geometry_suite/native/src/data.rs"]
mod data;
mod embedding_backward;
#[allow(dead_code)]
mod engine;
mod equivalence;
#[path = "../../../complex_geometry_suite/native/src/gemm.rs"]
mod gemm;
mod gemm_equivalence;
mod layout;
#[path = "../../../complex_geometry_suite/native/src/metrics.rs"]
mod metrics;
mod mhc;
mod optimizer;
mod optimizer_kernels;
#[allow(dead_code)]
mod oracle;
mod ordinary_attention;
mod ordinary_forward;
#[path = "../../tensor_model_native/src/attention.rs"]
mod parent_attention;
#[path = "../../tensor_model_native/src/mhc.rs"]
mod parent_mhc;
#[path = "../../tensor_model_native/src/base_ops.rs"]
mod parent_ops;
#[path = "../../tensor_model_native/src/ordinary_attention.rs"]
mod parent_ordinary;
#[path = "../../tensor_native/src/pipeline.rs"]
mod parent_pipeline;
#[path = "../../tensor_native/src/tensor.rs"]
mod parent_tensor;
#[path = "../../../complex_geometry_suite/native/src/pcg.rs"]
mod pcg;
mod pipeline;
mod pipeline_host;
mod placement;
#[path = "../../../complex_geometry_suite/diagnostic/src/plain_product.rs"]
mod plain_product;
mod precision;
#[path = "../../../complex_geometry_suite/native/src/random.rs"]
mod random;
mod rotary;
mod rotary_validation;
#[allow(dead_code)]
mod runner;
#[path = "../../wsd_suite_native/src/schedule.rs"]
mod schedule;
#[path = "../../../complex_geometry_suite/native/src/sequence_ops.rs"]
mod sequence_ops;
mod sources;
mod stable_attention;
mod stable_chain;
mod stable_loss;
#[allow(dead_code)]
mod stress;
mod tensor;
mod tiled_transpose;
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
mod ptx_audit;
#[path = "../../../complex_geometry_suite/native/src/attention.rs"]
mod reference_attention;
#[path = "../../native/src/mhc.rs"]
mod reference_mhc;
#[path = "../../../complex_geometry_suite/diagnostic/src/ordinary_attention.rs"]
mod reference_ordinary;
fn main() -> std::process::ExitCode {
    let args: Vec<_> = std::env::args().collect();
    let result = match args.get(1).map(String::as_str) {
        Some("equivalence") if args.len() == 3 => equivalence::run(std::path::Path::new(&args[2])),
        Some("stress") if args.len() == 3 => stress::run(std::path::Path::new(&args[2])),
        Some("numerical") if args.len() == 3 => numerical::run(std::path::Path::new(&args[2])),
        Some("ptx-audit") if args.len() == 3 => ptx_audit::run(std::path::Path::new(&args[2])),
        Some("rotary-validation") if args.len() == 3 => {
            rotary_validation::run(std::path::Path::new(&args[2]))
        }
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
mod pair_optimizer_tests;
#[cfg(test)]
#[allow(dead_code)]
mod reference_optimizer;
#[cfg(test)]
#[path = "../../wsd_native/src/optimizer_kernels.rs"]
mod reference_optimizer_kernels;
#[cfg(test)]
#[path = "../../wsd_lr001_native/src/schedule.rs"]
mod reference_schedule;
#[cfg(test)]
mod rotary_optimizer_tests;
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
