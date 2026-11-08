//! Current main model, quartic task only, historical AdamW/WSD recipe.
#[path = "../../../complex_geometry_suite/native/src/arithmetic_data.rs"]
#[expect(
    dead_code,
    reason = "Frozen arithmetic data port also contains an unused legacy RoPE helper"
)]
mod arithmetic_data;
#[path = "../../../complex_geometry_suite/native/src/attention.rs"]
mod attention;
#[path = "../../../complex_geometry_suite/native/src/base_ops.rs"]
mod base_ops;
#[path = "../../../complex_geometry_suite/diagnostic/src/centered_attention.rs"]
mod centered_attention;
#[path = "../../../complex_geometry_suite/native/src/complex.rs"]
mod complex;
#[path = "../../native/src/engine.rs"]
#[expect(
    dead_code,
    reason = "Reuse the immutable model; optimizer updates are supplied by this study"
)]
mod engine;
#[path = "../../../complex_geometry_suite/native/src/gemm.rs"]
mod gemm;
#[path = "../../native/src/layout.rs"]
#[allow(dead_code)]
mod layout;
#[path = "../../native/src/mhc.rs"]
mod mhc;
mod optimizer;
mod optimizer_kernels;
#[path = "../../../complex_geometry_suite/diagnostic/src/ordinary_attention.rs"]
mod ordinary_attention;
#[path = "../../../complex_geometry_suite/diagnostic/src/plain_product.rs"]
mod plain_product;
#[path = "../../../complex_geometry_suite/native/src/random.rs"]
#[expect(
    dead_code,
    reason = "Frozen RNG also supports sequence datasets outside this arithmetic study"
)]
mod random;
mod runner;
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
use engine::Result;
use sha2::{Digest, Sha256};
fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|x| format!("{x:02x}"))
        .collect()
}
fn main() -> std::process::ExitCode {
    match runner::run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            std::process::ExitCode::FAILURE
        }
    }
}
#[cfg(test)]
mod tests;
#[cfg(test)]
mod loss_precision {
    // Same independent confidence oracle used by the current main's loss test.
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
        let mut gradient: Vec<_> = mass.iter().map(|v| v / total).collect();
        gradient[gold] = -other / total;
        let loss = if logits[gold] as f64 == max {
            other.ln_1p()
        } else {
            max - logits[gold] as f64 + total.ln()
        };
        (loss, gradient)
    }
}
