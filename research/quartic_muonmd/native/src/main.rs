//! Frozen quartic study with complex-linear model and native complex MuonMD.
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
#[path = "../../../complex_geometry_suite/native/src/complex.rs"]
mod complex;
#[path = "../../../complex_geometry_suite/native/src/engine.rs"]
#[expect(
    dead_code,
    reason = "Reuse the immutable model; optimizer updates are supplied by this study"
)]
mod engine;
#[path = "../../../complex_geometry_suite/native/src/gemm.rs"]
mod gemm;
#[path = "../../../complex_geometry_suite/native/src/layout.rs"]
mod layout;
mod md_ops;
mod optimizer;
#[path = "../../../complex_geometry_suite/native/src/random.rs"]
#[expect(
    dead_code,
    reason = "Frozen RNG also supports sequence datasets outside this arithmetic study"
)]
mod random;
mod runner;
#[path = "../../../complex_geometry_suite/native/src/sequence_ops.rs"]
mod sequence_ops;
mod sources;
use engine::Result;
use sha2::{Digest, Sha256};
fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|x| format!("{x:02x}"))
        .collect()
}
#[cfg(test)]
mod tests;
fn main() -> std::process::ExitCode {
    match runner::run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            std::process::ExitCode::FAILURE
        }
    }
}
