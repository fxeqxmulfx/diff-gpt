//! General sequence backend for the frozen transformer suite; no Python runtime.
#[expect(
    dead_code,
    reason = "Reuse frozen SplitMix64; other entry points belong to arithmetic backend"
)]
mod arithmetic_data;
mod attention;
mod base_ops;
mod blake;
mod complex;
mod data;
mod engine;
mod gemm;
mod layout;
mod metrics;
mod pcg;
mod random;
mod runner;
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
        Err(e) => {
            eprintln!("{e}");
            std::process::ExitCode::FAILURE
        }
    }
}
