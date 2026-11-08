//! Native complex Peri-RMS transformer with Kernite CUDA kernels.
//!
//! The default architecture is the independently audited 75-short/30-long
//! research main: complex product FFN, normalized Q/K, squared-imaginary
//! SSMax score, phase-conjugated four-stream mHC, tied readout, and LeRoPE.
//! Arithmetic is statically FP16 forward affine products, TF32 backward
//! affine products, and FP32 accumulation/state/remaining kernels.
//!
//! [`Model::train_step`] performs no file I/O or device-to-host metric copy.
//! [`CheckpointCadence`] provides the requested 500-or-more epoch save cadence.
//! [`codec`] supplies all three native DiffGPT encoders. [`DiffGpt`] uses only
//! axial time-feature LeRoPE, with inference-only YaRN beyond actual training
//! context. The generic [`Model`] retains the independently validated 1D main.
#![recursion_limit = "256"]

mod api;
mod build_info;
mod checkpoint;
mod config;
mod diffgpt;

// Keep the validated kernel bundle intact, including private alternate entry
// points. The public API only constructs the selected score/mHC architecture.
#[allow(
    dead_code,
    reason = "Validated private kernel bundle retains alternate entry points"
)]
mod attention;
#[allow(
    dead_code,
    reason = "Validated private kernel bundle retains alternate entry points"
)]
mod base_ops;
#[allow(
    dead_code,
    reason = "Validated private kernel bundle retains alternate entry points"
)]
mod centered_attention;
#[allow(
    dead_code,
    reason = "Validated private kernel bundle retains alternate entry points"
)]
mod complex;
#[allow(
    dead_code,
    reason = "Validated private kernel bundle retains alternate entry points"
)]
mod covariance;
mod embedding_backward;
#[allow(
    dead_code,
    reason = "The safe public API wraps the general validated engine"
)]
mod engine;
mod gemm;
mod layout;
mod mhc;
#[allow(dead_code, reason = "Optimizer diagnostics stay private")]
mod optimizer;
mod optimizer_kernels;
mod ordinary_attention;
mod ordinary_forward;
mod phase_mhc;
mod phase_score;
mod phase_score_forward;
mod pipeline;
mod pipeline_host;
mod placement;
mod plain_product;
#[allow(
    dead_code,
    reason = "Independent positional reference remains available for local tests"
)]
mod position;
mod precision;
#[allow(
    dead_code,
    reason = "Frozen initialization RNG retains its original implementation"
)]
mod random;
mod rotary;
mod schedule;
mod sequence_ops;
mod stable_attention;
mod stable_chain;
mod stable_loss;
mod tensor;
mod tiled_transpose;

use engine::{Config, Engine};

/// Errors from shape validation, checkpoint I/O, and the CUDA driver.
pub type Result<T> = engine::Result<T>;
pub use api::{EpochSampler, Evaluation, Forecast, Model, Step};
pub use build_info::{build_info, kernel_sha256};
pub use checkpoint::{Checkpoint, CheckpointCadence};
pub use config::{ModelConfig, Positions};
pub use diffgpt::DiffGpt;
/// Exact native MASH, LGT and MASH-half tokenization and decoding.
pub use diffgpt_mash_native as codec;
/// The default main model's research identity.
pub const MODEL_ID: &str = "complex-peri-rms-squared-imaginary-phase-mhc4-lerope-fp16tf32-v1";
/// The user-selected 2D-only DiffGPT architecture; distinct from generic-main promotion.
pub const DIFFGPT_ID: &str =
    "complex-peri-rms-squared-imaginary-phase-mhc4-axial-lerope-yarn-fp16tf32-v1";
/// Fixed arithmetic placement used by both forward and backward.
pub const PRECISION: &str = precision::TAG;
/// A masked target. Other out-of-vocabulary target values are rejected.
pub const IGNORE_TARGET: u32 = u32::MAX;

pub(crate) fn hash(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

#[cfg(test)]
mod loss_precision {
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
