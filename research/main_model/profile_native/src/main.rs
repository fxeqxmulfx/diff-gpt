//! Executable current main model and its preregistered four-stream mHC candidate.
#![recursion_limit = "256"]
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
#[path = "../../native/src/engine.rs"]
mod engine;
#[path = "../../../complex_geometry_suite/native/src/gemm.rs"]
mod gemm;
#[path = "../../native/src/layout.rs"]
mod layout;
#[path = "../../../complex_geometry_suite/native/src/metrics.rs"]
mod metrics;
#[path = "../../native/src/mhc.rs"]
mod mhc;
#[path = "../../wsd_suite_native/src/optimizer.rs"]
mod optimizer;
#[path = "../../wsd_native/src/optimizer_kernels.rs"]
mod optimizer_kernels;
#[allow(dead_code)]
#[path = "../../native/src/oracle.rs"]
mod oracle;
#[path = "../../../complex_geometry_suite/diagnostic/src/ordinary_attention.rs"]
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
#[path = "../../wsd_suite_native/src/sources.rs"]
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
mod profile;
fn main() -> std::process::ExitCode {
    match profile::run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            std::process::ExitCode::FAILURE
        }
    }
}
