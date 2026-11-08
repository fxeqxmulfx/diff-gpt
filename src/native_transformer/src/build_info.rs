//! CPU-only inventory of the Kernite PTX embedded in this library.
use serde_json::{Value, json};
const MODULES: &[(&str, &str)] = &[
    ("attention", crate::attention::kernels::PTX),
    ("base_ops", crate::base_ops::kernels::PTX),
    (
        "centered_attention",
        crate::centered_attention::kernels::PTX,
    ),
    ("complex", crate::complex::kernels::PTX),
    ("covariance", crate::covariance::kernels::PTX),
    (
        "embedding_backward",
        crate::embedding_backward::kernels::PTX,
    ),
    ("gemm", crate::gemm::kernels::PTX),
    ("mhc", crate::mhc::kernels::PTX),
    ("optimizer", crate::optimizer_kernels::kernels::PTX),
    (
        "ordinary_attention",
        crate::ordinary_attention::kernels::PTX,
    ),
    ("ordinary_forward", crate::ordinary_forward::kernels::PTX),
    ("phase_mhc", crate::phase_mhc::kernels::PTX),
    ("phase_score", crate::phase_score::kernels::PTX),
    (
        "phase_score_forward",
        crate::phase_score_forward::kernels::PTX,
    ),
    ("pipeline", crate::pipeline::kernels::PTX),
    ("plain_product", crate::plain_product::kernels::PTX),
    ("rotary", crate::rotary::kernels::PTX),
    ("sequence_ops", crate::sequence_ops::kernels::PTX),
    ("stable_attention", crate::stable_attention::kernels::PTX),
    ("stable_chain", crate::stable_chain::kernels::PTX),
    ("stable_loss", crate::stable_loss::kernels::PTX),
    ("tensor", crate::tensor::kernels::PTX),
    ("tiled_transpose", crate::tiled_transpose::kernels::PTX),
];
/// Digest of named PTX modules and the fixed arithmetic policy.
pub fn kernel_sha256() -> String {
    use sha2::{Digest, Sha256};
    let mut sha = Sha256::new();
    for (name, ptx) in MODULES {
        sha.update((name.len() as u64).to_le_bytes());
        sha.update(name.as_bytes());
        sha.update((ptx.len() as u64).to_le_bytes());
        sha.update(ptx.as_bytes());
    }
    sha.update(crate::PRECISION.as_bytes());
    sha.finalize().iter().map(|b| format!("{b:02x}")).collect()
}
/// Embedded module hashes and an explicit GPU-FP64 scan; no CUDA access.
pub fn build_info() -> Value {
    json!({"model":crate::MODEL_ID,"diffgpt":crate::DIFFGPT_ID,"diffgpt_positions":"axial_time_feature_only",
        "precision":crate::PRECISION,"kernel_sha256":kernel_sha256(),
        "gpu_fp64":MODULES.iter().any(|(_,ptx)|ptx.contains(".f64")),
        "modules":MODULES.iter().map(|(name,ptx)|json!({"name":name,"sha256":crate::hash(ptx.as_bytes()),"bytes":ptx.len()})).collect::<Vec<_>>()})
}
#[cfg(test)]
mod tests {
    #[test]
    fn embedded_bundle_contains_no_gpu_fp64() {
        assert!(!super::MODULES.is_empty());
        for (name, ptx) in super::MODULES {
            assert!(!ptx.contains(".f64"), "{name}");
        }
    }
}
