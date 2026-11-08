//! Precision choices are Cargo compile-time features, never data-dependent.
pub const BF16: bool = cfg!(feature = "bf16-forward");
pub const TF32_BACKWARD: bool = cfg!(feature = "tf32-backward");
pub const TAG: &str = match (BF16, TF32_BACKWARD) {
    (false, false) => {
        "FP16 forward GEMM; FP32 backward GEMM; FP32 accumulation, norms, master weights and moments"
    }
    (true, false) => {
        "BF16 forward GEMM; FP32 backward GEMM; FP32 accumulation, norms, master weights and moments"
    }
    (false, true) => {
        "FP16 forward GEMM; TF32 backward GEMM; FP32 accumulation, norms, master weights and moments"
    }
    (true, true) => {
        "BF16 forward GEMM; TF32 backward GEMM; FP32 accumulation, norms, master weights and moments"
    }
};
