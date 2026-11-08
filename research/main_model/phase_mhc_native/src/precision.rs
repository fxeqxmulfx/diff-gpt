//! Precision choices are Cargo compile-time features, never data-dependent.
pub const BF16: bool = cfg!(feature = "bf16-forward");
pub const TF32_BACKWARD: bool = cfg!(feature = "tf32-backward");
pub const FP32_AFFINE_REFERENCE: bool = cfg!(feature = "fp32-affine-reference");
pub const TAG: &str = if FP32_AFFINE_REFERENCE {
    "all affine forward/backward FP32: numerical diagnostic only, training disabled"
} else {
    match (BF16, TF32_BACKWARD) {
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
    }
};
