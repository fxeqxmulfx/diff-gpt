//! Fixed production arithmetic; no runtime precision selection or fallback.
pub const BF16: bool = false;
pub const TF32_BACKWARD: bool = true;
pub const FP32_AFFINE_REFERENCE: bool = false;
pub const TAG: &str =
    "FP16 forward GEMM; TF32 backward GEMM; FP32 accumulation, norms, master weights and moments";
