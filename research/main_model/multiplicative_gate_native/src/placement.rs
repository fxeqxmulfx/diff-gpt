pub const FUSED_ATTENTION: bool = cfg!(feature = "fused-attention");
pub const TILED_TRANSPOSE: bool = cfg!(feature = "tiled-transpose");
pub const TAG: &str = match (FUSED_ATTENTION, TILED_TRANSPOSE) {
    (false, false) => {
        "packed query/norm/key/routing/RMS warps, ordered ballot embedding gradient and single-partition GEMM finish fusion"
    }
    (true, false) => {
        "paired-D16 attention; packed forward/query/norm/key/routing/RMS warps; ordered ballot embedding gradient; single-partition GEMM finish fusion"
    }
    (false, true) => {
        "tiled transpose; packed query/norm/key/routing/RMS warps; ordered ballot embedding gradient; single-partition GEMM finish fusion"
    }
    (true, true) => {
        "paired-D16 ordinary attention fusion, tiled transpose, packed forward/query/norm/key/routing/RMS warps, ordered ballot embedding gradient and single-partition GEMM finish fusion"
    }
};
