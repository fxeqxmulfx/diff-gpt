//! The kernels of the training engine, written in Kernite's language and
//! compiled to x86-64-v3 assembly (AVX2 and FMA): the inner loop of the
//! matrix product, where almost all the time of training goes.
//!
//! The micro-kernel is that of Kernite's `matmul` example
//! (<https://github.com/fxeqxmulfx/kernite>, MIT), a tile of 6 rows by 16
//! columns kept in twelve registers, as BLIS has it for Haswell; here it is a
//! plain kernel for one thread, and [`crate::gemm`] packs and blocks around
//! it.

// Kernite writes each kernel as a naked function, and calling one is unsafe:
// the callers are in `gemm`, which checks every length first.
#![allow(unsafe_code)]

pub use x86::{MR, NR, micro};

/// The kernels for x86-64-v3.
#[kernite_macros::kernels]
pub mod x86 {
    /// Rows of the product the micro-kernel keeps in registers.
    pub const MR: usize = 6;
    /// Columns of the product it keeps in registers: two vectors of eight a
    /// row.
    pub const NR: usize = 16;
    /// Steps of the inner dimension a turn of the loop takes.
    pub const UNROLL: usize = 4;

    /// The tile of `MR` rows of `NR` floats at `c`, rows `ld` bytes apart,
    /// becomes the product of the `kc` steps of the slivers at `x` and `y`,
    /// plus what the tile held if `keep` is not 0.
    ///
    /// `x` holds `kc` steps of `MR` floats, a column of the left operand
    /// each; `y` holds `kc` steps of `NR` floats, a row of the right operand
    /// each. Each element of the tile is a chain of fused multiply-adds over
    /// the steps in order.
    pub unsafe extern "sysv64" fn micro(
        c: *mut f32,
        ld: usize,
        x: *const f32,
        y: *const f32,
        kc: usize,
        keep: usize,
    ) {
        let acc: [ymm; 2 * MR];
        for i in 0..2 * MR {
            zero(acc[i]);
        }
        test(keep, keep);
        jz(steps);
        let from: gpr;
        mov(from, c);
        for r in 0..MR {
            for half in 0..2 {
                vmovups(acc[2 * r + half], [from + 32 * half]);
            }
            add(from, ld);
        }

        label(steps);
        let (px, py, left): (gpr, gpr, gpr);
        mov(px, x);
        mov(py, y);
        mov(left, kc);
        sub(left, UNROLL);
        jb(rest);
        label(turn);
        for s in 0..UNROLL {
            step::<s>(acc, px, py);
        }
        add(px, 4 * MR * UNROLL);
        add(py, 4 * NR * UNROLL);
        sub(left, UNROLL);
        jae(turn);
        label(rest);
        add(left, UNROLL);
        jz(store);
        label(one);
        step::<0>(acc, px, py);
        add(px, 4 * MR);
        add(py, 4 * NR);
        dec(left);
        jnz(one);

        label(store);
        let to: gpr;
        mov(to, c);
        for r in 0..MR {
            for half in 0..2 {
                vmovups([to + 32 * half], acc[2 * r + half]);
            }
            add(to, ld);
        }
    }

    /// Step `S` of a turn: the row of the right sliver at `y` by each of the
    /// `MR` floats of the left sliver at `x`, added to `acc`.
    fn step<const S: usize>(acc: [ymm; 2 * MR], x: gpr, y: gpr) {
        let (b0, b1): (ymm, ymm);
        vmovups(b0, [y + 4 * NR * S]);
        vmovups(b1, [y + 4 * NR * S + 32]);
        for r in 0..MR {
            let s: ymm;
            vbroadcastss(s, [x + 4 * (MR * S + r)]);
            vfmadd231ps(acc[2 * r], s, b0);
            vfmadd231ps(acc[2 * r + 1], s, b1);
        }
    }
}

/// Whether this CPU has what the kernels need: AVX2 and FMA.
#[must_use]
pub fn supported() -> bool {
    std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
}

/// The tile of [`MR`] rows of [`NR`] floats at the start of `tile`, its rows
/// `stride` floats apart, becomes the product of the `steps` steps of the
/// slivers `left` and `right`, plus what it held if `keep`: [`micro`], with
/// its lengths checked.
///
/// # Panics
///
/// If a slice is shorter than the kernel reads or writes.
pub fn tile(
    tile: &mut [f32],
    stride: usize,
    left: &[f32],
    right: &[f32],
    steps: usize,
    keep: bool,
) {
    assert!(
        tile.len() >= (MR - 1) * stride + NR,
        "the tile lies in its slice"
    );
    assert!(left.len() >= steps * MR, "the left sliver has its steps");
    assert!(right.len() >= steps * NR, "the right sliver has its steps");
    // SAFETY: the kernel reads `steps * MR` floats of `left` and
    // `steps * NR` of `right`, and reads and writes `MR` rows of `NR` floats
    // of `tile`, `4 * stride` bytes apart: all checked above. The slices do
    // not overlap, as one is borrowed mutably. The CPU has AVX2 and FMA: the
    // crate is compiled for x86-64-v3, and `Gemm::new` asks the CPU.
    unsafe {
        micro(
            tile.as_mut_ptr(),
            4 * stride,
            left.as_ptr(),
            right.as_ptr(),
            steps,
            usize::from(keep),
        );
    }
}
