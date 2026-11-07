//! GPU regression cases independent of the Python whole-model oracle.
use crate::{gemm, ops};
use kernite_cuda::{Cuda, Grid};

#[test]
#[ignore = "requires a CUDA GPU; run explicitly with --ignored"]
fn repaired_rms_keeps_zero_and_radial_derivatives() -> Result<(), Box<dyn std::error::Error>> {
    let cuda = Cuda::new()?;
    let k = ops::kernels::Kernels::load(&cuda)?;
    for width in [128, 256] {
        for amplitude in [0.0f32, 1e-7, 0.02, 1.0, 1e10] {
            for radial in [true, false] {
                let x: Vec<f32> = (0..width)
                    .map(|i| amplitude * if i % 2 == 0 { 1.0 } else { -1.0 })
                    .collect();
                let g: Vec<f32> = if radial {
                    x.clone()
                } else {
                    (0..width).map(|i| (i as f32 * 0.17).sin()).collect()
                };
                let bx = cuda.upload(&x)?;
                let bg = cuda.upload(&g)?;
                let out = cuda.alloc::<f32>(width)?;
                // SAFETY: one 128-thread block contains four warps; only the first row is
                // in range. Both kernels use exactly the width elements allocated above.
                unsafe {
                    if width == 128 {
                        k.rms_bwd::<128>(Grid::new(1, 128), out.ptr(), bx.ptr(), bg.ptr(), 1)?;
                    } else {
                        k.rms_bwd::<256>(Grid::new(1, 128), out.ptr(), bx.ptr(), bg.ptr(), 1)?;
                    }
                }
                let e = x.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / width as f64;
                let dot = x
                    .iter()
                    .zip(&g)
                    .map(|(&v, &u)| v as f64 * u as f64)
                    .sum::<f64>()
                    / width as f64;
                let inv = (e + f32::EPSILON as f64).sqrt().recip();
                let reference: Vec<f64> = x
                    .iter()
                    .zip(&g)
                    .map(|(&v, &u)| {
                        let parallel = v as f64 * dot / if e > 0.0 { e } else { 1.0 };
                        inv * (u as f64 - parallel) + f32::EPSILON as f64 * inv.powi(3) * parallel
                    })
                    .collect();
                let actual = out.read()?;
                let error = actual
                    .iter()
                    .zip(&reference)
                    .map(|(&a, &b)| (a as f64 - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let norm = reference.iter().map(|v| v * v).sum::<f64>().sqrt();
                assert!(actual.iter().all(|v| v.is_finite()));
                assert!(
                    if norm == 0.0 {
                        error == 0.0
                    } else {
                        error / norm < 1e-6
                    },
                    "width={width}, amplitude={amplitude}, radial={radial}, error={error}, norm={norm}"
                );
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires a CUDA GPU; run explicitly with --ignored"]
fn softmax_gradient_and_argmax_write_misses_after_hits() -> Result<(), Box<dyn std::error::Error>> {
    let cuda = Cuda::new()?;
    let k = ops::kernels::Kernels::load(&cuda)?;
    let logits = cuda.alloc::<f32>(194)?;
    let grad = cuda.alloc::<f32>(194)?;
    let losses = cuda.alloc::<f32>(1)?;
    let hits = cuda.alloc::<u32>(1)?;
    let mut targets = cuda.upload(&[0u32])?;
    for target in [0, 96, 0, 1] {
        targets.write(&[target])?;
        // SAFETY: the input/output buffers cover one example and all 97 classes. Invalid
        // warps return before reading; all lanes of the valid warp participate in reductions.
        unsafe {
            k.loss(
                Grid::new(1, 128),
                grad.ptr(),
                losses.ptr(),
                hits.ptr(),
                logits.ptr(),
                targets.ptr(),
                1,
            )?;
        }
        let g = grad.read()?;
        assert!(g[..97].iter().all(|&x| x == 0.0));
        for c in 0..97 {
            let want = 1.0f32 / 97.0 - if c == target as usize { 1.0 } else { 0.0 };
            assert!((g[97 + c] - want).abs() < 1e-7);
        }
        assert_eq!(hits.read()?, vec![u32::from(target == 0)]);
        assert!((losses.read()?[0] - 97f32.ln()).abs() < 1e-6);
    }
    Ok(())
}

#[test]
#[ignore = "requires a CUDA GPU; run explicitly with --ignored"]
fn sgemm_uses_fp32_products_and_handles_partial_tiles() -> Result<(), Box<dyn std::error::Error>> {
    let cuda = Cuda::new()?;
    let kernels = gemm::kernels::Kernels::load(&cuda)?;
    for (m, n, k) in [(5usize, 7usize, 9usize), (96, 97, 128), (192, 256, 683)] {
        let a: Vec<f32> = (0..m * k)
            .map(|i| (i as f32 * 0.013).sin() * 0.02)
            .collect();
        let b: Vec<f32> = (0..k * n)
            .map(|i| (i as f32 * 0.019).cos() * 0.02)
            .collect();
        let aa = cuda.upload(&a)?;
        let bb = cuda.upload(&b)?;
        let c = cuda.alloc::<f32>(m * n)?;
        let grid = Grid::new((m.div_ceil(128) * n.div_ceil(128)) as u32, 256);
        // SAFETY: m*k, k*n, and m*n initialized elements cover the exact product; the
        // unaligned instance supports the odd reduction and partial output tiles.
        unsafe {
            kernels.sgemm::<false>(
                grid,
                c.ptr(),
                aa.ptr(),
                bb.ptr(),
                m as u32,
                n as u32,
                k as u32,
            )?;
        }
        let actual = c.read()?;
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for q in 0..k {
                    s = a[i * k + q].mul_add(b[q * n + j], s);
                }
                assert_eq!(actual[i * n + j], s, "m={m} n={n} k={k} i={i} j={j}");
            }
        }
    }
    Ok(())
}
