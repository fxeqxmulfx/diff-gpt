//! Native tensor-core numerical and throughput screen; no scientific training.
mod pipeline;
mod pipeline_host;
#[path = "../../../complex_geometry_suite/native/src/random.rs"]
#[allow(dead_code)]
mod random;
#[path = "../../../complex_geometry_suite/native/src/gemm.rs"]
mod reference;
mod tensor;
use kernite_cuda::{Buffer, Cuda, Grid};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{fs, path::Path, time::Instant};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|v| format!("{v:02x}"))
        .collect()
}
fn round_half(x: f32) -> f32 {
    let bits = x.to_bits();
    let sign = bits & 0x80000000;
    let exponent = ((bits >> 23) & 255) as i32 - 127 + 15;
    let mantissa = bits & 0x7fffff;
    let half = if exponent >= 31 {
        0x7c00
    } else if exponent < -10 {
        0
    } else if exponent <= 0 {
        let m = mantissa | 0x800000;
        let shift = (14 - exponent) as u32;
        (m + (1 << (shift - 1)) - 1 + ((m >> shift) & 1)) >> shift
    } else {
        ((exponent as u32) << 10) + ((mantissa + 4095 + ((mantissa >> 13) & 1)) >> 13)
    };
    let e = (half >> 10) & 31;
    let m = half & 1023;
    let magnitude = if e == 0 {
        (m as f32) * (2.0f32).powi(-24)
    } else if e == 31 {
        f32::INFINITY
    } else {
        f32::from_bits(((e + 112) << 23) | (m << 13))
    };
    f32::from_bits(sign | magnitude.to_bits())
}
fn quantize(x: f32, format: usize) -> f32 {
    match format {
        0 => x,
        1 => f32::from_bits(x.to_bits().wrapping_add(4096) & !8191), // TF32 RNA
        2 => round_half(x),
        3 => {
            f32::from_bits(x.to_bits().wrapping_add(32767 + ((x.to_bits() >> 16) & 1)) & 0xffff0000)
        }
        _ => unreachable!(),
    }
}
fn exact(a: &[f32], b: &[f32], m: usize, n: usize, k: usize, format: usize) -> Vec<f64> {
    let a: Vec<_> = a.iter().map(|&x| quantize(x, format) as f64).collect();
    let b: Vec<_> = b.iter().map(|&x| quantize(x, format) as f64).collect();
    let mut out = vec![0.; m * n];
    for i in 0..m {
        for p in 0..k {
            for j in 0..n {
                out[i * n + j] += a[i * k + p] * b[p * n + j];
            }
        }
    }
    out
}
fn raw(path: &Path, values: &[f32]) -> Result<()> {
    fs::write(
        path,
        values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    Ok(())
}
struct Matrices {
    a: Buffer<f32>,
    b: Buffer<f32>,
    out: Buffer<f32>,
    m: usize,
    n: usize,
    k: usize,
    pipeline: pipeline_host::Pipeline,
}
fn launch(
    t: &tensor::kernels::Kernels,
    r: &reference::kernels::Kernels,
    x: &Matrices,
    format: usize,
) -> Result<()> {
    unsafe {
        match format {
            0 if x.n.is_multiple_of(4) && x.k.is_multiple_of(4) => r.sgemm::<true>(
                Grid::new(
                    (x.m.div_ceil(128) * x.n.div_ceil(128)) as u32,
                    reference::kernels::THREADS as u32,
                ),
                x.out.ptr(),
                x.a.ptr(),
                x.b.ptr(),
                x.m as u32,
                x.n as u32,
                x.k as u32,
            )?,
            0 => r.sgemm::<false>(
                Grid::new(
                    (x.m.div_ceil(128) * x.n.div_ceil(128)) as u32,
                    reference::kernels::THREADS as u32,
                ),
                x.out.ptr(),
                x.a.ptr(),
                x.b.ptr(),
                x.m as u32,
                x.n as u32,
                x.k as u32,
            )?,
            1 => x
                .pipeline
                .multiply_tf32(t, x.out.ptr(), x.a.ptr(), x.b.ptr(), (x.m, x.n, x.k))?,
            2 => x
                .pipeline
                .multiply::<false>(x.out.ptr(), x.a.ptr(), x.b.ptr(), x.m, x.n, x.k)?,
            3 => x
                .pipeline
                .multiply::<true>(x.out.ptr(), x.a.ptr(), x.b.ptr(), x.m, x.n, x.k)?,
            _ => unreachable!(),
        }
    }
    Ok(())
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 || args[1] != "screen" {
        return Err("usage: main-tensor-screen screen FRESH_ROOT".into());
    }
    let root = Path::new(&args[2]);
    if root.exists() {
        return Err("screen already exists".into());
    }
    fs::create_dir_all(root)?;
    for (name, bytes) in [
        ("main.rs", include_str!("main.rs")),
        ("tensor.rs", include_str!("tensor.rs")),
        ("pipeline.rs", include_str!("pipeline.rs")),
        ("pipeline_host.rs", include_str!("pipeline_host.rs")),
        ("Cargo.toml", include_str!("../Cargo.toml")),
        ("Cargo.lock", include_str!("../Cargo.lock")),
        ("PLAN.txt", include_str!("../../TENSOR_TILE_PLAN.txt")),
        (
            "reference.rs",
            include_str!("../../../complex_geometry_suite/native/src/gemm.rs"),
        ),
        (
            "random.rs",
            include_str!("../../../complex_geometry_suite/native/src/random.rs"),
        ),
        ("tensor.ptx", tensor::kernels::PTX),
        ("pipeline.ptx", pipeline::kernels::PTX),
        ("reference.ptx", reference::kernels::PTX),
    ] {
        fs::write(root.join(name), bytes)?;
    }
    let cuda = Cuda::new()?;
    let t = tensor::kernels::Kernels::load(&cuda)?;
    let r = reference::kernels::Kernels::load(&cuda)?;
    let mut rng = random::Random::new(8821);
    let mut rows = Vec::new();
    for (m, n, k) in [
        (1, 1, 1),
        (7, 9, 5),
        (16, 8, 8),
        (65, 67, 33),
        (128, 97, 128),
        (512, 384, 128),
        (512, 684, 128),
        (128, 128, 512),
        (64, 24, 512),
        (4096, 192, 64),
        (128, 128, 4096),
    ] {
        let dir = root.join(format!("{m}x{n}x{k}"));
        fs::create_dir_all(&dir)?;
        let a: Vec<f32> = (0..m * k).map(|_| rng.gauss(0., 0.1) as f32).collect();
        let b: Vec<f32> = (0..k * n).map(|_| rng.gauss(0., 0.1) as f32).collect();
        raw(&dir.join("a.f32"), &a)?;
        raw(&dir.join("b.f32"), &b)?;
        let (np, kp, _, splits) = pipeline_host::shape(m, n, k);
        let x = Matrices {
            a: cuda.upload(&a)?,
            b: cuda.upload(&b)?,
            out: cuda.alloc(m * n)?,
            m,
            n,
            k,
            pipeline: pipeline_host::Pipeline::new(&cuda, m * kp, kp * np, splits * m * np)?,
        };
        let original = exact(&a, &b, m, n, k, 0);
        let mut formats = Vec::new();
        for (format, name) in ["fp32", "tf32", "fp16", "bf16"].iter().enumerate() {
            let wanted = if format == 0 {
                original.clone()
            } else {
                exact(&a, &b, m, n, k, format)
            };
            launch(&t, &r, &x, format)?;
            let got = x.out.read()?;
            raw(&dir.join(format!("{name}.f32")), &got)?;
            let norm = wanted.iter().map(|v| v * v).sum::<f64>().sqrt();
            let error = got
                .iter()
                .zip(&wanted)
                .map(|(&v, &w)| (v as f64 - w).powi(2))
                .sum::<f64>()
                .sqrt();
            let original_norm = original.iter().map(|v| v * v).sum::<f64>().sqrt();
            let quantization = wanted
                .iter()
                .zip(&original)
                .map(|(&v, &w)| (v - w).powi(2))
                .sum::<f64>()
                .sqrt();
            let finite = got.iter().all(|v| v.is_finite());
            let tolerance = 2e-5 * norm + 1e-7;
            formats.push(json!({"format":name,"finite":finite,"rounded_input_reference_error_norm":error,
                "reference_norm":norm,"tolerance":tolerance,"passed":finite&&error<=tolerance,
                "input_quantization_relative_l2":quantization/original_norm.max(1e-300),"timings_ms":[]}));
        }
        // Alternating format order, three repetitions; includes conversion in the kernel.
        for rep in 0..3 {
            let order = if rep % 2 == 0 {
                [0, 1, 2, 3]
            } else {
                [3, 2, 1, 0]
            };
            for format in order {
                for _ in 0..5 {
                    launch(&t, &r, &x, format)?;
                }
                cuda.sync()?;
                let start = Instant::now();
                for _ in 0..50 {
                    launch(&t, &r, &x, format)?;
                }
                cuda.sync()?;
                formats[format]["timings_ms"]
                    .as_array_mut()
                    .unwrap()
                    .push(json!(start.elapsed().as_secs_f64() * 1000. / 50.));
            }
        }
        for format in &mut formats {
            let mut times: Vec<f64> = format["timings_ms"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect();
            times.sort_by(f64::total_cmp);
            format["median_ms"] = json!(times[1]);
        }
        let row = json!({"m":m,"n":n,"k":k,"pipeline_shape":pipeline_host::shape(m,n,k),"formats":formats});
        fs::write(dir.join("report.json"), serde_json::to_vec_pretty(&row)?)?;
        rows.push(row);
    }
    let passed = rows.iter().all(|r| {
        r["formats"]
            .as_array()
            .unwrap()
            .iter()
            .all(|f| f["passed"] == true)
    });
    let report = json!({"passed":passed,"gpu":cuda.name(),"sm":cuda.sm(),"rows":rows,
        "ptx_sha256":hash(tensor::kernels::PTX.as_bytes()),"pipeline_ptx_sha256":hash(pipeline::kernels::PTX.as_bytes()),
        "fp64_in_candidate_ptx":tensor::kernels::PTX.contains(".f64")||pipeline::kernels::PTX.contains(".f64"),
        "executable_sha256":hash(&fs::read(std::env::current_exe()?)?),
        "scope":"isolated GEMM, conversion included, FP32 accumulation; not whole-model throughput"});
    fs::write(
        root.join("report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{}", json!({"passed":passed,"shapes":rows.len()}));
    if passed {
        Ok(())
    } else {
        Err("tensor numerical failures retained".into())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn independent_input_rounding_boundaries() {
        use super::{quantize, round_half};
        assert_eq!(round_half(1.0), 1.0);
        assert_eq!(round_half(1. + 2f32.powi(-11)), 1.0);
        assert_eq!(round_half(1. + 3. * 2f32.powi(-11)), 1. + 2f32.powi(-9));
        assert_eq!(round_half(2f32.powi(-24)), 2f32.powi(-24));
        assert_eq!(round_half(2f32.powi(-25)), 0.0);
        assert_eq!(round_half(65504.), 65504.);
        assert_eq!(round_half(65520.), f32::INFINITY);
        assert_eq!(round_half(-0.).to_bits(), (-0f32).to_bits());
        assert_eq!(quantize(1. + 2f32.powi(-11), 1), 1. + 2f32.powi(-10));
        assert_eq!(quantize(1. + 2f32.powi(-8), 3), 1.);
        assert_eq!(quantize(1. + 3. * 2f32.powi(-8), 3), 1. + 2f32.powi(-6));
    }
}
