//! Numerical candidate checks at unchanged scalar-AD primitive tolerances.
//! Attention remains bit exact; Sinkhorn rounding is explicitly measured.
use crate::{Result, hash, mhc, random::Random, reference_mhc, runner};
use kernite_cuda::{Cuda, Grid};
use serde_json::json;
use std::{fs, path::Path};

pub fn run(output: &Path) -> Result<()> {
    if output.exists() {
        return Err("equivalence output already exists".into());
    }
    fs::create_dir_all(output)?;
    for (name, body) in crate::sources::FILES {
        let path = output.join("source").join(name);
        fs::create_dir_all(path.parent().unwrap())?;
        fs::write(path, body)?;
    }
    let cuda = Cuda::new()?;
    let old = reference_mhc::kernels::Kernels::load(&cuda)?;
    let new = mhc::kernels::Kernels::load(&cuda)?;
    let mut rng = Random::new(9281);
    let mut cases = Vec::new();
    for rows in [1usize, 2, 3, 7, 16, 17, 127, 512, 1025] {
        for pattern in [
            "zero",
            "random",
            "tiny",
            "large",
            "row-offset",
            "saturated",
            "permutation",
            "ill-conditioned",
        ] {
            let x: Vec<f32> = (0..rows * 24)
                .map(|i| {
                    let c = i % 24;
                    let r = rng.gauss(0., 2.) as f32;
                    match pattern {
                        "zero" => 0.,
                        "tiny" => r * 1e-20,
                        "large" => r * 20.,
                        "row-offset" if c >= 8 => 1000. * ((c - 8) / 4) as f32 + r,
                        "saturated" => {
                            if i % 2 == 0 {
                                10000.
                            } else {
                                -10000.
                            }
                        }
                        "permutation" if c >= 8 && (c - 8) / 4 == (c - 8) % 4 => 80.,
                        "ill-conditioned" if c >= 8 && (c - 8) % 4 == 0 => 80.,
                        _ => r,
                    }
                })
                .collect();
            let upstream: Vec<f64> = (0..rows * 24)
                .map(|i| rng.gauss(0., if i % 3 == 0 { 1e-20 } else { 1. }))
                .collect();
            let input = cuda.upload(&x)?;
            let up = cuda.upload(&upstream)?;
            let original = cuda.alloc::<f32>(rows * 24)?;
            let packed = cuda.alloc::<f32>(rows * 24)?;
            let original_grad = cuda.alloc::<f32>(rows * 24)?;
            let packed_grad = cuda.alloc::<f32>(rows * 24)?;
            // Same mathematics and arguments, different declared fixed arithmetic.
            unsafe {
                old.sinkhorn::<false>(
                    Grid::new(rows as u32, 32),
                    original.ptr(),
                    input.ptr(),
                    up.ptr(),
                    rows as u32,
                )?;
                new.sinkhorn::<false>(
                    Grid::new(rows.div_ceil(2) as u32, 32),
                    packed.ptr(),
                    input.ptr(),
                    up.ptr(),
                    rows as u32,
                )?;
                old.sinkhorn::<true>(
                    Grid::new(rows as u32, 32),
                    original_grad.ptr(),
                    input.ptr(),
                    up.ptr(),
                    rows as u32,
                )?;
                new.sinkhorn::<true>(
                    Grid::new(rows.div_ceil(2) as u32, 32),
                    packed_grad.ptr(),
                    input.ptr(),
                    up.ptr(),
                    rows as u32,
                )?;
            }
            let a = original.read()?;
            let b = packed.read()?;
            let da = original_grad.read()?;
            let db = packed_grad.read()?;
            let forward_mismatches = a
                .iter()
                .zip(&b)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            let backward_mismatches = da
                .iter()
                .zip(&db)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            let finite = a
                .iter()
                .chain(&b)
                .chain(&da)
                .chain(&db)
                .all(|x| x.is_finite());
            let forward_error = a
                .iter()
                .zip(&b)
                .map(|(&a, &b)| (a as f64 - b as f64).abs())
                .fold(0., f64::max);
            let backward_error = da
                .iter()
                .zip(&db)
                .map(|(&a, &b)| (a as f64 - b as f64).abs())
                .fold(0., f64::max);
            let row_error = b
                .chunks_exact(24)
                .flat_map(|row| {
                    (0..4).map(move |i| {
                        (row[8 + i * 4..12 + i * 4]
                            .iter()
                            .map(|&x| x as f64)
                            .sum::<f64>()
                            - 1.)
                            .abs()
                    })
                })
                .fold(0., f64::max);
            let raw = output.join(format!("rows{rows}-{pattern}"));
            fs::create_dir_all(&raw)?;
            for (name, values) in [
                ("logits", &x),
                ("reference", &a),
                ("candidate", &b),
                ("reference-gradient", &da),
                ("candidate-gradient", &db),
            ] {
                fs::write(
                    raw.join(format!("{name}.f32")),
                    values
                        .iter()
                        .flat_map(|x| x.to_le_bytes())
                        .collect::<Vec<_>>(),
                )?;
            }
            fs::write(
                raw.join("upstream.f64"),
                upstream
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            cases.push(
                json!({"rows":rows,"pattern":pattern,"forward_mismatches":forward_mismatches,
                "backward_mismatches":backward_mismatches,"finite":finite,
                "forward_max_error":forward_error,"backward_max_error":backward_error,
                "row_residual":row_error,
                "passed":finite && forward_error<2e-6 && backward_error<2e-6 && row_error<2e-7}),
            );
        }
    }
    cases.extend(attention_cases(&cuda, &mut rng)?);
    let passed = cases.iter().all(|c| c["passed"] == true);
    let report = json!({"passed":passed,"cases":cases,"criterion":"Sinkhorn absolute forward/backward error<2e-6 and row residual<2e-7; unchanged attention bit identical",
        "source_sha256":runner::source_hash(),"ptx_sha256":runner::ptx_hash(),
        "reference_mhc_ptx_sha256":hash(reference_mhc::kernels::PTX.as_bytes()),
        "optimized_mhc_ptx_sha256":hash(mhc::kernels::PTX.as_bytes())});
    fs::write(
        output.join("report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{}", json!({"passed":passed,"cases":cases.len()}));
    if passed {
        Ok(())
    } else {
        Err("mixed Sinkhorn numerical checks failed; evidence retained".into())
    }
}

fn attention_cases(cuda: &Cuda, rng: &mut Random) -> Result<Vec<serde_json::Value>> {
    let old = crate::reference_ordinary::kernels::Kernels::load(cuda)?;
    let new = crate::ordinary_attention::kernels::Kernels::load(cuda)?;
    let old_norm = crate::reference_attention::kernels::Kernels::load(cuda)?;
    let new_norm = crate::attention::kernels::Kernels::load(cuda)?;
    let mut results = Vec::new();
    for (d, heads) in [(16usize, 4usize), (32, 4), (128, 1)] {
        for length in [1usize, 2, 5, 64] {
            for pattern in ["random", "tiny", "large", "repeated"] {
                let batch = 2;
                let rows = batch * length;
                let w = heads * d;
                let scale = match pattern {
                    "tiny" => 1e-20,
                    "large" => 1e6,
                    _ => 1.,
                };
                let qkv: Vec<f32> = (0..rows * 3 * w)
                    .map(|i| {
                        if pattern == "repeated" {
                            ((i % (3 * w)) as f32 - w as f32) * 0.01
                        } else {
                            (rng.gauss(0., 0.3) * scale) as f32
                        }
                    })
                    .collect();
                let dy: Vec<f32> = (0..rows * w)
                    .map(|_| (rng.gauss(0., 0.1) * scale) as f32)
                    .collect();
                let norm_dy: Vec<f32> = (0..rows * 3 * w)
                    .map(|_| rng.gauss(0., 0.1) as f32)
                    .collect();
                let mut probs = vec![0f32; batch * heads * length * length];
                for bh in 0..batch * heads {
                    for t in 0..length {
                        for key in 0..=t {
                            probs[(bh * length + t) * length + key] = 1. / (t + 1) as f32;
                        }
                    }
                }
                let x = cuda.upload(&qkv)?;
                let grad = cuda.upload(&dy)?;
                let ng = cuda.upload(&norm_dy)?;
                let p = cuda.upload(&probs)?;
                let mut reference = Vec::new();
                let mut packed = Vec::new();
                for size in [rows * 3 * w, probs.len(), rows * w, rows * w, rows * 3 * w] {
                    reference.push(cuda.upload(&vec![0f32; size])?);
                    packed.push(cuda.upload(&vec![0f32; size])?);
                }
                macro_rules! launch {
                    ($d:literal) => {
                        unsafe {
                            old.query_backward::<$d>(
                                Grid::new((rows * heads) as u32, 32),
                                reference[0].ptr(),
                                reference[1].ptr(),
                                reference[2].ptr(),
                                reference[3].ptr(),
                                grad.ptr(),
                                p.ptr(),
                                x.ptr(),
                                heads as u32,
                                length as u32,
                            )?;
                            new.query_backward::<$d>(
                                Grid::new((rows * heads / if $d == 16 { 2 } else { 1 }) as u32, 32),
                                packed[0].ptr(),
                                packed[1].ptr(),
                                packed[2].ptr(),
                                packed[3].ptr(),
                                grad.ptr(),
                                p.ptr(),
                                x.ptr(),
                                heads as u32,
                                length as u32,
                            )?;
                            old_norm.norm_backward::<$d>(
                                Grid::new((rows * 3 * heads) as u32, 32),
                                reference[4].ptr(),
                                x.ptr(),
                                ng.ptr(),
                                heads as u32,
                            )?;
                            new_norm.norm_backward::<$d>(
                                Grid::new(
                                    (rows * 3 * heads / if $d == 16 { 2 } else { 1 }) as u32,
                                    32,
                                ),
                                packed[4].ptr(),
                                x.ptr(),
                                ng.ptr(),
                                heads as u32,
                            )?;
                        }
                    };
                }
                match d {
                    16 => launch!(16),
                    32 => launch!(32),
                    128 => launch!(128),
                    _ => unreachable!(),
                }
                let mut mismatches = 0;
                let mut finite = true;
                for (a, b) in reference.iter().zip(&packed) {
                    let a = a.read()?;
                    let b = b.read()?;
                    finite &= a.iter().chain(&b).all(|x| x.is_finite());
                    mismatches += a
                        .iter()
                        .zip(&b)
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count();
                }
                results.push(
                    json!({"kind":"attention_query_and_qk_norm_backward","head_width":d,
                    "heads":heads,"length":length,"pattern":pattern,"bit_mismatches":mismatches,
                    "finite":finite,"passed":finite && mismatches==0}),
                );
            }
        }
    }
    Ok(results)
}
