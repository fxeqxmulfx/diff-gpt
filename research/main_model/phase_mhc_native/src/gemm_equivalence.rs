//! Compare finish fusion with the frozen tensor kernels, including split-K,
//! output padding and canaries. Both sides use the same fixed geometry policy.
use crate::{Result, hash, pipeline_host, random::Random};
use kernite_cuda::{Cuda, Grid};
use serde_json::{Value, json};
use std::{fs, path::Path};

fn bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

pub fn run(cuda: &Cuda, root: &Path) -> Result<Vec<Value>> {
    let old_tensor = crate::parent_tensor::kernels::Kernels::load(cuda)?;
    let old_pipeline = crate::parent_pipeline::kernels::Kernels::load(cuda)?;
    let new_tensor = crate::tensor::kernels::Kernels::load(cuda)?;
    let mut rng = Random::new(84113);
    let mut records = Vec::new();
    for (m, n, k) in [
        (1usize, 1usize, 1usize),
        (3, 7, 5),
        (17, 24, 63),
        (31, 32, 32),
        (65, 128, 65),
        (512, 128, 97),
        (512, 384, 128),
        (128, 128, 512),
        (64, 24, 4096),
        (4096, 64, 128),
        (128, 342, 512),
        (342, 128, 1024),
        (17, 128, 512),
        (512, 128, 511),
        (129, 97, 1025),
    ] {
        let (np, kp, chunk, half_splits) = pipeline_host::shape(m, n, k);
        for pattern in ["zero", "signed-zero", "cancellation", "tiny", "random"] {
            let a: Vec<f32> = (0..m * k)
                .map(|i| match pattern {
                    "zero" => 0.,
                    "signed-zero" => {
                        if i % 2 == 0 {
                            -0.
                        } else {
                            0.
                        }
                    }
                    "cancellation" => {
                        if i % 2 == 0 {
                            1000.
                        } else {
                            -1000.
                        }
                    }
                    "tiny" => rng.gauss(0., 1e-8) as f32,
                    _ => rng.gauss(0., 0.4) as f32,
                })
                .collect();
            let b: Vec<f32> = (0..k * n)
                .map(|i| match pattern {
                    "zero" => 0.,
                    "signed-zero" => {
                        if i % 3 == 0 {
                            -0.
                        } else {
                            0.
                        }
                    }
                    "cancellation" => {
                        if i % 7 == 0 {
                            0.25
                        } else {
                            0.5
                        }
                    }
                    "tiny" => rng.gauss(0., 1e-8) as f32,
                    _ => rng.gauss(0., 0.4) as f32,
                })
                .collect();
            let ga = cuda.upload(&a)?;
            let gb = cuda.upload(&b)?;
            let packed_a = cuda.alloc::<u16>(m * kp)?;
            let packed_b = cuda.alloc::<u16>(kp * np)?;
            let partial = cuda.alloc::<f32>(half_splits * m * np)?;
            let candidate =
                pipeline_host::Pipeline::new(cuda, m * kp, kp * np, half_splits * m * np)?;
            for format in ["fp16", "bf16", "tf32"] {
                let sentinel = f32::from_bits(0x42edcba9);
                let parent_out = cuda.upload(&vec![sentinel; m * n + 17])?;
                let new_out = cuda.upload(&vec![sentinel; m * n + 17])?;
                let splits = if format == "tf32" {
                    k.div_ceil(chunk)
                } else {
                    half_splits
                };
                let direct = splits == 1 && (format == "tf32" || n == np);
                let grid = |size: usize| Grid::new(size.div_ceil(256) as u32, 256);
                macro_rules! half {
                    ($bf16:literal) => {
                        unsafe {
                            old_pipeline.pack::<$bf16>(
                                grid(m * kp),
                                packed_a.ptr(),
                                ga.ptr(),
                                m as u32,
                                k as u32,
                                m as u32,
                                kp as u32,
                            )?;
                            old_pipeline.pack::<$bf16>(
                                grid(kp * np),
                                packed_b.ptr(),
                                gb.ptr(),
                                k as u32,
                                n as u32,
                                kp as u32,
                                np as u32,
                            )?;
                            old_pipeline.hgemm::<$bf16>(
                                Grid::new((m.div_ceil(64) * np.div_ceil(64) * splits) as u32, 128)
                                    .with_shared(
                                        (crate::parent_pipeline::kernels::STAGES
                                            * crate::parent_pipeline::kernels::STAGE)
                                            as u32,
                                    ),
                                partial.ptr(),
                                packed_a.ptr(),
                                packed_b.ptr(),
                                m as u32,
                                np as u32,
                                kp as u32,
                                chunk as u32,
                            )?;
                            old_pipeline.finish(
                                grid(m * n),
                                parent_out.ptr(),
                                partial.ptr(),
                                m as u32,
                                n as u32,
                                np as u32,
                                splits as u32,
                            )?;
                        }
                        candidate.multiply::<$bf16>(new_out.ptr(), ga.ptr(), gb.ptr(), m, n, k)?;
                    };
                }
                match format {
                    "fp16" => {
                        half!(false);
                    }
                    "bf16" => {
                        half!(true);
                    }
                    "tf32" => {
                        unsafe {
                            old_tensor.gemm::<0>(
                                Grid::new((m.div_ceil(64) * n.div_ceil(64) * splits) as u32, 128),
                                partial.ptr(),
                                ga.ptr(),
                                gb.ptr(),
                                m as u32,
                                n as u32,
                                k as u32,
                                chunk as u32,
                            )?;
                            old_pipeline.finish(
                                grid(m * n),
                                parent_out.ptr(),
                                partial.ptr(),
                                m as u32,
                                n as u32,
                                n as u32,
                                splits as u32,
                            )?;
                        }
                        candidate.multiply_tf32(
                            &new_tensor,
                            new_out.ptr(),
                            ga.ptr(),
                            gb.ptr(),
                            (m, n, k),
                        )?;
                    }
                    _ => unreachable!(),
                }
                let (parent, new) = (parent_out.read()?, new_out.read()?);
                let different = parent
                    .iter()
                    .zip(&new)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                let finite = parent.iter().chain(&new).all(|x| x.is_finite());
                let canaries = parent[m * n..]
                    .iter()
                    .chain(&new[m * n..])
                    .all(|v| v.to_bits() == sentinel.to_bits());
                if different > 0 || !finite || !canaries {
                    let path = root.join(format!("gemm-{format}-{m}x{n}x{k}-{pattern}"));
                    fs::create_dir_all(&path)?;
                    for (name, values) in [("a", &a), ("b", &b), ("old", &parent), ("new", &new)] {
                        fs::write(path.join(format!("{name}.f32")), bytes(values))?;
                    }
                }
                records.push(json!({"kind":"gemm_finish","format":format,"m":m,"n":n,"k":k,"pattern":pattern,
                    "splits":splits,"direct_output":direct,"mismatches":different,"finite":finite,"canaries_intact":canaries,
                    "passed":different==0&&finite&&canaries,"input_a_sha256":hash(&bytes(&a)),"input_b_sha256":hash(&bytes(&b)),
                    "parent_output_sha256":hash(&bytes(&parent)),"new_output_sha256":hash(&bytes(&new))}));
            }
        }
    }
    Ok(records)
}
