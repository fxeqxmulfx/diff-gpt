//! Independent scalar AD checks: every base and added parameter group, including
//! both real and imaginary tied directions. Engineering runs never train a task.
use crate::{
    Config, Engine, Result, hash, layout, mhc,
    oracle::{self, Jet},
    random::Random,
    runner,
};
use kernite_cuda::{Cuda, Grid};
use serde_json::{Value, json};
use std::{fs, path::Path, time::Instant};
fn cfg(w: usize, heads: usize, layers: usize, mhc: bool) -> Config {
    Config {
        width: w,
        heads,
        layers,
        vocab: 11,
        max_batch: 2,
        max_length: 5,
        normalized: true,
        ssmax: true,
        gain: heads as f32 / w as f32,
        lr: 0.002,
        decay: 0.1,
        warmup: 50,
        architecture: if mhc { "complex-mhc" } else { "complex-linear" }.into(),
        phase_optimizer: false,
    }
}
fn blocks(c: &Config) -> Vec<(String, usize, usize)> {
    let w = c.width;
    let h = layout::hidden(w, "");
    let mut b = vec![("embedding/readout".into(), 0, c.vocab * w)];
    for l in 0..c.layers {
        let o = c.vocab * w + l * (4 * w * w + 3 * w * h);
        for (name, start, n) in [
            ("Q", o, w * w),
            ("K", o + w * w, w * w),
            ("V", o + 2 * w * w, w * w),
            ("attention output", o + 3 * w * w, w * w),
            ("product gate", o + 4 * w * w, h * w),
            ("product value", o + 4 * w * w + h * w, h * w),
            ("product output", o + 4 * w * w + 2 * h * w, w * h),
        ] {
            b.push((format!("layer{l}/{name}"), start, n));
        }
    }
    if layout::is_mhc(c) {
        for l in 0..2 * c.layers {
            let o = layout::router_offset(c, l);
            for (name, start, n) in [
                ("read projection", o, 16 * w),
                ("write projection", o + 16 * w, 16 * w),
                ("skip projection", o + 32 * w, 64 * w),
                ("read bias", o + 96 * w, 4),
                ("write bias", o + 96 * w + 4, 4),
                ("skip bias", o + 96 * w + 8, 16),
                ("read scale", o + 96 * w + 24, 1),
                ("write scale", o + 96 * w + 25, 1),
                ("skip scale", o + 96 * w + 26, 1),
            ] {
                b.push((format!("router{l}/{name}"), start, n));
            }
        }
    }
    b
}
fn scalar_gates() -> Result<Vec<Value>> {
    let cuda = Cuda::new()?;
    let k = mhc::kernels::Kernels::load(&cuda)?;
    let mut rng = Random::new(812);
    let mut result = vec![];
    for kind in [
        "zero",
        "random",
        "tiny",
        "large",
        "row-offset",
        "saturated",
        "permutation",
        "ill-conditioned",
    ] {
        let z: Vec<f32> = (0..24)
            .map(|i| match kind {
                "zero" => 0.,
                "tiny" => (rng.gauss(0., 1.) * 1e-8) as f32,
                "large" => rng.gauss(0., 20.) as f32,
                "row-offset" => {
                    if i >= 8 {
                        ((i - 8) / 4) as f32 * 1000. + rng.gauss(0., 1.) as f32
                    } else {
                        0.
                    }
                }
                "saturated" => {
                    if i % 2 == 0 {
                        10000.
                    } else {
                        -10000.
                    }
                }
                "permutation" => {
                    if i >= 8 && (i - 8) / 4 == (i - 8) % 4 {
                        80.
                    } else {
                        0.
                    }
                }
                "ill-conditioned" => {
                    if i >= 8 && (i - 8) % 4 == 0 {
                        80.
                    } else {
                        rng.gauss(0., 20.) as f32
                    }
                }
                _ => rng.gauss(0., 2.) as f32,
            })
            .collect();
        let upstream: Vec<f32> = (0..24).map(|_| rng.gauss(0., 1.) as f32).collect();
        let input = cuda.upload(&z)?;
        let up = cuda.upload(&upstream)?;
        let out = cuda.alloc::<f32>(24)?;
        let grad = cuda.alloc::<f32>(24)?;
        unsafe {
            k.sinkhorn::<false>(Grid::new(1, 32), out.ptr(), input.ptr(), up.ptr(), 1)?;
            k.sinkhorn::<true>(Grid::new(1, 32), grad.ptr(), input.ptr(), up.ptr(), 1)?;
        }
        let actual = out.read()?;
        let gradients = grad.read()?;
        let base: Vec<_> = z.iter().map(|&v| Jet::from(v as f64)).collect();
        let expected = oracle::mhc_gates(&base);
        let forward_error = actual
            .iter()
            .zip(&expected)
            .map(|(&a, b)| (a as f64 - b.v).abs())
            .fold(0., f64::max);
        let mut errors = vec![];
        for i in 0..24 {
            let mut dual = base.clone();
            dual[i].d = 1.;
            let ys = oracle::mhc_gates(&dual);
            let want: f64 = ys.iter().zip(&upstream).map(|(a, &b)| a.d * b as f64).sum();
            errors.push((gradients[i] as f64 - want).abs());
        }
        let max_error = errors.iter().copied().fold(0., f64::max);
        let row_error = (0..4)
            .map(|i| ((0..4).map(|j| actual[8 + i * 4 + j] as f64).sum::<f64>() - 1.).abs())
            .fold(0., f64::max);
        let column_error = (0..4)
            .map(|j| ((0..4).map(|i| actual[8 + i * 4 + j] as f64).sum::<f64>() - 1.).abs())
            .fold(0., f64::max);
        result.push(json!({"case":kind,"forward_max_error":forward_error,"gradient_max_error":max_error,"row_residual":row_error,"column_residual":column_error,"coordinate_checks":24,"passed":actual.iter().chain(&gradients).all(|v|v.is_finite())&&forward_error<2e-6&&max_error<2e-6&&row_error<2e-7}));
    }
    Ok(result)
}
fn full_case(w: usize, heads: usize, layers: usize, mhc: bool, variant: &str) -> Result<Value> {
    let c = cfg(w, heads, layers, mhc);
    let mut p = layout::initialize_model(&c, 1);
    if variant == "tiny-base" {
        for v in &mut p[..layout::base_count(&c)] {
            *v *= 0.001;
        }
    }
    if variant == "zero-embedding" {
        p[..c.vocab * w].fill(0.);
    }
    if mhc && variant == "dynamic" {
        let mut rng = Random::new(192);
        for l in 0..2 * layers {
            let o = layout::router_offset(&c, l);
            for v in &mut p[o..o + 96 * w] {
                *v *= 5.;
            }
            for v in &mut p[o + 96 * w..o + 96 * w + 24] {
                *v = rng.gauss(0., 2.) as f32;
            }
            p[o + 96 * w + 24..o + 96 * w + 27].fill(0.7);
        }
    }
    let (batch, length) = (2, if variant == "length-one" { 1 } else { 3 });
    let tokens: Vec<_> = (0..batch * length)
        .map(|i| {
            if variant == "repeated" {
                2
            } else {
                (i % 7 + 1) as u32
            }
        })
        .collect();
    let labels: Vec<_> = (0..batch * length)
        .map(|i| {
            if i % length + 1 == length {
                ((i + 3) % 11) as u32
            } else {
                u32::MAX
            }
        })
        .collect();
    let mut engine = Engine::new(c.clone(), &p)?;
    engine.forward(&tokens, &labels, batch, length)?;
    engine.backward(batch, length)?;
    let g = engine.gradients.read()?;
    let native = engine.logits.read()?;
    let zero: Vec<_> = p.iter().map(|&v| Jet::from(v as f64)).collect();
    let logits = oracle::main_model(&c, &zero, &tokens, batch, length);
    let forward_error = logits
        .iter()
        .zip(&native)
        .map(|(a, &b)| (a.v - b as f64).abs())
        .fold(0., f64::max);
    let forward_scale = logits.iter().map(|a| a.v.abs()).fold(1., f64::max);
    let mut contracted = g.iter().map(|&v| v as f64).collect::<Vec<_>>();
    let mut owners = vec![true; p.len()];
    for [ar, bi, br, ai, kind] in layout::pairs(&c) {
        if kind == 0 {
            contracted[ar as usize] += g[ai as usize] as f64;
            contracted[bi as usize] -= g[br as usize] as f64;
            owners[ai as usize] = false;
            owners[br as usize] = false;
        }
    }
    let mut directions = vec![];
    let mut rng = Random::new(51);
    for (name, start, n) in blocks(&c) {
        let ids: Vec<_> = (start..start + n).filter(|&i| owners[i]).collect();
        let norm = ids
            .iter()
            .map(|&i| contracted[i].powi(2))
            .sum::<f64>()
            .sqrt();
        for mode in 0..3 {
            let mut d = vec![0.; p.len()];
            if mode == 1 {
                let &idx = ids
                    .iter()
                    .max_by(|&&a, &&b| contracted[a].abs().total_cmp(&contracted[b].abs()))
                    .unwrap();
                d[idx] = 1.;
            } else {
                for &i in &ids {
                    d[i] = if mode == 2 {
                        contracted[i]
                    } else {
                        rng.gauss(0., 1.)
                    };
                }
                let scale = ids.iter().map(|&i| d[i] * d[i]).sum::<f64>().sqrt();
                if scale > 0. {
                    for &i in &ids {
                        d[i] /= scale;
                    }
                }
            }
            let actual: f64 = ids.iter().map(|&i| contracted[i] * d[i]).sum();
            for [ar, bi, br, ai, kind] in layout::pairs(&c) {
                if kind == 0 {
                    d[ai as usize] = d[ar as usize];
                    d[br as usize] = -d[bi as usize];
                }
            }
            let dual: Vec<_> = p
                .iter()
                .zip(&d)
                .map(|(&v, &d)| Jet { v: v as f64, d })
                .collect();
            let expected = oracle::cross_entropy(
                &oracle::main_model(&c, &dual, &tokens, batch, length),
                &labels,
                c.vocab,
            )
            .d;
            let error = (actual - expected).abs();
            let tolerance = 5e-4 * norm + 1e-12;
            directions.push(json!({"block":name,"direction":mode,"native":actual,"reference":expected,"block_norm":norm,"error":error,"tolerance":tolerance,"passed":actual.is_finite()&&expected.is_finite()&&error<=tolerance}));
        }
    }
    let passed =
        forward_error <= 2e-5 * forward_scale && directions.iter().all(|v| v["passed"] == true);
    Ok(
        json!({"arm":if mhc{"mhc"}else{"main"},"width":w,"heads":heads,"layers":layers,"variant":variant,"logit_error":forward_error,"directions":directions,"passed":passed}),
    )
}
pub fn run(output: &Path) -> Result<()> {
    if output.exists() {
        return Err("stress output exists; retain it and choose a new path".into());
    }
    fs::create_dir_all(output)?;
    for (name, body) in crate::sources::FILES {
        let path = output.join("source").join(name);
        fs::create_dir_all(path.parent().unwrap())?;
        fs::write(path, body)?;
    }
    let start = Instant::now();
    let mut primitive = scalar_gates()?;
    primitive.extend(rms_primitives()?);
    fs::write(
        output.join("primitives.json"),
        serde_json::to_vec_pretty(&primitive)?,
    )?;
    let mut cases = vec![];
    for (w, h, l, variant) in [
        (64, 4, 1, "normal"),
        (64, 4, 2, "normal"),
        (128, 4, 1, "normal"),
        (128, 1, 1, "normal"),
        (256, 8, 1, "normal"),
        (64, 4, 2, "dynamic"),
        (64, 4, 1, "tiny-base"),
        (64, 4, 1, "zero-embedding"),
        (64, 4, 1, "length-one"),
        (64, 4, 2, "repeated"),
    ] {
        for mhc in [false, true] {
            let row = full_case(w, h, l, mhc, variant)?;
            println!(
                "{}",
                json!({"width":w,"heads":h,"layers":l,"variant":variant,"mhc":mhc,"passed":row["passed"]})
            );
            fs::write(
                output.join(format!("case-{}.json", cases.len())),
                serde_json::to_vec_pretty(&row)?,
            )?;
            cases.push(row);
        }
    }
    let passed = primitive.iter().chain(&cases).all(|r| r["passed"] == true);
    let report = json!({"passed":passed,"source_sha256":runner::source_hash(),"ptx_sha256":runner::ptx_hash(),"seconds":start.elapsed().as_secs_f64(),"primitives":primitive,"models":cases,"oracle":"independent scalar FP64 forward-mode AD","training_updates":0});
    let bytes = serde_json::to_vec_pretty(&report)?;
    fs::write(output.join("validation.json"), &bytes)?;
    fs::write(output.join("validation.sha256"), hash(&bytes))?;
    if passed {
        Ok(())
    } else {
        Err("gradient validation failed; evidence retained".into())
    }
}

fn rms_primitives() -> Result<Vec<Value>> {
    let cuda = Cuda::new()?;
    let k = mhc::kernels::Kernels::load(&cuda)?;
    let mut rows = vec![];
    let mut rng = Random::new(831);
    for width in [256, 512, 1024] {
        for (name, amplitude, radial) in [
            ("zero", 0., false),
            ("tiny", 1e-20, false),
            ("epsilon-scale", 0.0003, false),
            ("normal", 1., false),
            ("large", 1e6, false),
            ("radial-large", 1e6, true),
            ("radial-huge", 1e20, true),
        ] {
            let x: Vec<f32> = (0..width)
                .map(|_| (rng.gauss(0., 1.) * amplitude) as f32)
                .collect();
            let dy: Vec<f32> = if radial {
                x.clone()
            } else {
                (0..width).map(|_| rng.gauss(0., 1.) as f32).collect()
            };
            let xgpu = cuda.upload(&x)?;
            let dygpu = cuda.upload(&dy)?;
            let y = cuda.alloc::<f32>(width)?;
            let dx = cuda.alloc::<f32>(width)?;
            unsafe {
                k.rms::<false>(
                    Grid::new(1, 128),
                    y.ptr(),
                    xgpu.ptr(),
                    dygpu.ptr(),
                    width as u32,
                    1,
                )?;
                k.rms::<true>(
                    Grid::new(1, 128),
                    dx.ptr(),
                    xgpu.ptr(),
                    dygpu.ptr(),
                    width as u32,
                    1,
                )?;
            }
            let actual = y.read()?;
            let gradient = dx.read()?;
            let energy = x.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / width as f64;
            let dot = x
                .iter()
                .zip(&dy)
                .map(|(&a, &b)| a as f64 * b as f64)
                .sum::<f64>()
                / width as f64;
            let inv = (energy + f32::EPSILON as f64).sqrt().recip();
            let expected: Vec<_> = x
                .iter()
                .zip(&dy)
                .map(|(&x, &dy)| {
                    if radial {
                        x as f64 * f32::EPSILON as f64 * inv.powi(3)
                    } else {
                        let parallel = if energy > 0. {
                            x as f64 * dot / energy
                        } else {
                            0.
                        };
                        (dy as f64 - parallel) * inv + parallel * f32::EPSILON as f64 * inv.powi(3)
                    }
                })
                .collect();
            let error = gradient
                .iter()
                .zip(&expected)
                .map(|(&a, &b)| (a as f64 - b).powi(2))
                .sum::<f64>()
                .sqrt();
            let norm = expected.iter().map(|v| v * v).sum::<f64>().sqrt();
            let forward = actual
                .iter()
                .zip(&x)
                .map(|(&a, &b)| (a as f64 - b as f64 * inv).abs())
                .fold(0., f64::max);
            let tolerance = 3e-6 * norm + (width as f64).sqrt() * f32::from_bits(1) as f64;
            rows.push(json!({"case":format!("router-rms/{width}/{name}"),"forward_max_error":forward,"gradient_error_norm":error,"reference_gradient_norm":norm,"tolerance":tolerance,"passed":forward<2e-6&&error<=tolerance&&gradient.iter().all(|v|v.is_finite())}));
        }
    }
    Ok(rows)
}
#[test]
fn paired_initialization_and_promoted_reference() {
    for (w, heads, layers) in [
        (64, 4, 2),
        (128, 4, 1),
        (128, 1, 1),
        (128, 4, 6),
        (256, 8, 1),
    ] {
        let mut base = cfg(w, heads, layers, false);
        base.vocab = 97;
        let mut candidate = base.clone();
        candidate.architecture = "complex-mhc".into();
        for seed in 0..3 {
            let b = layout::initialize_model(&base, seed);
            let c = layout::initialize_model(&candidate, seed);
            assert_eq!(&c[..b.len()], b);
            assert_eq!(c.len() - b.len(), 2 * layers * (96 * w + 27));
            layout::validate(&candidate, &c).unwrap();
            if w == 128 && heads == 4 && layers == 1 && seed == 1 {
                assert_eq!(
                    hash(&b.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>()),
                    "64eaf6fefaeb18df8ab77f6357529fe9a135fde0ad2b4dd9a7be4fed8bed4fd1"
                );
            }
        }
    }
}
