use crate::attention;
use crate::{
    engine::{Config, Engine},
    layout,
};
use kernite_cuda::{Cuda, Grid};
#[path = "geometry_tests.rs"]
mod geometry_tests;
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn rope(d: usize, t: usize) -> Vec<f32> {
    [false, true]
        .into_iter()
        .flat_map(|s| {
            (0..t).flat_map(move |p| {
                (0..d / 2).map(move |c| {
                    let a = p as f32 / 10000f32.powf(2. * c as f32 / d as f32);
                    if s { a.sin() } else { a.cos() }
                })
            })
        })
        .collect()
}

fn rms(x: &[f64]) -> Vec<f64> {
    let inv = (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64 + f32::EPSILON as f64)
        .sqrt()
        .recip();
    x.iter().map(|v| v * inv).collect()
}
fn linear(x: &[f64], p: &[f64], n: usize) -> Vec<f64> {
    p[..n * x.len()]
        .chunks_exact(x.len())
        .map(|r| r.iter().zip(x).map(|(a, b)| a * b).sum())
        .collect()
}

// Independent complex arithmetic oracle: read only A and B, never their
// expanded -B/A copies, and multiply (A+iB)(x+iy) directly.
fn complex_linear(x: &[f64], p: &[f64], n: usize, og: usize, ig: usize) -> Vec<f64> {
    let (op, ip) = (n / (2 * og), x.len() / (2 * ig));
    let mut y = vec![0.; n];
    for out_group in 0..og {
        for out_channel in 0..op {
            let r = out_group * 2 * op + out_channel;
            for in_group in 0..ig {
                for in_channel in 0..ip {
                    let c = in_group * 2 * ip + in_channel;
                    let (a, b) = (p[r * x.len() + c], p[(r + op) * x.len() + c]);
                    y[r] += a * x[c] - b * x[c + ip];
                    y[r + op] += a * x[c + ip] + b * x[c];
                }
            }
        }
    }
    y
}

fn projection(cfg: &Config, x: &[f64], p: &[f64], n: usize, og: usize, ig: usize) -> Vec<f64> {
    if layout::is_complex(&cfg.architecture) {
        complex_linear(x, p, n, og, ig)
    } else {
        linear(x, p, n)
    }
}

// Direct Hermitian inner product and complex multiplication. Unlike the GPU,
// this oracle uses the attended value itself, not centered deltas or J vectors.
fn hermitian_projection(v: &[f64], attended: &[f64]) -> Vec<f64> {
    let m = v.len() / 2;
    let denominator = v
        .iter()
        .map(|a| a * a)
        .sum::<f64>()
        .sqrt()
        .max(1e-12)
        .powi(2);
    let cr = (0..m)
        .map(|i| v[i] * attended[i] + v[i + m] * attended[i + m])
        .sum::<f64>()
        / denominator;
    let ci = (0..m)
        .map(|i| v[i] * attended[i + m] - v[i + m] * attended[i])
        .sum::<f64>()
        / denominator;
    let mut out = vec![0.; v.len()];
    for i in 0..m {
        out[i] = attended[i] - (v[i] * cr - v[i + m] * ci);
        out[i + m] = attended[i + m] - (v[i] * ci + v[i + m] * cr);
    }
    out
}
fn model(cfg: &Config, p: &[f64], tokens: &[u32], b: usize, t: usize) -> Vec<f64> {
    let w = cfg.width;
    let h = layout::hidden(w, "");
    let d = w / cfg.heads;
    let scale = (2. * cfg.layers as f64).sqrt().recip();
    let ro = rope(d, t);
    let mut x: Vec<Vec<f64>> = tokens
        .iter()
        .map(|&token| rms(&p[token as usize * w..(token as usize + 1) * w]))
        .collect();
    let mut flags: Vec<bool> = (0..b * t)
        .map(|i| {
            tokens[(i / t) * t..=i]
                .iter()
                .all(|&v| v == tokens[i / t * t])
        })
        .collect();
    for l in 0..cfg.layers {
        let o = cfg.vocab * w + l * (4 * w * w + 3 * w * h);
        let qkv: Vec<f64> = x
            .iter()
            .flat_map(|r| projection(cfg, &rms(r), &p[o..], 3 * w, 3 * cfg.heads, cfg.heads))
            .collect();
        let att = reference(
            &qkv,
            cfg.heads,
            d,
            t,
            b,
            &ro,
            cfg.architecture == "complex-hermitian",
        );
        for batch in 0..b {
            let ok = (0..cfg.heads).all(|head| {
                let start = batch * t * 3 * w + 2 * w + head * d;
                qkv[start..start + d]
                    .iter()
                    .map(|v| v * v)
                    .sum::<f64>()
                    .sqrt()
                    > 1e-12
            });
            for pos in 0..t {
                flags[batch * t + pos] &= ok;
            }
        }
        for i in 0..b * t {
            if i % t != 0 && !flags[i] {
                let branch = rms(&projection(
                    cfg,
                    &att[i * w..(i + 1) * w],
                    &p[o + 3 * w * w..],
                    w,
                    cfg.heads,
                    cfg.heads,
                ));
                for j in 0..w {
                    x[i][j] += scale * branch[j];
                }
            }
        }
        let ff: Vec<Vec<f64>> = x
            .iter()
            .map(|r| {
                let fc = projection(cfg, &rms(r), &p[o + 4 * w * w..], 2 * h, 2, cfg.heads);
                let mut hidden = vec![0.; h];
                if cfg.architecture == "real" {
                    for j in 0..h {
                        hidden[j] = fc[j] / (1. + (-fc[j]).exp()) * fc[h + j];
                    }
                } else {
                    let f = h / 2;
                    for j in 0..f {
                        let (a, bi, u, v) = (fc[j], fc[f + j], fc[2 * f + j], fc[3 * f + j]);
                        let s = (1. + (-(a * a + bi * bi + f32::EPSILON as f64).sqrt()).exp())
                            .recip()
                            / 2f64.sqrt();
                        hidden[j] = s * (a * u - bi * v);
                        hidden[f + j] = s * (a * v + bi * u);
                    }
                }
                rms(&projection(
                    cfg,
                    &hidden,
                    &p[o + 4 * w * w + 2 * h * w..],
                    w,
                    cfg.heads,
                    1,
                ))
            })
            .collect();
        for i in 0..b * t {
            for j in 0..w {
                x[i][j] += scale * ff[if flags[i] { i / t * t } else { i }][j];
            }
        }
    }
    (0..b * t)
        .flat_map(|i| linear(&rms(&x[if flags[i] { i / t * t } else { i }]), p, cfg.vocab))
        .collect()
}
fn model_loss(cfg: &Config, p: &[f64], tokens: &[u32], labels: &[u32], b: usize, t: usize) -> f64 {
    let z = model(cfg, p, tokens, b, t);
    let mut n = 0;
    let mut loss = 0.;
    for (row, &label) in labels.iter().enumerate() {
        if label as usize >= cfg.vocab {
            continue;
        }
        n += 1;
        let z = &z[row * cfg.vocab..(row + 1) * cfg.vocab];
        let max = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        loss += max + z.iter().map(|v| (v - max).exp()).sum::<f64>().ln() - z[label as usize];
    }
    loss / n as f64
}
#[test]
#[ignore = "requires CUDA"]
fn full_sequence_model_matches_scalar_logits_loss_and_parameter_gradients() -> Result<()> {
    let tokens = [1, 1, 1, 1, 2, 2, 3, 4, 5, 6, 7, 8];
    let b = 3;
    let t = 4;
    let labels = [u32::MAX, 2, 3, 4, u32::MAX, 4, 3, 5, 2, u32::MAX, 7, 8];
    for (w, heads, vocab, layers) in [
        (64, 4, 11, 2),
        (64, 4, 548, 2),
        (128, 4, 11, 1),
        (128, 1, 256, 1),
    ] {
        for architecture in [
            "real",
            "real-product",
            "complex-linear",
            "complex-hermitian",
        ] {
            let cfg = Config {
                width: w,
                heads,
                vocab,
                layers,
                max_batch: 4,
                max_length: 7,
                normalized: true,
                ssmax: true,
                gain: heads as f32 / w as f32,
                lr: 0.002,
                decay: 0.1,
                warmup: 50,
                architecture: architecture.into(),
                phase_optimizer: false,
            };
            let p = layout::initialize_model(&cfg, 1);
            let fp64: Vec<f64> = p.iter().map(|&v| v as f64).collect();
            let mut engine = Engine::new(cfg.clone(), &p)?;
            engine.forward(&tokens, &labels, b, t)?;
            engine.backward(b, t)?;
            let z = engine.logits.read()?;
            let expected = model(&cfg, &fp64, &tokens, b, t);
            let error = z
                .iter()
                .zip(&expected)
                .map(|(a, b)| (*a as f64 - b).abs())
                .fold(0., f64::max);
            assert!(error < 2e-5, "{architecture} W{w} V{vocab} logits {error}");
            let actual = engine.losses(b * t)?.iter().map(|&v| v as f64).sum::<f64>()
                / labels.iter().filter(|&&v| (v as usize) < vocab).count() as f64;
            let expected = model_loss(&cfg, &fp64, &tokens, &labels, b, t);
            assert!(
                (actual - expected).abs() < 2e-6,
                "loss {actual} != {expected}"
            );
            let grad = engine.gradients.read()?;
            assert!(grad.iter().all(|v| v.is_finite()));
            let h = layout::hidden(w, "");
            let mut groups = std::iter::once(0..vocab * w).collect::<Vec<_>>();
            let mut at = vocab * w;
            for _ in 0..layers {
                for n in [3 * w * w, w * w, 2 * h * w, h * w] {
                    groups.push(at..at + n);
                    at += n;
                }
            }
            for (group, indices) in groups.into_iter().enumerate() {
                let mut plus = fp64.clone();
                let mut minus = fp64.clone();
                let eps = 2e-6;
                let n = indices.len() as f64;
                let mut analytic = 0.;
                let mut direction = vec![0.; p.len()];
                for j in indices.clone() {
                    direction[j] = (j as f64 * 0.819 + group as f64).sin() / n.sqrt();
                }
                if layout::is_complex(architecture) {
                    for [ar, bi, br, ai, kind] in layout::pairs(&cfg) {
                        if kind == 0 && indices.contains(&(ar as usize)) {
                            direction[ai as usize] = direction[ar as usize];
                            direction[br as usize] = -direction[bi as usize];
                        }
                    }
                }
                for j in indices {
                    plus[j] += eps * direction[j];
                    minus[j] -= eps * direction[j];
                    analytic += grad[j] as f64 * direction[j];
                }
                let fd = (model_loss(&cfg, &plus, &tokens, &labels, b, t)
                    - model_loss(&cfg, &minus, &tokens, &labels, b, t))
                    / (2. * eps);
                assert!(
                    (analytic - fd).abs() < 2e-4 + 0.004 * fd.abs(),
                    "{architecture} W{w} V{vocab} group{group}: analytic{analytic} fd{fd}"
                );
            }
            // Warmup starts with a zero-rate update, while Adam moments advance.
            engine.update(0)?;
            assert_eq!(engine.parameters.read()?, p);
            assert!(engine.first_moment.read()?.iter().any(|&v| v != 0.));
        }
    }
    Ok(())
}

#[test]
fn every_sequence_dataset_matches_frozen_fingerprints() -> Result<()> {
    let suite: serde_json::Value = serde_json::from_str(include_str!("../../suite.json"))?;
    let mut count = 0;
    for row in suite["tasks"].as_array().unwrap() {
        {
            crate::data::build(row)?;
            count += 1;
        }
    }
    assert_eq!(count, 35);
    Ok(())
}
fn reference(
    x: &[f64],
    h: usize,
    d: usize,
    t: usize,
    b: usize,
    r: &[f32],
    hermitian: bool,
) -> Vec<f64> {
    let w = h * d;
    let mut y = vec![0.; b * t * w];
    let mut qkv = x.to_vec();
    for row in 0..b * t {
        for part in 0..2 {
            for head in 0..h {
                let at = row * 3 * w + part * w + head * d;
                let inv = (x[at..at + d].iter().map(|v| v * v).sum::<f64>() / d as f64
                    + f32::EPSILON as f64)
                    .sqrt()
                    .recip();
                for c in 0..d / 2 {
                    let co = r[(row % t) * d / 2 + c] as f64;
                    let si = r[t * d / 2 + (row % t) * d / 2 + c] as f64;
                    qkv[at + c] = (x[at + c] * co - x[at + c + d / 2] * si) * inv;
                    qkv[at + c + d / 2] = (x[at + c + d / 2] * co + x[at + c] * si) * inv;
                }
            }
        }
    }
    for batch in 0..b {
        for pos in 1..t {
            for head in 0..h {
                let q = (batch * t + pos) * 3 * w + head * d;
                let own = q + 2 * w;
                let anchor = batch * t * 3 * w + 2 * w + head * d;
                let mut p: Vec<f64> = (0..=pos)
                    .map(|key| {
                        let k = (batch * t + key) * 3 * w + w + head * d;
                        (0..d).map(|c| qkv[q + c] * qkv[k + c]).sum::<f64>()
                            * ((pos + 1) as f64).ln()
                            / d as f64
                    })
                    .collect();
                let max = p.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                for v in &mut p {
                    *v = (*v - max).exp();
                }
                let total = p.iter().sum::<f64>();
                for v in &mut p {
                    *v /= total;
                }
                if hermitian {
                    let attended: Vec<f64> = (0..d)
                        .map(|c| {
                            p.iter()
                                .enumerate()
                                .map(|(k, p)| p * x[(batch * t + k) * 3 * w + 2 * w + head * d + c])
                                .sum()
                        })
                        .collect();
                    let projected = hermitian_projection(&x[own..own + d], &attended);
                    let at = (batch * t + pos) * w + head * d;
                    y[at..at + d].copy_from_slice(&projected);
                    continue;
                }
                let norm = x[own..own + d].iter().map(|v| v * v).sum::<f64>().sqrt();
                let den = norm.max(1e-12);
                let delta: Vec<_> = (0..d)
                    .map(|c| {
                        p.iter()
                            .enumerate()
                            .map(|(k, p)| {
                                p * (x[(batch * t + k) * 3 * w + 2 * w + head * d + c]
                                    - x[anchor + c])
                            })
                            .sum::<f64>()
                            - (x[own + c] - x[anchor + c])
                    })
                    .collect();
                let dot = delta
                    .iter()
                    .enumerate()
                    .map(|(c, v)| v * x[own + c] / den)
                    .sum::<f64>();
                let rem = if norm > 1e-12 {
                    0.
                } else {
                    1. - (norm / den).powi(2)
                };
                for c in 0..d {
                    y[(batch * t + pos) * w + head * d + c] =
                        delta[c] - dot * x[own + c] / den + rem * x[own + c];
                }
            }
        }
    }
    y
}

#[test]
#[ignore = "requires CUDA"]
fn sequence_attention_matches_reference_and_gradients() -> Result<()> {
    let cuda = Cuda::new()?;
    let kernels = attention::kernels::Kernels::load(&cuda)?;
    for (hermitian, value_scale) in [(false, 1.0f64), (true, 1.0), (true, 1e-13), (true, 0.0)] {
        for (d, h, t, b) in [
            (16usize, 4usize, 4usize, 2usize),
            (32, 4, 7, 1),
            (128, 1, 3, 2),
            (16, 4, 128, 1),
        ] {
            let w = d * h;
            let r = rope(d, t);
            let x: Vec<f32> = (0..b * t * 3 * w)
                .map(|i| {
                    let v = 0.07 * (i as f32 * 0.193).sin() + 0.013;
                    if i % (3 * w) >= 2 * w {
                        v * value_scale as f32
                    } else {
                        v
                    }
                })
                .collect();
            let dy: Vec<f32> = (0..b * t * w)
                .map(|i| 0.03 * (i as f32 * 0.271).cos())
                .collect();
            let raw = cuda.upload(&x)?;
            let norm = cuda.alloc::<f32>(x.len())?;
            let qkv = cuda.alloc::<f32>(x.len())?;
            let rope = cuda.upload(&r)?;
            let out = cuda.alloc::<f32>(dy.len())?;
            let delta = cuda.alloc::<f32>(dy.len())?;
            let prob = cuda.alloc::<f32>(b * h * t * t)?;
            let ok = cuda.alloc::<u32>(b * h)?;
            let gout = cuda.upload(&dy)?;
            let ds = cuda.alloc::<f32>(b * h * t * t)?;
            let gd = cuda.alloc::<f32>(dy.len())?;
            let dv = cuda.alloc::<f32>(dy.len())?;
            let dqkv = cuda.alloc::<f32>(x.len())?;
            let dnorm = cuda.alloc::<f32>(x.len())?;
            let dx = cuda.alloc::<f32>(x.len())?;
            macro_rules! run {
                ($d:literal) => {
                    // SAFETY: exact row/head grids and disjoint buffers sized above.
                    unsafe {
                        kernels.norm::<$d>(
                            Grid::new((b * t * 3 * h) as u32, 32),
                            norm.ptr(),
                            raw.ptr(),
                            h as u32,
                        )?;
                        kernels.rotate::<false>(
                            Grid::new(x.len().div_ceil(256) as u32, 256),
                            qkv.ptr(),
                            norm.ptr(),
                            rope.ptr(),
                            d as u32,
                            h as u32,
                            t as u32,
                            x.len() as u32,
                        )?;
                        kernels.forward::<$d>(
                            Grid::new((b * t * h) as u32, 32),
                            out.ptr(),
                            prob.ptr(),
                            delta.ptr(),
                            ok.ptr(),
                            qkv.ptr(),
                            h as u32,
                            t as u32,
                            u32::from(hermitian),
                        )?;
                        kernels.query_backward::<$d>(
                            Grid::new((b * t * h) as u32, 32),
                            dqkv.ptr(),
                            ds.ptr(),
                            gd.ptr(),
                            dv.ptr(),
                            gout.ptr(),
                            delta.ptr(),
                            prob.ptr(),
                            qkv.ptr(),
                            h as u32,
                            t as u32,
                            u32::from(hermitian),
                        )?;
                        kernels.key_backward::<$d>(
                            Grid::new((b * t * h) as u32, 32),
                            dqkv.ptr(),
                            ds.ptr(),
                            gd.ptr(),
                            dv.ptr(),
                            prob.ptr(),
                            qkv.ptr(),
                            h as u32,
                            t as u32,
                        )?;
                        kernels.rotate::<true>(
                            Grid::new(x.len().div_ceil(256) as u32, 256),
                            dnorm.ptr(),
                            dqkv.ptr(),
                            rope.ptr(),
                            d as u32,
                            h as u32,
                            t as u32,
                            x.len() as u32,
                        )?;
                        kernels.norm_backward::<$d>(
                            Grid::new((b * t * 3 * h) as u32, 32),
                            dx.ptr(),
                            raw.ptr(),
                            dnorm.ptr(),
                            h as u32,
                        )?;
                    }
                };
            }
            match d {
                16 => run!(16),
                32 => run!(32),
                128 => run!(128),
                _ => unreachable!(),
            };
            let fp64: Vec<_> = x.iter().map(|&x| x as f64).collect();
            let expected = reference(&fp64, h, d, t, b, &r, hermitian);
            let actual = out.read()?;
            let error = actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| (*a as f64 - b).abs())
                .fold(0., f64::max);
            assert!(error < 2e-6, "D{d} forward {error}");
            let actual = dx.read()?;
            for i in (0..x.len()).step_by(if t > 16 { x.len() / 12 } else { 13 }) {
                let mut plus = fp64.clone();
                let mut minus = fp64.clone();
                let eps = if value_scale < 1.0 && i % (3 * w) >= 2 * w {
                    1e-18
                } else {
                    1e-6
                };
                plus[i] += eps;
                minus[i] -= eps;
                let a = reference(&plus, h, d, t, b, &r, hermitian);
                let c = reference(&minus, h, d, t, b, &r, hermitian);
                let fd = a
                    .iter()
                    .zip(&c)
                    .zip(&dy)
                    .map(|((a, b), g)| (a - b) * (*g as f64) / (2. * eps))
                    .sum::<f64>();
                assert!(
                    (actual[i] as f64 - fd).abs() < 5e-5 + 0.001 * fd.abs(),
                    "D{d} input{i}: actual{} expected{fd}",
                    actual[i]
                );
            }
        }
    }
    Ok(())
}
