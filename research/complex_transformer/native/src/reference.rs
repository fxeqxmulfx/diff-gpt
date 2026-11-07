//! Independent scalar FP64 forward oracle for the native two-token experiment.
//! It uses direct equations and scalar matrix products, never the GPU kernels.

use crate::{engine::Config, layout};

fn rms(x: &[f64]) -> Vec<f64> {
    let inv = (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64 + f32::EPSILON as f64)
        .sqrt()
        .recip();
    x.iter().map(|v| v * inv).collect()
}
fn linear(x: &[f64], p: &[f64], n: usize) -> Vec<f64> {
    p[..n * x.len()]
        .chunks_exact(x.len())
        .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
        .collect()
}
fn dot(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y).map(|(a, b)| a * b).sum()
}
fn rotate(x: &[f64], position: usize, rope: &[f32]) -> Vec<f64> {
    let x = rms(x);
    let mut y = vec![0.; 32];
    for j in 0..16 {
        let (c, s) = (
            rope[position * 16 + j] as f64,
            rope[32 + position * 16 + j] as f64,
        );
        y[j] = x[j] * c - x[j + 16] * s;
        y[j + 16] = x[j] * s + x[j + 16] * c;
    }
    y
}

pub fn product(x: [f64; 4], cross: bool) -> [f64; 2] {
    let [a, b, u, v] = x;
    let s = 1.0 / (1.0 + (-(a * a + b * b + f32::EPSILON as f64).sqrt()).exp());
    if cross {
        [
            s * (a * u - b * v) / 2.0f64.sqrt(),
            s * (a * v + b * u) / 2.0f64.sqrt(),
        ]
    } else {
        [s * a * u, s * b * v]
    }
}

pub fn logits(config: &Config, p: &[f64], rows: &[[u32; 3]], rope: &[f32]) -> Vec<f64> {
    let w = config.width;
    let h = layout::hidden(w, &config.architecture);
    let scale = if config.normalized {
        (2.0 * config.layers as f64).sqrt().recip()
    } else {
        1.0
    };
    let mut outputs = Vec::new();
    for &[a, b, _] in rows {
        let mut x = [
            p[a as usize * w..(a as usize + 1) * w].to_vec(),
            p[b as usize * w..(b as usize + 1) * w].to_vec(),
        ];
        if config.normalized {
            x = x.map(|v| rms(&v));
        }
        let mut constant = a == b;
        for layer in 0..config.layers {
            let o = 97 * w + layer * (4 * w * w + 3 * w * h);
            let qkv = x.each_ref().map(|v| linear(&rms(v), &p[o..], 3 * w));
            let mut att = [vec![0.; w], vec![0.; w]];
            for head in 0..w / 32 {
                let c = head * 32;
                let q = rotate(&qkv[1][c..c + 32], 1, rope);
                let k0 = rotate(&qkv[0][w + c..w + c + 32], 0, rope);
                let k1 = rotate(&qkv[1][w + c..w + c + 32], 1, rope);
                let gain = config.gain as f64
                    * if config.ssmax {
                        std::f32::consts::LN_2 as f64
                    } else {
                        1.
                    };
                let p1 = 1.0 / (1.0 + ((dot(&q, &k0) - dot(&q, &k1)) * gain).exp());
                let v0 = &qkv[0][2 * w + c..2 * w + c + 32];
                let v1 = &qkv[1][2 * w + c..2 * w + c + 32];
                constant &= dot(v0, v0).sqrt() > 1e-12;
                let norm = dot(v1, v1).sqrt();
                let den = norm.max(1e-12);
                let unit: Vec<_> = v1.iter().map(|v| v / den).collect();
                let delta: Vec<_> = v1
                    .iter()
                    .zip(v0)
                    .map(|(a, b)| (a - b) * p1 - (a - b))
                    .collect();
                let proj = dot(&delta, &unit);
                let rem = if norm > 1e-12 {
                    0.0
                } else {
                    1.0 - (norm / den).powi(2)
                };
                for j in 0..32 {
                    att[1][c + j] = delta[j] - proj * unit[j] + rem * v1[j];
                }
            }
            for t in 0..2 {
                let mut branch = linear(&att[t], &p[o + 3 * w * w..], w);
                if config.normalized {
                    branch = rms(&branch);
                }
                if t == 1 && !constant {
                    for j in 0..w {
                        x[t][j] += scale * branch[j];
                    }
                }
            }
            let mut ff = [vec![0.; w], vec![0.; w]];
            for t in 0..2 {
                let fc = linear(&rms(&x[t]), &p[o + 4 * w * w..], 2 * h);
                let hidden = if config.architecture == "real" {
                    (0..h)
                        .map(|j| fc[j] / (1. + (-fc[j]).exp()) * fc[h + j])
                        .collect::<Vec<_>>()
                } else {
                    let f = h / 2;
                    let mut out = vec![0.; h];
                    for j in 0..f {
                        let y = product(
                            [fc[j], fc[f + j], fc[2 * f + j], fc[3 * f + j]],
                            config.architecture == "complex-product",
                        );
                        out[j] = y[0];
                        out[f + j] = y[1];
                    }
                    out
                };
                ff[t] = linear(&hidden, &p[o + 4 * w * w + 2 * h * w..], w);
                if config.normalized {
                    ff[t] = rms(&ff[t]);
                }
            }
            for t in 0..2 {
                for j in 0..w {
                    x[t][j] += scale * ff[if constant { 0 } else { t }][j];
                }
            }
        }
        if constant {
            x[1] = x[0].clone();
        }
        for value in x {
            outputs.extend(linear(&rms(&value), p, 97));
        }
    }
    outputs
}

pub fn loss(config: &Config, p: &[f64], rows: &[[u32; 3]], rope: &[f32]) -> f64 {
    let logits = logits(config, p, rows, rope);
    rows.iter()
        .enumerate()
        .map(|(i, row)| {
            let z = &logits[i * 194 + 97..(i + 1) * 194];
            let max = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            max + z.iter().map(|v| (v - max).exp()).sum::<f64>().ln() - z[row[2] as usize]
        })
        .sum::<f64>()
        / rows.len() as f64
}
