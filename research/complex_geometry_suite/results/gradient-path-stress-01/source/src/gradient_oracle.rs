//! Independent scalar FP64 forward-mode automatic differentiation. No CUDA
//! kernels or manually coded transformer backward formulas are reused.
use crate::{Config, layout};
use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Clone, Copy, Debug, Default)]
pub struct Jet {
    pub v: f64,
    pub d: f64,
}
impl From<f64> for Jet {
    fn from(v: f64) -> Self {
        Self { v, d: 0. }
    }
}
impl Add for Jet {
    type Output = Self;
    fn add(self, b: Self) -> Self {
        Self {
            v: self.v + b.v,
            d: self.d + b.d,
        }
    }
}
impl Sub for Jet {
    type Output = Self;
    fn sub(self, b: Self) -> Self {
        Self {
            v: self.v - b.v,
            d: self.d - b.d,
        }
    }
}
#[allow(clippy::suspicious_arithmetic_impl)]
impl Mul for Jet {
    type Output = Self;
    fn mul(self, b: Self) -> Self {
        Self {
            v: self.v * b.v,
            d: self.d * b.v + self.v * b.d,
        }
    }
}
impl Div for Jet {
    type Output = Self;
    fn div(self, b: Self) -> Self {
        let v = self.v / b.v;
        Self {
            v,
            d: (self.d - v * b.d) / b.v,
        }
    }
}
impl Neg for Jet {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            v: -self.v,
            d: -self.d,
        }
    }
}
impl Add<f64> for Jet {
    type Output = Self;
    fn add(self, b: f64) -> Self {
        self + Self::from(b)
    }
}
impl Sub<f64> for Jet {
    type Output = Self;
    fn sub(self, b: f64) -> Self {
        self - Self::from(b)
    }
}
impl Mul<f64> for Jet {
    type Output = Self;
    fn mul(self, b: f64) -> Self {
        Self {
            v: self.v * b,
            d: self.d * b,
        }
    }
}
impl Div<f64> for Jet {
    type Output = Self;
    fn div(self, b: f64) -> Self {
        Self {
            v: self.v / b,
            d: self.d / b,
        }
    }
}
impl Jet {
    pub fn sqrt(self) -> Self {
        let v = self.v.sqrt();
        Self {
            v,
            d: self.d / (2. * v),
        }
    }
    pub fn exp(self) -> Self {
        let v = self.v.exp();
        Self { v, d: v * self.d }
    }
    pub fn ln(self) -> Self {
        Self {
            v: self.v.ln(),
            d: self.d / self.v,
        }
    }
    pub fn ln_1p(self) -> Self {
        Self {
            v: self.v.ln_1p(),
            d: self.d / (1. + self.v),
        }
    }
    pub fn recip(self) -> Self {
        Self::from(1.) / self
    }
}
fn sum(values: impl Iterator<Item = Jet>) -> Jet {
    values.fold(Jet::default(), |a, b| a + b)
}
pub fn rms(x: &[Jet]) -> Vec<Jet> {
    let inv = (sum(x.iter().map(|&v| v * v)) / x.len() as f64 + f32::EPSILON as f64)
        .sqrt()
        .recip();
    x.iter().map(|&v| v * inv).collect()
}
fn linear(x: &[Jet], p: &[Jet], n: usize) -> Vec<Jet> {
    p[..n * x.len()]
        .chunks_exact(x.len())
        .map(|r| sum(r.iter().zip(x).map(|(&a, &b)| a * b)))
        .collect()
}
fn projection(cfg: &Config, x: &[Jet], p: &[Jet], n: usize, og: usize, ig: usize) -> Vec<Jet> {
    if !layout::is_complex(&cfg.architecture) {
        return linear(x, p, n);
    }
    let (op, ip) = (n / (2 * og), x.len() / (2 * ig));
    let mut y = vec![Jet::default(); n];
    for group in 0..og {
        for j in 0..op {
            let r = group * 2 * op + j;
            for input in 0..ig {
                for k in 0..ip {
                    let c = input * 2 * ip + k;
                    let (a, b) = (p[r * x.len() + c], p[(r + op) * x.len() + c]);
                    y[r] = y[r] + a * x[c] - b * x[c + ip];
                    y[r + op] = y[r + op] + a * x[c + ip] + b * x[c];
                }
            }
        }
    }
    y
}
pub fn product(fc: &[Jet]) -> Vec<Jet> {
    let m = fc.len() / 4;
    let mut y = vec![Jet::default(); 2 * m];
    for j in 0..m {
        let (a, b, u, v) = (fc[j], fc[m + j], fc[2 * m + j], fc[3 * m + j]);
        let s =
            ((-((a * a + b * b + f32::EPSILON as f64).sqrt())).exp() + 1.).recip() / 2f64.sqrt();
        y[j] = s * (a * u - b * v);
        y[m + j] = s * (a * v + b * u);
    }
    y
}
fn exclusion(v: &[Jet], a: &[Jet], hermitian: bool) -> Vec<Jet> {
    let norm2 = sum(v.iter().map(|&v| v * v));
    let den = if norm2.v > 1e-24 {
        norm2
    } else {
        Jet::from(1e-24)
    };
    let re = sum(v.iter().zip(a).map(|(&v, &a)| v * a)) / den;
    let half = v.len() / 2;
    let im = if hermitian {
        sum((0..half).map(|c| v[c] * a[c + half] - v[c + half] * a[c])) / den
    } else {
        Jet::default()
    };
    (0..v.len())
        .map(|c| {
            let jv = if c < half { -v[c + half] } else { v[c - half] };
            a[c] - v[c] * re - jv * im
        })
        .collect()
}
fn attention(cfg: &Config, raw: &[Jet], batch: usize, length: usize) -> Vec<Jet> {
    let w = cfg.width;
    let d = w / cfg.heads;
    let mut qkv = raw.to_vec();
    for row in 0..batch * length {
        for part in 0..2 {
            for head in 0..cfg.heads {
                let at = row * 3 * w + part * w + head * d;
                let normalized = rms(&raw[at..at + d]);
                for c in 0..d / 2 {
                    // Match the forward's stored FP32 RoPE constants, not its algebra.
                    let angle = (row % length) as f32 / 10000f32.powf(2. * c as f32 / d as f32);
                    let (si, co) = (angle.sin() as f64, angle.cos() as f64);
                    qkv[at + c] = normalized[c] * co - normalized[c + d / 2] * si;
                    qkv[at + c + d / 2] = normalized[c] * si + normalized[c + d / 2] * co;
                }
            }
        }
    }
    let mut result = vec![Jet::default(); batch * length * w];
    for b in 0..batch {
        for pos in 1..length {
            for head in 0..cfg.heads {
                let q = (b * length + pos) * 3 * w + head * d;
                let scores: Vec<_> = (0..=pos)
                    .map(|key| {
                        let k = (b * length + key) * 3 * w + w + head * d;
                        sum((0..d).map(|c| qkv[q + c] * qkv[k + c]))
                            * ((pos + 1) as f64).ln()
                            * cfg.gain as f64
                    })
                    .collect();
                let max = *scores.iter().max_by(|a, b| a.v.total_cmp(&b.v)).unwrap();
                let mass: Vec<_> = scores.iter().map(|&v| (v - max).exp()).collect();
                let denominator = sum(mass.iter().copied());
                let attended: Vec<_> = (0..d)
                    .map(|c| {
                        sum((0..=pos).map(|key| {
                            mass[key] / denominator
                                * raw[(b * length + key) * 3 * w + 2 * w + head * d + c]
                        }))
                    })
                    .collect();
                let own = q + 2 * w;
                let y = exclusion(
                    &raw[own..own + d],
                    &attended,
                    cfg.architecture == "complex-hermitian",
                );
                let start = (b * length + pos) * w + head * d;
                result[start..start + d].copy_from_slice(&y);
            }
        }
    }
    result
}
pub fn model(cfg: &Config, p: &[Jet], tokens: &[u32], batch: usize, length: usize) -> Vec<Jet> {
    let w = cfg.width;
    let h = layout::hidden(w, "");
    let d = w / cfg.heads;
    let scale = (2. * cfg.layers as f64).sqrt().recip();
    let mut x: Vec<Vec<Jet>> = tokens
        .iter()
        .map(|&t| rms(&p[t as usize * w..(t as usize + 1) * w]))
        .collect();
    let mut flags: Vec<bool> = (0..batch * length)
        .map(|i| {
            tokens[i / length * length..=i]
                .iter()
                .all(|&t| t == tokens[i / length * length])
        })
        .collect();
    for l in 0..cfg.layers {
        let o = cfg.vocab * w + l * (4 * w * w + 3 * w * h);
        let qkv: Vec<_> = x
            .iter()
            .flat_map(|r| projection(cfg, &rms(r), &p[o..], 3 * w, 3 * cfg.heads, cfg.heads))
            .collect();
        let a = attention(cfg, &qkv, batch, length);
        for b in 0..batch {
            let ok = (0..cfg.heads).all(|head| {
                let start = b * length * 3 * w + 2 * w + head * d;
                qkv[start..start + d].iter().map(|v| v.v * v.v).sum::<f64>() > 1e-24
            });
            for pos in 0..length {
                flags[b * length + pos] &= ok;
            }
        }
        for i in 0..batch * length {
            if i % length != 0 && !flags[i] {
                let branch = rms(&projection(
                    cfg,
                    &a[i * w..(i + 1) * w],
                    &p[o + 3 * w * w..],
                    w,
                    cfg.heads,
                    cfg.heads,
                ));
                for (v, branch) in x[i].iter_mut().zip(branch) {
                    *v = *v + branch * scale;
                }
            }
        }
        let ffn: Vec<Vec<_>> = x
            .iter()
            .map(|r| {
                let fc = projection(cfg, &rms(r), &p[o + 4 * w * w..], 2 * h, 2, cfg.heads);
                let hidden = if cfg.architecture == "real" {
                    (0..h)
                        .map(|c| fc[c] * ((-fc[c]).exp() + 1.).recip() * fc[h + c])
                        .collect()
                } else {
                    product(&fc)
                };
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
        for i in 0..batch * length {
            let branch = &ffn[if flags[i] { i / length * length } else { i }];
            for (v, &branch) in x[i].iter_mut().zip(branch) {
                *v = *v + branch * scale;
            }
        }
    }
    (0..batch * length)
        .flat_map(|i| {
            linear(
                &rms(&x[if flags[i] { i / length * length } else { i }]),
                p,
                cfg.vocab,
            )
        })
        .collect()
}
pub fn cross_entropy(logits: &[Jet], labels: &[u32], vocab: usize) -> Jet {
    let mut n = 0;
    let mut loss = Jet::default();
    for (row, &gold) in labels.iter().enumerate() {
        let gold = gold as usize;
        if gold >= vocab {
            continue;
        }
        n += 1;
        let z = &logits[row * vocab..(row + 1) * vocab];
        let max = *z.iter().max_by(|a, b| a.v.total_cmp(&b.v)).unwrap();
        let mass: Vec<_> = z.iter().map(|&v| (v - max).exp()).collect();
        let other = sum(mass
            .iter()
            .enumerate()
            .filter(|(c, _)| *c != gold)
            .map(|(_, &v)| v));
        loss = loss
            + if z[gold].v == max.v {
                (other / mass[gold]).ln_1p()
            } else {
                (max - z[gold]) + (other + mass[gold]).ln()
            };
    }
    loss / n.max(1) as f64
}
#[test]
fn dual_oracle_rules_and_confident_cross_entropy() {
    let x = Jet { v: 0.7, d: 1. };
    let y = ((x * x + 0.1).sqrt().exp() + x / (x + 1.)).ln_1p();
    let f = |v: f64| ((v * v + 0.1).sqrt().exp() + v / (v + 1.)).ln_1p();
    assert!((y.d - (f(0.700001) - f(0.699999)) / 0.000002).abs() < 1e-9);
    let z = [Jet { v: 80., d: 1. }, Jet::from(0.)];
    let ce = cross_entropy(&z, &[0], 2);
    assert!(ce.v > 0. && ce.d < 0.);
    assert!((ce.d / (-80f64).exp() + 1.).abs() < 1e-14);
}
