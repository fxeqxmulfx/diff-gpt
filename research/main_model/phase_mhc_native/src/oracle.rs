//! Independent scalar FP64 forward-mode automatic differentiation. No CUDA
//! kernels or manually coded transformer backward formulas are reused.
use crate::{AttentionMode, Config, layout};
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
    pub fn sin(self) -> Self {
        Self {
            v: self.v.sin(),
            d: self.d * self.v.cos(),
        }
    }
    pub fn cos(self) -> Self {
        Self {
            v: self.v.cos(),
            d: -self.d * self.v.sin(),
        }
    }
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
pub fn projection(cfg: &Config, x: &[Jet], p: &[Jet], n: usize, og: usize, ig: usize) -> Vec<Jet> {
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
pub fn plain_product(fc: &[Jet]) -> Vec<Jet> {
    let m = fc.len() / 4;
    let mut y = vec![Jet::default(); 2 * m];
    for j in 0..m {
        let (a, b, u, v) = (fc[j], fc[m + j], fc[2 * m + j], fc[3 * m + j]);
        y[j] = (a * u - b * v) / 2f64.sqrt();
        y[m + j] = (a * v + b * u) / 2f64.sqrt();
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
pub fn attention(cfg: &Config, raw: &[Jet], batch: usize, length: usize) -> Vec<Jet> {
    attention_mode(cfg, raw, batch, length, AttentionMode::Xsa)
}
fn attention_mode(
    cfg: &Config,
    raw: &[Jet],
    batch: usize,
    length: usize,
    mode: AttentionMode,
) -> Vec<Jet> {
    attention_frequency(cfg, raw, batch, length, mode, &[])
}
fn attention_frequency(
    cfg: &Config,
    raw: &[Jet],
    batch: usize,
    length: usize,
    mode: AttentionMode,
    alpha: &[Jet],
) -> Vec<Jet> {
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
                    let mut angle = (row % length) as f32 / 10000f32.powf(2. * c as f32 / d as f32);
                    if !alpha.is_empty() && layout::log_positions(cfg) {
                        angle *= ((row % length + 1) as f32).ln();
                    }
                    let (s0, c0) = (angle.sin() as f64, angle.cos() as f64);
                    let (si, co) = if alpha.is_empty() {
                        (Jet::from(s0), Jet::from(c0))
                    } else {
                        // Independent scalar AD of the declared rotation composition.
                        let delta = (alpha[c].exp() - 1.) * angle as f64;
                        (
                            delta.cos() * s0 + delta.sin() * c0,
                            delta.cos() * c0 - delta.sin() * s0,
                        )
                    };
                    qkv[at + c] = normalized[c] * co - normalized[c + d / 2] * si;
                    qkv[at + c + d / 2] = normalized[c] * si + normalized[c + d / 2] * co;
                }
            }
        }
    }
    attention_qkv_kind(cfg, &qkv, batch, length, mode)
}

/// Attention on already-normalized and rotated Q/K; V remains unnormalized.
/// This separation also checks the attention adjoint at the actual GPU inputs.
pub fn attention_qkv(cfg: &Config, qkv: &[Jet], batch: usize, length: usize) -> Vec<Jet> {
    attention_qkv_mode(cfg, qkv, batch, length, false)
}
pub fn attention_qkv_mode(
    cfg: &Config,
    qkv: &[Jet],
    batch: usize,
    length: usize,
    ordinary: bool,
) -> Vec<Jet> {
    attention_qkv_kind(
        cfg,
        qkv,
        batch,
        length,
        AttentionMode::from_ordinary(ordinary),
    )
}
pub fn attention_qkv_kind(
    cfg: &Config,
    qkv: &[Jet],
    batch: usize,
    length: usize,
    mode: AttentionMode,
) -> Vec<Jet> {
    let ordinary = mode != AttentionMode::Xsa;
    let w = cfg.width;
    let d = w / cfg.heads;
    let mut result = vec![Jet::default(); batch * length * w];
    for b in 0..batch {
        for pos in usize::from(!ordinary)..length {
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
                        if mode == AttentionMode::Centered {
                            let first = qkv[b * length * 3 * w + 2 * w + head * d + c];
                            let differences: Vec<_> = (0..=pos)
                                .map(|key| {
                                    qkv[(b * length + key) * 3 * w + 2 * w + head * d + c] - first
                                })
                                .collect();
                            return sum(
                                (0..=pos).map(|key| mass[key] / denominator * differences[key])
                            ) - sum(differences.into_iter()) / Jet::from((pos + 1) as f64);
                        }
                        sum((0..=pos).map(|key| {
                            mass[key] / denominator
                                * qkv[(b * length + key) * 3 * w + 2 * w + head * d + c]
                        }))
                    })
                    .collect();
                let own = q + 2 * w;
                let y = if ordinary {
                    attended
                } else {
                    exclusion(
                        &qkv[own..own + d],
                        &attended,
                        cfg.architecture == "complex-hermitian",
                    )
                };
                let start = (b * length + pos) * w + head * d;
                result[start..start + d].copy_from_slice(&y);
            }
        }
    }
    result
}
pub fn model(cfg: &Config, p: &[Jet], tokens: &[u32], batch: usize, length: usize) -> Vec<Jet> {
    model_with_product(cfg, p, tokens, batch, length, false)
}
pub fn model_with_product(
    cfg: &Config,
    p: &[Jet],
    tokens: &[u32],
    batch: usize,
    length: usize,
    plain: bool,
) -> Vec<Jet> {
    model_variant(cfg, p, tokens, batch, length, plain, false)
}
pub fn model_variant(
    cfg: &Config,
    p: &[Jet],
    tokens: &[u32],
    batch: usize,
    length: usize,
    plain: bool,
    ordinary: bool,
) -> Vec<Jet> {
    model_mode(
        cfg,
        p,
        tokens,
        batch,
        length,
        plain,
        AttentionMode::from_ordinary(ordinary),
    )
}
pub fn model_mode(
    cfg: &Config,
    p: &[Jet],
    tokens: &[u32],
    batch: usize,
    length: usize,
    plain: bool,
    mode: AttentionMode,
) -> Vec<Jet> {
    // Centered reference deliberately keeps all token paths: shared-value
    // cancellation follows from Jet algebra, not native prefix shortcuts.
    let ordinary = mode != AttentionMode::Xsa;
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
            !ordinary
                && tokens[i / length * length..=i]
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
        let a = attention_mode(cfg, &qkv, batch, length, mode);
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
            if ordinary || i % length != 0 && !flags[i] {
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
                } else if plain {
                    plain_product(&fc)
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

/// Independent Jet implementation of the paper's finite map. The GPU has no
/// access to this code; each normalization is differentiated by scalar AD.
pub fn mhc_gates(z: &[Jet]) -> Vec<Jet> {
    let mut gates = vec![Jet::default(); 24];
    for i in 0..8 {
        let e = if z[i].v >= 0. {
            (-z[i]).exp()
        } else {
            z[i].exp()
        };
        gates[i] = if z[i].v >= 0. {
            (e + 1.).recip()
        } else {
            e / (e + 1.)
        };
        if i >= 4 {
            gates[i] = gates[i] * 2.;
        }
    }
    let mut log = z[8..24].to_vec();
    for _ in 0..20 {
        for col in 0..4 {
            let max = (0..4)
                .map(|row| log[row * 4 + col])
                .max_by(|a, b| a.v.total_cmp(&b.v))
                .unwrap();
            let den = sum((0..4).map(|row| (log[row * 4 + col] - max).exp())).ln();
            for row in 0..4 {
                log[row * 4 + col] = log[row * 4 + col] - max - den;
            }
        }
        for row in 0..4 {
            let max = log[row * 4..row * 4 + 4]
                .iter()
                .copied()
                .max_by(|a, b| a.v.total_cmp(&b.v))
                .unwrap();
            let den = sum(log[row * 4..row * 4 + 4].iter().map(|&v| (v - max).exp())).ln();
            for col in 0..4 {
                log[row * 4 + col] = log[row * 4 + col] - max - den;
            }
        }
    }
    for i in 0..16 {
        gates[8 + i] = log[i].exp();
    }
    gates
}
fn route_mhc(x: &[Jet], p: &[Jet], w: usize) -> (Vec<Jet>, Vec<Jet>) {
    let projected = linear(&rms(x), p, 24);
    let z: Vec<_> = (0..24)
        .map(|i| projected[i] * p[96 * w + 24 + (i / 4).min(2)] + p[96 * w + i])
        .collect();
    let gates = mhc_gates(&z);
    let read = (0..w)
        .map(|c| sum((0..4).map(|j| gates[j] * x[j * w + c])))
        .collect();
    (gates, read)
}
fn write_mhc(x: &[Jet], b: &[Jet], g: &[Jet], scale: f64) -> Vec<Jet> {
    let w = b.len();
    (0..4 * w)
        .map(|i| {
            sum((0..4).map(|j| g[8 + (i / w) * 4 + j] * x[j * w + i % w]))
                + g[4 + i / w] * b[i % w] * scale
        })
        .collect()
}
// Independent factorized D * P * D* expression. The GPU instead forms each
// edge coefficient and its manually coded adjoint; this oracle uses scalar AD.
pub fn write_phase_mhc(
    cfg: &Config,
    x: &[Jet],
    b: &[Jet],
    g: &[Jet],
    a: &[Jet],
    scale: f64,
) -> Vec<Jet> {
    let w = cfg.width;
    let d = w / cfg.heads;
    let mut u = vec![(Jet::from(1.), Jet::from(0.))];
    for &value in &a[..3] {
        let den = (value * value + 1.).sqrt();
        u.push((den.recip(), value / den));
    }
    let mut out = vec![Jet::default(); 4 * w];
    for head in 0..cfg.heads {
        for band in 0..d / 2 {
            let re = head * d + band;
            let im = re + d / 2;
            let input: Vec<_> = (0..4)
                .map(|j| {
                    let (c, s) = u[j];
                    (
                        c * x[j * w + re] + s * x[j * w + im],
                        c * x[j * w + im] - s * x[j * w + re],
                    )
                })
                .collect();
            for i in 0..4 {
                let vr = sum((0..4).map(|j| g[8 + 4 * i + j] * input[j].0));
                let vi = sum((0..4).map(|j| g[8 + 4 * i + j] * input[j].1));
                out[i * w + re] = u[i].0 * vr - u[i].1 * vi + g[4 + i] * b[re] * scale;
                out[i * w + im] = u[i].1 * vr + u[i].0 * vi + g[4 + i] * b[im] * scale;
            }
        }
    }
    out
}
pub fn main_model(
    cfg: &Config,
    p: &[Jet],
    tokens: &[u32],
    batch: usize,
    length: usize,
) -> Vec<Jet> {
    if !layout::is_mhc(cfg) {
        return model_mode(cfg, p, tokens, batch, length, true, AttentionMode::Ordinary);
    }
    let w = cfg.width;
    let h = layout::hidden(w, "");
    let scale = (2. * cfg.layers as f64).sqrt().recip();
    let mut x: Vec<Vec<Jet>> = tokens
        .iter()
        .map(|&t| rms(&p[t as usize * w..(t as usize + 1) * w]).repeat(4))
        .collect();
    for layer in 0..cfg.layers {
        let o = cfg.vocab * w + layer * (4 * w * w + 3 * w * h);
        let routes: Vec<_> = x
            .iter()
            .map(|r| route_mhc(r, &p[layout::router_offset(cfg, 2 * layer)..], w))
            .collect();
        let qkv: Vec<_> = routes
            .iter()
            .flat_map(|(_, read)| {
                projection(cfg, &rms(read), &p[o..], 3 * w, 3 * cfg.heads, cfg.heads)
            })
            .collect();
        let alpha = if layout::learned_rotary(cfg) {
            &p[layout::frequency_offset(cfg)..]
        } else {
            &[]
        };
        let a = attention_frequency(cfg, &qkv, batch, length, AttentionMode::Ordinary, alpha);
        for i in 0..x.len() {
            let branch = rms(&projection(
                cfg,
                &a[i * w..(i + 1) * w],
                &p[o + 3 * w * w..],
                w,
                cfg.heads,
                cfg.heads,
            ));
            x[i] = if layout::phase_routing(cfg) {
                write_phase_mhc(
                    cfg,
                    &x[i],
                    &branch,
                    &routes[i].0,
                    &p[layout::phase_offset(cfg) + 6 * layer..],
                    scale,
                )
            } else {
                write_mhc(&x[i], &branch, &routes[i].0, scale)
            };
        }
        for state in &mut x {
            let (g, read) = route_mhc(state, &p[layout::router_offset(cfg, 2 * layer + 1)..], w);
            let fc = projection(cfg, &rms(&read), &p[o + 4 * w * w..], 2 * h, 2, cfg.heads);
            let product = plain_product(&fc);
            let branch = rms(&projection(
                cfg,
                &product,
                &p[o + 4 * w * w + 2 * h * w..],
                w,
                cfg.heads,
                1,
            ));
            *state = if layout::phase_routing(cfg) {
                write_phase_mhc(
                    cfg,
                    state,
                    &branch,
                    &g,
                    &p[layout::phase_offset(cfg) + 6 * layer + 3..],
                    scale,
                )
            } else {
                write_mhc(state, &branch, &g, scale)
            };
        }
    }
    x.iter()
        .flat_map(|state| {
            let mean: Vec<_> = (0..w)
                .map(|c| sum((0..4).map(|j| state[j * w + c])) / 4.)
                .collect();
            linear(&rms(&mean), p, cfg.vocab)
        })
        .collect()
}
