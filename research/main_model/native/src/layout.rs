//! Independent complex coordinates expanded as [[A,-B],[B,A]] for real GEMM.
//! Generalizes complex_transformer/native/src/layout.rs at GPT commit 786696b:
//! arbitrary vocabulary and 16/32/128-real-coordinate heads, fixed hidden width.
use crate::engine::{Config, Result};
use crate::random::Random;

pub fn is_complex(architecture: &str) -> bool {
    matches!(
        architecture,
        "complex-linear" | "complex-hermitian" | "complex-mhc"
    )
}

pub fn is_mhc(cfg: &Config) -> bool {
    cfg.architecture == "complex-mhc"
}
pub fn router_count(width: usize) -> usize {
    96 * width + 27
}
pub fn base_count(cfg: &Config) -> usize {
    count(cfg.width, cfg.layers, cfg.vocab)
}
pub fn total_count(cfg: &Config) -> usize {
    base_count(cfg)
        + if is_mhc(cfg) {
            2 * cfg.layers * router_count(cfg.width)
        } else {
            0
        }
}
pub fn router_offset(cfg: &Config, sublayer: usize) -> usize {
    base_count(cfg) + sublayer * router_count(cfg.width)
}

pub fn hidden(width: usize, _architecture: &str) -> usize {
    let old = (8 * width + 1) / 3;
    old.div_ceil(2) * 2
}

pub fn count(width: usize, layers: usize, vocab: usize) -> usize {
    vocab * width + layers * (4 * width * width + 3 * width * hidden(width, ""))
}

pub fn initialize(width: usize, layers: usize, vocab: usize, seed: u32) -> Vec<f32> {
    let mut rng = Random::new(seed);
    (0..count(width, layers, vocab))
        .map(|_| rng.gauss(0.0, 0.02) as f32)
        .collect()
}

/// Each owner is [A(real row,real col), B(imag row,real col), -B, A, kind].
/// kind=1 is an embedding pair without duplicated entries; kind=0 is a matrix.
fn matrix(map: &mut Vec<[u32; 5]>, offset: usize, n: usize, k: usize, og: usize, ig: usize) {
    let (np, kp) = (n / (2 * og), k / (2 * ig));
    for r in 0..n / 2 {
        let rr = r / np * 2 * np + r % np;
        let ri = rr + np;
        for c in 0..k / 2 {
            let cr = c / kp * 2 * kp + c % kp;
            let ci = cr + kp;
            map.push([
                (offset + rr * k + cr) as u32,
                (offset + ri * k + cr) as u32,
                (offset + rr * k + ci) as u32,
                (offset + ri * k + ci) as u32,
                0,
            ]);
        }
    }
}

pub fn pairs(cfg: &Config) -> Vec<[u32; 5]> {
    let (w, heads) = (cfg.width, cfg.heads);
    let d = w / heads;
    let h = hidden(w, &cfg.architecture);
    let mut map = Vec::new();
    for token in 0..cfg.vocab {
        for head in 0..heads {
            for c in 0..d / 2 {
                let at = token * w + head * d + c;
                map.push([at as u32, (at + d / 2) as u32, 0, 0, 1]);
            }
        }
    }
    for l in 0..cfg.layers {
        let o = cfg.vocab * w + l * (4 * w * w + 3 * w * h);
        matrix(&mut map, o, 3 * w, w, 3 * heads, heads);
        matrix(&mut map, o + 3 * w * w, w, w, heads, heads);
        matrix(&mut map, o + 4 * w * w, 2 * h, w, 2, heads);
        matrix(&mut map, o + 4 * w * w + 2 * h * w, w, h, heads, 1);
    }
    map
}

pub fn independent_count(cfg: &Config) -> usize {
    if is_complex(&cfg.architecture) {
        2 * pairs(cfg).len() + total_count(cfg) - base_count(cfg)
    } else {
        count(cfg.width, cfg.layers, cfg.vocab)
    }
}

pub fn initialize_model(cfg: &Config, seed: u32) -> Vec<f32> {
    let mut p = initialize(cfg.width, cfg.layers, cfg.vocab, seed);
    if is_complex(&cfg.architecture) {
        // Keep the real arm's A/B entries and embedding. The two complex arms
        // start bit-identically; projection input variance remains unchanged.
        for [ar, bi, br, ai, kind] in pairs(cfg) {
            if kind == 0 {
                p[ai as usize] = p[ar as usize];
                p[br as usize] = -p[bi as usize];
            }
        }
    }
    if is_mhc(cfg) {
        let mut rng = Random::new(seed + 20000);
        for _ in 0..2 * cfg.layers {
            p.extend((0..96 * cfg.width).map(|_| rng.gauss(0.0, 0.02) as f32));
            p.extend((0..24).map(|i| {
                if i >= 8 && (i - 8) / 4 == (i - 8) % 4 {
                    2.
                } else {
                    0.
                }
            }));
            p.extend([0.01; 3]);
        }
    }
    p
}

pub fn validate(cfg: &Config, p: &[f32]) -> Result<()> {
    if p.len() != total_count(cfg) || !p.iter().all(|v| v.is_finite()) {
        return Err("invalid parameter length or nonfinite value".into());
    }
    if is_complex(&cfg.architecture) {
        for [ar, bi, br, ai, kind] in pairs(cfg) {
            if kind == 0 && (p[ar as usize] != p[ai as usize] || p[bi as usize] != -p[br as usize])
            {
                return Err("broken complex-linear parameter constraint".into());
            }
        }
    }
    Ok(())
}

pub fn validate_state(cfg: &Config, p: &[f32], m: &[f32], v: &[f32]) -> Result<()> {
    validate(cfg, p)?;
    validate(cfg, m)?;
    if v.len() != p.len() || !v.iter().all(|x| x.is_finite() && *x >= 0.) {
        return Err("invalid AdamW second moment".into());
    }
    if is_complex(&cfg.architecture) {
        for [ar, bi, br, ai, kind] in pairs(cfg) {
            if cfg.phase_optimizer && v[ar as usize] != v[bi as usize] {
                return Err("phase AdamW requires a shared second moment".into());
            }
            if kind == 0 && (v[ar as usize] != v[ai as usize] || v[bi as usize] != v[br as usize]) {
                return Err("broken complex second-moment constraint".into());
            }
        }
    }
    Ok(())
}
