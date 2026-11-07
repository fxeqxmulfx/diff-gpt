//! Complex parameters in head-major real storage; RoPE pairs halves within each head.
//! A matrix W=A+iB is expanded as [[A,-B],[B,A]], with explicit head permutations.

use crate::random::Random;

pub fn hidden(width: usize, architecture: &str) -> usize {
    let real = (8 * width + 1) / 3;
    if architecture == "real" {
        real
    } else {
        2 * (real + 2 * width / 3)
    }
}

pub fn count(width: usize, layers: usize, architecture: &str) -> usize {
    97 * width + layers * (4 * width * width + 3 * width * hidden(width, architecture))
}

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

pub fn pairs(w: usize, layers: usize, h: usize) -> Vec<[u32; 5]> {
    let mut map = Vec::new();
    for token in 0..97 {
        for head in 0..w / 32 {
            for c in 0..16 {
                let at = token * w + head * 32 + c;
                map.push([at as u32, (at + 16) as u32, 0, 0, 1]);
            }
        }
    }
    for layer in 0..layers {
        let o = 97 * w + layer * (4 * w * w + 3 * w * h);
        matrix(&mut map, o, 3 * w, w, 3 * w / 32, w / 32);
        matrix(&mut map, o + 3 * w * w, w, w, w / 32, w / 32);
        matrix(&mut map, o + 4 * w * w, 2 * h, w, 2, w / 32);
        matrix(&mut map, o + 4 * w * w + 2 * h * w, w, h, w / 32, 1);
    }
    map
}

pub fn initialize(w: usize, layers: usize, architecture: &str, seed: u32) -> Vec<f32> {
    let mut rng = Random::new(seed);
    let mut p = vec![0.; count(w, layers, architecture)];
    for value in &mut p[..97 * w] {
        *value = rng.gauss(0., 0.02) as f32;
    }
    if architecture == "real" {
        for value in &mut p[97 * w..] {
            *value = rng.gauss(0., 0.02) as f32;
        }
    } else {
        for [ar, bi, br, ai, kind] in pairs(w, layers, hidden(w, architecture)) {
            if kind == 1 {
                continue;
            }
            let (a, b) = (rng.gauss(0., 0.02) as f32, rng.gauss(0., 0.02) as f32);
            p[ar as usize] = a;
            p[ai as usize] = a;
            p[bi as usize] = b;
            p[br as usize] = -b;
        }
    }
    p
}

pub fn validate(p: &[f32], w: usize, layers: usize, architecture: &str) -> bool {
    p.len() == count(w, layers, architecture)
        && p.iter().all(|x| x.is_finite())
        && (architecture == "real"
            || pairs(w, layers, hidden(w, architecture))
                .iter()
                .all(|&[ar, bi, br, ai, kind]| {
                    kind == 1
                        || (p[ar as usize] == p[ai as usize] && p[bi as usize] == -p[br as usize])
                }))
}
