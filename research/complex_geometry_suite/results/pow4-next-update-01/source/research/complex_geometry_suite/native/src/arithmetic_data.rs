//! Port of the frozen arithmetic oracle and SplitMix64 split, checked against suite hashes.
use crate::{Result, hash};
use serde_json::Value;

pub struct SplitMix64(pub u64);
impl SplitMix64 {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    pub fn shuffle<T>(&mut self, xs: &mut [T]) {
        for end in (1..xs.len()).rev() {
            let n = end as u64 + 1;
            let threshold = n.wrapping_neg() % n;
            loop {
                let x = self.next();
                if x >= threshold {
                    xs.swap(end, (x % n) as usize);
                    break;
                }
            }
        }
    }
}

pub fn power(mut x: u32, mut n: u32, p: u32) -> u32 {
    let mut out = 1;
    while n > 0 {
        if n & 1 != 0 {
            out = out * x % p;
        }
        x = x * x % p;
        n >>= 1;
    }
    out
}

pub fn fingerprint(rows: &[[u32; 3]]) -> String {
    let mut bytes = format!("torch.int64:({}, 3, 2):", rows.len()).into_bytes();
    for &[a, b, y] in rows {
        for x in [a as i64, b as i64, -100, y as i64, 0, 0] {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
    }
    hash(&bytes)
}

pub struct Arithmetic {
    pub train: Vec<[u32; 3]>,
    pub validation: Vec<[u32; 3]>,
    pub vocab: usize,
}

pub fn arithmetic(row: &Value) -> Result<Arithmetic> {
    if row["case"]["kind"] != "arithmetic" {
        return Err("not an arithmetic case".into());
    }
    let spec = &row["case"]["data"];
    let p = spec["prime"].as_u64().ok_or("missing prime")? as u32;
    let op = spec["operation"].as_str().ok_or("missing operation")?;
    let mut permutation: Vec<u32> = (0..p).collect();
    SplitMix64(0x7065726D75746531).shuffle(&mut permutation);
    let first = u32::from(matches!(op, "div" | "power-division"));
    let mut rows = Vec::new();
    for a in 0..p {
        for b in first..p {
            let y = match op {
                "add" => (a + b) % p,
                "sub" => (a + p - b) % p,
                "mul" => a * b % p,
                "div" => a * power(b, p - 2, p) % p,
                "power-division" => power(
                    a * power(b, p - 2, p) % p,
                    spec["power"].as_u64().ok_or("missing power")? as u32,
                    p,
                ),
                "cube-plus" => (power(a, 3, p) + b) % p,
                "square-times" => a * a * b % p,
                "square-sum" => (a * a + b * b) % p,
                "half-sum" => (a + b) * power(2, p - 2, p) % p,
                "sum-square" => power((a + b) % p, 2, p),
                "sum-cube" => power((a + b) % p, 3, p),
                "sum-pow4" => power((a + b) % p, 4, p),
                "affine-product" => (a + a * b) % p,
                "perm-add" => permutation[((a + b) % p) as usize],
                "abs-diff" => a.abs_diff(b),
                "abs-diff/blind" => a.abs_diff(b) / 2,
                _ => return Err(format!("unported operation {op}").into()),
            };
            rows.push([a, b, y]);
        }
    }
    SplitMix64(
        row["case"]["data_seed"]
            .as_u64()
            .ok_or("missing data seed")?,
    )
    .shuffle(&mut rows);
    let mid = rows.len() / 2;
    let train = rows[..mid].to_vec();
    let validation = rows[mid..].to_vec();
    for (name, split) in [("train", &train), ("validation", &validation)] {
        if fingerprint(split)
            != row["data_fingerprints"][name]
                .as_str()
                .ok_or("missing frozen fingerprint")?
        {
            return Err(format!("frozen {name} fingerprint mismatch for {}", row["name"]).into());
        }
    }
    if train.len() as u64 != row["train_examples"].as_u64().ok_or("missing size")?
        || validation.len() as u64 != row["heldout_examples"].as_u64().ok_or("missing size")?
    {
        return Err("frozen dataset sizes changed".into());
    }
    Ok(Arithmetic {
        train,
        validation,
        vocab: p as usize,
    })
}

pub fn rope() -> Vec<f32> {
    let mut out = Vec::new();
    for sine in [false, true] {
        for position in [0.0f32, 1.0] {
            for channel in 0..16 {
                let angle = position / 10000.0f32.powf(channel as f32 / 16.0);
                out.push(if sine { angle.sin() } else { angle.cos() });
            }
        }
    }
    out
}
