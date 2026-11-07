//! Native input preparation: exact arithmetic, fixed split, paired weights and orders.
//! SplitMix64 split is ported from grokking's tasks.py. Initialization/sampling use
//! Kernite microgpt's Random, hence this cohort has fresh controls, not historical Torch seeds.

use crate::{Result, hash, layout, random::Random, record, write_floats};
use serde_json::json;
use std::{fs, path::Path};

struct SplitMix64(u64);
impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn shuffle<T>(&mut self, xs: &mut [T]) {
        for end in (1..xs.len()).rev() {
            let n = end as u64 + 1;
            let threshold = n.wrapping_neg() % n;
            loop {
                let v = self.next();
                if v >= threshold {
                    xs.swap(end, (v % n) as usize);
                    break;
                }
            }
        }
    }
}

pub fn dataset() -> Vec<[u32; 3]> {
    let mut rows: Vec<_> = (0u32..97)
        .flat_map(|a| (0..97).map(move |b| [a, b, (a + b).pow(3) % 97]))
        .collect();
    SplitMix64(1).shuffle(&mut rows);
    rows
}

pub fn rope() -> Vec<f32> {
    let mut out = Vec::new();
    for sine in [false, true] {
        for position in [0.0f32, 1.0] {
            for channel in 0..16 {
                let frequency = 1.0f32 / 10000.0f32.powf(channel as f32 / 16.0);
                let theta = position * frequency;
                out.push(if sine { theta.sin() } else { theta.cos() });
            }
        }
    }
    out
}

pub fn write(
    dir: &Path,
    architecture: &str,
    optimizer: &str,
    seed: u32,
    updates: usize,
) -> Result<()> {
    if !["real", "complex-diagonal", "complex-product"].contains(&architecture)
        || !["adamw", "phase"].contains(&optimizer)
        || (architecture == "real" && optimizer == "phase")
        || updates == 0
        || updates > 100_000
    {
        return Err("invalid native preparation protocol".into());
    }
    fs::create_dir(dir)?;
    let (w, l) = (128, 1);
    let h = layout::hidden(w, architecture);
    let parameters = layout::initialize(w, l, architecture, seed);
    write_floats(&dir.join("parameters.f32"), &parameters)?;
    write_floats(&dir.join("rope.f32"), &rope())?;
    let rows = dataset();
    let words = |name: &str, xs: &[u32]| -> Result<()> {
        fs::write(
            dir.join(name),
            xs.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
        )?;
        Ok(())
    };
    words(
        "train.u32",
        &rows[..4704].iter().flatten().copied().collect::<Vec<_>>(),
    )?;
    words(
        "validation.u32",
        &rows[4704..].iter().flatten().copied().collect::<Vec<_>>(),
    )?;
    let mut rng = Random::new(seed + 10_000);
    let mut order = Vec::new();
    let mut boundaries = vec![0u32];
    while boundaries.len() <= updates {
        let mut epoch: Vec<u32> = (0..4704).collect();
        rng.shuffle(&mut epoch);
        for batch in epoch.chunks(256) {
            if boundaries.len() > updates {
                break;
            }
            order.extend_from_slice(batch);
            boundaries.push(order.len() as u32);
        }
    }
    words("order.u32", &order)?;
    words("boundaries.u32", &boundaries)?;
    let mut files = serde_json::Map::new();
    for name in [
        "parameters.f32",
        "rope.f32",
        "train.u32",
        "validation.u32",
        "order.u32",
        "boundaries.u32",
    ] {
        let bytes = fs::read(dir.join(name))?;
        files.insert(
            name.into(),
            json!({"sha256":hash(&bytes),"bytes":bytes.len()}),
        );
    }
    let mut params =
        vec![json!({"name":"token_embedding_table.weight","shape":[97,w],"offset":0,"count":97*w})];
    let mut offset = 97 * w;
    for layer in 0..l {
        for (name, shape) in [
            ("sa.c_attn.weight", [3 * w, w]),
            ("sa.c_proj.weight", [w, w]),
            ("ffwd.c_fc.weight", [2 * h, w]),
            ("ffwd.c_proj.weight", [w, h]),
        ] {
            let count = shape[0] * shape[1];
            params.push(json!({"name":format!("blocks.{layer}.{name}"),"shape":shape,"offset":offset,"count":count}));
            offset += count;
        }
    }
    record(
        &dir.join("manifest.json"),
        &json!({
            "version":1,"architecture":architecture,"phase_optimizer":optimizer=="phase","seed":seed,
            "width":w,"layers":l,"heads":4,"vocab":97,"batch":256,"observe":19,"target":0.99,
            "normalized":true,"ssmax":true,"attention_gain":1.0/32.0,"learning_rate":0.002,
            "weight_decay":0.1,"warmup":50,"updates":updates,"train_examples":4704,"validation_examples":4705,
            "parameter_count":parameters.len(),"independent_parameters":if architecture=="real" {parameters.len()}else{2*layout::pairs(w,l,h).len()},
            "parameters":params,"files":files,"initialization":"Kernite microgpt Random.gauss(0,.02); fresh matched controls",
            "sampler":"Kernite microgpt Random.shuffle each epoch, seed+10000; common across arms",
            "split":"SplitMix64 seed1, same disjoint half table as current cubic study",
            "task":"(a+b)^3 mod 97","flush_epochs":500,"seconds_cap":120.0,
            "scope":"250-epoch exploratory pilot; no eventual-convergence conclusion"
        }),
    )
}
