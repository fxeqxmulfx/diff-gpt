//! Exact primitive comparisons against the frozen tensor candidate, plus a CPU
//! bit-layout oracle for transpose. Scientific training is not performed here.
use crate::{Result, hash, random::Random, runner};
use kernite_cuda::{Cuda, Grid};
use serde_json::{Value, json};
use std::{fs, path::Path};
fn bits(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn mismatches(a: &[f32], b: &[f32]) -> usize {
    a.iter()
        .zip(b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count()
}
pub fn run(root: &Path) -> Result<()> {
    if root.exists() {
        return Err("equivalence output already exists".into());
    }
    fs::create_dir_all(root)?;
    for (name, body) in crate::sources::FILES {
        let path = root.join("source").join(name);
        fs::create_dir_all(path.parent().unwrap())?;
        fs::write(path, body)?;
    }
    let cuda = Cuda::new()?;
    let old = crate::parent_attention::kernels::Kernels::load(&cuda)?;
    let ordinary = crate::parent_ordinary::kernels::Kernels::load(&cuda)?;
    let new_ordinary = crate::ordinary_attention::kernels::Kernels::load(&cuda)?;
    let fused = crate::ordinary_forward::kernels::Kernels::load(&cuda)?;
    let old_ops = crate::parent_ops::kernels::Kernels::load(&cuda)?;
    let tiled = crate::tiled_transpose::kernels::Kernels::load(&cuda)?;
    let old_mhc = crate::parent_mhc::kernels::Kernels::load(&cuda)?;
    let new_mhc = crate::mhc::kernels::Kernels::load(&cuda)?;
    let new_attention = crate::attention::kernels::Kernels::load(&cuda)?;
    let embedding = crate::embedding_backward::kernels::Kernels::load(&cuda)?;
    let mut rng = Random::new(71913);
    let mut records: Vec<Value> = Vec::new();
    for (d, h) in [(16usize, 4usize), (32, 4), (128, 1), (16, 8), (16, 16)] {
        for length in [1usize, 2, 3, 7, 31, 32, 33, 64, 65, 96, 129] {
            for pattern in ["zero", "random", "tiny", "large-values", "repeated"] {
                let batch = 3usize;
                let w = d * h;
                let n = batch * length * w;
                let scores = batch * h * length * length;
                let mut input: Vec<f32> = (0..3 * n)
                    .map(|i| {
                        let sample = rng.gauss(0., 0.4) as f32;
                        match pattern {
                            "zero" => 0.,
                            "tiny" => sample * 1e-20,
                            "large-values" if (i / w) % 3 == 2 => sample * 1e20,
                            _ => sample,
                        }
                    })
                    .collect();
                if pattern == "repeated" {
                    for b in 0..batch {
                        for t in 1..length {
                            for i in 0..3 * w {
                                input[(b * length + t) * 3 * w + i] = input[b * length * 3 * w + i];
                            }
                        }
                    }
                }
                let qkv = cuda.upload(&input)?;
                let a = cuda.alloc::<f32>(n)?;
                let b = cuda.alloc::<f32>(n)?;
                let pa = cuda.upload(&vec![0f32; scores])?;
                let pb = cuda.upload(&vec![0f32; scores])?;
                let delta = cuda.alloc::<f32>(n)?;
                let flags = cuda.alloc::<u32>(batch * h)?;
                macro_rules! launch {
                    ($d:literal) => {
                        unsafe {
                            old.forward::<$d>(
                                Grid::new((batch * length * h) as u32, 32),
                                a.ptr(),
                                pa.ptr(),
                                delta.ptr(),
                                flags.ptr(),
                                qkv.ptr(),
                                h as u32,
                                length as u32,
                                0,
                            )?;
                            ordinary.values::<$d>(
                                Grid::new((batch * length * h) as u32, 32),
                                a.ptr(),
                                pa.ptr(),
                                qkv.ptr(),
                                h as u32,
                                length as u32,
                            )?;
                            fused.forward::<$d>(
                                Grid::new(
                                    ((batch * length * h) / if $d == 16 { 2 } else { 1 })
                                        .div_ceil(4) as u32,
                                    128,
                                ),
                                b.ptr(),
                                pb.ptr(),
                                qkv.ptr(),
                                h as u32,
                                length as u32,
                                (batch * length * h) as u32,
                            )?;
                        }
                    };
                }
                match d {
                    16 => launch!(16),
                    32 => launch!(32),
                    128 => launch!(128),
                    _ => unreachable!(),
                }
                let (a, b, pa, pb) = (a.read()?, b.read()?, pa.read()?, pb.read()?);
                let output_mismatches = mismatches(&a, &b);
                let probability_mismatches = mismatches(&pa, &pb);
                let finite = a
                    .iter()
                    .chain(&b)
                    .chain(&pa)
                    .chain(&pb)
                    .all(|x| x.is_finite());
                let path = root.join(format!("attention-d{d}-h{h}-t{length}-{pattern}"));
                fs::create_dir_all(&path)?;
                for (name, values) in [
                    ("input", &input),
                    ("old-output", &a),
                    ("new-output", &b),
                    ("old-prob", &pa),
                    ("new-prob", &pb),
                ] {
                    fs::write(path.join(format!("{name}.f32")), bits(values))?;
                }
                records.push(json!({"kind":"attention","d":d,"heads":h,"batch":batch,"length":length,"pattern":pattern,
                    "output_mismatches":output_mismatches,"probability_mismatches":probability_mismatches,"finite":finite,
                    "passed":output_mismatches==0&&probability_mismatches==0&&finite,"input_sha256":hash(&bits(&input))}));
                let na = cuda.alloc::<f32>(3 * n)?;
                let nb = cuda.alloc::<f32>(3 * n)?;
                let ka = cuda.upload(&vec![0f32; 3 * n])?;
                let kb = cuda.upload(&vec![0f32; 3 * n])?;
                let ds = cuda.upload(
                    &(0..scores)
                        .map(|_| rng.gauss(0., 0.1) as f32)
                        .collect::<Vec<_>>(),
                )?;
                let gd = cuda.upload(
                    &(0..n)
                        .map(|_| rng.gauss(0., 0.2) as f32)
                        .collect::<Vec<_>>(),
                )?;
                let dv = cuda.upload(
                    &(0..n)
                        .map(|_| rng.gauss(0., 0.3) as f32)
                        .collect::<Vec<_>>(),
                )?;
                let probabilities = cuda.upload(&pa)?;
                macro_rules! extra {
                    ($d:literal) => {
                        unsafe {
                            old.norm::<$d>(
                                Grid::new((batch * length * 3 * h) as u32, 32),
                                na.ptr(),
                                qkv.ptr(),
                                h as u32,
                            )?;
                            new_attention.norm::<$d>(
                                Grid::new(
                                    ((batch * length * 3 * h) / if $d == 16 { 2 } else { 1 })
                                        .div_ceil(4) as u32,
                                    128,
                                ),
                                nb.ptr(),
                                qkv.ptr(),
                                h as u32,
                                (batch * length * 3 * h) as u32,
                            )?;
                            old.key_backward::<$d>(
                                Grid::new((batch * length * h) as u32, 32),
                                ka.ptr(),
                                ds.ptr(),
                                gd.ptr(),
                                dv.ptr(),
                                probabilities.ptr(),
                                qkv.ptr(),
                                h as u32,
                                length as u32,
                            )?;
                            new_attention.key_backward::<$d>(
                                Grid::new(
                                    ((batch * length * h) / if $d == 16 { 2 } else { 1 })
                                        .div_ceil(4) as u32,
                                    128,
                                ),
                                kb.ptr(),
                                ds.ptr(),
                                gd.ptr(),
                                dv.ptr(),
                                probabilities.ptr(),
                                qkv.ptr(),
                                h as u32,
                                length as u32,
                                (batch * length * h) as u32,
                            )?;
                        }
                    };
                }
                match d {
                    16 => extra!(16),
                    32 => extra!(32),
                    128 => extra!(128),
                    _ => unreachable!(),
                }
                for (kind, a, b) in [
                    ("qk_norm", na.read()?, nb.read()?),
                    ("key_value_backward", ka.read()?, kb.read()?),
                ] {
                    let different = mismatches(&a, &b);
                    if different > 0 {
                        fs::write(path.join(format!("{kind}-old.f32")), bits(&a))?;
                        fs::write(path.join(format!("{kind}-new.f32")), bits(&b))?;
                    }
                    records.push(json!({"kind":kind,"d":d,"heads":h,"batch":batch,"length":length,"pattern":pattern,"mismatches":different,"passed":different==0,
                        "parent_output_sha256":hash(&bits(&a)),"new_output_sha256":hash(&bits(&b))}));
                }
                let mut qa = Vec::new();
                let mut qb = Vec::new();
                for count in [3 * n, scores, n, n] {
                    qa.push(cuda.upload(&vec![0f32; count])?);
                    qb.push(cuda.upload(&vec![0f32; count])?);
                }
                macro_rules! query {
                    ($d:literal) => {
                        unsafe {
                            ordinary.query_backward::<$d>(
                                Grid::new(
                                    ((batch * length * h) / if $d == 16 { 2 } else { 1 }) as u32,
                                    32,
                                ),
                                qa[0].ptr(),
                                qa[1].ptr(),
                                qa[2].ptr(),
                                qa[3].ptr(),
                                gd.ptr(),
                                probabilities.ptr(),
                                qkv.ptr(),
                                h as u32,
                                length as u32,
                            )?;
                            new_ordinary.query_backward::<$d>(
                                Grid::new(
                                    ((batch * length * h) / if $d == 16 { 2 } else { 1 })
                                        .div_ceil(4) as u32,
                                    128,
                                ),
                                qb[0].ptr(),
                                qb[1].ptr(),
                                qb[2].ptr(),
                                qb[3].ptr(),
                                gd.ptr(),
                                probabilities.ptr(),
                                qkv.ptr(),
                                h as u32,
                                length as u32,
                                (batch * length * h) as u32,
                            )?;
                        }
                    };
                }
                match d {
                    16 => query!(16),
                    32 => query!(32),
                    128 => query!(128),
                    _ => unreachable!(),
                }
                let mut query_outputs = Vec::new();
                for (i, (a, b)) in qa.iter().zip(&qb).enumerate() {
                    let (a, b) = (a.read()?, b.read()?);
                    let different = mismatches(&a, &b);
                    let finite = a.iter().chain(&b).all(|x| x.is_finite());
                    if different > 0 || !finite {
                        fs::write(path.join(format!("query-{i}-old.f32")), bits(&a))?;
                        fs::write(path.join(format!("query-{i}-new.f32")), bits(&b))?;
                    }
                    query_outputs.push(json!({"buffer":i,"mismatches":different,"finite":finite,
                        "passed":different==0&&finite,"parent_output_sha256":hash(&bits(&a)),"new_output_sha256":hash(&bits(&b))}));
                }
                records.push(json!({"kind":"query_backward","d":d,"heads":h,"batch":batch,"length":length,"pattern":pattern,
                    "passed":query_outputs.iter().all(|v|v["passed"]==true),"outputs":query_outputs}));
            }
        }
    }
    for (rows, cols) in [
        (1usize, 1usize),
        (1, 97),
        (97, 1),
        (7, 9),
        (31, 33),
        (33, 31),
        (32, 32),
        (64, 24),
        (128, 97),
        (128, 342),
        (684, 128),
        (1025, 548),
        (4096, 192),
    ] {
        let input: Vec<f32> = (0..rows * cols)
            .map(|i| match i % 8 {
                0 => -0.,
                1 => f32::from_bits(1),
                2 => f32::from_bits(0x7fc01234),
                3 => f32::INFINITY,
                4 => f32::NEG_INFINITY,
                _ => rng.gauss(0., 20.) as f32,
            })
            .collect();
        let x = cuda.upload(&input)?;
        let a = cuda.alloc::<f32>(rows * cols)?;
        let b = cuda.alloc::<f32>(rows * cols)?;
        unsafe {
            old_ops.transpose(
                Grid::new((rows * cols).div_ceil(256) as u32, 256),
                a.ptr(),
                x.ptr(),
                rows as u32,
                cols as u32,
            )?;
            tiled.transpose(
                Grid::new((rows.div_ceil(32) * cols.div_ceil(32)) as u32, 256),
                b.ptr(),
                x.ptr(),
                rows as u32,
                cols as u32,
            )?;
        }
        let (a, b) = (a.read()?, b.read()?);
        let mut expected = vec![0f32; input.len()];
        for r in 0..rows {
            for c in 0..cols {
                expected[c * rows + r] = input[r * cols + c];
            }
        }
        let parent_mismatches = mismatches(&a, &b);
        let cpu_mismatches = mismatches(&expected, &b);
        let path = root.join(format!("transpose-{rows}x{cols}"));
        fs::create_dir_all(&path)?;
        for (name, values) in [("input", &input), ("parent", &a), ("new", &b)] {
            fs::write(path.join(format!("{name}.f32")), bits(values))?;
        }
        records.push(
            json!({"kind":"transpose","rows":rows,"cols":cols,"parent_mismatches":parent_mismatches,
            "cpu_mismatches":cpu_mismatches,"passed":parent_mismatches==0&&cpu_mismatches==0}),
        );
    }
    for width in [64usize, 128, 256] {
        for rows in [1usize, 3, 17, 512] {
            for (pattern, scale) in [("normal", 1f64), ("tiny", 1e-20), ("large", 1e8)] {
                let x: Vec<f32> = (0..rows * 4 * width)
                    .map(|_| rng.gauss(0., scale) as f32)
                    .collect();
                let dy: Vec<f32> = (0..rows * 4 * width)
                    .map(|_| rng.gauss(0., scale) as f32)
                    .collect();
                let branch: Vec<f32> = (0..rows * width)
                    .map(|_| rng.gauss(0., scale) as f32)
                    .collect();
                let dr: Vec<f32> = (0..rows * width)
                    .map(|_| rng.gauss(0., scale) as f32)
                    .collect();
                let (gx, gy, gb, gr) = (
                    cuda.upload(&x)?,
                    cuda.upload(&dy)?,
                    cuda.upload(&branch)?,
                    cuda.upload(&dr)?,
                );
                let a = cuda.alloc::<f32>(rows * 24)?;
                let b = cuda.alloc::<f32>(rows * 24)?;
                unsafe {
                    old_mhc.gate_grad(
                        Grid::new((rows * 24) as u32, 32),
                        a.ptr(),
                        gx.ptr(),
                        gb.ptr(),
                        gy.ptr(),
                        gr.ptr(),
                        width as u32,
                        rows as u32,
                        0.5,
                    )?;
                    new_mhc.gate_grad(
                        Grid::new((rows * 6) as u32, 128),
                        b.ptr(),
                        gx.ptr(),
                        gb.ptr(),
                        gy.ptr(),
                        gr.ptr(),
                        width as u32,
                        rows as u32,
                        0.5,
                    )?;
                }
                let (a, b) = (a.read()?, b.read()?);
                let different = mismatches(&a, &b);
                let finite = a.iter().chain(&b).all(|x| x.is_finite());
                let path = root.join(format!("gate-w{width}-r{rows}-{pattern}"));
                fs::create_dir_all(&path)?;
                for (name, values) in [
                    ("input", &x),
                    ("dy", &dy),
                    ("branch", &branch),
                    ("dr", &dr),
                    ("old", &a),
                    ("new", &b),
                ] {
                    fs::write(path.join(format!("{name}.f32")), bits(values))?;
                }
                records.push(json!({"kind":"routing_gradient","width":width,"rows":rows,"pattern":pattern,"mismatches":different,"finite":finite,"passed":different==0&&finite}));
            }
        }
    }
    for width in [64usize, 128, 256] {
        for rows in [1usize, 31, 32, 33, 65, 512, 4097] {
            for pattern in ["spread", "one-token", "cancellation"] {
                let vocab = 97usize;
                let tokens: Vec<u32> = (0..rows)
                    .map(|r| {
                        if pattern == "one-token" {
                            13
                        } else {
                            ((r * 31 + r / 7) % vocab) as u32
                        }
                    })
                    .collect();
                let dx: Vec<f32> = (0..rows * width)
                    .map(|i| {
                        if pattern == "cancellation" {
                            match (i / width) % 4 {
                                0 => 1e20,
                                1 => 1.0,
                                2 => -1e20,
                                _ => -0.0,
                            }
                        } else {
                            rng.gauss(0., 0.1) as f32
                        }
                    })
                    .collect();
                let initial: Vec<f32> = (0..vocab * width)
                    .map(|_| rng.gauss(0., 0.1) as f32)
                    .collect();
                let (gx, gt) = (cuda.upload(&dx)?, cuda.upload(&tokens)?);
                let (a, b) = (cuda.upload(&initial)?, cuda.upload(&initial)?);
                unsafe {
                    old_ops.embed_bwd(
                        Grid::new((vocab * width).div_ceil(256) as u32, 256),
                        a.ptr(),
                        gx.ptr(),
                        gt.ptr(),
                        width as u32,
                        vocab as u32,
                        rows as u32,
                    )?;
                    embedding.accumulate(
                        Grid::new(vocab as u32, width as u32),
                        b.ptr(),
                        gx.ptr(),
                        gt.ptr(),
                        width as u32,
                        rows as u32,
                    )?;
                }
                let (a, b) = (a.read()?, b.read()?);
                let different = mismatches(&a, &b);
                if different > 0 {
                    let path = root.join(format!("embedding-w{width}-r{rows}-{pattern}"));
                    fs::create_dir_all(&path)?;
                    for (name, values) in
                        [("dx", &dx), ("initial", &initial), ("old", &a), ("new", &b)]
                    {
                        fs::write(path.join(format!("{name}.f32")), bits(values))?;
                    }
                    fs::write(
                        path.join("tokens.u32"),
                        tokens
                            .iter()
                            .flat_map(|x| x.to_le_bytes())
                            .collect::<Vec<_>>(),
                    )?;
                }
                records.push(json!({"kind":"embedding_backward","width":width,"rows":rows,"vocab":vocab,"pattern":pattern,"mismatches":different,"passed":different==0,
                    "parent_output_sha256":hash(&bits(&a)),"new_output_sha256":hash(&bits(&b))}));
            }
        }
    }
    for width in [64usize, 128, 256] {
        for rows in [1usize, 2, 3, 4, 5, 17, 65, 512] {
            for pattern in ["zero", "tiny", "normal", "cancellation", "huge-router"] {
                for router in [true, false] {
                    // The scaled router normalizer has a wider input range than
                    // the unchanged read normalizer. Compare each in its range.
                    if !router && pattern == "huge-router" {
                        continue;
                    }
                    let cols = if router { 4 * width } else { width };
                    let input: Vec<f32> = (0..rows * cols)
                        .map(|i| {
                            let v = rng.gauss(0., 0.5) as f32;
                            match pattern {
                                "zero" => 0.,
                                "tiny" => v * 1e-20,
                                "huge-router" => v * 1e30,
                                "cancellation" => {
                                    if i % 2 == 0 {
                                        1.
                                    } else {
                                        -1.
                                    }
                                }
                                _ => v,
                            }
                        })
                        .collect();
                    let gradient: Vec<f32> = (0..input.len())
                        .map(|i| match pattern {
                            "huge-router" => input[i],
                            "cancellation" => 1.,
                            _ => rng.gauss(0., 0.4) as f32,
                        })
                        .collect();
                    let x = cuda.upload(&input)?;
                    let dy = cuda.upload(&gradient)?;
                    let a = cuda.alloc::<f32>(input.len())?;
                    let b = cuda.alloc::<f32>(input.len())?;
                    let da = cuda.alloc::<f32>(input.len())?;
                    let db = cuda.alloc::<f32>(input.len())?;
                    unsafe {
                        if router {
                            old_mhc.rms::<false>(
                                Grid::new(rows as u32, 32),
                                a.ptr(),
                                x.ptr(),
                                dy.ptr(),
                                cols as u32,
                                rows as u32,
                            )?;
                            new_mhc.rms::<false>(
                                Grid::new(rows.div_ceil(4) as u32, 128),
                                b.ptr(),
                                x.ptr(),
                                dy.ptr(),
                                cols as u32,
                                rows as u32,
                            )?;
                            old_mhc.rms::<true>(
                                Grid::new(rows as u32, 32),
                                da.ptr(),
                                x.ptr(),
                                dy.ptr(),
                                cols as u32,
                                rows as u32,
                            )?;
                            new_mhc.rms::<true>(
                                Grid::new(rows.div_ceil(4) as u32, 128),
                                db.ptr(),
                                x.ptr(),
                                dy.ptr(),
                                cols as u32,
                                rows as u32,
                            )?;
                        } else {
                            old_mhc.read_rms::<false>(
                                Grid::new(rows as u32, 32),
                                a.ptr(),
                                da.ptr(),
                                x.ptr(),
                                dy.ptr(),
                                cols as u32,
                                rows as u32,
                            )?;
                            new_mhc.read_rms::<false>(
                                Grid::new(rows.div_ceil(4) as u32, 128),
                                b.ptr(),
                                db.ptr(),
                                x.ptr(),
                                dy.ptr(),
                                cols as u32,
                                rows as u32,
                            )?;
                            old_mhc.read_rms::<true>(
                                Grid::new(rows as u32, 32),
                                a.ptr(),
                                da.ptr(),
                                x.ptr(),
                                dy.ptr(),
                                cols as u32,
                                rows as u32,
                            )?;
                            new_mhc.read_rms::<true>(
                                Grid::new(rows.div_ceil(4) as u32, 128),
                                b.ptr(),
                                db.ptr(),
                                x.ptr(),
                                dy.ptr(),
                                cols as u32,
                                rows as u32,
                            )?;
                        }
                    }
                    for (direction, a, b) in [
                        ("forward", a.read()?, b.read()?),
                        ("backward", da.read()?, db.read()?),
                    ] {
                        let different = mismatches(&a, &b);
                        let finite = a.iter().chain(&b).all(|x| x.is_finite());
                        if different > 0 || !finite {
                            let path = root.join(format!(
                                "rms-router{router}-w{width}-r{rows}-{pattern}-{direction}"
                            ));
                            fs::create_dir_all(&path)?;
                            for (name, values) in
                                [("x", &input), ("dy", &gradient), ("old", &a), ("new", &b)]
                            {
                                fs::write(path.join(format!("{name}.f32")), bits(values))?;
                            }
                        }
                        records.push(json!({"kind":"mhc_rms","router":router,"width":width,"rows":rows,"pattern":pattern,"direction":direction,
                            "mismatches":different,"finite":finite,"passed":different==0&&finite,
                            "parent_output_sha256":hash(&bits(&a)),"new_output_sha256":hash(&bits(&b))}));
                    }
                }
            }
        }
    }
    records.extend(crate::gemm_equivalence::run(&cuda, root)?);
    let passed = records.iter().all(|r| r["passed"] == true);
    fs::write(
        root.join("report.json"),
        serde_json::to_vec_pretty(&json!({"passed":passed,"cases":records,
        "source_sha256":runner::source_hash(),"ptx_sha256":runner::ptx_hash(),"criterion":"bit-identical primitive outputs and gradients against the frozen numerical parent; no tolerance relaxation"}))?,
    )?;
    if passed {
        println!("{} exact primitive cases passed", records.len());
        Ok(())
    } else {
        Err("placement mismatch; raw evidence retained".into())
    }
}
