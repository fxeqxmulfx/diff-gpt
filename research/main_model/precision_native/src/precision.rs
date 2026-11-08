//! Offline, exact-oracle screening. No training or runtime precision fallback.
use crate::{
    Engine, Result, data, exact, hash, layout, metrics, precision_kernels, random::Random, runner,
};
use kernite_cuda::{Cuda, Grid};
use serde_json::{Value, json};
use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::Path,
};
#[path = "hard_cases.rs"]
mod hard_cases;

struct Pair {
    label: String,
    origin: String,
    a: Vec<f32>,
    b: Vec<f32>,
}
fn push(pairs: &mut Vec<Pair>, label: String, origin: &str, a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    pairs.push(Pair {
        label,
        origin: origin.into(),
        a: a.to_vec(),
        b: b.to_vec(),
    });
}
fn synthetic(pairs: &mut Vec<Pair>) {
    let mut rng = Random::new(812030);
    for d in [16, 32, 128] {
        for (label, small) in [
            ("half_cancellation", 1. / 4096.),
            ("bfloat_cancellation", 1. / 512.),
            ("float_cancellation", 2f32.powi(-26)),
        ] {
            let mut a = vec![0.; d];
            let mut b = vec![0.; d];
            a[0] = 1.;
            a[1] = small;
            a[4] = -1.;
            b[0] = 1.;
            b[1] = 1.;
            b[4] = 1.;
            push(pairs, format!("{label}/d{d}"), "synthetic", &a, &b);
        }
        let mut a = vec![0.; d];
        let mut b = vec![0.; d];
        a[0] = 1. + 2f32.powi(-13);
        b[0] = 1. - 2f32.powi(-13);
        a[4] = -1.;
        b[4] = 1.;
        push(
            pairs,
            format!("product_cancellation/d{d}"),
            "synthetic",
            &a,
            &b,
        );
        for (label, scale_a, scale_b) in [
            ("normalized", 1., 1.),
            ("tiny_gradient", 1., 1e-10),
            ("underflow_product", 1e-22, 1e-22),
            ("large_product", 1e20, 1e20),
        ] {
            for i in 0..32 {
                let mut a: Vec<f32> = (0..d).map(|_| rng.gauss(0., 1.) as f32).collect();
                let mut b: Vec<f32> = (0..d).map(|_| rng.gauss(0., 1.) as f32).collect();
                for (v, scale) in [(&mut a, scale_a), (&mut b, scale_b)] {
                    let rms = (v.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / d as f64
                        + 2f64.powi(-23))
                    .sqrt();
                    for x in v {
                        *x = (*x as f64 / rms * scale) as f32;
                    }
                }
                push(pairs, format!("{label}/d{d}/{i}"), "synthetic", &a, &b);
            }
        }
        push(
            pairs,
            format!("signed_zero/d{d}"),
            "synthetic",
            &vec![-0.; d],
            &vec![1.; d],
        );
        push(
            pairs,
            format!("subnormal/d{d}"),
            "synthetic",
            &vec![f32::from_bits(1); d],
            &vec![f32::from_bits(1); d],
        );
    }
}
fn checkpoints(pairs: &mut Vec<Pair>) -> Result<Vec<Value>> {
    let mut provenance = vec![];
    let suite = runner::suite();
    for (name, seed) in [("arithmetic-sum-pow4", 0), ("basis-recall-easy", 2)] {
        let dir = Path::new("runs/main-model-mhc-wsd-01/mhc")
            .join(name)
            .join(format!("seed{seed}"));
        let cp: Value = serde_json::from_slice(&fs::read(dir.join("checkpoint.json"))?)?;
        let execution: Value = serde_json::from_slice(&fs::read(dir.join("execution.json"))?)?;
        let bytes = fs::read(dir.join("checkpoint.bin"))?;
        if cp["state_sha256"] != hash(&bytes)
            || execution["clean"] != true
            || cp["source_sha256"]
                != "07763652ff0889d3e4b171425bf3e5f7a8490a14273f1b915f6013acf2068c58"
            || cp["ptx_sha256"]
                != "9b3a413469e02a32037b76d404ec5df8c59aa63f012b74020ef5b2404a4227c9"
        {
            return Err("checkpoint provenance mismatch".into());
        }
        let row = suite["tasks"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["name"] == name)
            .ok_or("missing task")?;
        let task = data::build(row)?;
        let cfg = runner::config(row, &task, "mhc")?;
        let n = layout::total_count(&cfg);
        if bytes.len() != 12 * n {
            return Err("checkpoint layout mismatch".into());
        }
        let state: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        layout::validate_state(&cfg, &state[..n], &state[n..2 * n], &state[2 * n..])?;
        let train = &task.splits["train"];
        let step = cp["completed"].as_u64().ok_or("missing checkpoint step")? as usize;
        let batch = row["case"]["batch"].as_u64().unwrap() as usize;
        let (order, edges) = runner::order(train.len(), batch, step + 1, seed);
        if cp["examples"] != json!(edges[step]) {
            return Err("sampler offset mismatch".into());
        }
        let indices = &order[edges[step]..edges[step + 1]];
        let mut packed = metrics::Packed::default();
        packed.fill(train, indices);
        let mut engine = Engine::new(cfg.clone(), &state[..n])?;
        engine.forward(&packed.tokens, &packed.labels, indices.len(), packed.length)?;
        engine.backward(indices.len(), packed.length)?;
        let snapshot = engine.diagnostic_snapshot(indices.len() * packed.length)?;
        let d = cfg.width / cfg.heads;
        let w = cfg.width;
        let len = packed.length;
        let start = pairs.len();
        for layer in 0..cfg.layers {
            // Only correctly dimensioned QKV arrays are used from this older
            // diagnostic API; its two-token probability slice is not used.
            let qkv = &snapshot[&format!("layer{layer}.qkv")];
            let rotated = &snapshot[&format!("layer{layer}.qkv_rotated")];
            let grad = &snapshot[&format!("layer{layer}.grad_qkv_normalized")];
            for batch_index in 0..indices.len().min(4) {
                let mut positions = vec![0, len / 2, len - 1];
                positions.sort_unstable();
                positions.dedup();
                for t in positions {
                    for h in 0..cfg.heads {
                        let at = (batch_index * len + t) * 3 * w + h * d;
                        let mut keys = vec![0, t / 2, t];
                        keys.sort_unstable();
                        keys.dedup();
                        for k in keys {
                            let kat = (batch_index * len + k) * 3 * w + w + h * d;
                            push(
                                pairs,
                                format!(
                                    "{name}/seed{seed}/layer{layer}/b{batch_index}/t{t}/h{h}/k{k}"
                                ),
                                "trained_QK_dot",
                                &rotated[at..at + d],
                                &rotated[kat..kat + d],
                            );
                        }
                        for part in 0..2 {
                            let q = at + part * w;
                            push(
                                pairs,
                                format!(
                                    "{name}/seed{seed}/layer{layer}/b{batch_index}/t{t}/h{h}/part{part}"
                                ),
                                "trained_norm_backward_dot",
                                &qkv[q..q + d],
                                &grad[q..q + d],
                            );
                        }
                    }
                }
            }
        }
        if engine.parameters.read()? != state[..n] {
            return Err("screen changed model weights".into());
        }
        provenance.push(json!({"case":name,"seed":seed,"checkpoint":dir,"checkpoint_sha256":cp["state_sha256"],"completed":step,"examples":edges[step],"batch":indices.len(),"length":len,"sampled_pairs":pairs.len()-start,"training_updates":0}));
    }
    Ok(provenance)
}
pub fn run(root: &Path) -> Result<()> {
    if root.exists() {
        return Err("screen output exists; choose a fresh root".into());
    }
    fs::create_dir_all(root)?;
    for (name, body) in crate::sources::FILES {
        let path = root.join("source").join(name);
        fs::create_dir_all(path.parent().unwrap())?;
        fs::write(path, body)?;
    }
    let mut pairs = vec![];
    synthetic(&mut pairs);
    hard_cases::append(&mut pairs);
    let checkpoints = checkpoints(&mut pairs)?;
    let cuda = Cuda::new()?;
    let norm_controls = hard_cases::norm_controls(&cuda)?;
    fs::write(
        root.join("norm-controls.json"),
        serde_json::to_vec_pretty(&norm_controls)?,
    )?;
    let kernels = precision_kernels::kernels::Kernels::load(&cuda)?;
    let mut summaries = vec![];
    let mut observations = BufWriter::new(File::create_new(root.join("cases.jsonl"))?);
    for d in [16, 32, 128] {
        let group: Vec<_> = pairs.iter().filter(|p| p.a.len() == d).collect();
        let a: Vec<f32> = group.iter().flat_map(|p| p.a.iter().copied()).collect();
        let b: Vec<f32> = group.iter().flat_map(|p| p.b.iter().copied()).collect();
        let aa = cuda.upload(&a)?;
        let bb = cuda.upload(&b)?;
        let output = cuda.alloc::<f64>(group.len())?;
        let exact: Vec<_> = group.iter().map(|p| exact::dot(&p.a, &p.b)).collect();
        let mut input = BufWriter::new(File::create_new(root.join(format!("inputs-d{d}.f32")))?);
        for v in a.iter().chain(&b) {
            input.write_all(&v.to_le_bytes())?;
        }
        input.flush()?;
        for mode in 0..9 {
            macro_rules! launch {
                ($d:literal,$m:literal) => {
                    unsafe {
                        kernels.dot::<$d, $m>(
                            Grid::new(group.len() as u32, 32),
                            output.ptr(),
                            aa.ptr(),
                            bb.ptr(),
                        )?;
                    }
                };
            }
            macro_rules! width {
                ($d:literal) => {
                    match mode {
                        0 => launch!($d, 0),
                        1 => launch!($d, 1),
                        2 => launch!($d, 2),
                        3 => launch!($d, 3),
                        4 => launch!($d, 4),
                        5 => launch!($d, 5),
                        6 => launch!($d, 6),
                        7 => launch!($d, 7),
                        8 => launch!($d, 8),
                        _ => unreachable!(),
                    }
                };
            }
            match d {
                16 => width!(16),
                32 => width!(32),
                128 => width!(128),
                _ => unreachable!(),
            }
            let values = output.read()?;
            let mut raw = BufWriter::new(File::create_new(
                root.join(format!("outputs-d{d}-mode{mode}.f64")),
            )?);
            let mut forward_failures = 0;
            let mut backward_failures = 0;
            let mut nonfinite = 0;
            for (i, ((pair, target), &actual)) in group.iter().zip(&exact).zip(&values).enumerate()
            {
                raw.write_all(&actual.to_le_bytes())?;
                let (f, b, e, t) = exact::assess(target, actual);
                forward_failures += usize::from(!f);
                backward_failures += usize::from(!b);
                nonfinite += usize::from(!actual.is_finite());
                let max_abs_operand = pair
                    .a
                    .iter()
                    .chain(&pair.b)
                    .fold(0f32, |a, b| a.max(b.abs()));
                let sum_abs_products = pair
                    .a
                    .iter()
                    .zip(&pair.b)
                    .map(|(&x, &y)| ((x as f64) * (y as f64)).abs())
                    .sum::<f64>();
                let point = json!({"width":d,"mode":mode,"index":i,"label":pair.label,"origin":pair.origin,"forward_pass":f,"backward_pass":b,"finite":actual.is_finite(),"actual":actual,"actual_f64_bits":format!("{:016x}",actual.to_bits()),"exact_dot":t,"absolute_error":e,"max_abs_operand":max_abs_operand,"sum_abs_products_display":sum_abs_products});
                serde_json::to_writer(&mut observations, &point)?;
                observations.write_all(b"\n")?;
            }
            raw.flush()?;
            summaries.push(json!({"width":d,"mode":mode,"pairs":group.len(),"forward_failures":forward_failures,"backward_failures":backward_failures,"nonfinite":nonfinite}));
        }
    }
    observations.flush()?;
    let baseline_passed = summaries
        .iter()
        .filter(|r| r["mode"] == 0)
        .all(|r| r["forward_failures"] == 0 && r["backward_failures"] == 0 && r["nonfinite"] == 0);
    let result = json!({"purpose":"static_reduction_screen_not_whole_model_certification","gpu":cuda.name(),"sm":cuda.sm(),"source_sha256":runner::source_hash(),"base_ptx_sha256":runner::ptx_hash(),"screen_ptx_sha256":hash(precision_kernels::kernels::PTX.as_bytes()),"executable_sha256":hash(&fs::read(std::env::current_exe()?)?),"checkpoints":checkpoints,"unique_pairs":pairs.len(),"summaries":summaries,"training_updates":0,"oracle":"exact_big_integer_FP32_products_and_exact_rational_error_budgets","runtime_precision_selection":false});
    fs::write(
        root.join("report.json"),
        serde_json::to_vec_pretty(&result)?,
    )?;
    fs::write(root.join("screen.ptx"), precision_kernels::kernels::PTX)?;
    println!(
        "{}",
        json!({"pairs":pairs.len(),"variants":9,"baseline_passed":baseline_passed,"output":root})
    );
    if !baseline_passed {
        return Err("FP64 reference fails exact-oracle budget; screen invalid".into());
    }
    Ok(())
}
