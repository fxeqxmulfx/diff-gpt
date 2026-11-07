//! Causal centering preflight against the immutable successful ordinary control.
use crate::continuation::{durable, record};
use crate::gradient_oracle::{self as oracle, Jet};
use crate::*;
use kernite_cuda::{Cuda, Grid};

pub(crate) const PLAN: &str = include_str!("../CENTERING_PLAN.txt");

fn primitive(
    d: usize,
    length: usize,
    batch: usize,
    scale: f32,
    repeated: bool,
    uniform: bool,
    qk_scale: f32,
) -> Result<Value> {
    let cuda = Cuda::new()?;
    let base = attention::kernels::Kernels::load(&cuda)?;
    let ordinary = ordinary_attention::kernels::Kernels::load(&cuda)?;
    let centered = centered_attention::kernels::Kernels::load(&cuda)?;
    let r = batch * length;
    let mut x = vec![0f32; r * 3 * d];
    let dy: Vec<_> = (0..r * d)
        .map(|i| 0.1 * (0.217 * (i + 1) as f32).sin())
        .collect();
    for row in 0..r {
        for part in 0..3 {
            for c in 0..d {
                let index = if repeated && part == 2 {
                    row / length * 3 * d + part * d + c
                } else {
                    row * 3 * d + part * d + c
                };
                x[row * 3 * d + part * d + c] =
                    (0.173 * (index + 1) as f32).sin() * if part == 2 { scale } else { 1. };
                if uniform && part < 2 {
                    x[row * 3 * d + part * d + c] = 0.;
                }
                if part < 2 {
                    x[row * 3 * d + part * d + c] *= qk_scale;
                }
            }
        }
    }
    let sentinel = 12345.25;
    let qkv = cuda.upload(&x)?;
    let upstream = cuda.upload(&dy)?;
    let out = cuda.upload(&vec![sentinel; r * d + 32])?;
    let prob = cuda.upload(&vec![sentinel; batch * length * length + 32])?;
    let coefficients = cuda.upload(&vec![sentinel as f64; batch * length * length + 32])?;
    let delta = cuda.upload(&vec![sentinel; r * d + 32])?;
    let guard = cuda.upload(&vec![0u32; batch + 32])?;
    let dqkv = cuda.upload(&vec![sentinel; x.len() + 32])?;
    let ds = cuda.upload(&vec![sentinel; batch * length * length + 32])?;
    let gd = cuda.upload(&vec![sentinel; r * d + 32])?;
    let dv = cuda.upload(&vec![sentinel; r * d + 32])?;
    macro_rules! forward {
        ($d:literal) => {
            unsafe {
                base.forward::<$d>(
                    Grid::new(r as u32, 32),
                    out.ptr(),
                    prob.ptr(),
                    delta.ptr(),
                    guard.ptr(),
                    qkv.ptr(),
                    1,
                    length as u32,
                    1,
                )?;
            }
        };
    }
    match d {
        16 => forward!(16),
        32 => forward!(32),
        128 => forward!(128),
        _ => unreachable!(),
    }
    let probabilities = prob.read()?;
    macro_rules! run {
        ($d:literal) => {
            unsafe {
                centered.coefficients::<$d>(
                    Grid::new(r as u32, 32),
                    coefficients.ptr(),
                    prob.ptr(),
                    qkv.ptr(),
                    1,
                    length as u32,
                )?;
                centered.values::<$d>(
                    Grid::new(r as u32, 32),
                    out.ptr(),
                    coefficients.ptr(),
                    qkv.ptr(),
                    1,
                    length as u32,
                )?;
                ordinary.query_backward::<$d>(
                    Grid::new(r as u32, 32),
                    dqkv.ptr(),
                    ds.ptr(),
                    gd.ptr(),
                    dv.ptr(),
                    upstream.ptr(),
                    prob.ptr(),
                    qkv.ptr(),
                    1,
                    length as u32,
                )?;
                centered.key_backward::<$d>(
                    Grid::new(r as u32, 32),
                    dqkv.ptr(),
                    ds.ptr(),
                    gd.ptr(),
                    coefficients.ptr(),
                    qkv.ptr(),
                    1,
                    length as u32,
                )?;
            }
        };
    }
    match d {
        16 => run!(16),
        32 => run!(32),
        128 => run!(128),
        _ => unreachable!(),
    }
    let y = out.read()?;
    let g = dqkv.read()?;
    let score_gradient = ds.read()?;
    let value_gradient = gd.read()?;
    let correction_gradient = dv.read()?;
    let centered_coefficients = coefficients.read()?;
    require(
        y[r * d..]
            .iter()
            .chain(&g[x.len()..])
            .chain(&score_gradient[batch * length * length..])
            .chain(&value_gradient[r * d..])
            .chain(&correction_gradient[r * d..])
            .all(|&v| v == sentinel)
            && centered_coefficients[batch * length * length..]
                .iter()
                .all(|&v| v == sentinel as f64)
            && prob.read()? == probabilities
            && qkv.read()? == x
            && upstream.read()? == dy,
        "ordinary primitive changed inputs or launch tails",
    )?;
    require(
        y.iter().chain(&g).all(|v| v.is_finite()),
        "nonfinite ordinary primitive",
    )?;
    let mut cfg = gradient_stress::synthetic("primitive", 64, 4, 1, batch, length).cfg;
    cfg.width = d;
    cfg.heads = 1;
    cfg.gain = 1. / d as f32;
    let input: Vec<_> = x.iter().map(|&v| Jet::from(v as f64)).collect();
    let expected = oracle::attention_qkv_kind(&cfg, &input, batch, length, AttentionMode::Centered);
    let error: Vec<_> = y
        .iter()
        .zip(&expected)
        .map(|(&v, e)| v as f64 - e.v)
        .collect();
    let norm = stats::norm(&expected.iter().map(|v| v.v).collect::<Vec<_>>());
    let forward_error = stats::norm(&error);
    let mut checks = Vec::new();
    let mut coordinates = 0;
    if length <= 7 {
        let mut reference = Vec::with_capacity(x.len());
        let mut seeded = input.clone();
        for i in 0..x.len() {
            seeded[i].d = 1.;
            reference.push(
                oracle::attention_qkv_kind(&cfg, &seeded, batch, length, AttentionMode::Centered)
                    .iter()
                    .zip(&dy)
                    .map(|(v, &u)| v.d * u as f64)
                    .sum::<f64>(),
            );
            seeded[i].d = 0.;
        }
        coordinates = reference.len();
        for (part, name) in ["Q", "K", "V"].iter().enumerate() {
            let ids: Vec<_> = (0..r)
                .flat_map(|row| (0..d).map(move |c| row * 3 * d + part * d + c))
                .collect();
            let e = stats::norm(
                &ids.iter()
                    .map(|&i| g[i] as f64 - reference[i])
                    .collect::<Vec<_>>(),
            );
            let n = stats::norm(&ids.iter().map(|&i| reference[i]).collect::<Vec<_>>());
            let tolerance = 5e-6 * n + 1e-12;
            checks.push(json!({"part":name,"gradient_error":e,"reference_norm":n,"tolerance":tolerance,"passed":e<=tolerance}));
        }
    } else {
        for (part, name) in ["Q", "K", "V"].iter().enumerate() {
            let ids: Vec<_> = (0..r)
                .flat_map(|row| (0..d).map(move |c| row * 3 * d + part * d + c))
                .collect();
            let n = stats::norm(&ids.iter().map(|&i| g[i] as f64).collect::<Vec<_>>());
            for direction in 0..4 {
                if direction == 3 && n == 0. {
                    continue;
                }
                let mut seeded = input.clone();
                if direction < 2 {
                    let mut total = 0.;
                    for &i in &ids {
                        seeded[i].d = ((i + 1) as f64 * (0.19 + direction as f64 * 0.31)).sin();
                        total += seeded[i].d.powi(2);
                    }
                    for &i in &ids {
                        seeded[i].d /= total.sqrt();
                    }
                } else if direction == 2 {
                    let &i = ids
                        .iter()
                        .max_by(|&&a, &&b| g[a].abs().total_cmp(&g[b].abs()))
                        .unwrap();
                    seeded[i].d = 1.;
                } else {
                    for &i in &ids {
                        seeded[i].d = g[i] as f64 / n;
                    }
                }
                let expected = oracle::attention_qkv_kind(
                    &cfg,
                    &seeded,
                    batch,
                    length,
                    AttentionMode::Centered,
                )
                .iter()
                .zip(&dy)
                .map(|(v, &u)| v.d * u as f64)
                .sum::<f64>();
                let actual: f64 = ids.iter().map(|&i| g[i] as f64 * seeded[i].d).sum();
                let e = (actual - expected).abs();
                let tolerance = 5e-6 * n + 1e-12;
                checks.push(json!({"part":name,"direction":direction,"actual":actual,"expected":expected,"absolute_error":e,"tolerance":tolerance,"passed":e<=tolerance}));
            }
        }
    }
    // Exact numerical equality: +0 and -0 represent the same value. The
    // anchored sum may canonicalize signed zero without changing the model.
    let equal_values_canceled = !repeated || (0..r).all(|row| (0..d).all(|c| y[row * d + c] == 0.));
    let first_values_canceled = (0..batch).all(|b| (0..d).all(|c| y[b * length * d + c] == 0.));
    let uniform_rows_canceled = !uniform || y[..r * d].iter().all(|&v| v == 0.);
    let passed = forward_error <= 5e-6 * norm + 1e-12
        && checks.iter().all(|c| c["passed"] == true)
        && equal_values_canceled
        && first_values_canceled
        && uniform_rows_canceled;
    Ok(
        json!({"head_width":d,"length":length,"batch":batch,"value_scale":scale,"repeated_values":repeated,
        "forward_error":forward_error,"reference_norm":norm,"checks":checks,"checked_gradient_coordinates":coordinates,
        "equal_values_canceled":equal_values_canceled,"first_values_canceled":first_values_canceled,
        "uniform_scores":uniform,"qk_scale":qk_scale,"uniform_rows_canceled":uniform_rows_canceled,
        "probabilities_unchanged":true,"input_and_tails_unchanged":true,"passed":passed}),
    )
}

pub fn stress(parent_root: &Path, control: &Path, output: &Path) -> Result<()> {
    require(
        !output.exists() && !output.starts_with(control),
        "choose a new centering stress output outside the control",
    )?;
    let before = next_update::files(control)?;
    let control_audit =
        plain_ffn::control_input_mode(parent_root, control, AttentionMode::Centered)?;
    let (parent, task, _) = retrain::setup(parent_root)?;
    let original = retrain::protected(&parent)?;
    let mut cases: Vec<_> = gradient_stress::synthetic_cases()
        .into_iter()
        .filter(|c| layout::is_complex(&c.cfg.architecture))
        .collect();
    for (name, w, h, l, b, t) in [
        ("centered_wide_depth2", 128, 1, 2, 1, 4),
        ("centered_wide_masks", 128, 1, 1, 3, 7),
        ("centered_repeated_two", 128, 4, 1, 3, 2),
        ("centered_repeated_long", 64, 4, 1, 1, 128),
        ("centered_first_only", 64, 4, 2, 3, 7),
    ] {
        let mut c = gradient_stress::synthetic(name, w, h, l, b, t);
        if name.contains("repeated") {
            for row in 0..b {
                c.tokens[row * t..(row + 1) * t].fill(row as u32 + 1);
            }
        }
        if name == "centered_first_only" {
            c.labels.fill(u32::MAX);
            for row in 0..b {
                c.labels[row * t] = (row as u32 + 7) % 17;
            }
        }
        cases.push(c);
    }
    for (name, scale) in [
        ("centered_wide_value_zero", 0.),
        ("centered_wide_value_tiny", 1e-13),
        ("centered_wide_value_small", 1e-9),
    ] {
        let mut c = gradient_stress::synthetic(name, 128, 1, 1, 2, 4);
        gradient_stress::scale(&mut c, &["V"], scale);
        cases.push(c);
    }
    let synthetic_count = cases.len();
    for c in cases.iter_mut().filter(|c| c.name == "confident_readout") {
        let p: Vec<_> = c.p.iter().map(|&v| Jet::from(v as f64)).collect();
        let z = oracle::model_mode(
            &c.cfg,
            &p,
            &c.tokens,
            c.batch,
            c.length,
            true,
            AttentionMode::Centered,
        );
        for (row, values) in z.chunks_exact(c.cfg.vocab).enumerate() {
            if c.labels[row] != u32::MAX {
                c.labels[row] = values
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.v.total_cmp(&b.1.v))
                    .unwrap()
                    .0 as u32;
            }
        }
    }
    let train = &task.splits["train"];
    let mut packed = metrics::Packed::default();
    packed.fill(
        train,
        &[0, (train.len() / 3) as u32, (2 * train.len() / 3) as u32],
    );
    for cp in read(&control.join("result.json"))?["snapshots"]
        .as_array()
        .unwrap()
    {
        let step = cp["step"].as_u64().unwrap();
        let (p, _, _) = retrain::decode(
            &control.join(format!("checkpoint-{step:06}.bin")),
            parent.p.len(),
        )?;
        let mut cfg = parent.cfg.clone();
        cfg.max_batch = 3;
        cases.push(gradient_stress::Case {
            name: format!("ordinary-saved-step{step}"),
            cfg,
            p,
            tokens: packed.tokens.clone(),
            labels: packed.labels.clone(),
            batch: 3,
            length: 2,
        });
    }
    fs::create_dir(output)?;
    let source = retrain::archive(output)?;
    let protocol = json!({"purpose":"causal centering independent AD preflight; no training","attention":"centered_causal","ordinary_attention":true,"centered_attention":true,
        "source":source,"plan_sha256":hash(PLAN.as_bytes()),"input_files":before,"original_files":original,"control_audit":control_audit,
        "plain_product_ptx_sha256":hash(plain_product::kernels::PTX.as_bytes()),"ordinary_attention_ptx_sha256":hash(ordinary_attention::kernels::PTX.as_bytes()),"centered_attention_ptx_sha256":hash(centered_attention::kernels::PTX.as_bytes()),
        "synthetic_cases":synthetic_count,"saved_cases":cases.len()-synthetic_count,"model_cases":cases.len(),
        "direction_bound":"5e-4 * independent block gradient norm + 1e-12","primitive_bound":"5e-6 * block/reference norm + 1e-12",
        "python_invoked":false});
    record(&output.join("protocol.json"), &protocol)?;
    durable(&output.join("PLAN.txt"), PLAN.as_bytes())?;
    let mut primitives = Vec::new();
    for d in [16, 32, 128] {
        for t in [1, 2, 7, 17, 128] {
            primitives.push(primitive(d, t, 1, 1., false, false, 1.)?);
        }
        for scale in [0., 1e-13, 1e8] {
            primitives.push(primitive(d, 7, 1, scale, false, false, 1.)?);
        }
        primitives.push(primitive(d, 7, 1, 1., true, false, 1.)?);
        primitives.push(primitive(d, 2, 3, 1., true, false, 1.)?);
        primitives.push(primitive(d, 7, 1, 1., false, true, 1.)?);
        primitives.push(primitive(d, 7, 1, 1., false, false, 1e-3)?);
    }
    record(&output.join("primitives.json"), &json!(primitives))?;
    let mut rows = Vec::new();
    for (index, c) in cases.iter().enumerate() {
        let row = gradient_stress::check_case_mode(c, true, AttentionMode::Centered)?;
        record(&output.join(format!("centered-{index:03}.json")), &row)?;
        println!("{}", json!({"stress":c.name,"passed":row["passed"]}));
        rows.push(row);
    }
    let repeated = cases
        .iter()
        .position(|c| c.name == "centered_repeated_two")
        .unwrap();
    let mut regression = Vec::new();
    for index in [0, 4, repeated, cases.len() - 1] {
        let row = gradient_stress::check_case_variant(&cases[index], true, true)?;
        record(&output.join(format!("ordinary-{index:03}.json")), &row)?;
        regression.push(row);
    }
    let mut restored = Vec::new();
    for path in ["layer0.V", "layer0.attention_output"] {
        let new = rows[repeated]["directional_checks"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["path"] == path)
            .unwrap()["native_block_gradient_norm"]
            .as_f64()
            .unwrap();
        let old = regression[2]["directional_checks"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["path"] == path)
            .unwrap()["native_block_gradient_norm"]
            .as_f64()
            .unwrap();
        restored.push(json!({"path":path,"centered_gradient_norm":new,"ordinary_gradient_norm":old,"passed":new==0.&&old>1e-10}));
    }
    require(
        next_update::files(control)? == before && retrain::protected(&parent)? == original,
        "centering stress changed inputs",
    )?;
    let passed = rows
        .iter()
        .chain(&regression)
        .chain(&primitives)
        .chain(&restored)
        .all(|r| r["passed"] == true);
    record(
        &output.join("summary.json"),
        &json!({"complete":true,"passed":passed,"protocol":protocol,"rows":rows,"ordinary_regression":regression,
        "primitives":primitives,"canceled_equal_token_paths":restored,"input_artifacts_unchanged":true,"training_updates":0}),
    )?;
    require(
        passed,
        "centered attention stress failed; preserve output and do not train",
    )
}

#[test]
fn centered_oracle_cancels_shared_values_but_not_independent_perturbations() {
    let c = gradient_stress::synthetic("centered-oracle", 64, 4, 1, 1, 3);
    let w = c.cfg.width;
    let mut x = vec![Jet::from(0.); 3 * 3 * w];
    for row in 0..3 {
        for col in 0..w {
            x[row * 3 * w + 2 * w + col] = Jet::from((col + 1) as f64 / 64.);
        }
    }
    x[3 * w] = Jet::from(2.);
    x[w] = Jet::from(1.);
    x[4 * w] = Jet::from(-1.);
    let run = |x: &[Jet]| oracle::attention_qkv_kind(&c.cfg, x, 1, 3, AttentionMode::Centered);
    assert!(run(&x).iter().all(|v| v.v == 0. && v.d == 0.));
    x[2 * w].d = 1.;
    let independent = run(&x);
    assert_eq!(independent[0].d, 0.);
    assert!(independent[w].d.abs() > 0.01);
    x[5 * w].d = 1.;
    x[8 * w].d = 1.;
    assert!(run(&x).iter().all(|v| v.v == 0. && v.d == 0.));
    let before = run(&x);
    for value in &mut x[6 * w..] {
        *value = Jet { v: 1000., d: 17. };
    }
    let after = run(&x);
    assert!(
        before[..2 * w]
            .iter()
            .zip(&after)
            .all(|(a, b)| a.v == b.v && a.d == b.d)
    );
}

#[test]
fn centered_oracle_removes_uniform_rows_with_unequal_values() {
    let c = gradient_stress::synthetic("centered-uniform", 64, 4, 1, 1, 2);
    let w = c.cfg.width;
    let mut x = vec![Jet::from(0.); 6 * w];
    for col in 0..w {
        x[2 * w + col] = Jet::from(col as f64 + 1.);
        x[5 * w + col] = Jet::from(-2. * col as f64);
    }
    let z = oracle::attention_qkv_kind(&c.cfg, &x, 1, 2, AttentionMode::Centered);
    assert!(z.iter().all(|v| v.v == 0. && v.d == 0.));
}
