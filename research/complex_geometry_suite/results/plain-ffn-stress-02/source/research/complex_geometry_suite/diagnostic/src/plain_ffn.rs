//! Validated, single-variable plain-product FFN experiment; no baseline mutation.
use crate::continuation::{durable, record};
use crate::gradient_oracle::{self as oracle, Jet};
use crate::*;
use kernite_cuda::{Cuda, Grid};
use std::time::Instant;

const PLAN: &str = include_str!("../PLAIN_FFN_PLAN.txt");
const BUDGET: usize = 100000;
const OBSERVE: usize = 19;
const FLUSH: usize = 500 * OBSERVE;

fn primitive_checks() -> Result<Vec<Value>> {
    let cuda = Cuda::new()?;
    let kernel = plain_product::kernels::Kernels::load(&cuda)?;
    let scales = [0., 1e-10, 1e-4, 1., 20., 1e4, 1e8, 1.];
    let rows = scales.len();
    let mut result = Vec::new();
    for pairs in [1, 3, 17, 341] {
        let mut x = vec![0f32; rows * 4 * pairs];
        let mut dy = vec![0f32; rows * 2 * pairs];
        for (row, &scale) in scales.iter().enumerate() {
            for c in 0..4 * pairs {
                let s = if row == 0 && c >= 2 * pairs {
                    1.
                } else if row == 7 {
                    if c < 2 * pairs { 1e20 } else { 1e-20 }
                } else {
                    scale
                };
                x[row * 4 * pairs + c] = s * ((c as f32 * 0.173).sin() + 0.21);
            }
            for c in 0..2 * pairs {
                dy[row * 2 * pairs + c] = 0.1 * (c as f32 * 0.317).cos();
            }
        }
        let input = cuda.upload(&x)?;
        let upstream = cuda.upload(&dy)?;
        let sentinel = 12345.25f32;
        let output = cuda.upload(&vec![sentinel; dy.len() + 32])?;
        let gradient = cuda.upload(&vec![sentinel; x.len() + 32])?;
        let grid = Grid::new((rows * pairs).div_ceil(256) as u32, 256);
        unsafe {
            kernel.forward(
                grid,
                output.ptr(),
                input.ptr(),
                pairs as u32,
                (rows * pairs) as u32,
            )?;
            kernel.backward(
                Grid::new((rows * pairs).div_ceil(256) as u32, 256),
                gradient.ptr(),
                input.ptr(),
                upstream.ptr(),
                pairs as u32,
                (rows * pairs) as u32,
            )?;
        }
        let y = output.read()?;
        let g = gradient.read()?;
        require(
            y[dy.len()..]
                .iter()
                .chain(&g[x.len()..])
                .all(|&v| v.to_bits() == sentinel.to_bits()),
            "plain kernel wrote beyond its tail",
        )?;
        require(
            input.read()? == x && upstream.read()? == dy,
            "plain kernel mutated input",
        )?;
        require(
            y.iter().chain(&g).all(|v| v.is_finite()),
            "nonfinite plain primitive",
        )?;
        for (row, &scale) in scales.iter().enumerate() {
            let base: Vec<_> = x[row * 4 * pairs..(row + 1) * 4 * pairs]
                .iter()
                .map(|&v| Jet::from(v as f64))
                .collect();
            let reference = oracle::plain_product(&base);
            let fwd_error: Vec<_> = reference
                .iter()
                .enumerate()
                .map(|(c, v)| y[row * 2 * pairs + c] as f64 - v.v)
                .collect();
            let fwd: Vec<_> = reference.iter().map(|v| v.v).collect();
            let mut reference_gradient = Vec::new();
            for c in 0..4 * pairs {
                let mut seeded = base.clone();
                seeded[c].d = 1.;
                reference_gradient.push(
                    oracle::plain_product(&seeded)
                        .iter()
                        .enumerate()
                        .map(|(j, v)| v.d * dy[row * 2 * pairs + j] as f64)
                        .sum::<f64>(),
                );
            }
            let grad_error: Vec<_> = reference_gradient
                .iter()
                .enumerate()
                .map(|(c, &v)| g[row * 4 * pairs + c] as f64 - v)
                .collect();
            let forward = stats::norm(&fwd_error) / stats::norm(&fwd).max(1e-300);
            let backward = stats::norm(&grad_error) / stats::norm(&reference_gradient).max(1e-300);
            result.push(json!({"pairs":pairs,"row":row,"scale":scale,"mixed_scale":row==7,"forward_relative_error":forward,
                "gradient_relative_error":backward,"checked_gradient_coordinates":4*pairs,"guarded_tails":true,"passed":forward<=5e-6&&backward<=5e-6}));
        }
    }
    Ok(result)
}
fn radial_input(parent: &Path, radial: &Path) -> Result<Value> {
    require(
        read(&radial.join("protocol.json"))?["ffn"] != "plain_complex_product",
        "control must use the radial FFN",
    )?;
    retrain::audit_readonly(parent, radial)
}
pub fn stress(parent_root: &Path, radial: &Path, output: &Path) -> Result<()> {
    require(
        !output.exists() && !output.starts_with(radial),
        "choose a new stress output outside the control",
    )?;
    let before = next_update::files(radial)?;
    let control_audit = radial_input(parent_root, radial)?;
    let (parent, task, _) = retrain::setup(parent_root)?;
    let original = retrain::protected(&parent)?;
    let mut cases: Vec<_> = gradient_stress::synthetic_cases()
        .into_iter()
        .filter(|c| layout::is_complex(&c.cfg.architecture))
        .collect();
    for (name, layers, batch, length) in [("wide_depth2", 2, 1, 4), ("wide_masks_tail", 1, 3, 7)] {
        cases.push(gradient_stress::synthetic(
            name, 128, 1, layers, batch, length,
        ));
    }
    let mut real_xsa = gradient_stress::synthetic("wide_real_xsa", 128, 1, 1, 2, 4);
    real_xsa.cfg.architecture = "complex-linear".into();
    real_xsa.p = layout::initialize_model(&real_xsa.cfg, 7);
    cases.push(real_xsa);
    for (name, scale) in [
        ("wide_value_zero", 0.),
        ("wide_value_clipped", 1e-13),
        ("wide_value_small", 1e-9),
    ] {
        let mut c = gradient_stress::synthetic(name, 128, 1, 1, 2, 4);
        gradient_stress::scale(&mut c, &["V"], scale);
        cases.push(c);
    }
    let synthetic_count = cases.len();
    // Reassign the confident labels using the plain model, retaining its intended stress.
    for c in cases.iter_mut().filter(|c| c.name == "confident_readout") {
        let p: Vec<_> = c.p.iter().map(|&v| Jet::from(v as f64)).collect();
        let z = oracle::model_with_product(&c.cfg, &p, &c.tokens, c.batch, c.length, true);
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
    let result = read(&radial.join("result.json"))?;
    let mut packed = metrics::Packed::default();
    let train = &task.splits["train"];
    packed.fill(
        train,
        &[0, (train.len() / 3) as u32, (2 * train.len() / 3) as u32],
    );
    for cp in result["snapshots"].as_array().unwrap() {
        let step = cp["step"].as_u64().unwrap();
        let (p, _, _) = retrain::decode(
            &radial.join(format!("checkpoint-{step:06}.bin")),
            parent.p.len(),
        )?;
        let mut cfg = parent.cfg.clone();
        cfg.max_batch = 3;
        cases.push(gradient_stress::Case {
            name: format!("radial-saved-step{step}"),
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
    let plan = json!({"purpose":"plain FFN independent AD preflight; no training","source":source,"plan_sha256":hash(PLAN.as_bytes()),
        "plain_product_ptx_sha256":hash(plain_product::kernels::PTX.as_bytes()),"control_audit":control_audit,
        "input_files":before,"original_files":original,"synthetic_cases":synthetic_count,"saved_cases":cases.len()-synthetic_count,
        "plain_cases":cases.len(),"radial_regression_cases":3,"primitive_rows":32,
        "wide_chain_ptx_sha256":hash(stable_chain::kernels::PTX.as_bytes()),
        "direction_bound":"5e-4 * block gradient L2 + 1e-12","forward_bound":"5e-5 of scale","primitive_bound":5e-6,"python_invoked":false});
    record(&output.join("protocol.json"), &plan)?;
    durable(&output.join("PLAN.txt"), PLAN.as_bytes())?;
    let primitive = primitive_checks()?;
    record(&output.join("primitives.json"), &json!(primitive))?;
    let mut rows = Vec::new();
    for (index, case) in cases.iter().enumerate() {
        let row = gradient_stress::check_case_product(case, true)?;
        record(&output.join(format!("plain-{index:03}.json")), &row)?;
        println!("{}", json!({"stress":case.name,"passed":row["passed"]}));
        rows.push(row);
    }
    let mut radial_rows = Vec::new();
    for &index in &[0, 4, cases.len() - 1] {
        let row = gradient_stress::check_case_product(&cases[index], false)?;
        record(&output.join(format!("radial-{index:03}.json")), &row)?;
        radial_rows.push(row);
    }
    require(
        next_update::files(radial)? == before && retrain::protected(&parent)? == original,
        "stress changed inputs",
    )?;
    let trace = gradient_stress::trace_head_width_128_product(true, true)?;
    record(&output.join("wide-composed-adjoint.json"), &trace)?;
    let passed = trace["local_adjoints"]
        .as_array()
        .unwrap()
        .iter()
        .all(|r| r["passed"] == true)
        && primitive
            .iter()
            .chain(&rows)
            .chain(&radial_rows)
            .all(|r| r["passed"] == true);
    let summary = json!({"complete":true,"passed":passed,"protocol":plan,"primitives":primitive,"rows":rows,"radial_regression":radial_rows,
        "wide_composed_adjoint":trace,"input_artifacts_unchanged":true,"training_updates":0});
    record(&output.join("summary.json"), &summary)?;
    require(
        passed,
        "plain FFN stress failed; retain results and do not train",
    )
}
pub fn train(parent_root: &Path, radial: &Path, stress_root: &Path, output: &Path) -> Result<()> {
    require(
        !output.exists() && !output.starts_with(radial),
        "refuse overwrite or training inside control",
    )?;
    let before = next_update::files(radial)?;
    let control_audit = radial_input(parent_root, radial)?;
    let stress = read(&stress_root.join("summary.json"))?;
    require(
        stress["complete"] == true
            && stress["passed"] == true
            && stress["protocol"]["input_files"] == before
            && stress["protocol"]["plain_product_ptx_sha256"]
                == hash(plain_product::kernels::PTX.as_bytes()),
        "preflight not passed for these inputs and kernels",
    )?;
    let (parent, task, samples) = retrain::setup(parent_root)?;
    let original = retrain::protected(&parent)?;
    fs::create_dir(output)?;
    let source = retrain::archive(output)?;
    require(
        source == stress["protocol"]["source"],
        "compiled source differs from stress-tested source",
    )?;
    let initial: Vec<_> = parent
        .initial
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let protocol = json!({"purpose":"plain versus radial complex-product FFN, same seed1 initial state and samples","case":parent.protocol["case"],
        "seed":1,"arm":"complex-hermitian","ffn":"plain_complex_product","baseline_protocol":parent.protocol,"original_files":original,
        "radial_control_files":before,"radial_control_audit":control_audit,"source":source,"plan_sha256":hash(PLAN.as_bytes()),
        "budget":BUDGET,"observe":OBSERVE,"checkpoint_epochs":500,"initial_parameters_sha256":hash(&initial),"sampler_sha256":samples.future_hash,
        "original_ptx_sha256":ptx_hash(),"corrected_loss_ptx_sha256":hash(stable_loss::kernels::PTX.as_bytes()),
        "corrected_attention_ptx_sha256":hash(stable_attention::kernels::PTX.as_bytes()),"plain_product_ptx_sha256":hash(plain_product::kernels::PTX.as_bytes()),
        "wide_chain_ptx_sha256":hash(stable_chain::kernels::PTX.as_bytes()),"training_uses_wide_chain":false,
        "stress_source_sha256":stress["protocol"]["source"]["sha256"],"stress_summary_sha256":hash(&fs::read(stress_root.join("summary.json"))?),
        "stress_root":stress_root.canonicalize()?.to_string_lossy(),"optimizer_moments":"zero at original initialization",
        "stopping_rule":"first observed heldout exact>=99%,100000updates,or explicit STOP","python_invoked":false,"benchmark_modified":false});
    require(
        protocol["initial_parameters_sha256"] == parent.protocol["initial_parameters_sha256"],
        "initial weights differ",
    )?;
    record(&output.join("protocol.json"), &protocol)?;
    durable(&output.join("initial.f32"), &initial)?;
    durable(&output.join("PLAN.txt"), PLAN.as_bytes())?;
    let zeros = vec![0.; parent.initial.len()];
    let mut e = retrain::corrected(parent.cfg.clone(), &parent.initial, &zeros, &zeros)?;
    e.enable_plain_product()?;
    let started = Instant::now();
    let mut history = Vec::new();
    let mut snapshots = Vec::new();
    let (mut completed, mut window_end, mut previous) = (0, 0, 0);
    let mut packed = metrics::Packed::default();
    loop {
        let point = retrain::observe(&mut e, &task, &samples, completed, &started)?;
        history.extend(serde_json::to_vec(&point)?);
        history.push(b'\n');
        let passed = point["validation"]["exact"].as_u64().unwrap() * 100 >= 99 * 4705;
        let stopped = output.join("STOP").exists();
        let done = passed || stopped || completed == BUDGET;
        if completed == 0 || completed == window_end || done {
            snapshots.push(retrain::checkpoint(
                &e, output, &history, &point, &protocol,
            )?);
            println!(
                "{}",
                json!({"updates":completed,"epochs":point["epochs"],"heldout":point["validation"]["sequence_accuracy"],
                "diagonal_correct":point["validation"]["diagonal"]["exact"],"train":point["train"]["sequence_accuracy"],"seconds":point["seconds"]})
            );
            if !done {
                window_end = ((completed / FLUSH + 1) * FLUSH).min(BUDGET);
                record(
                    &output.join("execution.json"),
                    &json!({"clean":false,"durable_completed":completed,"possible_completed":window_end}),
                )?;
            }
        }
        if done {
            let result = json!({"case":CASE,"seed":1,"arm":"complex-hermitian","ffn":"plain_complex_product",
                "status":if passed{"passed"}else if stopped{"stopped"}else{"budget_exhausted"},"completed":completed,"examples":samples.edges[completed],
                "epochs":point["epochs"],"first99_epochs":if passed{Some(point["epochs"].clone())}else{None},
                "previous_observed_epochs":if passed{Some(samples.edges[previous] as f64/4704.)}else{None},
                "endpoint":point,"snapshots":snapshots,"source_sha256":source["sha256"],"zero_rate_first_update_checked":true});
            record(&output.join("result.json"), &result)?;
            require(
                next_update::files(radial)? == before && retrain::protected(&parent)? == original,
                "training changed control files",
            )?;
            return audit(parent_root, radial, output);
        }
        previous = completed;
        let next = (completed + OBSERVE).min(window_end).min(BUDGET);
        while completed < next {
            let ids = &samples.order[samples.edges[completed]..samples.edges[completed + 1]];
            packed.fill(&task.splits["train"], ids);
            e.forward(&packed.tokens, &packed.labels, ids.len(), packed.length)?;
            e.backward(ids.len(), packed.length)?;
            e.update(completed)?;
            if completed == 0 {
                require(
                    e.parameters
                        .read()?
                        .iter()
                        .zip(&parent.initial)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "zero-rate first update changed parameters",
                )?;
                require(
                    e.first_moment.read()?.iter().any(|&x| x != 0.)
                        && e.second_moment.read()?.iter().any(|&x| x > 0.),
                    "zero-rate first update omitted moments",
                )?;
            }
            completed += 1;
        }
    }
}
fn points(root: &Path) -> Result<Vec<Value>> {
    fs::read_to_string(root.join("history.jsonl"))?
        .lines()
        .map(|s| Ok(serde_json::from_str(s)?))
        .collect()
}
pub fn audit(parent_root: &Path, radial: &Path, output: &Path) -> Result<()> {
    let before = next_update::files(radial)?;
    let control = radial_input(parent_root, radial)?;
    let candidate = retrain::audit_readonly(parent_root, output)?;
    let protocol = read(&output.join("protocol.json"))?;
    require(
        protocol["ffn"] == "plain_complex_product"
            && protocol["radial_control_files"] == before
            && candidate["ffn"] == "plain_complex_product",
        "wrong comparison provenance",
    )?;
    let stress_path = Path::new(
        protocol["stress_root"]
            .as_str()
            .ok_or("missing stress path")?,
    );
    let stress_bytes = fs::read(stress_path.join("summary.json"))?;
    let stress: Value = serde_json::from_slice(&stress_bytes)?;
    require(
        hash(&stress_bytes) == protocol["stress_summary_sha256"]
            && stress["passed"] == true
            && stress["protocol"]["source"] == protocol["source"],
        "stress provenance changed",
    )?;
    let (parent, task, samples) = retrain::setup(parent_root)?;
    let rows = points(output)?;
    let radial_rows = points(radial)?;
    let paired = rows
        .iter()
        .zip(&radial_rows)
        .take_while(|(a, b)| a["step"] == b["step"])
        .count();
    require(
        paired == rows.len(),
        "candidate observations are not paired with control",
    )?;
    for row in &rows {
        for (split, n, d) in [("train", 4704, 55), ("validation", 4705, 42)] {
            let s = &row[split];
            let diag = &s["diagonal"];
            let other = &s["non_diagonal"];
            require(
                diag["examples"] == d
                    && other["examples"] == n - d
                    && diag["exact"].as_u64().ok_or("no subgroup count")? <= d as u64
                    && other["exact"].as_u64().ok_or("no subgroup count")? <= (n - d) as u64
                    && s["exact"].as_u64().unwrap()
                        == diag["exact"].as_u64().unwrap() + other["exact"].as_u64().unwrap()
                    && (s["loss"].as_f64().unwrap() * n as f64
                        - diag["loss"].as_f64().unwrap() * d as f64
                        - other["loss"].as_f64().unwrap() * (n - d) as f64)
                        .abs()
                        < 1e-7,
                "subgroup accounting failed",
            )?;
        }
    }
    let mut replays = Vec::new();
    for cp in read(&output.join("result.json"))?["snapshots"]
        .as_array()
        .unwrap()
    {
        let step = cp["step"].as_u64().unwrap() as usize;
        let path = output.join(format!("checkpoint-{step:06}.bin"));
        let (p, m, v) = retrain::decode(&path, parent.p.len())?;
        let mut e = retrain::corrected(parent.cfg.clone(), &p, &m, &v)?;
        e.enable_plain_product()?;
        let actual = retrain::observe(&mut e, &task, &samples, step, &Instant::now())?;
        let expected = rows.iter().find(|r| r["step"] == step).unwrap();
        for split in ["train", "validation"] {
            for group in [None, Some("diagonal"), Some("non_diagonal")] {
                let (a, b) = if let Some(g) = group {
                    (&actual[split][g], &expected[split][g])
                } else {
                    (&actual[split], &expected[split])
                };
                require(
                    a["exact"] == b["exact"]
                        && (a["loss"].as_f64().unwrap() - b["loss"].as_f64().unwrap()).abs()
                            < 1e-10,
                    "plain retained-state replay differs",
                )?;
            }
        }
        require(
            e.parameters.read()? == p
                && e.first_moment.read()? == m
                && e.second_moment.read()? == v
                && hash(&fs::read(&path)?) == cp["state_sha256"],
            "replay mutated state",
        )?;
        replays.push(json!({"step":step,"passed":true,"validation":actual["validation"]}));
    }
    require(
        next_update::files(radial)? == before,
        "audit changed radial control",
    )?;
    let report = json!({"complete":true,"passed":true,"candidate":candidate,"radial_control":control,"paired_observations":paired,
        "radial_at_candidate_endpoint":radial_rows[paired-1],"all_retained_replays":replays,"subgroup_accounting":true,
        "input_artifacts_unchanged":true,"stress_source_matches_training":true,"python_invoked":false,"benchmark_modified":false});
    record(&output.join("audit.json"), &report)?;
    println!(
        "{}",
        json!({"audit":"passed","status":report["candidate"]["status"],"epochs":report["candidate"]["epochs"],"heldout":report["candidate"]["heldout_accuracy"]})
    );
    Ok(())
}

#[test]
fn plain_complex_product_has_cross_terms_and_zero_gate_derivatives() {
    let x = [2., 3., -4., 5.].map(Jet::from);
    let y = oracle::plain_product(&x);
    assert!((y[0].v + 23. / 2f64.sqrt()).abs() < 1e-14);
    assert!((y[1].v + 2. / 2f64.sqrt()).abs() < 1e-14);
    let mut z = [0., 0., -4., 5.].map(Jet::from);
    z[0].d = 1.;
    let dy = oracle::plain_product(&z);
    assert_eq!(dy[0].v, 0.);
    assert!((dy[0].d + 4. / 2f64.sqrt()).abs() < 1e-14);
    assert!((dy[1].d - 5. / 2f64.sqrt()).abs() < 1e-14);
}
