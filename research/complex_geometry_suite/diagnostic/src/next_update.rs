//! Disposable, matched next-minibatch AdamW counterfactuals at epoch4000.
use crate::continuation::record;
use crate::*;

const STEP: usize = 76000;
const STATE_SHA: &str = "ec4acff217e8b267a4947aa45a2bd409455ec9bb2c67b5264de8bdb4e7cda3d5";
const PLAN: &str = include_str!("../NEXT_UPDATE_PLAN.txt");

#[derive(Clone, Copy)]
enum Reset {
    Keep,
    First,
    Both,
}
impl Reset {
    fn name(self) -> &'static str {
        match self {
            Self::Keep => "keep",
            Self::First => "zero_first_keep_second_and_clock",
            Self::Both => "zero_both_fresh_adam_clock_current_lr",
        }
    }
}
struct State {
    p: Vec<f32>,
    m: Vec<f32>,
    v: Vec<f32>,
}
impl State {
    fn read(e: &Engine) -> Result<Self> {
        Ok(Self {
            p: e.parameters.read()?,
            m: e.first_moment.read()?,
            v: e.second_moment.read()?,
        })
    }
    fn bytes(&self) -> Vec<u8> {
        self.p
            .iter()
            .chain(&self.m)
            .chain(&self.v)
            .flat_map(|v| v.to_le_bytes())
            .collect()
    }
}
fn fingerprint(x: &[f32]) -> String {
    hash(&x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
}
pub(crate) fn files(root: &Path) -> Result<Value> {
    fn visit(root: &Path, dir: &Path, out: &mut serde_json::Map<String, Value>) -> Result<()> {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            let kind = entry.file_type()?;
            if kind.is_dir() {
                visit(root, &path, out)?;
            } else if kind.is_file() {
                out.insert(
                    path.strip_prefix(root)?.to_string_lossy().into_owned(),
                    hash(&fs::read(&path)?).into(),
                );
            }
        }
        Ok(())
    }
    let mut out = serde_json::Map::new();
    visit(root, root, &mut out)?;
    Ok(Value::Object(out))
}
fn batch_metrics(e: &mut Engine, batch: &metrics::Packed) -> Result<Value> {
    let n = batch.tokens.len() / 2;
    e.forward(&batch.tokens, &batch.labels, n, 2)?;
    let loss = e.losses(2 * n)?.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let predictions = e.predictions.read()?;
    let exact = (0..n)
        .filter(|&i| predictions[2 * i + 1] == batch.labels[2 * i + 1])
        .count();
    Ok(json!({"examples":n,"exact":exact,"accuracy":exact as f64/n as f64,"loss":loss}))
}
fn scores(e: &mut Engine, task: &data::Task, batch: &metrics::Packed) -> Result<Value> {
    Ok(
        json!({"batch":batch_metrics(e,batch)?,"train":metrics::evaluate(e,&task.splits["train"],false)?,
        "validation":metrics::evaluate(e,&task.splits["validation"],true)?}),
    )
}
fn full_gradient(e: &mut Engine, examples: &[data::Example]) -> Result<Vec<f64>> {
    let mut result = vec![0.; e.parameters.len()];
    let mut packed = metrics::Packed::default();
    for start in (0..examples.len()).step_by(e.config.max_batch) {
        let end = (start + e.config.max_batch).min(examples.len());
        let ids: Vec<_> = (start as u32..end as u32).collect();
        packed.fill(examples, &ids);
        e.forward(&packed.tokens, &packed.labels, ids.len(), 2)?;
        e.backward(ids.len(), 2)?;
        for (to, g) in result.iter_mut().zip(e.gradients.read()?) {
            *to += g as f64 * ids.len() as f64 / examples.len() as f64;
        }
    }
    require(
        result.iter().all(|v| v.is_finite()),
        "nonfinite full-training gradient",
    )?;
    Ok(projection_gradient(&e.config, &result))
}
fn delta(p: &[f32], q: &[f32]) -> Vec<f64> {
    p.iter()
        .zip(q)
        .map(|(&a, &b)| b as f64 - a as f64)
        .collect()
}
fn magnitude(cfg: &Config, d: &[f64]) -> f64 {
    layout::pairs(cfg)
        .iter()
        .flat_map(|p| [p[0] as usize, p[1] as usize])
        .map(|i| d[i] * d[i])
        .sum::<f64>()
        .sqrt()
}

#[expect(
    clippy::too_many_arguments,
    reason = "scalar AdamW oracle explicitly exposes moments and bias-correction inputs"
)]
fn adam_scalar(
    p: f64,
    m: f64,
    v: f64,
    g: f64,
    lr: f64,
    decay: f64,
    bc1: f64,
    bc2root: f64,
) -> (f64, f64, f64) {
    let m = 0.9 * m + 0.1 * g;
    let v = 0.98 * v + 0.02 * g * g;
    (
        p * (1. - lr * decay) - lr * (m / bc1) / (v.sqrt() / bc2root + 1e-8),
        m,
        v,
    )
}
fn scalar_check(
    cfg: &Config,
    completed: usize,
    before: &State,
    g: &[f32],
    after: &State,
) -> Result<Value> {
    let t = (completed + 1) as f64;
    let bc1 = (1. - 0.9f64.powf(t)) as f32 as f64;
    let bc2root = (1. - 0.98f64.powf(t)).sqrt() as f32 as f64;
    let lr = (cfg.lr
        * if cfg.warmup == 0 {
            1.
        } else {
            (completed as f64 / cfg.warmup as f64).min(1.)
        }) as f32 as f64;
    let mut worst = [0f64; 3];
    let mut normalized = [0f64; 3];
    let mut checked = 0;
    let mut errors = Vec::new();
    for [ar, bi, br, ai, kind] in layout::pairs(cfg) {
        let ga = g[ar as usize] as f64 + if kind == 0 { g[ai as usize] as f64 } else { 0. };
        let gb = g[bi as usize] as f64 - if kind == 0 { g[br as usize] as f64 } else { 0. };
        for (i, gradient) in [(ar as usize, ga), (bi as usize, gb)] {
            let (p, m, v) = (before.p[i] as f64, before.m[i] as f64, before.v[i] as f64);
            let (ep, em, ev) = adam_scalar(p, m, v, gradient, lr, cfg.decay as f64, bc1, bc2root);
            let expected = [ep, em, ev];
            let actual = [after.p[i] as f64, after.m[i] as f64, after.v[i] as f64];
            let bound = [
                1e-6 * (p.abs() + (ep - p).abs()) + 1e-30,
                2e-6 * (m.abs() + gradient.abs()) + 1e-30,
                2e-6 * (v + gradient * gradient) + 1e-38,
            ];
            for j in 0..3 {
                let error = (actual[j] - expected[j]).abs();
                worst[j] = worst[j].max(error);
                normalized[j] = normalized[j].max(error / bound[j]);
                if error > bound[j] && errors.len() < 12 {
                    errors.push(json!({"index":i,"array":j,"expected":expected[j],"actual":actual[j],"bound":bound[j]}));
                }
            }
            checked += 1;
        }
    }
    layout::validate_state(cfg, &after.p, &after.m, &after.v)?;
    Ok(
        json!({"passed":normalized.iter().all(|&v|v<=1.),"independent_coordinates":checked,"max_abs_error_parameter_m_v":worst,
        "max_fraction_of_bound_parameter_m_v":normalized,"first_failures":errors,"effective_lr_f32":lr,"bias_correction_completed":completed}),
    )
}
fn geometry(cfg: &Config, p: &[f32], d: &[f64], batch: &[f64], full: &[f64]) -> Value {
    let mut rows = Vec::new();
    let mut train_dot = 0.;
    let mut batch_dot = 0.;
    let mut all_d = Vec::new();
    let mut all_g = Vec::new();
    let mut all_b = Vec::new();
    for block in blocks(cfg) {
        let ids = owner_indices(cfg, &block);
        let ds: Vec<_> = ids.iter().map(|&i| d[i]).collect();
        let gs: Vec<_> = ids.iter().map(|&i| full[i]).collect();
        let bs: Vec<_> = ids.iter().map(|&i| batch[i]).collect();
        let ws: Vec<_> = ids.iter().map(|&i| p[i] as f64).collect();
        let descent: Vec<_> = ds.iter().map(|x| -x).collect();
        let td = stats::dot(&ds, &gs);
        let bd = stats::dot(&ds, &bs);
        train_dot += td;
        batch_dot += bd;
        rows.push(json!({"block":block.name,"displacement_l2":stats::norm(&ds),"relative_displacement_l2":stats::norm(&ds)/stats::norm(&ws),
            "train_gradient_dot_displacement":td,"batch_gradient_dot_displacement":bd,"descent_alignment_train":stats::cosine(&descent,&gs),
            "descent_alignment_batch":stats::cosine(&descent,&bs),"batch_train_gradient_cosine":stats::cosine(&bs,&gs)}));
        all_d.extend(descent);
        all_g.extend(gs);
        all_b.extend(bs);
    }
    json!({"displacement_l2":stats::norm(&all_d),"train_gradient_dot_displacement":train_dot,"batch_gradient_dot_displacement":batch_dot,
        "descent_alignment_train":stats::cosine(&all_d,&all_g),"descent_alignment_batch":stats::cosine(&all_d,&all_b),
        "batch_train_gradient_cosine":stats::cosine(&all_b,&all_g),"blocks":rows})
}
struct Experiment<'a> {
    cfg: &'a Config,
    state: &'a State,
    task: &'a data::Task,
    batch: &'a metrics::Packed,
    gradient: &'a [f32],
    batch_independent: &'a [f64],
    full: &'a [f64],
    before: &'a Value,
    output: &'a Path,
}
impl Experiment<'_> {
    fn branch(
        &self,
        name: &str,
        reset: Reset,
        lr_scale: f64,
        match_norm: Option<f64>,
    ) -> Result<(Value, State)> {
        let mut cfg = self.cfg.clone();
        cfg.lr *= lr_scale;
        let completed = if matches!(reset, Reset::Both) {
            cfg.warmup = 0;
            0
        } else {
            STEP
        };
        let m = if matches!(reset, Reset::Keep) {
            self.state.m.clone()
        } else {
            vec![0.; self.state.m.len()]
        };
        let v = if matches!(reset, Reset::Both) {
            vec![0.; self.state.v.len()]
        } else {
            self.state.v.clone()
        };
        let input = State {
            p: self.state.p.clone(),
            m,
            v,
        };
        let mut e = retrain::corrected(cfg.clone(), &input.p, &input.m, &input.v)?;
        require(
            e.parameters.read()? == self.state.p,
            "pre-update weights changed",
        )?;
        e.forward(
            &self.batch.tokens,
            &self.batch.labels,
            self.batch.tokens.len() / 2,
            2,
        )?;
        e.backward(self.batch.tokens.len() / 2, 2)?;
        require(
            e.gradients.read()? == self.gradient,
            "counterfactual batch gradients differ",
        )?;
        e.update(completed)?;
        let after = State::read(&e)?;
        let reference = scalar_check(&cfg, completed, &input, self.gradient, &after)?;
        let displacement = delta(&self.state.p, &after.p);
        let norm = magnitude(&cfg, &displacement);
        let matching_error = match_norm.map(|target| (norm / target - 1.).abs());
        let after_scores = scores(&mut e, self.task, self.batch)?;
        require(
            e.parameters.read()? == after.p
                && e.first_moment.read()? == after.m
                && e.second_moment.read()? == after.v,
            "scoring changed updated state",
        )?;
        let mut changes = serde_json::Map::new();
        for split in ["batch", "train", "validation"] {
            let before = self.before[split]["loss"].as_f64().unwrap();
            let now = after_scores[split]["loss"].as_f64().unwrap();
            changes.insert(split.into(),json!({"loss_before":before,"loss_after":now,"loss_change":now-before,"relative_loss_change":now/before-1.,
                "exact_before":self.before[split]["exact"],"exact_after":after_scores[split]["exact"]}));
        }
        let path = self.output.join(format!("{name}.bin"));
        let bytes = after.bytes();
        continuation::durable(&path, &bytes)?;
        let row = json!({"branch":name,"moment_policy":reset.name(),"lr_scale":lr_scale,"effective_lr":reference["effective_lr_f32"],
            "optimizer_completed_before_update":completed,"global_completed_before_update":STEP,"restart_lr_warmup":false,
            "native_optimizer_updates":1,"identical_input_parameters_and_batch_gradient":true,"batch_gradient_sha256":fingerprint(self.gradient),
            "state_file":format!("{name}.bin"),"state_sha256":hash(&bytes),"state_layout":"parameters, first moment, second moment; f32 little endian",
            "scalar_optimizer_reference":reference,"matched_displacement_relative_error":matching_error,
            "geometry":geometry(&cfg,&self.state.p,&displacement,self.batch_independent,self.full),"scores":after_scores,"changes":changes,
            "passed":reference["passed"]==true&&matching_error.is_none_or(|v|v<=1e-3)});
        record(&self.output.join(format!("{name}.json")), &row)?;
        println!(
            "{}",
            json!({"branch":name,"train_loss_change":row["changes"]["train"]["loss_change"],"batch_loss_change":row["changes"]["batch"]["loss_change"],"checks":row["passed"]})
        );
        Ok((row, after))
    }
}
pub fn run(parent_root: &Path, root: &Path, output: &Path) -> Result<()> {
    require(
        !output.exists() && !output.starts_with(root),
        "choose a fresh output outside the input run",
    )?;
    let protected_run = files(root)?;
    let input_audit = retrain::audit_readonly(parent_root, root)?;
    let (parent, task, samples) = retrain::setup(parent_root)?;
    let original_files = retrain::protected(&parent)?;
    let protocol = read(&root.join("protocol.json"))?;
    let meta = read(&root.join(format!("checkpoint-{STEP:06}.json")))?;
    let input_path = root.join(format!("checkpoint-{STEP:06}.bin"));
    require(
        meta["state_sha256"] == STATE_SHA
            && hash(&fs::read(&input_path)?) == STATE_SHA
            && meta["epochs"] == 4000.,
        "wrong epoch4000 checkpoint",
    )?;
    let (p, m, v) = retrain::decode(&input_path, parent.p.len())?;
    let state = State { p, m, v };
    layout::validate_state(&parent.cfg, &state.p, &state.m, &state.v)?;
    let indices = &samples.order[samples.edges[STEP]..samples.edges[STEP + 1]];
    require(
        indices.len() == 256 && samples.edges[STEP] == 4000 * 4704,
        "unexpected sampler boundary",
    )?;
    let mut packed = metrics::Packed::default();
    packed.fill(&task.splits["train"], indices);
    fs::create_dir(output)?;
    let sources = retrain::archive(output)?;
    continuation::durable(&output.join("PLAN.txt"), PLAN.as_bytes())?;
    record(
        &output.join("protocol.json"),
        &json!({"purpose":"actual next-update counterfactual at epoch4000","plan_sha256":hash(PLAN.as_bytes()),
        "source":sources,"input_audit":input_audit,"input_protocol":protocol,"input_snapshot":meta,"input_files":protected_run,"original_files":original_files,
        "next_batch_indices":indices,"next_batch_indices_sha256":hash(&indices.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>()),
        "batch_examples":256,"branches":["keep_lr1","keep_lr_half","keep_lr_quarter","keep_lr_tenth","reset_first_lr1","reset_both_lr1","reset_first_norm_matched","reset_both_norm_matched"],
        "diagnostic_optimizer_updates":8,"continuation_training_updates":0,"benchmark_modified":false,"python_invoked":false}),
    )?;
    let mut base = retrain::corrected(parent.cfg.clone(), &state.p, &state.m, &state.v)?;
    let before = scores(&mut base, &task, &packed)?;
    let full = full_gradient(&mut base, &task.splits["train"])?;
    base.forward(&packed.tokens, &packed.labels, 256, 2)?;
    base.backward(256, 2)?;
    let gradient = base.gradients.read()?;
    let independent = projection_gradient(
        &parent.cfg,
        &gradient.iter().map(|&v| v as f64).collect::<Vec<_>>(),
    );
    require(
        gradient.iter().all(|v| v.is_finite()),
        "nonfinite batch gradient",
    )?;
    let control = scores(&mut base, &task, &packed)?;
    require(
        before == control
            && base.parameters.read()? == state.p
            && base.first_moment.read()? == state.m
            && base.second_moment.read()? == state.v,
        "unchanged-state control failed",
    )?;
    let history = fs::read_to_string(root.join("history.jsonl"))?;
    let point = history
        .lines()
        .map(serde_json::from_str::<Value>)
        .collect::<std::result::Result<Vec<_>, _>>()?
        .into_iter()
        .find(|r| r["step"] == STEP)
        .ok_or("missing saved observation")?;
    for split in ["train", "validation"] {
        require(
            before[split]["exact"] == point[split]["exact"]
                && (before[split]["loss"].as_f64().unwrap()
                    - point[split]["loss"].as_f64().unwrap())
                .abs()
                    < 1e-10,
            "input checkpoint score replay failed",
        )?;
    }
    continuation::durable(
        &output.join("minibatch-gradient.f32"),
        &gradient
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    continuation::durable(
        &output.join("full-train-independent-gradient.f64"),
        &full
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    record(
        &output.join("before.json"),
        &json!({"scores":before,"unchanged_control_passed":true,"batch_gradient_sha256":fingerprint(&gradient),
        "full_gradient_layout":"expanded-length array of independent-coordinate gradients, non-owner entries zero; little-endian f64"}),
    )?;
    let experiment = Experiment {
        cfg: &parent.cfg,
        state: &state,
        task: &task,
        batch: &packed,
        gradient: &gradient,
        batch_independent: &independent,
        full: &full,
        before: &before,
        output,
    };
    let mut rows = Vec::new();
    let mut reference_moments = None;
    let mut normal_norm = 0.;
    let mut reset_norms = [0.; 2];
    for (name, reset, scale) in [
        ("keep_lr1", Reset::Keep, 1.),
        ("keep_lr_half", Reset::Keep, 0.5),
        ("keep_lr_quarter", Reset::Keep, 0.25),
        ("keep_lr_tenth", Reset::Keep, 0.1),
        ("reset_first_lr1", Reset::First, 1.),
        ("reset_both_lr1", Reset::Both, 1.),
    ] {
        let (row, after) = experiment.branch(name, reset, scale, None)?;
        let norm = row["geometry"]["displacement_l2"].as_f64().unwrap();
        if name == "keep_lr1" {
            normal_norm = norm;
            reference_moments = Some((after.m.clone(), after.v.clone()));
        }
        let (baseline_m, baseline_v) = reference_moments.as_ref().unwrap();
        match reset {
            Reset::Keep => require(
                after.m == *baseline_m && after.v == *baseline_v,
                "LR-only arms changed moments",
            )?,
            Reset::First => {
                require(
                    after.v == *baseline_v,
                    "first-moment reset changed second moment",
                )?;
                reset_norms[0] = norm;
            }
            Reset::Both => {
                reset_norms[1] = norm;
            }
        }
        rows.push(row);
    }
    require(
        normal_norm > 0. && reset_norms.iter().all(|&n| n > 0.),
        "cannot match a zero update",
    )?;
    for (name, reset, norm) in [
        ("reset_first_norm_matched", Reset::First, reset_norms[0]),
        ("reset_both_norm_matched", Reset::Both, reset_norms[1]),
    ] {
        let (row, _) = experiment.branch(name, reset, normal_norm / norm, Some(normal_norm))?;
        rows.push(row);
    }
    require(
        files(root)? == protected_run && retrain::protected(&parent)? == original_files,
        "diagnostic changed input artifacts",
    )?;
    let passed = rows.iter().all(|r| r["passed"] == true);
    record(
        &output.join("summary.json"),
        &json!({"complete":true,"passed":passed,"case":CASE,"seed":1,"epoch":4000,"global_completed":STEP,
        "before":before,"branches":rows,"diagnostic_optimizer_updates":8,"continuation_training_updates":0,"original_artifacts_unchanged":true,
        "lr_only_moments_identical":true,"reset_first_preserves_second_moment":true,"limits":"one historical state and one next batch; neither sustained training nor universal convergence is established"}),
    )?;
    require(
        passed,
        "one or more next-update checks failed; all branch artifacts retained",
    )
}

#[test]
fn reset_clock_and_first_moment_have_distinct_effects() {
    let old = adam_scalar(0.5, 0.3, 0.09, -0.2, 0.002, 0., 1., 1.);
    assert!(old.0 < 0.5); // stale positive momentum moves against the negative gradient
    assert!((old.1 - 0.25).abs() < 1e-15 && (old.2 - 0.089).abs() < 1e-15);
    let first = adam_scalar(0.5, 0., 0.09, -0.2, 0.002, 0., 1., 1.);
    assert!(first.0 > 0.5 && first.2 == old.2);
    let fresh = adam_scalar(0.5, 0., 0., -0.2, 0.002, 0., 0.1, 0.02f64.sqrt());
    assert!((fresh.0 - 0.5019999999).abs() < 1e-14);
    let old_clock = adam_scalar(0.5, 0., 0., -0.2, 0.002, 0., 1., 1.);
    assert!(old_clock.0 < fresh.0 && old_clock.0 > 0.5);
}
