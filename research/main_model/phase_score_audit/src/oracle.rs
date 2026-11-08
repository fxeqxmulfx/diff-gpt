// CPU-only geometry oracle. Do not depend on the GPU engine or its layout code.
type ComplexPair = (usize, usize, Option<(usize, usize)>);
struct Geometry {
    base_parameters: usize,
    router_shape: Option<(usize, usize)>,
    parameters: usize,
    independent: usize,
    hidden: usize,
    frequency_parameters: usize,
    phase_parameters: usize,
    phase_offset: usize,
    score_parameters: usize,
    score_offset: usize,
    // real, imaginary, optional (-imaginary, real) matrix duplicates
    pairs: Vec<ComplexPair>,
}
impl Geometry {
    fn new(width: usize, heads: usize, layers: usize, vocab: usize, score: bool) -> Self {
        let hidden = ((8 * width + 1) / 3).div_ceil(2) * 2;
        let per_layer = 4 * width * width + 3 * width * hidden;
        let parameters = vocab * width + layers * per_layer;
        let mut pairs = Vec::new();
        let d = width / heads;
        for token in 0..vocab {
            for head in 0..heads {
                for i in 0..d / 2 {
                    let re = token * width + head * d + i;
                    pairs.push((re, re + d / 2, None));
                }
            }
        }
        for layer in 0..layers {
            let offset = vocab * width + layer * per_layer;
            for (base, rows, cols, row_groups, col_groups) in [
                (offset, 3 * width, width, 3 * heads, heads),
                (offset + 3 * width * width, width, width, heads, heads),
                (offset + 4 * width * width, 2 * hidden, width, 2, heads),
                (
                    offset + 4 * width * width + 2 * hidden * width,
                    width,
                    hidden,
                    heads,
                    1,
                ),
            ] {
                let rh = rows / row_groups / 2;
                let ch = cols / col_groups / 2;
                for rg in 0..row_groups {
                    for r in 0..rh {
                        for cg in 0..col_groups {
                            for c in 0..ch {
                                let re = base + (rg * 2 * rh + r) * cols + cg * 2 * ch + c;
                                let im = re + rh * cols;
                                pairs.push((re, im, Some((re + ch, im + ch))));
                            }
                        }
                    }
                }
            }
        }
        let extra = 2 * layers * (96 * width + 27);
        let frequencies = width / heads / 2;
        let phases = 6 * layers;
        let scores = if score { layers * heads } else { 0 };
        Self {
            base_parameters: parameters,
            router_shape: Some((width, layers)),
            parameters: parameters + extra + phases + scores + frequencies,
            independent: 2 * pairs.len() + extra + phases + scores + frequencies,
            frequency_parameters: frequencies,
            phase_parameters: phases,
            phase_offset: parameters + extra,
            score_parameters: scores,
            score_offset: parameters + extra + phases,
            hidden,
            pairs,
        }
    }
    fn initial(&self, seed: u32, complex: bool) -> Vec<f32> {
        let mut rng = random::Random::new(seed);
        let mut result: Vec<f32> = (0..self.base_parameters)
            .map(|_| rng.gauss(0., 0.02) as f32)
            .collect();
        if let Some((width, layers)) = self.router_shape {
            let mut rng = random::Random::new(seed + 20000);
            for _ in 0..2 * layers {
                result.extend((0..24 * 4 * width).map(|_| rng.gauss(0., 0.02) as f32));
                result.extend([0.; 8]);
                for row in 0..4 {
                    for col in 0..4 {
                        result.push(if row == col { 2. } else { 0. });
                    }
                }
                result.extend([0.01; 3]);
            }
        }
        result.resize(self.parameters, 0.);
        if complex {
            for &(re, im, duplicate) in &self.pairs {
                if let Some((neg_im, copy_re)) = duplicate {
                    result[neg_im] = -result[im];
                    result[copy_re] = result[re];
                }
            }
        }
        result
    }
    fn check_state(&self, state: &[f32], arm: &str) -> Result<()> {
        require(state.len() == 3 * self.parameters, "state size mismatch")?;
        require(
            state.iter().all(|x| x.is_finite()),
            "nonfinite optimizer state",
        )?;
        let (p, rest) = state.split_at(self.parameters);
        let (m, v) = rest.split_at(self.parameters);
        require(v.iter().all(|x| *x >= 0.), "negative AdamW variance")?;
        if arm != "real-product" {
            for &(re, im, duplicate) in &self.pairs {
                if let Some((neg_im, copy_re)) = duplicate {
                    for values in [p, m] {
                        require(
                            values[re] == values[copy_re] && values[im] == -values[neg_im],
                            "broken complex-linear weights or first moments",
                        )?;
                    }
                    require(
                        v[re] == v[copy_re] && v[im] == v[neg_im],
                        "broken complex matrix variance ties",
                    )?;
                }
                if arm == "complex-hermitian-phase" {
                    require(v[re] == v[im], "broken shared complex-pair variance")?;
                }
            }
        }
        Ok(())
    }
}
fn sampler_hash(n: usize, batch: usize, updates: usize, seed: u32) -> String {
    let mut rng = random::Random::new(seed + 10000);
    let mut sha = Sha256::new();
    let mut done = 0;
    while done < updates {
        let mut order: Vec<u32> = (0..n as u32).collect();
        rng.shuffle(&mut order);
        for chunk in order.chunks(batch) {
            if done == updates {
                break;
            }
            for value in chunk {
                sha.update(value.to_le_bytes());
            }
            done += 1;
        }
    }
    sha.finalize().iter().map(|x| format!("{x:02x}")).collect()
}
fn examples(step: usize, n: usize, batch: usize) -> usize {
    let per = n.div_ceil(batch);
    (step / per) * n + ((step % per) * batch).min(n)
}
// Independent CPU reconstruction; no training optimizer or CUDA dependency.
fn expected_rate(budget: usize, step: usize) -> f32 {
    let t = step.min(budget - 1);
    let warmup = budget / 50;
    let start = budget * 4 / 5;
    if t < warmup {
        return (0.001 * t as f64 / warmup as f64) as f32;
    }
    let decay = if t < start {
        0.
    } else {
        let fraction = (t - start) as f64 / (budget - 1 - start) as f64;
        (1. - (std::f64::consts::PI * fraction).cos()) / 2.
    };
    (0.001 + (1e-8 - 0.001) * decay) as f32
}
fn check_phase_metadata(protocol: &Value, geometry: &Geometry) -> Result<()> {
    require(
        protocol["phase_routing"] == true
            && protocol["phase_routing_parameters"] == geometry.phase_parameters
            && protocol["phase_routing_offset"] == geometry.phase_offset
            && protocol["phase_routing_initial_values"]
                == json!(vec![0f32; geometry.phase_parameters])
            && protocol["phase_routing_definition"]
                == json!({"skip":"D P D*",
            "phase":"(1+i*a)/sqrt(1+a*a)","stream0_fixed":true,
            "per_sublayer_real_parameters":3,"read_write_gates_unchanged":true,
            "optimizer_group":"base clip1 wd0.1 ordinary real AdamW"}),
        "declared phase routing,initialization or optimizer treatment changed",
    )
}

fn check_score_metadata(protocol: &Value, geometry: &Geometry) -> Result<()> {
    require(
        protocol["phase_score"] == true
            && protocol["phase_score_parameters"] == geometry.score_parameters
            && protocol["phase_score_offset"] == geometry.score_offset
            && protocol["phase_score_initial_values"]
                == json!(vec![0f32; geometry.score_parameters])
            && protocol["phase_score_definition"]
                == json!({
            "score":"ln(N)/d*(Re(Z)+beta*Im(Z)^2/d)","product":"sum(conj(q)*k)",
            "beta":"a/sqrt(1+a*a)","real_parameters":"layers*heads","initialization":0,
            "optimizer_group":"base clip1 wd0.1 ordinary real AdamW"}),
        "declared score,initialization or optimizer treatment changed",
    )
}

fn audit_cell(
    root: &Path,
    row: &Value,
    seed: u32,
    arm: &str,
    source: &Value,
    data: &Value,
    sampler: &str,
) -> Result<Value> {
    require(
        arm == CANDIDATE_ARM,
        "only the registered phase-score-lerope arm is auditable",
    )?;
    let name = row["name"].as_str().unwrap();
    let dir = root.join(arm).join(name).join(format!("seed{seed}"));
    let result = read(&dir.join("result.json"))?;
    let protocol = read(&dir.join("protocol.json"))?;
    let checkpoint = read(&dir.join("checkpoint.json"))?;
    let execution = read(&dir.join("execution.json"))?;
    require(
        protocol["case"] == *row && protocol["seed"] == seed && protocol["arm"] == arm,
        "declared case/seed/arm changed",
    )?;
    require(
        result["metadata"] == protocol,
        "result metadata disagrees with protocol",
    )?;
    require(
        protocol["source_sha256"] == source["source_sha256"]
            && protocol["ptx_sha256"] == source["ptx_sha256"],
        "source/PTX provenance mismatch",
    )?;
    require(
        checkpoint["source_sha256"] == source["source_sha256"]
            && checkpoint["ptx_sha256"] == source["ptx_sha256"],
        "checkpoint provenance mismatch",
    )?;
    require(
        protocol["sampler_sha256"] == sampler,
        "native sampler fingerprint changed",
    )?;
    let initial = fs::read(dir.join("initial.f32"))?;
    require(
        protocol["initial_parameters_sha256"] == hash(&initial),
        "initial parameter hash mismatch",
    )?;
    let c = &row["case"];
    let geometry = Geometry::new(
        number(c, "width")?,
        number(c, "heads")?,
        number(c, "layers")?,
        number(data, "vocab")?,
        true,
    );
    let params = number(&protocol, "expanded_parameters")?;
    require(
        params == geometry.parameters && number(&protocol, "hidden")? == geometry.hidden,
        "expanded model dimensions changed",
    )?;
    let independent = if arm == "real-product" {
        params
    } else {
        geometry.independent
    };
    require(
        number(&protocol, "independent_parameters")? == independent,
        "independent parameter count changed",
    )?;
    require(
        protocol["learning_rate"].as_f64() == Some(0.001)
            && protocol["warmup"] == number(c, "max_updates")? / 50
            && protocol["betas"] == json!([0.9, 0.95])
            && protocol["epsilon"] == json!(1e-8)
            && protocol["weight_decay"].as_f64() == Some(0.1f32 as f64)
            && protocol["schedule"] == "wsd-cosine20"
            && protocol["warmup_fraction"] == json!(0.02)
            && protocol["decay_fraction"] == json!(0.2)
            && protocol["cooldown_start"] == number(c, "max_updates")? * 4 / 5
            && protocol["schedule_floor"] == json!(1e-8)
            && protocol["global_independent_gradient_clip"] == 1
            && protocol["router_coordinates_in_clip_and_adamw"] == true
            && protocol["json_float_roundtrip"] == true
            && protocol["checkpoint_epochs"] == 500
            && protocol["gain"].as_f64()
                == Some(number(c, "heads")? as f64 / number(c, "width")? as f64),
        "optimizer, gain or save cadence changed",
    )?;
    require(
        protocol["hermitian_xsa"] == false
            && protocol["ordinary_attention"] == true
            && protocol["plain_complex_product"] == true
            && protocol["mhc"] == true
            && protocol["streams"] == 4
            && protocol["sinkhorn_cycles"] == 20
            && protocol["primary"] == data["primary"]
            && protocol["optimizer"]
                == if arm.ends_with("-phase") {
                    "AdamW shared complex-pair variance"
                } else {
                    "AdamW separate real/imaginary variances"
                },
        "declared architecture, optimizer or primary metric changed",
    )?;
    require(initial.len() == 4 * params, "initial state layout mismatch")?;
    let frequency_offset = geometry.parameters - geometry.frequency_parameters;
    require(
        protocol["position_encoding"] == "LeRoPE"
            && protocol["frequency_parameters"] == geometry.frequency_parameters
            && protocol["frequency_offset"] == frequency_offset
            && protocol["frequency_initial_values"]
                == json!(vec![0f32; geometry.frequency_parameters])
            && protocol["frequency_optimizer"]
                == json!({"separate_group":true,"weight_decay":0,"l2_clip":1,"same_schedule_and_betas":true})
            && protocol["base_gradient_clip_excludes_frequencies"] == true
            && protocol.get("attention_output_gate").is_none(),
        "declared LeRoPE frequency or optimizer group changed",
    )?;
    check_phase_metadata(&protocol, &geometry)?;
    check_score_metadata(&protocol, &geometry)?;
    let common_hash = hash(&initial[..4 * geometry.phase_offset]);
    require(
        protocol["common_initial_sha256"] == common_hash,
        "common initialization hash mismatch",
    )?;
    let shared_hash = hash(
        &[
            &initial[..4 * geometry.score_offset],
            &initial[4 * frequency_offset..],
        ]
        .concat(),
    );
    require(
        protocol["shared_full_initial_sha256"] == shared_hash,
        "shared complete initialization hash mismatch",
    )?;
    let expected: Vec<u8> = geometry
        .initial(seed, arm != "real-product")
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    require(
        initial == expected,
        "initial weights differ from declared native RNG",
    )?;
    require(
        protocol["base_initial_sha256"] == hash(&initial[..4 * geometry.base_parameters]),
        "base initialization hash mismatch",
    )?;
    let state = fs::read(dir.join("checkpoint.bin"))?;
    let history = fs::read(dir.join("history.jsonl"))?;
    require(state.len() == 12 * params, "checkpoint size mismatch")?;
    require(
        checkpoint["state_sha256"] == hash(&state)
            && checkpoint["history_sha256"] == hash(&history),
        "checkpoint/history integrity failure",
    )?;
    geometry.check_state(
        &state
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| f32::from_le_bytes(*v))
            .collect::<Vec<_>>(),
        arm,
    )?;
    require(
        execution["clean"] == true
            && execution["durable_completed"] == execution["possible_completed"],
        "uncertain execution exposure",
    )?;
    let n = number(row, "train_examples")?;
    let heldout = number(row, "heldout_examples")?;
    let batch = number(c, "batch")?;
    let observe = number(c, "observe")?;
    let budget = number(c, "max_updates")?;
    let flush = 500 * n.div_ceil(batch);
    let records: Vec<Value> = history
        .split(|&v| v == b'\n')
        .filter(|v| !v.is_empty())
        .map(serde_json::from_slice)
        .collect::<std::result::Result<_, _>>()?;
    require(!records.is_empty(), "empty history")?;
    let mut previous = 0;
    let mut previous_seconds = 0.;
    let mut first = None;
    for (i, point) in records.iter().enumerate() {
        let step = number(point, "step")?;
        let count = examples(step, n, batch);
        let expected_step = if i == 0 {
            0
        } else {
            (previous + observe)
                .min(budget)
                .min((previous / flush + 1) * flush)
        };
        require(
            step == expected_step,
            "observation cadence/update history changed",
        )?;
        require(
            step <= budget && number(point, "examples")? == count,
            "budget or example count mismatch",
        )?;
        require(
            point["next_lr"].as_f64() == Some(expected_rate(budget, step) as f64),
            "recorded WSD learning rate changed",
        )?;
        require(
            (point["epochs"].as_f64().ok_or("missing epochs")? - count as f64 / n as f64).abs()
                < 1e-9,
            "epoch denominator changed",
        )?;
        let seconds = point["seconds"].as_f64().ok_or("missing time")?;
        require(seconds >= previous_seconds, "elapsed time went backwards")?;
        for (which, size) in [("train", n), ("validation", heldout)] {
            let m = &point[which];
            require(
                number(m, "examples")? == size,
                "incomplete split measurement",
            )?;
            let exact = number(m, "exact")?;
            require(exact <= size, "impossible exact count")?;
            require(
                (m["sequence_accuracy"].as_f64().ok_or("missing accuracy")?
                    - exact as f64 / size as f64)
                    .abs()
                    < 1e-12,
                "accuracy denominator changed",
            )?;
            require(
                m["loss"].as_f64().is_some_and(f64::is_finite),
                "nonfinite loss",
            )?;
        }
        if first.is_none() && number(&point["validation"], "exact")? * 100 >= 99 * heldout {
            first = Some((step, point["epochs"].clone()));
        }
        previous = step;
        previous_seconds = seconds;
    }
    let last = records.last().unwrap();
    let completed = number(&result, "completed")?;
    require(
        completed == previous
            && completed == number(&checkpoint, "completed")?
            && completed == number(&execution, "durable_completed")?,
        "endpoint/update mismatch",
    )?;
    require(
        result["endpoint"] == *last,
        "endpoint differs from last observation",
    )?;
    require(
        result["examples"] == checkpoint["examples"] && result["examples"] == last["examples"],
        "checkpoint sample count mismatch",
    )?;
    match result["status"].as_str() {
        Some("passed") => {
            let (step, epoch) = first.as_ref().ok_or("pass without threshold crossing")?;
            require(
                *step == completed && result["first99_epochs"] == *epoch,
                "wrong first crossing",
            )?;
        }
        Some("budget_exhausted") => {
            require(
                completed == budget && first.is_none() && result["first99_epochs"].is_null(),
                "incorrect censored budget result",
            )?;
        }
        Some("stopped") => require(
            first.is_none() && result["first99_epochs"].is_null(),
            "incorrect stop result",
        )?,
        _ => return Err("unknown endpoint status".into()),
    }
    let first_prior = if first.is_some() && records.len() > 1 {
        records[records.len() - 2]["epochs"].clone()
    } else {
        json!(0.)
    };
    if first.is_some() {
        require(
            result["previous_observed_epochs"] == first_prior,
            "first crossing interval changed",
        )?;
    }
    Ok(
        json!({"case":name,"tier":row["tier"],"seed":seed,"arm":arm,"status":result["status"],"audit":"passed",
        "completed":completed,"examples":result["examples"],"epochs":result["epochs"],
        "first99_epochs":result["first99_epochs"],"previous_observed_epochs":result["previous_observed_epochs"],
        "sequence_accuracy":last["validation"]["sequence_accuracy"],"train_accuracy":last["train"]["sequence_accuracy"],
        "seconds":result["seconds"],"observations":records.len(),"initial_parameters_sha256":protocol["initial_parameters_sha256"],
        "expanded_parameters":params,"independent_parameters":independent,"base_initial_sha256":protocol["base_initial_sha256"],
        "common_initial_sha256":common_hash,"shared_full_initial_sha256":shared_hash,
        "frequency_parameters":geometry.frequency_parameters,"phase_routing_parameters":geometry.phase_parameters,
        "phase_score_parameters":geometry.score_parameters,
        "sampler_sha256":protocol["sampler_sha256"],"checkpoint_sha256":checkpoint["state_sha256"],
        "history_sha256":checkpoint["history_sha256"],"source_sha256":protocol["source_sha256"]}),
    )
}
