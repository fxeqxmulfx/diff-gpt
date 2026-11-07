use crate::*;
use std::io::{BufWriter, Write};
fn sample(metrics: &mut BTreeMap<String, Vec<f64>>, name: impl Into<String>, value: f64) {
    metrics.entry(name.into()).or_default().push(value);
}
fn rms(x: &[f32]) -> f64 {
    (x.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / x.len() as f64).sqrt()
}
fn cosine32(x: &[f32], y: &[f32]) -> Option<f64> {
    stats::cosine(
        &x.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        &y.iter().map(|&v| v as f64).collect::<Vec<_>>(),
    )
}
fn activations(
    metrics: &mut BTreeMap<String, Vec<f64>>,
    t: &BTreeMap<String, Vec<f32>>,
    examples: &[data::Example],
    cfg: &Config,
) {
    let w = cfg.width;
    let h = layout::hidden(w, "");
    let d = w / cfg.heads;
    for (b, e) in examples.iter().enumerate() {
        let row = 2 * b + 1;
        let diagonal = e.tokens[0] == e.tokens[1];
        for (name, cols) in [
            ("embedding", w),
            ("readout_input", w),
            ("layer0.mid", w),
            ("layer0.ffn_hidden", h),
            ("layer0.ffn_projected", w),
            ("layer0.output", w),
            ("layer0.grad_qkv", 3 * w),
            ("layer0.grad_attention", w),
            ("layer0.grad_ffn_gate_value", 2 * h),
            ("layer0.grad_ffn_hidden", h),
        ] {
            sample(
                metrics,
                format!("{name}.row_rms"),
                rms(&t[name][row * cols..(row + 1) * cols]),
            );
        }
        let embed = &t["embedding_normalized"][row * w..(row + 1) * w];
        let mid = &t["layer0.mid"][row * w..(row + 1) * w];
        let out = &t["layer0.output"][row * w..(row + 1) * w];
        if let Some(c) = cosine32(embed, mid) {
            sample(metrics, "embedding_mid_cosine", c);
        }
        if let Some(c) = cosine32(mid, out) {
            sample(metrics, "mid_output_cosine", c);
        }
        let fr = rms(&t["layer0.ffn_projected"][row * w..(row + 1) * w]);
        sample(
            metrics,
            "ffn_rms_inverse_scale",
            (fr * fr + f32::EPSILON as f64).sqrt().recip(),
        );
        let fc = &t["layer0.ffn_gate_value"][row * 2 * h..(row + 1) * 2 * h];
        for c in 0..h / 2 {
            let radius =
                ((fc[c] as f64).powi(2) + (fc[h / 2 + c] as f64).powi(2) + f32::EPSILON as f64)
                    .sqrt();
            sample(metrics, "gate_radius", radius);
            sample(metrics, "gate_sigmoid", 1. / (1. + (-radius).exp()));
        }
        if !diagonal {
            let ar = rms(&t["layer0.attention_projected"][row * w..(row + 1) * w]);
            sample(metrics, "attention_projected_rms_nonconstant", ar);
            sample(
                metrics,
                "attention_rms_inverse_scale_nonconstant",
                (ar * ar + f32::EPSILON as f64).sqrt().recip(),
            );
        }
        for head in 0..cfg.heads {
            let offset = row * 3 * w + head * d;
            for (part, name) in ["q", "k", "v"].iter().enumerate() {
                sample(
                    metrics,
                    format!("head{head}.{name}_raw_rms"),
                    rms(&t["layer0.qkv"][offset + part * w..offset + part * w + d]),
                );
            }
            let prob = t["layer0.probabilities"][(b * cfg.heads + head) * 4 + 2] as f64;
            sample(
                metrics,
                format!("head{head}.previous_attention_probability"),
                prob,
            );
            if !diagonal {
                let v0 = &t["layer0.qkv"]
                    [2 * b * 3 * w + 2 * w + head * d..2 * b * 3 * w + 2 * w + (head + 1) * d];
                let v1 = &t["layer0.qkv"][offset + 2 * w..offset + 2 * w + d];
                let a = &t["layer0.attention"][row * w + head * d..row * w + (head + 1) * d];
                let attended: Vec<_> = v0
                    .iter()
                    .zip(v1)
                    .map(|(&x, &y)| prob * x as f64 + (1. - prob) * y as f64)
                    .collect();
                let delta: Vec<_> = v0
                    .iter()
                    .zip(v1)
                    .map(|(&x, &y)| x as f64 - y as f64)
                    .collect();
                let anorm = a.iter().map(|&v| (v as f64).powi(2)).sum::<f64>().sqrt();
                let before = stats::norm(&attended);
                if before > 0. {
                    sample(
                        metrics,
                        format!("head{head}.xsa_norm_over_attended_norm"),
                        anorm / before,
                    );
                }
                let dn = stats::norm(&delta) * prob;
                if dn > 0. {
                    sample(
                        metrics,
                        format!("head{head}.xsa_norm_over_centered_delta_norm"),
                        anorm / dn,
                    );
                }
                let vn = v1.iter().map(|&v| (v as f64).powi(2)).sum::<f64>().sqrt();
                let re: f64 = a.iter().zip(v1).map(|(&x, &y)| x as f64 * y as f64).sum();
                let im: f64 = (0..d / 2)
                    .map(|c| {
                        v1[c] as f64 * a[c + d / 2] as f64 - v1[c + d / 2] as f64 * a[c] as f64
                    })
                    .sum();
                if anorm * vn > 0. {
                    sample(
                        metrics,
                        format!("head{head}.complex_orthogonality_residual"),
                        re.hypot(im) / (anorm * vn),
                    );
                }
            }
        }
    }
}
struct Split {
    report: Value,
    gradient: Vec<f64>,
    predictions: Vec<u32>,
}
fn split(
    cp: &Checkpoint,
    engine: &mut Engine,
    examples: &[data::Example],
    name: &str,
    errors: &mut impl Write,
) -> Result<Split> {
    let n = examples.len();
    let vocab = cp.cfg.vocab;
    let mut gradient = vec![0.; cp.p.len()];
    let mut predictions = Vec::with_capacity(n);
    let mut activation_metrics = BTreeMap::new();
    let mut margins = Vec::new();
    let mut wrong_margins = Vec::new();
    let mut confidence = Vec::new();
    let mut correct = 0;
    let mut loss = 0.;
    let mut diagonal = (0, 0);
    let mut confusions: BTreeMap<(usize, usize), usize> = BTreeMap::new();
    let mut by_sum = vec![(0usize, 0usize); vocab];
    let mut wrong_pairs = Vec::new();
    let mut packed = metrics::Packed::default();
    for start in (0..n).step_by(cp.cfg.max_batch) {
        let end = (start + cp.cfg.max_batch).min(n);
        let batch = end - start;
        let indices: Vec<_> = (start as u32..end as u32).collect();
        packed.fill(examples, &indices);
        require(packed.length == 2, "non-arithmetic input in probe")?;
        engine.forward(&packed.tokens, &packed.labels, batch, 2)?;
        let p = engine.predictions.read()?;
        let z = engine.logits.read()?;
        let losses = engine.losses(batch * 2)?;
        engine.backward(batch, 2)?;
        let g = engine.gradients.read()?;
        require(
            g.iter().all(|x| x.is_finite()),
            "nonfinite endpoint gradient",
        )?;
        for (sum, x) in gradient.iter_mut().zip(g) {
            *sum += x as f64 * batch as f64 / n as f64;
        }
        if name == "validation" {
            activations(
                &mut activation_metrics,
                &engine.diagnostic_snapshot(batch * 2)?,
                &examples[start..end],
                &cp.cfg,
            );
        }
        for (local, e) in examples[start..end].iter().enumerate() {
            let row = 2 * local + 1;
            let gold = e.targets[1] as usize;
            let pred = p[row] as usize;
            let logits = &z[row * vocab..(row + 1) * vocab];
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            let mass: f64 = logits.iter().map(|&v| (v as f64 - max).exp()).sum();
            let prob = (logits[gold] as f64 - max).exp() / mass;
            let competitor = logits
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != gold)
                .map(|(_, &v)| v as f64)
                .fold(f64::NEG_INFINITY, f64::max);
            let margin = logits[gold] as f64 - competitor;
            let hit = pred == gold;
            predictions.push(pred as u32);
            correct += usize::from(hit);
            loss += losses[row] as f64;
            margins.push(margin);
            confidence.push(prob);
            let sum = (e.tokens[0] + e.tokens[1]) as usize % vocab;
            by_sum[sum].0 += 1;
            by_sum[sum].1 += usize::from(hit);
            if e.tokens[0] == e.tokens[1] {
                diagonal.0 += 1;
                diagonal.1 += usize::from(hit);
            }
            if !hit {
                *confusions.entry((gold, pred)).or_default() += 1;
                wrong_margins.push(margin);
                wrong_pairs.push(json!({"a":e.tokens[0],"b":e.tokens[1],"target":gold,"predicted":pred,"margin":margin}));
                writeln!(
                    errors,
                    "{}\t{}\t{name}\t{}\t{}\t{gold}\t{pred}\t{margin}\t{prob}",
                    cp.protocol["arm"].as_str().unwrap(),
                    cp.protocol["seed"],
                    e.tokens[0],
                    e.tokens[1]
                )?;
            }
        }
    }
    let expected = &cp.result["endpoint"][if name == "train" {
        "train"
    } else {
        "validation"
    }];
    require(
        expected["exact"] == correct,
        "checkpoint replay prediction count differs from original endpoint",
    )?;
    require(
        (expected["loss"].as_f64().unwrap() - loss / n as f64).abs() < 1e-7,
        "checkpoint replay loss differs from original endpoint",
    )?;
    let mut confuse: Vec<_> = confusions.into_iter().collect();
    confuse.sort_by_key(|a| std::cmp::Reverse(a.1));
    let mut sums:Vec<_>=by_sum.iter().enumerate().map(|(sum,&(count,right))|json!({"sum_mod97":sum,"examples":count,"correct":right,"accuracy":right as f64/count as f64})).collect();
    sums.sort_by(|a, b| {
        a["accuracy"]
            .as_f64()
            .unwrap()
            .total_cmp(&b["accuracy"].as_f64().unwrap())
    });
    Ok(Split {
        gradient: projection_gradient(&cp.cfg, &gradient),
        predictions,
        report: json!({"examples":n,"exact":correct,"accuracy":correct as f64/n as f64,"loss":loss/n as f64,
        "reproduces_saved_endpoint":true,"gold_margin":stats::summary(&margins),"wrong_gold_margin":stats::summary(&wrong_margins),"gold_probability":stats::summary(&confidence),
        "diagonal_inputs":{"examples":diagonal.0,"correct":diagonal.1},"worst_sums":sums.into_iter().take(12).collect::<Vec<_>>(),
        "top_confusions":confuse.into_iter().take(12).map(|((gold,pred),n)|json!({"target":gold,"predicted":pred,"count":n})).collect::<Vec<_>>(),
        "first_errors":wrong_pairs.into_iter().take(20).collect::<Vec<_>>(),
        "activation_distributions":activation_metrics.into_iter().map(|(name,values)|(name,stats::summary(&values))).collect::<BTreeMap<_,_>>()}),
    })
}
fn geometry(task: &data::Task, train: &Split, heldout: &Split) -> Value {
    let n = task.vocab;
    let mut grid = vec![u32::MAX; n * n];
    let mut is_train = vec![false; n * n];
    for (split, predictions, training) in [
        ("train", &train.predictions, true),
        ("validation", &heldout.predictions, false),
    ] {
        for (e, &pred) in task.splits[split].iter().zip(predictions) {
            let i = e.tokens[0] as usize * n + e.tokens[1] as usize;
            grid[i] = pred;
            is_train[i] = training;
        }
    }
    assert!(grid.iter().all(|&v| v != u32::MAX));
    let mut swaps = 0;
    let mut consistency = 0;
    let mut by_swap_training = [(0usize, 0usize); 2];
    for a in 0..n {
        for b in 0..n {
            swaps += usize::from(grid[a * n + b] != grid[b * n + a]);
            if !is_train[a * n + b] {
                let bucket = usize::from(is_train[b * n + a]);
                by_swap_training[bucket].0 += 1;
                by_swap_training[bucket].1 +=
                    usize::from(grid[a * n + b] as usize == ((a + b) % n).pow(4) % n);
            }
        }
    }
    for sum in 0..n {
        let mut counts = vec![0usize; n];
        for a in 0..n {
            counts[grid[a * n + (sum + n - a) % n] as usize] += 1;
        }
        consistency += counts.into_iter().max().unwrap();
    }
    json!({"swap_disagreement_fraction":swaps as f64/(n*n) as f64,"constant_sum_modal_prediction_consistency":consistency as f64/(n*n) as f64,
        "heldout_swap_in_train":{"examples":by_swap_training[1].0,"correct":by_swap_training[1].1},
        "heldout_swap_not_in_train":{"examples":by_swap_training[0].0,"correct":by_swap_training[0].1}})
}
pub fn run(cp: &Checkpoint, output: &Path) -> Result<Value> {
    let task = data::build(&cp.protocol["case"])?;
    require(task.selection == "validation", "unexpected primary split")?;
    let mut engine = cp.engine()?;
    fs::create_dir_all(output)?;
    let error_path = output.join(format!(
        "pow4-errors-{}-seed{}.tsv",
        cp.protocol["arm"].as_str().unwrap(),
        cp.protocol["seed"]
    ));
    let mut errors = BufWriter::new(fs::File::create(error_path)?);
    writeln!(
        errors,
        "arm\tseed\tsplit\ta\tb\ttarget\tpredicted\tgold_margin\tgold_probability"
    )?;
    let train = split(cp, &mut engine, &task.splits["train"], "train", &mut errors)?;
    let heldout = split(
        cp,
        &mut engine,
        &task.splits["validation"],
        "validation",
        &mut errors,
    )?;
    errors.flush()?;
    let mut gradients = Vec::new();
    for b in blocks(&cp.cfg) {
        let ids = owner_indices(&cp.cfg, &b);
        let tr: Vec<_> = ids.iter().map(|&i| train.gradient[i]).collect();
        let va: Vec<_> = ids.iter().map(|&i| heldout.gradient[i]).collect();
        let momentum: Vec<_> = ids
            .iter()
            .map(|&i| {
                cp.m[i] as f64 / ((cp.v[i] as f64).sqrt() + 1e-8)
                    + cp.cfg.decay as f64 * cp.p[i] as f64
            })
            .collect();
        let weights: Vec<_> = ids.iter().map(|&i| cp.p[i] as f64).collect();
        gradients.push(json!({"block":b.name,"independent_parameters":ids.len(),"train_gradient_l2":stats::norm(&tr),"heldout_gradient_l2":stats::norm(&va),
            "train_heldout_cosine":stats::cosine(&tr,&va),"momentum_direction_train_cosine":stats::cosine(&momentum,&tr),
            "momentum_direction_heldout_cosine":stats::cosine(&momentum,&va),"train_gradient_over_weight_norm":stats::norm(&tr)/stats::norm(&weights)}));
    }
    require(
        engine.parameters.read()? == cp.p
            && engine.first_moment.read()? == cp.m
            && engine.second_moment.read()? == cp.v,
        "probe mutated model or optimizer",
    )?;
    let invariances = geometry(&task, &train, &heldout);
    Ok(
        json!({"arm":cp.protocol["arm"],"seed":cp.protocol["seed"],"checkpoint_sha256":cp.checkpoint["state_sha256"],"train":train.report,"heldout":heldout.report,
        "parameter_gradients":gradients,"function_geometry":invariances,"weights_and_optimizer_unchanged":true,
        "gradient_scope":"full-split cross entropy; contracted to independent complex coordinates before norms/cosines"}),
    )
}
