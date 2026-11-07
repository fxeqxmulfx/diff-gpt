//! Read binary tensors and test saved weights in memory. Never call Engine::update.
use crate::*;

const FILES: &[(&str, &str)] = &[
    ("src/main.rs", include_str!("main.rs")),
    ("src/raw.rs", include_str!("raw.rs")),
    ("src/loss_precision.rs", include_str!("loss_precision.rs")),
    ("src/loss_recheck.rs", include_str!("loss_recheck.rs")),
    ("src/stable_loss.rs", include_str!("stable_loss.rs")),
    ("src/gradient_oracle.rs", include_str!("gradient_oracle.rs")),
    ("src/gradient_stress.rs", include_str!("gradient_stress.rs")),
    ("src/engine.rs", include_str!("engine.rs")),
    ("src/probe.rs", include_str!("probe.rs")),
    ("src/stats.rs", include_str!("stats.rs")),
    ("src/continuation.rs", include_str!("continuation.rs")),
    ("Cargo.toml", include_str!("../Cargo.toml")),
    ("Cargo.lock", include_str!("../Cargo.lock")),
    (
        "CONTINUATION_PLAN.txt",
        include_str!("../CONTINUATION_PLAN.txt"),
    ),
];

pub(crate) struct Snapshot {
    pub(crate) name: String,
    pub(crate) cp: Checkpoint,
    files: Vec<(PathBuf, String)>,
}
impl Snapshot {
    pub(crate) fn unchanged(&self) -> Result<()> {
        for (path, sha) in &self.files {
            require(
                hash(&fs::read(path)?) == *sha,
                "raw diagnostic changed an input file",
            )?;
        }
        Ok(())
    }
}

fn floats(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    require(bytes.len() == 4 * n, "raw state has wrong byte length")?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| f32::from_le_bytes(*v))
        .collect())
}

pub(crate) fn load(parent: &Checkpoint, root: &Path, arm: &str, step: usize) -> Result<Snapshot> {
    let dir = root.join(arm);
    let meta_path = dir.join(format!("checkpoint-{step:06}.json"));
    let meta = read(&meta_path)?;
    let protocol_path = dir.join("protocol.json");
    let protocol = read(&protocol_path)?;
    require(
        meta["parent_checkpoint_sha256"] == parent.checkpoint["state_sha256"]
            && protocol["parent_checkpoint_sha256"] == parent.checkpoint["state_sha256"]
            && meta["source_sha256"] == protocol["source_sha256"]
            && meta["additional_updates"] == step
            && meta["global_updates"] == 100000 + step,
        "unexpected continuation snapshot provenance/counters",
    )?;
    let result = read(&dir.join("result.json"))?;
    require(
        result["snapshots"]
            .as_array()
            .ok_or("no snapshots")?
            .contains(&meta),
        "snapshot differs from audited result",
    )?;
    let state_path = dir.join(meta["state_file"].as_str().ok_or("no state path")?);
    let history_path = dir.join(meta["history_file"].as_str().ok_or("no history path")?);
    let state = fs::read(&state_path)?;
    let history = fs::read(&history_path)?;
    require(
        hash(&state) == meta["state_sha256"] && hash(&history) == meta["history_sha256"],
        "raw snapshot integrity failure",
    )?;
    let n = parent.p.len();
    let all = floats(&state, 3 * n)?;
    let (p, m, v) = (
        all[..n].to_vec(),
        all[n..2 * n].to_vec(),
        all[2 * n..].to_vec(),
    );
    let mut cfg = parent.cfg.clone();
    cfg.lr = protocol["learning_rate"].as_f64().ok_or("missing LR")?;
    require(
        cfg.lr == if arm == "unchanged" { 0.002 } else { 0.0002 },
        "unexpected arm LR",
    )?;
    layout::validate_state(&cfg, &p, &m, &v)?;
    if step == 0 {
        require(
            p == parent.p && m == parent.m && v == parent.v,
            "initial state differs from parent",
        )?;
    }
    let mut history: Vec<Value> = history
        .split(|&b| b == b'\n')
        .filter(|l| !l.is_empty())
        .map(serde_json::from_slice)
        .collect::<std::result::Result<_, _>>()?;
    require(
        history.last() == Some(&meta["point"]),
        "raw history endpoint mismatch",
    )?;
    for point in &mut history {
        point["epochs"] = point["global_epochs"].clone();
    }
    let cp = Checkpoint {
        dir,
        protocol: parent.protocol.clone(),
        result: json!({"status":"saved-continuation","epochs":meta["point"]["global_epochs"],"endpoint":meta["point"]}),
        checkpoint: json!({"completed":meta["global_updates"],"state_sha256":meta["state_sha256"]}),
        cfg,
        p,
        m,
        v,
        initial: parent.initial.clone(),
        history,
    };
    let paths = [
        meta_path,
        protocol_path,
        state_path,
        history_path,
        cp.dir.join("result.json"),
    ];
    let files = paths
        .into_iter()
        .map(|p| Ok((p.clone(), hash(&fs::read(p)?))))
        .collect::<Result<Vec<_>>>()?;
    Ok(Snapshot {
        name: format!("{arm}-{step:06}"),
        cp,
        files,
    })
}

fn radial_tangent(weights: &[f64], displacement: &[f64]) -> Value {
    let ww = stats::dot(weights, weights);
    let dd = stats::dot(displacement, displacement);
    let coefficient = if ww > 0. {
        stats::dot(weights, displacement) / ww
    } else {
        0.
    };
    let radial = coefficient * coefficient * ww;
    json!({"signed_radial_fraction":coefficient,
        "relative_l2":(dd/ww).sqrt(),"tangent_over_weight_norm":((dd-radial).max(0.)/ww).sqrt(),
        "tangent_energy_fraction":if dd>0.{Some((1.-radial/dd).clamp(0.,1.))}else{None}})
}

fn raw_blocks(cp: &Checkpoint, parent: Option<&Checkpoint>) -> Vec<Value> {
    let pairs = layout::pairs(&cp.cfg);
    blocks(&cp.cfg).iter().map(|b| {
        let ids = owner_indices(&cp.cfg,b);
        let w:Vec<_> = ids.iter().map(|&i|cp.p[i] as f64).collect();
        let m:Vec<_> = ids.iter().map(|&i|cp.m[i] as f64).collect();
        let sqrtv:Vec<_> = ids.iter().map(|&i|(cp.v[i] as f64).sqrt()).collect();
        let adapt:Vec<_> = m.iter().zip(&sqrtv).map(|(m,v)|cp.cfg.lr*m/(v+1e-8)).collect();
        let whole:Vec<_> = adapt.iter().zip(&w).map(|(a,w)|a+cp.cfg.lr*cp.cfg.decay as f64*w).collect();
        let mut order:Vec<_> = (0..ids.len()).collect();
        order.sort_by(|&a,&b|adapt[b].abs().total_cmp(&adapt[a].abs()));
        let energy = stats::dot(&adapt,&adapt);
        let top_fraction = |f:f64| if energy>0. {
            Some(order.iter().take((ids.len() as f64*f).ceil() as usize).map(|&i|adapt[i].powi(2)).sum::<f64>()/energy)
        } else {None};
        let mut weight_order:Vec<_> = (0..ids.len()).collect();
        weight_order.sort_by(|&a,&b|w[b].abs().total_cmp(&w[a].abs()));
        let top_weights:Vec<_> = weight_order.iter().take(5).map(|&j|json!({"flat_index":ids[j],
            "row":(ids[j]-b.start)/b.cols,"column":(ids[j]-b.start)%b.cols,"weight":w[j]})).collect();
        let top_steps:Vec<_> = order.iter().take(5).map(|&j|json!({"flat_index":ids[j],
            "row":(ids[j]-b.start)/b.cols,"column":(ids[j]-b.start)%b.cols,
            "weight":w[j],"first_moment":m[j],"sqrt_second_moment":sqrtv[j],"historical_adaptive_step":adapt[j]})).collect();
        let anisotropy:Vec<_> = pairs.iter().filter(|p|(p[0] as usize)>=b.start&&(p[0] as usize)<b.end())
            .map(|p| {let a=(cp.v[p[0] as usize] as f64).sqrt()+1e-8;
                let c=(cp.v[p[1] as usize] as f64).sqrt()+1e-8;a.max(c)/a.min(c)}).collect();
        let drift = parent.map(|p| {
            let base:Vec<_>=ids.iter().map(|&i|p.p[i] as f64).collect();
            let d:Vec<_>=w.iter().zip(&base).map(|(a,b)|a-b).collect();
            json!({"cosine":stats::cosine(&w,&base),"norm_ratio":stats::norm(&w)/stats::norm(&base),
                "displacement":radial_tangent(&base,&d)})
        });
        json!({"block":b.name,"first_flat_index":b.start,"expanded_shape":[b.rows,b.cols],"independent_entries":ids.len(),
            "raw_weight":stats::summary(&w),"raw_first_moment":stats::summary(&m),"sqrt_second_moment":stats::summary(&sqrtv),
            "epsilon_dominated_fraction":sqrtv.iter().filter(|&&v|v<=1e-8).count() as f64/ids.len() as f64,
            "historical_adaptive_step":stats::summary(&adapt),"historical_step_geometry":radial_tangent(&w,&whole),
            "adaptive_step_geometry":radial_tangent(&w,&adapt),"adaptive_step_energy_top_1pct":top_fraction(0.01),
            "adaptive_step_energy_top_0_1pct":top_fraction(0.001),"pair_denominator_anisotropy":stats::summary(&anisotropy),
            "largest_abs_weights":top_weights,"largest_abs_historical_steps":top_steps,"drift_from_parent":drift})
    }).collect()
}

fn difference(a: &Checkpoint, b: &Checkpoint) -> Vec<Value> {
    blocks(&a.cfg).iter().map(|block| {
        let ids=owner_indices(&a.cfg,block);
        let x:Vec<_>=ids.iter().map(|&i|a.p[i] as f64).collect();
        let y:Vec<_>=ids.iter().map(|&i|b.p[i] as f64).collect();
        let d:Vec<_>=y.iter().zip(&x).map(|(y,x)|y-x).collect();
        let rows:Vec<_>=(0..block.rows).filter_map(|r| {
            let s=block.start+r*block.cols;
            let x:Vec<_>=a.p[s..s+block.cols].iter().map(|&x|x as f64).collect();
            let y:Vec<_>=b.p[s..s+block.cols].iter().map(|&x|x as f64).collect();
            stats::cosine(&x,&y)
        }).collect();
        json!({"block":block.name,"cosine":stats::cosine(&x,&y),"norm_ratio":stats::norm(&y)/stats::norm(&x),
            "row_direction_cosine":stats::summary(&rows),"difference":radial_tangent(&x,&d)})
    }).collect()
}

fn embedding(cp: &Checkpoint) -> Value {
    let w = cp.cfg.width;
    let valid: std::collections::BTreeSet<_> = (0..97usize).map(|s| s.pow(4) % 97).collect();
    let vectors: Vec<Vec<_>> = cp.p[..cp.cfg.vocab * w]
        .chunks_exact(w)
        .map(|row| row.iter().map(|&v| v as f64).collect())
        .collect();
    let norms: Vec<_> = vectors.iter().map(|v| stats::norm(v)).collect();
    let output: Vec<_> = valid.iter().map(|&i| norms[i]).collect();
    let other: Vec<_> = (0..97)
        .filter(|i| !valid.contains(i))
        .map(|i| norms[i])
        .collect();
    let mut similar = Vec::new();
    for &a in &valid {
        for &b in &valid {
            if a < b {
                similar.push((stats::cosine(&vectors[a], &vectors[b]).unwrap_or(0.), a, b));
            }
        }
    }
    similar.sort_by(|a, b| b.0.total_cmp(&a.0));
    json!({"valid_output_token_norms":stats::summary(&output),"other_token_norms":stats::summary(&other),
        "most_similar_valid_readout_rows":similar.iter().take(10).map(|&(cos,a,b)|json!({"tokens":[a,b],"cosine":cos,"norms":[norms[a],norms[b]]})).collect::<Vec<_>>()})
}

fn group_indices(cfg: &Config, names: &[&str]) -> Vec<usize> {
    blocks(cfg)
        .iter()
        .filter(|b| names.contains(&b.name))
        .flat_map(|b| b.start..b.end())
        .collect()
}

fn inference(engine: &mut Engine, task: &data::Task) -> Result<Value> {
    let mut result = serde_json::Map::new();
    for split in ["train", "validation"] {
        let metric = metrics::evaluate(engine, &task.splits[split], false)?;
        result.insert(
            split.into(),
            json!({"exact":metric["exact"],"examples":metric["examples"],
            "accuracy":metric["sequence_accuracy"],"loss":metric["loss"]}),
        );
    }
    Ok(Value::Object(result))
}

fn groups() -> Vec<Vec<&'static str>> {
    let mut result: Vec<_> = [
        "embedding_readout",
        "Q",
        "K",
        "V",
        "attention_output",
        "ffn_gate",
        "ffn_value",
        "ffn_output",
    ]
    .into_iter()
    .map(|n| vec![n])
    .collect();
    result.extend([
        vec!["Q", "K"],
        vec!["V", "attention_output"],
        vec!["ffn_gate", "ffn_value", "ffn_output"],
        vec!["Q", "K", "V", "attention_output"],
        vec![
            "embedding_readout",
            "Q",
            "K",
            "V",
            "attention_output",
            "ffn_gate",
            "ffn_value",
            "ffn_output",
        ],
    ]);
    result
}

fn interventions(a: &Snapshot, b: &Snapshot, task: &data::Task) -> Result<Value> {
    let mut swaps = Vec::new();
    let mut sensitivities = Vec::new();
    for (recipient, donor) in [(a, b), (b, a)] {
        let cp = &recipient.cp;
        let mut engine = cp.engine()?;
        let baseline = inference(&mut engine, task)?;
        require(
            baseline["validation"]["exact"] == cp.result["endpoint"]["validation"]["exact"]
                && baseline["train"]["exact"] == cp.result["endpoint"]["train"]["exact"],
            "inference baseline mismatch",
        )?;
        for names in groups() {
            let ids = group_indices(&cp.cfg, &names);
            let mut p = cp.p.clone();
            for &i in &ids {
                p[i] = donor.cp.p[i];
            }
            layout::validate(&cp.cfg, &p)?;
            engine.parameters.write(&p)?;
            let score = inference(&mut engine, task)?;
            if ids.len() == p.len() {
                require(
                    score["validation"]["exact"]
                        == donor.cp.result["endpoint"]["validation"]["exact"]
                        && score["train"]["exact"] == donor.cp.result["endpoint"]["train"]["exact"],
                    "whole-model transplant failed positive control",
                )?;
            }
            swaps.push(
                json!({"recipient":recipient.name,"donor":donor.name,"copied_blocks":names,
                "baseline":baseline,"scores":score}),
            );
        }
        // Compare sensitivity at an equal 0.1% block-weight norm. The direction
        // comes from saved moments; no current minibatch or optimizer update.
        for block in blocks(&cp.cfg) {
            let ids = owner_indices(&cp.cfg, &block);
            let w: Vec<_> = ids.iter().map(|&i| cp.p[i] as f64).collect();
            let direction: Vec<_> = ids
                .iter()
                .map(|&i| {
                    cp.m[i] as f64 / ((cp.v[i] as f64).sqrt() + 1e-8)
                        + cp.cfg.decay as f64 * cp.p[i] as f64
                })
                .collect();
            let length = stats::norm(&direction);
            require(length > 0., "zero sensitivity direction")?;
            let factor = 0.001 * stats::norm(&w) / length;
            for sign in [-1., 1.] {
                let mut p = cp.p.clone();
                for (&i, &d) in ids.iter().zip(&direction) {
                    p[i] = (p[i] as f64 - sign * factor * d) as f32;
                }
                for [ar, bi, br, ai, kind] in layout::pairs(&cp.cfg) {
                    if kind == 0 {
                        p[ai as usize] = p[ar as usize];
                        p[br as usize] = -p[bi as usize];
                    }
                }
                layout::validate(&cp.cfg, &p)?;
                engine.parameters.write(&p)?;
                let score = inference(&mut engine, task)?;
                let actual: Vec<_> = ids.iter().map(|&i| p[i] as f64 - cp.p[i] as f64).collect();
                sensitivities.push(json!({"snapshot":recipient.name,"block":block.name,
                    "direction":if sign>0.{"subtract saved optimizer direction"}else{"add saved optimizer direction"},
                    "intended_relative_displacement":0.001,"actual_relative_displacement":stats::norm(&actual)/stats::norm(&w),
                    "baseline":baseline,"scores":score}));
            }
        }
        engine.parameters.write(&cp.p)?;
        require(
            engine.parameters.read()? == cp.p
                && engine.first_moment.read()? == cp.m
                && engine.second_moment.read()? == cp.v,
            "intervention did not restore in-memory model or changed moments",
        )?;
        recipient.unchanged()?;
    }
    Ok(
        json!({"scope":"inference only, no learning or file mutation; copies are discarded",
        "transplants":swaps,"equal_norm_momentum_sensitivity":sensitivities,
        "limitations":"Transplant failures can reflect co-adapted representations, not faulty blocks. Fixed-direction local sensitivity is not a worst-case bound or proof of the original crash trigger."}),
    )
}

pub fn run(mode: &str, parent_root: &Path, continuation_root: &Path, output: &Path) -> Result<()> {
    require(
        mode == "weights" || mode == "probe" || mode == "precision" || mode == "fixed-loss",
        "unknown raw diagnostic mode",
    )?;
    require(
        !output.exists(),
        "refuse overwriting an existing raw diagnostic",
    )?;
    let parent = Checkpoint::load(parent_root, "complex-hermitian", 1)?;
    let summary = read(&continuation_root.join("summary.json"))?;
    require(
        summary["complete"] == true
            && summary["protocol"]["parent_checkpoint_sha256"] == parent.checkpoint["state_sha256"],
        "continuation is incomplete or from another parent",
    )?;
    let mut snapshots = Vec::new();
    for arm in ["unchanged", "lower_lr"] {
        for step in [0, 9500, 19000] {
            snapshots.push(load(&parent, continuation_root, arm, step)?);
        }
    }
    fs::create_dir_all(output)?;
    let mut bytes = Vec::new();
    for &(name, body) in FILES {
        let path = output.join("source").join(name);
        fs::create_dir_all(path.parent().unwrap())?;
        fs::write(path, body)?;
        bytes.extend((name.len() as u64).to_le_bytes());
        bytes.extend(name.as_bytes());
        bytes.extend((body.len() as u64).to_le_bytes());
        bytes.extend(body.as_bytes());
    }
    let plan = json!({"purpose":"Raw binary weight and moment diagnosis; no training or benchmark selection",
        "mode":mode,"parent_root":parent_root,"continuation_root":continuation_root,
        "format":"little-endian f32 arrays: expanded parameters, first moment, second moment",
        "expanded_entries_per_array":parent.p.len(),"independent_parameters":layout::independent_count(&parent.cfg),
        "source_sha256":hash(&bytes),"canonical_ptx_sha256":ptx_hash(),"parent_state_sha256":parent.checkpoint["state_sha256"],
        "snapshots":snapshots.iter().map(|s|json!({"name":s.name,"state_sha256":s.cp.checkpoint["state_sha256"]})).collect::<Vec<_>>(),
        "gpu_probe":mode!="weights","transplant_pair_additional_epoch":500,"transplant_groups":groups(),
        "local_sensitivity_relative_displacement":0.001,"local_sensitivity_signs":[-1,1],
        "reference_seeds":[0,2],"reference_limitation":"Success endpoints have different training ages and independent coordinate systems",
        "python_invoked":false,"training_updates":0,"benchmark_modified":false,
        "fixed_loss_kernel_ptx_sha256":if mode=="fixed-loss"{Some(hash(stable_loss::kernels::PTX.as_bytes()))}else{None},
        "fixed_loss_logit_relative_tolerance":loss_recheck::LOGIT_RELATIVE_TOLERANCE,
        "fixed_loss_parameter_relative_tolerance":loss_recheck::PARAMETER_RELATIVE_TOLERANCE,
        "fixed_loss_checkpoint_count":if mode=="fixed-loss"{18}else{0}});
    save(&output.join("protocol.json"), &plan)?;
    let mut rows = Vec::new();
    for s in &snapshots {
        let cp = &s.cp;
        let report = json!({"snapshot":s.name,"learning_rate":cp.cfg.lr,
            "weight_report":weight_report(cp),"raw_blocks":raw_blocks(cp,Some(&parent)),"readout_geometry":embedding(cp)});
        save(&output.join(format!("{}.json", s.name)), &report)?;
        rows.push(report);
        s.unchanged()?;
        eprintln!("read raw tensors {}", s.name);
    }
    let mut references = Vec::new();
    for seed in [0, 2] {
        let cp = Checkpoint::load(parent_root, "complex-hermitian", seed)?;
        references.push(json!({"seed":seed,"weight_report":weight_report(&cp),"raw_blocks":raw_blocks(&cp,None),"readout_geometry":embedding(&cp)}));
        cp.unchanged()?;
    }
    let contrasts: Vec<_> = [0, 1, 2]
        .iter()
        .map(|&i| {
            json!({"left":snapshots[i].name,"right":snapshots[i+3].name,
        "blocks":difference(&snapshots[i].cp,&snapshots[i+3].cp)})
        })
        .collect();
    save(
        &output.join("weights.json"),
        &json!({"protocol":plan,"rows":rows,"success_references":references,"paired_weight_differences":contrasts}),
    )?;
    if mode == "probe" {
        let mut probes = Vec::new();
        for s in &snapshots {
            let result = probe::run(&s.cp, &output.join(&s.name))?;
            save(&output.join(format!("{}-probe.json", s.name)), &result)?;
            probes.push(json!({"snapshot":s.name,"probe":result}));
            s.unchanged()?;
            eprintln!("replayed raw checkpoint {}", s.name);
        }
        let task = data::build(&parent.protocol["case"])?;
        let comparison = interventions(&snapshots[1], &snapshots[4], &task)?;
        save(
            &output.join("probes.json"),
            &json!({"protocol":plan,"rows":probes,"interventions":comparison}),
        )?;
    }
    if mode == "precision" {
        let mut rows = Vec::new();
        for s in &snapshots {
            let result = loss_precision::run(&s.cp, &s.name)?;
            save(&output.join(format!("{}-precision.json", s.name)), &result)?;
            rows.push(result);
            s.unchanged()?;
            eprintln!("checked loss/gradient precision {}", s.name);
        }
        save(
            &output.join("precision.json"),
            &json!({"protocol":plan,"rows":rows}),
        )?;
    }
    if mode == "fixed-loss" {
        let mut rows = Vec::new();
        for s in &snapshots {
            let result = loss_recheck::run(&s.cp, &s.name)?;
            save(&output.join(format!("{}-fixed-loss.json", s.name)), &result)?;
            rows.push(result);
            s.unchanged()?;
            eprintln!("rechecked corrected loss {}", s.name);
        }
        for arm in ARMS {
            for seed in 0..3 {
                let cp = Checkpoint::load(parent_root, arm, seed)?;
                let name = format!("{arm}-seed{seed}");
                let result = loss_recheck::run(&cp, &name)?;
                save(&output.join(format!("{name}-fixed-loss.json")), &result)?;
                rows.push(result);
                cp.unchanged()?;
                eprintln!("rechecked corrected loss {name}");
            }
        }
        save(
            &output.join("fixed-loss.json"),
            &json!({"complete":true,"expected_checkpoints":18,"protocol":plan,"rows":rows}),
        )?;
    }
    for s in &snapshots {
        s.unchanged()?;
    }
    parent.unchanged()?;
    save(
        &output.join("audit.json"),
        &json!({"complete":true,"raw_snapshots":6,"successful_reference_checkpoints":2,
        "hashes_and_complex_constraints":"passed","input_files_unchanged":true,"gpu_replays":if mode=="fixed-loss"{18}else if mode!="weights"{6}else{0},
        "corrected_loss_reference_split_checks":if mode=="fixed-loss"{36}else{0},
        "softmax_reference_gradient_comparisons":if mode=="precision"{12}else{0},
        "in_memory_transplants":if mode=="probe"{26}else{0},"local_sensitivity_probes":if mode=="probe"{32}else{0},
        "whole_model_transplant_controls":if mode=="probe"{"passed"}else{"not run"},
        "training_updates":0,"python_invoked":false}),
    )
}

#[test]
fn raw_decode_and_radial_geometry() {
    let bytes: Vec<_> = [1.25f32, -0.5, 0.]
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    assert_eq!(floats(&bytes, 3).unwrap(), vec![1.25, -0.5, 0.]);
    assert!(floats(&bytes, 2).is_err());
    let radial = radial_tangent(&[3., 4.], &[0.3, 0.4]);
    assert!((radial["relative_l2"].as_f64().unwrap() - 0.1).abs() < 1e-12);
    assert!(radial["tangent_over_weight_norm"].as_f64().unwrap() < 1e-8);
    let tangent = radial_tangent(&[3., 4.], &[-0.4, 0.3]);
    assert!((tangent["tangent_energy_fraction"].as_f64().unwrap() - 1.).abs() < 1e-12);
}

#[test]
fn block_transplants_preserve_all_complex_constraints() {
    let cfg = Config {
        width: 128,
        vocab: 97,
        heads: 4,
        layers: 1,
        max_batch: 256,
        max_length: 2,
        normalized: true,
        ssmax: true,
        gain: 1.,
        lr: 0.002,
        decay: 0.1,
        warmup: 50,
        architecture: "complex-hermitian".into(),
        phase_optimizer: false,
    };
    let a = layout::initialize_model(&cfg, 1);
    let b = layout::initialize_model(&cfg, 2);
    for names in groups() {
        let mut p = a.clone();
        let ids = group_indices(&cfg, &names);
        for &i in &ids {
            p[i] = b[i];
        }
        layout::validate(&cfg, &p).unwrap();
        if names.len() == 8 {
            assert_eq!(p, b);
            assert_eq!(ids.len(), a.len());
        }
    }
}
