//! Checkpoint inspection and separately authorized post-budget continuations.
#[path = "../../native/src/arithmetic_data.rs"]
#[allow(dead_code)]
mod arithmetic_data;
#[path = "../../native/src/attention.rs"]
mod attention;
#[path = "../../native/src/base_ops.rs"]
mod base_ops;
#[path = "../../native/src/blake.rs"]
mod blake;
#[path = "../../native/src/complex.rs"]
mod complex;
mod continuation;
#[path = "../../native/src/data.rs"]
mod data;
#[allow(dead_code)]
mod engine;
#[path = "../../native/src/gemm.rs"]
mod gemm;
mod gradient_oracle;
mod gradient_stress;
#[path = "../../native/src/layout.rs"]
mod layout;
mod loss_precision;
mod loss_recheck;
#[path = "../../native/src/metrics.rs"]
mod metrics;
mod momentum_continuation;
mod next_update;
mod no_xsa;
mod ordinary_attention;
#[path = "../../native/src/pcg.rs"]
mod pcg;
mod plain_ffn;
mod plain_product;
mod probe;
#[path = "../../native/src/random.rs"]
mod random;
mod raw;
mod retrain;
#[path = "../../native/src/sequence_ops.rs"]
mod sequence_ops;
mod stable_attention;
mod stable_chain;
mod stable_loss;
mod stats;
#[path = "../../native/src/sources.rs"]
mod training_sources;
use engine::{Config, Engine, Result};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};
const ARMS: [&str; 4] = [
    "real-product",
    "complex-linear",
    "complex-hermitian",
    "complex-hermitian-phase",
];
const CASE: &str = "arithmetic-sum-pow4";
fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
fn ptx_hash() -> String {
    hash(
        [
            attention::kernels::PTX,
            base_ops::kernels::PTX,
            sequence_ops::kernels::PTX,
            complex::kernels::PTX,
            gemm::kernels::PTX,
        ]
        .concat()
        // Kernite mangles workspace/crate names into kernel symbols. Only that
        // namespace differs in this separate inspector; require all other PTX
        // bytes to match the recorded training module exactly.
        .replace(
            "_ZN10diagnostic25complex_weight_diagnostic",
            "_ZN6native16complex_geometry",
        )
        .as_bytes(),
    )
}
fn read(path: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}
fn require(test: bool, message: &str) -> Result<()> {
    if test {
        Ok(())
    } else {
        Err(message.to_string().into())
    }
}
fn save(path: &Path, value: &Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let pending = path.with_extension("pending");
    fs::write(&pending, serde_json::to_vec_pretty(value)?)?;
    fs::rename(pending, path)?;
    Ok(())
}
fn config(protocol: &Value) -> Config {
    let c = &protocol["case"]["case"];
    let arm = protocol["arm"].as_str().unwrap();
    Config {
        width: c["width"].as_u64().unwrap() as usize,
        vocab: 97,
        heads: c["heads"].as_u64().unwrap() as usize,
        layers: c["layers"].as_u64().unwrap() as usize,
        max_batch: 256,
        max_length: 2,
        normalized: true,
        ssmax: true,
        gain: protocol["gain"].as_f64().unwrap() as f32,
        lr: protocol["learning_rate"].as_f64().unwrap(),
        decay: protocol["weight_decay"].as_f64().unwrap() as f32,
        warmup: protocol["warmup"].as_u64().unwrap() as usize,
        architecture: arm.trim_end_matches("-phase").into(),
        phase_optimizer: arm.ends_with("-phase"),
    }
}
struct Checkpoint {
    dir: PathBuf,
    protocol: Value,
    result: Value,
    checkpoint: Value,
    cfg: Config,
    p: Vec<f32>,
    m: Vec<f32>,
    v: Vec<f32>,
    initial: Vec<f32>,
    history: Vec<Value>,
}
impl Checkpoint {
    fn load(root: &Path, arm: &str, seed: u32) -> Result<Self> {
        let dir = root.join(arm).join(CASE).join(format!("seed{seed}"));
        let protocol = read(&dir.join("protocol.json"))?;
        let checkpoint = read(&dir.join("checkpoint.json"))?;
        let result = read(&dir.join("result.json"))?;
        let execution = read(&dir.join("execution.json"))?;
        require(
            execution["clean"] == true
                && execution["possible_completed"] == execution["durable_completed"],
            "unclean checkpoint",
        )?;
        let cfg = config(&protocol);
        require(
            cfg.width == 128 && cfg.layers == 1 && cfg.heads == 4,
            "diagnostic scope is fixed pow4 shape",
        )?;
        let bytes = fs::read(dir.join("checkpoint.bin"))?;
        let history = fs::read(dir.join("history.jsonl"))?;
        let initial = fs::read(dir.join("initial.f32"))?;
        require(
            hash(&bytes) == checkpoint["state_sha256"]
                && hash(&history) == checkpoint["history_sha256"]
                && hash(&initial) == protocol["initial_parameters_sha256"],
            "checkpoint or history hash mismatch",
        )?;
        let validated: Value = serde_json::from_str(include_str!("../../VALIDATION.json"))?;
        require(
            protocol["source_sha256"] == validated["source_sha256"]
                && protocol["ptx_sha256"] == validated["ptx_sha256"]
                && checkpoint["source_sha256"] == protocol["source_sha256"]
                && checkpoint["ptx_sha256"] == protocol["ptx_sha256"],
            "checkpoint provenance changed",
        )?;
        let n = layout::count(cfg.width, cfg.layers, cfg.vocab);
        require(
            protocol["ptx_sha256"] == ptx_hash(),
            "diagnostic PTX differs from scientific checkpoint",
        )?;
        require(
            protocol["independent_parameters"] == layout::independent_count(&cfg),
            "independent parameter count changed",
        )?;
        require(
            bytes.len() == 12 * n && initial.len() == 4 * n,
            "wrong state layout",
        )?;
        let values: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        let p = values[..n].to_vec();
        let m = values[n..2 * n].to_vec();
        let v = values[2 * n..].to_vec();
        layout::validate_state(&cfg, &p, &m, &v)?;
        let initial: Vec<f32> = initial
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        require(
            initial == layout::initialize_model(&cfg, seed),
            "unexpected initialization",
        )?;
        let history = history
            .split(|&b| b == b'\n')
            .filter(|line| !line.is_empty())
            .map(serde_json::from_slice)
            .collect::<std::result::Result<_, _>>()?;
        Ok(Self {
            dir,
            protocol,
            result,
            checkpoint,
            cfg,
            p,
            m,
            v,
            initial,
            history,
        })
    }
    fn engine(&self) -> Result<Engine> {
        let mut engine = Engine::new(self.cfg.clone(), &self.p)?;
        engine.first_moment.write(&self.m)?;
        engine.second_moment.write(&self.v)?;
        Ok(engine)
    }
    fn unchanged(&self) -> Result<()> {
        require(
            hash(&fs::read(self.dir.join("checkpoint.bin"))?) == self.checkpoint["state_sha256"],
            "input checkpoint was changed",
        )
    }
}

struct Block {
    name: &'static str,
    start: usize,
    rows: usize,
    cols: usize,
}
impl Block {
    fn end(&self) -> usize {
        self.start + self.rows * self.cols
    }
}
fn blocks(cfg: &Config) -> Vec<Block> {
    let w = cfg.width;
    let h = layout::hidden(w, "");
    let o = cfg.vocab * w;
    vec![
        Block {
            name: "embedding_readout",
            start: 0,
            rows: cfg.vocab,
            cols: w,
        },
        Block {
            name: "Q",
            start: o,
            rows: w,
            cols: w,
        },
        Block {
            name: "K",
            start: o + w * w,
            rows: w,
            cols: w,
        },
        Block {
            name: "V",
            start: o + 2 * w * w,
            rows: w,
            cols: w,
        },
        Block {
            name: "attention_output",
            start: o + 3 * w * w,
            rows: w,
            cols: w,
        },
        Block {
            name: "ffn_gate",
            start: o + 4 * w * w,
            rows: h,
            cols: w,
        },
        Block {
            name: "ffn_value",
            start: o + 4 * w * w + h * w,
            rows: h,
            cols: w,
        },
        Block {
            name: "ffn_output",
            start: o + 4 * w * w + 2 * h * w,
            rows: w,
            cols: h,
        },
    ]
}
fn projection_gradient(cfg: &Config, gradient: &[f64]) -> Vec<f64> {
    if !layout::is_complex(&cfg.architecture) {
        return gradient.to_vec();
    }
    let mut result = vec![0.; gradient.len()];
    for [ar, bi, br, ai, kind] in layout::pairs(cfg) {
        result[ar as usize] =
            gradient[ar as usize] + if kind == 0 { gradient[ai as usize] } else { 0. };
        result[bi as usize] =
            gradient[bi as usize] - if kind == 0 { gradient[br as usize] } else { 0. };
    }
    result
}
fn owner_indices(cfg: &Config, block: &Block) -> Vec<usize> {
    if !layout::is_complex(&cfg.architecture) {
        return (block.start..block.end()).collect();
    }
    layout::pairs(cfg)
        .into_iter()
        .flat_map(|p| [p[0] as usize, p[1] as usize])
        .filter(|&i| i >= block.start && i < block.end())
        .collect()
}
fn history_point(p: &Value) -> Value {
    json!({"epochs":p["epochs"],"train_accuracy":p["train"]["sequence_accuracy"],
    "train_loss":p["train"]["loss"],"heldout_accuracy":p["validation"]["sequence_accuracy"],"heldout_loss":p["validation"]["loss"]})
}
fn history_report(cp: &Checkpoint) -> Value {
    let best = cp
        .history
        .iter()
        .max_by(|a, b| {
            a["validation"]["sequence_accuracy"]
                .as_f64()
                .unwrap()
                .total_cmp(&b["validation"]["sequence_accuracy"].as_f64().unwrap())
        })
        .unwrap();
    let tail = &cp.history[cp.history.len().saturating_sub(1000)..];
    let mut drops: Vec<_> = cp
        .history
        .windows(2)
        .map(|w| {
            (
                w[0]["validation"]["sequence_accuracy"].as_f64().unwrap()
                    - w[1]["validation"]["sequence_accuracy"].as_f64().unwrap(),
                &w[0],
                &w[1],
            )
        })
        .collect();
    drops.sort_by(|a, b| b.0.total_cmp(&a.0));
    json!({"observations":cp.history.len(),"peak":history_point(best),"final":history_point(cp.history.last().unwrap()),
        "last1000_accuracy":stats::summary(&tail.iter().map(|p|p["validation"]["sequence_accuracy"].as_f64().unwrap()).collect::<Vec<_>>()),
        "largest_epoch_drops":drops.iter().take(5).map(|(drop,before,after)|json!({"accuracy_drop":drop,"before":history_point(before),"after":history_point(after)})).collect::<Vec<_>>()})
}
fn fourier_embedding(cp: &Checkpoint) -> Value {
    let w = cp.cfg.width;
    let n = cp.cfg.vocab;
    let mut modes = Vec::new();
    for k in 1..=n / 2 {
        let mut energy = 0.;
        for c in 0..w {
            let mut re = 0.;
            let mut im = 0.;
            for a in 0..n {
                let angle = std::f64::consts::TAU * (k * a) as f64 / n as f64;
                re += cp.p[a * w + c] as f64 * angle.cos();
                im += cp.p[a * w + c] as f64 * angle.sin();
            }
            energy += re * re + im * im;
        }
        modes.push((k, energy));
    }
    let total: f64 = modes.iter().map(|v| v.1).sum();
    let participation = total * total / modes.iter().map(|v| v.1 * v.1).sum::<f64>();
    modes.sort_by(|a, b| b.1.total_cmp(&a.1));
    json!({"non_dc_frequency_pair_participation":participation,"top_modes":modes.iter().take(8).map(|(k,e)|json!({"frequency":k,"fraction":e/total})).collect::<Vec<_>>()})
}
fn weight_report(cp: &Checkpoint) -> Value {
    let mut rows = Vec::new();
    let step = cp.checkpoint["completed"].as_u64().unwrap() as f64;
    for b in blocks(&cp.cfg) {
        let values: Vec<f64> = cp.p[b.start..b.end()].iter().map(|&x| x as f64).collect();
        let initial: Vec<f64> = cp.initial[b.start..b.end()]
            .iter()
            .map(|&x| x as f64)
            .collect();
        let displacement: Vec<_> = values.iter().zip(&initial).map(|(x, y)| x - y).collect();
        let ids = owner_indices(&cp.cfg, &b);
        let weights: Vec<_> = ids.iter().map(|&i| cp.p[i] as f64).collect();
        let first: Vec<_> = ids.iter().map(|&i| cp.m[i] as f64).collect();
        let second: Vec<_> = ids.iter().map(|&i| cp.v[i] as f64).collect();
        // Historical momentum direction only: not a next-minibatch update prediction.
        let update: Vec<_> = ids
            .iter()
            .map(|&i| {
                cp.cfg.lr
                    * (cp.m[i] as f64
                        / (1. - 0.9f64.powf(step))
                        / ((cp.v[i] as f64 / (1. - 0.98f64.powf(step))).sqrt() + 1e-8)
                        + cp.cfg.decay as f64 * cp.p[i] as f64)
            })
            .collect();
        rows.push(json!({"block":b.name,"expanded_shape":[b.rows,b.cols],"independent_count":ids.len(),
            "weights":stats::matrix(&values,b.rows,b.cols),"initial_rms":stats::norm(&initial)/(initial.len() as f64).sqrt(),
            "relative_displacement":stats::norm(&displacement)/stats::norm(&initial),"cosine_with_initial":stats::cosine(&values,&initial),
            "first_moment":stats::summary(&first),"second_moment":stats::summary(&second),
            "historical_momentum_plus_decay_step_over_weight_norm":stats::norm(&update)/stats::norm(&weights)}));
    }
    let normrows:Vec<_>=cp.p[..cp.cfg.vocab*cp.cfg.width].chunks_exact(cp.cfg.width).enumerate()
        .map(|(i,row)|json!({"token":i,"norm":row.iter().map(|&x|(x as f64).powi(2)).sum::<f64>().sqrt()})).collect();
    json!({"arm":cp.protocol["arm"],"seed":cp.protocol["seed"],"status":cp.result["status"],
        "checkpoint_sha256":cp.checkpoint["state_sha256"],"finite_and_complex_constraints":"passed",
        "epochs":cp.result["epochs"],"history":history_report(cp),"blocks":rows,
        "embedding_token_norms":normrows,"embedding_fourier":fourier_embedding(cp)})
}
fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args
        .get(1)
        .is_some_and(|s| s == "no-xsa-stress" || s == "no-xsa-audit")
    {
        require(
            args.len() == 5,
            "usage: no-xsa-stress|no-xsa-audit PARENT_ROOT PLAIN_XSA_ROOT OUTPUT_ROOT",
        )?;
        return if args[1] == "no-xsa-stress" {
            no_xsa::stress(
                Path::new(&args[2]),
                Path::new(&args[3]),
                Path::new(&args[4]),
            )
        } else {
            require(
                read(&Path::new(&args[4]).join("protocol.json"))?["attention"] == "ordinary_causal",
                "not an ordinary-attention run",
            )?;
            plain_ffn::audit(
                Path::new(&args[2]),
                Path::new(&args[3]),
                Path::new(&args[4]),
            )
        };
    }
    if args.get(1).is_some_and(|s| s == "no-xsa-train") {
        require(
            args.len() == 6,
            "usage: no-xsa-train PARENT_ROOT PLAIN_XSA_ROOT STRESS_ROOT NEW_OUTPUT_ROOT",
        )?;
        return plain_ffn::train_variant(
            Path::new(&args[2]),
            Path::new(&args[3]),
            Path::new(&args[4]),
            Path::new(&args[5]),
            true,
        );
    }
    if args.get(1).is_some_and(|s| s == "plain-ffn-trace") {
        require(
            args.len() == 3 && !Path::new(&args[2]).exists(),
            "usage: plain-ffn-trace NEW_OUTPUT.json",
        )?;
        return save(
            Path::new(&args[2]),
            &gradient_stress::trace_head_width_128_product(true, true)?,
        );
    }
    if args
        .get(1)
        .is_some_and(|s| s == "plain-ffn-stress" || s == "plain-ffn-audit")
    {
        require(
            args.len() == 5,
            "usage: complex-weight-diagnostic plain-ffn-stress|plain-ffn-audit PARENT_ROOT RADIAL_ROOT OUTPUT_ROOT",
        )?;
        return if args[1] == "plain-ffn-stress" {
            plain_ffn::stress(
                Path::new(&args[2]),
                Path::new(&args[3]),
                Path::new(&args[4]),
            )
        } else {
            plain_ffn::audit(
                Path::new(&args[2]),
                Path::new(&args[3]),
                Path::new(&args[4]),
            )
        };
    }
    if args.get(1).is_some_and(|s| s == "plain-ffn-train") {
        require(
            args.len() == 6,
            "usage: complex-weight-diagnostic plain-ffn-train PARENT_ROOT RADIAL_ROOT STRESS_ROOT NEW_OUTPUT_ROOT",
        )?;
        return plain_ffn::train(
            Path::new(&args[2]),
            Path::new(&args[3]),
            Path::new(&args[4]),
            Path::new(&args[5]),
        );
    }
    if args
        .get(1)
        .is_some_and(|s| s == "momentum-continue" || s == "momentum-audit")
    {
        require(
            args.len() == 6,
            "usage: complex-weight-diagnostic momentum-continue|momentum-audit PARENT_ROOT CORRECTED_ROOT ONE_STEP_ROOT OUTPUT_ROOT",
        )?;
        return if args[1] == "momentum-continue" {
            momentum_continuation::run(
                Path::new(&args[2]),
                Path::new(&args[3]),
                Path::new(&args[4]),
                Path::new(&args[5]),
            )
        } else {
            momentum_continuation::audit(
                Path::new(&args[2]),
                Path::new(&args[3]),
                Path::new(&args[4]),
                Path::new(&args[5]),
            )
        };
    }
    if args.get(1).is_some_and(|s| s == "next-update") {
        require(
            args.len() == 5,
            "usage: complex-weight-diagnostic next-update PARENT_ROOT CORRECTED_RUN_ROOT NEW_OUTPUT_DIR",
        )?;
        return next_update::run(
            Path::new(&args[2]),
            Path::new(&args[3]),
            Path::new(&args[4]),
        );
    }
    if args.get(1).is_some_and(|s| s == "retrain-weights") {
        require(
            args.len() == 5,
            "usage: complex-weight-diagnostic retrain-weights PARENT_ROOT RUN_ROOT NEW_OUTPUT_DIR",
        )?;
        return retrain::inspect_epochs(
            Path::new(&args[2]),
            Path::new(&args[3]),
            Path::new(&args[4]),
        );
    }
    if args
        .get(1)
        .is_some_and(|s| s == "retrain" || s == "retrain-audit")
    {
        require(
            args.len() == 4,
            "usage: complex-weight-diagnostic retrain|retrain-audit PARENT_ROOT OUTPUT_ROOT",
        )?;
        return if args[1] == "retrain" {
            retrain::run(Path::new(&args[2]), Path::new(&args[3]))
        } else {
            retrain::audit(Path::new(&args[2]), Path::new(&args[3]))
        };
    }
    if args.get(1).is_some_and(|s| s == "gradient-stress") {
        require(
            args.len() == 5,
            "usage: complex-weight-diagnostic gradient-stress PARENT_ROOT CONTINUATION_ROOT NEW_OUTPUT_DIR",
        )?;
        return gradient_stress::run(
            Path::new(&args[2]),
            Path::new(&args[3]),
            Path::new(&args[4]),
        );
    }
    if args.get(1).is_some_and(|s| s == "raw") {
        require(
            args.len() == 6,
            "usage: complex-weight-diagnostic raw weights|probe|precision|fixed-loss PARENT_ROOT CONTINUATION_ROOT NEW_OUTPUT_DIR",
        )?;
        return raw::run(
            &args[2],
            Path::new(&args[3]),
            Path::new(&args[4]),
            Path::new(&args[5]),
        );
    }
    require(
        args.len() == 4,
        "usage: complex-weight-diagnostic weights|probe ROOT OUTPUT.json | continue ROOT NEW_RUN_ROOT",
    )?;
    let root = Path::new(&args[2]);
    if args[1] == "continue" {
        return continuation::run(root, Path::new(&args[3]));
    }
    if args[1] == "audit-continuation" {
        return continuation::audit_saved(root, Path::new(&args[3]));
    }
    if args[1] == "finish-continuation" {
        return continuation::finish(root, Path::new(&args[3]));
    }
    let mut rows = Vec::new();
    for arm in ARMS {
        for seed in 0..3 {
            let cp = Checkpoint::load(root, arm, seed)?;
            let row = match args[1].as_str() {
                "weights" => weight_report(&cp),
                "probe" => probe::run(&cp, Path::new(&args[3]).parent().unwrap())?,
                _ => return Err("unknown diagnostic mode".into()),
            };
            cp.unchanged()?;
            rows.push(row);
            eprintln!("checked {arm} seed{seed}");
        }
    }
    let source_hash = hash(
        [
            include_str!("main.rs"),
            include_str!("engine.rs"),
            include_str!("stats.rs"),
            include_str!("probe.rs"),
        ]
        .concat()
        .as_bytes(),
    );
    save(
        Path::new(&args[3]),
        &json!({"case":CASE,"purpose":"read-only endpoint diagnosis; not training or candidate selection",
        "python_invoked":false,"diagnostic_source_sha256":source_hash,"source_model_sha256":serde_json::from_str::<Value>(include_str!("../../VALIDATION.json"))?["source_sha256"],
        "limitations":"Successful checkpoints stop at first99; missed checkpoints are later budget endpoints. Magnitudes are not age-matched causal evidence.","rows":rows}),
    )
}
fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            std::process::ExitCode::FAILURE
        }
    }
}
