//! Read-only checkpoint input, disposable training steps, unchanged model/PTX.
//! Native Nsight capture controls; no scientific outputs or checkpoint writes.
use crate::{Engine, Result, data, hash, layout, metrics, optimizer::Optimizer, runner};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    ffi::{c_char, c_void},
    fs,
    io::Write,
    path::Path,
    time::Instant,
};

const COHORT: &str = "runs/main-model-mhc-wsd-01";
const PTX: &str = "9b3a413469e02a32037b76d404ec5df8c59aa63f012b74020ef5b2404a4227c9";
const SOURCE: &str = "07763652ff0889d3e4b171425bf3e5f7a8490a14273f1b915f6013acf2068c58";

#[link(name = "dl")]
unsafe extern "C" {
    fn dlopen(name: *const c_char, flags: i32) -> *mut c_void;
    fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> i32;
}
struct Capture {
    library: *mut c_void,
    start: unsafe extern "C" fn() -> i32,
    stop: unsafe extern "C" fn() -> i32,
}
impl Capture {
    fn new() -> Result<Self> {
        // CUDA's stable native driver ABI. Keep the library alive for both calls.
        unsafe {
            let library = dlopen(c"libcuda.so.1".as_ptr(), 2);
            if library.is_null() {
                return Err("cannot load CUDA profiler API".into());
            }
            let start = dlsym(library, c"cuProfilerStart".as_ptr());
            let stop = dlsym(library, c"cuProfilerStop".as_ptr());
            if start.is_null() || stop.is_null() {
                dlclose(library);
                return Err("missing CUDA profiler entrypoint".into());
            }
            Ok(Self {
                library,
                start: std::mem::transmute::<*mut c_void, unsafe extern "C" fn() -> i32>(start),
                stop: std::mem::transmute::<*mut c_void, unsafe extern "C" fn() -> i32>(stop),
            })
        }
    }
}
impl Drop for Capture {
    fn drop(&mut self) {
        unsafe {
            dlclose(self.library);
        }
    }
}
fn read(path: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

pub fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 7 {
        return Err(
            "usage: mhc-profile CASE SEED main|mhc WARMUP_STEPS CAPTURE_STEPS FRESH_JSON".into(),
        );
    }
    let (name, arm) = (&args[1], &args[3]);
    let seed: u32 = args[2].parse()?;
    let warmup: usize = args[4].parse()?;
    let steps: usize = args[5].parse()?;
    let output = Path::new(&args[6]);
    if seed > 2
        || !runner::ARMS.contains(&arm.as_str())
        || warmup == 0
        || steps == 0
        || warmup + steps > 10000
        || output.exists()
    {
        return Err("invalid profile arguments or output already exists".into());
    }
    if runner::ptx_hash() != PTX {
        return Err("PTX differs from frozen mHC cohort".into());
    }
    let source_root = Path::new(COHORT).join("source");
    for (name, source) in crate::sources::FILES {
        // Only the live architecture/provenance manifest changed after selection.
        // All model, optimizer, data and original harness sources must match.
        if *name != "research/main_model/model.json"
            && fs::read(source_root.join(name))? != source.as_bytes()
        {
            return Err(format!("frozen dependency changed: {name}").into());
        }
    }
    let suite = runner::suite();
    let row = suite["tasks"]
        .as_array()
        .ok_or("missing suite tasks")?
        .iter()
        .find(|r| r["name"] == *name)
        .ok_or("unknown frozen task")?;
    let task = data::build(row)?;
    let cfg = runner::config(row, &task, arm)?;
    let budget = row["case"]["max_updates"]
        .as_u64()
        .ok_or("missing budget")? as usize;
    let batch = row["case"]["batch"].as_u64().ok_or("missing batch")? as usize;
    let dir = Path::new(COHORT)
        .join(arm)
        .join(name)
        .join(format!("seed{seed}"));
    let checkpoint = read(&dir.join("checkpoint.json"))?;
    let state_bytes = fs::read(dir.join("checkpoint.bin"))?;
    if checkpoint["state_sha256"] != hash(&state_bytes)
        || checkpoint["ptx_sha256"] != PTX
        || checkpoint["source_sha256"] != SOURCE
        || read(&dir.join("execution.json"))?["clean"] != true
    {
        return Err("checkpoint provenance/cleanliness mismatch".into());
    }
    let completed = checkpoint["completed"]
        .as_u64()
        .ok_or("missing checkpoint step")? as usize;
    if completed + warmup + steps > budget {
        return Err("profile would exceed original budget".into());
    }
    let n = layout::total_count(&cfg);
    if state_bytes.len() != 12 * n {
        return Err("checkpoint size mismatch".into());
    }
    let state: Vec<f32> = state_bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    layout::validate_state(&cfg, &state[..n], &state[n..2 * n], &state[2 * n..])?;
    let mut engine = Engine::new(cfg.clone(), &state[..n])?;
    engine.first_moment.write(&state[n..2 * n])?;
    engine.second_moment.write(&state[2 * n..])?;
    let mut optimizer = Optimizer::new(&engine, budget)?;
    let train = &task.splits["train"];
    let (order, edges) = runner::order(train.len(), batch, completed + warmup + steps, seed);
    if edges[completed] as u64
        != checkpoint["examples"]
            .as_u64()
            .ok_or("missing sample offset")?
    {
        return Err("sampler offset mismatch".into());
    }
    let mut shape_counts = BTreeMap::<String, usize>::new();
    let mut batches = Vec::new();
    for i in completed..completed + warmup + steps {
        let indices = &order[edges[i]..edges[i + 1]];
        let mut packed = metrics::Packed::default();
        packed.fill(train, indices);
        if i >= completed + warmup {
            *shape_counts
                .entry(format!("batch{}_length{}", indices.len(), packed.length))
                .or_default() += 1;
        }
        batches.push((indices.len(), packed));
    }
    let capture = Capture::new()?;
    for (i, (b, packed)) in batches.iter().take(warmup).enumerate() {
        engine.forward(&packed.tokens, &packed.labels, *b, packed.length)?;
        engine.backward(*b, packed.length)?;
        optimizer.step(&engine, completed + i)?;
    }
    engine.cuda.sync()?;
    if unsafe { (capture.start)() } != 0 {
        return Err("cuProfilerStart failed".into());
    }
    let started = Instant::now();
    for (i, (b, packed)) in batches.iter().enumerate().skip(warmup) {
        engine.forward(&packed.tokens, &packed.labels, *b, packed.length)?;
        engine.backward(*b, packed.length)?;
        optimizer.step(&engine, completed + i)?;
    }
    engine.cuda.sync()?;
    let seconds = started.elapsed().as_secs_f64();
    if unsafe { (capture.stop)() } != 0 {
        return Err("cuProfilerStop failed".into());
    }
    let p = engine.parameters.read()?;
    let m = engine.first_moment.read()?;
    let v = engine.second_moment.read()?;
    layout::validate_state(&cfg, &p, &m, &v)?;
    let last = batches.last().unwrap();
    let losses = engine.losses(last.0 * last.1.length)?;
    if losses.iter().any(|x| !x.is_finite()) {
        return Err("nonfinite profile loss".into());
    }
    let mean_loss = losses.iter().map(|&v| v as f64).sum::<f64>() / losses.len() as f64;
    let state_hash = hash(
        &p.iter()
            .chain(&m)
            .chain(&v)
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    let result = json!({
        "purpose":"profiling_only_disposable_steps_not_scientific_training",
        "case":name,"arm":arm,"seed":seed,"gpu":engine.cuda.name(),"sm":engine.cuda.sm(),
        "width":cfg.width,"heads":cfg.heads,"layers":cfg.layers,"vocab":cfg.vocab,
        "max_batch":cfg.max_batch,"max_length":cfg.max_length,"shape_counts":shape_counts,
        "checkpoint":dir,"checkpoint_sha256":checkpoint["state_sha256"],"checkpoint_step":completed,
        "original_source_sha256":SOURCE,"ptx_sha256":PTX,"ptx_matches_frozen_cohort":true,
        "shared_sources_match_frozen_cohort_except_updated_model_manifest":true,
        "profile_driver_sha256":hash(include_bytes!("profile.rs")),
        "warmup_updates":warmup,"captured_updates":steps,"captured_seconds":seconds,
        "milliseconds_per_update":1000.*seconds/steps as f64,"final_state_sha256":state_hash,
        "last_batch_mean_token_slot_loss":mean_loss,"state_and_loss_finite":true,
        "timing_scope":"warmed training forward/backward/AdamW, real next batches prepacked; includes engine H2D copies and final synchronization, excludes evaluation, dataset/sampler/setup/packing and all disk writes",
        "scientific_checkpoint_written":false
    });
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    file.write_all(serde_json::to_string_pretty(&result)?.as_bytes())?;
    file.write_all(b"\n")?;
    println!("{}", serde_json::to_string(&result)?);
    Ok(())
}
