//! Standalone Rust training; all GPU model operations are Kernite PTX.
mod attention;
mod cohort;
mod complex;
#[cfg(test)]
mod complex_tests;
mod continuation;
#[cfg(test)]
mod continuation_tests;
mod engine;
mod gemm;
mod layout;
mod long_audit;
mod ops;
mod prepare;
mod random;
#[cfg(test)]
mod reference;
#[cfg(test)]
mod tests;

use engine::{Config, Engine, Result};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
    time::Instant,
};

const FLUSH_EPOCHS: usize = 500;
const UPDATES_PER_EPOCH: usize = 19;
const FLUSH_UPDATES: usize = FLUSH_EPOCHS * UPDATES_PER_EPOCH;
const HISTORY_BUFFER_BYTES: usize = 1 << 20;

fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|x| format!("{x:02x}"))
        .collect()
}
fn source_hash() -> String {
    hash(
        concat!(
            include_str!("main.rs"),
            include_str!("engine.rs"),
            include_str!("ops.rs"),
            include_str!("attention.rs"),
            include_str!("gemm.rs"),
            include_str!("complex.rs"),
            include_str!("layout.rs"),
            include_str!("prepare.rs"),
            include_str!("random.rs"),
            include_str!("cohort.rs"),
            include_str!("continuation.rs"),
            include_str!("long_audit.rs"),
            include_str!("../Cargo.toml"),
            include_str!("../Cargo.lock")
        )
        .as_bytes(),
    )
}
fn ptx_hash() -> String {
    hash(
        [
            ops::kernels::PTX,
            attention::kernels::PTX,
            gemm::kernels::PTX,
            complex::kernels::PTX,
        ]
        .concat()
        .as_bytes(),
    )
}
fn integer(v: &Value, key: &str) -> Result<usize> {
    let n = v[key]
        .as_u64()
        .ok_or_else(|| format!("invalid integer: {key}"))?;
    Ok(usize::try_from(n)?)
}
fn number(v: &Value, key: &str) -> Result<f64> {
    let n = v[key]
        .as_f64()
        .ok_or_else(|| format!("invalid number: {key}"))?;
    if !n.is_finite() {
        return Err(format!("nonfinite {key}").into());
    }
    Ok(n)
}
fn words(path: &Path) -> Result<Vec<u32>> {
    let bytes = fs::read(path)?;
    if bytes.len() % 4 != 0 {
        return Err("truncated binary array".into());
    }
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|p| u32::from_le_bytes(*p))
        .collect())
}
fn floats(path: &Path) -> Result<Vec<f32>> {
    Ok(words(path)?.into_iter().map(f32::from_bits).collect())
}
fn write_floats(path: &Path, data: &[f32]) -> Result<()> {
    let mut f = File::create(path)?;
    let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
    f.write_all(&bytes)?;
    Ok(())
}
fn durable(path: &Path, bytes: &[u8]) -> Result<()> {
    let temp = path.with_extension("pending");
    let mut f = File::create(&temp)?;
    f.write_all(bytes)?;
    f.sync_all()?;
    fs::rename(&temp, path)?;
    File::open(path.parent().unwrap())?.sync_all()?;
    Ok(())
}
fn record(path: &Path, value: &Value) -> Result<()> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    durable(path, &bytes)
}

struct Input {
    manifest: Value,
    digest: String,
    boundaries: Vec<u32>,
    engine: Engine,
}
impl Input {
    fn load(path: &Path) -> Result<Self> {
        let bytes = fs::read(path.join("manifest.json"))?;
        let m: Value = serde_json::from_slice(&bytes)?;
        for (name, spec) in m["files"].as_object().ok_or("missing input files")? {
            if name.contains('/') || name.contains("..") {
                return Err("invalid input filename".into());
            }
            let b = fs::read(path.join(name))?;
            if hash(&b) != spec["sha256"].as_str().ok_or("missing input hash")?
                || b.len() != integer(spec, "bytes")?
            {
                return Err(format!("input changed: {name}").into());
            }
        }
        let expected = [
            "parameters.f32",
            "rope.f32",
            "train.u32",
            "validation.u32",
            "order.u32",
            "boundaries.u32",
        ];
        if expected.iter().any(|k| m["files"][k].is_null()) {
            return Err("incomplete input manifest".into());
        }
        let w = integer(&m, "width")?;
        let l = integer(&m, "layers")?;
        let normalized = m["normalized"].as_bool().ok_or("missing norm flag")?;
        let config = Config {
            width: w,
            layers: l,
            normalized,
            ssmax: m["ssmax"].as_bool().ok_or("missing SSMax flag")?,
            gain: number(&m, "attention_gain")? as f32,
            lr: number(&m, "learning_rate")?,
            decay: number(&m, "weight_decay")? as f32,
            warmup: integer(&m, "warmup")?,
            architecture: m["architecture"]
                .as_str()
                .ok_or("missing architecture")?
                .to_owned(),
            phase_optimizer: m["phase_optimizer"].as_bool().ok_or("missing optimizer")?,
        };
        if integer(&m, "version")? != 1
            || integer(&m, "vocab")? != 97
            || integer(&m, "heads")? * 32 != w
            || integer(&m, "batch")? != 256
            || integer(&m, "observe")? != 19
            || number(&m, "target")? != 0.99
            || ![128, 256].contains(&w)
            || ![1, 2].contains(&l)
            || config.lr <= 0.0
            || config.decay < 0.0
            || config.gain <= 0.0
        {
            return Err("unsupported protocol".into());
        }
        let parameters = floats(&path.join("parameters.f32"))?;
        let rope = floats(&path.join("rope.f32"))?;
        if parameters.iter().chain(&rope).any(|x| !x.is_finite()) {
            return Err("nonfinite input".into());
        }
        if !["real", "complex-diagonal", "complex-product"].contains(&config.architecture.as_str())
            || (config.architecture == "real" && config.phase_optimizer)
            || !layout::validate(&parameters, w, l, &config.architecture)
        {
            return Err("invalid complex architecture, parameter tying or optimizer".into());
        }
        let hidden = layout::hidden(w, &config.architecture);
        let mut expected_params = vec![("token_embedding_table.weight".to_string(), vec![97, w])];
        for layer in 0..l {
            for (name, shape) in [
                ("sa.c_attn.weight", vec![3 * w, w]),
                ("sa.c_proj.weight", vec![w, w]),
                ("ffwd.c_fc.weight", vec![2 * hidden, w]),
                ("ffwd.c_proj.weight", vec![w, hidden]),
            ] {
                expected_params.push((format!("blocks.{layer}.{name}"), shape));
            }
        }
        let rows = m["parameters"]
            .as_array()
            .ok_or("missing parameter layout")?;
        let mut offset = 0;
        if rows.len() != expected_params.len() {
            return Err("parameter count changed".into());
        }
        for (row, (name, shape)) in rows.iter().zip(expected_params) {
            let count: usize = shape.iter().product();
            if row["name"] != name
                || row["shape"] != json!(shape)
                || integer(row, "offset")? != offset
                || integer(row, "count")? != count
            {
                return Err("parameter order or shape changed".into());
            }
            offset += count;
        }
        if parameters.len() != offset || integer(&m, "parameter_count")? != offset {
            return Err("parameter length changed".into());
        }
        let train = words(&path.join("train.u32"))?;
        let validation = words(&path.join("validation.u32"))?;
        if train.len() != 4704 * 3
            || validation.len() != 4705 * 3
            || train.iter().chain(&validation).any(|&x| x >= 97)
            || integer(&m, "train_examples")? != 4704
            || integer(&m, "validation_examples")? != 4705
        {
            return Err("invalid two-token dataset".into());
        }
        let order = words(&path.join("order.u32"))?;
        let boundaries = words(&path.join("boundaries.u32"))?;
        if boundaries.len() != integer(&m, "updates")? + 1
            || boundaries.first() != Some(&0)
            || boundaries.last() != Some(&(order.len() as u32))
            || order.iter().any(|&x| x >= 4704)
        {
            return Err("invalid batch order".into());
        }
        for (step, range) in boundaries.windows(2).enumerate() {
            if range[1].checked_sub(range[0]) != Some(if step % 19 == 18 { 96 } else { 256 }) {
                return Err("batch boundary changed".into());
            }
        }
        let engine = Engine::new(config, &parameters, &rope, &train, &validation, &order)?;
        Ok(Self {
            manifest: m,
            digest: hash(&bytes),
            boundaries,
            engine,
        })
    }
    fn step(&self, step: usize) -> Result<usize> {
        let start = self.boundaries[step] as usize;
        let b = (self.boundaries[step + 1] - self.boundaries[step]) as usize;
        self.engine.forward(false, start, b, true)?;
        self.engine.backward(b)?;
        Ok(b)
    }
}

fn checkpoint(input: &Input, dir: &Path, completed: usize) -> Result<()> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(completed as u64).to_le_bytes());
    for buffer in [
        &input.engine.parameters,
        &input.engine.first_moment,
        &input.engine.second_moment,
    ] {
        let values = buffer.read()?;
        if values.iter().any(|x| !x.is_finite()) {
            return Err("nonfinite checkpoint state".into());
        }
        bytes.extend(values.iter().flat_map(|x| x.to_le_bytes()));
    }
    durable(&dir.join("checkpoint.bin"), &bytes)?;
    record(
        &dir.join("checkpoint.json"),
        &json!({"completed":completed,"sha256":hash(&bytes),"history_sha256":hash(&fs::read(dir.join("history.jsonl"))?),"input_sha256":input.digest,"source_sha256":source_hash(),"ptx_sha256":ptx_hash(),"optimizer":if input.engine.config.phase_optimizer {"Kernite phase-coupled AdamW, mean pair-square"}else{"Kernite coordinate AdamW"},"sampler_examples":input.boundaries[completed]}),
    )
}

fn restore(input: &mut Input, dir: &Path) -> Result<usize> {
    restore_from_source(input, dir, &source_hash())
}

// Ordinary resumes remain exact-source-only. The continuation module permits
// one pinned pilot source after separately checking unchanged kernels and host math.
fn restore_from_source(input: &mut Input, dir: &Path, expected_source: &str) -> Result<usize> {
    let saved: Value = serde_json::from_slice(&fs::read(dir.join("checkpoint.json"))?)?;
    let bytes = fs::read(dir.join("checkpoint.bin"))?;
    if saved["sha256"] != hash(&bytes)
        || saved["history_sha256"] != hash(&fs::read(dir.join("history.jsonl"))?)
        || saved["input_sha256"] != input.digest
        || saved["source_sha256"] != expected_source
        || saved["ptx_sha256"] != ptx_hash()
    {
        return Err("checkpoint, input or executable source changed".into());
    }
    let completed = integer(&saved, "completed")?;
    let n = input.engine.parameters.len();
    if bytes.len() != 8 + 12 * n
        || u64::from_le_bytes(bytes[..8].try_into()?) != completed as u64
        || completed >= input.boundaries.len()
        || integer(&saved, "sampler_examples")? != input.boundaries[completed] as usize
    {
        return Err("invalid checkpoint layout".into());
    }
    let journal: Value = serde_json::from_slice(&fs::read(dir.join("execution.json"))?)?;
    if integer(&journal, "completed")? != completed
        || integer(&journal, "possible_end")? != completed
    {
        return Err("checkpoint is behind issued work; preserved execution.json bounds possible replay; automatic resume refused".into());
    }
    let values: Vec<f32> = bytes[8..]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|p| f32::from_le_bytes(*p))
        .collect();
    if values.iter().any(|x| !x.is_finite()) {
        return Err("nonfinite restored state".into());
    }
    input.engine.parameters.write(&values[..n])?;
    input.engine.first_moment.write(&values[n..2 * n])?;
    input.engine.second_moment.write(&values[2 * n..])?;
    if !layout::validate(
        &values[..n],
        input.engine.config.width,
        input.engine.config.layers,
        &input.engine.config.architecture,
    ) || values[2 * n..].iter().any(|&v| v < 0.0)
    {
        return Err("invalid restored complex tying or second moment".into());
    }
    Ok(completed)
}

fn probe(input: &Input, dir: &Path, steps: usize) -> Result<()> {
    let started = Instant::now();
    let mut rows = Vec::new();
    for step in 0..steps {
        let batch = input.step(step)?;
        if step == 0 {
            input.engine.debug(dir, batch)?;
        }
        if [0, 1, 18, 19, 49, 50, 51].contains(&step) || step + 1 == steps {
            write_floats(
                &dir.join(format!("step{step}-parameters.f32")),
                &input.engine.parameters.read()?,
            )?;
            write_floats(
                &dir.join(format!("step{step}-logits.f32")),
                &input.engine.logits.read()?[..batch * 194],
            )?;
            write_floats(
                &dir.join(format!("step{step}-gradients.f32")),
                &input.engine.gradients.read()?,
            )?;
            let (loss, hits) = input.engine.metrics(batch)?;
            rows.push(json!({"step":step,"batch":batch,"loss":loss/batch as f64,"hits":hits}));
        }
        input.engine.update(step)?;
    }
    input.engine.cuda.sync()?;
    write_floats(
        &dir.join("parameters.f32"),
        &input.engine.parameters.read()?,
    )?;
    write_floats(
        &dir.join("first_moment.f32"),
        &input.engine.first_moment.read()?,
    )?;
    write_floats(
        &dir.join("second_moment.f32"),
        &input.engine.second_moment.read()?,
    )?;
    record(
        &dir.join("probe.json"),
        &json!({"steps":steps,"rows":rows,"seconds":started.elapsed().as_secs_f64(),"input_sha256":input.digest,"source_sha256":source_hash(),"ptx_sha256":ptx_hash(),"gpu":input.engine.cuda.name()}),
    )
}

fn train(input: &mut Input, dir: &Path, limit: usize, resume: bool) -> Result<()> {
    let started = Instant::now();
    let initial_prior = input.manifest["prior_seconds"].as_f64().unwrap_or(0.0);
    let prior_seconds = if resume && dir.join("segment.json").exists() {
        let previous: Value = serde_json::from_slice(&fs::read(dir.join("segment.json"))?)?;
        number(&previous, "total_seconds")?.max(initial_prior)
    } else {
        initial_prior
    };
    let seconds_cap = number(&input.manifest, "seconds_cap")? - (prior_seconds - initial_prior);
    let mut completed = if resume { restore(input, dir)? } else { 0 };
    if !limit.is_multiple_of(19) && limit + 1 != input.boundaries.len() {
        return Err("a requested stop must be an observation boundary or the full budget".into());
    }
    if limit < completed {
        return Err("stop precedes restored checkpoint".into());
    }
    let mut train_seconds = 0.0;
    let mut evaluation_seconds = 0.0;
    let mut io_seconds = 0.0;
    let mut history = Vec::with_capacity(HISTORY_BUFFER_BYTES);
    let mut initial_boundary = true;
    let mut first99 = None;
    let mut reuse_observation = resume;
    loop {
        let at = Instant::now();
        let (training, validation) = if reuse_observation {
            let text = fs::read_to_string(dir.join("history.jsonl"))?;
            let row: Value =
                serde_json::from_str(text.lines().last().ok_or("empty resumed history")?)?;
            if integer(&row, "step")? != completed {
                return Err("checkpoint and observation disagree".into());
            }
            (row["train"].clone(), row["validation"].clone())
        } else {
            (input.engine.evaluate(false)?, input.engine.evaluate(true)?)
        };
        evaluation_seconds += at.elapsed().as_secs_f64();
        if validation["exact"].as_u64().ok_or("missing exact count")? >= 4658 && first99.is_none() {
            first99 = Some(completed);
        }
        let row = json!({"step":completed,"examples_seen":input.boundaries[completed],"epochs":input.boundaries[completed] as f64/4704.0,"seconds":prior_seconds+started.elapsed().as_secs_f64(),"train":training,"validation":validation,"first99_updates":first99});
        if !reuse_observation {
            writeln!(history, "{row}")?;
        }
        reuse_observation = false;
        let timed_out = started.elapsed().as_secs_f64() >= seconds_cap;
        let requested_stop = dir.join("STOP").exists();
        let stopping = completed == limit || first99.is_some() || timed_out || requested_stop;
        let at = Instant::now();
        if initial_boundary || completed.is_multiple_of(FLUSH_UPDATES) || stopping {
            // The intent is durable before any update in the next window is issued.
            // Intermediate observations stay buffered. A crash can therefore leave at
            // most FLUSH_UPDATES possible updates, explicitly bounded by execution.json.
            let mut file = fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(dir.join("history.jsonl"))?;
            file.write_all(&history)?;
            file.sync_all()?;
            history.clear();
            checkpoint(input, dir, completed)?;
            let reserved_end = if stopping {
                completed
            } else {
                ((completed / FLUSH_UPDATES + 1) * FLUSH_UPDATES).min(limit)
            };
            record(
                &dir.join("execution.json"),
                &json!({
                    "version":2,"input_sha256":input.digest,"completed":completed,
                    "possible_end":reserved_end,"known_examples":input.boundaries[completed],
                    "possible_examples":input.boundaries[reserved_end],"flush_epochs":FLUSH_EPOCHS,
                    "source_sha256":source_hash()
                }),
            )?;
            println!("{row}");
            initial_boundary = false;
        }
        io_seconds += at.elapsed().as_secs_f64();
        if stopping {
            break;
        }
        let end = (completed + UPDATES_PER_EPOCH).min(limit);
        let at = Instant::now();
        for step in completed..end {
            input.step(step)?;
            input.engine.update(step)?;
        }
        input.engine.cuda.sync()?;
        train_seconds += at.elapsed().as_secs_f64();
        completed = end;
    }
    let result = json!({"completed":completed,"first99_updates":first99,"status":if first99.is_some(){"passed"}else if completed+1==input.boundaries.len(){"budget_exhausted"}else if started.elapsed().as_secs_f64() >= seconds_cap {"wall_time_cap"}else{"stopped_at_declared_boundary"},"epochs":input.boundaries[completed] as f64/4704.0,"input_sha256":input.digest,"source_sha256":source_hash(),"ptx_sha256":ptx_hash(),"gpu":input.engine.cuda.name(),"train_seconds":train_seconds,"evaluation_seconds":evaluation_seconds,"io_seconds":io_seconds,"segment_seconds":started.elapsed().as_secs_f64(),"total_seconds":prior_seconds+started.elapsed().as_secs_f64(),"prior_seconds":prior_seconds,"resumed":resume,"flush_epochs":FLUSH_EPOCHS,"history_in_memory":true,"manifest":input.manifest});
    record(
        &dir.join(
            if completed + 1 == input.boundaries.len() || first99.is_some() {
                "result.json"
            } else {
                "segment.json"
            },
        ),
        &result,
    )?;
    Ok(())
}

fn benchmark(input: &Input, dir: &Path, steps: usize) -> Result<()> {
    let begin = Instant::now();
    for step in 0..steps {
        input.step(step)?;
        input.engine.update(step)?;
    }
    input.engine.cuda.sync()?;
    let training = begin.elapsed().as_secs_f64();
    let begin = Instant::now();
    let train = input.engine.evaluate(false)?;
    let validation = input.engine.evaluate(true)?;
    let scoring = begin.elapsed().as_secs_f64();
    record(
        &dir.join("benchmark.json"),
        &json!({"updates":steps,"training_seconds":training,"milliseconds_per_update":1000.0*training/steps as f64,"full_train_and_validation_seconds":scoring,"train":train,"validation":validation,"input_sha256":input.digest,"source_sha256":source_hash(),"ptx_sha256":ptx_hash(),"gpu":input.engine.cuda.name(),"scope":"timing diagnostic, no scientific first99 observation claim"}),
    )
}

fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() == 4 && args[1] == "long" {
        return continuation::run(Path::new(&args[2]), Path::new(&args[3]));
    }
    if args.len() == 3 && args[1] == "audit-long" {
        return continuation::audit(Path::new(&args[2]));
    }
    if args.len() == 3 && args[1] == "pilot" {
        return cohort::run(Path::new(&args[2]));
    }
    if args.len() == 3 && args[1] == "audit" {
        return cohort::audit(Path::new(&args[2]));
    }
    if args.len() == 7 && args[1] == "prepare" {
        return prepare::write(
            Path::new(&args[2]),
            &args[3],
            &args[4],
            args[5].parse()?,
            args[6].parse()?,
        );
    }
    if args.len() != 5 || !["probe", "train", "resume", "bench"].contains(&args[1].as_str()) {
        return Err(
            "usage: astra-kernite probe|train|resume|bench INPUT_DIRECTORY OUTPUT_DIRECTORY UPDATES"
                .into(),
        );
    }
    let limit: usize = args[4].parse()?;
    let dir = PathBuf::from(&args[3]);
    let resume = args[1] == "resume";
    if !resume {
        fs::create_dir(&dir)?;
    } else if dir.join("result.json").exists() {
        return Err("completed runs cannot be resumed".into());
    }
    let start = Instant::now();
    let mut input = Input::load(Path::new(&args[2]))?;
    if limit == 0 || limit >= input.boundaries.len() {
        return Err("updates outside exported budget".into());
    }
    eprintln!(
        "native setup {:.3}s; GPU {}; {} parameters",
        start.elapsed().as_secs_f64(),
        input.engine.cuda.name(),
        input.engine.parameters.len()
    );
    if args[1] == "bench" {
        benchmark(&input, &dir, limit)
    } else if args[1] == "probe" {
        probe(&input, &dir, limit)
    } else {
        train(&mut input, &dir, limit, resume)
    }
}

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("astra-kernite: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}
