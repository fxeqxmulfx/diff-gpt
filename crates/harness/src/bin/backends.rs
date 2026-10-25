//! The bench of burn's backends at the sizes of the 5M model: a matmul, a
//! softmax, a cross-entropy and a causal attention, each forward on the
//! plain device, as forecasting runs it, and forward with its backward on
//! the Autodiff device, as a training step runs it.
//!
//! `backends <device> [--ops matmul,softmax,cross_entropy,attention]
//! [--passes forward,forward+backward] [--rows N] [--warmup N]
//! [--iterations N] [--warmup-seconds N] [--memory]`, the device one of
//! `ndarray`, `flex`, `libtorch`, `libtorch-cuda` and `cuda`.
//!
//! The ops, on `--rows` rows (12, the 5M model's batch) of its block of 1412
//! tokens: the tokens by its width of 256 times the width by the 1366 of the
//! first linear of its `SwiGLU`; a softmax on the last axis of scores of
//! [rows, 4, 1412, 1412]; the cross-entropy of logits of [tokens, 256]; and
//! `burn::tensor::module::attention`, causal, on q, k and v of
//! [rows, 4, 1412, 64]. The passes:
//! - `forward`: the op on the plain device, on tensors that track no
//!   gradient;
//! - `forward+backward`: the op and the backward of its sum on the Autodiff
//!   device, the gradients of its inputs read;
//! - `checkpointed`: the same on an Autodiff device with burn's balanced
//!   gradient checkpointing, which recomputes the memory-bound ops in the
//!   backward;
//! - `autodiff-forward` and `checkpointed-forward`: the op alone on the
//!   Autodiff device, without and with the checkpointing, on tensors that
//!   track gradients, with no backward: `--memory` then shows what each
//!   forward keeps for a backward.
//!
//! The first two are the default.
//!
//! It prints a line per op and pass: the device, the rows, the op, the pass,
//! and the median, the least and the most of the iterations' times in
//! milliseconds, each timed from the op's start to the device's sync. On
//! `cuda` it then reads the round's outputs, the gradients in a backward
//! pass: `CubeCL`'s sync reports no failed launch, such as one that found no
//! memory, which shows only on the read of an output it left unwritten. The
//! other backends fail at the op itself, and it reads none of their outputs:
//! with the reads, `LibTorch`'s matmul forward on the CPU took 27 to 30 ms a
//! round, and 15.7 without them. Before the rounds, `--warmup-seconds` of
//! plain matmuls (10 on a GPU, none on the CPU) bring a GPU out of idle, and
//! it prints the GPU's state after them as `nvidia-smi` gives it. With
//! `--memory`, it prints a line before the rounds of each pass and after
//! each round, its outputs dropped: the GPU's used memory as `nvidia-smi`
//! gives it, and the bytes that burn's allocator holds in use and reserved
//! where the backend reports them (`CubeCL`'s does, `LibTorch`'s does not),
//! all in MiB. The command that starts it sets the threads (`numactl`,
//! `RAYON_NUM_THREADS`, `OMP_NUM_THREADS`); the bench sets none.
#![forbid(unsafe_code)]

use std::process::{Command, ExitCode};
use std::time::Instant;

use burn::nn::loss::CrossEntropyLossConfig;
use burn::tensor::activation::softmax;
use burn::tensor::module::attention;
use burn::tensor::ops::AttentionModuleOptions;
use burn::tensor::{Device, Distribution, Int, Tensor, TensorData};

/// The 5M model's batch: 12 rows of its block of 1412 tokens.
const ROWS: usize = 12;
const BLOCK: usize = 1412;
/// Its width, heads and vocabulary, and the width of the first linear of
/// its `SwiGLU` (2 × 683).
const WIDTH: usize = 256;
const HEADS: usize = 4;
const VOCAB: usize = 256;
const HIDDEN: usize = 1366;
/// The seconds of matmuls that bring a GPU out of idle by default.
const GPU_WARMUP_SECONDS: u64 = 10;

const USAGE: &str = "usage: backends <ndarray|flex|libtorch|libtorch-cuda|cuda> \
                     [--ops matmul,softmax,cross_entropy,attention] \
                     [--passes forward,forward+backward,checkpointed,autodiff-forward,\
                     checkpointed-forward] \
                     [--rows N] [--warmup N] [--iterations N] [--warmup-seconds N] [--memory]";

#[derive(Clone, Copy, PartialEq, Eq)]
enum Op {
    Matmul,
    Softmax,
    CrossEntropy,
    Attention,
}

impl Op {
    const ALL: [Self; 4] = [
        Self::Matmul,
        Self::Softmax,
        Self::CrossEntropy,
        Self::Attention,
    ];

    const fn name(self) -> &'static str {
        match self {
            Self::Matmul => "matmul",
            Self::Softmax => "softmax",
            Self::CrossEntropy => "cross_entropy",
            Self::Attention => "attention",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Pass {
    Forward,
    ForwardBackward,
    Checkpointed,
    AutodiffForward,
    CheckpointedForward,
}

impl Pass {
    const ALL: [Self; 5] = [
        Self::Forward,
        Self::ForwardBackward,
        Self::Checkpointed,
        Self::AutodiffForward,
        Self::CheckpointedForward,
    ];

    const fn name(self) -> &'static str {
        match self {
            Self::Forward => "forward",
            Self::ForwardBackward => "forward+backward",
            Self::Checkpointed => "checkpointed",
            Self::AutodiffForward => "autodiff-forward",
            Self::CheckpointedForward => "checkpointed-forward",
        }
    }

    /// The device the pass runs on, made from the plain one.
    fn device(self, plain: &Device) -> Device {
        match self {
            Self::Forward => plain.clone(),
            Self::ForwardBackward | Self::AutodiffForward => plain.clone().autodiff(),
            Self::Checkpointed | Self::CheckpointedForward => {
                plain.clone().autodiff().gradient_checkpointing()
            }
        }
    }

    /// Whether the pass runs the backward.
    const fn backward(self) -> bool {
        matches!(self, Self::ForwardBackward | Self::Checkpointed)
    }
}

struct Settings {
    device: String,
    ops: Vec<Op>,
    passes: Vec<Pass>,
    rows: usize,
    warmup: usize,
    iterations: usize,
    warmup_seconds: u64,
    memory: bool,
    read: bool,
}

fn main() -> ExitCode {
    match settings(std::env::args().skip(1)).and_then(|settings| {
        let device = device(&settings.device)?;
        if settings.warmup_seconds > 0 {
            warm_up(&device, settings.warmup_seconds);
        }
        for &op in &settings.ops {
            for &pass in &settings.passes {
                bench(&settings, &pass.device(&device), op, pass);
            }
        }
        Ok(())
    }) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::from(2)
        }
    }
}

fn settings(mut args: impl Iterator<Item = String>) -> Result<Settings, String> {
    let device = args.next().ok_or(USAGE)?;
    let gpu = matches!(device.as_str(), "libtorch-cuda" | "cuda");
    let read = device == "cuda";
    let mut settings = Settings {
        device,
        ops: Op::ALL.to_vec(),
        passes: vec![Pass::Forward, Pass::ForwardBackward],
        rows: ROWS,
        warmup: 3,
        iterations: 11,
        warmup_seconds: if gpu { GPU_WARMUP_SECONDS } else { 0 },
        memory: false,
        read,
    };
    while let Some(flag) = args.next() {
        if flag == "--memory" {
            settings.memory = true;
            continue;
        }
        let value = args
            .next()
            .ok_or_else(|| format!("{flag} takes a value\n{USAGE}"))?;
        let count = || {
            value
                .parse::<usize>()
                .ok()
                .filter(|count| *count > 0)
                .ok_or_else(|| format!("{flag} takes a positive count, not {value}"))
        };
        match flag.as_str() {
            "--ops" => settings.ops = names(&value, &Op::ALL, Op::name)?,
            "--passes" => settings.passes = names(&value, &Pass::ALL, Pass::name)?,
            "--rows" => settings.rows = count()?,
            "--warmup" => settings.warmup = count()?,
            "--iterations" => settings.iterations = count()?,
            "--warmup-seconds" => {
                settings.warmup_seconds = value
                    .parse()
                    .map_err(|_| format!("{flag} takes a count of seconds, not {value}"))?;
            }
            _ => return Err(format!("no flag {flag}\n{USAGE}")),
        }
    }
    Ok(settings)
}

/// The items of `all` that the comma-separated `value` names.
fn names<T: Copy>(value: &str, all: &[T], name: fn(T) -> &'static str) -> Result<Vec<T>, String> {
    value
        .split(',')
        .map(|wanted| {
            all.iter()
                .copied()
                .find(|item| name(*item) == wanted)
                .ok_or_else(|| format!("no op or pass {wanted}\n{USAGE}"))
        })
        .collect()
}

#[expect(
    deprecated,
    reason = "NdArray and LibTorch are deprecated in burn 0.22; the bench measures them"
)]
fn device(name: &str) -> Result<Device, String> {
    Ok(match name {
        "ndarray" => Device::ndarray(),
        "flex" => Device::flex(),
        "libtorch" => Device::libtorch(),
        "libtorch-cuda" => Device::libtorch_cuda(0),
        "cuda" => Device::cuda(0),
        _ => return Err(format!("no device {name}\n{USAGE}")),
    })
}

/// Run plain matmuls for `seconds` to bring a GPU out of idle, and print the
/// GPU's state after them.
fn warm_up(device: &Device, seconds: u64) {
    let x = Tensor::<2>::random(
        [ROWS * BLOCK, WIDTH],
        Distribution::Normal(0.0, 1.0),
        device,
    );
    let w = Tensor::<2>::random([WIDTH, HIDDEN], Distribution::Normal(0.0, 0.02), device);
    let start = Instant::now();
    let mut matmuls = 0_u64;
    while start.elapsed().as_secs() < seconds {
        let y = x.clone().matmul(w.clone());
        device.sync().expect("the device runs the warm-up");
        drop(y);
        matmuls += 1;
    }
    eprintln!(
        "warm-up: {matmuls} matmuls in {:.1} s; GPU 0 after it (pstate, SM MHz, memory MHz, \
         degrees C, W): {}",
        start.elapsed().as_secs_f64(),
        nvidia_smi("pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw")
    );
}

/// Time `op` in `pass` on `device`, the pass's device.
fn bench(settings: &Settings, device: &Device, op: Op, pass: Pass) {
    let tokens = settings.rows * BLOCK;
    match op {
        Op::Matmul => {
            let inputs = [
                input([tokens, WIDTH], 1.0, device, pass),
                input([WIDTH, HIDDEN], 0.02, device, pass),
            ];
            measure(settings, device, op, pass, &inputs, || {
                inputs[0].clone().matmul(inputs[1].clone())
            });
        }
        Op::Softmax => {
            let inputs = [input(
                [settings.rows, HEADS, BLOCK, BLOCK],
                1.0,
                device,
                pass,
            )];
            measure(settings, device, op, pass, &inputs, || {
                softmax(inputs[0].clone(), 3)
            });
        }
        Op::CrossEntropy => {
            let inputs = [input([tokens, VOCAB], 1.0, device, pass)];
            let targets =
                Tensor::<1, Int>::from_data(TensorData::new(targets(tokens), [tokens]), device);
            let loss = CrossEntropyLossConfig::new().init(device);
            measure(settings, device, op, pass, &inputs, || {
                loss.forward(inputs[0].clone(), targets.clone())
            });
        }
        Op::Attention => {
            let shape = [settings.rows, HEADS, BLOCK, WIDTH / HEADS];
            let inputs = [0, 1, 2].map(|_| input(shape, 1.0, device, pass));
            let options = AttentionModuleOptions {
                is_causal: true,
                ..AttentionModuleOptions::default()
            };
            measure(settings, device, op, pass, &inputs, || {
                let [query, key, value] = inputs.clone();
                attention(query, key, value, None, None, options)
            });
        }
    }
}

/// A normal draw of `shape` on `device`, of mean 0 and deviation
/// `deviation`, that tracks its gradient in the passes on the Autodiff
/// device.
fn input<const D: usize>(
    shape: [usize; D],
    deviation: f64,
    device: &Device,
    pass: Pass,
) -> Tensor<D> {
    let tensor = Tensor::random(shape, Distribution::Normal(0.0, deviation), device);
    if pass == Pass::Forward {
        tensor
    } else {
        tensor.require_grad()
    }
}

/// The targets of the cross-entropy: a fixed draw of `tokens` tokens.
fn targets(tokens: usize) -> Vec<i64> {
    let vocab = u64::try_from(VOCAB).expect("the vocabulary fits");
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    (0..tokens)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            i64::try_from((state >> 33) % vocab).expect("a token fits")
        })
        .collect()
}

/// Time `forward` in `pass`: alone, or with the backward of its sum and the
/// gradients of `inputs`.
fn measure<const D: usize, const E: usize>(
    settings: &Settings,
    device: &Device,
    op: Op,
    pass: Pass,
    inputs: &[Tensor<D>],
    forward: impl Fn() -> Tensor<E>,
) {
    if pass.backward() {
        time(settings, device, op, pass, || {
            let grads = forward().sum().backward();
            inputs
                .iter()
                .map(|input| input.grad(&grads).expect("the input tracks its gradient"))
                .collect()
        });
    } else {
        time(settings, device, op, pass, || vec![forward()]);
    }
}

/// Run `run` the warm-up times and then the timed ones, each up to the
/// device's sync, its outputs kept until then and, where the settings read
/// them, read after it; print the times.
///
/// # Panics
///
/// If a round leaves an output unwritten, which its read reports.
fn time<const D: usize>(
    settings: &Settings,
    device: &Device,
    op: Op,
    pass: Pass,
    mut run: impl FnMut() -> Vec<Tensor<D>>,
) {
    eprintln!("{} {}: {device:?}", op.name(), pass.name());
    if settings.memory {
        memory(settings, device, op, pass, "-", "-");
    }
    let mut times = Vec::with_capacity(settings.iterations);
    for round in 0..settings.warmup + settings.iterations {
        let start = Instant::now();
        let outputs = run();
        device.sync().expect("the device runs the pass");
        let elapsed = start.elapsed().as_secs_f64() * 1e3;
        if settings.read {
            for output in outputs {
                if let Err(error) = output.try_into_data() {
                    panic!(
                        "round {round} of {} {} left an output unwritten: {error}",
                        op.name(),
                        pass.name()
                    );
                }
            }
        } else {
            drop(outputs);
        }
        if settings.memory {
            let round = round.to_string();
            memory(settings, device, op, pass, &round, &format!("{elapsed:.3}"));
        }
        if round >= settings.warmup {
            times.push(elapsed);
        }
    }
    times.sort_by(f64::total_cmp);
    println!(
        "{}\t{}\t{}\t{}\t{:.3}\t{:.3}\t{:.3}",
        settings.device,
        settings.rows,
        op.name(),
        pass.name(),
        times[times.len() / 2],
        times[0],
        times[times.len() - 1]
    );
}

/// Print, after a sync, the GPU's used memory and the bytes that burn's
/// allocator holds in use and reserved, in MiB, `-` where none is reported.
fn memory(settings: &Settings, device: &Device, op: Op, pass: Pass, round: &str, ms: &str) {
    device.sync().expect("the device runs the pass");
    let pool = device.memory_pool_usage().map_or_else(
        || "-\t-".to_owned(),
        |usage| {
            format!(
                "{}\t{}",
                usage.bytes_in_use >> 20,
                usage.bytes_reserved >> 20
            )
        },
    );
    println!(
        "memory\t{}\t{}\t{}\t{}\t{round}\t{ms}\t{}\t{pool}",
        settings.device,
        settings.rows,
        op.name(),
        pass.name(),
        nvidia_smi("memory.used")
    );
}

/// What `nvidia-smi` gives for `query` on GPU 0, or `-` if it gives nothing.
fn nvidia_smi(query: &str) -> String {
    Command::new("nvidia-smi")
        .arg(format!("--query-gpu={query}"))
        .args(["--format=csv,noheader,nounits", "-i", "0"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map_or_else(
            || "-".to_owned(),
            |output| String::from_utf8_lossy(&output.stdout).trim().to_owned(),
        )
}
