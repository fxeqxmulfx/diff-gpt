//! The bench of burn's backends: a matmul, a softmax and a cross-entropy at
//! the sizes of the 5M model, each forward alone and forward with its
//! backward, on an autodiff device.
//!
//! `backends <device> [--ops matmul,softmax,cross_entropy] [--warmup N]
//! [--iterations N] [--softmax-rows N]`, the device one of `ndarray`, `flex`,
//! `libtorch`, `libtorch-cuda` and `cuda`. It prints a line per op and pass:
//! the device, the op, the pass, and the median, the least and the most of
//! the iterations' times in milliseconds, each timed up to the device's sync.
//! The command that starts it sets the threads (`numactl`,
//! `RAYON_NUM_THREADS`, `OMP_NUM_THREADS`); the bench sets none.
#![forbid(unsafe_code)]

use std::process::ExitCode;
use std::time::Instant;

use burn::nn::loss::CrossEntropyLossConfig;
use burn::tensor::activation::softmax;
use burn::tensor::{Device, Distribution, Int, Tensor, TensorData};

/// The 5M model's batch: 12 rows of its block of 1412 tokens.
const ROWS: usize = 12;
const BLOCK: usize = 1412;
const TOKENS: usize = ROWS * BLOCK;
/// Its width, heads and vocabulary, and the width of the first linear of
/// its `SwiGLU` (2 × 683).
const WIDTH: usize = 256;
const HEADS: usize = 4;
const VOCAB: usize = 256;
const HIDDEN: usize = 1366;

const USAGE: &str = "usage: backends <ndarray|flex|libtorch|libtorch-cuda|cuda> \
                     [--ops matmul,softmax,cross_entropy] [--warmup N] [--iterations N] \
                     [--softmax-rows N]";

#[derive(Clone, Copy, PartialEq, Eq)]
enum Op {
    Matmul,
    Softmax,
    CrossEntropy,
}

struct Settings {
    device: String,
    ops: Vec<Op>,
    warmup: usize,
    iterations: usize,
    softmax_rows: usize,
}

fn main() -> ExitCode {
    match settings(std::env::args().skip(1)).and_then(|settings| {
        let device = device(&settings.device)?.autodiff();
        bench(&settings, &device);
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
    let mut settings = Settings {
        device,
        ops: vec![Op::Matmul, Op::Softmax, Op::CrossEntropy],
        warmup: 3,
        iterations: 11,
        softmax_rows: ROWS,
    };
    while let Some(flag) = args.next() {
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
            "--ops" => {
                settings.ops = value
                    .split(',')
                    .map(|name| match name {
                        "matmul" => Ok(Op::Matmul),
                        "softmax" => Ok(Op::Softmax),
                        "cross_entropy" => Ok(Op::CrossEntropy),
                        _ => Err(format!("no op {name}\n{USAGE}")),
                    })
                    .collect::<Result<_, _>>()?;
            }
            "--warmup" => settings.warmup = count()?,
            "--iterations" => settings.iterations = count()?,
            "--softmax-rows" => settings.softmax_rows = count()?,
            _ => return Err(format!("no flag {flag}\n{USAGE}")),
        }
    }
    Ok(settings)
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

fn bench(settings: &Settings, device: &Device) {
    eprintln!("{device:?}");
    if settings.ops.contains(&Op::Matmul) {
        let x = Tensor::<2>::random([TOKENS, WIDTH], Distribution::Normal(0.0, 1.0), device)
            .require_grad();
        let w = Tensor::<2>::random([WIDTH, HIDDEN], Distribution::Normal(0.0, 0.02), device)
            .require_grad();
        time(settings, device, "matmul", "forward", || {
            x.clone().matmul(w.clone())
        });
        time(settings, device, "matmul", "forward+backward", || {
            let grads = x.clone().matmul(w.clone()).sum().backward();
            (x.grad(&grads), w.grad(&grads))
        });
    }
    if settings.ops.contains(&Op::Softmax) {
        let scores = Tensor::<4>::random(
            [settings.softmax_rows, HEADS, BLOCK, BLOCK],
            Distribution::Normal(0.0, 1.0),
            device,
        )
        .require_grad();
        time(settings, device, "softmax", "forward", || {
            softmax(scores.clone(), 3)
        });
        time(settings, device, "softmax", "forward+backward", || {
            let grads = softmax(scores.clone(), 3).sum().backward();
            scores.grad(&grads)
        });
    }
    if settings.ops.contains(&Op::CrossEntropy) {
        let logits = Tensor::<2>::random([TOKENS, VOCAB], Distribution::Normal(0.0, 1.0), device)
            .require_grad();
        let targets = Tensor::<1, Int>::from_data(TensorData::new(targets(), [TOKENS]), device);
        let loss = CrossEntropyLossConfig::new().init(device);
        time(settings, device, "cross_entropy", "forward", || {
            loss.forward(logits.clone(), targets.clone())
        });
        time(
            settings,
            device,
            "cross_entropy",
            "forward+backward",
            || {
                let grads = loss.forward(logits.clone(), targets.clone()).backward();
                logits.grad(&grads)
            },
        );
    }
}

/// The targets of the cross-entropy: a fixed draw of tokens.
fn targets() -> Vec<i64> {
    let vocab = u64::try_from(VOCAB).expect("the vocabulary fits");
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    (0..TOKENS)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            i64::try_from((state >> 33) % vocab).expect("a token fits")
        })
        .collect()
}

/// Run `pass` the warm-up times and then the timed ones, each up to the
/// device's sync, its outputs kept until then; print the times.
fn time<T>(settings: &Settings, device: &Device, op: &str, pass: &str, mut run: impl FnMut() -> T) {
    let mut times = Vec::with_capacity(settings.iterations);
    for round in 0..settings.warmup + settings.iterations {
        let start = Instant::now();
        let outputs = run();
        device.sync().expect("the device runs the pass");
        let elapsed = start.elapsed().as_secs_f64() * 1e3;
        drop(outputs);
        if round >= settings.warmup {
            times.push(elapsed);
        }
    }
    times.sort_by(f64::total_cmp);
    println!(
        "{}\t{op}\t{pass}\t{:.3}\t{:.3}\t{:.3}",
        settings.device,
        times[times.len() / 2],
        times[0],
        times[times.len() - 1]
    );
}
