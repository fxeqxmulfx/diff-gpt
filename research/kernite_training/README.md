# Native Rust / Kernite training

The current two-token cubic-sum model has a standalone Rust trainer. All GPU
model arithmetic is Kernite-generated PTX, including FP32 SGEMM, forward and
backward normalization, attention/XSA/RoPE/SSMax, SwiGLU, tied embedding gradients,
cross entropy and AdamW. There is no Python or libtorch in its training loop.
Python exports the frozen initialization, data and batch order once and acts as
an independent correctness oracle.

History uses a 1 MiB `BufWriter`. The trainer flushes and fsyncs history and
checkpoints every **500 epochs**, and immediately on success, completion or an
explicit stop. It still evaluates every epoch: first-observed 99% is unchanged.
For this task, 500 epochs are 9,500 updates, including all 96-example tail batches.
Before each window a durable intent bounds potentially issued work. A crash can
lose up to 500 epochs of unsaved state; automatic recovery refuses an uncertain
window instead of silently replaying it. Clean checkpoints restore parameters,
both moments, batch cursor and observation history exactly.

The implementation supports the original model and the inverse-width normalized
model. Validation covers L2/W128, L1/W256 and L2/W256, both architectures, all
three seeds: **18 runs, 1,950 CUDA comparisons**, through 64 updates. This includes
all parameter gradients, logits, loss, exact predictions, zero-rate warmup and
AdamW state. Three independent GPU tests cover RMS zero/radial derivatives,
softmax gradients and FP32 SGEMM, including partial tiles. A 501-epoch execution
test retains all 502 observations while emitting only the initial, epoch-500
and final records. Buffered checkpoint continuation is bit-for-bit identical to
uninterrupted execution; corrupted checkpoints and uncertain restarts are refused.

On this RTX 3050, the 608-update diagnostic with every-epoch complete-split
scoring took about **2.13 seconds** in the buffered native loop versus **9.94
seconds after preparation** in the unchanged trainer, about **4.7× faster**.
The unchanged trainer took 21.73 seconds including setup. These are single
workload measurements, not a universal performance claim. Raw computation was
similar (2.37 vs 2.43 ms/update); removing per-update disk synchronization explains
most of the gain. The earlier concurrent-GPU timing is retained and excluded.
See [the results](results/summary.json) for the measurements, limitations and
source hashes.

Small FP32 rounding differences remain. At 608 updates the independent trainers
already have slightly different metrics; finite-trajectory parity does not prove
identical long-run grokking. This backend is separate execution evidence, not a
new scientific winner. The existing capacity cohort keeps its frozen sources;
its brief benchmark suspension is recorded. Contexts longer than two tokens,
KV caching, other optimizers and trainable affine normalizers are not ported.

## Build and run

Kernite is the local checkout at `/home/misha/rust_projects/kgen`, revision
`a1b6bc2c956e1a5f35960e26b209e1b6c213096c`. No Kernite patch was needed.
The small standalone Cargo workspace leaves the existing Burn workspace alone.
From the GPT repository root:

```sh
CARGO_BUILD_JOBS=4 cargo build --release --offline \
  --manifest-path research/kernite_training/native/Cargo.toml \
  --target-dir target/kernite

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. .venv/bin/python \
  -m research.kernite_training.export runs/native-cubic-input \
  --name cubic-depth-normalized-lr2 --seed 0 --updates 100000

target/kernite/release/astra-kernite train \
  runs/native-cubic-input runs/native-cubic-output 100000
```

Input and output directories must be distinct. New runs refuse existing output
directories. `resume` replaces `train` only for a clean, compatible checkpoint.
An explicit stop before the exported budget must land on a 19-update observation
boundary. Keep each new execution cohort separate from the original capacity
results. Exported inputs and checkpoint files are hashed before use.

## Reproduce validation

```sh
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. .venv/bin/python \
  -m research.kernite_training.validate target/kernite/release/astra-kernite \
  runs/native-validation

CARGO_BUILD_JOBS=4 cargo test --release --offline \
  --manifest-path research/kernite_training/native/Cargo.toml \
  --target-dir target/kernite -- --ignored --test-threads=1

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. .venv/bin/python \
  -m research.kernite_training.export runs/native-buffer-input --updates 9519
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. .venv/bin/python \
  -m research.kernite_training.check_buffered target/kernite/release/astra-kernite \
  runs/native-buffer-input runs/native-buffer-check
```

`validate --recovery-only` checks the host persistence path without repeating the
kernel matrix. `probe` writes the gradients and intermediate tensors used by
`check`; `bench` measures updates and full-split scoring without disk journaling.
`benchmark_reference` separates existing compiled CUDA performance from its
per-update journal. GPU timings require an idle GPU.
