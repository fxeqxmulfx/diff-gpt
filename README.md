# diff-gpt

Autoregressive time-series forecasting with a decoder-only transformer trained on
a derivative-based tokenization. The encoder rounds the signal onto a fixed grid
of step $\Delta$ and emits the $k$-th differences of the grid index
(level-grid tokenization, LGT: $q_t = \Delta^k\,\mathrm{round}(s\,x_t) + V/2$);
the decoder integrates the tokens exactly in integer arithmetic. The model is a
plain GPT that reads the token stream as ordinary text — the inductive bias
sits entirely in the tokenization.

The second tokenizer, MASH (`tokenizer="mash"`), rounds onto the same grid
anchored at a polynomial of degree $< k$ through the window prefix, the
rounding a MASH $\Sigma\Delta$ cascade started there performs. Both share the
scale, the bound and the exact decoding for any $k$ and $T$. With LGT the
token at $t$ depends only on $x_{t-k..t}$: encoding a window equals the slice
of the full encoding, and identical values give identical tokens anywhere in a
series. MASH tokens depend on the prefix as well; in exchange MASH puts the
prefix values on its grid, so a constant column comes back exactly, where LGT
rounds it to $2^{-21}$. Pass `tokenizer="mash"` to
`encode`/`decode`, `DiffGPT` and the data loaders, or set
`DIFF_GPT_TOKENIZER=mash` to make it the default. `DiffGPT` and the loaders
keep the tokenizer they were built with, `DiffGPT.train` refuses a loader that
encodes with another, and `DiffGPT.load` refuses weights trained on another
token scheme (`ENCODER_VERSION`).

## Signal assumption

```math
\exists m \ge 0 \quad \exists M \ge 0 \quad \forall n \ge 0 \quad \forall \mathbf{x} \in \mathbb{R}^n: \quad \|f^{(m)}(\mathbf{x})\|_{\infty} \le M
```

## Tokenization contract

Two vocabulary-scaling laws hold *at the tokenization level*, independent of the
model on top.

| quantity | per vocab doubling | reference |
|---|---|---|
| ideal CE $\mathcal L^\star(V)$ | $+\log 2 \approx 0.69$ nats | Prop. 2 |
| reconstruction $\|\hat x - x\|_\infty$ | $\times \tfrac{1}{2}$ | Thm. 1 |

**Theorem 1 (reconstruction, any $k$, both tokenizers).** For any signal with
$D = \max_t |\Delta^k x_t| > 0$ and $V > \max(2, 2^k)$, every token lies in
$[1, V-1]$ and

$$\|\hat x - x\|_\infty \le \frac{\Delta}{2} = \frac{D}{V - \max(2, 2^k)}$$

uniformly in horizon $T$. Vocabulary doubling halves reconstruction error at any
order $k$. The proof is checked in Lean 4: `lean/DiffGPT/LevelGrid.lean` for
exact arithmetic, and `Float64.lean`, `ExactLevels.lean`, `Int64.lean` and
`Machine.lean` next to it for the float64/int64 implementation, end to end in
`encode_decode`. There the levels are exact, so every token lies in
$[1, V-1]$ as in exact arithmetic, `decode` gets the levels back without an
int64 overflow, and the decode adds only the float64 rounding of $(R + F)/s$,
about $2^{-53}|\hat x|$ per operation:

$$|\hat x_t - x_t| \le \frac{\Delta}{2} + 2^{-52}\left(1 + 2^{-54}\right)\left(|x_t| + \frac{\Delta}{2}\right) + 2^{-1075}\left(1 + \left(1 + 2^{-53}\right)\Delta\right).$$

`encode` raises where the float64 product $s\,x$ reaches
$2^{\min(53,\,62-k)}$.

The bound needs a `domain_of_definition` of at least $D$. A smaller one (a
quantile, say) clips the larger steps, and `encode` reports how many with a
`TokenClippingWarning`. At $k \ge 1$ a clipped step offsets every decoded
value after it, with either tokenizer. `predict` decodes
the forecast from the last context rows, so clipped context steps do not shift
it.

**Proposition 2 (achievable CE).** In the fine-quantization limit,
$\mathcal L^\star(V) = h(X_t \mid \text{ctx}) + \log V - \log(2D) + o(1)$.
Ideal loss is linear in $\log V$, slope 1. Cross-vocab comparisons must subtract
out $\log V$ to get the vocab-invariant *differential-entropy residual*
$\tilde h(V) = \mathcal L(V) - \log V + \log(2D)$.

## Key features

- **Level-grid tokenization**: tight $\Delta/2$ reconstruction bound uniformly
  in $T$ at any order $k$, exact integer decoding, window-consistent tokens.
  The prefix-anchored MASH grid (`tokenizer="mash"`) keeps the same bound.
- **Bayes point readout** (`DiffGPT.predict_point`): the point forecast that
  minimises the expected metric (MAE/MASE, MSE, sMAPE or M4's OWA) under
  $N$ sampled trajectories, instead of feeding argmax/median tokens back.
- **Channel-aligned windows** (`DiffDataFrameDataLoader(align_windows=True)`):
  a multivariate stream interleaves $F$ channels, so training windows are cut
  on a channel boundary, like the inference context.
- **Free-run temperature calibration** (`DiffGPT.calibrate_temperature`):
  cross-entropy calibrates each one-step distribution, but a rollout feeds its
  own samples back and the joint forecast comes out over-dispersed. Pick the
  temperature on held-out windows by CRPS — *not* by matching interval
  coverage, which any wide enough band satisfies. On ETTh1 the CRPS criterion
  picks $T = 0.5$ and the sampled readout then beats the deterministic path;
  coverage-matching picks $T = 1$, where MSE is 2.4x worse.
- **Per-column derivative order**: `order_of_derivative` can be an array of shape
  $(F,)$ giving each channel its own $k_c$. Mix $k = 0$ (raw) with $k = 1, 2, \ldots$
  in a single model.
- **Plain GPT model** (as in nanoGPT with `bias=False`): learned token and
  position embeddings, pre-LayerNorm blocks with causal self-attention and a
  GELU MLP, no biases, an output head tied to the token embedding,
  cross-entropy loss and $\mathcal{N}(0, 0.02)$ initialization with residual
  projections scaled by $1/\sqrt{2L}$ — nothing series-specific. Trained with
  the Schedule-Free AMSGrad optimizer.
- **Inference engine**: KV-cache-accelerated generation, position-aware
  `force_schedule` for teacher-forced decoding (TimeXer-style exogenous
  covariates), best-val checkpoint auto-restore.
- **NUMA-aware training**: single-process CPU-affinity pin via
  `DiffGPT.train(pin_node=...)`, or one-worker-per-node DDP via
  `DiffGPT.train_numa` on multi-socket boxes.

The numbers in this README were measured with earlier versions of the model,
which carried model-side additions that have since been removed: Gaussian
ordinal soft-CE, a discretised logistic-mixture head, exact token features,
channel embeddings and a RoPE/RMSNorm/SwiGLU backbone with Attention
Residuals. The best of those versions reached sMAPE 13.11, MASE 1.018 and
OWA 0.569 on M4 Hourly. The plain GPT model has not been re-benchmarked yet;
`improvement_plan.md` carries the step-by-step ablation of the earlier
versions.

## Usage

```python
import numpy as np, pandas as pd, torch
from diff_gpt.diff_gpt import DiffGPT
from diff_gpt.model.gpt import GPT
from diff_gpt.data_loader import DiffDataFrameDataLoader
from diff_gpt.encoder_decoder import get_domain_of_definition
from diff_gpt.sampler.temperature import TemperatureSampler

# Your data as DataFrames (one per series).
dfs: list[pd.DataFrame] = [...]

# Compute domain (per-channel max |k-th diff|) from training data.
order = 1  # scalar, OR np.array([0, 1, 2]) for per-column
all_data = np.concatenate([df.to_numpy(dtype=np.float64) for df in dfs], axis=0)
domain = get_domain_of_definition(
    inp=all_data,
    order_of_derivative=order,
)

# Construct GPT + DiffGPT wrapper.
base = GPT(
    vocab_size=256,
    n_embd=64,
    block_size=138,
    n_head=4,
    n_layer=2,
).to("cuda")  # train() expects the model on the loader's device
model = DiffGPT(
    model=base,
    order_of_derivative=order,
    domain_of_definition=domain,
)

loader = DiffDataFrameDataLoader(
    dfs=dfs,
    block_size=138,
    batch_size=32,
    vocab_size=256,
    order_of_derivative=order,
    domain_of_definition=domain,
    device="cuda",
    train_part=0.8,
)

model.train(loader=loader, max_iters=20_000, eval_interval=500)

# Point forecast: Bayes-optimal readout for the metric you are scored on,
# from 100 sampled trajectories ("mae" -> marginal median, "mse" -> mean,
# "smape", or "owa" with M4's Naive2 references).
context = dfs[0].iloc[-96:]
prediction = model.predict_point(
    df=context,
    max_new_points=48,
    metric="mae",
    num_samples=100,
)
# Deterministic single-path readouts are still available:
from diff_gpt.sampler.median import MedianSampler

path = model.predict(df=context, max_new_points=48, sampler=MedianSampler())
```

## Probabilistic forecasting

Every trained model is already a probabilistic forecaster — the token head
outputs a calibrated distribution at each step (cross-entropy is a strictly
proper scoring rule). Point argmax is one readout; `num_samples` independent
temperature-1 trajectories give Monte-Carlo samples from the joint future
distribution, which we can reduce to any set of quantiles.

```python
from diff_gpt.sampler.temperature import TemperatureSampler
from diff_gpt.sampler.nucleus import NucleusSampler

# N Monte-Carlo trajectories, shape (N, H, F).
# Plain temperature-1 sampling — the most diverse readout.
samples = model.predict_samples(
    df=context,
    max_new_points=48,
    num_samples=100,
    sampler=TemperatureSampler(temperature=1.0),
)

# Nucleus top-p on top of temperature: drop the low-probability tail of
# each per-step distribution before sampling. Tightens forecast bands
# without any retraining (−25% MSIS on M4 Hourly at p=0.9).
nucleus = NucleusSampler(
    p=0.9,
    sampler=TemperatureSampler(temperature=1.0),
)
samples_nucleus = model.predict_samples(
    df=context,
    max_new_points=48,
    num_samples=100,
    sampler=nucleus,
)

# Or directly as quantile bands {0.1, 0.5, 0.9}: each a DataFrame of shape (H, F).
bands = model.predict_quantiles(
    df=context,
    max_new_points=48,
    quantiles=(0.1, 0.5, 0.9),
    num_samples=100,
    sampler=nucleus,
)
p10, p50, p90 = bands[0.1], bands[0.5], bands[0.9]
```

No architectural change and no auxiliary pinball-loss head: one model serves
arbitrary quantile sets at inference time. Since the model predicts the
*derivative* and the decoder integrates, confidence bands naturally widen
with horizon as an integral of random-walk uncertainty.

**First-pass M4 Hourly numbers** (an earlier soft-CE model, no
probabilistic-specific tuning; 414 series, horizon 48, N = 100 samples):

| metric | T = 1 | T = 1, top-p = 0.9 |
|---|---|---|
| sMAPE (median point) | 24.18 | **22.02** |
| MASE (median point) | 2.94 | **2.41** |
| CRPS | 332.0 | **293.6** |
| MSIS (α = 0.05) | 41.1 | **30.8** (−25%) |
| coverage@80 (nominal 0.80) | 0.946 | **0.919** |

Nucleus top-p = 0.9 at inference time (zero retraining) tightens every
metric, MSIS by 25%. Remaining over-coverage reflects the fact that the
model was tuned for argmax sMAPE; conformal calibration or a CRPS-tuned
temperature (`DiffGPT.calibrate_temperature`) closes most of the rest of the
gap vs M4 probabilistic winners (MSIS ~13–15).

## Anomaly detection

Cross-entropy training also gives a calibrated **per-step conditional
log-likelihood** for free. Feeding an observed signal through a single
forward pass and taking $-\log p(\text{token}_t \mid \text{ctx})$ at each
position yields a per-(time, channel) anomaly score: low where the
observation is typical, high where it surprises the model.

```python
# df: observed multivariate signal, shape (T, F).
scores = model.anomaly_scores(df)
# scores: DataFrame, same shape and columns as df.
# First max_k rows are NaN (the derivative prefix); the first encoded
# token at column 0 is also NaN (no preceding context).

row_surprise = scores.sum(axis=1, skipna=True)  # per-step aggregate
anomalies = row_surprise[row_surprise > row_surprise.quantile(0.99)]
```

Regression-based forecasters have to approximate this with $\|y_t - \hat
y_t\|$, which miscalibrates on heteroscedastic series; the categorical
head's log-likelihood is proper under the learned distribution.

**ETTh1 oil-temperature demo** ([`benchmarks/anomaly_demo.py`](benchmarks/anomaly_demo.py)):
train on 2880 clean hours, inject 5 synthetic ±5σ spikes into a 336-hour
test window, rank positions by NLL. After only 2000 training iterations
the top-10 contains **3 of 5 injected spikes (60% recall)**; the rest of
the top-10 are genuine irregular transitions in the unsynthetic signal.

## Compiled training (`torch.compile`)

PyTorch 2's `torch.compile` fuses the GPT forward/backward into a handful
of kernels — a sizable step-time speedup at this model scale, once the
one-time graph-tracing cost is amortized. Two rules:

- **Disable gradient checkpointing.** `use_checkpoint=True` introduces
  graph breaks that defeat the fusion. Pass `use_checkpoint=False` to
  `GPT` when compiling.
- **Compile the inner `GPT`, not the `DiffGPT` wrapper.** `DiffGPT` is a
  numpy-side encoder/decoder shell around an `nn.Module`; `torch.compile`
  only operates on the latter. Compile before constructing the wrapper.

```python
base = GPT(
    vocab_size=256,
    n_embd=64,
    block_size=138,
    n_head=4,
    n_layer=2,
    use_checkpoint=False,  # required alongside torch.compile
)
# GPU: max-autotune without cudagraphs — peak Triton/Inductor tuning,
# no CUDA-graph capture (which breaks under variable shapes, DDP, and
# KV-cache inference). CPU: max-autotune enables Inductor's CPU GEMM
# template / tile sweep; cudagraphs is a no-op on CPU either way.
compile_mode = (
    "max-autotune-no-cudagraphs" if torch.cuda.is_available() else "max-autotune"
)
base = torch.compile(base, mode=compile_mode)
model = DiffGPT(
    model=base,
    order_of_derivative=order,
    domain_of_definition=domain,
)
model.train(loader=loader, max_iters=20_000, eval_interval=500)
```

Combined with `DiffGPT.train_numa`, compile inside the factory so every
worker DDP-wraps an already-compiled module:

```python
def build_gpt(rank: int) -> GPT:
    gpt = GPT(
        vocab_size=256,
        n_embd=64,
        block_size=138,
        n_head=4,
        n_layer=2,
        use_checkpoint=False,
    )
    mode = "max-autotune-no-cudagraphs" if torch.cuda.is_available() else "max-autotune"
    return torch.compile(gpt, mode=mode)
```

The checkpoint that `train_numa` writes holds the bare `GPT`'s keys, so load
it into a plain `GPT` (or into the compiled module's `_orig_mod`).

The first 1–2 training iterations are dominated by tracing and kernel
compilation; benchmark step time from iteration 10+ for a fair number.
Inference (`predict`, `predict_samples`, `anomaly_scores`) still works
on a compiled model — `OptimizedModule` delegates attribute access — but
the KV-cache generation path grows `T` by one token per step, so the
compiled graph is recompiled or specialized frequently and the speedup
is smaller than at training time.

## NUMA-aware training

On multi-socket CPU boxes, cross-socket memory traffic and thread migration
dominate wall-clock for this size of model. Two entry points:

**Single-process pin.** Pass a NUMA node id to `train()` — pins CPU affinity
to the node's CPUs that the process may use (a cpuset or a SLURM job may allow
fewer than the node has) and aligns `torch.set_num_threads` /
`OMP_NUM_THREADS` to their count before the hot loop:

```python
model.train(loader=loader, max_iters=20_000, pin_node=0)
```

**One process per NUMA node (DDP).** For linear scaling across sockets,
`DiffGPT.train_numa` spawns one worker per NUMA node with a CPU the process
may use, pins each to those CPUs, and wraps the inner GPT in `DistributedDataParallel`. Both
factories run *inside* the pinned worker so allocations land on local
memory; seed the loader by `rank` so workers draw different batches
(otherwise DDP averages identical gradients).

```python
from diff_gpt.diff_gpt import DiffGPT
from diff_gpt.model.gpt import GPT
from diff_gpt.data_loader import DiffDataFrameDataLoader
import torch


def build_gpt(rank: int) -> GPT:
    return GPT(vocab_size=256, n_embd=64, block_size=138, n_head=4, n_layer=2)


def build_loader(rank: int, world: int) -> DiffDataFrameDataLoader:
    rng = torch.Generator().manual_seed(42 + rank)
    return DiffDataFrameDataLoader(
        dfs=dfs,
        block_size=138,
        batch_size=32,
        vocab_size=256,
        order_of_derivative=order,
        domain_of_definition=domain,
        device="cpu",
        train_part=0.8,
        rng=rng,
    )


val_loss, _ = DiffGPT.train_numa(
    model_factory=build_gpt,
    loader_factory=build_loader,
    train_kwargs={"max_iters": 20_000, "eval_interval": 500},
    checkpoint_path="best.pt",
)
# Rebuild DiffGPT around the trained weights:
gpt = build_gpt(0)
gpt.load_state_dict(torch.load("best.pt"))
model = DiffGPT(model=gpt, order_of_derivative=order, domain_of_definition=domain)
```

Backend: Gloo for a CPU model, NCCL for a CUDA model. NCCL needs a GPU per
rank: rank r runs on GPU r, so build the model with `.to("cuda")` in the
factory. Single-node systems short-circuit to the in-process pinning path — no
spawn, no DDP overhead.

## Benchmarks

Measured with an earlier version of the model: a RoPE/RMSNorm/SwiGLU backbone
trained with Gaussian ordinal soft-CE, both since removed. The plain GPT
model has not been re-benchmarked yet.

### M4 Hourly (short-term, global)

414 series, horizon 48, per-series z-normalization, single global model,
20k iters, argmax inference.

| config | sMAPE | MASE |
|---|---|---|
| baseline ($V=256$, plain CE) | 18.58 | — |
| + soft-CE $\sigma = 1$ | 17.55 | 1.663 |
| + soft-CE $\sigma = 2$ | **17.49** | **1.659** |
| + soft-CE $\sigma = 3$ | 17.35 | 1.949 (over-smoothed) |

Vocab sweep (U-curve, optimum at $V = 256$):

| $V$ | $\sigma$ | sMAPE |
|---|---|---|
| 64  | 0.5 | 19.96 |
| 128 | 1.0 | 19.39 |
| **256** | **2.0** | **17.49** |
| 512 | 4.0 | 18.21 |

Higher-order tokenization: $k=1 \to$ **17.49**, $k=2 \to 39.27$. Prediction
error compounds as $O(\Delta \cdot T^{k-1})$ — MASH gives tight *reconstruction*
but a noisy predictor integrated twice is much worse than once. $k = 1$ is the
forecasting sweet spot.

### ETTh1 (long-term, multivariate, iTransformer protocol)

7 channels, seq_len = 96, pred_len = 96, chronological 12/4/4-month split.

| config | MSE | MAE |
|---|---|---|
| $V = 256$, plain CE | 0.9861 | 0.6056 |
| $V = 64$, plain CE | 0.9787 | 0.6030 |
| $V = 64$, soft-CE $\sigma = 1.0$ | 0.9806 | 0.6034 |

ETTh1 is data-limited (~2 tokens/parameter), not loss-limited. Soft targets help
most in data-starved regimes.

## Install

```bash
pip install git+https://github.com/fxeqxmulfx/diff-gpt
```

## Develop & test

```bash
uv sync
uv run --no-sync pytest            # skips the tests marked slow
uv run --no-sync pytest --runslow  # runs everything
```

## References

- [youtube.com: Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [github.com: nanoGPT](https://github.com/karpathy/nanoGPT)
- [github.com: schedule_free](https://github.com/facebookresearch/schedule_free)
- [github.com: nanochat](https://github.com/karpathy/nanochat)
- [github.com: Time-Series-Library](https://github.com/thuml/Time-Series-Library)
- [arxiv.org: Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [openai.com: Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [arxiv.org: The Road Less Scheduled](https://arxiv.org/abs/2405.15682)
- [arxiv.org: The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)
- [arxiv.org: iTransformer: Inverted Transformers are Effective for Time Series Forecasting](https://arxiv.org/abs/2310.06625)
