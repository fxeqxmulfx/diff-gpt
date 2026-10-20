# diff-gpt

Autoregressive time-series forecasting with a decoder-only transformer trained on
a derivative-based tokenization. The encoder rounds the signal onto a grid
of step $\Delta$ and emits the grid index differenced by an integer operator
$c(B)$, the $k$-th difference $\Delta^k = (1 - B)^k$ or a seasonal one such as
$1 - B^{24}$ (level-grid tokenization, LGT:
$q_t = c(B)\,\mathrm{round}(s\,x_t) + V/2$); the decoder integrates the
tokens exactly in integer arithmetic. The model is a
plain GPT that reads the token stream as ordinary text — the inductive bias
sits entirely in the tokenization.

The default tokenizer, MASH (`tokenizer="mash"`), rounds onto the same grid
anchored at a sequence that $c(B)$ maps to zero through the window prefix;
for $\Delta^k$ it is a polynomial of degree $< k$, the rounding a MASH
$\Sigma\Delta$ cascade started there performs. Both share the scale, the
bound and the exact decoding for any operator and $T$. With LGT
(`tokenizer="lgt"`) the token at $t$ depends only on $x_{t-p..t}$ for the
degree $p$ of $c$ ($k$ for $\Delta^k$): encoding a window equals the slice
of the full encoding, and identical values give identical tokens anywhere in a
series. MASH tokens depend on the prefix as well; in exchange MASH puts the
prefix values on its grid, so a constant column comes back exactly, where LGT
rounds it to $2^{-21}$. Pass `tokenizer="lgt"` to
`encode`/`decode`, `DiffGPT` and the data loaders, or set
`DIFF_GPT_TOKENIZER=lgt` to make it the default. `DiffGPT` and the loaders
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
| ideal CE $\mathcal L^\star(V)$ | $+\log 2 \approx 0.69$ nats | Prop. 3 |
| reconstruction $\|\hat x - x\|_\infty$ | $\times \tfrac{1}{2}$ | Thm. 1 |

**Theorem 1 (reconstruction, any operator, both tokenizers).** Let
$c(B) = \prod_{i=1}^m (1 - B^{S_i})$, $m \ge 1$, with lags $S_i \ge 1$ ($k$
lags of 1 for $\Delta^k$) have $\|c\|_1 = \sum_j |c_j|$, which is $2^k$ for
$\Delta^k$. For
any signal with $D = \max_t |c(B)\,x_t| > 0$ and $V > \|c\|_1$, every token
lies in $[1, V-1]$ and

$$\|\hat x - x\|_\infty \le \frac{\Delta}{2} = \frac{D}{V - \|c\|_1}$$

uniformly in horizon $T$. No larger scale keeps every token in range: rounding
errors with the signs of the coefficients of $c$ push one out. The order 0,
$c = 1$, needs no such room: on the grid shifted by half a step the tokens take
all of $[0, V-1]$, and the error $\Delta/2$ is a hair above $D/V$, the least
that $V$ values allow on $[-D, D]$. Vocabulary doubling halves reconstruction
error for any operator. The proof is checked in Lean 4:
`lean/DiffGPT/LevelGrid.lean` for exact arithmetic, the sharpness and the
order 0 included, and `Float64.lean`, `ExactLevels.lean`, `Int64.lean` and
`Machine.lean` next to it for the float64/int64 implementation, end to end in
`encode_decode` (`encode_decode_zero` for the order 0). There the levels are
exact, so every token lies in
$[1, V-1]$ ($[0, V-1]$ at the order 0) as in exact arithmetic, `decode` gets
the levels back without an
int64 overflow, and the decode adds only the float64 rounding of $(R + F)/s$,
about $2^{-53}|\hat x|$ per operation:

$$|\hat x_t - x_t| \le \frac{\Delta}{2} + 2^{-52}\left(1 + 2^{-54}\right)\left(|x_t| + \frac{\Delta}{2}\right) + 2^{-1075}\left(1 + \left(1 + 2^{-53}\right)\Delta\right).$$

`encode` raises where the float64 product $s\,x$ reaches
$2^{\min(53,\,62-m)}$ for the $m$ factors of $c$.

The bound needs a `domain_of_definition` of at least $D$. A smaller one (a
quantile, say) clips the larger steps, and `encode` reports how many with a
`TokenClippingWarning`. Above the order 0 a clipped step offsets the decoded
values after it, with either tokenizer. `predict` decodes
the forecast from the last context rows, so clipped context steps do not shift
it.

**Theorem 2 (no encoder does better).** Take the decoder of Theorem 1: it
reads the first $p$ values raw and decodes $\hat x_t = A_t + L_t/s$, where the
levels $L$ integrate the tokens in $[0, V-1]$ and $c(B)$ maps the anchor $A$
to zero ($A = 0$ for LGT). For $V \ge 2\|c\|_1$, an encoder that keeps every
series with $\max_t |c(B)\,x_t| \le D$ within $\varepsilon$ has

$$\varepsilon \ge \frac{D}{V - \|c\|_1},$$

whatever the scale, the prefix and the anchor, even if it sees the whole
series and uses token 0. The level grid attains it, so the bound of Theorem 1
is the least this decoder allows. Three series force it: a value between two
levels, a ramp $c(B)\,x = D$ that the tokens have to keep up with, and errors
with the signs of the coefficients of $c$ at the edge of the domain. Checked
in Lean 4: `le_mul_of_encodes` and `isLeast_encodes` in
`lean/DiffGPT/Optimality.lean`.

**Proposition 3 (achievable CE).** In the fine-quantization limit,
$\mathcal L^\star(V) = h(X_t \mid \text{ctx}) + \log V - \log(2D) + o(1)$.
Ideal loss is linear in $\log V$, slope 1. Cross-vocab comparisons must subtract
out $\log V$ to get the vocab-invariant *differential-entropy residual*
$\tilde h(V) = \mathcal L(V) - \log V + \log(2D)$.

## Key features

- **Level-grid tokenization**: tight $\Delta/2$ reconstruction bound uniformly
  in $T$ for any operator, exact integer decoding. The default MASH grid is
  anchored at the window prefix; the absolute LGT grid (`tokenizer="lgt"`)
  keeps the same bound and makes the tokens window-consistent.
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
- **Per-column operator**: an order is an int $k$, for $\Delta^k$, or a
  `Difference(*lags)`, for $\prod (1 - B^{S})$: `Difference(24)` is the daily
  seasonal difference of hourly data, `Difference(1, 24)` the SARIMA one,
  $(1 - B)(1 - B^{24})$. `order_of_derivative` can be an array of shape $(F,)$
  giving each channel its own; mix $k = 0$ (raw) with $k = 1, 2, \ldots$ and
  seasonal operators in a single model.
- **Plain GPT model** (as in nanoGPT with `bias=False`): a learned token
  embedding, rotary position embeddings (RoPE) on queries and keys,
  Peri-RMSNorm blocks with causal exclusive self-attention (XSA: each head's
  output is orthogonal to the token's own value) and a SwiGLU MLP.
  Parameter-free RMSNorm normalizes the embedding stream and each branch's
  input and output before residual addition. No biases, an output head tied
  to the token embedding, cross-entropy loss and $\mathcal{N}(0, 0.02)$ initialization
  with residual projections scaled by $1/\sqrt{2L}$ — nothing
  series-specific. Queries and keys use per-head RMSNorm before RoPE and
  standard $1/\sqrt{d_{\mathrm{head}}}$ attention scaling, with no temperature
  parameter. Trained with Schedule-Free AdamW.
- **Inference engine**: KV-cache-accelerated generation, position-aware
  `force_schedule` for teacher-forced decoding (TimeXer-style exogenous
  covariates), best-val checkpoint auto-restore.
- **NUMA-aware training**: single-process CPU-affinity pin via
  `DiffGPT.train(pin_node=...)`, or one-worker-per-node DDP via
  `DiffGPT.train_numa` on multi-socket boxes.

The forecasting numbers in this README come from earlier model versions,
which carried model-side additions that have since been removed: Gaussian
ordinal soft-CE, a discretised logistic-mixture head, exact token features,
channel embeddings and Attention Residuals. The best
of those versions reached sMAPE 13.11, MASE 1.018 and OWA 0.569 on M4 Hourly.
The plain GPT has not been re-benchmarked on them yet; `improvement_plan.md`
carries the step-by-step ablation of the earlier versions.

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

# Compute domain (per-channel max |c(B) x|, the k-th difference at order k).
order = 1  # int k, OR Difference(1, 24), OR np.array([0, 1, 2]) per column
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
# max-autotune: peak Triton/Inductor tuning, plus CUDA graphs on GPU (on
# CPU, Inductor's GEMM template / tile sweep). Module.compile compiles the
# calls of the module, which training makes at one batch shape; inference
# calls its forward, so KV-cache decoding, whose length grows, runs eager.
base.compile(mode="max-autotune")
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
    return torch.compile(gpt, mode="max-autotune")
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

The grokking results below were measured at commit a5d16eb, before
Peri-RMSNorm and Q/K RMSNorm; their checkpoints replay only with that
commit's model.

### Modular division mod 97 (k=0)

The GPT shows delayed grokking on `q = a * inverse(b) mod 97` with
order-0 tokenization. Each independent equation supplies two raw values
`[a, b]`; only the next token `q` is supervised. The codec uses vocabulary
256 and domain 96: all 97 residues have distinct tokens and round back to
their original integers. Inputs never contain the answer.

The browser lab's seeded split and sampler are reproduced exactly: 4,656
training equations, 4,656 test equations, batches of 512 plus a 48-row tail.
The unmodified GPT has width 128, one XSA/SwiGLU block, four heads and 229,248
parameters. ScheduleFreeAdamW uses lr 0.01, weight decay 1, betas 0.9/0.98
and ten warmup updates. A dedicated benchmark loop evaluates both complete
splits every epoch, without validation-loss early stopping. It scores the
optimizer's averaged evaluation weights and records query accuracy separately.

Measured on 2026-10-20 in FP32 on an RTX 3050 Laptop GPU:

| Seed | Memorization plateau, epochs | Test 10% to 90%, epochs | First test >=99%, epoch | Final test accuracy |
|---|---:|---:|---:|---:|
| 11 | 50 | 25 | 118 | 100% |
| 17 | 38 | 43 | 184 | 100% |
| 23 | 71 | 51 | 173 | 99.98% |

The website criterion requires train >=99% and test <=10% for at least 25
epochs, then a rise to 90% within 250 epochs and a 99% test crossing. All
three averaged trajectories qualify and remain above 99% for the next 100
epochs. Query weights fluctuate and do not meet the 25-epoch plateau rule.
Settings were screened on seed 11 before the seed-17/23 confirmations. A
random-label control reaches 100% train and 1.03% test after 500 epochs.
Saved checkpoints replay both weight modes; `DiffGPT.predict_many` verifies
every test equation in the three confirmation runs. All 20 averaged/query
trajectories match the actual browser's classifier.

```bash
# Three seeds and a random-label control; completed runs are retained.
uv run --locked -m benchmarks.grokking.div97 --device cuda \
  --plan benchmarks/grokking/plans/div97_confirm.json \
  --output runs/grokking-div97-confirm

# Curves and compact results; --plot additionally needs matplotlib.
uv run --locked -m benchmarks.grokking.div97_report \
  runs/grokking-div97-confirm --output runs/grokking-div97-confirm/summary.json
```

The five-configuration ScheduleFreeAdamW screening plan is in
`benchmarks/grokking/plans/div97_screen.json`. Local run directories retain full CSV
curves, optimizer checkpoints, codec metadata and source hashes. The
`--verify DIR --full-predict` mode checks a checkpoint through the public
DiffGPT forecasting interface.

### Recurrent modular division across rows (k >= 1, ongoing)

`benchmarks/grokking/recurrence.py` uses one actual time-series column:
`x[t+2] = x[t] / x[t+1] mod 97`. Finite differences run down consecutive
rows; there is no synthetic zero prefix. Complete recurrence orbits are
held out, including every time shift of each test orbit. The seed-11 split
has 4,614 training states and 4,602 test states. By default, only the next
derivative token is supervised; evaluation measures both that token and the decoded
original integer on the complete splits.

With k=1 and 48 input derivative tokens (49 observed raw rows), the
GPT reaches **99.83% test accuracy**: 4,594/4,602 correct. Both token and
decoded accuracy first cross 99% at epoch 50 and stay above it throughout
the next 100 epochs. The saved epoch-150 checkpoint reproduces all 4,602
forecasts through `DiffGPT.predict_many` with the native argmax sampler.
The model has width 128, two blocks and four heads; ScheduleFreeAdamW uses
lr 0.003, weight decay 1, betas 0.9/0.98 and ten warmup updates. The integer
grid has vocabulary 256, domain 127 and scale 1.

This result exploits temporal periodicity: exhaustive orbit enumeration
shows that every division orbit has period dividing 48. Its next derivative
therefore equals the first observed derivative. Input interventions confirm
that the network learned this copy rule; transferring these weights to a
short context gives approximately chance accuracy. This is generalization
to unseen orbits, without the memorization plateau required for delayed
grokking. Learning the division rule with a short context remains an open
experiment. Seed-17/23 confirmations also pass complete public verification:
99.72% (4,593/4,606) at epoch 105 and 99.89% (4,621/4,626) at epoch 171.
Both final checkpoints follow 101 consecutive observations above 99%; seed
23 had an earlier transient crossing. At k=2, context 48 and seed 11, the
same model configuration reaches 99.24% (4,567/4,602) at epoch 128, after
101 consecutive observations above 99%. Every forecast also passes public
verification. This higher-order result uses the same period-copy mechanism.

MASH and LGT produce identical tokens on the exact integer grid. A separate
paired experiment uses vocabulary 512 and domain 96 (scale 2.65625), where
32% of their tokens differ and correct target tokens still decode to the
correct original integers. Samplers are compared on frozen checkpoints,
for both averaged and query weights. On the exact integer grid with four
heads, Fibonacci k=1/C=2 at epoch 4,250 gives 96.24% with argmax, 96.13%
with median, 95.11% with temperature 1, and argmax
restricted to the known original range 0..96 gives 98.45%. The range control
uses observed past rows and the support only; it never computes the
recurrence. These four-head sampler results have not reached the 99% target.

An additional finite-field readout sums model probabilities of equivalent
residues before emitting the canonical derivative token. It uses the known
modulus and observed prefix, without computing the recurrence. On the frozen
Fibonacci epoch-5,000 weights it reaches 98.79%, verified through all 4,704
public forecasts; native argmax at those weights gives 96.47%.

The non-unit-grid tokenizer comparison now runs both encoders with the same
seed, raw orbit partition, model, initialization and ScheduleFreeAdamW
settings. At matched epoch 1,000, decoded test accuracy is 92.39% for MASH
and 3.42% for LGT. At epoch 3,000 the corresponding values are 93.81% and
6.42%. These are exploratory seed-11 results; both long native-argmax runs
continue. `progress-tokenizers.json` records matched budgets and checks
that the raw data, model sizes and core source versions agree. A one-head
pair is also in `benchmarks/grokking/plans/recurrence_tokenizer_heads.json`.

The finite-field readout also supports this non-unit grid, using the actual
codec to encode its chosen canonical continuation. On frozen one-head MASH
weights at epoch 3,750 it reaches 97.96% (4,608/4,704), confirmed through every
public forecast; native argmax on the same weights gives 94.43%. The support
prior helps, but this non-unit-grid case still falls below the 99% target.

With **one attention head**, k=1/C=2, width 128 and one block, the field
readout shows **delayed grokking**. Training still uses the original token
cross-entropy and ScheduleFreeAdamW (lr 0.01, weight decay 1, betas 0.9/0.98,
ten warmup updates). The averaged curve has a 125-epoch memorization plateau,
then rises from 10% to 90% in 101 epochs; it first crosses 99% at epoch 482.
All next 100 observations remain above 99%. The final epoch-582 weights give
**99.21%** (4,667/4,704) for both true derivative tokens and decoded original
values, confirmed on every test orbit through `DiffGPT.predict_many`.

The field support prior is part of this result: native argmax on those same
weights gives 96.56%. The sampler sums probabilities of residues modulo 97
and uses observed decoder anchors to emit a canonical derivative. It never
evaluates the Fibonacci recurrence. Tests confirm that selecting the readout
preserves the training weights, optimizer and batch sampler state exactly.
The 48 Fibonacci orbits each have 196 states, with 24 complete orbits held
out. All three seeds now pass complete public verification and the site's
strict grokking criterion on the averaged evaluation weights:

| Seed | Plateau, epochs | Test 10% to 90%, epochs | First test >=99%, epoch | Final epoch | Field readout | Native argmax |
|---|---:|---:|---:|---:|---:|---:|
| 11 | 125 | 101 | 482 | 582 | 99.21% | 96.56% |
| 17 | 157 | 132 | 531 | 631 | 99.38% | 96.11% |
| 23 | 100 | 75 | 324 | 424 | 99.64% | 96.68% |

Each final checkpoint follows 100 additional epochs above 99%, and its
4,704 forecasts are rechecked through `DiffGPT.predict_many`. Query weights
and native-argmax curves remain separate in the reports. The native losses
and argmax accuracies match all 583 corresponding observations of the
seed-11 field run exactly.

```bash
# Reproduce the full-period result.
uv run --locked -m benchmarks.grokking.recurrence --device cuda \
  --task division --order 1 --context 48 --width 128 --layers 2 \
  --learning-rate 0.003 --weight-decay 1 --epochs 3000 \
  --output runs/grokking-div97-rows-period

# Verify every held-out forecast through the public API.
uv run --locked -m benchmarks.grokking.recurrence --device cuda --full \
  --verify runs/grokking-div97-rows-period/division-k1-c48-w128-l2-h4-sf-lr0.003-wd1-b20.98-wu10-s11

# Identical model, data split and initialization; only the tokenizer changes.
uv run --locked -m benchmarks.grokking.recurrence --device cuda \
  --plan benchmarks/grokking/plans/recurrence_compare.json \
  --output runs/grokking-div97-rows-tokenizers

# Reproduce positive-order Fibonacci grokking; verify every public forecast.
uv run --locked -m benchmarks.grokking.recurrence --device cuda --full \
  --plan benchmarks/grokking/plans/recurrence_field.json \
  --output runs/grokking-div97-rows-field
```

Checkpoints retain the complete optimizer, batch sampler and criterion state;
`--resume` extends the epoch budget without restarting training. New runs
require 101 consecutive observations with token and decoded test accuracy
at least 99% before stopping. `benchmarks.grokking.recurrence_readouts` saves exact
checkpoint copies for sampler comparisons; `benchmarks.grokking.recurrence_period_probe`
provides the copy controls. `benchmarks.grokking.recurrence_report` summarizes active
and completed curves and can export PNG/SVG plots with `--plot`.

`benchmarks.grokking.recurrence_report --confirmed-only` exports only results with
complete public verification and the required stable tail. Its JSON keeps
the native-argmax criterion separately from the configured sampler's curve;
the plot draws native argmax with a dashed line where the sampler differs.

`benchmarks.grokking.recurrence_audit` independently checks every saved raw state
and transition, closure of both splits under all orbit shifts, equality of
actual encoder inputs with encoding of observed rows alone, all supervised
tokens, optimizer iterates and batch sampler state. It then repeats every
public forecast and records file hashes. The seven confirmed temporal
results pass this audit; all 14 averaged/query curves match the actual
browser classifier.

The native four-head Fibonacci control was preserved at epoch 13,250 after
its last accuracy maximum at epoch 6,485. All 4,704 public forecasts replay
at 96.39%. Its complete optimizer and data sampler remain in the checkpoint,
with a hash and archived curve recorded in `deferred.json`; its compute slot
now trains short-context division with the compact vocabulary. Growing
controls continue independently.
The two-block Fibonacci C=4 control was also preserved at epoch 7,000,
and the balanced short-division control at epoch 2,750. Both retain complete
optimizer/sampler states and archived curves; their full public forecasts
were replayed before their slots were reassigned to depth and MASH warmup
experiments. The original long division baseline and growing controls continue.

Short-context division remains an open experiment. Two differences at k=1
have an information-theoretic next-token ceiling of 67.55%; three suffice
to identify every state. Current trials include larger/deeper models,
causal supervision of intermediate next-token targets, and balanced batches.
On the frozen two-block k=1/C=4 checkpoint at epoch 8,000, complete-test
sampler comparisons give 1.02% decoded accuracy with native argmax,
1.22% with range-restricted argmax and 1.04% with the field readout.
The Fibonacci sampler improvement does not resolve this short-division case.
The search prioritizes the earliest stable 99% generalization. A memorization
plateau is a diagnostic for delayed grokking, not a target to induce. New
non-unit-grid trials prioritize MASH; LGT remains a paired control.
`benchmarks/grokking/plans/recurrence_division_tokenizers.json` also compares MASH
with LGT directly on division, with k=1/C=4, width 128, two blocks and four
heads on the non-unit grid (vocabulary 512, domain 96). The paired preflight
checks every state: observed-only inputs match full-window inputs, target
tokens decode to the correct integers, and both splits and raw rows agree.
`benchmarks/grokking/plans/recurrence_compact.json` tests an exact unit grid with
vocabulary 192 and domain 95 for division. Its tokens represent the same
integer differences as the default vocabulary 256, shifted by 32 token
indices; 191 classes are used, leaving one unused class instead of 65.
All 9,216 states pass observed-only encoding and exact target-decoding checks.
The two settings use balanced batches and ScheduleFreeAdamW weight decay
1 or 3, with the same learning rate 0.01 and model sizes.
`benchmarks/grokking/plans/recurrence_depth.json` adds eight- and four-block
models at width 128 on this same compact grid, raw state space and orbit
split. Both use learning rate 0.01, weight decay 1 and balanced effective
batches of at most 512 rows, accumulated in 256-row microbatches. Each has
a 20,000-epoch budget. The eight-block model has 1,596,416 parameters;
the four-block model has 810,496. Their preflight checks all 9,216 observed
inputs and target decodes and exercises a finite forward/backward optimizer
update. The depth comparison is an experiment; its generalization is not
yet established.
`benchmarks/grokking/plans/recurrence_beta1.json` prepares a matched compact-grid
division comparison of ScheduleFreeAdamW beta1 0.5 and 0.95 against the
existing beta1 0.9 control. Only that optimizer parameter changes; default
run names and continuation of checkpoints without a beta1 config field
remain compatible. These configurations are separate from the depth trial.
The original seed-11 training set creates nine 512-row batches plus a
six-row tail. Balanced sampling uses six 461-row batches and four 462-row
batches, preserving the row order, complete corpus and ten optimizer
updates. All growing runs retain full resume state and extended budgets.

`benchmarks/grokking/plans/recurrence_full_batch.json` tests one optimizer update
over all 4,614 training rows per epoch, with a budget of 100,000 epochs.
Gradients accumulate in 512-row microbatches, weighted by their row counts;
there is one ScheduleFreeAdamW update after the complete effective batch.
The monolithic full batch exceeded the available GPU memory before its
first update; that failed attempt and its epoch-zero checkpoint are retained.
Tests compare accumulated and monolithic gradients and optimizer states,
including uneven microbatches and intermediate-target supervision, and
verify exact continuation of the full-batch optimizer and data sampler.

The current ScheduleFreeAdamW does not decay its nominal learning rate after
warmup. At the short-division epoch-8,000 checkpoint, its global step is
80,000, its base rate is still 0.01 and its averaging coefficient is
0.0000125077. The growing averaging history can slow changes to evaluation
weights even while base updates remain active.
`benchmarks.grokking.recurrence_restart` forks a frozen complete checkpoint and ramps
the learning rate again, without resetting global steps, moments, base and
evaluation iterates, batch order or RNG state. The original run continues.
The first trial branches from non-unit-grid MASH division at epoch 1,500,
raising the rate to 0.03 over 500 additional optimizer updates (50 epochs),
with an extended budget of 20,000 total epochs. It tests a higher rate with
a second warmup; this comparison alone does not isolate the ramp from the
higher peak rate. Its full CSV also records the actual nominal rate, base
rate and averaging coefficient. Historical scalar diagnostics are reconstructed
from the constant-rate parent and checked against its saved averaging total;
all accuracy/loss observations retain the original measured values.
Tests check exact fork-state preservation, source immutability and bitwise
agreement between continuous and resumed training interrupted during the ramp.

The M4 Hourly and ETTh1 measurements below use an earlier model with Gaussian
ordinal soft-CE and other additions since removed. Those forecasting datasets
have not been re-benchmarked with the current plain GPT yet.

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
- [github.com: nanochat](https://github.com/karpathy/nanochat)
- [github.com: Time-Series-Library](https://github.com/thuml/Time-Series-Library)
- [github.com: schedulefree](https://github.com/facebookresearch/schedulefree)
- [arxiv.org: Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [openai.com: Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [arxiv.org: RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [arxiv.org: Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- [arxiv.org: Query-Key Normalization for Transformers](https://arxiv.org/abs/2010.04245)
- [arxiv.org: Peri-LN: Revisiting Normalization Layer in the Transformer Architecture](https://arxiv.org/abs/2502.02732)
- [arxiv.org: Normalization in Attention Dynamics](https://arxiv.org/abs/2510.22026)
- [arxiv.org: The Road Less Scheduled](https://arxiv.org/abs/2405.15682)
- [arxiv.org: GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- [arxiv.org: Exclusive Self Attention](https://arxiv.org/abs/2603.09078)
- [arxiv.org: The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)
- [arxiv.org: Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets](https://arxiv.org/abs/2201.02177)
- [arxiv.org: iTransformer: Inverted Transformers are Effective for Time Series Forecasting](https://arxiv.org/abs/2310.06625)
