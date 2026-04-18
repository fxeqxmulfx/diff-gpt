# diff-gpt

Autoregressive time-series forecasting with a decoder-only transformer trained on
a derivative-based tokenization. The encoder differentiates the input, scales by
its observed range, and quantizes with a MASH (multi-stage noise-shaping)
$k$-th-order $\Sigma\Delta$ modulator; the decoder inverts this. The model is
a standard GPT — the inductive bias sits entirely in the tokenization.

See [papers/paper.pdf](papers/paper.pdf) (English) and [papers/paper_ru.pdf](papers/paper_ru.pdf)
(Russian) for the full write-up with proofs and benchmarks.

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
| ordinal smoothing $\sigma^\star$ | $\times 2$ (hold value-space constant) | §4 |

**Theorem 1 (MASH reconstruction, any $k$).** For any signal with
$D = \max_t |\Delta^k x_t| > 0$ and $V > 2^k + 2$,

$$\|\hat x - x\|_\infty \le \frac{D}{V - 2 - 2^k} = \frac{\Delta}{2}$$

uniformly in horizon $T$. Vocabulary doubling halves reconstruction error at any
order $k$.

**Proposition 2 (achievable CE).** In the fine-quantization limit,
$\mathcal L^\star(V) = h(X_t \mid \text{ctx}) + \log V - \log(2D) + o(1)$.
Ideal loss is linear in $\log V$, slope 1. Cross-vocab comparisons must subtract
out $\log V$ to get the vocab-invariant *differential-entropy residual*
$\tilde h(V) = \mathcal L(V) - \log V + \log(2D)$.

## Key features

- **MASH $k$-th-order $\Sigma\Delta$**: stable noise-shaping at arbitrary order $k$,
  tight $\Delta/2$ reconstruction bound uniformly in $T$. Falls back to the
  classical carry scheme when $V \le 2^k + 2$.
- **Gaussian ordinal soft-CE**: replaces one-hot targets with a discrete Gaussian
  over bin indices, $p^\text{soft}_i \propto \exp(-(i - y)^2 / 2\sigma^2)$.
  Teaches the model that "off by one bin is almost right" — matches the ordinal
  structure of derivative-quantized tokens. On M4 Hourly: sMAPE $18.58 \to 17.49$
  at $\sigma = 2$ with no architectural change.
- **Per-column derivative order**: `order_of_derivative` can be an array of shape
  $(F,)$ giving each channel its own $k_c$. Mix $k = 0$ (raw) with $k = 1, 2, \ldots$
  in a single model.
- **GPT architecture niceties**: RoPE, RMSNorm (no learnable scale), SwiGLU with
  soft limit, tied embeddings, logit softcap, Block Attention Residuals (Chen et
  al. 2026), Schedule-Free AMSGrad optimizer.
- **Inference engine**: KV-cache-accelerated generation, position-aware
  `force_schedule` for teacher-forced decoding (TimeXer-style exogenous
  covariates), best-val checkpoint auto-restore.

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
    inp=all_data, order_of_derivative=order, use_decimal=False,
)

# Construct GPT + DiffGPT wrapper.
base = GPT(
    vocab_size=256,
    n_embd=64,
    block_size=138,
    n_head=4,
    n_layer=2,
    label_smoothing_sigma=2.0,  # ordinal soft-CE
)
model = DiffGPT(
    model=base,
    order_of_derivative=order,
    domain_of_definition=domain,
    use_decimal=False,
)

loader = DiffDataFrameDataLoader(
    dfs=dfs,
    block_size=138,
    batch_size=32,
    vocab_size=256,
    order_of_derivative=order,
    domain_of_definition=domain,
    use_decimal=False,
    device="cuda",
    train_part=0.8,
)

model.train(loader=loader, max_iters=20_000, eval_interval=500)

# Forecast.
context = dfs[0].iloc[-96:]
prediction = model.predict(
    df=context,
    max_new_points=48,
    sampler=TemperatureSampler(temperature=0.0),  # argmax
)
```

## Benchmarks

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

## Install & test

```bash
uv sync
uv run --no-sync pytest
```

## References

- [youtube.com: Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [github.com: nanoGPT](https://github.com/karpathy/nanoGPT)
- [github.com: gpt-oss](https://github.com/openai/gpt-oss)
- [github.com: gemma_pytorch](https://github.com/google/gemma_pytorch)
- [github.com: schedule_free](https://github.com/facebookresearch/schedule_free)
- [github.com: nanochat](https://github.com/karpathy/nanochat)
- [github.com: Attention-Residuals](https://github.com/MoonshotAI/Attention-Residuals)
- [github.com: Time-Series-Library](https://github.com/thuml/Time-Series-Library)
- [arxiv.org: Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [arxiv.org: Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- [arxiv.org: RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [arxiv.org: GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- [arxiv.org: The Road Less Scheduled](https://arxiv.org/abs/2405.15682)
- [arxiv.org: The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)
- [arxiv.org: Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free](https://arxiv.org/abs/2505.06708)
- [arxiv.org: iTransformer: Inverted Transformers are Effective for Time Series Forecasting](https://arxiv.org/abs/2310.06625)
- [arxiv.org: Rethinking the Inception Architecture for Computer Vision (label smoothing)](https://arxiv.org/abs/1512.00567)
