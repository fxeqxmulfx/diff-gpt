# Two-token attention bounds

The [four checked Lean statements](Bounds.lean) show a concrete restriction of
Q/K-normalized inverse-width SSMax. If each score has magnitude at most `B`,
either probability of a two-key softmax lies in

`[1 / (1 + exp(2B)), exp(2B) / (1 + exp(2B))]`.

Finite-epsilon Q/K RMS bounds each vector's norm by `sqrt(d)`. The previously
proved stored-RoPE bound contributes a factor `M`. Thus the inverse-width
`log(N)/d` score has magnitude at most `M*log(N)`, independently of head width.
For two keys and ideal rotations (`M = 1`), the probability interval is exactly
**[1/5, 4/5]**. Each key receives at least 20% of the attention in this real model.

The [application record](results/application.json) checks all four declared
capacity shapes. Their context is two, head width is 32, weights remain tied,
and their actual stored rotary tables match the prior exact rational certificate.
With the certified stored-pair bound `M = 1 + 2^-20`, the inverse-width upper
probability bound is approximately **0.8000002115**. For comparison:

| Attention rule, head width 32 | Approximate upper probability bound |
| --- | ---: |
| Inverse-width SSMax | 0.8000002 |
| Usual-gain SSMax | 0.9996073 |
| Original softmax | 0.9999878 |

The application uses the actual FP32 gain as a real constant. Its decimal
endpoints approximate the real envelope; they do not certify rounding throughout
an actual forward pass. The usual gain retains the parent's existing
`exp(log_alpha)/d` calculation, representing `1/sqrt(d)`.

The per-head cap persists in the running depth/width study. Additional layers,
heads, value projections, XSA, residual paths and readout capacity can still help.
This bound does **not** show that sharp attention is necessary for cubic-sum,
that the complete network cannot solve it, or that Peri/embedding normalization
is wrong. It motivates the separately declared attention-gain test while retaining
those normalizations and tied embedding/readout weights.

The [proof record](results/guarantees.json) verifies all four statements with
standard axioms only and no warnings. The first unsuccessful compiler attempt
is retained with source text: unused simplification rules and two numeral
normalizations were corrected without changing the statements. There were no
optimizer updates, model forwards or new held-out scores in this supplement.

Reproduce from the repository root:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m research.two_token_attention_bounds.application
```

The proof checker runs under `lake env` in the external transformer project,
with `PYTHONPATH` set to this repository and an absolute path to `check.py`.
It writes compiled files to this workspace, leaving the proof library unchanged.
