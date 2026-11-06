# Initialization scale before finite-epsilon RMS

The user asks whether different model blocks need different initialization rules.
This [plan](plan.json) isolates a mathematical issue before choosing an initializer:
scaling a linear map immediately before RMS can change its parameter gradients
even when its normalized forward values barely move.

The existing Lean library already proves the exact identity
`R_eps(s*z) = R_(eps/s²)(z)` for positive `s`. The [four checked statements](Bounds.lean)
extend this to a fixed linear parameter map `A`, a downstream scalar loss, and
physical-parameter derivatives. If `F_eps(w) = loss(R_eps(A(w)))`, then

```text
F_eps(s*w) = F_(eps/s²)(w)
D F_eps(s*w) = (1/s) D F_(eps/s²)(w).
```

The second identity explicitly assumes the counterpart's derivative exists.
The Lean file includes a nonconstant finite-epsilon example. The epsilon change
is essential: this is not exact scale invariance at the original epsilon, or a
statement that two optimizers follow the same trajectory.

For mean-square input energy `m`, the local RMS tangent coefficient is
`1/sqrt(m+eps)` and the radial coefficient is `eps/(m+eps)^(3/2)`.
Their ratio is `eps/(m+eps)`. Increasing the input scale to `s>=1` cannot increase
this ratio: `eps/(s²*m+eps) <= eps/(m+eps)`. Thus smaller gradient amplification
can accompany more unequal directional sensitivities. Neither property alone
predicts whether learning improves.

All **96/96 numerical cases and 1,152 comparisons pass**. The grid covers widths
32/128/256, four amplitudes, four positive scales, and CPU FP32/FP64, retaining
the actual FP32 epsilon in both precisions. Each case checks two independent
physical matrix leaves, their forward outputs, and radial, tangent and mixed
parameter gradients against independent binary64 formulas. Maximum relative
error is **5.9604643e-8**, below the declared 0.001 threshold. The
[full results](results/runtime.json) and [96-row table](results/scaling-summary.csv)
retain every case.

For the synthetic width-128 FP32 input with RMS approximately 0.02:

| Scale | Forward amplitude / unscaled amplitude | Tangent coefficient | Radial coefficient |
| ---: | ---: | ---: | ---: |
| 0.25 | 0.997773 | 199.525 | 0.946894 |
| 1 | 1 | 49.9926 | 0.0148945 |
| 4 | 1.000140 | 12.4999 | 0.000232824 |
| 16 | 1.000148 | 3.12500 | 0.00000363797 |

These same-epsilon coefficients illustrate the tradeoff. At scale four, the
normalized amplitude moves only about 0.014%, while the tangent coefficient
falls about fourfold and the radial coefficient about 64-fold. These are local
synthetic measurements and formulas, not observations of a trained transformer.

All four statements compile without warnings or extra axioms. Five earlier
attempts failed while elaborating the concrete examples; one also had a linter
warning. Every source and compiler output is retained. The
[complete audit](results/complete-audit.json) verifies that the four exported
statements and their proofs stayed unchanged across all six attempts.

The linear input map, other parameters and downstream loss are fixed here.
This identity cannot describe the whole tied embedding/readout path when both
uses change together, or arbitrary rescaling inside SwiGLU. The moderate-scale
CPU checks do not resolve the ten existing full-stack numerical stress failures.
No optimizer updates, task evaluations or initializer selection occurred.
The capacity grid and queued usual-gain experiment retain their frozen sources.

```sh
# From the Lean project, with all output directed to the GPT workspace:
PYTHONPATH=/home/misha/python_projects/gpt lake env python3 /home/misha/python_projects/gpt/research/rms_initialization_scaling/check.py
# From the GPT repository:
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m research.rms_initialization_scaling.check_runtime > research/rms_initialization_scaling/results/runtime-output.txt 2>&1
PYTHONPATH=. .venv/bin/python -m research.rms_initialization_scaling.report
```
