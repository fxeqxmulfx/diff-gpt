# SSMax score strength with Q/K RMS retained

This round tests **one-half and one-quarter of the current SSMax score gain**
on the best completed Q/K-normalized short-suite arm. It retains scaled
Peri-RMS, embedding RMS, centered XSA, SwiGLU, ordinary fixed RoPE, tied weights,
parameter-free normalization and LR ×2. No tensors, parameter counts, random
draws, data, budgets or optimizer settings change.

The [plan](plan.json) was declared after round two's results were committed as
`1138c62`, and before new benchmark training. Short results motivate the test:
SSMax/QK passes 75/75 at 24.3271 geometric-mean epochs; the no-QK ablation passes
75/75 at 20.9798. Removing Q/K RMS changes both magnitude sensitivity and score
scale. That comparison does not establish which caused the speed difference.
This round tests whether softer attention helps while retaining normalization.
The completed long diagnostic does not select or tune these candidates.

The equation is `m*log(N)*g_h*dot(rotated_RMS(q), rotated_RMS(k))` for visible keys.
Thus the effective SSMax coefficient is **s=0.5 or s=0.25**, replacing the
parent's fixed s=1. Q/K normalization occurs before the multiplier. Both dense
and cached attention use the same coefficient and full causal key count;
the N=1 row remains zero before masking.

The real score bound becomes `m*log(N)*abs(g_h)*head_dim`, independent of Q/K
projection-weight norms, using positive epsilon and the checked
[normalized-score contract](../early99/guarantees/NormalizedScores.lean).
These two candidates tighten that bound by factors of two and four. They
preserve the existing residual-state and normalization bounds. Neither this
bound nor a finite-precision stress pass guarantees learning, a useful margin,
whole-network conditioning or generalization to larger tasks.

There is also a concentration tradeoff. The existing checked
[SSMax margin theorem](../early99/guarantees/ScalableSoftmax.lean) gives
`p(target) >= 1 / (1 + (N-1)*N^(-m*delta))` if its unscaled score exceeds every
competitor by at least `delta`. Reducing m weakens this lower bound for the
same gap. For a fixed positive gap, `m*delta > 1` is sufficient for this bound
to approach one as context grows: half-scale requires `delta > 2`, quarter-scale
requires `delta > 4`. These are sufficient conditions for the bound, not
necessary conditions for learning or an assumption that training supplies
the gap. Smaller bounded scores alone do not establish a better universal model.

The screen keeps the six existing short tasks and all three seeds. Advancement
requires 18/18 passes and at least three improved task median epoch ratios
**versus the Q/K parent**. This stronger parent comparison is declared before
training. Every promoted arm must complete all 75 short cells. The shared-model
choice can improve only with full coverage and a lower complete geometric mean
than the parent; baseline, LR ×2 and no-QK comparisons remain visible separately.
Every miss is retained and no filtered geometric mean is allowed. A new shared
winner, if selected, receives one reused long diagnostic after freezing choice.

Implementation and validation:

- [factories.py](factories.py) changes only the fixed gain and records it in the
  configuration. It adds no learned parameters and keeps every original matrix.
- [run.py](run.py) calls the unchanged original trainer with explicit factories,
  and extends its source archive to include this package. It rejects changed
  configuration, execution settings or archived sources.
- [stress.py](stress.py) compares 90 normalized attention rows with an independent
  FP64 forward/backward reference: widths 8/32/128, visible lengths through 4096,
  Q/K scales 1e-9/1/1e9, and both coefficients. This isolates score arithmetic;
  it does not replace the earlier full-stack extreme-magnitude tests or conceal
  their unresolved failures.
- The [implementation tests](../../tests/diff_gpt/application/test_early99_score_scale.py)
  check independent equations and all attention gradients, singleton/clipped XSA,
  unchanged initialization/RNG/capacity, causality and cache modes, source/archive
  guards, and CUDA graph updates against uncaptured native AdamW.

Reproduce pretraining checks from the repository root:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. .venv/bin/pytest -q tests/diff_gpt/application/test_early99_score_scale.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. .venv/bin/python -m research.early99_score_scale.stress
```

The frozen configurations are under [configs](configs). Use the isolated runner
with `--configuration PATH --write-plan`, then `--run --tier short --case NAME`
for each declared screen task. `--run --tier short` completes all 75 cells only
for a promoted arm; it reuses already completed screen cells. Preserve the
source snapshot before training and use a new output directory for any change.

The scientific source is frozen at **`dfecdc9`**. All **9 CPU/CUDA checks** passed,
and all **90 score probes** passed the predeclared bounds and accuracy limits.
Maximum probability error was `6.4692e-7`; maximum relative gradient error was
`1.5498e-5`. The [validation record](results/validation.json) and
[complete stress rows](results/score-stress.json) preserve the evidence and scope.
The half-scale screen is complete; the quarter-scale screen is running.
No complete-suite learning improvement is established yet.

Run both screens or regenerate their four-control reports with
`python -m research.early99_score_scale_reporting.run_screens --run` or
`--report`, using the environment shown above. Do not launch a duplicate active
driver. The [checkpoint auditor](../early99_score_scale_reporting/audit.py) accepts
`--name NAME --output PATH` for a screen, `--full` for 75 short cells, or
`--full --long` for 105 cells after a selected winner's diagnostic.

## Half-scale screen

The half-scale arm passes **18/18** and improves **4/6 paired task medians**
versus the Q/K parent, meeting the declared promotion gate. It has **13 earlier,
4 tied and 1 later** cells. The one slowdown is linear transport, with a worst
paired ratio of 1.5. No screen geometric mean is reported.

| Task | Parent median epochs | Half-scale median epochs | Median paired ratio | Worst paired ratio |
|---|---:|---:|---:|---:|
| Boolean AND | 3 | 2 | 1.0000 | 1.0000 |
| Division p193 | 35 | 27 | 0.8571 | 1.0000 |
| Division p97 | 28 | 16 | 0.5357 | 0.5714 |
| Key-value binding | 16 | 11 | 0.7500 | 0.8462 |
| Square sum | 311 | 218 | 0.7554 | 0.7878 |
| Linear transport | 2 | 3 | 1.0000 | 1.5000 |

The gate uses the median of **paired seed ratios**, which need not equal the
ratio of the two separate medians. All rows and the baseline/LR ×2/no-QK
comparisons are retained beside the
[parent screen](results/screen-peri-rms-scaled-ssmax-gain-half-lr2-vs-parent.json).
The [checkpoint audit](results/audit-peri-rms-scaled-ssmax-gain-half-lr2-screen.json)
verifies all 18 cells and 31 archived sources, with no resumptions, execution
errors or capacity changes. Complete the quarter-scale screen, then run every
promoted arm on all 75 short cells with these same frozen sources.
