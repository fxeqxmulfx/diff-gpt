# SSMax with inverse head-width gain

This [separately declared trial](plan.json) keeps Q/K RMS and changes only the
fixed gain from the selected quarter-scale parent's `(1/4)*exp(log_alpha)/d`
to **`1/d`**. The resulting real score equation is

\[
z_{ij}=\frac{\log(i+1)}d\langle R_i\operatorname{RMS}(q_i),
R_j\operatorname{RMS}(k_j)\rangle,\qquad j\le i.
\]

The original matrices, initialization draws, parameter ordering/count, frozen
buffers, Q/K RMS, ordinary fixed RoPE, centered XSA, SwiGLU, scaled Peri-RMS,
embedding RMS, tied weights, LR ×2 and native AdamW remain intact. A deterministic
nonlearned buffer stores the new gain; the original `log_alpha` buffer is retained.
Dense, tokenwise, chunked and shared-prefix attention use the original forward
implementation and global visible-key counts. No residual-decay or activation
change from the independent stress study is combined with this trial.

The motivation is the complete short comparison and the width-dependent score
bounds. No long-suite result is used for design or selection. At head width 16
the stored gain equals the quarter-scale parent exactly. It reduces the ideal
score envelope for widths above 16 and increases it below 16; it is not a
uniformly tighter bound at every width.

The [13 pretraining checks](results/validation.json) passed, including independent
forward and all parameter/input gradient equations, clipped XSA, singleton rows,
causality, cache paths, unchanged initialization/RNG/capacity, archive tampering,
width-16 gain equality, and two native CUDA graph update comparisons with short
batches and unchanged optimizer moments/RNG. The
[120 declared score probes](results/score-stress.json) all passed. Maximum
probability absolute error was **5.22919e-8** and maximum relative gradient error
**1.48669e-4**, within the fixed 2e-6 and 1e-3 limits. These isolate score arithmetic
at widths 2/4/8/16/32/128, visible counts 1/2/32/512/4096 and scales 0/1e-9/1/1e9.
They do not repair the previously recorded full-stack extreme-input failures.

[Four Lean statements](Guarantees.lean) compile without warnings or extra axioms,
with concrete examples using FP32 epsilon. For positive width and epsilon in
real arithmetic, they establish:

- Absolute score at most `log(N)`, independent of width.
- Per-key probability at most `1/(1+(N-1)/N²)`. This is a selectivity ceiling;
  it does not ensure concentration or a positive margin.
- The algebraic inverse-width sensitivity coefficient.
- Complete probability-row query sensitivity at most
  `log(N)/(sqrt(d)*sqrt(epsilon))`, holding projected keys and rotary phases fixed.

The [proof record](results/guarantees.json) binds sources, compiler output, exact
external Lean/mathlib revisions and the reviewed implementation. These are not
whole-network, finite-precision, generalization or convergence theorems. The exact
real gain and rounded stored gain are distinguished; all tested widths are powers
of two, so `1/d` is exactly representable there. The earlier general theorems
still cover arbitrary fixed rounded gains. Compile the record's dependency order
to a common temporary Lean import root before checking `Guarantees.lean`.

The screen completed **18/18 passes**, with **four improved paired task medians**
against the quarter-scale parent. It passes the unchanged three-improved-task
gate. Source and pretraining evidence were committed as `a5aaa89` before training.
All 18 checkpoints and 34 archived sources passed the [audit](results/audit-screen.json),
with no execution errors or resumptions. All four control comparisons were
[reconstructed](results/screen-integrity.json) from raw histories and metadata.

| Screen task | Head width | Median paired epoch ratio vs quarter-scale |
|---|---:|---:|
| Boolean AND | 16 | 1.0000 |
| Division p193 | 32 | 0.8235 |
| Division p97 | 32 | 0.9375 |
| Key-value binding | 16 | 1.1000 |
| Square sum | 32 | 0.9021 |
| Linear transport | 16 | 0.8000 |

There are 11 earlier, four tied and three later paired cells versus the parent.
Preserve the regressions: key-value binding seeds 1/2 take 16→20 and 70→77 epochs;
linear transport seed 1 takes 4→5. Against LR ×2 the worst screen slowdown is
key-value binding seed 2, 4→77 epochs (19.25×). No screen geometric mean is used.

The width-16 cases exposed a numerical limitation of the expected unchanged
controls. Their fixed gains are exactly equal, but their learning trajectories
can differ. A [read-only CUDA probe](results/width16-probe.json) of all three
seed-0 first batches found identical eager losses and gradients, while compiled
parent/candidate gradients differ by up to 3.71223e-7 in relative L2 norm.
Even repeated compiled calls within one factory differ, up to 1.01934e-7 here.
No optimizer updates were issued by the probe. These observations establish
numerical differences and nondeterminism, not a causal replay of entire training
trajectories. Matched seeds do not supply bitwise training reproducibility.
All three width-32 task medians improve, so promotion does not rely on the
width-16 linear-transport improvement. The original declaration and every
observed row remain intact.

The full short suite completed **75/75**, selecting inverse-width SSMax under
the original policy. Its geometric mean is **16.2660501255 epochs**, versus
17.3257083726 for quarter-scale: **6.12% fewer epochs**. This is an observed
three-seed benchmark result, not a deterministic or universal convergence claim.

| Reference, all 75 passed | Candidate/reference GM epoch ratio | Earlier / tied / later | Worst paired slowdown |
|---|---:|---:|---:|
| Quarter-scale Q/K parent | 0.938839 | 40 / 22 / 13 | 4.3333× |
| LR ×2 | 0.474922 | 48 / 10 / 17 | 19.25× |
| Original baseline | 0.284544 | 55 / 6 / 14 | 15.4× |
| No-Q/K ablation | 0.775320 | 57 / 7 / 11 | 4.6667× |

Preserve every regression. Against the parent, depth-hard seed 0 takes 1.2→5.2
epochs and permutation-addition seed 0 takes 108→371. Against LR ×2, key-value
binding seed 2 takes 4→77. Against no-Q/K, branch-amplitudes seed 1 takes 3→14.
All paired rows and per-task medians/worst values are in the four `*-short.json`
and CSV comparisons under [results](results/).

A supplemental grouping of the complete 75 rows by head width shows a 0.937374
ratio on all 57 width-32 cells (36 earlier, 13 tied, eight later), and 0.943494
on all 18 width-16 cells (four earlier, nine tied, five later). These are complete
groups, not filtered successes, and were not used to change the selection rule.
The numerical caveat for equal width-16 gains remains.

Parameter counts remain unchanged at 100,736–1,183,488. Recorded trainer time is
1,205.27 seconds, versus the parent's 1,133.38; time does not determine promotion.
All 75 checkpoints and 34 archived sources passed the full audit, with zero
errors or resumptions. The [selection](results/shared-selection.json) and
[integrity manifest](results/selection-integrity.json) reconstruct every full
comparison and freeze the short-only choice. The checks inspect saved weights,
optimizer states, histories and provenance; they do not replay all full-GPT
held-out forwards.

The short decision was committed as `dc4411b` before the single reused long
regression diagnostic launched at **2026-11-06 04:26:05 UTC**. It completed
**19/30 passes**, with **11 original-budget misses**. This is worse long coverage
than quarter-scale (21/30), LR ×2 (26/30), the original baseline (29/30), and
no-Q/K (20/30). The short-only selection remains frozen; these results do not
establish a universal improvement or change subsequent optimizer rates.

| Long control | Passes | New misses / recoveries / joint misses | Common passes: earlier / tied / later | Worst observed successful ratio |
|---|---:|---:|---:|---:|
| Quarter-scale parent | 21/30 | 3 / 1 / 8 | 6 / 1 / 11 | 1.5714× |
| LR ×2 | 26/30 | 7 / 0 / 4 | 12 / 0 / 7 | 2.8559× |
| Original baseline | 29/30 | 10 / 0 / 1 | 12 / 0 / 7 | 9.3235× |
| No-Q/K ablation | 20/30 | 1 / 0 / 10 | 11 / 0 / 8 | 1.8538× |

All three fourth-power and all three cube seeds miss 100,000 updates. The other
misses are recall-easy seed 0 (20,000), Fibonacci seeds 0/2 (100,000), and
parity24 seeds 0/2 (100,000). Relative to quarter-scale, fourth-power seed 0,
recall-easy seed 0 and parity24 seed 2 become misses; parity24 seed 1 recovers at
45.8 epochs. The worst common-pass slowdown is associative-divisor seed 0,
7→11 epochs. Against LR ×2, parity20 seed 1 takes 22.2→63.4 epochs. Every cell,
final accuracy, censoring budget and task median is retained in the
[complete long report](../early99_dimension_gain_recovery/results/long-summary.json)
and its four JSON/CSV comparisons. No filtered long geometric mean is reported.

The pause/resume interruption required recovery of recall-easy seed 0 from its
1,000-update checkpoint, while preserving all 18 previously completed long
cells. Its discarded history reaches update 1,050 without a crossing. Recovery
therefore repeated at least 50 updates / 3,200 examples / 0.16 epochs. The final
canonical trajectory is 63.8992 epochs; actual executed exposure is at least
64.0592 epochs. Further unobserved work before process loss cannot be counted
exactly. Both raw segments, the old checkpoint/history, and all hashes are
retained. Summed observed trainer segments total 4,782.73 seconds for the long
suite; unobserved time before process loss is excluded and resumed time ratios
are omitted.

The frozen original auditor rejected resumptions as expected; its error is
preserved. A separate [recovery-aware audit](../early99_dimension_gain_recovery/results/complete-audit.json)
verifies all 105 checkpoints (94 passes, 11 misses), the unchanged 34-source
experiment plan, all prior completed-cell hashes, the restored history prefix,
and the four complete long comparisons. It records one resumption and zero
training execution errors. The original launcher, plan and short selection
hashes remain unchanged. This diagnostic is complete; do not restart it.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. .venv/bin/pytest -q tests/diff_gpt/application/test_early99_dimension_gain.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. .venv/bin/python -m research.early99_dimension_gain.stress
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. .venv/bin/python -m research.early99_dimension_gain_reporting.run_screens --run
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. .venv/bin/python -m research.early99_dimension_gain_reporting.run_full --run
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. .venv/bin/python -m research.early99_dimension_gain_reporting.verify_results --freeze
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. .venv/bin/python -m research.early99_dimension_gain_reporting.run_long --run
```

The runner requires committed source, configurations and passing evidence before
any training update. It preserves the original trainer/scorer and archives all
inherited scientific sources. Reports live in a separate package so they can be
completed without changing the frozen training implementation.
