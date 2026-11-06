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

The new winner now requires its single already observed 30-cell long diagnostic,
after the short decision is committed. Compare the quarter-scale parent, LR ×2,
baseline and no-Q/K controls. Preserve all budget misses and do not use long
results to tune or reselect the model.

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
