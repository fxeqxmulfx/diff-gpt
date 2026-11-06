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

Benchmark training is pending. The unchanged screen uses six tasks × three seeds
and requires 18/18 passes and improved paired medians on at least three tasks
versus the quarter-scale parent. Three tasks have unchanged width-16 gains; the
other three have width 32. No screen geometric mean is used. Only a promoted
candidate receives the remaining short cells, and only all 75 successes plus a
lower complete geometric mean can select it. A new short winner receives one
already observed long diagnostic after its decision is committed. Otherwise the
parent and its existing long result are retained without another long run.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. .venv/bin/pytest -q tests/diff_gpt/application/test_early99_dimension_gain.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. .venv/bin/python -m research.early99_dimension_gain.stress
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. .venv/bin/python -m research.early99_dimension_gain_reporting.run_screens --run
```

The runner requires committed source, configurations and passing evidence before
any training update. It preserves the original trainer/scorer and archives all
inherited scientific sources. Reports live in a separate package so they can be
completed without changing the frozen training implementation.
