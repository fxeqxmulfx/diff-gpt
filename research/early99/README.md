# First-99% benchmark

This is the user's fixed list of 35 tasks from the current-GPT sweep. The
historical seed-1 time to first 99% held-out exact accuracy selects two disjoint
tiers: **25 short tasks** at most 60 seconds, and **10 long tasks** above 60 and
at most 300 seconds. The three constant-output controls are excluded.
`suite.json` records every task, dataset fingerprint, model size, batch, budget,
reference crossing and training-set size. Tier membership stays fixed.

Every task runs model/shuffle seeds **0, 1, 2**, with fixed data seed 1. The short
suite has 75 cells; long has 30; their union has 105. Optimization uses the short
suite. The long suite is a subsequent regression check.

**Round 2 is complete; its short selection remains frozen.** All eight promoted candidates
completed 75/75 passes. The declared coverage/epoch rule selects
`peri-rms-scaled-ssmax-noqk-lr2`: geometric mean **20.9798 epochs**, **38.7%
fewer** than the LR ×2 incumbent and **63.3% fewer** than the original baseline.
The [frozen selection](results/round2/selection.json) and
[complete short summary](results/round2/final-short-summary.json) retain all
eligible arms, rejected screens, regressions and mathematical tradeoffs.
Its one reused long diagnostic passes **20/30**, versus **26/30 for LR ×2**
and **29/30 for baseline**, with six new budget misses versus LR ×2. The
short winner has substantial long regressions; no long geometric mean is
reported and no reselection or tuning uses those outcomes.
**Q/K RMS remains in the shared-model research
candidate**; the small-benchmark winner is an ablation, not a universal-model
recommendation. SSMax/QK is the strongest completed Q/K-normalized arm at
**24.3271 epochs**, **29.0% fewer** than LR ×2.

The [reporting audit](../early99_reporting/README.md) reconstructs every screen
decision and full short comparison from raw histories, verifies JSON/CSV and
frozen hashes, and has checked both complete long comparisons and their summary.
The [round-two report](ROUND2_REPORT.md) collects the complete short table,
rejected screens, mathematical tradeoffs, all long failures and the 105-cell
checkpoint audit. The user-requested next round will use short evidence and
mathematics, preserving the original suite and all completed results.

The [next declared study](../early99_score_scale/README.md) is now running two
short screens with **Q/K RMS retained**, reducing the SSMax score coefficient
to one-half or one-quarter. All nine CPU/CUDA checks and 90 score-stress cases
passed before training. Its independently declared gate compares task medians
with the Q/K parent, while the completed round-two selection remains unchanged.

Success is **one exhaustive held-out observation with at least 99% exact
sequences**. Delay and final retention do not enter this criterion. Numeric
probes now use exact generated vectors, not their former MSE target. Validation
uses the original frozen primary held-out split; final test, where present, is
reported by the underlying runner but does not alter the first-crossing rule.

Epochs are actual training examples presented divided by training-set size.
The counter includes the short final batch and the documented zero-rate first
update. CUDA capture warmup consumes no scientific steps. Observation is every
`min(original_interval, updates_per_epoch)` updates, so the first-crossing bracket
is at most one epoch wide. This is finer than the historical sweep: compare
candidates against a fresh three-seed baseline with this same schedule.

The original update budgets remain fixed. Historical time thresholds select
tasks; they do not impose wall-clock timeouts on a new run. Setup/compilation
and evaluation overhead remain visible in recorded seconds, while epochs are
the optimization measure. A resumed run's clock resets; its epoch counter does
not. Misses remain budget-censored. Per-task median/worst and aggregate geometric
mean require all expected seeds to pass.

All previous sweep processes were stopped at the user's request; the old queue
remains stopped. The fresh short baseline is now complete: **75/75 passed**.
Historical seed-1 crossings in the manifest remain references, not comparison
baselines.
[`historical_seed1.csv`](historical_seed1.csv) gives the available epoch counts;
[`validation.json`](validation.json) records the harness checks.

```bash
# Plan and reports do not train.
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. .venv/bin/python -m research.early99.run --write-plan
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. .venv/bin/python -m research.early99.run --report --tier short

# Explicit training commands for the next authorized experiment.
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. .venv/bin/python -m research.early99.run --run --tier short
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. .venv/bin/python -m research.early99.run --run --tier long

.venv/bin/pytest -q tests/diff_gpt/application/test_early99_benchmark.py
```

The dispatcher uses a fresh process per case/seed, resumes its own checkpoints,
and stops on execution errors. `--case NAME` runs all three seeds for one case;
`--case NAME --seed N` is an explicit single-cell diagnostic. Use `--tier all` for
the union. Outputs go to `runs/early99/baseline`, with archived suite/driver
sources, per-seed `early99_result.json`, and `summary-short.json/.csv` (or long/all).
An existing output rejects changed suite/driver sources. The baseline reuses
the verified public GPT/native AdamW runner; it preserves its frozen model check.

The optimization handoff is [`ASTRA_TASK.txt`](ASTRA_TASK.txt). Candidate
model/optimizer factories are in [`experiments/factories.py`](experiments/factories.py),
with separate output directories and recorded configurations. The original
baseline runner, model reference, manifest and data/scoring code are unchanged.
The candidate loop reuses the original sampler, stepper, evaluators and stopping
logic. Exact CPU checkpoint equivalence and CUDA capture/update checks cover
weights, moments, RNG, short batches, zero-rate first updates and recapture.

## Round 1: measured short-suite result

The round-1 selected candidate is [`configs/adamw-lr2.json`](configs/adamw-lr2.json): multiply
every task's native AdamW learning rate by **2**. Betas remain `(0.9, 0.98)`, epsilon
`1e-8`, weight decay `0.1`, and warmup 50 completed updates. Model architecture,
initialization, parameter counts, dtype and all task/scoring settings match the
baseline. This is the best of the baseline and one candidate tested, not an
exhaustive search.

| Short suite, all 75 paired cells | Baseline | AdamW LR ×2 |
| --- | ---: | ---: |
| Exact held-out first-99 coverage | 75/75 | 75/75 |
| Geometric mean first-99 epochs | 57.1653 | 34.2499 |
| Summed trainer seconds | 2066.37 | 1297.23 |

The geometric mean paired epoch ratio is **0.599139**: **40.1% fewer epochs**.
There are **55 earlier, 7 tied and 13 later** cells. The worst regression is
`power-division-p193-k2`, seed 2, at **5.8×** baseline epochs. Key-value recall and
some larger-prime division tasks also regress. Per-task three-seed medians and
worst cases, all paired ratios and observation brackets are retained in
[`results/adamw-lr2-short.json`](results/adamw-lr2-short.json),
[`cell CSV`](results/adamw-lr2-short.csv) and
[`task CSV`](results/adamw-lr2-short-tasks.csv).

Trainer seconds include model/data setup, compilation, exhaustive scoring and
final test scoring; they exclude dispatcher/process startup and the separate CPU
data preflight. No measured cell was resumed. Parameter counts are unchanged
in every pair, ranging from 100,736 to 1,183,488 depending on the fixed task.

The 15-cell screen and its regressions are in
[`results/screen-adamw-lr2.json`](results/screen-adamw-lr2.json). The selection was
frozen in [`results/selection.json`](results/selection.json) before starting any
long-suite training. The long check is complete: one fresh baseline and one
candidate run for each of its 30 cells, with no tuning against that check.

## Round 1: long-suite regression check

Coverage **regressed from 29/30 baseline passes to 26/30 candidate passes**.
The candidate introduced four budget misses, each at 100,000 updates
(5263.1633 epochs):

| Task | Censored candidate seeds |
| --- | --- |
| `arithmetic-sum-pow4` | 0, 1 |
| `arithmetic-sum-cube` | 1, 2 |

It recovered the baseline miss on `recurrence-fibonacci-k1-c2-lgt`, seed 0:
the candidate crossed at 3577 epochs, while baseline exhausted 10,000 epochs.
Among the 25 pairs in which both runs passed, 13 candidate crossings were earlier
and 12 were later. The worst observed paired ratio was 3.2647× on `basis-parity20`,
seed 1. The five comparisons containing a censored run are retained, and **no long
paired geometric mean is reported**. Summed trainer time was 3628.01 seconds for
baseline and 3238.05 seconds for the candidate, including the budget misses.

The short-suite improvement therefore comes with measured long-suite coverage
and epoch regressions. The selected configuration was not changed after this
check. All 30 comparisons, three-seed task statistics and censoring bounds are in
[`results/adamw-lr2-long.json`](results/adamw-lr2-long.json),
[`cell CSV`](results/adamw-lr2-long.csv) and
[`task CSV`](results/adamw-lr2-long-tasks.csv). The compact combined result is
[`results/summary.json`](results/summary.json).

All **210 round-1 scientific runs** completed their first crossing or original budget.
The final audit checked all 105 pairs, preserved checkpoints/histories/source
snapshots, optimizer update counts and unchanged model/task/candidate provenance.
There were no execution errors or resumed clocks. Validation details are in
[`results/validation.json`](results/validation.json).

```bash
# Recompute comparisons from the preserved round-1 histories.
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. .venv/bin/python -m research.early99.experiments.compare --candidate runs/early99/candidates/adamw-lr2 --tier short
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. .venv/bin/python -m research.early99.experiments.compare --candidate runs/early99/candidates/adamw-lr2 --tier long

# Full integrity, oracle, CPU equivalence and CUDA capture checks.
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. .venv/bin/pytest -q tests/diff_gpt/application/test_early99_experiments.py tests/diff_gpt/application/test_early99_benchmark.py tests/diff_gpt/application/test_current_grokking.py tests/diff_gpt/application/test_current_grokking_adapters.py
```

All raw histories, checkpoints, source snapshots and fingerprints remain under
`runs/early99/baseline` and `runs/early99/candidates/adamw-lr2`.

## Round 2: further optimizer experiments

The [predeclared round-2 plan](results/round2/plan.json) tests two separate changes
from LR ×2: increase the rate multiplier to **3**, or retain LR ×2 and reduce
AdamW **beta2 to 0.95**. Both use beta1 0.9 and retain the original warmup,
weight decay, model, data, seeds, cadence and budgets. The six-task screen uses
all three seeds and requires 18/18 passes plus improved median paired epochs on
at least three tasks relative to LR ×2. No screen-only aggregate is reported.

Both configurations passed 18/18 screen cells. LR ×3 improved three task medians
and advanced to all 75 short cells. Beta2 0.95 improved only two and was rejected
at the declared gate. All screen observations, including regressions, are in
[`screen-adamw-lr3.json`](results/round2/screen-adamw-lr3.json) and
[`screen-adamw-lr2-beta95.json`](results/round2/screen-adamw-lr2-beta95.json).

Constant LR ×3 completed **75/75** short cells, but did not beat LR ×2.
Its geometric mean was **35.3367 epochs**, versus **34.2499** for LR ×2:
the paired ratio **1.031730** means **3.17% more epochs**. There were 41 earlier,
10 tied and 24 later cells; the worst ratio was **20.625×** on
`power-division-p97-k3`, seed 0. Against the original baseline it used 38.19%
fewer epochs (ratio 0.618149). Summed trainer time was 1470.06 seconds.
All paired records and task medians/worst cases are in
[`LR ×3 versus LR ×2`](results/round2/adamw-lr3-vs-lr2-short.json) and
[`LR ×3 versus baseline`](results/round2/adamw-lr3-vs-baseline-short.json), with
matching cell/task CSV files. The [short summary](results/round2/short-summary.json)
retains the regressions. All 93 new runs passed the
[checkpoint and source audit](results/round2/audit-before-schedule.json).

The user then requested a learning-rate schedule. The separate
[cosine plan](results/round2/schedule-plan.json) retains the 50-update warmup,
starts at LR ×3, and decays to LR ×1.5 over 200 actual epochs after warmup.
Final round-2 selection waits for the additional short-suite experiments.

The user also suggested PeriNorm and inspection of
`/home/misha/lean_projects/transformer`. Its older architecture documentation
describes Peri-RMS with depth scaling, whereas the frozen benchmark GPT uses
pre-branch RMS and unit residual coefficients. The isolated
[`peri-rms-lr2`](configs/peri-rms-lr2.json) candidate adds parameter-free RMS after
the attention and SwiGLU output projections, at the incumbent LR ×2. It reuses
every original weight and RNG draw, the repaired FP64 RMS backward and the
existing attention/prefix behavior. It adds no trainable parameters. The cosine
candidate keeps the original model, so the two mechanisms remain separate.

The normalization placement follows Table 1 of
[Normalization in Attention Dynamics](https://arxiv.org/abs/2510.22026v2), with
parameter-free RMS as the explicit adaptation. The inspected Lean
`Transformer.Normalization.Scheme.peri` describes idealized attention dynamics;
it does not prove faster trained-GPT convergence. The
[model experiment plan](results/round2/peri-rms-plan.json) records the external
source revision/hashes, hypothesis, deviations and the same 18-cell screen gate.
The transformer repository was consulted read-only; training uses this fixed
early99 benchmark and its native AdamW contract.

The cosine screen completed **18/18 passes**, but improved only **1/6** task
medians versus LR ×2, below the required three. Its square-sum median paired
ratio was 1.3104. It also improved no task median versus constant LR ×3.
The [screen](results/round2/screen-adamw-lr3-cosine200.json) and the
[constant-LR parent comparison](results/round2/screen-adamw-lr3-cosine200-vs-lr3.json)
retain every seed. This schedule was rejected without a full-suite aggregate.

Branch-only Peri-RMS completed **12/18 passes** and improved no complete task
median; four were available and two were suppressed by censoring. All three
key-value recall seeds exhausted 30,000 updates
(468.75 epochs); all three square-sum seeds exhausted 100,000 updates
(5263.1633 epochs). The numeric-vector task passed all three seeds, with a tied
median. The [Peri-RMS screen](results/round2/screen-peri-rms-lr2.json) preserves
all six censored outcomes; this variant was rejected.

An [initialization-only training-input probe](results/round2/peri-initial-scales.json)
found a first-attention-branch/residual RMS ratio of 48.26 on recall with the
branch-only norms, versus 0.578 in the baseline. Adding embedding RMS makes it
0.957. This motivates a separately [declared follow-up](results/round2/peri-embedding-plan.json)
that adds only parameter-free embedding RMS to Peri-RMS, retaining the tied
weight, unit residual coefficients and LR ×2. These initial scales do not
establish why training failed or prove that the follow-up learns faster.
The follow-up configuration is
[`peri-rms-embedding-lr2`](configs/peri-rms-embedding-lr2.json).

The embedding-normalized screen completed **18/18 passes**, recovering all six
branch-only Peri-RMS misses. It improved two task medians relative to LR ×2:
larger-prime squared division (ratio 0.2947) and square-sum (0.8807). Recall
(3.125), smaller-prime squared division (1.3333), and numeric transport (1.3333)
were slower; the boolean median tied. It therefore did not pass the predeclared
three-task promotion gate. The [screen](results/round2/screen-peri-rms-embedding-lr2.json)
and [parent comparison](results/round2/screen-peri-rms-embedding-lr2-vs-peri-rms.json)
retain all outcomes. The [training/held-out diagnosis](results/round2/peri-generalization-diagnosis.json)
shows that the failed recall seeds had reached 99% training accuracy, while
square-sum reached 100% training and about 98.72–98.79% held-out accuracy.

The [next declared refinements](results/round2/refinement-plan.json) test a
constant LR ×2.5 and fixed depth scaling after the embedding-normalized Peri-RMS
branches, each as a separate change. The task now explicitly values stronger
proved mathematical guarantees alongside first-99 performance, with exact
assumptions and a clear distinction between forward bounds and learning claims.
Both refinements passed all 18 screen cells and advance to the complete short
suite. LR ×2.5 improved three task medians, with 9 earlier, 5 tied and 4 later
cells against LR ×2. Scaled Peri-RMS improved four task medians, with 11 earlier,
4 tied and 3 later cells. Its recall median ratio is 0.9375 but the worst seed
is still 5× slower; smaller-prime squared division has median ratio 1.3333.
The [LR ×2.5 screen](results/round2/screen-adamw-lr2p5.json),
[scaled screen](results/round2/screen-peri-rms-scaled-lr2.json), and
[scaled-versus-embedding comparison](results/round2/screen-peri-rms-scaled-lr2-vs-peri-embedding.json)
preserve every pair. Full 75-cell evaluation is required before a new winner
can be selected; these screens are not aggregate speed claims.

The round-2 factories record explicit moment coefficients; their default settings
still match the frozen baseline. CUDA checks cover the baseline, LR ×2, LR ×3
and LR ×2 with beta2 0.95 against native eager AdamW. Adding the cosine and
Peri-RMS mechanisms expanded the combined integrity, oracle, CPU and CUDA suite
from 33 to 38 passing tests. The embedding follow-up passes **41 tests**, adding
explicit tied-weight/RNG checks, a native embedding equation and cached
generation for the additional model. The LR ×2.5 and scaled Peri-RMS refinements
bring this to **45 passing tests**, including scaling after output RMS and nine
configurations checked against native eager updates. See
[validation](results/round2/validation.json).

Candidate source/configuration archives are immutable. The current driver
therefore rejects training into outputs made by older driver revisions. Their
histories remain readable by the comparison tool. Replaying round 1 requires
its archived sources or commit `fde92f2`; constant LR ×3 and beta2 0.95 use
`de711da`; the cosine and branch-only Peri-RMS experiments use `5389a30`;
the embedding-normalized Peri-RMS experiment uses `6e31141`.
Use a fresh output directory for a repeat; do not replace an existing plan or
expected hashes.

```bash
# Current candidate plans; these commands do not train.
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. .venv/bin/python -m research.early99.experiments.run --write-plan --configuration research/early99/configs/peri-rms-scaled-csa-lr2.json

# Separate report destinations preserve both controls.
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. .venv/bin/python -m research.early99.experiments.compare --candidate runs/early99/candidates/adamw-lr3 --tier short --output runs/early99/candidates/adamw-lr3/comparisons/baseline
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. .venv/bin/python -m research.early99.experiments.compare --candidate runs/early99/candidates/adamw-lr3 --baseline runs/early99/candidates/adamw-lr2 --tier short --output runs/early99/candidates/adamw-lr3/comparisons/incumbent
```

Short-suite observations alone determine selection. The long suite has already
been observed in round 1; a later check of a new winner is a reused regression
diagnostic, not new holdout evidence, and cannot affect selection or tuning.

## Mathematical guarantees and the scaled candidate

The complete scaled candidate is Peri-RMS + centered XSA + SwiGLU + per-head
Q/K RMS normalization + residual scaling **after output RMS** + tied
embedding/readout weights. Every RMSNorm is parameter-free, including the
embedding and final normalization. RoPE, fixed attention temperature and guarded
prefix stabilization remain as in the frozen GPT. Embedding RMS acts on lookup
activations; the readout still uses the shared raw embedding matrix.

Write `N(x) = x / sqrt(mean(x²) + epsilon)`. Each of the two branches in a layer
contributes `alpha * N(branch_output)`, where `alpha = 1/sqrt(2L)` for `L` layers.
The [Lean fixture](guarantees/PeriBounds.lean) proves, for positive width and
epsilon and nonnegative alpha:

- `||N(x)||₂ <= sqrt(width)` (from the checked transformer normalization module).
- Arbitrary initial state: `||h_k||₂ <= ||h_0||₂ + k*alpha*sqrt(width)`.
- Normalized embedding: `RMS(h_k) <= 1 + k*alpha`.
- After `L` layers: `RMS(h_2L) <= 1 + sqrt(2L)` with the scaled branches,
  compared with `1 + 2L` for unit coefficients.

The raw branch sequence may depend on all causal inputs. First-position zeroing
and guarded selection of another prefix branch/state preserve the same uniform
bound. These are real-arithmetic trace proofs; the link to the PyTorch code is
an algebraic review and equation/cached-generation tests, not a formally verified
translation. The scientific FP32 epsilon is `1/8388608`. All six new theorems
have concrete nonzero examples. Their dependency audits contain only
`propext`, `Classical.choice` and `Quot.sound`.

The imported normalization module also proves a global Lipschitz bound
`2/sqrt(epsilon)` for a single RMS operation; the fixture proves that scaling
after RMS multiplies this constant by `alpha`. This is a non-tight bound and can
be large near zero. None of these results bounds unconstrained tied readout
weights, the full network Jacobian, floating-point roundoff, or optimization
and grokking. Empirical speed claims still require the complete fixed benchmark.
The [proof record](results/round2/guarantees.json) pins the external source,
toolchain, statements, assumptions and [axiom output](results/round2/lean-check.txt).

Verify using the existing transformer checkout at commit `8890a985` with its
pinned Lean 4.35.0 and Mathlib dependencies:

```bash
cd /home/misha/lean_projects/transformer
lake build Transformer.GPTMini.RMSNorm
lake env lean /home/misha/python_projects/gpt/research/early99/guarantees/PeriBounds.lean
```

The fixture lives in this repository; the external project's sources are unchanged.

The baseline's existing centered XSA and repaired RMS backward also have
[checked identities](results/round2/existing-mechanisms.json). Centering averages
`v_j - anchor` and retains the clipped own-value remainder; normalized attention
weights give exactly the original XSA projection over the reals. Constant values
above the clipping threshold therefore give zero. The separated RMS backward is
algebraically the same derivative, and its radial/tangent coefficient ratio
remains `epsilon/(mean_square + epsilon)`. These facts support the numerical
rearrangements but do not establish a roundoff or training-convergence bound.
The [dependency fixture](guarantees/ExistingMechanisms.lean) and
[Lean output](results/round2/existing-mechanisms-check.txt) retain the checks.

With finite epsilon, scaling a vector before RMS is not exactly canceled; it
changes epsilon's relative effect. Applying the coefficient after RMS is what
makes the stated branch-size and Lipschitz bounds apply to the intended residual
coefficient.

## Further short-suite result and causal centering

Constant LR ×2.5 completed **75/75 short passes** with geometric-mean first-99
exposure **33.2585 epochs**. Its paired ratio is **0.971054** versus LR ×2
(2.9% fewer epochs) and **0.581796** versus the original baseline (41.8% fewer).
Against LR ×2, 38 cells are earlier, 13 tied and 24 later; the worst regression
is 5.7895× on modulus-193 fifth-power division, seed 1 (19 → 110 epochs).
Summed trainer time is 1181.87 seconds, including setup and scoring. Parameters
and initial matrices match in every pair. The
[full LR ×2 comparison](results/round2/adamw-lr2p5-vs-lr2-short.json),
[baseline comparison](results/round2/adamw-lr2p5-vs-baseline-short.json), matching
CSV files, and [checkpoint audit](results/round2/audit-lr2p5-full-short.json)
retain every cell. Final selection waits for the scaled model and the newly
requested attention experiment; no new long cells have been run.

The user supplied [Centered Self-Attention Layers](https://arxiv.org/html/2306.01610)
and [Mind the Gap](https://arxiv.org/html/2410.07799v3), and requested adding the
mechanism to attention. Both motivate subtracting the uniform mixing component
after softmax. The [separate experiment plan](results/round2/csa-plan.json)
adds **causal prefix-mean subtraction** to the scaled Peri-RMS parent, before
its existing own-direction XSA projection. For query `i`, it mixes values with
`C_ij = A_ij - 1/(i+1)` for `j <= i`, and zero for future positions. Output RMS
and residual scaling follow the projection as before.

This changes the attention function. The baseline's numerical centered XSA
only rearranges its evaluation. The causal correction also differs from the
papers' global rank-one uniform matrix: each query has its own prefix average.
Their initialization and asymptotic spectral results do not establish a
spectral-gap or training guarantee for this causal, trained model.

The [new Lean fixture](guarantees/CausalCentering.lean) proves zero row sums,
the weighted-minus-uniform-mean identity, constant-signal cancellation, anchor
invariance of the linear mixing stage, and prefix-only dependence for fixed
causal weights. Nonzero examples accompany the hypotheses. The
[proof record](results/round2/csa-guarantees.json) and
[axiom output](results/round2/csa-lean-check.txt) retain the exact scope.
The existing Peri-RMS state bounds still apply to its normalized branch outputs.

Scaled Peri-RMS also completed **75/75 short passes**. Its geometric-mean
first-99 exposure is **28.3606 epochs**, with paired ratios **0.828049** versus
LR ×2 (17.2% fewer epochs) and **0.496116** versus the original baseline (50.4%
fewer). Against LR ×2, 47 cells are earlier, 9 tied and 19 later. The worst
regression is 6× on the hard depth probe, seed 1 (0.4 → 2.4 epochs); recall seed
2 takes 5× as many epochs. Summed trainer time is **1371.78 seconds**, about
5.7% above LR ×2, so the epoch improvement has a runtime tradeoff. All learned
tensors, initial states and parameter counts remain matched. See the
[full comparison](results/round2/peri-rms-scaled-lr2-vs-lr2-short.json),
[baseline comparison](results/round2/peri-rms-scaled-lr2-vs-baseline-short.json),
and [checkpoint/source audit](results/round2/audit-peri-scaled-full-short.json).

The LR ×2.5 and scaled Peri-RMS scientific source revision is `9e57d7e`.
Causal-centering work uses a subsequent source snapshot. Continue or replay
older outputs from their pinned revision; never replace their plans or hashes.
An isolated checkout at `/tmp/early99-refinement-replay-9e57` has verified that
it reproduces the recorded LR ×2.5 plan and can run an earlier winner's long
cells with `--output` and `--baseline` pointing at the existing absolute paths.

The causal-centering candidate is implemented in
[`experiments/causal_centering.py`](experiments/causal_centering.py), selected by
[`peri-rms-scaled-csa-lr2.json`](configs/peri-rms-scaled-csa-lr2.json).
All **51 CPU/CUDA checks pass**, including an independent matrix reference,
future-gradient exclusion, constant-signal cancellation, cache equivalence and
native eager versus captured updates. Its frozen 18-cell screen starts after
the complete scaled-parent short suite, as requested. The screen retains the
same six tasks, three seeds, promotion rule, evaluation cadence and budgets.

Centering at LR ×2 completed **18/18 screen passes** but improved only one
of the six task medians against LR ×2, below the required three. Every task
median was slower than the scaled Peri-RMS parent. Against LR ×2, median ratios
are 0.5684 for modulus-193 squared division, 8.0345 for modulus-97 squared
division, 3.0 for recall, 7.8234 for square-sum, 1.3333 for numeric transport,
and 1.0 for the boolean task. The worst observed ratio is 27.5× on recall seed 2.
All rows are in the [screen](results/round2/screen-peri-rms-scaled-csa-lr2.json)
and [parent comparison](results/round2/screen-peri-rms-scaled-csa-lr2-vs-peri-scaled.json),
with the [checkpoint/source audit](results/round2/audit-csa-lr2-screen.json).
There is no screen-only geometric-mean speed claim.

The user's learning-rate question motivates a separately
[declared LR ×1 sensitivity trial](results/round2/csa-lr-plan.json), halving only
the rate. Its additional native-eager/CUDA-capture check passed; the 18-cell
screen follows the completed LR ×2 centering screen. Both centering rates use
scientific source revision `51e15ed`. At LR ×1, **15/18** screen cells passed;
all three square-sum seeds missed 99% at the original 100,000-update budget.
Only one of six task medians improved against LR ×2, and none improved against
the scaled parent. This arm also fails the declared gate. The complete
[screen](results/round2/screen-peri-rms-scaled-csa-lr1.json) and
[audit](results/round2/audit-csa-lr1-screen.json) preserve these outcomes.
Two tested rates do not establish the best possible performance of centering.


## SSMax and LeRoPE extensions

The user's next request replaces causal centering with
[Scalable-Softmax](https://arxiv.org/html/2501.19399v1). The
[declared experiment](results/round2/ssmax-plan.json) uses the paper's fixed
`s=1` variant: multiply the existing attention scores by `log(N)` for each
query's complete visible key count. This retains positive, normalized weights
and the original numerical centered-XSA evaluation. A matched second arm
bypasses only parameter-free Q/K RMS; it keeps the same fixed temperature,
initial weights and LR ×2. Both arms retain the scaled Peri-RMS parent and
add no parameters. Queries are scaled after normalization and before masking,
including the `N=1` case and shared-prefix caches.

[LeRoPE](https://arxiv.org/html/2607.10134v1) is a separate
[declared arm](results/round2/lerope-plan.json) on scaled Peri-RMS, with Q/K RMS
and ordinary softmax. Frequencies become `base_frequency * exp(log_scale)`,
with one zero-initialized scalar per band shared across all layers and heads.
This adds **head_dim/2 parameters**, explicitly recorded alongside a separate
hash of every unchanged parent tensor. Only the new frequency vector has zero
weight decay and independent gradient-norm clipping at 1.0; its LR matches the
other parameters. Original parameters retain their native AdamW treatment.
Differentiable rotary tables are recomputed on each forward call.

The two [checked fixtures](results/round2/attention-extension-guarantees.json)
separate structural guarantees from training hypotheses. SSMax has a finite
probability lower bound conditional on a gap above every competing logit;
equal logits remain uniform. LeRoPE retains positive frequencies, the ordinary
RoPE initialization, norm-preserving rotations and relative-position score
structure. Its frequency and local-score derivatives are also dependency
checked. These are real-arithmetic properties, without optimization or
extrapolation promises. The existing scaled residual-state bound applies to
all three new arms. LeRoPE here uses one-dimensional token positions and
two-dimensional channel rotations; no second token coordinate is introduced.

Each new candidate receives the same 18-cell screen, gate and original budgets.
Every promoted arm must complete all 75 short cells before final selection;
only the selected new winner receives the reused long regression diagnostic.

All **69 relevant CPU/CUDA checks pass** for the frozen attention-extension
implementation. The [validation record](results/round2/validation.json) includes
the command, independent equations, optimizer/capture checks, and the
loss-unit correction to the initial clipping-stress assertion.

Both SSMax arms completed **18/18 screen passes**. The Q/K-RMS arm improves
four of six medians against LR ×2 and five against the same scaled parent;
its direct parent pairs are 10 earlier, 4 tied and 4 later. The no-Q/K arm
improves three medians against LR ×2. Both satisfy the predeclared gate and
advance to the complete 75-cell short suite after LeRoPE's screen. Their
screen JSON/CSV files and checkpoint audits retain every result. No aggregate
screen speed claim is made. The user explicitly defers runtime optimization;
selection uses coverage and first-99% epochs, with seconds retained as records.

The additional [Q/K score fixture](guarantees/NormalizedScores.lean) proves
`abs(dot(Q_norm,K_norm)) <= head_dim` after arbitrary rotary angles. With
fixed gain `g`, scores satisfy `abs(score) <= abs(g)*head_dim`; SSMax multiplies
this bound by `log(N)`. The no-Q/K arm loses this bound independent of learned
projection norms. Neither upper bound establishes the positive winning margin
needed for SSMax concentration. Both variants retain the scaled residual-state
bound. See the [updated guarantee record](results/round2/attention-extension-guarantees.json).


## Combined SSMax and LeRoPE candidate

LeRoPE alone completed **18/18 screen passes**, improving four task medians
against LR ×2 and two against the same scaled parent. Against that parent,
6 cells are earlier, 5 tied and 7 later. It advances under the unchanged rule;
its [complete screen](results/round2/screen-peri-rms-scaled-lerope-lr2.json),
[parent comparison](results/round2/screen-peri-rms-scaled-lerope-lr2-vs-peri-scaled.json)
and [checkpoint audit](results/round2/audit-peri-rms-scaled-lerope-lr2-screen.json)
retain the positive and negative pairs. Its screen models add 8–16 learned
frequency scalars, with every original tensor still matched.

The user's confirmed next architecture is **Peri-RMS + XSA + SwiGLU + SSMax +
Q/K RMS normalization + residual scaling after RMS + tied embedding/readout
weights + parameter-free RMSNorm, including embedding RMSNorm + LeRoPE**.
The [combined plan](results/round2/ssmax-lerope-plan.json) adds LeRoPE to the
SSMax/QK parent with the same fixed `s=1`, `1/sqrt(2L)` branch coefficient,
LR ×2 and frequency optimizer treatment. It introduces no causal centering.
The parameter-free statement applies to RMSNorm; LeRoPE's shared frequency
vector is learned and counted.

The existing real-arithmetic hidden-state, Q/K score, conditional SSMax-margin
and learned-rotation guarantees apply together with their original hypotheses.
The combined mechanism is checked against independent equations and both
single-mechanism parents. Earlier promoted arms replay from the verified
`7cff292` checkout at `/tmp/early99-attention-replay-7cff`; their source plans
remain unchanged. After the combined screen, every promoted arm must finish
all 75 short cells before final selection and the one reused long diagnostic.

All **77 relevant CPU/CUDA checks pass** with the combined arm enabled. The
[validation record](results/round2/validation.json) preserves the full command,
independent combined score/gradient checks, cache equivalence and both ordinary
and clipping-stress native versus captured updates.

The combination completed **18/18 screen passes** and improved three of six
medians against LR ×2, meeting the unchanged promotion gate. Against SSMax
with Q/K RMS alone, however, only one median improves: 4 cells are earlier,
6 tied and 8 later. Recall's median is 1.375× that parent's exposure. The
[full screen](results/round2/screen-peri-rms-scaled-ssmax-lerope-lr2.json),
[SSMax parent comparison](results/round2/screen-peri-rms-scaled-ssmax-lerope-lr2-vs-ssmax-qk.json),
[LeRoPE parent comparison](results/round2/screen-peri-rms-scaled-ssmax-lerope-lr2-vs-lerope.json)
and [audit](results/round2/audit-peri-rms-scaled-ssmax-lerope-lr2-screen.json)
retain every seed. Composition therefore does not automatically preserve each
component's measured learning gains.

Median paired screen epoch ratios against LR ×2 (lower is better):

| Task | SSMax + Q/K RMS | SSMax without Q/K RMS | LeRoPE | SSMax + Q/K RMS + LeRoPE |
|---|---:|---:|---:|---:|
| Boolean AND | 1.0000 | 1.0000 | 0.7500 | 1.0000 |
| Division p193 k2 | 0.3684 | 0.4316 | 0.3789 | 0.4000 |
| Division p97 k2 | 1.3333 | 0.7619 | 1.2857 | 1.3810 |
| Key-value recall | 0.8125 | 1.3750 | 0.6875 | 2.1875 |
| Square-sum | 0.6896 | 0.4988 | 0.7190 | 0.7162 |
| Numeric transport | 0.5000 | 1.0000 | 1.0000 | 0.5000 |

All four arms advance to full 75-cell evaluation. This table is a declared
screen, not an aggregate benchmark speed claim. The combined scientific source
revision is `57aa3c9`; the three component arms retain `7cff292`.

The [recall diagnosis](results/round2/combined-recall-diagnosis.json) compares
saved histories without retraining. First held-out 99% epochs are:

| Seed | Scaled Peri-RMS | + SSMax | + LeRoPE | + SSMax and LeRoPE |
|---|---:|---:|---:|---:|
| 0 | 15 | 13 | 11 | 35 |
| 1 | 9 | 18 | 14 | 12 |
| 2 | 20 | 16 | 12 | 22 |

For seed 0, the combination first fits the training split at epoch 9, but
needs 35 epochs for the held-out crossing. At SSMax's epoch-13 crossing,
the combination has 99.66% training and 96.48% held-out accuracy. Seed 2
likewise has 99.51% training and 97.27% held-out accuracy when SSMax first
passes at epoch 16. This locates a generalization delay in those two seeds;
seed 1 improves. The combined endpoint frequency multipliers span
0.899–1.084 across these seeds, so the saved endpoints do not show exploding
frequencies. They do not bound earlier trajectories or positional phase changes.

A plausible interaction follows from the equations: the SSMax logit is
`log(N) * g * qᵀ R(delta * theta * exp(alpha)) k`. Holding q and k fixed,
its frequency derivative contains both `log(N)` and
`delta * theta * exp(alpha)`. Q/K RMS controls vector lengths, but does not
fix the learned angles or the winning attention key. The softmax derivative
also changes with the resulting probabilities, so this does not establish
a universal gradient amplification. It motivates testing a smaller frequency
LR or SSMax scale in future work; neither is a tested explanation here.
The current comparison measures no attention entropy or gradient trajectories
and does not identify a unique cause. No extra tuning arm is introduced.
LeRoPE itself computes `position * base_frequency * exp(log_scale)` without
`log(N)`; SSMax scales scores after rotation. The existing LeRoPE-only arm
already tests these learned angles with ordinary softmax. Removing the
SSMax factor only from frequency backpropagation would change the update
rule, not correct the current forward calculation.


## First complete attention-extension result

SSMax with Q/K RMS completed **75/75 short passes**, with geometric-mean
first-99% exposure **24.3271 epochs**. Its ratios are **0.710281** versus
LR ×2 (**29.0% fewer epochs**), **0.857777** versus the same scaled Peri-RMS
parent (**14.2% fewer**), and **0.425557** versus the original baseline
(**57.4% fewer**). Against LR ×2, 53 cells are earlier, 8 tied and 14 later;
against the parent, 52 are earlier, 10 tied and 13 later. Parameters remain
matched at 100736–1183488 across cases.

The improvements are not uniform. Recall seed 2 takes 4× LR ×2's exposure
(4 → 16 epochs). Division p193 seed 2 takes 3.5× LR ×2 (12 → 42) and
4.6667× the scaled parent (9 → 42). Hard-depth seed 2 takes 3.3333× LR ×2
(0.6 → 2.0). The [LR ×2 comparison](results/round2/peri-rms-scaled-ssmax-lr2-vs-lr2-short.json),
[parent comparison](results/round2/peri-rms-scaled-ssmax-lr2-vs-peri-scaled-short.json),
[baseline comparison](results/round2/peri-rms-scaled-ssmax-lr2-vs-baseline-short.json),
CSV files and [complete audit](results/round2/audit-peri-rms-scaled-ssmax-lr2-full-short.json)
preserve every cell.

The first audit attempt used a relative result path inside the replay checkout.
Training had used absolute paths throughout and was unaffected. Correcting the
temporary audit helper's result paths produced a clean 75-cell audit, with no
retraining or resumed cells. At that checkpoint, final selection awaited the
remaining three promoted attention candidates and the long diagnostic had
not started. The completed round is summarized at the top of this document.

The **SSMax without Q/K RMS** arm has now also completed and passed its
[75-cell audit](results/round2/audit-peri-rms-scaled-ssmax-noqk-lr2-full-short.json).
Its geometric mean is **20.9798 epochs**: **38.7% fewer** than LR ×2,
**26.0% fewer** than the scaled Peri-RMS parent, **13.8% fewer** than the
otherwise matched SSMax/QK arm, and **63.3% fewer** than the original baseline.
Against LR ×2 it has 49 earlier, 9 tied and 17 later cells. Against SSMax/QK
it has 47 earlier, 4 tied and 24 later cells. Capacity remains matched at
100736–1183488 parameters. Summed trainer time is 1279.04 seconds; the queue
pause interrupted no training cell, and all 75 cells have no resumptions.

The [direct Q/K ablation](results/round2/peri-rms-scaled-ssmax-noqk-lr2-vs-ssmax-qk-short.json)
shows large regressions too: hard-depth seed 1 takes **0.4 → 5.6 epochs (14×)**,
division p193 seed 1 takes **9 → 78 (8.6667×)**, and numeric transport seed 0
takes **2 → 13 (6.5×)**. Its
[LR ×2 comparison](results/round2/peri-rms-scaled-ssmax-noqk-lr2-vs-lr2-short.json),
[scaled-parent comparison](results/round2/peri-rms-scaled-ssmax-noqk-lr2-vs-peri-scaled-short.json)
and [baseline comparison](results/round2/peri-rms-scaled-ssmax-noqk-lr2-vs-baseline-short.json)
retain every row and complete task medians/worst cases. This arm became the
empirical leader under the predeclared coverage/epoch rule. It retains the
scaled residual-state bound but loses Q/K RMS's score bound independent of
projection norms. Faster observed learning and stronger score control are
therefore distinct choices here. At that checkpoint, final selection still
awaited the remaining promoted arms and the long diagnostic had not started.

The user's clarified research objective is one shared architecture for small
and large tasks. **Q/K RMS remains in the main candidate**; the no-Q/K arm
is an ablation whose small-suite speed does not establish scalable stability.
The declared benchmark ranking remains intact, while a shared-model choice
requires separate evidence across sizes and context lengths. This distinction
is supported by large-model studies: [ViT-22B, Section 2](https://proceedings.mlr.press/v202/dehghani23a/dehghani23a.pdf)
reports Q/K normalization preventing attention-logit-driven divergence in an
8B model; [Chameleon, Section 2.3](https://arxiv.org/html/2405.09818v1#S2.SS3)
reports a 7B ablation diverging without Q/K normalization and uses additional
stabilization measures. These are evidence from other training settings, not
proof that our exact parameter-free RMS variant is sufficient at every scale.

## Log-Scaled LeRoPE with SSMax retained

The final **SSMax + Log-Scaled LeRoPE + Q/K RMS** arm completed **75/75 short
passes** and its [full audit](results/round2/audit-peri-rms-scaled-ssmax-lerope-log-angle-lr2-full-short.json).
Its geometric mean is **28.3632 epochs**: **17.2% fewer** than LR ×2, but
**16.59% more** than SSMax/QK and **9.11% more** than ordinary SSMax + LeRoPE.
The aggregate is effectively unchanged from scaled Peri-RMS (ratio 1.0000905).
Against LR ×2, 47 cells are earlier, 4 tied and 24 later; against SSMax/QK,
23 are earlier, 13 tied and 39 later. Hard-depth seed 2 takes **0.6 → 9.2
epochs (15.3333×)** versus LR ×2. The complete
[LR ×2 comparison](results/round2/peri-rms-scaled-ssmax-lerope-log-angle-lr2-vs-lr2-short.json),
[SSMax comparison](results/round2/peri-rms-scaled-ssmax-lerope-log-angle-lr2-vs-ssmax-qk-short.json)
and [direct-parent comparison](results/round2/peri-rms-scaled-ssmax-lerope-log-angle-lr2-vs-ssmax-lerope-short.json)
preserve every seed. Parameters are 100744–1183504; summed trainer time is
1687.52 seconds. This full result supersedes any inference from its favorable
recall screen alone. All required short comparisons are now complete.

The ordinary **SSMax + LeRoPE + Q/K RMS** combination also completed
**75/75 short passes**, with geometric mean **25.9959 epochs**. This is
**24.1% fewer** than LR ×2 (51 earlier, 4 tied, 20 later), but **6.86% more**
than SSMax/QK alone (20 earlier, 27 tied, 28 later). It improves over ordinary
LeRoPE by **8.32%**, and over the scaled Peri-RMS parent by **8.34%**.
Hard-depth seed 2 takes 6.6667× LR ×2's exposure; recall seed 2 takes 5.5×.
Summed trainer time is 1574.27 seconds. The
[full comparison](results/round2/peri-rms-scaled-ssmax-lerope-lr2-vs-lr2-short.json),
[SSMax parent comparison](results/round2/peri-rms-scaled-ssmax-lerope-lr2-vs-ssmax-qk-short.json)
and [75-cell audit](results/round2/audit-peri-rms-scaled-ssmax-lerope-lr2-full-short.json)
retain every cell. The final Log-Scaled LeRoPE arm remains in progress.

The ordinary **LeRoPE with Q/K RMS** arm completed **75/75 short passes**.
Its geometric mean is **28.3558 epochs**, versus 28.3606 for its scaled Peri-RMS
parent: ratio **0.999830**, an effectively unchanged aggregate. It has 24 earlier,
27 tied and 24 later cells against that parent. Against LR ×2 it uses **17.2%
fewer epochs** (ratio 0.827908; 48 earlier, 6 tied, 21 later), and against the
original baseline **50.4% fewer**. Summed trainer time is 1525.97 seconds;
parameter counts are 100744–1183504, including the declared frequency vector.
Hard-depth seeds 1 and 2 take 5.5× and 5× LR ×2's exposure, respectively.
The [complete comparison](results/round2/peri-rms-scaled-lerope-lr2-vs-lr2-short.json),
[parent comparison](results/round2/peri-rms-scaled-lerope-lr2-vs-peri-scaled-short.json)
and [75-cell audit](results/round2/audit-peri-rms-scaled-lerope-lr2-full-short.json)
retain all outcomes. SSMax/QK is still the strongest completed Q/K-normalized
arm by the declared aggregate. Both combined rotary arms remain in progress.

The user requested a further experiment before resuming the current queue:
**SSMax + Log-Scaled LeRoPE**, under the
[declared plan](results/round2/ssmax-lerope-log-angle-plan.json).
Its direct parent is the combined SSMax + LeRoPE candidate. The only change
is the per-token angle `p * log(p+1) * base_frequency * exp(log_scale)`, where
`p` is each token's zero-based global position. SSMax separately retains
`log(visible keys)` on scores. All normalization, residual scaling, tied weights,
frequency parameters, LR and optimizer treatment remain matched. Initial
weights match, but initial outputs intentionally change with the angle map.

The [queue pause record](results/round2/log-angle-pause.json) preserves its
boundary at 65/75 completed no-Q/K short cells. Only dispatchers were paused;
the active cell finished normally, so no training clock or scientific cell was
interrupted. Earlier arms retain their frozen source snapshots. The combined
parent can now replay from `57aa3c9` in `/tmp/early99-combined-replay-57aa`.
After this new 18-cell screen, resume the unfinished full suites and include
this arm's full 75 cells if it meets the unchanged gate.

All **86 relevant CPU/CUDA checks pass**, including independent angle equations,
frequency gradients, cached decoding and both ordinary and clipping-stress
updates. The [Lean fixture](guarantees/LogScaledRotary.lean) and
[proof record](results/round2/log-angle-guarantees.json) check the zero first
angle, norm preservation, and rotary identity with warped displacement
`w(t)-w(s)`. Equal unit token gaps can have unequal warped gaps. Ordinary
relative-position invariance is therefore not claimed. Existing Q/K score,
conditional SSMax-margin and scaled residual-state bounds still apply;
none establishes improved learning or extrapolation.

The [completed screen](results/round2/screen-peri-rms-scaled-ssmax-lerope-log-angle-lr2.json)
passes **18/18** and improves three of six task medians against LR ×2, so it
advances to the full 75 cells. Against that control, 10 cells are earlier,
1 tied and 7 later. Against the
[direct SSMax + LeRoPE parent](results/round2/screen-peri-rms-scaled-ssmax-lerope-log-angle-lr2-vs-ssmax-lerope.json),
7 are earlier, 5 tied and 6 later, with two improved task medians. Recall
improves in every seed: **35 → 15, 12 → 9, 22 → 13 epochs**. Numeric transport
regresses in every seed: **2 → 7, 2 → 10, 5 → 11 epochs**, including a 5×
regression for seed 1. The paired task medians against the direct parent are
1.0 (Boolean), 1.0 (p193), 1.0 (p97), 0.5909 (recall), 0.9793 (square-sum)
and 3.5 (transport). These are screen results, with no aggregate speed claim.
The [18-cell audit](results/round2/audit-peri-rms-scaled-ssmax-lerope-log-angle-lr2-screen.json)
verifies every original stopping rule, checkpoint, source and optimizer group.
The original queue resumed after this audit, reusing all 65 completed no-Q/K
cells; this new arm is included among the promoted full-suite comparisons.

After completing this round's frozen short selection, reused long diagnostic,
reports and commits, the user's latest instruction is to continue research in
a separately declared short-suite round. Long outcomes remain regression
evidence and cannot guide the next candidates.

### Replaying the attention candidates

The immutable run plans hash every research Python source. Resume an old
candidate from its recorded scientific revision, using absolute output and
baseline paths; do not replace its expected hashes with the current tree's.

| Candidates | Scientific revision |
|---|---|
| SSMax with/without Q/K RMS; ordinary LeRoPE | `7cff292` |
| SSMax + LeRoPE | `57aa3c9` |
| SSMax + Log-Scaled LeRoPE | `c6c24d8` |

For example, from a checkout of `7cff292`, the SSMax/QK dispatcher command is:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. /home/misha/python_projects/gpt/.venv/bin/python -m research.early99.experiments.run \
  --run --tier short \
  --configuration research/early99/configs/peri-rms-scaled-ssmax-lr2.json \
  --output /home/misha/python_projects/gpt/runs/early99/candidates/peri-rms-scaled-ssmax-lr2 \
  --baseline /home/misha/python_projects/gpt/runs/early99/baseline
```

Completed cells are reused; changed sources or settings are rejected. Use a
separate output directory for an independent repeat. The committed comparison
entrypoint can regenerate the complete JSON and both CSV files from preserved
records, without training, from the current checkout:

```bash
PYTHONPATH=. .venv/bin/python -m research.early99.experiments.compare \
  --baseline runs/early99/candidates/adamw-lr2 \
  --candidate runs/early99/candidates/peri-rms-scaled-ssmax-lr2 \
  --output /tmp/early99-recomputed-ssmax-vs-lr2 --tier short
```

Use the corresponding candidate name for other arms. All plan, configuration,
source, initialization, checkpoint, history and data hashes remain archived
under each candidate's run directory. Raw checkpoints are retained locally;
the compact comparison reports, configurations, implementation and proofs are
committed.
