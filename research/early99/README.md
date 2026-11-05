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
