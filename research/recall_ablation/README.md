# GPTMini hard-recall feature ablation

Restore the historically successful model and AdamW first, commit that baseline,
then add one feature at a time. Stop at the first feature that loses the
baseline's fast 99% validation crossing. A finite-budget miss is a speed
regression, not a proof that the model can never generalize.

Current accepted model: **SwiGLU + repaired RMS backward + Q/K RMS + ordinary AdamW**,
with raw embeddings, unscaled projection initialization and no Peri branch
norms. Fixed forward residual scaling missed its screen and remains off.
Head temperatures are fixed at their initial values. Fused SDPA missed its
fast screen and remains off. Native dtype input/final RMS epsilon passes
and is now retained, with repaired backward and explicit dense attention.
The XSA clipping threshold 1e-12 and centered projection pass and are retained.
Explicit zeroing of the first global attention position also passes and is
retained. Guarded repeated-prefix stabilization also passes and is retained.
The final architecture passes the same 3,000-update criterion on all three
model seeds. Public `GPT` now matches this confirmed `prefix-stable` model.

| Latest continuation, ordinary AdamW, seed 1 | First 99% exact | Exact endpoint |
| --- | --- | --- |
| Repaired SwiGLU control, unit L2 Q/K | 2,400 | 99.02% |
| Add repaired per-head Q/K RMS, keep learned temperature | 2,050 | 99.02% |
| Fix head temperature at its initial value | 2,500 | 99.22% |
| Replace explicit attention with fused SDPA | Miss by 3,000 | 97.46% (best 98.83%) |
| Native dtype input/final RMS epsilon, explicit attention | 1,850 | 99.02% |
| XSA clipping threshold 1e-12 | 2,650 | 99.02% |
| Centered XSA with the same threshold | 2,400 | 99.02% |
| Explicit first global attention zero | 2,800 | 99.22% |
| Guarded repeated-prefix stabilization | 2,600 | 99.80% |

The compiled CPU embedding backward is not bitwise deterministic: repeated
identical calls on the same unchanged model differ by up to 2.24e-8 in the
embedding gradient. Loss and all other parameter gradients agree bitwise in
the probe, at initialization and at the trained control endpoint. Matching
seeds therefore does not guarantee identical numerical trajectories. Crossing
times and near-threshold misses here are individual finite-run observations,
not strong causal proof of permanent failure. The original training protocol
stays fixed. See `results/xsa_epsilon_compiled_control_probe.json`.

## Passed screen: native dtype RMS epsilon

The `rms-epsilon` candidate starts fresh from the passing
`fixed-temperature` model (99% at 2,500), skipping the rejected SDPA path.
It changes epsilon at exactly thirteen existing branch-input/final RMS
sites from 1e-5 to the native dtype default (FP32 epsilon, approximately
1.19209e-7, in this run). The twelve per-head Q/K norms already use that
default and stay unchanged. All 25 sites keep repaired FP64 radial/tangent
backward. Parameter bytes, fixed temperatures, construction RNG, embedding
and residual behavior remain identical to the parent. Ordinary AdamW, data,
seed 1, 50-update observations and the 3,000-update criterion stay fixed.
No trained endpoint initializes the candidate; no final-test data are used.

Isolation checks verify all thirteen changed sites, unchanged Q/K epsilon,
native forward equality and independent positive radial derivatives at
embedding-scale and unit-scale inputs. Public GPT now matches the passing
`rms-epsilon` candidate.

```bash
.venv/bin/python -m research.recall_ablation.run --stage rms-epsilon --seeds 1 --max-updates 3000
```

**Native epsilon passed at 1,850 updates:** 507/512 exact sequences (99.02%)
and 4,090/4,096 correct queries (99.85%), compared with its parent's 2,500
crossing. Independent eager replay reproduces every count; the latest-value
oracle finds 2,711/2,714 overwritten-key answers and 1,379/1,382 unchanged-key
answers correct, with no stale-value answers. Source, sampler, optimizer and
cadence checks pass. Results, complete history and audit are retained as
`results/rms_epsilon_seed1*`. This is one matched CPU FP32 seed, with no
final-test access.

## Passed screen: XSA clipping threshold

The `xsa-epsilon` candidate changes only the own-value normalization clipping
threshold from 1e-6 to 1e-12, matching the archived newer GPT. It keeps the
ordinary subtraction-based XSA projection; centering and first-position
zeroing are later features. Its parent is the passing `rms-epsilon` model
(first 99% at 1,850). All weights, buffers and construction RNG are identical;
all 25 repaired RMS sites keep native dtype epsilon. Fixed head gain,
explicit attention, ordinary AdamW and the same seed/data/3,000-update
criterion remain. No rejected feature is reintroduced.

Initial model predictions match the parent exactly. Independent clipped
parallel/perpendicular examples cover zero vectors, vectors below either
threshold and ordinary vectors; full-stack gradients and the frozen original
GPTMini fixture pass. The public default now uses 1e-12 clipping.
Source snapshots now also include the XSA helper and cached-attention
module, with no training recipe change.

```bash
.venv/bin/python -m research.recall_ablation.run --stage xsa-epsilon --seeds 1 --max-updates 3000
```

**Clipping passed at 2,650 updates:** 507/512 exact sequences (99.02%) and
4,091/4,096 correct queries (99.88%). The native-epsilon parent crosses at
1,850. Eager checkpoint replay, independent latest-value targets, source
hashes, optimizer steps, observation cadence and full sampler replay verify.
Overwritten-key queries score 2,711/2,714, unchanged-key queries 1,380/1,382,
and no stale values are predicted. The compiled-control probe finds the
same small embedding-gradient variation on repeated calls to either the old
or refactored control; the observed speed difference cannot be attributed
entirely to clipping from this single run. Public GPT retains all passing
features and ordinary AdamW. Evidence is retained as `results/xsa_epsilon_seed1*`.

## Passed screen: centered XSA

The `centered-xsa` candidate branches from the passed `xsa-epsilon` parent
(first 99% at 2,650). It centers values on the first available cached/key
value before explicit attention, then uses the archived `centered_projection`
equation, including its below-epsilon own-value remainder. This changes
floating-point evaluation while preserving the real attention/projection
function. Raw values stay in the cache, including shared prefixes. The
threshold remains 1e-12 and all RMS epsilons remain native with repaired
backward. Weights, fixed head temperatures, RNG, ordinary AdamW and data
remain matched. No embedding norm, Peri output norm, scaling or SDPA returns.

Explicit first-position zeroing and repeated-prefix rewrites remain later
features. A tiny below-clipping own vector therefore retains its legitimate
first-position remainder in this candidate. Checks cover parameter/RNG
isolation, FP64 parent forward and parameter gradients at ordinary/tiny
value scales, raw KV-cache continuation in tokens/chunks, the optional guard
return and the compiled fullgraph loss/gradients. The usual seed-1,
3,000-update fast criterion and 50-update observations apply. Public GPT
now matches the passing centered candidate.

```bash
.venv/bin/python -m research.recall_ablation.run --stage centered-xsa --seeds 1 --max-updates 3000
```

**Centered XSA passed at 2,400 updates:** 507/512 exact sequences (99.02%)
and 4,091/4,096 correct queries (99.88%). Independent eager replay matches
every count; the latest-value oracle verifies 2,712/2,714 overwritten-key
answers and 1,379/1,382 unchanged-key answers, with two stale-value predictions.
Source hashes, sampler replay, optimizer steps and observation cadence pass.
Results, history and audit are retained as `results/centered_xsa_seed1*`.

This run experienced substantial CPU contention. Three read-only debugger
samples and separate throughput probes were used to investigate. After the
last observed update 1,250, its four OpenMP workers were placed on distinct
allowed CPUs (0, 2, 4, 6). Model/data/optimizer/thread-count settings stayed
fixed. These scheduling interventions and system load make wall-clock
comparisons inappropriate. A retrospective session note records the runtime
details; the model source snapshots and trained state are complete.

## Passed screen: exact zero at the first attention position

The `first-zero` candidate starts from the passed `centered-xsa` parent
(first 99% at 2,400). It explicitly sets the first global attention output
to zero, even when the own value is below the clipping threshold. This is
separate from centered evaluation, which preserves the clipped own-value
remainder. The bias-free finite output projection commutes with zeroing,
so the shared attention path is reused and its projected first row is zeroed.
Later positions and cached calls after global position zero are preserved.

All parameters, fixed head temperatures and RNG stay identical. Centering,
1e-12 clipping, native-epsilon repaired RMS, explicit attention, ordinary
AdamW and the same task/sampler/budget remain. Repeated-prefix stabilization
is still a later feature. Checks cover parameter/buffer isolation, exact
first output and gradient zeros with tiny values, unchanged later outputs,
token/chunk decoding and compiled training gradients. Public GPT now matches
the passing `first-zero` candidate.

```bash
.venv/bin/python -m research.recall_ablation.run --stage first-zero --seeds 1 --max-updates 3000
```

**Explicit first zero passed at 2,800 updates:** 508/512 exact sequences
(99.22%) and 4,092/4,096 correct queries (99.90%). The centered parent
crosses at 2,400. Independent eager replay matches, with 2,713/2,714 correct
overwritten-key answers, 1,379/1,382 unchanged-key answers and no stale-value
predictions. Source hashes, sampler/optimizer state and observation cadence
verify. Complete evidence is retained as `results/first_zero_seed1*`.

## Passed final feature screen: repeated-prefix stabilization

The `prefix-stable` candidate starts from the passing `first-zero` parent
(first 99% at 2,800). For uncached full-context calls it shares each constant
token prefix's attention/MLP branch outputs and final residual states with
the first position. If any first-position value head is clipped in a layer,
that row's rewrite is disabled permanently for the remaining layers.
Cached calls retain the ordinary parent path.

The adapter borrows every existing parameter and buffer without random draws,
preserves weight tying, model dimensions, mode and checkpointing settings,
and keeps centered XSA, first-position zeroing, native-epsilon repaired RMS,
fixed head gain, explicit attention and ordinary AdamW. No rejected feature
returns. Tests cover FP64 parent forward/gradient equivalence, exact prefix
states, permanent clipping-guard disablement, mode/parameter identity,
token/chunk cache continuation and compiled checkpointing in both modes.

The seed-1 3,000-update screen remains fixed. The public default includes
this final feature after the three-seed confirmation below. No final-test
data are loaded by the screen or confirmations.

```bash
.venv/bin/python -m research.recall_ablation.run --stage prefix-stable --seeds 1 --max-updates 3000
```

**The final feature passed at 2,600 updates:** 511/512 exact sequences (99.80%)
and 4,095/4,096 correct queries (99.98%). Independent eager replay agrees:
all 2,714 overwritten-key answers are correct, 1,381/1,382 unchanged-key
answers are correct, and no stale values are predicted. Source, sampler,
optimizer and cadence checks pass. Portable evidence is retained as
`results/prefix_stable_seed1*`.

## Final architecture confirmation

`confirm.py` repeats the passed `prefix-stable` architecture from fresh model
seeds 0 and 2, alongside the existing seed-1 screen. It uses the original
training/evaluation loop, native AdamW and the exact same 3,000-update budget,
data fingerprints, thread count, compiler setting, architecture and model/data
source files. Only the model initialization and its seed-derived shuffle
change. It does not require or fabricate per-feature parent runs for the new
seeds. This is architecture confirmation rather than another feature ablation.

Eligibility and provenance handling were extended in `run.py`; an AST
fingerprint verifies that the imports, task/training constants, split builder,
model/optimizer/loss/compiler/sampler setup, update loop and endpoint handling
match the frozen accepted driver. Recipe and source guards must pass before
training. The model files stay frozen through both confirmations, and the
model is never initialized from the reference checkpoint. Each requested seed
is evaluated within its own fixed budget; no further feature work follows.
No final-test data are accessed.

**All three seeds passed within 3,000 updates.** Seed 1 is the original final
feature screen; seeds 0 and 2 are fresh confirmations with different model
initialization and seed-derived shuffle order.

| Model seed | First 99% exact | Exact endpoint | Correct queries |
| --- | --- | --- | --- |
| 0 | 2,550 | 508/512 (99.22%) | 4,092/4,096 |
| 1 | 2,600 | 511/512 (99.80%) | 4,095/4,096 |
| 2 | 2,650 | 508/512 (99.22%) | 4,092/4,096 |

Independent eager checkpoint replay matches every endpoint count. Latest-value
oracles verify all 4,096 query targets per seed, and source hashes, full sampler
state, optimizer steps and first-crossing cadence pass. Portable results,
histories and audits are retained as `results/prefix_stable_seed{0,1,2}*`;
`results/final_confirmation_summary.json` combines the cohort. The recorded
confirmation-driver commit is `a2764d8`; use that revision to rerun the frozen
experiment, since public-model promotion changes the guarded source hash.

All planned features have been screened. Previously rejected features stay
excluded. Public `GPT` now includes every passing feature, including the guarded
prefix adapter, with exact initialization/RNG and repeated-prefix output checks
against the confirmed factory. These are three model seeds on one fixed data
split under the finite speed criterion; no final-test or new data-seed claim
is made, and the measured CPU embedding-gradient nondeterminism still applies.

Final public-model validation: **896 domain/application tests passed, 40 skipped**;
Ruff passes. The model promotion is committed as `01f18f8`.

```bash
.venv/bin/python -m research.recall_ablation.confirm --seeds 0 2
```

Confirmations require the model/data source revision of their reference;
use the recorded confirmation-driver commit to repeat this frozen cohort.

## Requested continuation: Q/K RMS with learned temperature retained

The `qk-rms` candidate starts fresh from the passed repaired-SwiGLU control
(`rms-repaired`, first 99% at 2,400). It changes only per-head Q/K
normalization from unit L2 to repaired RMSNorm, before RoPE. All existing
weights, learned head-temperature parameters and construction RNG state stay
identical. Embeddings, branch outputs, initialization and forward residual
coefficients stay at the passing settings. Ordinary AdamW and the same
seed-1 3,000-update screen remain fixed.

An RMS-normalized head has L2 length about `sqrt(head_dim)`. The learned
score gain therefore becomes `exp(log_alpha)/head_dim`, compensating both
head norms while retaining the original learned **cosine** temperature.
Temperature removal and fused SDPA are later, separate features. New head
norms use native dtype epsilon and the repaired FP64 backward; the thirteen
existing input/final norms retain epsilon 1e-5. There are 25 repaired RMS
sites in total.

Focused checks verify identical initial parameters/RNG, native RMS forward
equations, finite gradients to every parameter, retained learnable scales,
independent FP64 cosine-score equivalence, cached-vs-full predictions and
the frozen original GPTMini initialization/forward/backward fixture.

```bash
.venv/bin/python -m research.recall_ablation.run --stage qk-rms --seeds 1 --max-updates 3000
```

**Q/K RMS passed:** first 99% at **2,050 updates**, with 507/512 exact
sequences (99.02%) and 4,091/4,096 correct queries (99.88%). The repaired
L2-Q/K control reaches the same threshold at 2,400. Independent eager replay
matches every count, including 2,713/2,714 overwritten-key answers and
1,378/1,382 unchanged-key answers; no superseded-value answers remain.
Source hashes, optimizer steps, cadence and full sampler state verify.
The public `GPT` now matches the passing `qk-rms` candidate. This is one
matched seed, with no final-test evaluation. Results, history and audit are
retained as `results/qk_rms_seed1*`.

## Continuation: fixed attention temperature

The `fixed-temperature` candidate starts fresh from the passed `qk-rms`
parent. Each head keeps its initial `log_alpha = log(sqrt(head_dim))` as a
nonpersistent buffer, removing its trainable parameter from AdamW. With Q/K
RMS and the existing gain compensation this is fixed `1/sqrt(head_dim)`
score scaling, as in the archived newer GPT. The exact existing exponential
and division operations remain, so initial forward predictions match the
learned-temperature parent bit for bit. All 24 removed scalars have the same
initial values; every remaining parameter, RNG draw and epsilon is unchanged.
No trained weights initialize this run. The 25 repaired RMS sites, ordinary
XSA and explicit dense attention remain. Fused SDPA is a later feature.

The same seed 1, AdamW recipe, data, 50-update observations and 3,000-update
budget apply. Initial forward equality, removed parameter names, retained
matrix values, fixed buffers and full-stack gradients are checked before
training. Public GPT kept learned temperatures while this test ran.

```bash
.venv/bin/python -m research.recall_ablation.run --stage fixed-temperature --seeds 1 --max-updates 3000
```

**Fixed temperature passed at 2,500 updates:** 508/512 exact sequences
(99.22%) and 4,092/4,096 correct queries (99.90%). This is slower than the
learned-temperature parent's 2,050, but remains within the declared fast
criterion. Eager checkpoint replay reproduces those counts, including
2,711/2,714 overwritten-key queries, 1,381/1,382 unchanged-key queries
and one stale-value answer. Full sampler, optimizer, source and cadence
checks pass. Public `GPT` now matches `fixed-temperature`, with 1,249,024
parameters (24 fewer temperature scalars). Results, history and audit are
retained as `results/fixed_temperature_seed1*`. This remains a single-seed
screen; no final-test data were accessed.

## Stopped screen: fused SDPA

The `sdpa` candidate branches from the passing `fixed-temperature` parent.
It replaces only explicit dense score/softmax/value evaluation with
`scaled_dot_product_attention`; the historical single-query cached path
stays the same. It supplies `q * (gain * sqrt(head_dim))` to compensate
SDPA's built-in `1/sqrt(head_dim)`, preserving the parent's real score
gain. Fresh parameters and all construction RNG state are identical.
All previous passing features, 25 repaired RMS sites, epsilons and ordinary
XSA remain. CPU kernels and floating-point reduction order can change.

A full prefill uses causal SDPA; a cached multi-token chunk gets an explicit
offset causal mask. Independent FP64 forward/gradient oracles cover both,
and whole-model tests cover token and chunk cache continuation. The frozen
GPTMini fixture verifies that extracting the old explicit attention into a
method did not change the original control. Public GPT retains explicit
attention during the matched seed-1, 3,000-update screen.

```bash
.venv/bin/python -m research.recall_ablation.run --stage sdpa --seeds 1 --max-updates 3000
```

**Fused SDPA missed the fast screen.** It never reaches 99% by 3,000 updates:
the endpoint is 499/512 exact sequences (97.46%), with 4,083/4,096 correct
queries (99.68%). Its best is 506/512 exact (98.83%) at 2,350; at its parent's
2,500-update crossing it has 497/512 exact (97.07%). These are CPU FP32,
single-seed observations under the fixed budget. The independent FP64 math
and gradient tests pass; the missed threshold concerns the trained numerical
trajectory, not a changed real attention function or eventual impossibility
of generalization. No further feature was started.

The public default retains explicit attention, fixed head temperature, Q/K
RMS, repaired RMS backward, SwiGLU and ordinary AdamW. Full history, source
snapshot, optimizer and sampler state are retained, with portable result,
history and audit in `results/sdpa_seed1*`. The broader suite passes 852
tests, with 40 skipped; no final-test data were loaded.

## Continuation: ordinary AdamW, architectural features

The next cohort keeps ordinary AdamW throughout and resumes the architectural
sequence after the failed optimizer screen. Each fresh candidate adds one
feature to the last passing model. The same seed 1, historical data streams,
50-update observations and 3,000-update 99% exact-sequence screen apply.
The passing control remains the recorded 2,900-update AdamW run. A missed
screen stops this sequence; it does not establish permanent inability to learn.

The first candidate replaces only the ReLU-squared MLP with the existing
SwiGLU implementation. It keeps every non-MLP parameter byte-identical to
the recorded historical initialization. The new MLP matrices receive fresh
Normal(0, 0.02) initialization after constructing the historical model; there
is no initialization from trained weights. Its parameter count is 1,249,048,
compared with 1,249,816 for the control. Shape changes therefore do not shift
the random initialization of unrelated embedding and attention matrices.

**SwiGLU passed:** first 99% at 2,300 updates, with 508/512 exact sequences
(99.22%) and 4,092/4,096 correct queries. Its endpoint, independent
latest-value replay and full history are retained in `results/`.
The public `GPT` now matches this passing candidate exactly; `GPTMini`
retains the original ReLU-squared control. The optimizer remains ordinary
AdamW.

The next candidate, `scaled-init`, changes only the initial attention and MLP
output matrices: multiply the parent's same Gaussian draws by
`1/sqrt(2*n_layer)`. All other tensors, including the MLP input matrices,
remain byte-identical. This isolates depth-scaled projection initialization
from residual scaling during the forward pass. It requires a passed SwiGLU
parent and verifies the parent's initial-parameter digest before training.

**Scaled initialization missed the fast screen.** It never reached 99% by
3,000 updates: its best exact accuracy was 481/512 (93.95%) at 2,800, and
its endpoint was 453/512 (88.48%). At its parent's 2,300-update crossing it
scored 89.26%. Eager replay exactly reproduces 4,036/4,096 correct endpoint
queries and 453 exact sequences. This is a finite speed regression in the
tested order; it does not establish inability to grok later, and interactions
with later normalization were not tested. Public GPT retains unscaled
initialization. Full results and audits are in `results/architecture_summary.json`.

The user then requested the next test while leaving scaled initialization out.
`embedding-norm` therefore branches from the passed **unscaled SwiGLU** parent,
not from the failed `scaled-init` model. Every parameter is byte-identical to
that parent, including the tied embedding/readout object. Only the embedding
output gains parameter-free native RMSNorm, using native dtype epsilon and
native backward. Other norm sites, attention, branch scaling, initialization,
optimizer and data remain unchanged. The separated radial RMS backward is
a later feature and is not included in this candidate. The same seed 1 and
3,000-update screen apply; another miss stops the sequence again.

**Embedding normalization missed the fast screen:** 375/512 exact sequences
(73.24%) and 3,935/4,096 correct queries (96.07%) at 3,000 updates. Its best
observation is that endpoint; it never reaches 99%. At the parent's
2,300-update crossing it scores only 9/512 exact sequences (1.76%). Eager
checkpoint replay reproduces every prediction count, and the independent
latest-value oracle, source hashes, optimizer steps and complete sampler
state all verify. All parameter bytes at initialization match the passed
SwiGLU parent; the failed initialization change is absent.

**Decision for the embedding screen:** stop again and retain public GPTMini + SwiGLU with
ordinary AdamW, raw embeddings and unscaled initialization (`b91fdf8`).
Embedding normalization is clearly learning by the endpoint; the result is
a regression under the declared fast criterion on one fresh seed, not a
claim of eventual impossibility. Effects of later normalization or different
training rates remain untested. Six focused isolation/gradient tests pass;
the preceding public-model promotion passed 830 domain/application tests
with 40 skipped. Ruff passes for the changed Python files.

| Ordinary AdamW, same seed and data | First 99% exact | Exact endpoint |
| --- | --- | --- |
| Historical ReLU-squared | 2,900 | 99.02% |
| SwiGLU | 2,300 | 99.22% |
| SwiGLU + scaled projection initialization | Miss by 3,000 | 88.48% (best 93.95%) |
| SwiGLU + embedding RMS, unscaled initialization | Miss by 3,000 | 73.24% |

## Requested continuation: Peri-RMSNorm with raw embeddings

The next requested test skips the failed embedding normalization and again
starts from the passed unscaled SwiGLU parent. `peri-norm` keeps each existing
branch-input RMSNorm and adds a parameter-free native RMSNorm after each
attention and MLP output projection, before residual addition:

```text
x <- x + RMS(attention(RMS(x)))
x <- x + RMS(MLP(RMS(x)))
```

This tests Peri-RMSNorm in the transformer branches. Embeddings stay raw;
projection initialization and forward residual coefficients stay unscaled.
Existing norm sites and all parameter values remain byte-identical to the
parent; added norms use native dtype epsilon and native backward. Learned
temperatures, ordinary XSA and the rest of the passing architecture remain
unchanged. The separate RMS backward repair and first-position/prefix
stabilization are not included. Interactions with skipped or later features
are outside this single-feature screen.

Every added norm wraps an existing projection and retains its exact parameter
objects. Construction draws no random numbers, and the initial parameter
digest and RNG state match the parent. Eight focused tests verify isolation,
independent post-projection RMS equations and finite full-stack gradients.
The fixed ordinary-AdamW recipe, seed 1 and 3,000-update 99% exact-sequence
criterion apply again; another miss stops the sequence.

## Requested switch to repaired RMS backward

The active screen is now `peri-norm-repaired`, using the existing FP64
radial/tangent backward from `domain/model/rms_norm.py` at **all 25 RMS
sites**: twelve branch inputs, twelve branch outputs and the final norm.
Every existing epsilon is preserved: 1e-5 at branch inputs/final readout,
and native dtype epsilon at the new outputs. Forward outputs, all initial
parameter bytes and construction RNG state match native Peri-RMSNorm exactly.
The complete compiled validation metrics at initialization match too.
Ten focused tests include all-site radial derivatives and native/repaired
forward equality.

The native-backward `peri-norm` run was interrupted at the user's request
to switch immediately. Its last completed observation was update 2,050:
0/512 exact sequences and 7.28% token accuracy. This is **not a completed
failure** at the declared 3,000-update horizon. Its full observed history,
protocol and interruption record are retained in `results/`; no endpoint
weights or optimizer state were saved. It is not used as a passed parent.

The repaired case starts fresh from the passing SwiGLU initialization under
the same ordinary AdamW recipe. Raw embeddings, unscaled initialization and
forward residual coefficient 1 remain fixed. This is the explicitly requested
Peri-plus-backward-repair combination; no conclusions about the numerical
repair follow from the interrupted native run. The remaining stability,
attention and epsilon changes remain separate. The 3,000-update screen and
first-99% stopping rule are unchanged.

**Repaired Peri-RMSNorm missed the screen:** 0/512 exact sequences and
286/4,096 correct queries (6.98%) at 3,000 updates. Eager replay reproduces
the endpoint exactly; source, sampler and optimizer state checks pass.
The repair was verified at all 25 sites and by compiled radial-gradient
probes at actual pre- and post-norm sites. Its failure therefore cannot be
reported as a run that accidentally used native RMS backward. This result
applies to Peri branches with raw embeddings and residual coefficient 1;
joint changes or a longer training horizon remain untested.

The next requested continuation skips Peri branch norms. Since the RMS repair
is now requested as a fixed choice, `rms-repaired` first checks the passing
SwiGLU architecture with only its thirteen existing input/final RMS backward
paths repaired. All epsilons remain 1e-5, and all forward outputs, parameter
bytes and RNG state match the prior SwiGLU control. There are no embedding
or branch-output norms and no scaling changes. Twelve focused tests pass.

If this repaired-only control passes, the next architectural feature is fixed
**forward residual scaling** by `1/sqrt(2*n_layer)`. It will multiply branch
outputs during the forward pass, while leaving the projection initialization
unchanged. This is a distinct parameterization and AdamW update geometry from
the earlier scaled-initialization test. The repaired-only control must pass
before that feature is implemented and trained.

**The repaired-only control passed:** first 99% at 2,400 updates, with
507/512 exact sequences and 4,091/4,096 correct queries. Eager replay,
optimizer/sampler state and all source hashes verify. The public `GPT`
now uses this passing repaired-SwiGLU architecture. All thirteen input/final
norm epsilons remain 1e-5; numerical repair adds no parameters and changes no
forward function. The broad domain/application suite passes 842 tests,
with 40 skipped; fourteen focused isolation/gradient tests pass.

The active `residual-scale` candidate multiplies every attention and MLP
output by `1/sqrt(12)` before residual addition. It reuses every initial
parameter and the parent's RNG state, keeps all thirteen repaired RMS sites,
and adds no normalization. Both the initializer scale and embedding/Peri
norm settings remain unchanged. The raw parameter digest therefore equals
the repaired parent, unlike the earlier scaled-initialization candidate.
The ordinary-AdamW recipe and 3,000-update screen remain fixed.

**Forward residual scaling missed the screen:** 478/512 exact sequences
(93.36%) and 4,062/4,096 correct queries (99.17%) after 3,000 updates.
Its best exact accuracy was 481/512 (93.95%) at update 2,650; it never
reached 99%. At the repaired parent's 2,400-update crossing it scored
451/512 exact sequences (88.09%). Initial weights, data, optimizer settings
and all thirteen repaired RMS sites match the parent. Only the forward
branch coefficient changes, from 1 to `1/sqrt(12)`. Eager replay matches
every endpoint count; complete sampler/optimizer state and source hashes
verify.

**Current decision:** stop here and retain ordinary AdamW + SwiGLU + repaired
RMS, raw embeddings, unscaled initialization and forward coefficient 1
(`21ed2f6`). Fourteen focused tests and the 842-test domain/application
suite pass; 40 tests are skipped. This is a single-seed finite speed
regression, not a claim that scaling cannot eventually learn. The next
untested feature is Q/K RMS normalization, retaining learned temperatures;
no later feature has been trained after this miss.

```bash
.venv/bin/python -m research.recall_ablation.run --stage swiglu --seeds 1 --max-updates 3000
.venv/bin/python -m research.recall_ablation.run --stage scaled-init --seeds 1 --max-updates 3000
.venv/bin/python -m research.recall_ablation.run --stage embedding-norm --seeds 1 --max-updates 3000
.venv/bin/python -m research.recall_ablation.run --stage peri-norm --seeds 1 --max-updates 3000
.venv/bin/python -m research.recall_ablation.run --stage peri-norm-repaired --seeds 1 --max-updates 3000
.venv/bin/python -m research.recall_ablation.run --stage rms-repaired --seeds 1 --max-updates 3000
.venv/bin/python -m research.recall_ablation.run --stage residual-scale --seeds 1 --max-updates 3000
```

After the Peri-RMSNorm screen, the untested remaining sequence is
residual scaling, Q/K normalization, removal of learned temperatures,
centered XSA and its clipping threshold, fused SDPA, first-position and
repeated-prefix stabilization, native dtype epsilon, and the separated RMS
backward. Normalization and temperature removal remain separate changes.
Later features are implemented and trained only after their predecessor passes.
The original optimizer-only evidence below remains unchanged. Its stop decision
is historical; the requested continuations above explicitly skip failed features.

## Completed result: stop at Schedule-Free AdamW

The restored baseline is committed as `b7b8e89`. In the fresh CPU comparison,
changing only the optimizer loses the rapid hard-recall transition:

| Unchanged GPTMini, seed 1 | First 99% exact validation | Endpoint exact sequences | Endpoint query accuracy |
| --- | --- | --- | --- |
| AdamW | 2,900 updates | 507/512 = 99.02% | 4,091/4,096 = 99.88% |
| Schedule-Free AdamW | Not reached by 3,000 | 0/512 = 0% | 411/4,096 = 10.03% |

At the same 2,900 updates, Schedule-Free also has 0/512 exact sequences.
Both arms start from byte-identical parameters and use identical splits,
batch order, learning rates, warmup, betas and decay. Only the optimizer
changes. The baseline's independent latest-value oracle checks every query;
it answers 2,712/2,714 queries of overwritten keys correctly and predicts
no superseded values. Eager checkpoint replay reproduces the compiled
endpoint prediction counts. Optimizer steps, epoch tails, sampler RNG state,
first-crossing observations and source snapshot hashes are verified.

**Original decision at `5bec856`:** retain GPTMini + AdamW and stop adding features. SwiGLU,
normalization changes, residual scaling, temperature removal and the remaining
numerical changes have not been tested in this ablation. The result is one
fresh matched seed and a finite speed screen; it does not establish permanent
inability to learn or explain all effects of the newer architecture.

[`results/summary.json`](results/summary.json) records the decision, matching
conditions, test coverage and limitations. The same folder retains both full
50-update histories, complete result metadata and endpoint replay audits.
Raw checkpoints and exact executed source snapshots remain under
`runs/recall-ablation/`. The replay script is retained there as
`audit_replay.py`. The baseline started before its restoration commit was
created; its metadata therefore records parent commit `df1e9f7` plus verified
snapshots of the restored source. Training trajectories need not reproduce
historical GPU/PyTorch crossing counts bit for bit.

Validation: 828 domain/application tests passed, 40 skipped; changed Python
files pass Ruff. The whole-repository suite could not collect optional
corpus/leaderboard tests because dependencies including `pyarrow` are absent.

The frozen `GPTMini` implements the original baseline: PreNorm, native parameter-free
RMSNorm with epsilon 1e-5, unit Q/K with learned head temperatures, RoPE,
ordinary XSA, ReLU squared with hidden width 4d, tied embedding/readout,
and one Normal(0, 0.02) initialization per distinct matrix. Default
`train`/`DiffGPT.train` uses native AdamW, betas (0.9, 0.98), decay 0.1,
rate 0.0003 and the historical 50-update warmup starting at zero.
The public `GPT` additionally includes the passing SwiGLU change described above.

The former production architecture remains available as `NormalizedGPT`
only through its explicit module. Existing numerical and learned-division
regressions, and research runners whose checkpoints require that architecture,
continue to use it. No constructor switch changes the canonical model.

## Frozen task and decision

| Setting | Value |
| --- | --- |
| Model | width 128, depth 6, four heads; 1,249,816 parameters |
| Task | 256 keys and 256 values; 16 writes, eight overwrites, eight queries |
| Data | historical streams, data seed 1; 20,000 training and 512 validation |
| Length | uniform 48–64; right-padded to 64 for compiled execution |
| Vocabulary | historical 548 tokens, without padding |
| Batch order | shuffled epochs; generator seed = model seed + 10,000 |
| Training batch | 64, retaining the 32-example epoch tail |
| Evaluation | all 512 validation rows, every 50 updates |
| Pass | at least 99% exact sequences, all eight latest-value answers correct |
| Baseline safety budget | 4,800 updates; first crossing retained |
| First-feature screen | 3,000 updates, covering all historical 2,150–2,950 crossings |
| Test set | never constructed or read by the screen |
| Execution | CPU TorchInductor, FP32; four threads |

The quick fresh comparison uses seed 1, with the all-three-seed historical
AdamW evidence retained in `historical_baseline.json`. It establishes a matched
seed result, not a fresh three-seed replication. The initial parameters and
training/validation fingerprints must match before the candidate can run.
Every 50-update observation is retained; there is no selection by training loss
or restoration of a best checkpoint.

1. Restore GPTMini and verify its initialization, logits and gradients against
   an independent fixture generated by archived `gpt_mini.py`; validate native
   AdamW updates, inference, checkpointing and the existing regressions. Commit.
2. Reproduce a fresh GPTMini + AdamW crossing, then change only the optimizer
   to the repository's Schedule-Free AdamW. Apply exactly the same external
   warmup to both arms; disable Schedule-Free's internal warmup. If the
   candidate misses 99% within 3,000 updates, stop and retain GPTMini + AdamW.
   This cap was chosen before candidate training, to keep the experiment short;
   it must be at least the fresh baseline's crossing time.
3. Only if the optimizer passes, introduce the newer architectural changes
   separately: SwiGLU, initialization, embedding normalization, output
   normalization, residual scaling, Q/K normalization and temperature removal,
   centered XSA, repeated-prefix stabilization and the separated RMS backward.
   Later changes are not tested after the first failure.

## Running

From the repository root:

```bash
.venv/bin/python -m research.recall_ablation.run --stage baseline --seeds 1
.venv/bin/python -m research.recall_ablation.run --stage schedule-free --seeds 1 --max-updates 3000
```

The second command requires a passed matching baseline. A run refuses to
overwrite an existing seed directory. Results live under ignored
`runs/recall-ablation/<stage>/seed1/`: protocol and source snapshots,
`history.jsonl`, `result.json` and the complete `endpoint.pt`. Timing includes
compilation and evaluation. The 50-update speed check under
`runs/recall-speedcheck` is only a runtime measurement, not a failed learning
experiment.

## Historical evidence

`historical_baseline.json` preserves the lab's committed record of AdamW
crossings at 2,950 / 2,150 / 2,750 updates. `historical_schedule_free.json`
extracts the later unchanged-GPTMini controls: seed 0 crossed at 12,200;
seed 1 was still at 0% exact sequences after 100,000. It records the original
source hashes and the distinction between historical observations and this
fresh CPU comparison. Those long runs motivate testing the optimizer first;
they do not replace the matched fresh screen.
