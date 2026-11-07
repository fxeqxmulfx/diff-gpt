Complex geometry transformer

A separate native Rust/Kernite model built while complex_ffn_suite trains.
The existing experiment's sources and outputs are not dependencies of this
crate. UPSTREAM.sha256 records the source snapshot used for the port.

The candidate uses complex embeddings, complex-linear Q/K/V, attention-output
and FFN projections, magnitude-based RMS normalization, complex RoPE, a radial
complex-product FFN, Hermitian XSA and a real Hermitian tied readout. Real CUDA
GEMM implements each complex matrix through tied blocks [[A,-B],[B,A]].
All forward/backward kernels and the runner are native Rust/Kernite; no Python.

Comparison variants

arm                        projections       XSA         AdamW variance
real-product               unrestricted real real        per real coordinate
complex-linear             complex-linear   real        per real coordinate
complex-hermitian           complex-linear   Hermitian   per real coordinate
complex-hermitian-phase     complex-linear   Hermitian   shared per complex pair

The first contrast isolates complex-linear constraints, the second isolates
Hermitian exclusion, and the third isolates the optimizer's variance rule.
All four use the same gated complex product, width, hidden width, depth and
batch order. The complex variants share identical initial weights. Dimensions
are matched; parameter counts are not. For the cubic task, width128, 4 heads,
one block, vocabulary97 and real hidden width342 give 209280 expanded real
storage entries: 209280 independent parameters in the real control and 110848
in each complex variant. The expanded implementation saves no GEMM FLOPs.

Which components need complex treatment

For a complex-linear projection, independent A and B determine all four real
blocks. Its real gradient contracts as gA=gRR+gII, gB=gIR-gRI BEFORE AdamW's
moments are updated. Checkpoint load/save enforces exact ties in weights and
optimizer state. Ordinary AdamW is valid for complex variables represented by
real/imaginary parts, but its two variance estimates depend on those axes.
The phase variant uses v <- .98*v+.02*(gR^2+gI^2)/2 with a complex first moment.
The same positive denominator scales both components, making this update
covariant under rotations of complex coordinate axes.

For nonzero v above the norm clip, Hermitian XSA is
    a - v * sum(conj(v_j)*a_j) / sum(|v_j|^2).
It removes both real directions v and i*v, unlike the original real XSA.
The denominator is clipped at norm1e-12; the first query is explicitly zero,
and the original stable constant-prefix handling remains. The GPU computes
centered deltas. The validation oracle computes direct complex inner products.

RMS already depends on complex magnitudes. Preserve its original scale
sqrt(f32::EPSILON + sum(|z|^2)/(2m)), where m is complex width. Changing to a
mean over m would introduce an additional scale change. RoPE is already complex
phase multiplication. Attention uses real scores Re<q,k> and real softmax
probabilities to mix complex values. Classification uses real logits Re<E_y,x>.
These real quantities are appropriate; complex-valued probabilities or losses
are unnecessary. Componentwise ReLU/SwiGLU are absent from the four study arms;
the legacy SwiGLU path exists only for regression tests.

The product sigma(sqrt(|g|^2+eps))*g*u/sqrt(2) adds phases and is deliberately
quadratic. It is not globally phase-equivariant through residual addition.
Magnitude gates, RMS and real attention scores are nonholomorphic but are
differentiable as real maps. The manually implemented backward follows those
real derivatives. No claim of optimality, convergence or a 99% result is made.
The Lean flat-spectrum bridge remains conditional; it does not prove that this
architecture or either optimizer will learn the frozen suite.

Build and validate

From research/complex_geometry_suite/native:
    cargo build --release --offline --locked
    cargo fmt --check
    cargo clippy --offline --locked --all-targets -- -D warnings
    cargo test --release --offline --locked
    cargo test --release --offline --locked -- --ignored --test-threads=1
    target/release/complex-geometry preflight
    target/release/complex-geometry smoke /tmp/complex-geometry-smoke-new

The CUDA tests require driver access. The small smoke command performs eight
updates per arm on the real (a+b)^3 mod97 data, scores the complete splits and
saves once at the end. It writes under engineering/, outside scientific cells.
It checks training execution; eight updates are not a generalization study.
VALIDATION.json records the build checks and smoke evidence.

Kernite path dependencies point to /home/misha/rust_projects/kgen/crates.
The validated backend is a1b6bc2c956e1a5f35960e26b209e1b6c213096c.
The older independent-coordinate layout/optimizer derives from GPT's
complex_transformer at commit786696b. The frozen data/scorers, arbitrary-length
engine and radial product derive from the current complex_ffn_suite snapshot;
its exact original file hashes are retained. THIRD_PARTY.txt preserves the
data-generator compatibility code's attribution and license.

Scientific runs when GPU time is available

    target/release/complex-geometry suite ../runs/cohort-01 short
    target/release/complex-geometry suite ../runs/cohort-01 long
    target/release/complex-geometry report ../runs/cohort-01

Or run the cubic candidate alone with the frozen budget:
    target/release/complex-geometry cell ../runs/cubic-01 arithmetic-sum-cube 0 complex-hermitian-phase long

The manifest contains 25 short and 10 long tasks; 3 seeds and 4 arms give420
cells. The score remains complete held-out exact accuracy >=99%. Budgets,
data seeds, splits and observation cadence remain frozen. Arm order rotates
with the seed. Misses, pending cells and execution errors remain visible.
Long tasks are diagnostics, not a tuning set. No full cohort was launched
as part of building this version.

Observations stay in memory between saves. Checkpoints occur every500 epochs,
plus initialization, first99 crossing, budget end or clean stop. Create STOP
in the cohort or cell directory to request a clean stop at an observation.
Remove STOP and use the same cell arguments with resume instead of cell.
Only clean stopped cells can resume; existing cells are never overwritten.
An unclean work window refuses silent replay. Parameters, both moments, history,
sampler position, hashes and a compile-time source snapshot accompany each run.

The user subsequently authorized testing this model on 2026-11-07. See
QUEUED_STUDY.txt: run-after-ffn.sh schedules all420 cells after the current
210-cell complex_ffn_suite finishes and is audited. Runtime status is in
/home/misha/python_projects/gpt/runs/complex-geometry-suite/queue-state.json.
The independent native audit writes results/audit.json and results/audit.tsv.

All420 cells subsequently completed and passed the integrity audit. Long passes:
real-product23/30, complex-linear29/30, complex-hermitian29/30, phase25/30;
all arms passed75/75 short. The user's fourth-power seed1 weight inspection is
in diagnostic/ and results/pow4-diagnosis.txt. It preserves the frozen results.

The subsequently authorized matched continuation of failed Hermitian pow4 seed1
is complete:1000 additional epochs per arm, identical initial state and samples.
LR0.0002 reduces final200 accuracy standard deviation from3.2312 to0.04723
percentage points versus LR0.002, but neither reaches99%. Final held-out
accuracies are97.7683% and97.7258%. This isolates a strong sensitivity to the
whole AdamW step, including its LR-scaled decay; it does not localize a faulty
block or change the frozen benchmark. See results/pow4-lr-continuation.txt,
the audited JSON, paired TSV and SVG/PNG learning curves.

The subsequent raw-weight diagnosis reads all six continuation states and
checks inference-only block substitutions. It finds no exploding saved weights;
stored momentum is locally uphill in every block at the weak +500 snapshot.
A separate precision oracle confirms severe loss/gradient cancellation at the
high-confidence lower-LR endpoint, including lost target derivatives on3628/4704
training examples. It does not yet establish the cause of the earlier crashes
or a convergence fix. See results/pow4-raw-diagnosis.txt and its complete raw,
intervention and precision artifacts. Frozen kernels and outcomes are preserved.

The requested gradient fix and stress audit are complete. The separate
diagnostic candidate fixes softmax target-gradient/loss cancellation and an
XSA Q/K adjoint cancellation found by the stress suite. All18 saved pow4
checkpoints pass complete-split rechecks with bitwise-identical logits and
predictions. Lost target derivatives at the lower-LR endpoint fall3628 to0.
Independent FP64 AD checks pass42 model cases and1477 parameter-direction
checks, plus28 primitive rows and five local adjoints. The first failing
head-width128 case is retained; FP64 XSA accumulation fixes it without changing
the original tolerances. See results/pow4-fixed-loss.txt and
results/gradient-path-stress.txt. No new training or convergence claim is made.

The subsequent corrected-gradient sum-pow4 seed1 rerun is complete and audited.
Identical initial weights, zero moments, sample order and100000-update budget:
best98.2784%, final93.9426%, train100%;99% remains unreached. Original best/final
were98.2572%/90.6057%. All12 retained epochs passed raw-state and full-split
gradient inspection. At the weak4000-epoch state, saved directions oppose the
training gradient in all8 blocks and FFN historical relative steps reach0.81%;
weights remain finite and normalization/projection checks are healthy. This
suggests a targeted optimizer-update diagnostic, not an established causal fix.
See results/pow4-fixed-grad-seed1.txt, paired curves and pow4-fixed-grad-epochs-01.

The subsequent actual next-update diagnostic confirms a harmful momentum
direction at the corrected epoch4000 checkpoint: normal AdamW raises full-train
CE 27.5657%, and one tenth LR still raises it 2.1431%. Resetting only the first
moment at equal displacement norm lowers CE 10.0749%, with all eight blocks'
gradient/displacement products becoming negative. All eight disposable branches
pass independent scalar optimizer checks and input-state preservation audits.
Norms, residual connections and tied weights remain identical. This isolates
a local optimizer-state effect; it does not establish a long-run fix or compare
alternative normalizers/residuals. See results/pow4-next-update.txt and the full
pow4-next-update-01 artifacts. No sustained continuation was performed.

The subsequently authorized500-epoch continuation is complete: keep moments
versus resetting only m once, with identical epoch4000 state, LR and batches.
Neither reaches99%. Best held-out is98.2147%/98.1084%; final98.1722%/90.4145%;
final200 mean94.1218%/91.6777% (keep/reset). The local one-step reset benefit
does not persist as a better trajectory. All501 control observations and final
p/m/v bytes reproduce the original interval. Full subgroup curves, validation,
source and state references: results/pow4-momentum-500.txt.

The requested flat-spectrum reread finds a source distinction: the latest Lean
CP bridge covers plain complex multiplication; our FFN has a radial gate.
That suggests a single-gate ablation, not a proved convergence fix. Its recovery
and margin hypotheses do not imply AdamW discovers the required features.
A Rust arithmetic check also shows the quartic task has an available rank-four
scalar label readout; its existence does not prove learned classification.
See results/pow4-flat-spectrum-ffn-review.txt and pow4-spectrum-review.rs/.json.
At that review stage no plain-product transformer experiment had run; the
subsequently authorized single-gate ablation is recorded below.

The plain-product seed1 ablation completed100000 matched updates and passed its
audit. Best held-out98.8098% exceeds corrected radial98.2784%, but neither
reaches99%; final90.0531% versus93.9426%. At the plain peak, off-diagonal accuracy
is4633/4663 (99.3566%), while held-out diagonals are16/42 and training is100%.
Oscillations remain; the largest one-observation drop is26.865 percentage points.
All12 retained states replay and all5265 observations match control exposure.
The numerical preflight passes40 plain cases,3 radial controls,1534 parameter
directions,32 product rows and6 local adjoints. The wide-head precision fix is
inactive for this four-head32 training comparison. See results/pow4-plain-ffn.txt,
the full curves/JSON and results/plain-ffn-stress.txt. This remains a single known-
miss diagnostic; norms/residuals/tied weights and frozen results are retained.

The subsequent diagonal/Lean review identifies an attention mechanism: equal
values have ordinary attention y=v but Hermitian XSA cancels that value. Native
first/shared-prefix suppression preserves this cancellation. The plain product
itself does not cancel equal nonzero factors. Checked algebra, the new answer-
class theory, caveats and the recommended ordinary-attention ablation are in
results/pow4-diagonal-lean-review.txt. No no-XSA training has run; this is a
specific hypothesis rather than a demonstrated explanation of the99% failure.

The subsequently authorized ordinary-attention ablation reaches99.0011%
(4658/4705) at2648 epochs/50312 updates and stops atfirst99. It retains the
plain FFN, norms,residual scale,tied weights,QK/RoPE and SSMax. Held-out diagonals
are32/42; the matched-window XSA best has10/42 and equal non-diagonal correctness
(4626/4663). XSA never reaches99 in100000 updates. All7 retained states replay,
2649 observations pair by exposure, and full preflight/provenance checks pass.
The run still oscillates and does not establish sustained99% or universal
superiority. See results/pow4-no-xsa.txt, exact statistics,validation JSON and
full SVG/PNG/TSV curves. Frozen benchmark and the stopped capacity queue remain.

The user-requested causal adaptation of arXiv:2410.07799v3 is rejected for this
seed1 comparison. Pure prefix centering peaks at4.1870% and ends at2.9968%
held-out accuracy after100000updates. At the ordinary control's50312-update
99.0011% crossing,centering scores3.2306%,with1/42 diagonals versus32/42.
Non-diagonal generalization also fails,despite100% training at the peak.
The Lean proof establishes attention-path cancellation,not full-model
impossibility. The passing Rust/Kernite preflight includes37 models,4 ordinary
regressions,1480 parameter directions and30624 primitive coordinates; two
precision failures were fixed and preserved. All12 saved states replay and all
provenance checks pass. Control comparisons use only2649 common observations.
See results/pow4-centering.txt,complete curves,validation and Lean certificates.
The ordinary model remains the incumbent for this comparison; frozen benchmarks
and the stopped capacity queue remain unchanged.
