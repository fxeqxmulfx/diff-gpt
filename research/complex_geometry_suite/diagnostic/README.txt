Fourth-power checkpoint diagnosis and matched continuation

User requested inspecting the failed complex-hermitian arithmetic-sum-pow4,
seed1 weights. This inspector reads all12 fixed case checkpoints (four arms,
three seeds), including initial parameters, final parameters, both AdamW moments,
metadata and complete training history. The weights/probe modes never call
Engine::update. The separately authorized continue mode implements the fixed
post-budget experiment in CONTINUATION_PLAN.txt.

The model's engine.rs is copied from the frozen scientific implementation.
It adds read-only activation snapshots and two explicit numerical-fix switches:
enable_stable_loss() and enable_stable_attention(). Ordinary replay and the
original continuation modes retain the frozen kernels; fixed-loss and gradient-
stress enable both corrected kernels. Layouts, data and unchanged operations
are imported from native/. Kernite embeds workspace/crate names in
PTX symbols. After replacing only the diagnostic namespace prefix with the
original prefix, the full PTX hash must match the checkpoint exactly.

Checks: original checkpoint/history/initial SHA256, declared initialization,
finite weights/moments, complex-linear matrix ties and phase variance ties.
GPU inspection reproduces complete train and held-out counts and losses from
each endpoint (loss tolerance1e-7). Both moment arrays and parameters are checked
unchanged after probing, and the input checkpoint file hash is rechecked.

Measurements: matrix norms and singular-energy participation, power-iteration
spectral norm estimate, embedding Fourier energies, saved-moment step scale,
full-split cross-entropy gradients contracted to independent complex parameters,
activation distributions, attention probabilities, Hermitian orthogonality,
errors by input pair/target/sum, and swap/constant-sum consistency on all97^2 pairs.
Relative saved-moment step scale includes decay; it is NOT the next minibatch's
AdamW update, which would incorporate a newly computed gradient and moments.
Success endpoints and exhausted endpoints have different ages; their comparisons
are descriptive and cannot by themselves establish a causal hyperparameter fix.

From this directory:
    cargo build --release --offline --locked
    cargo test --release --offline --locked
    cargo clippy --offline --locked --all-targets -- -D warnings
    target/release/complex-weight-diagnostic weights RUN_ROOT OUTPUT.json
    target/release/complex-weight-diagnostic probe RUN_ROOT OUTPUT.json
    target/release/complex-weight-diagnostic continue RUN_ROOT NEW_RUN_ROOT
    target/release/complex-weight-diagnostic audit-continuation RUN_ROOT EXISTING_RUN_ROOT
    target/release/complex-weight-diagnostic finish-continuation RUN_ROOT EXISTING_RUN_ROOT

The probe command requires CUDA access and should run when training is idle.
Eleven CPU tests pass, including sampler continuation across a partial epoch,
tail batches and rejection of meaningful summary differences while accepting
decimal roundoff. Both inspection modes completed all12 checkpoints; all
full-split replays passed. Inspection performs no training or state mutation.
No mode executes Python or overwrites the original benchmark.
Outputs are ../results/pow4-weights.json, pow4-probes.json and per-cell error TSVs.
Interpretation and limitations are in ../results/pow4-diagnosis.txt.

The continue command performs new training only in a fresh output directory.
It retains the source checkpoint/moments/global step, verifies the entire old
sampler prefix, and compares fixed LR0.002 versus0.0002 on the same next1000
epochs. It verifies identical first-step gradients/moments, the LR displacement
ratio, original checkpoint immutability and all saved continuation histories.
Each500-epoch snapshot is retained. The original benchmark miss is unchanged.
The exact first inspection sources are archived in ../results/pow4-inspection-source.

The authorized1000-epoch comparison is complete and audited. Lowering LR reduces
the final200 held-out standard deviation from3.2312 to0.04723 percentage points;
neither arm reaches99%. Fixed endpoints are97.7258% and97.7683%. Full findings
and learning curves are in ../results/pow4-lr-continuation.txt and its artifacts.

The first control finished before an exact float-summary comparison rejected
one-ULP decimal roundoff. audit-continuation checks the existing control without
training; finish-continuation only permits the documented recovery with a fully
completed control and absent lower-LR output. It verifies identical numerical
source, archives the audit fix, regenerates one discarded first-step reference,
then runs only the missing lower-LR arm. No control trajectory is replayed.
Both arms now exist, so finish-continuation refuses to rerun this completed study.

Raw continuation diagnosis

From the repository root, use this crate's release binary:
    complex-weight-diagnostic raw weights PARENT_ROOT CONTINUATION_ROOT NEW_OUTPUT_DIR
    complex-weight-diagnostic raw probe PARENT_ROOT CONTINUATION_ROOT NEW_OUTPUT_DIR
    complex-weight-diagnostic raw precision PARENT_ROOT CONTINUATION_ROOT NEW_OUTPUT_DIR

All three commands read the six retained continuation states and two successful
reference endpoints. They refuse an existing output directory and archive their
sources. weights decodes binary tensors and measures spectra, independent
coordinates, moments and directional changes. probe additionally reproduces
complete scores and evaluates temporary block transplants and equal-norm local
perturbations. precision compares native loss/gradients with a cancellation-
resistant FP64 reference on identical native logits and propagates both through
the original backward kernels. No mode performs optimizer updates or writes
input checkpoints. The last two modes require CUDA.

These checks completed in ../results/pow4-raw-weights-01, pow4-raw-probe-01 and
pow4-raw-precision-01. See ../results/pow4-raw-diagnosis.txt. The high-confidence
loss/gradient precision defect is verified; its role in the earlier original-LR
accuracy crashes and the effect of corrected-loss training remain unproven.

Corrected gradients and full-path stress audit, 2026-11-08

From the repository root, using this crate's release binary:
    complex-weight-diagnostic raw fixed-loss PARENT_ROOT CONTINUATION_ROOT NEW_OUTPUT_DIR
    complex-weight-diagnostic gradient-stress PARENT_ROOT CONTINUATION_ROOT NEW_OUTPUT_DIR
From this directory, the isolated corrected-loss CUDA stress test is:
    cargo test --release --offline --locked stable_loss::tests -- --ignored --test-threads=1

stable_loss.rs separately accumulates non-target softmax mass in FP64, evaluates
the target derivative as -other/total, and uses cancellation-resistant log1p.
It removes the lost representable gradients without changing logits or labels.
stable_attention.rs evaluates the XSA query adjoint's projections, dot products
and reductions in FP64, casting only at the existing FP32 buffer boundaries.
The original forward, model dimensions, normalizations and weight ties remain.

The initial loss-only recheck passed all18 checkpoints. The first full-path
stress run then exposed Q/K cancellation in a128-wide single-head case. Its
failed artifacts remain in ../results/gradient-path-stress-01. The XSA fix
reduces local Q/K relative error from0.00323 to2.55e-8/3.29e-8. Tolerances were
not loosened. With both fixes enabled, ../results/gradient-path-stress-02 passes
all42 full-model cases and1477 directional checks over378 parameter-block paths,
28 primitive rows and five local adjoints. It uses scalar FP64 forward-mode AD
independent of the manual backward kernels, plus analytic extreme RMS checks.
Cases include18 checkpoints and24 synthetic variants: widths64--256,
depths1--4, sequence lengths1--128, head widths16/32/128, masks, repeated-token
prefixes, zero/tiny/large branches, clipped values and confident readouts.

Checkpoint AD probes use three fixed examples each. The separate fixed-loss
command evaluates the entire training and held-out sets of all18 checkpoints,
compares every logit derivative with a stable FP64 loss reference, and propagates
both through the candidate backward kernels. This latter propagation isolates
the loss fix; the independent AD suite validates model-backward paths.
It checks parameters, moments, input hashes and bitwise forward invariance.
Both commands refuse existing output directories and never update weights.

See ../results/gradient-path-stress.txt and ../results/pow4-fixed-loss.txt.
Finite stress coverage is not a proof for every tensor, coordinate or clipping
boundary. The worst remaining full-model relative discrepancy is about0.0396%
in the nearly scale-invariant Q/K path, within the original0.05% norm-scaled
bound. FP32 tensor boundaries remain; no corrected training or convergence
claim follows from these checks. Performance of the FP64 adjoint is unmeasured.

Fresh corrected-gradient seed1 rerun

The user subsequently confirmed rerunning the EXISTING (a+b)^4 mod97 task,
not a new division task. RETRAIN_PLAN.txt fixes the protocol before training.
From the repository root, use the release binary:
    complex-weight-diagnostic retrain PARENT_ROOT NEW_RUN_ROOT
    complex-weight-diagnostic retrain-audit PARENT_ROOT EXISTING_RUN_ROOT

retrain restores the exact original initial parameters, zeroes moments and
verifies the complete100000-update sampler hash before training. It enables
both numerical fixes, preserves model/optimizer settings, stops at first99
or the original budget and retains states every500 epochs. No existing output
directory can be reused or silently replayed. The final audit checks every
retained state, history/exposure, source hashes and original-file immutability,
then reloads the endpoint and reproduces complete training and held-out scores.

The rerun completed100000 updates without99%: best98.2784%, final93.9426%,
train100%. The original best/final are98.2572%/90.6057%; accuracy fluctuations
remain. All12 retained states passed a subsequent full-split weight/gradient
inspection using the corrected kernels. Command:
    complex-weight-diagnostic retrain-weights PARENT_ROOT RUN_ROOT NEW_OUTPUT_DIR
This checks every retained epoch, writes raw block spectra/moments, full-split
gradients, activation summaries and error tables, and performs no updates.
See ../results/pow4-fixed-grad-seed1.txt and pow4-fixed-grad-epochs-01/.

Actual next-update diagnostic

NEXT_UPDATE_PLAN.txt fixes eight one-update counterfactuals at corrected
sum-pow4 seed1 epoch4000. From the repository root, use the release binary:
    complex-weight-diagnostic next-update PARENT_ROOT CORRECTED_RUN_ROOT NEW_OUTPUT_DIR
The input run is audited read-only. Every branch restores the same parameters
and uses the exact next batch from the original sampler. Compare retained
moments at LR scales 1/.5/.25/.1, first-moment reset, fresh Adam restart, and
both reset directions matched to the original displacement norm. The fresh
restart uses current LR with no warmup restart. Matching uses no loss metric.
The command saves all eight p/m/v states, scores, gradients, source fingerprints
and independent scalar FP64 AdamW checks; it refuses an existing output path.

Completed artifact: ../results/pow4-next-update-01/. All eight branches pass.
The ordinary update raises full-training CE 27.5657%; one tenth LR still raises
it 2.1431%. Resetting only m lowers CE 2.6166% at the same LR and 10.0749% at
matched displacement norm. All eight blocks' first-order contributions flip
from uphill to downhill in the matched first-moment reset. Norms and residual
connections are unchanged. This is a local momentum diagnosis, not a sustained
convergence claim. See ../results/pow4-next-update.txt and its validation JSON.
The current CPU suite passes 12 tests; one CUDA test is ignored in that suite.

Matched 500-epoch first-moment continuation

MOMENTUM_CONTINUATION_PLAN.txt fixes the two subsequent trajectories, from
corrected epoch4000 through epoch4500. From the repository root:
    complex-weight-diagnostic momentum-continue PARENT_ROOT CORRECTED_ROOT ONE_STEP_ROOT NEW_OUTPUT_ROOT
    complex-weight-diagnostic momentum-audit PARENT_ROOT CORRECTED_ROOT ONE_STEP_ROOT EXISTING_OUTPUT_ROOT
keep restores all p/m/v; reset_first zeros m once, retaining v, Adam counter,
LR and every subsequent sample. Both run exactly 9500 updates. Each epoch's
complete splits include diagonal and non-diagonal scores. History is buffered
and state is saved only at the start/end, 500 epochs apart, or explicit STOP.
First99 is recorded without shortening the fixed diagnostic comparison.

Completed output: runs/complex-pow4-momentum-500-01 at repository root.
Both miss99: best98.2147%/98.1084%, final98.1722%/90.4145% (keep/reset).
The control reproduces all501 original observations and final p/m/v bytes.
Both first updates reproduce the earlier diagnostic bytes. All sampler,
source/state hashes, subgroup accounting and endpoint reloads pass. The current
CPU suite passes13 tests, with one CUDA-only test ignored. One reset is not a
lasting fix. See ../results/pow4-momentum-500.txt and full TSV/SVG/PNG curves.
The separate ../results/pow4-flat-spectrum-ffn-review.txt compares the current
radial gate to the latest plain-product source theorem; it changes no kernels.

Plain complex-product FFN candidate

PLAIN_FFN_PLAN.txt fixes the single-gate ablation. The new plain_product.rs
implements g*u/sqrt(2) and its real-coordinate adjoint in Kernite, with the
existing half-split hidden layout. Engine::enable_plain_product opts in;
default radial behavior and frozen native sources remain unchanged. The
FP64 scalar AD oracle selects the same mathematical variant explicitly.
The training protocol records FFN choice and plain-kernel PTX hash. Reload
audits use that choice, and the older radial-only weight probe rejects plain
checkpoints instead of silently using the wrong model.

From the repository root, with the diagnostic release binary:
    complex-weight-diagnostic plain-ffn-stress PARENT_ROOT RADIAL_ROOT NEW_STRESS_ROOT
    complex-weight-diagnostic plain-ffn-train PARENT_ROOT RADIAL_ROOT STRESS_ROOT NEW_OUTPUT_ROOT
    complex-weight-diagnostic plain-ffn-audit PARENT_ROOT RADIAL_ROOT EXISTING_OUTPUT_ROOT
The training command requires a passed preflight with identical compiled-source
and kernel fingerprints. It restores the original seed1 initialization and
zero moments, matches the100000-update sampler and optimizer settings, stops
at first99 or budget, and retains500-epoch buffered checkpoints. Complete
scores include diagonal/non-diagonal counts. Every retained checkpoint is
reloaded, and original/control files must remain unchanged. This targeted
known-miss experiment is not universal-model selection. No Python is used.

The initial plain preflight exposed a composed Q/K precision failure in the
single128-wide-head case, despite passing individual product/RMS/XSA adjoints.
Its source and failure remain in ../results/plain-ffn-stress-01. stable_chain.rs
recomputes the XSA/output-projection/RMS Q/K adjoint together in FP64 for that
single-head configuration. The ordinary forward and other adjoints remain.
This dispatch is inactive for the actual four-head32 training comparison.
The full preflight adds six wide-head cases and a radial wide-head regression;
the original error bounds are unchanged. Traces and the unsuccessful narrower
projection/RMS-only attempt remain in the results directory.

Completed plain run: runs/complex-pow4-plain-ffn-seed1-01 at repository root.
It uses the full100000-update budget and never reaches99%. Best98.8098%,
final90.0531%, versus corrected radial best98.2784%, final93.9426%. All12
states replay and5265 observations have matching control exposure. At its
best epoch4233, train100%, held-out diagonal16/42 and off-diagonal4633/4663.
See ../results/pow4-plain-ffn.txt, exact JSON, validation JSON and full curves.
The current CPU suite passes14 tests; preflight/provenance audits pass.

Ordinary attention candidate

NO_XSA_PLAN.txt fixes the next single-component ablation against the completed
plain-product XSA run. ordinary_attention.rs consumes the unchanged frozen
SSMax probability kernel, overwrites its unused XSA output with ordinary causal
values and supplies the ordinary softmax adjoint. First/shared attention is
restored; zero prefix flags disable shared-output and backward masking. The
existing Peri/embedding norms, residual scale, complex product and tied readout
remain. Protocol attention/FFN tags and PTX hashes select the correct replay.

Commands, from repository root with the diagnostic release binary:
    complex-weight-diagnostic no-xsa-stress PARENT_ROOT PLAIN_XSA_ROOT NEW_STRESS_ROOT
    complex-weight-diagnostic no-xsa-train PARENT_ROOT PLAIN_XSA_ROOT STRESS_ROOT NEW_OUTPUT_ROOT
    complex-weight-diagnostic no-xsa-audit PARENT_ROOT PLAIN_XSA_ROOT EXISTING_OUTPUT_ROOT
Training requires the exact source fingerprint that passed preflight. The
preflight includes scalar AD, all compatible complex full-model cases, every
retained plain-XSA state and explicit first/repeated-token V/output gradients.

Completed output: runs/complex-pow4-no-xsa-seed1-01 at repository root.
First99 is2648 epochs/50312 updates:4658/4705 correct,32/42 diagonals,train100%.
The matched-window XSA best is4636/4705 with10/42 diagonals. All7 retained
states replay and2649 exposures pair exactly. Fifteen CPU tests and the full
GPU preflight pass. The first preflight's signed-zero bit-check issue and its
correction are preserved; no kernel or tolerance changed. Oscillations remain
and the trajectory stops at first99. See ../results/pow4-no-xsa.txt and full
curves/validation. This candidate has not replaced the frozen universal model.

Causal centering candidate

CENTERING_PLAN.txt freezes the user-requested causal adaptation of
arXiv:2410.07799v3 against the successful ordinary-attention seed1 control.
The standalone Lean checks in ../results/pow4-centering/ prove cancellation of
constant values and shared-parameter gradients, while retaining the residual;
they do not prove that the complete model cannot solve the task.

Commands, from repository root with the diagnostic release binary:
    complex-weight-diagnostic centering-stress PARENT_ROOT ORDINARY_ROOT NEW_STRESS_ROOT
    complex-weight-diagnostic centering-train PARENT_ROOT ORDINARY_ROOT STRESS_ROOT NEW_OUTPUT_ROOT
    complex-weight-diagnostic centering-audit PARENT_ROOT ORDINARY_ROOT EXISTING_OUTPUT_ROOT

centered_attention.rs evaluates centered values and their V adjoint with FP64
accumulation. The first preflight exposed near-uniform probability cancellation
in the qk_tiny model; all original failures remain in centering-stress-01.
The correction computes small centered coefficients before FP32 probability
rounding, using anchored score differences and a six-term expm1 polynomial
only within1e-3. Other rows use centered frozen probabilities. The ordinary
Q/score adjoint and SSMax scale are retained. No tolerance has been relaxed.

The independent FP64 model computes all token paths without the native
constant-prefix shortcut. Full preflight covers37 candidate models,4 ordinary
regressions and36 primitive rows, including uniform and near-uniform scores.
Training requires the exact source that passes all checks. It saves buffered
states every500epochs and stops atfirst99 or100000updates. Comparison audits
use only the common observed prefix: the ordinary control stopped at50312
updates, so its later trajectory must be reported as unobserved.

The second preflight fixes near-uniform scores but exposes coefficient-storage
rounding in two Q/K directions of the wide two-layer case. FP64 score and
coefficient buffers remove that extra rounding. The complete third preflight
passes all cases and1480 parameter directions/30624 primitive coordinates;
both earlier attempts remain archived. See ../results/centering-stress.txt.

Completed output: runs/complex-pow4-centered-seed1-01 at repository root.
The candidate exhausts100000updates,peaks at4.1870% held-out and ends at2.9968%.
All12 retained states replay. At50312 matched updates,ordinary scores99.0011%
and centering3.2306%;the control is unobserved afterward. Do not promote this
candidate. See ../results/pow4-centering.txt,exact JSON,audits and full curves.
