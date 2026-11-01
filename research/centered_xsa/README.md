# Centered XSA preserves the real model and removes repeated-value cancellation

The original XSA computes an attention average and subtracts its projection
onto the current value. When all values coincide, this subtracts two equal
nonzero vectors. Roundoff then enters Peri-RMSNorm and can amplify over depth.
The source already zeroed the first position explicitly; repeated values at
later positions expose the same issue.

Centered MHA centers values before attention, computes the residual
relative to the own value, then projects. A separate own-value remainder
preserves the original clipped normalization below epsilon. The accompanying
Lean file `SequenceModel/Attention/CenteredXSA.lean` proves exact equivalence
for arbitrary normalized weights and for the actual normalized RoPE head.
Q/K normalization, attention weights, RoPE, residual addition and trainable
parameter counts retain their mathematical meaning. `GPT(centered_xsa=False)`
selects the complete legacy path for historical replay. Direct MHA/Block
construction retains its legacy default: centering alone is insufficient for
deep backward accuracy.

Independent float64 equations and all score/value/own gradients agree,
including zero, below-epsilon, boundary and above-epsilon cases. Finite
differences validate both smooth clipping regions. Whole-module input and
parameter gradients agree. The clipped boundary uses the same clamp backward
convention as F.normalize; equality of real formulas alone would not select
that autodiff convention. Cached raw values stay intact; full prefill,
chunked decode and shared-prefix paths pass the existing tests.

`checkpoint_replay.json` audits eight frozen lab checkpoints on the same first
512 heldout rows, without training. FP64 copies retain the original FP32
averaged parameter values, rotary tables and RMS epsilon2^-23. On repeated
inputs, original deep-model FP32/FP64 logit differences range5.33--8.09 and
change seven predictions across four batches. With centering, differences
are below5.4e-6 and no predictions disagree across precisions. Every repeated
first attention output is exactly zero. Even original FP64 has small amplified
cancellation, up to.000943 relative to centered FP64. No result is discarded.

This verifies a numerical correction on the tested domain. It does not prove
uniform error bounds or improved generalization. Changing arithmetic can
change a historical trajectory or prediction. The long lab experiments retain
their frozen original implementation. Centering cached prefixes adds value
transforms and temporary storage; a cache-speed improvement is not claimed.
The CPU source-model suite and four additional centered-XSA checks pass.
Compiled CUDA checks and the historical div97 learning regression are tracked
as follow-up validation after the running frozen GPU phase.

A subsequent full-model parameter-gradient audit found an additional failure:
for three deep checkpoints, centered FP32 gradient directions disagreed strongly
with FP64 (cosines.0066--.0624), despite accurate forwards. Centering alone was therefore restored to opt-in before the common-path
correction was implemented. This failure is retained in
`parameter_gradient_audit.json`; no successful forward test is treated as a
complete backward-accuracy guarantee.

The full GPT now defaults to centered XSA **plus guarded common-prefix states**.
After each block, still-constant token prefixes share the already computed
first-position result. Every first-position value head must be strictly above
1e-12; clipping permanently disables sharing for that row. The guard reuses
the actual projection values, with no extra matrix multiplication. Future
nonconstant tokens are allowed. Lean's `ConstantTokens` and `ConstantPrefixes`
prove the required source-state equalities for arbitrary RoPE and finite depth.
They do not claim arbitrary independent-position gradient equivalence or a
floating-point error bound. Cached forwards use centering without prefix-state
sharing, so cache backward stability is outside this diagnostic claim.

`guarded_parameter_replay.json` is the reproducible audit of the **actual GPT**,
not just the lab prototype. All eight frozen checkpoints retain all six
legacy/centered/guarded FP32/FP64 comparisons. Centering-only parameter-gradient
RMS error reaches2.284; guarded errors are1.78e-10--2.57e-8. Guarded all-row
logit errors are below1.88e-5 and predictions agree across precisions. The
companion script is `experiments/modular_composition/parameter_precision.py`
in the transformer repository; source and checkpoint hashes are in the JSON.
The final affected CPU suite passes123 tests with14 skips. Three new tests
check a separate FFN-only common-path oracle and all shared-parameter gradients,
arbitrary suffixes, and the clipped case that must disable sharing. No result
here establishes a learning improvement; the compiled regression records that
question separately.
