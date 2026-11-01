# Centered XSA preserves the real model and removes repeated-value cancellation

The original XSA computes an attention average and subtracts its projection
onto the current value. When all values coincide, this subtracts two equal
nonzero vectors. Roundoff then enters Peri-RMSNorm and can amplify over depth.
The source already zeroed the first position explicitly; repeated values at
later positions expose the same issue.

The optional centered MHA centers values before attention, computes the residual
relative to the own value, then projects. A separate own-value remainder
preserves the original clipped normalization below epsilon. The accompanying
Lean file `SequenceModel/Attention/CenteredXSA.lean` proves exact equivalence
for arbitrary normalized weights and for the actual normalized RoPE head.
Q/K normalization, attention weights, RoPE, residual addition and trainable
parameter counts retain their mathematical meaning. `centered_xsa=False`
selects legacy arithmetic in `MultiHeadAttention` for historical replay.

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
with FP64 (cosines.0066--.0624), despite accurate forwards. Centering alone is
therefore opt-in. The original model remains the default while a mathematically
justified repeated-token output path is checked. This failure is retained in
`parameter_gradient_audit.json`; no successful forward test is treated as a
complete backward-accuracy guarantee.
