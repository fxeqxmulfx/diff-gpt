# Prepared inverse-width numerical repair

The [plan](plan.json) prepares the requested follow-up to the existing 40/50
complete-model numerical result. Scientific execution waits for the Muon study's
committed closure. No new stress result or benchmark learning improvement has
been measured here.

Four variants isolate range-safe RMS, widened XSA, widened SwiGLU, and their
combination. Each preserves the real equation, original weights, normalization
epsilon, fixed rotary tables and residual scale. The attention adapter must
retain the exact `1 / head_width` gain; borrowing the old generic repair without
that override would change the model.

After implementation validation and a source freeze, repeat all 50 applicable
parent cases for every variant. The reference is an unrepaired FP64 copy of the
original inverse-width model. Compare logits, loss and every parameter gradient
using the unchanged 0.001 thresholds. Preserve all ten parent failures, ordinary
controls and any new regression. The 15 old learned-frequency cases remain
inapplicable to fixed RoPE.

These probes will test arithmetic at the declared magnitudes. They will not
establish universal stability, faster learning, or a new short-suite winner.
