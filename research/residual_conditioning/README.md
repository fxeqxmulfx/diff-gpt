# Conditioning the residual-attention query

This separately declared synthetic cohort tests whether the remaining huge-query
failure can be repaired while retaining the original attention function and the
complete Schedule-Free history. It does not replace residuals in the full GPT.
The original 21-case experiment and its failures stay under `research/residual_attention/`.

The correction stores a coordinate query `u` and a positive, nontrainable scale
`s`, representing physical query `q = s*u`. It evaluates query RMSNorm with
`eps/s^2`, retaining the exact finite-epsilon function. Keys keep their original
RMS epsilon, values stay raw, and neither normalization has learned gain or bias.
When the largest RMS of query, base and implied average exceeds 2, a common factor
rescales all three iterates and the gradient second moment. Averaging clocks,
optimizer steps, data and sampler history remain intact. Gradients are ordinary
autodiff derivatives in the current frame. Future physical step sizes change;
this is a declared optimization correction, not identical original trajectories.

| Cases | Variants | Scales | Seeds | Updates |
| --- | --- | --- | --- | --- |
| 24 fresh | Original Q/K, conditioned Q/K | 0, 1e-8, 1, 1e6 | 11, 17, 23 | 2000 each |
| 6 continuations | Original Q/K, conditioned Q/K | Original saved 1e6 failures | 11, 17, 23 | 10000 additional, ending at 12000 |

`protocol.json` freezes cases, source/checkpoint hashes, thresholds and the same
Schedule-Free AdamW settings before learning. The separate synthetic wall budget
is 900 seconds, checked at step/case boundaries; compilation or final checkpoint
work can extend the last boundary and is reported. Every outcome, curve, chart
change, gradient, state and failed/unfinished case is retained. No final checkpoint
is selected by heldout performance. The previous long cohort is not resumed.

The six independent unit checks cover finite-epsilon equations and all gradients,
query/base/average preservation, transformed gradient history, unchanged small
queries, the actual huge-query gradient, and exact saved-state continuation.
The compiled no-update comparison uses a separate FP64 execution and retains
all three repetitions. The companion Lean modules `QueryCoordinates` and
`QueryConditioning` formalize the real identities and immediate coordinate bound.
There is no universal convergence or floating-point error theorem.

Run in the requested Python 3.15/Torch 2.15 environment:

```sh
python -m diff_gpt.interfaces.grokking.residual_conditioning \
  --output runs/residual-conditioning-check-20261101
```

Status: protocol and implementation prepared; learning outcomes will be recorded
only after the complete declared execution and saved-state replay.
