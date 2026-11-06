# Single additions to the original model

The [plan](plan.json) follows the requested return to the old architecture.
The control is the completed original-model AdamW LR×2 run. Three candidates
individually add embedding RMS, scaled Peri-RMS, or SSMax with the original gain.
All retain Q/K RMS, centered XSA, SwiGLU, fixed RoPE and tied weights. Original
parameters, initialization, data, seeds, budgets and scoring remain fixed.

This benchmark was selected from the original model's successful seed-1 runs.
The historical long subset passed 10/10 with that seed; the later three-seed
baseline passed 29/30. These comparisons measure preservation and acceleration
of known abilities. They do not establish superiority on an independently
selected task population or suitability as a universal model.

| Arm | Embedding | Residual branches | Attention coefficient |
|---|---|---|---|
| Existing LR×2 control | Original | Original | Original fixed gain |
| Embedding RMS | Add RMS | Original | Original fixed gain |
| Scaled Peri-RMS | Original | Output RMS, then `1/sqrt(2L)` | Original fixed gain |
| SSMax | Original | Original | `log(visible keys)` times original fixed gain |

Each new arm completes an 18-cell screen against the old LR×2 control; all pass
and at least three of six paired task medians must improve to advance. Every
promoted arm then completes all 75 short cells. Compare every outcome with the
old control, baseline, inverse-width incumbent and applicable combined-stack
controls. The earlier unscaled Peri-RMS negative remains recorded.

The global short-suite incumbent remains inverse-width at 16.2660501255
geometric-mean epochs unless a new complete candidate improves it. Only a newly
frozen shared winner gets one reused long diagnostic. An improvement over the old
control alone does not silently change the selection criterion. No long outcomes
choose model settings, rates, promotion or the short winner.

The guarantees differ across these models. Embedding RMS bounds the initial
state only; scaled Peri-RMS without it bounds growth relative to the original
embedding norm. SSMax retains the original gain, with its corresponding score
bound. None of these properties proves successful optimization or explains the
previous long failures by itself.

The implementation reuses the frozen native training loop, execution journal and
individual model adapters. Validation passed 33 checks, including all 35 task
shapes, independent full forward and parameter gradients, causal caches, three
real CUDA cases and three CPU checkpoint continuations. Lean checked four scoped
specializations of the existing bounds with only standard axioms. The initial
failed proof examples and CUDA comparison remain in `results/`.

The initial CUDA comparison combined compiler rounding with graph capture. The
unchanged scaled-Peri fixture differed by at most 0.0215% per-tensor relative L2
over 13 updates between compiled and eager evaluation, and 0.0022% between
captured and uncaptured compiled evaluation. The final checks retain the original
elementwise capture tolerances and separately check compiler rounding at the
existing numerical-stress relative L2 threshold of 0.001. This is a local update
check, not a whole-model numerical stress pass or proof of equivalent training
trajectories. No scientific model code changed in response to this diagnostic.

Status: validated implementation; source freeze precedes benchmark training.
The incomplete Muon study and proposed numerical repair remain deferred.
