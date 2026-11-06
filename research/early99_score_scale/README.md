# SSMax score strength with Q/K RMS retained

This round tests **one-half and one-quarter of the current SSMax score gain**
on the best completed Q/K-normalized short-suite arm. It retains scaled
Peri-RMS, embedding RMS, centered XSA, SwiGLU, ordinary fixed RoPE, tied weights,
parameter-free normalization and LR ×2. No tensors, parameter counts, random
draws, data, budgets or optimizer settings change.

**Both full short suites are complete and audited. The frozen shared-model
choice is quarter-scale SSMax, with Q/K RMS retained:** **75/75 passes** and
**17.3257 geometric-mean epochs**, **49.4% fewer than LR ×2** and **28.8% fewer
than the Q/K parent**. It is also 17.4% earlier than the previous no-Q/K ablation
on the complete aggregate. Its worst slowdown against LR ×2 is **17.5×**;
every regression is retained. The single reused long diagnostic is running;
short selection does not establish a universal replacement.

The [plan](plan.json) was declared after round two's results were committed as
`1138c62`, and before new benchmark training. Short results motivate the test:
SSMax/QK passes 75/75 at 24.3271 geometric-mean epochs; the no-QK ablation passes
75/75 at 20.9798. Removing Q/K RMS changes both magnitude sensitivity and score
scale. That comparison does not establish which caused the speed difference.
This round tests whether softer attention helps while retaining normalization.
The completed long diagnostic does not select or tune these candidates.

The equation is `m*log(N)*g_h*dot(rotated_RMS(q), rotated_RMS(k))` for visible keys.
Thus the effective SSMax coefficient is **s=0.5 or s=0.25**, replacing the
parent's fixed s=1. Q/K normalization occurs before the multiplier. Both dense
and cached attention use the same coefficient and full causal key count;
the N=1 row remains zero before masking.

The real score bound becomes `m*log(N)*abs(g_h)*head_dim`, independent of Q/K
projection-weight norms, using positive epsilon and the checked
[normalized-score contract](../early99/guarantees/NormalizedScores.lean).
These two candidates tighten that bound by factors of two and four. They
preserve the existing residual-state and normalization bounds. Neither this
bound nor a finite-precision stress pass guarantees learning, a useful margin,
whole-network conditioning or generalization to larger tasks.

There is also a concentration tradeoff. The existing checked
[SSMax margin theorem](../early99/guarantees/ScalableSoftmax.lean) gives
`p(target) >= 1 / (1 + (N-1)*N^(-m*delta))` if its unscaled score exceeds every
competitor by at least `delta`. Reducing m weakens this lower bound for the
same gap. For a fixed positive gap, `m*delta > 1` is sufficient for this bound
to approach one as context grows: half-scale requires `delta > 2`, quarter-scale
requires `delta > 4`. These are sufficient conditions for the bound, not
necessary conditions for learning or an assumption that training supplies
the gap. Smaller bounded scores alone do not establish a better universal model.

The screen keeps the six existing short tasks and all three seeds. Advancement
requires 18/18 passes and at least three improved task median epoch ratios
**versus the Q/K parent**. This stronger parent comparison is declared before
training. Every promoted arm must complete all 75 short cells. The shared-model
choice can improve only with full coverage and a lower complete geometric mean
than the parent; baseline, LR ×2 and no-QK comparisons remain visible separately.
Every miss is retained and no filtered geometric mean is allowed. A new shared
winner, if selected, receives one reused long diagnostic after freezing choice.

Implementation and validation:

- [factories.py](factories.py) changes only the fixed gain and records it in the
  configuration. It adds no learned parameters and keeps every original matrix.
- [run.py](run.py) calls the unchanged original trainer with explicit factories,
  and extends its source archive to include this package. It rejects changed
  configuration, execution settings or archived sources.
- [stress.py](stress.py) compares 90 normalized attention rows with an independent
  FP64 forward/backward reference: widths 8/32/128, visible lengths through 4096,
  Q/K scales 1e-9/1/1e9, and both coefficients. This isolates score arithmetic;
  it does not replace the earlier full-stack extreme-magnitude tests or conceal
  their unresolved failures.
- The [implementation tests](../../tests/diff_gpt/application/test_early99_score_scale.py)
  check independent equations and all attention gradients, singleton/clipped XSA,
  unchanged initialization/RNG/capacity, causality and cache modes, source/archive
  guards, and CUDA graph updates against uncaptured native AdamW.

Reproduce pretraining checks from the repository root:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. .venv/bin/pytest -q tests/diff_gpt/application/test_early99_score_scale.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. .venv/bin/python -m research.early99_score_scale.stress
```

The frozen configurations are under [configs](configs). Use the isolated runner
with `--configuration PATH --write-plan`, then `--run --tier short --case NAME`
for each declared screen task. `--run --tier short` completes all 75 cells only
for a promoted arm; it reuses already completed screen cells. Preserve the
source snapshot before training and use a new output directory for any change.

The scientific source is frozen at **`dfecdc9`**. All **9 CPU/CUDA checks** passed,
and all **90 score probes** passed the predeclared bounds and accuracy limits.
Maximum probability error was `6.4692e-7`; maximum relative gradient error was
`1.5498e-5`. The [validation record](results/validation.json) and
[complete stress rows](results/score-stress.json) preserve the evidence and scope.
Both 18-cell screens are complete and audited; both meet the declared gate.
Both full 75-cell suites passed. The complete shared-model selection is frozen
in [shared-selection.json](results/shared-selection.json), with independently
reconstructed evidence in [selection-integrity.json](results/selection-integrity.json).

Run both screens or regenerate their four-control reports with
`python -m research.early99_score_scale_reporting.run_screens --run` or
`--report`, using the environment shown above. Do not launch a duplicate active
driver. The [checkpoint auditor](../early99_score_scale_reporting/audit.py) accepts
`--name NAME --output PATH` for a screen, `--full` for 75 short cells, or
`--full --long` for 105 cells after a selected winner's diagnostic.

## Half-scale screen

The half-scale arm passes **18/18** and improves **4/6 paired task medians**
versus the Q/K parent, meeting the declared promotion gate. It has **13 earlier,
4 tied and 1 later** cells. The one slowdown is linear transport, with a worst
paired ratio of 1.5. No screen geometric mean is reported.

| Task | Parent median epochs | Half-scale median epochs | Median paired ratio | Worst paired ratio |
|---|---:|---:|---:|---:|
| Boolean AND | 3 | 2 | 1.0000 | 1.0000 |
| Division p193 | 35 | 27 | 0.8571 | 1.0000 |
| Division p97 | 28 | 16 | 0.5357 | 0.5714 |
| Key-value binding | 16 | 11 | 0.7500 | 0.8462 |
| Square sum | 311 | 218 | 0.7554 | 0.7878 |
| Linear transport | 2 | 3 | 1.0000 | 1.5000 |

The gate uses the median of **paired seed ratios**, which need not equal the
ratio of the two separate medians. All rows and the baseline/LR ×2/no-QK
comparisons are retained beside the
[parent screen](results/screen-peri-rms-scaled-ssmax-gain-half-lr2-vs-parent.json).
The [checkpoint audit](results/audit-peri-rms-scaled-ssmax-gain-half-lr2-screen.json)
verifies all 18 cells and 31 archived sources, with no resumptions, execution
errors or capacity changes.

## Quarter-scale screen

Quarter-scale also passes **18/18**, improving **3/6 paired task medians** and
meeting the declared gate. It has **9 earlier, 3 tied and 6 later** cells versus
the Q/K parent. Its largest regression is key-value binding at **6.2308×** the
parent's epochs. A tighter score bound does not remove the learning tradeoff.

| Task | Parent median epochs | Quarter-scale median epochs | Median paired ratio | Worst paired ratio |
|---|---:|---:|---:|---:|
| Boolean AND | 3 | 3 | 1.0000 | 1.3333 |
| Division p193 | 35 | 31 | 0.9714 | 1.1111 |
| Division p97 | 28 | 16 | 0.5357 | 0.5714 |
| Key-value binding | 16 | 70 | 4.3750 | 6.2308 |
| Square sum | 311 | 141 | 0.4534 | 0.5144 |
| Linear transport | 2 | 5 | 2.0000 | 2.5000 |

The [complete parent screen](results/screen-peri-rms-scaled-ssmax-gain-quarter-lr2-vs-parent.json)
and adjacent baseline/LR ×2/no-QK reports retain every row and regression.
Its [checkpoint audit](results/audit-peri-rms-scaled-ssmax-gain-quarter-lr2-screen.json)
also verifies all 18 cells and 31 archived sources, with no errors, resumptions
or capacity changes.

Both arms completed **all 75 short cells** with the same frozen sources.
The [full-suite driver](../early99_score_scale_reporting/run_full.py) reconstructs
both promotion decisions, runs each full suite, checks its checkpoints and all
four paired comparisons, then freezes a shared-model choice. It performs no
long training before that choice. Reproduce with
`python -m research.early99_score_scale_reporting.run_full --run` in the environment
above; do not launch a duplicate active queue.

## Half-scale complete short result

Half-scale passes **75/75**, with geometric mean **18.5238 epochs**: **23.9%
fewer than the Q/K parent**, **45.9% fewer than LR ×2**, **67.6% fewer than
baseline**, and **11.7% fewer than the previous no-Q/K ablation**. It retains
Q/K RMS and adds no parameters. Its summed trainer time, including setup and
scoring, is 1110.40 seconds; wall time does not select the candidate.

Against the parent, 53 cells cross earlier, 12 tie and 10 cross later. The
largest regression is division p193, seed 1, **9 → 64 epochs (7.1111×)**.
The [parent comparison](results/peri-rms-scaled-ssmax-gain-half-lr2-vs-parent-short.json)
and adjacent baseline/LR ×2/no-Q/K JSON/CSV reports retain every cell and each
task's median and worst ratio. Its [75-cell checkpoint audit](results/audit-peri-rms-scaled-ssmax-gain-half-lr2-full-short.json)
verifies 31 archived sources, original capacity/initialization, native optimizer
state, first crossings and all histories, with no errors or resumptions. The
four comparisons were independently reconstructed from raw histories.

The [score guarantee record](results/score-guarantees.json) specializes existing
checked Lean theorems to each new coefficient and links the improved RMS
Lipschitz proof. It explicitly records the weaker conditional concentration
bound. These algebraic specializations are not a new Lean compilation or a
formally verified translation of the PyTorch implementation.

## Complete comparison and frozen choice

All five configurations below pass the same 75 short cells. The newly selected
quarter-scale coefficient changes only fixed attention-score strength; the
full shared architecture retains scaled Peri-RMS, embedding RMS, Q/K RMS,
centered XSA, SwiGLU, tied weights and ordinary fixed RoPE. It does not combine
the prior LeRoPE or independent numerical/activation repairs with this trial.

| Configuration | GM first-99 epochs | Ratio to LR ×2 | Ratio to Q/K parent |
|---|---:|---:|---:|
| LR ×2 control | 34.2499 | 1.0000 | 1.4079 |
| SSMax/QK parent, s=1 | 24.3271 | 0.7103 | 1.0000 |
| Previous no-Q/K ablation | 20.9798 | 0.6125 | 0.8624 |
| Half-scale, Q/K retained | 18.5238 | 0.5408 | 0.7614 |
| **Quarter-scale, Q/K retained** | **17.3257** | **0.5059** | **0.7122** |

The original baseline passes 75/75 at 57.1653 epochs; quarter-scale uses 69.7%
fewer aggregate epochs. Original parameter counts remain 100,736–1,183,488.
Quarter-scale's summed trainer time is 1133.38 seconds, including setup and
scoring. Runtime does not determine selection.

| Quarter-scale comparison | Earlier / tied / later cells | Worst paired slowdown |
|---|---:|---:|
| Q/K parent | 50 / 7 / 18 | 6.2308× |
| LR ×2 | 49 / 9 / 17 | 17.5× |
| Original baseline | 55 / 6 / 14 | 14× |
| Previous no-Q/K ablation | 55 / 8 / 12 | 4.3333× |

The largest regression versus LR ×2 is key-value binding, seed 2, **4 → 70
epochs**; versus the Q/K parent it is seed 0 of that task, **13 → 81 epochs**.
The [complete summary](results/full-short-summary.json) and adjacent four-control
JSON/CSV reports preserve every seed, per-task median/worst, runtime and failure
field. The [quarter-scale checkpoint audit](results/audit-peri-rms-scaled-ssmax-gain-quarter-lr2-full-short.json)
verifies all 75 cells and 31 archived sources with no execution errors or
resumptions. Together with half-scale, all eight complete paired comparisons
were independently reconstructed before choice was frozen. The old round-two
selection is unchanged; the new shared-model choice retains Q/K RMS throughout.

## Audit and subsequent long diagnostic

The separate [report verifier](../early99_score_scale_reporting/verify_results.py)
reconstructs all screens, complete comparisons, selection and coverage from raw
histories. It checks source/configuration/proof hashes and every declared
numerical stress row, error flag and aggregate. After both full suites finish,
run `python -m research.early99_score_scale_reporting.verify_results --freeze`
to create `results/selection-integrity.json`, then commit the audited decision.

The [long launcher](../early99_score_scale_reporting/run_long.py) requires that
complete frozen decision and its commit before running a newly selected shared
winner once. It rejects duplicate live launchers, uses the unchanged scientific
runner and original budgets, and reconstructs all long comparisons afterward.
Use `python -m research.early99_score_scale_reporting.run_long --run` only after
the full short decision is committed. It compares baseline, LR ×2 and the
previous no-Q/K ablation, whose long suites already exist. The Q/K parent has
no long suite, so no paired long comparison against that parent is claimed.
All misses remain visible; incomplete coverage prevents a complete geometric
mean. Long outcomes cannot alter the choice or tune subsequent candidates.

The reporting guards are tested separately in
[test_early99_score_reporting.py](../../tests/diff_gpt/application/test_early99_score_reporting.py),
including missing/changed selection, undeclared configurations, pending cells,
censored rows, misleading stress summaries and overwriting completed evidence.

The short selection and evidence were committed as **`8256d95`** before the
[long launch](results/long-launch.json). Its selected configuration and source
hashes remain fixed. The supplemental
[attention sensitivity proofs](../normalization_guarantees/README.md#normalized-attention-sensitivity)
now also bound changes in one probability row under projected query/key
perturbations with fixed rotations. Seven additional checked statements improve
the earlier softmax constant from four to one and extend the query-only result
to uniformly bounded changes in all keys. They change no model behavior.
It is separate from the already frozen selection manifest and changes no model
or experiment. Long outcomes will be reported once the complete diagnostic and
checkpoint audits finish.

A supplemental [selectivity bound](../normalization_guarantees/README.md#attention-selectivity-ceiling)
also limits the probability any single key can receive from bounded normalized
scores. It makes the sensitivity/concentration tradeoff explicit, including a
checked width-four limit on concentrated quarter-scale
attention. Width four is outside this benchmark's head widths; the result
changes no experiment and is not a whole-model learning impossibility.
