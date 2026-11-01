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

The six CPU unit checks cover finite-epsilon equations and all gradients,
query/base/average preservation, transformed gradient history, unchanged small
queries, the actual huge-query gradient, and exact saved-state continuation.
The compiled no-update comparison uses a separate FP64 execution and retains
all three repetitions. A seventh, reproducible CUDA regression separately checks
physical-query FP64 equations and both query/value gradients without calling
the model or conditioner for the reference. The companion Lean modules `QueryCoordinates` and
`QueryConditioning` formalize the real identities and immediate coordinate bound.
There is no universal convergence or floating-point error theorem.

Run in the requested Python 3.15/Torch 2.15 environment:

```sh
python -m diff_gpt.interfaces.grokking.residual_conditioning \
  --output runs/residual-conditioning-check-20261101
```

Use the conditioned component with its state adapter:

```python
from diff_gpt.application.residual_conditioning import DepthQueryConditioner
from diff_gpt.domain.model.conditioned_attention_residual import ConditionedQKAttentionResidual
from diff_gpt.domain.optimizer.schedule_free import ScheduleFreeAdamW

model = ConditionedQKAttentionResidual(width)
optimizer = ScheduleFreeAdamW(model.parameters(), lr=0.01)
conditioner = DepthQueryConditioner(model, optimizer, compiled=True)

optimizer.train()
optimizer.zero_grad(set_to_none=True)
conditioner()
loss = objective(model(source_states))
loss.backward()
optimizer.step()
```

Save both `model.state_dict()` and `optimizer.state_dict()`: the model contains
nontrainable coordinate metadata, and the optimizer contains the matching history.
The adapter must run before each forward/backward, after clearing gradients. Its
ownership and mode checks reject a stale gradient or evaluation-mode state. The
verified numerical path uses FP32 (compiled CUDA) and FP64 references. This
component is not wired into the full GPT residual path by this synthetic repair.

The first attempt completed all six zero-query cases and failed during CUDA
graph capture when starting the first tiny-query legacy case. `attempt1.json`
and `attempt1.log` retain its results and verbatim exception; `retry.json`
declares a complete unchanged retry after the concurrent repository tests finish.
The concurrency is recorded, but its causal role in the failure is not established.

All30 cases completed in the unchanged retry, in105.05 seconds. The failed
attempt took27.62 seconds; both attempts together took132.67 seconds of execution.
Python3.15.2, Torch2.15.0+cu132, RTX3050 Laptop, max-autotune, Schedule-Free AdamW.
The retry ran after the repository test jobs ended. State-kernel CUDA graph
capture was skipped for mutated inputs; compilation itself remained enabled.
These tiny-query timings include compilation/evaluation and are not a full-model
throughput comparison.

| Query cases | Original success | Conditioned success | Original final heldout MSE | Conditioned final heldout MSE |
| --- | ---: | ---: | --- | --- |
| Zero and tiny, 2000 updates | 6/6 | 6/6 | 2.16e-9--2.52e-9 | 2.16e-9--2.52e-9 |
| Unit, 2000 updates | 3/3 | 3/3 | 6.73e-6--8.69e-6 | 1.81e-9--2.72e-9 |
| Huge, fresh 2000 updates | 0/3 | 3/3 | 4.216--4.226 | 1.81e-9--2.72e-9 |
| Huge saved failures, +10000 updates | 0/3 | 3/3 | 4.216--4.226 | 8.52e-15--9.24e-15 |

All six corrected huge-query cases reach100% heldout routing; every matched
original huge-query case remains at0%, including the longer continuations.
The initial coordinate gradient grows from7.65e-9--7.94e-9 to.00765--.00794
without changing the represented function. This factor of about1e6 comes from
the coordinate chain rule; it is not an increase in the physical-query derivative.
Unit-scale runs also trigger two later reconditionings, fresh huge runs three,
and resumed huge runs four. Zero and tiny runs trigger none.

`results.json` retains every curve and the independent FP64 replay of all initial,
conditioned-initial and final states. Source/checkpoint/data hashes, exact resumed
initial model/optimizer/sampler states, reconstructed averaging clocks and all
subsequent sampler draws pass. Routing counts and success classifications agree
exactly. The largest scalar metric error is6.28e-7, below the declared replay
tolerances. Gradient-norm errors are retained without silently thresholding near
zero; the largest absolute error,1.67e-4, occurs at gradient norm1463.36 and has
relative error1.14e-7. No checkpoint selection or model retuning followed outcomes.
The original21-case failure report and the failed first attempt remain intact.

Validation:167 affected GPT tests pass, with one existing skip, including the
compiled physical-equation/all-gradient regression. The companion15 Lean theorems
pass build/audit/index/forbidden and explicit axiom checks. The full lab suite
passes452 tests with three skips. Its optimizer stage retains two pre-existing
ScheduleFreeMuon CUDA allocation failures, reproduced on unchanged source;
the CPU optimizer suite passes104 tests with14 skips. Schedule-Free AdamW passes.

This repairs the demonstrated query-conditioning failure on the synthetic task.
It does not show better full-transformer learning than ordinary residual addition,
or prove convergence for arbitrary data, width, depth or floating-point range.
That matched full-model experiment remains task #145 in the companion project.

Replay all recorded outcomes without training:

```sh
python -m diff_gpt.interfaces.grokking.residual_conditioning_replay \
  --input runs/residual-conditioning-check-20261101-retry1 \
  --output /tmp/residual-conditioning-replay.json
```


Requested author self-review reran the affected167 tests (one existing skip)
and all30 checkpoint checks at `24b3b49`. The complete replay JSON reproduces
byte-for-byte. The companion Lean acceptance checks and15 axiom lists pass again
at `9ec890da`. The same assistant performed this review; it is not independent
review. Task#150 is complete, with full-model comparison remaining task#145.
