# Known-solution residual attention: saturation and conditioning

Does depth attention become stuck despite an explicit solution? The frozen
21-case probe at386ae01 trains only a query, using Schedule-Free AdamW on
Python3.15.2/Torch2.15.0+cu132 and CUDA max-autotune. Its independent teacher
is a soft mixture of eight width16 source vectors. One source has a positive
marker; payloads and its position vary. The query4e0 is an explicit solution.
Every case runs2000 updates; none is selected or stopped early.

| Scoring | Initial query RMS | Successes / cases | Final heldout MSE range | Routing accuracy |
| --- | ---: | ---: | ---: | ---: |
| Key-only, unit key gain | 0 | 3/3 | 1.96e-12--3.39e-12 | 100% |
| Key-only, wrong direction | 1 | 0/3 | 4.219--4.226 | 0% |
| Key-only, wrong direction | 1e6 | 0/3 | 4.401--4.420 | 0% |
| Q/K, parameter-free RMSNorm | 0 | 3/3 | 2.16e-9--2.52e-9 | 100% |
| Q/K, parameter-free RMSNorm | 1e-8 | 3/3 | 2.16e-9--2.52e-9 | 100% |
| Q/K, wrong direction | 1 | 3/3 | 6.73e-6--8.69e-6 | 100% |
| Q/K, wrong direction | 1e6 | 0/3 | 4.216--4.226 | 0% |

Success requires final heldout MSE<=.001 and routing accuracy>=99%.
Seeds11/17/23 vary data and minibatches; the wrong initial direction is
fixed, so these are not three independent random query initializations.
The scoring comparison changes both query normalization and the sqrt(width)
score divisor. It does not isolate their individual effects.

At RMS1e6, key-only scores saturate and the training-query gradient is
exactly zero in FP32. Q/K scores remain below4, but their query-gradient norm
is only7.65e-9--7.94e-9; the raw query stays near RMS1e6. A bounded score does
not guarantee practical trainability from every parameter scale. With unit
RMS, Q/K escapes the wrong direction in all three cases. The primitive uses
unit-RMS initialization. This is a tested default, not a convergence proof.
The successful key-only zero-initialization controls are retained: this task
does not establish that Q/K normalization is universally necessary.

All21 final checkpoints replay on CPU with exactly matching routing counts
and success classifications. Source/input hashes, optimizer step and exact
sampler state are checked. The portable `results.json` retains every learning
curve, final result, checkpoint hash and replay discrepancy. Mixture metrics
use averaged weights; gradients use training queries. The extra evaluation
roundtrip can introduce FP32 rounding. Raw checkpoints remain under
`runs/residual-attention-check-20261101/` in the main repository.

Run the replay in the requested Python3.15/Torch2.15 environment:

```sh
python -m diff_gpt.interfaces.grokking.residual_attention_replay \
  --input runs/residual-attention-check-20261101 \
  --output research/residual_attention/results.json
```

The companion Lean commit1a9b522e proves unbounded original-query/gain scores,
normalized score and sensitivity bounds, convex aggregation, and depth-weight
caps. It assumes no optimizer convergence. Five independent equation/gradient
checks and three task/saturation tests pass; Ruff passes for the new modules.
This is a query-only mechanism test. Full-transformer residual replacement
still needs the separate controlled cohort; the ordinary GPT recurrence and
its div97 regression are unchanged.
