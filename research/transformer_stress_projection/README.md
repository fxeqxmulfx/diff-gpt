# Bounded branch projection

The [declared plan](plan.json), committed as `3c44c2c`, tests one change to the
best completed residual learner (17/24 passes): replace branch-output RMS by
`P(z) = z / max(1, RMS(z))`. Small outputs retain their magnitude; large outputs
are bounded at RMS one. Input RMS, the learned-even activation, hidden/output
biases, residual gates and scaling, all parameters and the learning protocol
remain fixed. This changes the model equation. The completed comparison below
does not improve learning coverage. It is independent of the inverse-width
attention diagnostic.

The model, dispatcher and copied training loop are implemented. All seven
independent implementation/protocol checks and lint pass. [Six Lean statements](Guarantees.lean) compile with zero warnings
and only `propext`, `Classical.choice` and `Quot.sound`. They establish identity
inside the ball, the radius bound, the projection variational inequality,
global Euclidean Lipschitz constant one, sensitivity after a fixed scalar and
the conditional one-step residual radius bound. The actual width-16 branch
uses Euclidean radius four. Projection is not classically differentiable on
the boundary; the implementation chooses the exterior derivative there.

The [80 numerical cases](results/numeric.json) all pass across widths
1/2/16/64/256, FP32/FP64 and scales zero through 1e30. Maximum absolute
forward error is 1.16690e-7 and gradient error 9.32924e-8 versus independent
NumPy equations and analytic derivatives, below the declared 1e-6 limits.
The maximum sampled distance ratio is one. These are isolated projection
checks, not complete-model numerical or learning guarantees.

A directed boundary test caught an initial mismatch: this PyTorch build's
`clamp_min` selected the interior derivative at exact equality. The initial
[failing output](results/validation-initial-failure.txt) is retained. An explicit
conditional now selects the declared exterior derivative, preserving the
forward function. All seven checks and the full numerical grid passed after
that repair. The proof does not require classical differentiability there.

Validated source was committed as `b0cbb60` before training. All 24 new cells
completed the original 512-update budget and remain finite. The complete
[48-cell comparison](results/summary.json) retains the RMS parent: projection
passes **16/24**, versus the parent's **17/24**.

| Branch output operation | Odd teacher | Mixed teacher | Total | Worst final held-out relative MSE |
|---|---:|---:|---:|---:|
| RMS, reused control | 11/12 | 6/12 | **17/24** | **0.130742** |
| Ball projection | 10/12 | 6/12 | 16/24 | 0.144459 |

Projection recovers the previously near-threshold odd depth-4, strength-0.25,
seed-1 failure, reaching the threshold at 352 updates. It introduces two misses:
odd depth-4, strength-1, seeds 0/1, ending at 0.054539/0.051469. Among the 15
common passes, six are earlier, one ties and eight are later. The largest
successful-pair slowdown is 96→224 updates (2.3333×), on odd depth-16,
strength-0.25, seeds 0/2. Every paired cell is in [paired.json](results/paired.json).

All six strong mixed-teacher cases still fail, with final held-out errors
0.127347–0.144459. All 24 final training errors are below 0.05, with maximum
0.020712. Preserving branch magnitude did not resolve the generalization gap
under this fixed protocol. The stronger projection sensitivity bound therefore
does not justify replacing the better measured RMS parent. Neither arm passes
24/24, so no pooled first-crossing geometric mean is reported.

The [audit](results/audit.json) replays full training and held-out scores for every
new checkpoint, verifies histories, crossings, exposure epochs, initial states,
RNG, optimizer and sampler states, and checks 71 archived source/evidence files.
The full grid, controls and pairs are reconstructed. Parameter counts stay at
8,672/34,688 for depths 4/16; recorded trainer time totals 91.38 seconds. Raw
checkpoints remain under `runs/transformer-stress/projection-v1/`. All eight new
budget misses are preserved. The proved branch bound does not control the
learned FFN, input normalization, free residual gates, full network or
generalization. No full-GPT change is made.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/pytest -q tests/diff_gpt/application/test_transformer_stress_projection.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m research.transformer_stress_projection.stress
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m research.transformer_stress_projection.run
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m research.stress_audits.projection
```
