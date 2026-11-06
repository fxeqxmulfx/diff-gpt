# Bounded branch projection

The [declared plan](plan.json), committed as `3c44c2c`, tests one change to the
best completed residual learner (17/24 passes): replace branch-output RMS by
`P(z) = z / max(1, RMS(z))`. Small outputs retain their magnitude; large outputs
are bounded at RMS one. Input RMS, the learned-even activation, hidden/output
biases, residual gates and scaling, all parameters and the learning protocol
remain fixed. This changes the model equation and is an untested learning
hypothesis. It is independent of the running inverse-width attention diagnostic.

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

Training must use the committed source and validation records. The plan requires
24 new learning cells, all 24 reused controls, the complete original budget and
paired reporting. The separate checkpoint auditor replays both complete splits.
No learning improvement is claimed before those results. The proved branch
bound does not control the learned FFN, input normalization, free residual
gates, full network or generalization.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/pytest -q tests/diff_gpt/application/test_transformer_stress_projection.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m research.transformer_stress_projection.stress
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m research.transformer_stress_projection.run
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m research.stress_audits.projection
```
