# Bounded branch projection: implementation in progress

The [declared plan](plan.json), committed as `3c44c2c`, tests one change to the
best completed residual learner (17/24 passes): replace branch-output RMS by
`P(z) = z / max(1, RMS(z))`. Small outputs retain their magnitude; large outputs
are bounded at RMS one. Input RMS, the learned-even activation, hidden/output
biases, residual gates and scaling, all parameters and the learning protocol
remain fixed. This changes the model equation and is an untested learning
hypothesis. It is independent of the running inverse-width attention diagnostic.

The model and copied training loop are implemented. Python syntax and lint
checks pass. [Six Lean statements](Guarantees.lean) compile with zero warnings
and only `propext`, `Classical.choice` and `Quot.sound`. They establish identity
inside the ball, the radius bound, the projection variational inequality,
global Euclidean Lipschitz constant one, sensitivity after a fixed scalar and
the conditional one-step residual radius bound. The actual width-16 branch
uses Euclidean radius four. Projection is not classically differentiable on
the boundary; the implementation chooses the exterior derivative there.

**Training has not started.** The experiment dispatcher, independent numerical
and implementation tests, and checkpoint auditor still need completion before
the source can be frozen for training. The plan requires 80 numerical cases,
24 new learning cells, all 24 reused controls and complete paired reporting.
No learning improvement or floating-point validation is claimed yet. The
proved branch bound does not control the learned FFN, input normalization,
free residual gates, full network or generalization.
