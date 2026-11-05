# Small retrieval test of Q/K scale robustness

The user requested a small synthetic test in which removing Q/K normalization
can break learning. This is a targeted attention-level stress experiment,
separate from the frozen 75-cell early99 benchmark. It does not include the
full GPT, Peri-RMS branches, SwiGLU, XSA or learned rotary positions.

An example has eight random 32-dimensional keys, each paired with a distinct
random one-hot value. The query repeats one key exactly; the target is that
key's value. Two learned 32×32 projections give **2,048 parameters**. SSMax
attention averages the values and a fixed readout produces classification
logits. Training and held-out examples are independent. A separate distance
oracle checks every label, and values are independently permuted per example.

The [predeclared plan](plan.json) compares parameter-free Q/K RMS on/off,
projection multipliers 1, 4, 16 and 64, and seeds 0, 1 and 2: **24 cells**.
All conditions share data, original weights, shuffle order, optimizer, learning
rate, observation cadence and budget. Each cell stops at the first exhaustive
99% held-out observation or at 1,024 updates (32 epochs). All ordinary-scale
controls and failed stresses are retained. No result-dependent retuning occurs.

The multiplier `c` applies to both projected queries and keys before optional
RMS. Without normalization, scores multiply by `c²`. With positive `c`,
finite-epsilon RMS obeys
`RMS_eps(c*x) = x / sqrt(mean(x²) + eps/c²)`; scale cancellation is approximate
when epsilon is nonzero. The inherited real-arithmetic norm bound gives
`abs(SSMax_score) <= sqrt(32)*log(8)` for the normalized variant, independent
of projection magnitudes. Large raw scores can make softmax nearly one-hot,
reducing gradients through the value mixture. This is the mechanism being
tested, not a claim of naturally occurring norm growth in the full GPT.

The preflight checks an independent label oracle, value permutations, the RMS
scaling equation and its gradients, and independently computed attention/readout
equations. CPU execution uses one thread so the current GPU research can continue.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m research.qk_scale_stress.run
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m research.qk_scale_stress.run --run
```

Raw tensors, source snapshots, histories and checkpoints are preserved in
`runs/qk-scale-stress/v1`. The runner rejects changed settings or sources in an
existing output. Results will report every seed's coverage, first-99 epochs,
final accuracy, attention entropy, score magnitude, saturation and gradient
diagnostics. Failure to reach 99% is distinct from nonfinite numerical failure.

## Result

All **24 declared cells completed and passed the checkpoint/source audit**.
Q/K RMS reached 99% in one epoch in all 12 conditions. Without Q/K RMS,
both ordinary-scale and scale-4 controls passed all seeds, while scales 16
and 64 missed the target in every seed at the 32-epoch budget.

| Q/K multiplier | With Q/K RMS: epochs for seeds 0/1/2 | Without Q/K RMS: epochs or final accuracy |
|---|---|---|
| 1 | 1 / 1 / 1 | 1 / 1 / 1 epochs |
| 4 | 1 / 1 / 1 | 3 / 2 / 3 epochs |
| 16 | 1 / 1 / 1 | Censored: 89.36% / 85.74% / 88.18% |
| 64 | 1 / 1 / 1 | Censored: 23.24% / 24.07% / 15.14% |

Normalized final accuracies are 99.90%, 99.85% and 99.80% for the three seeds
at every scale. Normalized endpoint maximum score magnitudes stay below
10.51. Without normalization, scale-16 endpoints reach score magnitudes
6,562–9,063; scale-64 endpoints reach 87,614–94,255. At scale 64, 305–310
updates per seed have exactly zero projection gradient, versus zero such
updates in every normalized run. Attention entropy is near zero in the
failed raw-score conditions. These observations support the intended saturation
mechanism; no NaN/Inf failure occurred.

This is a small, trained counterexample to robustness under prescribed Q/K
scale growth. It does not show that the same growth occurs spontaneously in
the full GPT. The ordinary controls are successful, and the selected scale
grid, learning rate and budget were frozen before training.

[Complete results](results/summary.json), [all cell metrics](results/cells.csv)
and [audit](results/audit.json) preserve all six budget misses. Scientific
source revision: `9c71eee`. The full transformer study continued on the GPU
while this one-thread CPU experiment ran.
