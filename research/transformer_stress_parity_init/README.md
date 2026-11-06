# Initialization of the learned even path

Starting the [learned even coefficient](../transformer_stress_parity/README.md)
at zero helped the original odd teacher but regressed on the mixed teacher.
This [separately frozen plan](plan.json), source `0b83bef`, changes only its
initial value to 0.5 or 1. Both teachers, all seeds, matrices, parameter counts,
optimizer settings, 512 updates and the 0.05 relative-MSE threshold stay fixed.
All 48 earlier controls and 48 new cells are retained.

| Arm | Original odd passes / 12 | Mixed passes / 12 | Total / 24 | Worst final relative MSE |
|---|---:|---:|---:|---:|
| Shifted parent | 6 | 3 | 9 | 0.15534 |
| Learned coefficient, initial 0 | **10** | 0 | **10** | 0.16344 |
| Learned coefficient, initial 0.5 | 6 | 3 | 9 | 0.15871 |
| Learned coefficient, initial 1 | 3 | 3 | 6 | **0.14973** |

Neither new initialization fixes the tradeoff. Initial 0.5 has exactly the
parent's pass/miss set: it recovers the three mixed-teacher passes lost at zero
but gives up the four gains on the original teacher. Initial 1 additionally
loses all three depth-16, strength-1 original-teacher passes. All 48 new runs
remain finite and complete the full budget; they total 157.63 CPU seconds.

Initial zero remains the declared coverage winner at 10/24, with its three
regressions and worse worst-case error retained. No arm passes all 24, and no
successful-only geometric mean is reported. There is no new shared-model
recommendation. This adaptive synthetic comparison is not fresh transfer
evidence or a full-GPT improvement.

Four checks verify unchanged matrices, random draws and capacity, recovery of
the parent function at initial one with active residual gates, the unchanged
training-loop AST, and the incomplete-selection guard. The complete
[results](results/summary.json), [96 rows](results/cells.csv) and
[audit](results/audit.json) retain every condition. The audit reproduces all 48
new final full-split scores and checks initial states, optimizer/sampler states,
data, sources and the complete comparison.

The earlier [Lean activation identities](../transformer_stress_parity/ActivationParity.lean)
apply unchanged for every coefficient. They guarantee that coefficient one
recovers the parent function in real arithmetic; they do not guarantee the
same optimization trajectory once that coefficient is learnable.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m research.transformer_stress_parity_init.run
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python research/stress_audits/parity_init.py
```

Raw histories, checkpoints and copied scientific sources remain in
`runs/transformer-stress/parity-init-v1/`. The original early99 benchmark,
frozen short selection and active long diagnostic are unchanged.
