# Residual stress learning-rate follow-up

The [declared plan](plan.json) changes only peak LR in the selected shifted
SwiGLU / zero-initialized post-RMS scalar gate / `1/sqrt(D)` candidate.
The 12 cells at LR 0.003 are reused; 24 new cells test 0.01 and 0.03. Data,
seeds, initialization, optimizer moments/decay, warmup, 512 updates and the
5% relative-MSE threshold remain fixed. This is an adaptive synthetic
diagnostic, not fresh held-out evidence or a full-GPT improvement claim.

| Peak LR | Passed / 12 | Worst final relative MSE | Median final relative MSE |
|---|---:|---:|---:|
| 0.003 | 5 | 0.07459 | 0.06050 |
| **0.01** | **6** | 0.09002 | 0.06514 |
| 0.03 | 3 | 0.09392 | 0.07362 |

LR 0.01 wins the declared coverage-first rule. All six 16-branch cases pass,
but all six 4-branch cases miss. Higher rates decrease training error in those
shallow cases while worsening held-out error. This does not solve the full
test, and no geometric mean of successful-only crossings is reported.

[Complete results](results/summary.json), [all-cell CSV](results/cells.csv),
and the [24-cell audit](results/audit.json) preserve every result. The audit
reproduces final full-split metrics and verifies source/data hashes,
initialization, all optimizer steps, sampler states and reused comparison rows.
Raw histories and checkpoints remain in `runs/transformer-stress/rates-v1/`.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m research.transformer_stress_rates.run
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python research/stress_audits/followups.py rates
```

The next [capacity study](../transformer_stress_width/README.md) preserves
these rate results and tests a separate change.
