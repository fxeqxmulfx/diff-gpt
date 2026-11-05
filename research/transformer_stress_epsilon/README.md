# Branch-output RMS epsilon follow-up

The [plan](plan.json) changes only output RMS epsilon in the shifted SwiGLU /
zero-initialized scalar-gate model at LR 0.01 and original hidden width 43.
Input normalization remains unchanged. All 36 new cells retain the original
data, seeds, matrices, optimizer, 512 updates and 5% relative-MSE target;
the 12 original-epsilon controls are reused.

| Branch epsilon | Passed / 12 | Worst final relative MSE | Median final relative MSE |
|---|---:|---:|---:|
| **FP32 epsilon, 1.19209e-7** | **6** | **0.09002** | **0.06514** |
| 0.001 | 6 | 0.09026 | 0.06498 |
| 0.1 | 3 | 0.09642 | 0.06735 |
| 1.0 | 3 | 0.09863 | 0.06638 |

The original epsilon remains the winner under the declared coverage/worst-error
rule. Larger epsilons do not fix the six shallow failures and lose some deeper
passes. All runs stayed finite. No successful-only first-crossing mean is used.

For every positive epsilon, the existing RMS norm bound remains valid, and
the normalization-only Lipschitz bound `2/sqrt(epsilon)` becomes smaller as
epsilon increases. The radial derivative also retains more signal. Those
mathematical improvements **did not produce a learning improvement here**.
The result does not establish the best epsilon at other rates, on other data,
or in the complete GPT.

[Complete results](results/summary.json), [all 48 comparison rows](results/cells.csv)
and the [audit](results/audit.json) preserve every case. Three tests verify
parent equivalence with active gates, actual radial derivatives and the
unchanged training-loop AST. The audit reproduces full-split final metrics
and checks sources, data, optimizer/sampler states and original control rows.
Raw checkpoints remain in `runs/transformer-stress/epsilon-v1/`.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m research.transformer_stress_epsilon.run
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python research/stress_audits/epsilon.py
```

This closes the declared epsilon comparison without promoting an unsuccessful
fix. Numerical repairs have a separate successful 65-case result; the residual
learning diagnostic still has unresolved shallow cases.
