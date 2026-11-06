# Residual weight decay after the generalization diagnosis

The [frozen plan](plan.json) changes only native AdamW weight decay from 0.01
to 0.1 or 1.0. It uses both existing activations: shifted SwiGLU and the learned
even coefficient initialized at zero. The previous [96-cell diagnosis](../stress_audits/results/residual-generalization.json)
found that all runs fit the training threshold while only 34 passed held-out
tests. This motivates testing regularization; it does not prove the cause.

The study reuses all **48 parent controls** and runs **96 new cells**. Each of
the six activation/decay configurations spans both teachers, depths 4/16,
strengths 0.25/1, and seeds 0/1/2. Data, matrices, RNG draws, learning rate 0.01,
batch order, native optimizer settings, 512 updates and the 0.05 relative-MSE
threshold remain fixed. All parameters receive the same declared decay,
including residual gates and learned even coefficients.

The runner calls the existing `train_parity_cell` and model constructors
directly. It requires committed source and passing validation before training,
archives inherited sources and data, rejects duplicate live dispatchers, and
requires the exact 144-cell grid before selection. The earlier coefficient
0.5/1 studies remain preserved as separate negative evidence.

The four checks cover unchanged initialization/RNG/capacity, nontrivial active
residual outputs and gradients, retained controls, missing/duplicate cells,
and coverage-first selection without a successful-only mean. The final audit
replays every new checkpoint on both complete splits and verifies histories,
first crossings, optimizer state/settings, sampler state and source hashes.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/pytest -q tests/diff_gpt/application/test_transformer_stress_decay.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m research.transformer_stress_decay.run
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m research.stress_audits.decay
```

The completed [144-cell comparison](results/summary.json) selects **learned-even
activation with weight decay 0.1** under the declared pooled coverage rule.
It improves from 10/24 to **15/24**, recovering five cells without losing a
previous passing cell. Four common passes cross earlier and six tie. The odd
teacher now passes all 12 cells; the mixed teacher still fails nine. This is a
partial repair, not a universally successful residual block.

| Activation | Decay | Odd teacher | Mixed teacher | All cells | Worst final held-out relative MSE |
|---|---:|---:|---:|---:|---:|
| Shifted | 0.01, reused | 6/12 | 3/12 | 9/24 | 0.155336 |
| Shifted | 0.1 | 6/12 | 3/12 | 9/24 | 0.137495 |
| Shifted | 1 | 6/12 | 0/12 | 6/24 | 0.158372 |
| Learned-even, initial coefficient 0 | 0.01, reused | 10/12 | 0/12 | 10/24 | 0.163438 |
| Learned-even, initial coefficient 0 | **0.1** | **12/12** | **3/12** | **15/24** | **0.141903** |
| Learned-even, initial coefficient 0 | 1 | 6/12 | 0/12 | 6/24 | 0.144597 |

No configuration passes all 24 cells, so no pooled geometric mean of first
crossings is reported. The complete odd-teacher subgroup of the selected arm
has geometric mean 184.732 updates. The nine remaining mixed-teacher misses
include every depth-4 cell and all three depth-16, strength-1 cells. Their final
held-out errors range from 0.057161 to 0.141903; all run the original 512 updates.
Every selected-arm final training error is below 0.05 (maximum 0.024165), so a
substantial generalization gap remains.

Preserve the negative findings: decay 1 loses three previous mixed passes for
shifted activation and four previous odd passes for learned-even activation.
The latter also slows three of its six common passes. Shifted decay 0.1 improves
six common crossing times and ties three, with no coverage gain. Every paired
outcome is in [paired.json](results/paired.json); all 144 rows are in
[cells.csv](results/cells.csv).

The [audit](results/audit.json) reproduced full train/held-out final scores from
all 96 new checkpoints, verified 50 archived source/control/validation files,
and checked initialization, capacity, histories, optimizer and sampler state.
All 96 runs are finite: 36 passes and 60 budget misses. The source and validation
were committed as `8d83f9e` before training. Raw evidence remains under
`runs/transformer-stress/decay-v1/`; no old result was overwritten.

The unchanged forward equations retain their existing mathematical bounds.
Weight decay alone does not prove a smaller function Jacobian, generalization
or convergence. These adaptively reused synthetic tasks are independent of the
inverse-width attention experiment and do not change any completed early99 result.
