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

Raw evidence is stored under `runs/transformer-stress/decay-v1/`. Results are
pending; no repair is established yet. The unchanged forward equations retain
their existing mathematical bounds. Weight decay alone does not prove a smaller
function Jacobian, generalization or convergence. These adaptively reused
synthetic tasks are independent of the inverse-width attention experiment and
do not change any completed early99 result.
