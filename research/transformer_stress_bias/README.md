# Zero-initialized FFN biases in the residual stress test

The [frozen plan](plan.json), committed as `0810591`, tests whether biases improve
the remaining generalization failures after weight decay 0.1. The selected
learned-even control passes 15/24; shifted activation passes 9/24. A complete
reflection diagnosis finds mixed-teacher errors in both odd and even components.
Biases may help represent offsets and shifted features; this is a hypothesis.

Two new arms add either an output projection bias alone or both hidden and output
biases. All new vectors initialize at zero around the same existing bias-free
linear modules. Original matrices, even coefficients, residual gates, parameter
ordering relative to one another and RNG draws are preserved. Depth D adds
16D or 102D scalar parameters, respectively. This is not a parameter-matched
capacity comparison. Every bias receives the same native AdamW decay 0.1 as the
original parameters.

The study reuses all 48 decay-0.1 controls and runs 96 new cells. Every one of six
activation/bias configurations spans both teachers, depths 4/16, strengths
0.25/1, and seeds 0/1/2. The fixed data, batch order, learning rate 0.01, optimizer
settings, 512-update budget, cadence and 0.05 relative-MSE threshold remain intact.
All cells finish their budget even after crossing, except numerical failures.
Only the complete 144-cell grid can select a configuration; all failures remain.

Five pretraining checks cover unchanged initialization/function/common gradients
with active residual gates, exact added capacity, nonzero-bias equations and all
gradients against an independent reference, identical training-loop syntax apart
from construction and capacity metadata, retained controls and complete-grid
selection without excluding misses. The final audit replays all new checkpoints
on both full splits and checks histories, optimizer state/settings, sampler state
and the actual added capacity.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/pytest -q tests/diff_gpt/application/test_transformer_stress_bias.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m research.transformer_stress_bias.run
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m research.stress_audits.bias
```

Training results are pending. Raw records belong under
`runs/transformer-stress/bias-v1/`. Generic finite-epsilon RMS and post-RMS
residual-radius bounds still apply to arbitrary biased branch outputs. Trained
biases can break input oddness, so bias-free oddness statements do not transfer.
No whole-network, generalization or convergence theorem is inferred. This
adaptive synthetic study changes neither the full GPT nor the independently
running inverse-width benchmark.
