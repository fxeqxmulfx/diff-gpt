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

The complete [144-cell comparison](results/summary.json) selects learned-even
activation with hidden and output biases: **17/24**, versus 15/24 without biases.
It recovers all three shallow, strength-0.25 mixed-teacher cases, but loses the
odd-teacher depth-4, strength-0.25, seed-1 case. That miss finishes at
**0.0500355988**, above the unchanged 0.05 threshold; it remains a failure.

| Activation | Biases | Odd teacher | Mixed teacher | Total | Worst final held-out relative MSE |
|---|---|---:|---:|---:|---:|
| Shifted | None, reused | 6/12 | 3/12 | 9/24 | 0.137495 |
| Shifted | Output | 6/12 | 4/12 | 10/24 | 0.135877 |
| Shifted | Hidden + output | 6/12 | 6/12 | 12/24 | 0.128899 |
| Learned-even | None, reused | 12/12 | 3/12 | 15/24 | 0.141903 |
| Learned-even | Output | 12/12 | 3/12 | 15/24 | 0.135563 |
| Learned-even | **Hidden + output** | **11/12** | **6/12** | **17/24** | **0.130742** |

The selected arm has two earlier, seven tied and five later common passes, plus
three recoveries and one new miss. Its six remaining mixed-teacher failures are
every strength-1 case at both depths. Their final held-out errors are
0.115274–0.130742, despite final training errors of 0.011837–0.022319. The
generalization failure remains substantial. No configuration passes 24/24, so
no pooled first-crossing geometric mean is reported.

The output-only learned-even arm preserves coverage and improves three crossing
times, tying the other twelve. Shifted output-only recovers one case; shifted
hidden-plus-output recovers three, without new misses. Every paired seed and
capacity change is in [paired.json](results/paired.json), with all 144 cells in
[cells.csv](results/cells.csv). The selected architecture has 8,672/34,688 total
parameters at depths 4/16, including 408/1,632 added bias scalars (about 4.94%
more than its no-bias control).

All **96 new runs are finite**: 54 passes and 42 budget misses. The
[checkpoint audit](results/audit.json) reproduces all new full-split final scores
and verifies 58 archived source/control/validation files, original matrices,
added capacity, histories, optimizer states/settings and sampler states. The
validated implementation was committed as `a0d0135` before training. This is
another partial repair with a documented regression, not a fully solved stress
suite or authorization to replace the full GPT.

Raw records remain under `runs/transformer-stress/bias-v1/`.
Generic finite-epsilon RMS and post-RMS
residual-radius bounds still apply to arbitrary biased branch outputs. Trained
biases can break input oddness, so bias-free oddness statements do not transfer.
No whole-network, generalization or convergence theorem is inferred. This
adaptive synthetic study changes neither the full GPT nor the independently
running inverse-width benchmark.
