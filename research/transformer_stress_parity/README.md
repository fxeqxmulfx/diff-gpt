# Learning SwiGLU's even component separately

The previous rate, width and epsilon studies left all six shallow residual
cases unsolved. An [error decomposition](results/diagnosis.json) found an even
component with relative energy 0.0224–0.0249 in the LR 0.01 shallow models, despite
the teacher being odd. Removing that component after training would still leave
odd-part error above the unchanged 0.05 threshold, so this was only a diagnosis.

The [frozen plan](plan.json), source `95ce273`, changes shifted SwiGLU from
`(1+SiLU(g))*v` to

```
(1 + 0.5*g*tanh(0.5*g) + 0.5*b*g)*v
```

Each branch learns one new scalar `b`, initialized to zero. At `b=1`, the
expression equals the parent in real arithmetic. All matrices, zero-initialized
residual gates, normalization, optimizer, data, 512 updates and scoring rules
stay the same. The four/16-branch parameter counts rise from 8,260/33,040 to
8,264/33,056. This is a changed model and initialization, not a numerical repair.

To check the symmetry bias, both arms also learn the separately declared mixed
teacher `tanh(xR)+0.5*((xR)^2-1)`. It uses the same inputs and rotation and needs
an even component. The original teacher arrays remain unchanged. All 36 new
cells and 12 reused controls are retained in the [complete results](results/summary.json).

| Teacher | Shifted parent passes / 12 | Learned even path passes / 12 | Parent worst final error | New worst final error |
|---|---:|---:|---:|---:|
| Original odd | 6 | **10** | 0.09002 | **0.05328** |
| Mixed odd/even | **3** | 0 | **0.15534** | 0.16344 |
| Both, out of 24 | 9 | **10** | **0.15534** | 0.16344 |

The new arm fixes four original shallow failures. Two still miss: depth 4,
strength 0.25/seed 1 ends at 0.05328; strength 1/seed 0 ends at 0.05014.
All six deeper original cells pass. On the mixed teacher, the new arm loses
the parent's three deeper, weak-residual passes. All cells remain finite.

The new arm wins the predeclared pooled coverage rule by one cell, but has worse
worst-case error and three explicit regressions. **It is not a demonstrated
replacement for a shared transformer.** Representing the parent at `b=1` does
not ensure that optimization from `b=0` finds it quickly. No successful-only
geometric mean is reported. The new training cells total 113.29 CPU seconds;
runtime remains descriptive, not the selection objective.

Six tests verify common matrices and parameter counts, recovery of the parent
with active residual gates, odd initialization, a nonzero gradient for the even
path, unchanged original data, the training-loop AST and incomplete-report
handling. The [audit](results/audit.json) checks 44 source snapshots, all 36 new
checkpoints, full-split final scores, optimizer/sampler state and all 48 rows.
Raw histories and checkpoints are in `runs/transformer-stress/parity-v1/`.

The [Lean fixture](ActivationParity.lean) proves the SiLU decomposition,
recovery of the shifted parent, initial oddness, and exact odd/even components
of the pre-projection product. The [proof record](results/guarantees.json)
and [checker output](results/lean-check.txt) report zero warnings and only
standard Lean axioms. These statements do not assert learning success or
odd symmetry of the full attention transformer.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m research.transformer_stress_parity.run
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python research/stress_audits/parity.py
```
