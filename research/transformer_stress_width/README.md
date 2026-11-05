# Residual stress internal-width follow-up

The [plan](plan.json) increases only SwiGLU's internal feature width, retaining
the state dimension 16, selected LR 0.01, post-RMS zero-initialized scalar
gates, data, seeds, optimizer, 512 updates and the 5% relative-MSE target.
Every original gate/value row and output column is copied exactly; added
features have a separately seeded initialization. This changes capacity and
initial outputs. Three validation tests check the retained weights, extra
parameter count, reproducibility and unchanged training-loop AST.

| Hidden width | Parameters at 4 / 16 branches | Passed / 12 | Worst final relative MSE |
|---|---:|---:|---:|
| 43, reused control | 8260 / 33040 | 6 | 0.09002 |
| 128 | 24580 / 98320 | 6 | 0.07253 |
| **256** | **49156 / 196624** | **6** | **0.06444** |

Width 256 ranks first by the declared worst-error tie-break, but all six
shallow cases still miss. It improves errors at nearly six times the control's
parameter count; it is not an equal-size or complete-coverage win. No filtered
first-crossing mean is reported. These are outcomes on a repeatedly observed
synthetic diagnostic, not evidence of universal or full-GPT superiority.

[Complete results](results/summary.json), [all-cell CSV](results/cells.csv),
and the [24-cell audit](results/audit.json) retain every condition. Source/data
hashes, checkpoints, full-split scores, optimizer/sampler states and reused
control rows are verified. Raw outputs remain in
`runs/transformer-stress/width-v1/`.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m research.transformer_stress_width.run
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python research/stress_audits/followups.py width
```

The separately declared output-epsilon study returns to width 43 to isolate
amplitude preservation in branch normalization without adding parameters.
