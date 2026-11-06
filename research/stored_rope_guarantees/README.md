# Bounds for stored RoPE tables

The [plan](plan.json) supplements the architecture guarantees without changing
any experiment. Actual FP32 cosine and sine pairs need not have exactly unit
energy. The existing Lean `tableRotate` map already models those independent
stored coefficients in the source half-split layout.

The checked bound uses `M >= 1` and `c² + s² <= M` for every pair. The norm
factor is `sqrt(M)`; the Q/K dot-product factor is `M`. Ordinary normalized
scores are bounded by `abs(gain) * M * d`, and SSMax adds `log(visible keys)`.
The actual fixed gain remains explicit. Query and key may use different rows;
odd leftover coordinates are also covered by the general theorem.

Lean checked all four statements with only standard axioms and no warnings.
The [proof record](results/guarantees.json) pins the source and imported library.
The [exact rational audit](results/application.json) passed **35/35 models**:
7,640 stored pairs, or 15,280 coefficients, all satisfy the predeclared
`M = 1048577/1048576 = 1 + 2^-20`. No learning outcome determined this constant.
The largest exact pair energy is `1125899990935417/1125899906842624`, approximately
**1.000000074689404**, at position 100, pair 1 of `basis-depth-hard`.
The smallest is approximately 0.9999999111747471: these are not exact isometries.
Seventy archived binary table files and their hashes preserve the actual bytes.

This addresses stored-table distortion over the reals. Arithmetic rounding when
applying the tables, other model operations and learning remain outside the bound.
The initial environment, example-lint and parser issues are recorded in
[attempt notes](results/attempt-notes.json), with every compiler output retained.
No benchmark model, training setting or selection rule changed.

```bash
# From /home/misha/lean_projects/transformer:
lake env python3 /home/misha/python_projects/gpt/research/stored_rope_guarantees/check.py
# From the GPT repository:
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m research.stored_rope_guarantees.audit_tables
```
