# Bounds for stored RoPE tables

The [plan](plan.json) supplements the architecture guarantees without changing
any experiment. Actual FP32 cosine and sine pairs need not have exactly unit
energy. The existing Lean `tableRotate` map already models those independent
stored coefficients in the source half-split layout.

The proposed bound uses `M >= 1` and `c² + s² <= M` for every pair. The norm
factor is `sqrt(M)`; the Q/K dot-product factor is `M`. An exact rational audit
will check all tables in the 35 fixed task specifications against `M = 1 + 2^-20`.
No learning outcomes determine this constant or change the active screen.

This addresses stored-table distortion over the reals. Arithmetic rounding when
applying the tables, other model operations and learning remain outside the bound.
Status: declared; proof and application checks pending.
