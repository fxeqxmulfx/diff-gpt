# Finite Muon matrix bounds

These sibling proofs extend the [frozen scalar bounds](../early99_muon/Guarantees.lean)
without changing the optimizer or its running experiments. They formalize the
declared epsilon floor and five quintic Newton–Schulz steps, including the
transpose used for tall matrices.

[Bounds.lean](Bounds.lean) proves eight statements. With an explicit representation
`X = U diag(s) Vᵀ`, orthonormal columns of `U` and `V`, and nonnegative `s`, the
direction has squared Frobenius norm at most `25r/16`, where `r` is the number of
columns in the representation. Zero coefficients and rank deficiency are allowed.
For positive matrix dimensions and `r <= min(rows, columns)`, multiplying by
`0.2 * sqrt(max(rows, columns))` gives RMS at most `1/4`. At a nonnegative learning
rate, the parameter displacement from its decayed value has RMS at most `rate/4`.

These are real-arithmetic bounds, conditional on the stated representation.
They do not include floating-point rounding, auxiliary AdamW, or the displacement
from weight decay itself. They do not establish faster learning, convergence of
the transformer, or robustness of its other blocks. The existing full-model
stress result remains 40/50, with ten unresolved failures.

Run from `/home/misha/lean_projects/transformer`:

```sh
lake env python3 /home/misha/python_projects/gpt/research/early99_muon_math/check.py
```

The checker compiles the unchanged prerequisite and these bounds into a private
cache under `runs/`, verifies the frozen dependency hashes, and records the exact
sources and output in [results/guarantees.json](results/guarantees.json). All printed
theorems must use only standard Lean axioms, with no warnings or errors. Earlier
compilation logs are retained.

[DecayBounds.lean](DecayBounds.lean) adds four statements about the complete
decayed parameter trajectory. For positive decay and epsilon, nonnegative rates
with `rate * decay <= 1`, and the same explicit singular representation at each
step, every real iterate satisfies:

```text
RMS(parameter at step n) <= max(RMS(initial parameter), 1 / (4 * decay))
```

The proof uses convexity of squared Frobenius energy and the already checked
finite direction bound. It includes zero-rate warmup steps and rank-deficient
directions. It applies to the Muon partition; the hybrid's auxiliary AdamW
parameters are outside this theorem. Floating-point accumulation remains
outside the bound, and bounded parameters do not establish convergence or
successful generalization.

Run `lake env python3 /home/misha/python_projects/gpt/research/early99_muon_math/check_decay.py`
from the external Lean project, then run
`PYTHONPATH=. .venv/bin/python -m research.early99_muon_math.audit_decay` from this
repository. The separate [proof record](results/decay-guarantees.json) preserves
the original eight-statement record. The
[application audit](results/decay-application.json) verifies all 140 combinations
of four declared configurations and 35 tasks: decay is 0.1 throughout, and
`peak_rate * decay` is at most 0.0004. Thus the real trajectory cap is
`max(initial RMS, 2.5)` for these settings. This audit reads declarations and
checks warmup boundary rates; it runs no training or outcome-based selection.

The first extension compile failed because a nonnegativity lemma was not imported
and two tactics followed goals that were already closed. Its log is retained.
The final four statements compile without warnings or extra axioms.
