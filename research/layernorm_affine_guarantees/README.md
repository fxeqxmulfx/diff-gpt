# Finite-epsilon LayerNorm and affine normalization bounds

The user keeps Peri-normalization, embedding normalization and tied weights, while
leaving RMSNorm, LayerNorm and learned parameters open. This [plan](plan.json)
examines those choices locally without selecting or training a new architecture.

Use the actual population-variance LayerNorm equation: subtract the coordinate
mean, then apply finite-epsilon RMS. Prove its radius, input-sensitivity and
constant-shift properties. For either normalizer, make the learned diagonal
scale and bias explicit in the output and input-sensitivity bounds.

After the proofs, a declared 160-case CPU comparison will check existing RMS and
native LayerNorm outputs and input/affine gradients against independent 80-digit
Decimal formulas. It includes singleton and constant inputs, and retains every
numerical failure under the fixed 0.001 relative-error threshold. No model change,
optimizer update or task evaluation is part of this supplement.

All eight target statements now compile without warnings or extra axioms. The
five [centering and LayerNorm statements](Centering.lean) reuse the improved
finite-epsilon RMS bound. The three [affine statements](Affine.lean) explicitly
include the learned parameters: if every `abs(gamma_i) <= G`, with `G>=0`, then

```text
||gamma * N(x) + beta|| <= G*sqrt(d) + ||beta||
||gamma * N(x) + beta - (gamma * N(y) + beta)||
    <= (G/sqrt(epsilon)) * ||x-y||.
```

Here `N` is either RMS or LayerNorm, the product is coordinatewise, and gamma/beta
are fixed during the input comparison. These are conditional bounds. Unconstrained
training does not provide a fixed bound on their magnitude. LayerNorm additionally
removes every constant coordinate shift; the radius/sensitivity bound alone does
not decide which normalizer is better for learning.

The independent Decimal reference passes three checks: analytic input/gamma/beta
derivatives agree with high-precision finite differences for both normalizers,
and singleton LayerNorm plus zero-input RMS match exact elementary values.
The comparison driver requires these checks, the proof record and committed
sources before any of the 160 measurements. All proof-development failures are
retained in the results directory.

The comparison launched from committed source **2be94d2** and completed all
**160 cases and 480 vector comparisons**, with no execution errors or nonfinite
vectors. **146/160 cases pass** the declared rule. All **160 forward outputs pass**;
the 14 misses are gradient discrepancies. Both implementations use the benchmark
epsilon `2^-23` in this local study.

Each table entry covers all 20 declared width/input combinations:

| CPU dtype | Existing RMS, no affine | Existing RMS, affine | Native LayerNorm, no affine | Native LayerNorm, affine |
| --- | ---: | ---: | ---: | ---: |
| FP32 | 20/20 | 20/20 | 14/20 | 14/20 |
| FP64 | 20/20 | 20/20 | 20/20 | 18/20 |

The full [measurements](results/measurements.json), [160 cases](results/cells.csv),
[480 metrics](results/metrics.csv) and [failure table](results/summary.json) preserve
all vectors and outcomes. The [audit](results/complete-audit.json) reconstructs
every represented input and every Decimal reference vector, recomputes errors
independently with scalar sums, and checks the source archive and raw cell files.

Six LayerNorm misses concern FP32 input gradients without affine parameters:

- At unit-RMS balanced inputs, the true radial gradient is only about `1.1921e-7`
  per coordinate. Native results lose its direction, giving about 100.78% relative
  error at widths 8/128/256. The absolute errors remain small.
- At `3 + 0.02*direction`, input-gradient relative errors are about 1.64%, 3.68%
  and 16.46% at those widths; the reference magnitude is about 0.014895 per coordinate.

Eight misses concern affine-scale gradients whose reference is exactly zero.
Six FP32 cases have maximum component errors from `3.79e-7` to `3.82e-4`.
The two FP64 singleton cases have absolute errors `2.34e-15` and `2.03e-13`.
They remain failures because the predeclared rule requires exact zero for an
identically zero reference. Every bias gradient passes.

The existing RMS implementation already uses a repaired FP64 backward. Native
LayerNorm uses its own CPU kernels. These results compare those implementations;
they do not prove that RMS is intrinsically a better normalizer or that LayerNorm
would fail training. The real-arithmetic bounds remain valid. A future LayerNorm
model needs its own declared implementation validation before a learning comparison.
No normalizer is selected here, and no numerical threshold or implementation was
changed after observing these outcomes.

All four failed proof-development attempts and the final clean attempt are
retained. No optimizer updates or task evaluations occurred. The capacity grid,
usual-gain test and deferred full-stack repairs stay intact.

```sh
# Recheck proofs from the Lean project:
PYTHONPATH=/home/misha/python_projects/gpt lake env python3 /home/misha/python_projects/gpt/research/layernorm_affine_guarantees/check.py
# Recheck the scalar reference from the GPT repository:
.venv/bin/pytest -q tests/diff_gpt/application/test_layernorm_affine_reference.py
# Audit the already completed comparison without rerunning measurements:
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m research.layernorm_affine_reporting.run
```

The frozen measurement command is
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m research.layernorm_affine_guarantees.compare`.
Its guard refuses to overwrite an existing comparison.
