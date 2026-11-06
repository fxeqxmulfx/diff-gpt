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

Status: proofs and reference validation complete; native local comparison pending.
The capacity grid, usual-gain test and deferred full-stack repairs stay intact.
