import Transformer.XSA
import Transformer.Grokking.FlatSpectrum.FeatureCodes.CubicQuotient
import Transformer.Rotate90.ComplexProduct.Defs

/-!
Local review checks for the seed-1 quartic diagnostic. These statements concern
exact algebra, not convergence or the trained weights. They are kept beside the
experiment without modifying the separate Lean project.

Sources: Transformer/XSA.lean (projection definition), ComplexProduct/Defs.lean
(plain product), and FeatureCodes/FiberCodes.lean (answer-class readout).
The native Hermitian projector has the complex projection form below whenever
the value norm is above its clipping threshold. The repeated-prefix path
explicitly enforces its cancellation in that regime.
-/

namespace QuarticDiagonalReview

open scoped BigOperators
open Matrix
open Transformer.Grokking.FlatSpectrum.FeatureCodes

variable {I E : Type*} [Fintype I] [NormedAddCommGroup E] [InnerProductSpace ℂ E]

/-- Normalized attention weights retain a common value, independently of Q/K. -/
theorem equal_values_survive_attention (p : I → ℂ) (hp : ∑ i, p i = 1) (v : E) :
    (∑ i, p i • v) = v := by
  rw [← Finset.sum_smul, hp, one_smul]

/-- Removing the Hermitian component along the common value cancels it. -/
theorem hermitian_exclusion_cancels_self (v : E) :
    v - (inner (𝕜 := ℂ) v v / (‖v‖ : ℂ) ^ 2) • v = 0 := by
  by_cases hv : v = 0
  · simp [hv]
  · have hn : (‖v‖ : ℂ) ^ 2 ≠ 0 :=
      pow_ne_zero 2 (Complex.ofReal_ne_zero.mpr (norm_ne_zero_iff.mpr hv))
    have hi : inner (𝕜 := ℂ) v v = (‖v‖ : ℂ) ^ 2 := by
      simp
    rw [hi, div_self hn, one_smul, sub_self]

/-- Equal values vanish under Hermitian XSA even if positional Q/K change p. -/
theorem equal_values_vanish_under_xsa (p : I → ℂ) (hp : ∑ i, p i = 1) (v : E) :
    (∑ i, p i • v) -
      (inner (𝕜 := ℂ) v (∑ i, p i • v) / (‖v‖ : ℂ) ^ 2) • v = 0 := by
  rw [equal_values_survive_attention p hp v]
  exact hermitian_exclusion_cancels_self v

/-- Multiplication itself does not cancel equal nonzero scalar factors. -/
theorem complex_product_self_nonzero {z : ℂ} (hz : z ≠ 0) :
    z * z / (Real.sqrt 2 : ℂ) ≠ 0 := by
  apply div_ne_zero (mul_ne_zero hz hz)
  exact_mod_cast (ne_of_gt (Real.sqrt_pos.mpr (by norm_num : (0 : ℝ) < 2)))

private instance : Fact (Nat.Prime 97) := ⟨by norm_num⟩

/-- Quartic answer equivalence has four roots, unlike the cubic example. -/
theorem quartic97_factor (s t : ZMod 97) :
    (s - t) * (s - 22 * t) * (s - 75 * t) * (s - 96 * t) = s ^ 4 - t ^ 4 := by
  ring_nf
  rw [show (169362 : ZMod 97) = 0 by decide, show (11155 : ZMod 97) = 0 by decide,
    show (194 : ZMod 97) = 0 by decide, show (158400 : ZMod 97) = -1 by decide]
  ring

/-- This includes zero and hence applies to all operand sums. -/
theorem quartic97_eq_iff (s t : ZMod 97) :
    s ^ 4 = t ^ 4 ↔ s = t ∨ s = 22 * t ∨ s = 75 * t ∨ s = 96 * t := by
  rw [← sub_eq_zero, ← quartic97_factor, mul_eq_zero, mul_eq_zero, mul_eq_zero]
  simp only [sub_eq_zero]
  tauto

/-- The existing generic fiber theorem applies to quartic answers as well.
`cubicFourier97` is just the normalized additive Fourier matrix; its isometry
does not depend on the choice of answer polynomial. -/
theorem quartic_distinct_features_same_readouts :
    cubicFourier97 *ᵥ Pi.single 1 1 ≠ cubicFourier97 *ᵥ Pi.single 22 1 ∧
    ∀ a : ZMod 97,
      fiberReadout cubicFourier97 (fun s => s ^ 4) a
          (cubicFourier97 *ᵥ Pi.single 1 1) =
        fiberReadout cubicFourier97 (fun s => s ^ 4) a
          (cubicFourier97 *ᵥ Pi.single 22 1) := by
  refine ⟨feature_columns_ne cubicFourier97_isometry (by decide), ?_⟩
  exact fiber_readouts_eq_of_same_answer cubicFourier97_isometry
    (fun s : ZMod 97 => s ^ 4) (s := 1) (t := 22) (by decide)

example := equal_values_survive_attention (fun _ : Fin 2 => (1 / 2 : ℂ))
  (by norm_num [Fin.sum_univ_two]) (1 : ℂ)
example := hermitian_exclusion_cancels_self (1 : ℂ)
example := equal_values_vanish_under_xsa (fun _ : Fin 2 => (1 / 2 : ℂ))
  (by norm_num [Fin.sum_univ_two]) (1 : ℂ)
example := complex_product_self_nonzero (z := 1) (by norm_num)
example := quartic97_factor 1 22
example := quartic97_eq_iff 0 0

#print axioms equal_values_survive_attention
#print axioms hermitian_exclusion_cancels_self
#print axioms equal_values_vanish_under_xsa
#print axioms complex_product_self_nonzero
#print axioms quartic97_factor
#print axioms quartic97_eq_iff
#print axioms quartic_distinct_features_same_readouts

end QuarticDiagonalReview
