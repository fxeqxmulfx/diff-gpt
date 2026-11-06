import Transformer.Landscape.Neural.Basic
import Mathlib.Analysis.Complex.Trigonometric

/-!
Real-arithmetic contracts for models.py at 95ce273. Scalar g and v are the
two linear projections of an input; changing the input's sign negates both.
These identities describe expressivity and initialization, not optimization
or full-transformer odd symmetry (embeddings and attention need not be odd).
-/

namespace TransformerParityStress

open Transformer.Landscape.Neural

/-- Exact decomposition of the unit-slope SiLU used by the parent. -/
theorem silu_decomposition (g : ℝ) :
    silu g = g / 2 + g / 2 * Real.tanh (g / 2) := by
  have he : Real.exp (g / 2) ≠ 0 := ne_of_gt (Real.exp_pos (g / 2))
  have hd : Real.exp (g / 2) + (Real.exp (g / 2))⁻¹ ≠ 0 := by positivity
  have hr : 1 + ((Real.exp (g / 2))⁻¹) ^ 2 ≠ 0 := by positivity
  have hdecay : Real.exp (-g) = ((Real.exp (g / 2))⁻¹) ^ 2 := by
    rw [show -g = -(g / 2) + -(g / 2) by ring, Real.exp_add, Real.exp_neg]
    ring
  rw [silu, Real.tanh_eq, hdecay, Real.exp_neg]
  field_simp
  ring

/-- The proposed product before the unchanged output projection and RMS. -/
noncomputable def product (g v b : ℝ) : ℝ :=
  (1 + g / 2 * Real.tanh (g / 2) + b * g / 2) * v

/-- Setting the learned coefficient to one recovers the shifted parent exactly. -/
theorem recovers_parent (g v : ℝ) : product g v 1 = (1 + silu g) * v := by
  rw [silu_decomposition]
  unfold product
  ring

/-- The zero initial coefficient makes the branch product odd in its input. -/
theorem initial_product_odd (g v : ℝ) :
    product (-g) (-v) 0 = -product g v 0 := by
  unfold product
  rw [show -g / 2 = -(g / 2) by ring, Real.tanh_neg]
  ring

/-- The learned coefficient controls the quadratic even part, without deleting it. -/
theorem even_component (g v b : ℝ) :
    (product g v b + product (-g) (-v) b) / 2 = b * g * v / 2 := by
  unfold product
  rw [show -g / 2 = -(g / 2) by ring, Real.tanh_neg]
  ring

/-- The remaining component is exactly the odd part of shifted SwiGLU. -/
theorem odd_component (g v b : ℝ) :
    (product g v b - product (-g) (-v) b) / 2 =
      (1 + g / 2 * Real.tanh (g / 2)) * v := by
  unfold product
  rw [show -g / 2 = -(g / 2) by ring, Real.tanh_neg]
  ring

#print axioms TransformerParityStress.silu_decomposition
#print axioms TransformerParityStress.recovers_parent
#print axioms TransformerParityStress.initial_product_odd
#print axioms TransformerParityStress.even_component
#print axioms TransformerParityStress.odd_component

end TransformerParityStress
