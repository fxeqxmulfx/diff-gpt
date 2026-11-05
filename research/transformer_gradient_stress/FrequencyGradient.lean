import Transformer.RecurrentGradients.Section2_Mechanics

/-!
Real-arithmetic contract for frequency.py at ed3c65f.
The forward remains exp(4*tanh(a/4)). Its VJP replaces subtraction of tanh²
with an equivalent exponential quotient. This does not prove floating-point
accuracy, a nonzero representable optimizer update, or learning success.
-/

namespace TransformerGradientStress

/-- An exponential form of tanh used to identify the implemented VJP. -/
theorem tanh_decay (z : ℝ) :
    Real.tanh z = (1 - Real.exp (-2 * z)) / (1 + Real.exp (-2 * z)) := by
  have he : Real.exp z ≠ 0 := ne_of_gt (Real.exp_pos z)
  have hd : Real.exp z + (Real.exp z)⁻¹ ≠ 0 := by positivity
  have hr : 1 + ((Real.exp z)⁻¹) ^ 2 ≠ 0 := by positivity
  have hdecay : Real.exp (-2 * z) = ((Real.exp z)⁻¹) ^ 2 := by
    rw [show -2 * z = -z + -z by ring, Real.exp_add, Real.exp_neg]
    ring
  rw [Real.tanh_eq, Real.exp_neg, hdecay]
  field_simp

/-- The quotient avoids cancellation in 1-tanh² but is the same real value. -/
theorem sech_squared_decay (z : ℝ) :
    1 - Real.tanh z ^ 2 =
      4 * Real.exp (-2 * z) / (1 + Real.exp (-2 * z)) ^ 2 := by
  rw [tanh_decay]
  have hd : 1 + Real.exp (-2 * z) ≠ 0 := by positivity
  field_simp
  ring

/-- Absolute value keeps the exponential argument nonpositive for either sign. -/
theorem stable_sech_squared (z : ℝ) :
    1 - Real.tanh z ^ 2 =
      4 * Real.exp (-2 * |z|) / (1 + Real.exp (-2 * |z|)) ^ 2 := by
  by_cases hz : 0 ≤ z
  · rw [abs_of_nonneg hz]
    exact sech_squared_decay z
  · have hneg : z < 0 := by linarith
    rw [abs_of_neg hneg]
    simpa using sech_squared_decay (-z)

/-- Chain rule gives exactly the replacement backward expression for every a. -/
theorem bounded_frequency_hasDerivAt (a : ℝ) :
    HasDerivAt (fun x : ℝ => Real.exp (4 * Real.tanh (x / 4)))
      (Real.exp (4 * Real.tanh (a / 4)) *
        (4 * Real.exp (-2 * |a / 4|) / (1 + Real.exp (-2 * |a / 4|)) ^ 2)) a := by
  have hinner := (hasDerivAt_id a).div_const (4 : ℝ)
  have htan := (Transformer.RecurrentGradients.hasDerivAt_tanh (a / 4)).comp a hinner
  have h := (htan.const_mul (4 : ℝ)).exp
  change HasDerivAt (fun x : ℝ => Real.exp (4 * Real.tanh (x / 4)))
    (Real.exp (4 * Real.tanh (a / 4)) *
      (4 * ((1 - Real.tanh (a / 4) ^ 2) * (1 / 4)))) a at h
  rw [stable_sech_squared] at h
  convert h using 1
  ring

/-- The corrected derivative is strictly positive in real arithmetic. -/
theorem bounded_frequency_derivative_positive (a : ℝ) :
    0 < Real.exp (4 * Real.tanh (a / 4)) *
      (4 * Real.exp (-2 * |a / 4|) / (1 + Real.exp (-2 * |a / 4|)) ^ 2) := by
  positivity

#print axioms TransformerGradientStress.tanh_decay
#print axioms TransformerGradientStress.sech_squared_decay
#print axioms TransformerGradientStress.stable_sech_squared
#print axioms TransformerGradientStress.bounded_frequency_hasDerivAt
#print axioms TransformerGradientStress.bounded_frequency_derivative_positive

end TransformerGradientStress
