import Transformer.Precision.ArithmeticError

/-! A finite RMS forward control does not bound raw backward products.
This matches static-precision-03's x=2^60 and zero-mean tangent g of magnitude2^80.
All claims are exact-real/IEEE representability claims; CUDA roundoff is checked
independently by the native RMS forward/backward control. -/
open Transformer.Precision
namespace MainModelRawGradientRange

theorem finite_inputs_and_energy_but_large_product :
    (2 : ℝ)^60 ∈ grid 8 23 ∧ (2 : ℝ)^80 ∈ grid 8 23 ∧
    128 * ((2 : ℝ)^60)^2 ∈ grid 8 23 ∧
    128 * ((2 : ℝ)^60)^2 < (2 : ℝ)^128 - 2^104 ∧
    (2 : ℝ)^60 * 2^80 > (2 : ℝ)^128 - 2^104 := by
  refine ⟨?_, ?_, ?_, by norm_num, by norm_num⟩
  · convert binary32_value_mem false 187 0 (by norm_num) (by norm_num) using 1
    norm_num [binary32Value]
  · convert binary32_value_mem false 207 0 (by norm_num) (by norm_num) using 1
    norm_num [binary32Value]
  · convert binary32_value_mem false 254 0 (by norm_num) (by norm_num) using 1
    norm_num [binary32Value]

/-- The exact normalized scalar remains in the unit interval despite the
raw finite product in the previous theorem exceeding the largest binary32. -/
theorem normalized_high_input_bounded :
    0 < (2 : ℝ)^60 / Real.sqrt ((2 : ℝ)^120 + 1/8388608) ∧
    (2 : ℝ)^60 / Real.sqrt ((2 : ℝ)^120 + 1/8388608) ≤ 1 := by
  have hs : 0 < Real.sqrt ((2 : ℝ)^120 + 1/8388608) := by positivity
  constructor
  · positivity
  · apply (div_le_iff₀ hs).2
    simp only [one_mul]
    apply (Real.le_sqrt (by positivity) (by positivity)).2
    norm_num

#print axioms finite_inputs_and_energy_but_large_product
#print axioms normalized_high_input_bounded
end MainModelRawGradientRange
