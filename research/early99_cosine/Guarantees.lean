import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Tactic

/-!
Real scalar bounds for the implemented cosine schedule. These apply pointwise
to the existing exposure-clock formula, including its held floor after progress1.
They establish neither nonconvex convergence nor earlier held-out accuracy.
-/

namespace Early99Cosine

noncomputable def factor (floor progress : ℝ) : ℝ :=
  floor + (1 - floor) * (1 + Real.cos (Real.pi * progress)) / 2

/-- A floor at most one bounds the cosine multiplier for every real progress. -/
theorem factor_bounds (floor progress : ℝ) (hfloor : floor ≤ 1) :
    floor ≤ factor floor progress ∧ factor floor progress ≤ 1 := by
  have hs : 0 ≤ 1 - floor := by linarith
  have hc0 := Real.neg_one_le_cos (Real.pi * progress)
  have hc1 := Real.cos_le_one (Real.pi * progress)
  have h0 := mul_nonneg hs (by linarith : 0 ≤ 1 + Real.cos (Real.pi * progress))
  have h1 := mul_nonneg hs (by linarith : 0 ≤ 1 - Real.cos (Real.pi * progress))
  unfold factor
  constructor <;> nlinarith

/-- Nonnegative peak/floor give a nonnegative rate between floor*peak and peak. -/
theorem rate_bounds (peak floor progress : ℝ) (hp : 0 ≤ peak)
    (hf0 : 0 ≤ floor) (hf1 : floor ≤ 1) :
    0 ≤ peak * factor floor progress ∧
      floor * peak ≤ peak * factor floor progress ∧
      peak * factor floor progress ≤ peak := by
  obtain ⟨hl, hu⟩ := factor_bounds floor progress hf1
  constructor
  · exact mul_nonneg hp (le_trans hf0 hl)
  constructor
  · nlinarith [mul_nonneg hp (sub_nonneg.mpr hl)]
  · nlinarith [mul_nonneg hp (sub_nonneg.mpr hu)]

/-- Decay starts at the peak rather than applying a second warmup. -/
theorem at_start (floor : ℝ) : factor floor 0 = 1 := by
  simp [factor]
  ring

/-- Reaching one declared decay horizon gives the configured floor. -/
theorem at_floor (floor : ℝ) : factor floor 1 = floor := by
  simp [factor]

example (progress : ℝ) := factor_bounds (1/10) progress (by norm_num)
example (progress : ℝ) := rate_bounds (1/500) (1/10) progress
  (by norm_num) (by norm_num) (by norm_num)
example : factor (1/10) 0 = 1 := at_start _
example : factor (1/10) 1 = 1/10 := at_floor _

#print axioms factor_bounds
#print axioms rate_bounds
#print axioms at_start
#print axioms at_floor
end Early99Cosine
