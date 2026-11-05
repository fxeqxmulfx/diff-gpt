import Transformer.RecurrentGradients.Section2_Mechanics
import Transformer.Residual.GradientPaths
import Transformer.MHC.Section4_Stability
import Transformer.MHC.Section4_Sinkhorn

/-!
Real-arithmetic contracts for the independently declared stress fixes.
Source: models.py at 60ad06f. The bounded frequency is a model change;
the pairwise RMS identity is an alternative evaluation of the same VJP.
No theorem below asserts floating-point correctness or learning success.
-/

namespace TransformerStress

open scoped BigOperators

/-- The exact multiplier used by the explicitly bounded rotary candidate. -/
noncomputable def boundedFrequency (a : ℝ) : ℝ :=
  Real.exp (4 * Real.tanh (a / 4))

/-- Unlike exp(a), the tested frequency multiplier has a uniform finite range. -/
theorem frequency_bounds (a : ℝ) :
    Real.exp (-4) ≤ boundedFrequency a ∧ boundedFrequency a ≤ Real.exp 4 := by
  have hlo := Real.neg_one_lt_tanh (a / 4)
  have hhi := Real.tanh_lt_one (a / 4)
  constructor
  · apply Real.exp_le_exp.mpr
    nlinarith
  · apply Real.exp_le_exp.mpr
    nlinarith

/-- Chain rule for the actual 4*tanh(a/4) log-frequency, not a clipped exp. -/
theorem frequency_hasDerivAt (a : ℝ) :
    HasDerivAt boundedFrequency
      (boundedFrequency a * (1 - Real.tanh (a / 4) ^ 2)) a := by
  have hinner := (hasDerivAt_id a).div_const (4 : ℝ)
  have htan := (Transformer.RecurrentGradients.hasDerivAt_tanh (a / 4)).comp a hinner
  have h := (htan.const_mul (4 : ℝ)).exp
  change HasDerivAt boundedFrequency
    (boundedFrequency a * (4 * ((1 - Real.tanh (a / 4) ^ 2) * (1 / 4)))) a at h
  convert h using 1
  ring

/-- The multiplier derivative is also bounded, independently of its raw parameter. -/
theorem frequency_derivative_bound (a : ℝ) :
    |boundedFrequency a * (1 - Real.tanh (a / 4) ^ 2)| ≤ Real.exp 4 := by
  have hp : 0 ≤ boundedFrequency a := (Real.exp_pos _).le
  have ht : 0 ≤ 1 - Real.tanh (a / 4) ^ 2 := by
    have h := Real.tanh_sq_lt_one (a / 4)
    linarith
  rw [abs_of_nonneg (mul_nonneg hp ht)]
  calc
    boundedFrequency a * (1 - Real.tanh (a / 4) ^ 2) ≤ boundedFrequency a * 1 :=
      mul_le_mul_of_nonneg_left (by nlinarith [sq_nonneg (Real.tanh (a / 4))]) hp
    _ = boundedFrequency a := mul_one _
    _ ≤ Real.exp 4 := (frequency_bounds a).2

/-- Initialization keeps both the unbounded parent's frequency and first derivative. -/
theorem initial_frequency_and_derivative :
    boundedFrequency 0 = 1 ∧ HasDerivAt boundedFrequency 1 0 := by
  constructor
  · simp [boundedFrequency]
  · simpa [boundedFrequency] using frequency_hasDerivAt 0

/-- Pairwise differences compute the same tangent numerator as a projection.
The implementation promotes products before evaluating this identity. -/
theorem pairwise_tangent_numerator {d : ℕ} (x g : Fin d → ℝ) (i : Fin d) :
    (∑ j, x j * (x j * g i - x i * g j)) =
      (∑ j, (x j) ^ 2) * g i - x i * (∑ j, x j * g j) := by
  calc
    (∑ j, x j * (x j * g i - x i * g j)) =
        ∑ j, ((x j) ^ 2 * g i - x i * (x j * g j)) := by
      apply Finset.sum_congr rfl
      intro j _
      ring
    _ = (∑ j, (x j) ^ 2) * g i - x i * (∑ j, x j * g j) := by
      rw [Finset.sum_sub_distrib, ← Finset.sum_mul, ← Finset.mul_sum]

/-- Division gives g minus its projection onto x for every nonzero energy.
At zero energy the implementation uses its separate finite-epsilon expression. -/
theorem pairwise_tangent_eq {d : ℕ} (x g : Fin d → ℝ) (i : Fin d)
    (h : (∑ j, (x j) ^ 2) ≠ 0) :
    (∑ j, x j * (x j * g i - x i * g j)) / (∑ j, (x j) ^ 2) =
      g i - x i * (∑ j, x j * g j) / (∑ j, (x j) ^ 2) := by
  rw [pairwise_tangent_numerator]
  field_simp

/-- Exactly parallel gradients have zero tangent numerator before any division. -/
theorem parallel_pairwise_zero {d : ℕ} (x : Fin d → ℝ) (c : ℝ) (i : Fin d) :
    (∑ j, x j * (x j * (c * x i) - x i * (c * x j))) = 0 := by
  apply Finset.sum_eq_zero
  intro j _
  ring

/-- A nonzero one-dimensional input satisfies the tangent-equivalence hypothesis. -/
example : (∑ _ : Fin 1, (2 : ℝ) ^ 2) ≠ 0 := by norm_num

example :
    (∑ _ : Fin 1, (2 : ℝ) * (2 * 3 - 2 * 3)) / (∑ _ : Fin 1, (2 : ℝ) ^ 2) =
      3 - 2 * (∑ _ : Fin 1, (2 : ℝ) * 3) / (∑ _ : Fin 1, (2 : ℝ) ^ 2) := by
  exact pairwise_tangent_eq (fun _ => 2) (fun _ => 3) 0 (by norm_num)

example : Real.exp (-4) ≤ boundedFrequency 90 ∧ boundedFrequency 90 ≤ Real.exp 4 :=
  frequency_bounds 90

#print axioms TransformerStress.frequency_bounds
#print axioms TransformerStress.frequency_hasDerivAt
#print axioms TransformerStress.frequency_derivative_bound
#print axioms TransformerStress.initial_frequency_and_derivative
#print axioms TransformerStress.pairwise_tangent_numerator
#print axioms TransformerStress.pairwise_tangent_eq
#print axioms TransformerStress.parallel_pairwise_zero
#print axioms Transformer.Residual.identity_skip_zero_derivative
#print axioms Transformer.Residual.gain_derivative_ratio
#print axioms Transformer.MHC.stochastic_can_erase_disagreement
#print axioms Transformer.MHC.one_cycle_not_doubly_stochastic

end TransformerStress
