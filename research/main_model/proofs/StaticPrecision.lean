import Transformer.Precision.ArithmeticError

/-!
Fixed-format screening contracts, 2026-11-08.
The binary32 example matches the four-lane low-precision block in
precision_native/src/precision_kernels.rs: [1,2^-26,0,0] is reduced first,
then the block containing -1 is combined in a precise final accumulator.
These statements concern nearest rounding on the IEEE grid. The Rust screen
independently checks the implemented GPU result and its exact error budget.
-/
open Transformer.Precision
namespace MainModelStaticPrecision

/-- An IEEE binary32 partial sum absorbs this representable increment.
It is strictly below a half-ulp; either nearest tie convention gives the result. -/
theorem f32_absorbs_small (Q : ℝ → ℝ) (hQ : IsNearest (grid 8 23) Q) :
    Q (1 + 1 / (2 : ℝ)^26) = 1 := by
  have h1 : (1 : ℝ) ∈ grid 8 23 := by
    exact ⟨1065353216, by norm_num, by norm_num [ieee, bias]⟩
  apply hQ.add_eq h1
  · intro z hz hz1
    exact (grid_hasSignificand 8 23).gap h1 hz (s := -23) (by norm_num) hz1
  · positivity
  · norm_num

/-- RMS-bounded summands can still violate the registered backward diagnostic
budget after a binary32 partial sum and an EXACT final subtraction. Thus a
universal FP32-partial backward claim does not follow from normalization. -/
theorem f32_bounded_backward_counterexample (Q : ℝ → ℝ)
    (hQ : IsNearest (grid 8 23) Q) :
    (1 : ℝ) ∈ grid 8 23 ∧ (1 / 67108864 : ℝ) ∈ grid 8 23 ∧
    (-1 : ℝ) ∈ grid 8 23 ∧
    ((1 : ℝ)^2 + (1/67108864 : ℝ)^2 + (-1 : ℝ)^2) / 3 < 1 ∧
    Q (1 + 1/67108864) - 1 = 0 ∧
    |(Q (1 + 1/67108864) - 1) - 1/67108864| >
      |(1/67108864 : ℝ)|/2000 + 1/1000000000000 := by
  have h := f32_absorbs_small Q hQ
  simp only [show (2 : ℝ)^26 = 67108864 by norm_num] at h
  refine ⟨⟨1065353216, by norm_num, by norm_num [ieee, bias]⟩,
    ⟨847249408, by norm_num, by norm_num [ieee, bias]⟩,
    ⟨3212836864, by norm_num, by norm_num [ieee, bias]⟩,
    by norm_num, ?_, ?_⟩
  · rw [h]; ring
  · rw [h]; norm_num

/-- A positive static certificate: if an absolute error bound fits the fixed
gradient floor, the registered backward diagnostic budget is met for ANY
exact result, including exact cancellation. No runtime branch is needed. -/
theorem uniform_absolute_backward_certificate (exact approx bound : ℝ)
    (he : |approx - exact| ≤ bound) (hb : bound ≤ 1/1000000000000) :
    |approx - exact| ≤ |exact|/2000 + 1/1000000000000 := by
  have hn := abs_nonneg exact
  linarith

example : ∃ Q, IsNearest (grid 8 23) Q := exists_isNearest_grid (by norm_num) 23
example : |(1/2000000000000 : ℝ) - 0| ≤ |(0 : ℝ)|/2000 + 1/1000000000000 := by
  exact uniform_absolute_backward_certificate 0 (1/2000000000000)
    (1/2000000000000) (by norm_num) (by norm_num)

#print axioms f32_absorbs_small
#print axioms f32_bounded_backward_counterexample
#print axioms uniform_absolute_backward_certificate
end MainModelStaticPrecision
