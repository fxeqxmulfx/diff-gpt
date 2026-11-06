import research.stored_rope_guarantees.Bounds
import Transformer.Perspective.Softmax

/-!
Real-arithmetic consequences of the fixed inverse-width SSMax implementation.
Sources: research/early99_dimension_gain/factories.py and
research/early99/experiments/scalable_softmax.py. The stored-table score theorem
is the previously checked research/stored_rope_guarantees/Bounds.lean.
This bounds a single two-token head's probabilities; it does not prove a training
failure or any limitation of the complete multihead residual network.
-/

namespace TwoTokenAttention
open Transformer Transformer.GPTMini Transformer.Perspective
open Transformer.SequenceModel.Attention.Execution

/-- The existing two-score softmax equals the logistic function of their gap. -/
theorem binary_gap (a b : ℝ) :
    softmaxWeight ![a, b] 0 = 1 / (1 + Real.exp (b - a)) := by
  simp only [softmaxWeight, Fin.sum_univ_two, Matrix.cons_val_zero,
    Matrix.cons_val_one]
  rw [Real.exp_sub]
  have ha := (Real.exp_pos a).ne'
  have hab : Real.exp a + Real.exp b ≠ 0 := by positivity
  field_simp

/-- Scores bounded in magnitude by B impose an explicit two-key probability
interval. This covers either coordinate by swapping a and b. -/
theorem binary_bounds (a b B : ℝ) (ha : |a| ≤ B) (hb : |b| ≤ B) :
    1 / (1 + Real.exp (2 * B)) ≤ softmaxWeight ![a, b] 0 ∧
    softmaxWeight ![a, b] 0 ≤ Real.exp (2 * B) / (1 + Real.exp (2 * B)) := by
  have hab : a ≤ 2 * B + b := by
    linarith [(abs_le.mp ha).2, (abs_le.mp hb).1]
  have hba : b ≤ 2 * B + a := by
    linarith [(abs_le.mp hb).2, (abs_le.mp ha).1]
  have hea : Real.exp a ≤ Real.exp (2 * B) * Real.exp b := by
    rw [← Real.exp_add]
    exact Real.exp_le_exp.mpr hab
  have heb : Real.exp b ≤ Real.exp (2 * B) * Real.exp a := by
    rw [← Real.exp_add]
    exact Real.exp_le_exp.mpr hba
  simp only [softmaxWeight, Fin.sum_univ_two, Matrix.cons_val_zero,
    Matrix.cons_val_one]
  constructor
  · apply (div_le_div_iff₀ (by positivity) (by positivity)).mpr
    nlinarith
  · apply (div_le_div_iff₀ (by positivity) (by positivity)).mpr
    nlinarith

/-- Ideal normalized inverse-width SSMax has |score| <= log(2) for two visible
keys, so no head can assign more than 4/5 or less than 1/5 to either key. -/
theorem inverse_two_key_bounds (a b : ℝ)
    (ha : |a| ≤ Real.log 2) (hb : |b| ≤ Real.log 2) :
    (1 / 5 : ℝ) ≤ softmaxWeight ![a, b] 0 ∧
    softmaxWeight ![a, b] 0 ≤ (4 / 5 : ℝ) := by
  have he : Real.exp (2 * Real.log 2) = 4 := by
    rw [show (2 : ℝ) * Real.log 2 = Real.log 2 + Real.log 2 by ring,
      Real.exp_add, Real.exp_log (by norm_num : (0 : ℝ) < 2)]
    norm_num
  convert binary_bounds a b (Real.log 2) ha hb using 1 <;> norm_num [he]

/-- Finite-epsilon Q/K RMS with gain 1/d cancels head width in the SSMax bound.
For stored RoPE coefficient energy <= M, every score is bounded by log(N)*M.
The residual scale, depth and number of heads are absent from this per-head bound. -/
theorem inverse_stored_score_bound {d : ℕ} (eps M : ℝ)
    (heps : 0 < eps) (hd : 0 < d) (hM : 1 ≤ M) (n : ℕ)
    (cq sq ck sk : Fin (d / 2) → ℝ)
    (hq : ∀ k, (cq k)^2 + (sq k)^2 ≤ M)
    (hk : ∀ k, (ck k)^2 + (sk k)^2 ≤ M) (q k : EucSpace d) :
    |Real.log (n + 1) * ((1 / (d : ℝ)) *
      inner ℝ (tableRotate d cq sq (rmsNormEps eps q))
        (tableRotate d ck sk (rmsNormEps eps k)))| ≤ Real.log (n + 1) * M := by
  have hd' : (0 : ℝ) < d := by exact_mod_cast hd
  have hg : |1 / (d : ℝ)| * M * (d : ℝ) = M := by
    rw [abs_of_pos (by positivity : 0 < 1 / (d : ℝ))]
    field_simp
  simpa only [hg] using
    StoredRoPE.ssmax_score_bound eps (1 / (d : ℝ)) M heps hd hM n cq sq ck sk hq hk q k

-- Nonempty hypotheses, including asymmetric nonzero score and vector examples.
example : softmaxWeight ![(1 : ℝ), -1] 0 = 1 / (1 + Real.exp (-1 - 1)) :=
  binary_gap 1 (-1)
example :
    1 / (1 + Real.exp (2 * 3)) ≤ softmaxWeight ![(1 : ℝ), -2] 0 ∧
    softmaxWeight ![(1 : ℝ), -2] 0 ≤ Real.exp (2 * 3) / (1 + Real.exp (2 * 3)) :=
  binary_bounds 1 (-2) 3 (by norm_num) (by norm_num)
example : (1 / 5 : ℝ) ≤ softmaxWeight ![Real.log 2, -Real.log 2] 0 ∧
    softmaxWeight ![Real.log 2, -Real.log 2] 0 ≤ (4 / 5 : ℝ) :=
  inverse_two_key_bounds _ _
    (by rw [abs_of_nonneg (Real.log_nonneg (by norm_num))])
    (by rw [abs_neg, abs_of_nonneg (Real.log_nonneg (by norm_num))])
example (q k : EucSpace 32) :
    |Real.log 2 * ((1 / 32 : ℝ) *
      inner ℝ (tableRotate 32 (fun _ => 1) (fun _ => 0) (rmsNormEps (1 / 8388608) q))
        (tableRotate 32 (fun _ => 1) (fun _ => 0) (rmsNormEps (1 / 8388608) k)))|
      ≤ Real.log 2 * 1 := by
  convert inverse_stored_score_bound (1 / 8388608) 1 (by norm_num)
    (by norm_num : 0 < 32) (by norm_num) 1
    (fun _ => 1) (fun _ => 0) (fun _ => 1) (fun _ => 0)
    (by intro j; norm_num) (by intro j; norm_num) q k using 1 <;> norm_num

#print axioms binary_gap
#print axioms binary_bounds
#print axioms inverse_two_key_bounds
#print axioms inverse_stored_score_bound
end TwoTokenAttention
