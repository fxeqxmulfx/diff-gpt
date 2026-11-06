import research.normalization_guarantees.AttentionSelectivity
import research.normalization_guarantees.AttentionSensitivityImproved

/-!
Specialize the checked finite-epsilon Q/K bounds to the declared inverse-width
gain (plan.json, equation and guarantees). The coefficient is exactly 1/d in
real arithmetic. The separate numerical study checks the actual stored buffer.
These are score/probability bounds, not convergence or whole-network theorems.
-/

namespace Early99DimensionGain
open Transformer Transformer.Perspective Early99.NormalizedScores
open Early99AttentionSensitivity
open scoped BigOperators

/-- The declared log(N)/d coefficient gives a width-independent log(N) envelope. -/
theorem score_bound {d : ℕ} (eps : ℝ) (heps : 0 < eps) (hd : 0 < d)
    (n : ℕ) (a b : Fin (d / 2) → ℝ) (q k : EucSpace d) :
    |Real.log (n + 1) * ((1 / (d : ℝ)) *
      inner ℝ (rotatedQK eps a q) (rotatedQK eps b k))| ≤ Real.log (n + 1) := by
  have hd' : (0 : ℝ) < d := by exact_mod_cast hd
  have h := ssmax_score eps (1 / (d : ℝ)) heps hd n a b q k
  simpa [abs_of_pos (div_pos zero_lt_one hd'), ne_of_gt hd'] using h

/-- The corresponding selectivity ceiling is 1/(1+(N-1)/N^2), at every width.
This is only an upper bound; it does not guarantee a winning margin. -/
theorem probability_ceiling {d n : ℕ} (eps : ℝ) (heps : 0 < eps) (hd : 0 < d)
    (a : Fin (d / 2) → ℝ) (b : Fin (n + 1) → Fin (d / 2) → ℝ)
    (q : EucSpace d) (keys : Fin (n + 1) → EucSpace d) (i : Fin (n + 1)) :
    softmaxWeight (fun j => Real.log (n + 1) * ((1 / (d : ℝ)) *
      inner ℝ (rotatedQK eps a q) (rotatedQK eps (b j) (keys j)))) i ≤
      1 / (1 + (n : ℝ) / ((n : ℝ) + 1)^2) := by
  have h := Early99AttentionSelectivity.softmax_bounded_ceiling
    (fun j => Real.log (n + 1) * ((1 / (d : ℝ)) *
      inner ℝ (rotatedQK eps a q) (rotatedQK eps (b j) (keys j)))) i
    (Real.log (n + 1)) (fun j => score_bound eps heps hd n a (b j) q (keys j))
  have he : Real.exp (2 * Real.log ((n : ℝ) + 1)) = ((n : ℝ) + 1)^2 := by
    simpa using (Real.exp_nat_mul (Real.log ((n : ℝ) + 1)) 2).trans
      (by rw [Real.exp_log (by positivity)])
  simpa [Real.exp_neg, he, div_eq_mul_inv] using h

/-- Algebraic specialization of the existing RMS/softmax sensitivity constant. -/
theorem inverse_width_coefficient {d : ℕ} (eps c : ℝ)
    (heps : 0 < eps) (hd : 0 < d) (hc : 0 ≤ c) :
    (|c / (d : ℝ)| * Real.sqrt (d : ℝ) / Real.sqrt eps) =
      c / (Real.sqrt (d : ℝ) * Real.sqrt eps) := by
  have hd' : (0 : ℝ) < d := by exact_mod_cast hd
  have hs : 0 < Real.sqrt (d : ℝ) := Real.sqrt_pos.mpr hd'
  have he : 0 < Real.sqrt eps := Real.sqrt_pos.mpr heps
  rw [abs_of_nonneg (div_nonneg hc hd'.le)]
  field_simp [ne_of_gt hd', ne_of_gt hs, ne_of_gt he]
  nlinarith [congrArg (fun x : ℝ => c * x) (Real.mul_self_sqrt hd'.le)]

/-- With fixed keys and phases, the complete row has coefficient
log(N)/(sqrt(d)*sqrt(eps)) with no additional visible-key factor. -/
theorem query_sensitivity {d n : ℕ} (eps : ℝ) (heps : 0 < eps) (hd : 0 < d)
    (a : Fin (d / 2) → ℝ) (b : Fin (n + 1) → Fin (d / 2) → ℝ)
    (keys : Fin (n + 1) → EucSpace d) (q q' : EucSpace d) :
    ∑ j : Fin (n + 1), |softmaxWeight
        (fun l => (Real.log (n + 1) / (d : ℝ)) *
          inner ℝ (normalized eps a q) (normalized eps (b l) (keys l))) j -
      softmaxWeight (fun l => (Real.log (n + 1) / (d : ℝ)) *
          inner ℝ (normalized eps a q') (normalized eps (b l) (keys l))) j| ≤
      (Real.log (n + 1) / (Real.sqrt (d : ℝ) * Real.sqrt eps)) * ‖q - q'‖ := by
  have h := Early99AttentionSensitivityImproved.query_lipschitz
    eps (Real.log (n + 1) / (d : ℝ)) heps hd (Nat.zero_lt_succ n) a b keys q q'
  have hn : 0 ≤ Real.log (n + 1 : ℝ) :=
    Real.log_nonneg (by exact_mod_cast Nat.succ_le_succ (Nat.zero_le n))
  rwa [inverse_width_coefficient eps (Real.log (n + 1)) heps hd hn] at h

-- Actual positive FP32 epsilon and nonempty rows, with unrestricted vectors.
example (q k : EucSpace 32) := score_bound (1 / 8388608)
  (by norm_num) (by norm_num) 63 (fun _ => 2) (fun _ => 3) q k

example (q : EucSpace 32) (keys : Fin 64 → EucSpace 32) (i : Fin 64) :=
  probability_ceiling (1 / 8388608) (by norm_num) (by norm_num)
    (fun _ => 2) (fun j k => (j.val + k.val : ℕ)) q keys i

example : (|Real.log 64 / (32 : ℝ)| * Real.sqrt 32 / Real.sqrt (1 / 8388608)) =
    Real.log 64 / (Real.sqrt 32 * Real.sqrt (1 / 8388608)) :=
  inverse_width_coefficient (d := 32) (1 / 8388608) (Real.log 64)
    (by norm_num) (by norm_num) (Real.log_nonneg (by norm_num))

example (keys : Fin 64 → EucSpace 32) (q q' : EucSpace 32) :=
  query_sensitivity (1 / 8388608) (by norm_num) (by norm_num)
    (fun _ => 2) (fun j k => (j.val + k.val : ℕ)) keys q q'

#print axioms score_bound
#print axioms probability_ceiling
#print axioms inverse_width_coefficient
#print axioms query_sensitivity
end Early99DimensionGain
