import research.early99.guarantees.NormalizedScores
import Transformer.Perspective.Softmax

/-!
A complementary limitation of the score bound: a bounded gap also bounds how
much probability any one key can receive. This sharpens the elementary
exp(2B)/N estimate by retaining the selected key in the denominator. Applied
to the actual Q/K RMS and rotary equations, it exposes a width/SSMax-gain
tradeoff. It is a real-arithmetic statement about one probability row, not
an impossibility theorem for the full transformer, XSA or its learned readout.
-/

namespace Early99AttentionSelectivity
open Transformer Transformer.SequenceModel Transformer.Perspective
open Early99.NormalizedScores
open scoped BigOperators

/-- If no competitor is more than G below a key, that key has a finite ceiling.
This is the upper-bound counterpart of the existing SSMax margin lower bound. -/
theorem softmax_gap_ceiling {n : ℕ} (z : Fin (n + 1) → ℝ)
    (i : Fin (n + 1)) (G : ℝ)
    (hgap : ∀ j : Fin n, z i - z (i.succAbove j) ≤ G) :
    softmaxWeight z i ≤ 1 / (1 + (n : ℝ) * Real.exp (-G)) := by
  have each (j : Fin n) : Real.exp (z i) * Real.exp (-G) ≤
      Real.exp (z (i.succAbove j)) := by
    rw [← Real.exp_add]
    apply Real.exp_le_exp.mpr
    linarith [hgap j]
  have total : Real.exp (z i) * (1 + (n : ℝ) * Real.exp (-G)) ≤
      ∑ j : Fin (n + 1), Real.exp (z j) := by
    rw [Fin.sum_univ_succAbove _ i]
    have hs := Finset.sum_le_sum (fun (j : Fin n) (_ : j ∈ Finset.univ) => each j)
    simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] at hs
    nlinarith
  unfold softmaxWeight
  apply (div_le_div_iff₀ (softmaxPartition_pos (Nat.zero_lt_succ n) _)
    (by positivity)).mpr
  simpa only [one_mul] using total

/-- A uniform absolute score bound supplies the largest possible pairwise gap. -/
theorem softmax_bounded_ceiling {n : ℕ} (z : Fin (n + 1) → ℝ)
    (i : Fin (n + 1)) (B : ℝ) (hB : ∀ j, |z j| ≤ B) :
    softmaxWeight z i ≤ 1 / (1 + (n : ℝ) * Real.exp (-(2 * B))) := by
  apply softmax_gap_ceiling z i (2 * B)
  intro j
  have hi := (abs_le.mp (hB i)).2
  have hj := (abs_le.mp (hB (i.succAbove j))).1
  linarith

/-- The ceiling for finite-epsilon Q/K RMS, arbitrary phases, and fixed SSMax m.
The gain can be the actual rounded buffer; no ideal 1/sqrt(d) equality is assumed. -/
theorem ssmax_normalized_ceiling {d n : ℕ} (eps gain m : ℝ)
    (heps : 0 < eps) (hd : 0 < d) (hm : 0 ≤ m)
    (a : Fin (d / 2) → ℝ) (b : Fin (n + 1) → Fin (d / 2) → ℝ)
    (q : EucSpace d) (keys : Fin (n + 1) → EucSpace d) (i : Fin (n + 1)) :
    softmaxWeight (fun j => (m * Real.log (n + 1)) *
      (gain * inner ℝ (rotatedQK eps a q) (rotatedQK eps (b j) (keys j)))) i ≤
      1 / (1 + (n : ℝ) * Real.exp (-(2 *
        ((m * Real.log (n + 1)) * (|gain| * (d : ℝ)))))) := by
  apply softmax_bounded_ceiling
  intro j
  have hn : 0 ≤ Real.log (n + 1 : ℝ) :=
    Real.log_nonneg (by exact_mod_cast Nat.succ_le_succ (Nat.zero_le n))
  rw [abs_mul, abs_of_nonneg (mul_nonneg hm hn)]
  exact mul_le_mul_of_nonneg_left
    (normalized_score eps gain heps hd a (b j) q (keys j)) (mul_nonneg hm hn)

/-- At width four, m=1/4 and gain=1/2 impose the ceiling N/(2N-1).
This holds for all projected queries/keys and phases, with positive epsilon. -/
theorem quarter_width_four_ceiling {n : ℕ} (eps : ℝ) (heps : 0 < eps)
    (a : Fin 2 → ℝ) (b : Fin (n + 1) → Fin 2 → ℝ)
    (q : EucSpace 4) (keys : Fin (n + 1) → EucSpace 4) (i : Fin (n + 1)) :
    softmaxWeight (fun j => ((1 / 4 : ℝ) * Real.log (n + 1)) *
      ((1 / 2 : ℝ) * inner ℝ (rotatedQK eps a q) (rotatedQK eps (b j) (keys j)))) i ≤
      ((n : ℝ) + 1) / (2 * (n : ℝ) + 1) := by
  have h := ssmax_normalized_ceiling (d := 4) eps (1 / 2) (1 / 4)
    heps (by norm_num) (by norm_num) a b q keys i
  have harg : -(2 * (((1 / 4 : ℝ) * Real.log (n + 1)) *
      (|(1 / 2 : ℝ)| * (4 : ℕ)))) =
      -Real.log (n + 1) := by norm_num; ring
  rw [harg, Real.exp_neg, Real.exp_log (by positivity)] at h
  convert h using 1
  have hn : (n : ℝ) + 1 ≠ 0 := by positivity
  have hd : 2 * (n : ℝ) + 1 ≠ 0 := by positivity
  field_simp [hn, hd]
  ring

-- Unequal scores and actual positive FP32 epsilon make the hypotheses concrete.
example : softmaxWeight (![1, -1] : Fin 2 → ℝ) 0 ≤
    1 / (1 + Real.exp (-2)) := by
  convert softmax_gap_ceiling (![1, -1] : Fin 2 → ℝ) 0 2 (by
    intro j; fin_cases j; norm_num) using 1; norm_num

example (i : Fin 2) : softmaxWeight (![1, -1] : Fin 2 → ℝ) i ≤
    1 / (1 + Real.exp (-2)) := by
  convert softmax_bounded_ceiling (![1, -1] : Fin 2 → ℝ) i 1 (by
    intro j; fin_cases j <;> norm_num) using 1; norm_num

example (q : EucSpace 8) (keys : Fin 4 → EucSpace 8) (i : Fin 4) :=
  ssmax_normalized_ceiling (n := 3) (1 / 8388608) (1 / Real.sqrt 8) (1 / 4)
    (by norm_num) (by norm_num) (by norm_num) (fun _ => 2)
    (fun j k => ((j.val + k.val : ℕ) : ℝ)) q keys i

example (q : EucSpace 4) (keys : Fin 64 → EucSpace 4) (i : Fin 64) :
    softmaxWeight (fun j => ((1 / 4 : ℝ) * Real.log 64) *
      ((1 / 2 : ℝ) * inner ℝ (rotatedQK (1 / 8388608) (fun _ => 2) q)
        (rotatedQK (1 / 8388608) (fun k => (j.val + k.val : ℕ)) (keys j)))) i ≤
      (64 / 127 : ℝ) := by
  convert quarter_width_four_ceiling (n := 63) (1 / 8388608) (by norm_num)
    (fun _ => 2) (fun j k => (j.val + k.val : ℕ)) q keys i using 1 <;> norm_num

#print axioms softmax_gap_ceiling
#print axioms softmax_bounded_ceiling
#print axioms ssmax_normalized_ceiling
#print axioms quarter_width_four_ceiling
end Early99AttentionSelectivity
