import Transformer.Perspective.Softmax

/-!
Finite-row guarantees for arXiv:2501.19399v1, Eqs. (2) and (9).
We prove the exponential form of the margin bound; with beta=s*log(N),
exp(-beta*gap)=N^(-s*gap). No uniform positive logit gap is assumed implicitly.
The target key is arbitrary (Fin.succAbove enumerates all other visible keys).
-/
namespace Early99.ScalableSoftmax
open scoped BigOperators
open Transformer.Perspective

noncomputable def weight {n : ℕ} (s : ℝ) (z : Fin (n + 1) → ℝ)
    (i : Fin (n + 1)) : ℝ :=
  softmaxWeight (fun j => (s * Real.log (n + 1)) * z j) i

/-- SSMax retains the normalized, nonnegative attention contract for any real s. -/
theorem probability_row {n : ℕ} (s : ℝ) (z : Fin (n + 1) → ℝ) :
    (∑ i, weight s z i) = 1 ∧ ∀ i, 0 ≤ weight s z i := by
  constructor
  · exact sum_softmaxWeight (Nat.zero_lt_succ n) _
  · intro i
    exact softmaxWeight_nonneg _ i

/-- Explicit finite logit margin implies a lower bound for the chosen key.
Source: arXiv:2501.19399v1 Eq. (9), generalized to any nonnegative temperature.
The hypothesis quantifies over every other key; a large maximum alone is insufficient. -/
theorem softmax_margin {n : ℕ} (z : Fin (n + 1) → ℝ)
    (i : Fin (n + 1)) (beta gap : ℝ) (hb : 0 ≤ beta)
    (hgap : ∀ j : Fin n, z (i.succAbove j) ≤ z i - gap) :
    1 / (1 + (n : ℝ) * Real.exp (-beta * gap)) ≤
      softmaxWeight (fun j => beta * z j) i := by
  have each (j : Fin n) : Real.exp (beta * z (i.succAbove j)) ≤
      Real.exp (beta * z i) * Real.exp (-beta * gap) := by
    rw [← Real.exp_add]
    apply Real.exp_le_exp.mpr
    have := mul_le_mul_of_nonneg_left (hgap j) hb
    nlinarith
  have total : (∑ j : Fin (n + 1), Real.exp (beta * z j)) ≤
      Real.exp (beta * z i) * (1 + (n : ℝ) * Real.exp (-beta * gap)) := by
    rw [Fin.sum_univ_succAbove _ i]
    have hs := Finset.sum_le_sum (fun (j : Fin n) (_ : j ∈ Finset.univ) => each j)
    simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] at hs
    nlinarith
  unfold softmaxWeight
  apply (div_le_div_iff₀ (by positivity)
    (softmaxPartition_pos (Nat.zero_lt_succ n) _)).mpr
  simpa only [one_mul] using total

/-- The implemented s=1 case and general s>=0 satisfy the finite SSMax bound. -/
theorem ssmax_margin {n : ℕ} (s : ℝ) (hs : 0 ≤ s)
    (z : Fin (n + 1) → ℝ) (i : Fin (n + 1)) (gap : ℝ)
    (hgap : ∀ j : Fin n, z (i.succAbove j) ≤ z i - gap) :
    1 / (1 + (n : ℝ) * Real.exp (-(s * Real.log (n + 1)) * gap)) ≤
      weight s z i := by
  apply softmax_margin z i (s * Real.log (n + 1)) gap
  · exact mul_nonneg hs (Real.log_nonneg (by exact_mod_cast Nat.succ_le_succ (Nat.zero_le n)))
  · exact hgap

/-- Equal logits remain uniform, including N=1; concentration is conditional. -/
theorem constant_logits {n : ℕ} (s c : ℝ) (i : Fin (n + 1)) :
    weight s (fun _ => c) i = 1 / (n + 1) := by
  unfold weight softmaxWeight
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul,
    Nat.cast_add, Nat.cast_one]
  have he := Real.exp_ne_zero ((s * Real.log (n + 1)) * c)
  field_simp

-- Nonvacuous witnesses: two unequal logits and a strictly positive margin.
def witness : Fin 2 → ℝ := ![2, 0]
example : ∀ j : Fin 1, witness ((0 : Fin 2).succAbove j) ≤ witness 0 - 2 := by
  intro j
  fin_cases j
  norm_num [witness]
example : 1 / (1 + Real.exp (-(Real.log 2) * 2)) ≤ weight 1 witness 0 := by
  convert ssmax_margin 1 (by norm_num) witness 0 2 (by
    intro j
    fin_cases j
    norm_num [witness]) using 1; norm_num
example : weight (n := 0) 1 (fun _ => 7) 0 = 1 := by
  simpa using constant_logits (n := 0) 1 7 0
example : (∑ i, weight 1 witness i) = 1 ∧ ∀ i, 0 ≤ weight 1 witness i :=
  probability_row 1 witness

#print axioms probability_row
#print axioms softmax_margin
#print axioms ssmax_margin
#print axioms constant_logits
end Early99.ScalableSoftmax
