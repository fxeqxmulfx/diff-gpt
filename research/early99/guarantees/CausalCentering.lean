import Transformer.GPTMini.RMSNorm

/-!
Causal adaptation of arXiv:2306.01610 Section 3.2 and arXiv:2410.07799v3
Eqs. (7)-(9), (15)-(18). At query n the visible prefix is indexed by Fin (n+1).
Subtract its uniform average AFTER softmax. This differs from a global rank-one
correction. We prove the algebraic identities needed by the implementation;
no asymptotic spectral, optimization, or floating-point theorem is asserted.
-/

namespace Early99.CSA

open Transformer
open scoped BigOperators

/-- Zero-sum candidate weights on the nonempty prefix visible to query n. -/
noncomputable def centeredWeight {n : ℕ} (weight : Fin (n + 1) → ℝ) (j : Fin (n + 1)) : ℝ :=
  weight j - 1 / ((n : ℝ) + 1)

/-- The new mixing stage before the existing per-head XSA projection. -/
noncomputable def mixing {n d : ℕ} (weight : Fin (n + 1) → ℝ)
    (value : Fin (n + 1) → EucSpace d) : EucSpace d :=
  ∑ j, centeredWeight weight j • value j

/-- Subtracting the uniform prefix weights makes each normalized row sum to zero. -/
theorem centered_row_sum {n : ℕ} (weight : Fin (n + 1) → ℝ)
    (hweight : ∑ j, weight j = 1) : ∑ j, centeredWeight weight j = 0 := by
  have hn : (n : ℝ) + 1 ≠ 0 := by positivity
  simp [centeredWeight, Finset.sum_sub_distrib, hweight, hn]

/-- The explicit centered matrix equals weighted values minus the prefix mean. -/
theorem mixing_eq_mean_difference {n d : ℕ} (weight : Fin (n + 1) → ℝ)
    (value : Fin (n + 1) → EucSpace d) :
    mixing weight value = (∑ j, weight j • value j) -
      (1 / ((n : ℝ) + 1)) • ∑ j, value j := by
  unfold mixing centeredWeight
  simp only [sub_smul, Finset.sum_sub_distrib, Finset.smul_sum]

/-- Constant value signals vanish before XSA, including tiny or zero values. -/
theorem mixing_constant {n d : ℕ} (weight : Fin (n + 1) → ℝ)
    (hweight : ∑ j, weight j = 1) (value : EucSpace d) :
    mixing weight (fun _ => value) = 0 := by
  unfold mixing
  rw [← Finset.sum_smul, centered_row_sum weight hweight]
  simp

/-- A common anchor cancels from zero-sum mixing. This justifies evaluating both
averages on value differences in the implementation, including cached prefixes. -/
theorem mixing_anchor_invariant {n d : ℕ} (weight : Fin (n + 1) → ℝ)
    (hweight : ∑ j, weight j = 1) (value : Fin (n + 1) → EucSpace d)
    (anchor : EucSpace d) :
    mixing weight (fun j => value j - anchor) = mixing weight value := by
  unfold mixing
  simp_rw [smul_sub]
  rw [Finset.sum_sub_distrib, ← Finset.sum_smul, centered_row_sum weight hweight]
  simp

/-- The mixing stage consumes only values in the visible prefix. This theorem
holds for fixed causal weights; causal generation of the weights is separate. -/
theorem mixing_prefix_congr {n d : ℕ} (weight : Fin (n + 1) → ℝ)
    (value other : ℕ → EucSpace d) (hprefix : ∀ j ≤ n, value j = other j) :
    mixing weight (fun j => value j.val) = mixing weight (fun j => other j.val) := by
  unfold mixing
  apply Finset.sum_congr rfl
  intro j hj
  change centeredWeight weight j • value j.val = centeredWeight weight j • other j.val
  rw [hprefix j.val (Nat.lt_succ_iff.mp j.isLt)]

/-- A normalized, nonuniform row for concrete non-vacuity checks. -/
noncomputable def witnessWeight : Fin 2 → ℝ := fun j => if j = 0 then 3 / 4 else 1 / 4

noncomputable def witnessValue : EucSpace 1 := EuclideanSpace.single (0 : Fin 1) 1

theorem witnessWeight_sum : ∑ j, witnessWeight j = 1 := by
  norm_num [witnessWeight, Fin.sum_univ_two]

example : ∑ j, centeredWeight (n := 1) witnessWeight j = 0 :=
  centered_row_sum (n := 1) witnessWeight witnessWeight_sum

example : mixing (n := 1) witnessWeight (fun _ => witnessValue) = 0 :=
  mixing_constant (n := 1) witnessWeight witnessWeight_sum witnessValue

example : mixing (n := 1) witnessWeight (fun j => (j.val + 1 : ℝ) • witnessValue - witnessValue) =
    mixing (n := 1) witnessWeight (fun j => (j.val + 1 : ℝ) • witnessValue) :=
  mixing_anchor_invariant (n := 1) witnessWeight witnessWeight_sum
    (fun j => (j.val + 1 : ℝ) • witnessValue) witnessValue

example : mixing (n := 1) witnessWeight (fun j => if j.val < 2 then witnessValue else 0) =
    mixing (n := 1) witnessWeight (fun j => if j.val < 2 then witnessValue else 2 • witnessValue) := by
  apply mixing_prefix_congr (n := 1) witnessWeight
    (fun j => if j < 2 then witnessValue else 0)
    (fun j => if j < 2 then witnessValue else 2 • witnessValue)
  intro j hj
  have hlt : j < 2 := by omega
  simp [hlt]

/-- The mixing map is not identically zero on nonconstant values. -/
example : mixing (n := 1) witnessWeight (fun j => if j = 0 then witnessValue else 3 • witnessValue) =
    (-1 / 2 : ℝ) • witnessValue := by
  unfold mixing
  rw [Fin.sum_univ_two]
  norm_num [centeredWeight, witnessWeight]
  module

#print axioms Early99.CSA.centered_row_sum
#print axioms Early99.CSA.mixing_eq_mean_difference
#print axioms Early99.CSA.mixing_constant
#print axioms Early99.CSA.mixing_anchor_invariant
#print axioms Early99.CSA.mixing_prefix_congr

end Early99.CSA
