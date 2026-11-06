import Transformer.Homogenized.SoftmaxDerivatives
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Analysis.Calculus.Deriv.Prod

/-!
Improve the checked softmax infinity-to-one Lipschitz constant from four to one.
For any subset of keys, its probability mass changes at speed at most M/2 when
each score changes at speed at most M. The positive-difference subset converts
this scalar estimate into the full l1 distance. This is a proof about the same
real softmax equation; no change to the model or floating-point claim is made.
-/

namespace Early99SoftmaxLipschitz
open Transformer Perspective
open scoped BigOperators

/-- Actual softmax derivative along an arbitrary affine path of logits. -/
theorem affine_derivative {n : ℕ} (hn : 0 < n) (u a : Fin n → ℝ) (j : Fin n) (t : ℝ) :
    HasDerivAt (fun r : ℝ => softmaxWeight (fun k => u k + r * a k) j)
      (softmaxWeight (fun k => u k + t * a k) j *
        (a j - ∑ k, softmaxWeight (fun l => u l + t * a l) k * a k)) t := by
  have hl : HasDerivAt (fun r : ℝ => fun k => u k + r * a k) a t := by
    apply hasDerivAt_pi.mpr
    intro k
    simpa only [id_eq, one_mul] using
      ((hasDerivAt_id t).mul_const (a k)).const_add (u k)
  have h := (Homogenized.hasFDerivAt_softmaxWeight hn
    (fun k => u k + t * a k) j).comp_hasDerivAt t hl
  simpa only [Function.comp_def, smul_apply, sub_apply, sum_apply,
    ContinuousLinearMap.proj_apply, smul_eq_mul] using h

/-- A subset's probability mass has derivative at most half the score-speed bound. -/
theorem subset_derivative_bound {n : ℕ} (hn : 0 < n) (u a : Fin n → ℝ)
    (M : ℝ) (hM : 0 ≤ M) (ha : ∀ j, |a j| ≤ M) (s : Finset (Fin n)) (t : ℝ) :
    |deriv (fun r : ℝ => ∑ j ∈ s, softmaxWeight (fun k => u k + r * a k) j) t| ≤ M / 2 := by
  classical
  let w := softmaxWeight (fun k => u k + t * a k)
  let c := Finset.univ \ s
  let p := ∑ j ∈ s, w j
  let q := ∑ j ∈ c, w j
  let A := ∑ j ∈ s, w j * a j
  let B := ∑ j ∈ c, w j * a j
  have hsum : q + p = 1 := by
    dsimp [p, q, c, w]
    rw [Finset.sum_sdiff (Finset.subset_univ s), sum_softmaxWeight hn]
  have hmean : B + A = ∑ j, w j * a j := by
    exact Finset.sum_sdiff (Finset.subset_univ s)
  have hp : 0 ≤ p := Finset.sum_nonneg fun j _ => softmaxWeight_nonneg _ j
  have hq : 0 ≤ q := Finset.sum_nonneg fun j _ => softmaxWeight_nonneg _ j
  have bounded (v : Finset (Fin n)) : |∑ j ∈ v, w j * a j| ≤ M * ∑ j ∈ v, w j := by
    calc
      _ ≤ ∑ j ∈ v, |w j * a j| := Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ j ∈ v, M * w j := by
        apply Finset.sum_le_sum
        intro j hj
        rw [abs_mul, abs_of_nonneg (softmaxWeight_nonneg _ j)]
        exact (mul_le_mul_of_nonneg_left (ha j) (softmaxWeight_nonneg _ j)).trans_eq (mul_comm _ _)
      _ = _ := (Finset.mul_sum _ _ _).symm
  have hA : |A| ≤ M * p := bounded s
  have hB : |B| ≤ M * q := bounded c
  have hderiv := HasDerivAt.fun_sum (u := s) fun j _ => affine_derivative hn u a j t
  rw [hderiv.deriv]
  change |∑ j ∈ s, w j * (a j - ∑ k, w k * a k)| ≤ _
  have heq : (∑ j ∈ s, w j * (a j - ∑ k, w k * a k)) = A * q - p * B := by
    simp_rw [mul_sub]
    rw [Finset.sum_sub_distrib, ← Finset.sum_mul, ← hmean]
    change A - p * (B + A) = _
    have hq_eq : q = 1 - p := by linarith
    rw [hq_eq]
    ring
  rw [heq]
  calc
    |A * q - p * B| ≤ |A * q| + |p * B| := abs_sub _ _
    _ = |A| * q + p * |B| := by rw [abs_mul, abs_mul, abs_of_nonneg hp, abs_of_nonneg hq]
    _ ≤ (M * p) * q + p * (M * q) :=
      add_le_add (mul_le_mul_of_nonneg_right hA hq) (mul_le_mul_of_nonneg_left hB hp)
    _ ≤ M / 2 := by
      have hsq := congrArg (fun z : ℝ => z ^ 2) hsum
      have hpq : 4 * p * q ≤ 1 := by nlinarith [sq_nonneg (p - q)]
      have hprod := mul_le_mul_of_nonneg_left hpq hM
      nlinarith [hprod]

/-- Any key subset's probability mass is 1/2-Lipschitz in logit sup distance. -/
theorem subset_lipschitz {n : ℕ} (hn : 0 < n) (u v : Fin n → ℝ)
    (M : ℝ) (hM : 0 ≤ M) (huv : ∀ j, |u j - v j| ≤ M) (s : Finset (Fin n)) :
    |(∑ j ∈ s, softmaxWeight u j) - ∑ j ∈ s, softmaxWeight v j| ≤ M / 2 := by
  let f := fun r : ℝ => ∑ j ∈ s, softmaxWeight (fun k => v k + r * (u k - v k)) j
  have hlip : LipschitzWith ⟨M / 2, by positivity⟩ f := by
    apply lipschitzWith_of_nnnorm_deriv_le
    · intro t
      exact (HasDerivAt.fun_sum (u := s) fun j _ =>
        affine_derivative hn v (fun k => u k - v k) j t).differentiableAt
    · intro t
      change |deriv f t| ≤ M / 2
      exact subset_derivative_bound hn v (fun k => u k - v k) M hM huv s t
  have h := hlip.norm_sub_le 1 0
  change ‖f 1 - f 0‖ ≤ (M / 2) * ‖(1 : ℝ) - 0‖ at h
  convert! h using 1 <;> simp [f, Real.norm_eq_abs]

/-- Complete real softmax has infinity-to-one Lipschitz constant one. -/
theorem softmax_l1_lipschitz {n : ℕ} (hn : 0 < n) (u v : Fin n → ℝ)
    (M : ℝ) (hM : 0 ≤ M) (huv : ∀ j, |u j - v j| ≤ M) :
    ∑ j, |softmaxWeight u j - softmaxWeight v j| ≤ M := by
  classical
  let f := fun j => softmaxWeight u j - softmaxWeight v j
  let s := Finset.univ.filter (fun j => 0 ≤ f j)
  have hzero : ∑ j, f j = 0 := by
    dsimp [f]
    rw [Finset.sum_sub_distrib, sum_softmaxWeight hn, sum_softmaxWeight hn, sub_self]
  have habs : ∀ j, |f j| = 2 * (if 0 ≤ f j then f j else 0) - f j := by
    intro j
    split_ifs with h
    · rw [abs_of_nonneg h]; ring
    · rw [abs_of_neg (lt_of_not_ge h)]; ring
  have heq : (∑ j, |f j|) = 2 * ∑ j ∈ s, f j := by
    simp_rw [habs]
    rw [Finset.sum_sub_distrib, ← Finset.mul_sum, hzero, sub_zero]
    congr 1
    exact (Finset.sum_filter _ _).symm
  change (∑ j, |f j|) ≤ M
  rw [heq]
  have hs := subset_lipschitz hn u v M hM huv s
  have hs' : |∑ j ∈ s, f j| ≤ M / 2 := by
    simpa only [f, Finset.sum_sub_distrib] using hs
  linarith [le_abs_self (∑ j ∈ s, f j)]

-- Nonempty rows, nonzero score changes, and a genuine proper subset.
example (u a : Fin 2 → ℝ) (j : Fin 2) (t : ℝ) :
    HasDerivAt (fun r : ℝ => softmaxWeight (fun k => u k + r * a k) j)
      (softmaxWeight (fun k => u k + t * a k) j *
        (a j - ∑ k, softmaxWeight (fun l => u l + t * a l) k * a k)) t :=
  affine_derivative (by norm_num) u a j t

example (u : Fin 2 → ℝ) (t : ℝ) :
    |deriv (fun r : ℝ => ∑ j ∈ ({0} : Finset (Fin 2)),
      softmaxWeight (fun k => u k + r * (if k = 0 then 1 else -1)) j) t| ≤ 1 / 2 := by
  apply subset_derivative_bound (by norm_num) u _ 1 (by norm_num) _ {0} t
  intro j; split_ifs <;> norm_num

example : |(∑ j ∈ ({0} : Finset (Fin 2)), softmaxWeight (fun k => if k = 0 then 1 else -1) j) -
    ∑ j ∈ ({0} : Finset (Fin 2)), softmaxWeight (fun _ => 0) j| ≤ 1 / 2 := by
  apply subset_lipschitz (by norm_num) _ _ 1 (by norm_num)
  intro j; split_ifs <;> norm_num

example : ∑ j : Fin 2, |softmaxWeight (fun k => if k = 0 then 1 else -1) j -
    softmaxWeight (fun _ => 0) j| ≤ 1 := by
  apply softmax_l1_lipschitz (by norm_num) _ _ 1 (by norm_num)
  intro j; split_ifs <;> norm_num

#print axioms affine_derivative
#print axioms subset_derivative_bound
#print axioms subset_lipschitz
#print axioms softmax_l1_lipschitz
end Early99SoftmaxLipschitz
