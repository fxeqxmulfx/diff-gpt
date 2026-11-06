import Transformer.SequenceModel.Attention.Execution.Tables
import Transformer.GPTMini.RMSNorm

/-!
Stored fixed-RoPE coefficients need not lie exactly on the unit circle.
Source: src/diff_gpt/domain/model/rope.py, `apply_rope`, and the existing
Transformer.SequenceModel.Attention.Execution.tableRotate definition. The split
sum proof follows Execution/Tables.lean at transformer commit 90b1df2e, replacing
componentwise trigonometric approximation assumptions by a directly auditable
bound on each stored pair's exact squared magnitude. All arithmetic here is real;
rounding in the operations which apply these coefficients is outside the claim.
-/

namespace StoredRoPE
open Transformer Transformer.GPTMini Transformer.SequenceModel.Attention.Execution
open scoped BigOperators

/-- Pair energy bounds imply the whole half-split operator's squared norm bound.
M >= 1 also covers an unchanged leftover coordinate when d is odd. -/
theorem energy_bound {d : ℕ} (c s : Fin (d / 2) → ℝ) (M : ℝ)
    (hM : 1 ≤ M) (hpair : ∀ k, (c k)^2 + (s k)^2 ≤ M) (x : EucSpace d) :
    ‖tableRotate d c s x‖ ^ 2 ≤ M * ‖x‖ ^ 2 := by
  rw [EuclideanSpace.real_norm_sq_eq, EuclideanSpace.real_norm_sq_eq]
  rw [sum_split d (fun i => (tableRotate d c s x i)^2),
    sum_split d (fun i => (x i)^2)]
  have hf (k : Fin (d / 2)) :
      tableRotate d c s x (ropeFst d k) =
        c k * x (ropeFst d k) - s k * x (ropeSnd d k) := by
    simp only [tableRotate, ropeFst, PiLp.toLp_apply, Equiv.apply_symm_apply]
  have ht (k : Fin (d / 2)) :
      tableRotate d c s x (ropeSnd d k) =
        s k * x (ropeFst d k) + c k * x (ropeSnd d k) := by
    simp only [tableRotate, ropeSnd, PiLp.toLp_apply, Equiv.apply_symm_apply]
  have hl (j : Fin (d - 2 * (d / 2))) :
      tableRotate d c s x ((ropeSplit d).symm (.inr j)) =
        x ((ropeSplit d).symm (.inr j)) := by
    simp only [tableRotate, PiLp.toLp_apply, Equiv.apply_symm_apply]
  simp only [hf, ht, hl]
  rw [← Finset.sum_add_distrib, ← Finset.sum_add_distrib, mul_add,
    Finset.mul_sum, Finset.mul_sum]
  apply add_le_add
  · apply Finset.sum_le_sum
    intro k hk
    calc
      _ = ((c k)^2 + (s k)^2) * ((x (ropeFst d k))^2 + (x (ropeSnd d k))^2) := by ring
      _ ≤ M * ((x (ropeFst d k))^2 + (x (ropeSnd d k))^2) :=
        mul_le_mul_of_nonneg_right (hpair k) (by positivity)
  · apply Finset.sum_le_sum
    intro j hj
    simpa using mul_le_mul_of_nonneg_right hM (sq_nonneg (x ((ropeSplit d).symm (.inr j))))

/-- The stored-table norm multiplier is sqrt(M), without ideal phase assumptions. -/
theorem norm_bound {d : ℕ} (c s : Fin (d / 2) → ℝ) (M : ℝ)
    (hM : 1 ≤ M) (hpair : ∀ k, (c k)^2 + (s k)^2 ≤ M) (x : EucSpace d) :
    ‖tableRotate d c s x‖ ≤ Real.sqrt M * ‖x‖ := by
  apply (sq_le_sq₀ (norm_nonneg _) (by positivity : 0 ≤ Real.sqrt M * ‖x‖)).mp
  rw [mul_pow, Real.sq_sqrt (zero_le_one.trans hM)]
  exact energy_bound c s M hM hpair x

/-- Finite-epsilon Q/K RMS and two independently stored rows give score bound
abs(gain)*M*d. The two sqrt(M) norm factors multiply to M, not M squared. -/
theorem score_bound {d : ℕ} (eps gain M : ℝ) (heps : 0 < eps) (hd : 0 < d)
    (hM : 1 ≤ M) (cq sq ck sk : Fin (d / 2) → ℝ)
    (hq : ∀ k, (cq k)^2 + (sq k)^2 ≤ M)
    (hk : ∀ k, (ck k)^2 + (sk k)^2 ≤ M) (q k : EucSpace d) :
    |gain * inner ℝ (tableRotate d cq sq (rmsNormEps eps q))
      (tableRotate d ck sk (rmsNormEps eps k))| ≤ |gain| * M * (d : ℝ) := by
  have bound (c s : Fin (d / 2) → ℝ) (h : ∀ j, (c j)^2 + (s j)^2 ≤ M)
      (x : EucSpace d) :
      ‖tableRotate d c s (rmsNormEps eps x)‖ ≤ Real.sqrt M * Real.sqrt (d : ℝ) :=
    (norm_bound c s M hM h _).trans
      (mul_le_mul_of_nonneg_left (rmsNormEps_norm_le eps heps hd x) (Real.sqrt_nonneg _))
  have dot : |inner ℝ (tableRotate d cq sq (rmsNormEps eps q))
      (tableRotate d ck sk (rmsNormEps eps k))| ≤ M * (d : ℝ) := by
    calc
      _ ≤ ‖tableRotate d cq sq (rmsNormEps eps q)‖ *
          ‖tableRotate d ck sk (rmsNormEps eps k)‖ := abs_real_inner_le_norm _ _
      _ ≤ (Real.sqrt M * Real.sqrt (d : ℝ)) *
          (Real.sqrt M * Real.sqrt (d : ℝ)) :=
        mul_le_mul (bound cq sq hq q) (bound ck sk hk k) (norm_nonneg _) (by positivity)
      _ = (Real.sqrt M * Real.sqrt M) *
          (Real.sqrt (d : ℝ) * Real.sqrt (d : ℝ)) := by ring
      _ = M * (d : ℝ) := by
        rw [Real.mul_self_sqrt (zero_le_one.trans hM), Real.mul_self_sqrt (Nat.cast_nonneg d)]
  rw [abs_mul]
  simpa only [mul_assoc] using mul_le_mul_of_nonneg_left dot (abs_nonneg gain)

/-- SSMax multiplies the stored-coefficient score bound by log(N), N=n+1. -/
theorem ssmax_score_bound {d : ℕ} (eps gain M : ℝ) (heps : 0 < eps) (hd : 0 < d)
    (hM : 1 ≤ M) (n : ℕ) (cq sq ck sk : Fin (d / 2) → ℝ)
    (hq : ∀ k, (cq k)^2 + (sq k)^2 ≤ M)
    (hk : ∀ k, (ck k)^2 + (sk k)^2 ≤ M) (q k : EucSpace d) :
    |Real.log (n + 1) * (gain * inner ℝ (tableRotate d cq sq (rmsNormEps eps q))
      (tableRotate d ck sk (rmsNormEps eps k)))| ≤
      Real.log (n + 1) * (|gain| * M * (d : ℝ)) := by
  have hn : 0 ≤ Real.log (n + 1 : ℝ) :=
    Real.log_nonneg (by exact_mod_cast Nat.succ_le_succ (Nat.zero_le n))
  rw [abs_mul, abs_of_nonneg hn]
  exact mul_le_mul_of_nonneg_left (score_bound eps gain M heps hd hM cq sq ck sk hq hk q k) hn

-- Non-isometric stored coefficients and arbitrary vectors, including nonzero inputs.
example (x : EucSpace 5) : ‖tableRotate 5 (fun _ => 1) (fun _ => 1) x‖^2 ≤ 2 * ‖x‖^2 :=
  energy_bound (fun _ => 1) (fun _ => 1) 2 (by norm_num) (by intro j; norm_num) x
example (x : EucSpace 5) : ‖tableRotate 5 (fun _ => 1) (fun _ => 1) x‖ ≤ Real.sqrt 2 * ‖x‖ :=
  norm_bound (fun _ => 1) (fun _ => 1) 2 (by norm_num) (by intro j; norm_num) x
example (q k : EucSpace 4) :
    |(1 / 2 : ℝ) * inner ℝ (tableRotate 4 (fun _ => 1) (fun _ => 1) (rmsNormEps (1 / 8388608) q))
      (tableRotate 4 (fun _ => 1) (fun _ => 1) (rmsNormEps (1 / 8388608) k))| ≤ 4 := by
  convert score_bound (1 / 8388608) (1 / 2) 2 (by norm_num) (by norm_num) (by norm_num)
    (fun _ => 1) (fun _ => 1) (fun _ => 1) (fun _ => 1)
    (by intro j; norm_num) (by intro j; norm_num) q k using 1; norm_num
example (q k : EucSpace 4) :
    |Real.log 4 * ((1 / 2 : ℝ) * inner ℝ
      (tableRotate 4 (fun _ => 1) (fun _ => 1) (rmsNormEps (1 / 8388608) q))
      (tableRotate 4 (fun _ => 1) (fun _ => 1) (rmsNormEps (1 / 8388608) k)))| ≤ Real.log 4 * 4 := by
  convert ssmax_score_bound (1 / 8388608) (1 / 2) 2 (by norm_num) (by norm_num) (by norm_num) 3
    (fun _ => 1) (fun _ => 1) (fun _ => 1) (fun _ => 1)
    (by intro j; norm_num) (by intro j; norm_num) q k using 1 <;> norm_num

#print axioms energy_bound
#print axioms norm_bound
#print axioms score_bound
#print axioms ssmax_score_bound
end StoredRoPE
