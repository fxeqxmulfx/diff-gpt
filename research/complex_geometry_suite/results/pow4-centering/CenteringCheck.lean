import CausalCentering
import Mathlib.Analysis.Calculus.FDeriv.Const
import Mathlib.Data.ZMod.Basic
import Mathlib.LinearAlgebra.Matrix.Determinant.Basic

/-!
Exact-algebra review of the causal adaptation of arXiv:2410.07799v3,
Sections 2.4 and 3.2, equations (7)-(9) and (15)-(18).

The paper subtracts the global uniform matrix. Here each causal row subtracts
the uniform distribution over its own visible prefix, as in the earlier local
CausalCentering.lean fixture. The results below prove cancellation and its
parameter-gradient consequence, not the paper's random-matrix asymptotics,
floating-point correctness, AdamW convergence, or full-model impossibility.

The model connection assumes shared token/value maps and Q/K-only positional
rotation: equal tokens then have equal values. Output projection is linear,
RMSNorm is parameter-free, and residual scaling follows RMSNorm. A downstream
loss theorem holds with the residual input and downstream parameters fixed.
-/

namespace QuarticCenteringReview

open Transformer Early99.CSA
open scoped BigOperators

/-- Ordinary normalized attention preserves a common value. -/
theorem ordinary_constant {n d : ℕ} (p : Fin (n + 1) → ℝ)
    (hp : ∑ j, p j = 1) (v : EucSpace d) : (∑ j, p j • v) = v := by
  rw [← Finset.sum_smul, hp, one_smul]

/-- Causal centering also suppresses the one-token prefix. -/
theorem first_prefix_zero {d : ℕ} (p : Fin 1 → ℝ) (hp : ∑ j, p j = 1)
    (v : Fin 1 → EucSpace d) : mixing (n := 0) p v = 0 := by
  have hp0 : p 0 = 1 := by simpa using hp
  simp [mixing, centeredWeight, hp0]

/-- At the supervised second position the branch keeps only a value difference.
This is independent of how normalized p was obtained (softmax, SSMax, QK/RoPE). -/
theorem two_token_difference {d : ℕ} (p : Fin 2 → ℝ) (hp : ∑ j, p j = 1)
    (v : Fin 2 → EucSpace d) :
    mixing (n := 1) p v = (p 0 - 1 / 2) • (v 0 - v 1) := by
  have hp1 : p 1 = 1 - p 0 := by
    rw [Fin.sum_univ_two] at hp
    linarith
  unfold mixing
  rw [Fin.sum_univ_two]
  norm_num [centeredWeight, hp1]
  module

/-- Uniform attention is killed even when the values differ. -/
theorem uniform_row_zero {n d : ℕ} (v : Fin (n + 1) → EucSpace d) :
    mixing (fun _ => 1 / ((n : ℝ) + 1)) v = 0 := by
  unfold mixing centeredWeight
  simp only [sub_self, zero_smul, Finset.sum_const_zero]

/-- The actual order of the bias-free projected, post-normalized attention branch. -/
noncomputable def periBranch {n d e : ℕ} (p : Fin (n + 1) → ℝ)
    (v : Fin (n + 1) → EucSpace d) (w : EucSpace d →ₗ[ℝ] EucSpace e)
    (eps scale : ℝ) : EucSpace e :=
  scale • GPTMini.rmsNormEps eps (w (mixing p v))

/-- Projection, parameter-free RMSNorm, and residual scaling cannot restore
the constant-value component that centering has already removed. -/
theorem repeated_branch_zero {n d e : ℕ} (p : Fin (n + 1) → ℝ)
    (hp : ∑ j, p j = 1) (v : EucSpace d)
    (w : EucSpace d →ₗ[ℝ] EucSpace e) (eps scale : ℝ) :
    periBranch p (fun _ => v) w eps scale = 0 := by
  unfold periBranch
  rw [mixing_constant p hp v, map_zero]
  simp [GPTMini.rmsNormEps]

/-- The residual preserves its input exactly on the canceled branch. -/
theorem residual_repeated_eq {n d e : ℕ} (p : Fin (n + 1) → ℝ)
    (hp : ∑ j, p j = 1) (v : EucSpace d)
    (w : EucSpace d →ₗ[ℝ] EucSpace e) (eps scale : ℝ) (x : EucSpace e) :
    x + periBranch p (fun _ => v) w eps scale = x := by
  rw [repeated_branch_zero p hp v w eps scale, add_zero]

section Parameters

variable {Θ : Type*} [NormedAddCommGroup Θ] [NormedSpace ℝ Θ]
variable {n d e : ℕ}

/-- The shared-parameter branch is identically zero, so its Frechet derivative
really exists and equals zero. This is stronger than fderiv's zero fallback
for nondifferentiable functions. No differentiability of p/v/w is needed. -/
theorem repeated_branch_hasFDerivAt_zero (p : Θ → Fin (n + 1) → ℝ)
    (hp : ∀ θ, ∑ j, p θ j = 1) (v : Θ → EucSpace d)
    (w : Θ → EucSpace d →ₗ[ℝ] EucSpace e) (eps scale : Θ → ℝ) (θ : Θ) :
    HasFDerivAt (fun t => periBranch (p t) (fun _ => v t) (w t) (eps t) (scale t))
      (0 : Θ →L[ℝ] EucSpace e) θ := by
  have hfun : (fun t => periBranch (p t) (fun _ => v t) (w t) (eps t) (scale t)) =
      (fun _ : Θ => (0 : EucSpace e)) := by
    funext t
    exact repeated_branch_zero (p t) (hp t) (v t) (w t) (eps t) (scale t)
  rw [hfun]
  exact hasFDerivAt_const 0 θ

/-- Diagonal examples give no loss gradient to parameters occurring only in
this attention branch. The residual input and downstream function are fixed;
this makes no zero-gradient claim about embeddings or FFN/readout parameters. -/
theorem attention_only_loss_hasFDerivAt_zero (p : Θ → Fin (n + 1) → ℝ)
    (hp : ∀ θ, ∑ j, p θ j = 1) (v : Θ → EucSpace d)
    (w : Θ → EucSpace d →ₗ[ℝ] EucSpace e) (eps scale : Θ → ℝ)
    (x : EucSpace e) (lossAfter : EucSpace e → ℝ) (θ : Θ) :
    HasFDerivAt
      (fun t => lossAfter (x + periBranch (p t) (fun _ => v t) (w t) (eps t) (scale t)))
      (0 : Θ →L[ℝ] ℝ) θ := by
  have hfun :
      (fun t => lossAfter (x + periBranch (p t) (fun _ => v t) (w t) (eps t) (scale t))) =
      (fun _ : Θ => lossAfter x) := by
    funext t
    rw [residual_repeated_eq (p t) (hp t) (v t) (w t) (eps t) (scale t) x]
  rw [hfun]
  exact hasFDerivAt_const (lossAfter x) θ

end Parameters

/-- Cancellation does not mean the derivative with respect to each independent
token value is zero: perturbing just the first value produces a nonzero slope.
The parameter-gradient theorem above relies on the values remaining tied. -/
theorem independent_value_perturbation (t : ℝ) :
    mixing (n := 1) witnessWeight
      (fun j => if j = 0 then (1 + t) • witnessValue else witnessValue) =
        (t / 4) • witnessValue := by
  rw [two_token_difference witnessWeight witnessWeight_sum]
  norm_num [witnessWeight]
  module

/-- Any decoder that sees ONLY the centered diagonal attention output cannot
solve every quartic diagonal: the inputs a=0 and a=1 both map to zero, but
their answers are 0 and 16. This is not a theorem about the full residual model. -/
theorem no_decoder_from_centered_diagonal {d : ℕ}
    (p : ZMod 97 → Fin 2 → ℝ) (hp : ∀ a, ∑ j, p a j = 1)
    (value : ZMod 97 → EucSpace d) (decode : EucSpace d → ZMod 97) :
    ¬ (∀ a, decode (mixing (n := 1) (p a) (fun _ => value a)) = (a + a) ^ 4) := by
  intro h
  have h0 := h 0
  have h1 := h 1
  rw [mixing_constant (p 0) (hp 0) (value 0)] at h0
  rw [mixing_constant (p 1) (hp 1) (value 1)] at h1
  have hbad : ((0 : ZMod 97) + 0) ^ 4 = (1 + 1) ^ 4 := h0.symm.trans h1
  norm_num at hbad
  exact (by decide : (0 : ZMod 97) ≠ 16) hbad

/-- The task on a=b is a unary function of a. A surviving residual can carry a;
this identity does not assert that the current FFN learns that function. -/
theorem diagonal_target_unary (a : ZMod 97) : (a + a) ^ 4 = (2 * a) ^ 4 := by
  congr 1
  ring

/-- A causal uniform correction is not the paper's global rank-one correction:
already at length two the prefix-uniform matrix is invertible. -/
theorem causal_uniform_det :
    Matrix.det (!![(1 : ℝ), 0; 1 / 2, 1 / 2]) = 1 / 2 := by
  rw [Matrix.det_fin_two]
  norm_num

theorem global_uniform_det :
    Matrix.det (!![(1 / 2 : ℝ), 1 / 2; 1 / 2, 1 / 2]) = 0 := by
  rw [Matrix.det_fin_two]
  norm_num

/- Concrete witnesses for every conditional theorem, including nonuniform
weights and a nonzero value. The parameter family may vary its shared value. -/
example := ordinary_constant witnessWeight witnessWeight_sum witnessValue
example := first_prefix_zero (fun _ : Fin 1 => (1 : ℝ)) (by simp)
  (fun _ => witnessValue)
example := two_token_difference witnessWeight witnessWeight_sum
  (fun j => if j = 0 then witnessValue else 3 • witnessValue)
example := repeated_branch_zero witnessWeight witnessWeight_sum witnessValue
  (LinearMap.id : EucSpace 1 →ₗ[ℝ] EucSpace 1) 1 1
example := residual_repeated_eq witnessWeight witnessWeight_sum witnessValue
  (LinearMap.id : EucSpace 1 →ₗ[ℝ] EucSpace 1) 1 1 witnessValue
example := repeated_branch_hasFDerivAt_zero (fun _ : ℝ => witnessWeight)
  (fun _ => witnessWeight_sum) (fun t => t • witnessValue)
  (fun _ => (LinearMap.id : EucSpace 1 →ₗ[ℝ] EucSpace 1))
  (fun _ => 1) (fun _ => 1) 1
example := attention_only_loss_hasFDerivAt_zero (fun _ : ℝ => witnessWeight)
  (fun _ => witnessWeight_sum) (fun t => t • witnessValue)
  (fun _ => (LinearMap.id : EucSpace 1 →ₗ[ℝ] EucSpace 1))
  (fun _ => 1) (fun _ => 1) witnessValue (fun x => ‖x‖ ^ 2) 1
example := no_decoder_from_centered_diagonal (fun _ => witnessWeight)
  (fun _ => witnessWeight_sum) (fun _ => witnessValue) (fun _ => (0 : ZMod 97))

#print axioms ordinary_constant
#print axioms first_prefix_zero
#print axioms two_token_difference
#print axioms uniform_row_zero
#print axioms repeated_branch_zero
#print axioms residual_repeated_eq
#print axioms repeated_branch_hasFDerivAt_zero
#print axioms attention_only_loss_hasFDerivAt_zero
#print axioms independent_value_perturbation
#print axioms no_decoder_from_centered_diagonal
#print axioms diagonal_target_unary
#print axioms causal_uniform_det
#print axioms global_uniform_det

end QuarticCenteringReview
