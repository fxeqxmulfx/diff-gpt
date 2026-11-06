import research.layernorm_affine_guarantees.Centering

/-!
Local normalizer-choice analysis, plan.json targets 6--8. Affine parameters are
the actual coordinatewise gamma and beta, fixed during an input comparison.
The bound on gamma is explicit; unconstrained training does not supply a
uniform bound across optimizer steps. No learning guarantee follows.
-/

namespace LayerNormAffine
open Transformer Transformer.GPTMini Early99Normalization
open scoped BigOperators

/-- The ordinary coordinatewise learned scale, with no new hidden parameters. -/
noncomputable def diagonal {d : ℕ} (gamma : Fin d → ℝ) (x : EucSpace d) : EucSpace d :=
  WithLp.toLp 2 (fun i => gamma i * x.ofLp i)

/-- The ordinary learned diagonal affine operation following normalization. -/
noncomputable def affine {d : ℕ} (gamma : Fin d → ℝ) (beta x : EucSpace d) : EucSpace d :=
  diagonal gamma x + beta

/-- Target 6: bounded coordinatewise gain controls the Euclidean operator norm. -/
theorem diagonal_bound {d : ℕ} (gamma : Fin d → ℝ) (G : ℝ) (hG : 0 ≤ G)
    (hgamma : ∀ i, |gamma i| ≤ G) (x : EucSpace d) :
    ‖diagonal gamma x‖ ≤ G * ‖x‖ := by
  have hs : ‖diagonal gamma x‖ ^ 2 ≤ G ^ 2 * ‖x‖ ^ 2 := by
    rw [EuclideanSpace.real_norm_sq_eq, EuclideanSpace.real_norm_sq_eq, Finset.mul_sum]
    apply Finset.sum_le_sum
    intro i _
    change (gamma i * x.ofLp i) ^ 2 ≤ G ^ 2 * (x.ofLp i) ^ 2
    rw [mul_pow]
    apply mul_le_mul_of_nonneg_right _ (sq_nonneg _)
    apply sq_le_sq.mpr
    simpa only [abs_of_nonneg hG] using hgamma i
  apply (sq_le_sq₀ (norm_nonneg _) (mul_nonneg hG (norm_nonneg _))).mp
  simpa only [mul_pow] using hs

private theorem diagonal_sub {d : ℕ} (gamma : Fin d → ℝ) (x y : EucSpace d) :
    diagonal gamma (x - y) = diagonal gamma x - diagonal gamma y := by
  ext i
  simp [diagonal, mul_sub]

private theorem transfer_bounds {d : ℕ} (N : EucSpace d → EucSpace d)
    (R L G : ℝ) (hG : 0 ≤ G)
    (hR : ∀ x, ‖N x‖ ≤ R) (hL : ∀ x y, ‖N x - N y‖ ≤ L * ‖x - y‖)
    (gamma : Fin d → ℝ) (hgamma : ∀ i, |gamma i| ≤ G)
    (beta x y : EucSpace d) :
    ‖affine gamma beta (N x)‖ ≤ G * R + ‖beta‖ ∧
      ‖affine gamma beta (N x) - affine gamma beta (N y)‖ ≤ (G * L) * ‖x - y‖ := by
  constructor
  · change ‖diagonal gamma (N x) + beta‖ ≤ G * R + ‖beta‖
    calc
      _ ≤ ‖diagonal gamma (N x)‖ + ‖beta‖ := norm_add_le _ _
      _ ≤ G * ‖N x‖ + ‖beta‖ :=
        add_le_add (diagonal_bound gamma G hG hgamma (N x)) (le_refl ‖beta‖)
      _ ≤ G * R + ‖beta‖ :=
        add_le_add (mul_le_mul_of_nonneg_left (hR x) hG) (le_refl ‖beta‖)
  · change ‖(diagonal gamma (N x) + beta) - (diagonal gamma (N y) + beta)‖ ≤
      (G * L) * ‖x - y‖
    rw [add_sub_add_right_eq_sub, ← diagonal_sub]
    calc
      _ ≤ G * ‖N x - N y‖ := diagonal_bound gamma G hG hgamma (N x - N y)
      _ ≤ G * (L * ‖x - y‖) := mul_le_mul_of_nonneg_left (hL x y) hG
      _ = (G * L) * ‖x - y‖ := by ring

/-- Target 7: learned affine RMS retains bounds that explicitly depend on gamma
and beta. Bias cancels in fixed-parameter input differences, not output radii. -/
theorem affine_rms_bounds {d : ℕ} (hd : 0 < d) (eps : ℝ) (heps : 0 < eps)
    (gamma : Fin d → ℝ) (G : ℝ) (hG : 0 ≤ G) (hgamma : ∀ i, |gamma i| ≤ G)
    (beta x y : EucSpace d) :
    ‖affine gamma beta (rmsNormEps eps x)‖ ≤ G * Real.sqrt (d : ℝ) + ‖beta‖ ∧
      ‖affine gamma beta (rmsNormEps eps x) - affine gamma beta (rmsNormEps eps y)‖ ≤
        (G / Real.sqrt eps) * ‖x - y‖ := by
  have h := transfer_bounds (rmsNormEps eps) (Real.sqrt (d : ℝ)) (1 / Real.sqrt eps)
    G hG (rmsNormEps_norm_le eps heps hd)
    (rmsNormEps_lipschitz_improved eps heps hd) gamma hgamma beta x y
  simpa only [mul_one_div] using h

/-- Target 8: learned affine LayerNorm has the same explicit radius and input
sensitivity bounds; its shift invariance is a separate structural property. -/
theorem affine_layerNorm_bounds {d : ℕ} (hd : 0 < d) (eps : ℝ) (heps : 0 < eps)
    (gamma : Fin d → ℝ) (G : ℝ) (hG : 0 ≤ G) (hgamma : ∀ i, |gamma i| ≤ G)
    (beta x y : EucSpace d) :
    ‖affine gamma beta (layerNormEps eps x)‖ ≤ G * Real.sqrt (d : ℝ) + ‖beta‖ ∧
      ‖affine gamma beta (layerNormEps eps x) - affine gamma beta (layerNormEps eps y)‖ ≤
        (G / Real.sqrt eps) * ‖x - y‖ := by
  have h := transfer_bounds (layerNormEps eps) (Real.sqrt (d : ℝ)) (1 / Real.sqrt eps)
    G hG (layerNorm_radius hd eps heps) (layerNorm_lipschitz hd eps heps)
    gamma hgamma beta x y
  simpa only [mul_one_div] using h

example (x : EucSpace 8) : ‖diagonal (fun _ => (2 : ℝ)) x‖ ≤ 2 * ‖x‖ :=
  diagonal_bound _ _ (by norm_num) (by intro i; norm_num) x

example (x y : EucSpace 8) :=
  affine_rms_bounds (by norm_num : 0 < 8) (1 / 8388608) (by norm_num)
    (fun _ => (2 : ℝ)) 2 (by norm_num) (by intro i; norm_num) (ones 8) x y

example (x y : EucSpace 8) :=
  affine_layerNorm_bounds (by norm_num : 0 < 8) (1 / 8388608) (by norm_num)
    (fun _ => (2 : ℝ)) 2 (by norm_num) (by intro i; norm_num) (ones 8) x y

#print axioms diagonal_bound
#print axioms affine_rms_bounds
#print axioms affine_layerNorm_bounds
end LayerNormAffine
