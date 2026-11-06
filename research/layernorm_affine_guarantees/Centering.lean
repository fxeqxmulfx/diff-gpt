import research.normalization_guarantees.RMSLipschitz

/-!
Local normalizer-choice analysis, plan.json targets 1--5. LayerNorm uses the
actual population variance: subtract the coordinate mean, then apply the
existing finite-epsilon RMS definition. No abstract sphere replacement or
unbiased-variance factor is used. These are real-arithmetic input bounds,
not statements about floating-point implementations or learning.
-/

namespace LayerNormAffine
open Transformer Transformer.GPTMini Early99Normalization
open scoped BigOperators

/-- The constant coordinate direction removed by featurewise mean subtraction. -/
noncomputable def ones (d : ℕ) : EucSpace d := WithLp.toLp 2 (fun _ => 1)

/-- Population coordinate mean, matching the declared LayerNorm equation. -/
noncomputable def coordinateMean {d : ℕ} (x : EucSpace d) : ℝ :=
  (∑ i, x.ofLp i) / (d : ℝ)

/-- Mean subtraction retains every centered coordinate; it is not attention centering. -/
noncomputable def center {d : ℕ} (x : EucSpace d) : EucSpace d :=
  x - coordinateMean x • ones d

/-- Finite-epsilon population-variance LayerNorm, before any affine parameters. -/
noncomputable def layerNormEps {d : ℕ} (eps : ℝ) (x : EucSpace d) : EucSpace d :=
  rmsNormEps eps (center x)

private theorem ones_norm_sq (d : ℕ) : ‖ones d‖ ^ 2 = (d : ℝ) := by
  rw [EuclideanSpace.real_norm_sq_eq]
  simp [ones]

private theorem inner_ones {d : ℕ} (x : EucSpace d) :
    inner ℝ (ones d) x = ∑ i, x.ofLp i := by
  rw [PiLp.inner_apply]
  simp [ones]

/-- Target 1: explicit mean subtraction removes exactly the constant-direction
energy. Positive width avoids division by zero. -/
theorem center_energy {d : ℕ} (hd : 0 < d) (x : EucSpace d) :
    ‖center x‖ ^ 2 = ‖x‖ ^ 2 - (∑ i, x.ofLp i) ^ 2 / (d : ℝ) := by
  have hd0 : (d : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hd.ne'
  rw [center, norm_sub_sq_real, norm_smul, Real.norm_eq_abs,
    mul_pow, sq_abs, ones_norm_sq, real_inner_smul_right,
    real_inner_comm (ones d) x, inner_ones]
  unfold coordinateMean
  field_simp
  ring

private theorem center_sub {d : ℕ} (x y : EucSpace d) :
    center x - center y = center (x - y) := by
  have hm : coordinateMean (x - y) = coordinateMean x - coordinateMean y := by
    simp [coordinateMean, Finset.sum_sub_distrib, sub_div]
  simp only [center, hm, sub_smul]
  module

/-- Target 2: mean subtraction is a Euclidean nonexpansion. -/
theorem center_nonexpansive {d : ℕ} (hd : 0 < d) (x y : EucSpace d) :
    ‖center x - center y‖ ≤ ‖x - y‖ := by
  rw [center_sub]
  have h := center_energy hd (x - y)
  have hremoved : 0 ≤ (∑ i, (x - y).ofLp i) ^ 2 / (d : ℝ) := by positivity
  nlinarith [norm_nonneg (center (x - y)), norm_nonneg (x - y)]

/-- Target 3: actual positive-epsilon LayerNorm retains the RMS radius bound. -/
theorem layerNorm_radius {d : ℕ} (hd : 0 < d) (eps : ℝ) (heps : 0 < eps)
    (x : EucSpace d) : ‖layerNormEps eps x‖ ≤ Real.sqrt (d : ℝ) := by
  exact rmsNormEps_norm_le eps heps hd (center x)

/-- Target 4: centering does not enlarge the improved 1/sqrt(epsilon) RMS
input Lipschitz constant. This does not compare trained model conditioning. -/
theorem layerNorm_lipschitz {d : ℕ} (hd : 0 < d) (eps : ℝ) (heps : 0 < eps)
    (x y : EucSpace d) :
    ‖layerNormEps eps x - layerNormEps eps y‖ ≤
      (1 / Real.sqrt eps) * ‖x - y‖ := by
  calc
    _ ≤ (1 / Real.sqrt eps) * ‖center x - center y‖ :=
      rmsNormEps_lipschitz_improved eps heps hd _ _
    _ ≤ (1 / Real.sqrt eps) * ‖x - y‖ :=
      mul_le_mul_of_nonneg_left (center_nonexpansive hd x y) (by positivity)

/-- Target 5: adding any scalar constant to every coordinate leaves LayerNorm
unchanged, even with finite epsilon. This is a local information loss, not a
claim that a residual transformer or tied readout cannot learn. -/
theorem layerNorm_shift {d : ℕ} (hd : 0 < d) (eps c : ℝ) (x : EucSpace d) :
    layerNormEps eps (x + c • ones d) = layerNormEps eps x := by
  have hd0 : (d : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hd.ne'
  have hm : coordinateMean (x + c • ones d) = coordinateMean x + c := by
    simp [coordinateMean, ones, Finset.sum_add_distrib, add_div, hd0]
  have hc : center (x + c • ones d) = center x := by
    simp only [center, hm, add_smul]
    module
  unfold layerNormEps
  rw [hc]

-- Concrete positive width/epsilon witnesses; vectors and shifts remain arbitrary.
example (x : EucSpace 8) :
    ‖center x‖ ^ 2 = ‖x‖ ^ 2 - (∑ i, x.ofLp i) ^ 2 / 8 :=
  center_energy (by norm_num) x

example (x y : EucSpace 8) : ‖center x - center y‖ ≤ ‖x - y‖ :=
  center_nonexpansive (by norm_num) x y

example (x : EucSpace 8) : ‖layerNormEps (1 / 8388608) x‖ ≤ Real.sqrt 8 :=
  layerNorm_radius (by norm_num) _ (by norm_num) x

example (x y : EucSpace 8) :
    ‖layerNormEps (1 / 8388608) x - layerNormEps (1 / 8388608) y‖ ≤
      (1 / Real.sqrt (1 / 8388608)) * ‖x - y‖ :=
  layerNorm_lipschitz (by norm_num) _ (by norm_num) x y

example (x : EucSpace 8) (c : ℝ) :
    layerNormEps (1 / 8388608) (x + c • ones 8) = layerNormEps (1 / 8388608) x :=
  layerNorm_shift (by norm_num) _ c x

#print axioms center_energy
#print axioms center_nonexpansive
#print axioms layerNorm_radius
#print axioms layerNorm_lipschitz
#print axioms layerNorm_shift
end LayerNormAffine
