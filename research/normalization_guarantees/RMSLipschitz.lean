import Transformer.GPTMini.RMSNorm
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Analysis.Calculus.Deriv.Inv
import Mathlib.Analysis.SpecialFunctions.Sqrt

/-!
A sharper real-arithmetic Lipschitz bound for the unchanged parameter-free
finite-epsilon RMS operation. The previous checked bound was 2/sqrt(eps).
The proof keeps the radial and angular contributions separate instead of
bounding both by a triangle inequality. No floating-point or learning claim.
-/

namespace Early99Normalization

open Transformer Real

/-- Radial scalar map whose derivative is at most one. -/
noncomputable def radialScalar (c t : ℝ) : ℝ :=
  Real.sqrt c * t / Real.sqrt (t ^ 2 + c)

/-- Exact derivative of the scalar map for every real t and positive c. -/
theorem radialScalar_hasDerivAt (c : ℝ) (hc : 0 < c) (t : ℝ) :
    HasDerivAt (radialScalar c)
      (Real.sqrt c * c / (Real.sqrt (t ^ 2 + c)) ^ 3) t := by
  have hp : 0 < t ^ 2 + c := by positivity
  have hs : 0 < Real.sqrt (t ^ 2 + c) := Real.sqrt_pos.mpr hp
  have hnum := (hasDerivAt_id t).const_mul (Real.sqrt c)
  have hden := (((hasDerivAt_id t).pow 2).add_const c).sqrt hp.ne'
  have h : HasDerivAt (radialScalar c)
    ((Real.sqrt c * Real.sqrt (t ^ 2 + c) -
      (Real.sqrt c * t) * (2 * t / (2 * Real.sqrt (t ^ 2 + c)))) /
      Real.sqrt (t ^ 2 + c) ^ 2) t := by
    convert! hnum.fun_div hden hs.ne' using 1
    simp
  convert h using 1
  field_simp [hs.ne']
  nlinarith [Real.sq_sqrt hp.le]

/-- Positive epsilon makes the radial scalar function globally nonexpansive. -/
theorem radialScalar_lipschitz (c : ℝ) (hc : 0 < c) :
    LipschitzWith 1 (radialScalar c) := by
  apply lipschitzWith_of_nnnorm_deriv_le
  · intro t
    exact (radialScalar_hasDerivAt c hc t).differentiableAt
  · intro t
    change ‖deriv (radialScalar c) t‖ ≤ (1 : ℝ)
    rw [(radialScalar_hasDerivAt c hc t).deriv, Real.norm_eq_abs,
      abs_of_nonneg (by positivity)]
    have hp : 0 < t ^ 2 + c := by positivity
    have hs : 0 < Real.sqrt (t ^ 2 + c) := Real.sqrt_pos.mpr hp
    have hlow : Real.sqrt c ≤ Real.sqrt (t ^ 2 + c) :=
      Real.sqrt_le_sqrt (by nlinarith [sq_nonneg t])
    have henergy : c ≤ Real.sqrt (t ^ 2 + c) ^ 2 := by
      rw [Real.sq_sqrt hp.le]
      nlinarith [sq_nonneg t]
    apply (div_le_iff₀ (pow_pos hs 3)).mpr
    have hm := mul_le_mul hlow henergy hc.le hs.le
    nlinarith [hm]

/-- Vector map with the same radial scalar function, before RMS rescaling. -/
noncomputable def radialMap {d : ℕ} (c : ℝ) (x : EucSpace d) : EucSpace d :=
  (Real.sqrt c / Real.sqrt (‖x‖ ^ 2 + c)) • x

/-- Radial and angular distances both contract; no triangle factor is lost. -/
theorem radialMap_nonexpansive {d : ℕ} (c : ℝ) (hc : 0 < c)
    (x y : EucSpace d) : ‖radialMap c x - radialMap c y‖ ≤ ‖x - y‖ := by
  set a := Real.sqrt (‖x‖ ^ 2 + c) with ha
  set b := Real.sqrt (‖y‖ ^ 2 + c) with hb
  have ha0 : 0 < a := by dsimp [a]; positivity
  have hb0 : 0 < b := by dsimp [b]; positivity
  have hca : Real.sqrt c ≤ a := by
    apply Real.sqrt_le_sqrt
    nlinarith [sq_nonneg ‖x‖]
  have hcb : Real.sqrt c ≤ b := by
    apply Real.sqrt_le_sqrt
    nlinarith [sq_nonneg ‖y‖]
  set u := Real.sqrt c / a with hu
  set v := Real.sqrt c / b with hv
  have hu0 : 0 ≤ u := by dsimp [u]; positivity
  have hv0 : 0 ≤ v := by dsimp [v]; positivity
  have huv : u * v ≤ 1 := by
    dsimp [u, v]
    rw [div_mul_div_comm]
    apply (div_le_iff₀ (mul_pos ha0 hb0)).mpr
    have h := mul_le_mul hca hcb (Real.sqrt_nonneg _) ha0.le
    nlinarith [h]
  have hscalar := (radialScalar_lipschitz c hc).norm_sub_le ‖x‖ ‖y‖
  have hscalar' : |u * ‖x‖ - v * ‖y‖| ≤ |‖x‖ - ‖y‖| := by
    simpa [radialScalar, u, v, a, b, Real.norm_eq_abs, div_mul_eq_mul_div] using hscalar
  have hsquare := sq_le_sq.mpr hscalar'
  have hgap : 0 ≤ ‖x‖ * ‖y‖ - inner ℝ x y := by
    linarith [real_inner_le_norm x y]
  have hangular :
      2 * (u * v) * (‖x‖ * ‖y‖ - inner ℝ x y) ≤
      2 * (‖x‖ * ‖y‖ - inner ℝ x y) := by
    nlinarith [mul_nonneg (sub_nonneg.mpr huv) hgap]
  have hresult : ‖u • x - v • y‖ ^ 2 ≤ ‖x - y‖ ^ 2 := by
    calc
      ‖u • x - v • y‖ ^ 2 =
          (u * ‖x‖ - v * ‖y‖) ^ 2 +
            2 * (u * v) * (‖x‖ * ‖y‖ - inner ℝ x y) := by
        rw [norm_sub_sq_real, norm_smul, norm_smul, Real.norm_eq_abs,
          Real.norm_eq_abs, abs_of_nonneg hu0, abs_of_nonneg hv0,
          real_inner_smul_left, real_inner_smul_right]
        ring
      _ ≤ (‖x‖ - ‖y‖) ^ 2 + 2 * (‖x‖ * ‖y‖ - inner ℝ x y) :=
        add_le_add hsquare hangular
      _ = ‖x - y‖ ^ 2 := by rw [norm_sub_sq_real]; ring
  exact (sq_le_sq₀ (norm_nonneg _) (norm_nonneg _)).mp hresult

/-- Exact relation to the existing GPTMini definition, with finite epsilon. -/
theorem rmsNormEps_eq_scaled_radial {d : ℕ} (eps : ℝ) (heps : 0 < eps)
    (hd : 0 < d) (x : EucSpace d) :
    GPTMini.rmsNormEps eps x = (1 / Real.sqrt eps) • radialMap ((d : ℝ) * eps) x := by
  have hd' : (0 : ℝ) < (d : ℝ) := Nat.cast_pos.mpr hd
  have he : Real.sqrt eps ≠ 0 := (Real.sqrt_pos.mpr heps).ne'
  unfold GPTMini.rmsNormEps radialMap
  rw [smul_smul, Real.sqrt_mul hd'.le]
  congr 1
  field_simp

/-- Improved global Lipschitz bound for the actual finite-epsilon RMS equation. -/
theorem rmsNormEps_lipschitz_improved {d : ℕ} (eps : ℝ) (heps : 0 < eps)
    (hd : 0 < d) (x y : EucSpace d) :
    ‖GPTMini.rmsNormEps eps x - GPTMini.rmsNormEps eps y‖ ≤
      (1 / Real.sqrt eps) * ‖x - y‖ := by
  have hd' : (0 : ℝ) < (d : ℝ) := Nat.cast_pos.mpr hd
  rw [rmsNormEps_eq_scaled_radial eps heps hd x,
    rmsNormEps_eq_scaled_radial eps heps hd y, ← smul_sub, norm_smul,
    Real.norm_eq_abs, abs_of_nonneg (by positivity)]
  exact mul_le_mul_of_nonneg_left
    (radialMap_nonexpansive ((d : ℝ) * eps) (mul_pos hd' heps) x y) (by positivity)

/-- Post-RMS residual scaling multiplies the improved bound by its absolute value. -/
theorem scaled_rmsNormEps_lipschitz {d : ℕ} (eps : ℝ) (heps : 0 < eps)
    (hd : 0 < d) (alpha : ℝ) (x y : EucSpace d) :
    ‖alpha • GPTMini.rmsNormEps eps x - alpha • GPTMini.rmsNormEps eps y‖ ≤
      (|alpha| / Real.sqrt eps) * ‖x - y‖ := by
  rw [← smul_sub, norm_smul, Real.norm_eq_abs]
  calc
    |alpha| * ‖GPTMini.rmsNormEps eps x - GPTMini.rmsNormEps eps y‖ ≤
        |alpha| * ((1 / Real.sqrt eps) * ‖x - y‖) :=
      mul_le_mul_of_nonneg_left (rmsNormEps_lipschitz_improved eps heps hd x y)
        (abs_nonneg alpha)
    _ = (|alpha| / Real.sqrt eps) * ‖x - y‖ := by ring

/-- Nonvacuity: actual FP32 epsilon, positive width and arbitrary vectors. -/
example (x y : EucSpace 8) :
    ‖GPTMini.rmsNormEps (1 / 8388608) x - GPTMini.rmsNormEps (1 / 8388608) y‖ ≤
      (1 / Real.sqrt (1 / 8388608)) * ‖x - y‖ :=
  rmsNormEps_lipschitz_improved _ (by norm_num) (by norm_num) x y

example (t : ℝ) : HasDerivAt (radialScalar 1)
    (1 / (Real.sqrt (t ^ 2 + 1)) ^ 3) t := by
  simpa using radialScalar_hasDerivAt 1 (by norm_num) t

example : LipschitzWith 1 (radialScalar 1) := radialScalar_lipschitz 1 (by norm_num)

example (x y : EucSpace 8) : ‖radialMap 1 x - radialMap 1 y‖ ≤ ‖x - y‖ :=
  radialMap_nonexpansive 1 (by norm_num) x y

example (x : EucSpace 8) :
    GPTMini.rmsNormEps (1 / 8388608) x =
      (1 / Real.sqrt (1 / 8388608)) • radialMap (8 * (1 / 8388608)) x :=
  rmsNormEps_eq_scaled_radial _ (by norm_num) (by norm_num) x

example (x y : EucSpace 8) :
    ‖(1 / 4 : ℝ) • GPTMini.rmsNormEps (1 / 8388608) x -
      (1 / 4 : ℝ) • GPTMini.rmsNormEps (1 / 8388608) y‖ ≤
      (|(1 / 4 : ℝ)| / Real.sqrt (1 / 8388608)) * ‖x - y‖ :=
  scaled_rmsNormEps_lipschitz _ (by norm_num) (by norm_num) _ x y

#print axioms Early99Normalization.radialScalar_hasDerivAt
#print axioms Early99Normalization.radialScalar_lipschitz
#print axioms Early99Normalization.radialMap_nonexpansive
#print axioms Early99Normalization.rmsNormEps_eq_scaled_radial
#print axioms Early99Normalization.rmsNormEps_lipschitz_improved
#print axioms Early99Normalization.scaled_rmsNormEps_lipschitz

end Early99Normalization
