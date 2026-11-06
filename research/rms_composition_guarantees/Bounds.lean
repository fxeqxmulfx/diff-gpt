import research.normalization_guarantees.RMSLipschitz
import Transformer.Residual.RMSBackward

/-!
Two adjacent finite-epsilon RMS operations, as at the normalized embedding and
first block input in the declared early99 ablations. Source definitions:
Transformer.GPTMini.RMSNorm and Transformer.Residual.GradientPaths/RMSBackward
at 90b1df2e; improved one-operation bound in the pinned RMSLipschitz.lean.
All arithmetic is real. No learned projection, residual/readout contribution,
floating-point operation, optimization trajectory or generalization is certified.
-/

namespace RMSComposition
open Transformer Transformer.GPTMini Transformer.Residual Early99Normalization

/-- Exact finite-epsilon composition on a unit-RMS amplitude. No idealized
scale invariance discards either occurrence of epsilon. -/
theorem radial_comp_eq (eps : ℝ) (heps : 0 < eps) (a : ℝ) :
    radialRMS eps (radialRMS eps a) =
      a / Real.sqrt ((1 + eps) * a ^ 2 + eps ^ 2) := by
  have hp : 0 < a ^ 2 + eps := by positivity
  have hs : 0 < Real.sqrt (a ^ 2 + eps) := Real.sqrt_pos.mpr hp
  have ht : 0 < (1 + eps) * a ^ 2 + eps ^ 2 := by positivity
  have hinner : (a / Real.sqrt (a ^ 2 + eps)) ^ 2 + eps =
      ((1 + eps) * a ^ 2 + eps ^ 2) / (a ^ 2 + eps) := by
    rw [div_pow, Real.sq_sqrt hp.le]
    field_simp
    ring
  unfold radialRMS
  rw [hinner, Real.sqrt_div ht.le]
  field_simp

/-- The scalar composition is the actual double vector RMS on every unit-RMS
direction, including zero amplitude. This reuses the existing vector definition. -/
theorem double_on_unit_direction {d : ℕ} (eps a : ℝ) (hd : 0 < d)
    (v : EucSpace d) (hv : ‖v‖ ^ 2 = (d : ℝ)) :
    rmsNormEps eps (rmsNormEps eps (a • v)) =
      radialRMS eps (radialRMS eps a) • v := by
  rw [rmsNormEps_unit_direction hd v hv eps a,
    rmsNormEps_unit_direction hd v hv eps (radialRMS eps a)]

/-- Two adjacent positive-epsilon RMS maps have global sensitivity bound 1/eps.
The width is positive, but the bound is independent of its value. -/
theorem double_lipschitz {d : ℕ} (eps : ℝ) (heps : 0 < eps) (hd : 0 < d)
    (x y : EucSpace d) :
    ‖rmsNormEps eps (rmsNormEps eps x) - rmsNormEps eps (rmsNormEps eps y)‖ ≤
      (1 / eps) * ‖x - y‖ := by
  have hs : Real.sqrt eps ≠ 0 := (Real.sqrt_pos.mpr heps).ne'
  have hc : (1 / Real.sqrt eps) * (1 / Real.sqrt eps) = 1 / eps := by
    field_simp
    nlinarith [Real.sq_sqrt heps.le]
  calc
    _ ≤ (1 / Real.sqrt eps) * ‖rmsNormEps eps x - rmsNormEps eps y‖ :=
      rmsNormEps_lipschitz_improved eps heps hd _ _
    _ ≤ (1 / Real.sqrt eps) * ((1 / Real.sqrt eps) * ‖x - y‖) :=
      mul_le_mul_of_nonneg_left (rmsNormEps_lipschitz_improved eps heps hd x y)
        (by positivity)
    _ = (1 / eps) * ‖x - y‖ := by rw [← mul_assoc, hc]

/-- The double scalar map's derivative at zero equals 1/eps. A second
normalization can therefore increase local sensitivity, despite bounded outputs. -/
theorem radial_double_zero_derivative (eps : ℝ) (heps : 0 < eps) :
    HasDerivAt (fun a => radialRMS eps (radialRMS eps a)) (1 / eps) 0 := by
  have hs : Real.sqrt eps ≠ 0 := (Real.sqrt_pos.mpr heps).ne'
  have h := (radialRMS_zero_hasDerivAt eps heps).comp_of_eq (0 : ℝ)
    (radialRMS_zero_hasDerivAt eps heps) (by simp [radialRMS])
  apply h.congr_deriv
  field_simp
  nlinarith [Real.sq_sqrt heps.le]

/-- The 1/eps derivative is attained along actual nonzero unit-RMS vector
directions. This does not assert that learned embeddings are near zero. -/
theorem double_zero_direction_derivative {d : ℕ} (eps : ℝ) (heps : 0 < eps)
    (hd : 0 < d) (v : EucSpace d) (hv : ‖v‖ ^ 2 = (d : ℝ)) :
    HasDerivAt (fun a : ℝ => rmsNormEps eps (rmsNormEps eps (a • v)))
      ((1 / eps) • v) 0 := by
  have h := (radial_double_zero_derivative eps heps).smul_const v
  have hfun : (fun a : ℝ => rmsNormEps eps (rmsNormEps eps (a • v))) =
      (fun a : ℝ => radialRMS eps (radialRMS eps a) • v) := by
    funext a
    exact double_on_unit_direction eps a hd v hv
  rw [hfun]
  exact h

-- Concrete positive epsilon and positive-dimensional examples, including the
-- exact native FP32 epsilon. The direction in dimension one is nonzero.
example (a : ℝ) :
    radialRMS (1 / 8388608) (radialRMS (1 / 8388608) a) =
      a / Real.sqrt ((1 + 1 / 8388608) * a ^ 2 + (1 / 8388608) ^ 2) :=
  radial_comp_eq _ (by norm_num) a

example (a : ℝ) :
    rmsNormEps (1 / 8388608)
      (rmsNormEps (1 / 8388608) (a • (PiLp.single 2 (0 : Fin 1) (1 : ℝ)))) =
      radialRMS (1 / 8388608) (radialRMS (1 / 8388608) a) •
        (PiLp.single 2 (0 : Fin 1) (1 : ℝ)) :=
  double_on_unit_direction _ a (by norm_num) _ (by norm_num [PiLp.norm_single])

example (x y : EucSpace 8) :
    ‖rmsNormEps (1 / 8388608) (rmsNormEps (1 / 8388608) x) -
      rmsNormEps (1 / 8388608) (rmsNormEps (1 / 8388608) y)‖ ≤
      (1 / (1 / 8388608)) * ‖x - y‖ :=
  double_lipschitz _ (by norm_num) (by norm_num) x y

example : HasDerivAt
    (fun a => radialRMS (1 / 8388608) (radialRMS (1 / 8388608) a))
    (1 / (1 / 8388608)) 0 :=
  radial_double_zero_derivative _ (by norm_num)

example : HasDerivAt
    (fun a : ℝ => rmsNormEps (1 / 8388608)
      (rmsNormEps (1 / 8388608) (a • (PiLp.single 2 (0 : Fin 1) (1 : ℝ)))))
    ((1 / (1 / 8388608) : ℝ) • (PiLp.single 2 (0 : Fin 1) (1 : ℝ))) 0 :=
  double_zero_direction_derivative _ (by norm_num) (by norm_num) _
    (by norm_num [PiLp.norm_single])

#print axioms radial_comp_eq
#print axioms double_on_unit_direction
#print axioms double_lipschitz
#print axioms radial_double_zero_derivative
#print axioms double_zero_direction_derivative
end RMSComposition
