import Transformer.AttentionResiduals.QueryCoordinates
import Transformer.Residual.RMSBackward

/-!
Local initialization analysis, plan.json targets 1--4. Reuse the exact RMS
coordinate identity in Transformer.AttentionResiduals.QueryCoordinates QC2;
the coordinate VJP and conditioning use Transformer.Residual.RMSBackward.
The parameter-to-activation map and downstream loss are fixed. A physical
parameter rescaling is not a coordinate change during optimizer execution.
Every identity retains epsilon. No model convergence or FP guarantee follows.
-/

namespace RMSInitialization
open Transformer Transformer.GPTMini Transformer.AttentionResiduals

variable {P : Type*} [NormedAddCommGroup P] [NormedSpace ℝ P] {d : ℕ}

/-- Target 1: for a fixed linear map from parameters to activations, positive
parameter rescaling changes the effective epsilon. Source QC2 is reused, not
an assumption of ideal scale invariance. -/
theorem linear_forward (eps scale : ℝ) (hs : 0 < scale)
    (A : P →L[ℝ] EucSpace d) (w : P) :
    rmsNormEps eps (A (scale • w)) = rmsNormEps (eps / scale ^ 2) (A w) := by
  rw [map_smul, ← rmsNormEps_coordinates eps scale hs]

/-- Target 2: the physical-parameter derivative at scale*w is 1/scale times
the counterpart at w with epsilon/scale². The counterpart derivative is an
explicit hypothesis; this does not assert identical same-epsilon losses. -/
theorem linear_loss_derivative (eps scale : ℝ) (hs : 0 < scale)
    (A : P →L[ℝ] EucSpace d) (loss : EucSpace d → ℝ) (w : P)
    (D : P →L[ℝ] ℝ)
    (h : HasFDerivAt (fun u => loss (rmsNormEps (eps / scale ^ 2) (A u))) D w) :
    HasFDerivAt (fun u => loss (rmsNormEps eps (A u)))
      (scale⁻¹ • D) (scale • w) := by
  have hinv : scale⁻¹ • (scale • w) = w := by
    rw [smul_smul, inv_mul_cancel₀ hs.ne', one_smul]
  have h' : HasFDerivAt (fun u => loss (rmsNormEps (eps / scale ^ 2) (A u)))
      D (scale⁻¹ • (scale • w)) := by rw [hinv]; exact h
  have hc := h'.comp (scale • w)
    ((hasFDerivAt_id (𝕜 := ℝ) (scale • w)).const_smul scale⁻¹)
  have hfun : (fun u => loss (rmsNormEps (eps / scale ^ 2) (A (scale⁻¹ • u)))) =
      (fun u => loss (rmsNormEps eps (A u))) := by
    funext u
    rw [map_smul, rmsNormEps_coordinates eps scale hs,
      smul_smul, mul_inv_cancel₀ hs.ne', one_smul]
  change HasFDerivAt
    (fun u => loss (rmsNormEps (eps / scale ^ 2) (A (scale⁻¹ • u))))
    (D.comp (scale⁻¹ • ContinuousLinearMap.id ℝ P)) (scale • w) at hc
  rw [hfun] at hc
  simpa only [ContinuousLinearMap.comp_smul, ContinuousLinearMap.comp_id] using hc

/-- Target 3: one coordinate of the RMS VJP obeys the same inverse-scale
identity. m is mean(x²), c is mean(x*g), and x/g are one coordinate. Under
input rescaling, the energy becomes scale²*m and the product becomes scale*c.
Source: the existing RMSBackward coordinate expression, with finite epsilon. -/
theorem vjp_rescale (m eps scale x g c : ℝ) (hm : 0 ≤ m) (heps : 0 < eps)
    (hs : 0 < scale) :
    g / Real.sqrt (scale ^ 2 * m + eps) -
        (scale * x) * (scale * c) / Real.sqrt (scale ^ 2 * m + eps) ^ 3 =
      scale⁻¹ * (g / Real.sqrt (m + eps / scale ^ 2) -
        x * c / Real.sqrt (m + eps / scale ^ 2) ^ 3) := by
  have hp : 0 < m + eps / scale ^ 2 := by positivity
  have hr : Real.sqrt (m + eps / scale ^ 2) ≠ 0 := (Real.sqrt_pos.mpr hp).ne'
  have henergy : scale ^ 2 * m + eps = scale ^ 2 * (m + eps / scale ^ 2) := by
    field_simp
  rw [henergy, Real.sqrt_mul (sq_nonneg scale), Real.sqrt_sq_eq_abs, abs_of_pos hs]
  field_simp

/-- Target 4: increasing the input scale worsens the radial/tangent coefficient
ratio (or leaves it equal), although both coefficients remain finite. The ratio
is the exact existing RMSBackward ratio, not a full-network condition number. -/
theorem coefficient_ratio_decreases (m eps scale : ℝ) (hm : 0 ≤ m)
    (heps : 0 < eps) (hs : 1 ≤ scale) :
    eps / (scale ^ 2 * m + eps) ≤ eps / (m + eps) := by
  have hsq : 1 ≤ scale ^ 2 := by nlinarith
  have henergy : m + eps ≤ scale ^ 2 * m + eps := by
    nlinarith [mul_nonneg (sub_nonneg.mpr hsq) hm]
  exact div_le_div_of_nonneg_left heps.le (by positivity) henergy

-- A concrete nonzero vector and positive scale instantiate the forward identity.
example : rmsNormEps (1 / 8388608)
      ((ContinuousLinearMap.id ℝ (EucSpace 1))
        ((4 : ℝ) • PiLp.single 2 (0 : Fin 1) (1 : ℝ))) =
    rmsNormEps ((1 / 8388608) / 4 ^ 2)
      (PiLp.single 2 (0 : Fin 1) (1 : ℝ) : EucSpace 1) :=
  linear_forward _ _ (by norm_num) _ _

-- A nonconstant scalar loss on a unit direction has a positive derivative;
-- its existence is supplied by the actual finite-epsilon radial RMS theorem.
example (a : ℝ) :=
    let v : EucSpace 1 := PiLp.single 2 (0 : Fin 1) (1 : ℝ)
    let A : ℝ →L[ℝ] EucSpace 1 := (ContinuousLinearMap.id ℝ ℝ).smulRight v
    let e : ℝ := (1 / 8388608) / 2 ^ 2
    have hv : ‖v‖ ^ 2 = ((1 : ℕ) : ℝ) := by norm_num [v, PiLp.norm_single]
    have hfun : (fun u : ℝ => inner ℝ v (rmsNormEps e (A u))) =
        Transformer.Residual.radialRMS e := by
      funext u
      change inner ℝ v (rmsNormEps e (u • v)) = _
      rw [Transformer.Residual.rmsNormEps_unit_direction (by decide) v hv,
        real_inner_smul_right, real_inner_self_eq_norm_sq, hv]
      simp
    have h : HasFDerivAt (fun u : ℝ => inner ℝ v (rmsNormEps e (A u)))
        ((1 : ℝ →L[ℝ] ℝ).smulRight (e / Real.sqrt (a ^ 2 + e) ^ 3)) a := by
      rw [hfun]
      exact (Transformer.Residual.radialRMS_hasDerivAt e a (by norm_num [e])).hasFDerivAt
    linear_loss_derivative (1 / 8388608) 2 (by norm_num) A
      (fun z => inner ℝ v z) a _ h

example (x g c : ℝ) :
    g / Real.sqrt (4 ^ 2 * 1 + 1 / 8388608) -
        (4 * x) * (4 * c) / Real.sqrt (4 ^ 2 * 1 + 1 / 8388608) ^ 3 =
      (4 : ℝ)⁻¹ * (g / Real.sqrt (1 + (1 / 8388608) / 4 ^ 2) -
        x * c / Real.sqrt (1 + (1 / 8388608) / 4 ^ 2) ^ 3) :=
  vjp_rescale _ _ _ x g c (by norm_num) (by norm_num) (by norm_num)

example : (1 / 8388608 : ℝ) / (4 ^ 2 * 1 + 1 / 8388608) ≤
    (1 / 8388608) / (1 + 1 / 8388608) :=
  coefficient_ratio_decreases _ _ _ (by norm_num) (by norm_num) (by norm_num)

#print axioms linear_forward
#print axioms linear_loss_derivative
#print axioms vjp_rescale
#print axioms coefficient_ratio_decreases
end RMSInitialization
