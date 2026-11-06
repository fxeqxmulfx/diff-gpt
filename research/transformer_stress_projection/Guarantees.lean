import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Tactic

/-! Real bounds for the declared post-branch map z / max(1, RMS(z)).
For a d-dimensional Euclidean space its norm radius is sqrt(d). The general
radius form below is algebraically identical. No floating-point, boundary
differentiability, whole-network or learning claim is made. -/

namespace ResidualProjection

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Radial projection onto the closed Euclidean ball with positive radius r. -/
noncomputable def project (r : ℝ) (x : E) : E := (r / max r ‖x‖) • x

/-- Small branch outputs retain their magnitude, unlike post-RMS. -/
theorem identity_inside {r : ℝ} (hr : 0 < r) (x : E) (hx : ‖x‖ ≤ r) :
    project r x = x := by
  simp only [project, max_eq_left hx, div_self (ne_of_gt hr), one_smul]

/-- The output norm is at most r, independently of the raw FFN magnitude. -/
theorem radius_bound {r : ℝ} (hr : 0 < r) (x : E) : ‖project r x‖ ≤ r := by
  have hm : 0 < max r ‖x‖ := lt_of_lt_of_le hr (le_max_left _ _)
  rw [project, norm_smul, Real.norm_eq_abs, abs_of_nonneg (div_nonneg hr.le hm.le)]
  calc
    r / max r ‖x‖ * ‖x‖ ≤ r / max r ‖x‖ * max r ‖x‖ :=
      mul_le_mul_of_nonneg_left (le_max_right _ _) (div_nonneg hr.le hm.le)
    _ = r := div_mul_cancel₀ _ (ne_of_gt hm)

/-- Projection variational inequality, proved for the implemented radial map. -/
theorem variational {r : ℝ} (hr : 0 < r) (x z : E) (hz : ‖z‖ ≤ r) :
    inner ℝ (x - project r x) (z - project r x) ≤ 0 := by
  by_cases hx : ‖x‖ ≤ r
  · rw [identity_inside hr x hx, sub_self, inner_zero_left]
  · have hn : 0 < ‖x‖ := lt_trans hr (lt_of_not_ge hx)
    have ha : 0 ≤ r / ‖x‖ := div_nonneg hr.le hn.le
    have hb : r / ‖x‖ ≤ 1 := (div_le_one hn).2 (le_of_lt (lt_of_not_ge hx))
    have hc : inner ℝ x z ≤ ‖x‖ * r :=
      (real_inner_le_norm x z).trans (mul_le_mul_of_nonneg_left hz (norm_nonneg x))
    have he : r / ‖x‖ * ‖x‖ ^ 2 = ‖x‖ * r := by field_simp
    have hsmall : inner ℝ x z - r / ‖x‖ * ‖x‖ ^ 2 ≤ 0 := by
      rw [he]
      exact sub_nonpos.mpr hc
    have hg := mul_nonpos_of_nonneg_of_nonpos (sub_nonneg.mpr hb) hsmall
    simp only [project, max_eq_right (le_of_lt (lt_of_not_ge hx)),
      inner_sub_left, inner_sub_right, real_inner_smul_left,
      real_inner_smul_right, real_inner_self_eq_norm_sq,
      norm_smul, Real.norm_eq_abs, abs_of_nonneg ha]
    nlinarith

/-- The same norm bound permits a one-Lipschitz map, without 1/sqrt(epsilon). -/
theorem nonexpansive {r : ℝ} (hr : 0 < r) (x y : E) :
    ‖project r x - project r y‖ ≤ ‖x - y‖ := by
  have hxy := variational hr x (project r y) (radius_bound hr y)
  have hyx := variational hr y (project r x) (radius_bound hr x)
  have hinner : ‖project r x - project r y‖ ^ 2 ≤
      inner ℝ (x - y) (project r x - project r y) := by
    rw [← real_inner_self_eq_norm_sq]
    simp only [inner_sub_left, inner_sub_right] at hxy hyx ⊢
    linarith
  have hc := real_inner_le_norm (x - y) (project r x - project r y)
  have hn := norm_nonneg (project r x - project r y)
  by_cases hzero : ‖project r x - project r y‖ = 0
  · rw [hzero]; exact norm_nonneg _
  · have hpos : 0 < ‖project r x - project r y‖ := lt_of_le_of_ne hn (Ne.symm hzero)
    nlinarith

/-- A fixed post-projection scalar contributes exactly its absolute magnitude. -/
theorem scaled_sensitivity {r : ℝ} (hr : 0 < r) (a : ℝ) (x y : E) :
    ‖a • project r x - a • project r y‖ ≤ |a| * ‖x - y‖ := by
  rw [← smul_sub, norm_smul, Real.norm_eq_abs]
  exact mul_le_mul_of_nonneg_left (nonexpansive hr x y) (abs_nonneg a)

/-- One residual step has an additive radius bound, conditional on its gate. -/
theorem residual_radius {r : ℝ} (hr : 0 < r) (a : ℝ) (x z : E) :
    ‖x + a • project r z‖ ≤ ‖x‖ + |a| * r := by
  calc
    ‖x + a • project r z‖ ≤ ‖x‖ + ‖a • project r z‖ := norm_add_le _ _
    _ = ‖x‖ + |a| * ‖project r z‖ := by rw [norm_smul, Real.norm_eq_abs]
    _ ≤ ‖x‖ + |a| * r := by
      have h := mul_le_mul_of_nonneg_left (radius_bound hr z) (abs_nonneg a)
      linarith

-- Radius 4 corresponds to RMS radius 1 in the actual width-16 learner.
example (x : E) (hx : ‖x‖ ≤ 4) : project 4 x = x := identity_inside (by norm_num) x hx
example (x : E) : ‖project 4 x‖ ≤ 4 := radius_bound (by norm_num) x
example (x z : E) (hz : ‖z‖ ≤ 4) :
    inner ℝ (x - project 4 x) (z - project 4 x) ≤ 0 := variational (by norm_num) x z hz
example (x y : E) : ‖project 4 x - project 4 y‖ ≤ ‖x - y‖ :=
  nonexpansive (by norm_num) x y
example (x y : E) : ‖(1/2 : ℝ) • project 4 x - (1/2 : ℝ) • project 4 y‖ ≤
    |(1/2 : ℝ)| * ‖x - y‖ := scaled_sensitivity (by norm_num) _ x y
example (x z : E) : ‖x + (1/2 : ℝ) • project 4 z‖ ≤ ‖x‖ + |(1/2 : ℝ)| * 4 :=
  residual_radius (by norm_num) _ x z

#print axioms identity_inside
#print axioms radius_bound
#print axioms variational
#print axioms nonexpansive
#print axioms scaled_sensitivity
#print axioms residual_radius

end ResidualProjection
