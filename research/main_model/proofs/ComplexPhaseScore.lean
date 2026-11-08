import Mathlib

/-!
Exact-real local statements for COMPLEX_PHASE_SCORE_PLAN.txt. The current
Hermitian real part is the ordinary real dot. The new score adds a bounded
squared-imaginary term; its unit-disk bound is 5/4, not the parent's 1.
These statements do not establish floating-point or convergence guarantees.
-/
namespace MainModelComplexPhaseScore
noncomputable section
open scoped BigOperators
open ComplexConjugate
variable {n : ℕ}

def hermitian (q k : Fin n → ℂ) : ℂ := ∑ i, conj (q i) * k i
def energy (q : Fin n → ℂ) : ℝ := ∑ i, ‖q i‖^2
def coefficient (a : ℝ) : ℝ := a / Real.sqrt (1+a^2)
def core (β : ℝ) (z : ℂ) : ℝ := z.re + β*z.im^2

theorem hermitian_real (q k : Fin n → ℂ) :
    (hermitian q k).re = ∑ i, ((q i).re*(k i).re + (q i).im*(k i).im) := by
  simp [hermitian, Complex.mul_re]

theorem hermitian_imaginary (q k : Fin n → ℂ) :
    (hermitian q k).im = ∑ i, ((q i).re*(k i).im - (q i).im*(k i).re) := by
  simp [hermitian, Complex.mul_im, sub_eq_add_neg]

theorem coefficient_strict_bound (a : ℝ) : |coefficient a| < 1 := by
  have hs := Real.sq_sqrt (by positivity : 0 ≤ 1+a^2)
  have hp : 0 < Real.sqrt (1+a^2) := Real.sqrt_pos.2 (by positivity)
  have ha : |a| < Real.sqrt (1+a^2) := by nlinarith [sq_abs a]
  simpa [coefficient, abs_div, abs_of_pos hp] using (div_lt_one hp).2 ha

theorem zero_coefficient (z : ℂ) : core (coefficient 0) z = z.re := by
  norm_num [core, coefficient]

/-- Cauchy-Schwarz applies to the complete complex product, not just its real part. -/
theorem hermitian_energy_bound (q k : Fin n → ℂ) :
    ‖hermitian q k‖^2 ≤ energy q * energy k := by
  have ht : ‖hermitian q k‖ ≤ ∑ i, ‖q i‖ * ‖k i‖ := by
    simpa only [hermitian, norm_mul, Complex.norm_conj] using
      norm_sum_le Finset.univ (fun i => conj (q i) * k i)
  have hn : 0 ≤ ∑ i, ‖q i‖ * ‖k i‖ := by positivity
  calc
    _ ≤ (∑ i, ‖q i‖ * ‖k i‖)^2 := by nlinarith [norm_nonneg (hermitian q k)]
    _ ≤ energy q * energy k := Finset.sum_mul_sq_le_sq_mul_sq _ _ _

/-- Q/K RMS and exact unit rotations supply these squared-norm hypotheses.
The positive d is the real head width, not the number of complex coordinates. -/
theorem normalized_hermitian_disk (q k : Fin n → ℂ) (d : ℝ) (hd : 0 < d)
    (hq : energy q ≤ d) (hk : energy k ≤ d) :
    Complex.normSq (hermitian q k / (d : ℂ)) ≤ 1 := by
  have he := hermitian_energy_bound q k
  have hn : 0 ≤ energy k := by unfold energy; positivity
  have hb : energy q * energy k ≤ d*d := mul_le_mul hq hk hn hd.le
  rw [Complex.normSq_eq_norm_sq, norm_div, Complex.norm_real, Real.norm_eq_abs,
    abs_of_pos hd, div_pow]
  apply (div_le_one (sq_pos_of_pos hd)).2
  nlinarith
example : (0 : ℝ) < 2 ∧ energy (fun _ : Fin 1 => (1 : ℂ)) ≤ 2 ∧
    energy (fun _ : Fin 1 => Complex.I) ≤ 2 := by norm_num [energy]

/-- Sharp bound for the enlarged closed family |beta|<=1; the implemented
coefficient has strict |beta|<1. This is a score bound, not a Jacobian claim. -/
theorem core_disk_bound (β : ℝ) (z : ℂ) (hb : |β| ≤ 1)
    (hz : Complex.normSq z ≤ 1) : |core β z| ≤ 5/4 := by
  have hz' : z.re^2 + z.im^2 ≤ 1 := by simpa [Complex.normSq_apply, pow_two] using hz
  have hi : 0 ≤ z.im^2 := sq_nonneg _
  have hm : |β| * z.im^2 ≤ z.im^2 := by nlinarith
  have ha : |z.re| + z.im^2 ≤ 5/4 := by
    nlinarith [sq_nonneg (|z.re|-1/2), sq_abs z.re]
  calc
    |core β z| ≤ |z.re| + |β*z.im^2| := abs_add_le _ _
    _ = |z.re| + |β| * z.im^2 := by rw [abs_mul, abs_of_nonneg hi]
    _ ≤ 5/4 := by linarith
example : |(1/2 : ℝ)| ≤ 1 ∧ Complex.normSq Complex.I ≤ 1 := by norm_num

theorem normalized_score_bound (a : ℝ) (q k : Fin n → ℂ) (d : ℝ)
    (hd : 0 < d) (hq : energy q ≤ d) (hk : energy k ≤ d) :
    |core (coefficient a) (hermitian q k / (d : ℂ))| ≤ 5/4 :=
  core_disk_bound _ _ (coefficient_strict_bound a).le (normalized_hermitian_disk q k d hd hq hk)
example : (0 : ℝ) < 2 ∧ energy (fun _ : Fin 1 => (1 : ℂ)) ≤ 2 ∧
    energy (fun _ : Fin 1 => Complex.I) ≤ 2 := by norm_num [energy]

/-- The unit-disk form is algebraically the registered gain times the ordinary
real dot plus the squared-imaginary correction. No change from real width d. -/
theorem registered_score_formula (N a d : ℝ) (z : ℂ) :
    Real.log N * core (coefficient a) (z / (d : ℂ)) =
      Real.log N / d * (z.re + coefficient a * (z.im/d) * z.im) := by
  by_cases hd : d = 0
  · simp [hd, core]
  · simp [core, div_eq_mul_inv, Complex.mul_re, Complex.mul_im]
    field_simp [hd]

theorem causal_score_bound (N a d : ℝ) (q k : Fin n → ℂ)
    (hN : 1 ≤ N) (hd : 0 < d) (hq : energy q ≤ d) (hk : energy k ≤ d) :
    |Real.log N / d * ((hermitian q k).re +
      coefficient a * ((hermitian q k).im/d) * (hermitian q k).im)| ≤
      (5/4)*Real.log N := by
  rw [← registered_score_formula]
  rw [abs_mul, abs_of_nonneg (Real.log_nonneg hN)]
  calc
    _ ≤ Real.log N * (5/4) := mul_le_mul_of_nonneg_left
      (normalized_score_bound a q k d hd hq hk) (Real.log_nonneg hN)
    _ = _ := mul_comm _ _
example : (1 : ℝ) ≤ 2 ∧ (0 : ℝ) < 2 ∧ energy (fun _ : Fin 1 => (1 : ℂ)) ≤ 2 ∧
    energy (fun _ : Fin 1 => Complex.I) ≤ 2 := by norm_num [energy]

theorem common_phase_hermitian (u : ℂ) (q k : Fin n → ℂ) (hu : ‖u‖ = 1) :
    hermitian (fun i => u*q i) (fun i => u*k i) = hermitian q k := by
  unfold hermitian
  apply Finset.sum_congr rfl
  intro i _
  simp only [map_mul]
  calc
    conj u * conj (q i) * (u*k i) = (conj u*u)*(conj (q i)*k i) := by ring
    _ = conj (q i)*k i := by rw [Complex.conj_mul', hu]; norm_num
example : ‖(1 : ℂ)‖ = 1 := by norm_num

/-- Simultaneous conjugation of Q/K conjugates their Hermitian product.
The imaginary-square score then preserves the real output. -/
theorem conjugate_core (β : ℝ) (z : ℂ) : core β (conj z) = core β z := by
  simp [core]

/-- A reachable nonzero coefficient is not odd in the key. Every bilinear
score is odd in either input; this witness excludes a hidden dot/phase identity. -/
theorem nonlinear_witness :
    core (coefficient 1) Complex.I + core (coefficient 1) (-Complex.I) > 0 := by
  norm_num [core, coefficient]

/-- Exact polynomial remainder identifies the first variation in a complex
score input. This is not presented as a formal full-model derivative theorem. -/
theorem differential_remainder (β t : ℝ) (z dz : ℂ) :
    core β (z + (t : ℂ)*dz) = core β z +
      t*(dz.re + 2*β*z.im*dz.im) + t^2*β*dz.im^2 := by
  simp only [core, Complex.add_re, Complex.add_im, Complex.mul_re, Complex.mul_im,
    Complex.ofReal_re, Complex.ofReal_im, zero_mul, sub_zero, add_zero]
  ring

#print axioms hermitian_real
#print axioms hermitian_imaginary
#print axioms coefficient_strict_bound
#print axioms zero_coefficient
#print axioms hermitian_energy_bound
#print axioms normalized_hermitian_disk
#print axioms core_disk_bound
#print axioms normalized_score_bound
#print axioms registered_score_formula
#print axioms causal_score_bound
#print axioms common_phase_hermitian
#print axioms conjugate_core
#print axioms nonlinear_witness
#print axioms differential_remainder
end
end MainModelComplexPhaseScore
