import Mathlib.Analysis.Real.Sqrt
import Mathlib.Analysis.Complex.Exponential
import Mathlib.Tactic

/-!
Exact-real local algebra for COMPLEX_MULTIPLICATIVE_GATE_PLAN.txt.
The gate is sigmoid(u) * (1 + i*v) / sqrt(1+v^2). The claims concern its
norm and explicit local derivative coefficients, not floating-point rounding,
an input-dependent whole transformer, or convergence. Native scalar AD checks
the implemented derivatives independently. No formal HasFDerivAt claim is made.
-/
namespace MainModelComplexGate
noncomputable section

def amplitude (u : ℝ) : ℝ := 1 / (1 + Real.exp (-u))
def phaseRe (v : ℝ) : ℝ := 1 / Real.sqrt (1 + v^2)
def phaseIm (v : ℝ) : ℝ := v / Real.sqrt (1 + v^2)
def gateRe (u v : ℝ) : ℝ := amplitude u * phaseRe v
def gateIm (u v : ℝ) : ℝ := amplitude u * phaseIm v

/-- The registered sigmoid amplitude lies strictly between zero and one. -/
theorem amplitude_bounds (u : ℝ) : 0 < amplitude u ∧ amplitude u < 1 := by
  have he := Real.exp_pos (-u)
  unfold amplitude
  constructor
  · positivity
  · apply (div_lt_one (by positivity : 0 < 1 + Real.exp (-u))).2
    linarith

/-- The regularized phase denominator cannot vanish. -/
theorem denominator_positive (v : ℝ) : 0 < Real.sqrt (1 + v^2) := by
  apply Real.sqrt_pos.2
  positivity

/-- The registered real/imaginary phase coordinates have unit squared norm. -/
theorem phase_unit (v : ℝ) : phaseRe v ^ 2 + phaseIm v ^ 2 = 1 := by
  have hs := Real.sq_sqrt (by positivity : 0 ≤ 1 + v^2)
  have hn : Real.sqrt (1 + v^2) ≠ 0 := ne_of_gt (denominator_positive v)
  unfold phaseRe phaseIm
  field_simp
  nlinarith

/-- Gate magnitude equals the sigmoid amplitude in squared norm. -/
theorem gate_norm_identity (u v : ℝ) : gateRe u v ^ 2 + gateIm u v ^ 2 = amplitude u ^ 2 := by
  have h := phase_unit v
  unfold gateRe gateIm
  calc
    (amplitude u * phaseRe v)^2 + (amplitude u * phaseIm v)^2 =
        amplitude u ^ 2 * (phaseRe v ^ 2 + phaseIm v ^ 2) := by ring
    _ = amplitude u ^ 2 := by rw [h]; ring

/-- A fixed registered gate cannot increase a complex input pair's norm. -/
theorem multiplication_nonexpansive (u v ar ai : ℝ) :
    (gateRe u v * ar - gateIm u v * ai)^2 +
      (gateIm u v * ar + gateRe u v * ai)^2 ≤ ar^2 + ai^2 := by
  have hb := amplitude_bounds u
  have hg := gate_norm_identity u v
  have ha : amplitude u ^ 2 ≤ 1 := by nlinarith
  have hn : 0 ≤ ar^2 + ai^2 := by positivity
  calc
    _ = (gateRe u v ^ 2 + gateIm u v ^ 2) * (ar^2 + ai^2) := by ring
    _ = amplitude u ^ 2 * (ar^2 + ai^2) := by rw [hg]
    _ ≤ ar^2 + ai^2 := by nlinarith [mul_nonneg (sub_nonneg.mpr ha) hn]

/-- Algebraic bounds for the sigmoid and normalized-phase derivative factors. -/
theorem coefficient_bounds (u v : ℝ) :
    0 ≤ amplitude u * (1-amplitude u) ∧
      amplitude u * (1-amplitude u) ≤ 1/4 ∧
      0 ≤ amplitude u / (1+v^2) ∧ amplitude u / (1+v^2) ≤ 1 := by
  have hb := amplitude_bounds u
  have ha : 0 ≤ amplitude u := hb.1.le
  have hv : 0 ≤ v^2 := sq_nonneg v
  refine ⟨by nlinarith, ?_, by positivity, ?_⟩
  · nlinarith [sq_nonneg (amplitude u - 1/2)]
  · apply (div_le_one (by positivity : 0 < 1+v^2)).2
    linarith

/-- Explicit local Jacobian coefficients have operator norm at most one.
This algebraic statement does not assert a formal differentiation theorem. -/
theorem local_coefficients_bound (u v du dv : ℝ) :
    let a := du * (amplitude u * (1-amplitude u))
    let b := dv * (amplitude u / (1+v^2))
    (a * phaseRe v - b * phaseIm v)^2 +
      (a * phaseIm v + b * phaseRe v)^2 ≤ du^2/16 + dv^2 := by
  dsimp
  have hp := phase_unit v
  obtain ⟨ha0, ha1, hb0, hb1⟩ := coefficient_bounds u v
  have ha2 : (amplitude u * (1-amplitude u))^2 ≤ 1/16 := by nlinarith
  have hb2 : (amplitude u / (1+v^2))^2 ≤ 1 := by nlinarith
  have hx := mul_nonneg (sq_nonneg du) (sub_nonneg.mpr ha2)
  have hy := mul_nonneg (sq_nonneg dv) (sub_nonneg.mpr hb2)
  calc
    _ = ((du * (amplitude u * (1-amplitude u)))^2 +
        (dv * (amplitude u / (1+v^2)))^2) *
        (phaseRe v ^ 2 + phaseIm v ^ 2) := by ring
    _ ≤ du^2/16 + dv^2 := by rw [hp]; nlinarith

-- A concrete nonzero input example, with no additional hypotheses.
example : (gateRe 0 1 * 2 - gateIm 0 1 * 3)^2 +
    (gateIm 0 1 * 2 + gateRe 0 1 * 3)^2 ≤ 13 := by
  have h := multiplication_nonexpansive 0 1 2 3
  norm_num at h ⊢
  exact h

#print axioms amplitude_bounds
#print axioms denominator_positive
#print axioms phase_unit
#print axioms gate_norm_identity
#print axioms multiplication_nonexpansive
#print axioms coefficient_bounds
#print axioms local_coefficients_bound
end
end MainModelComplexGate
