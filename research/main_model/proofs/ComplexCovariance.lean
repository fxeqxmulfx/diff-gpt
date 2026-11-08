import Mathlib

/-! Local exact-real algebra for COMPLEX_COVARIANCE_PLAN.txt. These statements
do not assert floating-point accuracy, model convergence, or a Rust refinement. -/
namespace MainModelComplexCovariance
noncomputable section

def eigenReg (a b e : ℝ) : ℝ := (31*a+b)/32+e

theorem eigen_positive (a b e : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b)
    (he : 0 < e) : 0 < eigenReg a b e := by
  unfold eigenReg
  positivity

theorem condition_bound (a b e : ℝ) (hb : 0 ≤ b) (he : 0 ≤ e) :
    eigenReg a b e ≤ 31*eigenReg b a e := by
  unfold eigenReg
  linarith

theorem condition_ratio (a b e : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b)
    (he : 0 < e) : eigenReg a b e / eigenReg b a e ≤ 31 := by
  exact (div_le_iff₀ (eigen_positive b a e hb ha he)).2
    (condition_bound a b e hb he.le)

theorem trace_gap_identity (a b e : ℝ) :
    2*eigenReg a b e*eigenReg b a e -
      (a*eigenReg b a e+b*eigenReg a b e) =
      15/512*(a-b)^2+e*(a+b)+2*e^2 := by
  unfold eigenReg
  ring

/-- Given nonnegative eigenvalues of the uncentered moment, the average real
coordinate energy of the normalized vector is at most one. -/
theorem normalized_trace_bound (a b e : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b)
    (he : 0 < e) : a/eigenReg a b e+b/eigenReg b a e ≤ 2 := by
  have hp := eigen_positive a b e ha hb he
  have hq := eigen_positive b a e hb ha he
  rw [div_add_div _ _ (ne_of_gt hp) (ne_of_gt hq)]
  apply (div_le_iff₀ (mul_pos hp hq)).2
  have hg := trace_gap_identity a b e
  have hz : 0 ≤ 15/512*(a-b)^2+e*(a+b)+2*e^2 := by positivity
  nlinarith

theorem isotropic (a e : ℝ) : eigenReg a a e = a+e := by
  unfold eigenReg
  ring

def momentA (a b c u v : ℝ) : ℝ := u^2*a-2*u*v*b+v^2*c
def momentB (a b c u v : ℝ) : ℝ := u*v*(a-c)+(u^2-v^2)*b
def momentC (a b c u v : ℝ) : ℝ := v^2*a+2*u*v*b+u^2*c

theorem rotated_trace (a b c u v : ℝ) (h : u^2+v^2=1) :
    momentA a b c u v+momentC a b c u v=a+c := by
  unfold momentA momentC
  nlinarith [congrArg (fun z : ℝ => z*a) h, congrArg (fun z : ℝ => z*c) h]

theorem regularizer_rotation (a b c e u v : ℝ) (h : u^2+v^2=1) :
    momentA (eigenReg a c e) (15/16*b) (eigenReg c a e) u v =
      eigenReg (momentA a b c u v) (momentC a b c u v) e := by
  unfold momentA momentC eigenReg
  nlinarith [congrArg (fun z : ℝ => z*e) h]

/-- The adjugate formula used by the kernel squares to M^-1. Positive-root
hypotheses select the positive symmetric inverse square root in applications. -/
theorem inverse_sqrt_diagonal (a b c s t : ℝ)
    (hs : s^2=a*c-b^2) (ht : t^2=a+c+2*s)
    (hs0 : s ≠ 0) (ht0 : t ≠ 0) :
    ((c+s)^2+b^2)/(s*t)^2 = c/s^2 := by
  field_simp
  nlinarith [congrArg (fun z : ℝ => c*z) ht]

theorem inverse_sqrt_off_diagonal (a b c s t : ℝ)
    (ht : t^2=a+c+2*s) (hs0 : s ≠ 0) (ht0 : t ≠ 0) :
    (-b*(a+c+2*s))/(s*t)^2 = -b/s^2 := by
  rw [← ht]
  field_simp

theorem rank_one_regularization (a e : ℝ) :
    eigenReg a 0 e = 31*a/32+e ∧ eigenReg 0 a e = a/32+e := by
  constructor <;> unfold eigenReg <;> ring

end
end MainModelComplexCovariance

#print axioms MainModelComplexCovariance.eigen_positive
#print axioms MainModelComplexCovariance.condition_ratio
#print axioms MainModelComplexCovariance.normalized_trace_bound
#print axioms MainModelComplexCovariance.regularizer_rotation
#print axioms MainModelComplexCovariance.inverse_sqrt_diagonal
