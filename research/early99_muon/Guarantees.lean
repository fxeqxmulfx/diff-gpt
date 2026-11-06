import Transformer.Muon.Section2_Approximation
import Mathlib.Tactic

/-!
Finite quintic bounds for the exact real coefficients used in this experiment.
A nonnegative singular coefficient initially at most one remains at most5/4.
This is not orthogonalization convergence or a finite-precision theorem.
The Bernstein certificates below are explicit polynomial identities checked by ring.
-/
namespace Early99Muon
open Transformer.Muon

noncomputable def quintic (x : ℝ) : ℝ :=
  schulzPolynomial (34445/10000) (-47750/10000) (20315/10000) x

noncomputable def bernstein5 (a b c d e f t : ℝ) : ℝ :=
  a * (1-t)^5 + 5*b*t*(1-t)^4 + 10*c*t^2*(1-t)^3 +
    10*d*t^3*(1-t)^2 + 5*e*t^4*(1-t) + f*t^5

/-- Nonnegative Bernstein coefficients certify nonnegativity on the unit interval. -/
theorem bernstein5_nonneg (a b c d e f t : ℝ)
    (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hd : 0 ≤ d)
    (he : 0 ≤ e) (hf : 0 ≤ f) (ht0 : 0 ≤ t) (ht1 : t ≤ 1) :
    0 ≤ bernstein5 a b c d e f t := by
  have ht : 0 ≤ 1-t := by linarith
  unfold bernstein5
  positivity

example : 0 ≤ bernstein5 0 0 0 0 0 0 0 :=
  bernstein5_nonneg 0 0 0 0 0 0 0
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- Exact Bernstein certificate on interval0, [0,1/2]. -/
theorem interval0 (t : ℝ) (h0 : 0 ≤ t) (h1 : t ≤ 1) :
    0 ≤ quintic (0 + (1/2)*t) ∧ quintic (0 + (1/2)*t) ≤ 5/4 := by
  have hlow := bernstein5_nonneg 0 (6889/20000) (6889/10000) (77893/80000) (22781/20000) (76087/64000) t
    (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) h0 h1
  have elow : quintic (0 + (1/2)*t) =
      bernstein5 0 (6889/20000) (6889/10000) (77893/80000) (22781/20000) (76087/64000) t := by
    unfold quintic schulzPolynomial bernstein5
    ring
  rw [← elow] at hlow
  have hhigh := bernstein5_nonneg (5/4) (18111/20000) (5611/10000) (22107/80000) (2219/20000) (3913/64000) t
    (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) h0 h1
  have ehigh : (5/4:ℝ) - quintic (0 + (1/2)*t) =
      bernstein5 (5/4) (18111/20000) (5611/10000) (22107/80000) (2219/20000) (3913/64000) t := by
    unfold quintic schulzPolynomial bernstein5
    ring
  rw [← ehigh] at hhigh
  constructor <;> linarith

example : 0 ≤ quintic 0 ∧ quintic 0 ≤ 5/4 := by
  convert interval0 0 (by norm_num) (by norm_num) using 1 <;> norm_num

/-- Exact Bernstein certificate on interval1, [1/2,3/4]. -/
theorem interval1 (t : ℝ) (h0 : 0 ≤ t) (h1 : t ≤ 1) :
    0 ≤ quintic ((1/2) + (1/4)*t) ∧ quintic ((1/2) + (1/4)*t) ≤ 5/4 := by
  have hlow := bernstein5_nonneg (76087/64000) (776809/640000) (1548511/1280000) (3014053/2560000) (5739403/5120000) (2152461/2048000) t
    (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) h0 h1
  have elow : quintic ((1/2) + (1/4)*t) =
      bernstein5 (76087/64000) (776809/640000) (1548511/1280000) (3014053/2560000) (5739403/5120000) (2152461/2048000) t := by
    unfold quintic schulzPolynomial bernstein5
    ring
  rw [← elow] at hlow
  have hhigh := bernstein5_nonneg (3913/64000) (23191/640000) (51489/1280000) (185947/2560000) (660597/5120000) (407539/2048000) t
    (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) h0 h1
  have ehigh : (5/4:ℝ) - quintic ((1/2) + (1/4)*t) =
      bernstein5 (3913/64000) (23191/640000) (51489/1280000) (185947/2560000) (660597/5120000) (407539/2048000) t := by
    unfold quintic schulzPolynomial bernstein5
    ring
  rw [← ehigh] at hhigh
  constructor <;> linarith

example : 0 ≤ quintic (1/2) ∧ quintic (1/2) ≤ 5/4 := by
  convert interval1 0 (by norm_num) (by norm_num) using 1 <;> norm_num

/-- Exact Bernstein certificate on interval2, [3/4,1]. -/
theorem interval2 (t : ℝ) (h0 : 0 ≤ t) (h1 : t ≤ 1) :
    0 ≤ quintic ((3/4) + (1/4)*t) ∧ quintic ((3/4) + (1/4)*t) ≤ 5/4 := by
  have hlow := bernstein5_nonneg (2152461/2048000) (2511451/2560000) (143597/160000) (64859/80000) (14743/20000) (701/1000) t
    (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) h0 h1
  have elow : quintic ((3/4) + (1/4)*t) =
      bernstein5 (2152461/2048000) (2511451/2560000) (143597/160000) (64859/80000) (14743/20000) (701/1000) t := by
    unfold quintic schulzPolynomial bernstein5
    ring
  rw [← elow] at hlow
  have hhigh := bernstein5_nonneg (407539/2048000) (688549/2560000) (56403/160000) (35141/80000) (10257/20000) (549/1000) t
    (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) h0 h1
  have ehigh : (5/4:ℝ) - quintic ((3/4) + (1/4)*t) =
      bernstein5 (407539/2048000) (688549/2560000) (56403/160000) (35141/80000) (10257/20000) (549/1000) t := by
    unfold quintic schulzPolynomial bernstein5
    ring
  rw [← ehigh] at hhigh
  constructor <;> linarith

example : 0 ≤ quintic (3/4) ∧ quintic (3/4) ≤ 5/4 := by
  convert interval2 0 (by norm_num) (by norm_num) using 1 <;> norm_num

/-- Exact Bernstein certificate on interval3, [1,5/4]. -/
theorem interval3 (t : ℝ) (h0 : 0 ≤ t) (h1 : t ≤ 1) :
    0 ≤ quintic (1 + (1/4)*t) ∧ quintic (1 + (1/4)*t) ≤ 5/4 := by
  have hlow := bernstein5_nonneg (701/1000) (13297/20000) (53291/80000) (116663/160000) (2268379/2560000) (482959/409600) t
    (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) h0 h1
  have elow : quintic (1 + (1/4)*t) =
      bernstein5 (701/1000) (13297/20000) (53291/80000) (116663/160000) (2268379/2560000) (482959/409600) t := by
    unfold quintic schulzPolynomial bernstein5
    ring
  rw [← elow] at hlow
  have hhigh := bernstein5_nonneg (549/1000) (11703/20000) (46709/80000) (83337/160000) (931621/2560000) (29041/409600) t
    (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) h0 h1
  have ehigh : (5/4:ℝ) - quintic (1 + (1/4)*t) =
      bernstein5 (549/1000) (11703/20000) (46709/80000) (83337/160000) (931621/2560000) (29041/409600) t := by
    unfold quintic schulzPolynomial bernstein5
    ring
  rw [← ehigh] at hhigh
  constructor <;> linarith

example : 0 ≤ quintic 1 ∧ quintic 1 ≤ 5/4 := by
  convert interval3 0 (by norm_num) (by norm_num) using 1 <;> norm_num

/-- The implemented quintic preserves [0,5/4], not the single value one. -/
theorem quintic_bounds (x : ℝ) (h0 : 0 ≤ x) (h1 : x ≤ 5/4) :
    0 ≤ quintic x ∧ quintic x ≤ 5/4 := by
  rcases le_total x (1/2) with h | h
  · have bound := interval0 (2*x) (by linarith) (by linarith)
    have e : (0:ℝ) + (1/2)*(2*x) = x := by ring
    rw [e] at bound
    exact bound
  · rcases le_total x (3/4) with h' | h'
    · have bound := interval1 (4*x-2) (by linarith) (by linarith)
      have e : (1/2:ℝ) + (1/4)*(4*x-2) = x := by ring
      rw [e] at bound
      exact bound
    · rcases le_total x 1 with h'' | h''
      · have bound := interval2 (4*x-3) (by linarith) (by linarith)
        have e : (3/4:ℝ) + (1/4)*(4*x-3) = x := by ring
        rw [e] at bound
        exact bound
      · have bound := interval3 (4*x-4) (by linarith) (by linarith)
        have e : (1:ℝ) + (1/4)*(4*x-4) = x := by ring
        rw [e] at bound
        exact bound

/-- Every finite scalar iteration stays bounded from any normalized coefficient. -/
theorem finite_scalar_bounds (x : ℝ) (h0 : 0 ≤ x) (h1 : x ≤ 1) (n : ℕ) :
    0 ≤ scalarSchulz (34445/10000) (-47750/10000) (20315/10000) x n ∧
    scalarSchulz (34445/10000) (-47750/10000) (20315/10000) x n ≤ 5/4 := by
  induction n with
  | zero => simpa [scalarSchulz] using And.intro h0 (by linarith : x ≤ 5/4)
  | succ n ih => exact quintic_bounds _ ih.1 ih.2

/-- A unit singular coefficient is not fixed by these actual coefficients. -/
theorem unit_not_fixed : quintic 1 = 701/1000 ∧ quintic 1 ≠ 1 := by
  norm_num [quintic, schulzPolynomial]

example := quintic_bounds (1/2) (by norm_num) (by norm_num)
example := finite_scalar_bounds (1/2) (by norm_num) (by norm_num) 5

#print axioms quintic_bounds
#print axioms finite_scalar_bounds
#print axioms unit_not_fixed
#print axioms Transformer.Muon.schulzStep_spectrum
#print axioms Transformer.Muon.schulzIterate_zero
#print axioms Transformer.Muon.printed_step_identity
end Early99Muon
