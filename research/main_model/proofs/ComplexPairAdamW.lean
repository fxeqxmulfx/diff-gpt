import Mathlib

/-!
Exact-real consequences of COMPLEX_PAIR_ADAMW_PLAN.txt. These are local
optimizer statements, not floating-point or whole-network convergence claims.
Coordinates denote independent real/imaginary parameters after contraction.
-/
namespace MainModelComplexPairAdamW
noncomputable section
open scoped BigOperators
abbrev Pair := ℝ × ℝ
def rotate (c s : ℝ) (x : Pair) : Pair :=
  (c * x.1 - s * x.2, s * x.1 + c * x.2)
def energy (g : Pair) : ℝ := (g.1 ^ 2 + g.2 ^ 2) / 2
def clipped (d : ℝ) (g : Pair) : Pair := (g.1 / d, g.2 / d)
def first (b : ℝ) (m g : Pair) : Pair :=
  (b * m.1 + (1 - b) * g.1, b * m.2 + (1 - b) * g.2)
def second (b v : ℝ) (g : Pair) : ℝ := b * v + (1 - b) * energy g
def finish (lr wd bc1 den : ℝ) (p m : Pair) : Pair :=
  ((1 - lr * wd) * p.1 - (lr / bc1) * m.1 / den,
   (1 - lr * wd) * p.2 - (lr / bc1) * m.2 / den)
def step (lr wd bc1 bc2sqrt eps : ℝ) (p m : Pair) (v : ℝ) (g : Pair) : Pair :=
  finish lr wd bc1 (Real.sqrt (second (19/20) v g) / bc2sqrt + eps)
    p (first (9/10) m g)

theorem energy_nonnegative (g : Pair) : 0 ≤ energy g := by
  exact div_nonneg (add_nonneg (sq_nonneg _) (sq_nonneg _)) (by norm_num)

theorem energy_rotate (c s : ℝ) (g : Pair) (h : c^2 + s^2 = 1) :
    energy (rotate c s g) = energy g := by
  have algebra : energy (rotate c s g) = (c^2+s^2) * energy g := by
    dsimp [energy, rotate]
    ring
  rw [algebra, h, one_mul]
example : (0 : ℝ)^2 + 1^2 = 1 := by norm_num

theorem global_squared_norm_rotate {n : ℕ} (g : Fin n → Pair)
    (c s : Fin n → ℝ) (h : ∀ i, (c i)^2+(s i)^2=1) :
    (∑ i, 2 * energy (rotate (c i) (s i) (g i))) = ∑ i, 2 * energy (g i) := by
  apply Finset.sum_congr rfl
  intro i _
  rw [energy_rotate _ _ _ (h i)]
example : ∀ _i : Fin 3, (0 : ℝ)^2 + 1^2 = 1 := by intro i; norm_num

theorem clip_rotate (c s d : ℝ) (g : Pair) :
    clipped d (rotate c s g) = rotate c s (clipped d g) := by
  apply Prod.ext <;> dsimp [clipped, rotate] <;> ring

theorem first_rotate (c s b : ℝ) (m g : Pair) :
    first b (rotate c s m) (rotate c s g) = rotate c s (first b m g) := by
  apply Prod.ext <;> dsimp [first, rotate] <;> ring

theorem second_rotate (c s b v : ℝ) (g : Pair) (h : c^2+s^2=1) :
    second b v (rotate c s g) = second b v g := by
  simp only [second, energy_rotate _ _ _ h]
example : (0 : ℝ)^2 + 1^2 = 1 := by norm_num

theorem second_nonnegative (b v : ℝ) (g : Pair)
    (hb : 0 ≤ b) (hb1 : b ≤ 1) (hv : 0 ≤ v) : 0 ≤ second b v g := by
  exact add_nonneg (mul_nonneg hb hv)
    (mul_nonneg (sub_nonneg.mpr hb1) (energy_nonnegative g))
example : (0 : ℝ) ≤ 19/20 ∧ (19/20 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1/10 := by norm_num

theorem denominator_positive (v bc2sqrt eps : ℝ) (hb : 0 < bc2sqrt)
    (he : 0 < eps) : 0 < Real.sqrt v / bc2sqrt + eps := by
  exact add_pos_of_nonneg_of_pos (div_nonneg (Real.sqrt_nonneg _) hb.le) he
example : (0 : ℝ) < 1 ∧ (0 : ℝ) < 1/100000000 := by norm_num

theorem finish_rotate (c s lr wd bc1 den : ℝ) (p m : Pair) :
    finish lr wd bc1 den (rotate c s p) (rotate c s m) =
    rotate c s (finish lr wd bc1 den p m) := by
  apply Prod.ext <;> dsimp [finish, rotate] <;> ring

theorem step_rotate (c s lr wd bc1 bc2sqrt eps : ℝ)
    (p m : Pair) (v : ℝ) (g : Pair) (h : c^2+s^2=1) :
    step lr wd bc1 bc2sqrt eps (rotate c s p) (rotate c s m) v (rotate c s g) =
    rotate c s (step lr wd bc1 bc2sqrt eps p m v g) := by
  simp only [step, first_rotate, second_rotate _ _ _ _ _ h, finish_rotate]
example : (0 : ℝ)^2 + 1^2 = 1 := by norm_num

#print axioms energy_nonnegative
#print axioms energy_rotate
#print axioms global_squared_norm_rotate
#print axioms clip_rotate
#print axioms first_rotate
#print axioms second_rotate
#print axioms second_nonnegative
#print axioms denominator_positive
#print axioms finish_rotate
#print axioms step_rotate
end
end MainModelComplexPairAdamW
