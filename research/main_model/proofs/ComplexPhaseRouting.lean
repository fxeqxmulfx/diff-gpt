import Mathlib
import Transformer.Residual.Basic

/-!
Exact-arithmetic guarantees for COMPLEX_PHASE_ROUTING_PLAN.txt. Coefficients
are frozen at a token. These do not bound the input-dependent router Jacobian,
prove finite Sinkhorn double stochasticity, or establish FP32/convergence claims.
The row-bound proof uses Transformer.Residual.weightedSum_convex_norm_le;
the phase conjugation is the independently specified new construction.
-/
namespace MainModelComplexPhaseRouting
noncomputable section
open scoped BigOperators
open ComplexConjugate
variable {n : ℕ}

def phase (a : ℝ) : ℂ :=
  ⟨1 / Real.sqrt (1 + a^2), a / Real.sqrt (1 + a^2)⟩

theorem phase_normSq (a : ℝ) : Complex.normSq (phase a) = 1 := by
  have hp : 0 < 1 + a^2 := by positivity
  have hs := Real.sq_sqrt hp.le
  have hn : Real.sqrt (1 + a^2) ≠ 0 := ne_of_gt (Real.sqrt_pos.2 hp)
  dsimp [phase]
  field_simp [hn]
  nlinarith

theorem phase_norm (a : ℝ) : ‖phase a‖ = 1 := by
  rw [Complex.norm_def, phase_normSq, Real.sqrt_one]

def action (u x : Fin n → ℂ) : Fin n → ℂ := fun i => u i * x i
def undo (u x : Fin n → ℂ) : Fin n → ℂ := fun i => conj (u i) * x i
def energy (x : Fin n → ℂ) : ℝ := ∑ i, ‖x i‖^2
def routed (P : Matrix (Fin n) (Fin n) ℝ) (u x : Fin n → ℂ) : Fin n → ℂ :=
  action u (Transformer.Residual.mix P (undo u x))

theorem action_energy (u x : Fin n → ℂ) (hu : ∀ i, ‖u i‖ = 1) :
    energy (action u x) = energy x := by
  unfold energy
  apply Finset.sum_congr rfl
  intro i _
  rw [action, norm_mul, hu i, one_mul]
example : ∀ i : Fin 4, ‖phase (i : ℝ)‖ = 1 := fun _ => phase_norm _

theorem undo_energy (u x : Fin n → ℂ) (hu : ∀ i, ‖u i‖ = 1) :
    energy (undo u x) = energy x := by
  unfold energy
  apply Finset.sum_congr rfl
  intro i _
  rw [undo, norm_mul, Complex.norm_conj, hu i, one_mul]
example : ∀ i : Fin 4, ‖phase (i : ℝ)‖ = 1 := fun _ => phase_norm _

theorem undo_action (u x : Fin n → ℂ) (hu : ∀ i, ‖u i‖ = 1) :
    undo u (action u x) = x := by
  funext i
  dsimp [undo, action]
  rw [← mul_assoc, Complex.conj_mul', hu i]
  norm_num
example : ∀ i : Fin 4, ‖phase (i : ℝ)‖ = 1 := fun _ => phase_norm _

/-- Exact equivalence of every frozen-map squared-energy upper bound. This
does not assume that the implemented finite Sinkhorn matrix is doubly stochastic. -/
theorem energy_bound_iff (P : Matrix (Fin n) (Fin n) ℝ) (u : Fin n → ℂ)
    (C : ℝ) (hu : ∀ i, ‖u i‖ = 1) :
    (∀ x, energy (routed P u x) ≤ C * energy x) ↔
    (∀ x, energy (Transformer.Residual.mix P x) ≤ C * energy x) := by
  constructor
  · intro h x
    have hx := h (action u x)
    simpa only [routed, undo_action u x hu, action_energy u _ hu] using hx
  · intro h x
    unfold routed
    rw [action_energy u _ hu]
    simpa only [undo_energy u x hu] using h (undo u x)
example : ∀ i : Fin 4, ‖phase (i : ℝ)‖ = 1 := fun _ => phase_norm _

/-- Nonnegative row-normalized coefficients retain the maximum-stream norm
bound. No column normalization or ideal Sinkhorn limit is assumed. -/
theorem row_norm_bound (P : Matrix (Fin n) (Fin n) ℝ) (u x : Fin n → ℂ)
    (R : ℝ) (hu : ∀ i, ‖u i‖ = 1) (hp : ∀ i j, 0 ≤ P i j)
    (hs : ∀ i, ∑ j, P i j = 1) (hx : ∀ i, ‖x i‖ ≤ R) (i : Fin n) :
    ‖routed P u x i‖ ≤ R := by
  change ‖u i * Transformer.Residual.weightedSum (P i) (undo u x)‖ ≤ R
  rw [norm_mul, hu i, one_mul]
  apply Transformer.Residual.weightedSum_convex_norm_le _ _ _ (hp i) (hs i)
  intro j
  simpa only [undo, norm_mul, Complex.norm_conj, hu j, one_mul] using hx j
example : (∀ i : Fin 4, ‖phase (i : ℝ)‖ = 1) ∧
    (∀ _i _j : Fin 4, (0 : ℝ) ≤ 1/4) ∧
    (∀ _i : Fin 4, ∑ _j : Fin 4, (1/4 : ℝ) = 1) ∧
    (∀ _i : Fin 4, ‖(1 : ℂ)‖ ≤ 1) := by
  exact ⟨fun _ => phase_norm _, by norm_num, by norm_num, by norm_num⟩

/-- A phase-aligned constant mode replaces ordinary identical-stream
conservation for this conjugated skip map. Write-back is outside this claim. -/
theorem aligned_mode (P : Matrix (Fin n) (Fin n) ℝ) (u : Fin n → ℂ)
    (z : ℂ) (hu : ∀ i, ‖u i‖ = 1) (hs : ∀ i, ∑ j, P i j = 1) :
    routed P u (action u (fun _ => z)) = action u (fun _ => z) := by
  unfold routed
  rw [undo_action u _ hu, Transformer.Residual.mix_const P z hs]
example : (∀ i : Fin 4, ‖phase (i : ℝ)‖ = 1) ∧
    (∀ _i : Fin 4, ∑ _j : Fin 4, (1/4 : ℝ) = 1) := by
  exact ⟨fun _ => phase_norm _, by norm_num⟩

/-- The zero-initialized phases give the original mixer for every input. -/
theorem zero_phase (P : Matrix (Fin n) (Fin n) ℝ) (x : Fin n → ℂ) :
    routed P (fun _ => phase 0) x = Transformer.Residual.mix P x := by
  have h0 : phase 0 = 1 := by apply Complex.ext <;> norm_num [phase]
  have hx : undo (fun _ => phase 0) x = x := by
    funext i
    simp [undo, h0]
  rw [routed, hx]
  funext i
  simp [action, h0]

/-- A strictly positive doubly stochastic two-stream mixer need not preserve
ordinary constant streams after nontrivial finite phase conjugation. -/
theorem ordinary_constant_not_preserved :
    routed (fun _ _ : Fin 2 => (1/2 : ℝ)) ![phase 0, phase 1]
      (fun _ => (1 : ℂ)) 0 ≠ 1 := by
  intro h
  have hi := congrArg Complex.im h
  have hn : Real.sqrt 2 ≠ 0 := ne_of_gt (Real.sqrt_pos.2 (by norm_num))
  norm_num [routed, action, undo, Transformer.Residual.mix,
    Transformer.Residual.weightedSum, phase, Fin.sum_univ_two] at hi

#print axioms phase_normSq
#print axioms phase_norm
#print axioms action_energy
#print axioms undo_energy
#print axioms undo_action
#print axioms energy_bound_iff
#print axioms row_norm_bound
#print axioms aligned_mode
#print axioms zero_phase
#print axioms ordinary_constant_not_preserved
end
end MainModelComplexPhaseRouting
