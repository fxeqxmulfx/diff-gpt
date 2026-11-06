import research.early99_muon_math.Bounds

/-!
Real parameter-growth bounds for the declared finite Muon update. The direction
is the actual five-step polynomial with an epsilon floor, not an exact polar
factor. Positive decay and rate * decay <= 1 are explicit hypotheses. These
statements concern Muon matrices only and do not include floating-point error.
-/
namespace Early99MuonDecay
open Transformer.Muon Early99MuonMath
open scoped BigOperators

variable {a b r : ℕ}

/-- Entrywise squared Frobenius energy is convex; no matrix rank is assumed. -/
theorem energy_convex (t : ℝ) (ht : 0 ≤ t) (ht1 : t ≤ 1)
    (X Y : Matrix (Fin a) (Fin b) ℝ) :
    squaredFrobenius ((1-t) • X + t • Y) ≤
      (1-t) * squaredFrobenius X + t * squaredFrobenius Y := by
  simp only [squaredFrobenius, Matrix.add_apply, Matrix.smul_apply, smul_eq_mul]
  simp only [Finset.mul_sum, ← Finset.sum_add_distrib]
  apply Finset.sum_le_sum
  intro i _
  apply Finset.sum_le_sum
  intro j _
  nlinarith [mul_nonneg (mul_nonneg ht (sub_nonneg.mpr ht1))
    (sq_nonneg (X i j - Y i j))]

/-- Convex combinations preserve a common RMS cap for nonempty matrices. -/
theorem rms_convex_bound (t cap : ℝ) (ht : 0 ≤ t) (ht1 : t ≤ 1)
    (X Y : Matrix (Fin a) (Fin b) ℝ) (ha : 0 < a) (hb : 0 < b)
    (hX : matrixRMS X ≤ cap) (hY : matrixRMS Y ≤ cap) :
    matrixRMS ((1-t) • X + t • Y) ≤ cap := by
  have hcap : 0 ≤ cap := (Real.sqrt_nonneg _).trans hX
  have hab : 0 < (a:ℝ) * b := mul_pos (by exact_mod_cast ha) (by exact_mod_cast hb)
  have hx := (div_le_iff₀ hab).mp (Real.sqrt_le_iff.mp hX).2
  have hy := (div_le_iff₀ hab).mp (Real.sqrt_le_iff.mp hY).2
  have hc := energy_convex t ht ht1 X Y
  unfold matrixRMS
  apply Real.sqrt_le_iff.mpr
  refine ⟨hcap, (div_le_iff₀ hab).mpr ?_⟩
  nlinarith [mul_le_mul_of_nonneg_left hx (sub_nonneg.mpr ht1),
    mul_le_mul_of_nonneg_left hy ht]

/-- The frozen Muon equation preserves any RMS cap at least 1/(4*decay). -/
theorem step_bound (rate decay eps cap : ℝ) (hrate : 0 ≤ rate)
    (hdecay : 0 < decay) (hstep : rate * decay ≤ 1) (heps : 0 < eps)
    (P : Matrix (Fin a) (Fin b) ℝ) (hP : matrixRMS P ≤ cap)
    (hcap : 1/(4*decay) ≤ cap)
    (U : Matrix (Fin a) (Fin r) ℝ) (s : Fin r → ℝ) (V : Matrix (Fin b) (Fin r) ℝ)
    (hU : OrthonormalColumns U) (hV : OrthonormalColumns V) (hs : ∀ k, 0 ≤ s k)
    (ha : 0 < a) (hb : 0 < b) (hra : r ≤ a) (hrb : r ≤ b) :
    matrixRMS (matrixUpdate rate decay eps P (singularMatrix U s V)) ≤ cap := by
  let D := adjustedUpdate (direction eps (singularMatrix U s V))
  have hD : matrixRMS D ≤ 1/4 :=
    adjusted_direction_rms eps heps U s V hU hV hs ha hb hra hrb
  have hQ : matrixRMS ((-decay⁻¹) • D) ≤ cap := by
    rw [matrixRMS_smul, abs_neg, abs_of_pos (inv_pos.mpr hdecay)]
    calc
      _ ≤ decay⁻¹ * (1/4) := mul_le_mul_of_nonneg_left hD (by positivity)
      _ = 1/(4*decay) := by field_simp
      _ ≤ cap := hcap
  have heq : matrixUpdate rate decay eps P (singularMatrix U s V) =
      (1-rate*decay) • P + (rate*decay) • ((-decay⁻¹) • D) := by
    have hc : (rate*decay)*(-decay⁻¹) = -rate := by
      field_simp
    rw [smul_smul, hc, neg_smul]
    rfl
  rw [heq]
  exact rms_convex_bound (rate*decay) cap (mul_nonneg hrate hdecay.le)
    hstep P _ ha hb hP hQ

noncomputable def trajectory (rate : ℕ → ℝ) (decay eps : ℝ)
    (P : Matrix (Fin a) (Fin b) ℝ) (H : ℕ → Matrix (Fin a) (Fin b) ℝ) :
    ℕ → Matrix (Fin a) (Fin b) ℝ
  | 0 => P
  | n+1 => matrixUpdate (rate n) decay eps (trajectory rate decay eps P H n) (H n)

/-- Every real iterate stays below max(initial RMS, 1/(4*decay)), including warmup. -/
theorem trajectory_bound (rate : ℕ → ℝ) (decay eps : ℝ)
    (hrate : ∀ n, 0 ≤ rate n) (hdecay : 0 < decay)
    (hstep : ∀ n, rate n * decay ≤ 1) (heps : 0 < eps)
    (P : Matrix (Fin a) (Fin b) ℝ)
    (U : ℕ → Matrix (Fin a) (Fin r) ℝ) (s : ℕ → Fin r → ℝ)
    (V : ℕ → Matrix (Fin b) (Fin r) ℝ)
    (hU : ∀ n, OrthonormalColumns (U n)) (hV : ∀ n, OrthonormalColumns (V n))
    (hs : ∀ n k, 0 ≤ s n k) (ha : 0 < a) (hb : 0 < b)
    (hra : r ≤ a) (hrb : r ≤ b) (n : ℕ) :
    matrixRMS (trajectory rate decay eps P (fun n => singularMatrix (U n) (s n) (V n)) n)
      ≤ max (matrixRMS P) (1/(4*decay)) := by
  induction n with
  | zero => exact le_max_left _ _
  | succ n ih =>
    rw [trajectory]
    exact step_bound (rate n) decay eps _ (hrate n) hdecay (hstep n) heps
      _ ih (le_max_right _ _) (U n) (s n) (V n) (hU n) (hV n) (hs n) ha hb hra hrb

private theorem unit_columns : OrthonormalColumns (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
  simp [OrthonormalColumns]

private theorem unit_rms : matrixRMS (1 : Matrix (Fin 1) (Fin 1) ℝ) = 1 := by
  norm_num [matrixRMS, squaredFrobenius]

example := energy_convex (1/2) (by norm_num) (by norm_num)
  (1 : Matrix (Fin 1) (Fin 1) ℝ) (-1)
example := rms_convex_bound (1/2) 1 (by norm_num) (by norm_num)
  (1 : Matrix (Fin 1) (Fin 1) ℝ) 1 (by norm_num) (by norm_num)
  (by rw [unit_rms]) (by rw [unit_rms])
example := step_bound (1/500) (1/10) (1/10000000) (5/2)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (1 : Matrix (Fin 1) (Fin 1) ℝ) (by rw [unit_rms]; norm_num) (by norm_num)
  1 (fun _ => 1) 1 unit_columns unit_columns (fun _ => by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
example (n : ℕ) := trajectory_bound (fun k => if k = 0 then 0 else 1/500)
  (1/10) (1/10000000) (by intro k; split_ifs <;> norm_num) (by norm_num)
  (by intro k; split_ifs <;> norm_num) (by norm_num)
  (1 : Matrix (Fin 1) (Fin 1) ℝ) (fun _ => 1) (fun _ _ => 1) (fun _ => 1)
  (fun _ => unit_columns) (fun _ => unit_columns) (fun _ _ => by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num) n

#print axioms energy_convex
#print axioms rms_convex_bound
#print axioms step_bound
#print axioms trajectory_bound
end Early99MuonDecay
