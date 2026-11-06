import research.early99_muon.Guarantees
import Transformer.Muon.Section2_PolarIdentities

/-!
Real matrix bounds for the frozen FP32-Muon experiment's mathematical equation.
The initial denominator is max(Frobenius norm, epsilon), including zero inputs.
The five finite quintic steps are not replaced by an exact polar factor.
Floating-point casts/products and auxiliary AdamW are outside these statements.
-/
namespace Early99MuonMath
open Transformer.Muon
open scoped BigOperators

variable {a b r : ℕ}

noncomputable def floorNorm (eps : ℝ) (X : Matrix (Fin a) (Fin b) ℝ) :=
  (max (Real.sqrt (squaredFrobenius X)) eps)⁻¹ • X

noncomputable def finiteSteps (X : Matrix (Fin a) (Fin b) ℝ) : ℕ →
    Matrix (Fin a) (Fin b) ℝ
  | 0 => X
  | n+1 => schulzStep (34445/10000) (-47750/10000) (20315/10000) (finiteSteps X n)

noncomputable def direction (eps : ℝ) (X : Matrix (Fin a) (Fin b) ℝ) :=
  finiteSteps (floorNorm eps X) 5

/-- The explicit epsilon-floor denominator is positive even for the zero matrix. -/
theorem denominator_pos (eps : ℝ) (heps : 0 < eps)
    (X : Matrix (Fin a) (Fin b) ℝ) :
    0 < max (Real.sqrt (squaredFrobenius X)) eps :=
  lt_of_lt_of_le heps (le_max_right _ _)

/-- Nonnegative singular coefficients of the floor-normalized matrix lie in [0,1]. -/
theorem normalized_coefficients (eps : ℝ) (heps : 0 < eps)
    (U : Matrix (Fin a) (Fin r) ℝ) (s : Fin r → ℝ) (V : Matrix (Fin b) (Fin r) ℝ)
    (hU : OrthonormalColumns U) (hV : OrthonormalColumns V)
    (hs : ∀ k, 0 ≤ s k) (k : Fin r) :
    0 ≤ s k / max (Real.sqrt (squaredFrobenius (singularMatrix U s V))) eps ∧
    s k / max (Real.sqrt (squaredFrobenius (singularMatrix U s V))) eps ≤ 1 := by
  have hp := denominator_pos eps heps (singularMatrix U s V)
  have hsq : s k ^ 2 ≤ squaredFrobenius (singularMatrix U s V) := by
    rw [squaredFrobenius_singularMatrix U s V hU hV]
    exact Finset.single_le_sum (fun i _ => sq_nonneg (s i)) (Finset.mem_univ k)
  have hle := (Real.le_sqrt (hs k) (squaredFrobenius_nonneg _)).2 hsq
  exact ⟨div_nonneg (hs k) (le_of_lt hp),
    (div_le_one₀ hp).2 (hle.trans (le_max_left _ _))⟩

/-- This normalization has the declared singular coefficients, including below epsilon. -/
theorem floorNorm_spectrum (eps : ℝ) (U : Matrix (Fin a) (Fin r) ℝ)
    (s : Fin r → ℝ) (V : Matrix (Fin b) (Fin r) ℝ) :
    floorNorm eps (singularMatrix U s V) =
      singularMatrix U
        (fun k => s k / max (Real.sqrt (squaredFrobenius (singularMatrix U s V))) eps) V := by
  unfold floorNorm
  rw [singularMatrix_smul]
  congr 1
  funext k
  simp only [div_eq_mul_inv]
  ring

/-- Every finite step acts by the same scalar polynomial; no convergence limit is taken. -/
theorem finiteSteps_spectrum (U : Matrix (Fin a) (Fin r) ℝ) (s : Fin r → ℝ)
    (V : Matrix (Fin b) (Fin r) ℝ) (hU : OrthonormalColumns U)
    (hV : OrthonormalColumns V) (n : ℕ) :
    finiteSteps (singularMatrix U s V) n =
      singularMatrix U
        (fun k => scalarSchulz (34445/10000) (-47750/10000) (20315/10000) (s k) n) V := by
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [finiteSteps, ih, schulzStep_spectrum _ _ _ U _ V hU hV]
    rfl

/-- Transposing before the five steps and back afterwards preserves the direction. -/
theorem oriented_direction (eps : ℝ) (U : Matrix (Fin a) (Fin r) ℝ)
    (s : Fin r → ℝ) (V : Matrix (Fin b) (Fin r) ℝ)
    (hU : OrthonormalColumns U) (hV : OrthonormalColumns V) :
    (finiteSteps (floorNorm eps (singularMatrix U s V)).transpose 5).transpose =
      direction eps (singularMatrix U s V) := by
  rw [floorNorm_spectrum, singularMatrix_transpose,
    finiteSteps_spectrum V _ U hV hU, singularMatrix_transpose]
  unfold direction
  rw [floorNorm_spectrum, finiteSteps_spectrum U _ V hU hV]

/-- The five-step direction has energy at most 25r/16 for an r-column representation. -/
theorem direction_energy (eps : ℝ) (heps : 0 < eps)
    (U : Matrix (Fin a) (Fin r) ℝ) (s : Fin r → ℝ) (V : Matrix (Fin b) (Fin r) ℝ)
    (hU : OrthonormalColumns U) (hV : OrthonormalColumns V) (hs : ∀ k, 0 ≤ s k) :
    squaredFrobenius (direction eps (singularMatrix U s V)) ≤ (25/16:ℝ) * r := by
  unfold direction
  rw [floorNorm_spectrum, finiteSteps_spectrum U _ V hU hV,
    squaredFrobenius_singularMatrix U _ V hU hV]
  calc
    _ ≤ ∑ _ : Fin r, (25/16:ℝ) := by
      apply Finset.sum_le_sum
      intro k _
      obtain ⟨hl, hu⟩ := normalized_coefficients eps heps U s V hU hV hs k
      obtain ⟨hn, hb⟩ := Early99Muon.finite_scalar_bounds _ hl hu 5
      nlinarith [sq_nonneg
        (scalarSchulz (34445/10000) (-47750/10000) (20315/10000)
          (s k / max (Real.sqrt (squaredFrobenius (singularMatrix U s V))) eps) 5 - 5/4)]
    _ = (25/16:ℝ) * r := by simp; ring

/-- The implemented 0.2sqrt(max(shape)) factor gives RMS at most 0.25. -/
theorem adjusted_direction_rms (eps : ℝ) (heps : 0 < eps)
    (U : Matrix (Fin a) (Fin r) ℝ) (s : Fin r → ℝ) (V : Matrix (Fin b) (Fin r) ℝ)
    (hU : OrthonormalColumns U) (hV : OrthonormalColumns V) (hs : ∀ k, 0 ≤ s k)
    (ha : 0 < a) (hb : 0 < b) (hra : r ≤ a) (hrb : r ≤ b) :
    matrixRMS (adjustedUpdate (direction eps (singularMatrix U s V))) ≤ 1/4 := by
  have henergy := direction_energy eps heps U s V hU hV hs
  have hmax : 0 ≤ max (a:ℝ) b := le_trans (by positivity) (le_max_left _ _)
  have hrmin : (r:ℝ) ≤ min (a:ℝ) b := by
    exact_mod_cast (le_min hra hrb)
  have hdim : max (a:ℝ) b * r ≤ (a:ℝ) * b := by
    calc
      _ ≤ max (a:ℝ) b * min (a:ℝ) b := mul_le_mul_of_nonneg_left hrmin hmax
      _ = _ := max_mul_min _ _
  have hab : 0 < (a:ℝ) * b := mul_pos (by exact_mod_cast ha) (by exact_mod_cast hb)
  have hscaled := mul_le_mul_of_nonneg_left henergy (show 0 ≤ max (a:ℝ) b / 25 by positivity)
  unfold matrixRMS adjustedUpdate shapeScale
  rw [squaredFrobenius_smul]
  apply Real.sqrt_le_iff.mpr
  constructor
  · norm_num
  · apply (div_le_iff₀ hab).2
    rw [mul_pow, Real.sq_sqrt hmax]
    nlinarith

noncomputable def matrixUpdate (rate decay eps : ℝ) (P H : Matrix (Fin a) (Fin b) ℝ) :=
  (1-rate*decay) • P - rate • adjustedUpdate (direction eps H)

/-- After separating the declared decoupled decay, update RMS is bounded by rate/4. -/
theorem update_rms (rate decay eps : ℝ) (hrate : 0 ≤ rate) (heps : 0 < eps)
    (P : Matrix (Fin a) (Fin b) ℝ)
    (U : Matrix (Fin a) (Fin r) ℝ) (s : Fin r → ℝ) (V : Matrix (Fin b) (Fin r) ℝ)
    (hU : OrthonormalColumns U) (hV : OrthonormalColumns V) (hs : ∀ k, 0 ≤ s k)
    (ha : 0 < a) (hb : 0 < b) (hra : r ≤ a) (hrb : r ≤ b) :
    matrixRMS (matrixUpdate rate decay eps P (singularMatrix U s V) -
      (1-rate*decay) • P) ≤ rate/4 := by
  have h := adjusted_direction_rms eps heps U s V hU hV hs ha hb hra hrb
  have e : matrixUpdate rate decay eps P (singularMatrix U s V) -
      (1-rate*decay) • P = (-rate) • adjustedUpdate (direction eps (singularMatrix U s V)) := by
    unfold matrixUpdate
    module
  rw [e, matrixRMS_smul, abs_neg, abs_of_nonneg hrate]
  nlinarith [mul_le_mul_of_nonneg_left h hrate]

-- Nonvacuous 1x1 instances satisfy every hypothesis, including the actual epsilon.
private theorem unit_columns : OrthonormalColumns (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
  simp [OrthonormalColumns]

example := denominator_pos (1/10000000) (by norm_num) (0 : Matrix (Fin 1) (Fin 1) ℝ)
example := normalized_coefficients (1/10000000) (by norm_num)
  (1 : Matrix (Fin 1) (Fin 1) ℝ) (fun _ => 0) 1 unit_columns unit_columns
  (fun _ => by norm_num) 0
example := finiteSteps_spectrum (1 : Matrix (Fin 1) (Fin 1) ℝ) (fun _ => 1) 1
  unit_columns unit_columns 5
example := oriented_direction (1/10000000)
  (1 : Matrix (Fin 1) (Fin 1) ℝ) (fun _ => 1) 1 unit_columns unit_columns
example := direction_energy (1/10000000) (by norm_num)
  (1 : Matrix (Fin 1) (Fin 1) ℝ) (fun _ => 1) 1 unit_columns unit_columns
  (fun _ => by norm_num)
example := adjusted_direction_rms (1/10000000) (by norm_num)
  (1 : Matrix (Fin 1) (Fin 1) ℝ) (fun _ => 1) 1 unit_columns unit_columns
  (fun _ => by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
example := update_rms (1/500) (1/10) (1/10000000) (by norm_num) (by norm_num)
  (1 : Matrix (Fin 1) (Fin 1) ℝ) 1 (fun _ => 1) 1 unit_columns unit_columns
  (fun _ => by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)

#print axioms denominator_pos
#print axioms normalized_coefficients
#print axioms floorNorm_spectrum
#print axioms finiteSteps_spectrum
#print axioms oriented_direction
#print axioms direction_energy
#print axioms adjusted_direction_rms
#print axioms update_rms
end Early99MuonMath
