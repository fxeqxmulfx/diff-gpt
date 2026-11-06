import research.mhc_scaled.CurrentModel

/-!
Concrete positive-epsilon / positive-depth examples and axiom checks for
every new theorem. The nonzero scalar fixture has four streams, dynamic
logits, a nonzero branch, and the actual finite twenty-cycle skip map.
The current model example uses width 128, head width 32 and six blocks.
-/

namespace MHCScaled

open Transformer

/-- Nonzero stream feature with norm one. -/
noncomputable def witness : EucSpace 1 := EuclideanSpace.single (0 : Fin 1) 1

/-- A nontrivial branch whose routing and raw output depend on its inputs. -/
noncomputable def witnessBranch : Sublayer 1 where
  routing x := {
    read := fun i => ‖x i‖ - 1
    write := fun i => ‖x i‖ - 1
    skip := fun i j => if i = j then ‖x i‖ else 0
  }
  raw x z := z + x 0

example : ‖witness‖ = 1 := by simp [witness, PiLp.norm_single]

example : ‖step (1 / 8388608) (1 / 2) witnessBranch (fun _ => witness) 0‖ ≤
    (1 : ℝ) + 2 * (1 / 2) * Real.sqrt (1 : ℝ) := by
  simpa only [Nat.cast_one, Nat.cast_ofNat] using
    step_norm_le (d := 1) (1 / 8388608) (1 / 2) (by norm_num) (by norm_num)
      (by norm_num) witnessBranch (fun _ => witness) 1
      (by intro i; simp [witness, PiLp.norm_single]) 0

example : ∀ i,
    ‖run (1 / 8388608) (1 / 2) (fun _ => witnessBranch) (fun _ => witnessBranch)
        (fun _ => witness) 2 i‖ ≤ (1 : ℝ) + (2 : ℝ) * (4 * (1 / 2) * Real.sqrt 1) := by
  simpa only [Nat.cast_one, Nat.cast_ofNat] using
    run_norm_le (d := 1) (1 / 8388608) (1 / 2) (by norm_num) (by norm_num)
      (by norm_num) (fun _ => witnessBranch) (fun _ => witnessBranch)
      (fun _ => witness) 1 (by intro i; simp [witness, PiLp.norm_single]) 2

example :=
  scaled_model_rms_bound (d := 1) (1 / 8388608) (by norm_num) (by norm_num)
    (fun _ => witnessBranch) (fun _ => witnessBranch) witness 2 (by norm_num) 0

example :=
  unscaled_model_rms_bound (d := 1) (1 / 8388608) (by norm_num) (by norm_num)
    (fun _ => witnessBranch) (fun _ => witnessBranch) witness 2 0

example : ‖collapse (fun _ => witness)‖ ≤ 1 := by
  exact collapse_norm_le (fun _ => witness) 1
    (by intro i; simp [witness, PiLp.norm_single])

example (attention mlp : ℕ → Sublayer 128) (embedding : EucSpace 128)
    (q k : EucSpace 32) :=
  current_model_bounds (1 / 8388608) (by norm_num) (by norm_num) (by norm_num)
    attention mlp embedding 6 (by norm_num) 63 (fun _ => 2) (fun _ => 3) q k

example (attention mlp : ℕ → Sublayer 128) (embedding : EucSpace 128)
    (q k : EucSpace 32) :=
  unscaled_current_model_bounds (1 / 8388608) (by norm_num) (by norm_num) (by norm_num)
    attention mlp embedding 6 63 (fun _ => 2) (fun _ => 3) q k

example : 1 + 2 * Real.sqrt (2 * (6 : ℝ)) < 1 + 4 * (6 : ℝ) := by
  simpa only [Nat.cast_ofNat] using scaled_bound_lt_unscaled_bound 6 (by norm_num)

example (attention mlp : ℕ → Sublayer 128) (embedding : EucSpace 128) :
    ‖finalFeature (1 / 8388608) (1 / Real.sqrt 12) attention mlp embedding 6‖ /
        Real.sqrt (128 : ℝ) ≤ 1 := by
  exact final_feature_rms_le_one (1 / 8388608) (1 / Real.sqrt 12) (by norm_num) (by norm_num)
    attention mlp embedding 6

example (attention mlp : ℕ → Sublayer 128) (embedding : EucSpace 128) :
    ‖finalFeature (1 / 8388608) 1 attention mlp embedding 6‖ / Real.sqrt (128 : ℝ) ≤ 1 := by
  exact final_feature_rms_le_one (1 / 8388608) 1 (by norm_num) (by norm_num)
    attention mlp embedding 6

#print axioms step_norm_le
#print axioms run_norm_le
#print axioms scaled_model_rms_bound
#print axioms unscaled_model_rms_bound
#print axioms collapse_norm_le
#print axioms current_model_bounds
#print axioms unscaled_current_model_bounds
#print axioms scaled_bound_lt_unscaled_bound
#print axioms final_feature_rms_le_one
#print axioms Early99DimensionGain.score_bound
#print axioms Transformer.MHC.skipProduct_spectral_norm_le
#print axioms Transformer.MHC.stochastic_can_erase_disagreement
#print axioms Transformer.MHC.one_cycle_not_doubly_stochastic

end MHCScaled
