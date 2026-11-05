import Transformer.RoPE.LeRoPE.Section3_LearnedFrequencies
import Transformer.SequenceModel.Rotary

/-!
The user's proposed Log-Scaled LeRoPE changes position p to p*log(p+1).
This is a new experiment, not a theorem attributed to either source paper.
The rotary identities below instantiate the arbitrary-angle and arbitrary-real-
position contracts used for LeRoPE (arXiv:2607.10134v1, Section 3.1).
They hold in real arithmetic; the implementation separately checks float32
tables and cached computation. No training or extrapolation claim follows.
-/
open scoped BigOperators
open Transformer Transformer.RoPE.LeRoPE Transformer.SequenceModel

namespace Early99.LogScaledRotary

/-- Zero-based token position warped by its own visible key count. -/
noncomputable def logPosition (p : ℝ) : ℝ := p * Real.log (p + 1)

/-- The first token's angle is finite and zero, including log(1). -/
theorem log_position_zero : logPosition 0 = 0 := by
  simp [logPosition]

/-- Arbitrary learned frequencies retain exact rotary norm preservation. -/
theorem warped_rotation_norm (d : ℕ) (frequency : Fin (d / 2) → ℝ)
    (p : ℝ) (x : EucSpace d) :
    ‖rotateHead d (fun k => logPosition p * frequency k) x‖ = ‖x‖ := by
  exact rotateHead_isometry d _ x

/-- Scores depend on the warped displacement, not necessarily on t-s. -/
theorem warped_relative (K : Type*) [Fintype K] (frequency : K → ℝ)
    (s t : ℝ) (x y : Transformer.Landscape.Neural.Channels K → ℝ) :
    (∑ i, rope frequency (logPosition s) x i *
      rope frequency (logPosition t) y i) =
    ∑ i, x i * rope frequency (logPosition t - logPosition s) y i := by
  exact rope_relative frequency (logPosition s) (logPosition t) x y

/-- Equal unit token gaps already differ at positions 0,1 and 1,2.
This is a coordinate counterexample, not a claim that every pair of vectors
has unequal rotary scores (zero vectors and periodic phases can coincide). -/
theorem unequal_unit_gaps :
    logPosition 1 - logPosition 0 < logPosition 2 - logPosition 1 := by
  have h : Real.log 2 < Real.log 3 :=
    Real.strictMonoOn_log (by norm_num) (by norm_num) (by norm_num)
  norm_num [logPosition]
  linarith

-- Nonzero learned frequencies and genuine vectors remain arbitrary here.
example (x : EucSpace 8) (alpha : Fin 4 → ℝ) :
    ‖rotateHead 8
      (fun k => logPosition 7 * learnedFrequency (1 / 10000) (alpha k)) x‖ = ‖x‖ :=
  warped_rotation_norm 8 _ 7 x
example (x y : Transformer.Landscape.Neural.Channels (Fin 4) → ℝ)
    (alpha : Fin 4 → ℝ) :
    (∑ i, rope (fun k => learnedFrequency (1 / 10000) (alpha k)) (logPosition 3) x i *
      rope (fun k => learnedFrequency (1 / 10000) (alpha k)) (logPosition 8) y i) =
    ∑ i, x i * rope (fun k => learnedFrequency (1 / 10000) (alpha k))
      (logPosition 8 - logPosition 3) y i :=
  warped_relative (Fin 4) _ 3 8 x y

#print axioms log_position_zero
#print axioms warped_rotation_norm
#print axioms warped_relative
#print axioms unequal_unit_gaps

end Early99.LogScaledRotary
