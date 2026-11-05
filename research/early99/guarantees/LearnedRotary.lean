import Transformer.RoPE.LeRoPE.Section3_LearnedFrequencies
import Transformer.SequenceModel.Rotary

/-!
Dependency audit for arXiv:2607.10134v1 Section 3.1 and Eq. (3).
The existing source theorems accept arbitrary real learned log-scales and
arbitrary rotary frequencies. They cover real arithmetic, not rounded tables,
optimizer convergence, a bounded frequency range or length extrapolation.
-/
open scoped BigOperators
open Transformer Transformer.RoPE.LeRoPE Transformer.SequenceModel

-- Positive base frequency and a nonzero learned scale, rather than a zero vector.
example : 0 < learnedFrequency (1 / 10000) (-2) :=
  learned_frequency_pos (by norm_num) (-2)
example : learnedFrequency (1 / 10000) 0 = 1 / 10000 :=
  learned_frequency_init _
example : HasDerivAt (learnedFrequency (1 / 10000))
    (learnedFrequency (1 / 10000) (-2)) (-2) :=
  learned_frequency_hasDerivAt _ _

-- Instantiations preserve the complete arbitrary-frequency theorem contracts.
example (x : EucSpace 8) (alpha : Fin 4 → ℝ) :
    ‖rotateHead 8 (fun k => 7 * learnedFrequency (1 / 10000) (alpha k)) x‖ = ‖x‖ :=
  rotateHead_isometry _ _ _
example (x y : Transformer.Landscape.Neural.Channels (Fin 4) → ℝ)
    (alpha : Fin 4 → ℝ) (s t : ℝ) :
    (∑ i, rope (fun k => learnedFrequency (1 / 10000) (alpha k)) s x i *
      rope (fun k => learnedFrequency (1 / 10000) (alpha k)) t y i) =
    ∑ i, x i * rope (fun k => learnedFrequency (1 / 10000) (alpha k)) (t - s) y i :=
  rope_relative _ _ _ _ _

#print axioms learned_frequency_pos
#print axioms learned_frequency_init
#print axioms learned_frequency_hasDerivAt
#print axioms score_log_frequency_hasDerivAt
#print axioms rotateHead_isometry
#print axioms rope_relative
