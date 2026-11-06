import research.mhc_scaled.Bounds
import research.early99_dimension_gain.Guarantees

/-!
# Bounds specialized to the latest inverse-width SSMax model

The base architecture is `peri-rms-scaled-ssmax-inverse-width-lr2`:
embedding RMS, per-head Q/K RMS, fixed RoPE, SSMax gain `1/head_width`,
centered XSA, input/output Peri-RMS, SwiGLU, depth scaling, final RMS,
and a tied embedding/readout matrix. The latest Schedule-Free experiment
uses the same model. These bounds do not depend on the optimizer.

For the proposed mHC adaptation, `Bounds.run` replaces each of the two
residual additions per block by four-stream finite mHC. The raw attention
branch includes Q/K projections, normalization, RoPE, SSMax, centered XSA
and output projection; the raw MLP is SwiGLU and output projection. Their
exact internal operations need no extra hypothesis for the output-RMS bound.
The final stream mean is passed through the current final RMS and tied head.

The score theorem explicitly retains the existing `log(N)/head_width`
equation. Its projected query, key, and rotary phases are arbitrary, so
routing before those projections cannot invalidate this pointwise bound.
Conjoining the bounds is not a formal translation of the PyTorch forward,
a bound on learned readout weights, or a theorem about training dynamics.
-/

namespace MHCScaled

open Transformer
open Early99.NormalizedScores

/-- The mHC proposal's final features for either residual coefficient,
immediately before the unchanged tied readout: stream mean, then final RMS. -/
noncomputable def finalFeature {d : ℕ} (eps scale : ℝ)
    (attention mlp : ℕ → Sublayer d) (embedding : EucSpace d) (layers : ℕ) : EucSpace d :=
  GPTMini.rmsNormEps eps (collapse
    (run eps scale attention mlp
      (embeddingState eps embedding) layers))

/-- Current-model contract: each stream and their mean have RMS at most
`1 + 2*sqrt(2L)`; Q/K-normalized inverse-width SSMax has score magnitude
at most `log(N)`. Feature width and attention head width are distinct.
The same positive epsilon is used at the current model's RMS sites. -/
theorem current_model_bounds {width headWidth : ℕ} (eps : ℝ) (heps : 0 < eps)
    (hwidth : 0 < width) (hhead : 0 < headWidth)
    (attention mlp : ℕ → Sublayer width) (embedding : EucSpace width)
    (layers : ℕ) (hlayers : 0 < layers) (n : ℕ)
    (a b : Fin (headWidth / 2) → ℝ) (q k : EucSpace headWidth) :
    (∀ i, ‖run eps (1 / Real.sqrt (2 * (layers : ℝ))) attention mlp
        (embeddingState eps embedding) layers i‖ / Real.sqrt (width : ℝ) ≤
      1 + 2 * Real.sqrt (2 * (layers : ℝ))) ∧
    (‖collapse (run eps (1 / Real.sqrt (2 * (layers : ℝ))) attention mlp
        (embeddingState eps embedding) layers)‖ / Real.sqrt (width : ℝ) ≤
      1 + 2 * Real.sqrt (2 * (layers : ℝ))) ∧
    |Real.log (n + 1) * ((1 / (headWidth : ℝ)) *
      inner ℝ (rotatedQK eps a q) (rotatedQK eps b k))| ≤ Real.log (n + 1) := by
  have hs : 0 < Real.sqrt (width : ℝ) := by positivity
  have hstreams := scaled_model_rms_bound eps heps hwidth attention mlp embedding
    layers hlayers
  refine ⟨hstreams, ?_, Early99DimensionGain.score_bound eps heps hhead n a b q k⟩
  apply (div_le_iff₀ hs).mpr
  apply collapse_norm_le
  intro i
  exact (div_le_iff₀ hs).mp (hstreams i)

/-- Matched unscaled contract: set only the post-RMS depth coefficient to one.
The four streams, finite Sinkhorn, sigmoid / twice-sigmoid gates, branch
normalizations, and inverse-width Q/K-normalized SSMax equation are identical. -/
theorem unscaled_current_model_bounds {width headWidth : ℕ} (eps : ℝ) (heps : 0 < eps)
    (hwidth : 0 < width) (hhead : 0 < headWidth)
    (attention mlp : ℕ → Sublayer width) (embedding : EucSpace width)
    (layers : ℕ) (n : ℕ) (a b : Fin (headWidth / 2) → ℝ) (q k : EucSpace headWidth) :
    (∀ i, ‖run eps 1 attention mlp (embeddingState eps embedding) layers i‖ /
        Real.sqrt (width : ℝ) ≤ 1 + 4 * (layers : ℝ)) ∧
    (‖collapse (run eps 1 attention mlp (embeddingState eps embedding) layers)‖ /
        Real.sqrt (width : ℝ) ≤ 1 + 4 * (layers : ℝ)) ∧
    |Real.log (n + 1) * ((1 / (headWidth : ℝ)) *
      inner ℝ (rotatedQK eps a q) (rotatedQK eps b k))| ≤ Real.log (n + 1) := by
  have hs : 0 < Real.sqrt (width : ℝ) := by positivity
  have hstreams := unscaled_model_rms_bound eps heps hwidth attention mlp embedding layers
  refine ⟨hstreams, ?_, Early99DimensionGain.score_bound eps heps hhead n a b q k⟩
  apply (div_le_iff₀ hs).mpr
  apply collapse_norm_le
  intro i
  exact (div_le_iff₀ hs).mp (hstreams i)

/-- Positive depth gives a strictly smaller uniform upper bound with scaling.
This compares the two proved bounds, not actual activations, gradients,
or learning outcomes of the scaled and unscaled models. -/
theorem scaled_bound_lt_unscaled_bound (layers : ℕ) (hlayers : 0 < layers) :
    1 + 2 * Real.sqrt (2 * (layers : ℝ)) < 1 + 4 * (layers : ℝ) := by
  have hl : (1 : ℝ) ≤ (layers : ℝ) := by exact_mod_cast hlayers
  have hs : Real.sqrt (2 * (layers : ℝ)) < 2 * (layers : ℝ) :=
    Real.sqrt_lt_self_iff.mpr (by linarith)
  linarith

/-- The current final RMS bounds the features supplied to the tied matrix,
independently of depth and the chosen residual coefficient. This does not
bound logits independently of the readout matrix or full-network gradients. -/
theorem final_feature_rms_le_one {d : ℕ} (eps scale : ℝ) (heps : 0 < eps) (hd : 0 < d)
    (attention mlp : ℕ → Sublayer d) (embedding : EucSpace d) (layers : ℕ) :
    ‖finalFeature eps scale attention mlp embedding layers‖ / Real.sqrt (d : ℝ) ≤ 1 := by
  apply (div_le_iff₀ (by positivity : 0 < Real.sqrt (d : ℝ))).mpr
  simpa only [finalFeature, one_mul] using GPTMini.rmsNormEps_norm_le eps heps hd
    (collapse (run eps scale attention mlp
      (embeddingState eps embedding) layers))

end MHCScaled
