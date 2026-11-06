import Transformer.MHC.Section4_Parameterization
import Transformer.GPTMini.RMSNorm

/-!
# mHC with the current inverse-width model's Peri-RMS residual branches

Adaptation of arXiv:2512.24880v2, Eqs. (3), (8), (9), to
`early99_dimension_gain.dimension_model`: input RMS, attention / SwiGLU,
output RMS, then the fixed coefficient `1 / sqrt(2 * layers)`.
The mHC write gate remains `2 * sigmoid`; it is outside output RMS.

There are four streams, separate attention and MLP connections, and exactly
twenty column-then-row Sinkhorn cycles. These are real-arithmetic bounds for
the proposed combination, not a claim that the last Python experiment used
mHC. The current Python model and its optimizer are not modified.

Routing predictors are arbitrary functions of the current stream state;
the bounds therefore include normalized affine, input-dependent predictors.
Raw branches are arbitrary functions, with the current causal context held
in their closure along a forward trace. Neither independence nor a bound on
projection weights is assumed. Floating-point arithmetic, the complete
input-dependent Jacobian, and learning performance are outside this result.
-/

namespace MHCScaled

open Transformer Transformer.Residual Transformer.MHC
open scoped BigOperators

/-- Four residual streams; feature width is the current task's model width. -/
abbrev State (d : ℕ) := Fin 4 → EucSpace d

/-- Logits after the routing predictor, before sigmoid / finite Sinkhorn. -/
structure Routing where
  read : Fin 4 → ℝ
  write : Fin 4 → ℝ
  skip : Matrix (Fin 4) (Fin 4) ℝ

/-- A raw attention or SwiGLU branch and its input-dependent routing logits.
`raw` excludes both Peri-RMS operations and the depth coefficient. -/
structure Sublayer (d : ℕ) where
  routing : State d → Routing
  raw : State d → EucSpace d → EucSpace d

/-- mHC Eq. (3) with both current Peri-RMS sites and post-RMS scaling.
Read gates are sigmoid, write gates are twice sigmoid; the skip matrix is
the paper's finite twenty-cycle map, without denominator regularization. -/
noncomputable def step {d : ℕ} (eps scale : ℝ) (branch : Sublayer d)
    (x : State d) : State d :=
  let r := branch.routing x
  finiteMHCStep r.read r.write r.skip
    (fun z => scale • GPTMini.rmsNormEps eps
      (branch.raw x (GPTMini.rmsNormEps eps z))) x

/-- Current decoder order: attention addition, then SwiGLU addition.
Layer count counts decoder blocks, not the two residual sublayers. -/
noncomputable def run {d : ℕ} (eps scale : ℝ)
    (attention mlp : ℕ → Sublayer d) (initial : State d) : ℕ → State d
  | 0 => initial
  | k + 1 =>
    step eps scale (mlp k) (step eps scale (attention k)
      (run eps scale attention mlp initial k))

/-- Finite row normalization suffices for the combined per-stream bound.
The factor two comes from mHC's actual write gate, not the stream count.
No exact double stochasticity assumption is made for finite Sinkhorn. -/
theorem step_norm_le {d : ℕ} (eps scale : ℝ) (heps : 0 < eps)
    (hd : 0 < d) (hscale : 0 ≤ scale) (branch : Sublayer d)
    (x : State d) (R : ℝ) (hx : ∀ i, ‖x i‖ ≤ R) (i : Fin 4) :
    ‖step eps scale branch x i‖ ≤ R + 2 * scale * Real.sqrt (d : ℝ) := by
  let r := branch.routing x
  let H := finiteSinkhorn r.skip 20
  let v := branch.raw x (GPTMini.rmsNormEps eps
    (weightedSum (fun j => gate (r.read j)) x))
  have hpos : ∀ j k, 0 < H j k :=
    sinkhornIterate_pos _ (fun j k => Real.exp_pos (r.skip j k)) 20
  have hmix : ‖mix H x i‖ ≤ R :=
    weightedSum_convex_norm_le (H i) x R (fun j => (hpos i j).le)
      (twenty_iterations_row_sum r.skip i) hx
  have hc := writeGate_bounds (r.write i)
  have hv : ‖scale • GPTMini.rmsNormEps eps v‖ ≤ scale * Real.sqrt (d : ℝ) := by
    rw [norm_smul, Real.norm_of_nonneg hscale]
    exact mul_le_mul_of_nonneg_left (GPTMini.rmsNormEps_norm_le eps heps hd v) hscale
  have hw : ‖writeGate (r.write i) • (scale • GPTMini.rmsNormEps eps v)‖ ≤
      2 * scale * Real.sqrt (d : ℝ) := by
    rw [norm_smul, Real.norm_of_nonneg hc.1.le]
    calc
      writeGate (r.write i) * ‖scale • GPTMini.rmsNormEps eps v‖ ≤
          writeGate (r.write i) * (scale * Real.sqrt (d : ℝ)) :=
        mul_le_mul_of_nonneg_left hv hc.1.le
      _ ≤ 2 * (scale * Real.sqrt (d : ℝ)) :=
        mul_le_mul_of_nonneg_right hc.2.le (by positivity)
      _ = 2 * scale * Real.sqrt (d : ℝ) := by ring
  change ‖mix H x i + writeGate (r.write i) •
    (scale • GPTMini.rmsNormEps eps v)‖ ≤ R + 2 * scale * Real.sqrt (d : ℝ)
  exact (norm_add_le _ _).trans (add_le_add hmix hw)

/-- Both sublayers contribute at most `2 * scale * sqrt(width)` per block.
The bound permits correlated branches and input-dependent routing. -/
theorem run_norm_le {d : ℕ} (eps scale : ℝ) (heps : 0 < eps)
    (hd : 0 < d) (hscale : 0 ≤ scale) (attention mlp : ℕ → Sublayer d)
    (initial : State d) (R : ℝ) (hx : ∀ i, ‖initial i‖ ≤ R) (layers : ℕ) :
    ∀ i, ‖run eps scale attention mlp initial layers i‖ ≤
      R + (layers : ℝ) * (4 * scale * Real.sqrt (d : ℝ)) := by
  induction layers with
  | zero => simpa only [run, Nat.cast_zero, zero_mul, add_zero] using hx
  | succ k ih =>
    have ha := step_norm_le eps scale heps hd hscale (attention k)
      (run eps scale attention mlp initial k) _ ih
    have hm := step_norm_le eps scale heps hd hscale (mlp k)
      (step eps scale (attention k) (run eps scale attention mlp initial k)) _ ha
    intro i
    calc
      ‖run eps scale attention mlp initial (k + 1) i‖ ≤
          R + (k : ℝ) * (4 * scale * Real.sqrt (d : ℝ)) +
            2 * scale * Real.sqrt (d : ℝ) + 2 * scale * Real.sqrt (d : ℝ) := hm i
      _ = R + ((k + 1 : ℕ) : ℝ) * (4 * scale * Real.sqrt (d : ℝ)) := by
        push_cast
        ring

/-- The current normalized embedding is copied into all four streams. -/
noncomputable def embeddingState {d : ℕ} (eps : ℝ) (embedding : EucSpace d) : State d :=
  fun _ => GPTMini.rmsNormEps eps embedding

/-- Specialization to `1/sqrt(2L)` after output RMS and twice-sigmoid write-back.
The per-stream RMS bound is `1 + 2*sqrt(2L)`, not the single-stream
unit-write bound `1 + sqrt(2L)`. Positive width, epsilon and depth are explicit. -/
theorem scaled_model_rms_bound {d : ℕ} (eps : ℝ) (heps : 0 < eps)
    (hd : 0 < d) (attention mlp : ℕ → Sublayer d) (embedding : EucSpace d)
    (layers : ℕ) (hlayers : 0 < layers) (i : Fin 4) :
    ‖run eps (1 / Real.sqrt (2 * (layers : ℝ))) attention mlp
        (embeddingState eps embedding) layers i‖ / Real.sqrt (d : ℝ) ≤
      1 + 2 * Real.sqrt (2 * (layers : ℝ)) := by
  have hpos : (0 : ℝ) < 2 * (layers : ℝ) := by positivity
  have hspos : 0 < Real.sqrt (2 * (layers : ℝ)) := Real.sqrt_pos.mpr hpos
  have hsquare := Real.sq_sqrt hpos.le
  have hcoef : (2 * (layers : ℝ)) * (1 / Real.sqrt (2 * (layers : ℝ))) =
      Real.sqrt (2 * (layers : ℝ)) := by
    field_simp
    nlinarith
  have hr := run_norm_le eps (1 / Real.sqrt (2 * (layers : ℝ))) heps hd
    (by positivity) attention mlp (embeddingState eps embedding) (Real.sqrt (d : ℝ))
    (fun _ => GPTMini.rmsNormEps_norm_le eps heps hd embedding) layers i
  apply (div_le_iff₀ (by positivity : 0 < Real.sqrt (d : ℝ))).mpr
  calc
    ‖run eps (1 / Real.sqrt (2 * (layers : ℝ))) attention mlp
        (embeddingState eps embedding) layers i‖ ≤
        Real.sqrt (d : ℝ) + (layers : ℝ) *
          (4 * (1 / Real.sqrt (2 * (layers : ℝ))) * Real.sqrt (d : ℝ)) := hr
    _ = (1 + 2 * ((2 * (layers : ℝ)) * (1 / Real.sqrt (2 * (layers : ℝ))))) *
        Real.sqrt (d : ℝ) := by ring
    _ = (1 + 2 * Real.sqrt (2 * (layers : ℝ))) * Real.sqrt (d : ℝ) := by rw [hcoef]

/-- The same mHC / Peri-RMS model without depth scaling: `scale = 1`.
Two sublayers per block and write gates bounded by two give `1 + 4L`.
Depth zero is permitted; all raw branches and routing predictors remain arbitrary. -/
theorem unscaled_model_rms_bound {d : ℕ} (eps : ℝ) (heps : 0 < eps)
    (hd : 0 < d) (attention mlp : ℕ → Sublayer d) (embedding : EucSpace d)
    (layers : ℕ) (i : Fin 4) :
    ‖run eps 1 attention mlp (embeddingState eps embedding) layers i‖ /
        Real.sqrt (d : ℝ) ≤ 1 + 4 * (layers : ℝ) := by
  have hr := run_norm_le eps 1 heps hd (by norm_num) attention mlp
    (embeddingState eps embedding) (Real.sqrt (d : ℝ))
    (fun _ => GPTMini.rmsNormEps_norm_le eps heps hd embedding) layers i
  apply (div_le_iff₀ (by positivity : 0 < Real.sqrt (d : ℝ))).mpr
  calc
    ‖run eps 1 attention mlp (embeddingState eps embedding) layers i‖ ≤
        Real.sqrt (d : ℝ) + (layers : ℝ) * (4 * 1 * Real.sqrt (d : ℝ)) := hr
    _ = (1 + 4 * (layers : ℝ)) * Real.sqrt (d : ℝ) := by ring

/-- An unweighted mean collapses the four streams before the existing final RMS.
This is the explicit readout convention for the proposed adaptation. -/
noncomputable def collapse {d : ℕ} (x : State d) : EucSpace d :=
  weightedSum (fun _ => 1 / 4) x

/-- Stream averaging preserves a common norm bound; no learned readout is assumed. -/
theorem collapse_norm_le {d : ℕ} (x : State d) (R : ℝ)
    (hx : ∀ i, ‖x i‖ ≤ R) : ‖collapse x‖ ≤ R := by
  apply weightedSum_convex_norm_le
  · intro i
    norm_num
  · norm_num
  · exact hx

end MHCScaled
