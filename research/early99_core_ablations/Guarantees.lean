import research.early99.guarantees.PeriBounds
import research.early99.guarantees.NormalizedScores

/-!
Bounds for the three additions declared in plan.json, over real arithmetic.
Source: factories.py installs only one mechanism; the exact reused equations
are experiments/models.py and experiments/scalable_softmax.py. These statements
specialize the previously checked RMS, residual-trace and normalized-score
bounds. They do not assert convergence, generalization, finite-precision
stability, or a uniform hidden-state bound for unnormalized residual branches.
-/

namespace Early99Core
open Transformer Early99.NormalizedScores

/-- Embedding-only RMS bounds the initial token state, for finite epsilon > 0. -/
theorem embedding_bound {d : ℕ} (eps : ℝ) (heps : 0 < eps) (hd : 0 < d)
    (embedding : EucSpace d) :
    ‖GPTMini.rmsNormEps eps embedding‖ ≤ Real.sqrt (d : ℝ) :=
  GPTMini.rmsNormEps_norm_le eps heps hd embedding

/-- Scaled Peri-RMS without embedding RMS retains dependence on the initial norm.
There are two normalized branches per layer, scaled AFTER normalization.
The raw branch outputs may depend on the complete preceding causal trace. -/
theorem scaled_peri_bound {d : ℕ} (eps : ℝ) (heps : 0 < eps) (hd : 0 < d)
    (embedding : EucSpace d) (raw : ℕ → EucSpace d)
    (layers : ℕ) (hlayers : 0 < layers) :
    ‖Early99.residualTrace eps (1 / Real.sqrt (2 * (layers : ℝ)))
      embedding raw (2 * layers)‖ ≤
      ‖embedding‖ + Real.sqrt (2 * (layers : ℝ)) * Real.sqrt (d : ℝ) := by
  have hpos : (0 : ℝ) < 2 * (layers : ℝ) := by positivity
  have hspos : 0 < Real.sqrt (2 * (layers : ℝ)) := Real.sqrt_pos.mpr hpos
  have hsquare := Real.sq_sqrt hpos.le
  have hcoef : (2 * (layers : ℝ)) * (1 / Real.sqrt (2 * (layers : ℝ))) =
      Real.sqrt (2 * (layers : ℝ)) := by
    field_simp
    nlinarith
  have bound := Early99.residualTrace_bound eps
    (1 / Real.sqrt (2 * (layers : ℝ))) heps hd (by positivity)
    embedding raw (2 * layers)
  simpa only [Nat.cast_mul, Nat.cast_ofNat, hcoef] using bound

/-- Original attention retains the actual fixed gain; no exact square-root
identity is assumed for its rounded exp(log_alpha)/d buffer calculation. -/
theorem original_score_bound {d : ℕ} (eps gain : ℝ) (heps : 0 < eps) (hd : 0 < d)
    (a b : Fin (d / 2) → ℝ) (q k : EucSpace d) :
    |gain * inner ℝ (rotatedQK eps a q) (rotatedQK eps b k)| ≤ |gain| * (d : ℝ) :=
  normalized_score eps gain heps hd a b q k

/-- SSMax alone multiplies the original normalized-score bound by log(N).
The visible key count is N=n+1, including the singleton N=1. -/
theorem ssmax_score_bound {d : ℕ} (eps gain : ℝ) (heps : 0 < eps) (hd : 0 < d)
    (n : ℕ) (a b : Fin (d / 2) → ℝ) (q k : EucSpace d) :
    |Real.log (n + 1) *
      (gain * inner ℝ (rotatedQK eps a q) (rotatedQK eps b k))| ≤
      Real.log (n + 1) * (|gain| * (d : ℝ)) :=
  ssmax_score eps gain heps hd n a b q k

-- Positive scientific epsilon and dimensions; no zero-vector-only assumptions.
example (embedding : EucSpace 8) :
    ‖GPTMini.rmsNormEps (1 / 8388608) embedding‖ ≤ Real.sqrt 8 :=
  embedding_bound (1 / 8388608) (by norm_num) (by norm_num) embedding

example (embedding : EucSpace 8) (raw : ℕ → EucSpace 8) :
    ‖Early99.residualTrace (1 / 8388608) (1 / Real.sqrt 4) embedding raw 4‖ ≤
      ‖embedding‖ + Real.sqrt 4 * Real.sqrt 8 := by
  convert scaled_peri_bound (1 / 8388608) (by norm_num) (by norm_num)
    embedding raw 2 (by norm_num) using 1 <;> norm_num

example (q k : EucSpace 8) :
    |(1 / Real.sqrt 8) * inner ℝ (rotatedQK (1 / 8388608) (fun _ => 2) q)
      (rotatedQK (1 / 8388608) (fun _ => 3) k)| ≤ |1 / Real.sqrt 8| * 8 :=
  original_score_bound (1 / 8388608) (1 / Real.sqrt 8) (by norm_num) (by norm_num)
    (fun _ => 2) (fun _ => 3) q k

example (q k : EucSpace 8) :
    |Real.log 4 * ((1 / Real.sqrt 8) * inner ℝ
      (rotatedQK (1 / 8388608) (fun _ => 2) q)
      (rotatedQK (1 / 8388608) (fun _ => 3) k))| ≤
      Real.log 4 * (|1 / Real.sqrt 8| * 8) := by
  convert ssmax_score_bound (1 / 8388608) (1 / Real.sqrt 8)
    (by norm_num) (by norm_num) 3 (fun _ => 2) (fun _ => 3) q k using 1 <;> norm_num

#print axioms embedding_bound
#print axioms scaled_peri_bound
#print axioms original_score_bound
#print axioms ssmax_score_bound
end Early99Core
