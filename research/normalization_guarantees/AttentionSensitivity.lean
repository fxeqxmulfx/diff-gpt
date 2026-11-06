import research.normalization_guarantees.RMSLipschitz
import Transformer.SequenceModel.Rotary
import Transformer.Perspective.Softmax

/-!
Local real-arithmetic sensitivity of the implemented normalized attention row.
Combine the improved finite-epsilon RMS bound with arbitrary fixed rotations,
Cauchy-Schwarz, and the existing conservative softmax infinity-to-one bound.
Keys, rotations and score multiplier are fixed. The variable is a projected
query before its RMS operation, not an embedding before an unbounded matrix.
This does not bound XSA, the full network, floating-point errors or learning.
-/

namespace Early99AttentionSensitivity
open Transformer Transformer.SequenceModel
open scoped BigOperators

/-- The implemented split-half rotation is linear on vector differences. -/
theorem rotateHead_sub {d : ℕ} (angle : Fin (d / 2) → ℝ) (x y : EucSpace d) :
    rotateHead d angle (x - y) = rotateHead d angle x - rotateHead d angle y := by
  ext i
  change (match GPTMini.ropeSplit d i with
    | .inl (.inl k) => rotate angle (splitPairs d (x - y)) (k, 0)
    | .inl (.inr k) => rotate angle (splitPairs d (x - y)) (k, 1)
    | .inr j => (x - y) ((GPTMini.ropeSplit d).symm (.inr j))) = _
  rcases h : GPTMini.ropeSplit d i with (k | k) | j <;>
    simp [rotateHead, h, rotate, splitPairs] <;> ring

/-- The actual epsilon-RMS followed by the actual arbitrary-angle rotation. -/
noncomputable def normalized {d : ℕ} (eps : ℝ) (angle : Fin (d / 2) → ℝ)
    (x : EucSpace d) : EucSpace d :=
  rotateHead d angle (GPTMini.rmsNormEps eps x)

/-- Fixed rotary phases preserve the improved global RMS Lipschitz constant. -/
theorem normalized_lipschitz {d : ℕ} (eps : ℝ) (heps : 0 < eps) (hd : 0 < d)
    (angle : Fin (d / 2) → ℝ) (x y : EucSpace d) :
    ‖normalized eps angle x - normalized eps angle y‖ ≤
      (1 / Real.sqrt eps) * ‖x - y‖ := by
  unfold normalized
  rw [← rotateHead_sub, rotateHead_isometry]
  exact Early99Normalization.rmsNormEps_lipschitz_improved eps heps hd x y

/-- Holding a normalized key fixed, each score is globally Lipschitz in raw q.
The gain may include the actual SSMax coefficient m*log(N)*g. -/
theorem score_query_lipschitz {d : ℕ} (eps gain : ℝ) (heps : 0 < eps) (hd : 0 < d)
    (a b : Fin (d / 2) → ℝ) (q q' k : EucSpace d) :
    |gain * inner ℝ (normalized eps a q) (normalized eps b k) -
      gain * inner ℝ (normalized eps a q') (normalized eps b k)| ≤
      (|gain| * Real.sqrt (d : ℝ) / Real.sqrt eps) * ‖q - q'‖ := by
  have hkey : ‖normalized eps b k‖ ≤ Real.sqrt (d : ℝ) := by
    rw [normalized, rotateHead_isometry]
    exact GPTMini.rmsNormEps_norm_le eps heps hd k
  calc
    _ = |gain| * |inner ℝ
        (normalized eps a q - normalized eps a q') (normalized eps b k)| := by
      rw [← mul_sub, ← inner_sub_left, abs_mul]
    _ ≤ |gain| * (‖normalized eps a q - normalized eps a q'‖ *
        ‖normalized eps b k‖) :=
      mul_le_mul_of_nonneg_left (abs_real_inner_le_norm _ _) (abs_nonneg gain)
    _ ≤ |gain| * (((1 / Real.sqrt eps) * ‖q - q'‖) * Real.sqrt (d : ℝ)) := by
      apply mul_le_mul_of_nonneg_left _ (abs_nonneg gain)
      exact mul_le_mul (normalized_lipschitz eps heps hd a q q') hkey
        (norm_nonneg _) (by positivity)
    _ = _ := by ring

/-- A complete nonempty row has an l1 sensitivity bound without a factor N.
The factor four is conservative, inherited from the checked softmax theorem. -/
theorem attention_row_query_lipschitz {d n : ℕ} (eps gain : ℝ)
    (heps : 0 < eps) (hd : 0 < d) (hn : 0 < n)
    (a : Fin (d / 2) → ℝ) (b : Fin n → Fin (d / 2) → ℝ)
    (keys : Fin n → EucSpace d) (q q' : EucSpace d) :
    ∑ j : Fin n, |Perspective.softmaxWeight
        (fun l => gain * inner ℝ (normalized eps a q) (normalized eps (b l) (keys l))) j -
      Perspective.softmaxWeight
        (fun l => gain * inner ℝ (normalized eps a q') (normalized eps (b l) (keys l))) j| ≤
      4 * ((|gain| * Real.sqrt (d : ℝ) / Real.sqrt eps) * ‖q - q'‖) := by
  apply Perspective.sum_abs_softmaxWeight_sub_le hn _ _ _ (by positivity)
  intro j
  exact score_query_lipschitz eps gain heps hd a (b j) q q' (keys j)

-- Concrete positive width/epsilon, nonzero phases, nonempty row, arbitrary inputs.
example (x y : EucSpace 8) :
    rotateHead 8 (fun _ => 2) (x - y) =
      rotateHead 8 (fun _ => 2) x - rotateHead 8 (fun _ => 2) y :=
  rotateHead_sub _ x y

example (x y : EucSpace 8) :
    ‖normalized (1 / 8388608) (fun _ => 2) x -
      normalized (1 / 8388608) (fun _ => 2) y‖ ≤
      (1 / Real.sqrt (1 / 8388608)) * ‖x - y‖ :=
  normalized_lipschitz _ (by norm_num) (by norm_num) _ x y

example (q q' k : EucSpace 8) :
    |(Real.log 4 / 2) * inner ℝ
        (normalized (1 / 8388608) (fun _ => 2) q)
        (normalized (1 / 8388608) (fun _ => 3) k) -
      (Real.log 4 / 2) * inner ℝ
        (normalized (1 / 8388608) (fun _ => 2) q')
        (normalized (1 / 8388608) (fun _ => 3) k)| ≤
      (|Real.log 4 / 2| * Real.sqrt 8 / Real.sqrt (1 / 8388608)) * ‖q - q'‖ :=
  score_query_lipschitz _ _ (by norm_num) (by norm_num) _ _ q q' k

example (keys : Fin 4 → EucSpace 8) (q q' : EucSpace 8) :
    ∑ j : Fin 4, |Perspective.softmaxWeight
        (fun l => (Real.log 4 / 2) * inner ℝ
          (normalized (1 / 8388608) (fun _ => 2) q)
          (normalized (1 / 8388608) (fun _ => 3) (keys l))) j -
      Perspective.softmaxWeight
        (fun l => (Real.log 4 / 2) * inner ℝ
          (normalized (1 / 8388608) (fun _ => 2) q')
          (normalized (1 / 8388608) (fun _ => 3) (keys l))) j| ≤
      4 * ((|Real.log 4 / 2| * Real.sqrt 8 / Real.sqrt (1 / 8388608)) * ‖q - q'‖) :=
  attention_row_query_lipschitz _ _ (by norm_num) (by norm_num) (by norm_num) _ _ keys q q'

#print axioms rotateHead_sub
#print axioms normalized_lipschitz
#print axioms score_query_lipschitz
#print axioms attention_row_query_lipschitz
end Early99AttentionSensitivity
