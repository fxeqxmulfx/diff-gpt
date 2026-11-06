import research.normalization_guarantees.AttentionSensitivity
import research.normalization_guarantees.SoftmaxLipschitz

/-!
Apply the improved softmax bound to the unchanged finite-epsilon Q/K equation.
The query-only constant improves from 4*C to C. A joint query/key perturbation
bound uses the same C and a uniform bound on key changes, with fixed phases.
These are real bounds on projected vectors, not on unbounded projection weights,
XSA, value/readout paths, learned phase changes, floating-point errors or learning.
-/

namespace Early99AttentionSensitivityImproved
open Transformer Early99AttentionSensitivity
open scoped BigOperators

/-- The same actual probability row has query sensitivity C instead of 4*C. -/
theorem query_lipschitz {d n : ℕ} (eps gain : ℝ)
    (heps : 0 < eps) (hd : 0 < d) (hn : 0 < n)
    (a : Fin (d / 2) → ℝ) (b : Fin n → Fin (d / 2) → ℝ)
    (keys : Fin n → EucSpace d) (q q' : EucSpace d) :
    ∑ j : Fin n, |Perspective.softmaxWeight
        (fun l => gain * inner ℝ (normalized eps a q) (normalized eps (b l) (keys l))) j -
      Perspective.softmaxWeight
        (fun l => gain * inner ℝ (normalized eps a q') (normalized eps (b l) (keys l))) j| ≤
      (|gain| * Real.sqrt (d : ℝ) / Real.sqrt eps) * ‖q - q'‖ := by
  apply Early99SoftmaxLipschitz.softmax_l1_lipschitz hn _ _ _ (by positivity)
  intro j
  exact score_query_lipschitz eps gain heps hd a (b j) q q' (keys j)

/-- Both projected vectors can change; normalization controls both terms. -/
theorem score_pair_lipschitz {d : ℕ} (eps gain : ℝ) (heps : 0 < eps) (hd : 0 < d)
    (a b : Fin (d / 2) → ℝ) (q q' k k' : EucSpace d) :
    |gain * inner ℝ (normalized eps a q) (normalized eps b k) -
      gain * inner ℝ (normalized eps a q') (normalized eps b k')| ≤
      (|gain| * Real.sqrt (d : ℝ) / Real.sqrt eps) * (‖q - q'‖ + ‖k - k'‖) := by
  have hq := score_query_lipschitz eps gain heps hd a b q q' k
  have hk := score_query_lipschitz eps gain heps hd b a k k' q'
  simp only [real_inner_comm] at hk
  calc
    _ ≤ |gain * inner ℝ (normalized eps a q) (normalized eps b k) -
          gain * inner ℝ (normalized eps a q') (normalized eps b k)| +
        |gain * inner ℝ (normalized eps a q') (normalized eps b k) -
          gain * inner ℝ (normalized eps a q') (normalized eps b k')| :=
      abs_sub_le _ _ _
    _ ≤ _ := by nlinarith [hq, hk]

/-- A uniform bound on all key changes controls the entire row without a factor N. -/
theorem joint_lipschitz {d n : ℕ} (eps gain : ℝ)
    (heps : 0 < eps) (hd : 0 < d) (hn : 0 < n)
    (a : Fin (d / 2) → ℝ) (b : Fin n → Fin (d / 2) → ℝ)
    (keys keys' : Fin n → EucSpace d) (q q' : EucSpace d)
    (E : ℝ) (hE : 0 ≤ E) (hkeys : ∀ j, ‖keys j - keys' j‖ ≤ E) :
    ∑ j : Fin n, |Perspective.softmaxWeight
        (fun l => gain * inner ℝ (normalized eps a q) (normalized eps (b l) (keys l))) j -
      Perspective.softmaxWeight
        (fun l => gain * inner ℝ (normalized eps a q') (normalized eps (b l) (keys' l))) j| ≤
      (|gain| * Real.sqrt (d : ℝ) / Real.sqrt eps) * (‖q - q'‖ + E) := by
  apply Early99SoftmaxLipschitz.softmax_l1_lipschitz hn _ _ _ (by positivity)
  intro j
  exact (score_pair_lipschitz eps gain heps hd a (b j) q q' (keys j) (keys' j)).trans
    (mul_le_mul_of_nonneg_left (add_le_add le_rfl (hkeys j)) (by positivity))

-- Concrete actual FP32 epsilon, positive width, nonempty row and arbitrary inputs.
example (keys : Fin 4 → EucSpace 8) (q q' : EucSpace 8) :=
  query_lipschitz (1 / 8388608) (Real.log 4 / (4 * Real.sqrt 8))
    (by norm_num) (by norm_num) (by norm_num)
    (fun _ => 2) (fun j _ => j.val) keys q q'

example (q q' k k' : EucSpace 8) :=
  score_pair_lipschitz (1 / 8388608) (Real.log 4 / (4 * Real.sqrt 8))
    (by norm_num) (by norm_num) (fun _ => 2) (fun _ => 3) q q' k k'

example (q q' k k' : EucSpace 8) :=
  joint_lipschitz (1 / 8388608) (Real.log 4 / (4 * Real.sqrt 8))
    (by norm_num) (by norm_num) (by norm_num)
    (fun _ => 2) (fun j : Fin 4 => fun _ => (j.val : ℝ))
    (fun _ => k) (fun _ => k') q q' ‖k - k'‖ (norm_nonneg _) (fun _ => le_rfl)

#print axioms query_lipschitz
#print axioms score_pair_lipschitz
#print axioms joint_lipschitz
end Early99AttentionSensitivityImproved
