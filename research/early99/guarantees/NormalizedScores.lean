import Transformer.GPTMini.RMSNorm
import Transformer.SequenceModel.Rotary

/-!
Q/K RMS gives uniform score control independent of projection matrix norms.
These bounds use the actual finite-epsilon normalization, arbitrary rotary
angles (including LeRoPE), fixed attention gain and SSMax's visible-key scale.
They do not imply a positive winning margin or a full-network gradient bound.
-/
namespace Early99.NormalizedScores
open Transformer Transformer.SequenceModel

noncomputable def rotatedQK {d : ℕ} (eps : ℝ) (angles : Fin (d / 2) → ℝ)
    (x : EucSpace d) : EucSpace d :=
  rotateHead d angles (GPTMini.rmsNormEps eps x)

/-- Post-RMS rotations preserve the uniform sqrt(d) query/key norm bound. -/
theorem normalized_norm {d : ℕ} (eps : ℝ) (heps : 0 < eps) (hd : 0 < d)
    (angles : Fin (d / 2) → ℝ) (x : EucSpace d) :
    ‖rotatedQK eps angles x‖ ≤ Real.sqrt (d : ℝ) := by
  rw [rotatedQK, rotateHead_isometry]
  exact GPTMini.rmsNormEps_norm_le eps heps hd x

/-- Cauchy–Schwarz controls every finite-epsilon normalized Q/K dot product. -/
theorem normalized_dot {d : ℕ} (eps : ℝ) (heps : 0 < eps) (hd : 0 < d)
    (a b : Fin (d / 2) → ℝ) (q k : EucSpace d) :
    |inner ℝ (rotatedQK eps a q) (rotatedQK eps b k)| ≤ (d : ℝ) := by
  calc
    _ ≤ ‖rotatedQK eps a q‖ * ‖rotatedQK eps b k‖ := abs_real_inner_le_norm _ _
    _ ≤ Real.sqrt (d : ℝ) * Real.sqrt (d : ℝ) :=
      mul_le_mul (normalized_norm eps heps hd a q) (normalized_norm eps heps hd b k)
        (norm_nonneg _) (Real.sqrt_nonneg _)
    _ = (d : ℝ) := Real.mul_self_sqrt (Nat.cast_nonneg d)

/-- A fixed multiplier g bounds each ordinary attention score by |g|*d. -/
theorem normalized_score {d : ℕ} (eps gain : ℝ) (heps : 0 < eps) (hd : 0 < d)
    (a b : Fin (d / 2) → ℝ) (q k : EucSpace d) :
    |gain * inner ℝ (rotatedQK eps a q) (rotatedQK eps b k)| ≤ |gain| * (d : ℝ) := by
  rw [abs_mul]
  exact mul_le_mul_of_nonneg_left (normalized_dot eps heps hd a b q k) (abs_nonneg gain)

/-- Fixed s=1 SSMax changes this uniform bound by log(N), for N visible keys. -/
theorem ssmax_score {d : ℕ} (eps gain : ℝ) (heps : 0 < eps) (hd : 0 < d)
    (n : ℕ) (a b : Fin (d / 2) → ℝ) (q k : EucSpace d) :
    |Real.log (n + 1) *
      (gain * inner ℝ (rotatedQK eps a q) (rotatedQK eps b k))| ≤
      Real.log (n + 1) * (|gain| * (d : ℝ)) := by
  have hn : 0 ≤ Real.log (n + 1 : ℝ) :=
    Real.log_nonneg (by exact_mod_cast Nat.succ_le_succ (Nat.zero_le n))
  rw [abs_mul, abs_of_nonneg hn]
  exact mul_le_mul_of_nonneg_left (normalized_score eps gain heps hd a b q k) hn

-- Positive scientific epsilon, positive width and unrestricted vectors, including nonzero ones.
example (q k : EucSpace 8) :
    |(1 / 8 : ℝ) * inner ℝ
      (rotatedQK (1 / 8388608) (fun _ => 2) q)
      (rotatedQK (1 / 8388608) (fun _ => 3) k)| ≤ 1 := by
  convert normalized_score (d := 8) (1 / 8388608) (1 / 8)
    (by norm_num) (by norm_num) (fun _ => 2) (fun _ => 3) q k using 1; norm_num
example (q k : EucSpace 8) :
    |Real.log 4 * ((1 / 8 : ℝ) * inner ℝ
      (rotatedQK (1 / 8388608) (fun _ => 2) q)
      (rotatedQK (1 / 8388608) (fun _ => 3) k))| ≤ Real.log 4 := by
  convert ssmax_score (d := 8) (1 / 8388608) (1 / 8)
    (by norm_num) (by norm_num) 3 (fun _ => 2) (fun _ => 3) q k using 1 <;> norm_num

#print axioms normalized_norm
#print axioms normalized_dot
#print axioms normalized_score
#print axioms ssmax_score
end Early99.NormalizedScores
