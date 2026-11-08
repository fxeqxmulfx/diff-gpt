import Transformer.MHC.PeriRMS
import Transformer.Precision.ArithmeticError

/-!
User-requested adaptive precision for the selected mHC model, 2026-11-08.
Local certificates using the repository's IEEE nearest-rounding semantics and
existing Peri-RMS/QK bounds. These are exact-real error contracts, not a verified
translation of a CUDA kernel or a training-convergence guarantee.

FP16/BF16 counterexamples use bounded, representable summands; even an EXACT
final accumulator cannot recover an increment absorbed in a low-precision block.
The positive result accounts for every block and final-accumulator error, then
certifies a relative budget using only the approximate result and its bound.
-/
open scoped BigOperators
open Transformer Transformer.Precision
namespace MainModelPrecision

/-- Errors in low-precision blocks persist when their totals are combined.
The final accumulation has its own independently certified error. -/
theorem blocked_sum_error {ι : Type*} [Fintype ι]
    (exact approx bound : ι → ℝ) (total finalBound : ℝ)
    (hb : ∀ i, |approx i - exact i| ≤ bound i)
    (hf : |total - ∑ i, approx i| ≤ finalBound) :
    |total - ∑ i, exact i| ≤ finalBound + ∑ i, bound i := by
  have he : total - ∑ i, exact i =
      (total - ∑ i, approx i) + ∑ i, (approx i - exact i) := by
    rw [Finset.sum_sub_distrib]
    ring
  rw [he]
  calc
    _ ≤ |total - ∑ i, approx i| + |∑ i, (approx i - exact i)| := abs_add_le _ _
    _ ≤ finalBound + ∑ i, |approx i - exact i| :=
      add_le_add hf (Finset.abs_sum_le_sum_abs _ _)
    _ ≤ finalBound + ∑ i, bound i :=
      add_le_add le_rfl (Finset.sum_le_sum (fun i _ => hb i))

example : |(3 : ℝ) - ∑ _i : Fin 2, (3 / 2 : ℝ)| ≤ 0 + ∑ _i : Fin 2, (1 / 100 : ℝ) := by
  apply blocked_sum_error (fun _ : Fin 2 => 3 / 2) (fun _ => 3 / 2)
    (fun _ => 1 / 100) 3 0
  · intro i
    norm_num
  · norm_num

/-- A computable relative-error guard: when the certified error budget fits
tau*|approx|/(1+tau), the relative error against the unknown exact value is <=tau.
Near cancellation makes the guard stricter automatically. -/
theorem relative_guard (exact approx bound tau : ℝ) (ht : 0 ≤ tau)
    (he : |approx - exact| ≤ bound)
    (hg : (1 + tau) * bound ≤ tau * |approx|) :
    |approx - exact| ≤ tau * |exact| := by
  have ha : |approx| ≤ |exact| + |approx - exact| := by
    have h := abs_add_le exact (approx - exact)
    have heq : exact + (approx - exact) = approx := by ring
    simpa only [heq] using h
  have hmul := mul_le_mul_of_nonneg_left he (by linarith : 0 ≤ 1 + tau)
  have htri := mul_le_mul_of_nonneg_left ha ht
  nlinarith

example : |(1 : ℝ) - (1 + 1 / 8192)| ≤ (1 / 1000) * |1 + 1 / 8192| := by
  exact relative_guard (1 + 1 / 8192) 1 (1 / 8192) (1 / 1000)
    (by norm_num) (by norm_num) (by norm_num)

/-- The current model's Q/K normalization remains useful: a certified dot
error eta enlarges the exact log(N) score envelope by only log(N)*eta/d.
The error premise must include input conversion, products and all reductions. -/
theorem normalized_score_error {d : ℕ} (eps : ℝ) (heps : 0 < eps)
    (hd : 0 < d) (n : ℕ) (a b : Fin (d / 2) → ℝ) (q k : EucSpace d)
    (approx eta : ℝ)
    (he : |approx - inner ℝ (MHC.PeriRMS.rotatedQK eps a q)
        (MHC.PeriRMS.rotatedQK eps b k)| ≤ eta) :
    |(Real.log (n + 1) / (d : ℝ)) * approx| ≤
      Real.log (n + 1) + (Real.log (n + 1) / (d : ℝ)) * eta := by
  let dot := inner ℝ (MHC.PeriRMS.rotatedQK eps a q) (MHC.PeriRMS.rotatedQK eps b k)
  let c := Real.log (n + 1) / (d : ℝ)
  have hl : 0 ≤ Real.log (n + 1 : ℝ) :=
    Real.log_nonneg (by exact_mod_cast Nat.succ_le_succ (Nat.zero_le n))
  have hc : 0 ≤ c := div_nonneg hl (Nat.cast_nonneg d)
  have hs : |c * dot| ≤ Real.log (n + 1) := by
    simpa [c, dot, div_eq_mul_inv, mul_assoc] using
      MHC.PeriRMS.score_bound eps heps hd n a b q k
  have herr : |c * (approx - dot)| ≤ c * eta := by
    rw [abs_mul, abs_of_nonneg hc]
    exact mul_le_mul_of_nonneg_left he hc
  calc
    |c * approx| = |c * dot + c * (approx - dot)| := by congr 1; ring
    _ ≤ |c * dot| + |c * (approx - dot)| := abs_add_le _ _
    _ ≤ Real.log (n + 1) + c * eta := add_le_add hs herr

example (q k : EucSpace 16) :
    |(Real.log (64 : ℝ) / 16) * inner ℝ
      (MHC.PeriRMS.rotatedQK (1 / 8388608) (fun _ => 0) q)
      (MHC.PeriRMS.rotatedQK (1 / 8388608) (fun _ => 0) k)| ≤ Real.log 64 := by
  have h := normalized_score_error (1 / 8388608) (by norm_num) (by norm_num)
    63 (fun _ => 0) (fun _ => 0) q k
    (inner ℝ (MHC.PeriRMS.rotatedQK (1 / 8388608) (fun _ => 0) q)
      (MHC.PeriRMS.rotatedQK (1 / 8388608) (fun _ => 0) k)) 0 (by simp)
  norm_num at h ⊢
  exact h

/-- Absorption for actual binary16 nearest rounding, far from overflow. -/
theorem f16_absorbs_small (Q : ℝ → ℝ) (hQ : IsNearest (grid 5 10) Q) :
    Q (1 + 1 / (2 : ℝ) ^ 12) = 1 := by
  have h1 : (1 : ℝ) ∈ grid 5 10 := by
    exact ⟨15360, by norm_num, by norm_num [ieee, bias]⟩
  apply hQ.add_eq h1
  · intro z hz hz1
    exact (grid_hasSignificand 5 10).gap h1 hz (s := -10) (by norm_num) hz1
  · positivity
  · norm_num

/-- Absorption for actual bfloat16 nearest rounding, far from overflow. -/
theorem bf16_absorbs_small (Q : ℝ → ℝ) (hQ : IsNearest (grid 8 7) Q) :
    Q (1 + 1 / (2 : ℝ) ^ 9) = 1 := by
  have h1 : (1 : ℝ) ∈ grid 8 7 := by
    exact ⟨16256, by norm_num, by norm_num [ieee, bias]⟩
  apply hQ.add_eq h1
  · intro z hz hz1
    exact (grid_hasSignificand 8 7).gap h1 hz (s := -7) (by norm_num) hz1
  · positivity
  · norm_num

/-- Three representable summands have mean square<1, but a rounded first
block followed by EXACT final summation loses the entire nonzero result.
Thus norm bounds alone do not certify low-precision partial sums. -/
theorem f16_bounded_cancellation (Q : ℝ → ℝ) (hQ : IsNearest (grid 5 10) Q) :
    (1 : ℝ) ∈ grid 5 10 ∧ (1 / 4096 : ℝ) ∈ grid 5 10 ∧ (-1 : ℝ) ∈ grid 5 10 ∧
    ((1 : ℝ)^2 + (1/4096 : ℝ)^2 + (-1 : ℝ)^2) / 3 < 1 ∧
    Q (1 + 1/4096) - 1 = 0 ∧ (1 + 1/4096 : ℝ) - 1 = 1/4096 := by
  refine ⟨⟨15360, by norm_num, by norm_num [ieee, bias]⟩,
    ⟨3072, by norm_num, by norm_num [ieee, bias]⟩,
    ⟨48128, by norm_num, by norm_num [ieee, bias]⟩, by norm_num, ?_, by norm_num⟩
  have h := f16_absorbs_small Q hQ
  simp only [show (2 : ℝ)^12 = 4096 by norm_num] at h
  rw [h]
  ring

/-- The same mean-square<1 counterexample in bfloat16. -/
theorem bf16_bounded_cancellation (Q : ℝ → ℝ) (hQ : IsNearest (grid 8 7) Q) :
    (1 : ℝ) ∈ grid 8 7 ∧ (1 / 512 : ℝ) ∈ grid 8 7 ∧ (-1 : ℝ) ∈ grid 8 7 ∧
    ((1 : ℝ)^2 + (1/512 : ℝ)^2 + (-1 : ℝ)^2) / 3 < 1 ∧
    Q (1 + 1/512) - 1 = 0 ∧ (1 + 1/512 : ℝ) - 1 = 1/512 := by
  refine ⟨⟨16256, by norm_num, by norm_num [ieee, bias]⟩,
    ⟨15104, by norm_num, by norm_num [ieee, bias]⟩,
    ⟨49024, by norm_num, by norm_num [ieee, bias]⟩, by norm_num, ?_, by norm_num⟩
  have h := bf16_absorbs_small Q hQ
  simp only [show (2 : ℝ)^9 = 512 by norm_num] at h
  rw [h]
  ring

example : ∃ Q, IsNearest (grid 5 10) Q := exists_isNearest_grid (by norm_num) 10
example : ∃ Q, IsNearest (grid 8 7) Q := exists_isNearest_grid (by norm_num) 7

#print axioms blocked_sum_error
#print axioms relative_guard
#print axioms normalized_score_error
#print axioms f16_absorbs_small
#print axioms bf16_absorbs_small
#print axioms f16_bounded_cancellation
#print axioms bf16_bounded_cancellation
end MainModelPrecision
