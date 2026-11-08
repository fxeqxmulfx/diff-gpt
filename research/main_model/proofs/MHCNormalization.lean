import Transformer.MHC.Section4_Sinkhorn

/-!
Exact-real safety of the positive-space part of the native four-stream kernel.
This is a local consequence of the paper's operators (2512.24880v2, §4.2),
not a floating-point, convergence, or whole-network Jacobian theorem.
The first cycle may have arbitrarily small row denominators and is kept in log
space. Subsequent column and row denominators are between 1/4 and 4.
-/

open scoped BigOperators
open Transformer.MHC
namespace MainModelMHC

/-- Normalizing rows of a positive column-stochastic 4x4 matrix leaves every
column sum in [1/4,4]. Exact-real local bound for the paper's row operator. -/
theorem row_normalize_column_bounds (M : Matrix (Fin 4) (Fin 4) ℝ)
    (hpos : ∀ i j, 0 < M i j) (hcols : ∀ j, ∑ i, M i j = 1) (j : Fin 4) :
    1 / 4 ≤ ∑ i, rowNormalize M i j ∧ ∑ i, rowNormalize M i j ≤ 4 := by
  have entry_le (i k : Fin 4) : M i k ≤ 1 := by
    calc
      M i k ≤ ∑ a, M a k := Finset.single_le_sum
        (fun a _ => (hpos a k).le) (Finset.mem_univ i)
      _ = 1 := hcols k
  have rows_le (i : Fin 4) : ∑ k, M i k ≤ 4 := by
    calc
      ∑ k, M i k ≤ ∑ _k : Fin 4, (1 : ℝ) := Finset.sum_le_sum
        (fun k _ => entry_le i k)
      _ = 4 := by norm_num
  constructor
  · calc
      (1 : ℝ) / 4 = (∑ i, M i j) / 4 := by rw [hcols j]
      _ = ∑ i, M i j / 4 := Finset.sum_div _ _ _
      _ ≤ ∑ i, rowNormalize M i j := by
        apply Finset.sum_le_sum
        intro i _
        exact div_le_div_of_nonneg_left (hpos i j).le
          (row_sum_pos M hpos i) (rows_le i)
  · calc
      ∑ i, rowNormalize M i j ≤ ∑ _i : Fin 4, (1 : ℝ) := by
        apply Finset.sum_le_sum
        intro i _
        calc
          rowNormalize M i j ≤ ∑ k, rowNormalize M i k :=
            Finset.single_le_sum (fun k _ => (rowNormalize_pos M hpos i k).le)
              (Finset.mem_univ j)
          _ = 1 := rowNormalize_sum M hpos i
      _ = 4 := by norm_num

/-- The column-stochastic and positivity hypotheses have a concrete witness. -/
example : (∀ i j : Fin 4, (0 : ℝ) < (fun _ _ : Fin 4 => (1 : ℝ) / 4) i j) ∧
    (∀ j : Fin 4, ∑ i : Fin 4, (fun _ _ : Fin 4 => (1 : ℝ) / 4) i j = 1) := by
  constructor <;> intros <;> norm_num

/-- Symmetric bound for the column operator when the input rows sum to one.
It bounds the next row denominators without assuming double stochasticity. -/
theorem column_normalize_row_bounds (M : Matrix (Fin 4) (Fin 4) ℝ)
    (hpos : ∀ i j, 0 < M i j) (hrows : ∀ i, ∑ j, M i j = 1) (i : Fin 4) :
    1 / 4 ≤ ∑ j, columnNormalize M i j ∧ ∑ j, columnNormalize M i j ≤ 4 := by
  simpa only [rowNormalize, columnNormalize, Matrix.of_apply, Matrix.transpose_apply] using
    row_normalize_column_bounds M.transpose (fun i j => hpos j i) hrows i

example : (∀ i j : Fin 4, (0 : ℝ) < (fun _ _ : Fin 4 => (1 : ℝ) / 4) i j) ∧
    (∀ i : Fin 4, ∑ j : Fin 4, (fun _ _ : Fin 4 => (1 : ℝ) / 4) i j = 1) := by
  constructor <;> intros <;> norm_num

/-- Any completed positive Sinkhorn cycle supplies bounded column denominators
for the following cycle. This is not exact double stochasticity. -/
theorem cycle_column_bounds (M : Matrix (Fin 4) (Fin 4) ℝ)
    (hpos : ∀ i j, 0 < M i j) (j : Fin 4) :
    1 / 4 ≤ ∑ i, sinkhornCycle M i j ∧ ∑ i, sinkhornCycle M i j ≤ 4 := by
  exact row_normalize_column_bounds (columnNormalize M)
    (columnNormalize_pos M hpos) (columnNormalize_sum M hpos) j

example : ∀ i j : Fin 4, (0 : ℝ) < (fun _ _ : Fin 4 => (1 : ℝ)) i j := by
  intros
  norm_num

/-- After the first cycle, both normalizations' denominators are in [1/4,4].
Thus the native optimization has no need for further log/exp evaluations in
exact arithmetic; rounding and underflow are tested independently in Rust. -/
theorem later_denominators (logits : Matrix (Fin 4) (Fin 4) ℝ) (t : ℕ) :
    (∀ j, 1 / 4 ≤ ∑ i, finiteSinkhorn logits (t + 1) i j ∧
      ∑ i, finiteSinkhorn logits (t + 1) i j ≤ 4) ∧
    (∀ i, 1 / 4 ≤ ∑ j, columnNormalize (finiteSinkhorn logits (t + 1)) i j ∧
      ∑ j, columnNormalize (finiteSinkhorn logits (t + 1)) i j ≤ 4) := by
  have hp (n : ℕ) : ∀ i j, 0 < finiteSinkhorn logits n i j :=
    sinkhornIterate_pos _ (fun i j => Real.exp_pos (logits i j)) n
  constructor
  · intro j
    exact cycle_column_bounds (finiteSinkhorn logits t) (hp t) j
  · intro i
    exact column_normalize_row_bounds _ (hp (t + 1))
      (finiteSinkhorn_row_sum logits t) i

example : ∀ j : Fin 4, 1 / 4 ≤ ∑ i, finiteSinkhorn (fun _ _ => (0 : ℝ)) 20 i j := by
  intro j
  exact ((later_denominators (fun _ _ => (0 : ℝ)) 19).1 j).1

end MainModelMHC

#print axioms MainModelMHC.row_normalize_column_bounds
#print axioms MainModelMHC.column_normalize_row_bounds
#print axioms MainModelMHC.cycle_column_bounds
#print axioms MainModelMHC.later_denominators
