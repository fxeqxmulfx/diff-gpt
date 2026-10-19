module

public import Mathlib
public import DiffGPT.LevelGrid

/-!
# int64 and uint64 arithmetic

numpy's int64 and uint64 arrays add and subtract modulo `2^64` without a check, as Lean's
`Int64` and `UInt64` do, so these are their model. The tokenizer relies on three facts.

* int64 sums and differences that stay in range are exact (`toInt_add_of_inInt64`,
  `toInt_sub_of_inInt64`), so `np.diff(L, n=k)` is exact while every difference up to the
  `k`-th stays in range (`toInt_diffI64_iter`): for `|L| ≤ 2^(62-k)`
  (`toInt_diffI64_iter_of_abs_le`) and for values in `[0, 2^G)` with `k + G ≤ 64`
  (`toInt_diffI64_iter_of_mem_Ico`).
* `_cumsum`'s test, terms of one sign with a wrapped sum of the other, fires exactly at the
  int64 sums that wrap (`toInt_add_eq_iff_not_sumWraps`): an int64 cumulative sum is the
  integer one unless it fires (`toInt_cumsumI64`), and it fires when an integer partial sum
  leaves int64 (`exists_sumWraps`). `_integrate_levels`, k such sums, is then exact in int64
  while the levels' differences stay in range (`toInt_integrateI64`).
* uint64 is a ring modulo `2^64`, so `_integrate_levels` in uint64 is the integer
  `integrate` modulo `2^64` (`map_integrate`), and masking with `2^G - 1` leaves it modulo
  `2^G` (`anchor_numerators`). The residues modulo `2^G` of an integer sequence with zero
  `k`-th differences have `k`-th differences divisible by `2^G` (`fwdDiff_iter_emod_eq`), below
  `2^(k-1+G)` in magnitude (`abs_fwdDiff_iter_emod_lt`).
-/

@[expose] public section

open fwdDiff

namespace DiffGPT

/-- The int64 range. -/
def InInt64 (n : ℤ) : Prop := -2 ^ 63 ≤ n ∧ n < 2 ^ 63

lemma inInt64_toInt (a : Int64) : InInt64 a.toInt := ⟨a.le_toInt, a.toInt_lt⟩

/-- `Int.bmod` modulo `2^64` fixes the int64 range. -/
lemma bmod_eq_of_inInt64 {n : ℤ} (h : InInt64 n) : n.bmod (2 ^ 64) = n := by
  obtain ⟨h1, h2⟩ := h
  exact Int.bmod_eq_of_le_mul_two (by push_cast; omega) (by push_cast; omega)

theorem toInt_add_of_inInt64 (a b : Int64) (h : InInt64 (a.toInt + b.toInt)) :
    (a + b).toInt = a.toInt + b.toInt := by
  rw [Int64.toInt_add, bmod_eq_of_inInt64 h]

theorem toInt_sub_of_inInt64 (a b : Int64) (h : InInt64 (a.toInt - b.toInt)) :
    (a - b).toInt = a.toInt - b.toInt := by
  rw [Int64.toInt_sub, bmod_eq_of_inInt64 h]

/-! ### `_cumsum` -/

/-- The test `_cumsum` applies to an int64 sum `a + b`: the terms share a sign that the
wrapped sum lacks. -/
def SumWraps (a b : Int64) : Prop :=
  (a.toInt < 0 ↔ b.toInt < 0) ∧ ¬((a + b).toInt < 0 ↔ a.toInt < 0)

/-- numpy computes the sign comparisons of `SumWraps` as `(a ^ b) >= 0`. -/
theorem toInt_xor_nonneg_iff (a b : Int64) :
    0 ≤ (a ^^^ b).toInt ↔ (a.toInt < 0 ↔ b.toInt < 0) := by
  have h (x : Int64) : x.toInt < 0 ↔ x.toBitVec.msb = true := by
    rw [BitVec.msb_eq_toInt, decide_eq_true_iff]
    rfl
  rw [← not_lt, h, h, h, Int64.toBitVec_xor, BitVec.msb_xor]
  cases a.toBitVec.msb <;> cases b.toBitVec.msb <;> simp

/-- The test fires exactly at the sums that wrap. -/
theorem toInt_add_eq_iff_not_sumWraps (a b : Int64) :
    (a + b).toInt = a.toInt + b.toInt ↔ ¬SumWraps a b := by
  have ha := inInt64_toInt a
  have hb := inInt64_toInt b
  unfold SumWraps InInt64 at *
  rw [Int64.toInt_add, Int.bmod_def]
  generalize a.toInt = x at *
  generalize b.toInt = y at *
  push_cast
  split <;> omega

/-- `np.cumsum` in int64. -/
def cumsumI64 (v : ℕ → Int64) : ℕ → Int64
  | 0 => v 0
  | n + 1 => cumsumI64 v n + v (n + 1)

/-- Unless the test fires, the int64 cumulative sum is the integer one. -/
theorem toInt_cumsumI64 (v : ℕ → Int64) (n : ℕ)
    (h : ∀ m < n, ¬SumWraps (cumsumI64 v m) (v (m + 1))) :
    (cumsumI64 v n).toInt = ∑ i ∈ Finset.range (n + 1), (v i).toInt := by
  induction n with
  | zero => simp [cumsumI64]
  | succ n ih =>
    rw [cumsumI64, (toInt_add_eq_iff_not_sumWraps _ _).mpr (h n (by omega)),
      ih fun m hm => h m (by omega), Finset.sum_range_succ _ (n + 1)]

/-- The test fires when an integer partial sum leaves int64. -/
theorem exists_sumWraps (v : ℕ → Int64) (n : ℕ)
    (h : ¬InInt64 (∑ i ∈ Finset.range (n + 1), (v i).toInt)) :
    ∃ m < n, SumWraps (cumsumI64 v m) (v (m + 1)) := by
  by_contra hc
  exact h (toInt_cumsumI64 v n (fun m hm hw => hc ⟨m, hm, hw⟩) ▸ inInt64_toInt _)

/-! ### `np.diff` -/

/-- `np.diff` in int64. -/
def diffI64 (f : ℕ → Int64) (n : ℕ) : Int64 := f (n + 1) - f n

lemma abs_fwdDiff_iter_le {M : ℤ} {f : ℕ → ℤ} (h : ∀ n, |f n| ≤ M) (k n : ℕ) :
    |Δ_[1] ^[k] f n| ≤ 2 ^ k * M := by
  induction k generalizing f M with
  | zero => simpa using h n
  | succ k ih =>
    rw [Function.iterate_succ_apply]
    have h2 : ∀ m, |Δ_[1] f m| ≤ 2 * M := fun m => by
      obtain ⟨b1, b2⟩ := abs_le.mp (h (m + 1))
      obtain ⟨c1, c2⟩ := abs_le.mp (h m)
      exact abs_le.mpr ⟨by simp only [fwdDiff]; linarith, by simp only [fwdDiff]; linarith⟩
    calc |Δ_[1] ^[k] (Δ_[1] f) n| ≤ 2 ^ k * (2 * M) := ih h2
      _ = 2 ^ (k + 1) * M := by ring

/-- `k ≥ 1` differences of integers in `[0, B)` stay below `2^(k-1)·B` in magnitude. -/
lemma abs_fwdDiff_iter_lt_of_mem_Ico {B : ℤ} {f : ℕ → ℤ} (h : ∀ n, 0 ≤ f n ∧ f n < B)
    {k : ℕ} (hk : 1 ≤ k) (n : ℕ) : |Δ_[1] ^[k] f n| < 2 ^ (k - 1) * B := by
  obtain ⟨j, rfl⟩ : ∃ j, k = j + 1 := ⟨k - 1, by omega⟩
  rw [Function.iterate_succ_apply, Nat.add_sub_cancel]
  have h1 : ∀ m, |Δ_[1] f m| ≤ B - 1 := fun m => by
    obtain ⟨a1, a2⟩ := h m
    obtain ⟨b1, b2⟩ := h (m + 1)
    exact abs_le.mpr ⟨by simp only [fwdDiff]; linarith, by simp only [fwdDiff]; linarith⟩
  calc |Δ_[1] ^[j] (Δ_[1] f) n| ≤ 2 ^ j * (B - 1) := abs_fwdDiff_iter_le h1 j n
    _ < 2 ^ j * B := by have := pow_pos (two_pos : (0 : ℤ) < 2) j; nlinarith

/-- `np.diff(f, n=k)` in int64 is exact while every difference up to the `k`-th fits. -/
theorem toInt_diffI64_iter {k : ℕ} {f : ℕ → Int64}
    (h : ∀ j ≤ k, ∀ n, InInt64 (Δ_[1] ^[j] (fun m => (f m).toInt) n)) (n : ℕ) :
    (diffI64^[k] f n).toInt = Δ_[1] ^[k] (fun m => (f m).toInt) n := by
  induction k generalizing f with
  | zero => rfl
  | succ k ih =>
    have hd : (fun m => (diffI64 f m).toInt) = Δ_[1] (fun m => (f m).toInt) := funext fun m =>
      toInt_sub_of_inInt64 _ _ (h 1 (by omega) m)
    rw [Function.iterate_succ_apply, Function.iterate_succ_apply, ← hd]
    exact ih fun j hj m => by
      rw [hd, ← Function.iterate_succ_apply]
      exact h (j + 1) (by omega) m

/-- The levels: `|L| ≤ 2^(62-k)`. -/
theorem toInt_diffI64_iter_of_abs_le {k : ℕ} (hk : k ≤ 62) {f : ℕ → Int64}
    (h : ∀ n, |(f n).toInt| ≤ 2 ^ (62 - k)) (n : ℕ) :
    (diffI64^[k] f n).toInt = Δ_[1] ^[k] (fun m => (f m).toInt) n := by
  refine toInt_diffI64_iter (fun j hj m => ?_) n
  have hp : (2 : ℤ) ^ j * 2 ^ (62 - k) ≤ 2 ^ 62 := by
    rw [← pow_add]
    exact pow_le_pow_right₀ (by norm_num) (by omega)
  obtain ⟨b1, b2⟩ := abs_le.mp (abs_fwdDiff_iter_le h j m)
  exact ⟨by linarith, by linarith⟩

/-- The anchor numerators: values in `[0, 2^G)` with `k + G ≤ 64`. -/
theorem toInt_diffI64_iter_of_mem_Ico {k G : ℕ} (hkG : k + G ≤ 64) {f : ℕ → Int64}
    (h : ∀ n, 0 ≤ (f n).toInt ∧ (f n).toInt < 2 ^ G) (n : ℕ) :
    (diffI64^[k] f n).toInt = Δ_[1] ^[k] (fun m => (f m).toInt) n := by
  refine toInt_diffI64_iter (fun j hj m => ?_) n
  rcases Nat.eq_zero_or_pos j with rfl | hj1
  · exact inInt64_toInt (f m)
  have hp : (2 : ℤ) ^ (j - 1) * 2 ^ G ≤ 2 ^ 63 := by
    rw [← pow_add]
    exact pow_le_pow_right₀ (by norm_num) (by omega)
  obtain ⟨b1, b2⟩ := abs_lt.mp (abs_fwdDiff_iter_lt_of_mem_Ico h hj1 m)
  exact ⟨by linarith, by linarith⟩

/-! ### `_integrate_levels` -/

/-- `_integrate_levels`: the sequence that starts with `pre 0, …, pre (k-1)` and has the
`k`-th differences `d`, by `k` cumulative sums from the first differences of `pre`. -/
def integrate {R : Type*} [AddCommGroup R] : ℕ → (ℕ → R) → (ℕ → R) → ℕ → R
  | 0, _, d => d
  | k + 1, pre, d => fun n => pre 0 + ∑ i ∈ Finset.range n, integrate k (Δ_[1] pre) d i

/-- One difference of `integrate (k + 1)` is `integrate k` of the differences of the prefix. -/
lemma fwdDiff_integrate_succ {R : Type*} [AddCommGroup R] (k : ℕ) (pre d : ℕ → R) :
    Δ_[1] (integrate (k + 1) pre d) = integrate k (Δ_[1] pre) d := by
  funext n
  simp only [fwdDiff, integrate, Finset.sum_range_succ]
  abel

theorem fwdDiff_iter_integrate {R : Type*} [AddCommGroup R] (k : ℕ) (pre d : ℕ → R) :
    Δ_[1] ^[k] (integrate k pre d) = d := by
  induction k generalizing pre with
  | zero => rfl
  | succ k ih => rw [Function.iterate_succ_apply, fwdDiff_integrate_succ, ih]

theorem integrate_of_lt {R : Type*} [AddCommGroup R] {k n : ℕ} (hn : n < k) (pre d : ℕ → R) :
    integrate k pre d n = pre n := by
  induction k generalizing pre n with
  | zero => omega
  | succ k ih =>
    simp only [integrate]
    rw [Finset.sum_congr rfl fun i hi => ih (by simp at hi; omega) (Δ_[1] pre)]
    simp only [fwdDiff]
    rw [Finset.sum_range_sub pre n]
    abel

/-- `integrate k` reads the prefix only below `k`. -/
lemma integrate_congr {R : Type*} [AddCommGroup R] {k : ℕ} {pre pre' : ℕ → R}
    (h : ∀ i < k, pre i = pre' i) (d : ℕ → R) : integrate k pre d = integrate k pre' d := by
  induction k generalizing pre pre' with
  | zero => rfl
  | succ k ih =>
    have hd : integrate k (Δ_[1] pre) d = integrate k (Δ_[1] pre') d :=
      ih fun i hi => by simp only [fwdDiff]; rw [h i (by omega), h (i + 1) (by omega)]
    funext n
    simp only [integrate, hd, h 0 (by omega)]

/-- An additive map, such as the reduction of integers modulo `2^64`, commutes with
`integrate`. -/
theorem map_integrate {R S : Type*} [AddCommGroup R] [AddCommGroup S] (φ : R →+ S) (k : ℕ)
    (pre d : ℕ → R) (n : ℕ) :
    φ (integrate k pre d n) = integrate k (φ ∘ pre) (φ ∘ d) n := by
  induction k generalizing pre n with
  | zero => rfl
  | succ k ih =>
    have hφ : φ ∘ Δ_[1] pre = Δ_[1] (φ ∘ pre) := by
      funext m
      simp [fwdDiff]
    simp only [integrate, map_add, map_sum, ih, hφ]
    rfl

/-- `_integrate_levels` in int64: `_cumsum(concatenate([first, levels]))` per order, with the
firsts from `np.diff` of the prefix levels. -/
def integrateI64 : ℕ → (ℕ → Int64) → (ℕ → Int64) → ℕ → Int64
  | 0, _, d => d
  | k + 1, pre, d =>
    cumsumI64 fun n => match n with
      | 0 => pre 0
      | m + 1 => integrateI64 k (diffI64 pre) d m

/-- `_integrate_levels` is exact in int64 on the first `M` rows while the differences of the
levels it returns, up to the `k`-th, stay in range there: the decoder gets the encoder's
levels back. (The `j`-th differences span the rows `n, …, n + j`.) -/
theorem toInt_integrateI64 {k M : ℕ} (hkM : k ≤ M) {pre d : ℕ → Int64}
    (h : ∀ j ≤ k, ∀ n, n + j < M → InInt64
      (Δ_[1] ^[j] (integrate k (fun m => (pre m).toInt) (fun m => (d m).toInt)) n))
    {n : ℕ} (hn : n < M) :
    (integrateI64 k pre d n).toInt =
      integrate k (fun m => (pre m).toInt) (fun m => (d m).toInt) n := by
  induction k generalizing M pre n with
  | zero => rfl
  | succ k ih =>
    set P : ℕ → ℤ := fun m => (pre m).toInt
    set D : ℕ → ℤ := fun m => (d m).toInt
    set L := integrate (k + 1) P D with hL
    -- below `k` the prefix is the levels, so its int64 first differences are exact
    have hpre : ∀ i < k, (diffI64 pre i).toInt = Δ_[1] P i := fun i hi => by
      have h1 := h 1 (by omega) i (by omega)
      rw [Function.iterate_one] at h1
      have e1 : L i = P i := integrate_of_lt (by omega) P D
      have e2 : L (i + 1) = P (i + 1) := integrate_of_lt (by omega) P D
      simp only [fwdDiff] at h1 ⊢
      rw [e1, e2] at h1
      exact toInt_sub_of_inInt64 _ _ h1
    -- the inner call integrates the first differences of the levels
    have hinner : integrate k (fun m => (diffI64 pre m).toInt) D = Δ_[1] L := by
      rw [hL, fwdDiff_integrate_succ]
      exact integrate_congr hpre D
    have hIH : ∀ m, m + 1 < M → (integrateI64 k (diffI64 pre) d m).toInt = Δ_[1] L m := by
      intro m hm
      rw [← hinner]
      refine ih (M := M - 1) (by omega) (fun j hj m' hm' => ?_) (by omega)
      rw [hinner, ← Function.iterate_succ_apply]
      exact h (j + 1) (by omega) m' (by omega)
    -- the outer cumulative sum stays on the levels, which are in range
    induction n with
    | zero => exact (integrate_of_lt (by omega) P D).symm
    | succ m ihm =>
      have e : integrateI64 (k + 1) pre d (m + 1) =
          integrateI64 (k + 1) pre d m + integrateI64 k (diffI64 pre) d m := rfl
      have hm : L m + Δ_[1] L m = L (m + 1) := by simp [fwdDiff]
      rw [e, toInt_add_of_inInt64, ihm (by omega), hIH m hn, hm]
      rw [ihm (by omega), hIH m hn, hm]
      simpa using h 0 (by omega) (m + 1) (by omega)

/-! ### The "mash" anchor -/

/-- The residues modulo `2^G` of a sequence with zero `k`-th differences have `k`-th
differences `2^G` times those of the quotients, negated. -/
theorem fwdDiff_iter_emod_eq {P : ℕ → ℤ} {k : ℕ} (hP : Δ_[1] ^[k] P = 0) (G n : ℕ) :
    Δ_[1] ^[k] (fun m => P m % 2 ^ G) n = -(2 ^ G * Δ_[1] ^[k] (fun m => P m / 2 ^ G) n) := by
  have e : (fun m => P m % 2 ^ G) = P + (-(2 : ℤ) ^ G) • fun m => P m / 2 ^ G := by
    funext m
    simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul]
    have := Int.mul_ediv_add_emod (P m) (2 ^ G)
    linarith
  rw [e, fwdDiff_iter_add, fwdDiff_iter_const_smul, hP]
  simp

/-- The `k`-th differences of residues modulo `2^G` stay below `2^(k-1+G)` in magnitude. -/
theorem abs_fwdDiff_iter_emod_lt (P : ℕ → ℤ) {k : ℕ} (hk : 1 ≤ k) (G n : ℕ) :
    |Δ_[1] ^[k] (fun m => P m % 2 ^ G) n| < 2 ^ (k - 1 + G) := by
  rw [pow_add]
  exact abs_fwdDiff_iter_lt_of_mem_Ico
    (fun m => ⟨Int.emod_nonneg _ (by positivity), Int.emod_lt_of_pos _ (by positivity)⟩) hk n

open scoped UInt64.CommRing in
/-- The anchor numerators: `_integrate_levels` of the seeds in uint64, masked with `2^G - 1`,
are the integer anchor `P = integrate k seeds 0` modulo `2^G`. -/
theorem anchor_numerators {G : ℕ} (hG : G ≤ 64) (k : ℕ) (seeds : ℕ → UInt64) (n : ℕ) :
    (((integrate k seeds 0 n) &&& UInt64.ofNat (2 ^ G - 1)).toNat : ℤ) =
      integrate k (fun m => ((seeds m).toNat : ℤ)) 0 n % 2 ^ G := by
  set Z := integrate k (fun m => ((seeds m).toNat : ℤ)) 0 n
  -- the uint64 integration is the integer one cast to uint64
  have hZ : integrate k seeds 0 n = ((Z : ℤ) : UInt64) := by
    have h := map_integrate (Int.castRingHom UInt64).toAddMonoidHom k
      (fun m => ((seeds m).toNat : ℤ)) 0 n
    have hs : (Int.castRingHom UInt64).toAddMonoidHom ∘ (fun m => ((seeds m).toNat : ℤ)) =
        seeds := by
      funext m
      simp only [Function.comp_apply, RingHom.toAddMonoidHom_eq_coe, AddMonoidHom.coe_ofClass,
        eq_intCast, Int.cast_natCast]
      exact UInt64.ofNat_toNat
    have h0 : (Int.castRingHom UInt64).toAddMonoidHom ∘ (0 : ℕ → ℤ) = 0 := by
      funext m
      simp
    rw [hs, h0] at h
    exact h.symm
  have hmask : (UInt64.ofNat (2 ^ G - 1)).toNat = 2 ^ G - 1 := by
    rw [UInt64.toNat_ofNat']
    apply Nat.mod_eq_of_lt
    have := Nat.pow_le_pow_right two_pos hG
    have := Nat.one_le_two_pow (n := G)
    omega
  have hcast : ((Z : ℤ) : UInt64).toNat = (Z % 2 ^ 64).toNat := by
    rw [← UInt64.toNat_toBitVec, UInt64.toBitVec_intCast]
    simp
  rw [hZ, UInt64.toNat_and, hmask, Nat.and_two_pow_sub_one_eq_mod, hcast]
  push_cast
  rw [Int.toNat_of_nonneg (Int.emod_nonneg _ (by norm_num))]
  exact Int.emod_emod_of_dvd Z (pow_dvd_pow 2 hG)

end DiffGPT
