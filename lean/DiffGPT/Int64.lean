module

public import Mathlib
public import DiffGPT.LevelGrid

/-!
# int64 and uint64 arithmetic

numpy's int64 and uint64 arrays add and subtract modulo `2^64` without a check, as Lean's
`Int64` and `UInt64` do, so these are their model. The tokenizer relies on four facts.

* int64 sums and differences are the integer ones modulo `2^64` (`toInt_add_modEq`, and
  `toInt_diffsI64_modEq` for `_difference`), so a result that lies in int64 is the integer one
  (`eq_of_modEq_of_inInt64`), however its parts wrapped. Levels in `[-2^b - 1, 2^b]` keep their
  differences by any `m` factors `1 - B^S` in int64 for `b + m ≤ 62`
  (`inInt64_diffs_of_mem_Icc`), and values in `[0, 2^G)` theirs for `m + G ≤ 64`
  (`toInt_diffsI64_of_mem_Ico`).
* `_cumsum`'s test, terms of one sign with a wrapped sum of the other, fires exactly at the
  int64 sums that wrap (`toInt_add_eq_iff_not_sumWraps`): an int64 sum along every `S`-th value
  is the integer one unless it fires (`toInt_lagSum`), and it fires when an integer sum leaves
  int64 (`lagSumRaises_of_not_inInt64`). `_invert_difference`, one such sum per factor, is then
  exact in int64 and does not raise (`InvertRaises`) while the differences of the levels it
  returns by the first factors stay in range (`toInt_invertI64`).
* uint64 is a ring modulo `2^64`, so `_invert_difference` in uint64 is the integer `invert`
  modulo `2^64` (`map_invert`), and masking with `2^G - 1` leaves it modulo `2^G`
  (`anchor_numerators`). The residues modulo `2^G` of an integer sequence in the kernel of
  `c(B)` have `c(B)` divisible by `2^G` (`diffs_emod_eq`), below `2^(m-1+G)` in magnitude
  (`abs_diffs_emod_lt`).
* `>> G` is the floor division by `2^G` (`toInt_shiftRight_ofNat`), and `.astype(np.int64)`
  keeps uint64 values below `2^63` (`toInt_toInt64_of_lt`).
-/@[expose] public section

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

/-! ### Modulo `2^64` -/

lemma toInt_add_modEq (a b : Int64) : (a + b).toInt ≡ a.toInt + b.toInt [ZMOD 2 ^ 64] := by
  rw [Int64.toInt_add]
  exact Int.bmod_emod

lemma toInt_sub_modEq (a b : Int64) : (a - b).toInt ≡ a.toInt - b.toInt [ZMOD 2 ^ 64] := by
  rw [Int64.toInt_sub]
  exact Int.bmod_emod

/-- numpy casts a Python int operand to int64 modulo `2^64`. -/
lemma toInt_ofInt_modEq (n : ℤ) : (Int64.ofInt n).toInt ≡ n [ZMOD 2 ^ 64] := by
  rw [Int64.toInt_ofInt]
  exact Int.bmod_emod

/-- Two integers in int64 that agree modulo `2^64` are equal. -/
lemma eq_of_modEq_of_inInt64 {a b : ℤ} (h : a ≡ b [ZMOD 2 ^ 64]) (ha : InInt64 a)
    (hb : InInt64 b) : a = b := by
  unfold Int.ModEq InInt64 at *
  omega

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

/-- `_cumsum` along every `S`-th of `M` int64 values raises: the test fires at one of its sums.
(The zeros that pad its last row never fire it.) -/
def LagSumRaises (S : ℕ) (v : ℕ → Int64) (M : ℕ) : Prop :=
  ∃ t, S ≤ t ∧ t < M ∧ SumWraps (lagSum S v (t - S)) (v t)

/-- Unless the test fires, the int64 sums along every `S`-th value are the integer ones. -/
theorem toInt_lagSum {S M : ℕ} {v : ℕ → Int64} (h : ¬LagSumRaises S v M) {n : ℕ}
    (hn : n < M) : (lagSum S v n).toInt = lagSum S (fun m => (v m).toInt) n := by
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    by_cases hc : 0 < S ∧ S ≤ n
    · rw [lagSum_of_le hc.1 hc.2, lagSum_of_le hc.1 hc.2,
        (toInt_add_eq_iff_not_sumWraps _ _).mpr fun hw => h ⟨n, hc.2, hn, hw⟩,
        ih (n - S) (by omega) (by omega)]
    · rw [lagSum, dite_eq_right hc, lagSum, dite_eq_right hc]

/-- The test fires when an integer sum leaves int64. -/
theorem lagSumRaises_of_not_inInt64 {S M : ℕ} (v : ℕ → Int64) {n : ℕ} (hn : n < M)
    (h : ¬InInt64 (lagSum S (fun m => (v m).toInt) n)) : LagSumRaises S v M := by
  by_contra hc
  exact h (toInt_lagSum hc hn ▸ inInt64_toInt _)

/-! ### `_difference` -/

/-- `values[S:] - values[:-S]` in int64. -/
def lagDiffI64 (S : ℕ) (f : ℕ → Int64) (n : ℕ) : Int64 := f (n + S) - f n

/-- `_difference` in int64: the factors in turn. -/
def diffsI64 : List ℕ → (ℕ → Int64) → ℕ → Int64
  | [], f => f
  | S :: l, f => diffsI64 l (lagDiffI64 S f)

/-- `c(B)` keeps congruences. -/
lemma diffs_modEq {c : ℤ} {f g : ℕ → ℤ} (h : ∀ i, f i ≡ g i [ZMOD c]) (l : List ℕ) (n : ℕ) :
    diffs l f n ≡ diffs l g n [ZMOD c] := by
  induction l generalizing f g with
  | nil => exact h n
  | cons S l ih => exact ih fun i => (h (i + S)).sub (h i)

/-- `_difference` in int64 is the integer `c(B)` modulo `2^64`, overflow or not. -/
theorem toInt_diffsI64_modEq (l : List ℕ) (f : ℕ → Int64) (n : ℕ) :
    (diffsI64 l f n).toInt ≡ diffs l (fun m => (f m).toInt) n [ZMOD 2 ^ 64] := by
  induction l generalizing f with
  | nil => rfl
  | cons S l ih =>
    rw [diffsI64, diffs_cons]
    exact (ih (lagDiffI64 S f)).trans (diffs_modEq (fun i => toInt_sub_modEq _ _) l n)

/-- Each factor at most doubles the magnitude. -/
lemma abs_diffs_le {M : ℤ} {f : ℕ → ℤ} (h : ∀ n, |f n| ≤ M) (l : List ℕ) (n : ℕ) :
    |diffs l f n| ≤ 2 ^ l.length * M := by
  induction l generalizing f M with
  | nil => simpa using h n
  | cons S l ih =>
    have h2 : ∀ m, |Δ_[S] f m| ≤ 2 * M := fun m => by
      obtain ⟨b1, b2⟩ := abs_le.mp (h (m + S))
      obtain ⟨c1, c2⟩ := abs_le.mp (h m)
      exact abs_le.mpr ⟨by simp only [fwdDiff]; linarith, by simp only [fwdDiff]; linarith⟩
    calc |diffs (S :: l) f n| = |diffs l (Δ_[S] f) n| := rfl
      _ ≤ 2 ^ l.length * (2 * M) := ih h2
      _ = 2 ^ (S :: l).length * M := by rw [List.length_cons, pow_succ]; ring

/-- At least one factor: values in `[a, b]` have differences within `2^(m-1)·(b - a)`. -/
lemma abs_diffs_le_of_mem_Icc {a b : ℤ} {f : ℕ → ℤ} (h : ∀ n, a ≤ f n ∧ f n ≤ b)
    {l : List ℕ} (hl : l ≠ []) (n : ℕ) : |diffs l f n| ≤ 2 ^ (l.length - 1) * (b - a) := by
  obtain ⟨S, l, rfl⟩ := List.exists_cons_of_ne_nil hl
  rw [diffs_cons, List.length_cons, Nat.add_sub_cancel]
  refine abs_diffs_le (fun m => ?_) l n
  obtain ⟨a1, a2⟩ := h m
  obtain ⟨b1, b2⟩ := h (m + S)
  exact abs_le.mpr ⟨by simp only [fwdDiff]; linarith, by simp only [fwdDiff]; linarith⟩

/-- At least one factor: values in `[0, B)` have differences below `2^(m-1)·B` in magnitude. -/
lemma abs_diffs_lt_of_mem_Ico {B : ℤ} {f : ℕ → ℤ} (h : ∀ n, 0 ≤ f n ∧ f n < B)
    {l : List ℕ} (hl : l ≠ []) (n : ℕ) : |diffs l f n| < 2 ^ (l.length - 1) * B := by
  have h1 := abs_diffs_le_of_mem_Icc (b := B - 1)
    (fun m => ⟨(h m).1, by have := (h m).2; omega⟩) hl n
  have := pow_pos (two_pos : (0 : ℤ) < 2) (l.length - 1)
  linarith

/-- `abs_diffs_le_of_mem_Icc` on the first `M` values, which the differences there read. -/
lemma abs_diffs_le_of_mem_Icc_of_lt {a b : ℤ} {M : ℕ} {f : ℕ → ℤ}
    (h : ∀ m < M, a ≤ f m ∧ f m ≤ b) {l : List ℕ} (hl : l ≠ []) {n : ℕ} (hn : n + l.sum < M) :
    |diffs l f n| ≤ 2 ^ (l.length - 1) * (b - a) := by
  have hab : a ≤ b := (h n (by omega)).1.trans (h n (by omega)).2
  set f' : ℕ → ℤ := fun m => if m < M then f m else a
  have hc : diffs l f n = diffs l f' n :=
    diffs_congr fun i hi => by simp only [f']; rw [ite_eq_left (by omega)]
  rw [hc]
  refine abs_diffs_le_of_mem_Icc (fun m => ?_) hl n
  simp only [f']
  split_ifs with hm
  exacts [h m hm, ⟨le_rfl, hab⟩]

/-- The levels under `_levels`' guard `|p| < 2^b`: on the first `M` values in
`[-2^b - 1, 2^b]` with `b + m ≤ 62` for `m` factors, their differences stay below `3·2^61`
there. -/
lemma inInt64_diffs_of_mem_Icc {l : List ℕ} {b M : ℕ} (hbl : b + l.length ≤ 62) {f : ℕ → ℤ}
    (h : ∀ m < M, -2 ^ b - 1 ≤ f m ∧ f m ≤ 2 ^ b) {n : ℕ} (hn : n + l.sum < M) :
    InInt64 (diffs l f n) := by
  have hb2 : (2 : ℤ) ^ b ≤ 2 ^ 62 := pow_le_pow_right₀ (by norm_num) (by omega)
  rcases eq_or_ne l [] with rfl | hl
  · obtain ⟨h1, h2⟩ := h n (by simpa using hn)
    exact ⟨by simp only [diffs_nil]; linarith, by simp only [diffs_nil]; linarith⟩
  have hl1 := List.length_pos_of_ne_nil hl
  have h1 : (2 : ℤ) ^ (l.length - 1) * 2 ^ b ≤ 2 ^ 61 := by
    rw [← pow_add]
    exact pow_le_pow_right₀ (by norm_num) (by omega)
  have h2 : (2 : ℤ) ^ (l.length - 1) ≤ 2 ^ 61 := pow_le_pow_right₀ (by norm_num) (by omega)
  have h3 : (2 : ℤ) ^ (l.length - 1) * (2 ^ b - (-2 ^ b - 1)) =
      2 * (2 ^ (l.length - 1) * 2 ^ b) + 2 ^ (l.length - 1) := by ring
  obtain ⟨b1, b2⟩ := abs_le.mp (abs_diffs_le_of_mem_Icc_of_lt h hl hn)
  exact ⟨by linarith, by linarith⟩

/-- The anchor numerators: values in `[0, 2^G)` with `m + G ≤ 64`. -/
theorem toInt_diffsI64_of_mem_Ico {l : List ℕ} {G : ℕ} (hlG : l.length + G ≤ 64)
    {f : ℕ → Int64} (h : ∀ n, 0 ≤ (f n).toInt ∧ (f n).toInt < 2 ^ G) (n : ℕ) :
    (diffsI64 l f n).toInt = diffs l (fun m => (f m).toInt) n := by
  rcases eq_or_ne l [] with rfl | hl
  · rfl
  refine eq_of_modEq_of_inInt64 (toInt_diffsI64_modEq l f n) (inInt64_toInt _) ?_
  have hp : (2 : ℤ) ^ (l.length - 1) * 2 ^ G ≤ 2 ^ 63 := by
    rw [← pow_add]
    exact pow_le_pow_right₀ (by norm_num) (by have := List.length_pos_of_ne_nil hl; omega)
  obtain ⟨b1, b2⟩ := abs_lt.mp (abs_diffs_lt_of_mem_Ico h hl n)
  exact ⟨by linarith, by linarith⟩

/-! ### `_invert_difference` -/

/-- `_invert_difference` in int64: `_cumsum` along the first lag `S` of the first `S` values of
the prefix followed by the values the other lags give, from the prefix differenced by `S` in
int64. -/
def invertI64 : List ℕ → (ℕ → Int64) → (ℕ → Int64) → ℕ → Int64
  | [], _, d => d
  | S :: l, pre, d =>
    lagSum S fun n => if n < S then pre n else invertI64 l (lagDiffI64 S pre) d (n - S)

/-- `_invert_difference` raises on `M` values: one of its `_cumsum`s, over `M`, `M - S₁`, …
values from the outermost in, fires. -/
def InvertRaises : List ℕ → (ℕ → Int64) → (ℕ → Int64) → ℕ → Prop
  | [], _, _, _ => False
  | S :: l, pre, d, M =>
    LagSumRaises S (fun n => if n < S then pre n else invertI64 l (lagDiffI64 S pre) d (n - S))
      M ∨ InvertRaises l (lagDiffI64 S pre) d (M - S)

/-- `_invert_difference` on the first `M` values is exact in int64 and does not raise while
the differences of the levels it returns by the first factors, `l₁ <+: l`, stay in range there:
the decoder gets the encoder's levels back. (Those differences span the values
`n, …, n + l₁.sum`.) -/
theorem toInt_invertI64 {l : List ℕ} (hl : ∀ S ∈ l, 0 < S) {M : ℕ} (hM : l.sum ≤ M)
    {pre d : ℕ → Int64}
    (h : ∀ l₁, l₁ <+: l → ∀ n, n + l₁.sum < M → InInt64
      (diffs l₁ (invert l (fun m => (pre m).toInt) (fun m => (d m).toInt)) n)) :
    (∀ n < M, (invertI64 l pre d n).toInt =
      invert l (fun m => (pre m).toInt) (fun m => (d m).toInt) n) ∧
      ¬InvertRaises l pre d M := by
  induction l generalizing M pre with
  | nil => exact ⟨fun n _ => rfl, id⟩
  | cons S l ih =>
    have hS : 0 < S := hl S (by simp)
    have hl' : ∀ T ∈ l, 0 < T := fun T hT => hl T (by simp [hT])
    rw [List.sum_cons] at hM
    set P : ℕ → ℤ := fun m => (pre m).toInt
    set D : ℕ → ℤ := fun m => (d m).toInt
    set L := invert (S :: l) P D with hL
    -- the levels are in range
    have hL0 : ∀ n < M, InInt64 (L n) := fun n hn =>
      h [] List.nil_prefix n (by simpa using hn)
    -- below `S + l.sum` the levels are the prefix, so its int64 differences along `S` are exact
    have hpre : ∀ i < l.sum, (lagDiffI64 S pre i).toInt = Δ_[S] P i := fun i hi => by
      have h1 := h [S] (by simp) i (by simp; omega)
      have e1 : L i = P i := invert_of_lt hl P D (by simp; omega)
      have e2 : L (i + S) = P (i + S) := invert_of_lt hl P D (by simp; omega)
      simp only [diffs_cons, diffs_nil, fwdDiff] at h1
      rw [e1, e2] at h1
      exact toInt_sub_of_inInt64 _ _ h1
    -- the inner call inverts the differences of the levels along `S`
    have hinner : invert l (fun m => (lagDiffI64 S pre m).toInt) D = Δ_[S] L := by
      rw [hL, fwdDiff_invert_cons hS]
      exact invert_congr hpre D
    obtain ⟨hIH, hnr⟩ := ih hl' (M := M - S) (pre := lagDiffI64 S pre) (by omega)
      fun l₁ hl₁ n hn => by
        rw [hinner]
        exact h (S :: l₁) (List.cons_prefix_cons.mpr ⟨rfl, hl₁⟩) n
          (by rw [List.sum_cons]; omega)
    rw [hinner] at hIH
    -- the outer sums along `S` stay on the levels, which are in range
    set v : ℕ → Int64 := fun n =>
      if n < S then pre n else invertI64 l (lagDiffI64 S pre) d (n - S) with hv_def
    have e : invertI64 (S :: l) pre d = lagSum S v := rfl
    have hv : ∀ n, S ≤ n → v n = invertI64 l (lagDiffI64 S pre) d (n - S) := fun n hn => by
      simp only [hv_def]
      rw [ite_eq_right (by omega)]
    have hval : ∀ n < M, (lagSum S v n).toInt = L n := by
      intro n hn
      induction n using Nat.strong_induction_on with
      | _ n ihn =>
        by_cases hnS : n < S
        · simp only [lagSum_of_lt hnS, hv_def]
          rw [ite_eq_left hnS]
          exact (invert_of_lt hl P D (by simp; omega)).symm
        · have h1 := ihn (n - S) (by omega) (by omega)
          have h2 := hIH (n - S) (by omega)
          have hsum : L (n - S) + Δ_[S] L (n - S) = L n := by
            simp only [fwdDiff, Nat.sub_add_cancel (not_lt.mp hnS)]
            abel
          rw [lagSum_of_le hS (not_lt.mp hnS), hv n (not_lt.mp hnS), toInt_add_of_inInt64, h1,
            h2, hsum]
          rw [h1, h2, hsum]
          exact hL0 n hn
    -- so no outer sum wraps
    refine ⟨fun n hn => by rw [e]; exact hval n hn, ?_⟩
    rintro (⟨t, ht1, ht2, hw⟩ | hr)
    · have hw' : SumWraps (lagSum S v (t - S)) (v t) := hw
      refine (toInt_add_eq_iff_not_sumWraps _ _).mp ?_ hw'
      rw [← lagSum_of_le hS ht1 v, hval t ht2, hval (t - S) (by omega), hv t ht1,
        hIH (t - S) (by omega)]
      simp only [fwdDiff, Nat.sub_add_cancel ht1]
      abel
    · exact hnr hr

/-! ### The "mash" anchor -/

/-- The residues modulo `2^G` of a sequence in the kernel of `c(B)` have `c(B)` `2^G` times
that of the quotients, negated. -/
theorem diffs_emod_eq {l : List ℕ} {P : ℕ → ℤ} (hP : diffs l P = 0) (G n : ℕ) :
    diffs l (fun m => P m % 2 ^ G) n = -(2 ^ G * diffs l (fun m => P m / 2 ^ G) n) := by
  have e : (fun m => P m % 2 ^ G) = P + (-(2 : ℤ) ^ G) • fun m => P m / 2 ^ G := by
    funext m
    simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul]
    have := Int.mul_ediv_add_emod (P m) (2 ^ G)
    linarith
  rw [e, diffs_add, diffs_const_smul, hP]
  simp

/-- At least one factor: `c(B)` of residues modulo `2^G` stays below `2^(m-1+G)` in
magnitude. -/
theorem abs_diffs_emod_lt (P : ℕ → ℤ) {l : List ℕ} (hl : l ≠ []) (G n : ℕ) :
    |diffs l (fun m => P m % 2 ^ G) n| < 2 ^ (l.length - 1 + G) := by
  rw [pow_add]
  exact abs_diffs_lt_of_mem_Ico
    (fun m => ⟨Int.emod_nonneg _ (by positivity), Int.emod_lt_of_pos _ (by positivity)⟩) hl n

open scoped UInt64.CommRing in
/-- The anchor numerators: `_invert_difference` of the seeds in uint64, masked with `2^G - 1`,
are the integer anchor `P = invert l seeds 0` modulo `2^G`. -/
theorem anchor_numerators {G : ℕ} (hG : G ≤ 64) (l : List ℕ) (seeds : ℕ → UInt64) (n : ℕ) :
    (((invert l seeds 0 n) &&& UInt64.ofNat (2 ^ G - 1)).toNat : ℤ) =
      invert l (fun m => ((seeds m).toNat : ℤ)) 0 n % 2 ^ G := by
  set Z := invert l (fun m => ((seeds m).toNat : ℤ)) 0 n
  -- the uint64 inversion is the integer one cast to uint64
  have hZ : invert l seeds 0 n = ((Z : ℤ) : UInt64) := by
    have h := map_invert (Int.castRingHom UInt64).toAddMonoidHom l
      (fun m => ((seeds m).toNat : ℤ)) 0 n
    have hs : (fun m => (Int.castRingHom UInt64).toAddMonoidHom ((seeds m).toNat : ℤ)) =
        seeds := by
      funext m
      simp only [RingHom.toAddMonoidHom_eq_coe, AddMonoidHom.coe_ofClass, eq_intCast,
        Int.cast_natCast]
      exact UInt64.ofNat_toNat
    have h0 : (fun m => (Int.castRingHom UInt64).toAddMonoidHom ((0 : ℕ → ℤ) m)) = 0 := by
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

/-- `.astype(np.int64)` keeps uint64 values below `2^63`. -/
theorem toInt_toInt64_of_lt (u : UInt64) (h : u.toNat < 2 ^ 63) :
    u.toInt64.toInt = u.toNat := by
  show u.toBitVec.toInt = _
  rw [BitVec.toInt_eq_toNat_of_lt (by simp; omega)]
  simp

/-- `a >> G` in int64, `G < 64`, is the floor division by `2^G`. -/
theorem toInt_shiftRight_ofNat (a : Int64) {G : ℕ} (hG : G < 64) :
    (a >>> Int64.ofNat G).toInt = a.toInt / 2 ^ G := by
  have h1 : (a >>> Int64.ofNat G).toInt =
      a.toBitVec.toInt >>> ((Int64.ofNat G).toBitVec.smod 64).toNat := by
    show (BitVec.sshiftRight' a.toBitVec ((Int64.ofNat G).toBitVec.smod 64)).toInt = _
    rw [BitVec.toInt_sshiftRight']
  have h2 : ((Int64.ofNat G).toBitVec.smod 64).toNat = G := by
    have hx : (Int64.ofNat G).toBitVec.toNat = G := by
      simp
      omega
    have hmx : (Int64.ofNat G).toBitVec.msb = false := by
      rw [BitVec.msb_eq_false_iff_two_mul_lt, hx]
      omega
    have hmy : (64 : BitVec 64).msb = false := by decide
    rw [BitVec.toNat_smod, hmx, hmy]
    simp
    omega
  rw [h1, h2, Int.shiftRight_eq_div_pow]
  rfl

end DiffGPT
