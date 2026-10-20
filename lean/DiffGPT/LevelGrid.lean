module

public import Mathlib

/-!
# Level-grid tokens fit the vocabulary

Both tokenizers of `diff_gpt.encoder_decoder` difference the levels by an integer operator
`c(B) = (1 - B^S₁)⋯(1 - B^Sₘ)` over the lags `l = [S₁, …, Sₘ]`, `Sᵢ ≥ 1`: `Δᵏ` is `k` lags of
`1` (`diffs_replicate`), `1 - B²⁴` the seasonal difference. With the forward differences
`Δ_[S] f n = f (n + S) - f n`, `diffs l f n` reads `f n, …, f (n + p)` for the degree
`p = l.sum`, and gives the token of row `n + p`:

    diffs l f n = ∑ j ≤ p, coeff l j • f (n + j)          (`diffs_eq_sum`)

The coefficients sum to `0` and the last one is `1` (`sum_coeff_eq_zero`, `coeff_sum_self`):
`c(1) = 0` and `c₀ = 1`. `norm1 l = ∑ |coeff l j|` is `‖c‖₁`, at most `2^m` (`norm1_le`) and
`2^k` for `Δᵏ` (`norm1_replicate`).

The tokens are `c(B) L n + V/2`, where `L n = ⌊s·(x n - A n) + 1/2⌋` rounds the scaled series
half up onto the integer grid, measured from an anchor `A` in the kernel, `c(B) A = 0`: `A = 0`
for "lgt", for "mash" the `A` through the first `p` values (`invert l pre 0`).

* `abs_diffs_levels_le`: if `|c(B) x n| ≤ D` and `s·D ≤ (V - ‖c‖₁)/2`, then
  `|c(B) L n| ≤ V/2 - 1`: every token lies in `[1, V - 1]`. `scale_mul_domain`: the scale
  `s = (V - ‖c‖₁) / (2D)` meets this with equality, and no larger one does
  (`exists_levels_ge`): errors of the signs of the coefficients push the token out.
  `DiffGPT.Optimality` goes further: for `V ≥ 2‖c‖₁` no encoder for this decoder does better.
* `lgt_token_bound`, `mash_token_bound`: the two anchors.
* `invert`, `_invert_difference`, undoes the factors by sums along every `S`-th value
  (`lagSum`): it gives the sequence with the differences `d` and the first `p` values of `pre`
  (`diffs_invert`, `invert_of_lt`), and no other sequence has them (`eq_of_diffs_eq_of_lt`),
  so the decoder gets the levels back exactly; `roundHalfUp_levels_eq_zero`: "mash" starts it
  from zeros.
* `abs_decode_sub_le`: the decoded `A n + L n / s` is within `1/(2s)` of `x n`;
  `half_step_eq`: with that scale `1/(2s) = D / (V - ‖c‖₁)`.
* The order `0`, `c = 1`, needs no room for the noise: the levels `⌊s·x⌋` on the grid shifted
  by half a step lie in `[-V/2, V/2 - 1]` for `s·D < V/2`, so the tokens take all of
  `[0, V - 1]` (`floor_mem_of_domain`), and `(⌊s·x⌋ + 1/2)/s` is within `1/(2s)` of `x`
  (`abs_floor_decode_sub_le`), `D/V` for `s = V/(2D)`: no `V` values come closer to every
  point of `[-D, D]` (`le_mul_of_forall_exists_abs_sub_le`).

The key fact is `abs_diffs_lt_of_mem`: rounding half up leaves an error in `(-1/2, 1/2]`,
half-open, and `c` has coefficients of both signs, so `c(B)` of the errors stays strictly inside
`(-‖c‖₁/2, ‖c‖₁/2)`, and the integer `c(B) L` gets one level more than a closed interval would
allow.
-/

@[expose] public section

open fwdDiff

namespace DiffGPT

/-- Round half up: `⌊u + 1/2⌋`. -/
noncomputable def roundHalfUp (u : ℝ) : ℤ := ⌊u + 1 / 2⌋

/-- The error of `roundHalfUp` lies in `(-1/2, 1/2]`. -/
lemma roundHalfUp_sub_mem (u : ℝ) :
    -(1 / 2 : ℝ) < roundHalfUp u - u ∧ (roundHalfUp u : ℝ) - u ≤ 1 / 2 := by
  unfold roundHalfUp
  constructor
  · linarith [Int.lt_floor_add_one (u + 1 / 2)]
  · linarith [Int.floor_le (u + 1 / 2)]

/-! ### The operator -/

section

variable {G H : Type*} [AddCommGroup G] [AddCommGroup H]

/-- `_difference`: `c(B) = (1 - B^S₁)⋯(1 - B^Sₘ)` over the lags `l`, one factor after another. -/
def diffs : List ℕ → (ℕ → G) → ℕ → G
  | [], f => f
  | S :: l, f => diffs l (Δ_[S] f)

@[simp] lemma diffs_nil (f : ℕ → G) : diffs [] f = f := rfl

@[simp] lemma diffs_cons (S : ℕ) (l : List ℕ) (f : ℕ → G) :
    diffs (S :: l) f = diffs l (Δ_[S] f) := rfl

/-- `Δᵏ` is the operator of `k` lags of `1`. -/
lemma diffs_replicate (k : ℕ) (f : ℕ → G) : diffs (List.replicate k 1) f = Δ_[1] ^[k] f := by
  induction k generalizing f with
  | zero => rfl
  | succ k ih => rw [List.replicate_succ, diffs_cons, ih, Function.iterate_succ_apply]

/-- An additive map, such as a cast or the reduction modulo `2^64`, commutes with `c(B)`. -/
lemma map_diffs (φ : G →+ H) (l : List ℕ) (f : ℕ → G) (n : ℕ) :
    φ (diffs l f n) = diffs l (fun m => φ (f m)) n := by
  induction l generalizing f with
  | nil => rfl
  | cons S l ih =>
    have hφ : (fun m => φ (Δ_[S] f m)) = Δ_[S] (fun m => φ (f m)) := by
      funext m
      simp [fwdDiff]
    rw [diffs_cons, diffs_cons, ih, hφ]

lemma diffs_add (l : List ℕ) (f g : ℕ → G) : diffs l (f + g) = diffs l f + diffs l g := by
  induction l generalizing f g with
  | nil => rfl
  | cons S l ih => rw [diffs_cons, diffs_cons, diffs_cons, fwdDiff_add, ih]

lemma diffs_const_smul {R : Type*} [Monoid R] [DistribMulAction R G] (l : List ℕ) (r : R)
    (f : ℕ → G) : diffs l (r • f) = r • diffs l f := by
  induction l generalizing f with
  | nil => rfl
  | cons S l ih => rw [diffs_cons, diffs_cons, fwdDiff_const_smul, ih]

lemma diffs_zero (l : List ℕ) : diffs l (0 : ℕ → G) = 0 := by
  have h := diffs_const_smul l (0 : ℤ) (0 : ℕ → G)
  rwa [zero_smul, zero_smul] at h

/-- The coefficient of `f (n + j)` in `diffs l f n`: the factor `1 - B^S` turns the
coefficients `c` into `j ↦ c (j - S) - c j`. -/
def coeff : List ℕ → ℕ → ℤ
  | [], j => if j = 0 then 1 else 0
  | S :: l, j => (if S ≤ j then coeff l (j - S) else 0) - coeff l j

/-- `‖c‖₁`, the sum of the magnitudes of the coefficients. -/
def norm1 (l : List ℕ) : ℤ := ∑ j ∈ Finset.range (l.sum + 1), |coeff l j|

lemma coeff_eq_zero_of_lt {l : List ℕ} {j : ℕ} (h : l.sum < j) : coeff l j = 0 := by
  induction l generalizing j with
  | nil => rw [coeff, ite_eq_right (by simp at h; omega)]
  | cons S l ih =>
    rw [List.sum_cons] at h
    rw [coeff, ih (j := j) (by omega), sub_zero]
    split_ifs
    · exact ih (by omega)
    · rfl

/-- `diffs l f n` reads `f n, …, f (n + p)`, `p = l.sum`, with the weights `coeff l`. -/
theorem diffs_eq_sum (l : List ℕ) (f : ℕ → G) (n : ℕ) :
    diffs l f n = ∑ j ∈ Finset.range (l.sum + 1), coeff l j • f (n + j) := by
  induction l generalizing f with
  | nil => simp [coeff]
  | cons S l ih =>
    have h1 : ∑ j ∈ Finset.range (S + l.sum + 1),
        (if S ≤ j then coeff l (j - S) else 0) • f (n + j) =
        ∑ j ∈ Finset.range (l.sum + 1), coeff l j • f (n + j + S) := by
      rw [show S + l.sum + 1 = S + (l.sum + 1) by ring, Finset.sum_range_add _ S (l.sum + 1),
        Finset.sum_eq_zero fun i hi => by
          rw [ite_eq_right (by simp at hi; omega), zero_smul], zero_add]
      refine Finset.sum_congr rfl fun j _ => ?_
      rw [ite_eq_left (by omega), Nat.add_sub_cancel_left, show n + (S + j) = n + j + S by ring]
    have h2 : ∑ j ∈ Finset.range (S + l.sum + 1), coeff l j • f (n + j) =
        ∑ j ∈ Finset.range (l.sum + 1), coeff l j • f (n + j) := by
      have h0 : ∑ i ∈ Finset.range S, coeff l (l.sum + 1 + i) • f (n + (l.sum + 1 + i)) = 0 :=
        Finset.sum_eq_zero fun i _ => by rw [coeff_eq_zero_of_lt (by omega), zero_smul]
      rw [show S + l.sum + 1 = l.sum + 1 + S by ring, Finset.sum_range_add _ (l.sum + 1) S, h0,
        add_zero]
    rw [diffs_cons, ih, List.sum_cons]
    simp only [coeff, sub_smul, Finset.sum_sub_distrib, fwdDiff, smul_sub, h1, h2]

/-- `c(B) f n` reads the values at `n, …, n + p` alone. -/
lemma diffs_congr {l : List ℕ} {f g : ℕ → G} {n : ℕ} (h : ∀ i ≤ l.sum, f (n + i) = g (n + i)) :
    diffs l f n = diffs l g n := by
  rw [diffs_eq_sum, diffs_eq_sum]
  exact Finset.sum_congr rfl fun j hj => by
    rw [h j (Nat.lt_succ_iff.mp (Finset.mem_range.mp hj))]

/-- `c(1) = 0`: the coefficients of `1 - B^S`, and so of `c`, sum to `0`. -/
lemma sum_coeff_eq_zero {l : List ℕ} (hl : l ≠ []) :
    ∑ j ∈ Finset.range (l.sum + 1), coeff l j = 0 := by
  obtain ⟨S, l, rfl⟩ := List.exists_cons_of_ne_nil hl
  have h := diffs_eq_sum (S :: l) (fun _ => (1 : ℤ)) 0
  have h1 : Δ_[S] (fun _ : ℕ => (1 : ℤ)) = 0 := by
    funext m
    simp [fwdDiff]
  rw [diffs_cons, h1, diffs_zero] at h
  simpa using h.symm

/-- `c₀ = 1`: the coefficient of the last value is `1`. -/
lemma coeff_sum_self {l : List ℕ} (hl : ∀ S ∈ l, 0 < S) : coeff l l.sum = 1 := by
  induction l with
  | nil => rfl
  | cons S l ih =>
    have hS : 0 < S := hl S (by simp)
    rw [coeff, List.sum_cons, ite_eq_left (by omega), Nat.add_sub_cancel_left,
      ih fun T hT => hl T (by simp [hT]), coeff_eq_zero_of_lt (by omega), sub_zero]

/-- Each factor at most doubles `‖c‖₁`. -/
lemma norm1_le (l : List ℕ) : norm1 l ≤ 2 ^ l.length := by
  induction l with
  | nil => simp [norm1, coeff]
  | cons S l ih =>
    have h1 : ∑ j ∈ Finset.range (S + l.sum + 1), |if S ≤ j then coeff l (j - S) else 0| =
        norm1 l := by
      rw [show S + l.sum + 1 = S + (l.sum + 1) by ring, Finset.sum_range_add _ S (l.sum + 1),
        Finset.sum_eq_zero fun i hi => by
          rw [ite_eq_right (by simp at hi; omega), abs_zero], zero_add, norm1]
      exact Finset.sum_congr rfl fun j _ => by
        rw [ite_eq_left (by omega), Nat.add_sub_cancel_left]
    have h2 : ∑ j ∈ Finset.range (S + l.sum + 1), |coeff l j| = norm1 l := by
      have h0 : ∑ i ∈ Finset.range S, |coeff l (l.sum + 1 + i)| = 0 :=
        Finset.sum_eq_zero fun i _ => by rw [coeff_eq_zero_of_lt (by omega), abs_zero]
      rw [show S + l.sum + 1 = l.sum + 1 + S by ring, Finset.sum_range_add _ (l.sum + 1) S, h0,
        add_zero, norm1]
    have h3 : norm1 (S :: l) ≤
        ∑ j ∈ Finset.range (S + l.sum + 1), |if S ≤ j then coeff l (j - S) else 0| +
          ∑ j ∈ Finset.range (S + l.sum + 1), |coeff l j| := by
      rw [norm1, List.sum_cons, ← Finset.sum_add_distrib]
      exact Finset.sum_le_sum fun j _ => abs_sub _ _
    rw [List.length_cons, pow_succ]
    linarith

/-- `Δᵏ` has the binomial coefficients with alternating signs. -/
lemma coeff_replicate (k j : ℕ) :
    coeff (List.replicate k 1) j = if j ≤ k then (-1) ^ (k - j) * (k.choose j : ℤ) else 0 := by
  induction k generalizing j with
  | zero =>
    rcases j with _ | j
    · rfl
    · simp [coeff]
  | succ k ih =>
    rw [List.replicate_succ, coeff]
    rcases j with _ | j
    · rw [ih 0]
      simp [pow_succ]
    · rw [ite_eq_left (by omega), Nat.add_sub_cancel, ih j, ih (j + 1), Nat.choose_succ_succ']
      rcases lt_trichotomy j k with h | rfl | h
      · rw [ite_eq_left h.le, ite_eq_left (by omega : j + 1 ≤ k),
          ite_eq_left (by omega : j + 1 ≤ k + 1), show k + 1 - (j + 1) = k - j by omega,
          show k - j = k - (j + 1) + 1 by omega]
        push_cast
        ring
      · simp
      · rw [ite_eq_right (by omega : ¬j ≤ k), ite_eq_right (by omega : ¬j + 1 ≤ k),
          ite_eq_right (by omega : ¬j + 1 ≤ k + 1)]
        simp

/-- `‖Δᵏ‖₁ = 2^k`. -/
lemma norm1_replicate (k : ℕ) : norm1 (List.replicate k 1) = 2 ^ k := by
  rw [norm1, List.sum_replicate, smul_eq_mul, mul_one]
  have h : ∀ j ∈ Finset.range (k + 1), |coeff (List.replicate k 1) j| = (k.choose j : ℤ) :=
    fun j hj => by
      rw [coeff_replicate, ite_eq_left (Nat.lt_succ_iff.mp (Finset.mem_range.mp hj)), abs_mul,
        abs_pow, abs_neg, abs_one, one_pow, one_mul, Nat.abs_cast]
  rw [Finset.sum_congr rfl h]
  exact_mod_cast Nat.sum_range_choose k

/-- Weights of both signs that sum to `0`: against errors in `(-1/2, 1/2]` their sum stays
strictly inside `(-∑|a|/2, ∑|a|/2)`, since the errors reach `1/2` but not `-1/2`. -/
lemma abs_sum_mul_lt {s : Finset ℕ} {a : ℕ → ℤ} {ζ : ℕ → ℝ}
    (hζ : ∀ j ∈ s, -(1 / 2 : ℝ) < ζ j ∧ ζ j ≤ 1 / 2) (ha : ∑ j ∈ s, a j = 0) {j₀ : ℕ}
    (hj₀ : j₀ ∈ s) (ha₀ : a j₀ ≠ 0) :
    |∑ j ∈ s, (a j : ℝ) * ζ j| < ((∑ j ∈ s, |a j| : ℤ) : ℝ) / 2 := by
  obtain ⟨i, hi, hai⟩ : ∃ i ∈ s, a i < 0 := by
    by_contra! h
    exact ha₀ ((Finset.sum_eq_zero_iff_of_nonneg h).mp ha j₀ hj₀)
  obtain ⟨i', hi', hai'⟩ : ∃ i ∈ s, 0 < a i := by
    by_contra! h
    have h' := (Finset.sum_eq_zero_iff_of_nonneg fun i hi => neg_nonneg.mpr (h i hi)).mp
      (by rw [Finset.sum_neg_distrib, ha, neg_zero]) j₀ hj₀
    exact ha₀ (neg_eq_zero.mp h')
  have hcast : ((∑ j ∈ s, |a j| : ℤ) : ℝ) / 2 = ∑ j ∈ s, |(a j : ℝ)| / 2 := by
    push_cast
    rw [Finset.sum_div]
  have hle (j : ℕ) (hj : j ∈ s) :
      -(|(a j : ℝ)| / 2) ≤ a j * ζ j ∧ (a j : ℝ) * ζ j ≤ |(a j : ℝ)| / 2 := by
    obtain ⟨h1, h2⟩ := hζ j hj
    rcases le_or_gt 0 (a j : ℝ) with h | h
    · rw [abs_of_nonneg h]
      constructor <;> nlinarith
    · rw [abs_of_neg h]
      constructor <;> nlinarith
  rw [hcast, abs_lt]
  constructor
  · rw [← Finset.sum_neg_distrib]
    refine Finset.sum_lt_sum (fun j hj => (hle j hj).1) ⟨i', hi', ?_⟩
    have h : (0 : ℝ) < a i' := by exact_mod_cast hai'
    rw [abs_of_pos h]
    nlinarith [(hζ i' hi').1]
  · refine Finset.sum_lt_sum (fun j hj => (hle j hj).2) ⟨i, hi, ?_⟩
    have h : (a i : ℝ) < 0 := by exact_mod_cast hai
    rw [abs_of_neg h]
    nlinarith [(hζ i hi).1]

/-- `c(B)` of errors in `(-1/2, 1/2]` stays strictly inside `(-‖c‖₁/2, ‖c‖₁/2)`. -/
theorem abs_diffs_lt_of_mem {l : List ℕ} (hl : ∀ S ∈ l, 0 < S) (hl0 : l ≠ []) {ζ : ℕ → ℝ}
    (hζ : ∀ n, -(1 / 2 : ℝ) < ζ n ∧ ζ n ≤ 1 / 2) (n : ℕ) :
    |diffs l ζ n| < (norm1 l : ℝ) / 2 := by
  rw [diffs_eq_sum, norm1]
  simp only [zsmul_eq_mul]
  exact abs_sum_mul_lt (fun j _ => hζ (n + j)) (sum_coeff_eq_zero hl0)
    (Finset.self_mem_range_succ l.sum) (by rw [coeff_sum_self hl]; norm_num)

/-- `c(B)` commutes with the cast from `ℤ` to `ℝ`. -/
lemma cast_diffs (l : List ℕ) (f : ℕ → ℤ) (n : ℕ) :
    ((diffs l f n : ℤ) : ℝ) = diffs l (fun m => (f m : ℝ)) n := by
  simpa using map_diffs (Int.castAddHom ℝ) l f n

end

/-! ### The tokens -/

/-- **Theorem 1.** Every token `c(B) L n + V/2` lies in `[1, V - 1]`. -/
theorem abs_diffs_levels_le {V : ℤ} (hV : Even V) {l : List ℕ} (hl : ∀ S ∈ l, 0 < S)
    (hl0 : l ≠ []) {s D : ℝ} (hs : 0 ≤ s) {x A : ℕ → ℝ} (hA : diffs l A = 0) {n : ℕ}
    (hD : |diffs l x n| ≤ D) (hsD : s * D ≤ ((V : ℝ) - norm1 l) / 2) :
    |diffs l (fun m => roundHalfUp (s * (x m - A m))) n| ≤ V / 2 - 1 := by
  set L : ℕ → ℤ := fun m => roundHalfUp (s * (x m - A m)) with hL
  set ζ : ℕ → ℝ := fun m => (L m : ℝ) - s * (x m - A m) with hζdef
  have hζ : ∀ m, -(1 / 2 : ℝ) < ζ m ∧ ζ m ≤ 1 / 2 := fun m => roundHalfUp_sub_mem _
  have hLR : (fun m => (L m : ℝ)) = s • x + (-s) • A + ζ := by
    funext m
    simp only [hζdef, Pi.add_apply, Pi.smul_apply, smul_eq_mul]
    ring
  have hdiff : ((diffs l L n : ℤ) : ℝ) = s * diffs l x n + diffs l ζ n := by
    rw [cast_diffs, hLR, diffs_add, diffs_add, diffs_const_smul, diffs_const_smul, hA]
    simp
  have hx : |s * diffs l x n| ≤ s * D := by
    rw [abs_mul, abs_of_nonneg hs]
    exact mul_le_mul_of_nonneg_left hD hs
  have hlt : |((diffs l L n : ℤ) : ℝ)| < (V : ℝ) / 2 := by
    rw [hdiff]
    calc |s * diffs l x n + diffs l ζ n| ≤ |s * diffs l x n| + |diffs l ζ n| := abs_add_le _ _
      _ < s * D + (norm1 l : ℝ) / 2 := add_lt_add_of_le_of_lt hx (abs_diffs_lt_of_mem hl hl0 hζ n)
      _ ≤ (V : ℝ) / 2 := by linarith
  obtain ⟨m, hm⟩ := hV
  have hV2 : V / 2 = m := by omega
  have hVm : (V : ℝ) / 2 = m := by
    rw [hm]
    push_cast
    ring
  have hcast : ((|diffs l L n| : ℤ) : ℝ) < (m : ℝ) := by
    rw [Int.cast_abs]
    linarith
  have := Int.cast_lt.mp hcast
  rw [hV2]
  omega

/-- The scale `s = (V - h) / (2D)` gives `s·D = (V - h)/2`. -/
lemma scale_mul_domain (V h : ℝ) {D : ℝ} (hD : D ≠ 0) : (V - h) / (2 * D) * D = (V - h) / 2 := by
  field_simp

/-- `‖c‖₁ ≥ 2`: `c` has a coefficient `1` and sums to `0`. -/
lemma two_le_norm1 {l : List ℕ} (hl : ∀ S ∈ l, 0 < S) (hl0 : l ≠ []) : 2 ≤ norm1 l := by
  have hsum := sum_coeff_eq_zero hl0
  obtain ⟨i, hi, hai⟩ : ∃ i ∈ Finset.range (l.sum + 1), coeff l i < 0 := by
    by_contra! h
    have := (Finset.sum_eq_zero_iff_of_nonneg h).mp hsum l.sum (Finset.self_mem_range_succ _)
    rw [coeff_sum_self hl] at this
    exact one_ne_zero this
  have h : norm1 l = ∑ j ∈ Finset.range (l.sum + 1), (|coeff l j| - coeff l j) := by
    rw [Finset.sum_sub_distrib, hsum, sub_zero, norm1]
  have h1 := Finset.single_le_sum (f := fun j => |coeff l j| - coeff l j)
    (fun j _ => sub_nonneg.mpr (le_abs_self (coeff l j))) hi
  rw [abs_of_neg hai] at h1
  linarith

/-- `a · sign a = |a|`, and `|sign a| ≤ 1`, in the reals. -/
lemma cast_sign (a : ℤ) : |((a.sign : ℤ) : ℝ)| ≤ 1 ∧ (a : ℝ) * (a.sign : ℤ) = |(a : ℝ)| := by
  rcases lt_trichotomy a 0 with h | rfl | h
  · have h' : (a : ℝ) < 0 := by exact_mod_cast h
    rw [Int.sign_eq_neg_one_of_neg h, abs_of_neg h']
    norm_num
  · simp
  · have h' : (0 : ℝ) < a := by exact_mod_cast h
    rw [Int.sign_eq_one_of_pos h, abs_of_pos h']
    norm_num

/-- No scale above `(V - ‖c‖₁) / (2D)` keeps every token in range: errors of the signs of the
coefficients, of magnitude `t / ‖c‖₁ < 1/2`, add `t` to `c(B)` of the levels, so a step of `D`
gets the token `V`, or one above it. -/
theorem exists_levels_ge {V : ℤ} (hV : Even V) {l : List ℕ} (hl : ∀ S ∈ l, 0 < S)
    (hl0 : l ≠ []) {s D : ℝ} (hs : 0 < s) (hsD : ((V : ℝ) - norm1 l) / 2 < s * D) :
    ∃ x : ℕ → ℝ, diffs l x 0 = D ∧ V / 2 ≤ diffs l (fun m => roundHalfUp (s * x m)) 0 := by
  obtain ⟨v, rfl⟩ := hV
  rw [show (v + v) / 2 = v by omega]
  push_cast at hsD
  have hc2 : (2 : ℝ) ≤ norm1 l := by exact_mod_cast two_le_norm1 hl hl0
  set c : ℝ := (norm1 l : ℝ) with hc
  -- the level `K ≥ V/2` that `c(B) L` takes, and the excess `t` of `c(B)` of the errors
  set K : ℤ := max v ⌈s * D⌉ with hK
  set t : ℝ := K - s * D with ht
  have ht0 : 0 ≤ t := by
    have : s * D ≤ K := (Int.le_ceil _).trans (by exact_mod_cast le_max_right _ _)
    linarith
  have htc : t < c / 2 := by
    rcases le_total ⌈s * D⌉ v with h | h
    · have : (K : ℝ) = v := by rw [hK, max_eq_left h]
      linarith
    · have : (K : ℝ) = ⌈s * D⌉ := by rw [hK, max_eq_right h]
      linarith [Int.ceil_lt_add_one (s * D)]
  set ζ : ℕ → ℝ := fun j => t / c * ((coeff l j).sign : ℤ) with hζ
  have hζ2 : ∀ j, |ζ j| < 1 / 2 := fun j => by
    have h2 : |t / c| < 1 / 2 := by
      rw [abs_of_nonneg (by positivity), div_lt_iff₀ (by linarith)]
      linarith
    rw [hζ, abs_mul]
    calc |t / c| * |(((coeff l j).sign : ℤ) : ℝ)| ≤ |t / c| * 1 :=
          mul_le_mul_of_nonneg_left (cast_sign _).1 (abs_nonneg _)
      _ < 1 / 2 := by linarith
  -- the levels: `K` at the last value, `0` before it
  set n : ℕ → ℤ := fun j => if j = l.sum then K else 0 with hn
  have hdn : diffs l n 0 = K := by
    rw [diffs_eq_sum, Finset.sum_eq_single l.sum (fun j _ hj => by simp [hn, hj])
      (fun h => absurd (Finset.self_mem_range_succ _) h)]
    simp [hn, coeff_sum_self hl]
  have hdζ : diffs l ζ 0 = t := by
    rw [diffs_eq_sum]
    simp only [hζ, zsmul_eq_mul, zero_add]
    have h : ∀ j ∈ Finset.range (l.sum + 1),
        (coeff l j : ℝ) * (t / c * ((coeff l j).sign : ℤ)) = t / c * |((coeff l j : ℤ) : ℝ)| :=
      fun j _ => by rw [← (cast_sign (coeff l j)).2]; ring
    have hc' : ∑ j ∈ Finset.range (l.sum + 1), |((coeff l j : ℤ) : ℝ)| = c := by
      simp [hc, norm1]
    rw [Finset.sum_congr rfl h, ← Finset.mul_sum, hc', div_mul_cancel₀ _ (by linarith : 0 < c).ne']
  refine ⟨fun j => (n j - ζ j) / s, ?_, ?_⟩
  · have hx : (fun j => ((n j : ℤ) - ζ j) / s) =
        s⁻¹ • ((fun j => ((n j : ℤ) : ℝ)) + (-1 : ℝ) • ζ) := by
      funext j
      simp only [Pi.smul_apply, Pi.add_apply, smul_eq_mul]
      ring
    rw [hx, diffs_const_smul, diffs_add, diffs_const_smul, Pi.smul_apply, Pi.add_apply,
      Pi.smul_apply, hdζ, ← cast_diffs, hdn, ht, smul_eq_mul, smul_eq_mul]
    field_simp
    ring
  · have hL : (fun m => roundHalfUp (s * (((n m : ℤ) : ℝ) - ζ m) / s)) = n := by
      funext m
      rw [mul_div_cancel_left₀ _ hs.ne', roundHalfUp, Int.floor_eq_iff]
      obtain ⟨h1, h2⟩ := abs_lt.mp (hζ2 m)
      constructor <;> linarith
    simp only [mul_div_assoc'] at hL ⊢
    rw [hL, hdn]
    exact le_max_left _ _

/-- "lgt": levels on the absolute grid, `A = 0`. -/
theorem lgt_token_bound {V : ℤ} (hV : Even V) {l : List ℕ} (hl : ∀ S ∈ l, 0 < S)
    (hl0 : l ≠ []) {s D : ℝ} (hs : 0 ≤ s) {x : ℕ → ℝ} (hD : ∀ n, |diffs l x n| ≤ D)
    (hsD : s * D ≤ ((V : ℝ) - norm1 l) / 2) (n : ℕ) :
    |diffs l (fun m => roundHalfUp (s * x m)) n| ≤ V / 2 - 1 := by
  simpa using abs_diffs_levels_le hV hl hl0 hs (diffs_zero l) (hD n) hsD

/-! ### Inverting `c(B)` -/

/-- `_cumsum`: the sums along every `S`-th value, `lagSum S v n = lagSum S v (n - S) + v n`
from `n ≥ S` on. -/
def lagSum {G : Type*} [Add G] (S : ℕ) (v : ℕ → G) (n : ℕ) : G :=
  if _h : 0 < S ∧ S ≤ n then lagSum S v (n - S) + v n else v n
termination_by n
decreasing_by omega

lemma lagSum_of_lt {G : Type*} [Add G] {S n : ℕ} (h : n < S) (v : ℕ → G) :
    lagSum S v n = v n := by
  rw [lagSum, dite_eq_right (by omega)]

lemma lagSum_of_le {G : Type*} [Add G] {S n : ℕ} (hS : 0 < S) (h : S ≤ n) (v : ℕ → G) :
    lagSum S v n = lagSum S v (n - S) + v n := by
  rw [lagSum, dite_eq_left ⟨hS, h⟩]

section

variable {G H : Type*} [AddCommGroup G] [AddCommGroup H]

lemma fwdDiff_lagSum {S : ℕ} (hS : 0 < S) (v : ℕ → G) :
    Δ_[S] (lagSum S v) = fun n => v (n + S) := by
  funext n
  simp only [fwdDiff]
  rw [lagSum_of_le hS (n := n + S) (by omega), Nat.add_sub_cancel]
  abel

lemma map_lagSum (φ : G →+ H) (S : ℕ) (v : ℕ → G) (n : ℕ) :
    φ (lagSum S v n) = lagSum S (fun m => φ (v m)) n := by
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    by_cases h : 0 < S ∧ S ≤ n
    · rw [lagSum_of_le h.1 h.2, lagSum_of_le h.1 h.2, map_add, ih _ (by omega)]
    · rw [lagSum, dite_eq_right h, lagSum, dite_eq_right h]

/-- `_invert_difference`: the sequence with the differences `d` that starts with `pre 0, …,
pre (p - 1)`. The outermost factor `1 - B^S` is undone by `lagSum S` over the first `S` values
of `pre` followed by the inner sequence: `invert` of the other factors, from the prefix
differenced by `S`. -/
def invert : List ℕ → (ℕ → G) → (ℕ → G) → ℕ → G
  | [], _, d => d
  | S :: l, pre, d => lagSum S fun n => if n < S then pre n else invert l (Δ_[S] pre) d (n - S)

/-- One factor of `invert` undone: `Δ_[S]` of it is `invert` of the other factors, from the
prefix differenced by `S`. -/
lemma fwdDiff_invert_cons {S : ℕ} (hS : 0 < S) (l : List ℕ) (pre d : ℕ → G) :
    Δ_[S] (invert (S :: l) pre d) = invert l (Δ_[S] pre) d := by
  rw [invert, fwdDiff_lagSum hS]
  funext n
  rw [ite_eq_right (by omega), Nat.add_sub_cancel]

/-- `invert` has the differences `d`. -/
theorem diffs_invert {l : List ℕ} (hl : ∀ S ∈ l, 0 < S) (pre d : ℕ → G) :
    diffs l (invert l pre d) = d := by
  induction l generalizing pre with
  | nil => rfl
  | cons S l ih =>
    rw [diffs_cons, fwdDiff_invert_cons (hl S (by simp))]
    exact ih (fun T hT => hl T (by simp [hT])) _

/-- `invert` starts with the first `p` values of `pre`. -/
theorem invert_of_lt {l : List ℕ} (hl : ∀ S ∈ l, 0 < S) (pre d : ℕ → G) {n : ℕ}
    (hn : n < l.sum) : invert l pre d n = pre n := by
  induction l generalizing pre n with
  | nil => simp at hn
  | cons S l ih =>
    have hS : 0 < S := hl S (by simp)
    have ih' : ∀ m < l.sum, invert l (Δ_[S] pre) d m = Δ_[S] pre m := fun m hm =>
      ih (fun T hT => hl T (by simp [hT])) (Δ_[S] pre) hm
    rw [List.sum_cons] at hn
    induction n using Nat.strong_induction_on with
    | _ n ihn =>
      by_cases hnS : n < S
      · rw [invert, lagSum_of_lt hnS, ite_eq_left hnS]
      · have h1 := ihn (n - S) (by omega) (by omega)
        rw [invert] at h1 ⊢
        rw [lagSum_of_le hS (n := n) (by omega), ite_eq_right hnS, h1, ih' (n - S) (by omega)]
        simp only [fwdDiff, Nat.sub_add_cancel (by omega : S ≤ n)]
        abel

/-- `invert` reads the prefix only below `p`. -/
lemma invert_congr {l : List ℕ} {pre pre' : ℕ → G} (h : ∀ i < l.sum, pre i = pre' i)
    (d : ℕ → G) : invert l pre d = invert l pre' d := by
  induction l generalizing pre pre' with
  | nil => rfl
  | cons S l ih =>
    rw [List.sum_cons] at h
    have hd : invert l (Δ_[S] pre) d = invert l (Δ_[S] pre') d :=
      ih fun i hi => by simp only [fwdDiff]; rw [h i (by omega), h (i + S) (by omega)]
    simp only [invert, hd]
    congr 1
    funext n
    split_ifs with hn
    · exact h n (by omega)
    · rfl

/-- An additive map, such as the reduction of integers modulo `2^64`, commutes with
`invert`. -/
theorem map_invert (φ : G →+ H) (l : List ℕ) (pre d : ℕ → G) (n : ℕ) :
    φ (invert l pre d n) = invert l (fun m => φ (pre m)) (fun m => φ (d m)) n := by
  induction l generalizing pre n with
  | nil => rfl
  | cons S l ih =>
    have hφ : (fun m => φ (Δ_[S] pre m)) = Δ_[S] (fun m => φ (pre m)) := by
      funext m
      simp [fwdDiff]
    simp only [invert]
    rw [map_lagSum]
    congr 1
    funext m
    split_ifs
    · rfl
    · rw [ih, hφ]

lemma invert_zero (l : List ℕ) : invert l (0 : ℕ → G) 0 = 0 := by
  funext n
  have h := map_invert (0 : G →+ G) l 0 0 n
  simp only [AddMonoidHom.zero_apply] at h
  exact h.symm

/-- `c(B)` on the first `M` values and the first `p` values determine a sequence there: since
`c₀ = 1`, each value is a difference less the values before it. -/
theorem eq_of_diffs_eq_of_lt {l : List ℕ} (hl : ∀ S ∈ l, 0 < S) {M : ℕ} {L L' : ℕ → G}
    (h : ∀ n, n + l.sum < M → diffs l L n = diffs l L' n) (h0 : ∀ m < l.sum, L m = L' m)
    {m : ℕ} (hm : m < M) : L m = L' m := by
  induction l generalizing L L' M m with
  | nil => simpa using h m (by simpa using hm)
  | cons S l ih =>
    have hS : 0 < S := hl S (by simp)
    simp only [List.sum_cons] at h h0
    have hd : ∀ n < M - S, Δ_[S] L n = Δ_[S] L' n := fun n hn =>
      ih (fun T hT => hl T (by simp [hT])) (M := M - S) (fun n' hn' => h n' (by omega))
        (fun m' hm' => by simp only [fwdDiff]; rw [h0 m' (by omega), h0 (m' + S) (by omega)]) hn
    induction m using Nat.strong_induction_on with
    | _ m ihm =>
      by_cases hmS : m < S
      · exact h0 m (by omega)
      · have h1 := hd (m - S) (by omega)
        have h2 := ihm (m - S) (by omega) (by omega)
        simp only [fwdDiff, Nat.sub_add_cancel (by omega : S ≤ m)] at h1
        rw [h2] at h1
        exact sub_left_injective h1

end

/-- "mash": levels measured from the anchor `A = invert l pre 0` in the kernel, through the
first `p` values `pre`. -/
theorem mash_token_bound {V : ℤ} (hV : Even V) {l : List ℕ} (hl : ∀ S ∈ l, 0 < S)
    (hl0 : l ≠ []) {s D : ℝ} (hs : 0 ≤ s) {x : ℕ → ℝ} (pre : ℕ → ℝ)
    (hD : ∀ n, |diffs l x n| ≤ D) (hsD : s * D ≤ ((V : ℝ) - norm1 l) / 2) (n : ℕ) :
    |diffs l (fun m => roundHalfUp (s * (x m - invert l pre 0 m))) n| ≤ V / 2 - 1 :=
  abs_diffs_levels_le hV hl hl0 hs (diffs_invert hl pre 0) (hD n) hsD

/-! ### Decoding -/

/-- Where the anchor passes through the series, the level is `0`. -/
lemma roundHalfUp_levels_eq_zero {s : ℝ} {x A : ℕ → ℝ} {m : ℕ} (h : A m = x m) :
    roundHalfUp (s * (x m - A m)) = 0 := by
  rw [h, sub_self, mul_zero, roundHalfUp, zero_add, Int.floor_eq_zero_iff]
  norm_num

/-- Decoding `A n + L n / s` lands within `1/(2s)` of `x n`. -/
theorem abs_decode_sub_le {s : ℝ} (hs : 0 < s) (x A : ℕ → ℝ) (n : ℕ) :
    |A n + roundHalfUp (s * (x n - A n)) / s - x n| ≤ 1 / (2 * s) := by
  obtain ⟨h1, h2⟩ := roundHalfUp_sub_mem (s * (x n - A n))
  have he : A n + roundHalfUp (s * (x n - A n)) / s - x n
      = ((roundHalfUp (s * (x n - A n)) : ℝ) - s * (x n - A n)) / s := by
    field_simp
    ring
  rw [he, abs_div, abs_of_pos hs]
  calc |(roundHalfUp (s * (x n - A n)) : ℝ) - s * (x n - A n)| / s ≤ (1 / 2) / s := by
        gcongr
        exact abs_le.mpr ⟨by linarith, h2⟩
    _ = 1 / (2 * s) := by ring

/-- With the scale of `scale_mul_domain`, `1/(2s) = D / (V - h)`. -/
lemma half_step_eq {V h D : ℝ} (hD : D ≠ 0) (hV : V - h ≠ 0) :
    1 / (2 * ((V - h) / (2 * D))) = D / (V - h) := by
  field_simp

/-! ### The order `0` -/

/-- The order `0` takes the levels `⌊s·x⌋`, which `_levels` returns for `F = 1/2`. -/
lemma floor_eq_roundHalfUp (u : ℝ) : ⌊u⌋ = roundHalfUp (u - 1 / 2) := by
  rw [roundHalfUp, sub_add_cancel]

/-- The order-0 token `⌊s·x⌋ + V/2` lies in `[0, V - 1]` when `|x| ≤ D` and `s·D < V/2`. -/
theorem floor_mem_of_domain {V : ℤ} (hV : Even V) {s D x : ℝ} (hs : 0 ≤ s) (hD : |x| ≤ D)
    (hsD : s * D < (V : ℝ) / 2) : 0 ≤ ⌊s * x⌋ + V / 2 ∧ ⌊s * x⌋ + V / 2 ≤ V - 1 := by
  obtain ⟨m, rfl⟩ := hV
  have hm : (m + m) / 2 = m := by omega
  have hVm : ((m + m : ℤ) : ℝ) / 2 = m := by
    push_cast
    ring
  rw [hm]
  rw [hVm] at hsD
  have hx : |s * x| ≤ s * D := by
    rw [abs_mul, abs_of_nonneg hs]
    exact mul_le_mul_of_nonneg_left hD hs
  obtain ⟨h1, h2⟩ := abs_le.mp hx
  constructor
  · have : -m ≤ ⌊s * x⌋ := Int.le_floor.mpr (by push_cast; linarith)
    omega
  · have : ⌊s * x⌋ < m := Int.floor_lt.mpr (by linarith)
    omega

/-- The order-0 decode `(⌊s·x⌋ + 1/2)/s`, the middle of the cell `[⌊s·x⌋, ⌊s·x⌋ + 1)/s`, lies
within `1/(2s)` of `x`. -/
theorem abs_floor_decode_sub_le {s : ℝ} (hs : 0 < s) (x : ℝ) :
    |(⌊s * x⌋ + 1 / 2) / s - x| ≤ 1 / (2 * s) := by
  have h1 := Int.floor_le (s * x)
  have h2 := Int.lt_floor_add_one (s * x)
  rw [show (⌊s * x⌋ + 1 / 2) / s - x = (⌊s * x⌋ + 1 / 2 - s * x) / s by field_simp,
    abs_div, abs_of_pos hs, show 1 / (2 * s) = 1 / 2 / s by ring]
  exact div_le_div_of_nonneg_right (abs_le.mpr ⟨by linarith, by linarith⟩) hs.le

/-- No `V` values come within less than `D/V` of every point of `[-D, D]`: the intervals of
radius `r` around them cover it only if `2rV ≥ 2D`. -/
theorem le_mul_of_forall_exists_abs_sub_le {V : ℕ} {D r : ℝ} (hD : 0 ≤ D) (y : Fin V → ℝ)
    (h : ∀ x ∈ Set.Icc (-D) D, ∃ i, |x - y i| ≤ r) : D ≤ V * r := by
  obtain ⟨i, hi⟩ := h 0 ⟨by linarith, hD⟩
  have hr : 0 ≤ r := (abs_nonneg _).trans hi
  have hcover : Set.Icc (-D) D ⊆ ⋃ i, Set.Icc (y i - r) (y i + r) := fun x hx => by
    obtain ⟨i, hi⟩ := h x hx
    obtain ⟨h1, h2⟩ := abs_le.mp hi
    exact Set.mem_iUnion.mpr ⟨i, by constructor <;> linarith⟩
  have h1 := (MeasureTheory.measure_mono (μ := MeasureTheory.volume) hcover).trans
    (MeasureTheory.measure_iUnion_fintype_le MeasureTheory.volume _)
  have h2 (i : Fin V) : ENNReal.ofReal (y i + r - (y i - r)) = ENNReal.ofReal (2 * r) := by
    ring_nf
  simp only [Real.volume_Icc, h2, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
    nsmul_eq_mul] at h1
  rw [← ENNReal.ofReal_natCast, ← ENNReal.ofReal_mul (by positivity),
    ENNReal.ofReal_le_ofReal_iff (by positivity)] at h1
  linarith

end DiffGPT
