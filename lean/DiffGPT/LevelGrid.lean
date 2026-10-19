module

public import Mathlib

/-!
# Level-grid tokens fit the vocabulary

Both tokenizers of `diff_gpt.encoder_decoder` emit the token `Δᵏ L n + V/2`, where
`L n = ⌊s·(x n - A n) + 1/2⌋` rounds the scaled series half up onto the integer
grid, measured from an anchor `A` that the `k`-th difference removes
(`Δᵏ A = 0`): `A = 0` for "lgt", the polynomial through the first `k` values for
"mash".

* `abs_fwdDiff_iter_levels_le`: if `|Δᵏ x| ≤ D` and `s·D ≤ V/2 - 2^(k-1)`, with
  `k - 1` truncated (so `V/2 - 1` at `k = 0`), then `|Δᵏ L n| ≤ V/2 - 1`: every
  token lies in `[1, V - 1]`. `scale_mul_domain`: the scale
  `s = (V - 2·2^(k-1)) / (2D) = (V - max(2, 2^k)) / (2D)` meets this with equality.
* `lgt_token_bound`, `mash_token_bound`: the two anchors.
* `eq_of_fwdDiff_iter_eq`: the tokens and the first `k` levels determine `L`, so the
  decoder's integer `k`-fold cumulative sum recovers the levels exactly;
  `roundHalfUp_levels_eq_zero`: "mash" starts it from zeros.
* `abs_decode_sub_le`: the decoded `A n + L n / s` is within `1/(2s)` of `x n`;
  `half_step_eq`: with that scale `1/(2s) = D / (V - max(2, 2^k))`.

The key fact is `abs_fwdDiff_iter_lt_of_mem`: rounding half up leaves an error in
`(-1/2, 1/2]`, half-open, so its `k`-th difference stays strictly inside
`(-2^(k-1), 2^(k-1))` for `k ≥ 1`, and the integer `Δᵏ L` gets one level more
than a closed interval would allow.
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

/-- `k` differences of a sequence bounded by `a` stay below `2^k·a`. -/
lemma abs_fwdDiff_iter_lt {a : ℝ} (k : ℕ) {g : ℕ → ℝ} (hg : ∀ n, |g n| < a) (n : ℕ) :
    |Δ_[1] ^[k] g n| < 2 ^ k * a := by
  induction k generalizing g a with
  | zero => simpa using hg n
  | succ k ih =>
    rw [Function.iterate_succ_apply]
    have h2 : ∀ m, |Δ_[1] g m| < 2 * a := fun m => by
      obtain ⟨b1, b2⟩ := abs_lt.mp (hg (m + 1))
      obtain ⟨c1, c2⟩ := abs_lt.mp (hg m)
      exact abs_lt.mpr ⟨by simp only [fwdDiff]; linarith, by simp only [fwdDiff]; linarith⟩
    calc |Δ_[1] ^[k] (Δ_[1] g) n| < 2 ^ k * (2 * a) := ih h2
      _ = 2 ^ (k + 1) * a := by ring

/-- For `k ≥ 1` the `k`-th difference of errors in `(-1/2, 1/2]` stays strictly
inside `(-2^(k-1), 2^(k-1))`. -/
lemma abs_fwdDiff_iter_lt_of_mem {ζ : ℕ → ℝ} (hζ : ∀ n, -(1 / 2 : ℝ) < ζ n ∧ ζ n ≤ 1 / 2)
    {k : ℕ} (hk : 1 ≤ k) (n : ℕ) : |Δ_[1] ^[k] ζ n| < 2 ^ (k - 1) := by
  obtain ⟨j, rfl⟩ : ∃ j, k = j + 1 := ⟨k - 1, by omega⟩
  rw [Function.iterate_succ_apply, Nat.add_sub_cancel]
  have h1 : ∀ m, |Δ_[1] ζ m| < 1 := fun m => by
    obtain ⟨a1, a2⟩ := hζ m
    obtain ⟨b1, b2⟩ := hζ (m + 1)
    exact abs_lt.mpr ⟨by simp only [fwdDiff]; linarith, by simp only [fwdDiff]; linarith⟩
  simpa using abs_fwdDiff_iter_lt j h1 n

/-- Differences commute with the cast from `ℤ` to `ℝ`. -/
lemma cast_fwdDiff_iter (f : ℕ → ℤ) (k n : ℕ) :
    ((Δ_[1] ^[k] f n : ℤ) : ℝ) = Δ_[1] ^[k] (fun m => (f m : ℝ)) n := by
  induction k generalizing f with
  | zero => rfl
  | succ k ih =>
    have hf : (fun m => ((Δ_[1] f m : ℤ) : ℝ)) = Δ_[1] (fun m => (f m : ℝ)) := by
      funext m
      simp [fwdDiff]
    rw [Function.iterate_succ_apply, Function.iterate_succ_apply, ih, hf]

/-- Every token `Δᵏ L n + V/2` lies in `[1, V - 1]`. -/
theorem abs_fwdDiff_iter_levels_le {V : ℤ} (hV : Even V) {k : ℕ} {s D : ℝ} (hs : 0 ≤ s)
    {x A : ℕ → ℝ} (hA : Δ_[1] ^[k] A = 0) (hD : ∀ n, |Δ_[1] ^[k] x n| ≤ D)
    (hsD : s * D ≤ (V : ℝ) / 2 - 2 ^ (k - 1)) (n : ℕ) :
    |Δ_[1] ^[k] (fun m => roundHalfUp (s * (x m - A m))) n| ≤ V / 2 - 1 := by
  set L : ℕ → ℤ := fun m => roundHalfUp (s * (x m - A m)) with hL
  set ζ : ℕ → ℝ := fun m => (L m : ℝ) - s * (x m - A m) with hζdef
  have hζ : ∀ m, -(1 / 2 : ℝ) < ζ m ∧ ζ m ≤ 1 / 2 := fun m => roundHalfUp_sub_mem _
  have hLR : (fun m => (L m : ℝ)) = s • x + (-s) • A + ζ := by
    funext m
    simp only [hζdef, Pi.add_apply, Pi.smul_apply, smul_eq_mul]
    ring
  have hdiff : ((Δ_[1] ^[k] L n : ℤ) : ℝ) = s * Δ_[1] ^[k] x n + Δ_[1] ^[k] ζ n := by
    rw [cast_fwdDiff_iter, hLR, fwdDiff_iter_add, fwdDiff_iter_add, fwdDiff_iter_const_smul,
      fwdDiff_iter_const_smul, hA]
    simp
  have hx : |s * Δ_[1] ^[k] x n| ≤ s * D := by
    rw [abs_mul, abs_of_nonneg hs]
    exact mul_le_mul_of_nonneg_left (hD n) hs
  have hlt : |((Δ_[1] ^[k] L n : ℤ) : ℝ)| < (V : ℝ) / 2 := by
    rw [hdiff]
    rcases Nat.eq_zero_or_pos k with rfl | hk
    · have hz : |ζ n| ≤ 1 / 2 := abs_le.mpr ⟨by linarith [(hζ n).1], (hζ n).2⟩
      norm_num at hsD
      simp only [Function.iterate_zero, id_eq] at hx ⊢
      calc |s * x n + ζ n| ≤ |s * x n| + |ζ n| := abs_add_le _ _
        _ ≤ s * D + 1 / 2 := add_le_add hx hz
        _ < (V : ℝ) / 2 := by linarith
    · calc |s * Δ_[1] ^[k] x n + Δ_[1] ^[k] ζ n|
          ≤ |s * Δ_[1] ^[k] x n| + |Δ_[1] ^[k] ζ n| := abs_add_le _ _
        _ < s * D + 2 ^ (k - 1) := add_lt_add_of_le_of_lt hx (abs_fwdDiff_iter_lt_of_mem hζ hk n)
        _ ≤ (V : ℝ) / 2 := by linarith
  obtain ⟨m, hm⟩ := hV
  have hV2 : V / 2 = m := by omega
  have hVm : (V : ℝ) / 2 = m := by
    rw [hm]
    push_cast
    ring
  have hcast : ((|Δ_[1] ^[k] L n| : ℤ) : ℝ) < (m : ℝ) := by
    rw [Int.cast_abs]
    linarith
  have := Int.cast_lt.mp hcast
  rw [hV2]
  omega

/-- The scale `s = (V - 2·2^(k-1)) / (2D)` gives `s·D = V/2 - 2^(k-1)`. -/
lemma scale_mul_domain (V : ℝ) {D : ℝ} (hD : D ≠ 0) (k : ℕ) :
    (V - 2 * 2 ^ (k - 1)) / (2 * D) * D = V / 2 - 2 ^ (k - 1) := by
  field_simp

/-- The zero sequence has zero differences. -/
lemma fwdDiff_iter_zero (k : ℕ) : Δ_[1] ^[k] (0 : ℕ → ℝ) = 0 := by
  induction k with
  | zero => rfl
  | succ k ih =>
    have h0 : Δ_[1] (0 : ℕ → ℝ) = 0 := by
      funext m
      simp [fwdDiff]
    rw [Function.iterate_succ_apply, h0, ih]

/-- "lgt": levels on the absolute grid, `A = 0`. -/
theorem lgt_token_bound {V : ℤ} (hV : Even V) {k : ℕ} {s D : ℝ} (hs : 0 ≤ s) {x : ℕ → ℝ}
    (hD : ∀ n, |Δ_[1] ^[k] x n| ≤ D) (hsD : s * D ≤ (V : ℝ) / 2 - 2 ^ (k - 1)) (n : ℕ) :
    |Δ_[1] ^[k] (fun m => roundHalfUp (s * x m)) n| ≤ V / 2 - 1 := by
  simpa using abs_fwdDiff_iter_levels_le hV hs (fwdDiff_iter_zero k) hD hsD n

/-- The `k`-th difference of a polynomial of degree `< k`, read at `0, 1, 2, …`,
vanishes. -/
lemma fwdDiff_iter_eval_eq_zero {P : Polynomial ℝ} {k : ℕ} (hP : P.natDegree < k) :
    Δ_[1] ^[k] (fun n : ℕ => P.eval (n : ℝ)) = 0 := by
  funext n
  have h := congrFun (Polynomial.fwdDiff_iter_eq_zero_of_degree_lt hP) (n : ℝ)
  rw [fwdDiff_iter_eq_sum_shift] at h
  rw [fwdDiff_iter_eq_sum_shift]
  simpa [Nat.cast_add] using h

/-- "mash": levels measured from the polynomial `P` through the first `k`
values, so `deg P < k`. -/
theorem mash_token_bound {V : ℤ} (hV : Even V) {k : ℕ} {s D : ℝ} (hs : 0 ≤ s)
    {x : ℕ → ℝ} {P : Polynomial ℝ} (hP : P.natDegree < k)
    (hD : ∀ n, |Δ_[1] ^[k] x n| ≤ D) (hsD : s * D ≤ (V : ℝ) / 2 - 2 ^ (k - 1)) (n : ℕ) :
    |Δ_[1] ^[k] (fun m => roundHalfUp (s * (x m - P.eval (m : ℝ)))) n| ≤ V / 2 - 1 :=
  abs_fwdDiff_iter_levels_le hV hs (fwdDiff_iter_eval_eq_zero hP) hD hsD n

/-- The `k`-th differences and the first `k` values determine an integer sequence. -/
lemma eq_of_fwdDiff_iter_eq {k : ℕ} {L L' : ℕ → ℤ} (h : Δ_[1] ^[k] L = Δ_[1] ^[k] L')
    (h0 : ∀ m < k, L m = L' m) : L = L' := by
  induction k generalizing L L' with
  | zero => simpa using h
  | succ k ih =>
    rw [Function.iterate_succ_apply, Function.iterate_succ_apply] at h
    have hd : Δ_[1] L = Δ_[1] L' := ih h fun m hm => by
      simp only [fwdDiff]
      rw [h0 m (by omega), h0 (m + 1) (by omega)]
    funext m
    induction m with
    | zero => exact h0 0 (by omega)
    | succ m ihm =>
      have hm := congrFun hd m
      simp only [fwdDiff] at hm
      omega

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

/-- With the scale of `scale_mul_domain`, `1/(2s) = D / (V - 2·2^(k-1))`. -/
lemma half_step_eq {V D : ℝ} (hD : D ≠ 0) {k : ℕ} (hV : V - 2 * 2 ^ (k - 1) ≠ 0) :
    1 / (2 * ((V - 2 * 2 ^ (k - 1)) / (2 * D))) = D / (V - 2 * 2 ^ (k - 1)) := by
  field_simp

end DiffGPT
