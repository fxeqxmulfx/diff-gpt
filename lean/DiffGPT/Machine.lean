module

public import Mathlib
public import DiffGPT.LevelGrid
public import DiffGPT.Float64
public import DiffGPT.Int64

/-!
# The tokenizer in float64 and int64

For a column `x` of floats with scale `s` and order `k`, `encode` computes over its `k` prefix
rows and the rows after them (index `m`), with `fl` the float64 rounding:

    u m = fl (s * x m)                               `scaled = inp * scale`
    F m = (P m mod 2^G) / 2^G                         `_anchor_fraction`, exact
    w m = fl (u m - F m)                              `scaled - fraction`
    R m = ⌊w m + 1/2⌋                                 `_round_levels`, exact (`roundLevels_eq`)
    q n = Δᵏ R n + Δᵏ (P mod 2^G) n / 2^G + V/2       the token of row `n + k` (`token`)

with `P = 0` for "lgt" and, for "mash", the integer sequence with `Δᵏ P = 0` that
`_anchor_fraction` continues from the prefix in uint64 (`anchor_numerators`). `decode`
integrates `q - V/2 - Δᵏ (P mod 2^G) / 2^G` from the prefix levels and returns
`fl (fl (fl R + F) / s)`.

* `token_eq`: `q - V/2` is the `k`-th difference of the levels `⌊ũ - B + 1/2⌋` of the
  effective series `ũ = w + F` from the anchor `B = P / 2^G`, so the exact theory applies to
  `ũ` (`token_mem`). `ũ` is within two roundings of `s·x` (`abs_effective_sub_le`), so every
  token lies in `[1, V-1]` once the scale leaves room for `2^k` such errors
  (`token_mem_of_domain`).
* `decode_levels_eq`: the decoder's integration returns the encoder's levels, and
  `abs_decode_sub_le_f64`: the decoded value is within `1/(2s)` of `x`, up to the error of `ũ`
  and three more roundings.
* `abs_fwdDiff_iter_levels_le_two_pow`: under `_round_levels`' guard `|w| < 2^(62-k)` the
  levels and their differences up to the `k`-th lie within `2^62`, so `np.diff` and the
  decoder's `_cumsum` are exact in int64 (`toInt_diffI64_iter_of_abs_le`,
  `toInt_integrateI64`), and `abs_fracSteps_lt` bounds the anchor's term of the token.
  `fl_roundHalfUp_eq`: under the guard's other bound, `|w| < 2^53`, every level is a float.
-/

@[expose] public section

open fwdDiff

namespace DiffGPT

/-- The anchor's fraction at row `m`, `(P m mod 2^G) / 2^G`, as `_anchor_fraction` returns it. -/
noncomputable def anchorFrac (G : ℕ) (P : ℕ → ℤ) (m : ℕ) : ℝ :=
  ((P m % 2 ^ G : ℤ) : ℝ) / 2 ^ G

/-- `_anchor_fraction`'s `fraction_steps`: `np.diff(numerators, n=k) >> G`. -/
def fracSteps (k G : ℕ) (P : ℕ → ℤ) (n : ℕ) : ℤ :=
  Δ_[1] ^[k] (fun m => P m % 2 ^ G) n / 2 ^ G

/-- The token `encode` emits for row `n + k`, before clipping:
`np.diff(levels, n=k) + fraction_steps + vocab_size // 2` with `levels = _round_levels(w)`. -/
noncomputable def token (V : ℤ) (k G : ℕ) (P : ℕ → ℤ) (w : ℕ → ℝ) (n : ℕ) : ℤ :=
  Δ_[1] ^[k] (fun m => roundHalfUp (w m)) n + fracSteps k G P n + V / 2

/-- With `Δᵏ P = 0` the anchor's term is exact: `fracSteps = -Δᵏ ⌊P / 2^G⌋`. -/
lemma fracSteps_eq {k G : ℕ} {P : ℕ → ℤ} (hP : Δ_[1] ^[k] P = 0) (n : ℕ) :
    fracSteps k G P n = -Δ_[1] ^[k] (fun m => P m / 2 ^ G) n := by
  rw [fracSteps, fwdDiff_iter_emod_eq hP G n, ← mul_neg,
    Int.mul_ediv_cancel_left _ (by positivity)]

/-- The anchor's term of the token is exact, `-Δᵏ ⌊B⌋`, and below `2^(k-1)`. -/
theorem abs_fracSteps_lt {k G : ℕ} (hk : 1 ≤ k) {P : ℕ → ℤ} (hP : Δ_[1] ^[k] P = 0) (n : ℕ) :
    |fracSteps k G P n| < 2 ^ (k - 1) := by
  have h := abs_fwdDiff_iter_emod_lt P hk G n
  have h2 : (0 : ℤ) < 2 ^ G := by positivity
  rw [fwdDiff_iter_emod_eq hP G n, abs_neg, abs_mul, abs_of_pos h2, pow_add,
    mul_comm (2 ^ (k - 1))] at h
  rw [fracSteps_eq hP, abs_neg]
  exact lt_of_mul_lt_mul_left h h2.le

/-- The tokens are the `k`-th differences of the levels of the effective series `ũ = w + F`,
measured from the anchor `B = P / 2^G`, offset by `V/2`. -/
theorem token_eq {k G : ℕ} {P : ℕ → ℤ} (hP : Δ_[1] ^[k] P = 0) (V : ℤ) (w : ℕ → ℝ) (n : ℕ) :
    token V k G P w n =
      Δ_[1] ^[k] (fun m => roundHalfUp (w m + anchorFrac G P m - (P m : ℝ) / 2 ^ G)) n
        + V / 2 := by
  have hR : (fun m => roundHalfUp (w m + anchorFrac G P m - (P m : ℝ) / 2 ^ G)) =
      (fun m => roundHalfUp (w m)) + (-1 : ℤ) • fun m => P m / 2 ^ G := by
    funext m
    have hq : w m + anchorFrac G P m - (P m : ℝ) / 2 ^ G + 1 / 2 =
        w m + 1 / 2 - ((P m / 2 ^ G : ℤ) : ℝ) := by
      have h := congrArg (fun z : ℤ => (z : ℝ)) (Int.emod_def (P m) (2 ^ G))
      push_cast at h
      rw [anchorFrac, h]
      field_simp
      ring
    rw [Pi.add_apply, Pi.smul_apply, smul_eq_mul, roundHalfUp, roundHalfUp, hq,
      Int.floor_sub_intCast]
    ring
  rw [token, hR, fwdDiff_iter_add, fwdDiff_iter_const_smul, Pi.add_apply, Pi.smul_apply,
    smul_eq_mul, fracSteps_eq hP]
  ring

/-- Every token lies in `[1, V-1]` when the effective series' `k`-th differences stay within
`V/2 - 2^(k-1)` (`abs_fwdDiff_iter_levels_le`). -/
theorem token_mem {V : ℤ} (hV : Even V) {k G : ℕ} {P : ℕ → ℤ} (hP : Δ_[1] ^[k] P = 0)
    {w : ℕ → ℝ}
    (hw : ∀ n, |Δ_[1] ^[k] (fun m => w m + anchorFrac G P m) n| ≤ (V : ℝ) / 2 - 2 ^ (k - 1))
    (n : ℕ) : 1 ≤ token V k G P w n ∧ token V k G P w n ≤ V - 1 := by
  have hA : Δ_[1] ^[k] (fun m => (P m : ℝ) / 2 ^ G) = 0 := by
    have h1 : (fun m => (P m : ℝ) / 2 ^ G) = ((2 : ℝ) ^ G)⁻¹ • fun m => (P m : ℝ) := by
      funext m
      rw [Pi.smul_apply, smul_eq_mul, div_eq_inv_mul]
    funext m
    rw [h1, fwdDiff_iter_const_smul, Pi.smul_apply, ← cast_fwdDiff_iter, hP]
    simp only [Pi.zero_apply, Int.cast_zero, smul_zero]
  have h := abs_fwdDiff_iter_levels_le hV zero_le_one hA hw (by rw [one_mul]) n
  simp only [one_mul] at h
  rw [token_eq hP]
  obtain ⟨r, rfl⟩ := hV
  rw [abs_le] at h
  constructor <;> omega

/-- The effective series `fl (fl (s·x) - F) + F` is within two roundings of `s·x`. -/
theorem abs_effective_sub_le {fl : ℝ → ℝ} (hfl : IsRoundNearest fl) (s x F : ℝ) :
    |fl (fl (s * x) - F) + F - s * x| ≤ rndErr (s * x) + rndErr (fl (s * x) - F) := by
  calc |fl (fl (s * x) - F) + F - s * x|
      = |(fl (fl (s * x) - F) - (fl (s * x) - F)) + (fl (s * x) - s * x)| := by
        congr 1
        ring
    _ ≤ |fl (fl (s * x) - F) - (fl (s * x) - F)| + |fl (s * x) - s * x| := abs_add_le _ _
    _ ≤ rndErr (fl (s * x) - F) + rndErr (s * x) :=
        add_le_add (hfl.abs_sub_le _) (hfl.abs_sub_le _)
    _ = rndErr (s * x) + rndErr (fl (s * x) - F) := add_comm _ _

lemma abs_fwdDiff_iter_le_of_abs_le {η : ℝ} {g : ℕ → ℝ} (h : ∀ n, |g n| ≤ η) (k n : ℕ) :
    |Δ_[1] ^[k] g n| ≤ 2 ^ k * η := by
  induction k generalizing g η with
  | zero =>
    simp only [Function.iterate_zero, id_eq, pow_zero, one_mul]
    exact h n
  | succ k ih =>
    rw [Function.iterate_succ_apply]
    have h2 : ∀ m, |Δ_[1] g m| ≤ 2 * η := fun m => by
      obtain ⟨b1, b2⟩ := abs_le.mp (h (m + 1))
      obtain ⟨c1, c2⟩ := abs_le.mp (h m)
      exact abs_le.mpr ⟨by simp only [fwdDiff]; linarith, by simp only [fwdDiff]; linarith⟩
    calc |Δ_[1] ^[k] (Δ_[1] g) n| ≤ 2 ^ k * (2 * η) := ih h2
      _ = 2 ^ (k + 1) * η := by ring

/-- The token range in float64: if `|Δᵏ x| ≤ D`, the effective series is within `η` of `s·x`
and the scale leaves room for `2^k` such errors, `s·D + 2^k·η ≤ V/2 - 2^(k-1)`, every token
lies in `[1, V-1]`. -/
theorem token_mem_of_domain {fl : ℝ → ℝ} {V : ℤ} (hV : Even V) {k G : ℕ} {P : ℕ → ℤ}
    (hP : Δ_[1] ^[k] P = 0) {s D η : ℝ} (hs : 0 ≤ s) {x : ℕ → ℝ}
    (hD : ∀ n, |Δ_[1] ^[k] x n| ≤ D)
    (hη : ∀ m, |fl (fl (s * x m) - anchorFrac G P m) + anchorFrac G P m - s * x m| ≤ η)
    (hsD : s * D + 2 ^ k * η ≤ (V : ℝ) / 2 - 2 ^ (k - 1)) (n : ℕ) :
    1 ≤ token V k G P (fun m => fl (fl (s * x m) - anchorFrac G P m)) n ∧
      token V k G P (fun m => fl (fl (s * x m) - anchorFrac G P m)) n ≤ V - 1 := by
  refine token_mem hV hP (fun n => ?_) n
  set e : ℕ → ℝ :=
    fun m => fl (fl (s * x m) - anchorFrac G P m) + anchorFrac G P m - s * x m with he
  have hsplit :
      (fun m => fl (fl (s * x m) - anchorFrac G P m) + anchorFrac G P m) = s • x + e := by
    funext m
    simp only [he, Pi.add_apply, Pi.smul_apply, smul_eq_mul]
    ring
  rw [hsplit, fwdDiff_iter_add, fwdDiff_iter_const_smul, Pi.add_apply, Pi.smul_apply,
    smul_eq_mul]
  have hx : |s * Δ_[1] ^[k] x n| ≤ s * D := by
    rw [abs_mul, abs_of_nonneg hs]
    exact mul_le_mul_of_nonneg_left (hD n) hs
  calc |s * Δ_[1] ^[k] x n + Δ_[1] ^[k] e n|
      ≤ |s * Δ_[1] ^[k] x n| + |Δ_[1] ^[k] e n| := abs_add_le _ _
    _ ≤ s * D + 2 ^ k * η := add_le_add hx (abs_fwdDiff_iter_le_of_abs_le hη k n)
    _ ≤ (V : ℝ) / 2 - 2 ^ (k - 1) := hsD

/-- The decoder's integration of `token - V/2 - fracSteps` from the prefix levels returns the
encoder's levels (`eq_of_fwdDiff_iter_eq`). -/
theorem decode_levels_eq (V : ℤ) (k G : ℕ) (P : ℕ → ℤ) (w : ℕ → ℝ) :
    integrate k (fun m => roundHalfUp (w m))
        (fun n => token V k G P w n - V / 2 - fracSteps k G P n) =
      fun m => roundHalfUp (w m) := by
  have hd : (fun n => token V k G P w n - V / 2 - fracSteps k G P n) =
      Δ_[1] ^[k] (fun m => roundHalfUp (w m)) := by
    funext n
    rw [token]
    ring
  rw [hd]
  exact eq_of_fwdDiff_iter_eq (fwdDiff_iter_integrate k _ _) fun m hm => integrate_of_lt hm _ _

/-- The decoded value `fl (fl (fl R + F) / s)` is within `1/(2s)` of `x`, up to the error of
the effective series `w + F` and the roundings of the three operations. -/
theorem abs_decode_sub_le_f64 {fl : ℝ → ℝ} (hfl : IsRoundNearest fl) {s : ℝ} (hs : 0 < s)
    (x w F : ℝ) :
    |fl (fl (fl (roundHalfUp w) + F) / s) - x| ≤
      1 / (2 * s)
        + (|w + F - s * x| + rndErr (roundHalfUp w) + rndErr (fl (roundHalfUp w) + F)) / s
        + rndErr (fl (fl (roundHalfUp w) + F) / s) := by
  set R : ℝ := (roundHalfUp w : ℝ)
  set y := fl (fl R + F)
  obtain ⟨h1, h2⟩ := abs_le.mp (hfl.abs_sub_le (fl R + F))
  obtain ⟨h3, h4⟩ := abs_le.mp (hfl.abs_sub_le R)
  obtain ⟨h5, h6⟩ := roundHalfUp_sub_mem w
  have h7 := neg_abs_le (w + F - s * x)
  have h8 := le_abs_self (w + F - s * x)
  have hys : |y - s * x| ≤ 1 / 2 + (|w + F - s * x| + rndErr R + rndErr (fl R + F)) :=
    abs_le.mpr ⟨by linarith, by linarith⟩
  have hdiv :
      |y / s - x| ≤ 1 / (2 * s) + (|w + F - s * x| + rndErr R + rndErr (fl R + F)) / s := by
    rw [show y / s - x = (y - s * x) / s by field_simp, abs_div, abs_of_pos hs,
      show 1 / (2 * s) = 1 / 2 / s by ring, ← add_div]
    exact div_le_div_of_nonneg_right hys hs.le
  calc |fl (y / s) - x| ≤ |fl (y / s) - y / s| + |y / s - x| := abs_sub_le _ _ _
    _ ≤ rndErr (y / s) + (1 / (2 * s) + (|w + F - s * x| + rndErr R + rndErr (fl R + F)) / s) :=
        add_le_add (hfl.abs_sub_le _) hdiv
    _ = _ := by ring

/-- `_round_levels`' guard bounds the levels: `|u| < 2^j` gives `|⌊u + 1/2⌋| ≤ 2^j`. -/
lemma abs_roundHalfUp_le {u : ℝ} {j : ℕ} (hu : |u| < 2 ^ j) : |roundHalfUp u| ≤ 2 ^ j := by
  obtain ⟨h1, h2⟩ := roundHalfUp_sub_mem u
  obtain ⟨hu1, hu2⟩ := abs_lt.mp hu
  have hlt1 : ((roundHalfUp u : ℤ) : ℝ) < ((2 ^ j + 1 : ℤ) : ℝ) := by
    push_cast
    linarith
  have hlt2 : ((-2 ^ j - 1 : ℤ) : ℝ) < ((roundHalfUp u : ℤ) : ℝ) := by
    push_cast
    linarith
  have h3 := Int.cast_lt.mp hlt1
  have h4 := Int.cast_lt.mp hlt2
  rw [abs_le]
  constructor <;> omega

/-- `_round_levels`' other bound, `|w| < 2^53`: every level is a float, so `fl R = R` and the
first of the decode's roundings, `rndErr R` in `abs_decode_sub_le_f64`, vanishes. -/
theorem fl_roundHalfUp_eq {fl : ℝ → ℝ} (hfl : IsRoundNearest fl) {w : ℝ} (hw : |w| < 2 ^ 53) :
    fl (roundHalfUp w) = roundHalfUp w :=
  hfl.eq_self (isF64_intCast (abs_roundHalfUp_le hw))

/-- Under the guard `|w| < 2^(62-k)` the levels and their differences up to the `k`-th lie
within `2^62`, inside int64. -/
theorem abs_fwdDiff_iter_levels_le_two_pow {k : ℕ} (hk : k ≤ 62) {w : ℕ → ℝ}
    (hw : ∀ m, |w m| < 2 ^ (62 - k)) {j : ℕ} (hj : j ≤ k) (n : ℕ) :
    |Δ_[1] ^[j] (fun m => roundHalfUp (w m)) n| ≤ 2 ^ 62 :=
  calc |Δ_[1] ^[j] (fun m => roundHalfUp (w m)) n| ≤ 2 ^ j * 2 ^ (62 - k) :=
        abs_fwdDiff_iter_le (fun m => abs_roundHalfUp_le (hw m)) j n
    _ = 2 ^ (j + (62 - k)) := (pow_add _ _ _).symm
    _ ≤ 2 ^ 62 := pow_le_pow_right₀ (by norm_num) (by omega)

end DiffGPT
