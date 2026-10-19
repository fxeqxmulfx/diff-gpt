module

public import Mathlib
public import DiffGPT.LevelGrid
public import DiffGPT.Float64
public import DiffGPT.ExactLevels
public import DiffGPT.Int64

/-!
# The tokenizer in float64 and int64

For a column `x` of floats with scale `s` and order `k`, `encode` computes over its `k` prefix
rows and the rows after them (index `m`):

    F m = (P m mod 2^G) / 2^G                         `_anchor_fraction`, exact
    R m = ⌊s·x m - F m + 1/2⌋                         `_levels`, exact (`levels_eq`)
    q n = Δᵏ R n + Δᵏ (P mod 2^G) n / 2^G + V/2       the token of row `n + k` (`token`)

with `P = 0` for "lgt" and, for "mash", the integer sequence with `Δᵏ P = 0` that
`_anchor_fraction` continues from the prefix in uint64 (`anchor_numerators`). `decode`
integrates `q - V/2 - Δᵏ (P mod 2^G) / 2^G` from the prefix levels and returns
`fl (fl (R + F) / s)`.

* `token_eq`: `q - V/2` is the `k`-th difference of the levels `⌊s·x - B + 1/2⌋` from the
  anchor `B = P / 2^G`, so the exact theory applies (`token_mem`): every token lies in
  `[1, V-1]` when `|Δᵏ x| ≤ D` and `s·D ≤ V/2 - 2^(k-1)` (`token_mem_of_domain`), which the
  scale, rounded down, and the max of `get_domain_of_definition`, rounded up, give exactly.
* `encode_token_mem`: so do the tokens as `encode` computes them, `_levels` in float64 and
  `np.diff` in int64. Under `_levels`' guard `|p| < 2^b` with `b ≤ 53` and `b + k ≤ 62`, which
  `b = min(53, 62 - k)` meets, the levels lie in `[-2^b - 1, 2^b]` (`levels_mem`), so
  `np.diff` is exact (`toInt_diffI64_iter_of_mem_Icc`), and `abs_fracSteps_lt` bounds the
  anchor's term.
* `decode_levels_eq`: the decoder's integration returns the encoder's levels, exactly in int64
  (`toInt_integrateI64`), and `abs_decode_sub_le_f64`: the decoded value is within `1/(2s)` of
  `x` up to the roundings of `R + F` and of the quotient, the level being a float
  (`abs_levels_le`).
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
`np.diff(levels, n=k) + fraction_steps + vocab_size // 2` with the levels `⌊w + 1/2⌋`, which
`_levels` returns for `w = s·x - F` (`levels_eq`). -/
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

/-- The tokens are the `k`-th differences of the levels of `w + F`, `s·x` for `w = s·x - F`,
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

/-- Every token lies in `[1, V-1]` when the `k`-th differences of `w + F` stay within
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

/-- The token range: every token of the levels of `s·x - F` lies in `[1, V-1]` when
`|Δᵏ x| ≤ D` and `s·D ≤ V/2 - 2^(k-1)`. -/
theorem token_mem_of_domain {V : ℤ} (hV : Even V) {k G : ℕ} {P : ℕ → ℤ}
    (hP : Δ_[1] ^[k] P = 0) {s D : ℝ} (hs : 0 ≤ s) {x : ℕ → ℝ}
    (hD : ∀ n, |Δ_[1] ^[k] x n| ≤ D) (hsD : s * D ≤ (V : ℝ) / 2 - 2 ^ (k - 1)) (n : ℕ) :
    1 ≤ token V k G P (fun m => s * x m - anchorFrac G P m) n ∧
      token V k G P (fun m => s * x m - anchorFrac G P m) n ≤ V - 1 := by
  refine token_mem hV hP (fun n => ?_) n
  have h : (fun m => s * x m - anchorFrac G P m + anchorFrac G P m) = s • x := by
    funext m
    simp only [Pi.smul_apply, smul_eq_mul]
    ring
  rw [h, fwdDiff_iter_const_smul, Pi.smul_apply, smul_eq_mul, abs_mul, abs_of_nonneg hs]
  exact (mul_le_mul_of_nonneg_left (hD n) hs).trans hsD

/-- `encode`'s tokens in float64 and int64: for `s = m_s·2^e_s` and `x m = m_x m·2^(e_x m)` as
`np.frexp` splits them, the int64 levels `L` that `_levels` returns and its guard
`|p| < 2^b`, `b ≤ 53`, `b + k ≤ 62`, every token
`np.diff(L, n=k) + fraction_steps + vocab_size // 2` lies in `[1, V-1]` when `|Δᵏ x| ≤ D` and
`s·D ≤ V/2 - 2^(k-1)`. -/
theorem encode_token_mem {fl : ℝ → ℝ} (hfl : IsRoundNearest fl) {ri : ℝ → ℤ}
    (hri : IsRoundInt ri) {V : ℤ} (hV : Even V) {k G b : ℕ} (hG : G ≤ 53) (hb : b ≤ 53)
    (hbk : b + k ≤ 62) {P : ℕ → ℤ} (hP : Δ_[1] ^[k] P = 0) {ms : ℝ} {es : ℤ}
    (hms : IsMantissa ms) {mx : ℕ → ℝ} {ex : ℕ → ℤ} (hmx : ∀ m, IsMantissa (mx m))
    (hguard : ∀ m, |levelHi fl ms es (mx m) (ex m)| < 2 ^ b) {s D : ℝ} (hs : s = ms * 2 ^ es)
    (hs0 : 0 ≤ s) {x : ℕ → ℝ} (hx : ∀ m, x m = mx m * 2 ^ ex m)
    (hD : ∀ n, |Δ_[1] ^[k] x n| ≤ D) (hsD : s * D ≤ (V : ℝ) / 2 - 2 ^ (k - 1))
    {L : ℕ → Int64}
    (hL : ∀ m, (L m).toInt = levels fl ri ms es (mx m) (ex m) (anchorFrac G P m)) (n : ℕ) :
    1 ≤ (diffI64^[k] L n).toInt + fracSteps k G P n + V / 2 ∧
      (diffI64^[k] L n).toInt + fracSteps k G P n + V / 2 ≤ V - 1 := by
  have hF (m : ℕ) : ∃ N : ℤ, anchorFrac G P m = N / 2 ^ G ∧ 0 ≤ N ∧ N < 2 ^ G :=
    ⟨P m % 2 ^ G, rfl, Int.emod_nonneg _ (by positivity), Int.emod_lt_of_pos _ (by positivity)⟩
  have hR (m : ℕ) : (L m).toInt = roundHalfUp (s * x m - anchorFrac G P m) := by
    obtain ⟨N, hN, hN0, hN1⟩ := hF m
    rw [hL m, hN, levels_eq hfl hri hms (hmx m) es (ex m) hG hN0 hN1
      ((hguard m).trans_le (pow_le_pow_right₀ (by norm_num) hb)), hs, hx m]
  have hdiff : (diffI64^[k] L n).toInt = Δ_[1] ^[k] (fun m => (L m).toInt) n := by
    refine toInt_diffI64_iter_of_mem_Icc hbk (fun m => ?_) n
    obtain ⟨N, hN, hN0, hN1⟩ := hF m
    rw [hL m, hN]
    exact levels_mem hfl hri hms (hmx m) es (ex m) hG hN0 hN1 hb (hguard m)
  rw [hdiff, show (fun m => (L m).toInt) = fun m => roundHalfUp (s * x m - anchorFrac G P m)
    from funext hR]
  exact token_mem_of_domain hV hP hs0 hD hsD n

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

/-- The decoded value `fl (fl (fl R + F) / s)` of the level `R = ⌊s·x - F + 1/2⌋` is within
`1/(2s)` of `x`, up to the roundings of the sum and the quotient, when `R` lies within `2^53`,
as under `_levels`' guard (`abs_levels_le`): then `fl R = R`. -/
theorem abs_decode_sub_le_f64 {fl : ℝ → ℝ} (hfl : IsRoundNearest fl) {s : ℝ} (hs : 0 < s)
    (x F : ℝ) (hR : |roundHalfUp (s * x - F)| ≤ 2 ^ 53) :
    |fl (fl (fl (roundHalfUp (s * x - F)) + F) / s) - x| ≤
      1 / (2 * s) + rndErr (roundHalfUp (s * x - F) + F) / s
        + rndErr (fl (roundHalfUp (s * x - F) + F) / s) := by
  set R : ℝ := (roundHalfUp (s * x - F) : ℝ)
  rw [hfl.eq_self (isF64_intCast hR)]
  set y := fl (R + F)
  obtain ⟨h1, h2⟩ := abs_le.mp (hfl.abs_sub_le (R + F))
  obtain ⟨h3, h4⟩ := roundHalfUp_sub_mem (s * x - F)
  have hys : |y - s * x| ≤ 1 / 2 + rndErr (R + F) := abs_le.mpr ⟨by linarith, by linarith⟩
  have hdiv : |y / s - x| ≤ 1 / (2 * s) + rndErr (R + F) / s := by
    rw [show y / s - x = (y - s * x) / s by field_simp, abs_div, abs_of_pos hs,
      show 1 / (2 * s) = 1 / 2 / s by ring, ← add_div]
    exact div_le_div_of_nonneg_right hys hs.le
  calc |fl (y / s) - x| ≤ |fl (y / s) - y / s| + |y / s - x| := abs_sub_le _ _ _
    _ ≤ rndErr (y / s) + (1 / (2 * s) + rndErr (R + F) / s) :=
        add_le_add (hfl.abs_sub_le _) hdiv
    _ = _ := by ring

end DiffGPT
