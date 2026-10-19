module

public import Mathlib
public import DiffGPT.LevelGrid
public import DiffGPT.Float64
public import DiffGPT.ExactLevels
public import DiffGPT.Int64

/-!
# The tokenizer in float64 and int64

`encode` and `decode` as the code computes them, for one column `x` of order `k` with scale `s`:
its `k` prefix rows and the `N` rows after them (index `m`), the token of row `n + k` at `n`.

    P = integrate k seeds 0 mod 2^64          `_integrate_levels` in uint64
    F m = (P m mod 2^G) / 2^G                 `anchorFraction`, exact
    FS n = Δᵏ (P mod 2^G) n >> G              `anchorSteps`, exact
    L m = ⌊s·x m - F m + 1/2⌋                 `levelsI64`: `_levels`, exact (`levels_eq`)
    q n = clip (Δᵏ L n + FS n + V/2)          `encodeTokens`, in int64
    L' = integrate k L (q - V/2 - FS)         `decodeLevels`: `_integrate_levels` in int64
    x̂ m = fl (fl (fl (L' m) + F m) / s)       `decodeOut`, `x m` on the prefix rows

The `seeds` are the uint64 values `_anchor_fraction` takes off the prefix: for "mash" the
fractional parts of its `s·x` in units of `2^-G`, rounded; "lgt" is `seeds = 0`, which gives
`F = FS = 0` (`anchor_zero`). `P` is the integer `anchorInt`, with `Δᵏ P = 0`, modulo `2^64`.
`decode` takes the anchor and the prefix levels off the same floats, `start`, so it gets the
same `F`, `FS` and `L m`, `m < k`.

* `token_eq`, `token_mem`, `token_mem_of_domain`: the integer token `Δᵏ L + FS + V/2` is the
  `k`-th difference of the levels `⌊s·x - B + 1/2⌋` measured from the anchor `B = P / 2^G`,
  offset by `V/2`, so the exact theory applies: it lies in `[1, V-1]` where `|Δᵏ x| ≤ D` and
  `s·D ≤ V/2 - 2^(k-1)`, which the scale, rounded down, and the max of
  `get_domain_of_definition`, rounded up, give exactly.
* `encode_token_mem`: the int64 token sum is that token. The two agree modulo `2^64` and both
  lie in int64, so this needs only `_levels`' float64 guard `|p| < 2^53`. Neither clip then
  changes the sum (`encodeTokens_eq`).
* `toInt_tokenSum`: under the guard `|p| < 2^b` with `b + k ≤ 62`, which `b = min(53, 62 - k)`
  meets, the token sum is the integer one on every row, in the domain or not, so the clip puts a
  token outside `[1, V-1]` on its own side.
* `decode_levels_eq`: `decode` gets the levels `L` back from `encode`'s tokens, and
  `_integrate_levels` does not raise: its cumulative sums pass through the differences `Δʲ L`,
  `j ≤ k`, which int64 holds for `b + k ≤ 62` (`toInt_integrateI64`).
* `abs_decode_sub_le_f64`, `abs_decode_sub_le_closed`: the decoded value is within `1/(2s)` of
  `x` up to the roundings of `L + F` and of the quotient, at most
  `2^-52·(1 + 2^-54)·(|x| + 1/(2s)) + 2^-1075·(1 + (1 + 2^-53)/s)` more, the level being a
  float (`abs_levels_le`).
* `encode_decode`: all of it for a column, from the floats to the decoded values.
-/

@[expose] public section

open fwdDiff

namespace DiffGPT

/-! ### The tokens in integers -/

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

/-- The token at `n` lies in `[1, V-1]` when the `k`-th difference of `w + F` there stays within
`V/2 - 2^(k-1)` (`abs_fwdDiff_iter_levels_le`). -/
theorem token_mem {V : ℤ} (hV : Even V) {k G : ℕ} {P : ℕ → ℤ} (hP : Δ_[1] ^[k] P = 0)
    {w : ℕ → ℝ} {n : ℕ}
    (hw : |Δ_[1] ^[k] (fun m => w m + anchorFrac G P m) n| ≤ (V : ℝ) / 2 - 2 ^ (k - 1)) :
    1 ≤ token V k G P w n ∧ token V k G P w n ≤ V - 1 := by
  have hA : Δ_[1] ^[k] (fun m => (P m : ℝ) / 2 ^ G) = 0 := by
    have h1 : (fun m => (P m : ℝ) / 2 ^ G) = ((2 : ℝ) ^ G)⁻¹ • fun m => (P m : ℝ) := by
      funext m
      rw [Pi.smul_apply, smul_eq_mul, div_eq_inv_mul]
    funext m
    rw [h1, fwdDiff_iter_const_smul, Pi.smul_apply, ← cast_fwdDiff_iter, hP]
    simp only [Pi.zero_apply, Int.cast_zero, smul_zero]
  have h := abs_fwdDiff_iter_levels_le hV zero_le_one hA hw (by rw [one_mul])
  simp only [one_mul] at h
  rw [token_eq hP]
  obtain ⟨r, rfl⟩ := hV
  rw [abs_le] at h
  constructor <;> omega

/-- The token range: the token of the levels of `s·x - F` at `n` lies in `[1, V-1]` when
`|Δᵏ x n| ≤ D` and `s·D ≤ V/2 - 2^(k-1)`. -/
theorem token_mem_of_domain {V : ℤ} (hV : Even V) {k G : ℕ} {P : ℕ → ℤ}
    (hP : Δ_[1] ^[k] P = 0) {s D : ℝ} (hs : 0 ≤ s) {x : ℕ → ℝ} {n : ℕ}
    (hD : |Δ_[1] ^[k] x n| ≤ D) (hsD : s * D ≤ (V : ℝ) / 2 - 2 ^ (k - 1)) :
    1 ≤ token V k G P (fun m => s * x m - anchorFrac G P m) n ∧
      token V k G P (fun m => s * x m - anchorFrac G P m) n ≤ V - 1 := by
  refine token_mem hV hP ?_
  have h : (fun m => s * x m - anchorFrac G P m + anchorFrac G P m) = s • x := by
    funext m
    simp only [Pi.smul_apply, smul_eq_mul]
    ring
  rw [h, fwdDiff_iter_const_smul, Pi.smul_apply, smul_eq_mul, abs_mul, abs_of_nonneg hs]
  exact (mul_le_mul_of_nonneg_left hD hs).trans hsD

/-! ### The code -/

open scoped UInt64.CommRing in
/-- `_anchor_fraction`'s `numerators`: `_integrate_levels` of the uint64 `seeds` with zero
steps, which wraps modulo `2^64`, masked with `2^G - 1` and cast to int64. -/
noncomputable def anchorNum (k G : ℕ) (seeds : ℕ → UInt64) (m : ℕ) : Int64 :=
  ((integrate k seeds 0 m) &&& UInt64.ofNat (2 ^ G - 1)).toInt64

/-- The integer anchor `P`, which `anchorNum` reduces modulo `2^G`: `integrate k seeds 0` over
the integers. -/
def anchorInt (k : ℕ) (seeds : ℕ → UInt64) : ℕ → ℤ :=
  integrate k (fun m => ((seeds m).toNat : ℤ)) 0

/-- `_anchor_fraction`'s `fraction`: `np.ldexp(numerators.astype(np.float64), -G)`. -/
noncomputable def anchorFraction (fl : ℝ → ℝ) (k G : ℕ) (seeds : ℕ → UInt64) (m : ℕ) : ℝ :=
  fl ((anchorNum k G seeds m).toInt : ℝ) * 2 ^ (-(G : ℤ))

/-- `_anchor_fraction`'s `fraction_steps`: `np.diff(numerators, n=k) >> G` in int64. -/
noncomputable def anchorSteps (k G : ℕ) (seeds : ℕ → UInt64) (n : ℕ) : Int64 :=
  diffI64^[k] (anchorNum k G seeds) n >>> Int64.ofNat G

/-- `_levels`' int64 levels of the rows `x m = m_x m·2^(e_x m)` with the fractions `F`. -/
noncomputable def levelsI64 (fl : ℝ → ℝ) (ri : ℝ → ℤ) (ms : ℝ) (es : ℤ) (mx : ℕ → ℝ)
    (ex : ℕ → ℤ) (F : ℕ → ℝ) (m : ℕ) : Int64 :=
  Int64.ofInt (levels fl ri ms es (mx m) (ex m) (F m))

/-- `np.clip(a, lo, hi)` in int64. -/
def clipI64 (lo hi a : Int64) : Int64 := if a < lo then lo else if hi < a then hi else a

/-- `encode`'s token sums: `np.diff(levels, n=k) + fraction_steps + vocab_size // 2` in int64. -/
def tokenSum (V : ℤ) (k : ℕ) (L FS : ℕ → Int64) (n : ℕ) : Int64 :=
  diffI64^[k] L n + FS n + Int64.ofInt (V / 2)

/-- `encode`'s tokens: the sums clipped to `[-1, V]`, then by `_clip_tokens` to `[0, V-1]`. -/
def encodeTokens (V : ℤ) (k : ℕ) (L FS : ℕ → Int64) (n : ℕ) : Int64 :=
  clipI64 0 (Int64.ofInt (V - 1)) (clipI64 (-1) (Int64.ofInt V) (tokenSum V k L FS n))

/-- `decode`'s steps: `tokens - vocab_size // 2 - fraction_steps` in int64. -/
def decodeSteps (V : ℤ) (FS q : ℕ → Int64) (n : ℕ) : Int64 :=
  q n - Int64.ofInt (V / 2) - FS n

/-- `decode`'s levels: `_integrate_levels` of the prefix levels and the steps, in int64. -/
def decodeLevels (V : ℤ) (k : ℕ) (L FS q : ℕ → Int64) : ℕ → Int64 :=
  integrateI64 k L (decodeSteps V FS q)

/-- `decode`'s output: `start` on the `k` prefix rows, `(levels + fraction) / scale` in float64
after them. -/
noncomputable def decodeOut (fl : ℝ → ℝ) (s : ℝ) (k : ℕ) (x F : ℕ → ℝ) (L : ℕ → Int64)
    (m : ℕ) : ℝ :=
  if m < k then x m else fl (fl (fl ((L m).toInt : ℝ) + F m) / s)

/-! ### The anchor -/

lemma fwdDiff_iter_anchorInt (k : ℕ) (seeds : ℕ → UInt64) :
    Δ_[1] ^[k] (anchorInt k seeds) = 0 :=
  fwdDiff_iter_integrate k (fun m => ((seeds m).toNat : ℤ)) 0

open scoped UInt64.CommRing in
/-- The numerators are the integer anchor modulo `2^G` (`anchor_numerators`). -/
lemma toInt_anchorNum {G : ℕ} (hG : G ≤ 63) (k : ℕ) (seeds : ℕ → UInt64) (m : ℕ) :
    (anchorNum k G seeds m).toInt = anchorInt k seeds m % 2 ^ G := by
  have h := anchor_numerators (by omega : G ≤ 64) k seeds m
  have hu : ((integrate k seeds 0 m &&& UInt64.ofNat (2 ^ G - 1)).toNat : ℤ) < 2 ^ 63 := by
    rw [h]
    exact (Int.emod_lt_of_pos _ (by positivity)).trans_le (pow_le_pow_right₀ (by norm_num) hG)
  rw [anchorNum, toInt_toInt64_of_lt _ (by exact_mod_cast hu), h]
  rfl

lemma anchorFraction_eq {fl : ℝ → ℝ} (hfl : IsRoundNearest fl) {G : ℕ} (hG : G ≤ 53) (k : ℕ)
    (seeds : ℕ → UInt64) (m : ℕ) :
    anchorFraction fl k G seeds m = anchorFrac G (anchorInt k seeds) m := by
  have h0 := Int.emod_nonneg (anchorInt k seeds m) (by positivity : (2 : ℤ) ^ G ≠ 0)
  have h1 := Int.emod_lt_of_pos (anchorInt k seeds m) (by positivity : (0 : ℤ) < 2 ^ G)
  have h53 : |anchorInt k seeds m % 2 ^ G| ≤ 2 ^ 53 := by
    rw [abs_of_nonneg h0]
    exact h1.le.trans (pow_le_pow_right₀ (by norm_num) hG)
  rw [anchorFraction, toInt_anchorNum (by omega), hfl.eq_self (isF64_intCast h53), anchorFrac,
    zpow_neg, zpow_natCast, div_eq_mul_inv]

/-- `fraction_steps` is exact: the numerators lie in `[0, 2^G)`, so `np.diff` is exact for
`k + G ≤ 64`, and `>>` divides by `2^G`. -/
lemma toInt_anchorSteps {k G : ℕ} (hG : G ≤ 53) (hkG : k + G ≤ 64) (seeds : ℕ → UInt64)
    (n : ℕ) : (anchorSteps k G seeds n).toInt = fracSteps k G (anchorInt k seeds) n := by
  have hnum : (fun m => (anchorNum k G seeds m).toInt) = fun m => anchorInt k seeds m % 2 ^ G :=
    funext fun m => toInt_anchorNum (by omega) k seeds m
  rw [anchorSteps, toInt_shiftRight_ofNat _ (by omega), fracSteps,
    toInt_diffI64_iter_of_mem_Ico hkG (fun m => ?_) n, hnum]
  rw [toInt_anchorNum (by omega)]
  exact ⟨Int.emod_nonneg _ (by positivity), Int.emod_lt_of_pos _ (by positivity)⟩

/-- "lgt": zero seeds give the zero anchor, which `_anchor_fraction` returns for "lgt". -/
lemma anchor_zero {fl : ℝ → ℝ} (hfl : IsRoundNearest fl) {k G : ℕ} (hG : G ≤ 53)
    (hkG : k + G ≤ 64) : anchorFraction fl k G 0 = 0 ∧ anchorSteps k G 0 = 0 := by
  have hP : anchorInt k 0 = 0 := by
    have h : (fun m => (((0 : ℕ → UInt64) m).toNat : ℤ)) = 0 := by
      funext m
      simp
    rw [anchorInt, h, integrate_zero]
  constructor
  · funext m
    rw [anchorFraction_eq hfl hG, anchorFrac, hP]
    simp
  · funext n
    apply Int64.toInt_inj.mp
    rw [toInt_anchorSteps hG hkG, fracSteps, hP]
    have h : (fun m => (0 : ℕ → ℤ) m % 2 ^ G) = 0 := by
      funext m
      simp
    rw [h, fwdDiff_iter_zero]
    simp

/-! ### `encode` -/

/-- Under `_levels`' guard `|p| < 2^b`, `b ≤ 53`, the int64 levels are the exact
`⌊s·x - F + 1/2⌋` (`levels_eq`), which lie in `[-2^b - 1, 2^b]` (`levels_mem`) and within
`2^53` (`abs_levels_le`). -/
lemma toInt_levelsI64 {fl : ℝ → ℝ} (hfl : IsRoundNearest fl) {ri : ℝ → ℤ} (hri : IsRoundInt ri)
    {k G b : ℕ} (hG : G ≤ 53) (hb : b ≤ 53) (seeds : ℕ → UInt64) {ms : ℝ} {es : ℤ}
    (hms : IsMantissa ms) {mx : ℕ → ℝ} {ex : ℕ → ℤ} {m : ℕ} (hmx : IsMantissa (mx m))
    (hguard : |levelHi fl ms es (mx m) (ex m)| < 2 ^ b) {s x : ℝ} (hs : s = ms * 2 ^ es)
    (hx : x = mx m * 2 ^ ex m) :
    (levelsI64 fl ri ms es mx ex (anchorFraction fl k G seeds) m).toInt =
        roundHalfUp (s * x - anchorFrac G (anchorInt k seeds) m) ∧
      (-2 ^ b - 1 ≤ roundHalfUp (s * x - anchorFrac G (anchorInt k seeds) m) ∧
        roundHalfUp (s * x - anchorFrac G (anchorInt k seeds) m) ≤ 2 ^ b) ∧
      |roundHalfUp (s * x - anchorFrac G (anchorInt k seeds) m)| ≤ 2 ^ 53 := by
  have h0 := Int.emod_nonneg (anchorInt k seeds m) (by positivity : (2 : ℤ) ^ G ≠ 0)
  have h1 := Int.emod_lt_of_pos (anchorInt k seeds m) (by positivity : (0 : ℤ) < 2 ^ G)
  have h53 : |levelHi fl ms es (mx m) (ex m)| < 2 ^ 53 :=
    hguard.trans_le (pow_le_pow_right₀ (by norm_num) hb)
  have heq := levels_eq hfl hri hms hmx es (ex m) hG h0 h1 h53
  have hmem := levels_mem hfl hri hms hmx es (ex m) hG h0 h1 hb hguard
  have habs := abs_levels_le hfl hri hms hmx es (ex m) hG h0 h1 h53
  rw [heq, ← hs, ← hx] at hmem habs
  refine ⟨?_, hmem, habs⟩
  rw [levelsI64, Int64.toInt_ofInt, anchorFraction_eq hfl hG, anchorFrac, heq, ← hs, ← hx]
  exact bmod_eq_of_inInt64 ⟨by rw [abs_le] at habs; omega, by rw [abs_le] at habs; omega⟩

lemma clipI64_eq {lo hi a : Int64} (h1 : lo.toInt ≤ a.toInt) (h2 : a.toInt ≤ hi.toInt) :
    clipI64 lo hi a = a := by
  unfold clipI64
  split_ifs with h3 h4
  · rw [Int64.lt_iff_toInt_lt] at h3
    omega
  · rw [Int64.lt_iff_toInt_lt] at h4
    omega
  · rfl

/-- Neither clip changes a token sum in `[1, V-1]`, nor does `_clip_tokens` count it. -/
lemma encodeTokens_eq {V : ℤ} (hV63 : V < 2 ^ 63) {k : ℕ} {L FS : ℕ → Int64} {n : ℕ}
    (h1 : 1 ≤ (tokenSum V k L FS n).toInt) (h2 : (tokenSum V k L FS n).toInt ≤ V - 1) :
    encodeTokens V k L FS n = tokenSum V k L FS n := by
  have hV : (Int64.ofInt V).toInt = V := by
    rw [Int64.toInt_ofInt]
    exact bmod_eq_of_inInt64 ⟨by omega, hV63⟩
  have hV1 : (Int64.ofInt (V - 1)).toInt = V - 1 := by
    rw [Int64.toInt_ofInt]
    exact bmod_eq_of_inInt64 ⟨by omega, by omega⟩
  have hm1 : (-1 : Int64).toInt = -1 := by decide
  have hin : clipI64 (-1) (Int64.ofInt V) (tokenSum V k L FS n) = tokenSum V k L FS n :=
    clipI64_eq (by rw [hm1]; omega) (by rw [hV]; omega)
  rw [encodeTokens, hin]
  exact clipI64_eq (by rw [Int64.toInt_zero]; omega) (by rw [hV1]; omega)

/-- On rows within `_levels`' float64 guard the int64 token sum is the integer `token` modulo
`2^64`, overflow or not. -/
lemma toInt_tokenSum_modEq {fl : ℝ → ℝ} (hfl : IsRoundNearest fl) {ri : ℝ → ℤ}
    (hri : IsRoundInt ri) (V : ℤ) {k G M : ℕ} (hG : G ≤ 53) (hkG : k + G ≤ 64)
    (seeds : ℕ → UInt64) {ms : ℝ} {es : ℤ} (hms : IsMantissa ms) {mx : ℕ → ℝ} {ex : ℕ → ℤ}
    (hmx : ∀ m < M, IsMantissa (mx m))
    (hguard : ∀ m < M, |levelHi fl ms es (mx m) (ex m)| < 2 ^ 53) {s : ℝ}
    (hs : s = ms * 2 ^ es) {x : ℕ → ℝ} (hx : ∀ m < M, x m = mx m * 2 ^ ex m) {n : ℕ}
    (hn : n + k < M) :
    (tokenSum V k (levelsI64 fl ri ms es mx ex (anchorFraction fl k G seeds))
        (anchorSteps k G seeds) n).toInt ≡
      token V k G (anchorInt k seeds) (fun m => s * x m - anchorFrac G (anchorInt k seeds) m) n
        [ZMOD 2 ^ 64] := by
  set L := levelsI64 fl ri ms es mx ex (anchorFraction fl k G seeds)
  have e1 : Δ_[1] ^[k] (fun m => (L m).toInt) n =
      Δ_[1] ^[k] (fun m => roundHalfUp (s * x m - anchorFrac G (anchorInt k seeds) m)) n :=
    fwdDiff_iter_congr fun i hi =>
      (toInt_levelsI64 hfl hri hG le_rfl seeds hms (hmx (n + i) (by omega))
        (hguard (n + i) (by omega)) hs (hx (n + i) (by omega))).1
  unfold tokenSum
  calc _ ≡ (diffI64^[k] L n + anchorSteps k G seeds n).toInt + (Int64.ofInt (V / 2)).toInt
        [ZMOD 2 ^ 64] := toInt_add_modEq _ _
    _ ≡ ((diffI64^[k] L n).toInt + (anchorSteps k G seeds n).toInt) + V / 2 [ZMOD 2 ^ 64] :=
        (toInt_add_modEq _ _).add (toInt_ofInt_modEq _)
    _ ≡ (Δ_[1] ^[k] (fun m => (L m).toInt) n + (anchorSteps k G seeds n).toInt) + V / 2
        [ZMOD 2 ^ 64] := ((toInt_diffI64_iter_modEq L k n).add_right _).add_right _
    _ = _ := by rw [e1, toInt_anchorSteps hG hkG, token]

/-- `encode`'s tokens on the rows `n + k < M` of a column within `_levels`' float64 guard
`|p| < 2^53`: where `|Δᵏ x n| ≤ D` and `s·D ≤ V/2 - 2^(k-1)`, the int64 token sum is the
integer `token`, which lies in `[1, V-1]`. -/
theorem encode_token_mem {fl : ℝ → ℝ} (hfl : IsRoundNearest fl) {ri : ℝ → ℤ}
    (hri : IsRoundInt ri) {V : ℤ} (hV : Even V) (hV63 : V < 2 ^ 63) {k G M : ℕ} (hG : G ≤ 53)
    (hkG : k + G ≤ 64) (seeds : ℕ → UInt64) {ms : ℝ} {es : ℤ} (hms : IsMantissa ms)
    {mx : ℕ → ℝ} {ex : ℕ → ℤ} (hmx : ∀ m < M, IsMantissa (mx m))
    (hguard : ∀ m < M, |levelHi fl ms es (mx m) (ex m)| < 2 ^ 53) {s D : ℝ}
    (hs : s = ms * 2 ^ es) (hs0 : 0 ≤ s) {x : ℕ → ℝ} (hx : ∀ m < M, x m = mx m * 2 ^ ex m)
    {n : ℕ} (hn : n + k < M) (hD : |Δ_[1] ^[k] x n| ≤ D)
    (hsD : s * D ≤ (V : ℝ) / 2 - 2 ^ (k - 1)) :
    (tokenSum V k (levelsI64 fl ri ms es mx ex (anchorFraction fl k G seeds))
        (anchorSteps k G seeds) n).toInt =
      token V k G (anchorInt k seeds) (fun m => s * x m - anchorFrac G (anchorInt k seeds) m) n ∧
    1 ≤ token V k G (anchorInt k seeds) (fun m => s * x m - anchorFrac G (anchorInt k seeds) m) n ∧
    token V k G (anchorInt k seeds) (fun m => s * x m - anchorFrac G (anchorInt k seeds) m) n ≤
      V - 1 := by
  have htok := token_mem_of_domain (G := G) hV (fwdDiff_iter_anchorInt k seeds) hs0 hD hsD
  refine ⟨eq_of_modEq_of_inInt64
    (toInt_tokenSum_modEq hfl hri V hG hkG seeds hms hmx hguard hs hx hn) (inInt64_toInt _)
    ⟨by omega, by omega⟩, htok⟩

/-- Under `_levels`' guard `|p| < 2^b` with `b + k ≤ 62` the int64 token sum is the integer one
on every row, in the domain or not, for `2^k < V ≤ 2^61`: the clip to `[-1, V]` puts a token
outside `[1, V-1]` on its own side. -/
theorem toInt_tokenSum {fl : ℝ → ℝ} (hfl : IsRoundNearest fl) {ri : ℝ → ℤ}
    (hri : IsRoundInt ri) {V : ℤ} {k G b M : ℕ} (hkV : 2 ^ k < V) (hV61 : V ≤ 2 ^ 61)
    (hG : G ≤ 53) (hkG : k + G ≤ 64) (hb : b ≤ 53) (hbk : b + k ≤ 62) (seeds : ℕ → UInt64)
    {ms : ℝ} {es : ℤ} (hms : IsMantissa ms) {mx : ℕ → ℝ} {ex : ℕ → ℤ}
    (hmx : ∀ m < M, IsMantissa (mx m))
    (hguard : ∀ m < M, |levelHi fl ms es (mx m) (ex m)| < 2 ^ b) {s : ℝ}
    (hs : s = ms * 2 ^ es) {x : ℕ → ℝ} (hx : ∀ m < M, x m = mx m * 2 ^ ex m) {n : ℕ}
    (hn : n + k < M) :
    (tokenSum V k (levelsI64 fl ri ms es mx ex (anchorFraction fl k G seeds))
        (anchorSteps k G seeds) n).toInt =
      token V k G (anchorInt k seeds) (fun m => s * x m - anchorFrac G (anchorInt k seeds) m) n := by
  have h53 : ∀ m < M, |levelHi fl ms es (mx m) (ex m)| < 2 ^ 53 := fun m hm =>
    (hguard m hm).trans_le (pow_le_pow_right₀ (by norm_num) hb)
  refine eq_of_modEq_of_inInt64
    (toInt_tokenSum_modEq hfl hri V hG hkG seeds hms hmx h53 hs hx hn) (inInt64_toInt _) ?_
  set P := anchorInt k seeds
  have hR : ∀ m < M, -2 ^ b - 1 ≤ roundHalfUp (s * x m - anchorFrac G P m) ∧
      roundHalfUp (s * x m - anchorFrac G P m) ≤ 2 ^ b := fun m hm =>
    (toInt_levelsI64 hfl hri hG hb seeds hms (hmx m hm) (hguard m hm) hs (hx m hm)).2.1
  have hb2 : (2 : ℤ) ^ b ≤ 2 ^ 53 := pow_le_pow_right₀ (by norm_num) hb
  have hV2 : 0 ≤ V / 2 ∧ V / 2 ≤ 2 ^ 60 := by
    have : (0 : ℤ) < 2 ^ k := by positivity
    omega
  rw [token]
  rcases Nat.eq_zero_or_pos k with rfl | hk
  · have hf : fracSteps 0 G P n = 0 := by
      rw [fracSteps, Function.iterate_zero, id_eq]
      exact Int.ediv_eq_zero_of_lt (Int.emod_nonneg _ (by positivity))
        (Int.emod_lt_of_pos _ (by positivity))
    obtain ⟨h1, h2⟩ := hR n (by omega)
    simp only [Function.iterate_zero, id_eq, hf]
    exact ⟨by linarith, by linarith⟩
  -- `|Δᵏ R| ≤ 2^(k+b) + 2^(k-1)`, `|fracSteps| < 2^(k-1)` and `2·2^(k-1) < V ≤ 2^61`
  have hf := abs_fracSteps_lt (G := G) hk (fwdDiff_iter_anchorInt k seeds) n
  have hd := abs_fwdDiff_iter_le_of_mem_Icc_of_lt hR hk hn
  have h1 : (2 : ℤ) ^ (k - 1) * 2 ^ b ≤ 2 ^ 61 := by
    rw [← pow_add]
    exact pow_le_pow_right₀ (by norm_num) (by omega)
  have h2 : (2 : ℤ) ^ k = 2 * 2 ^ (k - 1) := by
    rw [← pow_succ']
    congr 1
    omega
  have h3 : (2 : ℤ) ^ (k - 1) * (2 ^ b - (-2 ^ b - 1)) =
      2 * (2 ^ (k - 1) * 2 ^ b) + 2 ^ (k - 1) := by ring
  obtain ⟨b1, b2⟩ := abs_le.mp hd
  obtain ⟨c1, c2⟩ := abs_lt.mp hf
  exact ⟨by linarith, by linarith⟩

/-! ### `decode` -/

/-- `decode` of `encode`'s tokens, for a column whose `k + N` rows lie within `_levels`' guard
`|p| < 2^b`, `b ≤ 53`, `b + k ≤ 62`, and whose `N` steps lie within the domain:
`_integrate_levels` returns the encoder's levels `⌊s·x - F + 1/2⌋` and does not raise. -/
theorem decode_levels_eq {fl : ℝ → ℝ} (hfl : IsRoundNearest fl) {ri : ℝ → ℤ}
    (hri : IsRoundInt ri) {V : ℤ} (hV : Even V) (hV63 : V < 2 ^ 63) {k G b N : ℕ}
    (hG : G ≤ 53) (hkG : k + G ≤ 64) (hb : b ≤ 53) (hbk : b + k ≤ 62) (seeds : ℕ → UInt64)
    {ms : ℝ} {es : ℤ} (hms : IsMantissa ms) {mx : ℕ → ℝ} {ex : ℕ → ℤ}
    (hmx : ∀ m < k + N, IsMantissa (mx m))
    (hguard : ∀ m < k + N, |levelHi fl ms es (mx m) (ex m)| < 2 ^ b) {s D : ℝ}
    (hs : s = ms * 2 ^ es) (hs0 : 0 ≤ s) {x : ℕ → ℝ} (hx : ∀ m < k + N, x m = mx m * 2 ^ ex m)
    (hD : ∀ n < N, |Δ_[1] ^[k] x n| ≤ D) (hsD : s * D ≤ (V : ℝ) / 2 - 2 ^ (k - 1))
    {L FS q : ℕ → Int64} (hL : L = levelsI64 fl ri ms es mx ex (anchorFraction fl k G seeds))
    (hFS : FS = anchorSteps k G seeds) (hq : q = encodeTokens V k L FS) :
    (∀ m < k + N, (decodeLevels V k L FS q m).toInt =
      roundHalfUp (s * x m - anchorFrac G (anchorInt k seeds) m)) ∧
      ¬IntegrateRaises k L (decodeSteps V FS q) (k + N) := by
  subst hL hFS hq
  set P := anchorInt k seeds
  set L := levelsI64 fl ri ms es mx ex (anchorFraction fl k G seeds)
  set FS := anchorSteps k G seeds
  set R : ℕ → ℤ := fun m => roundHalfUp (s * x m - anchorFrac G P m)
  have h53 : ∀ m < k + N, |levelHi fl ms es (mx m) (ex m)| < 2 ^ 53 := fun m hm =>
    (hguard m hm).trans_le (pow_le_pow_right₀ (by norm_num) hb)
  have hLR : ∀ m < k + N, (L m).toInt = R m := fun m hm =>
    (toInt_levelsI64 hfl hri hG hb seeds hms (hmx m hm) (hguard m hm) hs (hx m hm)).1
  have hRmem : ∀ m < k + N, -2 ^ b - 1 ≤ R m ∧ R m ≤ 2 ^ b := fun m hm =>
    (toInt_levelsI64 hfl hri hG hb seeds hms (hmx m hm) (hguard m hm) hs (hx m hm)).2.1
  -- the tokens are the integer ones
  have htok : ∀ n < N,
      (encodeTokens V k L FS n).toInt = Δ_[1] ^[k] R n + fracSteps k G P n + V / 2 :=
    fun n hn => by
      obtain ⟨h1, h2, h3⟩ := encode_token_mem hfl hri hV hV63 hG hkG seeds hms hmx h53 hs hs0 hx
        (by omega : n + k < k + N) (hD n hn) hsD
      rw [encodeTokens_eq hV63 (by rw [h1]; exact h2) (by rw [h1]; exact h3), h1, token]
  -- so the steps are the `k`-th differences of the levels
  have hstep : ∀ n < N, (decodeSteps V FS (encodeTokens V k L FS) n).toInt = Δ_[1] ^[k] R n :=
    fun n hn => by
      refine eq_of_modEq_of_inInt64 ?_ (inInt64_toInt _)
        (inInt64_fwdDiff_iter_of_mem_Icc hbk hRmem le_rfl (by omega))
      unfold decodeSteps
      calc _ ≡ (encodeTokens V k L FS n - Int64.ofInt (V / 2)).toInt - (FS n).toInt
            [ZMOD 2 ^ 64] := toInt_sub_modEq _ _
        _ ≡ ((encodeTokens V k L FS n).toInt - V / 2) - (FS n).toInt [ZMOD 2 ^ 64] :=
            ((toInt_sub_modEq _ _).trans ((Int.ModEq.refl _).sub (toInt_ofInt_modEq _))).sub_right _
        _ = Δ_[1] ^[k] R n := by rw [htok n hn, toInt_anchorSteps hG hkG]; ring
  -- and their integration in integers is the levels
  have hI : ∀ m < k + N, integrate k (fun m => (L m).toInt)
      (fun n => (decodeSteps V FS (encodeTokens V k L FS) n).toInt) m = R m := fun m hm =>
    eq_of_fwdDiff_iter_eq_of_lt (fun n hn => by
        rw [fwdDiff_iter_integrate]
        exact hstep n (by omega))
      (fun m' hm' => by
        rw [integrate_of_lt hm']
        exact hLR m' (by omega)) hm
  obtain ⟨hval, hnr⟩ := toInt_integrateI64 (k := k) (M := k + N) (by omega) (pre := L)
    (d := decodeSteps V FS (encodeTokens V k L FS)) fun j hj n hn => by
      rw [fwdDiff_iter_congr (g := R) fun i hi => hI (n + i) (by omega)]
      exact inInt64_fwdDiff_iter_of_mem_Icc hbk hRmem hj hn
  exact ⟨fun m hm => (hval m hm).trans (hI m hm), hnr⟩

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

/-- `abs_decode_sub_le_f64` in closed form: with `u = 2^-53`, `|R + F| ≤ s·|x| + 1/2` gives
`1/(2s) + u·(2 + u)·(|x| + 1/(2s)) + 2^-1075·(1 + (1 + u)/s)`. -/
theorem abs_decode_sub_le_closed {fl : ℝ → ℝ} (hfl : IsRoundNearest fl) {s : ℝ} (hs : 0 < s)
    (x F : ℝ) (hR : |roundHalfUp (s * x - F)| ≤ 2 ^ 53) :
    |fl (fl (fl (roundHalfUp (s * x - F)) + F) / s) - x| ≤
      1 / (2 * s) + 2 ^ (-52 : ℤ) * (1 + 2 ^ (-54 : ℤ)) * (|x| + 1 / (2 * s)) +
        2 ^ (-1075 : ℤ) * (1 + (1 + 2 ^ (-53 : ℤ)) / s) := by
  have h := abs_decode_sub_le_f64 hfl hs x F hR
  have hu : (2 : ℝ) ^ (-52 : ℤ) * (1 + 2 ^ (-54 : ℤ)) = 2 ^ (-53 : ℤ) * (2 + 2 ^ (-53 : ℤ)) := by
    norm_num
  rw [hu]
  set R : ℝ := (roundHalfUp (s * x - F) : ℝ)
  set u : ℝ := 2 ^ (-53 : ℤ)
  set η : ℝ := 2 ^ (-1075 : ℤ)
  have hu0 : 0 < u := by positivity
  obtain ⟨h3, h4⟩ := roundHalfUp_sub_mem (s * x - F)
  have hRF : |R + F| ≤ s * |x| + 1 / 2 := by
    have e : R + F = s * x + (R - (s * x - F)) := by ring
    rw [e]
    calc |s * x + (R - (s * x - F))| ≤ |s * x| + |R - (s * x - F)| := abs_add_le _ _
      _ ≤ s * |x| + 1 / 2 := by
        rw [abs_mul, abs_of_pos hs]
        gcongr
        exact abs_le.mpr ⟨by linarith, h4⟩
  set y := fl (R + F)
  have hy : |y| ≤ (1 + u) * (s * |x| + 1 / 2) + η := by
    have h5 : |y - (R + F)| ≤ u * |R + F| + η := hfl.abs_sub_le (R + F)
    calc |y| = |(R + F) + (y - (R + F))| := by ring_nf
      _ ≤ |R + F| + |y - (R + F)| := abs_add_le _ _
      _ ≤ |R + F| + (u * |R + F| + η) := add_le_add le_rfl h5
      _ ≤ s * |x| + 1 / 2 + (u * (s * |x| + 1 / 2) + η) :=
          add_le_add hRF (add_le_add (mul_le_mul_of_nonneg_left hRF hu0.le) le_rfl)
      _ = (1 + u) * (s * |x| + 1 / 2) + η := by ring
  have hys : |y / s| ≤ (1 + u) * (|x| + 1 / (2 * s)) + η / s := by
    rw [abs_div, abs_of_pos hs, div_le_iff₀ hs]
    have e : ((1 + u) * (|x| + 1 / (2 * s)) + η / s) * s = (1 + u) * (s * |x| + 1 / 2) + η := by
      field_simp
    rw [e]
    exact hy
  unfold rndErr at h
  calc _ ≤ 1 / (2 * s) + (u * |R + F| + η) / s + (u * |y / s| + η) := h
    _ ≤ 1 / (2 * s) + (u * (s * |x| + 1 / 2) + η) / s +
        (u * ((1 + u) * (|x| + 1 / (2 * s)) + η / s) + η) := by gcongr
    _ = _ := by
      field_simp
      ring

/-! ### End to end -/

/-- **The tokenizer in float64 and int64.** Take a column of order `k`: its `k + N` rows
`x m = m_x m·2^(e_x m)` and its scale `s = m_s·2^e_s` as `np.frexp` splits them, within
`_levels`' guard `|p| < 2^b`, `b ≤ 53`, `b + k ≤ 62`. Take the anchor of any `seeds` with
`G ≤ 53`, `k + G ≤ 64` fraction bits (`b = min(53, 62 - k)` and `G = min(53, 64 - k)` meet
these), and steps `|Δᵏ x| ≤ D` with `s·D ≤ V/2 - 2^(k-1)`. Then:

* every token `encode` emits is its token sum and lies in `[1, V-1]`, so neither clip changes
  it and `_clip_tokens` counts none;
* `decode`'s `_integrate_levels` does not raise;
* every decoded value lies within `1/(2s) + 2^-52·(1 + 2^-54)·(|x| + 1/(2s)) +
  2^-1075·(1 + (1 + 2^-53)/s)` of `x`, and the prefix rows come back exactly. -/
theorem encode_decode {fl : ℝ → ℝ} (hfl : IsRoundNearest fl) {ri : ℝ → ℤ} (hri : IsRoundInt ri)
    {V : ℤ} (hV : Even V) (hV63 : V < 2 ^ 63) {k G b N : ℕ} (hG : G ≤ 53) (hkG : k + G ≤ 64)
    (hb : b ≤ 53) (hbk : b + k ≤ 62) (seeds : ℕ → UInt64) {ms : ℝ} {es : ℤ}
    (hms : IsMantissa ms) {mx : ℕ → ℝ} {ex : ℕ → ℤ} (hmx : ∀ m < k + N, IsMantissa (mx m))
    (hguard : ∀ m < k + N, |levelHi fl ms es (mx m) (ex m)| < 2 ^ b) {s D : ℝ}
    (hs : s = ms * 2 ^ es) (hs0 : 0 < s) {x : ℕ → ℝ} (hx : ∀ m < k + N, x m = mx m * 2 ^ ex m)
    (hD : ∀ n < N, |Δ_[1] ^[k] x n| ≤ D) (hsD : s * D ≤ (V : ℝ) / 2 - 2 ^ (k - 1))
    {F : ℕ → ℝ} (hF : F = anchorFraction fl k G seeds) {FS : ℕ → Int64}
    (hFS : FS = anchorSteps k G seeds) {L : ℕ → Int64} (hL : L = levelsI64 fl ri ms es mx ex F)
    {q : ℕ → Int64} (hq : q = encodeTokens V k L FS) :
    (∀ n < N, q n = tokenSum V k L FS n ∧ 1 ≤ (q n).toInt ∧ (q n).toInt ≤ V - 1) ∧
      ¬IntegrateRaises k L (decodeSteps V FS q) (k + N) ∧
      ∀ m < k + N, |decodeOut fl s k x F (decodeLevels V k L FS q) m - x m| ≤
        1 / (2 * s) + 2 ^ (-52 : ℤ) * (1 + 2 ^ (-54 : ℤ)) * (|x m| + 1 / (2 * s)) +
          2 ^ (-1075 : ℤ) * (1 + (1 + 2 ^ (-53 : ℤ)) / s) := by
  subst hF
  obtain ⟨hlev, hnr⟩ := decode_levels_eq hfl hri hV hV63 hG hkG hb hbk seeds hms hmx hguard hs
    hs0.le hx hD hsD hL hFS hq
  subst hL hFS hq
  have h53 : ∀ m < k + N, |levelHi fl ms es (mx m) (ex m)| < 2 ^ 53 := fun m hm =>
    (hguard m hm).trans_le (pow_le_pow_right₀ (by norm_num) hb)
  refine ⟨fun n hn => ?_, hnr, fun m hm => ?_⟩
  · obtain ⟨h1, h2, h3⟩ := encode_token_mem hfl hri hV hV63 hG hkG seeds hms hmx h53 hs hs0.le
      hx (by omega : n + k < k + N) (hD n hn) hsD
    rw [encodeTokens_eq hV63 (by rw [h1]; exact h2) (by rw [h1]; exact h3), h1]
    exact ⟨rfl, h2, h3⟩
  · unfold decodeOut
    split_ifs with hmk
    · rw [sub_self, abs_zero]
      positivity
    · rw [hlev m hm, anchorFraction_eq hfl hG]
      exact abs_decode_sub_le_closed hfl hs0 (x m) _
        (toInt_levelsI64 hfl hri hG hb seeds hms (hmx m hm) (hguard m hm) hs (hx m hm)).2.2

end DiffGPT
