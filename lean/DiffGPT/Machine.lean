module

public import Mathlib
public import DiffGPT.LevelGrid
public import DiffGPT.Float64
public import DiffGPT.ExactLevels
public import DiffGPT.Int64

/-!
# The tokenizer in float64 and int64

`encode` and `decode` as the code computes them, for one column `x` with the operator
`c(B) = (1 - B^S₁)⋯(1 - B^Sₘ)` over the lags `l` of degree `p = l.sum` and the scale `s`: its `p`
prefix rows and the `N` rows after them (index `m`), the token of row `n + p` at `n`.

    P = invert l seeds 0 mod 2^64             `_invert_difference` in uint64
    F m = (P m mod 2^G) / 2^G                 `anchorFraction`, exact
    FS n = c(B) (P mod 2^G) n >> G            `anchorSteps`, exact
    L m = ⌊s·x m - F m + 1/2⌋                 `levelsI64`: `_levels`, exact (`levels_eq`)
    q n = clip (c(B) L n + FS n + V/2)        `encodeTokens`, in int64
    L' = invert l L (q - V/2 - FS)            `decodeLevels`: `_invert_difference` in int64
    x̂ m = fl (fl (fl (L' m) + F m) / s)       `decodeOut`, `x m` on the prefix rows

The `seeds` are the uint64 values `_anchor_fraction` takes off the prefix: for "mash" the
fractional parts of its `s·x` in units of `2^-G`, rounded; "lgt" is `seeds = 0`, which gives
`F = FS = 0` (`anchor_zero`). `P` is the integer `anchorInt`, with `c(B) P = 0`, modulo `2^64`.
`decode` takes the anchor and the prefix levels off the same floats, `start`, so it gets the
same `F`, `FS` and `L m`, `m < p`.

* `token_eq`, `token_mem`, `token_mem_of_domain`: the integer token `c(B) L + FS + V/2` is
  `c(B)` of the levels `⌊s·x - B + 1/2⌋` measured from the anchor `B = P / 2^G`, offset by
  `V/2`, so the exact theory applies: it lies in `[1, V-1]` where `|c(B) x| ≤ D` and
  `s·D ≤ (V - ‖c‖₁)/2`, which the scale, rounded down, and the max of
  `get_domain_of_definition`, rounded up, give exactly.
* `encode_token_mem`: the int64 token sum is that token. The two agree modulo `2^64` and both
  lie in int64, so this needs only `_levels`' float64 guard `|p| < 2^53`. Neither clip then
  changes the sum (`encodeTokens_eq`).
* `toInt_tokenSum`: under the guard `|p| < 2^b` with `b + m ≤ 62`, which `b = min(53, 62 - m)`
  meets, the token sum is the integer one on every row, in the domain or not, so the clip puts a
  token outside `[1, V-1]` on its own side.
* `decode_levels_eq`: `decode` gets the levels `L` back from `encode`'s tokens, and
  `_invert_difference` does not raise: its sums pass through the differences of `L` by the
  first factors, which int64 holds for `b + m ≤ 62` (`toInt_invertI64`).
* `abs_decode_sub_le_f64`, `abs_decode_sub_le_closed`: the decoded value is within `1/(2s)` of
  `x` up to the roundings of `L + F` and of the quotient, at most
  `2^-52·(1 + 2^-54)·(|x| + 1/(2s)) + 2^-1075·(1 + (1 + 2^-53)/s)` more, the level being a
  float (`abs_levels_le`).
* `encode_decode`: all of it for a column, from the floats to the decoded values.
* `encode_decode_zero`: the order `0`, `c = 1`, with the fraction `1/2` of both tokenizers:
  the tokens `⌊s·x⌋ + V/2` lie in `[0, V-1]` for `s·D < V/2` (`floor_mem_of_domain`), and the
  decoded `(⌊s·x⌋ + 1/2)/s` is within `1/(2s)` of `x` up to the same roundings.
-/

@[expose] public section

open fwdDiff

namespace DiffGPT

/-! ### The tokens in integers -/

/-- The anchor's fraction at row `m`, `(P m mod 2^G) / 2^G`, as `_anchor_fraction` returns it. -/
noncomputable def anchorFrac (G : ℕ) (P : ℕ → ℤ) (m : ℕ) : ℝ :=
  ((P m % 2 ^ G : ℤ) : ℝ) / 2 ^ G

/-- `_anchor_fraction`'s `fraction_steps`: `_difference(numerators, lags) >> G`. -/
def fracSteps (l : List ℕ) (G : ℕ) (P : ℕ → ℤ) (n : ℕ) : ℤ :=
  diffs l (fun m => P m % 2 ^ G) n / 2 ^ G

/-- The token `encode` emits for row `n + p`, before clipping:
`_difference(levels, lags) + fraction_steps + vocab_size // 2` with the levels `⌊w + 1/2⌋`,
which `_levels` returns for `w = s·x - F` (`levels_eq`). -/
noncomputable def token (V : ℤ) (l : List ℕ) (G : ℕ) (P : ℕ → ℤ) (w : ℕ → ℝ) (n : ℕ) : ℤ :=
  diffs l (fun m => roundHalfUp (w m)) n + fracSteps l G P n + V / 2

/-- With `c(B) P = 0` the anchor's term is exact: `fracSteps = -c(B) ⌊P / 2^G⌋`. -/
lemma fracSteps_eq {l : List ℕ} {G : ℕ} {P : ℕ → ℤ} (hP : diffs l P = 0) (n : ℕ) :
    fracSteps l G P n = -diffs l (fun m => P m / 2 ^ G) n := by
  rw [fracSteps, diffs_emod_eq hP G n, ← mul_neg,
    Int.mul_ediv_cancel_left _ (by positivity)]

/-- The anchor's term of the token is exact, `-c(B) ⌊B⌋`, and below `2^(m-1)`. -/
theorem abs_fracSteps_lt {l : List ℕ} (hl : l ≠ []) {G : ℕ} {P : ℕ → ℤ} (hP : diffs l P = 0)
    (n : ℕ) : |fracSteps l G P n| < 2 ^ (l.length - 1) := by
  have h := abs_diffs_emod_lt P hl G n
  have h2 : (0 : ℤ) < 2 ^ G := by positivity
  rw [diffs_emod_eq hP G n, abs_neg, abs_mul, abs_of_pos h2, pow_add,
    mul_comm (2 ^ (l.length - 1))] at h
  rw [fracSteps_eq hP, abs_neg]
  exact lt_of_mul_lt_mul_left h h2.le

/-- The tokens are `c(B)` of the levels of `w + F`, `s·x` for `w = s·x - F`, measured from the
anchor `B = P / 2^G`, offset by `V/2`. -/
theorem token_eq {l : List ℕ} {G : ℕ} {P : ℕ → ℤ} (hP : diffs l P = 0) (V : ℤ) (w : ℕ → ℝ)
    (n : ℕ) :
    token V l G P w n =
      diffs l (fun m => roundHalfUp (w m + anchorFrac G P m - (P m : ℝ) / 2 ^ G)) n + V / 2 := by
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
  rw [token, hR, diffs_add, diffs_const_smul, Pi.add_apply, Pi.smul_apply, smul_eq_mul,
    fracSteps_eq hP]
  ring

/-- The token at `n` lies in `[1, V-1]` when `c(B)` of `w + F` there stays within
`(V - ‖c‖₁)/2` (`abs_diffs_levels_le`). -/
theorem token_mem {V : ℤ} (hV : Even V) {l : List ℕ} (hl : ∀ S ∈ l, 0 < S) (hl0 : l ≠ [])
    {G : ℕ} {P : ℕ → ℤ} (hP : diffs l P = 0) {w : ℕ → ℝ} {n : ℕ}
    (hw : |diffs l (fun m => w m + anchorFrac G P m) n| ≤ ((V : ℝ) - norm1 l) / 2) :
    1 ≤ token V l G P w n ∧ token V l G P w n ≤ V - 1 := by
  have hA : diffs l (fun m => (P m : ℝ) / 2 ^ G) = 0 := by
    have h1 : (fun m => (P m : ℝ) / 2 ^ G) = ((2 : ℝ) ^ G)⁻¹ • fun m => (P m : ℝ) := by
      funext m
      rw [Pi.smul_apply, smul_eq_mul, div_eq_inv_mul]
    funext m
    rw [h1, diffs_const_smul, Pi.smul_apply, ← cast_diffs, hP]
    simp only [Pi.zero_apply, Int.cast_zero, smul_zero]
  have h := abs_diffs_levels_le hV hl hl0 zero_le_one hA hw (by rw [one_mul])
  simp only [one_mul] at h
  rw [token_eq hP]
  obtain ⟨r, rfl⟩ := hV
  rw [abs_le] at h
  constructor <;> omega

/-- The token range: the token of the levels of `s·x - F` at `n` lies in `[1, V-1]` when
`|c(B) x n| ≤ D` and `s·D ≤ (V - ‖c‖₁)/2`. -/
theorem token_mem_of_domain {V : ℤ} (hV : Even V) {l : List ℕ} (hl : ∀ S ∈ l, 0 < S)
    (hl0 : l ≠ []) {G : ℕ} {P : ℕ → ℤ} (hP : diffs l P = 0) {s D : ℝ} (hs : 0 ≤ s)
    {x : ℕ → ℝ} {n : ℕ} (hD : |diffs l x n| ≤ D) (hsD : s * D ≤ ((V : ℝ) - norm1 l) / 2) :
    1 ≤ token V l G P (fun m => s * x m - anchorFrac G P m) n ∧
      token V l G P (fun m => s * x m - anchorFrac G P m) n ≤ V - 1 := by
  refine token_mem hV hl hl0 hP ?_
  have h : (fun m => s * x m - anchorFrac G P m + anchorFrac G P m) = s • x := by
    funext m
    simp only [Pi.smul_apply, smul_eq_mul]
    ring
  rw [h, diffs_const_smul, Pi.smul_apply, smul_eq_mul, abs_mul, abs_of_nonneg hs]
  exact (mul_le_mul_of_nonneg_left hD hs).trans hsD

/-! ### The code -/

open scoped UInt64.CommRing in
/-- `_anchor_fraction`'s `numerators`: `_invert_difference` of the uint64 `seeds` with zero
steps, which wraps modulo `2^64`, masked with `2^G - 1` and cast to int64. -/
noncomputable def anchorNum (l : List ℕ) (G : ℕ) (seeds : ℕ → UInt64) (m : ℕ) : Int64 :=
  ((invert l seeds 0 m) &&& UInt64.ofNat (2 ^ G - 1)).toInt64

/-- The integer anchor `P`, which `anchorNum` reduces modulo `2^G`: `invert l seeds 0` over the
integers. -/
def anchorInt (l : List ℕ) (seeds : ℕ → UInt64) : ℕ → ℤ :=
  invert l (fun m => ((seeds m).toNat : ℤ)) 0

/-- `_anchor_fraction`'s `fraction`: `np.ldexp(numerators.astype(np.float64), -G)`. -/
noncomputable def anchorFraction (fl : ℝ → ℝ) (l : List ℕ) (G : ℕ) (seeds : ℕ → UInt64)
    (m : ℕ) : ℝ :=
  fl ((anchorNum l G seeds m).toInt : ℝ) * 2 ^ (-(G : ℤ))

/-- `_anchor_fraction`'s `fraction_steps`: `_difference(numerators, lags) >> G` in int64. -/
noncomputable def anchorSteps (l : List ℕ) (G : ℕ) (seeds : ℕ → UInt64) (n : ℕ) : Int64 :=
  diffsI64 l (anchorNum l G seeds) n >>> Int64.ofNat G

/-- `_levels`' int64 levels of the rows `x m = m_x m·2^(e_x m)` with the fractions `F`. -/
noncomputable def levelsI64 (fl : ℝ → ℝ) (ri : ℝ → ℤ) (ms : ℝ) (es : ℤ) (mx : ℕ → ℝ)
    (ex : ℕ → ℤ) (F : ℕ → ℝ) (m : ℕ) : Int64 :=
  Int64.ofInt (levels fl ri ms es (mx m) (ex m) (F m))

/-- `np.clip(a, lo, hi)` in int64. -/
def clipI64 (lo hi a : Int64) : Int64 := if a < lo then lo else if hi < a then hi else a

/-- `encode`'s token sums: `_difference(levels, lags) + fraction_steps + vocab_size // 2` in
int64. -/
def tokenSum (V : ℤ) (l : List ℕ) (L FS : ℕ → Int64) (n : ℕ) : Int64 :=
  diffsI64 l L n + FS n + Int64.ofInt (V / 2)

/-- `encode`'s tokens: the sums clipped to `[-1, V]`, then by `_clip_tokens` to `[0, V-1]`. -/
def encodeTokens (V : ℤ) (l : List ℕ) (L FS : ℕ → Int64) (n : ℕ) : Int64 :=
  clipI64 0 (Int64.ofInt (V - 1)) (clipI64 (-1) (Int64.ofInt V) (tokenSum V l L FS n))

/-- `decode`'s steps: `tokens - vocab_size // 2 - fraction_steps` in int64. -/
def decodeSteps (V : ℤ) (FS q : ℕ → Int64) (n : ℕ) : Int64 :=
  q n - Int64.ofInt (V / 2) - FS n

/-- `decode`'s levels: `_invert_difference` of the prefix levels and the steps, in int64. -/
def decodeLevels (V : ℤ) (l : List ℕ) (L FS q : ℕ → Int64) : ℕ → Int64 :=
  invertI64 l L (decodeSteps V FS q)

/-- `decode`'s output: `start` on the `p` prefix rows, `(levels + fraction) / scale` in float64
after them. -/
noncomputable def decodeOut (fl : ℝ → ℝ) (s : ℝ) (p : ℕ) (x F : ℕ → ℝ) (L : ℕ → Int64)
    (m : ℕ) : ℝ :=
  if m < p then x m else fl (fl (fl ((L m).toInt : ℝ) + F m) / s)

/-! ### The anchor -/

lemma diffs_anchorInt {l : List ℕ} (hl : ∀ S ∈ l, 0 < S) (seeds : ℕ → UInt64) :
    diffs l (anchorInt l seeds) = 0 :=
  diffs_invert hl _ 0

open scoped UInt64.CommRing in
/-- The numerators are the integer anchor modulo `2^G` (`anchor_numerators`). -/
lemma toInt_anchorNum {G : ℕ} (hG : G ≤ 63) (l : List ℕ) (seeds : ℕ → UInt64) (m : ℕ) :
    (anchorNum l G seeds m).toInt = anchorInt l seeds m % 2 ^ G := by
  have h := anchor_numerators (by omega : G ≤ 64) l seeds m
  have hu : ((invert l seeds 0 m &&& UInt64.ofNat (2 ^ G - 1)).toNat : ℤ) < 2 ^ 63 := by
    rw [h]
    exact (Int.emod_lt_of_pos _ (by positivity)).trans_le (pow_le_pow_right₀ (by norm_num) hG)
  rw [anchorNum, toInt_toInt64_of_lt _ (by exact_mod_cast hu), h]
  rfl

lemma anchorFraction_eq {fl : ℝ → ℝ} (hfl : IsRoundNearest fl) {G : ℕ} (hG : G ≤ 53)
    (l : List ℕ) (seeds : ℕ → UInt64) (m : ℕ) :
    anchorFraction fl l G seeds m = anchorFrac G (anchorInt l seeds) m := by
  have h0 := Int.emod_nonneg (anchorInt l seeds m) (by positivity : (2 : ℤ) ^ G ≠ 0)
  have h1 := Int.emod_lt_of_pos (anchorInt l seeds m) (by positivity : (0 : ℤ) < 2 ^ G)
  have h53 : |anchorInt l seeds m % 2 ^ G| ≤ 2 ^ 53 := by
    rw [abs_of_nonneg h0]
    exact h1.le.trans (pow_le_pow_right₀ (by norm_num) hG)
  rw [anchorFraction, toInt_anchorNum (by omega), hfl.eq_self (isF64_intCast h53), anchorFrac,
    zpow_neg, zpow_natCast, div_eq_mul_inv]

/-- `fraction_steps` is exact: the numerators lie in `[0, 2^G)`, so `_difference` is exact for
`m + G ≤ 64`, and `>>` divides by `2^G`. -/
lemma toInt_anchorSteps {l : List ℕ} {G : ℕ} (hG : G ≤ 53) (hlG : l.length + G ≤ 64)
    (seeds : ℕ → UInt64) (n : ℕ) :
    (anchorSteps l G seeds n).toInt = fracSteps l G (anchorInt l seeds) n := by
  have hnum : (fun m => (anchorNum l G seeds m).toInt) = fun m => anchorInt l seeds m % 2 ^ G :=
    funext fun m => toInt_anchorNum (by omega) l seeds m
  rw [anchorSteps, toInt_shiftRight_ofNat _ (by omega), fracSteps,
    toInt_diffsI64_of_mem_Ico hlG (fun m => ?_) n, hnum]
  rw [toInt_anchorNum (by omega)]
  exact ⟨Int.emod_nonneg _ (by positivity), Int.emod_lt_of_pos _ (by positivity)⟩

/-- "lgt": zero seeds give the zero anchor, which `_anchor_fraction` returns for "lgt". -/
lemma anchor_zero {fl : ℝ → ℝ} (hfl : IsRoundNearest fl) {l : List ℕ} {G : ℕ} (hG : G ≤ 53)
    (hlG : l.length + G ≤ 64) : anchorFraction fl l G 0 = 0 ∧ anchorSteps l G 0 = 0 := by
  have hP : anchorInt l 0 = 0 := by
    have h : (fun m => (((0 : ℕ → UInt64) m).toNat : ℤ)) = 0 := by
      funext m
      simp
    rw [anchorInt, h, invert_zero]
  constructor
  · funext m
    rw [anchorFraction_eq hfl hG, anchorFrac, hP]
    simp
  · funext n
    apply Int64.toInt_inj.mp
    rw [toInt_anchorSteps hG hlG, fracSteps, hP]
    have h : (fun m => (0 : ℕ → ℤ) m % 2 ^ G) = 0 := by
      funext m
      simp
    rw [h, diffs_zero]
    simp

/-! ### `encode` -/

/-- Under `_levels`' guard `|p| < 2^b`, `b ≤ 53`, the int64 levels of a row with the fraction
`F m = N / 2^G`, `0 ≤ N < 2^G`, are the exact `⌊s·x - F m + 1/2⌋` (`levels_eq`), which lie in
`[-2^b - 1, 2^b]` (`levels_mem`) and within `2^53` (`abs_levels_le`). -/
lemma toInt_levelsI64 {fl : ℝ → ℝ} (hfl : IsRoundNearest fl) {ri : ℝ → ℤ} (hri : IsRoundInt ri)
    {G b : ℕ} (hG : G ≤ 53) (hb : b ≤ 53) {F : ℕ → ℝ} {N : ℤ} (hN0 : 0 ≤ N) (hN : N < 2 ^ G)
    {ms : ℝ} {es : ℤ} (hms : IsMantissa ms) {mx : ℕ → ℝ} {ex : ℕ → ℤ} {m : ℕ}
    (hF : F m = N / 2 ^ G) (hmx : IsMantissa (mx m))
    (hguard : |levelHi fl ms es (mx m) (ex m)| < 2 ^ b) {s x : ℝ} (hs : s = ms * 2 ^ es)
    (hx : x = mx m * 2 ^ ex m) :
    (levelsI64 fl ri ms es mx ex F m).toInt = roundHalfUp (s * x - F m) ∧
      (-2 ^ b - 1 ≤ roundHalfUp (s * x - F m) ∧ roundHalfUp (s * x - F m) ≤ 2 ^ b) ∧
      |roundHalfUp (s * x - F m)| ≤ 2 ^ 53 := by
  have h53 : |levelHi fl ms es (mx m) (ex m)| < 2 ^ 53 :=
    hguard.trans_le (pow_le_pow_right₀ (by norm_num) hb)
  have heq := levels_eq hfl hri hms hmx es (ex m) hG hN0 hN h53
  have hmem := levels_mem hfl hri hms hmx es (ex m) hG hN0 hN hb hguard
  have habs := abs_levels_le hfl hri hms hmx es (ex m) hG hN0 hN h53
  rw [heq, ← hs, ← hx, ← hF] at hmem habs
  refine ⟨?_, hmem, habs⟩
  rw [levelsI64, Int64.toInt_ofInt, hF, heq, ← hs, ← hx, ← hF]
  exact bmod_eq_of_inInt64 ⟨by rw [abs_le] at habs; omega, by rw [abs_le] at habs; omega⟩

/-- `toInt_levelsI64` for the anchor of the seeds. -/
lemma toInt_levelsI64_anchor {fl : ℝ → ℝ} (hfl : IsRoundNearest fl) {ri : ℝ → ℤ}
    (hri : IsRoundInt ri) {l : List ℕ} {G b : ℕ} (hG : G ≤ 53) (hb : b ≤ 53)
    (seeds : ℕ → UInt64) {ms : ℝ} {es : ℤ} (hms : IsMantissa ms) {mx : ℕ → ℝ} {ex : ℕ → ℤ}
    {m : ℕ} (hmx : IsMantissa (mx m)) (hguard : |levelHi fl ms es (mx m) (ex m)| < 2 ^ b)
    {s x : ℝ} (hs : s = ms * 2 ^ es) (hx : x = mx m * 2 ^ ex m) :
    (levelsI64 fl ri ms es mx ex (anchorFraction fl l G seeds) m).toInt =
        roundHalfUp (s * x - anchorFrac G (anchorInt l seeds) m) ∧
      (-2 ^ b - 1 ≤ roundHalfUp (s * x - anchorFrac G (anchorInt l seeds) m) ∧
        roundHalfUp (s * x - anchorFrac G (anchorInt l seeds) m) ≤ 2 ^ b) ∧
      |roundHalfUp (s * x - anchorFrac G (anchorInt l seeds) m)| ≤ 2 ^ 53 := by
  have h := toInt_levelsI64 hfl hri hG hb
    (Int.emod_nonneg (anchorInt l seeds m) (by positivity : (2 : ℤ) ^ G ≠ 0))
    (Int.emod_lt_of_pos (anchorInt l seeds m) (by positivity : (0 : ℤ) < 2 ^ G)) hms
    (anchorFraction_eq hfl hG l seeds m) hmx hguard hs hx
  rwa [anchorFraction_eq hfl hG l seeds m] at h

lemma clipI64_eq {lo hi a : Int64} (h1 : lo.toInt ≤ a.toInt) (h2 : a.toInt ≤ hi.toInt) :
    clipI64 lo hi a = a := by
  unfold clipI64
  split_ifs with h3 h4
  · rw [Int64.lt_iff_toInt_lt] at h3
    omega
  · rw [Int64.lt_iff_toInt_lt] at h4
    omega
  · rfl

/-- Neither clip changes a token sum in `[0, V-1]`, nor does `_clip_tokens` count it. -/
lemma encodeTokens_eq {V : ℤ} (hV63 : V < 2 ^ 63) {l : List ℕ} {L FS : ℕ → Int64} {n : ℕ}
    (h1 : 0 ≤ (tokenSum V l L FS n).toInt) (h2 : (tokenSum V l L FS n).toInt ≤ V - 1) :
    encodeTokens V l L FS n = tokenSum V l L FS n := by
  have hV : (Int64.ofInt V).toInt = V := by
    rw [Int64.toInt_ofInt]
    exact bmod_eq_of_inInt64 ⟨by omega, hV63⟩
  have hV1 : (Int64.ofInt (V - 1)).toInt = V - 1 := by
    rw [Int64.toInt_ofInt]
    exact bmod_eq_of_inInt64 ⟨by omega, by omega⟩
  have hm1 : (-1 : Int64).toInt = -1 := by decide
  have hin : clipI64 (-1) (Int64.ofInt V) (tokenSum V l L FS n) = tokenSum V l L FS n :=
    clipI64_eq (by rw [hm1]; omega) (by rw [hV]; omega)
  rw [encodeTokens, hin]
  exact clipI64_eq (by rw [Int64.toInt_zero]; omega) (by rw [hV1]; omega)

/-- On rows within `_levels`' float64 guard the int64 token sum is the integer `token` modulo
`2^64`, overflow or not. -/
lemma toInt_tokenSum_modEq {fl : ℝ → ℝ} (hfl : IsRoundNearest fl) {ri : ℝ → ℤ}
    (hri : IsRoundInt ri) (V : ℤ) {l : List ℕ} {G M : ℕ} (hG : G ≤ 53) (hlG : l.length + G ≤ 64)
    (seeds : ℕ → UInt64) {ms : ℝ} {es : ℤ} (hms : IsMantissa ms) {mx : ℕ → ℝ} {ex : ℕ → ℤ}
    (hmx : ∀ m < M, IsMantissa (mx m))
    (hguard : ∀ m < M, |levelHi fl ms es (mx m) (ex m)| < 2 ^ 53) {s : ℝ}
    (hs : s = ms * 2 ^ es) {x : ℕ → ℝ} (hx : ∀ m < M, x m = mx m * 2 ^ ex m) {n : ℕ}
    (hn : n + l.sum < M) :
    (tokenSum V l (levelsI64 fl ri ms es mx ex (anchorFraction fl l G seeds))
        (anchorSteps l G seeds) n).toInt ≡
      token V l G (anchorInt l seeds) (fun m => s * x m - anchorFrac G (anchorInt l seeds) m) n
        [ZMOD 2 ^ 64] := by
  set L := levelsI64 fl ri ms es mx ex (anchorFraction fl l G seeds)
  have e1 : diffs l (fun m => (L m).toInt) n =
      diffs l (fun m => roundHalfUp (s * x m - anchorFrac G (anchorInt l seeds) m)) n :=
    diffs_congr fun i hi =>
      (toInt_levelsI64_anchor hfl hri hG le_rfl seeds hms (hmx (n + i) (by omega))
        (hguard (n + i) (by omega)) hs (hx (n + i) (by omega))).1
  unfold tokenSum
  calc _ ≡ (diffsI64 l L n + anchorSteps l G seeds n).toInt + (Int64.ofInt (V / 2)).toInt
        [ZMOD 2 ^ 64] := toInt_add_modEq _ _
    _ ≡ ((diffsI64 l L n).toInt + (anchorSteps l G seeds n).toInt) + V / 2 [ZMOD 2 ^ 64] :=
        (toInt_add_modEq _ _).add (toInt_ofInt_modEq _)
    _ ≡ (diffs l (fun m => (L m).toInt) n + (anchorSteps l G seeds n).toInt) + V / 2
        [ZMOD 2 ^ 64] := ((toInt_diffsI64_modEq l L n).add_right _).add_right _
    _ = _ := by rw [e1, toInt_anchorSteps hG hlG, token]

/-- `encode`'s tokens on the rows `n + p < M` of a column within `_levels`' float64 guard
`|p| < 2^53`: where `|c(B) x n| ≤ D` and `s·D ≤ (V - ‖c‖₁)/2`, the int64 token sum is the
integer `token`, which lies in `[1, V-1]`. -/
theorem encode_token_mem {fl : ℝ → ℝ} (hfl : IsRoundNearest fl) {ri : ℝ → ℤ}
    (hri : IsRoundInt ri) {V : ℤ} (hV : Even V) (hV63 : V < 2 ^ 63) {l : List ℕ}
    (hl : ∀ S ∈ l, 0 < S) (hl0 : l ≠ []) {G M : ℕ} (hG : G ≤ 53) (hlG : l.length + G ≤ 64)
    (seeds : ℕ → UInt64) {ms : ℝ} {es : ℤ} (hms : IsMantissa ms) {mx : ℕ → ℝ} {ex : ℕ → ℤ}
    (hmx : ∀ m < M, IsMantissa (mx m))
    (hguard : ∀ m < M, |levelHi fl ms es (mx m) (ex m)| < 2 ^ 53) {s D : ℝ}
    (hs : s = ms * 2 ^ es) (hs0 : 0 ≤ s) {x : ℕ → ℝ} (hx : ∀ m < M, x m = mx m * 2 ^ ex m)
    {n : ℕ} (hn : n + l.sum < M) (hD : |diffs l x n| ≤ D)
    (hsD : s * D ≤ ((V : ℝ) - norm1 l) / 2) :
    (tokenSum V l (levelsI64 fl ri ms es mx ex (anchorFraction fl l G seeds))
        (anchorSteps l G seeds) n).toInt =
      token V l G (anchorInt l seeds) (fun m => s * x m - anchorFrac G (anchorInt l seeds) m) n ∧
    1 ≤ token V l G (anchorInt l seeds) (fun m => s * x m - anchorFrac G (anchorInt l seeds) m) n ∧
    token V l G (anchorInt l seeds) (fun m => s * x m - anchorFrac G (anchorInt l seeds) m) n ≤
      V - 1 := by
  have htok := token_mem_of_domain (G := G) hV hl hl0 (diffs_anchorInt hl seeds) hs0 hD hsD
  refine ⟨eq_of_modEq_of_inInt64
    (toInt_tokenSum_modEq hfl hri V hG hlG seeds hms hmx hguard hs hx hn) (inInt64_toInt _)
    ⟨by omega, by omega⟩, htok⟩

/-- Under `_levels`' guard `|p| < 2^b` with `b + m ≤ 62` for `m ≤ 61` factors the int64 token
sum is the integer one on every row, in the domain or not, for `0 ≤ V ≤ 2^61`: the clip to
`[-1, V]` puts a token outside `[1, V-1]` on its own side. -/
theorem toInt_tokenSum {fl : ℝ → ℝ} (hfl : IsRoundNearest fl) {ri : ℝ → ℤ}
    (hri : IsRoundInt ri) {V : ℤ} (hV0 : 0 ≤ V) (hV61 : V ≤ 2 ^ 61) {l : List ℕ}
    (hl : ∀ S ∈ l, 0 < S) (hl0 : l ≠ []) (hl61 : l.length ≤ 61) {G b M : ℕ} (hG : G ≤ 53)
    (hlG : l.length + G ≤ 64) (hb : b ≤ 53) (hbl : b + l.length ≤ 62) (seeds : ℕ → UInt64)
    {ms : ℝ} {es : ℤ} (hms : IsMantissa ms) {mx : ℕ → ℝ} {ex : ℕ → ℤ}
    (hmx : ∀ m < M, IsMantissa (mx m))
    (hguard : ∀ m < M, |levelHi fl ms es (mx m) (ex m)| < 2 ^ b) {s : ℝ}
    (hs : s = ms * 2 ^ es) {x : ℕ → ℝ} (hx : ∀ m < M, x m = mx m * 2 ^ ex m) {n : ℕ}
    (hn : n + l.sum < M) :
    (tokenSum V l (levelsI64 fl ri ms es mx ex (anchorFraction fl l G seeds))
        (anchorSteps l G seeds) n).toInt =
      token V l G (anchorInt l seeds) (fun m => s * x m - anchorFrac G (anchorInt l seeds) m) n := by
  have h53 : ∀ m < M, |levelHi fl ms es (mx m) (ex m)| < 2 ^ 53 := fun m hm =>
    (hguard m hm).trans_le (pow_le_pow_right₀ (by norm_num) hb)
  refine eq_of_modEq_of_inInt64
    (toInt_tokenSum_modEq hfl hri V hG hlG seeds hms hmx h53 hs hx hn) (inInt64_toInt _) ?_
  set P := anchorInt l seeds
  have hR : ∀ m < M, -2 ^ b - 1 ≤ roundHalfUp (s * x m - anchorFrac G P m) ∧
      roundHalfUp (s * x m - anchorFrac G P m) ≤ 2 ^ b := fun m hm =>
    (toInt_levelsI64_anchor hfl hri hG hb seeds hms (hmx m hm) (hguard m hm) hs (hx m hm)).2.1
  have hV2 : 0 ≤ V / 2 ∧ V / 2 ≤ 2 ^ 60 := by omega
  rw [token]
  -- `|c(B) R| ≤ 2^(m+b) + 2^(m-1)`, `|fracSteps| < 2^(m-1)` and `2^(m-1), V/2 ≤ 2^60`
  have hf := abs_fracSteps_lt (G := G) hl0 (diffs_anchorInt hl seeds) n
  have hd := abs_diffs_le_of_mem_Icc_of_lt hR hl0 hn
  have h1 : (2 : ℤ) ^ (l.length - 1) * 2 ^ b ≤ 2 ^ 61 := by
    rw [← pow_add]
    exact pow_le_pow_right₀ (by norm_num) (by omega)
  have h2 : (2 : ℤ) ^ (l.length - 1) ≤ 2 ^ 60 := pow_le_pow_right₀ (by norm_num) (by omega)
  have h3 : (2 : ℤ) ^ (l.length - 1) * (2 ^ b - (-2 ^ b - 1)) =
      2 * (2 ^ (l.length - 1) * 2 ^ b) + 2 ^ (l.length - 1) := by ring
  obtain ⟨b1, b2⟩ := abs_le.mp hd
  obtain ⟨c1, c2⟩ := abs_lt.mp hf
  exact ⟨by linarith, by linarith⟩

/-! ### `decode` -/

/-- `decode` of `encode`'s tokens, for a column whose `p + N` rows lie within `_levels`' guard
`|p| < 2^b`, `b ≤ 53`, `b + m ≤ 62`, and whose `N` steps lie within the domain:
`_invert_difference` returns the encoder's levels `⌊s·x - F + 1/2⌋` and does not raise. -/
theorem decode_levels_eq {fl : ℝ → ℝ} (hfl : IsRoundNearest fl) {ri : ℝ → ℤ}
    (hri : IsRoundInt ri) {V : ℤ} (hV : Even V) (hV63 : V < 2 ^ 63) {l : List ℕ}
    (hl : ∀ S ∈ l, 0 < S) (hl0 : l ≠ []) {G b N : ℕ} (hG : G ≤ 53) (hlG : l.length + G ≤ 64)
    (hb : b ≤ 53) (hbl : b + l.length ≤ 62) (seeds : ℕ → UInt64) {ms : ℝ} {es : ℤ}
    (hms : IsMantissa ms) {mx : ℕ → ℝ} {ex : ℕ → ℤ}
    (hmx : ∀ m < l.sum + N, IsMantissa (mx m))
    (hguard : ∀ m < l.sum + N, |levelHi fl ms es (mx m) (ex m)| < 2 ^ b) {s D : ℝ}
    (hs : s = ms * 2 ^ es) (hs0 : 0 ≤ s) {x : ℕ → ℝ}
    (hx : ∀ m < l.sum + N, x m = mx m * 2 ^ ex m) (hD : ∀ n < N, |diffs l x n| ≤ D)
    (hsD : s * D ≤ ((V : ℝ) - norm1 l) / 2) {L FS q : ℕ → Int64}
    (hL : L = levelsI64 fl ri ms es mx ex (anchorFraction fl l G seeds))
    (hFS : FS = anchorSteps l G seeds) (hq : q = encodeTokens V l L FS) :
    (∀ m < l.sum + N, (decodeLevels V l L FS q m).toInt =
      roundHalfUp (s * x m - anchorFrac G (anchorInt l seeds) m)) ∧
      ¬InvertRaises l L (decodeSteps V FS q) (l.sum + N) := by
  subst hL hFS hq
  set P := anchorInt l seeds
  set L := levelsI64 fl ri ms es mx ex (anchorFraction fl l G seeds)
  set FS := anchorSteps l G seeds
  set R : ℕ → ℤ := fun m => roundHalfUp (s * x m - anchorFrac G P m)
  have h53 : ∀ m < l.sum + N, |levelHi fl ms es (mx m) (ex m)| < 2 ^ 53 := fun m hm =>
    (hguard m hm).trans_le (pow_le_pow_right₀ (by norm_num) hb)
  have hLR : ∀ m < l.sum + N, (L m).toInt = R m := fun m hm =>
    (toInt_levelsI64_anchor hfl hri hG hb seeds hms (hmx m hm) (hguard m hm) hs (hx m hm)).1
  have hRmem : ∀ m < l.sum + N, -2 ^ b - 1 ≤ R m ∧ R m ≤ 2 ^ b := fun m hm =>
    (toInt_levelsI64_anchor hfl hri hG hb seeds hms (hmx m hm) (hguard m hm) hs (hx m hm)).2.1
  -- the tokens are the integer ones
  have htok : ∀ n < N,
      (encodeTokens V l L FS n).toInt = diffs l R n + fracSteps l G P n + V / 2 :=
    fun n hn => by
      obtain ⟨h1, h2, h3⟩ := encode_token_mem hfl hri hV hV63 hl hl0 hG hlG seeds hms hmx h53 hs
        hs0 hx (by omega : n + l.sum < l.sum + N) (hD n hn) hsD
      rw [encodeTokens_eq hV63 (by rw [h1]; omega) (by rw [h1]; exact h3), h1, token]
  -- so the steps are `c(B)` of the levels
  have hstep : ∀ n < N, (decodeSteps V FS (encodeTokens V l L FS) n).toInt = diffs l R n :=
    fun n hn => by
      refine eq_of_modEq_of_inInt64 ?_ (inInt64_toInt _)
        (inInt64_diffs_of_mem_Icc hbl hRmem (by omega))
      unfold decodeSteps
      calc _ ≡ (encodeTokens V l L FS n - Int64.ofInt (V / 2)).toInt - (FS n).toInt
            [ZMOD 2 ^ 64] := toInt_sub_modEq _ _
        _ ≡ ((encodeTokens V l L FS n).toInt - V / 2) - (FS n).toInt [ZMOD 2 ^ 64] :=
            ((toInt_sub_modEq _ _).trans ((Int.ModEq.refl _).sub (toInt_ofInt_modEq _))).sub_right _
        _ = diffs l R n := by rw [htok n hn, toInt_anchorSteps hG hlG]; ring
  -- and their inversion in integers is the levels
  have hI : ∀ m < l.sum + N, invert l (fun m => (L m).toInt)
      (fun n => (decodeSteps V FS (encodeTokens V l L FS) n).toInt) m = R m := fun m hm =>
    eq_of_diffs_eq_of_lt hl (fun n hn => by
        rw [diffs_invert hl]
        exact hstep n (by omega))
      (fun m' hm' => by
        rw [invert_of_lt hl _ _ hm']
        exact hLR m' (by omega)) hm
  obtain ⟨hval, hnr⟩ := toInt_invertI64 hl (M := l.sum + N) (by omega) (pre := L)
    (d := decodeSteps V FS (encodeTokens V l L FS)) fun l₁ hl₁ n hn => by
      rw [diffs_congr (l := l₁) (n := n) fun i hi => hI (n + i) (by omega)]
      exact inInt64_diffs_of_mem_Icc (by have := hl₁.length_le; omega) hRmem hn
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

/-- **The tokenizer in float64 and int64.** Take a column with the operator `c(B)` of `m ≥ 1`
factors over the positive lags `l`, of degree `p = l.sum`: its `p + N` rows
`x m = m_x m·2^(e_x m)` and its scale `s = m_s·2^e_s` as `np.frexp` splits them, within
`_levels`' guard `|p| < 2^b`, `b ≤ 53`, `b + m ≤ 62`. Take the anchor of any `seeds` with
`G ≤ 53`, `m + G ≤ 64` fraction bits (`b = min(53, 62 - m)` and `G = min(53, 64 - m)` meet
these), and steps `|c(B) x| ≤ D` with `s·D ≤ (V - ‖c‖₁)/2`. Then:

* every token `encode` emits is its token sum and lies in `[1, V-1]`, so neither clip changes
  it and `_clip_tokens` counts none;
* `decode`'s `_invert_difference` does not raise;
* every decoded value lies within `1/(2s) + 2^-52·(1 + 2^-54)·(|x| + 1/(2s)) +
  2^-1075·(1 + (1 + 2^-53)/s)` of `x`, and the prefix rows come back exactly. -/
theorem encode_decode {fl : ℝ → ℝ} (hfl : IsRoundNearest fl) {ri : ℝ → ℤ} (hri : IsRoundInt ri)
    {V : ℤ} (hV : Even V) (hV63 : V < 2 ^ 63) {l : List ℕ} (hl : ∀ S ∈ l, 0 < S) (hl0 : l ≠ [])
    {G b N : ℕ} (hG : G ≤ 53) (hlG : l.length + G ≤ 64) (hb : b ≤ 53) (hbl : b + l.length ≤ 62)
    (seeds : ℕ → UInt64) {ms : ℝ} {es : ℤ} (hms : IsMantissa ms) {mx : ℕ → ℝ} {ex : ℕ → ℤ}
    (hmx : ∀ m < l.sum + N, IsMantissa (mx m))
    (hguard : ∀ m < l.sum + N, |levelHi fl ms es (mx m) (ex m)| < 2 ^ b) {s D : ℝ}
    (hs : s = ms * 2 ^ es) (hs0 : 0 < s) {x : ℕ → ℝ}
    (hx : ∀ m < l.sum + N, x m = mx m * 2 ^ ex m) (hD : ∀ n < N, |diffs l x n| ≤ D)
    (hsD : s * D ≤ ((V : ℝ) - norm1 l) / 2) {F : ℕ → ℝ} (hF : F = anchorFraction fl l G seeds)
    {FS : ℕ → Int64} (hFS : FS = anchorSteps l G seeds) {L : ℕ → Int64}
    (hL : L = levelsI64 fl ri ms es mx ex F) {q : ℕ → Int64} (hq : q = encodeTokens V l L FS) :
    (∀ n < N, q n = tokenSum V l L FS n ∧ 1 ≤ (q n).toInt ∧ (q n).toInt ≤ V - 1) ∧
      ¬InvertRaises l L (decodeSteps V FS q) (l.sum + N) ∧
      ∀ m < l.sum + N, |decodeOut fl s l.sum x F (decodeLevels V l L FS q) m - x m| ≤
        1 / (2 * s) + 2 ^ (-52 : ℤ) * (1 + 2 ^ (-54 : ℤ)) * (|x m| + 1 / (2 * s)) +
          2 ^ (-1075 : ℤ) * (1 + (1 + 2 ^ (-53 : ℤ)) / s) := by
  subst hF
  obtain ⟨hlev, hnr⟩ := decode_levels_eq hfl hri hV hV63 hl hl0 hG hlG hb hbl seeds hms hmx
    hguard hs hs0.le hx hD hsD hL hFS hq
  subst hL hFS hq
  have h53 : ∀ m < l.sum + N, |levelHi fl ms es (mx m) (ex m)| < 2 ^ 53 := fun m hm =>
    (hguard m hm).trans_le (pow_le_pow_right₀ (by norm_num) hb)
  refine ⟨fun n hn => ?_, hnr, fun m hm => ?_⟩
  · obtain ⟨h1, h2, h3⟩ := encode_token_mem hfl hri hV hV63 hl hl0 hG hlG seeds hms hmx h53 hs
      hs0.le hx (by omega : n + l.sum < l.sum + N) (hD n hn) hsD
    rw [encodeTokens_eq hV63 (by rw [h1]; omega) (by rw [h1]; exact h3), h1]
    exact ⟨rfl, h2, h3⟩
  · unfold decodeOut
    split_ifs with hmk
    · rw [sub_self, abs_zero]
      positivity
    · rw [hlev m hm, anchorFraction_eq hfl hG]
      exact abs_decode_sub_le_closed hfl hs0 (x m) _
        (toInt_levelsI64_anchor hfl hri hG hb seeds hms (hmx m hm) (hguard m hm) hs (hx m hm)).2.2

/-! ### The order `0` -/

/-- The order `0` takes the fraction `1/2`, so `_levels` returns `⌊s·x⌋`, within `2^53`. -/
lemma toInt_levelsI64_half {fl : ℝ → ℝ} (hfl : IsRoundNearest fl) {ri : ℝ → ℤ}
    (hri : IsRoundInt ri) {ms : ℝ} {es : ℤ} (hms : IsMantissa ms) {mx : ℕ → ℝ} {ex : ℕ → ℤ}
    {m : ℕ} (hmx : IsMantissa (mx m)) (hguard : |levelHi fl ms es (mx m) (ex m)| < 2 ^ 53)
    {s x : ℝ} (hs : s = ms * 2 ^ es) (hx : x = mx m * 2 ^ ex m) :
    (levelsI64 fl ri ms es mx ex (fun _ => 1 / 2) m).toInt = ⌊s * x⌋ ∧ |⌊s * x⌋| ≤ 2 ^ 53 := by
  obtain ⟨h1, -, h3⟩ := toInt_levelsI64 hfl hri (G := 1) (b := 53) (F := fun _ => 1 / 2)
    (N := 1) (by norm_num) le_rfl (by norm_num) (by norm_num) hms (by norm_num) hmx hguard hs hx
  rw [← floor_eq_roundHalfUp] at h1 h3
  exact ⟨h1, h3⟩

/-- The order `0`: under `_levels`' guard `|p| < 2^53` the int64 token sum is `⌊s·x⌋ + V/2` on
every row, in the domain or not, for `0 ≤ V < 2^63`: the clip to `[-1, V]` puts a token outside
`[0, V-1]` on its own side. -/
theorem toInt_tokenSum_zero {fl : ℝ → ℝ} (hfl : IsRoundNearest fl) {ri : ℝ → ℤ}
    (hri : IsRoundInt ri) {V : ℤ} (hV0 : 0 ≤ V) (hV63 : V < 2 ^ 63) {ms : ℝ} {es : ℤ}
    (hms : IsMantissa ms) {mx : ℕ → ℝ} {ex : ℕ → ℤ} {n : ℕ} (hmx : IsMantissa (mx n))
    (hguard : |levelHi fl ms es (mx n) (ex n)| < 2 ^ 53) {s : ℝ} (hs : s = ms * 2 ^ es)
    {x : ℕ → ℝ} (hx : x n = mx n * 2 ^ ex n) :
    (tokenSum V [] (levelsI64 fl ri ms es mx ex fun _ => 1 / 2) 0 n).toInt =
      ⌊s * x n⌋ + V / 2 := by
  obtain ⟨hL, hR⟩ := toInt_levelsI64_half hfl hri hms hmx hguard hs hx
  obtain ⟨h1, h2⟩ := abs_le.mp hR
  refine eq_of_modEq_of_inInt64 ?_ (inInt64_toInt _) ⟨by omega, by omega⟩
  unfold tokenSum
  calc _ ≡ (diffsI64 [] (levelsI64 fl ri ms es mx ex fun _ => 1 / 2) n +
        (0 : ℕ → Int64) n).toInt + (Int64.ofInt (V / 2)).toInt [ZMOD 2 ^ 64] :=
        toInt_add_modEq _ _
    _ ≡ ((diffsI64 [] (levelsI64 fl ri ms es mx ex fun _ => 1 / 2) n).toInt +
        ((0 : ℕ → Int64) n).toInt) + V / 2 [ZMOD 2 ^ 64] :=
        (toInt_add_modEq _ _).add (toInt_ofInt_modEq _)
    _ = ⌊s * x n⌋ + V / 2 := by
        rw [diffsI64, hL, Pi.zero_apply, Int64.toInt_zero, add_zero]

/-- **The order `0` in float64 and int64.** Take a column with `c = 1`: its `N` rows
`x m = m_x m·2^(e_x m)` and its scale `s = m_s·2^e_s` within `_levels`' guard `|p| < 2^53`,
the fraction `1/2` and zero steps that `_anchor_fraction` gives both tokenizers, and values
`|x| ≤ D` with `s·D < V/2`. Then every token `encode` emits is its token sum `⌊s·x⌋ + V/2` and
lies in `[0, V-1]`, so neither clip changes it and `_clip_tokens` counts none, and every
decoded value `(⌊s·x⌋ + 1/2)/s` lies within `1/(2s) + 2^-52·(1 + 2^-54)·(|x| + 1/(2s)) +
2^-1075·(1 + (1 + 2^-53)/s)` of `x`. -/
theorem encode_decode_zero {fl : ℝ → ℝ} (hfl : IsRoundNearest fl) {ri : ℝ → ℤ}
    (hri : IsRoundInt ri) {V : ℤ} (hV : Even V) (hV63 : V < 2 ^ 63) {N : ℕ} {ms : ℝ} {es : ℤ}
    (hms : IsMantissa ms) {mx : ℕ → ℝ} {ex : ℕ → ℤ} (hmx : ∀ m < N, IsMantissa (mx m))
    (hguard : ∀ m < N, |levelHi fl ms es (mx m) (ex m)| < 2 ^ 53) {s D : ℝ}
    (hs : s = ms * 2 ^ es) (hs0 : 0 < s) {x : ℕ → ℝ} (hx : ∀ m < N, x m = mx m * 2 ^ ex m)
    (hD : ∀ m < N, |x m| ≤ D) (hsD : s * D < (V : ℝ) / 2) {L : ℕ → Int64}
    (hL : L = levelsI64 fl ri ms es mx ex fun _ => 1 / 2) {q : ℕ → Int64}
    (hq : q = encodeTokens V [] L 0) :
    (∀ n < N, q n = tokenSum V [] L 0 n ∧ (q n).toInt = ⌊s * x n⌋ + V / 2 ∧
      0 ≤ (q n).toInt ∧ (q n).toInt ≤ V - 1) ∧
      ∀ m < N, |decodeOut fl s 0 x (fun _ => 1 / 2) (decodeLevels V [] L 0 q) m - x m| ≤
        1 / (2 * s) + 2 ^ (-52 : ℤ) * (1 + 2 ^ (-54 : ℤ)) * (|x m| + 1 / (2 * s)) +
          2 ^ (-1075 : ℤ) * (1 + (1 + 2 ^ (-53 : ℤ)) / s) := by
  subst hL hq
  -- the tokens are `⌊s·x⌋ + V/2`, in `[0, V-1]`, and neither clip changes them
  have htok : ∀ n < N, encodeTokens V [] (levelsI64 fl ri ms es mx ex fun _ => 1 / 2) 0 n =
      tokenSum V [] (levelsI64 fl ri ms es mx ex fun _ => 1 / 2) 0 n ∧
      (tokenSum V [] (levelsI64 fl ri ms es mx ex fun _ => 1 / 2) 0 n).toInt =
        ⌊s * x n⌋ + V / 2 ∧
      0 ≤ ⌊s * x n⌋ + V / 2 ∧ ⌊s * x n⌋ + V / 2 ≤ V - 1 := fun n hn => by
    obtain ⟨h1, h2⟩ := floor_mem_of_domain hV hs0.le (hD n hn) hsD
    have h := toInt_tokenSum_zero hfl hri (by omega) hV63 hms (hmx n hn) (hguard n hn) hs
      (hx n hn)
    exact ⟨encodeTokens_eq hV63 (by rw [h]; exact h1) (by rw [h]; exact h2), h, h1, h2⟩
  refine ⟨fun n hn => ?_, fun m hm => ?_⟩
  · obtain ⟨h1, h2, h3, h4⟩ := htok n hn
    rw [h1, h2]
    exact ⟨rfl, rfl, h3, h4⟩
  -- `decode` takes the steps for the levels
  obtain ⟨h1, h2, -, -⟩ := htok m hm
  obtain ⟨hL, hR⟩ := toInt_levelsI64_half hfl hri hms (hmx m hm) (hguard m hm) hs (hx m hm)
  obtain ⟨r1, r2⟩ := abs_le.mp hR
  have hdec : (decodeLevels V [] (levelsI64 fl ri ms es mx ex fun _ => 1 / 2) 0
      (encodeTokens V [] (levelsI64 fl ri ms es mx ex fun _ => 1 / 2) 0) m).toInt =
      ⌊s * x m⌋ := by
    refine eq_of_modEq_of_inInt64 ?_ (inInt64_toInt _) ⟨by omega, by omega⟩
    unfold decodeLevels invertI64 decodeSteps
    calc _ ≡ (encodeTokens V [] (levelsI64 fl ri ms es mx ex fun _ => 1 / 2) 0 m -
          Int64.ofInt (V / 2)).toInt - ((0 : ℕ → Int64) m).toInt [ZMOD 2 ^ 64] :=
          toInt_sub_modEq _ _
      _ ≡ ((encodeTokens V [] (levelsI64 fl ri ms es mx ex fun _ => 1 / 2) 0 m).toInt -
          V / 2) - ((0 : ℕ → Int64) m).toInt [ZMOD 2 ^ 64] :=
          ((toInt_sub_modEq _ _).trans ((Int.ModEq.refl _).sub (toInt_ofInt_modEq _))).sub_right _
      _ = ⌊s * x m⌋ := by rw [h1, h2, Pi.zero_apply, Int64.toInt_zero]; ring
  unfold decodeOut
  rw [ite_eq_right (Nat.not_lt_zero m), hdec, floor_eq_roundHalfUp]
  exact abs_decode_sub_le_closed hfl hs0 (x m) (1 / 2)
    (by rw [← floor_eq_roundHalfUp]; exact hR)

end DiffGPT
