module

public import Mathlib
public import DiffGPT.LevelGrid
public import DiffGPT.Float64
public import DiffGPT.ExactLevels
public import DiffGPT.Int64

/-!
# The tokenizer in float64 and int64

`encode` and `decode` as the code computes them, for one column `x` with the operator
`c(B) = (1 - B^S₁)⋯(1 - B^Sₘ)` over the lags `l` of degree `p = l.sum`, the scale `s` and the
zero point `z`: its `p` prefix rows and the `N` rows after them (index `m`), the token of row
`n + p` at `n`.

    P = invert l seeds step mod 2^64          `_invert_difference` in uint64
    F m = (P m mod 2^G) / 2^G                 `anchorFraction`, exact
    FS n = c(B) (P mod 2^G) n >> G            `anchorSteps`, exact
    L m = ⌊s·x m - F m + 1/2⌋                 `levelsI64`: `_levels`, exact (`levels_eq`)
    q n = clip (clip (c(B) L n + FS n) + z)   `encodeTokens`, in int64
    L' = invert l L (q - z - FS)              `decodeLevels`: `_invert_difference` in int64
    x̂ m = fl (fl (fl (L' m) + F m) / s)       `decodeOut`, `x m` on the prefix rows

The `seeds` are the uint64 values `_anchor_fraction` takes off the prefix, the fractional parts
of its `s·x` in units of `2^-G`, rounded, and the `step` is `0` for "mash" and `2^(G-1)` for
"mash-half"; "lgt" is `seeds = 0` and `step = 0`, which give `F = FS = 0` (`anchor_zero`). `P`
is the integer `anchorInt`, with `c(B) P = step`, modulo `2^64`. `decode` takes the anchor and
the prefix levels off the same floats, `start`, so it gets the same `F`, `FS` and `L m`,
`m < p`. The rows are finite floats `m·2^e`: `encode` raises on NaN and infinities.

* `token_eq`, `token_mem`: the integer token sum `c(B) L + FS` is `c(B)` of the levels
  `⌊s·x - B + 1/2⌋` measured from the anchor `B = P / 2^G`, which steps by `h = step / 2^G`
  levels, `0` or `1/2`, so the exact theory applies: with a zero point of
  `diffs_levels_add_mem` the token lies in `[t, V - 1]` where `lo ≤ c(B) x ≤ hi`. Where
  `[lo, hi]` fits (`Fits`), which `level_grid` checks in rationals, `_zero_point` returns one
  (`zeroPoint_mem`), and `get_domain_of_definition` returns the tightest float bounds, with
  `lo ≤ c(B) x ≤ hi` exactly (`differenceRange_mem` in `DiffGPT.DifferenceRange`).
* `encode_token_mem`: the int64 token sum is that token sum. The two agree modulo `2^64` and
  both lie in int64, so this needs only `_levels`' float64 guard `|p| < 2^53`. Neither clip
  then changes the token (`encodeTokens_eq`).
* `toInt_tokenSum`: under the guard `|p| < 2^b` with `b + m ≤ 62`, which `b = min(53, 62 - m)`
  meets, the token sum is the integer one on every row, in the range or not, and for
  `|z| < 2^62`, which `level_grid` checks, and `V < 2^62`, which `_check_vocab_size` checks,
  neither the clips nor `z` overflow: a token outside `[0, V - 1]` lands on its own side
  (`toInt_encodeTokens`).
* `decode_levels_eq`: `decode` gets the levels `L` back from `encode`'s tokens, and
  `_invert_difference` does not raise: its sums pass through the differences of `L` by the
  first factors, which int64 holds for `b + m ≤ 62` (`toInt_invertI64`).
* `abs_decode_sub_le_f64`, `abs_decode_sub_le_closed`: the decoded value is within `1/(2s)` of
  `x` up to the roundings of `L + F` and of the quotient, at most
  `2^-52·(1 + 2^-54)·(|x| + 1/(2s)) + 2^-1075·(1 + (1 + 2^-53)/s)` more, the level being a
  float (`abs_levels_le`).
* `encode_decode`: all of it for a column, from the floats to the decoded values.
* `encode_decode_zero`: the order `0`, `c = 1`, with the fraction `1/2` of every tokenizer:
  the tokens `⌊s·x⌋ + z` lie in `[0, V - 1]` for the zero points of `floor_add_mem`, which
  `_zero_point` returns (`zeroPoint0_mem`), and the decoded `(⌊s·x⌋ + 1/2)/s` is within
  `1/(2s)` of `x` up to the same roundings.
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

/-- The token sum `encode` forms for row `n + p` before the zero point and the clips:
`_difference(levels, lags) + fraction_steps` with the levels `⌊w + 1/2⌋`, which `_levels`
returns for `w = s·x - F` (`levels_eq`). -/
noncomputable def token (l : List ℕ) (G : ℕ) (P : ℕ → ℤ) (w : ℕ → ℝ) (n : ℕ) : ℤ :=
  diffs l (fun m => roundHalfUp (w m)) n + fracSteps l G P n

/-- Where `c(B) P` lies in `[0, 2^G)`, as the anchors' steps `0` and `2^(G-1)` do, the anchor's
term is exact: `fracSteps = -c(B) ⌊P / 2^G⌋`. -/
lemma fracSteps_eq {l : List ℕ} {G : ℕ} {P : ℕ → ℤ} {n : ℕ}
    (hP : 0 ≤ diffs l P n ∧ diffs l P n < 2 ^ G) :
    fracSteps l G P n = -diffs l (fun m => P m / 2 ^ G) n := by
  rw [fracSteps, diffs_emod_eq, sub_eq_add_neg, ← mul_neg,
    Int.add_mul_ediv_left _ _ (by positivity), Int.ediv_eq_zero_of_lt hP.1 hP.2, zero_add]

/-- The anchor's term lies in `[-2^(m-1), 2^(m-1))` for `m ≥ 1` factors, whatever `P`. -/
theorem fracSteps_mem {l : List ℕ} (hl : l ≠ []) (G : ℕ) (P : ℕ → ℤ) (n : ℕ) :
    -2 ^ (l.length - 1) ≤ fracSteps l G P n ∧ fracSteps l G P n < 2 ^ (l.length - 1) := by
  obtain ⟨h1, h2⟩ := abs_lt.mp (abs_diffs_emod_lt P hl G n)
  rw [pow_add] at h1 h2
  have hG : (0 : ℤ) < 2 ^ G := by positivity
  rw [fracSteps]
  exact ⟨Int.le_ediv_of_mul_le hG (by rw [neg_mul]; exact h1.le), Int.ediv_lt_of_lt_mul hG h2⟩

/-- The token sums are `c(B)` of the levels of `w + F`, `s·x` for `w = s·x - F`, measured from
the anchor `B = P / 2^G`. -/
theorem token_eq {l : List ℕ} {G : ℕ} {P : ℕ → ℤ} {n : ℕ}
    (hP : 0 ≤ diffs l P n ∧ diffs l P n < 2 ^ G) (w : ℕ → ℝ) :
    token l G P w n =
      diffs l (fun m => roundHalfUp (w m + anchorFrac G P m - (P m : ℝ) / 2 ^ G)) n := by
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

/-- The token at `n` lies in `[t, V - 1]` where `lo ≤ c(B) x n ≤ hi`, for the anchor's step
`h = c(B) P n / 2^G` levels and a zero point of `diffs_levels_add_mem`. -/
theorem token_mem {l : List ℕ} (hl : ∀ S ∈ l, 0 < S) (hl0 : l ≠ []) {G : ℕ} {P : ℕ → ℤ}
    {n : ℕ} (hP : 0 ≤ diffs l P n ∧ diffs l P n < 2 ^ G) {s h lo hi : ℝ} (hs : 0 ≤ s)
    (hh : ((diffs l P n : ℤ) : ℝ) / 2 ^ G = h) {x : ℕ → ℝ}
    (hx : lo ≤ diffs l x n ∧ diffs l x n ≤ hi) {V z t : ℤ}
    (hz : norm1 l / 2 + t - 1 - ⌊s * lo - h⌋ ≤ z ∧ z ≤ V - norm1 l / 2 - ⌈s * hi - h⌉) :
    t ≤ token l G P (fun m => s * x m - anchorFrac G P m) n + z ∧
      token l G P (fun m => s * x m - anchorFrac G P m) n + z ≤ V - 1 := by
  have hA : (1 : ℝ) * diffs l (fun m => (P m : ℝ) / 2 ^ G) n = h := by
    have h1 : (fun m => (P m : ℝ) / 2 ^ G) = ((2 : ℝ) ^ G)⁻¹ • fun m => (P m : ℝ) := by
      funext m
      rw [Pi.smul_apply, smul_eq_mul, div_eq_inv_mul]
    rw [one_mul, h1, diffs_const_smul, Pi.smul_apply, ← cast_diffs, smul_eq_mul,
      ← div_eq_inv_mul, hh]
  have hx' : s * lo ≤ diffs l (fun m => s * x m) n ∧ diffs l (fun m => s * x m) n ≤ s * hi := by
    have e : (fun m => s * x m) = s • x := by
      funext m
      rw [Pi.smul_apply, smul_eq_mul]
    rw [e, diffs_const_smul, Pi.smul_apply, smul_eq_mul]
    exact ⟨mul_le_mul_of_nonneg_left hx.1 hs, mul_le_mul_of_nonneg_left hx.2 hs⟩
  have H := diffs_levels_add_mem hl hl0 zero_le_one hA hx' (V := V) (z := z) (t := t)
    (by simpa only [one_mul] using hz)
  rw [token_eq hP]
  simpa only [sub_add_cancel, one_mul] using H

/-! ### The code -/

open scoped UInt64.CommRing in
/-- `_anchor_fraction`'s `numerators`: `_invert_difference` of the uint64 `seeds` with the
constant `steps`, which wraps modulo `2^64`, masked with `2^G - 1` and cast to int64. -/
noncomputable def anchorNum (l : List ℕ) (G : ℕ) (seeds : ℕ → UInt64) (step : UInt64)
    (m : ℕ) : Int64 :=
  ((invert l seeds (fun _ => step) m) &&& UInt64.ofNat (2 ^ G - 1)).toInt64

/-- The integer anchor `P`, which `anchorNum` reduces modulo `2^G`: `invert l seeds step` over
the integers. -/
def anchorInt (l : List ℕ) (seeds : ℕ → UInt64) (step : UInt64) : ℕ → ℤ :=
  invert l (fun m => ((seeds m).toNat : ℤ)) fun _ => (step.toNat : ℤ)

/-- `_anchor_fraction`'s `fraction`: `np.ldexp(numerators.astype(np.float64), -G)`. -/
noncomputable def anchorFraction (fl : ℝ → ℝ) (l : List ℕ) (G : ℕ) (seeds : ℕ → UInt64)
    (step : UInt64) (m : ℕ) : ℝ :=
  fl ((anchorNum l G seeds step m).toInt : ℝ) * 2 ^ (-(G : ℤ))

/-- `_anchor_fraction`'s `fraction_steps`: `_difference(numerators, lags) >> G` in int64. -/
noncomputable def anchorSteps (l : List ℕ) (G : ℕ) (seeds : ℕ → UInt64) (step : UInt64)
    (n : ℕ) : Int64 :=
  diffsI64 l (anchorNum l G seeds step) n >>> Int64.ofNat G

/-- `_levels`' int64 levels of the rows `x m = m_x m·2^(e_x m)` with the fractions `F`. -/
noncomputable def levelsI64 (fl : ℝ → ℝ) (ri : ℝ → ℤ) (ms : ℝ) (es : ℤ) (mx : ℕ → ℝ)
    (ex : ℕ → ℤ) (F : ℕ → ℝ) (m : ℕ) : Int64 :=
  Int64.ofInt (levels fl ri ms es (mx m) (ex m) (F m))

/-- `np.clip(a, lo, hi)` in int64. -/
def clipI64 (lo hi a : Int64) : Int64 := if a < lo then lo else if hi < a then hi else a

/-- `encode`'s token sums: `_difference(levels, lags) + fraction_steps` in int64. -/
def tokenSum (l : List ℕ) (L FS : ℕ → Int64) (n : ℕ) : Int64 :=
  diffsI64 l L n + FS n

/-- `encode`'s tokens: the sums clipped to `[-1 - z, V - z]` and `z` added, then clipped by
`_clip_tokens` to `[0, V - 1]`. -/
def encodeTokens (V z : ℤ) (l : List ℕ) (L FS : ℕ → Int64) (n : ℕ) : Int64 :=
  clipI64 0 (Int64.ofInt (V - 1))
    (clipI64 (Int64.ofInt (-1 - z)) (Int64.ofInt (V - z)) (tokenSum l L FS n) + Int64.ofInt z)

/-- `decode`'s steps: `tokens - grid.zero - fraction_steps` in int64. -/
def decodeSteps (z : ℤ) (FS q : ℕ → Int64) (n : ℕ) : Int64 :=
  q n - Int64.ofInt z - FS n

/-- `decode`'s levels: `_invert_difference` of the prefix levels and the steps, in int64. -/
def decodeLevels (z : ℤ) (l : List ℕ) (L FS q : ℕ → Int64) : ℕ → Int64 :=
  invertI64 l L (decodeSteps z FS q)

/-- `decode`'s output: `start` on the `p` prefix rows, `(levels + fraction) / scale` in float64
after them. -/
noncomputable def decodeOut (fl : ℝ → ℝ) (s : ℝ) (p : ℕ) (x F : ℕ → ℝ) (L : ℕ → Int64)
    (m : ℕ) : ℝ :=
  if m < p then x m else fl (fl (fl ((L m).toInt : ℝ) + F m) / s)

/-! ### The anchor -/

lemma diffs_anchorInt {l : List ℕ} (hl : ∀ S ∈ l, 0 < S) (seeds : ℕ → UInt64) (step : UInt64)
    (n : ℕ) : diffs l (anchorInt l seeds step) n = step.toNat :=
  congrFun (diffs_invert hl _ _) n

open scoped UInt64.CommRing in
/-- The numerators are the integer anchor modulo `2^G` (`anchor_numerators`). -/
lemma toInt_anchorNum {G : ℕ} (hG : G ≤ 63) (l : List ℕ) (seeds : ℕ → UInt64) (step : UInt64)
    (m : ℕ) : (anchorNum l G seeds step m).toInt = anchorInt l seeds step m % 2 ^ G := by
  have h := anchor_numerators (by omega : G ≤ 64) l seeds step m
  have hu : ((invert l seeds (fun _ => step) m &&& UInt64.ofNat (2 ^ G - 1)).toNat : ℤ) <
      2 ^ 63 := by
    rw [h]
    exact (Int.emod_lt_of_pos _ (by positivity)).trans_le (pow_le_pow_right₀ (by norm_num) hG)
  rw [anchorNum, toInt_toInt64_of_lt _ (by exact_mod_cast hu), h]
  rfl

lemma anchorFraction_eq {fl : ℝ → ℝ} (hfl : IsRoundNearest fl) {G : ℕ} (hG : G ≤ 53)
    (l : List ℕ) (seeds : ℕ → UInt64) (step : UInt64) (m : ℕ) :
    anchorFraction fl l G seeds step m = anchorFrac G (anchorInt l seeds step) m := by
  have h0 := Int.emod_nonneg (anchorInt l seeds step m) (by positivity : (2 : ℤ) ^ G ≠ 0)
  have h1 := Int.emod_lt_of_pos (anchorInt l seeds step m) (by positivity : (0 : ℤ) < 2 ^ G)
  have h53 : |anchorInt l seeds step m % 2 ^ G| ≤ 2 ^ 53 := by
    rw [abs_of_nonneg h0]
    exact h1.le.trans (pow_le_pow_right₀ (by norm_num) hG)
  rw [anchorFraction, toInt_anchorNum (by omega), hfl.eq_self (isF64_intCast h53), anchorFrac,
    zpow_neg, zpow_natCast, div_eq_mul_inv]

/-- `fraction_steps` is exact: the numerators lie in `[0, 2^G)`, so `_difference` is exact for
`m + G ≤ 64`, and `>>` divides by `2^G`. -/
lemma toInt_anchorSteps {l : List ℕ} {G : ℕ} (hG : G ≤ 53) (hlG : l.length + G ≤ 64)
    (seeds : ℕ → UInt64) (step : UInt64) (n : ℕ) :
    (anchorSteps l G seeds step n).toInt = fracSteps l G (anchorInt l seeds step) n := by
  have hnum : (fun m => (anchorNum l G seeds step m).toInt) =
      fun m => anchorInt l seeds step m % 2 ^ G :=
    funext fun m => toInt_anchorNum (by omega) l seeds step m
  rw [anchorSteps, toInt_shiftRight_ofNat _ (by omega), fracSteps,
    toInt_diffsI64_of_mem_Ico hlG (fun m => ?_) n, hnum]
  rw [toInt_anchorNum (by omega)]
  exact ⟨Int.emod_nonneg _ (by positivity), Int.emod_lt_of_pos _ (by positivity)⟩

/-- "lgt": zero seeds and step give the zero anchor, which `_anchor_fraction` returns for
"lgt". -/
lemma anchor_zero {fl : ℝ → ℝ} (hfl : IsRoundNearest fl) {l : List ℕ} {G : ℕ} (hG : G ≤ 53)
    (hlG : l.length + G ≤ 64) : anchorFraction fl l G 0 0 = 0 ∧ anchorSteps l G 0 0 = 0 := by
  have hP : anchorInt l 0 0 = 0 := by
    have h : anchorInt l 0 0 = invert l (0 : ℕ → ℤ) 0 := by
      unfold anchorInt
      congr 1
    rw [h, invert_zero]
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

/-- `toInt_levelsI64` for the anchor of the seeds and the step. -/
lemma toInt_levelsI64_anchor {fl : ℝ → ℝ} (hfl : IsRoundNearest fl) {ri : ℝ → ℤ}
    (hri : IsRoundInt ri) {l : List ℕ} {G b : ℕ} (hG : G ≤ 53) (hb : b ≤ 53)
    (seeds : ℕ → UInt64) (step : UInt64) {ms : ℝ} {es : ℤ} (hms : IsMantissa ms) {mx : ℕ → ℝ}
    {ex : ℕ → ℤ} {m : ℕ} (hmx : IsMantissa (mx m))
    (hguard : |levelHi fl ms es (mx m) (ex m)| < 2 ^ b) {s x : ℝ} (hs : s = ms * 2 ^ es)
    (hx : x = mx m * 2 ^ ex m) :
    (levelsI64 fl ri ms es mx ex (anchorFraction fl l G seeds step) m).toInt =
        roundHalfUp (s * x - anchorFrac G (anchorInt l seeds step) m) ∧
      (-2 ^ b - 1 ≤ roundHalfUp (s * x - anchorFrac G (anchorInt l seeds step) m) ∧
        roundHalfUp (s * x - anchorFrac G (anchorInt l seeds step) m) ≤ 2 ^ b) ∧
      |roundHalfUp (s * x - anchorFrac G (anchorInt l seeds step) m)| ≤ 2 ^ 53 := by
  have h := toInt_levelsI64 hfl hri hG hb
    (Int.emod_nonneg (anchorInt l seeds step m) (by positivity : (2 : ℤ) ^ G ≠ 0))
    (Int.emod_lt_of_pos (anchorInt l seeds step m) (by positivity : (0 : ℤ) < 2 ^ G)) hms
    (anchorFraction_eq hfl hG l seeds step m) hmx hguard hs hx
  rwa [anchorFraction_eq hfl hG l seeds step m] at h

/-- `np.clip` in int64, for `lo ≤ hi`. -/
lemma toInt_clipI64 {lo hi : Int64} (h : lo.toInt ≤ hi.toInt) (a : Int64) :
    (clipI64 lo hi a).toInt = max lo.toInt (min hi.toInt a.toInt) := by
  unfold clipI64
  split_ifs with h1 h2
  · rw [Int64.lt_iff_toInt_lt] at h1
    omega
  · rw [Int64.lt_iff_toInt_lt] at h2
    omega
  · rw [Int64.lt_iff_toInt_lt] at h1 h2
    omega

/-- For `0 < V < 2^62` and `|z| < 2^62` neither clip nor the zero point overflows: the token is
the token sum plus `z` clipped to `[0, V - 1]`, so one outside lands on its own side. -/
lemma toInt_encodeTokens {V z : ℤ} (hV0 : 0 < V) (hV62 : V < 2 ^ 62) (hz : |z| < 2 ^ 62)
    (l : List ℕ) (L FS : ℕ → Int64) (n : ℕ) :
    (encodeTokens V z l L FS n).toInt = max 0 (min (V - 1) ((tokenSum l L FS n).toInt + z)) := by
  obtain ⟨z1, z2⟩ := abs_lt.mp hz
  have e1 : (Int64.ofInt (-1 - z)).toInt = -1 - z := toInt_ofInt_of_inInt64 ⟨by omega, by omega⟩
  have e2 : (Int64.ofInt (V - z)).toInt = V - z := toInt_ofInt_of_inInt64 ⟨by omega, by omega⟩
  have e3 : (Int64.ofInt z).toInt = z := toInt_ofInt_of_inInt64 ⟨by omega, by omega⟩
  have e4 : (Int64.ofInt (V - 1)).toInt = V - 1 := toInt_ofInt_of_inInt64 ⟨by omega, by omega⟩
  have hin := toInt_clipI64 (lo := Int64.ofInt (-1 - z)) (hi := Int64.ofInt (V - z))
    (by rw [e1, e2]; omega) (tokenSum l L FS n)
  rw [e1, e2] at hin
  have hadd := toInt_add_of_inInt64
    (clipI64 (Int64.ofInt (-1 - z)) (Int64.ofInt (V - z)) (tokenSum l L FS n)) (Int64.ofInt z)
    (by rw [hin, e3]; exact ⟨by omega, by omega⟩)
  have hle : (0 : Int64).toInt ≤ (Int64.ofInt (V - 1)).toInt := by
    rw [Int64.toInt_zero, e4]
    omega
  rw [encodeTokens, toInt_clipI64 hle, Int64.toInt_zero, e4, hadd, hin, e3]
  omega

/-- A token sum `k` with `k + z` in `[0, V - 1]` passes both clips: the token is `k + z`, and
`_clip_tokens` does not count it. -/
lemma encodeTokens_eq {V z : ℤ} (hV62 : V < 2 ^ 62) (hz : |z| < 2 ^ 62) {l : List ℕ}
    {L FS : ℕ → Int64} {n : ℕ} {k : ℤ} (hk : (tokenSum l L FS n).toInt = k) (h1 : 0 ≤ k + z)
    (h2 : k + z ≤ V - 1) :
    encodeTokens V z l L FS n = tokenSum l L FS n + Int64.ofInt z ∧
      (encodeTokens V z l L FS n).toInt = k + z := by
  obtain ⟨z1, z2⟩ := abs_lt.mp hz
  have e3 : (Int64.ofInt z).toInt = z := toInt_ofInt_of_inInt64 ⟨by omega, by omega⟩
  have e : (encodeTokens V z l L FS n).toInt = k + z := by
    rw [toInt_encodeTokens (by omega) hV62 hz, hk]
    omega
  have e' : (tokenSum l L FS n + Int64.ofInt z).toInt = k + z := by
    rw [toInt_add_of_inInt64 (tokenSum l L FS n) (Int64.ofInt z)
      (by rw [hk, e3]; exact ⟨by omega, by omega⟩), hk, e3]
  exact ⟨Int64.toInt_inj.mp (e.trans e'.symm), e⟩

/-- On rows within `_levels`' float64 guard the int64 token sum is the integer `token` modulo
`2^64`, overflow or not. -/
lemma toInt_tokenSum_modEq {fl : ℝ → ℝ} (hfl : IsRoundNearest fl) {ri : ℝ → ℤ}
    (hri : IsRoundInt ri) {l : List ℕ} {G M : ℕ} (hG : G ≤ 53) (hlG : l.length + G ≤ 64)
    (seeds : ℕ → UInt64) (step : UInt64) {ms : ℝ} {es : ℤ} (hms : IsMantissa ms) {mx : ℕ → ℝ}
    {ex : ℕ → ℤ} (hmx : ∀ m < M, IsMantissa (mx m))
    (hguard : ∀ m < M, |levelHi fl ms es (mx m) (ex m)| < 2 ^ 53) {s : ℝ}
    (hs : s = ms * 2 ^ es) {x : ℕ → ℝ} (hx : ∀ m < M, x m = mx m * 2 ^ ex m) {n : ℕ}
    (hn : n + l.sum < M) :
    (tokenSum l (levelsI64 fl ri ms es mx ex (anchorFraction fl l G seeds step))
        (anchorSteps l G seeds step) n).toInt ≡
      token l G (anchorInt l seeds step)
        (fun m => s * x m - anchorFrac G (anchorInt l seeds step) m) n [ZMOD 2 ^ 64] := by
  set L := levelsI64 fl ri ms es mx ex (anchorFraction fl l G seeds step)
  have e1 : diffs l (fun m => (L m).toInt) n =
      diffs l (fun m => roundHalfUp (s * x m - anchorFrac G (anchorInt l seeds step) m)) n :=
    diffs_congr fun i hi =>
      (toInt_levelsI64_anchor hfl hri hG le_rfl seeds step hms (hmx (n + i) (by omega))
        (hguard (n + i) (by omega)) hs (hx (n + i) (by omega))).1
  unfold tokenSum
  calc _ ≡ (diffsI64 l L n).toInt + (anchorSteps l G seeds step n).toInt [ZMOD 2 ^ 64] :=
        toInt_add_modEq _ _
    _ ≡ diffs l (fun m => (L m).toInt) n + (anchorSteps l G seeds step n).toInt
        [ZMOD 2 ^ 64] := (toInt_diffsI64_modEq l L n).add_right _
    _ = _ := by rw [e1, toInt_anchorSteps hG hlG, token]

/-- `encode`'s tokens on the rows `n + p < M` of a column within `_levels`' float64 guard
`|p| < 2^53`: the int64 token sum is the integer `token`, and where `lo ≤ c(B) x n ≤ hi`, with
the anchor's step `h = step / 2^G` levels and a zero point of `diffs_levels_add_mem`, the token
lies in `[t, V - 1]`. -/
theorem encode_token_mem {fl : ℝ → ℝ} (hfl : IsRoundNearest fl) {ri : ℝ → ℤ}
    (hri : IsRoundInt ri) {l : List ℕ} (hl : ∀ S ∈ l, 0 < S) (hl0 : l ≠ []) {G M : ℕ}
    (hG : G ≤ 53) (hlG : l.length + G ≤ 64) (seeds : ℕ → UInt64) {step : UInt64}
    (hstep : step.toNat < 2 ^ G) {ms : ℝ} {es : ℤ} (hms : IsMantissa ms) {mx : ℕ → ℝ}
    {ex : ℕ → ℤ} (hmx : ∀ m < M, IsMantissa (mx m))
    (hguard : ∀ m < M, |levelHi fl ms es (mx m) (ex m)| < 2 ^ 53) {s h lo hi : ℝ}
    (hs : s = ms * 2 ^ es) (hs0 : 0 ≤ s) (hh : (step.toNat : ℝ) / 2 ^ G = h) {x : ℕ → ℝ}
    (hx : ∀ m < M, x m = mx m * 2 ^ ex m) {n : ℕ} (hn : n + l.sum < M)
    (hD : lo ≤ diffs l x n ∧ diffs l x n ≤ hi) {V z t : ℤ} (hV62 : V < 2 ^ 62)
    (hz62 : |z| < 2 ^ 62) (ht : 0 ≤ t)
    (hz : norm1 l / 2 + t - 1 - ⌊s * lo - h⌋ ≤ z ∧ z ≤ V - norm1 l / 2 - ⌈s * hi - h⌉) :
    (tokenSum l (levelsI64 fl ri ms es mx ex (anchorFraction fl l G seeds step))
        (anchorSteps l G seeds step) n).toInt =
      token l G (anchorInt l seeds step)
        (fun m => s * x m - anchorFrac G (anchorInt l seeds step) m) n ∧
    t ≤ token l G (anchorInt l seeds step)
        (fun m => s * x m - anchorFrac G (anchorInt l seeds step) m) n + z ∧
    token l G (anchorInt l seeds step)
        (fun m => s * x m - anchorFrac G (anchorInt l seeds step) m) n + z ≤ V - 1 := by
  obtain ⟨z1, z2⟩ := abs_lt.mp hz62
  have hP : 0 ≤ diffs l (anchorInt l seeds step) n ∧
      diffs l (anchorInt l seeds step) n < 2 ^ G := by
    rw [diffs_anchorInt hl seeds step n]
    exact ⟨by positivity, by exact_mod_cast hstep⟩
  have hh' : ((diffs l (anchorInt l seeds step) n : ℤ) : ℝ) / 2 ^ G = h := by
    rw [diffs_anchorInt hl seeds step n, Int.cast_natCast]
    exact hh
  obtain ⟨t1, t2⟩ := token_mem hl hl0 hP hs0 hh' hD hz
  exact ⟨eq_of_modEq_of_inInt64
    (toInt_tokenSum_modEq hfl hri hG hlG seeds step hms hmx hguard hs hx hn) (inInt64_toInt _)
    ⟨by omega, by omega⟩, t1, t2⟩

/-- Under `_levels`' guard `|p| < 2^b` with `b + m ≤ 62` the int64 token sum is the integer one
on every row, in the range or not, so the clips put a token outside `[0, V - 1]` on its own
side (`toInt_encodeTokens`). -/
theorem toInt_tokenSum {fl : ℝ → ℝ} (hfl : IsRoundNearest fl) {ri : ℝ → ℤ}
    (hri : IsRoundInt ri) {l : List ℕ} (hl0 : l ≠ []) {G b M : ℕ} (hG : G ≤ 53)
    (hlG : l.length + G ≤ 64) (hb : b ≤ 53) (hbl : b + l.length ≤ 62) (seeds : ℕ → UInt64)
    (step : UInt64) {ms : ℝ} {es : ℤ} (hms : IsMantissa ms) {mx : ℕ → ℝ} {ex : ℕ → ℤ}
    (hmx : ∀ m < M, IsMantissa (mx m))
    (hguard : ∀ m < M, |levelHi fl ms es (mx m) (ex m)| < 2 ^ b) {s : ℝ}
    (hs : s = ms * 2 ^ es) {x : ℕ → ℝ} (hx : ∀ m < M, x m = mx m * 2 ^ ex m) {n : ℕ}
    (hn : n + l.sum < M) :
    (tokenSum l (levelsI64 fl ri ms es mx ex (anchorFraction fl l G seeds step))
        (anchorSteps l G seeds step) n).toInt =
      token l G (anchorInt l seeds step)
        (fun m => s * x m - anchorFrac G (anchorInt l seeds step) m) n := by
  have h53 : ∀ m < M, |levelHi fl ms es (mx m) (ex m)| < 2 ^ 53 := fun m hm =>
    (hguard m hm).trans_le (pow_le_pow_right₀ (by norm_num) hb)
  refine eq_of_modEq_of_inInt64
    (toInt_tokenSum_modEq hfl hri hG hlG seeds step hms hmx h53 hs hx hn) (inInt64_toInt _) ?_
  set P := anchorInt l seeds step
  have hR : ∀ m < M, -2 ^ b - 1 ≤ roundHalfUp (s * x m - anchorFrac G P m) ∧
      roundHalfUp (s * x m - anchorFrac G P m) ≤ 2 ^ b := fun m hm =>
    (toInt_levelsI64_anchor hfl hri hG hb seeds step hms (hmx m hm) (hguard m hm) hs
      (hx m hm)).2.1
  rw [token]
  -- `|c(B) R| ≤ 2^(m+b) + 2^(m-1)`, `-2^(m-1) ≤ fracSteps < 2^(m-1)` and `2^(m-1+b) ≤ 2^61`
  have hf := fracSteps_mem hl0 G P n
  have hd := abs_diffs_le_of_mem_Icc_of_lt hR hl0 hn
  have h1 : (2 : ℤ) ^ (l.length - 1) * 2 ^ b ≤ 2 ^ 61 := by
    rw [← pow_add]
    exact pow_le_pow_right₀ (by norm_num) (by omega)
  have h2 : (2 : ℤ) ^ (l.length - 1) ≤ 2 ^ 61 := pow_le_pow_right₀ (by norm_num) (by omega)
  have h3 : (2 : ℤ) ^ (l.length - 1) * (2 ^ b - (-2 ^ b - 1)) =
      2 * (2 ^ (l.length - 1) * 2 ^ b) + 2 ^ (l.length - 1) := by ring
  obtain ⟨b1, b2⟩ := abs_le.mp hd
  obtain ⟨c1, c2⟩ := hf
  exact ⟨by linarith, by linarith⟩

/-! ### `decode` -/

/-- `decode` of `encode`'s tokens, for a column whose `p + N` rows lie within `_levels`' guard
`|p| < 2^b`, `b ≤ 53`, `b + m ≤ 62`, and whose `N` steps lie in `[lo, hi]`, with a zero point
of `diffs_levels_add_mem`: `_invert_difference` returns the encoder's levels
`⌊s·x - F + 1/2⌋` and does not raise. -/
theorem decode_levels_eq {fl : ℝ → ℝ} (hfl : IsRoundNearest fl) {ri : ℝ → ℤ}
    (hri : IsRoundInt ri) {l : List ℕ} (hl : ∀ S ∈ l, 0 < S) (hl0 : l ≠ []) {G b N : ℕ}
    (hG : G ≤ 53) (hlG : l.length + G ≤ 64) (hb : b ≤ 53) (hbl : b + l.length ≤ 62)
    (seeds : ℕ → UInt64) {step : UInt64} (hstep : step.toNat < 2 ^ G) {ms : ℝ} {es : ℤ}
    (hms : IsMantissa ms) {mx : ℕ → ℝ} {ex : ℕ → ℤ}
    (hmx : ∀ m < l.sum + N, IsMantissa (mx m))
    (hguard : ∀ m < l.sum + N, |levelHi fl ms es (mx m) (ex m)| < 2 ^ b) {s h lo hi : ℝ}
    (hs : s = ms * 2 ^ es) (hs0 : 0 ≤ s) (hh : (step.toNat : ℝ) / 2 ^ G = h) {x : ℕ → ℝ}
    (hx : ∀ m < l.sum + N, x m = mx m * 2 ^ ex m)
    (hD : ∀ n < N, lo ≤ diffs l x n ∧ diffs l x n ≤ hi) {V z t : ℤ} (hV62 : V < 2 ^ 62)
    (hz62 : |z| < 2 ^ 62) (ht : 0 ≤ t)
    (hz : norm1 l / 2 + t - 1 - ⌊s * lo - h⌋ ≤ z ∧ z ≤ V - norm1 l / 2 - ⌈s * hi - h⌉)
    {L FS q : ℕ → Int64}
    (hL : L = levelsI64 fl ri ms es mx ex (anchorFraction fl l G seeds step))
    (hFS : FS = anchorSteps l G seeds step) (hq : q = encodeTokens V z l L FS) :
    (∀ m < l.sum + N, (decodeLevels z l L FS q m).toInt =
      roundHalfUp (s * x m - anchorFrac G (anchorInt l seeds step) m)) ∧
      ¬InvertRaises l L (decodeSteps z FS q) (l.sum + N) := by
  subst hL hFS hq
  set P := anchorInt l seeds step
  set L := levelsI64 fl ri ms es mx ex (anchorFraction fl l G seeds step)
  set FS := anchorSteps l G seeds step
  set R : ℕ → ℤ := fun m => roundHalfUp (s * x m - anchorFrac G P m)
  have h53 : ∀ m < l.sum + N, |levelHi fl ms es (mx m) (ex m)| < 2 ^ 53 := fun m hm =>
    (hguard m hm).trans_le (pow_le_pow_right₀ (by norm_num) hb)
  have hLR : ∀ m < l.sum + N, (L m).toInt = R m := fun m hm =>
    (toInt_levelsI64_anchor hfl hri hG hb seeds step hms (hmx m hm) (hguard m hm) hs
      (hx m hm)).1
  have hRmem : ∀ m < l.sum + N, -2 ^ b - 1 ≤ R m ∧ R m ≤ 2 ^ b := fun m hm =>
    (toInt_levelsI64_anchor hfl hri hG hb seeds step hms (hmx m hm) (hguard m hm) hs
      (hx m hm)).2.1
  -- the tokens are the integer ones
  have htok : ∀ n < N,
      (encodeTokens V z l L FS n).toInt = diffs l R n + fracSteps l G P n + z := fun n hn => by
    obtain ⟨h1, h2, h3⟩ := encode_token_mem hfl hri hl hl0 hG hlG seeds hstep hms hmx h53 hs
      hs0 hh hx (by omega : n + l.sum < l.sum + N) (hD n hn) hV62 hz62 ht hz
    rw [(encodeTokens_eq hV62 hz62 h1 (by omega) h3).2, token]
  -- so the steps are `c(B)` of the levels
  have hsteps : ∀ n < N, (decodeSteps z FS (encodeTokens V z l L FS) n).toInt = diffs l R n :=
    fun n hn => by
      refine eq_of_modEq_of_inInt64 ?_ (inInt64_toInt _)
        (inInt64_diffs_of_mem_Icc hbl hRmem (by omega))
      unfold decodeSteps
      calc _ ≡ (encodeTokens V z l L FS n - Int64.ofInt z).toInt - (FS n).toInt
            [ZMOD 2 ^ 64] := toInt_sub_modEq _ _
        _ ≡ ((encodeTokens V z l L FS n).toInt - z) - (FS n).toInt [ZMOD 2 ^ 64] :=
            ((toInt_sub_modEq _ _).trans ((Int.ModEq.refl _).sub (toInt_ofInt_modEq _))).sub_right _
        _ = diffs l R n := by rw [htok n hn, toInt_anchorSteps hG hlG]; ring
  -- and their inversion in integers is the levels
  have hI : ∀ m < l.sum + N, invert l (fun m => (L m).toInt)
      (fun n => (decodeSteps z FS (encodeTokens V z l L FS) n).toInt) m = R m := fun m hm =>
    eq_of_diffs_eq_of_lt hl (fun n hn => by
        rw [diffs_invert hl]
        exact hsteps n (by omega))
      (fun m' hm' => by
        rw [invert_of_lt hl _ _ hm']
        exact hLR m' (by omega)) hm
  obtain ⟨hval, hnr⟩ := toInt_invertI64 hl (M := l.sum + N) (by omega) (pre := L)
    (d := decodeSteps z FS (encodeTokens V z l L FS)) fun l₁ hl₁ n hn => by
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
`_levels`' guard `|p| < 2^b`, `b ≤ 53`, `b + m ≤ 62`. Take the anchor of any `seeds` and
`step < 2^G`, `G ≤ 53`, `m + G ≤ 64` (`b = min(53, 62 - m)`, `G = min(53, 64 - m)` and the
steps `0` and `2^(G-1)` meet these), which steps by `h = step/2^G` levels, the steps
`lo ≤ c(B) x ≤ hi`, and a zero point `|z| < 2^62` of `diffs_levels_add_mem` for the tokens
`[t, V - 1]`, `0 ≤ t`, `V < 2^62`, such as `_zero_point` returns where `[lo, hi]` fits
(`zeroPoint_mem`); `level_grid` and `_check_vocab_size` check both bounds. Then:

* every token `encode` emits is its token sum plus `z` and lies in `[t, V - 1]`, so neither
  clip changes it and `_clip_tokens` counts none;
* `decode`'s `_invert_difference` does not raise;
* every decoded value lies within `1/(2s) + 2^-52·(1 + 2^-54)·(|x| + 1/(2s)) +
  2^-1075·(1 + (1 + 2^-53)/s)` of `x`, and the prefix rows come back exactly. -/
theorem encode_decode {fl : ℝ → ℝ} (hfl : IsRoundNearest fl) {ri : ℝ → ℤ} (hri : IsRoundInt ri)
    {l : List ℕ} (hl : ∀ S ∈ l, 0 < S) (hl0 : l ≠ []) {G b N : ℕ} (hG : G ≤ 53)
    (hlG : l.length + G ≤ 64) (hb : b ≤ 53) (hbl : b + l.length ≤ 62) (seeds : ℕ → UInt64)
    {step : UInt64} (hstep : step.toNat < 2 ^ G) {ms : ℝ} {es : ℤ} (hms : IsMantissa ms)
    {mx : ℕ → ℝ} {ex : ℕ → ℤ} (hmx : ∀ m < l.sum + N, IsMantissa (mx m))
    (hguard : ∀ m < l.sum + N, |levelHi fl ms es (mx m) (ex m)| < 2 ^ b) {s h lo hi : ℝ}
    (hs : s = ms * 2 ^ es) (hs0 : 0 < s) (hh : (step.toNat : ℝ) / 2 ^ G = h) {x : ℕ → ℝ}
    (hx : ∀ m < l.sum + N, x m = mx m * 2 ^ ex m)
    (hD : ∀ n < N, lo ≤ diffs l x n ∧ diffs l x n ≤ hi) {V z t : ℤ} (hV62 : V < 2 ^ 62)
    (hz62 : |z| < 2 ^ 62) (ht : 0 ≤ t)
    (hz : norm1 l / 2 + t - 1 - ⌊s * lo - h⌋ ≤ z ∧ z ≤ V - norm1 l / 2 - ⌈s * hi - h⌉)
    {F : ℕ → ℝ} (hF : F = anchorFraction fl l G seeds step) {FS : ℕ → Int64}
    (hFS : FS = anchorSteps l G seeds step) {L : ℕ → Int64}
    (hL : L = levelsI64 fl ri ms es mx ex F) {q : ℕ → Int64} (hq : q = encodeTokens V z l L FS) :
    (∀ n < N, q n = tokenSum l L FS n + Int64.ofInt z ∧ t ≤ (q n).toInt ∧ (q n).toInt ≤ V - 1) ∧
      ¬InvertRaises l L (decodeSteps z FS q) (l.sum + N) ∧
      ∀ m < l.sum + N, |decodeOut fl s l.sum x F (decodeLevels z l L FS q) m - x m| ≤
        1 / (2 * s) + 2 ^ (-52 : ℤ) * (1 + 2 ^ (-54 : ℤ)) * (|x m| + 1 / (2 * s)) +
          2 ^ (-1075 : ℤ) * (1 + (1 + 2 ^ (-53 : ℤ)) / s) := by
  subst hF
  obtain ⟨hlev, hnr⟩ := decode_levels_eq hfl hri hl hl0 hG hlG hb hbl seeds hstep hms hmx
    hguard hs hs0.le hh hx hD hV62 hz62 ht hz hL hFS hq
  subst hL hFS hq
  have h53 : ∀ m < l.sum + N, |levelHi fl ms es (mx m) (ex m)| < 2 ^ 53 := fun m hm =>
    (hguard m hm).trans_le (pow_le_pow_right₀ (by norm_num) hb)
  refine ⟨fun n hn => ?_, hnr, fun m hm => ?_⟩
  · obtain ⟨h1, h2, h3⟩ := encode_token_mem hfl hri hl hl0 hG hlG seeds hstep hms hmx h53 hs
      hs0.le hh hx (by omega : n + l.sum < l.sum + N) (hD n hn) hV62 hz62 ht hz
    obtain ⟨he, he'⟩ := encodeTokens_eq hV62 hz62 h1 (by omega) h3
    rw [he']
    exact ⟨he, h2, h3⟩
  · unfold decodeOut
    split_ifs with hmk
    · rw [sub_self, abs_zero]
      positivity
    · rw [hlev m hm, anchorFraction_eq hfl hG]
      exact abs_decode_sub_le_closed hfl hs0 (x m) _
        (toInt_levelsI64_anchor hfl hri hG hb seeds step hms (hmx m hm) (hguard m hm) hs
          (hx m hm)).2.2

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

/-- The order `0`: under `_levels`' guard `|p| < 2^53` the int64 token sum is `⌊s·x⌋` on every
row, in the range or not, so the clips put a token outside `[0, V - 1]` on its own side
(`toInt_encodeTokens`). -/
theorem toInt_tokenSum_zero {fl : ℝ → ℝ} (hfl : IsRoundNearest fl) {ri : ℝ → ℤ}
    (hri : IsRoundInt ri) {ms : ℝ} {es : ℤ} (hms : IsMantissa ms) {mx : ℕ → ℝ} {ex : ℕ → ℤ}
    {n : ℕ} (hmx : IsMantissa (mx n)) (hguard : |levelHi fl ms es (mx n) (ex n)| < 2 ^ 53)
    {s : ℝ} (hs : s = ms * 2 ^ es) {x : ℕ → ℝ} (hx : x n = mx n * 2 ^ ex n) :
    (tokenSum [] (levelsI64 fl ri ms es mx ex fun _ => 1 / 2) 0 n).toInt = ⌊s * x n⌋ := by
  obtain ⟨hL, -⟩ := toInt_levelsI64_half hfl hri hms hmx hguard hs hx
  rw [tokenSum, diffsI64, Pi.zero_apply, Int64.add_zero, hL]

/-- **The order `0` in float64 and int64.** Take a column with `c = 1`: its `N` rows
`x m = m_x m·2^(e_x m)` and its scale `s = m_s·2^e_s` within `_levels`' guard `|p| < 2^53`,
the fraction `1/2` and zero steps that `_anchor_fraction` gives every tokenizer, values
`lo ≤ x ≤ hi`, and a zero point `|z| < 2^62` of `floor_add_mem`, `V < 2^62`, such as
`_zero_point` returns where `[lo, hi]` fits (`zeroPoint0_mem`); `level_grid` and
`_check_vocab_size` check both bounds. Then every token `encode` emits
is its token sum `⌊s·x⌋` plus `z` and lies in `[0, V - 1]`, so neither clip changes it and
`_clip_tokens` counts none, and every decoded value `(⌊s·x⌋ + 1/2)/s` lies within
`1/(2s) + 2^-52·(1 + 2^-54)·(|x| + 1/(2s)) + 2^-1075·(1 + (1 + 2^-53)/s)` of `x`. -/
theorem encode_decode_zero {fl : ℝ → ℝ} (hfl : IsRoundNearest fl) {ri : ℝ → ℤ}
    (hri : IsRoundInt ri) {N : ℕ} {ms : ℝ} {es : ℤ} (hms : IsMantissa ms) {mx : ℕ → ℝ}
    {ex : ℕ → ℤ} (hmx : ∀ m < N, IsMantissa (mx m))
    (hguard : ∀ m < N, |levelHi fl ms es (mx m) (ex m)| < 2 ^ 53) {s lo hi : ℝ}
    (hs : s = ms * 2 ^ es) (hs0 : 0 < s) {x : ℕ → ℝ} (hx : ∀ m < N, x m = mx m * 2 ^ ex m)
    (hD : ∀ m < N, lo ≤ x m ∧ x m ≤ hi) {V z : ℤ} (hV62 : V < 2 ^ 62) (hz62 : |z| < 2 ^ 62)
    (hz : -⌊s * lo⌋ ≤ z ∧ z ≤ V - 1 - ⌊s * hi⌋) {L : ℕ → Int64}
    (hL : L = levelsI64 fl ri ms es mx ex fun _ => 1 / 2) {q : ℕ → Int64}
    (hq : q = encodeTokens V z [] L 0) :
    (∀ n < N, q n = tokenSum [] L 0 n + Int64.ofInt z ∧ (q n).toInt = ⌊s * x n⌋ + z ∧
      0 ≤ (q n).toInt ∧ (q n).toInt ≤ V - 1) ∧
      ∀ m < N, |decodeOut fl s 0 x (fun _ => 1 / 2) (decodeLevels z [] L 0 q) m - x m| ≤
        1 / (2 * s) + 2 ^ (-52 : ℤ) * (1 + 2 ^ (-54 : ℤ)) * (|x m| + 1 / (2 * s)) +
          2 ^ (-1075 : ℤ) * (1 + (1 + 2 ^ (-53 : ℤ)) / s) := by
  subst hL hq
  set L := levelsI64 fl ri ms es mx ex fun _ => 1 / 2
  -- the tokens are `⌊s·x⌋ + z`, in `[0, V - 1]`, and neither clip changes them
  have htok : ∀ n < N, encodeTokens V z [] L 0 n = tokenSum [] L 0 n + Int64.ofInt z ∧
      (encodeTokens V z [] L 0 n).toInt = ⌊s * x n⌋ + z ∧
      0 ≤ ⌊s * x n⌋ + z ∧ ⌊s * x n⌋ + z ≤ V - 1 := fun n hn => by
    obtain ⟨h1, h2⟩ := floor_add_mem hs0.le (hD n hn) hz
    have h := toInt_tokenSum_zero hfl hri hms (hmx n hn) (hguard n hn) hs (hx n hn)
    obtain ⟨he, he'⟩ := encodeTokens_eq hV62 hz62 h h1 h2
    exact ⟨he, he', h1, h2⟩
  refine ⟨fun n hn => ?_, fun m hm => ?_⟩
  · obtain ⟨h1, h2, h3, h4⟩ := htok n hn
    rw [h2]
    exact ⟨h1, rfl, h3, h4⟩
  -- `decode` takes the steps for the levels
  obtain ⟨-, h2, -, -⟩ := htok m hm
  obtain ⟨-, hR⟩ := toInt_levelsI64_half hfl hri hms (hmx m hm) (hguard m hm) hs (hx m hm)
  obtain ⟨r1, r2⟩ := abs_le.mp hR
  have hdec : (decodeLevels z [] L 0 (encodeTokens V z [] L 0) m).toInt = ⌊s * x m⌋ := by
    refine eq_of_modEq_of_inInt64 ?_ (inInt64_toInt _) ⟨by omega, by omega⟩
    unfold decodeLevels invertI64 decodeSteps
    calc _ ≡ (encodeTokens V z [] L 0 m - Int64.ofInt z).toInt - ((0 : ℕ → Int64) m).toInt
          [ZMOD 2 ^ 64] := toInt_sub_modEq _ _
      _ ≡ ((encodeTokens V z [] L 0 m).toInt - z) - ((0 : ℕ → Int64) m).toInt
          [ZMOD 2 ^ 64] :=
          ((toInt_sub_modEq _ _).trans ((Int.ModEq.refl _).sub (toInt_ofInt_modEq _))).sub_right _
      _ = ⌊s * x m⌋ := by rw [h2, Pi.zero_apply, Int64.toInt_zero]; ring
  unfold decodeOut
  rw [ite_eq_right (Nat.not_lt_zero m), hdec, floor_eq_roundHalfUp]
  exact abs_decode_sub_le_closed hfl hs0 (x m) (1 / 2)
    (by rw [← floor_eq_roundHalfUp]; exact hR)

end DiffGPT
