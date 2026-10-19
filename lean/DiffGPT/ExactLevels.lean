module

public import Mathlib
public import DiffGPT.LevelGrid
public import DiffGPT.Float64

/-!
# Exact levels in float64

`_levels` computes the levels `R = ⌊s·x - F + 1/2⌋ = ⌊s·x - c⌋`, `c = F - 1/2`, of floats `s`
and `x` without rounding `s·x` (`levels_eq`). `np.frexp` splits `s = m_s·2^e_s` and
`x = m_x·2^e_x` into mantissas `1/2 ≤ |m| < 1`, or `0` (`IsMantissa`), and then

    p0 = fl (m_s·m_x),  e0 = m_s·m_x - p0    Dekker's product, exact (`twoProductErr_eq`)
    E = max (e_s + e_x) (-53)                 `levelExp`
    p = p0·2^E,  e = e0·2^E                   exact: `np.ldexp` of floats that stay floats
    c = fl (F - 1/2)                          exact for `F = N / 2^G`, `G ≤ 53`

so `p + e = m_s·m_x·2^E`, which is `s·x` unless `E` was raised, and `p` is a float nearest to
it. `levelOf` reads `⌊p + e - c⌋` off `p`, `e` and `c` with float operations alone
(`levelOf_eq`). Where `|p| ≥ 1/2`, `n = rint(p)` and `w = (p - n) - c` are exact, `|e| ≤ 1/2`,
and each `(w - j) + e ≥ 0`, `j ∈ {-1, 0, 1}`, rounds to the sign of the exact sum. Where
`|p| < 1/2`, a float `c` lies between `p` and `p + e` only if `p = c`, so `p` decides, or the
sign of `e` where `p = c`. Where `E` was raised, `s·x` and `m_s·m_x·2^-53` both lie below
`2^-53` with one sign, which alone decides the level (`floor_sub_eq_of_abs_lt`).

Under `_levels`' guard `|p| < 2^b`, `b ≤ 53`, the levels lie in `[-2^b - 1, 2^b]`
(`levels_mem`) and within `2^53`, so float64 holds each of them (`abs_levels_le`).

`np.rint` is modelled as a rounding to a nearest integer, ties either way (`IsRoundInt`), and
`np.ldexp` as the exact product by `2^E`, which it is wherever that product is a float.
-/

@[expose] public section

namespace DiffGPT

/-! ### `np.frexp` and `np.rint` -/

/-- `np.frexp`'s mantissas: floats with `1/2 ≤ |m| < 1`, or `0`. -/
def IsMantissa (m : ℝ) : Prop := IsF64 m ∧ (m = 0 ∨ 1 / 2 ≤ |m| ∧ |m| < 1)

/-- A mantissa is `M / 2^53` with `2^52 ≤ |M| < 2^53`, or `0`. -/
lemma IsMantissa.exists_intCast_div {m : ℝ} (hm : IsMantissa m) :
    ∃ M : ℤ, m = M / 2 ^ 53 ∧ |M| < 2 ^ 53 ∧ (M = 0 ∨ 2 ^ 52 ≤ |M|) := by
  obtain ⟨hf, rfl | ⟨h1, h2⟩⟩ := hm
  · exact ⟨0, by simp, by norm_num, Or.inl rfl⟩
  obtain ⟨M, rfl⟩ := hf.exists_intCast_div (G := 53) (by norm_num; linarith)
  have h53 : (0 : ℝ) < 2 ^ 53 := by positivity
  rw [abs_div, abs_of_pos h53] at h1 h2
  refine ⟨M, rfl, ?_, Or.inr ?_⟩
  · have : |(M : ℝ)| < 2 ^ 53 := by rwa [div_lt_one h53] at h2
    exact_mod_cast this
  · have : (2 : ℝ) ^ 52 ≤ |(M : ℝ)| := by rw [le_div_iff₀ h53] at h1; linarith
    exact_mod_cast this

/-- A mantissa lies in `(-1, 1)`. -/
lemma IsMantissa.abs_lt_one {m : ℝ} (hm : IsMantissa m) : |m| < 1 := by
  rcases hm.2 with rfl | ⟨-, h⟩
  · norm_num
  · exact h

/-- `np.rint`: rounds to a nearest integer, ties either way (numpy breaks them to even). -/
def IsRoundInt (ri : ℝ → ℤ) : Prop := ∀ y, |y - ri y| ≤ 1 / 2

/-! ### Dekker's product -/

/-- `_split`'s high half: `np.ldexp(np.rint(np.ldexp(m, 26)), -26)`, whose scalings are
exact. -/
noncomputable def splitHi (ri : ℝ → ℤ) (m : ℝ) : ℝ := (ri (m * 2 ^ 26) : ℝ) / 2 ^ 26

/-- `_two_product`'s error: `(((a_hi*b_hi - p) + a_hi*b_lo) + a_lo*b_hi) + a_lo*b_lo` with
`p = a*b`, the low halves `m - hi`, and every operation rounded. -/
noncomputable def twoProductErr (fl : ℝ → ℝ) (ri : ℝ → ℤ) (a b : ℝ) : ℝ :=
  fl (fl (fl (fl (fl (splitHi ri a * splitHi ri b) - fl (a * b))
    + fl (splitHi ri a * fl (b - splitHi ri b)))
    + fl (fl (a - splitHi ri a) * splitHi ri b))
    + fl (fl (a - splitHi ri a) * fl (b - splitHi ri b)))

namespace IsRoundNearest

variable {fl : ℝ → ℝ} (hfl : IsRoundNearest fl)
include hfl

/-- `_split` of `A / 2^53`, `|A| < 2^53`: `hi = H / 2^26` and `lo = L / 2^53`, exactly, with
`A = H·2^27 + L` and `|H|, |L| ≤ 2^26`. -/
lemma split_eq {ri : ℝ → ℤ} (hri : IsRoundInt ri) {A : ℤ} (hA : |A| < 2 ^ 53) :
    ∃ H L : ℤ, |H| ≤ 2 ^ 26 ∧ |L| ≤ 2 ^ 26 ∧ A = H * 2 ^ 27 + L ∧
      splitHi ri (A / 2 ^ 53) = H / 2 ^ 26 ∧
      fl ((A : ℝ) / 2 ^ 53 - splitHi ri (A / 2 ^ 53)) = L / 2 ^ 53 := by
  set H := ri ((A : ℝ) / 2 ^ 53 * 2 ^ 26) with hH
  have hr := hri ((A : ℝ) / 2 ^ 53 * 2 ^ 26)
  rw [← hH] at hr
  have hL : |A - H * 2 ^ 27| ≤ 2 ^ 26 := by
    have h1 : ((A - H * 2 ^ 27 : ℤ) : ℝ) = 2 ^ 27 * ((A : ℝ) / 2 ^ 53 * 2 ^ 26 - H) := by
      push_cast
      ring
    have h2 : |((A - H * 2 ^ 27 : ℤ) : ℝ)| ≤ 2 ^ 26 := by
      rw [h1, abs_mul, abs_of_pos (by positivity : (0 : ℝ) < 2 ^ 27)]
      calc (2 : ℝ) ^ 27 * |(A : ℝ) / 2 ^ 53 * 2 ^ 26 - H| ≤ 2 ^ 27 * (1 / 2) := by gcongr
        _ = 2 ^ 26 := by norm_num
    exact_mod_cast h2
  refine ⟨H, A - H * 2 ^ 27, ?_, hL, by ring, rfl, ?_⟩
  · rw [abs_le] at hL ⊢
    rw [abs_lt] at hA
    omega
  · have h1 : (A : ℝ) / 2 ^ 53 - H / 2 ^ 26 = ((A - H * 2 ^ 27 : ℤ) : ℝ) / 2 ^ 53 := by
      push_cast
      ring
    rw [splitHi, ← hH, h1]
    exact hfl.eq_self (isF64_intCast_div_two_pow (hL.trans (by norm_num)) (by norm_num))

/-- Dekker's product is exact on `A / 2^53` and `B / 2^53`: given `fl (a·b) = P / 2^54` within
`2^-53` of `a·b`, every partial sum is a multiple of its grid within `2^53` steps, so float64
holds it, and the error comes out as `a·b - P / 2^54`. -/
lemma twoProductErr_eq_intCast_div {ri : ℝ → ℤ} (hri : IsRoundInt ri) {A B P : ℤ}
    (hA : |A| < 2 ^ 53) (hB : |B| < 2 ^ 53) (hP : fl ((A : ℝ) / 2 ^ 53 * (B / 2 ^ 53)) = P / 2 ^ 54)
    (hAB : |A * B - 2 ^ 52 * P| ≤ 2 ^ 53) :
    twoProductErr fl ri (A / 2 ^ 53) (B / 2 ^ 53) = ((A * B - 2 ^ 52 * P : ℤ) : ℝ) / 2 ^ 106 := by
  obtain ⟨Ha, La, hHa, hLa, rfl, hsa, hla⟩ := hfl.split_eq hri hA
  obtain ⟨Hb, Lb, hHb, hLb, rfl, hsb, hlb⟩ := hfl.split_eq hri hB
  have hprod {u v : ℤ} (hu : |u| ≤ 2 ^ 26) (hv : |v| ≤ 2 ^ 26) : |u * v| ≤ 2 ^ 52 := by
    rw [abs_mul]
    calc |u| * |v| ≤ (2 : ℤ) ^ 26 * 2 ^ 26 := mul_le_mul hu hv (abs_nonneg _) (by positivity)
      _ = 2 ^ 52 := by norm_num
  have ex (N : ℤ) (G : ℕ) (hN : |N| ≤ 2 ^ 53) (hG : G ≤ 1074) :
      fl ((N : ℝ) / 2 ^ G) = N / 2 ^ G :=
    hfl.eq_self (isF64_intCast_div_two_pow hN hG)
  obtain ⟨a1, a2⟩ := abs_le.mp (hprod hHa hHb)
  obtain ⟨b1, b2⟩ := abs_le.mp (hprod hHa hLb)
  obtain ⟨c1, c2⟩ := abs_le.mp (hprod hLa hHb)
  obtain ⟨d1, d2⟩ := abs_le.mp (hprod hLa hLb)
  obtain ⟨e1, e2⟩ := abs_le.mp hAB
  have s1 : fl ((Ha : ℝ) / 2 ^ 26 * (Hb / 2 ^ 26)) = ((Ha * Hb : ℤ) : ℝ) / 2 ^ 52 := by
    rw [show (Ha : ℝ) / 2 ^ 26 * (Hb / 2 ^ 26) = ((Ha * Hb : ℤ) : ℝ) / 2 ^ 52 by push_cast; ring]
    exact ex _ _ (hprod hHa hHb |>.trans (by norm_num)) (by norm_num)
  have s2 : fl ((Ha : ℝ) / 2 ^ 26 * (Lb / 2 ^ 53)) = ((Ha * Lb : ℤ) : ℝ) / 2 ^ 79 := by
    rw [show (Ha : ℝ) / 2 ^ 26 * (Lb / 2 ^ 53) = ((Ha * Lb : ℤ) : ℝ) / 2 ^ 79 by push_cast; ring]
    exact ex _ _ (hprod hHa hLb |>.trans (by norm_num)) (by norm_num)
  have s3 : fl ((La : ℝ) / 2 ^ 53 * (Hb / 2 ^ 26)) = ((La * Hb : ℤ) : ℝ) / 2 ^ 79 := by
    rw [show (La : ℝ) / 2 ^ 53 * (Hb / 2 ^ 26) = ((La * Hb : ℤ) : ℝ) / 2 ^ 79 by push_cast; ring]
    exact ex _ _ (hprod hLa hHb |>.trans (by norm_num)) (by norm_num)
  have s4 : fl ((La : ℝ) / 2 ^ 53 * (Lb / 2 ^ 53)) = ((La * Lb : ℤ) : ℝ) / 2 ^ 106 := by
    rw [show (La : ℝ) / 2 ^ 53 * (Lb / 2 ^ 53) = ((La * Lb : ℤ) : ℝ) / 2 ^ 106 by push_cast; ring]
    exact ex _ _ (hprod hLa hLb |>.trans (by norm_num)) (by norm_num)
  have t1 : fl (((Ha * Hb : ℤ) : ℝ) / 2 ^ 52 - P / 2 ^ 54) =
      ((4 * (Ha * Hb) - P : ℤ) : ℝ) / 2 ^ 54 := by
    rw [show ((Ha * Hb : ℤ) : ℝ) / 2 ^ 52 - P / 2 ^ 54 = ((4 * (Ha * Hb) - P : ℤ) : ℝ) / 2 ^ 54 by
      push_cast; ring]
    exact ex _ _ (abs_le.mpr ⟨by linarith, by linarith⟩) (by norm_num)
  have t2 : fl (((4 * (Ha * Hb) - P : ℤ) : ℝ) / 2 ^ 54 + ((Ha * Lb : ℤ) : ℝ) / 2 ^ 79) =
      ((2 ^ 25 * (4 * (Ha * Hb) - P) + Ha * Lb : ℤ) : ℝ) / 2 ^ 79 := by
    rw [show ((4 * (Ha * Hb) - P : ℤ) : ℝ) / 2 ^ 54 + ((Ha * Lb : ℤ) : ℝ) / 2 ^ 79 =
      ((2 ^ 25 * (4 * (Ha * Hb) - P) + Ha * Lb : ℤ) : ℝ) / 2 ^ 79 by push_cast; ring]
    exact ex _ _ (abs_le.mpr ⟨by linarith, by linarith⟩) (by norm_num)
  have t3 : fl (((2 ^ 25 * (4 * (Ha * Hb) - P) + Ha * Lb : ℤ) : ℝ) / 2 ^ 79 +
      ((La * Hb : ℤ) : ℝ) / 2 ^ 79) =
      ((2 ^ 25 * (4 * (Ha * Hb) - P) + Ha * Lb + La * Hb : ℤ) : ℝ) / 2 ^ 79 := by
    rw [show ((2 ^ 25 * (4 * (Ha * Hb) - P) + Ha * Lb : ℤ) : ℝ) / 2 ^ 79 +
      ((La * Hb : ℤ) : ℝ) / 2 ^ 79 =
      ((2 ^ 25 * (4 * (Ha * Hb) - P) + Ha * Lb + La * Hb : ℤ) : ℝ) / 2 ^ 79 by push_cast; ring]
    exact ex _ _ (abs_le.mpr ⟨by linarith, by linarith⟩) (by norm_num)
  have t4 : ((2 ^ 25 * (4 * (Ha * Hb) - P) + Ha * Lb + La * Hb : ℤ) : ℝ) / 2 ^ 79 +
      ((La * Lb : ℤ) : ℝ) / 2 ^ 106 =
      (((Ha * 2 ^ 27 + La) * (Hb * 2 ^ 27 + Lb) - 2 ^ 52 * P : ℤ) : ℝ) / 2 ^ 106 := by
    push_cast
    ring
  unfold twoProductErr
  rw [hla, hlb, hsa, hsb, hP, s1, s2, s3, s4, t1, t2, t3, t4]
  exact ex _ _ hAB (by norm_num)

/-- `fl (a·b)` for mantissas `a = A / 2^53`, `b = B / 2^53` is `P / 2^54`, within `2^53` steps of
`2^-106` from `a·b`, and `|P| ≥ 2^52` unless `a·b = 0`. -/
lemma exists_fl_mul_eq {A B : ℤ} (hA : |A| < 2 ^ 53) (hB : |B| < 2 ^ 53)
    (hA0 : A = 0 ∨ 2 ^ 52 ≤ |A|) (hB0 : B = 0 ∨ 2 ^ 52 ≤ |B|) :
    ∃ P : ℤ, fl ((A : ℝ) / 2 ^ 53 * (B / 2 ^ 53)) = P / 2 ^ 54 ∧
      |A * B - 2 ^ 52 * P| ≤ 2 ^ 53 ∧ (A * B ≠ 0 → 2 ^ 52 ≤ |P|) := by
  have hab : (A : ℝ) / 2 ^ 53 * (B / 2 ^ 53) = ((A * B : ℤ) : ℝ) / 2 ^ 106 := by
    push_cast
    ring
  rw [hab]
  by_cases h0 : A * B = 0
  · refine ⟨0, ?_, by simp [h0], fun h => absurd h0 h⟩
    rw [h0]
    simp [hfl.eq_self IsF64.zero]
  have hA' : 2 ^ 52 ≤ |A| := hA0.resolve_left fun h => h0 (by rw [h, zero_mul])
  have hB' : 2 ^ 52 ≤ |B| := hB0.resolve_left fun h => h0 (by rw [h, mul_zero])
  have hlo : (2 : ℤ) ^ 104 ≤ |A * B| := by
    rw [abs_mul]
    calc (2 : ℤ) ^ 104 = 2 ^ 52 * 2 ^ 52 := by norm_num
      _ ≤ |A| * |B| := mul_le_mul hA' hB' (by positivity) (abs_nonneg _)
  have hhi : |A * B| < 2 ^ 106 := by
    rw [abs_mul]
    calc |A| * |B| < 2 ^ 53 * 2 ^ 53 := mul_lt_mul'' hA hB (abs_nonneg _) (abs_nonneg _)
      _ = 2 ^ 106 := by norm_num
  have h106 : (0 : ℝ) < 2 ^ 106 := by positivity
  set ab := ((A * B : ℤ) : ℝ) / 2 ^ 106 with hab_def
  have hab4 : 1 / 4 ≤ |ab| := by
    rw [hab_def, abs_div, abs_of_pos h106, le_div_iff₀ h106]
    have : (2 : ℝ) ^ 104 ≤ |((A * B : ℤ) : ℝ)| := by rw [← Int.cast_abs]; exact_mod_cast hlo
    linarith
  have hab1 : |ab| < 1 := by
    rw [hab_def, abs_div, abs_of_pos h106, div_lt_one h106, ← Int.cast_abs]
    exact_mod_cast hhi
  have hq : IsF64 (1 / 4 : ℝ) := ⟨1, -2, by norm_num, by norm_num, by norm_num⟩
  have hfl4 : 1 / 4 ≤ |fl ab| := hfl.le_abs hq hab4
  obtain ⟨P, hP⟩ := (hfl.isF64 ab).exists_intCast_div (G := 54) (by norm_num; linarith)
  refine ⟨P, hP, ?_, fun _ => ?_⟩
  · have herr := hfl.abs_sub_le ab
    unfold rndErr at herr
    have hz1 : (2 : ℝ) ^ (-53 : ℤ) = 1 / 2 ^ 53 := by norm_num
    have hz2 : (2 : ℝ) ^ (-1075 : ℤ) < 1 / 2 ^ 106 := by
      rw [one_div, ← zpow_natCast, ← zpow_neg]
      exact zpow_lt_zpow_right₀ (by norm_num) (by norm_num)
    have h1 : |fl ab - ab| < 1 / 2 ^ 53 + 1 / 2 ^ 106 := by
      rw [hz1] at herr
      have : 1 / 2 ^ 53 * |ab| ≤ 1 / 2 ^ 53 := by
        have := mul_le_mul_of_nonneg_left hab1.le (by positivity : (0 : ℝ) ≤ 1 / 2 ^ 53)
        linarith
      linarith
    have h2 : fl ab - ab = ((2 ^ 52 * P - A * B : ℤ) : ℝ) / 2 ^ 106 := by
      rw [hP, hab_def]
      push_cast
      ring
    rw [h2, abs_div, abs_of_pos h106, div_lt_iff₀ h106] at h1
    have h3 : |((2 ^ 52 * P - A * B : ℤ) : ℝ)| < 2 ^ 53 + 1 :=
      h1.trans_eq (by norm_num)
    have h4 : |2 ^ 52 * P - A * B| < 2 ^ 53 + 1 := by exact_mod_cast h3
    rw [abs_sub_comm] at h4
    omega
  · rw [hP, abs_div, abs_of_pos (by positivity : (0 : ℝ) < 2 ^ 54),
      le_div_iff₀ (by positivity)] at hfl4
    have : (2 : ℝ) ^ 52 ≤ |(P : ℝ)| := by linarith
    rw [← Int.cast_abs] at this
    exact_mod_cast this

end IsRoundNearest

section

variable {fl : ℝ → ℝ} (hfl : IsRoundNearest fl)
include hfl

/-- Dekker's product is exact on mantissas: `twoProductErr a b = a·b - fl (a·b)`. -/
theorem twoProductErr_eq {ri : ℝ → ℤ} (hri : IsRoundInt ri) {a b : ℝ} (ha : IsMantissa a)
    (hb : IsMantissa b) : twoProductErr fl ri a b = a * b - fl (a * b) := by
  obtain ⟨A, rfl, hA, hA0⟩ := ha.exists_intCast_div
  obtain ⟨B, rfl, hB, hB0⟩ := hb.exists_intCast_div
  obtain ⟨P, hP, hAB, -⟩ := hfl.exists_fl_mul_eq hA hB hA0 hB0
  rw [hfl.twoProductErr_eq_intCast_div hri hA hB hP hAB, hP]
  push_cast
  ring

end

/-! ### The level of `p + e` from `p`, `e` and `c` -/

/-- If `p` lies no further from `v = p + e` than `y` does, `y ≤ v` gives `y ≤ p`. -/
lemma le_of_abs_le_abs_sub {p e y : ℝ} (h : |e| ≤ |y - (p + e)|) (hy : y ≤ p + e) : y ≤ p := by
  by_contra! hp
  rw [abs_of_nonpos (by linarith : y - (p + e) ≤ 0)] at h
  have := le_abs_self e
  linarith

/-- If `p` lies no further from `v = p + e` than `y` does, `v ≤ y` gives `p ≤ y`. -/
lemma le_of_abs_le_abs_sub' {p e y : ℝ} (h : |e| ≤ |y - (p + e)|) (hy : p + e ≤ y) : p ≤ y := by
  by_contra! hp
  rw [abs_of_nonneg (by linarith : 0 ≤ y - (p + e))] at h
  have := neg_abs_le e
  linarith

/-- `p` errs from `v = p + e` by at most `1/2` when no integer within `2^53` lies nearer to `v`
and `|p| < 2^53`. -/
lemma abs_le_half_of_forall_int {p e : ℝ} (hp53 : |p| < 2 ^ 53)
    (hint : ∀ N : ℤ, |N| ≤ 2 ^ 53 → |e| ≤ |N - (p + e)|) : |e| ≤ 1 / 2 := by
  obtain ⟨hp1, hp2⟩ := abs_lt.mp hp53
  have hv : |p + e| ≤ 2 ^ 53 := by
    rw [abs_le]
    constructor
    · by_contra! hlt
      have := le_of_abs_le_abs_sub' (hint (-2 ^ 53) (by norm_num)) (by push_cast; linarith)
      push_cast at this
      linarith
    · by_contra! hlt
      have := le_of_abs_le_abs_sub (hint (2 ^ 53) (by norm_num)) (by push_cast; linarith)
      push_cast at this
      linarith
  have hr := abs_sub_round (p + e)
  have hN : |round (p + e)| ≤ 2 ^ 53 := by
    have h1 : |((round (p + e) : ℤ) : ℝ)| < 2 ^ 53 + 1 :=
      calc |((round (p + e) : ℤ) : ℝ)| = |(p + e) - ((p + e) - round (p + e))| := by
            congr 1
            ring
        _ ≤ |p + e| + |(p + e) - round (p + e)| := abs_sub _ _
        _ < 2 ^ 53 + 1 := by linarith
    have h2 : |round (p + e)| < 2 ^ 53 + 1 := by exact_mod_cast h1
    omega
  calc |e| ≤ |(round (p + e) : ℝ) - (p + e)| := hint _ hN
    _ = |p + e - round (p + e)| := abs_sub_comm _ _
    _ ≤ 1 / 2 := hr

/-- `⌊t⌋ = -2 + [-1 ≤ t] + [0 ≤ t] + [1 ≤ t]` on `[-2, 2)`. -/
lemma floor_eq_ite {t : ℝ} (h1 : -2 ≤ t) (h2 : t < 2) :
    ⌊t⌋ = -2 + (if -1 ≤ t then 1 else 0) + (if 0 ≤ t then 1 else 0) +
      (if 1 ≤ t then 1 else 0) := by
  rw [Int.floor_eq_iff]
  split_ifs <;> push_cast <;> constructor <;> linarith

/-- The offset `c = C / 2^53` lies in `[-1/2, 1/2)`. -/
lemma offset_mem {C : ℤ} (hC1 : -2 ^ 52 ≤ C) (hC2 : C < 2 ^ 52) :
    -(1 / 2) ≤ (C : ℝ) / 2 ^ 53 ∧ (C : ℝ) / 2 ^ 53 < 1 / 2 := by
  have h53 : (0 : ℝ) < 2 ^ 53 := by positivity
  have h1 : ((-2 ^ 52 : ℤ) : ℝ) ≤ C := by exact_mod_cast hC1
  have h2 : (C : ℝ) < ((2 ^ 52 : ℤ) : ℝ) := by exact_mod_cast hC2
  push_cast at h1 h2
  constructor
  · rw [le_div_iff₀ h53]
    linarith
  · rw [div_lt_iff₀ h53]
    linarith

namespace IsRoundNearest

variable {fl : ℝ → ℝ} (hfl : IsRoundNearest fl)
include hfl

/-- `(w - j) + e >= 0` in float64, for `w - j = A / 2^53` and a float `e` with `|e| ≤ 1/2`:
`fl (fl (A / 2^53) + e)` has the sign of `A / 2^53 + e`. Where `|A / 2^53| < 1` it is a float,
and the sum of two floats rounds to its sign; elsewhere it decides the sign alone. -/
lemma nonneg_add_iff_of_abs_le_half {A : ℤ} {e : ℝ} (he : IsF64 e) (he2 : |e| ≤ 1 / 2) :
    0 ≤ fl (fl ((A : ℝ) / 2 ^ 53) + e) ↔ 0 ≤ (A : ℝ) / 2 ^ 53 + e := by
  obtain ⟨he₁, he₂⟩ := abs_le.mp he2
  have hone : IsF64 (1 : ℝ) := by exact_mod_cast isF64_intCast (n := 1) (by norm_num)
  have hhalf : IsF64 (1 / 2 : ℝ) := ⟨1, -1, by norm_num, by norm_num, by norm_num⟩
  rcases le_or_gt 1 ((A : ℝ) / 2 ^ 53) with h | h
  · have h1 := hfl.mono h
    rw [hfl.eq_self hone] at h1
    have h2 := hfl.mono (show 1 / 2 ≤ fl ((A : ℝ) / 2 ^ 53) + e by linarith)
    rw [hfl.eq_self hhalf] at h2
    exact iff_of_true (by linarith) (by linarith)
  rcases le_or_gt ((A : ℝ) / 2 ^ 53) (-1) with h' | h'
  · have h1 := hfl.mono h'
    rw [hfl.eq_self hone.neg] at h1
    have h2 := hfl.mono (show fl ((A : ℝ) / 2 ^ 53) + e ≤ -(1 / 2) by linarith)
    rw [hfl.eq_self hhalf.neg] at h2
    exact iff_of_false (by linarith) (by linarith)
  have ha : IsF64 ((A : ℝ) / 2 ^ 53) := by
    refine isF64_intCast_div_two_pow (le_of_lt ?_) (by norm_num)
    have : |(A : ℝ) / 2 ^ 53| < 1 := abs_lt.mpr ⟨h', h⟩
    rw [abs_div, abs_of_pos (by positivity : (0 : ℝ) < 2 ^ 53), div_lt_one (by positivity)] at this
    exact_mod_cast this
  rw [hfl.eq_self ha]
  exact hfl.nonneg_add_iff ha he

end IsRoundNearest

/-- One of `_levels`' `above` terms, `(w - j) + e >= 0`, every operation rounded. -/
noncomputable def above (fl : ℝ → ℝ) (w e : ℝ) (j : ℤ) : ℤ :=
  if 0 ≤ fl (fl (w - j) + e) then 1 else 0

/-- `_levels` from `p`, `e` and the offset `c = F - 1/2`: where `|p| ≥ 1/2`, `n - 2 + above`
with `n = rint(p)` and `w = (p - n) - c`, and `small` elsewhere. -/
noncomputable def levelOf (fl : ℝ → ℝ) (ri : ℝ → ℤ) (p e c : ℝ) : ℤ :=
  if 1 / 2 ≤ |p| then
    ri p - 2 + (above fl (fl (fl (p - ri p) - c)) e (-1) + above fl (fl (fl (p - ri p) - c)) e 0
      + above fl (fl (fl (p - ri p) - c)) e 1)
  else if c < p ∨ (p = c ∧ 0 ≤ e) then 0 else -1

section

variable {fl : ℝ → ℝ} (hfl : IsRoundNearest fl)
include hfl

/-- `levelOf` is `⌊p + e - c⌋` for floats `p` and `e`, `|p| < 2^53`, and the offset
`c = C / 2^53` in `[-1/2, 1/2)`, when `p` lies no further from `v = p + e` than `c`, `±1/2`
and the integers within `2^53` do. Then `|e| ≤ 1/2`, and `|p| < 1/2` gives `|v| < 1/2`. -/
theorem levelOf_eq {ri : ℝ → ℤ} (hri : IsRoundInt ri) {p e : ℝ} (hp : IsF64 p) (he : IsF64 e)
    (hp53 : |p| < 2 ^ 53) {C : ℤ} (hC1 : -2 ^ 52 ≤ C) (hC2 : C < 2 ^ 52)
    (hc : |e| ≤ |C / 2 ^ 53 - (p + e)|) (hhalf : |e| ≤ |1 / 2 - (p + e)|)
    (hhalf' : |e| ≤ |-(1 / 2) - (p + e)|) (hint : ∀ N : ℤ, |N| ≤ 2 ^ 53 → |e| ≤ |N - (p + e)|) :
    levelOf fl ri p e (C / 2 ^ 53) = ⌊p + e - C / 2 ^ 53⌋ := by
  have h53 : (0 : ℝ) < 2 ^ 53 := by positivity
  obtain ⟨hc1, hc2⟩ := offset_mem hC1 hC2
  have he2 : |e| ≤ 1 / 2 := abs_le_half_of_forall_int hp53 hint
  obtain ⟨he₁, he₂⟩ := abs_le.mp he2
  unfold levelOf
  split_ifs with hbig hcond
  · -- `|p| ≥ 1/2`: `p = P / 2^53`, and `p - n` and `w` are exact
    obtain ⟨P, rfl⟩ := hp.exists_intCast_div (G := 53) (by norm_num; linarith)
    set n := ri ((P : ℝ) / 2 ^ 53) with hn_def
    have hn := hri ((P : ℝ) / 2 ^ 53)
    rw [← hn_def] at hn
    obtain ⟨hn1, hn2⟩ := abs_le.mp hn
    have hPn : |P - n * 2 ^ 53| ≤ 2 ^ 52 := by
      have h1 : ((P - n * 2 ^ 53 : ℤ) : ℝ) = 2 ^ 53 * ((P : ℝ) / 2 ^ 53 - n) := by
        push_cast
        field_simp
        ring
      have h2 : |((P - n * 2 ^ 53 : ℤ) : ℝ)| ≤ 2 ^ 52 := by
        rw [h1, abs_mul, abs_of_pos h53]
        calc (2 : ℝ) ^ 53 * |(P : ℝ) / 2 ^ 53 - n| ≤ 2 ^ 53 * (1 / 2) := by gcongr
          _ = 2 ^ 52 := by norm_num
      exact_mod_cast h2
    have hsub : fl ((P : ℝ) / 2 ^ 53 - n) = ((P - n * 2 ^ 53 : ℤ) : ℝ) / 2 ^ 53 := by
      have : (P : ℝ) / 2 ^ 53 - n = ((P - n * 2 ^ 53 : ℤ) : ℝ) / 2 ^ 53 := by
        push_cast
        field_simp
        ring
      rw [this]
      exact hfl.eq_self (isF64_intCast_div_two_pow (hPn.trans (by norm_num)) (by norm_num))
    set W := P - n * 2 ^ 53 - C with hW
    have hWv : (W : ℝ) / 2 ^ 53 = (P : ℝ) / 2 ^ 53 - n - C / 2 ^ 53 := by
      rw [hW]
      push_cast
      field_simp
      ring
    have hw : fl (((P - n * 2 ^ 53 : ℤ) : ℝ) / 2 ^ 53 - C / 2 ^ 53) = (W : ℝ) / 2 ^ 53 := by
      have : ((P - n * 2 ^ 53 : ℤ) : ℝ) / 2 ^ 53 - C / 2 ^ 53 = (W : ℝ) / 2 ^ 53 := by
        rw [hW]
        push_cast
        ring
      rw [this]
      refine hfl.eq_self (isF64_intCast_div_two_pow ?_ (by norm_num))
      rw [abs_le] at hPn ⊢
      omega
    rw [hsub, hw]
    have habove (j : ℤ) :
        above fl ((W : ℝ) / 2 ^ 53) e j = if (j : ℝ) ≤ W / 2 ^ 53 + e then 1 else 0 := by
      have key : 0 ≤ fl (fl ((W : ℝ) / 2 ^ 53 - j) + e) ↔ (j : ℝ) ≤ W / 2 ^ 53 + e := by
        have : (W : ℝ) / 2 ^ 53 - j = ((W - j * 2 ^ 53 : ℤ) : ℝ) / 2 ^ 53 := by
          push_cast
          field_simp
          ring
        rw [this, hfl.nonneg_add_iff_of_abs_le_half he he2, ← this]
        constructor <;> intro h <;> linarith
      unfold above
      by_cases hj : (j : ℝ) ≤ W / 2 ^ 53 + e
      · simp [key.mpr hj, hj]
      · simp [mt key.mp hj, hj]
    rw [habove, habove, habove]
    have hfloor : ⌊(P : ℝ) / 2 ^ 53 + e - C / 2 ^ 53⌋ = n + ⌊(W : ℝ) / 2 ^ 53 + e⌋ := by
      rw [← Int.floor_intCast_add]
      congr 1
      rw [hWv]
      ring
    rw [hfloor, floor_eq_ite (by linarith) (by linarith)]
    push_cast
    ring
  all_goals
    obtain ⟨hp1, hp2⟩ := abs_lt.mp (not_le.mp hbig)
    have hv1 : -(1 / 2) < p + e := by
      by_contra! h
      have := le_of_abs_le_abs_sub' hhalf' h
      linarith
    have hv2 : p + e < 1 / 2 := by
      by_contra! h
      have := le_of_abs_le_abs_sub hhalf h
      linarith
    symm
    rw [Int.floor_eq_iff]
    push_cast
  · -- `|p| < 1/2` and `p > c`, or `p = c` and `e ≥ 0`: the level is `0`
    refine ⟨?_, by linarith⟩
    rcases hcond with h | ⟨h, h'⟩
    · by_contra! hlt
      have := le_of_abs_le_abs_sub' hc (by linarith)
      linarith
    · linarith
  · -- otherwise `-1`
    refine ⟨by linarith, ?_⟩
    obtain ⟨h1, h2⟩ := not_or.mp hcond
    rcases (not_lt.mp h1).lt_or_eq with h | h
    · by_contra! hge
      have := le_of_abs_le_abs_sub hc (by linarith)
      linarith
    · have : e < 0 := by
        by_contra! he0
        exact h2 ⟨h, he0⟩
      linarith

end

/-! ### `_levels` -/

/-- `_levels`' offset `c = F - 1/2` for `F = N / 2^G`, `0 ≤ N < 2^G`, `G ≤ 53`: `C / 2^53` with
`-2^52 ≤ C < 2^52`, so a float. -/
lemma exists_offset {G : ℕ} (hG : G ≤ 53) {N : ℤ} (hN0 : 0 ≤ N) (hN : N < 2 ^ G) :
    ∃ C : ℤ, (N : ℝ) / 2 ^ G - 1 / 2 = C / 2 ^ 53 ∧ -2 ^ 52 ≤ C ∧ C < 2 ^ 52 := by
  have h2 : (2 : ℤ) ^ 53 = 2 ^ G * 2 ^ (53 - G) := by rw [← pow_add, Nat.add_sub_cancel' hG]
  refine ⟨N * 2 ^ (53 - G) - 2 ^ 52, ?_, ?_, ?_⟩
  · have h : (N : ℝ) / 2 ^ G * 2 ^ 53 = N * 2 ^ (53 - G) := by
      rw [show (2 : ℝ) ^ 53 = 2 ^ G * 2 ^ (53 - G) by rw [← pow_add, Nat.add_sub_cancel' hG]]
      field_simp
    rw [eq_div_iff (by positivity), sub_mul, h]
    push_cast
    ring
  · have : 0 ≤ N * 2 ^ (53 - G) := mul_nonneg hN0 (by positivity)
    linarith
  · have : N * 2 ^ (53 - G) < 2 ^ G * 2 ^ (53 - G) := mul_lt_mul_of_pos_right hN (by positivity)
    linarith

/-- Below `2^-53` only the sign of `v` counts: `⌊v - C / 2^53⌋` is `0` if `C < 0`, or `C = 0`
and `v ≥ 0`, and `-1` otherwise. -/
lemma floor_sub_eq_of_abs_lt {v : ℝ} (hv : |v| < 1 / 2 ^ 53) {C : ℤ} (hC1 : -2 ^ 52 ≤ C)
    (hC2 : C < 2 ^ 52) :
    ⌊v - C / 2 ^ 53⌋ = if C < 0 ∨ (C = 0 ∧ 0 ≤ v) then 0 else -1 := by
  obtain ⟨hv1, hv2⟩ := abs_lt.mp hv
  obtain ⟨hc1, hc2⟩ := offset_mem hC1 hC2
  have h53 : (0 : ℝ) < 2 ^ 53 := by positivity
  have hsmall : (1 : ℝ) / 2 ^ 53 < 1 / 2 := by norm_num
  rw [Int.floor_eq_iff]
  split_ifs with h
  · push_cast
    rcases h with h | ⟨rfl, h⟩
    · have h1 : (C : ℝ) ≤ -1 := by exact_mod_cast (show C ≤ -1 by omega)
      have h2 : (C : ℝ) / 2 ^ 53 ≤ -1 / 2 ^ 53 := div_le_div_of_nonneg_right h1 h53.le
      rw [neg_div] at h2
      constructor <;> linarith
    · push_cast
      constructor <;> linarith
  · push_cast
    rcases lt_trichotomy C 0 with hC | rfl | hC
    · exact absurd (Or.inl hC) h
    · have hv0 : v < 0 := by
        by_contra! hv0
        exact h (Or.inr ⟨rfl, hv0⟩)
      push_cast
      constructor <;> linarith
    · have h1 : (1 : ℝ) ≤ C := by exact_mod_cast hC
      have h2 : 1 / 2 ^ 53 ≤ (C : ℝ) / 2 ^ 53 := div_le_div_of_nonneg_right h1 h53.le
      constructor <;> linarith

/-- `_levels`' exponent `E = max(e_s + e_x, -53)` (`_MIN_EXPONENT`). -/
def levelExp (es ex : ℤ) : ℤ := max (es + ex) (-53)

/-- `_levels`' `p = np.ldexp(fl(m_s·m_x), E)`, the value its guard bounds: the rounding of
`s·x = m_s·m_x·2^(e_s + e_x)`, or of `m_s·m_x·2^-53` where `E` is raised. -/
noncomputable def levelHi (fl : ℝ → ℝ) (ms : ℝ) (es : ℤ) (mx : ℝ) (ex : ℤ) : ℝ :=
  fl (ms * mx) * 2 ^ levelExp es ex

/-- `_levels(x, s, F)` for `s = m_s·2^e_s` and `x = m_x·2^e_x` as `np.frexp` splits them:
`levelOf` of `p = levelHi`, of `e = np.ldexp(e0, E)` with `e0` Dekker's error of `m_s·m_x`, and
of `c = fl(F - 1/2)`. `np.ldexp` is the exact product here, as it is wherever its result is a
float. -/
noncomputable def levels (fl : ℝ → ℝ) (ri : ℝ → ℤ) (ms : ℝ) (es : ℤ) (mx : ℝ) (ex : ℤ)
    (F : ℝ) : ℤ :=
  levelOf fl ri (levelHi fl ms es mx ex) (twoProductErr fl ri ms mx * 2 ^ levelExp es ex)
    (fl (F - 1 / 2))

namespace IsRoundNearest

variable {fl : ℝ → ℝ} (hfl : IsRoundNearest fl)
include hfl

/-- `fl a · 2^E` lies no further from `a · 2^E` than any `y` with `y · 2^-E` a float. -/
lemma abs_err_le_of_scaled (a : ℝ) (E : ℤ) {y : ℝ} (hy : IsF64 (y * 2 ^ (-E))) :
    |(a - fl a) * 2 ^ E| ≤ |y - (fl a * 2 ^ E + (a - fl a) * 2 ^ E)| := by
  have h2 : (0 : ℝ) < 2 ^ E := zpow_pos two_pos E
  have h : y - (fl a * 2 ^ E + (a - fl a) * 2 ^ E) = (y * 2 ^ (-E) - a) * 2 ^ E := by
    rw [zpow_neg]
    field_simp
    ring
  rw [h, abs_mul, abs_mul, abs_of_pos h2, abs_sub_comm a]
  exact mul_le_mul_of_nonneg_right (hfl.nearest a _ hy) h2.le

end IsRoundNearest

section

variable {fl : ℝ → ℝ} (hfl : IsRoundNearest fl)
include hfl

/-- Under the guard `|p| < 2^53`, `levels` is `⌊m_s·m_x·2^E - (F - 1/2)⌋`, and `p` is a float
within `1/2` of `m_s·m_x·2^E`. The landmarks of `levelOf_eq` scaled by `2^-E` are floats, for
`-53 ≤ E ≤ 54`, so `p` lies no further from `m_s·m_x·2^E` than they do. -/
lemma levels_eq_floor {ri : ℝ → ℤ} (hri : IsRoundInt ri) {ms mx : ℝ} (hms : IsMantissa ms)
    (hmx : IsMantissa mx) (es ex : ℤ) {G : ℕ} (hG : G ≤ 53) {N : ℤ} (hN0 : 0 ≤ N)
    (hN : N < 2 ^ G) (hguard : |levelHi fl ms es mx ex| < 2 ^ 53) :
    levels fl ri ms es mx ex (N / 2 ^ G) =
        ⌊ms * mx * 2 ^ levelExp es ex - (N / 2 ^ G - 1 / 2)⌋ ∧
      IsF64 (levelHi fl ms es mx ex) ∧
      |ms * mx * 2 ^ levelExp es ex - levelHi fl ms es mx ex| ≤ 1 / 2 := by
  obtain ⟨C, hcC, hC1, hC2⟩ := exists_offset hG hN0 hN
  have hc : fl ((N : ℝ) / 2 ^ G - 1 / 2) = C / 2 ^ 53 := by
    rw [hcC]
    exact hfl.eq_self (isF64_intCast_div_two_pow (abs_le.mpr ⟨by linarith, by linarith⟩)
      (by norm_num))
  have hE1 : -53 ≤ levelExp es ex := le_max_right _ _
  have hTP := twoProductErr_eq hfl hri hms hmx
  obtain ⟨A, rfl, hA, hA0⟩ := hms.exists_intCast_div
  obtain ⟨B, rfl, hB, hB0⟩ := hmx.exists_intCast_div
  obtain ⟨P, hP, hAB, hP0⟩ := hfl.exists_fl_mul_eq hA hB hA0 hB0
  have hTP' := hfl.twoProductErr_eq_intCast_div hri hA hB hP hAB
  set E := levelExp es ex
  set a := (A : ℝ) / 2 ^ 53 * (B / 2 ^ 53) with ha
  have hp_def : levelHi fl (A / 2 ^ 53) es (B / 2 ^ 53) ex = fl a * 2 ^ E := rfl
  unfold levels
  rw [hc, hp_def, hTP, hcC]
  rw [hp_def] at hguard
  -- `e = (A·B - 2^52·P) · 2^(E - 106)` is a float
  have he : IsF64 ((a - fl a) * 2 ^ E) := by
    rw [← hTP, hTP']
    convert isF64_intCast_mul_zpow hAB (e := E - 106) (by omega) using 1
    rw [zpow_sub₀ two_ne_zero, div_eq_mul_inv, div_eq_mul_inv]
    norm_num
    ring
  by_cases h0 : A * B = 0
  · -- `a = 0`: `p = e = 0`
    have ha0 : a = 0 := by
      rw [ha, div_mul_div_comm, ← Int.cast_mul, h0, Int.cast_zero, zero_div]
    have hfa : fl a = 0 := by
      rw [ha0]
      exact hfl.eq_self IsF64.zero
    rw [hfa, ha0]
    simp only [zero_mul, sub_self]
    have := levelOf_eq hfl hri IsF64.zero IsF64.zero (by norm_num) hC1 hC2 (by simp) (by simp)
      (by simp) (fun _ _ => by simp)
    rw [add_zero] at this
    exact ⟨this, IsF64.zero, by norm_num⟩
  -- `|fl a| ≥ 1/4`, so `E ≤ 54` under the guard
  have hP4 : (2 : ℝ) ^ (-2 : ℤ) ≤ |fl a| := by
    rw [hP, abs_div, abs_of_pos (by positivity : (0 : ℝ) < 2 ^ 54), le_div_iff₀ (by positivity)]
    have : (2 : ℝ) ^ 52 ≤ |(P : ℝ)| := by
      rw [← Int.cast_abs]
      exact_mod_cast hP0 h0
    norm_num
    linarith
  have hE2 : E ≤ 54 := by
    by_contra! hE
    have h1 : (2 : ℝ) ^ (55 : ℤ) ≤ 2 ^ E := zpow_le_zpow_right₀ (by norm_num) (by omega)
    rw [abs_mul, abs_of_pos (zpow_pos (two_pos : (0 : ℝ) < 2) E)] at hguard
    have := mul_le_mul hP4 h1 (by positivity) (abs_nonneg _)
    norm_num at this
    linarith
  have hp : IsF64 (fl a * 2 ^ E) := (hfl.isF64 a).mul_zpow hP4 (by omega)
  have hland {n : ℤ} (hn : |n| ≤ 2 ^ 53) {G' : ℕ} (hG' : G' ≤ 53) :
      |(a - fl a) * 2 ^ E| ≤ |(n : ℝ) / 2 ^ G' - (fl a * 2 ^ E + (a - fl a) * 2 ^ E)| :=
    hfl.abs_err_le_of_scaled a E (isF64_intCast_div_mul_zpow hn (by omega))
  have hint (N' : ℤ) (hN' : |N'| ≤ 2 ^ 53) :
      |(a - fl a) * 2 ^ E| ≤ |(N' : ℝ) - (fl a * 2 ^ E + (a - fl a) * 2 ^ E)| := by
    simpa using hland hN' (G' := 0) (by norm_num)
  have key := levelOf_eq hfl hri hp he hguard hC1 hC2
    (hland (abs_le.mpr ⟨by linarith, by linarith⟩) le_rfl)
    (by convert hland (n := 1) (by norm_num) (G' := 1) (by norm_num) using 3; norm_num)
    (by convert hland (n := -1) (by norm_num) (G' := 1) (by norm_num) using 3; norm_num) hint
  refine ⟨?_, hp, ?_⟩
  · rw [key]
    congr 1
    ring
  · rw [show a * 2 ^ E - fl a * 2 ^ E = (a - fl a) * 2 ^ E by ring]
    exact abs_le_half_of_forall_int hguard hint

/-- `_levels` is exact: under its guard `|p| < 2^53` it returns `⌊s·x - F + 1/2⌋` for
`s = m_s·2^e_s`, `x = m_x·2^e_x` and `F = N / 2^G` with `0 ≤ N < 2^G`, `G ≤ 53`. Where the
exponent is raised, `s·x` and `m_s·m_x·2^-53` both lie below `2^-53` with one sign, which alone
decides their levels (`floor_sub_eq_of_abs_lt`). -/
theorem levels_eq {ri : ℝ → ℤ} (hri : IsRoundInt ri) {ms mx : ℝ} (hms : IsMantissa ms)
    (hmx : IsMantissa mx) (es ex : ℤ) {G : ℕ} (hG : G ≤ 53) {N : ℤ} (hN0 : 0 ≤ N)
    (hN : N < 2 ^ G) (hguard : |levelHi fl ms es mx ex| < 2 ^ 53) :
    levels fl ri ms es mx ex (N / 2 ^ G) =
      roundHalfUp (ms * 2 ^ es * (mx * 2 ^ ex) - N / 2 ^ G) := by
  obtain ⟨h, -, -⟩ := levels_eq_floor hfl hri hms hmx es ex hG hN0 hN hguard
  obtain ⟨C, hcC, hC1, hC2⟩ := exists_offset hG hN0 hN
  have hsx : ms * 2 ^ es * (mx * 2 ^ ex) - N / 2 ^ G + 1 / 2 =
      ms * mx * 2 ^ (es + ex) - (N / 2 ^ G - 1 / 2) := by
    rw [zpow_add₀ two_ne_zero]
    ring
  rw [h, roundHalfUp, hsx, hcC]
  rcases le_or_gt (-53) (es + ex) with hE | hE
  · rw [levelExp, max_eq_left hE]
  rw [levelExp, max_eq_right hE.le]
  have ht : |ms * mx| < 1 := by
    rw [abs_mul]
    calc |ms| * |mx| ≤ 1 * |mx| := by gcongr; exact hms.abs_lt_one.le
      _ < 1 := by rw [one_mul]; exact hmx.abs_lt_one
  have hz1 : (0 : ℝ) < 2 ^ (-53 : ℤ) := zpow_pos two_pos _
  have hz2 : (0 : ℝ) < 2 ^ (es + ex) := zpow_pos two_pos _
  have h53 : (2 : ℝ) ^ (-53 : ℤ) = 1 / 2 ^ 53 := by norm_num
  have hv : |ms * mx * 2 ^ (-53 : ℤ)| < 1 / 2 ^ 53 := by
    rw [abs_mul, abs_of_pos hz1, ← h53]
    calc |ms * mx| * 2 ^ (-53 : ℤ) < 1 * 2 ^ (-53 : ℤ) := by gcongr
      _ = 2 ^ (-53 : ℤ) := one_mul _
  have hv' : |ms * mx * 2 ^ (es + ex)| < 1 / 2 ^ 53 := by
    rw [abs_mul, abs_of_pos hz2, ← h53]
    calc |ms * mx| * 2 ^ (es + ex) < 1 * 2 ^ (es + ex) := by gcongr
      _ = 2 ^ (es + ex) := one_mul _
      _ ≤ 2 ^ (-53 : ℤ) := zpow_le_zpow_right₀ (by norm_num) (by omega)
  rw [floor_sub_eq_of_abs_lt hv hC1 hC2, floor_sub_eq_of_abs_lt hv' hC1 hC2]
  exact if_congr (or_congr Iff.rfl (and_congr Iff.rfl ((mul_nonneg_iff_of_pos_right hz1).trans
    (mul_nonneg_iff_of_pos_right hz2).symm))) rfl rfl

/-- Under the guard `|p| < 2^b`, `b ≤ 53`, the levels lie in `[-2^b - 1, 2^b]`. -/
theorem levels_mem {ri : ℝ → ℤ} (hri : IsRoundInt ri) {ms mx : ℝ} (hms : IsMantissa ms)
    (hmx : IsMantissa mx) (es ex : ℤ) {G : ℕ} (hG : G ≤ 53) {N : ℤ} (hN0 : 0 ≤ N)
    (hN : N < 2 ^ G) {b : ℕ} (hb : b ≤ 53) (hguard : |levelHi fl ms es mx ex| < 2 ^ b) :
    -2 ^ b - 1 ≤ levels fl ri ms es mx ex (N / 2 ^ G) ∧
      levels fl ri ms es mx ex (N / 2 ^ G) ≤ 2 ^ b := by
  have h53 : |levelHi fl ms es mx ex| < 2 ^ 53 :=
    hguard.trans_le (pow_le_pow_right₀ (by norm_num) hb)
  obtain ⟨h, -, he⟩ := levels_eq_floor hfl hri hms hmx es ex hG hN0 hN h53
  obtain ⟨C, hcC, hC1, hC2⟩ := exists_offset hG hN0 hN
  obtain ⟨hc1, hc2⟩ := offset_mem hC1 hC2
  obtain ⟨hp1, hp2⟩ := abs_lt.mp hguard
  obtain ⟨he1, he2⟩ := abs_le.mp he
  rw [h, hcC]
  constructor
  · rw [Int.le_floor]
    push_cast
    linarith
  · rw [← Int.lt_add_one_iff, Int.floor_lt]
    push_cast
    linarith

/-- Under the guard `|p| < 2^53` the levels lie within `2^53`, so float64 holds each: `p` is a
float, and floats below `2^53` lie within `2^53 - 1`. -/
theorem abs_levels_le {ri : ℝ → ℤ} (hri : IsRoundInt ri) {ms mx : ℝ} (hms : IsMantissa ms)
    (hmx : IsMantissa mx) (es ex : ℤ) {G : ℕ} (hG : G ≤ 53) {N : ℤ} (hN0 : 0 ≤ N)
    (hN : N < 2 ^ G) (hguard : |levelHi fl ms es mx ex| < 2 ^ 53) :
    |levels fl ri ms es mx ex (N / 2 ^ G)| ≤ 2 ^ 53 := by
  obtain ⟨h, hp, he⟩ := levels_eq_floor hfl hri hms hmx es ex hG hN0 hN hguard
  obtain ⟨C, hcC, hC1, hC2⟩ := exists_offset hG hN0 hN
  obtain ⟨hc1, hc2⟩ := offset_mem hC1 hC2
  have hp1 : |levelHi fl ms es mx ex| ≤ 2 ^ 53 - 1 := by
    by_cases h52 : (2 : ℝ) ^ 52 ≤ |levelHi fl ms es mx ex|
    · obtain ⟨n, hn⟩ := hp.exists_int_eq h52
      rw [hn] at hguard ⊢
      have h1 : |n| < 2 ^ 53 := by exact_mod_cast hguard
      have h2 : |n| ≤ 2 ^ 53 - 1 := by omega
      exact_mod_cast h2
    · linarith [not_le.mp h52]
  obtain ⟨hp1', hp2'⟩ := abs_le.mp hp1
  obtain ⟨he1, he2⟩ := abs_le.mp he
  rw [h, hcC, abs_le]
  constructor
  · rw [Int.le_floor]
    push_cast
    linarith
  · rw [← Int.lt_add_one_iff, Int.floor_lt]
    push_cast
    linarith

end

end DiffGPT
