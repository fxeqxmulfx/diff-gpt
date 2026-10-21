module

public import Mathlib
public import DiffGPT.Float64

/-!
# Knuth's two-sum

`_difference_range` differences the column by each lag `S` in turn and takes the rounding error
of each difference by Knuth's two-sum, in this order, for the floats `a = d_t` and `b = -d_{t-S}`
of the column `d` differenced so far:

    diff = a + b;  b_rounded = diff - a;  error = (a - (diff - b_rounded)) + (b - b_rounded)

`twoSumErr` is that `error` in the binary64 model of `DiffGPT.Float64`, ties either way, and
`IsRoundNearest.twoSumErr_eq` proves it exact: `a + b - fl (a + b)`. The model has no overflow,
and IEEE 754 rounds as it does wherever no step overflows; a step that overflows makes `error`
infinite or `nan`.

* `isF64_of_eq_intCast_mul`: a multiple of `2^e`, `e ≥ -1074`, below `2^(e + 53)` is a float;
  `IsF64.exists_common`: two floats are multiples of one such `2^e`, the smaller of them below
  `2^(e + 53)`.
* `isF64_add_of_abs_le` (Sterbenz): a sum of two floats no larger than either summand is a
  float; so is `s - x` for `s` in `[x/2, 2x]` (`isF64_sub_of_mem`), and a sum below `2^-1021`
  (`isF64_add_of_abs_lt`).
* `IsRoundNearest.abs_add_sub_fl_le`, `IsRoundNearest.isF64_add_sub_fl`: the rounding error of a
  float sum is no larger than either summand, and a float.
* `IsRoundNearest.isF64_fl_add_sub` (Fast2Sum): for floats with `|y| ≤ |x|`, `fl (x + y) - x` is
  a float. `IsRoundNearest.reflect`: `u ↦ -fl (-u)` rounds to nearest as well, so `x > 0`
  suffices.
* `IsRoundNearest.twoSumErr_eq`: where `|a| ≤ |s|`, `s = fl (a + b)`, Fast2Sum makes
  `s - b_rounded` and `b - b_rounded` floats, the latter as `s - a = b - (a + b - s)`, and
  `a - (s - b_rounded)` is the rounding error of `s - a`, so every step is exact
  (`twoSumErr_eq_of_abs_le`). Where `|s| < |a|`, `s - a` is a float, by Fast2Sum for
  `|b| ≤ |a|` and by Sterbenz otherwise, so `b_rounded = s - a` and `error = b - b_rounded`
  (`twoSumErr_eq_of_isF64_sub`).
-/

@[expose] public section

namespace DiffGPT

/-- Knuth's two-sum as `_difference_range` computes it: `s = a + b`, `b' = s - a`, and the
error `(a - (s - b')) + (b - b')`. -/
noncomputable def twoSumErr (fl : ℝ → ℝ) (a b : ℝ) : ℝ :=
  fl (fl (a - fl (fl (a + b) - fl (fl (a + b) - a))) + fl (b - fl (fl (a + b) - a)))

/-- A float `m · 2^e` lies below `2^(e + 53)`. -/
lemma abs_intCast_mul_zpow_lt {m e : ℤ} (hm : |m| < 2 ^ 53) :
    |(m : ℝ) * 2 ^ e| < 2 ^ (e + 53) := by
  have h2e : (0 : ℝ) < 2 ^ e := zpow_pos two_pos e
  have hm' : |(m : ℝ)| < 2 ^ 53 := by exact_mod_cast hm
  calc |(m : ℝ) * 2 ^ e| = |(m : ℝ)| * 2 ^ e := by rw [abs_mul, abs_of_pos h2e]
    _ < 2 ^ 53 * 2 ^ e := mul_lt_mul_of_pos_right hm' h2e
    _ = 2 ^ (e + 53) := by rw [add_comm, zpow_add₀ two_ne_zero]; norm_num

/-- A multiple of `2^e`, `e ≥ -1074`, below `2^(e + 53)` is a float. -/
lemma isF64_of_eq_intCast_mul {x : ℝ} {n e : ℤ} (hx : x = n * 2 ^ e) (he : -1074 ≤ e)
    (h : |x| < 2 ^ (e + 53)) : IsF64 x := by
  have h2e : (0 : ℝ) < 2 ^ e := zpow_pos two_pos e
  have h' : |(n : ℝ)| * 2 ^ e < 2 ^ 53 * 2 ^ e :=
    calc |(n : ℝ)| * 2 ^ e = |x| := by rw [hx, abs_mul, abs_of_pos h2e]
      _ < 2 ^ (e + 53) := h
      _ = 2 ^ 53 * 2 ^ e := by rw [add_comm, zpow_add₀ two_ne_zero]; norm_num
  have hn : |(n : ℝ)| < 2 ^ 53 := lt_of_mul_lt_mul_right h' h2e.le
  exact ⟨n, e, by exact_mod_cast hn, he, hx⟩

/-- Two floats are multiples of one `2^e`, `e ≥ -1074`, the smaller of them below
`2^(e + 53)`: the `2^e` of the finer one. -/
lemma IsF64.exists_common {a b : ℝ} (ha : IsF64 a) (hb : IsF64 b) :
    ∃ e n₁ n₂ : ℤ, -1074 ≤ e ∧ a = n₁ * 2 ^ e ∧ b = n₂ * 2 ^ e ∧
      min |a| |b| < 2 ^ (e + 53) := by
  obtain ⟨m₁, e₁, hm₁, he₁, rfl⟩ := ha
  obtain ⟨m₂, e₂, hm₂, he₂, rfl⟩ := hb
  refine ⟨min e₁ e₂, _, _, le_min he₁ he₂, intCast_mul_zpow_eq_of_le m₁ (min_le_left _ _),
    intCast_mul_zpow_eq_of_le m₂ (min_le_right _ _), ?_⟩
  rcases le_total e₁ e₂ with h | h
  · rw [min_eq_left h]
    exact (min_le_left _ _).trans_lt (abs_intCast_mul_zpow_lt hm₁)
  · rw [min_eq_right h]
    exact (min_le_right _ _).trans_lt (abs_intCast_mul_zpow_lt hm₂)

/-- Sterbenz: a sum of two floats no larger than either summand is a float, a multiple of their
finer step `2^e` below `2^(e + 53)`. -/
lemma isF64_add_of_abs_le {a b : ℝ} (ha : IsF64 a) (hb : IsF64 b) (h₁ : |a + b| ≤ |a|)
    (h₂ : |a + b| ≤ |b|) : IsF64 (a + b) := by
  obtain ⟨e, n₁, n₂, he, rfl, rfl, hmin⟩ := ha.exists_common hb
  exact isF64_of_eq_intCast_mul (n := n₁ + n₂) (by push_cast; ring) he
    ((le_min h₁ h₂).trans_lt hmin)

/-- A sum of two floats below `2^-1021` is a float: both are multiples of `2^-1074`. -/
lemma isF64_add_of_abs_lt {a b : ℝ} (ha : IsF64 a) (hb : IsF64 b)
    (h : |a + b| < 2 ^ (-1021 : ℤ)) : IsF64 (a + b) := by
  obtain ⟨n₁, rfl⟩ := ha.exists_int_mul
  obtain ⟨n₂, rfl⟩ := hb.exists_int_mul
  exact isF64_of_eq_intCast_mul (n := n₁ + n₂) (by push_cast; ring) le_rfl
    (by rw [show (-1074 : ℤ) + 53 = -1021 by norm_num]; exact h)

/-- Sterbenz for a difference: for floats `x > 0` and `s` in `[x/2, 2x]`, `s - x` is a float. -/
lemma isF64_sub_of_mem {x s : ℝ} (hx : IsF64 x) (hs : IsF64 s) (hx0 : 0 < x)
    (h₁ : x / 2 ≤ s) (h₂ : s ≤ 2 * x) : IsF64 (s - x) := by
  rw [sub_eq_add_neg]
  refine isF64_add_of_abs_le hs hx.neg ?_ ?_
  · rw [abs_of_pos (by linarith : 0 < s), abs_le]
    constructor <;> linarith
  · rw [abs_neg, abs_of_pos hx0, abs_le]
    constructor <;> linarith

namespace IsRoundNearest

variable {fl : ℝ → ℝ} (hfl : IsRoundNearest fl)
include hfl

/-- Rounding the negation and negating back rounds to nearest as well. -/
lemma reflect : IsRoundNearest (fun u => -fl (-u)) :=
  ⟨fun a => (hfl.isF64 (-a)).neg, fun a y hy => by
    have := hfl.nearest (-a) (-y) hy.neg
    rwa [show -y - -a = -(y - a) by ring, abs_neg, show fl (-a) - -a = -(-fl (-a) - a) by ring,
      abs_neg] at this⟩

/-- The rounding error of a float sum is no larger than either summand: both are floats. -/
lemma abs_add_sub_fl_le {a b : ℝ} (ha : IsF64 a) (hb : IsF64 b) :
    |a + b - fl (a + b)| ≤ |a| ∧ |a + b - fl (a + b)| ≤ |b| := by
  rw [abs_sub_comm]
  constructor
  · have := hfl.nearest (a + b) b hb
    rwa [show b - (a + b) = -a by ring, abs_neg] at this
  · have := hfl.nearest (a + b) a ha
    rwa [show a - (a + b) = -b by ring, abs_neg] at this

/-- The rounding error of a float sum is a float: a multiple of the summands' finer step `2^e`,
since the sum is a float or at least `2^(e + 53)`, where floats are multiples of `2^(e + 1)`,
and below `2^(e + 53)`, being no larger than either summand. -/
lemma isF64_add_sub_fl {a b : ℝ} (ha : IsF64 a) (hb : IsF64 b) :
    IsF64 (a + b - fl (a + b)) := by
  obtain ⟨hδa, hδb⟩ := hfl.abs_add_sub_fl_le ha hb
  obtain ⟨e, n₁, n₂, he, h₁, h₂, hmin⟩ := ha.exists_common hb
  rcases lt_or_ge |a + b| (2 ^ (e + 53)) with h | h
  · have hab : IsF64 (a + b) :=
      isF64_of_eq_intCast_mul (n := n₁ + n₂) (by rw [h₁, h₂]; push_cast; ring) he h
    rw [hfl.eq_self hab, sub_self]
    exact IsF64.zero
  · have h2 : IsF64 ((2 : ℝ) ^ (e + 53)) := by
      have := isF64_intCast_mul_zpow (m := 1) (e := e + 53) (by norm_num) (by omega)
      simpa using this
    obtain ⟨m, e', -, he', hm⟩ := (hfl.isF64 (a + b)).exists_exponent_ge (hfl.le_abs h2 h)
    rw [intCast_mul_zpow_eq_of_le m (show e ≤ e' by omega)] at hm
    exact isF64_of_eq_intCast_mul (n := n₁ + n₂ - m * 2 ^ (e' - e).toNat)
      (by rw [hm, h₁, h₂]; push_cast; ring) he ((le_min hδa hδb).trans_lt hmin)

/-- Fast2Sum for `x > 0`: unless `x + y` is a float, `fl (x + y)` lies in `[x/2, 2x]`, so
Sterbenz applies. For `y < 0`, Sterbenz makes `x + y` a float unless `x + y ≥ x/2`, and `x/2`
is a float, as `x ≥ x + y ≥ 2^-1021`. -/
lemma isF64_fl_add_sub_of_pos {x y : ℝ} (hx : IsF64 x) (hy : IsF64 y) (hx0 : 0 < x)
    (h : |y| ≤ x) : IsF64 (fl (x + y) - x) := by
  by_cases hxy : IsF64 (x + y)
  · rw [hfl.eq_self hxy, add_sub_cancel_left]
    exact hy
  refine isF64_sub_of_mem hx (hfl.isF64 _) hx0 ?_ ?_
  · rcases le_or_gt 0 y with hy0 | hy0
    · have := hfl.mono (show x ≤ x + y by linarith)
      rw [hfl.eq_self hx] at this
      linarith
    rw [abs_of_neg hy0] at h
    rcases lt_or_ge (x / 2) (-y) with hy2 | hy2
    · refine absurd (isF64_add_of_abs_le hx hy ?_ ?_) hxy
      · rw [abs_of_pos hx0, abs_of_nonneg (by linarith)]
        linarith
      · rw [abs_of_neg hy0, abs_of_nonneg (by linarith)]
        linarith
    have h1021 : 2 ^ (-1021 : ℤ) ≤ x + y := by
      by_contra! hlt
      exact hxy (isF64_add_of_abs_lt hx hy (by rw [abs_of_nonneg (by linarith)]; exact hlt))
    have hx2 : IsF64 (x / 2) := by
      have := hx.mul_zpow (j := -1021) (E := -1) (by rw [abs_of_pos hx0]; linarith)
        (by norm_num)
      rwa [zpow_neg_one, ← div_eq_mul_inv] at this
    have := hfl.mono (show x / 2 ≤ x + y by linarith)
    rwa [hfl.eq_self hx2] at this
  · have := hfl.mono (show x + y ≤ 2 * x by linarith [le_abs_self y])
    rwa [hfl.eq_self hx.two_mul] at this

/-- Fast2Sum: for floats with `|y| ≤ |x|`, `fl (x + y) - x` is a float. -/
lemma isF64_fl_add_sub {x y : ℝ} (hx : IsF64 x) (hy : IsF64 y) (h : |y| ≤ |x|) :
    IsF64 (fl (x + y) - x) := by
  rcases lt_trichotomy x 0 with hx0 | rfl | hx0
  · have h1 : IsF64 (-fl (-(-x + -y)) - -x) :=
      hfl.reflect.isF64_fl_add_sub_of_pos hx.neg hy.neg (neg_pos.mpr hx0)
        (by rwa [abs_neg, ← abs_of_neg hx0])
    rw [show -(-x + -y) = x + y by ring] at h1
    convert h1.neg using 1
    ring
  · have hy0 : y = 0 := abs_nonpos_iff.mp (by simpa using h)
    subst hy0
    simpa using hfl.isF64 0
  · exact hfl.isF64_fl_add_sub_of_pos hx hy hx0 (by rwa [abs_of_pos hx0] at h)

/-- Where `s - a` is a float, `s = fl (a + b)`, the two-sum is exact step by step: `b' = s - a`,
`s - b' = a`, and the error is `b - b' = a + b - s`. -/
lemma twoSumErr_eq_of_isF64_sub {a b : ℝ} (ha : IsF64 a) (hb : IsF64 b)
    (h : IsF64 (fl (a + b) - a)) : twoSumErr fl a b = a + b - fl (a + b) := by
  have hδ := hfl.isF64_add_sub_fl ha hb
  unfold twoSumErr
  rw [hfl.eq_self h, sub_sub_cancel, hfl.eq_self ha, sub_self, hfl.eq_self IsF64.zero, zero_add,
    show b - (fl (a + b) - a) = a + b - fl (a + b) by ring, hfl.eq_self hδ, hfl.eq_self hδ]

/-- Where `|a| ≤ |s|`, `s = fl (a + b)`, Fast2Sum makes `s - b'` a float for `b' = fl (s - a)`;
`a - (s - b')` is the rounding error of `s - a`, a float; and since `s - a = b - δ` for the
error `δ = a + b - s`, `|δ| ≤ |b|`, Fast2Sum makes `b - b'` a float. Their sum is `δ`. -/
lemma twoSumErr_eq_of_abs_le {a b : ℝ} (ha : IsF64 a) (hb : IsF64 b)
    (h : |a| ≤ |fl (a + b)|) : twoSumErr fl a b = a + b - fl (a + b) := by
  have hs := hfl.isF64 (a + b)
  have hδ := hfl.isF64_add_sub_fl ha hb
  have hδb := (hfl.abs_add_sub_fl_le ha hb).2
  have h1 : IsF64 (fl (fl (a + b) - a) - fl (a + b)) := by
    have := hfl.isF64_fl_add_sub hs ha.neg (by rwa [abs_neg])
    rwa [← sub_eq_add_neg] at this
  have h2 : IsF64 (a - (fl (a + b) - fl (fl (a + b) - a))) := by
    have := (hfl.isF64_add_sub_fl hs ha.neg).neg
    rw [← sub_eq_add_neg] at this
    convert this using 1
    ring
  have h3 : IsF64 (b - fl (fl (a + b) - a)) := by
    have := (hfl.isF64_fl_add_sub hb hδ.neg (by rwa [abs_neg])).neg
    rw [show b + -(a + b - fl (a + b)) = fl (a + b) - a by ring] at this
    convert this using 1
    ring
  have h4 : IsF64 (fl (a + b) - fl (fl (a + b) - a)) := by
    convert h1.neg using 1
    ring
  unfold twoSumErr
  rw [hfl.eq_self h4, hfl.eq_self h2, hfl.eq_self h3,
    show a - (fl (a + b) - fl (fl (a + b) - a)) + (b - fl (fl (a + b) - a)) =
      a + b - fl (a + b) by ring, hfl.eq_self hδ]

end IsRoundNearest

/-- The two-sum error of two floats is the exact rounding error of their sum. Where
`|fl (a + b)| < |a|`, also `|a + b| < |a|`, so `fl (a + b) - a` is a float: by Fast2Sum for
`|b| ≤ |a|`, and otherwise because Sterbenz makes `a + b` a float. -/
theorem IsRoundNearest.twoSumErr_eq {fl : ℝ → ℝ} (hfl : IsRoundNearest fl) {a b : ℝ}
    (ha : IsF64 a) (hb : IsF64 b) : twoSumErr fl a b = a + b - fl (a + b) := by
  rcases le_or_gt |a| |fl (a + b)| with h | h
  · exact hfl.twoSumErr_eq_of_abs_le ha hb h
  refine hfl.twoSumErr_eq_of_isF64_sub ha hb ?_
  have hab : |a + b| < |a| := by
    by_contra! h'
    exact (not_le.mpr h) (hfl.le_abs ha.abs h')
  rcases le_or_gt |b| |a| with hba | hba
  · exact hfl.isF64_fl_add_sub ha hb hba
  · rw [hfl.eq_self (isF64_add_of_abs_le ha hb hab.le (hab.trans hba).le), add_sub_cancel_left]
    exact hb

end DiffGPT
