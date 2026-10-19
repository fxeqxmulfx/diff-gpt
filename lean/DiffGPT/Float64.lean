module

public import Mathlib

/-!
# A binary64 model

numpy's float64 arithmetic is IEEE 754 binary64 rounding to nearest, ties to even. A binary64
value is modelled as a real `m · 2^e` with `|m| < 2^53` and `e ≥ -1074` (`IsF64`), and an
operation as its exact real result rounded by a function `fl` that returns a nearest such
value (`IsRoundNearest`), ties broken either way, so IEEE's rule is one instance. The
exponent has no cap: the tokenizer's values stay below `2^63`, far from binary64's overflow
at `2^1024`, and below it the capped format rounds the same way.

* `IsRoundNearest.abs_sub_le`: an operation errs by at most `2^-53` of the magnitude of its
  exact result, plus `2^-1075` in the subnormal range.
* `IsRoundNearest.eq_self`, `IsRoundNearest.mono`: rounding keeps floats and order.
* `IsRoundNearest.nonneg_add_iff`: the rounded sum of two floats has the sign of the exact
  sum, since both are multiples of `2^-1074`.
* `IsF64.exists_int_eq`, `IsF64.exists_intCast_div`: floats of magnitude at least `2^52` are
  integers, and those of at least `2^(52 - G)` multiples of `2^-G`.
-/

@[expose] public section

namespace DiffGPT

/-- The finite binary64 values: `m · 2^e` with `|m| < 2^53` and `e ≥ -1074`. -/
def IsF64 (x : ℝ) : Prop :=
  ∃ m e : ℤ, |m| < 2 ^ 53 ∧ -1074 ≤ e ∧ x = m * (2 : ℝ) ^ e

/-- `fl` rounds every real to a nearest binary64 value, ties either way: IEEE 754's
round-to-nearest-even, numpy's float64 arithmetic, is one such. -/
structure IsRoundNearest (fl : ℝ → ℝ) : Prop where
  isF64 : ∀ a, IsF64 (fl a)
  nearest : ∀ a y, IsF64 y → |fl a - a| ≤ |y - a|

/-- The bound of `IsRoundNearest.abs_sub_le` on the rounding error of an exact result `a`. -/
noncomputable def rndErr (a : ℝ) : ℝ := 2 ^ (-53 : ℤ) * |a| + 2 ^ (-1075 : ℤ)

lemma IsF64.zero : IsF64 0 := ⟨0, 0, by norm_num, by norm_num, by simp⟩

lemma IsF64.neg {x : ℝ} (hx : IsF64 x) : IsF64 (-x) := by
  obtain ⟨m, e, hm, he, rfl⟩ := hx
  exact ⟨-m, e, by rwa [abs_neg], he, by push_cast; ring⟩

/-- `m · 2^e` with `|m| ≤ 2^53` is a float: at `|m| = 2^53` it is `(m/2) · 2^(e+1)`. -/
lemma isF64_intCast_mul_zpow {m e : ℤ} (hm : |m| ≤ 2 ^ 53) (he : -1074 ≤ e) :
    IsF64 (m * (2 : ℝ) ^ e) := by
  rcases hm.lt_or_eq with h | h
  · exact ⟨m, e, h, he, rfl⟩
  · rcases (abs_eq (by norm_num : (0 : ℤ) ≤ 2 ^ 53)).mp h with rfl | rfl
    · exact ⟨2 ^ 52, e + 1, by norm_num, by omega, by
        rw [zpow_add_one₀ two_ne_zero]; push_cast; ring⟩
    · exact ⟨-2 ^ 52, e + 1, by norm_num, by omega, by
        rw [zpow_add_one₀ two_ne_zero]; push_cast; ring⟩

/-- Integers up to `2^53` are floats. -/
lemma isF64_intCast {n : ℤ} (hn : |n| ≤ 2 ^ 53) : IsF64 n := by
  simpa using isF64_intCast_mul_zpow (e := 0) hn (by norm_num)

/-- `n / 2^G` with `|n| ≤ 2^53` is a float. -/
lemma isF64_intCast_div_two_pow {n : ℤ} (hn : |n| ≤ 2 ^ 53) {G : ℕ} (hG : G ≤ 1074) :
    IsF64 ((n : ℝ) / 2 ^ G) := by
  have := isF64_intCast_mul_zpow hn (e := -G) (by omega)
  rwa [zpow_neg, zpow_natCast, ← div_eq_mul_inv] at this

/-- `(n / 2^G) · 2^-E` is a float for `|n| ≤ 2^53` while `G + E ≤ 1074`. -/
lemma isF64_intCast_div_mul_zpow {n : ℤ} (hn : |n| ≤ 2 ^ 53) {G : ℕ} {E : ℤ}
    (h : G + E ≤ 1074) : IsF64 ((n : ℝ) / 2 ^ G * 2 ^ (-E)) := by
  have := isF64_intCast_mul_zpow hn (e := -G - E) (by omega)
  convert this using 1
  rw [sub_eq_add_neg, zpow_add₀ two_ne_zero, zpow_neg (2 : ℝ) (G : ℤ), zpow_natCast]
  ring

/-- Floats of magnitude at least `2^52` are integers. -/
lemma IsF64.exists_int_eq {x : ℝ} (hx : IsF64 x) (h : 2 ^ 52 ≤ |x|) : ∃ n : ℤ, x = n := by
  obtain ⟨m, e, hm, -, rfl⟩ := hx
  have he : 0 ≤ e := by
    by_contra! he
    have hm' : |(m : ℝ)| < 2 ^ 53 := by exact_mod_cast hm
    have h2e : (2 : ℝ) ^ e ≤ 1 / 2 :=
      calc (2 : ℝ) ^ e ≤ 2 ^ (-1 : ℤ) := zpow_le_zpow_right₀ (by norm_num) (by omega)
        _ = 1 / 2 := by norm_num
    rw [abs_mul, abs_of_pos (zpow_pos (two_pos : (0 : ℝ) < 2) e)] at h
    have := mul_le_mul_of_nonneg_left h2e (abs_nonneg (m : ℝ))
    linarith
  refine ⟨m * 2 ^ e.toNat, ?_⟩
  push_cast
  rw [← zpow_natCast, Int.toNat_of_nonneg he]

/-- A float of magnitude at least `2^j` has an exponent of at least `j - 52`. -/
lemma IsF64.exists_exponent_ge {y : ℝ} (hy : IsF64 y) {j : ℤ} (h : (2 : ℝ) ^ j ≤ |y|) :
    ∃ m e : ℤ, |m| < 2 ^ 53 ∧ j - 52 ≤ e ∧ y = m * (2 : ℝ) ^ e := by
  obtain ⟨m, e, hm, -, rfl⟩ := hy
  refine ⟨m, e, hm, ?_, rfl⟩
  by_contra! hlt
  have hm' : |(m : ℝ)| < 2 ^ 53 := by exact_mod_cast hm
  have h2e : (0 : ℝ) < 2 ^ e := zpow_pos two_pos e
  rw [abs_mul, abs_of_pos h2e] at h
  have : |(m : ℝ)| * 2 ^ e < 2 ^ j :=
    calc |(m : ℝ)| * 2 ^ e < 2 ^ 53 * 2 ^ e := mul_lt_mul_of_pos_right hm' h2e
      _ = 2 ^ ((53 : ℤ) + e) := by rw [zpow_add₀ two_ne_zero]; norm_num
      _ ≤ 2 ^ j := zpow_le_zpow_right₀ (by norm_num) (by omega)
  linarith

/-- A float of magnitude at least `2^(52 - G)` is a multiple of `2^-G`. -/
lemma IsF64.exists_intCast_div {y : ℝ} (hy : IsF64 y) {G : ℕ}
    (h : (2 : ℝ) ^ (52 - (G : ℤ)) ≤ |y|) : ∃ N : ℤ, y = N / 2 ^ G := by
  obtain ⟨m, e, -, he, rfl⟩ := hy.exists_exponent_ge h
  refine ⟨m * 2 ^ (e + G).toNat, ?_⟩
  rw [eq_div_iff (by positivity)]
  push_cast
  rw [← zpow_natCast (2 : ℝ) (e + G).toNat, Int.toNat_of_nonneg (by omega),
    zpow_add₀ two_ne_zero, zpow_natCast]
  ring

/-- A float of magnitude at least `2^j` stays a float when scaled by `2^E`, as long as the
smallest exponent it may need, `j - 52 + E`, stays normal or subnormal. -/
lemma IsF64.mul_zpow {y : ℝ} (hy : IsF64 y) {j : ℤ} (h : (2 : ℝ) ^ j ≤ |y|) {E : ℤ}
    (hE : -1074 ≤ j - 52 + E) : IsF64 (y * 2 ^ E) := by
  obtain ⟨m, e, hm, hje, rfl⟩ := hy.exists_exponent_ge h
  exact ⟨m, e + E, hm, by omega, by rw [zpow_add₀ two_ne_zero]; ring⟩

/-- A float `m · 2^e₁` is an integer multiple of `2^e` for every `e ≤ e₁`. -/
lemma intCast_mul_zpow_eq_of_le (m : ℤ) {e e₁ : ℤ} (h : e ≤ e₁) :
    (m : ℝ) * 2 ^ e₁ = ((m * 2 ^ (e₁ - e).toNat : ℤ) : ℝ) * 2 ^ e := by
  push_cast
  rw [← zpow_natCast, Int.toNat_of_nonneg (by omega), mul_assoc, ← zpow_add₀ two_ne_zero,
    sub_add_cancel]

namespace IsRoundNearest

variable {fl : ℝ → ℝ} (hfl : IsRoundNearest fl)
include hfl

/-- Rounding keeps a float. -/
lemma eq_self {y : ℝ} (hy : IsF64 y) : fl y = y := by
  have h := hfl.nearest y y hy
  rw [sub_self, abs_zero] at h
  exact sub_eq_zero.mp (abs_nonpos_iff.mp h)

/-- Rounding to nearest is monotone, whatever the ties. -/
lemma mono {a b : ℝ} (h : a ≤ b) : fl a ≤ fl b := by
  by_contra! hlt
  have hba : b ≤ a := by
    have e1 : fl a - a ≤ |fl b - a| := (le_abs_self _).trans (hfl.nearest a (fl b) (hfl.isF64 b))
    have e2 : -(fl b - b) ≤ |fl a - b| := (neg_le_abs _).trans (hfl.nearest b (fl a) (hfl.isF64 a))
    rcases abs_cases (fl b - a) with ⟨h3, _⟩ | ⟨h3, _⟩ <;>
    rcases abs_cases (fl a - b) with ⟨h4, _⟩ | ⟨h4, _⟩ <;> linarith
  obtain rfl := le_antisymm h hba
  exact lt_irrefl _ hlt

/-- Rounding keeps a float lower bound on the magnitude. -/
lemma le_abs {a y : ℝ} (hy : IsF64 y) (h : y ≤ |a|) : y ≤ |fl a| := by
  rcases le_or_gt 0 a with ha | ha
  · rw [abs_of_nonneg ha] at h
    have := hfl.mono h
    rw [hfl.eq_self hy] at this
    exact this.trans (le_abs_self _)
  · rw [abs_of_neg ha] at h
    have := hfl.mono (show a ≤ -y by linarith)
    rw [hfl.eq_self hy.neg] at this
    exact (show y ≤ -fl a by linarith).trans (neg_le_abs _)

/-- The rounded sum of two floats has the sign of the exact sum: both are multiples of
`2^-1074`, so a negative sum is at most `-2^-1074`, itself a float. -/
lemma nonneg_add_iff {a b : ℝ} (ha : IsF64 a) (hb : IsF64 b) : 0 ≤ fl (a + b) ↔ 0 ≤ a + b := by
  constructor
  · intro h
    by_contra! hlt
    obtain ⟨m₁, e₁, -, he₁, rfl⟩ := ha
    obtain ⟨m₂, e₂, -, he₂, rfl⟩ := hb
    rw [intCast_mul_zpow_eq_of_le m₁ he₁, intCast_mul_zpow_eq_of_le m₂ he₂, ← add_mul,
      ← Int.cast_add] at h hlt
    set K := m₁ * 2 ^ (e₁ - -1074).toNat + m₂ * 2 ^ (e₂ - -1074).toNat
    have h2 : (0 : ℝ) < 2 ^ (-1074 : ℤ) := zpow_pos two_pos _
    have hK : K ≤ -1 := by
      have : (K : ℝ) < 0 := by
        by_contra! h0
        have := mul_nonneg h0 h2.le
        linarith
      have : K < 0 := by exact_mod_cast this
      omega
    have hle : (K : ℝ) * 2 ^ (-1074 : ℤ) ≤ ((-1 : ℤ) : ℝ) * 2 ^ (-1074 : ℤ) :=
      mul_le_mul_of_nonneg_right (by exact_mod_cast hK) h2.le
    have hm1 : IsF64 (((-1 : ℤ) : ℝ) * 2 ^ (-1074 : ℤ)) := ⟨-1, -1074, by norm_num, le_refl _, rfl⟩
    have := hfl.mono hle
    rw [hfl.eq_self hm1] at this
    push_cast at this
    linarith
  · intro h
    have := hfl.mono h
    rwa [hfl.eq_self IsF64.zero] at this

/-- The rounding error: `2^-53` relative, plus `2^-1075` absolute for subnormals. -/
lemma abs_sub_le (a : ℝ) : |fl a - a| ≤ rndErr a := by
  suffices ∃ y, IsF64 y ∧ |y - a| ≤ rndErr a by
    obtain ⟨y, hy, hya⟩ := this
    exact (hfl.nearest a y hy).trans hya
  rcases eq_or_ne a 0 with rfl | ha
  · exact ⟨0, IsF64.zero, by simp [rndErr]⟩
  set L := Int.log 2 |a|
  set e := max (L - 52) (-1074) with he
  have ha' : 0 < |a| := abs_pos.mpr ha
  have hL1 : (2 : ℝ) ^ L ≤ |a| := by
    have := Int.zpow_log_le_self (b := 2) (by norm_num) ha'
    exact_mod_cast this
  have hL2 : |a| < (2 : ℝ) ^ (L + 1) := by
    have := Int.lt_zpow_succ_log_self (b := 2) (by norm_num) |a|
    exact_mod_cast this
  have h2e : (0 : ℝ) < 2 ^ e := zpow_pos two_pos e
  have hae : |a / 2 ^ e| < 2 ^ 53 := by
    rw [abs_div, abs_of_pos h2e, div_lt_iff₀ h2e]
    calc |a| < 2 ^ (L + 1) := hL2
      _ ≤ 2 ^ ((53 : ℕ) + e) := zpow_le_zpow_right₀ (by norm_num) (by omega)
      _ = 2 ^ 53 * 2 ^ e := by rw [zpow_add₀ two_ne_zero, zpow_natCast]
  set m := round (a / 2 ^ e)
  have hm1 : |a / 2 ^ e - m| ≤ 1 / 2 := abs_sub_round _
  have hm : |m| ≤ 2 ^ 53 := by
    have h1 : |(m : ℝ)| < 2 ^ 53 + 1 := by
      calc |(m : ℝ)| = |a / 2 ^ e - (a / 2 ^ e - m)| := by rw [sub_sub_cancel]
        _ ≤ |a / 2 ^ e| + |a / 2 ^ e - m| := abs_sub _ _
        _ < 2 ^ 53 + 1 := by linarith
    have h2 : |m| < 2 ^ 53 + 1 := by exact_mod_cast h1
    omega
  refine ⟨m * 2 ^ e, isF64_intCast_mul_zpow hm (le_max_right _ _), ?_⟩
  have hy : |(m : ℝ) * 2 ^ e - a| ≤ 2 ^ e / 2 := by
    have h1 : (m : ℝ) * 2 ^ e - a = -((a / 2 ^ e - m) * 2 ^ e) := by
      field_simp
      ring
    rw [h1, abs_neg, abs_mul, abs_of_pos h2e]
    calc |a / 2 ^ e - m| * 2 ^ e ≤ 1 / 2 * 2 ^ e := mul_le_mul_of_nonneg_right hm1 h2e.le
      _ = 2 ^ e / 2 := by ring
  refine hy.trans ?_
  unfold rndErr
  rcases le_total (L - 52) (-1074) with h | h
  · rw [he, max_eq_right h]
    have h3 : (2 : ℝ) ^ (-1074 : ℤ) / 2 = 2 ^ (-1075 : ℤ) := by
      rw [show (-1074 : ℤ) = -1075 + 1 by norm_num, zpow_add_one₀ two_ne_zero]
      ring
    rw [h3]
    have : (0 : ℝ) ≤ 2 ^ (-53 : ℤ) * |a| := by positivity
    linarith
  · rw [he, max_eq_left h]
    have h3 : (2 : ℝ) ^ (L - 52) / 2 = 2 ^ (-53 : ℤ) * 2 ^ L := by
      rw [← zpow_add₀ two_ne_zero, show L - 52 = -53 + L + 1 by ring, zpow_add_one₀ two_ne_zero]
      ring
    rw [h3]
    have : (2 : ℝ) ^ (-53 : ℤ) * 2 ^ L ≤ 2 ^ (-53 : ℤ) * |a| :=
      mul_le_mul_of_nonneg_left hL1 (by positivity)
    have : (0 : ℝ) ≤ 2 ^ (-1075 : ℤ) := by positivity
    linarith

end IsRoundNearest

end DiffGPT
