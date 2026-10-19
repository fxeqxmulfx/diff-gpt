module

public import Mathlib
public import DiffGPT.LevelGrid

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
* `IsF64.sterbenz`: `x - y` is a float, so exact, when `y/2 ≤ x ≤ 2y`.
* `IsF64.floor`: the floor of a float is a float, so `np.floor` is exact.
* `roundLevels_eq`: `_round_levels` computes `f + (fl(u - f) ≥ 1/2)` with `f = ⌊u⌋`, and
  this is exactly `⌊u + 1/2⌋` for every float `u`, where `⌊fl(u + 1/2)⌋` would round the
  sum first, to `1` at `u = 1/2 - 2^-54`.
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

/-- The anchor fractions `n / 2^G`, `|n| < 2^53`, are floats. -/
lemma isF64_intCast_div_two_pow {n : ℤ} (hn : |n| < 2 ^ 53) {G : ℕ} (hG : G ≤ 1074) :
    IsF64 ((n : ℝ) / 2 ^ G) :=
  ⟨n, -G, hn, by omega, by rw [zpow_neg, zpow_natCast, div_eq_mul_inv]⟩

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

/-- The floor of a float is a float: `np.floor` is exact. -/
lemma IsF64.floor {x : ℝ} (hx : IsF64 x) : IsF64 (⌊x⌋ : ℝ) := by
  by_cases h : 2 ^ 52 ≤ |x|
  · obtain ⟨n, rfl⟩ := hx.exists_int_eq h
    rwa [Int.floor_intCast]
  · obtain ⟨h1, h2⟩ := abs_lt.mp (not_le.mp h)
    have h3 : ⌊x⌋ < 2 ^ 52 := Int.floor_lt.mpr (by push_cast; linarith)
    have h4 : -2 ^ 52 ≤ ⌊x⌋ := Int.le_floor.mpr (by push_cast; linarith)
    exact isF64_intCast (abs_le.mpr ⟨by linarith, by linarith⟩)

/-- A float `m · 2^e₁` is an integer multiple of `2^e` for every `e ≤ e₁`. -/
lemma intCast_mul_zpow_eq_of_le (m : ℤ) {e e₁ : ℤ} (h : e ≤ e₁) :
    (m : ℝ) * 2 ^ e₁ = ((m * 2 ^ (e₁ - e).toNat : ℤ) : ℝ) * 2 ^ e := by
  push_cast
  rw [← zpow_natCast, Int.toNat_of_nonneg (by omega), mul_assoc, ← zpow_add₀ two_ne_zero,
    sub_add_cancel]

/-- An integer multiple `M · 2^e`, `e ≥ -1074`, of magnitude below `2^53 · 2^e` is a float. -/
lemma isF64_of_abs_lt {M e : ℤ} (he : -1074 ≤ e) (h : |(M : ℝ) * 2 ^ e| < 2 ^ 53 * 2 ^ e) :
    IsF64 (M * (2 : ℝ) ^ e) := by
  refine ⟨M, e, ?_, he, rfl⟩
  have h2 : (0 : ℝ) < 2 ^ e := zpow_pos two_pos e
  rw [abs_mul, abs_of_pos h2] at h
  have : |(M : ℝ)| < 2 ^ 53 := lt_of_mul_lt_mul_right h h2.le
  exact_mod_cast this

/-- Sterbenz's lemma: the difference of floats within a factor of two is a float. -/
lemma IsF64.sterbenz {x y : ℝ} (hx : IsF64 x) (hy : IsF64 y) (h₁ : y / 2 ≤ x) (h₂ : x ≤ 2 * y) :
    IsF64 (x - y) := by
  have hle : |x - y| ≤ x ∧ |x - y| ≤ y :=
    ⟨abs_le.mpr ⟨by linarith, by linarith⟩, abs_le.mpr ⟨by linarith, by linarith⟩⟩
  obtain ⟨m₁, e₁, hm₁, he₁, rfl⟩ := hx
  obtain ⟨m₂, e₂, hm₂, he₂, rfl⟩ := hy
  have hlt : |(m₁ : ℝ) * 2 ^ e₁ - m₂ * 2 ^ e₂| < 2 ^ 53 * 2 ^ min e₁ e₂ := by
    rcases min_choice e₁ e₂ with h | h <;> rw [h]
    · calc _ ≤ (m₁ : ℝ) * 2 ^ e₁ := hle.1
        _ ≤ |(m₁ : ℝ)| * 2 ^ e₁ :=
          mul_le_mul_of_nonneg_right (le_abs_self _) (zpow_pos two_pos _).le
        _ < 2 ^ 53 * 2 ^ e₁ :=
          mul_lt_mul_of_pos_right (by exact_mod_cast hm₁) (zpow_pos two_pos _)
    · calc _ ≤ (m₂ : ℝ) * 2 ^ e₂ := hle.2
        _ ≤ |(m₂ : ℝ)| * 2 ^ e₂ :=
          mul_le_mul_of_nonneg_right (le_abs_self _) (zpow_pos two_pos _).le
        _ < 2 ^ 53 * 2 ^ e₂ :=
          mul_lt_mul_of_pos_right (by exact_mod_cast hm₂) (zpow_pos two_pos _)
  rw [intCast_mul_zpow_eq_of_le m₁ (min_le_left e₁ e₂),
    intCast_mul_zpow_eq_of_le m₂ (min_le_right e₁ e₂), ← sub_mul, ← Int.cast_sub] at hlt ⊢
  exact isF64_of_abs_lt (le_min he₁ he₂) hlt

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

/-- `_round_levels` on a float `u`: `f = np.floor(u)`, exact, then `f + (u - f >= 0.5)`, with
the difference and the sum rounded. -/
noncomputable def roundLevels (fl : ℝ → ℝ) (u : ℝ) : ℝ :=
  fl ((⌊u⌋ : ℝ) + if (1 / 2 : ℝ) ≤ fl (u - ⌊u⌋) then 1 else 0)

/-- `_round_levels` rounds every float half up exactly. -/
theorem roundLevels_eq {fl : ℝ → ℝ} (hfl : IsRoundNearest fl) {u : ℝ} (hu : IsF64 u) :
    roundLevels fl u = roundHalfUp u := by
  have h1 : (⌊u⌋ : ℝ) ≤ u := Int.floor_le u
  have h2 : u < ⌊u⌋ + 1 := Int.lt_floor_add_one u
  have hcond : (1 / 2 : ℝ) ≤ fl (u - ⌊u⌋) ↔ (1 / 2 : ℝ) ≤ u - ⌊u⌋ := by
    by_cases hneg : -(1 / 2 : ℝ) < u ∧ u < 0
    · have hf : ⌊u⌋ = -1 := Int.floor_eq_iff.mpr ⟨by push_cast; linarith, by push_cast; linarith⟩
      have hhalf : IsF64 (1 / 2 : ℝ) := ⟨1, -1, by norm_num, by norm_num, by norm_num⟩
      have hmono : fl (1 / 2) ≤ fl (u - ⌊u⌋) := hfl.mono (by rw [hf]; push_cast; linarith)
      rw [hfl.eq_self hhalf] at hmono
      exact iff_of_true hmono (by rw [hf]; push_cast; linarith)
    · suffices hd : IsF64 (u - ⌊u⌋) by rw [hfl.eq_self hd]
      rcases lt_trichotomy ⌊u⌋ 0 with hf | hf | hf
      · -- `u ≤ -1/2`: `u - f = (-f) - (-u)`, Sterbenz with `-u/2 ≤ -f ≤ -2u`
        have hf' : (⌊u⌋ : ℝ) ≤ -1 := by exact_mod_cast Int.le_sub_one_of_lt hf
        have hu0 : u ≤ -(1 / 2) := by
          by_contra! h
          exact hneg ⟨h, by linarith⟩
        have h2u : 2 * u ≤ ⌊u⌋ := by
          rcases le_or_gt u (-1) with h | h
          · linarith
          · have : (-1 : ℤ) ≤ ⌊u⌋ := Int.le_floor.mpr (by push_cast; linarith)
            have : (-1 : ℝ) ≤ ⌊u⌋ := by exact_mod_cast this
            linarith
        have := hu.floor.neg.sterbenz hu.neg (by linarith) (by linarith)
        rwa [neg_sub_neg] at this
      · rw [hf]
        simpa using hu
      · -- `f ≥ 1`: Sterbenz with `f/2 ≤ u ≤ 2f`
        have hf' : (1 : ℝ) ≤ ⌊u⌋ := by exact_mod_cast hf
        exact hu.sterbenz hu.floor (by linarith) (by linarith)
  unfold roundLevels roundHalfUp
  by_cases h : (1 / 2 : ℝ) ≤ u - ⌊u⌋
  · rw [ite_eq_left (hcond.mpr h)]
    have hfr : ⌊u + 1 / 2⌋ = ⌊u⌋ + 1 :=
      Int.floor_eq_iff.mpr ⟨by push_cast; linarith, by push_cast; linarith⟩
    rw [hfr]
    push_cast
    apply hfl.eq_self
    have hsmall : |u| < 2 ^ 52 := by
      by_contra! h52
      obtain ⟨n, rfl⟩ := hu.exists_int_eq h52
      rw [Int.floor_intCast, sub_self] at h
      norm_num at h
    obtain ⟨h3, h4⟩ := abs_lt.mp hsmall
    have h5 : ⌊u⌋ + 1 ≤ 2 ^ 52 := by
      have : ⌊u⌋ < 2 ^ 52 := Int.floor_lt.mpr (by push_cast; linarith)
      omega
    have h6 : -2 ^ 52 ≤ ⌊u⌋ + 1 := by
      have : -2 ^ 52 ≤ ⌊u⌋ := Int.le_floor.mpr (by push_cast; linarith)
      omega
    have := isF64_intCast (n := ⌊u⌋ + 1) (abs_le.mpr ⟨by linarith, by linarith⟩)
    exact_mod_cast this
  · rw [ite_eq_right (mt hcond.mp h)]
    have hfr : ⌊u + 1 / 2⌋ = ⌊u⌋ :=
      Int.floor_eq_iff.mpr ⟨by linarith, by linarith⟩
    rw [hfr, add_zero]
    exact hfl.eq_self hu.floor

end DiffGPT
