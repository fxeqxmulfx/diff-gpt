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
  exact result, plus `2^-1075` in the subnormal range; the sum of two floats by at most
  `2^-53` of it (`IsRoundNearest.abs_add_sub_le`).
* `IsRoundNearest.eq_self`, `IsRoundNearest.mono`: rounding keeps floats and order.
* `IsRoundNearest.nonneg_add_iff`: the rounded sum of two floats has the sign of the exact
  sum, since both are multiples of `2^-1074`.
* `IsF64.exists_int_eq`, `IsF64.exists_intCast_div`: floats of magnitude at least `2^52` are
  integers, and those of at least `2^(52 - G)` multiples of `2^-G`.
* The floats next to a real: all floats are multiples of `2^-1074`, so there is a greatest at
  or below `q`, `floorF64`, and one below `x`, `predF64`, `math.nextafter(x, -inf)`; `ceilF64`
  and `succF64` mirror them. `_float_at_or_above` and `_float_at_or_below` return `ceilF64` and
  `floorF64` (`IsRoundNearest.atOrAbove_eq`, `IsRoundNearest.atOrBelow_eq`): a nearest float
  below `q` has no float between it and `q`.
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

/-- Floats are multiples of `2^-1074`. -/
lemma IsF64.exists_int_mul {y : ℝ} (hy : IsF64 y) : ∃ n : ℤ, y = n * (2 : ℝ) ^ (-1074 : ℤ) := by
  obtain ⟨m, e, -, he, rfl⟩ := hy
  exact ⟨_, intCast_mul_zpow_eq_of_le m he⟩

lemma IsF64.two_mul {x : ℝ} (hx : IsF64 x) : IsF64 (2 * x) := by
  obtain ⟨m, e, hm, he, rfl⟩ := hx
  exact ⟨m, e + 1, hm, by omega, by rw [zpow_add_one₀ two_ne_zero]; ring⟩

lemma IsF64.abs {x : ℝ} (hx : IsF64 x) : IsF64 |x| := by
  rcases abs_choice x with h | h <;> rw [h]
  · exact hx
  · exact hx.neg

/-- Within half a step `2^e`, `e ≥ -1074`, of a real below `2^(e + 53)` lies a float. -/
lemma exists_isF64_abs_sub_le {a : ℝ} {e : ℤ} (he : -1074 ≤ e) (ha : |a| < 2 ^ (e + 53)) :
    ∃ y, IsF64 y ∧ |y - a| ≤ 2 ^ e / 2 := by
  have h2e : (0 : ℝ) < 2 ^ e := zpow_pos two_pos e
  have hae : |a / 2 ^ e| < 2 ^ 53 := by
    rw [abs_div, abs_of_pos h2e, div_lt_iff₀ h2e]
    calc |a| < 2 ^ (e + 53) := ha
      _ = 2 ^ 53 * 2 ^ e := by rw [add_comm, zpow_add₀ two_ne_zero]; norm_num
  set m := round (a / 2 ^ e)
  have hm1 : |a / 2 ^ e - m| ≤ 1 / 2 := abs_sub_round _
  have hm : |m| ≤ 2 ^ 53 := by
    have h1 : |(m : ℝ)| < 2 ^ 53 + 1 := by
      calc |(m : ℝ)| = |a / 2 ^ e - (a / 2 ^ e - m)| := by rw [sub_sub_cancel]
        _ ≤ |a / 2 ^ e| + |a / 2 ^ e - m| := abs_sub _ _
        _ < 2 ^ 53 + 1 := by linarith
    have h2 : |m| < 2 ^ 53 + 1 := by exact_mod_cast h1
    omega
  refine ⟨m * 2 ^ e, isF64_intCast_mul_zpow hm he, ?_⟩
  have h1 : (m : ℝ) * 2 ^ e - a = -((a / 2 ^ e - m) * 2 ^ e) := by
    field_simp
    ring
  rw [h1, abs_neg, abs_mul, abs_of_pos h2e]
  calc |a / 2 ^ e - m| * 2 ^ e ≤ 1 / 2 * 2 ^ e := mul_le_mul_of_nonneg_right hm1 h2e.le
    _ = 2 ^ e / 2 := by ring

/-! ### The floats next to a real -/

/-- Every real has a float below it, `-2^k` for `2^k` above it. -/
lemma exists_isF64_lt (x : ℝ) : ∃ y, IsF64 y ∧ y < x := by
  obtain ⟨k, hk⟩ := pow_unbounded_of_one_lt (-x) (by norm_num : (1 : ℝ) < 2)
  refine ⟨-(2 : ℝ) ^ k, ?_, by linarith⟩
  have := isF64_intCast_mul_zpow (m := -1) (e := k) (by norm_num) (by omega)
  simpa [zpow_natCast] using this

/-- A nonempty set of floats bounded above has a greatest element: they are multiples of
`2^-1074`. -/
lemma exists_greatest_isF64 {S : Set ℝ} (hS : ∀ y ∈ S, IsF64 y) (hb : BddAbove S)
    (hne : S.Nonempty) : ∃ y ∈ S, ∀ y' ∈ S, y' ≤ y := by
  obtain ⟨B, hB⟩ := hb
  obtain ⟨y₀, hy₀⟩ := hne
  have h2 : (0 : ℝ) < 2 ^ (-1074 : ℤ) := zpow_pos two_pos _
  obtain ⟨n₀, rfl⟩ := (hS y₀ hy₀).exists_int_mul
  obtain ⟨N, hN, hmax⟩ := Int.exists_greatest_of_bdd
    (P := fun n : ℤ => (n : ℝ) * 2 ^ (-1074 : ℤ) ∈ S)
    ⟨⌈B / 2 ^ (-1074 : ℤ)⌉, fun n hn => by
      have h : (n : ℝ) ≤ B / 2 ^ (-1074 : ℤ) := by
        rw [le_div_iff₀ h2]
        exact hB hn
      exact_mod_cast h.trans (Int.le_ceil _)⟩
    ⟨n₀, hy₀⟩
  refine ⟨_, hN, fun y' hy' => ?_⟩
  obtain ⟨n', rfl⟩ := (hS y' hy').exists_int_mul
  exact mul_le_mul_of_nonneg_right (by exact_mod_cast hmax n' hy') h2.le

lemma exists_floorF64 (q : ℝ) :
    ∃ y, (IsF64 y ∧ y ≤ q) ∧ ∀ y', IsF64 y' ∧ y' ≤ q → y' ≤ y := by
  obtain ⟨y, hy, hyq⟩ := exists_isF64_lt q
  exact exists_greatest_isF64 (S := {y | IsF64 y ∧ y ≤ q}) (fun _ h => h.1)
    ⟨q, fun _ h => h.2⟩ ⟨y, hy, hyq.le⟩

lemma exists_predF64 (x : ℝ) :
    ∃ y, (IsF64 y ∧ y < x) ∧ ∀ y', IsF64 y' ∧ y' < x → y' ≤ y := by
  obtain ⟨y, hy, hyx⟩ := exists_isF64_lt x
  exact exists_greatest_isF64 (S := {y | IsF64 y ∧ y < x}) (fun _ h => h.1)
    ⟨x, fun _ h => h.2.le⟩ ⟨y, hy, hyx⟩

/-- The greatest float at or below `q`, which `_float_at_or_below` returns (`atOrBelow_eq`). -/
noncomputable def floorF64 (q : ℝ) : ℝ := (exists_floorF64 q).choose

/-- The least float at or above `q`, which `_float_at_or_above` returns (`atOrAbove_eq`). -/
noncomputable def ceilF64 (q : ℝ) : ℝ := -floorF64 (-q)

/-- `math.nextafter(x, -inf)`: the greatest float below `x`. -/
noncomputable def predF64 (x : ℝ) : ℝ := (exists_predF64 x).choose

/-- `math.nextafter(x, inf)`: the least float above `x`. -/
noncomputable def succF64 (x : ℝ) : ℝ := -predF64 (-x)

lemma isF64_floorF64 (q : ℝ) : IsF64 (floorF64 q) := (exists_floorF64 q).choose_spec.1.1

lemma floorF64_le (q : ℝ) : floorF64 q ≤ q := (exists_floorF64 q).choose_spec.1.2

lemma le_floorF64 {q y : ℝ} (hy : IsF64 y) (h : y ≤ q) : y ≤ floorF64 q :=
  (exists_floorF64 q).choose_spec.2 y ⟨hy, h⟩

lemma isF64_predF64 (x : ℝ) : IsF64 (predF64 x) := (exists_predF64 x).choose_spec.1.1

lemma predF64_lt (x : ℝ) : predF64 x < x := (exists_predF64 x).choose_spec.1.2

lemma le_predF64 {x y : ℝ} (hy : IsF64 y) (h : y < x) : y ≤ predF64 x :=
  (exists_predF64 x).choose_spec.2 y ⟨hy, h⟩

lemma isF64_ceilF64 (q : ℝ) : IsF64 (ceilF64 q) := (isF64_floorF64 _).neg

lemma le_ceilF64 (q : ℝ) : q ≤ ceilF64 q := by
  unfold ceilF64
  linarith [floorF64_le (-q)]

lemma ceilF64_le {q y : ℝ} (hy : IsF64 y) (h : q ≤ y) : ceilF64 q ≤ y := by
  unfold ceilF64
  linarith [le_floorF64 hy.neg (neg_le_neg h)]

lemma isF64_succF64 (x : ℝ) : IsF64 (succF64 x) := (isF64_predF64 _).neg

lemma lt_succF64 (x : ℝ) : x < succF64 x := by
  unfold succF64
  linarith [predF64_lt (-x)]

lemma succF64_le {x y : ℝ} (hy : IsF64 y) (h : x < y) : succF64 x ≤ y := by
  unfold succF64
  linarith [le_predF64 hy.neg (neg_lt_neg h)]

lemma floorF64_eq_self {y : ℝ} (hy : IsF64 y) : floorF64 y = y :=
  le_antisymm (floorF64_le y) (le_floorF64 hy le_rfl)

lemma floorF64_mono {p q : ℝ} (h : p ≤ q) : floorF64 p ≤ floorF64 q :=
  le_floorF64 (isF64_floorF64 p) ((floorF64_le p).trans h)

lemma predF64_mono {x y : ℝ} (h : x ≤ y) : predF64 x ≤ predF64 y :=
  le_predF64 (isF64_predF64 x) ((predF64_lt x).trans_le h)

lemma succF64_mono {x y : ℝ} (h : x ≤ y) : succF64 x ≤ succF64 y :=
  succF64_le (isF64_succF64 y) (h.trans_lt (lt_succF64 y))

/-- The float after the one at or below `q` lies above `q`. -/
lemma lt_succF64_floorF64 (q : ℝ) : q < succF64 (floorF64 q) := by
  by_contra! h
  linarith [le_floorF64 (isF64_succF64 _) h, lt_succF64 (floorF64 q)]

/-- The float before the one at or above `q` lies below `q`. -/
lemma predF64_ceilF64_lt (q : ℝ) : predF64 (ceilF64 q) < q := by
  by_contra! h
  linarith [ceilF64_le (isF64_predF64 _) h, predF64_lt (ceilF64 q)]

/-- `_float_at_or_above`: `float(q)`, a nearest float, then `math.nextafter` up from it if it
fell below `q`. -/
noncomputable def atOrAbove (fl : ℝ → ℝ) (q : ℝ) : ℝ :=
  if fl q < q then succF64 (fl q) else fl q

/-- `_float_at_or_below`: `0.0 - _float_at_or_above(-q)`, which negates exactly. -/
noncomputable def atOrBelow (fl : ℝ → ℝ) (q : ℝ) : ℝ := -atOrAbove fl (-q)

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
  have ha' : 0 < |a| := abs_pos.mpr ha
  have hL1 : (2 : ℝ) ^ L ≤ |a| := by
    have := Int.zpow_log_le_self (b := 2) (by norm_num) ha'
    exact_mod_cast this
  have hL2 : |a| < (2 : ℝ) ^ (L + 1) := by
    have := Int.lt_zpow_succ_log_self (b := 2) (by norm_num) |a|
    exact_mod_cast this
  obtain ⟨y, hy, hya⟩ := exists_isF64_abs_sub_le (le_max_right (L - 52) _)
    (hL2.trans_le (zpow_le_zpow_right₀ (by norm_num) (by omega)))
  refine ⟨y, hy, hya.trans ?_⟩
  unfold rndErr
  rcases le_total (L - 52) (-1074) with h | h
  · rw [max_eq_right h]
    have h3 : (2 : ℝ) ^ (-1074 : ℤ) / 2 = 2 ^ (-1075 : ℤ) := by
      rw [show (-1074 : ℤ) = -1075 + 1 by norm_num, zpow_add_one₀ two_ne_zero]
      ring
    rw [h3]
    have : (0 : ℝ) ≤ 2 ^ (-53 : ℤ) * |a| := by positivity
    linarith
  · rw [max_eq_left h]
    have h3 : (2 : ℝ) ^ (L - 52) / 2 = 2 ^ (-53 : ℤ) * 2 ^ L := by
      rw [← zpow_add₀ two_ne_zero, show L - 52 = -53 + L + 1 by ring, zpow_add_one₀ two_ne_zero]
      ring
    rw [h3]
    have : (2 : ℝ) ^ (-53 : ℤ) * 2 ^ L ≤ 2 ^ (-53 : ℤ) * |a| :=
      mul_le_mul_of_nonneg_left hL1 (by positivity)
    have : (0 : ℝ) ≤ 2 ^ (-1075 : ℤ) := by positivity
    linarith

/-- Above the subnormals, `|a| ≥ 2^-1022`, rounding errs by at most `2^-53` of `|a|`. -/
lemma abs_sub_le_mul {a : ℝ} (ha : 2 ^ (-1022 : ℤ) ≤ |a|) :
    |fl a - a| ≤ 2 ^ (-53 : ℤ) * |a| := by
  have ha' : 0 < |a| := lt_of_lt_of_le (zpow_pos two_pos _) ha
  set L := Int.log 2 |a|
  have hL1 : (2 : ℝ) ^ L ≤ |a| := by
    have := Int.zpow_log_le_self (b := 2) (by norm_num) ha'
    exact_mod_cast this
  have hL2 : |a| < (2 : ℝ) ^ (L + 1) := by
    have := Int.lt_zpow_succ_log_self (b := 2) (by norm_num) |a|
    exact_mod_cast this
  have hL : -1022 ≤ L := by
    by_contra! h
    have : (2 : ℝ) ^ (L + 1) ≤ 2 ^ (-1022 : ℤ) := zpow_le_zpow_right₀ (by norm_num) (by omega)
    linarith
  obtain ⟨y, hy, hya⟩ := exists_isF64_abs_sub_le (e := L - 52) (by omega)
    (by rwa [show L - 52 + 53 = L + 1 by ring])
  refine (hfl.nearest a y hy).trans (hya.trans ?_)
  have h3 : (2 : ℝ) ^ (L - 52) / 2 = 2 ^ (-53 : ℤ) * 2 ^ L := by
    rw [← zpow_add₀ two_ne_zero, show L - 52 = -53 + L + 1 by ring, zpow_add_one₀ two_ne_zero]
    ring
  rw [h3]
  exact mul_le_mul_of_nonneg_left hL1 (by positivity)

/-- The rounded sum of two floats errs by at most `2^-53` of the sum: below `2^-1022` the sum,
a multiple of `2^-1074`, is a float itself. -/
lemma abs_add_sub_le {a b : ℝ} (ha : IsF64 a) (hb : IsF64 b) :
    |fl (a + b) - (a + b)| ≤ 2 ^ (-53 : ℤ) * |a + b| := by
  rcases le_or_gt (2 ^ (-1022 : ℤ)) |a + b| with h | h
  · exact hfl.abs_sub_le_mul h
  obtain ⟨n₁, rfl⟩ := ha.exists_int_mul
  obtain ⟨n₂, rfl⟩ := hb.exists_int_mul
  rw [← add_mul, ← Int.cast_add] at h ⊢
  have h2 : (0 : ℝ) < 2 ^ (-1074 : ℤ) := zpow_pos two_pos _
  have hn : |((n₁ + n₂ : ℤ) : ℝ)| < 2 ^ 52 := by
    rw [abs_mul, abs_of_pos h2,
      show (2 : ℝ) ^ (-1022 : ℤ) = 2 ^ 52 * 2 ^ (-1074 : ℤ) by
        rw [show (-1022 : ℤ) = 52 + -1074 by norm_num, zpow_add₀ two_ne_zero]; norm_num] at h
    exact lt_of_mul_lt_mul_right h h2.le
  have hn' : |n₁ + n₂| < 2 ^ 52 := by exact_mod_cast hn
  rw [hfl.eq_self (isF64_intCast_mul_zpow (by omega) le_rfl), sub_self, abs_zero]
  positivity

/-- The float below the rounded `a` lies at or below `a`: above it, it would be nearer. -/
lemma predF64_le (a : ℝ) : predF64 (fl a) ≤ a := by
  by_contra! h
  have h1 := hfl.nearest a (predF64 (fl a)) (isF64_predF64 _)
  have h2 := predF64_lt (fl a)
  rw [abs_of_pos (by linarith : 0 < fl a - a), abs_of_pos (by linarith : 0 < predF64 (fl a) - a)]
    at h1
  linarith

/-- The float above the rounded `a` lies at or above `a`. -/
lemma le_succF64 (a : ℝ) : a ≤ succF64 (fl a) := by
  by_contra! h
  have h1 := hfl.nearest a (succF64 (fl a)) (isF64_succF64 _)
  have h2 := lt_succF64 (fl a)
  rw [abs_of_neg (by linarith : fl a - a < 0), abs_of_neg (by linarith : succF64 (fl a) - a < 0)]
    at h1
  linarith

/-- `_float_at_or_above` returns the least float at or above `q`: a nearest float below `q` has
no float between it and `q`, and one at or above `q` is the least. -/
lemma atOrAbove_eq (q : ℝ) : atOrAbove fl q = ceilF64 q := by
  unfold atOrAbove
  split_ifs with h
  · exact le_antisymm (succF64_le (isF64_ceilF64 q) (h.trans_le (le_ceilF64 q)))
      (ceilF64_le (isF64_succF64 _) (hfl.le_succF64 q))
  · rw [not_lt] at h
    refine le_antisymm ?_ (ceilF64_le (hfl.isF64 q) h)
    by_contra! hlt
    have h1 := hfl.nearest q (ceilF64 q) (isF64_ceilF64 q)
    have h2 := le_ceilF64 q
    rw [abs_of_nonneg (by linarith : 0 ≤ fl q - q), abs_of_nonneg (by linarith : 0 ≤ ceilF64 q - q)]
      at h1
    linarith

/-- `_float_at_or_below` returns the greatest float at or below `q`. -/
lemma atOrBelow_eq (q : ℝ) : atOrBelow fl q = floorF64 q := by
  rw [atOrBelow, hfl.atOrAbove_eq, ceilF64, neg_neg, neg_neg]

end IsRoundNearest

end DiffGPT
