module

public import Mathlib
public import DiffGPT.LevelGrid
public import DiffGPT.Machine

/-!
# "mash-half" on flat stretches

"mash-half" steps its anchor by half a level, `c(B) A = 1/2` in units of levels
(`mul_diffs_half_anchor`); in the code the anchor is `P / 2^G`, `P = anchorInt l seeds step`,
with `step / 2^G = 1/2` (`diffs_anchorInt_div`). On a constant series that has a price. "mash"
can take the constant anchor (`invert_const`), as the code does for the constant seeds of a
constant prefix (`anchorInt_const`): its levels of a constant series are constant, and `c(B)`
takes them to `0` (`diffs_const`). "mash-half" cannot:

* `not_const_mod_one`: `c(B)` takes integers to integers, so an anchor with `c(B) A n = 1/2` is
  on no window `n, …, n + p` an integer plus a constant. The decoded `A + L/s` with any integer
  levels `L` misses a constant `x₀` somewhere on every window (`exists_decode_ne`), by at least
  `1/(2‖c‖₁·s)` for positive lags (`exists_le_abs_add_sub`, `exists_le_abs_decode_sub`), where
  every decoded value is within `1/(2s)` (`abs_decode_sub_le`). In the code: `(L + F)/s` with
  the anchor's fractions `F`, whatever the seeds (`exists_decode_anchorFrac_ne`), before the
  roundings of float64.
* `Δᵏ`, `k ≥ 1`: the code's anchor of a constant series is `σ + step·C(m, k)`
  (`anchorInt_replicate`), `φ + C(m, k)/2` levels with `φ = σ / 2^G`, and its token sums are
  `c(B)` of the levels `⌊v - φ - C(m, k)/2 + 1/2⌋` of `v = s·x` (`token_replicate_half`):
  `2·T n = -1 + (1 - 2δ)·c(B) r n` (`two_mul_diffs_levels_replicate`) with the parities
  `r m = C(m, k) mod 2` and `δ = ⌊v - φ + 1/2⌋ - ⌊v - φ⌋`, `0` or `1`. By Lucas the parities
  repeat with any period `2^j > k` (`choose_add_two_pow_mod_two`), and so do the tokens
  (`diffs_levels_replicate_add_two_pow`, `token_replicate_half_add_two_pow`). They are not
  constant (`exists_diffs_levels_replicate_ne`): `c(B) r` is `1` at `0` (`diffs_parity_zero`)
  and sums to `0` over a period (`sum_diffs_parity`). At `Δ` they alternate `-δ` and `δ - 1`
  (`diffs_levels_one`): about the zero point `z`, the tokens `z, z - 1, z, …` for `δ = 0` and
  `z - 1, z, z - 1, …` for `δ = 1`, where "mash" gives `z, z, z, …`.
* Any lags: no anchor with `c(B) A = 1/2` at every row gives a constant series constant tokens
  (`not_forall_diffs_levels_eq`): `c(B)` of the rounding errors would be the constant
  `κ + 1/2 ≠ 0`, but `c(B)` of a bounded sequence is no nonzero constant (`diffs_ne_const`),
  since with `c = (1 - B^S)·c'` the bounded `c'(B)` of it would grow by it every `S` rows. In the
  code: the token sums of a constant series, whatever the seeds (`not_forall_token_half_eq`).
-/

@[expose] public section

open fwdDiff

namespace DiffGPT

/-! ### Constants -/

section

variable {G : Type*} [AddCommGroup G]

/-- `c(B)` takes constants to `0`: `c(1) = 0`. -/
lemma diffs_const {l : List ℕ} (hl0 : l ≠ []) (c : G) : diffs l (fun _ => c) = 0 := by
  obtain ⟨S, l, rfl⟩ := List.exists_cons_of_ne_nil hl0
  have h1 : Δ_[S] (fun _ : ℕ => c) = 0 := by
    funext m
    simp [fwdDiff]
  rw [diffs_cons, h1, diffs_zero]

/-- "mash" can take a constant anchor: `invert` of a constant prefix with the steps `0` is the
constant. -/
lemma invert_const {l : List ℕ} (hl : ∀ S ∈ l, 0 < S) (hl0 : l ≠ []) (c : G) :
    invert l (fun _ => c) 0 = fun _ => c := by
  funext m
  exact eq_of_diffs_eq_of_lt hl (M := m + 1)
    (fun n _ => by rw [diffs_invert hl, diffs_const hl0]) (fun i hi => invert_of_lt hl _ _ hi)
    (Nat.lt_succ_self m)

/-- `c(B)` commutes with `Δ_[S]`. -/
lemma diffs_fwdDiff (l : List ℕ) (S : ℕ) (f : ℕ → G) :
    diffs l (Δ_[S] f) = Δ_[S] (diffs l f) := by
  induction l generalizing f with
  | nil => rfl
  | cons T l ih =>
    have h : Δ_[T] (Δ_[S] f) = Δ_[S] (Δ_[T] f) := by
      funext m
      simp only [fwdDiff, add_right_comm m S T]
      abel
    rw [diffs_cons, diffs_cons, h, ih]

end

/-! ### A half step is no integer plus a constant -/

/-- An anchor that steps by half a level at `n`, `c(B) A n = 1/2`, is no integer plus a
constant on the window `n, …, n + p`: `c(B)` takes integers to integers and the constant
to `0`. -/
theorem not_const_mod_one {l : List ℕ} (hl0 : l ≠ []) {A : ℕ → ℝ} {n : ℕ}
    (hA : diffs l A n = 1 / 2) : ¬∃ φ : ℝ, ∀ j ≤ l.sum, ∃ k : ℤ, A (n + j) = k + φ := by
  rintro ⟨φ, hφ⟩
  choose! k hk using hφ
  have h : diffs l A n = diffs l ((fun m => ((k (m - n) : ℤ) : ℝ)) + fun _ => φ) n :=
    diffs_congr fun i hi => by simp only [hk i hi, Pi.add_apply, Nat.add_sub_cancel_left]
  rw [h, diffs_add, Pi.add_apply, diffs_const hl0, Pi.zero_apply, add_zero, ← cast_diffs] at hA
  have h2 : ((2 * diffs l (fun m => k (m - n)) n : ℤ) : ℝ) = 1 := by
    push_cast
    rw [hA]
    norm_num
  have h3 : 2 * diffs l (fun m => k (m - n)) n = 1 := by exact_mod_cast h2
  omega

/-- So the decoded `A + L/s`, `s·c(B) A n = 1/2` (`mul_diffs_half_anchor`), with any integer
levels `L` misses a constant `x₀` somewhere on the window: a constant series does not come back
exactly. -/
theorem exists_decode_ne {l : List ℕ} (hl0 : l ≠ []) {s : ℝ} (hs : s ≠ 0) {A : ℕ → ℝ} {n : ℕ}
    (hA : s * diffs l A n = 1 / 2) (L : ℕ → ℤ) (x₀ : ℝ) :
    ∃ j ≤ l.sum, A (n + j) + L (n + j) / s ≠ x₀ := by
  by_contra! h
  refine not_const_mod_one hl0 (A := s • A)
    (by rw [diffs_const_smul, Pi.smul_apply, smul_eq_mul, hA])
    ⟨s * x₀, fun j hj => ⟨-L (n + j), ?_⟩⟩
  have h1 := h j hj
  have h2 : s * ((L (n + j) : ℝ) / s) = L (n + j) := by field_simp
  rw [Pi.smul_apply, smul_eq_mul, Int.cast_neg]
  linear_combination s * h1 - h2

/-- By how much: some value of the window is at least `1/(2‖c‖₁)` from the constant, since
`c(B)` of the differences is `1/2` plus an integer, at least `1/2` in magnitude, and less than
`‖c‖₁ / (2‖c‖₁)` if they all were closer. -/
theorem exists_le_abs_add_sub {l : List ℕ} (hl : ∀ S ∈ l, 0 < S) (hl0 : l ≠ []) {A : ℕ → ℝ}
    {n : ℕ} (hA : diffs l A n = 1 / 2) (L : ℕ → ℤ) (c : ℝ) :
    ∃ j ≤ l.sum, 1 / (2 * norm1 l) ≤ |A (n + j) + L (n + j) - c| := by
  by_contra! h
  have hn : (2 : ℝ) ≤ norm1 l := by exact_mod_cast two_le_norm1 hl hl0
  have he : diffs l (fun m => A m + L m - c) n = 1 / 2 + diffs l L n := by
    have e : (fun m => A m + (L m : ℝ) - c) =
        A + ((fun m => (L m : ℝ)) + (-1 : ℝ) • fun _ => c) := by
      funext m
      simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul]
      ring
    rw [e, diffs_add, diffs_add, diffs_const_smul, diffs_const hl0, smul_zero, add_zero,
      Pi.add_apply, hA, cast_diffs]
  have hsum : ∑ j ∈ Finset.range (l.sum + 1), |(coeff l j : ℝ)| = norm1 l := by
    rw [norm1]
    push_cast
    rfl
  have hb : |diffs l (fun m => A m + L m - c) n| < 1 / 2 := by
    rw [diffs_eq_sum]
    calc |∑ j ∈ Finset.range (l.sum + 1), coeff l j • (A (n + j) + L (n + j) - c)|
        ≤ ∑ j ∈ Finset.range (l.sum + 1), |(coeff l j : ℝ)| * |A (n + j) + L (n + j) - c| := by
          refine (Finset.abs_sum_le_sum_abs _ _).trans_eq (Finset.sum_congr rfl fun j _ => ?_)
          rw [zsmul_eq_mul, abs_mul]
      _ < ∑ j ∈ Finset.range (l.sum + 1), |(coeff l j : ℝ)| * (1 / (2 * norm1 l)) := by
          refine Finset.sum_lt_sum (fun j hj => mul_le_mul_of_nonneg_left
            (h j (Nat.lt_succ_iff.mp (Finset.mem_range.mp hj))).le (abs_nonneg _))
            ⟨l.sum, Finset.self_mem_range_succ _, ?_⟩
          rw [coeff_sum_self hl, Int.cast_one, abs_one, one_mul, one_mul]
          exact h l.sum le_rfl
      _ = 1 / 2 := by
          rw [← Finset.sum_mul, hsum]
          field_simp
  rw [he] at hb
  obtain ⟨h1, h2⟩ := abs_lt.mp hb
  have h3 : ((diffs l L n : ℤ) : ℝ) < 0 := by linarith
  have h4 : (-1 : ℝ) < ((diffs l L n : ℤ) : ℝ) := by linarith
  have h5 : diffs l L n < 0 := by exact_mod_cast h3
  have h6 : -1 < diffs l L n := by exact_mod_cast h4
  omega

/-- In data units: some decoded value of the window is at least `1/(2‖c‖₁·s)` from `x₀`, where
every one is within `1/(2s)` (`abs_decode_sub_le`). -/
theorem exists_le_abs_decode_sub {l : List ℕ} (hl : ∀ S ∈ l, 0 < S) (hl0 : l ≠ []) {s : ℝ}
    (hs : 0 < s) {A : ℕ → ℝ} {n : ℕ} (hA : s * diffs l A n = 1 / 2) (L : ℕ → ℤ) (x₀ : ℝ) :
    ∃ j ≤ l.sum, 1 / (2 * norm1 l * s) ≤ |A (n + j) + L (n + j) / s - x₀| := by
  obtain ⟨j, hj, h⟩ := exists_le_abs_add_sub hl hl0 (A := s • A) (n := n)
    (by rw [diffs_const_smul, Pi.smul_apply, smul_eq_mul, hA]) L (s * x₀)
  refine ⟨j, hj, ?_⟩
  have h2 : s * ((L (n + j) : ℝ) / s) = L (n + j) := by field_simp
  have e : (s • A) (n + j) + L (n + j) - s * x₀ = s * (A (n + j) + L (n + j) / s - x₀) := by
    rw [Pi.smul_apply, smul_eq_mul]
    linear_combination -h2
  rw [e, abs_mul, abs_of_pos hs] at h
  rw [← div_div, div_le_iff₀ hs]
  linarith

/-! ### `Δᵏ` on a constant series -/

/-- `Δᵏ` of `m ↦ C(m, k)` is `1`. -/
lemma diffs_replicate_choose (k n : ℕ) :
    diffs (List.replicate k 1) (fun m => (m.choose k : ℤ)) n = 1 := by
  have h := fwdDiff_iter_choose (R := ℤ) 0 k
  rw [add_zero] at h
  rw [diffs_replicate, h]
  simp

/-- `invert` for `Δᵏ`, `k ≥ 1`, of a constant prefix and constant differences `d`: the constant
plus `d·C(m, k)`. -/
lemma invert_replicate_const {k : ℕ} (hk : 0 < k) (σ d : ℤ) (m : ℕ) :
    invert (List.replicate k 1) (fun _ => σ) (fun _ => d) m = σ + d * m.choose k := by
  have hl : ∀ S ∈ List.replicate k 1, 0 < S := fun S hS => by
    rw [List.eq_of_mem_replicate hS]
    exact Nat.one_pos
  have hl0 : List.replicate k 1 ≠ [] := by simp [hk.ne']
  refine eq_of_diffs_eq_of_lt hl (M := m + 1) (L' := fun m => σ + d * m.choose k)
    (fun n _ => ?_) (fun i hi => ?_) (Nat.lt_succ_self m)
  · have h : (fun m : ℕ => σ + d * (m.choose k : ℤ)) =
        (fun _ => σ) + d • fun m => (m.choose k : ℤ) := by
      funext m
      simp
    rw [diffs_invert hl, h, diffs_add, diffs_const_smul, diffs_const hl0, Pi.add_apply,
      Pi.smul_apply, diffs_replicate_choose]
    simp
  · have hik : i < k := by simpa using hi
    rw [invert_of_lt hl _ _ hi]
    simp [Nat.choose_eq_zero_of_lt hik]

open Finset.HasAntidiagonal in
/-- Lucas for the period: `C(m + 2^j, k) ≡ C(m, k) mod 2` for `k < 2^j`, by Vandermonde, since
`C(2^j, i)` is even for `0 < i < 2^j`. -/
lemma choose_add_two_pow_mod_two {j k : ℕ} (hk : k < 2 ^ j) (m : ℕ) :
    (m + 2 ^ j).choose k % 2 = m.choose k % 2 := by
  have hmem : ((k, 0) : ℕ × ℕ) ∈ antidiagonal k := by simp
  have h2 : 2 ∣ ∑ ij ∈ (antidiagonal k).erase (k, 0),
      m.choose ij.1 * (2 ^ j).choose ij.2 := by
    refine Finset.dvd_sum fun ij hij => Dvd.dvd.mul_left (Nat.prime_two.dvd_choose_pow ?_ ?_) _
    · intro h0
      obtain ⟨hne, hij'⟩ := Finset.mem_erase.mp hij
      rw [mem_antidiagonal, h0, add_zero] at hij'
      exact hne (Prod.ext hij' h0)
    · obtain ⟨-, hij'⟩ := Finset.mem_erase.mp hij
      rw [mem_antidiagonal] at hij'
      omega
  obtain ⟨c, hc⟩ := h2
  rw [Nat.add_choose_eq, ← Finset.add_sum_erase _ _ hmem, hc]
  simp only [Nat.choose_zero_right, mul_one]
  omega

/-- `c(B)` of the parities of `C(m, k)` has the period `2^j > k`. -/
lemma diffs_parity_add_two_pow (l : List ℕ) {j k : ℕ} (hk : k < 2 ^ j) (n : ℕ) :
    diffs l (fun m => ((m.choose k % 2 : ℕ) : ℤ)) (n + 2 ^ j) =
      diffs l (fun m => ((m.choose k % 2 : ℕ) : ℤ)) n := by
  rw [diffs_eq_sum, diffs_eq_sum]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [Nat.add_right_comm n (2 ^ j) i, choose_add_two_pow_mod_two hk]

/-- The level of `u - C/2`, `C = 2q + r`, is `⌊u + 1/2⌋ - q` for `r = 0` and `⌊u⌋ - q` for
`r = 1`: `2·⌊u - C/2 + 1/2⌋ = 2⌊u⌋ + 2δ - C + (1 - 2δ)·r` with `δ = ⌊u + 1/2⌋ - ⌊u⌋`. -/
lemma two_mul_roundHalfUp_sub_half (u : ℝ) (C : ℕ) :
    2 * roundHalfUp (u - (C : ℝ) / 2) = 2 * ⌊u⌋ + 2 * (⌊u + 1 / 2⌋ - ⌊u⌋) - C +
      (1 - 2 * (⌊u + 1 / 2⌋ - ⌊u⌋)) * ((C % 2 : ℕ) : ℤ) := by
  obtain ⟨q, hq⟩ : ∃ q, C = 2 * q + C % 2 := ⟨C / 2, (Nat.div_add_mod C 2).symm⟩
  rcases Nat.mod_two_eq_zero_or_one C with h | h <;> rw [h] at hq ⊢ <;> subst hq
  · have e : u - ((2 * q + 0 : ℕ) : ℝ) / 2 + 1 / 2 = u + 1 / 2 - ((q : ℤ) : ℝ) := by
      push_cast
      ring
    rw [roundHalfUp, e, Int.floor_sub_intCast]
    push_cast
    ring
  · have e : u - ((2 * q + 1 : ℕ) : ℝ) / 2 + 1 / 2 = u - ((q : ℤ) : ℝ) := by
      push_cast
      ring
    rw [roundHalfUp, e, Int.floor_sub_intCast]
    push_cast
    ring

/-- The levels of a constant series `v` about the anchor `φ + C(m, k)/2` of `Δᵏ`, `k ≥ 1`:
`2·c(B) L n = -1 + (1 - 2δ)·c(B) r n` with the parities `r m = C(m, k) mod 2` and
`δ = ⌊v - φ + 1/2⌋ - ⌊v - φ⌋`, `0` or `1`. -/
theorem two_mul_diffs_levels_replicate {k : ℕ} (hk : 0 < k) (φ v : ℝ) (n : ℕ) :
    2 * diffs (List.replicate k 1) (fun m => roundHalfUp (v - (φ + (m.choose k : ℝ) / 2))) n =
      -1 + (1 - 2 * (⌊v - φ + 1 / 2⌋ - ⌊v - φ⌋)) *
        diffs (List.replicate k 1) (fun m => ((m.choose k % 2 : ℕ) : ℤ)) n := by
  have hl0 : List.replicate k 1 ≠ [] := by simp [hk.ne']
  have h : (2 : ℤ) • (fun m : ℕ => roundHalfUp (v - (φ + (m.choose k : ℝ) / 2))) =
      (fun _ => 2 * ⌊v - φ⌋ + 2 * (⌊v - φ + 1 / 2⌋ - ⌊v - φ⌋)) +
        ((-1 : ℤ) • (fun m : ℕ => (m.choose k : ℤ)) +
          (1 - 2 * (⌊v - φ + 1 / 2⌋ - ⌊v - φ⌋)) • fun m : ℕ => ((m.choose k % 2 : ℕ) : ℤ)) := by
    funext m
    simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul]
    rw [sub_add_eq_sub_sub v φ, two_mul_roundHalfUp_sub_half]
    ring
  have h2 := congrFun (congrArg (diffs (List.replicate k 1)) h) n
  rw [diffs_const_smul, diffs_add, diffs_add, diffs_const_smul, diffs_const_smul,
    diffs_const hl0] at h2
  simp only [Pi.add_apply, Pi.smul_apply, Pi.zero_apply, smul_eq_mul] at h2
  rw [diffs_replicate_choose] at h2
  linear_combination h2

/-- The tokens have the period `2^j > k`: by Lucas the parities of `C(m, k)` have it. -/
theorem diffs_levels_replicate_add_two_pow {j k : ℕ} (hk : 0 < k) (hkj : k < 2 ^ j) (φ v : ℝ)
    (n : ℕ) :
    diffs (List.replicate k 1) (fun m => roundHalfUp (v - (φ + (m.choose k : ℝ) / 2)))
        (n + 2 ^ j) =
      diffs (List.replicate k 1) (fun m => roundHalfUp (v - (φ + (m.choose k : ℝ) / 2))) n := by
  have h := two_mul_diffs_levels_replicate hk φ v (n + 2 ^ j)
  rw [diffs_parity_add_two_pow _ hkj, ← two_mul_diffs_levels_replicate hk φ v n] at h
  exact mul_left_cancel₀ two_ne_zero h

/-- `c(B)` of the parities is `1` at `0`: `C(m, k)` is `0` below `k` and `1` at `k`. -/
lemma diffs_parity_zero (k : ℕ) :
    diffs (List.replicate k 1) (fun m => ((m.choose k % 2 : ℕ) : ℤ)) 0 = 1 := by
  rw [diffs_eq_sum, List.sum_replicate, smul_eq_mul, mul_one, Finset.sum_range_succ,
    Finset.sum_eq_zero fun i hi => by
      rw [zero_add, Nat.choose_eq_zero_of_lt (Finset.mem_range.mp hi)]
      simp]
  simp [coeff_replicate]

/-- Over the period `2^j > k + 1`, `c(B)` of the parities for `Δᵏ⁺¹` sums to `0`: it is the
difference of `Δᵏ` of them, which has the period as well. -/
lemma sum_diffs_parity {j k : ℕ} (hkj : k + 1 < 2 ^ j) :
    ∑ n ∈ Finset.range (2 ^ j),
      diffs (List.replicate (k + 1) 1) (fun m => ((m.choose (k + 1) % 2 : ℕ) : ℤ)) n = 0 := by
  have h (n : ℕ) :
      diffs (List.replicate (k + 1) 1) (fun m => ((m.choose (k + 1) % 2 : ℕ) : ℤ)) n =
        diffs (List.replicate k 1) (fun m => ((m.choose (k + 1) % 2 : ℕ) : ℤ)) (n + 1) -
          diffs (List.replicate k 1) (fun m => ((m.choose (k + 1) % 2 : ℕ) : ℤ)) n := by
    rw [diffs_replicate, diffs_replicate, Function.iterate_succ_apply']
    simp only [fwdDiff]
  have hp := diffs_parity_add_two_pow (List.replicate k 1) hkj 0
  rw [zero_add] at hp
  rw [Finset.sum_congr rfl fun n _ => h n, Finset.sum_range_sub, hp, sub_self]

/-- The tokens are not constant: equal tokens over the period `2^k` would make `c(B)` of the
parities `1` there (`two_mul_diffs_levels_replicate`, `1 - 2δ ≠ 0`), but it sums to `0`. -/
theorem exists_diffs_levels_replicate_ne {k : ℕ} (hk : 0 < k) (φ v : ℝ) :
    ∃ n < 2 ^ k,
      diffs (List.replicate k 1) (fun m => roundHalfUp (v - (φ + (m.choose k : ℝ) / 2))) n ≠
        diffs (List.replicate k 1) (fun m => roundHalfUp (v - (φ + (m.choose k : ℝ) / 2))) 0 := by
  by_contra! h
  have hδ : 1 - 2 * (⌊v - φ + 1 / 2⌋ - ⌊v - φ⌋) ≠ 0 := by omega
  have hr (n : ℕ) (hn : n < 2 ^ k) :
      diffs (List.replicate k 1) (fun m => ((m.choose k % 2 : ℕ) : ℤ)) n = 1 := by
    have h1 := two_mul_diffs_levels_replicate hk φ v n
    rw [h n hn, two_mul_diffs_levels_replicate hk φ v 0, diffs_parity_zero] at h1
    exact (mul_left_cancel₀ hδ (add_left_cancel h1)).symm
  obtain ⟨k, rfl⟩ : ∃ k', k = k' + 1 := ⟨k - 1, by omega⟩
  have hs := sum_diffs_parity (j := k + 1) Nat.lt_two_pow_self
  rw [Finset.sum_congr rfl fun n hn => hr n (Finset.mem_range.mp hn)] at hs
  simp at hs

/-- At `Δ` the tokens alternate `-δ` at even `n` and `δ - 1` at odd `n`. -/
theorem diffs_levels_one (φ v : ℝ) (n : ℕ) :
    diffs [1] (fun m => roundHalfUp (v - (φ + (m : ℝ) / 2))) n =
      if n % 2 = 0 then -(⌊v - φ + 1 / 2⌋ - ⌊v - φ⌋) else ⌊v - φ + 1 / 2⌋ - ⌊v - φ⌋ - 1 := by
  have h := two_mul_diffs_levels_replicate one_pos φ v n
  simp only [List.replicate_one, Nat.choose_one_right, diffs_cons, diffs_nil, fwdDiff] at h ⊢
  split_ifs with hn
  · rw [show (n + 1) % 2 = 1 by omega, hn] at h
    push_cast at h ⊢
    omega
  · rw [show (n + 1) % 2 = 0 by omega, show n % 2 = 1 by omega] at h
    push_cast at h ⊢
    omega

/-! ### Any lags -/

/-- `c(B)` of values within `1/2` of `0` stays within `‖c‖₁/2`. -/
lemma abs_diffs_le_half {l : List ℕ} {ζ : ℕ → ℝ} (hζ : ∀ m, |ζ m| ≤ 1 / 2) (n : ℕ) :
    |diffs l ζ n| ≤ (norm1 l : ℝ) / 2 := by
  rw [diffs_eq_sum, norm1]
  push_cast
  rw [Finset.sum_div]
  refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun j _ => ?_)
  rw [zsmul_eq_mul, abs_mul]
  calc |(coeff l j : ℝ)| * |ζ (n + j)| ≤ |(coeff l j : ℝ)| * (1 / 2) :=
        mul_le_mul_of_nonneg_left (hζ _) (abs_nonneg _)
    _ = |(coeff l j : ℝ)| / 2 := by ring

/-- `c(B)` of a bounded sequence is no nonzero constant: with `c = (1 - B^S)·c'`, `c'(B) ζ`
would grow by it every `S` rows. -/
lemma diffs_ne_const {l : List ℕ} (hl0 : l ≠ []) {ζ : ℕ → ℝ} (hζ : ∀ m, |ζ m| ≤ 1 / 2)
    {a : ℝ} (ha : a ≠ 0) : ¬∀ n, diffs l ζ n = a := by
  intro h
  obtain ⟨S, l, rfl⟩ := List.exists_cons_of_ne_nil hl0
  have hg (n : ℕ) : diffs l ζ (n + S) = diffs l ζ n + a := by
    have h1 := h n
    rw [diffs_cons, diffs_fwdDiff] at h1
    simp only [fwdDiff] at h1
    linarith
  have hN (N : ℕ) : diffs l ζ (N * S) = diffs l ζ 0 + N * a := by
    induction N with
    | zero => simp
    | succ N ih =>
      rw [add_one_mul, hg, ih]
      push_cast
      ring
  obtain ⟨N, hN'⟩ := exists_nat_gt ((norm1 l : ℝ) / |a|)
  have h1 := abs_diffs_le_half (l := l) hζ (N * S)
  have h2 := abs_diffs_le_half (l := l) hζ 0
  have h3 : |(N : ℝ) * a| ≤ norm1 l := by
    rw [show (N : ℝ) * a = diffs l ζ (N * S) - diffs l ζ 0 by rw [hN N]; ring]
    calc _ ≤ |diffs l ζ (N * S)| + |diffs l ζ 0| := abs_sub _ _
      _ ≤ _ := by linarith
  rw [abs_mul, Nat.abs_cast] at h3
  rw [div_lt_iff₀ (abs_pos.mpr ha)] at hN'
  linarith

/-- For any lags, an anchor that steps by half a level at every row, `c(B) A = 1/2`, gives the
constant series `v` tokens that are not all equal: `c(B)` of the rounding errors would be the
constant `κ + 1/2 ≠ 0`. -/
theorem not_forall_diffs_levels_eq {l : List ℕ} (hl0 : l ≠ []) {A : ℕ → ℝ}
    (hA : ∀ n, diffs l A n = 1 / 2) (v : ℝ) (κ : ℤ) :
    ¬∀ n, diffs l (fun m => roundHalfUp (v - A m)) n = κ := by
  intro h
  refine diffs_ne_const hl0 (ζ := fun m => (roundHalfUp (v - A m) : ℝ) - (v - A m))
    (fun m => ?_) (a := κ + 1 / 2) ?_ fun n => ?_
  · obtain ⟨h1, h2⟩ := roundHalfUp_sub_mem (v - A m)
    exact abs_le.mpr ⟨h1.le, h2⟩
  · intro h0
    have h1 : ((2 * κ + 1 : ℤ) : ℝ) = 0 := by
      push_cast
      linarith
    have h2 : 2 * κ + 1 = 0 := by exact_mod_cast h1
    omega
  · have e : (fun m => (roundHalfUp (v - A m) : ℝ) - (v - A m)) =
        (fun m => ((roundHalfUp (v - A m) : ℤ) : ℝ)) + ((-1 : ℝ) • fun _ => v) + A := by
      funext m
      simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul]
      ring
    rw [e, diffs_add, diffs_add, diffs_const_smul, diffs_const hl0, smul_zero, add_zero,
      Pi.add_apply, ← cast_diffs, h n, hA n]

/-! ### The code -/

/-- The code's "mash" anchor of a constant series is constant: constant seeds `σ`, step `0`. -/
lemma anchorInt_const {l : List ℕ} (hl : ∀ S ∈ l, 0 < S) (hl0 : l ≠ []) (σ : UInt64) :
    anchorInt l (fun _ => σ) 0 = fun _ => (σ.toNat : ℤ) := by
  have h : (fun _ : ℕ => ((0 : UInt64).toNat : ℤ)) = 0 := by
    funext m
    simp
  rw [anchorInt, h]
  exact invert_const hl hl0 _

/-- The code's anchor of a constant series for `Δᵏ`, `k ≥ 1`: `_anchor_fraction` takes constant
seeds `σ` off a constant prefix, and `invert` makes `σ + step·C(m, k)` of them. -/
theorem anchorInt_replicate {k : ℕ} (hk : 0 < k) (σ step : UInt64) (m : ℕ) :
    anchorInt (List.replicate k 1) (fun _ => σ) step m = σ.toNat + step.toNat * m.choose k :=
  invert_replicate_const hk _ _ m

/-- The anchor in levels, `P / 2^G`, steps by `step / 2^G` levels. -/
lemma diffs_anchorInt_div {l : List ℕ} (hl : ∀ S ∈ l, 0 < S) (G : ℕ) (seeds : ℕ → UInt64)
    (step : UInt64) (n : ℕ) :
    diffs l (fun m => (anchorInt l seeds step m : ℝ) / 2 ^ G) n = (step.toNat : ℝ) / 2 ^ G := by
  have e : (fun m => (anchorInt l seeds step m : ℝ) / 2 ^ G) =
      ((2 : ℝ) ^ G)⁻¹ • fun m => ((anchorInt l seeds step m : ℤ) : ℝ) := by
    funext m
    simp [div_eq_inv_mul]
  rw [e, diffs_const_smul, Pi.smul_apply, ← cast_diffs, diffs_anchorInt hl, smul_eq_mul,
    Int.cast_natCast, div_eq_inv_mul]

/-- "mash-half"'s step, `step / 2^G = 1/2`, lies in `[0, 2^G)`, as `token_eq` needs. -/
lemma diffs_anchorInt_mem_half {l : List ℕ} (hl : ∀ S ∈ l, 0 < S) {G : ℕ}
    (seeds : ℕ → UInt64) {step : UInt64} (hh : (step.toNat : ℝ) / 2 ^ G = 1 / 2) (n : ℕ) :
    0 ≤ diffs l (anchorInt l seeds step) n ∧ diffs l (anchorInt l seeds step) n < 2 ^ G := by
  rw [diffs_anchorInt hl]
  have h2 : (0 : ℝ) < 2 ^ G := by positivity
  have h : (step.toNat : ℝ) < 2 ^ G := by
    rw [div_eq_iff h2.ne'] at hh
    linarith
  exact ⟨by positivity, by exact_mod_cast h⟩

/-- "mash-half" decodes `(L + F)/s` from integer levels `L` and the anchor's fractions `F`: on
no window are these all `x₀`, whatever the seeds. -/
theorem exists_decode_anchorFrac_ne {l : List ℕ} (hl : ∀ S ∈ l, 0 < S) (hl0 : l ≠ []) {G : ℕ}
    (seeds : ℕ → UInt64) {step : UInt64} (hh : (step.toNat : ℝ) / 2 ^ G = 1 / 2) {s : ℝ}
    (hs : s ≠ 0) (L : ℕ → ℤ) (x₀ : ℝ) (n : ℕ) :
    ∃ j ≤ l.sum, (L (n + j) + anchorFrac G (anchorInt l seeds step) (n + j)) / s ≠ x₀ := by
  by_contra! h
  refine not_const_mod_one hl0 (A := fun m => (anchorInt l seeds step m : ℝ) / 2 ^ G) (n := n)
    (by rw [diffs_anchorInt_div hl, hh]) ⟨s * x₀, fun j hj =>
      ⟨anchorInt l seeds step (n + j) / 2 ^ G - L (n + j), ?_⟩⟩
  have ht : (2 : ℝ) ^ G ≠ 0 := by positivity
  have h1 := h j hj
  have h2 := congrArg (fun z : ℤ => (z : ℝ))
    (Int.emod_def (anchorInt l seeds step (n + j)) (2 ^ G))
  push_cast at h2
  rw [div_eq_iff hs, anchorFrac, h2, sub_div, mul_div_cancel_left₀ _ ht] at h1
  push_cast
  linarith

/-- "mash-half" on a constant series with `Δᵏ`, `s·x = v`: constant seeds `σ` and
`step / 2^G = 1/2` make the token sums `c(B)` of the levels of `v` about `φ + C(m, k)/2`,
`φ = σ / 2^G`. -/
theorem token_replicate_half {k G : ℕ} (hk : 0 < k) (σ : UInt64) {step : UInt64}
    (hh : (step.toNat : ℝ) / 2 ^ G = 1 / 2) (v : ℝ) (n : ℕ) :
    token (List.replicate k 1) G (anchorInt (List.replicate k 1) (fun _ => σ) step)
        (fun m => v - anchorFrac G (anchorInt (List.replicate k 1) (fun _ => σ) step) m) n =
      diffs (List.replicate k 1)
        (fun m => roundHalfUp (v - ((σ.toNat : ℝ) / 2 ^ G + (m.choose k : ℝ) / 2))) n := by
  have hl : ∀ S ∈ List.replicate k 1, 0 < S := fun S hS => by
    rw [List.eq_of_mem_replicate hS]
    exact Nat.one_pos
  rw [token_eq (diffs_anchorInt_mem_half hl _ hh n)]
  congr 1
  funext m
  congr 1
  rw [sub_add_cancel, anchorInt_replicate hk]
  push_cast
  linear_combination (-(m.choose k : ℝ)) * hh

/-- So those token sums repeat with the period `2^j > k`. -/
theorem token_replicate_half_add_two_pow {j k G : ℕ} (hk : 0 < k) (hkj : k < 2 ^ j)
    (σ : UInt64) {step : UInt64} (hh : (step.toNat : ℝ) / 2 ^ G = 1 / 2) (v : ℝ) (n : ℕ) :
    token (List.replicate k 1) G (anchorInt (List.replicate k 1) (fun _ => σ) step)
        (fun m => v - anchorFrac G (anchorInt (List.replicate k 1) (fun _ => σ) step) m)
        (n + 2 ^ j) =
      token (List.replicate k 1) G (anchorInt (List.replicate k 1) (fun _ => σ) step)
        (fun m => v - anchorFrac G (anchorInt (List.replicate k 1) (fun _ => σ) step) m) n := by
  rw [token_replicate_half hk σ hh, token_replicate_half hk σ hh]
  exact diffs_levels_replicate_add_two_pow hk hkj _ v n

/-- For any lags "mash-half" gives a constant series token sums that are not all equal, whatever
the seeds. -/
theorem not_forall_token_half_eq {l : List ℕ} (hl : ∀ S ∈ l, 0 < S) (hl0 : l ≠ []) {G : ℕ}
    (seeds : ℕ → UInt64) {step : UInt64} (hh : (step.toNat : ℝ) / 2 ^ G = 1 / 2) (v : ℝ)
    (κ : ℤ) :
    ¬∀ n, token l G (anchorInt l seeds step)
      (fun m => v - anchorFrac G (anchorInt l seeds step) m) n = κ := by
  intro h
  refine not_forall_diffs_levels_eq hl0 (A := fun m => (anchorInt l seeds step m : ℝ) / 2 ^ G)
    (fun n => by rw [diffs_anchorInt_div hl, hh]) v κ fun n => ?_
  rw [← h n, token_eq (diffs_anchorInt_mem_half hl seeds hh n)]
  congr 1
  funext m
  rw [sub_add_cancel]

end DiffGPT
