module

public import Mathlib
public import DiffGPT.LevelGrid

/-!
# No encoder beats the level grid

The decoder of the level grid reads the first `p = l.sum` values raw, integrates the tokens
`c(B) L n + V/2 ∈ [0, V - 1]` to the levels `L`, and decodes the later values to
`A m + L m / s`, for a scale `s` and an anchor `A` in the kernel, `c(B) A = 0`. An encoder for it
picks the levels of a series knowing all of it (`Encodes`).

**Theorem 2** (`le_mul_of_encodes`). For `V ≥ 2‖c‖₁`, an encoder that decodes every series with
`|c(B) x| ≤ D` to within `ε` has `D ≤ (V - ‖c‖₁)·ε`, whatever the scale, the prefix and the
anchor, "lgt" and "mash" alike. The level grid attains `ε = D / (V - ‖c‖₁)` with
`s = (V - ‖c‖₁)/(2D)` (`encodes_levels`), so no encoder does better. The order `0` has its own
bound, `le_mul_of_forall_exists_abs_sub_le`.

In units of levels the encoder's error is `E = s·ε`. Three series bound it:

* `min_le_of_encodes`: one step within the domain puts a value at least `min(1/2, s·D)` away
  from every level (`exists_far`), so `E ≥ min(1/2, s·D)`;
* `le_of_encodes_ramp`: on the series with `c(B) x = D` everywhere the errors stay bounded only
  if the tokens keep up, `s·D ≤ V/2 - 1`: the sums of `c(B)` of a bounded sequence are bounded,
  since `c(1) = 0` (`abs_sum_diffs_le`);
* `add_mul_lt_of_encodes`: values `η` off the levels with the signs of the coefficients, and
  `η + E < 1`, leave the encoder one level each, and with the step at the edge of the domain their
  token is `⌊s·D + ‖c‖₁·η⌋`, so `s·D + ‖c‖₁·η < V/2`.

At `E < 1/2` the first bound gives `E ≥ s·D ≥ s·D/(V - ‖c‖₁)`, at `E ≥ 1` the second gives
`s·D ≤ V/2 - 1 < (V - ‖c‖₁)·E`, and in between the third as `η → 1 - E` gives
`s·D ≤ V/2 - ‖c‖₁·(1 - E) ≤ (V - ‖c‖₁)·E`, where `V ≥ 2‖c‖₁` enters.
-/

@[expose] public section

namespace DiffGPT

/-- An encoder for the decoder of scale `s`, anchor `A` and prefix `P`, within `ε` on the series
with `|c(B) x| ≤ D`: for each such series it has levels `L` whose tokens `c(B) L n + V/2` lie in
`[0, V - 1]` and that decode to within `ε` of every value after the prefix. -/
def Encodes (V : ℤ) (l : List ℕ) (s D ε : ℝ) (P A : ℕ → ℝ) : Prop :=
  ∀ x : ℕ → ℝ, (∀ m < l.sum, x m = P m) → (∀ n, |diffs l x n| ≤ D) →
    ∃ L : ℕ → ℤ, (∀ n, -(V / 2) ≤ diffs l L n ∧ diffs l L n ≤ V / 2 - 1) ∧
      ∀ m, l.sum ≤ m → |A m + L m / s - x m| ≤ ε

lemma cast_half {V : ℤ} (hV : Even V) : ((V / 2 : ℤ) : ℝ) = (V : ℝ) / 2 := by
  obtain ⟨w, rfl⟩ := hV
  rw [show (w + w) / 2 = w by omega]
  push_cast
  ring

/-- A decoded value within `ε` of `x` is a level within `s·ε` of `s·(x - A)`. -/
lemma abs_level_sub_le {s ε A x : ℝ} {L : ℤ} (hs : 0 < s) (h : |A + L / s - x| ≤ ε) :
    |(L : ℝ) - s * (x - A)| ≤ s * ε := by
  rw [show (L : ℝ) - s * (x - A) = s * (A + L / s - x) by field_simp; ring, abs_mul,
    abs_of_pos hs]
  exact mul_le_mul_of_nonneg_left h hs.le

/-- The series with the prefix `P` that follows `invert l P 0 + v` up to the row `N + p`: its
differences are those of `v` up to `N` and `0` after. -/
lemma exists_of_prefix {l : List ℕ} (hl : ∀ S ∈ l, 0 < S) (P v : ℕ → ℝ)
    (hv : ∀ m < l.sum, v m = 0) (N : ℕ) :
    ∃ x : ℕ → ℝ, (∀ m < l.sum, x m = P m) ∧
      (∀ n, diffs l x n = if n ≤ N then diffs l v n else 0) ∧
      ∀ m ≤ N + l.sum, x m = invert l P 0 m + v m := by
  set d : ℕ → ℝ := fun n => if n ≤ N then diffs l v n else 0 with hd
  refine ⟨invert l P 0 + invert l 0 d, fun m hm => ?_, fun n => ?_, fun m hm => ?_⟩
  · rw [Pi.add_apply, invert_of_lt hl _ _ hm, invert_of_lt hl _ _ hm, Pi.zero_apply, add_zero]
  · rw [diffs_add, diffs_invert hl, diffs_invert hl, Pi.add_apply, Pi.zero_apply, zero_add]
  · rw [Pi.add_apply]
    congr 1
    refine eq_of_diffs_eq_of_lt hl (M := N + l.sum + 1) (fun n hn => ?_) (fun m hm => ?_)
      (by omega)
    · rw [diffs_invert hl]
      simp only [hd]
      rw [ite_eq_left (by omega)]
    · rw [invert_of_lt hl _ _ hm, hv m hm, Pi.zero_apply]

/-- Within `r` of any `z` lies a point at least `min(1/2, r)` away from every integer. -/
lemma exists_far (z : ℝ) {r : ℝ} (hr : 0 ≤ r) :
    ∃ u, |u| ≤ r ∧ ∀ K : ℤ, min (1 / 2) r ≤ |z + u - K| := by
  have h0 := Int.floor_le z
  have h1 := Int.lt_floor_add_one z
  have hm1 := min_le_left (1 / 2 : ℝ) r
  have hm2 := min_le_right (1 / 2 : ℝ) r
  rcases le_or_gt (z - ⌊z⌋) (1 / 2) with hf | hf
  · refine ⟨min r (1 / 2 - (z - ⌊z⌋)), ?_, fun K => ?_⟩
    · rw [abs_of_nonneg (le_min hr (by linarith))]
      exact min_le_left _ _
    · have hu := min_le_right r (1 / 2 - (z - ⌊z⌋))
      rcases le_or_gt K ⌊z⌋ with hK | hK
      · have hK' : (K : ℝ) ≤ ⌊z⌋ := by exact_mod_cast hK
        refine le_abs.mpr (Or.inl ?_)
        rcases le_total r (1 / 2 - (z - ⌊z⌋)) with h | h
        · rw [min_eq_left h]
          linarith
        · rw [min_eq_right h]
          linarith
      · have hK2 : ⌊z⌋ + 1 ≤ K := hK
        have hK' : (⌊z⌋ : ℝ) + 1 ≤ K := by exact_mod_cast hK2
        exact le_abs.mpr (Or.inr (by linarith))
  · refine ⟨-min r (z - ⌊z⌋ - 1 / 2), ?_, fun K => ?_⟩
    · rw [abs_neg, abs_of_nonneg (le_min hr (by linarith))]
      exact min_le_left _ _
    · have hu := min_le_right r (z - ⌊z⌋ - 1 / 2)
      rcases le_or_gt K ⌊z⌋ with hK | hK
      · have hK' : (K : ℝ) ≤ ⌊z⌋ := by exact_mod_cast hK
        exact le_abs.mpr (Or.inl (by linarith))
      · have hK2 : ⌊z⌋ + 1 ≤ K := hK
        have hK' : (⌊z⌋ : ℝ) + 1 ≤ K := by exact_mod_cast hK2
        refine le_abs.mpr (Or.inr ?_)
        rcases le_total r (z - ⌊z⌋ - 1 / 2) with h | h
        · rw [min_eq_left h]
          linarith
        · rw [min_eq_right h]
          linarith

/-- `c(1) = 0`, so the sums of `c(B) f` along a sequence bounded by `B` stay within
`2·p·‖c‖₁·B`: the sum of `c(B) f` over `N` rows is `∑ c_j` times the window sum of `f` shifted
by `j` less the unshifted one, which differ in `j` values at either end. -/
lemma abs_sum_diffs_le {l : List ℕ} (hl0 : l ≠ []) {f : ℕ → ℝ} {M : ℕ} {B : ℝ}
    (hB : ∀ m, M ≤ m → |f m| ≤ B) (N : ℕ) :
    |∑ n ∈ Finset.range N, diffs l f (M + n)| ≤ 2 * l.sum * norm1 l * B := by
  have hB0 : 0 ≤ B := (abs_nonneg _).trans (hB M le_rfl)
  have hsumB (k : ℕ) (g : ℕ → ℕ) (hg : ∀ n, M ≤ g n) :
      |∑ n ∈ Finset.range k, f (g n)| ≤ k * B :=
    calc |∑ n ∈ Finset.range k, f (g n)| ≤ ∑ n ∈ Finset.range k, |f (g n)| :=
          Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ n ∈ Finset.range k, B := Finset.sum_le_sum fun n _ => hB _ (hg n)
      _ = k * B := by simp
  have hwin (j : ℕ) : |∑ n ∈ Finset.range N, f (M + n + j) -
      ∑ n ∈ Finset.range N, f (M + n)| ≤ 2 * j * B := by
    have h1 := Finset.sum_range_add (fun n => f (M + n)) N j
    have h2 := Finset.sum_range_add (fun n => f (M + n)) j N
    rw [add_comm j N] at h2
    have h3 : ∑ n ∈ Finset.range N, f (M + n + j) = ∑ n ∈ Finset.range N, f (M + (j + n)) :=
      Finset.sum_congr rfl fun n _ => by rw [add_assoc, add_comm n j]
    have h4 := hsumB j (fun n => M + (N + n)) fun n => by omega
    have h5 := hsumB j (fun n => M + n) fun n => by omega
    rw [h3, show ∑ n ∈ Finset.range N, f (M + (j + n)) - ∑ n ∈ Finset.range N, f (M + n) =
      ∑ n ∈ Finset.range j, f (M + (N + n)) - ∑ n ∈ Finset.range j, f (M + n) by linarith]
    calc _ ≤ |∑ n ∈ Finset.range j, f (M + (N + n))| + |∑ n ∈ Finset.range j, f (M + n)| :=
          abs_sub _ _
      _ ≤ 2 * j * B := by linarith
  have hc : ∑ j ∈ Finset.range (l.sum + 1), (coeff l j : ℝ) = 0 := by
    exact_mod_cast sum_coeff_eq_zero hl0
  have hsum : ∑ n ∈ Finset.range N, diffs l f (M + n) =
      ∑ j ∈ Finset.range (l.sum + 1), (coeff l j : ℝ) *
        (∑ n ∈ Finset.range N, f (M + n + j) - ∑ n ∈ Finset.range N, f (M + n)) := by
    have h0 : ∑ j ∈ Finset.range (l.sum + 1), ∑ n ∈ Finset.range N,
        (coeff l j : ℝ) * f (M + n) = 0 := by
      rw [Finset.sum_comm]
      simp only [← Finset.sum_mul, hc, zero_mul, Finset.sum_const_zero]
    simp only [diffs_eq_sum, zsmul_eq_mul, mul_sub, Finset.sum_sub_distrib, Finset.mul_sum, h0,
      sub_zero]
    exact Finset.sum_comm
  rw [hsum]
  calc _ ≤ ∑ j ∈ Finset.range (l.sum + 1), |(coeff l j : ℝ) *
        (∑ n ∈ Finset.range N, f (M + n + j) - ∑ n ∈ Finset.range N, f (M + n))| :=
        Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ j ∈ Finset.range (l.sum + 1), |(coeff l j : ℝ)| * (2 * l.sum * B) :=
        Finset.sum_le_sum fun j hj => by
          rw [abs_mul]
          refine mul_le_mul_of_nonneg_left ((hwin j).trans ?_) (abs_nonneg _)
          have : (j : ℝ) ≤ l.sum := by
            exact_mod_cast Nat.lt_succ_iff.mp (Finset.mem_range.mp hj)
          nlinarith
    _ = 2 * l.sum * norm1 l * B := by
        rw [← Finset.sum_mul, norm1]
        push_cast
        ring

/-- One step within the domain puts a value at least `min(1/2, s·D)` away from every level. -/
theorem min_le_of_encodes {V : ℤ} {l : List ℕ} (hl : ∀ S ∈ l, 0 < S) {s D ε : ℝ} (hs : 0 < s)
    (hD : 0 ≤ D) {P A : ℕ → ℝ} (h : Encodes V l s D ε P A) : min (1 / 2) (s * D) ≤ s * ε := by
  set z : ℝ := s * (invert l P 0 l.sum - A l.sum) with hz
  obtain ⟨u, hu, hfar⟩ := exists_far z (r := s * D) (by positivity)
  set v : ℕ → ℝ := fun m => if m = l.sum then u / s else 0 with hv
  obtain ⟨x, hxP, hdx, hxv⟩ := exists_of_prefix hl P v (fun m hm => by
    simp only [hv]
    rw [ite_eq_right (by omega)]) 0
  have hv0 : diffs l v 0 = u / s := by
    rw [diffs_eq_sum, Finset.sum_eq_single l.sum (fun j _ hj => by simp [hv, hj])
      (fun h => absurd (Finset.self_mem_range_succ _) h)]
    simp [hv, coeff_sum_self hl]
  have hdom (n : ℕ) : |diffs l x n| ≤ D := by
    rw [hdx n]
    split_ifs with hn
    · rw [show n = 0 by omega, hv0, abs_div, abs_of_pos hs, div_le_iff₀ hs]
      linarith
    · simpa using hD
  obtain ⟨L, -, hLerr⟩ := h x hxP hdom
  have h1 := abs_level_sub_le hs (hLerr l.sum le_rfl)
  have h2 : s * (x l.sum - A l.sum) = z + u := by
    rw [hxv l.sum (by omega)]
    simp only [hv, hz, ite_true]
    field_simp
    ring
  rw [h2, abs_sub_comm] at h1
  exact (hfar (L l.sum)).trans h1

/-- On the series with `c(B) x = D` everywhere, bounded errors need `s·D ≤ V/2 - 1`: the tokens
must keep up with the steps. -/
theorem le_of_encodes_ramp {V : ℤ} (hV : Even V) {l : List ℕ} (hl : ∀ S ∈ l, 0 < S)
    (hl0 : l ≠ []) {s D ε : ℝ} (hs : 0 < s) (hD : 0 ≤ D) {P A : ℕ → ℝ} (hA : diffs l A = 0)
    (h : Encodes V l s D ε P A) : s * D ≤ (V : ℝ) / 2 - 1 := by
  set x : ℕ → ℝ := invert l P fun _ => D with hx
  have hdx : diffs l x = fun _ => D := diffs_invert hl P _
  obtain ⟨L, hLtok, hLerr⟩ := h x (fun m hm => invert_of_lt hl P _ hm) fun n => by
    rw [hdx, abs_of_nonneg hD]
  set e : ℕ → ℝ := fun m => (L m : ℝ) - s * (x m - A m) with he
  have hde (n : ℕ) : diffs l e n = ((diffs l L n : ℤ) : ℝ) - s * D := by
    have : e = (fun m => (L m : ℝ)) + (-s) • x + s • A := by
      funext m
      simp only [he, Pi.add_apply, Pi.smul_apply, smul_eq_mul]
      ring
    rw [this, diffs_add, diffs_add, diffs_const_smul, diffs_const_smul, hA, hdx, cast_diffs]
    simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul, Pi.zero_apply]
    ring
  have hE (m : ℕ) (hm : l.sum ≤ m) : |e m| ≤ s * ε := abs_level_sub_le hs (hLerr m hm)
  by_contra! hcon
  set δ := s * D - ((V : ℝ) / 2 - 1) with hδ
  have hδ0 : 0 < δ := by linarith
  obtain ⟨N, hN⟩ := exists_nat_gt (2 * l.sum * norm1 l * (s * ε) / δ)
  have hsum : ∑ n ∈ Finset.range N, diffs l e (l.sum + n) ≤ -(N * δ) := by
    calc _ ≤ ∑ n ∈ Finset.range N, -δ := Finset.sum_le_sum fun n _ => by
          have h1 : ((diffs l L (l.sum + n) : ℤ) : ℝ) ≤ ((V / 2 - 1 : ℤ) : ℝ) := by
            exact_mod_cast (hLtok _).2
          rw [Int.cast_sub, cast_half hV, Int.cast_one] at h1
          rw [hde]
          linarith
      _ = -(N * δ) := by simp
  have h2 := (abs_le.mp (abs_sum_diffs_le hl0 hE N)).1
  rw [div_lt_iff₀ hδ0] at hN
  linarith

/-- If the levels stay within `E = s·ε` of `s·(x - A)` after the prefix and `η + E < 1`, the
series whose values from the row `p` on lie `η` off the levels with the signs of the
coefficients, with the step at the edge of the domain, leaves the encoder one level for each of
them, and their token `⌊s·D + ‖c‖₁·η⌋` fits only if `s·D + ‖c‖₁·η < V/2`. -/
theorem add_mul_lt_of_encodes {V : ℤ} (hV : Even V) {l : List ℕ} (hl : ∀ S ∈ l, 0 < S)
    {s D ε η : ℝ} (hs : 0 < s) (hη : 0 ≤ η) (hηε : η + s * ε < 1) (hsD : 1 / 2 ≤ s * D)
    (hsDc : (norm1 l : ℝ) / 2 ≤ s * D) {P A : ℕ → ℝ} (hA : diffs l A = 0)
    (h : Encodes V l s D ε P A) : s * D + norm1 l * η < (V : ℝ) / 2 := by
  set x0 : ℕ → ℝ := invert l P 0 with hx0
  -- the levels of the series that follows the anchor through the prefix
  set z : ℕ → ℝ := fun m => s * (x0 m - A m) with hz
  have hdz : diffs l z = 0 := by
    have : z = s • x0 + (-s) • A := by
      funext m
      simp only [hz, Pi.add_apply, Pi.smul_apply, smul_eq_mul]
      ring
    rw [this, diffs_add, diffs_const_smul, diffs_const_smul, hA, hx0, diffs_invert hl]
    simp
  set σ : ℕ → ℝ := fun j => (((coeff l j).sign : ℤ) : ℝ) with hσ
  set Kt : ℤ := ⌊s * D + norm1 l * η⌋ with hKt
  set K0 : ℕ → ℤ := fun m => roundHalfUp (z m + σ (m - l.sum) * η) with hK0
  -- the levels: next to those of `z` on the rows `p, …, 2p - 1`, and on the row `2p` the one
  -- that makes `c(B) K` at `p` the token `Kt`
  set K : ℕ → ℤ := fun m =>
    if m = l.sum + l.sum then Kt - ∑ j ∈ Finset.range l.sum, coeff l j * K0 (l.sum + j)
    else K0 m with hK
  set t : ℕ → ℝ := fun m => (K m : ℝ) - σ (m - l.sum) * η with ht
  set v : ℕ → ℝ := fun m => if l.sum ≤ m then (t m - z m) / s else 0 with hv
  obtain ⟨x, hxP, hdx, hxv⟩ := exists_of_prefix hl P v (fun m hm => by
    simp only [hv]
    rw [ite_eq_right (by omega)]) l.sum
  have hKp : diffs l K l.sum = Kt := by
    rw [diffs_eq_sum, Finset.sum_range_succ, coeff_sum_self hl, one_smul]
    have h1 : ∀ j ∈ Finset.range l.sum, coeff l j • K (l.sum + j) = coeff l j * K0 (l.sum + j) :=
      fun j hj => by
        simp only [hK, smul_eq_mul]
        rw [ite_eq_right (by simp at hj; omega)]
    rw [Finset.sum_congr rfl h1]
    simp only [hK, ite_true]
    ring
  have htz (m : ℕ) (hm : m < l.sum + l.sum) : |t m - z m| ≤ 1 / 2 := by
    have h1 := roundHalfUp_sub_mem (z m + σ (m - l.sum) * η)
    simp only [ht, hK, hK0]
    rw [ite_eq_right (by omega)]
    rw [abs_le]
    constructor <;> linarith
  have hvb (m : ℕ) (hm : m < l.sum + l.sum) : |v m| ≤ 1 / (2 * s) := by
    simp only [hv]
    split_ifs
    · rw [abs_div, abs_of_pos hs]
      calc |t m - z m| / s ≤ 1 / 2 / s := div_le_div_of_nonneg_right (htz m hm) hs.le
        _ = 1 / (2 * s) := by ring
    · rw [abs_zero]
      positivity
  have hvp : s * diffs l v l.sum = Kt - norm1 l * η := by
    have h1 : diffs l v l.sum = diffs l (s⁻¹ • (t + (-1 : ℝ) • z)) l.sum :=
      diffs_congr fun i _ => by
        simp only [hv, Pi.smul_apply, Pi.add_apply, smul_eq_mul]
        rw [ite_eq_left (by omega)]
        ring
    have h2 : diffs l (fun m => σ (m - l.sum)) l.sum = norm1 l := by
      rw [diffs_eq_sum, norm1]
      push_cast
      exact Finset.sum_congr rfl fun j _ => by
        rw [Nat.add_sub_cancel_left, zsmul_eq_mul, hσ, (cast_sign (coeff l j)).2]
    have h3 : t = (fun m => (K m : ℝ)) + (-η) • fun m => σ (m - l.sum) := by
      funext m
      simp only [ht, Pi.add_apply, Pi.smul_apply, smul_eq_mul]
      ring
    rw [h1, diffs_const_smul, diffs_add, diffs_const_smul, hdz, h3, diffs_add, diffs_const_smul]
    simp only [Pi.smul_apply, Pi.add_apply, Pi.zero_apply, smul_eq_mul, h2, ← cast_diffs, hKp]
    field_simp
    ring
  have hdom (n : ℕ) : |diffs l x n| ≤ D := by
    rw [hdx n]
    split_ifs with hn
    · rcases lt_or_eq_of_le hn with hn | rfl
      · have h1 : |diffs l v n| ≤ norm1 l / (2 * s) := by
          rw [diffs_eq_sum]
          calc |∑ j ∈ Finset.range (l.sum + 1), coeff l j • v (n + j)|
              ≤ ∑ j ∈ Finset.range (l.sum + 1), |coeff l j • v (n + j)| :=
                Finset.abs_sum_le_sum_abs _ _
            _ ≤ ∑ j ∈ Finset.range (l.sum + 1), |(coeff l j : ℝ)| * (1 / (2 * s)) :=
                Finset.sum_le_sum fun j hj => by
                  rw [zsmul_eq_mul, abs_mul]
                  exact mul_le_mul_of_nonneg_left (hvb _ (by simp at hj; omega))
                    (abs_nonneg _)
            _ = norm1 l / (2 * s) := by
                rw [← Finset.sum_mul, norm1]
                push_cast
                ring
        refine h1.trans ?_
        rw [div_le_iff₀ (by positivity)]
        linarith
      · have h1 := Int.floor_le (s * D + norm1 l * η)
        have h2 := Int.lt_floor_add_one (s * D + norm1 l * η)
        rw [← hKt] at h1 h2
        rw [abs_le]
        constructor
        · rw [← mul_le_mul_iff_of_pos_left hs, hvp]
          linarith
        · rw [← mul_le_mul_iff_of_pos_left hs, hvp]
          linarith
    · rw [abs_zero]
      linarith [show 0 < D from pos_of_mul_pos_right (by linarith : 0 < s * D) hs.le]
  obtain ⟨L, hLtok, hLerr⟩ := h x hxP hdom
  -- within `η + E < 1` of `t m`, the only level is `K m`
  have hLK (m : ℕ) (hm1 : l.sum ≤ m) (hm2 : m ≤ l.sum + l.sum) : L m = K m := by
    have h1 := abs_level_sub_le hs (hLerr m hm1)
    have h2 : s * (x m - A m) = t m := by
      rw [hxv m (by omega)]
      simp only [hv, hz, ite_eq_left hm1]
      field_simp
      ring
    rw [h2] at h1
    have h3 : |t m - K m| ≤ η := by
      rw [show t m - K m = -(σ (m - l.sum) * η) by simp only [ht]; ring, abs_neg, abs_mul,
        abs_of_nonneg hη]
      exact mul_le_of_le_one_left hη (cast_sign _).1
    have h4 : |((L m - K m : ℤ) : ℝ)| < 1 := by
      push_cast
      calc |(L m : ℝ) - K m| ≤ |(L m : ℝ) - t m| + |t m - K m| := abs_sub_le _ _ _
        _ < 1 := by linarith
    have h5 : |L m - K m| < 1 := by exact_mod_cast h4
    have := Int.abs_lt_one_iff.mp h5
    omega
  have hLp : diffs l L l.sum = Kt := by
    rw [diffs_congr (g := K) fun i hi => hLK (l.sum + i) (by omega) (by omega), hKp]
  have h1 : ((Kt : ℤ) : ℝ) ≤ ((V / 2 - 1 : ℤ) : ℝ) := by
    rw [← hLp]
    exact_mod_cast (hLtok l.sum).2
  rw [Int.cast_sub, cast_half hV, Int.cast_one] at h1
  have h2 := Int.lt_floor_add_one (s * D + norm1 l * η)
  rw [← hKt] at h2
  linarith

/-- **Theorem 2.** No encoder does better than the level grid: one within `ε` on the series with
`|c(B) x| ≤ D` has `D ≤ (V - ‖c‖₁)·ε`, for any scale, prefix and anchor in the kernel, once
`V ≥ 2‖c‖₁`. -/
theorem le_mul_of_encodes {V : ℤ} (hV : Even V) {l : List ℕ} (hl : ∀ S ∈ l, 0 < S)
    (hl0 : l ≠ []) (hVc : 2 * norm1 l ≤ V) {s D ε : ℝ} (hs : 0 < s) (hD : 0 ≤ D)
    {P A : ℕ → ℝ} (hA : diffs l A = 0) (h : Encodes V l s D ε P A) :
    D ≤ (V - norm1 l) * ε := by
  have hc : (2 : ℝ) ≤ norm1 l := by exact_mod_cast two_le_norm1 hl hl0
  have hVc' : 2 * (norm1 l : ℝ) ≤ V := by exact_mod_cast hVc
  set c : ℝ := (norm1 l : ℝ) with hcdef
  set E : ℝ := s * ε with hE
  by_contra! hcon
  have hlt : (V - c) * E < s * D := by
    calc (V - c) * E = s * ((V - c) * ε) := by rw [hE]; ring
      _ < s * D := mul_lt_mul_of_pos_left hcon hs
  have hsD0 : 0 ≤ s * D := by positivity
  have h1 := min_le_of_encodes hl hs hD h
  have hE2 : 1 / 2 ≤ E := by
    rcases le_total (1 / 2) (s * D) with h' | h'
    · rwa [min_eq_left h'] at h1
    · rw [min_eq_right h'] at h1
      nlinarith [mul_nonneg (show (0 : ℝ) ≤ V - c - 1 by linarith) (show 0 ≤ E by linarith)]
  have hVcE := mul_nonneg (show (0 : ℝ) ≤ V - c by linarith) (sub_nonneg.mpr hE2)
  rcases lt_or_ge E 1 with hE1 | hE1
  · have hc0 : 0 < c := by linarith
    set η : ℝ := max 0 (((V : ℝ) / 2 - s * D) / c) with hη
    have hge : ((V : ℝ) / 2 - s * D) / c ≤ η := le_max_right _ _
    rw [div_le_iff₀ hc0] at hge
    have hηE : η + E < 1 := by
      rcases le_total (((V : ℝ) / 2 - s * D) / c) 0 with h' | h'
      · rw [hη, max_eq_left h']
        linarith
      · rw [hη, max_eq_right h']
        have : ((V : ℝ) / 2 - s * D) / c < 1 - E := by
          rw [div_lt_iff₀ hc0]
          nlinarith [mul_nonneg (show (0 : ℝ) ≤ V - 2 * c by linarith) (sub_nonneg.mpr hE2)]
        linarith
    have h2 := add_mul_lt_of_encodes hV hl hs (le_max_left _ _) hηE (by nlinarith)
      (by nlinarith) hA h
    linarith
  · have h3 := le_of_encodes_ramp hV hl hl0 hs hD hA h
    nlinarith [mul_le_mul_of_nonneg_left hE1 (show (0 : ℝ) ≤ V - c by linarith)]

/-- The level grid attains Theorem 2: with the scale `s = (V - ‖c‖₁)/(2D)` the levels
`roundHalfUp (s·(x - A))` decode every series with `|c(B) x| ≤ D` to within `D / (V - ‖c‖₁)`. -/
theorem encodes_levels {V : ℤ} (hV : Even V) {l : List ℕ} (hl : ∀ S ∈ l, 0 < S) (hl0 : l ≠ [])
    (hVc : norm1 l < V) {D : ℝ} (hD : 0 < D) (P : ℕ → ℝ) {A : ℕ → ℝ} (hA : diffs l A = 0) :
    Encodes V l ((V - norm1 l) / (2 * D)) D (D / (V - norm1 l)) P A := by
  intro x _ hx
  have hVc' : (0 : ℝ) < V - norm1 l := by
    have : (norm1 l : ℝ) < V := by exact_mod_cast hVc
    linarith
  have hs : (0 : ℝ) < (V - norm1 l) / (2 * D) := by positivity
  refine ⟨fun m => roundHalfUp ((V - norm1 l) / (2 * D) * (x m - A m)), fun n => ?_,
    fun m _ => ?_⟩
  · have := abs_diffs_levels_le hV hl hl0 hs.le hA (hx n)
      (le_of_eq (scale_mul_domain _ _ hD.ne'))
    obtain ⟨h1, h2⟩ := abs_le.mp this
    constructor <;> linarith
  · have := abs_decode_sub_le hs x A m
    rwa [half_step_eq hD.ne' hVc'.ne'] at this

/-- Over every scale and encoder, the least error within which the series with `|c(B) x| ≤ D`
decode is `D / (V - ‖c‖₁)`, the level grid's. -/
theorem isLeast_encodes {V : ℤ} (hV : Even V) {l : List ℕ} (hl : ∀ S ∈ l, 0 < S) (hl0 : l ≠ [])
    (hVc : 2 * norm1 l ≤ V) {D : ℝ} (hD : 0 < D) (P : ℕ → ℝ) {A : ℕ → ℝ} (hA : diffs l A = 0) :
    IsLeast {ε | ∃ s, 0 < s ∧ Encodes V l s D ε P A} (D / (V - norm1 l)) := by
  have hc := two_le_norm1 hl hl0
  have hVc' : (0 : ℝ) < V - norm1 l := by
    have : (norm1 l : ℝ) < V := by exact_mod_cast (show norm1 l < V by omega)
    linarith
  refine ⟨⟨(V - norm1 l) / (2 * D), by positivity,
    encodes_levels hV hl hl0 (by omega) hD P hA⟩, fun ε ⟨s, hs, h⟩ => ?_⟩
  rw [div_le_iff₀ hVc', mul_comm]
  exact le_mul_of_encodes hV hl hl0 hVc hs hD.le hA h

end DiffGPT
