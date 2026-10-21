module

public import Mathlib
public import DiffGPT.LevelGrid
public import DiffGPT.Float64

/-!
# `_max_scale` finds the largest float64 scale

`level_grid` sets the scale `s` of a column whose steps lie in `[lo, hi]`, `lo < hi`, with
`_max_scale`, which searches float64 for the largest `s` on which `[lo, hi]` fits. `_fits` is
`fitsRoom`: at the order `0` the levels `⌊s·x⌋` take `⌊s·hi⌋ - ⌊s·lo⌋ + 1` of the `room = V`
tokens (`zeroPoint0_mem`); above it the steps take `⌈s·hi - h⌉ - ⌊s·lo - h⌋` of the room
`V - ‖c‖₁ + 1 - t`, for the shift `h = c(B) A` and the least token `t`: `h = 0` and `t = 1`, or
`h = 1/2` and `t = 0` for "mash-half" (`Fits`, `fitsRoom_false_iff_fits`). At the order `0`,
`h = 0`.

A scale fits where its steps stay within `[p, p + room]` for some `p ∈ ℤ + h`, short of
`p + room` at the order `0` (`Within`, `within_of_fitsRoom`, `fitsRoom_of_within`). For
`lo < 0 < hi` the positive scales that do so for `p` are those up to the end
`min(p/lo, (p + room)/hi)` (`scaleEnd`, `le_scaleEnd_iff`), and the best `p` lies next to the
pivot `room·lo/(hi - lo)` (`scalePivot`), where both ends meet: `⌊pivot - h⌋ + h` or
`⌈pivot - h⌉ + h` (`within_candidate`). `maxScale` is `_max_scale`: for each of the two, it
rounds the end down to a float (`atOrBelow`, `_float_at_or_below`) and keeps it or the next float
toward `0` (`towardZero`, `math.nextafter(end, 0.0)`), the first that is positive and fits
(`firstFit`); the fallback `(room - 1)/(hi - lo)`, rounded down, fits whatever `p`
(`fitsRoom_of_mul_le`), and the result is the largest of them.

* `fitsRoom_maxScale`: the result fits.
* `lt_succF64_maxScale_mul`: it is at most a level short: the next float up, times `hi - lo`,
  exceeds `room - 1`.
* `le_maxScale`: for `lo < 0 < hi`, as on `[-D, D]`, every positive float that fits is at most
  the result. At the order `0` the end itself may put `s·hi` on `p + room`; then the float below
  it fits.
* `isF64_maxScale`, `maxScale_pos`: the result is a float, and positive for `room ≥ 2` and
  `hi - lo ≤ 2^1025`, which finite bounds meet; with `le_maxScale`, it is the largest positive
  float that fits (`isGreatest_maxScale`).

For `lo ≥ 0` or `hi ≤ 0` a candidate's scales form an interval away from `0` that may hold no
float, and the result may fall short of the best by up to a level (`lt_succF64_maxScale_mul`).

The floats are `IsF64`, with no exponent cap. `level_grid` passes `hi - lo > 2^-51` and
`room < 2^62`, so the positive ends and the fallback stay below `2^114`, where the model rounds
as float64 does. A negative end, which the code may round to `-inf`, is never kept in either
model, and neither is the float after it toward `0`.
-/

@[expose] public section

namespace DiffGPT

/-- `_fits`: every step in `[lo, hi]` gets a token in range on the scale `s`. At the order `0`
(`order0 = true`) the levels take `⌊s·hi⌋ - ⌊s·lo⌋ + 1` of the `room = V` tokens, the hypothesis
of `zeroPoint0_mem`; above it the steps take `⌈s·hi - h⌉ - ⌊s·lo - h⌋` of the room
`V - ‖c‖₁ + 1 - t` (`fitsRoom_false_iff_fits`). -/
def fitsRoom : Bool → ℤ → ℝ → ℝ → ℝ → ℝ → Prop
  | true, room, s, _, lo, hi => ⌊s * hi⌋ - ⌊s * lo⌋ < room
  | false, room, s, h, lo, hi => ⌈s * hi - h⌉ - ⌊s * lo - h⌋ ≤ room

lemma fitsRoom_true {room : ℤ} {s h lo hi : ℝ} :
    fitsRoom true room s h lo hi ↔ ⌊s * hi⌋ - ⌊s * lo⌋ < room := Iff.rfl

lemma fitsRoom_false {room : ℤ} {s h lo hi : ℝ} :
    fitsRoom false room s h lo hi ↔ ⌈s * hi - h⌉ - ⌊s * lo - h⌋ ≤ room := Iff.rfl

/-- Above the order `0`, `_fits` is `Fits` with the room `V - ‖c‖₁ + 1 - t`. -/
lemma fitsRoom_false_iff_fits {V : ℤ} {l : List ℕ} {s h lo hi : ℝ} {t : ℤ} :
    fitsRoom false (V - norm1 l + 1 - t) s h lo hi ↔ Fits V l s h lo hi t := Iff.rfl

/-- `math.nextafter(x, 0.0)`: the next float toward `0`, and `0` at `0`. -/
noncomputable def towardZero (x : ℝ) : ℝ :=
  if 0 < x then predF64 x else if x < 0 then succF64 x else 0

/-- `min(ends)` of `_max_scale` for the candidate `p`: `p/lo` where `lo < 0`, which bounds the
scales with `p ≤ s·lo`, and `(p + room)/hi` where `hi > 0`, which bounds those with
`s·hi ≤ p + room`. For `lo < hi` at least one is there. -/
noncomputable def scaleEnd (room : ℤ) (lo hi p : ℝ) : ℝ :=
  if lo < 0 then (if 0 < hi then min (p / lo) ((p + room) / hi) else p / lo)
  else (p + room) / hi

/-- `_max_scale`'s pivot `room·lo/(hi - lo)`: the `p` at which the ends `p/lo` and
`(p + room)/hi` meet, at `s = room/(hi - lo)`. -/
noncomputable def scalePivot (room : ℤ) (lo hi : ℝ) : ℝ := room * lo / (hi - lo)

/-- `_max_scale`'s fallback: `(room - 1)/(hi - lo)` rounded down to a float. -/
noncomputable def fallbackScale (fl : ℝ → ℝ) (room : ℤ) (lo hi : ℝ) : ℝ :=
  atOrBelow fl ((room - 1) / (hi - lo))

open Classical in
/-- The inner loop of `_max_scale`: `end`, then `math.nextafter(end, 0.0)`, the first that is
positive and fits, if either. -/
noncomputable def firstFit (order0 : Bool) (room : ℤ) (h lo hi e : ℝ) : Option ℝ :=
  if 0 < e ∧ fitsRoom order0 room e h lo hi then some e
  else if 0 < towardZero e ∧ fitsRoom order0 room (towardZero e) h lo hi then some (towardZero e)
  else none

/-- The scale `_max_scale` keeps for the candidate `p`, if any: its end rounded down to a float,
or the next float toward `0`. -/
noncomputable def candidateScale (fl : ℝ → ℝ) (order0 : Bool) (room : ℤ) (h lo hi p : ℝ) :
    Option ℝ :=
  firstFit order0 room h lo hi (atOrBelow fl (scaleEnd room lo hi p))

/-- `_max_scale`: the largest of the fallback and the scales kept for the candidates
`⌊pivot - h⌋ + h` and `⌈pivot - h⌉ + h`. -/
noncomputable def maxScale (fl : ℝ → ℝ) (order0 : Bool) (room : ℤ) (h lo hi : ℝ) : ℝ :=
  max (fallbackScale fl room lo hi)
    (max ((candidateScale fl order0 room h lo hi (⌊scalePivot room lo hi - h⌋ + h)).getD
        (fallbackScale fl room lo hi))
      ((candidateScale fl order0 room h lo hi (⌈scalePivot room lo hi - h⌉ + h)).getD
        (fallbackScale fl room lo hi)))

/-- The steps `s·x` of `[lo, hi]` stay within `[p, p + room]`, short of `p + room` at the order
`0`. -/
def Within (order0 : Bool) (room : ℤ) (s lo hi p : ℝ) : Prop :=
  p ≤ s * lo ∧ s * hi ≤ p + room ∧ (order0 = true → s * hi < p + room)

/-! ### Fitting scales -/

/-- A scale a level short of the room, `s·(hi - lo) ≤ room - 1`, fits at either order and for
any shift: `_max_scale`'s fallback (`fits_of_mul_le`). -/
lemma fitsRoom_of_mul_le {order0 : Bool} {room : ℤ} {s h lo hi : ℝ}
    (hs : s * (hi - lo) ≤ room - 1) : fitsRoom order0 room s h lo hi := by
  have e : s * (hi - lo) = s * hi - s * lo := by ring
  cases order0
  · rw [fitsRoom_false]
    have h1 := Int.ceil_lt_add_one (s * hi - h)
    have h2 := Int.sub_one_lt_floor (s * lo - h)
    have h3 : ((⌈s * hi - h⌉ - ⌊s * lo - h⌋ : ℤ) : ℝ) < ((room + 1 : ℤ) : ℝ) := by
      push_cast
      linarith
    have := Int.cast_lt.mp h3
    omega
  · rw [fitsRoom_true]
    have h1 := Int.floor_le (s * hi)
    have h2 := Int.sub_one_lt_floor (s * lo)
    have h3 : ((⌊s * hi⌋ - ⌊s * lo⌋ : ℤ) : ℝ) < ((room : ℤ) : ℝ) := by
      push_cast
      linarith
    exact_mod_cast h3

/-- A scale that fits keeps its steps within `[p, p + room]` for `p = ⌊s·lo - h⌋ + h`, the
largest `p ∈ ℤ + h` at or below `s·lo`. -/
lemma within_of_fitsRoom {order0 : Bool} {room : ℤ} {s h lo hi : ℝ}
    (h0 : order0 = true → h = 0) (hf : fitsRoom order0 room s h lo hi) :
    Within order0 room s lo hi (⌊s * lo - h⌋ + h) := by
  cases order0
  · rw [fitsRoom_false] at hf
    have h1 := Int.le_ceil (s * hi - h)
    have h2 := Int.floor_le (s * lo - h)
    have h3 : ((⌈s * hi - h⌉ : ℤ) : ℝ) ≤ ((⌊s * lo - h⌋ + room : ℤ) : ℝ) := by
      exact_mod_cast (by omega : ⌈s * hi - h⌉ ≤ ⌊s * lo - h⌋ + room)
    push_cast at h3
    exact ⟨by linarith, by linarith, fun h => absurd h Bool.false_ne_true⟩
  · obtain rfl := h0 rfl
    rw [fitsRoom_true] at hf
    have h1 := Int.lt_floor_add_one (s * hi)
    have h2 := Int.floor_le (s * lo)
    have h3 : ((⌊s * hi⌋ + 1 : ℤ) : ℝ) ≤ ((⌊s * lo⌋ + room : ℤ) : ℝ) := by
      exact_mod_cast (by omega : ⌊s * hi⌋ + 1 ≤ ⌊s * lo⌋ + room)
    push_cast at h3
    simp only [sub_zero, add_zero]
    exact ⟨h2, by linarith, fun _ => by linarith⟩

/-- A scale whose steps stay within `[p, p + room]`, `p ∈ ℤ + h`, fits. -/
lemma fitsRoom_of_within {order0 : Bool} {room : ℤ} {s h lo hi : ℝ}
    (h0 : order0 = true → h = 0) {n : ℤ} (hw : Within order0 room s lo hi (n + h)) :
    fitsRoom order0 room s h lo hi := by
  obtain ⟨h1, h2, h3⟩ := hw
  cases order0
  · rw [fitsRoom_false]
    have e1 : n ≤ ⌊s * lo - h⌋ := Int.le_floor.mpr (by linarith)
    have e2 : ⌈s * hi - h⌉ ≤ n + room := Int.ceil_le.mpr (by push_cast; linarith)
    omega
  · obtain rfl := h0 rfl
    rw [fitsRoom_true]
    have e1 : n ≤ ⌊s * lo⌋ := Int.le_floor.mpr (by linarith)
    have e2 : ⌊s * hi⌋ < n + room := Int.floor_lt.mpr (by push_cast; linarith [h3 rfl])
    omega

/-- For `lo < 0 < hi` the steps of `s` stay within `[p, p + room]` exactly when `s` is at most
the end `min(p/lo, (p + room)/hi)` (strictness at the order `0` aside). -/
lemma le_scaleEnd_iff {room : ℤ} {s lo hi p : ℝ} (hlo : lo < 0) (hhi : 0 < hi) :
    s ≤ scaleEnd room lo hi p ↔ p ≤ s * lo ∧ s * hi ≤ p + room := by
  rw [scaleEnd, ite_eq_left hlo, ite_eq_left hhi, le_min_iff, le_div_iff_of_neg hlo,
    le_div_iff₀ hhi]

/-- For `lo < 0 < hi` a scale within `[p, p + room]` for some `p ∈ ℤ + h` is within it for a
candidate: `⌊pivot - h⌋ + h` where `p` lies at or below the pivot, since the end `(p + room)/hi`
grows with `p` and stays below `p/lo` up to the pivot, and `⌈pivot - h⌉ + h` where `p` lies
above it, the other way round. -/
lemma within_candidate {order0 : Bool} {room : ℤ} {s h lo hi : ℝ} (hlo : lo < 0)
    (hhi : 0 < hi) {n : ℤ} (hw : Within order0 room s lo hi (n + h)) :
    Within order0 room s lo hi (⌊scalePivot room lo hi - h⌋ + h) ∨
      Within order0 room s lo hi (⌈scalePivot room lo hi - h⌉ + h) := by
  obtain ⟨h1, h2, h3⟩ := hw
  have hw0 : 0 < hi - lo := by linarith
  set P := scalePivot room lo hi
  have e1 : s * hi * lo = s * lo * hi := by ring
  rcases le_or_gt ((n : ℝ) + h) P with hnP | hnP
  · left
    set p := (⌊P - h⌋ : ℝ) + h
    have hnp : (n : ℝ) + h ≤ p := by
      have : n ≤ ⌊P - h⌋ := Int.le_floor.mpr (by linarith)
      have : (n : ℝ) ≤ ⌊P - h⌋ := by exact_mod_cast this
      linarith
    have hpP : p ≤ P := by linarith [Int.floor_le (P - h)]
    have a1 : p * (hi - lo) ≤ room * lo := (le_div_iff₀ hw0).mp hpP
    have hshi : s * hi ≤ p + room := by linarith
    have a2 : (p + room) * lo ≤ s * hi * lo := mul_le_mul_of_nonpos_right hshi hlo.le
    have a3 : p * hi ≤ s * lo * hi := by
      linarith [show p * (hi - lo) = p * hi - p * lo by ring,
        show (p + room) * lo = p * lo + room * lo by ring]
    exact ⟨le_of_mul_le_mul_right a3 hhi, hshi, fun ho => by linarith [h3 ho]⟩
  · right
    set p := (⌈P - h⌉ : ℝ) + h
    have hcn : ⌈P - h⌉ ≤ n := Int.ceil_le.mpr (by linarith)
    have hpn : p ≤ n + h := by
      have : (⌈P - h⌉ : ℝ) ≤ n := by exact_mod_cast hcn
      linarith
    have hPp : P ≤ p := by linarith [Int.le_ceil (P - h)]
    have b1 : room * lo ≤ p * (hi - lo) := (div_le_iff₀ hw0).mp hPp
    have hpl : p ≤ s * lo := hpn.trans h1
    have b2 : p * hi ≤ s * lo * hi := mul_le_mul_of_nonneg_right hpl hhi.le
    have hshi : s * hi ≤ p + room := by
      by_contra hc
      have b3 : s * hi * lo < (p + room) * lo := mul_lt_mul_of_neg_right (not_le.mp hc) hlo
      linarith [show p * (hi - lo) = p * hi - p * lo by ring,
        show (p + room) * lo = p * lo + room * lo by ring]
    refine ⟨hpl, hshi, fun ho => ?_⟩
    rcases hcn.lt_or_eq with hlt | heq
    · have hlt' : p < s * lo := by
        have : (⌈P - h⌉ : ℝ) < n := by exact_mod_cast hlt
        linarith
      have b4 : p * hi < s * lo * hi := mul_lt_mul_of_pos_right hlt' hhi
      by_contra hc
      have b3 : s * hi * lo ≤ (p + room) * lo := mul_le_mul_of_nonpos_right (not_lt.mp hc) hlo.le
      linarith [show p * (hi - lo) = p * hi - p * lo by ring,
        show (p + room) * lo = p * lo + room * lo by ring]
    · have : p = n + h := by rw [← heq]
      rw [this]
      exact h3 ho

/-! ### The search -/

lemma towardZero_of_pos {x : ℝ} (hx : 0 < x) : towardZero x = predF64 x := ite_eq_left hx

lemma isF64_towardZero {x : ℝ} : IsF64 (towardZero x) := by
  unfold towardZero
  split_ifs
  · exact isF64_predF64 x
  · exact isF64_succF64 x
  · exact IsF64.zero

/-- What `firstFit` keeps is positive and fits. -/
lemma firstFit_spec {order0 : Bool} {room : ℤ} {h lo hi e s : ℝ}
    (hk : firstFit order0 room h lo hi e = some s) : 0 < s ∧ fitsRoom order0 room s h lo hi := by
  unfold firstFit at hk
  split_ifs at hk with h1 h2
  · obtain rfl := Option.some.inj hk
    exact h1
  · obtain rfl := Option.some.inj hk
    exact h2

lemma isF64_getD_firstFit {order0 : Bool} {room : ℤ} {h lo hi e d : ℝ} (he : IsF64 e)
    (hd : IsF64 d) : IsF64 ((firstFit order0 room h lo hi e).getD d) := by
  unfold firstFit
  split_ifs
  · exact he
  · exact isF64_towardZero
  · exact hd

/-- `firstFit` keeps at least `y` when `end ≥ y` passes, or else the float toward `0` passes
and is at least `y`. -/
lemma le_getD_firstFit {order0 : Bool} {room : ℤ} {h lo hi e d y : ℝ} (hye : y ≤ e)
    (H : (0 < e ∧ fitsRoom order0 room e h lo hi) ∨
      (0 < towardZero e ∧ fitsRoom order0 room (towardZero e) h lo hi) ∧ y ≤ towardZero e) :
    y ≤ (firstFit order0 room h lo hi e).getD d := by
  unfold firstFit
  split_ifs with h1 h2
  · exact hye
  · exact (H.resolve_left h1).2
  · exact absurd (H.resolve_left h1).1 h2

variable {fl : ℝ → ℝ}

lemma fallbackScale_eq (hfl : IsRoundNearest fl) {room : ℤ} {lo hi : ℝ} :
    fallbackScale fl room lo hi = floorF64 ((room - 1) / (hi - lo)) :=
  hfl.atOrBelow_eq _

/-- `_max_scale`'s result is a float. -/
theorem isF64_maxScale (hfl : IsRoundNearest fl) {order0 : Bool} {room : ℤ} {h lo hi : ℝ} :
    IsF64 (maxScale fl order0 room h lo hi) := by
  have hfb : IsF64 (fallbackScale fl room lo hi) := by
    rw [fallbackScale_eq hfl]
    exact isF64_floorF64 _
  have hk : ∀ p, IsF64 ((candidateScale fl order0 room h lo hi p).getD
      (fallbackScale fl room lo hi)) := fun p =>
    isF64_getD_firstFit (by rw [hfl.atOrBelow_eq]; exact isF64_floorF64 _) hfb
  exact max_rec' _ hfb (max_rec' _ (hk _) (hk _))

/-- **`_max_scale`'s result fits**: the fallback does, and so does every scale kept. -/
theorem fitsRoom_maxScale (hfl : IsRoundNearest fl) {order0 : Bool} {room : ℤ} {h lo hi : ℝ}
    (hlh : lo < hi) : fitsRoom order0 room (maxScale fl order0 room h lo hi) h lo hi := by
  have hw : 0 < hi - lo := sub_pos.mpr hlh
  have hfb : fitsRoom order0 room (fallbackScale fl room lo hi) h lo hi := by
    apply fitsRoom_of_mul_le
    rw [fallbackScale_eq hfl]
    calc floorF64 (((room : ℝ) - 1) / (hi - lo)) * (hi - lo)
        ≤ ((room : ℝ) - 1) / (hi - lo) * (hi - lo) :=
          mul_le_mul_of_nonneg_right (floorF64_le _) hw.le
      _ = room - 1 := div_mul_cancel₀ _ hw.ne'
  have hk : ∀ p, fitsRoom order0 room ((candidateScale fl order0 room h lo hi p).getD
      (fallbackScale fl room lo hi)) h lo hi := by
    intro p
    cases hkp : candidateScale fl order0 room h lo hi p with
    | none => exact hfb
    | some s => exact (firstFit_spec hkp).2
  exact max_rec' (fitsRoom order0 room · h lo hi) hfb
    (max_rec' (fitsRoom order0 room · h lo hi) (hk _) (hk _))

/-- **`_max_scale` is at most a level short**: the float after its result, times `hi - lo`,
exceeds `room - 1`, since the fallback is the float at or below `(room - 1)/(hi - lo)`. -/
theorem lt_succF64_maxScale_mul (hfl : IsRoundNearest fl) {order0 : Bool} {room : ℤ}
    {h lo hi : ℝ} (hlh : lo < hi) :
    (room : ℝ) - 1 < succF64 (maxScale fl order0 room h lo hi) * (hi - lo) := by
  have hw : 0 < hi - lo := sub_pos.mpr hlh
  have h1 := lt_succF64_floorF64 (((room : ℝ) - 1) / (hi - lo))
  have h2 : succF64 (floorF64 (((room : ℝ) - 1) / (hi - lo))) ≤
      succF64 (maxScale fl order0 room h lo hi) := by
    apply succF64_mono
    rw [← fallbackScale_eq hfl]
    exact le_max_left _ _
  rw [← div_lt_iff₀ hw]
  linarith

/-- `_max_scale` keeps at least `y` for a candidate `p ∈ ℤ + h` that holds the steps of `y`: the
end rounded down, `e ≥ y`, fits, unless at the order `0` it puts `e·hi` on `p + room`; then
`y < e`, and the float below `e`, still at least `y`, fits. -/
lemma le_getD_candidateScale (hfl : IsRoundNearest fl) {order0 : Bool} {room : ℤ}
    {h lo hi : ℝ} (hlo : lo < 0) (hhi : 0 < hi) (h0 : order0 = true → h = 0) {y : ℝ}
    (hy : IsF64 y) (hy0 : 0 < y) {q : ℤ} (hq : Within order0 room y lo hi (q + h)) (d : ℝ) :
    y ≤ (candidateScale fl order0 room h lo hi (q + h)).getD d := by
  unfold candidateScale
  rw [hfl.atOrBelow_eq]
  set E := scaleEnd room lo hi (q + h)
  set e := floorF64 E
  have hye : y ≤ e := le_floorF64 hy ((le_scaleEnd_iff hlo hhi).mpr ⟨hq.1, hq.2.1⟩)
  have he0 : 0 < e := hy0.trans_le hye
  obtain ⟨he1, he2⟩ := (le_scaleEnd_iff hlo hhi).mp (floorF64_le E)
  apply le_getD_firstFit hye
  by_cases hlt : order0 = true → e * hi < q + h + room
  · exact Or.inl ⟨he0, fitsRoom_of_within (n := q) h0 ⟨he1, he2, hlt⟩⟩
  · right
    obtain ⟨ho, hge⟩ := not_imp.mp hlt
    have hyl : y < e := by
      have h3 := hq.2.2 ho
      by_contra hc
      have : e * hi ≤ y * hi := mul_le_mul_of_nonneg_right (not_lt.mp hc) hhi.le
      linarith [not_lt.mp hge]
    rw [towardZero_of_pos he0]
    have hp : y ≤ predF64 e := le_predF64 hy hyl
    have hpe : predF64 e < e := predF64_lt e
    refine ⟨⟨hy0.trans_le hp, fitsRoom_of_within (n := q) h0 ⟨?_, ?_, fun _ => ?_⟩⟩, hp⟩
    · have : e * lo ≤ predF64 e * lo := mul_le_mul_of_nonpos_right hpe.le hlo.le
      linarith
    · have : predF64 e * hi ≤ e * hi := mul_le_mul_of_nonneg_right hpe.le hhi.le
      linarith
    · have : predF64 e * hi < e * hi := mul_lt_mul_of_pos_right hpe hhi
      linarith

/-- **`_max_scale` is exact for `lo < 0 < hi`**: every positive float that fits is at most its
result. At the order `0` the shift is `0`, as in the code. -/
theorem le_maxScale (hfl : IsRoundNearest fl) {order0 : Bool} {room : ℤ} {h lo hi : ℝ}
    (hlo : lo < 0) (hhi : 0 < hi) (h0 : order0 = true → h = 0) {y : ℝ} (hy : IsF64 y)
    (hy0 : 0 < y) (hf : fitsRoom order0 room y h lo hi) :
    y ≤ maxScale fl order0 room h lo hi := by
  unfold maxScale
  rcases within_candidate hlo hhi (within_of_fitsRoom h0 hf) with hw | hw
  · exact le_max_of_le_right (le_max_of_le_left
      (le_getD_candidateScale hfl hlo hhi h0 hy hy0 hw _))
  · exact le_max_of_le_right (le_max_of_le_right
      (le_getD_candidateScale hfl hlo hhi h0 hy hy0 hw _))

/-- `_max_scale`'s result is positive where the fallback is: for `room ≥ 2` and
`hi - lo ≤ 2^1025`, `(room - 1)/(hi - lo)` is at least `2^-1074`, the least positive float. -/
theorem maxScale_pos (hfl : IsRoundNearest fl) {order0 : Bool} {room : ℤ} {h lo hi : ℝ}
    (hlh : lo < hi) (hroom : 2 ≤ room) (hw : hi - lo ≤ 2 ^ 1025) :
    0 < maxScale fl order0 room h lo hi := by
  have hw0 : 0 < hi - lo := sub_pos.mpr hlh
  have hr : (2 : ℝ) ≤ room := by exact_mod_cast hroom
  have hq : (2 : ℝ) ^ (-1074 : ℤ) ≤ ((room : ℝ) - 1) / (hi - lo) := by
    rw [le_div_iff₀ hw0]
    calc (2 : ℝ) ^ (-1074 : ℤ) * (hi - lo) ≤ 2 ^ (-1074 : ℤ) * 2 ^ 1025 :=
          mul_le_mul_of_nonneg_left hw (by positivity)
      _ ≤ 1 := by
          rw [← zpow_natCast, ← zpow_add₀ two_ne_zero]
          exact zpow_le_one_of_nonpos₀ one_le_two (by norm_num)
      _ ≤ room - 1 := by linarith
  have hf : IsF64 ((2 : ℝ) ^ (-1074 : ℤ)) := ⟨1, -1074, by norm_num, le_rfl, by simp⟩
  calc (0 : ℝ) < 2 ^ (-1074 : ℤ) := zpow_pos two_pos _
    _ ≤ floorF64 (((room : ℝ) - 1) / (hi - lo)) := le_floorF64 hf hq
    _ = fallbackScale fl room lo hi := (fallbackScale_eq hfl).symm
    _ ≤ maxScale fl order0 room h lo hi := le_max_left _ _

/-- **`_max_scale` returns the largest positive float on which `[lo, hi]` fits**, for
`lo < 0 < hi`, `room ≥ 2` and `hi - lo ≤ 2^1025`. -/
theorem isGreatest_maxScale (hfl : IsRoundNearest fl) {order0 : Bool} {room : ℤ} {h lo hi : ℝ}
    (hlo : lo < 0) (hhi : 0 < hi) (h0 : order0 = true → h = 0) (hroom : 2 ≤ room)
    (hw : hi - lo ≤ 2 ^ 1025) :
    IsGreatest {s | IsF64 s ∧ 0 < s ∧ fitsRoom order0 room s h lo hi}
      (maxScale fl order0 room h lo hi) :=
  ⟨⟨isF64_maxScale hfl, maxScale_pos hfl (hlo.trans hhi) hroom hw,
      fitsRoom_maxScale hfl (hlo.trans hhi)⟩,
    fun _ ⟨hy, hy0, hf⟩ => le_maxScale hfl hlo hhi h0 hy hy0 hf⟩

end DiffGPT
