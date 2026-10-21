module

public import Mathlib
public import DiffGPT.LevelGrid
public import DiffGPT.Float64
public import DiffGPT.TwoSum

/-!
# `_difference_range` returns the tightest float range of the steps

`get_domain_of_definition` takes the range `lo ≤ c(B) x ≤ hi` of a column `x` over its observed
rows from `_difference_range`, which differences the column lag by lag in float64, keeps a slack
that bounds how far each rounded difference is off, and sums in rationals only the rows whose
bounds reach the greatest value known to be attained. In the binary64 model of
`DiffGPT.Float64`:

* `lagStep`, `diffSlack`: the loop, first lag first, from `(column, 0)`. The lag `S` takes
  `diff = a + b` for `a = d (n + S)`, `b = -d n`, and `slack[lag:] + slack[:-lag] + |error|`,
  `error` Knuth's two-sum (`twoSumErr`). Index `n` is the row `n + p`, as in `diffs l x n`.
* `SlackBound`: after `k` lags `diff` is within `slack / (1 - 2^-53)^(2k)` of the exact
  differences (`SlackBound.step`, `slackBound_diffSlack`): two-sum gives the rounding error of
  each difference exactly, and each of the two rounded sums of the slack keeps `1 - 2^-53` of
  the exact sum (`IsRoundNearest.one_sub_mul_le_fl_add`). So `slack == 0` marks an exact
  difference (`SlackBound.eq_of_slack_eq_zero`), and for at most `2^51` lags, by Bernoulli,
  `|c(B) x - diff| ≤ 2·slack` (`SlackBound.abs_sub_le`).
* `lowerBound`, `upperBound`: `lower` and `upper`, `np.nextafter(size ∓ 2 * slack, ∓inf)` for
  `size = sign * diff`, bound `sign·c(B) x` (`lowerBound_le`, `le_upperBound`).
* `thresholdMax`: `greatest`'s filter, `none` for a NaN bound: `best` (`bestSize`), the
  greatest exact `size`; `threshold`, the greatest of `best` and the other rows' `lower`; then
  the greatest of `best` and the rational sums of the other rows whose `upper` is not below the
  threshold. `thresholdMax_eq`: given `size = value` on the exact rows and
  `lower ≤ value ≤ upper`, that is the greatest `value` over all rows.
* `greatest`, `differenceRange`: `greatest(sign)` and `_difference_range` on the rows `R`, the
  rational sums `Σ_j weights[j]·column[t + j]` with `weights = _coefficients(lags)[::-1]`, which
  is `coeff l` (`diffs_eq_sum`). `differenceRange_eq`: for floats `x`, at most `2^51` lags and
  nonempty `R`, `_difference_range` returns `floorF64` of the least `c(B) x` and `ceilF64` of
  the greatest. Every row lies in that range (`differenceRange_mem`), no float range inside it
  holds them all (`le_differenceRange_fst`, `differenceRange_snd_le`), and
  `get_domain_of_definition(symmetric=True)`'s `max(-lo, hi)` is the least float at or above
  every `|c(B) x|` (`abs_diffs_le_domain`, `domain_le`).

The model has no overflow. In the code an overflow makes the slack NaN or a bound infinite,
which the filter reads as no bound: `np.where` drops a NaN `lower` from the threshold, and a NaN
or infinite `upper` never compares below it, so by `thresholdMax_eq`, with `none` for those
bounds, such rows are summed exactly. `sign * diff` and `0.0 - f` negate exactly.
-/

@[expose] public section

open fwdDiff

namespace DiffGPT

/-- A rounded sum of two floats keeps at least `1 - 2^-53` of a nonnegative sum. -/
lemma IsRoundNearest.one_sub_mul_le_fl_add {fl : ℝ → ℝ} (hfl : IsRoundNearest fl) {a b : ℝ}
    (ha : IsF64 a) (hb : IsF64 b) (h : 0 ≤ a + b) :
    (1 - 2 ^ (-53 : ℤ)) * (a + b) ≤ fl (a + b) := by
  have h' := hfl.abs_add_sub_le ha hb
  rw [abs_of_nonneg h] at h'
  linarith [(abs_le.mp h').1]

/-! ### The loop over the lags -/

/-- One lag `S` of `_difference_range`'s loop on the differences and their slack, the pair
`de`: `diff = a + b` for `a = de.1 (n + S)`, `b = -de.1 n`, and
`slack[lag:] + slack[:-lag] + np.abs(error)`, `error` the two-sum error of `a + b`. -/
noncomputable def lagStep (fl : ℝ → ℝ) (S : ℕ) (de : (ℕ → ℝ) × (ℕ → ℝ)) :
    (ℕ → ℝ) × (ℕ → ℝ) :=
  (fun n => fl (de.1 (n + S) + -de.1 n),
    fun n => fl (fl (de.2 (n + S) + de.2 n) + |twoSumErr fl (de.1 (n + S)) (-de.1 n)|))

/-- `(diff, slack)` after `_difference_range`'s loop over the lags `l`, first lag first, from
`(column, 0)`. -/
noncomputable def diffSlack (fl : ℝ → ℝ) (l : List ℕ) (x : ℕ → ℝ) : (ℕ → ℝ) × (ℕ → ℝ) :=
  l.foldl (fun de S => lagStep fl S de) (x, 0)

/-- The loop's invariant after `k` lags, `f` the exact differences: the differences `de.1` and
the slack `de.2` are floats, the slack is nonnegative, and
`(1 - 2^-53)^(2k)·|f - de.1| ≤ de.2`. -/
structure SlackBound (f : ℕ → ℝ) (de : (ℕ → ℝ) × (ℕ → ℝ)) (k : ℕ) : Prop where
  isF64_diff : ∀ n, IsF64 (de.1 n)
  isF64_slack : ∀ n, IsF64 (de.2 n)
  slack_nonneg : ∀ n, 0 ≤ de.2 n
  le_slack : ∀ n, (1 - 2 ^ (-53 : ℤ)) ^ (2 * k) * |f n - de.1 n| ≤ de.2 n

/-- The arithmetic of one lag, `r = 1 - 2^-53` and `P = r^(2k)`. -/
private lemma lag_arith {r P X a b c e₁ e₂ A E : ℝ} (hr0 : 0 ≤ r) (hr1 : r ≤ 1) (hP0 : 0 ≤ P)
    (hP1 : P ≤ 1) (hc : 0 ≤ c) (hX : X ≤ a + b + c) (h₁ : P * a ≤ e₁) (h₂ : P * b ≤ e₂)
    (hA : r * (e₁ + e₂) ≤ A) (hE : r * (A + c) ≤ E) : r ^ 2 * (P * X) ≤ E := by
  have h3 : P * X ≤ e₁ + e₂ + c := by
    linarith [mul_le_mul_of_nonneg_left hX hP0, mul_le_of_le_one_left hc hP1]
  have h4 := mul_le_mul_of_nonneg_left h3 (sq_nonneg r)
  have h5 := mul_le_mul_of_nonneg_left hA hr0
  have h6 := mul_le_mul_of_nonneg_right (mul_le_of_le_one_left hr0 hr1) hc
  linarith

/-- One lag keeps the invariant, for `k + 1`: two-sum gives the rounding error of the difference
exactly, and the two rounded sums of the slack keep `(1 - 2^-53)²` of the exact sum. -/
lemma SlackBound.step {fl : ℝ → ℝ} (hfl : IsRoundNearest fl) {f : ℕ → ℝ}
    {de : (ℕ → ℝ) × (ℕ → ℝ)} {k : ℕ} (h : SlackBound f de k) (S : ℕ) :
    SlackBound (Δ_[S] f) (lagStep fl S de) (k + 1) where
  isF64_diff _ := hfl.isF64 _
  isF64_slack _ := hfl.isF64 _
  slack_nonneg n := by
    have h0 {a : ℝ} (ha : 0 ≤ a) : 0 ≤ fl a := (hfl.eq_self IsF64.zero).ge.trans (hfl.mono ha)
    exact h0 (add_nonneg (h0 (add_nonneg (h.slack_nonneg _) (h.slack_nonneg _))) (abs_nonneg _))
  le_slack n := by
    dsimp only [DiffGPT.lagStep]
    have hr0 : (0 : ℝ) ≤ 1 - 2 ^ (-53 : ℤ) := by norm_num
    have hr1 : (1 : ℝ) - 2 ^ (-53 : ℤ) ≤ 1 := by norm_num
    have hτ := hfl.twoSumErr_eq (h.isF64_diff (n + S)) (h.isF64_diff n).neg
    have hs := add_nonneg (h.slack_nonneg (n + S)) (h.slack_nonneg n)
    have hA := hfl.one_sub_mul_le_fl_add (h.isF64_slack (n + S)) (h.isF64_slack n) hs
    have hτf : IsF64 (twoSumErr fl (de.1 (n + S)) (-de.1 n)) := hfl.isF64 _
    have hE := hfl.one_sub_mul_le_fl_add (hfl.isF64 _) hτf.abs
      (add_nonneg ((hfl.eq_self IsF64.zero).ge.trans (hfl.mono hs)) (abs_nonneg _))
    have hX : |Δ_[S] f n - fl (de.1 (n + S) + -de.1 n)| ≤ |f (n + S) - de.1 (n + S)| +
        |f n - de.1 n| + |twoSumErr fl (de.1 (n + S)) (-de.1 n)| := by
      have e : Δ_[S] f n - fl (de.1 (n + S) + -de.1 n) = f (n + S) - de.1 (n + S) -
          (f n - de.1 n) + twoSumErr fl (de.1 (n + S)) (-de.1 n) := by
        rw [hτ, fwdDiff]; ring
      rw [e]
      exact (abs_add_le _ _).trans (add_le_add_left (abs_sub _ _) _)
    calc ((1 : ℝ) - 2 ^ (-53 : ℤ)) ^ (2 * (k + 1)) * |Δ_[S] f n - fl (de.1 (n + S) + -de.1 n)|
        = (1 - 2 ^ (-53 : ℤ)) ^ 2 *
          ((1 - 2 ^ (-53 : ℤ)) ^ (2 * k) * |Δ_[S] f n - fl (de.1 (n + S) + -de.1 n)|) := by
          ring
      _ ≤ _ := lag_arith hr0 hr1 (pow_nonneg hr0 _) (pow_le_one₀ hr0 hr1) (abs_nonneg _) hX
          (h.le_slack _) (h.le_slack _) hA hE

/-- The loop keeps the invariant, lag by lag. -/
lemma SlackBound.foldl {fl : ℝ → ℝ} (hfl : IsRoundNearest fl) (l : List ℕ) {f : ℕ → ℝ}
    {de : (ℕ → ℝ) × (ℕ → ℝ)} {k : ℕ} (h : SlackBound f de k) :
    SlackBound (diffs l f) (l.foldl (fun de S => lagStep fl S de) de) (k + l.length) := by
  induction l generalizing f de k with
  | nil => exact h
  | cons S l ih =>
    rw [List.foldl_cons, diffs_cons, List.length_cons,
      show k + (l.length + 1) = k + 1 + l.length by omega]
    exact ih (h.step hfl S)

/-- The invariant holds after the lags `l`, with `k = l.length`, from a column of floats. -/
lemma slackBound_diffSlack {fl : ℝ → ℝ} (hfl : IsRoundNearest fl) {x : ℕ → ℝ}
    (hx : ∀ n, IsF64 (x n)) (l : List ℕ) :
    SlackBound (diffs l x) (diffSlack fl l x) l.length := by
  have h0 : SlackBound x (x, 0) 0 :=
    ⟨hx, fun _ => IsF64.zero, fun _ => le_rfl, fun _ => by simp⟩
  have := h0.foldl hfl l
  rwa [zero_add] at this

/-- Where the slack is `0` the difference is exact. -/
lemma SlackBound.eq_of_slack_eq_zero {f : ℕ → ℝ} {de : (ℕ → ℝ) × (ℕ → ℝ)} {k n : ℕ}
    (h : SlackBound f de k) (hn : de.2 n = 0) : de.1 n = f n := by
  have hP : (0 : ℝ) < (1 - 2 ^ (-53 : ℤ)) ^ (2 * k) := pow_pos (by norm_num) _
  have h1 := h.le_slack n
  rw [hn] at h1
  have h0 : |f n - de.1 n| ≤ 0 := by
    by_contra! hc
    linarith [mul_pos hP hc]
  linarith [abs_nonpos_iff.mp h0]

/-- `(1 - 2^-53)^(2k) ≥ 1/2` for `k ≤ 2^51`, by Bernoulli. -/
lemma half_le_one_sub_pow {k : ℕ} (hk : k ≤ 2 ^ 51) :
    (1 / 2 : ℝ) ≤ (1 - 2 ^ (-53 : ℤ)) ^ (2 * k) := by
  have hb := one_add_mul_le_pow (a := -(2 : ℝ) ^ (-53 : ℤ)) (by norm_num) (2 * k)
  rw [← sub_eq_add_neg] at hb
  have hk' : ((2 * k : ℕ) : ℝ) ≤ 2 ^ 52 := by
    have : (k : ℝ) ≤ 2 ^ 51 := by exact_mod_cast hk
    push_cast
    linarith
  have hku : ((2 * k : ℕ) : ℝ) * 2 ^ (-53 : ℤ) ≤ 1 / 2 :=
    calc ((2 * k : ℕ) : ℝ) * 2 ^ (-53 : ℤ) ≤ 2 ^ 52 * 2 ^ (-53 : ℤ) :=
          mul_le_mul_of_nonneg_right hk' (by positivity)
      _ = 1 / 2 := by norm_num
  rw [mul_neg] at hb
  generalize (1 - (2 : ℝ) ^ (-53 : ℤ)) ^ (2 * k) = P at hb ⊢
  generalize ((2 * k : ℕ) : ℝ) * 2 ^ (-53 : ℤ) = m at hb hku
  linarith

/-- For at most `2^51` lags the difference is within twice the slack of the exact one. -/
lemma SlackBound.abs_sub_le {f : ℕ → ℝ} {de : (ℕ → ℝ) × (ℕ → ℝ)} {k : ℕ}
    (h : SlackBound f de k) (hk : k ≤ 2 ^ 51) (n : ℕ) : |f n - de.1 n| ≤ 2 * de.2 n := by
  have h1 := mul_le_mul_of_nonneg_right (half_le_one_sub_pow hk) (abs_nonneg (f n - de.1 n))
  have h2 := h.le_slack n
  generalize (1 - (2 : ℝ) ^ (-53 : ℤ)) ^ (2 * k) = P at h1 h2
  linarith

/-! ### The bounds of a row -/

/-- `lower = np.nextafter(size - 2 * slack, -np.inf)` for `size = σ·D`, `slack = E`. -/
noncomputable def lowerBound (fl : ℝ → ℝ) (σ D E : ℝ) : ℝ := predF64 (fl (σ * D - fl (2 * E)))

/-- `upper = np.nextafter(size + 2 * slack, np.inf)` for `size = σ·D`, `slack = E`. -/
noncomputable def upperBound (fl : ℝ → ℝ) (σ D E : ℝ) : ℝ := succF64 (fl (σ * D + fl (2 * E)))

/-- `lower ≤ σ·T` for `|σ| = 1`, a float slack `E` and `|T - D| ≤ 2·E`: `2 * slack` is exact,
`σ·D - 2·E ≤ σ·T`, and `np.nextafter` covers the rounding of the bound. -/
lemma lowerBound_le {fl : ℝ → ℝ} (hfl : IsRoundNearest fl) {σ T D E : ℝ} (hσ : |σ| = 1)
    (hE : IsF64 E) (h : |T - D| ≤ 2 * E) : lowerBound fl σ D E ≤ σ * T := by
  have h' : |σ * T - σ * D| ≤ 2 * E := by rwa [← mul_sub, abs_mul, hσ, one_mul]
  rw [lowerBound, hfl.eq_self hE.two_mul]
  linarith [hfl.predF64_le (σ * D - 2 * E), (abs_le.mp h').1]

/-- `σ·T ≤ upper` for `|σ| = 1`, a float slack `E` and `|T - D| ≤ 2·E`. -/
lemma le_upperBound {fl : ℝ → ℝ} (hfl : IsRoundNearest fl) {σ T D E : ℝ} (hσ : |σ| = 1)
    (hE : IsF64 E) (h : |T - D| ≤ 2 * E) : σ * T ≤ upperBound fl σ D E := by
  have h' : |σ * T - σ * D| ≤ 2 * E := by rwa [← mul_sub, abs_mul, hσ, one_mul]
  rw [upperBound, hfl.eq_self hE.two_mul]
  linarith [hfl.le_succF64 (σ * D + 2 * E), (abs_le.mp h').2]

/-! ### The threshold filter -/

section Filter

variable {ι : Type*}

open Classical in
/-- `best`: the greatest `size` over the exact rows of `R`, `⊥` (`-inf`) for none. -/
noncomputable def bestSize (R : Finset ι) (exact : ι → Prop) (size : ι → ℝ) : WithBot ℝ :=
  (R.filter exact).sup fun t => (size t : WithBot ℝ)

open Classical in
/-- `threshold`: the greatest of `best` and the `lower` bounds of the rows not exact, a NaN or
`-inf` bound, `none`, read as `⊥`. -/
noncomputable def threshold (R : Finset ι) (exact : ι → Prop) (size : ι → ℝ)
    (lower : ι → Option ℝ) : WithBot ℝ :=
  bestSize R exact size ⊔ (R.filter fun t => ¬exact t).sup fun t => (lower t).elim ⊥ (↑)

open Classical in
/-- `greatest`'s `max(values)`: the greatest of `best` and the `value` of every row that is not
exact and has no `upper < threshold`; a NaN or `inf` bound, `none`, is never below. `⊥` for
none. -/
noncomputable def thresholdMax (R : Finset ι) (exact : ι → Prop) (size : ι → ℝ)
    (lower upper : ι → Option ℝ) (value : ι → ℝ) : WithBot ℝ :=
  bestSize R exact size ⊔
    (R.filter fun t => ¬exact t ∧
      ∀ b ∈ upper t, ¬(b : WithBot ℝ) < threshold R exact size lower).sup
      fun t => (value t : WithBot ℝ)

/-- The filter keeps the maximum: given `size = value` on the exact rows and
`lower ≤ value ≤ upper` where the bounds are not `none`, `thresholdMax` is the greatest `value`
over `R`. The threshold is at most that maximum, so a row with `upper < threshold` does not
attain it. -/
theorem thresholdMax_eq {R : Finset ι} {exact : ι → Prop} {size : ι → ℝ}
    {lower upper : ι → Option ℝ} {value : ι → ℝ}
    (h_exact : ∀ t ∈ R, exact t → size t = value t)
    (h_lower : ∀ t ∈ R, ∀ a ∈ lower t, a ≤ value t)
    (h_upper : ∀ t ∈ R, ∀ b ∈ upper t, value t ≤ b) :
    thresholdMax R exact size lower upper value = R.sup fun t => (value t : WithBot ℝ) := by
  classical
  have hbest : bestSize R exact size ≤ R.sup fun t => (value t : WithBot ℝ) := by
    unfold bestSize
    refine Finset.sup_le fun t ht => ?_
    rw [Finset.mem_filter] at ht
    rw [h_exact t ht.1 ht.2]
    exact Finset.le_sup (f := fun t => (value t : WithBot ℝ)) ht.1
  have hthr : threshold R exact size lower ≤ R.sup fun t => (value t : WithBot ℝ) := by
    unfold threshold
    refine sup_le hbest (Finset.sup_le fun t ht => ?_)
    rw [Finset.mem_filter] at ht
    cases hl : lower t with
    | none => exact bot_le
    | some a =>
      exact (WithBot.coe_le_coe.mpr (h_lower t ht.1 a hl)).trans
        (Finset.le_sup (f := fun t => (value t : WithBot ℝ)) ht.1)
  unfold thresholdMax
  refine le_antisymm (sup_le hbest (Finset.sup_mono (Finset.filter_subset _ _))) ?_
  rcases R.eq_empty_or_nonempty with hR | hR
  · rw [hR, Finset.sup_empty]
    exact bot_le
  obtain ⟨t, ht, hMt⟩ := Finset.exists_mem_eq_sup R hR fun t => (value t : WithBot ℝ)
  refine hMt.le.trans ?_
  by_cases he : exact t
  · rw [← h_exact t ht he]
    exact le_sup_of_le_left (Finset.le_sup (f := fun t => (size t : WithBot ℝ))
      (Finset.mem_filter.mpr ⟨ht, he⟩))
  by_cases hk : ∀ b ∈ upper t, ¬(b : WithBot ℝ) < threshold R exact size lower
  · exact le_sup_of_le_right (Finset.le_sup (f := fun t => (value t : WithBot ℝ))
      (Finset.mem_filter.mpr ⟨ht, he, hk⟩))
  simp only [not_forall, not_not] at hk
  obtain ⟨b, hb, hlt⟩ := hk
  have := (WithBot.coe_le_coe.mpr (h_upper t ht b hb)).trans_lt (hlt.trans_le hthr)
  rw [hMt] at this
  exact absurd this (lt_irrefl _)

end Filter

/-! ### `_difference_range` -/

/-- `greatest(sign)` of `_difference_range` on the rows `R`, `σ = sign`: `exact = slack == 0`,
`size = σ·diff`, the bounds `lowerBound` and `upperBound`, and `σ·Σ_j weights[j]·column[t + j]`
summed in rationals, `weights = _coefficients(lags)[::-1]`, which is `coeff l`. -/
noncomputable def greatest (fl : ℝ → ℝ) (l : List ℕ) (x : ℕ → ℝ) (R : Finset ℕ) (σ : ℝ) :
    WithBot ℝ :=
  thresholdMax R (fun t => (diffSlack fl l x).2 t = 0) (fun t => σ * (diffSlack fl l x).1 t)
    (fun t => some (lowerBound fl σ ((diffSlack fl l x).1 t) ((diffSlack fl l x).2 t)))
    (fun t => some (upperBound fl σ ((diffSlack fl l x).1 t) ((diffSlack fl l x).2 t)))
    fun t => σ * ∑ j ∈ Finset.range (l.sum + 1), (coeff l j : ℝ) * x (t + j)

/-- `_difference_range` on the rows `R`:
`(_float_at_or_below(-greatest(-1)), _float_at_or_above(greatest(1)))`. An empty `max`, `⊥`,
reads as `0`; Python raises there, and nonempty rows never get one (`greatest_eq`). -/
noncomputable def differenceRange (fl : ℝ → ℝ) (l : List ℕ) (x : ℕ → ℝ) (R : Finset ℕ) :
    ℝ × ℝ :=
  (atOrBelow fl (-(greatest fl l x R (-1)).unbotD 0), atOrAbove fl ((greatest fl l x R 1).unbotD 0))

/-- `greatest(sign)` is the greatest `sign·c(B) x` over the rows: `diff` and `slack` bound it
(`SlackBound`), and `thresholdMax_eq` does the rest. -/
theorem greatest_eq {fl : ℝ → ℝ} (hfl : IsRoundNearest fl) {l : List ℕ} {x : ℕ → ℝ}
    (hx : ∀ n, IsF64 (x n)) (hl : l.length ≤ 2 ^ 51) (R : Finset ℕ) {σ : ℝ} (hσ : |σ| = 1) :
    greatest fl l x R σ = R.sup fun t => ((σ * diffs l x t : ℝ) : WithBot ℝ) := by
  have h := slackBound_diffSlack hfl hx l
  have hv : ∀ t, σ * ∑ j ∈ Finset.range (l.sum + 1), (coeff l j : ℝ) * x (t + j) =
      σ * diffs l x t := fun t => by rw [diffs_eq_sum]; simp only [zsmul_eq_mul]
  refine (thresholdMax_eq (fun t _ he => ?_) (fun t _ a ha => ?_) (fun t _ b hb => ?_)).trans ?_
  · rw [hv, h.eq_of_slack_eq_zero he]
  · rw [Option.mem_some_iff] at ha
    rw [← ha, hv]
    exact lowerBound_le hfl hσ (h.isF64_slack t) (h.abs_sub_le hl t)
  · rw [Option.mem_some_iff] at hb
    rw [← hb, hv]
    exact le_upperBound hfl hσ (h.isF64_slack t) (h.abs_sub_le hl t)
  · simp only [hv]

/-- `_difference_range` returns the greatest float at or below the least `c(B) x` over the rows
and the least float at or above the greatest. -/
theorem differenceRange_eq {fl : ℝ → ℝ} (hfl : IsRoundNearest fl) {l : List ℕ} {x : ℕ → ℝ}
    (hx : ∀ n, IsF64 (x n)) (hl : l.length ≤ 2 ^ 51) {R : Finset ℕ} (hR : R.Nonempty) :
    differenceRange fl l x R =
      (floorF64 (R.inf' hR (diffs l x)), ceilF64 (R.sup' hR (diffs l x))) := by
  have h1 : greatest fl l x R 1 = ((R.sup' hR (diffs l x) : ℝ) : WithBot ℝ) := by
    rw [greatest_eq hfl hx hl R (σ := 1) (by norm_num), Finset.coe_sup']
    simp only [one_mul]
    rfl
  have h2 : greatest fl l x R (-1) = ((-R.inf' hR (diffs l x) : ℝ) : WithBot ℝ) := by
    rw [greatest_eq hfl hx hl R (σ := -1) (by norm_num)]
    refine le_antisymm (Finset.sup_le fun t ht => WithBot.coe_le_coe.mpr ?_) ?_
    · linarith [Finset.inf'_le (diffs l x) ht]
    · obtain ⟨t, ht, hm⟩ := Finset.exists_mem_eq_inf' hR (diffs l x)
      have := Finset.le_sup (f := fun t => ((-1 * diffs l x t : ℝ) : WithBot ℝ)) ht
      rw [neg_one_mul] at this
      rwa [hm]
  rw [differenceRange, h1, h2, WithBot.unbotD_coe, WithBot.unbotD_coe, neg_neg,
    hfl.atOrBelow_eq, hfl.atOrAbove_eq]

/-- Every row's `c(B) x` lies in `_difference_range`'s `[lo, hi]`. -/
theorem differenceRange_mem {fl : ℝ → ℝ} (hfl : IsRoundNearest fl) {l : List ℕ} {x : ℕ → ℝ}
    (hx : ∀ n, IsF64 (x n)) (hl : l.length ≤ 2 ^ 51) {R : Finset ℕ} {t : ℕ} (ht : t ∈ R) :
    (differenceRange fl l x R).1 ≤ diffs l x t ∧ diffs l x t ≤ (differenceRange fl l x R).2 := by
  rw [differenceRange_eq hfl hx hl ⟨t, ht⟩]
  exact ⟨(floorF64_le _).trans (Finset.inf'_le _ ht), (Finset.le_sup' _ ht).trans (le_ceilF64 _)⟩

/-- `lo` is the greatest float at or below every row's `c(B) x`. -/
theorem le_differenceRange_fst {fl : ℝ → ℝ} (hfl : IsRoundNearest fl) {l : List ℕ}
    {x : ℕ → ℝ} (hx : ∀ n, IsF64 (x n)) (hl : l.length ≤ 2 ^ 51) {R : Finset ℕ}
    (hR : R.Nonempty) {y : ℝ} (hy : IsF64 y) (h : ∀ t ∈ R, y ≤ diffs l x t) :
    y ≤ (differenceRange fl l x R).1 := by
  rw [differenceRange_eq hfl hx hl hR]
  exact le_floorF64 hy (Finset.le_inf' hR _ h)

/-- `hi` is the least float at or above every row's `c(B) x`. -/
theorem differenceRange_snd_le {fl : ℝ → ℝ} (hfl : IsRoundNearest fl) {l : List ℕ}
    {x : ℕ → ℝ} (hx : ∀ n, IsF64 (x n)) (hl : l.length ≤ 2 ^ 51) {R : Finset ℕ}
    (hR : R.Nonempty) {y : ℝ} (hy : IsF64 y) (h : ∀ t ∈ R, diffs l x t ≤ y) :
    (differenceRange fl l x R).2 ≤ y := by
  rw [differenceRange_eq hfl hx hl hR]
  exact ceilF64_le hy (Finset.sup'_le hR _ h)

/-- `get_domain_of_definition(symmetric=True)`'s `D = max(-lo, hi)`,
`np.maximum(-result[:, 0], result[:, 1])`, bounds every row's `|c(B) x|`. -/
theorem abs_diffs_le_domain {fl : ℝ → ℝ} (hfl : IsRoundNearest fl) {l : List ℕ} {x : ℕ → ℝ}
    (hx : ∀ n, IsF64 (x n)) (hl : l.length ≤ 2 ^ 51) {R : Finset ℕ} {t : ℕ} (ht : t ∈ R) :
    |diffs l x t| ≤ max (-(differenceRange fl l x R).1) (differenceRange fl l x R).2 := by
  obtain ⟨h1, h2⟩ := differenceRange_mem hfl hx hl ht
  have h3 := le_max_left (-(differenceRange fl l x R).1) (differenceRange fl l x R).2
  exact abs_le.mpr ⟨by linarith, h2.trans (le_max_right _ _)⟩

/-- `D = max(-lo, hi)` is the least float at or above every row's `|c(B) x|`. -/
theorem domain_le {fl : ℝ → ℝ} (hfl : IsRoundNearest fl) {l : List ℕ} {x : ℕ → ℝ}
    (hx : ∀ n, IsF64 (x n)) (hl : l.length ≤ 2 ^ 51) {R : Finset ℕ} (hR : R.Nonempty) {y : ℝ}
    (hy : IsF64 y) (h : ∀ t ∈ R, |diffs l x t| ≤ y) :
    max (-(differenceRange fl l x R).1) (differenceRange fl l x R).2 ≤ y := by
  rw [differenceRange_eq hfl hx hl hR]
  have := le_floorF64 hy.neg (Finset.le_inf' hR _ fun t ht => (abs_le.mp (h t ht)).1)
  exact max_le (by linarith)
    (ceilF64_le hy (Finset.sup'_le hR _ fun t ht => (le_abs_self _).trans (h t ht)))

end DiffGPT
