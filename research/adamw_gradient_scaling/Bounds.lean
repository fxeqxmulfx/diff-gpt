import Transformer.Grokking.AdamW.ScalarRecurrence
import Transformer.Grokking.AdamW.LossDirection

/-!
Local extension of the existing native AdamW first-step scaling identity.
Source: the installed PyTorch AdamW equations pinned in plan.json and the
library's MomentRecurrence/ScalarRecurrence definitions. Gradients are supplied
sequences, not a hypothesis that model interventions preserve future gradients.
Finite optimizer epsilon is distinct from the model's RMS epsilon.
-/

namespace AdamWGradientScaling
open Transformer.Grokking.AdamW

/-- Target 1: scale every supplied input and the initial exponential buffer.
Source: the existing native momentAt recurrence; no reset is inserted. -/
theorem moment_scale (beta c initial : ℝ) (input : ℕ → ℝ) (n : ℕ) :
    momentAt beta (fun k => c * input k) (c * initial) n =
      c * momentAt beta input initial n := by
  induction n with
  | zero => simp only [momentAt]
  | succ n ih =>
    simp only [momentAt, ih]
    ring

private theorem first_scale (beta c : ℝ) (gradient : ℕ → ℝ) (n : ℕ) :
    firstMomentAt beta (fun k => c * gradient k) n =
      c * firstMomentAt beta gradient n := by
  simpa only [firstMomentAt, mul_zero] using moment_scale beta c 0 gradient n

private theorem second_scale (beta c : ℝ) (gradient : ℕ → ℝ) (n : ℕ) :
    secondMomentAt beta (fun k => c * gradient k) n =
      c ^ 2 * secondMomentAt beta gradient n := by
  simpa only [secondMomentAt, mul_zero, mul_pow] using
    moment_scale beta (c ^ 2) 0 (fun k => gradient k ^ 2) n

private theorem normalized_scale (eps c m v : ℝ) (he : 0 < eps) (hc : 0 < c) :
    (c * m) / (Real.sqrt (c ^ 2 * v) + eps) =
      m / (Real.sqrt v + eps / c) := by
  rw [Real.sqrt_mul (sq_nonneg c), Real.sqrt_sq hc.le]
  have hroot := Real.sqrt_nonneg v
  have hd1 : c * Real.sqrt v + eps ≠ 0 := by positivity
  have hd2 : Real.sqrt v + eps / c ≠ 0 := by positivity
  field_simp

/-- Target 2: a single positive factor applied to the whole zero-initialized
history changes effective epsilon, with both bias corrections and clock kept.
Source: native historyDirection; extends first_direction_positive_gradient_scaling.
The algebra includes the library's totalized clock-zero value. Physical AdamW
uses clock n>=1 and beta values in [0,1), as in the examples and CPU checks. -/
theorem history_scale (b1 b2 eps c : ℝ) (gradient : ℕ → ℝ) (n : ℕ)
    (he : 0 < eps) (hc : 0 < c) :
    historyDirection b1 b2 eps (fun k => c * gradient k) n =
      historyDirection b1 b2 (eps / c) gradient n := by
  unfold historyDirection
  rw [first_scale, second_scale]
  simpa only [mul_div_assoc] using normalized_scale eps c
    (firstMomentAt b1 gradient n / (1 - b1 ^ n))
    (secondMomentAt b2 gradient n / (1 - b2 ^ n)) he hc

private theorem buffer_scale (b1 b2 eps c m v g : ℝ) (clock : ℕ)
    (he : 0 < eps) (hc : 0 < c) :
    nextBufferDirection b1 b2 eps (c * m) (c ^ 2 * v) (c * g) clock =
      nextBufferDirection b1 b2 (eps / c) m v g clock := by
  have hm : b1 * (c * m) + (1 - b1) * (c * g) =
      c * (b1 * m + (1 - b1) * g) := by ring
  have hv : b2 * (c ^ 2 * v) + (1 - b2) * (c * g) ^ 2 =
      c ^ 2 * (b2 * v + (1 - b2) * g ^ 2) := by ring
  unfold nextBufferDirection
  rw [hm, hv]
  simpa only [mul_div_assoc] using normalized_scale eps c
    ((b1 * m + (1 - b1) * g) / (1 - b1 ^ (clock + 1)))
    ((b2 * v + (1 - b2) * g ^ 2) / (1 - b2 ^ (clock + 1))) he hc

/-- Actual native AdamW scalar recurrence from zero buffers, driven by a rate
schedule and supplied gradient sequence. Source: scalarNativeStep. A zero-rate
update still inserts its gradient and increments the completed-update clock. -/
noncomputable def nativeRun (b1 b2 eps decay : ℝ) (rate gradient : ℕ → ℝ)
    (initial : ℝ) : ℕ → ScalarState
  | 0 => seededScalarState initial
  | n + 1 => scalarNativeStep b1 b2 eps decay (rate n)
      (nativeRun b1 b2 eps decay rate gradient initial n) (gradient n)

private theorem step_scale (b1 b2 eps decay rate c g : ℝ) (s : ScalarState)
    (he : 0 < eps) (hc : 0 < c) :
    scalarNativeStep b1 b2 eps decay rate
        { s with moment := c * s.moment, variance := c ^ 2 * s.variance } (c * g) =
      let next := scalarNativeStep b1 b2 (eps / c) decay rate s g
      { next with moment := c * next.moment, variance := c ^ 2 * next.variance } := by
  have hd := buffer_scale b1 b2 eps c s.moment s.variance g s.clock he hc
  cases s
  simp only [scalarNativeStep, ScalarState.mk.injEq] at *
  exact ⟨by rw [hd], by ring, by ring, True.intro⟩

/-- Target 3: the whole native parameter path agrees at adjusted epsilon when
the supplied input histories differ by one fixed positive factor. Moment and
variance paths retain their c and c^2 factors, and clocks agree exactly.
Source: scalarNativeStep; includes arbitrary common rates and decoupled decay.
This is not a relation between models whose future gradients change differently. -/
theorem native_path_scale (b1 b2 eps decay c initial : ℝ) (rate gradient : ℕ → ℝ)
    (n : ℕ) (he : 0 < eps) (hc : 0 < c) :
    nativeRun b1 b2 eps decay rate (fun k => c * gradient k) initial n =
      let s := nativeRun b1 b2 (eps / c) decay rate gradient initial n
      { s with moment := c * s.moment, variance := c ^ 2 * s.variance } := by
  induction n with
  | zero => simp [nativeRun, seededScalarState]
  | succ n ih =>
    simp only [nativeRun]
    rw [ih]
    exact step_scale b1 b2 eps decay (rate n) c (gradient n) _ he hc

-- Nonconstant signed input, nonzero initial moment and actual benchmark betas.
example : momentAt (9 / 10) (fun k => (1 / 8 : ℝ) * (if k = 0 then 1 else -1))
      ((1 / 8) * 2) 4 =
    (1 / 8) * momentAt (9 / 10) (fun k => if k = 0 then 1 else -1) 2 4 :=
  moment_scale _ _ _ _ _

example : historyDirection (9 / 10) (49 / 50) (1 / 100000000)
      (fun k => (1 / 8 : ℝ) * (if k = 0 then 1 else -1)) 4 =
    historyDirection (9 / 10) (49 / 50) ((1 / 100000000) / (1 / 8))
      (fun k => if k = 0 then 1 else -1) 4 :=
  history_scale _ _ _ _ _ _ (by norm_num) (by norm_num)

-- Positive epsilon/scale, nonzero decay, signed inputs, and a zero-rate first step.
example : nativeRun (9 / 10) (49 / 50) (1 / 100000000) (1 / 10)
      (fun k => (1 / 500 : ℝ) * min 1 ((k : ℝ) / 50))
      (fun k => (1 / 8 : ℝ) * (if k = 0 then 1 else -1)) (1 / 50) 64 =
    let s := nativeRun (9 / 10) (49 / 50) ((1 / 100000000) / (1 / 8)) (1 / 10)
      (fun k => (1 / 500 : ℝ) * min 1 ((k : ℝ) / 50))
      (fun k => if k = 0 then 1 else -1) (1 / 50) 64
    { s with moment := (1 / 8) * s.moment, variance := (1 / 8 : ℝ) ^ 2 * s.variance } :=
  native_path_scale _ _ _ _ _ _ _ _ _ (by norm_num) (by norm_num)

#print axioms moment_scale
#print axioms history_scale
#print axioms native_path_scale
end AdamWGradientScaling
