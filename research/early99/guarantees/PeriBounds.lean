import Transformer.GPTMini.RMSNorm

/-!
Real-arithmetic bounds for the isolated early99 Peri-RMS candidates.

Source equations: research/early99/experiments/models.py and the frozen
parameter-free RMS implementation. The finite-epsilon normalization theorem
is Transformer.GPTMini.rmsNormEps_norm_le from transformer commit 8890a985.
There are two normalized residual branches per layer. The branch sequence
below is the actual sequence of raw outputs along a token's forward trace;
it may depend on all causal inputs. No independence assumption is imposed.
These bounds do not claim anything about optimization or floating-point error.
-/

namespace Early99

open Transformer

/-- A residual trace with arbitrary initial vector and normalized branch outputs. -/
noncomputable def residualTrace {d : ℕ} (eps scale : ℝ) (initial : EucSpace d)
    (raw : ℕ → EucSpace d) : ℕ → EucSpace d
  | 0 => initial
  | k + 1 => residualTrace eps scale initial raw k + scale • GPTMini.rmsNormEps eps (raw k)

/-- Triangle inequality and finite-epsilon RMS imply linear growth in branch count.
The initial vector is unrestricted, so this also covers branch-only Peri-RMS. -/
theorem residualTrace_bound {d : ℕ} (eps scale : ℝ) (heps : 0 < eps)
    (hd : 0 < d) (hscale : 0 ≤ scale) (initial : EucSpace d)
    (raw : ℕ → EucSpace d) (k : ℕ) :
    ‖residualTrace eps scale initial raw k‖ ≤
      ‖initial‖ + (k : ℝ) * scale * Real.sqrt (d : ℝ) := by
  induction k with
  | zero => simp [residualTrace]
  | succ k ih =>
    have hbranch : ‖scale • GPTMini.rmsNormEps eps (raw k)‖ ≤
        scale * Real.sqrt (d : ℝ) := by
      rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg hscale]
      exact mul_le_mul_of_nonneg_left (GPTMini.rmsNormEps_norm_le eps heps hd (raw k)) hscale
    calc
      ‖residualTrace eps scale initial raw (k + 1)‖ ≤
          ‖residualTrace eps scale initial raw k‖ + ‖scale • GPTMini.rmsNormEps eps (raw k)‖ :=
        norm_add_le _ _
      _ ≤ (‖initial‖ + (k : ℝ) * scale * Real.sqrt (d : ℝ)) +
          scale * Real.sqrt (d : ℝ) := add_le_add ih hbranch
      _ = ‖initial‖ + ((k + 1 : ℕ) : ℝ) * scale * Real.sqrt (d : ℝ) := by
        push_cast
        ring

/-- Embedding RMS removes the dependence of the bound on embedding weight size. -/
theorem embedding_trace_bound {d : ℕ} (eps scale : ℝ) (heps : 0 < eps)
    (hd : 0 < d) (hscale : 0 ≤ scale) (embedding : EucSpace d)
    (raw : ℕ → EucSpace d) (k : ℕ) :
    ‖residualTrace eps scale (GPTMini.rmsNormEps eps embedding) raw k‖ ≤
      (1 + (k : ℝ) * scale) * Real.sqrt (d : ℝ) := by
  have hi := GPTMini.rmsNormEps_norm_le eps heps hd embedding
  have ht := residualTrace_bound eps scale heps hd hscale
    (GPTMini.rmsNormEps eps embedding) raw k
  nlinarith

/-- After L layers, unit branch coefficients give the RMS-unit bound 1 + 2L. -/
theorem unit_depth_bound {d : ℕ} (eps : ℝ) (heps : 0 < eps) (hd : 0 < d)
    (embedding : EucSpace d) (raw : ℕ → EucSpace d) (layers : ℕ) :
    ‖residualTrace eps 1 (GPTMini.rmsNormEps eps embedding) raw (2 * layers)‖ ≤
      (1 + 2 * (layers : ℝ)) * Real.sqrt (d : ℝ) := by
  simpa using embedding_trace_bound eps 1 heps hd (by norm_num) embedding raw (2 * layers)

/-- Scaling AFTER output RMS by 1/sqrt(2L) gives the smaller bound 1 + sqrt(2L).
This is the scale in peri-rms-scaled-lr2; L must be positive. -/
theorem scaled_depth_bound {d : ℕ} (eps : ℝ) (heps : 0 < eps) (hd : 0 < d)
    (embedding : EucSpace d) (raw : ℕ → EucSpace d) (layers : ℕ) (hlayers : 0 < layers) :
    ‖residualTrace eps (1 / Real.sqrt (2 * (layers : ℝ)))
        (GPTMini.rmsNormEps eps embedding) raw (2 * layers)‖ ≤
      (1 + Real.sqrt (2 * (layers : ℝ))) * Real.sqrt (d : ℝ) := by
  have hpos : (0 : ℝ) < 2 * (layers : ℝ) := by positivity
  have hspos : 0 < Real.sqrt (2 * (layers : ℝ)) := Real.sqrt_pos.mpr hpos
  have hsquare := Real.sq_sqrt hpos.le
  have hcoef : (2 * (layers : ℝ)) * (1 / Real.sqrt (2 * (layers : ℝ))) =
      Real.sqrt (2 * (layers : ℝ)) := by
    field_simp
    nlinarith
  have ht := embedding_trace_bound eps (1 / Real.sqrt (2 * (layers : ℝ)))
    heps hd (by positivity) embedding raw (2 * layers)
  simpa only [Nat.cast_mul, Nat.cast_ofNat, hcoef] using ht

/-- The normalized residual bound stated directly in RMS units, norm/sqrt(d). -/
theorem scaled_depth_rms_bound {d : ℕ} (eps : ℝ) (heps : 0 < eps) (hd : 0 < d)
    (embedding : EucSpace d) (raw : ℕ → EucSpace d) (layers : ℕ) (hlayers : 0 < layers) :
    ‖residualTrace eps (1 / Real.sqrt (2 * (layers : ℝ)))
        (GPTMini.rmsNormEps eps embedding) raw (2 * layers)‖ / Real.sqrt (d : ℝ) ≤
      1 + Real.sqrt (2 * (layers : ℝ)) := by
  have hspos : 0 < Real.sqrt (d : ℝ) := by positivity
  exact (div_le_iff₀ hspos).mpr (scaled_depth_bound eps heps hd embedding raw layers hlayers)

/-- Scaling after normalization also scales its Lipschitz constant. This controls
only the normalization map; it supplies no bound for an arbitrary raw branch. -/
theorem scaled_rms_lipschitz {d : ℕ} (eps scale : ℝ) (heps : 0 < eps)
    (hd : 0 < d) (hscale : 0 ≤ scale) (x y : EucSpace d) :
    ‖scale • GPTMini.rmsNormEps eps x - scale • GPTMini.rmsNormEps eps y‖ ≤
      (2 * scale / Real.sqrt eps) * ‖x - y‖ := by
  calc
    ‖scale • GPTMini.rmsNormEps eps x - scale • GPTMini.rmsNormEps eps y‖ =
        scale * ‖GPTMini.rmsNormEps eps x - GPTMini.rmsNormEps eps y‖ := by
      rw [← smul_sub, norm_smul, Real.norm_eq_abs, abs_of_nonneg hscale]
    _ ≤ scale * ((2 / Real.sqrt eps) * ‖x - y‖) :=
      mul_le_mul_of_nonneg_left (GPTMini.rmsNormEps_lipschitz eps heps hd x y) hscale
    _ = (2 * scale / Real.sqrt eps) * ‖x - y‖ := by ring

/-- Nonzero inputs witness the hypotheses; epsilon is the FP32 machine epsilon
used by the scientific runs. The proof remains over real arithmetic. -/
noncomputable def witness : EucSpace 1 := EuclideanSpace.single (0 : Fin 1) 1

example : ‖witness‖ = 1 := by simp [witness, PiLp.norm_single]

example : ‖residualTrace (1 / 8388608) 1 witness (fun _ => witness) 8‖ ≤
    ‖witness‖ + (8 : ℝ) * 1 * Real.sqrt (1 : ℝ) := by
  simpa using residualTrace_bound (d := 1) (1 / 8388608) 1 (by norm_num) (by norm_num)
    (by norm_num) witness (fun _ => witness) 8

example : ‖residualTrace (1 / 8388608) 1 (GPTMini.rmsNormEps (1 / 8388608) witness)
    (fun _ => witness) 8‖ ≤ (1 + (8 : ℝ) * 1) * Real.sqrt (1 : ℝ) := by
  simpa using embedding_trace_bound (d := 1) (1 / 8388608) 1 (by norm_num) (by norm_num)
    (by norm_num) witness (fun _ => witness) 8

example : ‖residualTrace (1 / 8388608) 1 (GPTMini.rmsNormEps (1 / 8388608) witness)
    (fun _ => witness) (2 * 4)‖ ≤ (1 + 2 * (4 : ℝ)) * Real.sqrt (1 : ℝ) := by
  simpa using unit_depth_bound (d := 1) (1 / 8388608) (by norm_num) (by norm_num)
    witness (fun _ => witness) 4

example : ‖residualTrace (1 / 8388608) (1 / Real.sqrt (2 * (4 : ℝ)))
    (GPTMini.rmsNormEps (1 / 8388608) witness) (fun _ => witness) (2 * 4)‖ ≤
    (1 + Real.sqrt (2 * (4 : ℝ))) * Real.sqrt (1 : ℝ) := by
  simpa using scaled_depth_bound (d := 1) (1 / 8388608) (by norm_num) (by norm_num)
    witness (fun _ => witness) 4 (by norm_num)

example : ‖residualTrace (1 / 8388608) (1 / Real.sqrt (2 * (4 : ℝ)))
    (GPTMini.rmsNormEps (1 / 8388608) witness) (fun _ => witness) (2 * 4)‖ /
    Real.sqrt (1 : ℝ) ≤ 1 + Real.sqrt (2 * (4 : ℝ)) := by
  simpa using scaled_depth_rms_bound (d := 1) (1 / 8388608) (by norm_num) (by norm_num)
    witness (fun _ => witness) 4 (by norm_num)

example : ‖(1 / 2 : ℝ) • GPTMini.rmsNormEps (1 / 8388608) witness -
    (1 / 2 : ℝ) • GPTMini.rmsNormEps (1 / 8388608) (2 • witness)‖ ≤
    (2 * (1 / 2 : ℝ) / Real.sqrt (1 / 8388608)) * ‖witness - 2 • witness‖ := by
  simpa using scaled_rms_lipschitz (d := 1) (1 / 8388608) (1 / 2)
    (by norm_num) (by norm_num) (by norm_num) witness (2 • witness)

#print axioms Early99.scaled_rms_lipschitz
#print axioms Early99.residualTrace_bound
#print axioms Early99.embedding_trace_bound
#print axioms Early99.unit_depth_bound
#print axioms Early99.scaled_depth_bound
#print axioms Early99.scaled_depth_rms_bound
#print axioms Transformer.GPTMini.rmsNormEps_norm_le
#print axioms Transformer.GPTMini.rmsNormEps_lipschitz

end Early99
