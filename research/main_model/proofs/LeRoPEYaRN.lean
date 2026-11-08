import Mathlib

/-! Local real-algebra checks for LEROPE_YARN_PLAN.txt. These do not prove
FP32 kernel refinement, learned-model generalization, or epoch improvement. -/
namespace MainModelLeRoPEYaRN
noncomputable section

def ramp (r : ℝ) : ℝ := max 0 (min 1 ((r-1)/31))
def blend (u w : ℝ) : ℝ := u+w*(1-u)
def temperatureGain (s : ℝ) : ℝ := (1+Real.log s/10)^2

theorem ramp_bounds (r : ℝ) : 0 ≤ ramp r ∧ ramp r ≤ 1 := by
  constructor
  · exact le_max_left _ _
  · exact max_le (by norm_num) (min_le_left _ _)

theorem blend_bounds (u w : ℝ) (_hu : 0 < u) (hu1 : u ≤ 1)
    (hw : 0 ≤ w) (hw1 : w ≤ 1) : u ≤ blend u w ∧ blend u w ≤ 1 := by
  unfold blend
  constructor <;> nlinarith

theorem positive_frequency (omega u r : ℝ) (ho : 0 < omega)
    (hu : 0 < u) (hu1 : u ≤ 1) : 0 < omega*blend u (ramp r) := by
  have hb := blend_bounds u (ramp r) hu hu1 (ramp_bounds r).1 (ramp_bounds r).2
  exact mul_pos ho (lt_of_lt_of_le hu hb.1)

theorem unextended_frequency (omega r : ℝ) : omega*blend 1 (ramp r) = omega := by
  simp [blend]

theorem unextended_temperature : temperatureGain 1 = 1 := by
  norm_num [temperatureGain]

theorem temperature_ge_one (s : ℝ) (hs : 1 ≤ s) : 1 ≤ temperatureGain s := by
  have h := Real.log_nonneg hs
  unfold temperatureGain
  nlinarith [sq_nonneg (Real.log s)]

def rotatedReal (x y angle : ℝ) : ℝ := x*Real.cos angle-y*Real.sin angle
def rotatedImag (x y angle : ℝ) : ℝ := x*Real.sin angle+y*Real.cos angle

theorem rotation_energy (x y angle : ℝ) :
    (rotatedReal x y angle)^2+(rotatedImag x y angle)^2 = x^2+y^2 := by
  unfold rotatedReal rotatedImag
  nlinarith [Real.sin_sq_add_cos_sq angle,
    congrArg (fun z : ℝ => z*x^2) (Real.sin_sq_add_cos_sq angle),
    congrArg (fun z : ℝ => z*y^2) (Real.sin_sq_add_cos_sq angle)]

theorem axial_relative_phase (t1 t2 f1 f2 omega nu : ℝ) :
    t2*omega-t1*omega = (t2-t1)*omega ∧
    f2*nu-f1*nu = (f2-f1)*nu := by constructor <;> ring

/-- With squared-imaginary scores, amplitude-scaling Q/K adds a different term
than scaling the complete score. This identity states the exact discrepancy. -/
theorem amplitude_temperature_discrepancy (gain re im beta d : ℝ) :
    (gain*re+gain^2*beta*im^2/d)-gain*(re+beta*im^2/d) =
      gain*(gain-1)*beta*im^2/d := by ring

end
end MainModelLeRoPEYaRN

#print axioms MainModelLeRoPEYaRN.positive_frequency
#print axioms MainModelLeRoPEYaRN.rotation_energy
#print axioms MainModelLeRoPEYaRN.amplitude_temperature_discrepancy
