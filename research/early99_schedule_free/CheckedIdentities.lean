import Transformer.ScheduleFree.AdamW.Basic

/-!
Check the exact modern Schedule-Free two-buffer identities used by this study.
The external completed-update warmup supplies the effective rate to a step with
internal warmup zero. The packing map depends on beta1, which is constant here,
not on this changing rate. These checks establish real arithmetic identities,
not convergence, floating-point error bounds, or a complete program proof.
The library's fixed-parameter trainingRun_pack is printed for context; it is not
claimed to directly model our externally changing learning rate.
-/

namespace Early99ScheduleFree
open Transformer.ScheduleFree Transformer.ScheduleFree.AdamW

noncomputable def parameters (baseRate decay : ℝ) (completed : ℕ) : Parameters :=
  { adam := {
      learningRate := baseRate * min 1 ((completed : ℝ) / 50)
      decay := decay
      warmup := 0
      betaOne := 9 / 10
      betaTwo := 98 / 100
      epsilon := 1 / 100000000 } }

-- All coordinates and moments are unrestricted: the update identity applies
-- even on the zero-rate first update and independently of convexity.
example {ι : Type*} (baseRate decay : ℝ) (completed : ℕ)
    (s : State ι) (g : ι → ℝ) :
    trainingStep (parameters baseRate decay completed) completed
      (pack (parameters baseRate decay completed) s) g =
      pack (parameters baseRate decay completed)
        (step (parameters baseRate decay completed) completed s g) :=
  trainingStep_pack _ _ _ _

-- The actual beta1=.9 satisfies the nonzero hypothesis needed to recover x.
example {ι : Type*} (baseRate decay : ℝ) (completed : ℕ) (s : State ι) :
    evaluation (parameters baseRate decay completed)
      (pack (parameters baseRate decay completed) s) = s.evaluation := by
  apply evaluation_pack
  norm_num [parameters]

-- Changing the external rate cannot change the packing map itself.
example {ι : Type*} (r r' decay : ℝ) (t t' : ℕ) (s : State ι) :
    pack (parameters r decay t) s = pack (parameters r' decay t') s := by
  rfl

#check trainingStep_pack
#check evaluation_pack
#check trainingRun_pack
#print axioms trainingStep_pack
#print axioms evaluation_pack
#print axioms trainingRun_pack
end Early99ScheduleFree
