Existing mHC proofs checked natively for this experiment, 2026-11-08.

Source: /home/misha/lean_projects/transformer
Build checked at ca8e4bc748e5ed3dfec974725ced0f40413aef6d.
The local mhc-source.sha256 records the relevant files; this shared repository
can receive unrelated concurrent commits. No Lean source was changed for this
port. Generated logs, hash inventories and validation JSON remain local and
are excluded from commits; the commands below regenerate the proof checks.

Commands (from that repository):
  LAKE_ARTIFACT_CACHE=false lake build Transformer.MHC.PeriRMS
  LAKE_ARTIFACT_CACHE=false lake env lean scripts/MHCPeriRMSAxioms.lean

Both passed. The checked declarations use only propext, Classical.choice and
Quot.sound. See mhc-build.txt and mhc-axioms.txt.

Theorem: for positive epsilon/width/depth, four embedding copies, arbitrary
input-dependent read/write/skip predictors, twenty column-then-row cycles,
Peri-RMS sublayers and scale 1/sqrt(2L), each stream and their mean have RMS at
most 1+2*sqrt(2L); final RMS features have RMS<=1. Normalized Q/K and inverse-width
SSMax bound the absolute score by log(N). Finite Sinkhorn need not be exactly
doubly stochastic; the last row normalization suffices for this forward bound.

The old CurrentModel docstring mentions SwiGLU and XSA. The actual theorems
quantify over arbitrary raw sublayer functions, so their forward norm bounds
also apply to ordinary attention and the plain complex-product branch. They
are not a verified translation of our executable. Native numerical/AD checks
are separate evidence. There is no proof of optimizer convergence, improved
generalization, a bounded full nonlinear Jacobian, exact FP32 conservation,
or a readout-logit bound independent of learned embedding weights.

Additional local certificate: MHCNormalization.lean proves four exact-real
statements for the optimized kernel: row normalization bounds column sums,
the transposed column normalization bounds row sums, each completed cycle
bounds its column sums, and every later column/row denominator lies in[1/4,4].
These are derived consequences of the paper's operators, not assertions of
exact floating-point double stochasticity. All hypotheses have witnesses.
The four declarations checked with only the three standard axioms above.

Check from the Lean repository:
  LAKE_ARTIFACT_CACHE=false lake env lean /home/misha/python_projects/gpt/research/main_model/proofs/MHCNormalization.lean
Output: mhc-normalization-check.txt. The optimized first cycle remains in log
space because its initial row denominators do not have that lower bound.
