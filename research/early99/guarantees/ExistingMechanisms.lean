import Transformer.SequenceModel.Attention.CenteredXSA
import Transformer.Residual.RMSBackward

/-!
Dependency checks for mechanisms already present in the frozen baseline.

Centered XSA equals the clipped original XSA when attention weights sum to one,
including the small-own-value region. The RMS backward identities isolate the
radial coefficient without changing the real derivative. These are existing
results from transformer commit 8890a985, not improvements introduced by the
new candidate and not proofs of floating-point error or optimizer convergence.
-/

#print axioms Transformer.SequenceModel.Attention.centeredXSA_eq
#print axioms Transformer.SequenceModel.Attention.headOutput_constant_values
#print axioms Transformer.Residual.rms_vjp_decomposition
#print axioms Transformer.Residual.rms_vjp_parallel
#print axioms Transformer.Residual.rms_backward_coefficient_ratio
#print axioms Transformer.Residual.radialRMS_zero_hasDerivAt
