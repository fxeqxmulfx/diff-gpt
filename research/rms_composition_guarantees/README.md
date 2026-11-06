# Two adjacent RMS normalizations

This supplement studies the actual embedding-RMS followed by first-block input
RMS, with the same positive epsilon and channel width. The [plan](plan.json)
keeps the running architecture study unchanged.

For `R_e(x) = x / sqrt(mean(x^2) + e)`, the existing checked one-operation
Lipschitz bound is `1/sqrt(e)`. The planned statements check the two-operation
bound `1/e`, its exact derivative at zero along unit-RMS directions, and the
finite-epsilon scalar composition identity. These are sensitivity statements
about adjacent normalizers, not about the full attention/FFN or residual network.

The source definitions are `Transformer.GPTMini.rmsNormEps` and the existing
`Transformer.Residual.radialRMS`, whose unit-direction mapping is already proved.
The real-arithmetic identity keeps epsilon; no ideal scale invariance is assumed.
Arithmetic rounding, optimization and explanations of observed failures remain
outside this proof. No learned checkpoint or held-out outcome determines a bound.
