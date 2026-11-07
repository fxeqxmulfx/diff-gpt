Lean check of the causal adaptation of https://arxiv.org/html/2410.07799v3

Reproduce: bash research/complex_geometry_suite/results/pow4-centering/check.sh
Optional first argument: path to the existing transformer Lean project.
The script compiles the existing CausalCentering fixture and CenteringCheck in
a disposable /tmp module directory. It does not edit the separate Lean project.
See validation.json, version/commit, source hashes and the two axiom logs.
Thirteen new and five existing declarations pass, without warnings, holes,
or axioms beyond propext, Classical.choice and Quot.sound.

Proved for normalized causal rows:
  sum_j(p_j-1/N)V_j cancels a constant value and the first-token prefix.
  With two tokens it is(p_0-1/2)(V_0-V_1).
  Uniform attention cancels every value signal, including unequal values.
  Bias-free output projection, parameter-free RMSNorm and residual scaling
  preserve zero. The residual itself preserves its input.
  Along tied/shared parameter variations the canceled branch has an actual
  Frechet derivative of zero. Holding the residual input/downstream function
  fixed, the loss derivative for attention-only parameters is zero as well.
  Independent token-value perturbations can have a nonzero derivative; the
  result is not a statement that every primitive V gradient vanishes.
  A decoder seeing only the canceled attention output cannot solve all quartic
  diagonals: a=0 and a=1 require different answers,0 and16 mod97.

Limits: this is not full-model impossibility. The residual retains token
information, and(a+a)^4=(2a)^4 is a unary function the remaining network might
learn. It does not prove poor generalization, failure to reach99%, numerical
correctness, or AdamW convergence. QK normalization/SSMax/RoPE can change the
normalized probabilities but cannot defeat the constant-value identity when
positional rotation affects only Q/K. Read-only inspection confirms the current
model shares token/value maps and rotates only Q/K. The branch theorem applies
per head; all zero heads also give zero under their linear output projection.

The causal prefix-uniform matrix differs from the paper's global rank-one
matrix: at length two their determinants are1/2 and0, respectively. The paper's
random-matrix asymptotics have not been formalized here and do not transfer
automatically to our masked SSMax model. Its Section4 excludes training
dynamics. No result here blames the normalizer or residual connection.

User initially requested Lean-only checking, then explicitly authorized
"go rust". The validation.json fields about Rust/training describe this Lean
check only. The separate numerical/training scope is CENTERING_PLAN.txt.
