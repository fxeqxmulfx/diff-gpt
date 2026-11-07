Complex geometry transformer

A separate native Rust/Kernite model built while complex_ffn_suite trains.
The existing experiment's sources and outputs are not dependencies of this
crate. UPSTREAM.sha256 records the source snapshot used for the port.

The candidate uses complex embeddings, complex-linear Q/K/V, attention-output
and FFN projections, magnitude-based RMS normalization, complex RoPE, a radial
complex-product FFN, Hermitian XSA and a real Hermitian tied readout. Real CUDA
GEMM implements each complex matrix through tied blocks [[A,-B],[B,A]].
All forward/backward kernels and the runner are native Rust/Kernite; no Python.

Comparison variants

arm                        projections       XSA         AdamW variance
real-product               unrestricted real real        per real coordinate
complex-linear             complex-linear   real        per real coordinate
complex-hermitian           complex-linear   Hermitian   per real coordinate
complex-hermitian-phase     complex-linear   Hermitian   shared per complex pair

The first contrast isolates complex-linear constraints, the second isolates
Hermitian exclusion, and the third isolates the optimizer's variance rule.
All four use the same gated complex product, width, hidden width, depth and
batch order. The complex variants share identical initial weights. Dimensions
are matched; parameter counts are not. For the cubic task, width128, 4 heads,
one block, vocabulary97 and real hidden width342 give 209280 expanded real
storage entries: 209280 independent parameters in the real control and 110848
in each complex variant. The expanded implementation saves no GEMM FLOPs.

Which components need complex treatment

For a complex-linear projection, independent A and B determine all four real
blocks. Its real gradient contracts as gA=gRR+gII, gB=gIR-gRI BEFORE AdamW's
moments are updated. Checkpoint load/save enforces exact ties in weights and
optimizer state. Ordinary AdamW is valid for complex variables represented by
real/imaginary parts, but its two variance estimates depend on those axes.
The phase variant uses v <- .98*v+.02*(gR^2+gI^2)/2 with a complex first moment.
The same positive denominator scales both components, making this update
covariant under rotations of complex coordinate axes.

For nonzero v above the norm clip, Hermitian XSA is
    a - v * sum(conj(v_j)*a_j) / sum(|v_j|^2).
It removes both real directions v and i*v, unlike the original real XSA.
The denominator is clipped at norm1e-12; the first query is explicitly zero,
and the original stable constant-prefix handling remains. The GPU computes
centered deltas. The validation oracle computes direct complex inner products.

RMS already depends on complex magnitudes. Preserve its original scale
sqrt(f32::EPSILON + sum(|z|^2)/(2m)), where m is complex width. Changing to a
mean over m would introduce an additional scale change. RoPE is already complex
phase multiplication. Attention uses real scores Re<q,k> and real softmax
probabilities to mix complex values. Classification uses real logits Re<E_y,x>.
These real quantities are appropriate; complex-valued probabilities or losses
are unnecessary. Componentwise ReLU/SwiGLU are absent from the four study arms;
the legacy SwiGLU path exists only for regression tests.

The product sigma(sqrt(|g|^2+eps))*g*u/sqrt(2) adds phases and is deliberately
quadratic. It is not globally phase-equivariant through residual addition.
Magnitude gates, RMS and real attention scores are nonholomorphic but are
differentiable as real maps. The manually implemented backward follows those
real derivatives. No claim of optimality, convergence or a 99% result is made.
The Lean flat-spectrum bridge remains conditional; it does not prove that this
architecture or either optimizer will learn the frozen suite.

Build and validate

From research/complex_geometry_suite/native:
    cargo build --release --offline --locked
    cargo fmt --check
    cargo clippy --offline --locked --all-targets -- -D warnings
    cargo test --release --offline --locked
    cargo test --release --offline --locked -- --ignored --test-threads=1
    target/release/complex-geometry preflight
    target/release/complex-geometry smoke /tmp/complex-geometry-smoke-new

The CUDA tests require driver access. The small smoke command performs eight
updates per arm on the real (a+b)^3 mod97 data, scores the complete splits and
saves once at the end. It writes under engineering/, outside scientific cells.
It checks training execution; eight updates are not a generalization study.
VALIDATION.json records the build checks and smoke evidence.

Kernite path dependencies point to /home/misha/rust_projects/kgen/crates.
The validated backend is a1b6bc2c956e1a5f35960e26b209e1b6c213096c.
The older independent-coordinate layout/optimizer derives from GPT's
complex_transformer at commit786696b. The frozen data/scorers, arbitrary-length
engine and radial product derive from the current complex_ffn_suite snapshot;
its exact original file hashes are retained. THIRD_PARTY.txt preserves the
data-generator compatibility code's attribution and license.

Scientific runs when GPU time is available

    target/release/complex-geometry suite ../runs/cohort-01 short
    target/release/complex-geometry suite ../runs/cohort-01 long
    target/release/complex-geometry report ../runs/cohort-01

Or run the cubic candidate alone with the frozen budget:
    target/release/complex-geometry cell ../runs/cubic-01 arithmetic-sum-cube 0 complex-hermitian-phase long

The manifest contains 25 short and 10 long tasks; 3 seeds and 4 arms give420
cells. The score remains complete held-out exact accuracy >=99%. Budgets,
data seeds, splits and observation cadence remain frozen. Arm order rotates
with the seed. Misses, pending cells and execution errors remain visible.
Long tasks are diagnostics, not a tuning set. No full cohort was launched
as part of building this version.

Observations stay in memory between saves. Checkpoints occur every500 epochs,
plus initialization, first99 crossing, budget end or clean stop. Create STOP
in the cohort or cell directory to request a clean stop at an observation.
Remove STOP and use the same cell arguments with resume instead of cell.
Only clean stopped cells can resume; existing cells are never overwritten.
An unclean work window refuses silent replay. Parameters, both moments, history,
sampler position, hashes and a compile-time source snapshot accompany each run.
