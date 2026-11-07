# Complex transformer and phase-coupled AdamW

User request, 2026-11-07: reformulate the current cubic transformer in complex
space, compare AdamW with an optimizer suited to complex coordinates, and
implement the experiment in **Rust/Kernite**, using the existing examples.
Save every **500 epochs**, plus the initial state and completion/clean stop.
There is no Python preparation, training, scoring, or test dependency.

The question is whether complex-linear structure and phase-adding products
help learn `(a+b)^3 mod 97`, and whether sharing each complex parameter's
second moment helps independently of that architecture change.

| Arm | Projections and feed-forward operation | Optimizer |
| --- | --- | --- |
| real-adamw | Current normalized real GPT, ordinary SwiGLU | AdamW |
| complex-diagonal-adamw | Complex projections, radial gate, coordinate product | AdamW |
| complex-product-adamw | Same weights and gate, complex product | AdamW |
| complex-diagonal-phase | Same diagonal control | Shared pair second moment |
| complex-product-phase | Same complex product | Shared pair second moment |

All arms retain repaired RMS backward, embedding/branch RMS, residual scale
`1/sqrt(2L)`, geometric RoPE, inverse-width SSMax, real XSA, prefix guarding,
and the tied 97-class readout. The pilot uses width128, depth1, four heads,
batch256 (tail96), LR.002, betas(.9,.98), epsilon1e-8, decay.1, warmup50 and
seeds0/1/2. Labels and the disjoint 4704/4705 split match the existing task.
Both input residues are ordinary learned tokens; no Fourier codes, cube roots,
intermediate sums, or answer-specific features are supplied to the network.

RoPE pairs real and imaginary components **within each head**. In that layout,
RMS is `z/sqrt(sum |z|^2/(2m)+epsilon)`, and scores/readout are real parts of
Hermitian inner products. These operations already admit a complex description.
The actual restriction is `Wz`, whereas a general real projection is
`Wz+V conjugate(z)`. Matrices are stored in their real expansion, but their
four blocks share independent A/B parameters and gradients are contracted
before either optimizer updates them. This has 208768 independent real
parameters, versus 208896 in the real control; expanded storage is larger.
Complex hidden width426 replaces real hidden width341, so compute is not matched.

The two complex models share `s(g)=sigmoid(sqrt(|g|^2+epsilon))`. Their
products are `s(g)*(g_R*u_R, g_I*u_I)` and `s(g)*g*u/sqrt(2)` respectively.
The latter adds phases; sqrt(2) calibrates variance for independent isotropic
inputs. This is a radial replacement for SwiGLU, not analytic complex SiLU.
The diagonal control isolates cross-coordinate multiplication. Comparison
against the real arm also changes projection structure, gate and hidden width.
Real XSA is retained: it removes one real direction, not an entire complex line.

Ordinary AdamW is valid on real/imaginary coordinates but uses different second
moments for them. The alternative keeps the same complex first moment and uses
`v = beta2*v + (1-beta2)*(g_R^2+g_I^2)/2` for both coordinates. The factor1/2
matches the expected coordinate scale for isotropic gradients. This update
commutes with per-parameter phase rotations when parameters and gradients
are rotated together; it is not invariant to arbitrary unitary mixing and
does not imply faster convergence. Decay, bias corrections and epsilon are retained.

The local Lean basis is `Transformer.SequenceModel.rotate_pair_sq`,
`rotateHead_isometry`, `rope_relative`, and
`Transformer.Residual.rms_vjp_decomposition`. FlatSpectrum.FeatureCodes
supports classification from separating recovered features under explicit
conditions. It does not establish that this architecture or optimizer learns
those features. The new product and optimizer are algebraic proposals checked
numerically here, not newly proved Lean theorems.

Before observing outcomes, freeze a **250-epoch exploratory pilot** (4750
updates) for all15 cells, stop at first99% or120 seconds per cell including
evaluation, allow300 seconds setup per cell, and cap the whole cohort at1800
seconds. This tests early learning only; a miss does not rule out late grokking.
Rotate arm order across seeds; retain every error/censoring. No architecture
promotion or historical runtime comparison follows from this pilot.

Initialization and epoch shuffling use the existing Kernite microgpt example's
native RNG. They are shared across the complex arms; embeddings are shared
with the real control. These differ from Torch's RNG, so all controls are fresh.
The scalar FP64 Rust reference, finite differences, full/tail-batch state
continuation, and kernel tests validate execution before learning.

History stays in a Rust byte vector. At500-epoch boundaries or completion it
is written alongside parameters, both optimizer moments and sampler position.
A durable intent bounds possible unsaved updates after a crash; ambiguous
automatic restart is rejected. `STOP` in a run directory requests a clean stop.

```sh
cargo test --offline --release --manifest-path research/complex_transformer/native/Cargo.toml
cargo test --offline --release --manifest-path research/complex_transformer/native/Cargo.toml -- --ignored --test-threads=1
cargo build --offline --release --manifest-path research/complex_transformer/native/Cargo.toml
research/complex_transformer/native/target/release/complex-transformer-kernite pilot runs/complex-transformer
research/complex_transformer/native/target/release/complex-transformer-kernite audit runs/complex-transformer
```

Sources: current GPT73bd227; native GPT backend snapshot recorded in
`provenance.json`; Kernite a1b6bc2c956e1a5f35960e26b209e1b6c213096c,
including its microgpt RNG and matrix-multiplication example. Kernite is
currently checked out at `/home/misha/rust_projects/kgen`, which the manifest
uses directly. No Kernite library patch is required.

All15 pilot runs completed250 epochs; all saved endpoints reproduce their
exact correctness counts. The [raw summary](results.json) retains every cell,
checkpoint/history hashes, training accuracy, loss and wall time.

| Arm | Seed0 heldout | Seed1 heldout | Seed2 heldout | Mean |
| --- | ---: | ---: | ---: | ---: |
| real-adamw | 58.21% | 53.56% | 64.48% | 58.75% |
| complex-diagonal-adamw | 63.15% | 49.73% | 49.93% | 54.27% |
| complex-product-adamw | 95.52% | 94.11% | 94.84% | 94.82% |
| complex-diagonal-phase | 55.83% | 50.33% | 46.29% | 50.82% |
| complex-product-phase | 91.56% | 92.37% | 74.39% | 86.11% |

The complex product improves over its paired diagonal control in all three
seeds with either optimizer. The diagonal arm does not consistently improve
over the real control; it changes both projections and the gate, so that
comparison does not isolate projections alone. Ordinary AdamW beats the phase-coupled variant
on the product in all three seeds at the declared learning rate; phase
equivariance alone does not establish better optimization.

No cell reached99%. These results support faster early generalization from
the product in this cohort, not eventual recovery or an architecture promotion.
The phase-product seed2 also fits training less well (95.71% versus99.87% for
ordinary AdamW), so that optimizer contrast includes an optimization gap.

All9 native checks pass (2 CPU and7 CUDA), including full-model FP64/derivative
comparisons and exact checkpoint continuation with both optimizers. Clippy
passes with warnings denied. The15 training/evaluation/I/O segments total
166.80 seconds; input generation, compilation, tests and endpoint replay are
separate. Raw states are under `runs/complex-transformer/`. Since every run
ends at250 epochs, only initial and final states were saved; the regular
persistence interval remains500 epochs. Source/protocol commit: `1e58a9a`.

## Longer continuation

On the user's request to train for more epochs, continue all15 pilot cells to
**100000 absolute updates** (about5263.16 epochs), with their existing weights,
AdamW moments and sample cursor. Add three fresh original-GPT controls at
LR.001: ordinary attention scaling, no SSMax, and the original normalization
recipe. The historical Torch seeds reached99% at3531,3197 and4315 epochs;
these native controls use the cohort's native RNG and are not historical replays.
The full normalized real control retains LR.002, as do all complex cells.

Freeze all18 outcomes before further learning. Stop at the first observed
4658/4705 correct, the100000-update limit, or300 seconds per cell including
the pilot segment. Evaluation and checkpoint I/O count toward this time;
input preparation is recorded separately, with a300-second cohort setup cap
and5400-second overall cap checked between cells. Rotate six-arm execution
order across seeds. No optimizer, rate, gate, or mathematical kernel is changed.

Continuation is explicit because the input horizon and orchestration source
change. It accepts only the pinned pilot source after checking unchanged host
math and compiled GPU kernels. Initial weights, data, RoPE and the entire
issued sample/boundary prefix must match byte for byte. Parameters, both
moments and the update counter transfer without changing any checkpoint byte;
the original pilot remains intact. New checkpoints bind to the new source and
input hashes. A dedicated regression checks continuation past the old horizon
and through the end of warmup against uninterrupted execution with both
optimizers. The original-GPT control also passes the independent FP64 model
and directional derivative checks. Saves remain every500 epochs and at exit.

```sh
research/complex_transformer/native/target/release/complex-transformer-kernite long runs/complex-transformer runs/complex-transformer-long
research/complex_transformer/native/target/release/complex-transformer-kernite audit-long runs/complex-transformer-long
```

The long audit checks every endpoint against its saved correctness and loss,
all observation steps, first99 crossings, paired initializations and sample
orders, and the unchanged pilot-history prefix. All misses and errors remain
in the summary. More epochs test delayed generalization; they do not turn a
finite accuracy observation into a recovery or uniqueness theorem.
