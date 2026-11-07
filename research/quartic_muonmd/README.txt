Quartic complex-linear optimizer diagnostic

Question: does complex MuonMD improve held-out classification of (a+b)^4 mod97
over AdamW with the same linear learning-rate decay? The observed seed1 failure
in the historical constant-rate complex-linear study motivated this diagnostic.

Arm              Matrix optimizer     Peak matrix/gain LR   Embedding LR   WD
adamw-linear     coordinate AdamW     0.002 / none           0.002          0.1
muonmd-linear    complex MuonMD       0.01 / 0.01            0.002          0

The full frozen protocol is PLAN.txt. Both arms use model/sampler seeds0,1,2,
the identical 4704/4705 split and initialization, clipping at global independent
gradient norm1, Adam betas(0.9,0.99), and linear decay to1e-8. Each runs100000
updates even after crossing99%. These are new controls, not exact reruns of the
historical beta2=0.98, warmup50, unclipped constant-LR experiment.

Model: unchanged complex-linear weights and real XSA from GPT commit910f206;
128 real/64 complex coordinates, four heads, one block, real FFN width342,
radially gated complex product, Peri RMS, RoPE, SSMax, tied embedding/readout.
The architecture still contains real XSA. This is not the Hermitian-XSA arm.

MuonMD follows the magnitude/direction recipe in local paper2606.25971v2,
Section3, Section4.1, Appendices A/B/D. Its extension to complex matrices uses
Hermitian products, complex Frobenius norms, and one positive softplus gain
per complex row and column. All factor gradients use the old factors. Raw
Nesterov momentum is m=0.95m+G, followed by G+0.95m. Five Newton-Schulz steps
approximate the polar factor; they do not compute an exact polar decomposition.
The direction uses the fixed multiplier rho/sqrt(min(m,n)), then is reprojected
to its actual initial complex Frobenius radius rho. Tied embeddings use ordinary
coordinate Adam with no new row normalization. The selected0.01 MuonMD rate is
one diagnostic setting, not a tuned optimum. There is no complex MD convergence
theorem for this model in the Lean project.

Implementation: native Rust and Kernite-generated PTX only. Immutable model
modules are imported directly from research/complex_geometry_suite/native/src.
Complex matrix products use their constrained real expansions and FP32 SGEMM.
FP64 reductions accumulate squared norms and gain-gradient sums. Small softplus
arguments use a cubic log1p series to avoid cancellation; no gain floor is added.
All optimizer state is separate per matrix; duplicated real blocks are repaired
exactly. Checkpoints include every persistent optimizer buffer and exact sampler
position. Unclean windows, source changes and corrupt states are rejected.

Native validation:
  cargo fmt --manifest-path research/quartic_muonmd/native/Cargo.toml -- --check
  cargo clippy --offline --locked --all-targets --manifest-path research/quartic_muonmd/native/Cargo.toml -- -D warnings
  cargo test --release --offline --locked --manifest-path research/quartic_muonmd/native/Cargo.toml -- --test-threads=1

Tests compare full optimizer updates with an independent FP64 complex reference;
check Hermitian products for tall, wide, rank-deficient and zero matrices;
check gain derivatives by finite differences, stable positive gains, the frozen
dataset and epoch tails; and require bit-identical continuation across the96-row
tail batch. The model itself retains the checks of commit910f206 unchanged.

Runtime commands: archive ROOT; smoke ROOT; suite ROOT; cell ROOT ARM SEED;
resume ROOT ARM SEED; audit ROOT. A smoke run is19 updates per arm, explicitly
outside the scientific cells. Full runs evaluate all examples every19 updates,
buffer observations, and save every500 epochs, at first99 and at the end. STOP
in the cohort or cell directory makes a clean stop at the next observation.

The launch script run-after-geometry.sh waits for the existing geometry queue's
PID and start time, requires its completed420-cell audit, verifies frozen binary
hashes and reruns the native tests before starting. It never rebuilds or retries
failed scientific cells. It publishes all six cells, including misses and errors.

Results: all six cells completed 100000 updates on 2026-11-07. All ended at 100%
training accuracy; none reached 99% held-out accuracy at any observation.

Arm              Seed 0 final  Seed 1 final  Seed 2 final  Mean final
adamw-linear      98.363443%     98.214665%     96.301807%    97.626638%
muonmd-linear     95.812965%     95.494155%     95.345377%    95.550832%

At these fixed rates, MuonMD had lower final held-out accuracy on each paired
seed. This does not compare tuned optima. Final mean held-out cross-entropy was
0.133954620 for AdamW and 0.386356941 for MuonMD.

results/ contains the native audit, exact metrics, protocol and artifact hashes,
the original seven-test GPU validation log and the completed queue state. A
fresh CPU audit before the results commit found zero integrity failures.
Full histories and checkpoints remain in
/home/misha/python_projects/gpt/runs/quartic-muonmd. These six unchanged cells
are reused in the stopped research/quartic_md_schedules comparison.
