Read-only fourth-power checkpoint diagnosis

User requested inspecting the failed complex-hermitian arithmetic-sum-pow4,
seed1 weights. This inspector reads all12 fixed case checkpoints (four arms,
three seeds), including initial parameters, final parameters, both AdamW moments,
metadata and complete training history. It never calls Engine::update.

The model's engine.rs is copied from the frozen scientific implementation;
its only change is diagnostic_snapshot(), which copies activations to the host.
All Kernite kernels, layouts, data generators and packing code are imported
from the unchanged native/ sources. Kernite embeds workspace/crate names in
PTX symbols. After replacing only the diagnostic namespace prefix with the
original prefix, the full PTX hash must match the checkpoint exactly.

Checks: original checkpoint/history/initial SHA256, declared initialization,
finite weights/moments, complex-linear matrix ties and phase variance ties.
GPU inspection reproduces complete train and held-out counts and losses from
each endpoint (loss tolerance1e-7). Both moment arrays and parameters are checked
unchanged after probing, and the input checkpoint file hash is rechecked.

Measurements: matrix norms and singular-energy participation, power-iteration
spectral norm estimate, embedding Fourier energies, saved-moment step scale,
full-split cross-entropy gradients contracted to independent complex parameters,
activation distributions, attention probabilities, Hermitian orthogonality,
errors by input pair/target/sum, and swap/constant-sum consistency on all97^2 pairs.
Relative saved-moment step scale includes decay; it is NOT the next minibatch's
AdamW update, which would incorporate a newly computed gradient and moments.
Success endpoints and exhausted endpoints have different ages; their comparisons
are descriptive and cannot by themselves establish a causal hyperparameter fix.

From this directory:
    cargo build --release --offline --locked
    cargo test --release --offline --locked
    cargo clippy --offline --locked --all-targets -- -D warnings
    target/release/complex-weight-diagnostic weights RUN_ROOT OUTPUT.json
    target/release/complex-weight-diagnostic probe RUN_ROOT OUTPUT.json

The probe command requires CUDA access and should run when training is idle.
Five CPU tests pass. Both modes completed all12 checkpoints; all full-split
replays passed. No Python, training, resume, benchmark overwrite or retuning.
Outputs are ../results/pow4-weights.json, pow4-probes.json and per-cell error TSVs.
Interpretation and limitations are in ../results/pow4-diagnosis.txt.
