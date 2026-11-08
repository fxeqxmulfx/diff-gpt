Native main transformer and DiffGPT codecs

This is the Rust/Kernite implementation of the selected research transformer.
It provides a reusable library and a JSON command line, with no Python or
LibTorch execution. It is an independent workspace alongside the existing
Burn port, whose frozen reference and layer rules remain separate.

DiffGPT uses only axial 2D LeRoPE (time x feature), with inference-only YaRN
beyond the actual training context. This is the user's explicit interface
choice. The generic Model API separately retains the audited 1D main and its
checkpoints/context experiments. DiffGpt rejects 1D checkpoints; it never
silently changes the positional interpretation of trained weights.

The default is complex Peri-RMS with embedding/final/per-head QK RMS,
complex-linear projections, plain complex-product FFN, tied embedding/readout,
LeRoPE, phase-conjugated four-stream mHC and post-normalization residual scaling.
Attention uses ordinary causal value aggregation. For real head width d,
Z=sum(conj(q)*k), score=ln(N)/d*(Re(Z)+beta*Im(Z)^2/d),
beta=a/sqrt(1+a^2). This is the squared-imaginary-score research main, not XSA.
Fresh initialization and all independent optimizer coordinates match the
research implementation. AdamW/WSD uses LR.001,beta(.9,.95),wd.1,clip1,
2%warmup,20%cosine to1e-8; frequencies have separateclip1/wd0.

CUDA requires SM80 or later. Forward affine multiplication uses FP16,
backward affine multiplication TF32, and accumulation/weights/moments/
remaining GPU math FP32. There is no runtime precision fallback. CPU decoding
uses binary64/exact rationals and the schedule retains its binary64 reference.
Kernite CUDA/macros are supplied by the sibling rust_projects/kgen checkout;
Cargo.toml declares that dependency explicitly. Rust1.99.1 is used by the repo.

Build and check from the repository root:
  cargo build --offline --release --manifest-path src/native_transformer/Cargo.toml
  cargo test --offline --workspace --manifest-path src/native_transformer/Cargo.toml
  cargo clippy --offline --workspace --manifest-path src/native_transformer/Cargo.toml --all-targets -- -D warnings
  cargo run --offline --release --manifest-path src/native_transformer/Cargo.toml -- build-info

Library entry points:
  DiffGpt::new(vocab,features,max_batch,max_length,budget,training_input_length,seed)
  DiffGpt::with_config(config,seed), DiffGpt::from_checkpoint(checkpoint)
  DiffGpt::forecast(observed,domains,operators,tokenizer,future_rows)
  ModelConfig::main(vocab,max_batch,max_length,budget), Model::new(config,seed)
  Model::train_step(tokens,targets,batch,length)
  Model::evaluate(tokens,targets,batch,length)
  Model::generate(prompt,count,None)
  Model::forecast(observed,domains,operators,tokenizer,future_rows,None)
  Model::checkpoint(), Checkpoint::save/load(), Model::from_checkpoint()

train_step performs no disk writes or device-to-host metric copy. Evaluation
and checkpoint copying happen only when the caller requests them. Targets use
IGNORE_TARGET for a mask; all other out-of-range targets are rejected.
CheckpointCadence requires500or more epochs, measured from the actual training
split size. Save initialization, desired accuracy milestones, final state and
explicit stops separately. Atomic buffered checkpoints include configuration,
update/example counters, weights, both moments, kernel identity and checksums.
The data iterator is caller-owned; restore its dataset identity and sample
position when resuming. EpochSampler supplies the deterministic native shuffle.
Checkpoint::from_raw imports the research FP32 layout with caller-supplied
trusted geometry/progress. It is not a provenance verification substitute.

Supported real widths64/128/256,head widths16/32/128,positive depth and capacities.
The exposed WSD budgets are5000/20000/30000/100000, retaining the measured recipe.
ModelConfig::diffgpt defaults toD128/H1/L1 and axial positions. Its actual maximum
training input length is persisted with the feature count; for5inputtokens and
2features,the training time extent is3. Allocated capacity is never substituted.
DiffGPT training has no YaRN; forecast extension keeps the feature axis unchanged.
Generic Model generation defaults to1Dwithout extension and accepts an explicit
YaRN reference for context experiments. Do not attribute training-epoch savings
to inference-only YaRN. Experiments and their promotion decisions are recorded
in research/main_model/STATUS.txt and the registered result directories.

CLI:
  diff-gpt-kernite inspect CHECKPOINT
  diff-gpt-kernite init CONFIG_JSON SEED FRESH_CHECKPOINT
  diff-gpt-kernite init-diffgpt CONFIG_JSON FEATURES TRAIN_INPUT_LENGTH SEED FRESH_CHECKPOINT
  diff-gpt-kernite generate CHECKPOINT PROMPT_JSON COUNT
  diff-gpt-kernite forecast CHECKPOINT SERIES_JSON ROWS
  diff-gpt-kernite train SPEC_JSON FRESH_CHECKPOINT
  diff-gpt-kernite train-diffgpt SPEC_JSON FRESH_CHECKPOINT

Configuration JSON example:
{"width":128,"heads":4,"layers":1,"vocab":97,"max_batch":256,
 "max_length":2,"budget":100000,"positions":{"kind":"lerope"}}

PROMPT_JSON is an array of input token IDs. SPEC_JSON has config,seed and
examples:[{tokens:[...],targets:[...]}]. Uniform sequence lengths are required
by this small training CLI; the library supports variable lengths per call.
train-diffgpt additionally takes a top-level features count,defaults missing
positions to axial and rejects an explicit1Dconfiguration. It derives the
training reference from the actual examples. init-diffgpt requires that real
input length explicitly; both commands persist it with the axial feature count.
The CLI accepts-100 as a target-mask spelling. It saves initially,each500epochs,
at budget and on an explicit CHECKPOINT.stop file (checked every50updates).
It refuses an existing output at startup. This generic trainer supplies no
held-out benchmark or first99 claim; the research harness owns those metrics.

SERIES_JSON has features,row-major values,domains:[[low,high],...],orders:[...]
and tokenizer:mash|lgt|mash-half. Forecasts generate each feature from the
observed prefix and preceding predictions, then decode from the final raw
observation prefix. They return only future rows and report context clipping.
The forecast command loads DiffGpt and requires2Dcontext metadata. Generic
1Dcheckpoints remain usable through generate and the Model library API.
All three native codecs are exposed separately as diff_gpt_kernite::codec.

Validation and limitations
The frozen research main passed75short/30long under the registered budgets;
that is task evidence, not a convergence guarantee. Its147strict mixed-precision
AD misses and24near-zero FP32 diagnostic router misses remain documented.
Finite tests and local exact-real Lean statements do not establish universal
floating-point safety. Integration is checked through the public API by
research/main_model/production_check, including the trained quartic checkpoint's
next100updates, serialized continuation, generation and all three codecs.
Executed integration checks passed: 18 CPU tests, formatting, strict Clippy,
rustdoc, exact 100-update trained-checkpoint replay, serialized continuation,
all three encoders and CUDA memcheck/synccheck with zero errors. See
VALIDATION.txt for commands, evidence identities and the limits of these checks.

The separate 36-run DiffGPT learning study did not reach 90% decoded whole-row
accuracy in any 5000-update run, for either 1D or 2D. The best 2D final accuracy
was 89.328231% (MASH-half, one feature, seed 17); two-feature results were poor.
The 2D-only DiffGPT interface follows the user's request. It does not imply
successful learning or extrapolation on these additional tasks. Build outputs,
checkpoints and regenerable result tables stay out of git.
