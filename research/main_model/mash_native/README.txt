Native DiffGPT encoders: MASH,LGT,MASH-half

This CPU-only Rust library prepares the user's later2DLeRoPE+YaRN DiffGPT
comparison. All three encoders will be tested as separate model conditions.
It implements level_grid,encode and decode for explicit finite row-major
series,per-column difference operators and symmetric/asymmetric domains.
MASH and MASH-half preserve their prefix anchors; LGT uses its absolute grid.
Forecasting passes the raw tail as a new prefix,as the existing DiffGPT does.

No Python is executed. The source reference is encoder version4 in
src/diff_gpt/domain/encoder_decoder.py. Native BigRational arithmetic chooses
the same directed binary64 grid and exact quantized levels; modular uint64
arithmetic advances fractional anchors; checked int64 arithmetic reconstructs
levels. Decoding uses binary64. Exact rational arithmetic is CPU preprocessing,
not a GPU precision fallback. Codec errors and clipped-token counts are explicit.

Commands from repository root:
cargo test --offline --release --manifest-path research/main_model/mash_native/Cargo.toml
cargo clippy --offline --release --manifest-path research/main_model/mash_native/Cargo.toml --all-targets -- -D warnings
RUSTDOCFLAGS='-D warnings' cargo doc --offline --no-deps --manifest-path research/main_model/mash_native/Cargo.toml

Twelve tests pass. The independent dyadic-bit oracle covers2000level cases,
including rounding ties and scaled underflow. Unbounded integer recurrences
check modular anchors. Known repository rounding failures,constant streams,
seasonal/asymmetric/mixed-column cases,causal extension,clipping,overflow and
each encoder's forecast re-anchoring pass. Every9408Fibonacci97window passes
for each of the three encoders:84672token checks against independent formulas,
with zero clipping and reconstruction inside each grid's half-step bound
plus binary64 slack. These are codec checks,not model-learning results.

Scope limits: this typed API requires at least one feature,finite input,
positive finite grids,valid tokens and at most61difference factors. It does
not implement missing-value filling,automatic/quantile domain estimation,
pandas interfaces or the full DiffGPT forecaster. Domains are explicit so
the later training experiment can freeze them without test-derived tuning.
No Python/Rust differential execution or exhaustive floating-point theorem
is claimed. Existing Lean source explains the reference's mathematical
contract; this Rust implementation has not been formally verified in Lean.

Evidence: ../results/mash-native-preparation-01/.
This adapter stays in research until the requested final source integration.
