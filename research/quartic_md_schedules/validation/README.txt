Native validation, 2026-11-07

Twelve tests passed, zero failed. See native-tests.txt. This includes actual
CUDA execution, FP64 complex reference updates at schedule boundaries, zero
warmup, AdamMDW storage repair and zero-result handling, exact continuation,
all-step schedule checks and bit-identical parity with the original optimizer.
The combined-report test uses explicitly synthetic fixtures outside run paths.

All16 new arms passed a19-update wiring smoke test; all losses/state values
were finite and complex constraints remained valid. See smoke.json. These
are not generalization results or controlled performance benchmarks; the
prior scientific GPU cohort was active during these small checks.

Formatting, Clippy with -D warnings and shell syntax checks passed.
Original quartic and complex-geometry model sources remained unchanged.
The original queued executables and protocol passed their SHA256 manifest.
Kernite was clean at a1b6bc2c956e1a5f35960e26b209e1b6c213096c.
Compiler: rustc1.99.1; cargo1.99.1. No Python was executed.

provenance.json is the final compiled source/PTX identity. Scientific cells
will run only after the original six-cell quartic queue passes its final audit.
