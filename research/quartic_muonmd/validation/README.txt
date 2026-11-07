Native validation, 2026-11-07

Seven tests passed, zero failed. Native Rust tests include an independent FP64
complex reference, real-coordinate finite differences, CPU audit fixtures and
CUDA execution. See native-tests.txt and ../README.txt for commands and coverage.
Formatting, Clippy with -D warnings and bash -n also passed. All imported model
sources match master at910f206 exactly. Kernite was clean at
 a1b6bc2c956e1a5f35960e26b209e1b6c213096c.

The two smoke JSON files are19-update correctness runs, outside scientific cells.
They preceded final offline-audit additions; model and optimizer kernels were
unchanged by those additions. Their runtimes were measured while the prior GPU
cohort was active and are not controlled performance benchmarks.

provenance.json identifies the final compiled scientific sources and PTX.
There are no scientific training results yet. Full-budget cells will run only
when the existing geometry queue finishes and passes its420-cell audit.
