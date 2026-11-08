# Execution preference

The user requires Rust and Kernite and explicitly prohibits executing Python,
now and in future work. Do not invoke Python for tests, experiments, data
preparation, reporting, or editing scripts. Reading existing Python sources to
understand and port their behavior is allowed. Use native Rust tests and
Kernite-generated PTX for transformer experiments.

# Generated files and commits

Do not commit files that can be regenerated from the repository's sources and
build commands. PTX files and Kernite outputs are explicit exceptions and may
be committed even when they can be regenerated.
