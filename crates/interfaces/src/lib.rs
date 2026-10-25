//! The interfaces layer: the command lines.
//!
//! One binary per entry point of `src/diff_gpt/interfaces` in the frozen
//! Python reference: it parses and checks its arguments, makes the settings
//! of the whole process (threads, affinity, environment, signals, logging),
//! builds the adapters of the infrastructure, calls one use case and prints
//! what it returns. The harness's port map names the binary and the port
//! task of each.
//!
//! The rule. The interfaces may depend on every layer, and no layer depends
//! on them; only they have binaries. The harness's layer test checks the
//! rule on `cargo metadata` and on the source.
#![forbid(unsafe_code)]
