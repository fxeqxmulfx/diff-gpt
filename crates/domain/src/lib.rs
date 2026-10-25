//! The domain layer: pure computation on tensors and arrays.
//!
//! It will hold what `src/diff_gpt/domain` holds in the frozen Python
//! reference: the tokenizers (`encoder_decoder`), the GPT model, its KV cache
//! and engine (`model`), its optimizer (`optimizer`), the samplers
//! (`sampler`), the forecast metrics (`metrics`) and the per-window
//! normalization (`normalization`); the harness's port map names the module
//! and the port task of each.
//!
//! The rule. The domain depends on no other layer and on burn's tensor and nn
//! API only, with no backend: since burn 0.22 a tensor's type names none, so
//! the domain computes on the devices it is given, builds none and names no
//! backend. It declares no I/O, process, thread or device crate, opens no
//! file, reads no environment variable, starts no process or thread and
//! makes no setting of the whole process (threads, affinity, environment,
//! signals, logging), and it has no binary. The harness's layer test checks
//! the rule on `cargo metadata`, `cargo tree` and the source.
#![forbid(unsafe_code)]
