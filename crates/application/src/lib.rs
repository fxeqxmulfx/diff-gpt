//! The application layer: the use cases, which drive the domain.
//!
//! It will hold what `src/diff_gpt/application` holds in the frozen Python
//! reference: `DiffGPT`, the training loop, the in-memory data loaders, the
//! benchmarks, the grokking experiments, GIFT-Eval's forecasters and Track
//! B's pretraining; and the ports, the traits that the outer layers
//! implement (a data loader, a file lock, the git head, ...). The harness's
//! port map names the module and the port task of each.
//!
//! The rule. The application depends on the domain only and on burn's tensor
//! and nn API, with no backend: it computes on the devices it is given,
//! builds none and names no backend. It declares no backend, process,
//! thread, database, socket or device crate, starts no process or thread,
//! reads no environment variable and makes no setting of the whole process;
//! what it needs of them comes in through a port or as a value from the
//! command line. It may read and write the files under the paths it is
//! given. It has no binary. The harness's layer test checks the rule on
//! `cargo metadata`, `cargo tree` and the source.
#![forbid(unsafe_code)]
