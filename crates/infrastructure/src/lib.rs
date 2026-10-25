//! The infrastructure layer: the adapters that reach outside the process.
//!
//! It will implement the application's ports and hold what
//! `src/diff_gpt/infrastructure` holds in the frozen Python reference: the
//! `SQLite` data loader, the training across NUMA nodes, the file lock, the
//! provenance, the leaderboards' evaluation and the synthetic corpus; and
//! the burn backends, which only this layer and the interfaces pick. The
//! harness's port map names the module and the port task of each.
//!
//! The rule. The infrastructure depends on the domain and the application,
//! never on the interfaces. It makes no setting of the whole process
//! (threads, affinity, environment, signals, logging): the command lines
//! make them. It has no binary. The harness's layer test checks the rule on
//! `cargo metadata` and on the source.
#![forbid(unsafe_code)]
