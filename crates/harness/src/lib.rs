//! The port's tools.
//!
//! [`layers`] checks the dependency rule of the workspace's layers on
//! `cargo metadata`, `cargo tree`, the manifests and the source; the test
//! `tests/layers.rs` runs it on this workspace.
#![forbid(unsafe_code)]

pub mod layers;
