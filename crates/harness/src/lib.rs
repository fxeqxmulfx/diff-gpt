//! The port's tools.
//!
//! [`layers`] checks the dependency rule of the workspace's layers on
//! `cargo metadata`, `cargo tree`, the manifests and the source; the test
//! `tests/layers.rs` runs it on this workspace. [`port_map`] reads the port
//! map, `port_map.toml`, and checks it against the reference's commit; the
//! test `tests/port_map.rs` runs it on the map.
#![forbid(unsafe_code)]

pub mod layers;
pub mod port_map;
