//! The layer rule on this workspace: `cargo metadata`, `cargo tree`, the
//! manifests and the source of every crate.
#![forbid(unsafe_code)]

use std::path::Path;

use diff_gpt_harness::layers;

#[test]
fn the_workspace_keeps_the_layer_rule() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let breaks = layers::check_workspace(&root).expect("the workspace reads");
    if let Some(report) = layers::report(&breaks) {
        panic!("{report}");
    }
}
