//! Capture the reused main model and optimizer sources without editing either.
#[path = "../../native/src/sources.rs"]
mod main_sources;
pub fn files() -> Vec<(&'static str, &'static str)> {
    let mut files = main_sources::FILES.to_vec();
    files.extend([
        (
            "research/main_model/wsd_lr001_native/src/main.rs",
            include_str!("main.rs"),
        ),
        (
            "research/main_model/wsd_lr001_native/src/runner.rs",
            include_str!("runner.rs"),
        ),
        (
            "research/main_model/wsd_lr001_native/src/sources.rs",
            include_str!("sources.rs"),
        ),
        (
            "research/main_model/wsd_lr001_native/src/tests.rs",
            include_str!("tests.rs"),
        ),
        (
            "research/main_model/wsd_lr001_native/Cargo.toml",
            include_str!("../Cargo.toml"),
        ),
        (
            "research/main_model/wsd_lr001_native/Cargo.lock",
            include_str!("../Cargo.lock"),
        ),
        (
            "research/main_model/WSD_LR001_PLAN.txt",
            include_str!("../../WSD_LR001_PLAN.txt"),
        ),
        (
            "research/main_model/wsd_lr001_native/src/optimizer.rs",
            include_str!("optimizer.rs"),
        ),
        (
            "research/main_model/wsd_lr001_native/src/schedule.rs",
            include_str!("schedule.rs"),
        ),
        (
            "research/quartic_md_schedules/native/src/schedule.rs",
            include_str!("../../../quartic_md_schedules/native/src/schedule.rs"),
        ),
        (
            "research/main_model/wsd_lr001_native/src/optimizer_kernels.rs",
            include_str!("optimizer_kernels.rs"),
        ),
        (
            "research/quartic_muonmd/native/src/md_ops.rs",
            include_str!("../../../quartic_muonmd/native/src/md_ops.rs"),
        ),
    ]);
    files
}
