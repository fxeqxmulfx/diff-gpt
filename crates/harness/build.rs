//! The bench of the backends finds libtorch's libraries where `LIBTORCH`
//! (`.cargo/config.toml`) says, through its binary's rpath.
#![forbid(unsafe_code)]

fn main() {
    println!("cargo:rerun-if-env-changed=LIBTORCH");
    if std::env::var_os("CARGO_FEATURE_BACKENDS").is_some()
        && let Some(libtorch) = std::env::var_os("LIBTORCH")
    {
        let lib = std::path::Path::new(&libtorch).join("lib");
        println!(
            "cargo:rustc-link-arg-bin=backends=-Wl,-rpath,{}",
            lib.display()
        );
    }
}
