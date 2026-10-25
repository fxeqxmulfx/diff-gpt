//! The dependency rule of the workspace's layers, read off `cargo metadata`,
//! `cargo tree`, the manifests and the source.
//!
//! A crate's layer is its place in [`CRATES`]: the domain, the application,
//! the infrastructure, the interfaces (the command lines) or the tools (this
//! harness). On the manifests ([`check_members`], [`check_manifests`]):
//! - a crate depends, in any kind of dependency, only on the layers below
//!   it; nothing depends on the interfaces or the tools; every member of the
//!   workspace is one of [`CRATES`];
//! - the domain, the application and the infrastructure declare no crate
//!   that makes settings of the whole process ([`PROCESS_CRATES`]); the
//!   domain and the application none of the operating system, threads,
//!   databases or sockets ([`SYSTEM_CRATES`]); the domain none that opens
//!   files or reads file formats ([`FILE_CRATES`]);
//! - the domain and the application reach no backend or device crate
//!   ([`DEVICE_CRATES`]) through any chain of normal and build dependencies,
//!   each built alone with all its features on this platform (`cargo tree`),
//!   and take burn's crates without their default features and with no
//!   feature but [`BURN_FEATURES`]. Since burn 0.22 a tensor's type names no
//!   backend: burn's features compile the backends into one tensor type, and
//!   the `Device` a tensor is on picks one at run time. So the infrastructure
//!   and the interfaces pick the backends, as burn's features and the devices
//!   they build, and the domain and the application compute on the devices
//!   given them;
//! - no crate declares a mocking crate ([`MOCK_CRATES`]): test doubles
//!   implement the ports;
//! - only the interfaces and the tools have binaries or examples;
//! - every crate takes the workspace's lints, which allow nothing and deny
//!   [`REQUIRED_LINTS`].
//!
//! Dev-dependencies, which only the tests see, keep the direction and the
//! ban on mocks; the tests of a lower layer may run its code on a backend.
//!
//! On the source of every target ([`check_sources`], [`check_source`]):
//! - every crate root forbids `unsafe_code`, and no attribute allows it but
//!   on the `mod` items of [`UNSAFE_MODULES`] (none yet);
//! - every `allow` or `expect` covers one item and gives its reason: no
//!   inner `#![allow]`, no allow without `reason = "..."`;
//! - no script but Rust: no `.py` or `.sh` file in a crate.
//!
//! And on the source under `src/`:
//! - the domain, the application and the infrastructure name none of the
//!   settings of the whole process in [`PROCESS_SETTINGS`] (environment
//!   writes, the working directory, exit, panic hooks, loggers, threads,
//!   CPU affinity, signals, burn's and cubecl's global configurations): only
//!   the command lines make them;
//! - the domain and the application name nothing of `std::env`,
//!   `std::process`, `std::thread`, `std::net` and `std::os`: they read no
//!   environment variable and start no process or thread. A task that needs
//!   parallel loops in them changes this rule, its check and `RULES.md`
//!   together, and gives its reason;
//! - the domain and the application, outside their `#[cfg(test)]` modules,
//!   build no device ([`DEVICE_CONSTRUCTORS`]) and name no backend
//!   ([`BACKENDS`]);
//! - the domain names nothing of `std::fs`, uses no standard stream and
//!   prints nothing.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

use cargo_metadata::{DependencyKind, Metadata, MetadataCommand};
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;
use syn::visit::Visit;

/// A crate's place in the dependency rule, lowest first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Layer {
    /// Pure computation on tensors and arrays.
    Domain,
    /// The use cases and their ports.
    Application,
    /// The adapters of the ports, and the backends.
    Infrastructure,
    /// The command lines.
    Interfaces,
    /// The port's tools: the harness.
    Tools,
}

impl Layer {
    /// The layer's name, as the messages give it.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Domain => "domain",
            Self::Application => "application",
            Self::Infrastructure => "infrastructure",
            Self::Interfaces => "interfaces",
            Self::Tools => "tools",
        }
    }

    /// Whether a crate of this layer may depend on one of `other`: on the
    /// layers below it, and never on the interfaces or the tools.
    #[must_use]
    pub fn may_depend_on(self, other: Self) -> bool {
        other < self && other < Self::Interfaces
    }

    /// Whether this is the domain or the application, which pick no backend.
    #[must_use]
    pub const fn is_lower(self) -> bool {
        matches!(self, Self::Domain | Self::Application)
    }
}

/// The workspace's crates and their layers.
pub const CRATES: [(&str, Layer); 5] = [
    ("diff-gpt-domain", Layer::Domain),
    ("diff-gpt-application", Layer::Application),
    ("diff-gpt-infrastructure", Layer::Infrastructure),
    ("diff-gpt-interfaces", Layer::Interfaces),
    ("diff-gpt-harness", Layer::Tools),
];

/// Crates that make settings of the whole process: arguments, signals,
/// loggers, CPU affinity. Only the interfaces and the tools declare them.
pub const PROCESS_CRATES: &[&str] = &[
    "argh",
    "clap",
    "core_affinity",
    "ctrlc",
    "daemonize",
    "env_logger",
    "fern",
    "flexi_logger",
    "hwloc",
    "hwlocality",
    "lexopt",
    "log4rs",
    "pico-args",
    "pretty_env_logger",
    "signal-hook",
    "signal-hook-registry",
    "simplelog",
    "tracing-subscriber",
];

/// Crates of the operating system, threads, processes, databases, sockets
/// and file locks, which the domain and the application do not declare.
pub const SYSTEM_CRATES: &[&str] = &[
    "async-std",
    "cargo_metadata",
    "crossbeam",
    "crossbeam-channel",
    "diesel",
    "fd-lock",
    "fs2",
    "fs4",
    "git2",
    "gix",
    "hyper",
    "libc",
    "libloading",
    "mio",
    "nix",
    "num_cpus",
    "procfs",
    "rayon",
    "reqwest",
    "rusqlite",
    "rustix",
    "smol",
    "socket2",
    "sqlx",
    "sysinfo",
    "threadpool",
    "tokio",
    "ureq",
    "winapi",
    "windows-sys",
];

/// Crates that open files or read and write file formats, which the domain
/// does not declare.
pub const FILE_CRATES: &[&str] = &[
    "arrow",
    "burn-dataset",
    "burn-store",
    "bzip2",
    "csv",
    "flate2",
    "fs-err",
    "glob",
    "hdf5",
    "image",
    "memmap2",
    "ndarray-npy",
    "notify",
    "npyz",
    "parquet",
    "polars",
    "safetensors",
    "serde_json",
    "serde_yaml",
    "tar",
    "tempfile",
    "toml",
    "walkdir",
    "xz2",
    "zip",
    "zstd",
];

/// Backend and device crates, which the domain and the application reach
/// through no chain of dependencies.
pub const DEVICE_CRATES: &[&str] = &[
    "accelerate-src",
    "ash",
    "blas-src",
    "burn-autodiff",
    "burn-candle",
    "burn-cpu",
    "burn-cubecl",
    "burn-cuda",
    "burn-flex",
    "burn-fusion",
    "burn-ndarray",
    "burn-remote",
    "burn-rocm",
    "burn-router",
    "burn-tch",
    "burn-wgpu",
    "candle-core",
    "cubecl-cpu",
    "cubecl-cuda",
    "cubecl-hip",
    "cubecl-llvm",
    "cubecl-wgpu",
    "cudarc",
    "netlib-src",
    "openblas-src",
    "tch",
    "torch-sys",
    "wgpu",
    "wgpu-core",
    "wgpu-hal",
];

/// Mocking crates, which no crate declares: test doubles implement the ports.
pub const MOCK_CRATES: &[&str] = &[
    "double",
    "faux",
    "galvanic-mock",
    "httpmock",
    "mock_derive",
    "mockall",
    "mockers",
    "mockito",
    "mocktopus",
    "mry",
    "pseudo",
    "simulacrum",
    "unimock",
    "wiremock",
];

/// Burn's crates of the tensor and nn API.
pub const BURN_CRATES: &[&str] = &[
    "burn",
    "burn-backend",
    "burn-core",
    "burn-nn",
    "burn-optim",
    "burn-std",
    "burn-tensor",
];

/// The features of burn's crates that the domain and the application take.
pub const BURN_FEATURES: &[&str] = &["std"];

/// The constructors of burn's `Device`: each picks a backend, or (`default`,
/// `enumerate`) picks among the compiled ones by the process's environment.
/// The domain and the application build no device.
pub const DEVICE_CONSTRUCTORS: &[&str] = &[
    "capture",
    "cpu",
    "cuda",
    "default",
    "enumerate",
    "enumerate_physical",
    "flex",
    "libtorch",
    "libtorch_cuda",
    "libtorch_mps",
    "libtorch_vulkan",
    "metal",
    "ndarray",
    "rocm",
    "vulkan",
    "webgpu",
    "wgpu",
];

/// The backends, as the names of the types and modules of `burn::backend`
/// start (in any case): `Cuda`, `CudaDevice`, `ndarray`. The domain and the
/// application name none.
pub const BACKENDS: &[&str] = &[
    "capture", "cpu", "cube", "cuda", "flex", "libtorch", "metal", "ndarray", "remote", "rocm",
    "vulkan", "webgpu", "wgpu",
];

/// The modules that may allow `unsafe_code`, as (crate, module): none yet.
/// The root of such a crate denies `unsafe_code`, since a forbid cannot be
/// allowed below it.
pub const UNSAFE_MODULES: &[(&str, &str)] = &[];

/// The lints that the workspace denies (or forbids), as (tool, lint).
pub const REQUIRED_LINTS: &[(&str, &str)] = &[
    ("rust", "unsafe_code"),
    ("rust", "missing_docs"),
    ("clippy", "all"),
    ("clippy", "pedantic"),
    ("clippy", "allow_attributes_without_reason"),
];

/// The settings of the whole process, as runs of path segments: a path
/// that holds one of them names the setting. A one-segment run also names
/// it as a method (`.build_global()`).
pub const PROCESS_SETTINGS: &[&[&str]] = &[
    &["env", "set_var"],
    &["env", "remove_var"],
    &["env", "set_current_dir"],
    &["process", "exit"],
    &["process", "abort"],
    &["panic", "set_hook"],
    &["panic", "take_hook"],
    &["log", "set_logger"],
    &["log", "set_boxed_logger"],
    &["log", "set_max_level"],
    &["set_num_threads"],
    &["set_num_interop_threads"],
    &["sched_setaffinity"],
    &["pthread_setaffinity_np"],
    &["libc", "signal"],
    &["libc", "sigaction"],
    &["libc", "umask"],
    &["libc", "setrlimit"],
    &["BurnConfig", "set"],
    &["BurnConfig", "try_set"],
    &["CubeClRuntimeConfig", "set"],
    &["CubeClRuntimeConfig", "try_set"],
    &["RuntimeConfig", "set"],
    &["RuntimeConfig", "try_set"],
    &["build_global"],
    &["core_affinity"],
    &["ctrlc"],
    &["env_logger"],
    &["signal_hook"],
    &["tracing_subscriber"],
];

/// The kind of a declared dependency.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// `[dependencies]`.
    Normal,
    /// `[build-dependencies]`.
    Build,
    /// `[dev-dependencies]`.
    Dev,
}

/// A dependency that a workspace crate declares in its manifest.
#[derive(Clone, Debug)]
pub struct Declared {
    /// The package's name (not the name it is renamed to).
    pub name: String,
    /// Its kind.
    pub kind: Kind,
    /// Whether it takes the package's default features.
    pub default_features: bool,
    /// The features it asks for.
    pub features: Vec<String>,
}

/// A target of a workspace crate.
#[derive(Clone, Debug)]
pub struct Target {
    /// Its kinds (`lib`, `bin`, `test`, ...).
    pub kinds: Vec<String>,
    /// Its name.
    pub name: String,
    /// The source file of its crate root.
    pub root: PathBuf,
}

/// What the rule reads of a workspace crate.
#[derive(Clone, Debug)]
pub struct Member {
    /// The package's name.
    pub name: String,
    /// The dependencies its manifest declares.
    pub declared: Vec<Declared>,
    /// The packages it reaches through normal and build dependencies, built
    /// alone with all its features: filled for the domain and the application.
    pub reaches: BTreeSet<String>,
    /// Its targets.
    pub targets: Vec<Target>,
    /// Its manifest.
    pub manifest: PathBuf,
}

/// The layer of a workspace crate, if it is one of [`CRATES`].
#[must_use]
pub fn layer_of(name: &str) -> Option<Layer> {
    CRATES
        .iter()
        .find(|(crate_name, _)| *crate_name == name)
        .map(|(_, layer)| *layer)
}

/// Every break of the rule in the workspace whose manifest is in `root`.
///
/// # Errors
///
/// If `cargo metadata` or `cargo tree` fails (both run `--locked`), or a
/// manifest or a source file cannot be read or parsed.
pub fn check_workspace(root: &Path) -> Result<Vec<String>, String> {
    let metadata = MetadataCommand::new()
        .manifest_path(root.join("Cargo.toml"))
        .no_deps()
        .other_options(vec!["--locked".to_owned()])
        .exec()
        .map_err(|error| format!("cargo metadata: {error}"))?;
    let root = metadata.workspace_root.as_std_path();
    let mut members = members(&metadata);
    for member in &mut members {
        if layer_of(&member.name).is_some_and(Layer::is_lower) {
            member.reaches = reached(root, &member.name)?;
        }
    }
    let mut breaks = check_members(&members);
    breaks.extend(check_manifests(root, &members)?);
    breaks.extend(check_sources(root, &members)?);
    Ok(breaks)
}

/// The workspace's crates as `cargo metadata` gives them, with no `reaches`.
#[must_use]
pub fn members(metadata: &Metadata) -> Vec<Member> {
    metadata
        .workspace_packages()
        .into_iter()
        .map(|package| Member {
            name: package.name.to_string(),
            declared: package
                .dependencies
                .iter()
                .map(|dep| Declared {
                    name: dep.name.clone(),
                    kind: match dep.kind {
                        DependencyKind::Build => Kind::Build,
                        DependencyKind::Development => Kind::Dev,
                        _ => Kind::Normal,
                    },
                    default_features: dep.uses_default_features,
                    features: dep.features.clone(),
                })
                .collect(),
            reaches: BTreeSet::new(),
            targets: package
                .targets
                .iter()
                .map(|target| Target {
                    kinds: target.kind.iter().map(ToString::to_string).collect(),
                    name: target.name.clone(),
                    root: target.src_path.as_std_path().to_path_buf(),
                })
                .collect(),
            manifest: package.manifest_path.as_std_path().to_path_buf(),
        })
        .collect()
}

/// The packages that `package` reaches through normal and build
/// dependencies on this platform, built alone with all its features.
fn reached(root: &Path, package: &str) -> Result<BTreeSet<String>, String> {
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let output = Command::new(cargo)
        .args([
            "tree",
            "--locked",
            "--all-features",
            "--edges",
            "normal,build",
            "--prefix",
            "none",
            "--format",
            "{p}",
            "--package",
            package,
        ])
        .current_dir(root)
        .output()
        .map_err(|error| format!("cargo tree: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "cargo tree --package {package}: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter(|line| !line.starts_with('['))
        .filter_map(|line| line.split_whitespace().next())
        .filter(|name| *name != package)
        .map(str::to_owned)
        .collect())
}

/// Every break of the rule in the dependencies and targets of `members`.
#[must_use]
pub fn check_members(members: &[Member]) -> Vec<String> {
    let mut breaks = Vec::new();
    for member in members {
        let Some(layer) = layer_of(&member.name) else {
            breaks.push(format!("{}: a crate outside the layers", member.name));
            continue;
        };
        let say = |what: String| format!("{} ({}): {what}", member.name, layer.name());
        for dep in &member.declared {
            breaks.extend(check_dependency(layer, dep).into_iter().map(say));
        }
        if layer.is_lower() {
            for name in member
                .reaches
                .iter()
                .filter(|name| DEVICE_CRATES.contains(&name.as_str()))
            {
                breaks.push(say(format!("reaches {name}, a backend")));
            }
        }
        if layer < Layer::Interfaces {
            for target in &member.targets {
                if let Some(kind) = target
                    .kinds
                    .iter()
                    .find(|kind| *kind == "bin" || *kind == "example")
                {
                    breaks.push(say(format!("has the {kind} target {}", target.name)));
                }
            }
        }
    }
    breaks
}

fn check_dependency(layer: Layer, dep: &Declared) -> Vec<String> {
    let name = dep.name.as_str();
    let mut breaks = Vec::new();
    if let Some(other) = layer_of(name)
        && !layer.may_depend_on(other)
    {
        breaks.push(format!("depends on {name}, of the {}", other.name()));
    }
    if MOCK_CRATES.contains(&name) {
        breaks.push(format!(
            "declares {name}, a mocking crate: test doubles implement the ports"
        ));
    }
    if dep.kind == Kind::Dev {
        return breaks;
    }
    if layer < Layer::Interfaces && PROCESS_CRATES.contains(&name) {
        breaks.push(format!(
            "declares {name}, which makes settings of the whole process"
        ));
    }
    if layer.is_lower() && SYSTEM_CRATES.contains(&name) {
        breaks.push(format!("declares {name}, of the operating system"));
    }
    if layer == Layer::Domain && FILE_CRATES.contains(&name) {
        breaks.push(format!("declares {name}, which reads or writes files"));
    }
    if layer.is_lower() && DEVICE_CRATES.contains(&name) {
        breaks.push(format!("declares {name}, a backend"));
    }
    if layer.is_lower() && BURN_CRATES.contains(&name) {
        if dep.default_features {
            breaks.push(format!("takes the default features of {name}"));
        }
        for feature in &dep.features {
            if !BURN_FEATURES.contains(&feature.as_str()) {
                breaks.push(format!("takes the feature {feature} of {name}"));
            }
        }
    }
    breaks
}

/// Every break of the rule in the lint tables of the workspace's manifest
/// (in `root`) and of the manifests of `members`.
///
/// # Errors
///
/// If a manifest cannot be read or parsed.
pub fn check_manifests(root: &Path, members: &[Member]) -> Result<Vec<String>, String> {
    let mut breaks = workspace_lint_breaks(&read_toml(&root.join("Cargo.toml"))?);
    for member in members {
        breaks.extend(member_lint_break(
            &member.name,
            &read_toml(&member.manifest)?,
        ));
    }
    Ok(breaks)
}

fn read_toml(path: &Path) -> Result<toml::Table, String> {
    let text =
        std::fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
    text.parse()
        .map_err(|error| format!("{}: {error}", path.display()))
}

fn workspace_lint_breaks(manifest: &toml::Table) -> Vec<String> {
    let Some(tools) = manifest
        .get("workspace")
        .and_then(|workspace| workspace.get("lints"))
        .and_then(toml::Value::as_table)
    else {
        return vec!["Cargo.toml: no [workspace.lints]".to_owned()];
    };
    let level = |tool: &str, lint: &str| {
        let value = tools.get(tool)?.get(lint)?;
        value
            .as_str()
            .or_else(|| value.get("level").and_then(toml::Value::as_str))
    };
    let mut breaks = Vec::new();
    for (tool, lints) in tools {
        for (lint, _) in lints.as_table().into_iter().flatten() {
            if level(tool, lint) == Some("allow") {
                breaks.push(format!(
                    "Cargo.toml: the workspace allows {tool}::{lint}: an allow is local"
                ));
            }
        }
    }
    for (tool, lint) in REQUIRED_LINTS {
        if !matches!(level(tool, lint), Some("deny" | "forbid")) {
            breaks.push(format!(
                "Cargo.toml: the workspace does not deny {tool}::{lint}"
            ));
        }
    }
    breaks
}

fn member_lint_break(name: &str, manifest: &toml::Table) -> Option<String> {
    let inherits = manifest
        .get("lints")
        .and_then(toml::Value::as_table)
        .is_some_and(|lints| {
            lints.len() == 1 && lints.get("workspace").and_then(toml::Value::as_bool) == Some(true)
        });
    (!inherits).then(|| format!("{name}: its [lints] are not just the workspace's"))
}

/// Every break of the rule in the files of `members`, as `path:line: what`,
/// with the paths relative to `root`.
///
/// # Errors
///
/// If a folder or a source file cannot be read, or a file does not parse.
pub fn check_sources(root: &Path, members: &[Member]) -> Result<Vec<String>, String> {
    let mut breaks = Vec::new();
    for member in members {
        let Some(layer) = layer_of(&member.name) else {
            continue;
        };
        let folder = member.manifest.parent().unwrap_or(root);
        let src = folder.join("src");
        let mut files = Vec::new();
        files_under(folder, &mut files)?;
        for file in files {
            let shown = file
                .strip_prefix(root)
                .unwrap_or(&file)
                .display()
                .to_string();
            match file.extension().and_then(|extension| extension.to_str()) {
                Some("rs") => {
                    let text = std::fs::read_to_string(&file)
                        .map_err(|error| format!("{shown}: {error}"))?;
                    let is_root = member.targets.iter().any(|target| target.root == file);
                    let rules = file.starts_with(&src).then_some(layer);
                    breaks.extend(check_source(&member.name, rules, is_root, &shown, &text)?);
                }
                Some("py" | "sh" | "bash") => {
                    breaks.push(format!("{shown}: a script that is not Rust"));
                }
                _ => {}
            }
        }
    }
    Ok(breaks)
}

fn files_under(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    let mut paths = std::fs::read_dir(dir)
        .and_then(|entries| {
            entries
                .map(|entry| entry.map(|entry| entry.path()))
                .collect::<Result<Vec<_>, _>>()
        })
        .map_err(|error| format!("{}: {error}", dir.display()))?;
    paths.sort();
    for path in paths {
        if !path.is_dir() {
            out.push(path);
        } else if path.file_name().is_some_and(|name| name != "target") {
            files_under(&path, out)?;
        }
    }
    Ok(())
}

/// Every break of the rule in one source file of `crate_name`, as
/// `path:line: what`.
///
/// `layer`: the layer whose rules apply to the file's paths (`None` outside
/// `src/`); `root`: whether the file is a crate root.
///
/// # Errors
///
/// If the text does not parse as a Rust file.
pub fn check_source(
    crate_name: &str,
    layer: Option<Layer>,
    root: bool,
    path: &str,
    text: &str,
) -> Result<Vec<String>, String> {
    check_source_with(UNSAFE_MODULES, crate_name, layer, root, path, text)
}

fn check_source_with(
    unsafe_modules: &[(&str, &str)],
    crate_name: &str,
    layer: Option<Layer>,
    root: bool,
    path: &str,
    text: &str,
) -> Result<Vec<String>, String> {
    let file = syn::parse_file(text).map_err(|error| format!("{path}: {error}"))?;
    let mut scan = Scan {
        unsafe_modules,
        crate_name,
        layer,
        in_test: false,
        breaks: Vec::new(),
    };
    let named = unsafe_modules.iter().any(|(name, _)| *name == crate_name);
    if root && !named && !file.attrs.iter().any(forbids_unsafe) {
        scan.breaks
            .push((1, "the crate root does not forbid unsafe_code".to_owned()));
    }
    scan.visit_file(&file);
    let mut breaks = scan.breaks;
    breaks.sort();
    breaks.dedup();
    Ok(breaks
        .into_iter()
        .map(|(line, what)| format!("{path}:{line}: {what}"))
        .collect())
}

fn is_cfg_test(attr: &syn::Attribute) -> bool {
    attr.path().is_ident("cfg")
        && attr
            .parse_args::<syn::Ident>()
            .is_ok_and(|ident| ident == "test")
}

fn forbids_unsafe(attr: &syn::Attribute) -> bool {
    matches!(attr.style, syn::AttrStyle::Inner(_))
        && attr.path().is_ident("forbid")
        && lints(attr).0.iter().any(|name| name == "unsafe_code")
}

/// The lints that a lint attribute names, and whether it gives a reason.
fn lints(attr: &syn::Attribute) -> (Vec<String>, bool) {
    let Ok(metas) = attr.parse_args_with(Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated)
    else {
        return (Vec::new(), false);
    };
    let mut names = Vec::new();
    let mut reason = false;
    for meta in metas {
        match meta {
            syn::Meta::Path(path) => names.push(path_text(&path)),
            syn::Meta::NameValue(pair) if pair.path.is_ident("reason") => reason = true,
            _ => {}
        }
    }
    (names, reason)
}

fn path_text(path: &syn::Path) -> String {
    path.segments
        .iter()
        .map(|segment| segment.ident.to_string())
        .collect::<Vec<_>>()
        .join("::")
}

/// Whether `segments` holds `run` as consecutive segments.
fn holds(segments: &[String], run: &[&str]) -> bool {
    segments.windows(run.len()).any(|window| {
        window
            .iter()
            .zip(run)
            .all(|(segment, name)| segment == name)
    })
}

fn use_paths(tree: &syn::UseTree, prefix: &mut Vec<String>, out: &mut Vec<Vec<String>>) {
    let leaf = |prefix: &[String], last: String| {
        let mut path = prefix.to_vec();
        path.push(last);
        path
    };
    match tree {
        syn::UseTree::Path(path) => {
            prefix.push(path.ident.to_string());
            use_paths(&path.tree, prefix, out);
            prefix.pop();
        }
        syn::UseTree::Name(name) => out.push(leaf(prefix, name.ident.to_string())),
        syn::UseTree::Rename(rename) => out.push(leaf(prefix, rename.ident.to_string())),
        syn::UseTree::Glob(_) => out.push(leaf(prefix, "*".to_owned())),
        syn::UseTree::Group(group) => {
            for item in &group.items {
                use_paths(item, prefix, out);
            }
        }
    }
}

struct Scan<'a> {
    unsafe_modules: &'a [(&'a str, &'a str)],
    crate_name: &'a str,
    layer: Option<Layer>,
    /// Whether the scan is inside a `#[cfg(test)]` module.
    in_test: bool,
    breaks: Vec<(usize, String)>,
}

impl Scan<'_> {
    fn add(&mut self, span: proc_macro2::Span, what: String) {
        self.breaks.push((span.start().line, what));
    }

    fn attribute(&mut self, attr: &syn::Attribute, module: Option<&str>) {
        if !(attr.path().is_ident("allow") || attr.path().is_ident("expect")) {
            return;
        }
        let (names, reason) = lints(attr);
        if matches!(attr.style, syn::AttrStyle::Inner(_)) {
            self.add(
                attr.span(),
                "an inner allow: an allow covers the one item it is on".to_owned(),
            );
        }
        if !reason {
            self.add(attr.span(), "an allow without a reason".to_owned());
        }
        let in_named_module = module.is_some_and(|module| {
            self.unsafe_modules
                .iter()
                .any(|(name, path)| *name == self.crate_name && *path == module)
        });
        if !in_named_module && names.iter().any(|name| name == "unsafe_code") {
            self.add(
                attr.span(),
                "allows unsafe_code outside the named modules".to_owned(),
            );
        }
    }

    fn path(&mut self, segments: &[String], span: proc_macro2::Span) {
        let Some(layer) = self.layer else {
            return;
        };
        let text = segments.join("::");
        if layer < Layer::Interfaces
            && PROCESS_SETTINGS
                .iter()
                .any(|setting| holds(segments, setting))
        {
            self.add(
                span,
                format!("the {} sets the process: {text}", layer.name()),
            );
        }
        let std_module = |name: &str| {
            segments.len() >= 2
                && matches!(segments[0].as_str(), "std" | "core" | "alloc")
                && segments[1] == name
        };
        if layer.is_lower() {
            for module in ["env", "process", "thread", "net", "os"] {
                if std_module(module) {
                    self.add(
                        span,
                        format!("the {} uses std::{module}: {text}", layer.name()),
                    );
                }
            }
        }
        if layer.is_lower() && !self.in_test {
            let pairs = || segments.windows(2).map(|pair| (&pair[0], pair[1].as_str()));
            if pairs().any(|(type_name, constructor)| {
                type_name == "Device" && DEVICE_CONSTRUCTORS.contains(&constructor)
            }) {
                self.add(
                    span,
                    format!("the {} builds a device: {text}", layer.name()),
                );
            }
            if pairs().any(|(module, name)| {
                let name = name.to_lowercase();
                module == "backend" && BACKENDS.iter().any(|backend| name.starts_with(backend))
            }) {
                self.add(
                    span,
                    format!("the {} names a backend: {text}", layer.name()),
                );
            }
        }
        if layer == Layer::Domain {
            if std_module("fs") {
                self.add(span, format!("the domain uses std::fs: {text}"));
            }
            let stream = segments.get(2).is_some_and(|name| {
                matches!(name.to_lowercase().as_str(), "stdin" | "stdout" | "stderr")
            });
            if std_module("io") && stream {
                self.add(span, format!("the domain uses a standard stream: {text}"));
            }
        }
    }

    /// Scan the arguments of a macro that takes expressions, like
    /// `println!` or `assert_eq!`.
    fn macro_arguments(&mut self, node: &syn::Macro) {
        if let Ok(arguments) =
            node.parse_body_with(Punctuated::<syn::Expr, syn::Token![,]>::parse_terminated)
        {
            for argument in &arguments {
                Visit::visit_expr(self, argument);
            }
        }
    }
}

impl<'ast> Visit<'ast> for Scan<'_> {
    fn visit_attribute(&mut self, node: &'ast syn::Attribute) {
        self.attribute(node, None);
    }

    fn visit_item_mod(&mut self, node: &'ast syn::ItemMod) {
        let module = node.ident.to_string();
        for attr in &node.attrs {
            self.attribute(attr, Some(&module));
        }
        let outer = self.in_test;
        self.in_test |= node.attrs.iter().any(is_cfg_test);
        for item in node.content.iter().flat_map(|(_, items)| items) {
            self.visit_item(item);
        }
        self.in_test = outer;
    }

    fn visit_item_use(&mut self, node: &'ast syn::ItemUse) {
        for attr in &node.attrs {
            self.attribute(attr, None);
        }
        let mut paths = Vec::new();
        use_paths(&node.tree, &mut Vec::new(), &mut paths);
        for path in paths {
            self.path(&path, node.span());
        }
    }

    fn visit_path(&mut self, node: &'ast syn::Path) {
        let segments: Vec<String> = node
            .segments
            .iter()
            .map(|segment| segment.ident.to_string())
            .collect();
        self.path(&segments, node.span());
        syn::visit::visit_path(self, node);
    }

    fn visit_expr_method_call(&mut self, node: &'ast syn::ExprMethodCall) {
        let method = node.method.to_string();
        if let Some(layer) = self.layer
            && layer < Layer::Interfaces
            && PROCESS_SETTINGS
                .iter()
                .any(|setting| *setting == [method.as_str()])
        {
            self.add(
                node.method.span(),
                format!("the {} sets the process: .{method}()", layer.name()),
            );
        }
        syn::visit::visit_expr_method_call(self, node);
    }

    fn visit_macro(&mut self, node: &'ast syn::Macro) {
        if self.layer == Some(Layer::Domain)
            && let Some(last) = node.path.segments.last()
        {
            let name = last.ident.to_string();
            if matches!(
                name.as_str(),
                "print" | "println" | "eprint" | "eprintln" | "dbg"
            ) {
                self.add(node.path.span(), format!("the domain prints: {name}!"));
            }
        }
        self.macro_arguments(node);
        syn::visit::visit_macro(self, node);
    }
}

/// The breaks of the rule as one message, or `None` when there are none.
#[must_use]
pub fn report(breaks: &[String]) -> Option<String> {
    if breaks.is_empty() {
        return None;
    }
    let mut message = format!("{} breaks of the layer rule:", breaks.len());
    for line in breaks {
        message.push_str("\n  ");
        message.push_str(line);
    }
    Some(message)
}

#[cfg(test)]
mod tests {
    use super::{
        Declared, Kind, Layer, Member, Target, check_members, check_source, check_source_with,
        member_lint_break, workspace_lint_breaks,
    };
    use std::collections::BTreeSet;
    use std::path::PathBuf;

    fn dep(name: &str) -> Declared {
        let burn = name.starts_with("burn");
        Declared {
            name: name.to_owned(),
            kind: Kind::Normal,
            default_features: !burn,
            features: if burn {
                vec!["std".to_owned()]
            } else {
                Vec::new()
            },
        }
    }

    fn dev(name: &str) -> Declared {
        Declared {
            kind: Kind::Dev,
            ..dep(name)
        }
    }

    fn member(name: &str, deps: &[&str]) -> Member {
        Member {
            name: name.to_owned(),
            declared: deps.iter().copied().map(dep).collect(),
            reaches: BTreeSet::new(),
            targets: vec![Target {
                kinds: vec!["lib".to_owned()],
                name: name.replace('-', "_"),
                root: PathBuf::new(),
            }],
            manifest: PathBuf::new(),
        }
    }

    /// The workspace as it is meant to be, with a crate of each kind that
    /// each layer may declare.
    fn clean() -> Vec<Member> {
        let mut members = vec![
            member("diff-gpt-domain", &["burn"]),
            member("diff-gpt-application", &["diff-gpt-domain", "burn"]),
            member(
                "diff-gpt-infrastructure",
                &[
                    "diff-gpt-domain",
                    "diff-gpt-application",
                    "burn",
                    "rusqlite",
                    "serde_json",
                ],
            ),
            member(
                "diff-gpt-interfaces",
                &["diff-gpt-infrastructure", "clap", "ctrlc"],
            ),
            member(
                "diff-gpt-harness",
                &["diff-gpt-infrastructure", "cargo_metadata", "toml"],
            ),
        ];
        members[0].declared.push(Declared {
            features: vec!["flex".to_owned()],
            ..dev("burn")
        });
        members[0].reaches = ["burn-core", "burn-dispatch", "burn-tensor"]
            .map(str::to_owned)
            .into();
        members[2].declared[2]
            .features
            .extend(["autodiff", "cuda", "tch"].map(str::to_owned));
        members[3].targets[0].kinds = vec!["bin".to_owned()];
        members
    }

    fn breaks_after(edit: impl FnOnce(&mut Vec<Member>)) -> Vec<String> {
        let mut members = clean();
        edit(&mut members);
        check_members(&members)
    }

    #[test]
    fn the_clean_workspace_keeps_the_rule() {
        assert_eq!(check_members(&clean()), Vec::<String>::new());
    }

    #[test]
    fn a_dependency_on_a_layer_above_breaks_it() {
        let breaks = breaks_after(|members| {
            members[0].declared.push(dev("diff-gpt-infrastructure"));
            members[1].declared.push(dep("diff-gpt-application"));
            members[2].declared.push(dep("diff-gpt-harness"));
            members[4].declared.push(dep("diff-gpt-interfaces"));
        });
        assert_eq!(
            breaks,
            [
                "diff-gpt-domain (domain): depends on diff-gpt-infrastructure, of the \
                 infrastructure",
                "diff-gpt-application (application): depends on diff-gpt-application, of the \
                 application",
                "diff-gpt-infrastructure (infrastructure): depends on diff-gpt-harness, of the \
                 tools",
                "diff-gpt-harness (tools): depends on diff-gpt-interfaces, of the interfaces",
            ]
        );
    }

    #[test]
    fn io_process_and_device_crates_break_it_below() {
        let breaks = breaks_after(|members| {
            members[0].declared.push(dep("serde_json"));
            members[0].declared.push(dep("burn-ndarray"));
            members[1].declared.push(dep("clap"));
            members[1].declared.push(dep("rusqlite"));
            members[2].declared.push(dep("ctrlc"));
        });
        assert_eq!(
            breaks,
            [
                "diff-gpt-domain (domain): declares serde_json, which reads or writes files",
                "diff-gpt-domain (domain): declares burn-ndarray, a backend",
                "diff-gpt-application (application): declares clap, which makes settings of \
                 the whole process",
                "diff-gpt-application (application): declares rusqlite, of the operating system",
                "diff-gpt-infrastructure (infrastructure): declares ctrlc, which makes settings \
                 of the whole process",
            ]
        );
    }

    #[test]
    fn a_backend_reached_or_a_feature_of_burn_breaks_it() {
        let breaks = breaks_after(|members| {
            members[0].declared[0].features.push("ndarray".to_owned());
            members[1].declared[1].default_features = true;
            members[1].reaches.insert("burn-tch".to_owned());
        });
        assert_eq!(
            breaks,
            [
                "diff-gpt-domain (domain): takes the feature ndarray of burn",
                "diff-gpt-application (application): takes the default features of burn",
                "diff-gpt-application (application): reaches burn-tch, a backend",
            ]
        );
    }

    #[test]
    fn mocks_binaries_and_strangers_break_it() {
        let breaks = breaks_after(|members| {
            members[2].targets.push(Target {
                kinds: vec!["bin".to_owned()],
                name: "tool".to_owned(),
                root: PathBuf::new(),
            });
            members[3].declared.push(dev("mockall"));
            members.push(member("diff-gpt-extra", &[]));
        });
        assert_eq!(
            breaks,
            [
                "diff-gpt-infrastructure (infrastructure): has the bin target tool",
                "diff-gpt-interfaces (interfaces): declares mockall, a mocking crate: test \
                 doubles implement the ports",
                "diff-gpt-extra: a crate outside the layers",
            ]
        );
    }

    #[test]
    fn the_lints_are_the_workspace_s_and_allow_nothing() {
        let workspace: toml::Table = "[workspace.lints.rust]\nunsafe_code = \"deny\"\n\
             missing_docs = \"deny\"\n[workspace.lints.clippy]\n\
             all = { level = \"deny\", priority = -1 }\n\
             pedantic = { level = \"deny\", priority = -1 }\n\
             allow_attributes_without_reason = \"deny\"\n"
            .parse()
            .expect("the table parses");
        assert_eq!(workspace_lint_breaks(&workspace), Vec::<String>::new());
        let loose: toml::Table = "[workspace.lints.clippy]\n\
             pedantic = { level = \"allow\", priority = -1 }\n"
            .parse()
            .expect("the table parses");
        assert_eq!(
            workspace_lint_breaks(&loose),
            [
                "Cargo.toml: the workspace allows clippy::pedantic: an allow is local",
                "Cargo.toml: the workspace does not deny rust::unsafe_code",
                "Cargo.toml: the workspace does not deny rust::missing_docs",
                "Cargo.toml: the workspace does not deny clippy::all",
                "Cargo.toml: the workspace does not deny clippy::pedantic",
                "Cargo.toml: the workspace does not deny clippy::allow_attributes_without_reason",
            ]
        );
        let inherits: toml::Table = "[lints]\nworkspace = true\n".parse().expect("parses");
        assert_eq!(member_lint_break("x", &inherits), None);
        let own: toml::Table = "[lints]\nworkspace = true\n[lints.clippy]\nall = \"allow\"\n"
            .parse()
            .expect("parses");
        assert_eq!(
            member_lint_break("x", &own).as_deref(),
            Some("x: its [lints] are not just the workspace's")
        );
    }

    fn scan(layer: Option<Layer>, root: bool, text: &str) -> Vec<String> {
        check_source("diff-gpt-x", layer, root, "x.rs", text).expect("the snippet parses")
    }

    #[test]
    fn a_clean_source_keeps_the_rule() {
        let text = "//! Docs.\n#![forbid(unsafe_code)]\nuse std::collections::BTreeMap;\n\n\
                    /// A count.\n\
                    #[expect(clippy::cast_precision_loss, reason = \"the counts are small\")]\n\
                    pub fn f(n: usize) -> f64 {\n    let _ = BTreeMap::<u8, u8>::new();\n    \
                    n as f64\n}\n";
        assert_eq!(scan(Some(Layer::Domain), true, text), Vec::<String>::new());
    }

    #[test]
    fn unsafe_and_loose_allows_break_it() {
        assert_eq!(
            scan(
                Some(Layer::Domain),
                true,
                "//! Docs.\n#![allow(clippy::pedantic)]\n"
            ),
            [
                "x.rs:1: the crate root does not forbid unsafe_code",
                "x.rs:2: an allow without a reason",
                "x.rs:2: an inner allow: an allow covers the one item it is on",
            ]
        );
        let text = "#[allow(unsafe_code, reason = \"FFI\")]\nmod ffi {}\n\
                    #[allow(unsafe_code, reason = \"FFI\")]\nfn f() {}\n";
        assert_eq!(
            scan(None, false, text),
            [
                "x.rs:1: allows unsafe_code outside the named modules",
                "x.rs:3: allows unsafe_code outside the named modules",
            ]
        );
        let named = format!("//! Docs.\n#![deny(unsafe_code)]\n{text}");
        assert_eq!(
            check_source_with(
                &[("diff-gpt-x", "ffi")],
                "diff-gpt-x",
                None,
                true,
                "x.rs",
                &named
            )
            .expect("the snippet parses"),
            ["x.rs:5: allows unsafe_code outside the named modules"]
        );
    }

    #[test]
    fn process_settings_break_it_below_the_interfaces() {
        let text = "fn f() {\n    unsafe { std::env::set_var(\"A\", \"1\") };\n    \
                    rayon::ThreadPoolBuilder::new().num_threads(2).build_global().ok();\n    \
                    tch::set_num_threads(4);\n    \
                    burn::BurnConfig::set(burn::BurnConfig::default());\n    \
                    std::process::exit(1);\n}\n";
        assert_eq!(
            scan(Some(Layer::Infrastructure), false, text),
            [
                "x.rs:2: the infrastructure sets the process: std::env::set_var",
                "x.rs:3: the infrastructure sets the process: .build_global()",
                "x.rs:4: the infrastructure sets the process: tch::set_num_threads",
                "x.rs:5: the infrastructure sets the process: burn::BurnConfig::set",
                "x.rs:6: the infrastructure sets the process: std::process::exit",
            ]
        );
        assert_eq!(
            scan(Some(Layer::Interfaces), false, text),
            Vec::<String>::new()
        );
        assert_eq!(scan(None, false, text), Vec::<String>::new());
    }

    #[test]
    fn the_environment_processes_files_and_streams_break_it_below() {
        let text = "use std::{env, fs::File};\nfn f() -> String {\n    \
                    println!(\"{}\", std::env::var(\"B\").unwrap());\n    \
                    std::process::Command::new(\"git\");\n    \
                    let _ = std::io::stdout();\n    env::var(\"A\").unwrap()\n}\n";
        assert_eq!(
            scan(Some(Layer::Domain), false, text),
            [
                "x.rs:1: the domain uses std::env: std::env",
                "x.rs:1: the domain uses std::fs: std::fs::File",
                "x.rs:3: the domain prints: println!",
                "x.rs:3: the domain uses std::env: std::env::var",
                "x.rs:4: the domain uses std::process: std::process::Command::new",
                "x.rs:5: the domain uses a standard stream: std::io::stdout",
            ]
        );
        assert_eq!(
            scan(Some(Layer::Application), false, text),
            [
                "x.rs:1: the application uses std::env: std::env",
                "x.rs:3: the application uses std::env: std::env::var",
                "x.rs:4: the application uses std::process: std::process::Command::new",
            ]
        );
        assert_eq!(
            scan(Some(Layer::Infrastructure), false, text),
            Vec::<String>::new()
        );
    }

    #[test]
    fn a_thread_started_breaks_it_below() {
        let text = "use std::thread::JoinHandle;\n\
                    pub fn f() { std::thread::spawn(|| ()); }\n";
        assert_eq!(
            scan(Some(Layer::Domain), false, text),
            [
                "x.rs:1: the domain uses std::thread: std::thread::JoinHandle",
                "x.rs:2: the domain uses std::thread: std::thread::spawn",
            ]
        );
        assert_eq!(
            scan(Some(Layer::Application), false, text),
            [
                "x.rs:1: the application uses std::thread: std::thread::JoinHandle",
                "x.rs:2: the application uses std::thread: std::thread::spawn",
            ]
        );
        assert_eq!(
            scan(Some(Layer::Infrastructure), false, text),
            Vec::<String>::new()
        );
    }

    #[test]
    fn a_device_built_or_a_backend_named_breaks_it_below() {
        let text = "use burn::backend::{Backend, Cuda, ndarray::NdArrayDevice};\n\
                    use burn::tensor::Device;\nfn f() -> Device {\n    Device::cuda(0)\n}\n\
                    fn g(device: &Device) -> Device {\n    device.clone().autodiff()\n}\n\
                    #[cfg(test)]\nmod tests {\n    fn h() -> burn::tensor::Device {\n        \
                    burn::tensor::Device::flex()\n    }\n}\n";
        assert_eq!(
            scan(Some(Layer::Domain), false, text),
            [
                "x.rs:1: the domain names a backend: burn::backend::Cuda",
                "x.rs:1: the domain names a backend: burn::backend::ndarray::NdArrayDevice",
                "x.rs:4: the domain builds a device: Device::cuda",
            ]
        );
        assert_eq!(
            scan(
                Some(Layer::Application),
                false,
                "fn f() {\n    let _ = Device::default();\n}\n"
            ),
            ["x.rs:2: the application builds a device: Device::default"]
        );
        assert_eq!(
            scan(Some(Layer::Infrastructure), false, text),
            Vec::<String>::new()
        );
    }
}
