//! The port map, `port_map.toml` beside the harness's manifest: where each
//! part of the frozen Python reference goes in the workspace, which port
//! task brings it over and how its outputs are checked against the
//! reference's.
//!
//! [`read`] reads the map, and [`check`] finds what it misses or gets wrong
//! against the files of the reference's commit ([`reference_files`]):
//! - every module of the reference's package is in `[modules]` and every
//!   other file of it in `[data]`, every test file of its tests in `[tests]`
//!   and every other file of them in `[test_support]`; the map names no file
//!   that the commit does not hold;
//! - the test files hold the reference's `pytest_ids` ids, none twice in a
//!   file; each key of a `drop` or an `overrides` table names ids of its
//!   file, and no id is both dropped and moved; every slow id is one of its
//!   file's, and every reason to drop one is given;
//! - the tasks are those of [`TASKS`], each with its entries, and every
//!   entry names one of them; every check is one of [`CHECKS`], and names a
//!   tolerance or a distribution that the map defines; every crate is one of
//!   [`CRATES`]; every Rust module, test file, test and command is well
//!   formed, and no two ids take the same Rust test nor two modules the same
//!   command.
//!
//! [`check_functions`] holds the ids of a test file against its source,
//! [`fate`] gives the Rust test of an id or the reason it is dropped,
//! [`summary`] what each task ports, and [`tempopfn_walk`] the modules of
//! `TempoPFN` that the corpus imports, by a walk of the import statements.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;
use std::ops::RangeInclusive;
use std::path::Path;
use std::process::Command;

use serde::Deserialize;

/// The map's file, beside the harness's manifest.
pub const FILE: &str = "port_map.toml";

/// The tasks of the port on the board, the first to the last.
pub const TASKS: RangeInclusive<u32> = 27..=36;

/// The crates an entry may name: the members of the workspace by their
/// directories under `crates/`, and the workspace itself.
pub const CRATES: &[&str] = &[
    "domain",
    "application",
    "infrastructure",
    "interfaces",
    "harness",
    "workspace",
];

/// The kinds of check, the keys of `[checks]`: `exact`, `none`, `ulp:N`,
/// `tolerance:NAME` and `distribution:NAME`.
pub const CHECKS: &[&str] = &["distribution", "exact", "none", "tolerance", "ulp"];

/// The words Rust keeps, which no module or test may be named.
const KEYWORDS: &[&str] = &[
    "abstract", "as", "async", "await", "become", "box", "break", "const", "continue", "crate",
    "do", "dyn", "else", "enum", "extern", "false", "final", "fn", "for", "gen", "if", "impl",
    "in", "let", "loop", "macro", "match", "mod", "move", "mut", "override", "priv", "pub", "ref",
    "return", "self", "static", "struct", "super", "trait", "true", "try", "type", "typeof",
    "unsafe", "unsized", "use", "virtual", "where", "while", "yield",
];

/// The name under which the reference imports `TempoPFN`'s modules, the
/// stand-ins' package; `TempoPFN` names them `src.…`.
const TEMPOPFN_ALIAS: &str = "diff_gpt.infrastructure.corpus.tempopfn";

/// The port map.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Map {
    /// The reference: its commit, its package and its tests.
    pub reference: Reference,
    /// The tasks of the port by their numbers on the board, with their
    /// titles.
    pub tasks: BTreeMap<String, String>,
    /// The kinds of check, each with what it means.
    pub checks: BTreeMap<String, String>,
    /// The tolerances that `tolerance:NAME` names.
    pub tolerances: BTreeMap<String, Tolerance>,
    /// The statistical tests that `distribution:NAME` names.
    pub distributions: BTreeMap<String, String>,
    /// The modules of the reference's package, by their paths in it.
    pub modules: BTreeMap<String, Module>,
    /// The other files of the reference's package, by their paths in it.
    pub data: BTreeMap<String, Data>,
    /// The modules of `TempoPFN` that the corpus imports.
    pub tempopfn: TempoPfn,
    /// The Rust modules with no module of the reference, by their paths
    /// (`crate::module`).
    pub rust_only: BTreeMap<String, RustOnly>,
    /// The other files of the reference's tests, by their paths in the
    /// repository.
    pub test_support: BTreeMap<String, Support>,
    /// The test files of the reference, by their paths in the repository.
    pub tests: BTreeMap<String, TestFile>,
}

/// The reference: the commit of the frozen Python code that the port
/// follows.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Reference {
    /// The commit.
    pub commit: String,
    /// Its package's directory in the repository.
    pub package: String,
    /// Its tests' directory in the repository.
    pub tests: String,
    /// The number of ids that pytest collects from its tests.
    pub pytest_ids: usize,
}

/// A tolerance: `|rust - reference| <= atol + rtol * |reference|`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tolerance {
    /// The relative part.
    pub rtol: f64,
    /// The absolute part.
    pub atol: f64,
    /// Where it applies, and why it is this wide.
    pub note: String,
}

/// Where a module of the reference goes.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Module {
    /// The crate, one of [`CRATES`].
    #[serde(rename = "crate")]
    pub krate: String,
    /// The module in the crate (`lib` for the crate's root).
    pub rust: String,
    /// The port task that brings it over.
    pub task: u32,
    /// The check of its outputs.
    pub check: String,
    /// The check on the backends other than `LibTorch`.
    pub others: Option<String>,
    /// The subcommand of the `diff-gpt` binary that a command line becomes.
    pub command: Option<String>,
    /// The task that ports the rest of it.
    pub also: Option<u32>,
    /// What the entry needs said.
    pub note: Option<String>,
}

/// Where another file of the reference's package goes.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Data {
    /// The crate, one of [`CRATES`].
    #[serde(rename = "crate")]
    pub krate: String,
    /// The file's path in the repository.
    pub path: String,
    /// The port task that brings it over.
    pub task: u32,
    /// The check of the file.
    pub check: String,
    /// The task that ports the rest of it.
    pub also: Option<u32>,
    /// What the entry needs said.
    pub note: Option<String>,
}

/// The modules of the `TempoPFN` clone that the corpus imports.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TempoPfn {
    /// The clone's commit.
    pub commit: String,
    /// The modules, named as `TempoPFN` names them (`src.…`).
    pub modules: BTreeMap<String, Module>,
}

/// A Rust module with no module of the reference.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RustOnly {
    /// The port task that brings it.
    pub task: u32,
    /// The check of its outputs.
    pub check: String,
    /// What it is for.
    pub note: String,
}

/// Where another file of the reference's tests goes.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Support {
    /// The Rust file that takes its place, if one does.
    pub rust: Option<String>,
    /// The port task that brings it over.
    pub task: u32,
    /// What becomes of it.
    pub note: Option<String>,
}

/// Where the ids of a test file go.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TestFile {
    /// The Rust test file of its ids, `crates/<crate>/tests/<name>.rs`.
    pub rust: String,
    /// The port task that brings them over.
    pub task: u32,
    /// The kind of comparison their assertions make.
    pub check: String,
    /// The ids that pytest collects from the file, without the file.
    pub ids: Vec<String>,
    /// The ids that pytest marks slow.
    #[serde(default)]
    pub slow: Vec<String>,
    /// The ids, or the test functions, dropped, with the reason of each.
    #[serde(default)]
    pub drop: BTreeMap<String, String>,
    /// The ids, or the test functions, that go elsewhere.
    #[serde(default)]
    pub overrides: BTreeMap<String, Override>,
}

/// Where some ids of a test file go instead of where the file's go.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Override {
    /// Their Rust test, `<Rust test file>::<test>`.
    pub rust: Option<String>,
    /// Their port task.
    pub task: Option<u32>,
    /// The kind of comparison their assertions make.
    pub check: Option<String>,
    /// What the entry needs said.
    pub note: Option<String>,
}

/// What becomes of a pytest id.
#[derive(Debug, PartialEq, Eq)]
pub enum Fate<'a> {
    /// Ported to a Rust test.
    Ported {
        /// The Rust test, `<Rust test file>::<test>`.
        test: String,
        /// The case of the test, when pytest parametrizes it: the label
        /// that pytest gives it.
        case: Option<&'a str>,
        /// The port task that brings it over.
        task: u32,
        /// The kind of comparison its assertions make.
        check: &'a str,
        /// Whether the map moves it out of its file's Rust test file.
        moved: bool,
    },
    /// Dropped, for the reason given.
    Dropped {
        /// The reason.
        reason: &'a str,
        /// The port task of its file.
        task: u32,
    },
}

/// What a task ports, counted.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Share {
    /// The modules of the reference's package that it brings over.
    pub modules: usize,
    /// The modules and other files of the package of which it brings over
    /// the rest (`also`).
    pub rest_of: usize,
    /// The package's other files.
    pub data: usize,
    /// The modules of `TempoPFN`.
    pub tempopfn: usize,
    /// The Rust modules with no module of the reference.
    pub rust_only: usize,
    /// The other files of the reference's tests.
    pub test_support: usize,
    /// The pytest ids ported.
    pub ported: usize,
    /// The pytest ids dropped.
    pub dropped: usize,
}

impl fmt::Display for Share {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{} modules (and the rest of {}), {} data files, {} of TempoPFN, {} new in Rust, \
             {} test support files; {} pytest ids ported, {} dropped",
            self.modules,
            self.rest_of,
            self.data,
            self.tempopfn,
            self.rust_only,
            self.test_support,
            self.ported,
            self.dropped
        )
    }
}

/// Where a module of [`tempopfn_walk`] comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Origin {
    /// One of the reference's stand-ins, in place of `TempoPFN`'s.
    StandIn,
    /// The clone of `TempoPFN`.
    Clone,
}

/// Read the map at `path`.
///
/// # Errors
///
/// When the file does not read or does not hold a map.
pub fn read(path: &Path) -> Result<Map, String> {
    let text =
        std::fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
    toml::from_str(&text).map_err(|error| format!("{}: {error}", path.display()))
}

/// The files that the commit `commit` of the repository at `root` holds
/// under `dir`, by their paths in the repository.
///
/// # Errors
///
/// When git does not run or does not know the commit.
pub fn reference_files(root: &Path, commit: &str, dir: &str) -> Result<Vec<String>, String> {
    let listing = git(root, &["ls-tree", "-r", "--name-only", commit, "--", dir])?;
    Ok(listing.lines().map(str::to_owned).collect())
}

/// The text of the file that the commit `commit` of the repository at
/// `root` holds at `path`, or `None` when it holds none there.
///
/// # Errors
///
/// When git does not run, or the file is no text.
pub fn reference_text(root: &Path, commit: &str, path: &str) -> Result<Option<String>, String> {
    let object = format!("{commit}:{path}");
    let kind = Command::new("git")
        .args(["cat-file", "-t", &object])
        .current_dir(root)
        .output()
        .map_err(|error| format!("git cat-file -t {object}: {error}"))?;
    if !kind.status.success() || kind.stdout.trim_ascii() != b"blob" {
        return Ok(None);
    }
    git(root, &["show", &object]).map(Some)
}

/// The commit that the repository at `root` has checked out.
///
/// # Errors
///
/// When git does not run or `root` is no repository.
pub fn head(root: &Path) -> Result<String, String> {
    Ok(git(root, &["rev-parse", "HEAD"])?.trim().to_owned())
}

/// Run git in `root` and give what it prints.
fn git(root: &Path, args: &[&str]) -> Result<String, String> {
    let output = Command::new("git")
        .args(args)
        .current_dir(root)
        .output()
        .map_err(|error| format!("git {}: {error}", args.join(" ")))?;
    if !output.status.success() {
        return Err(format!(
            "git {}: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    String::from_utf8(output.stdout).map_err(|error| format!("git {}: {error}", args.join(" ")))
}

/// The fate of the id `id` of the test file `file`: dropped by its `drop`
/// entry, moved by its `overrides` entry, or else ported to the test of
/// the same name in the file's Rust test file: `test_NAME[CASE]` to the
/// test NAME, in lower case, and its case CASE.
#[must_use]
pub fn fate<'a>(file: &'a TestFile, id: &'a str) -> Fate<'a> {
    let (function, case) = split_id(id);
    let names = |key: &str| key == id || key == function;
    if let Some((_, reason)) = file.drop.iter().find(|(key, _)| names(key)) {
        return Fate::Dropped {
            reason,
            task: file.task,
        };
    }
    let moved = file
        .overrides
        .iter()
        .find(|(key, _)| names(key))
        .map(|(_, entry)| entry);
    let test = moved
        .and_then(|entry| entry.rust.clone())
        .unwrap_or_else(|| format!("{}::{}", file.rust, rust_name(function)));
    Fate::Ported {
        test,
        case,
        task: moved.and_then(|entry| entry.task).unwrap_or(file.task),
        check: moved
            .and_then(|entry| entry.check.as_deref())
            .unwrap_or(&file.check),
        moved: moved.is_some_and(|entry| entry.rust.is_some()),
    }
}

/// An id's test function and its case: `test_x[a-b]` is `test_x` and `a-b`.
fn split_id(id: &str) -> (&str, Option<&str>) {
    match id.split_once('[') {
        Some((function, case)) => (function, Some(case.strip_suffix(']').unwrap_or(case))),
        None => (id, None),
    }
}

/// The Rust name of a test function: without `test_`, in lower case.
fn rust_name(function: &str) -> String {
    function
        .strip_prefix("test_")
        .unwrap_or(function)
        .to_lowercase()
}

/// What each task ports.
#[must_use]
pub fn summary(map: &Map) -> BTreeMap<u32, Share> {
    let mut shares: BTreeMap<u32, Share> = TASKS.map(|task| (task, Share::default())).collect();
    for module in map.modules.values() {
        shares.entry(module.task).or_default().modules += 1;
        if let Some(task) = module.also {
            shares.entry(task).or_default().rest_of += 1;
        }
    }
    for data in map.data.values() {
        shares.entry(data.task).or_default().data += 1;
        if let Some(task) = data.also {
            shares.entry(task).or_default().rest_of += 1;
        }
    }
    for module in map.tempopfn.modules.values() {
        shares.entry(module.task).or_default().tempopfn += 1;
    }
    for module in map.rust_only.values() {
        shares.entry(module.task).or_default().rust_only += 1;
    }
    for support in map.test_support.values() {
        shares.entry(support.task).or_default().test_support += 1;
    }
    for file in map.tests.values() {
        for id in &file.ids {
            match fate(file, id) {
                Fate::Ported { task, .. } => shares.entry(task).or_default().ported += 1,
                Fate::Dropped { task, .. } => shares.entry(task).or_default().dropped += 1,
            }
        }
    }
    shares
}

/// The breaks of the map against the reference's files: `package` those of
/// its package, `tests` those of its tests, by their paths in the
/// repository.
#[must_use]
pub fn check(map: &Map, package: &[String], tests: &[String]) -> Vec<String> {
    let mut breaks = Vec::new();
    check_tables(map, &mut breaks);
    check_package(map, package, &mut breaks);
    check_test_files(map, tests, &mut breaks);
    check_modules(map, &mut breaks);
    check_others(map, &mut breaks);
    check_tests(map, &mut breaks);
    breaks
}

/// The breaks of the ids of the test file at `path` against its source
/// `text`: an id whose test function the file does not define at its top
/// level, and a test function there that no id names.
#[must_use]
pub fn check_functions(path: &str, file: &TestFile, text: &str) -> Vec<String> {
    let defined = test_functions(text);
    let named: BTreeSet<&str> = file.ids.iter().map(|id| split_id(id).0).collect();
    let mut breaks = Vec::new();
    for function in named.difference(&defined) {
        breaks.push(format!("[tests] {path}: the file defines no {function}"));
    }
    for function in defined.difference(&named) {
        breaks.push(format!("[tests] {path}: no id of {function}"));
    }
    breaks
}

/// The test functions that a test file of pytest's defines at its top
/// level, `def test_…(` and `async def test_…(`.
fn test_functions(text: &str) -> BTreeSet<&str> {
    text.lines()
        .filter_map(|line| {
            line.strip_prefix("def ")
                .or_else(|| line.strip_prefix("async def "))
        })
        .filter_map(|rest| rest.split_once('(').map(|(name, _)| name.trim()))
        .filter(|name| name.starts_with("test_"))
        .collect()
}

/// The tasks, the kinds of check, and an entry for every task.
fn check_tables(map: &Map, breaks: &mut Vec<String>) {
    let tasks: BTreeSet<String> = TASKS.map(|task| task.to_string()).collect();
    let named: BTreeSet<String> = map.tasks.keys().cloned().collect();
    if named != tasks {
        breaks.push(format!(
            "[tasks] holds {}, not the tasks {}..={}",
            join(&named),
            TASKS.start(),
            TASKS.end()
        ));
    }
    let checks: BTreeSet<&str> = map.checks.keys().map(String::as_str).collect();
    let kinds: BTreeSet<&str> = CHECKS.iter().copied().collect();
    if checks != kinds {
        breaks.push(format!(
            "[checks] holds {}, not {}",
            join(&checks),
            join(&kinds)
        ));
    }
    for (task, share) in summary(map) {
        if share == Share::default() {
            breaks.push(format!("task {task} has no entry"));
        }
    }
}

/// Every file of the package in `[modules]` or `[data]`, and no other.
fn check_package(map: &Map, package: &[String], breaks: &mut Vec<String>) {
    let prefix = format!("{}/", map.reference.package);
    let mut modules = BTreeSet::new();
    let mut data = BTreeSet::new();
    for file in package {
        let Some(path) = file.strip_prefix(&prefix) else {
            breaks.push(format!(
                "{file} is outside the package {}",
                map.reference.package
            ));
            continue;
        };
        if is_python(path) {
            modules.insert(path);
        } else {
            data.insert(path);
        }
    }
    same_files("[modules]", &modules, map.modules.keys(), &prefix, breaks);
    same_files("[data]", &data, map.data.keys(), &prefix, breaks);
}

/// Every file of the tests in `[tests]` or `[test_support]`, and no other.
fn check_test_files(map: &Map, tests: &[String], breaks: &mut Vec<String>) {
    let (test_files, support): (BTreeSet<&str>, BTreeSet<&str>) = tests
        .iter()
        .map(String::as_str)
        .partition(|file| is_test_file(file));
    same_files("[tests]", &test_files, map.tests.keys(), "", breaks);
    same_files(
        "[test_support]",
        &support,
        map.test_support.keys(),
        "",
        breaks,
    );
}

/// The breaks of a table whose keys are not the files `files`.
fn same_files<'a>(
    table: &str,
    files: &BTreeSet<&str>,
    keys: impl Iterator<Item = &'a String>,
    prefix: &str,
    breaks: &mut Vec<String>,
) {
    let keys: BTreeSet<&str> = keys.map(String::as_str).collect();
    for file in files.difference(&keys) {
        breaks.push(format!("{table} misses {prefix}{file}"));
    }
    for key in keys.difference(files) {
        breaks.push(format!(
            "{table} names {prefix}{key}, which the reference does not hold"
        ));
    }
}

/// Whether `path` names a Python file.
fn is_python(path: &str) -> bool {
    Path::new(path)
        .extension()
        .is_some_and(|extension| extension == "py")
}

/// Whether `path` names a test file of pytest's, `test_*.py`.
fn is_test_file(path: &str) -> bool {
    is_python(path)
        && Path::new(path)
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with("test_"))
}

/// The entries of `[modules]` and `[tempopfn]`.
fn check_modules(map: &Map, breaks: &mut Vec<String>) {
    let mut commands: BTreeMap<&str, &str> = BTreeMap::new();
    for (path, module) in &map.modules {
        let at = format!("[modules] {path}");
        check_module(map, &at, module, breaks);
        if let Some(command) = &module.command {
            if module.krate != "interfaces" {
                breaks.push(format!("{at}: only a command line has a command"));
            }
            if let Some(other) = commands.insert(command, path) {
                breaks.push(format!("{at}: {other} has its command too, {command}"));
            }
        }
    }
    if !is_commit(&map.tempopfn.commit) {
        breaks.push(format!(
            "[tempopfn] commit {} is no full commit",
            map.tempopfn.commit
        ));
    }
    for (name, module) in &map.tempopfn.modules {
        let at = format!("[tempopfn] {name}");
        if !name.starts_with("src.") {
            breaks.push(format!("{at}: TempoPFN's modules are named src.…"));
        }
        if module.krate != "infrastructure" || !module.rust.starts_with("corpus::tempopfn::") {
            breaks.push(format!(
                "{at}: TempoPFN's modules go to infrastructure::corpus::tempopfn"
            ));
        }
        if module.command.is_some() {
            breaks.push(format!("{at}: only a command line has a command"));
        }
        check_module(map, &at, module, breaks);
    }
}

/// An entry of `[modules]` or `[tempopfn]`.
fn check_module(map: &Map, at: &str, module: &Module, breaks: &mut Vec<String>) {
    check_crate(at, &module.krate, breaks);
    let rust_ok = if module.krate == "workspace" {
        module.rust == "Cargo.toml"
    } else {
        module.rust == "lib" || is_module_path(&module.rust)
    };
    if !rust_ok {
        breaks.push(format!("{at}: {} is no Rust module", module.rust));
    }
    check_task(at, module.task, breaks);
    if let Some(task) = module.also {
        check_task(at, task, breaks);
    }
    check_check(map, at, &module.check, breaks);
    if let Some(others) = &module.others {
        check_check(map, at, others, breaks);
    }
    if let Some(command) = &module.command {
        let subcommand = command
            .strip_prefix("diff-gpt ")
            .is_some_and(|words| words.split(' ').all(is_command_word));
        if !subcommand {
            breaks.push(format!("{at}: {command} is no subcommand of diff-gpt"));
        }
    }
}

/// The entries of `[data]`, `[rust_only]` and `[test_support]`.
fn check_others(map: &Map, breaks: &mut Vec<String>) {
    for (path, data) in &map.data {
        let at = format!("[data] {path}");
        check_crate(&at, &data.krate, breaks);
        if !data.path.starts_with(&format!("crates/{}/", data.krate)) {
            breaks.push(format!(
                "{at}: {} is not in the crate {}",
                data.path, data.krate
            ));
        }
        check_task(&at, data.task, breaks);
        if let Some(task) = data.also {
            check_task(&at, task, breaks);
        }
        check_check(map, &at, &data.check, breaks);
    }
    for (path, module) in &map.rust_only {
        let at = format!("[rust_only] {path}");
        match path.split_once("::") {
            Some((krate, rest)) if is_module_path(rest) => check_crate(&at, krate, breaks),
            _ => breaks.push(format!("{at}: no crate::module")),
        }
        check_task(&at, module.task, breaks);
        check_check(map, &at, &module.check, breaks);
    }
    for (path, support) in &map.test_support {
        let at = format!("[test_support] {path}");
        if let Some(rust) = &support.rust
            && !is_rust_test_file(rust)
        {
            breaks.push(format!("{at}: {rust} is no Rust test file"));
        }
        check_task(&at, support.task, breaks);
    }
}

/// The ids of `[tests]` and their Rust tests.
fn check_tests(map: &Map, breaks: &mut Vec<String>) {
    let count: usize = map.tests.values().map(|file| file.ids.len()).sum();
    if count != map.reference.pytest_ids {
        breaks.push(format!(
            "[tests] holds {count} ids, not the reference's {}",
            map.reference.pytest_ids
        ));
    }
    let mut taken: BTreeMap<(String, Option<&str>), String> = BTreeMap::new();
    for (path, file) in &map.tests {
        let at = format!("[tests] {path}");
        check_test_file(map, &at, file, breaks);
        for id in &file.ids {
            let Fate::Ported {
                test, case, moved, ..
            } = fate(file, id)
            else {
                continue;
            };
            if moved {
                continue;
            }
            let name = test.rsplit("::").next().unwrap_or_default();
            if !is_identifier(name) {
                breaks.push(format!(
                    "{at}: {id} takes the Rust name {name}, no identifier"
                ));
            }
            if let Some(other) = taken.insert((test.clone(), case), format!("{path}::{id}")) {
                breaks.push(format!(
                    "{at}: {id} and {other} take the same Rust test {test}"
                ));
            }
        }
    }
}

/// A test file's own entry, its ids, slow ids, drops and overrides.
fn check_test_file(map: &Map, at: &str, file: &TestFile, breaks: &mut Vec<String>) {
    if !is_rust_test_file(&file.rust) {
        breaks.push(format!("{at}: {} is no Rust test file", file.rust));
    }
    check_task(at, file.task, breaks);
    check_check(map, at, &file.check, breaks);
    let mut ids = BTreeSet::new();
    for id in &file.ids {
        if !ids.insert(id.as_str()) {
            breaks.push(format!("{at}: {id} twice"));
        }
        if !id.starts_with("test_") {
            breaks.push(format!("{at}: {id} is no pytest id"));
        }
    }
    for id in &file.slow {
        if !ids.contains(id.as_str()) {
            breaks.push(format!("{at}: the slow {id} is none of its ids"));
        }
    }
    let covers = |key: &str| {
        file.ids
            .iter()
            .any(|id| id.as_str() == key || split_id(id).0 == key)
    };
    let overlaps = |one: &str, other: &str| {
        one == other || split_id(one).0 == other || one == split_id(other).0
    };
    for (key, reason) in &file.drop {
        if !covers(key) {
            breaks.push(format!("{at}: drop names {key}, none of its ids"));
        }
        if reason.trim().is_empty() {
            breaks.push(format!("{at}: drop gives no reason for {key}"));
        }
        if file.overrides.keys().any(|other| overlaps(key, other)) {
            breaks.push(format!("{at}: {key} is both dropped and moved"));
        }
    }
    for (key, entry) in &file.overrides {
        if !covers(key) {
            breaks.push(format!("{at}: overrides names {key}, none of its ids"));
        }
        let at = format!("{at} overrides {key}");
        if let Some(rust) = &entry.rust {
            let test = rust
                .rsplit_once("::")
                .is_some_and(|(file, name)| is_rust_test_file(file) && is_identifier(name));
            if !test {
                breaks.push(format!("{at}: {rust} is no Rust test"));
            }
        }
        if let Some(task) = entry.task {
            check_task(&at, task, breaks);
        }
        if let Some(check) = &entry.check {
            check_check(map, &at, check, breaks);
        }
    }
}

/// A crate of [`CRATES`].
fn check_crate(at: &str, krate: &str, breaks: &mut Vec<String>) {
    if !CRATES.contains(&krate) {
        breaks.push(format!("{at}: {krate} is none of the crates"));
    }
}

/// A task of [`TASKS`].
fn check_task(at: &str, task: u32, breaks: &mut Vec<String>) {
    if !TASKS.contains(&task) {
        breaks.push(format!("{at}: {task} is none of the port's tasks"));
    }
}

/// A check of [`CHECKS`], naming a tolerance or distribution of the map.
fn check_check(map: &Map, at: &str, check: &str, breaks: &mut Vec<String>) {
    let known = match check.split_once(':') {
        None => check == "exact" || check == "none",
        Some(("ulp", count)) => count.parse::<u32>().is_ok_and(|count| count > 0),
        Some(("tolerance", name)) => map.tolerances.contains_key(name),
        Some(("distribution", name)) => map.distributions.contains_key(name),
        Some(_) => false,
    };
    if !known {
        breaks.push(format!("{at}: {check} is no check of the map"));
    }
}

/// `crates/<crate>/tests/…/<name>.rs`, of a member of the workspace.
fn is_rust_test_file(path: &str) -> bool {
    let parts: Vec<&str> = path.split('/').collect();
    let ["crates", krate, "tests", rest @ ..] = parts.as_slice() else {
        return false;
    };
    let Some((file, folders)) = rest.split_last() else {
        return false;
    };
    CRATES.contains(krate)
        && *krate != "workspace"
        && file.strip_suffix(".rs").is_some_and(is_identifier)
        && folders.iter().all(|folder| is_identifier(folder))
}

/// Identifiers joined by `::`.
fn is_module_path(path: &str) -> bool {
    path.split("::").all(is_identifier)
}

/// A Rust identifier in snake case that Rust does not keep.
fn is_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    chars
        .next()
        .is_some_and(|first| first.is_ascii_lowercase() || first == '_')
        && chars.all(|char| char.is_ascii_lowercase() || char.is_ascii_digit() || char == '_')
        && !KEYWORDS.contains(&name)
}

/// A word of a subcommand: lower case, digits and dashes.
fn is_command_word(word: &str) -> bool {
    !word.is_empty()
        && word
            .chars()
            .all(|char| char.is_ascii_lowercase() || char.is_ascii_digit() || char == '-')
}

/// A full commit: 40 hexadecimal digits.
fn is_commit(commit: &str) -> bool {
    commit.len() == 40 && commit.chars().all(|char| char.is_ascii_hexdigit())
}

/// The items of a set, joined by commas.
fn join<T: AsRef<str>>(items: &BTreeSet<T>) -> String {
    items
        .iter()
        .map(AsRef::<str>::as_ref)
        .collect::<Vec<_>>()
        .join(", ")
}

/// The modules of `TempoPFN` that the modules `roots` (each its name and its
/// source, none a package) import, and those that they import in turn: each
/// with where it comes from and the path of its file there. A module is the
/// stand-in's when `stand_in` reads its file and else the clone's when
/// `clone` does; both read a file by its path under their root
/// (`synthetic_generation/cauker/__init__.py`), and give `None` when there
/// is none. The walk reads the import statements outside
/// `if TYPE_CHECKING:` blocks, at any depth, and only the files that they
/// name.
#[must_use]
pub fn tempopfn_walk(
    roots: &[(String, String)],
    stand_in: impl Fn(&str) -> Option<String>,
    clone: impl Fn(&str) -> Option<String>,
) -> BTreeMap<String, (Origin, String)> {
    let mut queue: VecDeque<String> = VecDeque::new();
    for (module, text) in roots {
        queue.extend(imports(module, false, text));
    }
    let mut seen: BTreeMap<String, (Origin, String)> = BTreeMap::new();
    let mut absent: BTreeSet<String> = BTreeSet::new();
    while let Some(module) = queue.pop_front() {
        if seen.contains_key(&module) || absent.contains(&module) {
            continue;
        }
        let parts: Vec<&str> = module.split('.').collect();
        for end in 1..parts.len() {
            queue.push_back(parts[..end].join("."));
        }
        let Some((origin, path, text)) = find(&parts[1..], &stand_in, &clone) else {
            // An attribute that an import names, not a module.
            absent.insert(module);
            continue;
        };
        queue.extend(imports(&module, path.ends_with("__init__.py"), &text));
        seen.insert(module, (origin, path));
    }
    seen
}

/// A reader of the files of the stand-ins or of the clone, by their paths.
type Reader<'a> = &'a dyn Fn(&str) -> Option<String>;

/// The file of the module `parts` (under `src`): the package's
/// `__init__.py` or the module's file, the stand-in's before the clone's.
fn find(
    parts: &[&str],
    stand_in: &impl Fn(&str) -> Option<String>,
    clone: &impl Fn(&str) -> Option<String>,
) -> Option<(Origin, String, String)> {
    let base = parts.join("/");
    let package = if base.is_empty() {
        "__init__.py".to_owned()
    } else {
        format!("{base}/__init__.py")
    };
    let file = format!("{base}.py");
    let readers: [(Origin, Reader<'_>); 2] = [(Origin::StandIn, stand_in), (Origin::Clone, clone)];
    readers.into_iter().find_map(|(origin, read)| {
        if let Some(text) = read(&package) {
            return Some((origin, package.clone(), text));
        }
        if base.is_empty() {
            return None;
        }
        read(&file).map(|text| (origin, file.clone(), text))
    })
}

/// The `TempoPFN` modules that the module `module` imports, named as
/// `TempoPFN` names them, with the names that its `from … import` lists
/// (some of them modules, some attributes).
fn imports(module: &str, is_package: bool, text: &str) -> Vec<String> {
    let mut found = Vec::new();
    let mut skip_deeper: Option<usize> = None;
    for (indent, line) in logical_lines(text) {
        if let Some(level) = skip_deeper {
            if indent > level {
                continue;
            }
            skip_deeper = None;
        }
        if line.starts_with("if TYPE_CHECKING") {
            skip_deeper = Some(indent);
        } else if let Some(names) = line.strip_prefix("import ") {
            for part in names.split(',') {
                let name = part.split_whitespace().next().unwrap_or_default();
                found.extend(tempopfn_name(name));
            }
        } else if let Some(rest) = line.strip_prefix("from ")
            && let Some((base, names)) = rest.split_once(" import ")
            && let Some(base) = tempopfn_name(&absolute(module, is_package, base.trim()))
        {
            let names = names.trim().trim_start_matches('(').trim_end_matches(')');
            for part in names.split(',') {
                let name = part.split_whitespace().next().unwrap_or_default();
                if !name.is_empty() && name != "*" {
                    found.push(format!("{base}.{name}"));
                }
            }
            found.push(base);
        }
    }
    found
}

/// The absolute name of the base of a `from` import in the module `module`.
fn absolute(module: &str, is_package: bool, base: &str) -> String {
    let tail = base.trim_start_matches('.');
    let dots = base.len() - tail.len();
    if dots == 0 {
        return base.to_owned();
    }
    let mut parts: Vec<&str> = module.split('.').collect();
    let up = if is_package { dots - 1 } else { dots };
    parts.truncate(parts.len().saturating_sub(up));
    if tail.is_empty() {
        parts.join(".")
    } else {
        format!("{}.{tail}", parts.join("."))
    }
}

/// A module's name as `TempoPFN` names it, `src.…`, when it is one of its.
fn tempopfn_name(name: &str) -> Option<String> {
    if name == "src" || name.starts_with("src.") {
        Some(name.to_owned())
    } else if name == TEMPOPFN_ALIAS {
        Some("src".to_owned())
    } else {
        name.strip_prefix(TEMPOPFN_ALIAS)
            .and_then(|rest| rest.strip_prefix('.'))
            .map(|rest| format!("src.{rest}"))
    }
}

/// The logical lines of Python source, each with its indent: the physical
/// lines joined while a bracket is open or a line ends in a backslash, and
/// cut at a comment that no quote comes before.
fn logical_lines(text: &str) -> Vec<(usize, String)> {
    let mut lines = Vec::new();
    let mut current = String::new();
    let mut indent = 0;
    let mut depth = 0_usize;
    for raw in text.lines() {
        let line = match raw.find('#') {
            Some(at) if !raw[..at].contains(['"', '\'']) => &raw[..at],
            _ => raw,
        };
        if current.is_empty() {
            indent = line.len() - line.trim_start().len();
        }
        let continued = line.trim_end().ends_with('\\');
        current.push(' ');
        current.push_str(line.trim().trim_end_matches('\\'));
        depth = (depth + line.matches('(').count()).saturating_sub(line.matches(')').count());
        if depth == 0 && !continued {
            lines.push((indent, current.trim().to_owned()));
            current.clear();
        }
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A whole map: two modules, one data file, one module of `TempoPFN`,
    /// three new modules, one support file and one test file of three ids,
    /// with an entry for every task.
    const MAP: &str = r#"
[reference]
commit = "abc"
package = "src/p"
tests = "tests"
pytest_ids = 3

[tasks]
27 = "a"
28 = "b"
29 = "c"
30 = "d"
31 = "e"
32 = "f"
33 = "g"
34 = "h"
35 = "i"
36 = "j"

[checks]
exact = "e"
ulp = "u"
tolerance = "t"
distribution = "d"
none = "n"

[tolerances]
f32 = { rtol = 1e-6, atol = 1e-5, note = "n" }

[distributions]
seeds = "s"

[modules]
"domain/a.py" = { crate = "domain", rust = "a", task = 29, check = "ulp:1" }
"interfaces/b.py" = { crate = "interfaces", rust = "b", task = 30, check = "exact", command = "diff-gpt b", also = 35 }

[data]
"interfaces/plan.json" = { crate = "interfaces", path = "crates/interfaces/src/plan.json", task = 31, check = "exact" }

[tempopfn]
commit = "5969ec6ddbded5a8d2976d42504c1ae436a2f1b3"

[tempopfn.modules]
'src.x' = { crate = "infrastructure", rust = "corpus::tempopfn::x", task = 34, check = "distribution:seeds" }

[rust_only]
"harness::layers" = { task = 27, check = "none", note = "n" }
"harness::compare" = { task = 28, check = "exact", note = "n" }
"domain::rng" = { task = 32, check = "exact", note = "n" }

[test_support]
"tests/conftest.py" = { task = 33, note = "n" }

[tests."tests/test_a.py"]
rust = "crates/domain/tests/a.rs"
task = 29
check = "tolerance:f32"
ids = ['test_one[x-1]', 'test_one[y-2]', 'test_Two']
slow = ['test_Two']

[tests."tests/test_a.py".drop]
'test_one[y-2]' = "the reason"

[tests."tests/test_a.py".overrides]
test_Two = { task = 36 }
"#;

    const PACKAGE: &[&str] = &[
        "src/p/domain/a.py",
        "src/p/interfaces/b.py",
        "src/p/interfaces/plan.json",
    ];
    const TESTS: &[&str] = &["tests/conftest.py", "tests/test_a.py"];

    fn map(edit: impl Fn(&str) -> String) -> Map {
        toml::from_str(&edit(MAP)).expect("the map parses")
    }

    fn breaks(edit: impl Fn(&str) -> String, package: &[&str], tests: &[&str]) -> Vec<String> {
        let package: Vec<String> = package.iter().map(|file| (*file).to_owned()).collect();
        let tests: Vec<String> = tests.iter().map(|file| (*file).to_owned()).collect();
        check(&map(edit), &package, &tests)
    }

    #[test]
    fn the_whole_map_has_no_break() {
        assert_eq!(breaks(str::to_owned, PACKAGE, TESTS), Vec::<String>::new());
    }

    #[test]
    fn an_id_goes_to_the_test_of_its_name_or_where_the_map_says() {
        let map = map(str::to_owned);
        let file = &map.tests["tests/test_a.py"];
        assert_eq!(
            fate(file, "test_one[x-1]"),
            Fate::Ported {
                test: "crates/domain/tests/a.rs::one".to_owned(),
                case: Some("x-1"),
                task: 29,
                check: "tolerance:f32",
                moved: false,
            }
        );
        assert_eq!(
            fate(file, "test_one[y-2]"),
            Fate::Dropped {
                reason: "the reason",
                task: 29
            }
        );
        assert_eq!(
            fate(file, "test_Two"),
            Fate::Ported {
                test: "crates/domain/tests/a.rs::two".to_owned(),
                case: None,
                task: 36,
                check: "tolerance:f32",
                moved: false,
            }
        );
        let shares = summary(&map);
        assert_eq!(
            (shares[&29].ported, shares[&29].dropped, shares[&36].ported),
            (1, 1, 1)
        );
        assert_eq!((shares[&30].modules, shares[&35].rest_of), (1, 1));
    }

    #[test]
    fn a_file_missed_or_named_wrongly_breaks_it() {
        let package = [
            "src/p/domain/a.py",
            "src/p/domain/c.py",
            "src/p/interfaces/plan.json",
        ];
        let tests = [
            "tests/conftest.py",
            "tests/test_a.py",
            "tests/test_b.py",
            "tests/util.py",
        ];
        assert_eq!(
            breaks(str::to_owned, &package, &tests),
            [
                "[modules] misses src/p/domain/c.py",
                "[modules] names src/p/interfaces/b.py, which the reference does not hold",
                "[tests] misses tests/test_b.py",
                "[test_support] misses tests/util.py",
            ]
        );
    }

    #[test]
    fn a_wrong_entry_breaks_it() {
        let edit = |map: &str| {
            map.replace(
                "task = 29, check = \"ulp:1\"",
                "task = 26, check = \"ulp:0\"",
            )
            .replace("rust = \"b\"", "rust = \"b::type\"")
            .replace("command = \"diff-gpt b\"", "command = \"gpt b\"")
            .replace(
                "check = \"distribution:seeds\"",
                "check = \"distribution:other\"",
            )
            .replace("'test_one[y-2]' = \"the reason\"", "test_three = \"\"")
            .replace("slow = ['test_Two']", "slow = ['test_Two', 'test_four']")
            .replace(
                "test_Two = { task = 36 }",
                "test_Two = { rust = \"crates/x/tests/a.rs::t\" }",
            )
        };
        assert_eq!(
            breaks(edit, PACKAGE, TESTS),
            [
                "task 36 has no entry",
                "[modules] domain/a.py: 26 is none of the port's tasks",
                "[modules] domain/a.py: ulp:0 is no check of the map",
                "[modules] interfaces/b.py: b::type is no Rust module",
                "[modules] interfaces/b.py: gpt b is no subcommand of diff-gpt",
                "[tempopfn] src.x: distribution:other is no check of the map",
                "[tests] tests/test_a.py: the slow test_four is none of its ids",
                "[tests] tests/test_a.py: drop names test_three, none of its ids",
                "[tests] tests/test_a.py: drop gives no reason for test_three",
                "[tests] tests/test_a.py overrides test_Two: crates/x/tests/a.rs::t is no Rust test",
            ]
        );
    }

    #[test]
    fn ids_counted_otherwise_twice_or_on_one_test_break_it() {
        let edit = |map: &str| {
            map.replace("pytest_ids = 3", "pytest_ids = 6").replace(
                "'test_Two']\nslow",
                "'test_Two', 'test_two', 'test_two']\nslow",
            )
        };
        assert_eq!(
            breaks(edit, PACKAGE, TESTS),
            [
                "[tests] holds 5 ids, not the reference's 6",
                "[tests] tests/test_a.py: test_two twice",
                "[tests] tests/test_a.py: test_two and tests/test_a.py::test_Two take the same \
                 Rust test crates/domain/tests/a.rs::two",
                "[tests] tests/test_a.py: test_two and tests/test_a.py::test_two take the same \
                 Rust test crates/domain/tests/a.rs::two",
            ]
        );
    }

    #[test]
    fn the_ids_of_a_file_are_its_test_functions() {
        let map = map(str::to_owned);
        let file = &map.tests["tests/test_a.py"];
        let text = "import pytest\n\n@pytest.mark.parametrize(\"x\", [1, 2])\ndef test_one(x):\n    \
                    def test_inner():\n        pass\n\nasync def test_three(\n    y,\n):\n    pass\n";
        assert_eq!(
            check_functions("tests/test_a.py", file, text),
            [
                "[tests] tests/test_a.py: the file defines no test_Two",
                "[tests] tests/test_a.py: no id of test_three",
            ]
        );
    }

    #[test]
    fn the_walk_follows_the_imports_to_the_stand_ins_and_the_clone() {
        let stand_ins = |path: &str| match path {
            "__init__.py" => Some("TEMPOPFN_ROOT = 1\n".to_owned()),
            "gen/__init__.py" => Some(String::new()),
            "gen/stand.py" => Some("import numpy\n".to_owned()),
            _ => None,
        };
        let clone = |path: &str| match path {
            "gen/wrap.py" => Some(
                "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from src.gen.typed \
                 import T\nfrom .stand import (\n    Stand,  # the stand-in\n)\nfrom src.gen \
                 import \\\n    deep\n"
                    .to_owned(),
            ),
            "gen/deep.py" | "gen/typed.py" => Some(String::new()),
            _ => None,
        };
        let roots = [(
            "diff_gpt.infrastructure.corpus.generate".to_owned(),
            "from diff_gpt.infrastructure.corpus.tempopfn import TEMPOPFN_ROOT\nfrom \
             diff_gpt.infrastructure.corpus.tempopfn.gen.wrap import Wrapper\nimport os\n"
                .to_owned(),
        )];
        let walk = tempopfn_walk(&roots, stand_ins, clone);
        let found: Vec<(&str, Origin, &str)> = walk
            .iter()
            .map(|(name, (origin, path))| (name.as_str(), *origin, path.as_str()))
            .collect();
        assert_eq!(
            found,
            [
                ("src", Origin::StandIn, "__init__.py"),
                ("src.gen", Origin::StandIn, "gen/__init__.py"),
                ("src.gen.deep", Origin::Clone, "gen/deep.py"),
                ("src.gen.stand", Origin::StandIn, "gen/stand.py"),
                ("src.gen.wrap", Origin::Clone, "gen/wrap.py"),
            ]
        );
    }
}
