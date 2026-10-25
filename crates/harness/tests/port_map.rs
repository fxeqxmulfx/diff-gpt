//! The port map against the reference's commit: `port_map.toml` takes in
//! every file of the reference's package and tests and every pytest id, each
//! with its place in the workspace, its port task and its check; and, with
//! the clone of `TempoPFN` beside the repository, every module of it that the
//! corpus imports.
#![forbid(unsafe_code)]

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use diff_gpt_harness::port_map::{self, Map, Origin};

/// The corpus's package in the reference's package, and its import name.
const CORPUS: (&str, &str) = ("infrastructure/corpus", "diff_gpt.infrastructure.corpus");

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn map() -> Map {
    port_map::read(&Path::new(env!("CARGO_MANIFEST_DIR")).join(port_map::FILE))
        .unwrap_or_else(|error| panic!("{error}"))
}

/// The text of `path` at `commit` in the repository at `repo`.
fn text(repo: &Path, commit: &str, path: &str) -> Option<String> {
    port_map::reference_text(repo, commit, path).unwrap_or_else(|error| panic!("{error}"))
}

#[test]
fn the_map_takes_in_every_part_of_the_reference() {
    let root = root();
    let map = map();
    let commit = &map.reference.commit;
    let files = |dir: &str| {
        port_map::reference_files(&root, commit, dir).unwrap_or_else(|error| panic!("{error}"))
    };
    let package = files(&map.reference.package);
    let tests = files(&map.reference.tests);
    let mut breaks = port_map::check(&map, &package, &tests);
    for (path, file) in &map.tests {
        if let Some(source) = text(&root, commit, path) {
            breaks.extend(port_map::check_functions(path, file, &source));
        }
    }
    println!(
        "the reference at {commit}: {} files of {}, {} of {}; {} pytest ids in {} test files",
        package.len(),
        map.reference.package,
        tests.len(),
        map.reference.tests,
        map.tests.values().map(|file| file.ids.len()).sum::<usize>(),
        map.tests.len()
    );
    for (task, share) in port_map::summary(&map) {
        let title = map.tasks.get(&task.to_string()).map_or("", String::as_str);
        println!("#{task} {title}\n    {share}");
    }
    assert!(
        breaks.is_empty(),
        "{} breaks of the map:\n  {}",
        breaks.len(),
        breaks.join("\n  ")
    );
}

#[test]
fn the_repository_keeps_the_reference_frozen_and_its_tools_in_rust() {
    let root = root();
    let map = map();
    let commit = &map.reference.commit;
    let tracked = port_map::tracked_files(&root).unwrap_or_else(|error| panic!("{error}"));
    let reference =
        port_map::reference_files(&root, commit, ".").unwrap_or_else(|error| panic!("{error}"));
    let changed = port_map::changed_files(
        &root,
        commit,
        &[map.reference.package.as_str(), map.reference.tests.as_str()],
    )
    .unwrap_or_else(|error| panic!("{error}"));
    let breaks = port_map::check_repository(&map, &tracked, &reference, &changed);
    println!(
        "{} files tracked; {} in the reference at {commit}, {} of them changed since",
        tracked.len(),
        reference.len(),
        changed.len()
    );
    assert!(
        breaks.is_empty(),
        "{} breaks of the port's rules on the repository's files:\n  {}",
        breaks.len(),
        breaks.join("\n  ")
    );
}

#[test]
#[ignore = "reads the clone of TempoPFN at ../TempoPFN beside the repository"]
fn the_map_takes_in_every_module_of_tempopfn_that_the_corpus_imports() {
    let root = root();
    let map = map();
    let clone = root.join("../TempoPFN");
    let head = port_map::head(&clone).unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(
        head, map.tempopfn.commit,
        "the clone is not at the map's commit"
    );
    let commit = &map.reference.commit;
    let corpus = format!("{}/{}/", map.reference.package, CORPUS.0);
    let mut roots = Vec::new();
    for path in
        port_map::reference_files(&root, commit, &corpus).unwrap_or_else(|error| panic!("{error}"))
    {
        let Some(stem) = path
            .strip_prefix(&corpus)
            .and_then(|name| name.strip_suffix(".py"))
        else {
            continue;
        };
        if stem.contains('/') || stem == "__init__" {
            continue;
        }
        let source = text(&root, commit, &path).expect("a module of the corpus reads");
        roots.push((format!("{}.{stem}", CORPUS.1), source));
    }
    let stand_ins = format!("{corpus}tempopfn/");
    let walk = port_map::tempopfn_walk(
        &roots,
        |path| text(&root, commit, &format!("{stand_ins}{path}")),
        |path| text(&clone, &map.tempopfn.commit, &format!("src/{path}")),
    );
    let mut from_stand_ins = BTreeSet::new();
    let mut from_clone = BTreeSet::new();
    for (module, (origin, path)) in &walk {
        match origin {
            Origin::StandIn => from_stand_ins.insert(format!("{}/tempopfn/{path}", CORPUS.0)),
            Origin::Clone => from_clone.insert(module.clone()),
        };
    }
    println!(
        "from {} modules of the corpus: {} modules, {} of the clone at {head} and {} stand-ins",
        roots.len(),
        walk.len(),
        from_clone.len(),
        from_stand_ins.len()
    );
    let stand_ins_mapped: BTreeSet<String> = map
        .modules
        .keys()
        .filter(|path| path.starts_with(&format!("{}/tempopfn/", CORPUS.0)))
        .cloned()
        .collect();
    let clone_mapped: BTreeSet<String> = map.tempopfn.modules.keys().cloned().collect();
    assert_eq!(
        from_clone, clone_mapped,
        "the clone's modules and [tempopfn]"
    );
    assert_eq!(
        from_stand_ins, stand_ins_mapped,
        "the stand-ins and those of [modules]"
    );
}
