# The rules of the Rust port

The port rewrites diff-gpt in Rust on burn, in the tasks #27 to #36 of the
board. Until #36 removes it, the Python code at the commit of the
`[reference]` of `port_map.toml` is the frozen reference: the Rust code
follows it, and the comparisons check against it. These rules hold for every
port task. Each says what holds it: a test of this harness, which fails on a
break; the checks that every task runs; or the review of each task.

## Layers

1. **Clean architecture.** Four crates in layers, each depending only on the
   ones below it: `domain`, pure computation on tensors and arrays;
   `application`, the use cases and their ports, as traits; `infrastructure`,
   the adapters of the ports (files, formats, processes) and the backends;
   and `interfaces`, the command lines. The harness, the port's tools, sits
   above them, and nothing depends on it or on the interfaces. The domain
   and the application start no thread: a task that needs parallel loops in
   them changes this rule, the check of `src/layers.rs` and its docs
   together, and gives its reason.
   Held by `tests/layers.rs` for the dependencies, which it reads from
   `cargo metadata`, `cargo tree`, the manifests and the source (the docs of
   `src/layers.rs` give every part of the rule), and by the review for the
   ports and their adapters.
2. **The command lines only parse and wire.** A command line parses its
   flags, builds the adapters and the devices, hands them to a use case, and
   makes the settings of the whole process: threads, CPU affinity, the
   environment, signals, logging, burn's and cubecl's global configurations.
   No lower layer makes one. Held by `tests/layers.rs` for the settings, and
   by the review for the rest.
3. **burn without a backend below.** The domain and the application take
   burn's tensor and nn API without its default features, reach no backend or
   device crate, and outside their tests build no device and name no backend:
   they compute on the devices given them. The infrastructure and the
   interfaces pick the backends. Held by `tests/layers.rs`.

## Tests

4. **No mocks or monkeypatching.** A test double implements a port of the
   application. No crate declares a mocking crate, and unsafe code, the only
   way to patch code at run time, is forbidden (rule 8). Held by
   `tests/layers.rs` for the crates and the unsafe code, and by the review
   for the doubles.
5. **Every pytest id ported or dropped.** `port_map.toml` names the Rust test
   of every id of the reference's tests, or the reason to drop it, and the
   check of each part against the reference; each task ports its share as the
   map says. Held by `tests/port_map.rs` for the map, by each task's review
   for its share, and by #36 for the whole.
6. **The same math.** The plain GPT and ScheduleFreeAdamW keep the
   reference's math: their forward passes, losses, gradients and optimizer
   steps match the reference's within the map's tolerances,
   `tolerance:f32_rounding` on LibTorch and `tolerance:backend` on the other
   backends. Held by the parity tests of #30 on the reference's dumps of #28,
   and by the comparison harness of #28 for the commands' outputs.

## Code

7. **Strict lints.** Clippy's `all` and `pedantic` denied, warnings denied,
   rustdoc's lints and missing docs denied, and rustfmt. Every `allow` or
   `expect` covers one item and gives its reason. Held by
   `cargo clippy --workspace --all-targets -- -D warnings`,
   `cargo fmt --check` and `RUSTDOCFLAGS="-D warnings" cargo doc --workspace
   --no-deps`, which every task runs, and by `tests/layers.rs` for the lints'
   table and the allows.
8. **No unsafe code** but in the modules that `UNSAFE_MODULES` of
   `src/layers.rs` names (none yet), each with its reason. Held by every
   crate root's `#![forbid(unsafe_code)]` and by `tests/layers.rs`.
9. **The toolchain and the crates.** `rust-toolchain.toml` pins the Rust, the
   latest stable when the port began; a newer one is a commit of its own. A
   crate comes in at its latest version, and `Cargo.lock` is tracked.
   libtorch, which the LibTorch backend needs, lives outside the repository in
   a folder of its version, which `.cargo/config.toml` names. Held by the
   review.

## Tools, files and commits

10. **Every tool and script in Rust**: a binary of the harness or a command
    of the interfaces. One-off shell commands are fine. The only new Python
    is the reference dumps of #28, made by the frozen code and kept out of git
    under `runs/rust/reference/`. Held by `tests/port_map.rs`: git tracks no
    script but the reference's, and the reference's package and tests are
    unchanged since its commit.
11. **Nothing regenerable in git**: what a command makes again is not
    committed, only its source. `target/`, `runs/` and `datasets/` hold such
    files. Held by `tests/port_map.rs` for those folders, and by the review
    for the rest.
12. **A commit for each finished step**, its paths named
    (`git add -- <paths>`, never `-A` or `.`), its message
    `ADD|FIX|UPD <thing>: <summary>`, its body ending with the
    `Co-Authored-By` trailer. No amend, no push. Held by the review.

## Data and the machine

13. **The data.** GIFT-Eval's and TIME's data stay in the ignored
    `datasets/`. TIME's data (CC BY-NC 4.0) is never committed, copied or
    redistributed. Track B's pretraining reads only
    `datasets/tempopfn-synthetic`: it never reads
    `datasets/long_term_forecast`, `datasets/short_term_forecast` or the
    local illness data. Nothing trains on fev-bench's `epf_*`. TempoPFN's
    `data/nan_stats.json` is never read. Nothing is submitted or uploaded to
    GIFT-Eval or TIME without the user's explicit go-ahead. Held by the
    review, and by rule 11's check of `datasets/`.
14. **Two CPUs for the desktop.** CPUs 27 and 55 stay free for the desktop:
    a run takes `taskset -c 0-26,28-54`, socket 0's `0-13,28-41`, or socket 1
    without 27 and 55, `14-26,42-54`. Held by the review.

## Evaluation

15. **Scored once, tuned on validation.** Settings are chosen on validation
    only; for Track B, that is its pretraining holdout. A candidate's test is
    scored once. GIFT-Eval's validation is a go/no-go check, run once per
    candidate and never used for tuning. The GIFT blank-cell rule is never
    exploited. GIFT validation forecasts made for timing or for the port's
    checks are deleted unscored. The port scores no forecast of ours on
    GIFT's or TIME's data; the official baselines of #33's "G1 again" are
    scored as the reference scores them. Held by the review, and by rule 11's
    check of `datasets/`.
