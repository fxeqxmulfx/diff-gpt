//! `forecast corpus …`: make the synthetic corpus, or check its sources
//! against the corpus that TempoPFN's Python generators made.

use std::path::PathBuf;

use diff_gpt_forecast_application::Result;
use diff_gpt_forecast_application::corpus::{self, Shape, SourceCheck};
use diff_gpt_forecast_application::ports::Split;
use diff_gpt_forecast_domain::synth::{CHUNK, LENGTH, Source};
use diff_gpt_forecast_infrastructure::corpus_files::{Reference, Shards};
use diff_gpt_forecast_infrastructure::threads::Threads;

use crate::args::Options;

/// The usage of the commands of this module.
pub const USAGE: &str = "\
forecast corpus check    [--reference DIR] [--root DIR] [--split train|holdout]
                         [--sources A,B,…] [--series N] [--level P] [--threads N]
    Compare the series each source makes with those of the reference corpus,
    feature by feature (two-sample Kolmogorov-Smirnov); fail if any differs.
    The augmented sources read the generators' shards under --root.
forecast corpus generate [--root DIR] [--split train|holdout] [--sources A,B,…]
                         [--shards N] [--shard-series N] [--length N] [--threads N]
    Write the shards of the corpus that are not there yet: the generators'
    first, which the augmented sources read.";

fn split(options: &mut Options) -> Result<Split> {
    let name: String = options.take("split", "train".to_owned())?;
    Split::named(&name).ok_or_else(|| format!("no split {name:?}").into())
}

fn sources(options: &mut Options) -> Result<Vec<Source>> {
    let Some(names) = options.take_list("sources") else {
        return Ok(Source::ALL.to_vec());
    };
    names
        .iter()
        .map(|name| Source::named(name).ok_or_else(|| format!("no source {name:?}").into()))
        .collect()
}

fn threads(options: &mut Options) -> Result<Threads> {
    let count = options.take("threads", Threads::available().count())?;
    Ok(Threads::new(count))
}

fn print_check(check: &SourceCheck) {
    println!(
        "{}: {} series against {} of the reference, critical distance {:.4}",
        check.source.name(),
        check.candidate_series,
        check.reference_series,
        check.critical
    );
    println!(
        "  {:<14} {:>8}  {:>36}  {:>36}",
        "feature", "distance", "reference 5% 50% 95%", "candidate 5% 50% 95%"
    );
    for feature in &check.features {
        let quantiles = |q: [f64; 3]| format!("{:>11.4e} {:>11.4e} {:>11.4e}", q[0], q[1], q[2]);
        println!(
            "  {:<14} {:>8.4}  {}  {}{}",
            feature.feature,
            feature.distance,
            quantiles(feature.reference),
            quantiles(feature.candidate),
            if feature.distance > check.critical {
                "  DIFFERS"
            } else {
                ""
            }
        );
    }
}

/// `forecast corpus check`.
///
/// # Errors
///
/// If an option is wrong, the reference cannot be read or a source differs.
pub fn check(mut options: Options) -> Result<()> {
    let reference: PathBuf = options.take(
        "reference",
        PathBuf::from("datasets/tempopfn-synthetic/npy"),
    )?;
    let root: PathBuf = options.take("root", PathBuf::from("datasets/tempopfn-rust"))?;
    let split = split(&mut options)?;
    let sources = sources(&mut options)?;
    let series = options.take("series", 4096_usize)?;
    let level = options.take("level", 1e-3_f64)?;
    let threads = threads(&mut options)?;
    options.finish()?;

    let checks = corpus::check_all(
        &threads,
        &Reference::new(reference),
        &Shards::new(root),
        split,
        &sources,
        series,
        level,
        print_check,
    )?;
    let differing: Vec<String> = checks
        .iter()
        .flat_map(|check| {
            check
                .failures()
                .map(|feature| format!("{}/{}", check.source.name(), feature.feature))
        })
        .collect();
    if differing.is_empty() {
        println!("every feature of every source agrees with the reference");
        Ok(())
    } else {
        Err(format!(
            "{} features differ: {}",
            differing.len(),
            differing.join(" ")
        )
        .into())
    }
}

/// `forecast corpus generate`.
///
/// # Errors
///
/// If an option is wrong or a shard cannot be written.
pub fn generate(mut options: Options) -> Result<()> {
    let root: PathBuf = options.take("root", PathBuf::from("datasets/tempopfn-rust"))?;
    let split = split(&mut options)?;
    let sources = sources(&mut options)?;
    let shards = options.take("shards", 16_usize)?;
    let shard_series = options.take("shard-series", 4096_usize)?;
    let length = options.take("length", LENGTH)?;
    let threads = threads(&mut options)?;
    options.finish()?;
    if shard_series == 0 || !shard_series.is_multiple_of(CHUNK) {
        return Err(format!("--shard-series must be a positive multiple of {CHUNK}").into());
    }

    let shape = Shape {
        length,
        chunks_per_shard: shard_series / CHUNK,
        shards,
    };
    let writer = Shards::new(root);
    let started = std::time::Instant::now();
    let written = corpus::generate(
        &threads,
        &writer,
        split,
        &sources,
        shape,
        |source, shard| {
            println!(
                "{}/{} shard {shard} written, {:.1} s",
                split.name(),
                source.name(),
                started.elapsed().as_secs_f64()
            );
        },
    )?;
    println!("{written} shards written under {}", writer.root().display());
    Ok(())
}
