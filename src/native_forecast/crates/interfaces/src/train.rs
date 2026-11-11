//! `forecast train`: pretrain a model on the synthetic corpus.

use std::path::PathBuf;

use diff_gpt_forecast_application::Result;
use diff_gpt_forecast_application::ports::Crew;
use diff_gpt_forecast_application::pretrain::{self, Plan, Ports, Row};
use diff_gpt_forecast_domain::model::Config;
use diff_gpt_forecast_domain::model::optimizer::Settings;
use diff_gpt_forecast_infrastructure::corpus_files::Shards;
use diff_gpt_forecast_infrastructure::crew::Pinned;
use diff_gpt_forecast_infrastructure::gemm::Gemm;
use diff_gpt_forecast_infrastructure::run_files::{RunFiles, Stopwatch};

use crate::args::Options;
use crate::corpus::{sources, threads};

/// The usage of the command.
pub const USAGE: &str = "\
forecast train --output DIR [--root DIR] [--sources A,B,…]
               [--vocab N] [--width N] [--heads N] [--layers N] [--bins N]
               [--block N] [--batch N] [--steps N] [--every N] [--holdout N]
               [--rate X] [--decay X] [--warmup N] [--clip X]
               [--least-context N] [--domain-windows N] [--domain-quantile P]
               [--seed N] [--workers N] [--cpus A-B,C,…] [--threads N]
    Pretrain a model on the train split of the corpus under --root, a worker
    on each core, or on each of --cpus, and write to DIR its record
    (run.txt), a row for each evaluation on the holdout split (history.csv),
    its state and its averaged parameters. A DIR that holds a run goes on
    from its state.";

/// The processors of a list such as `0-13,28,30-41`.
fn cpus(list: &[String]) -> Result<Vec<usize>> {
    let mut cpus = Vec::new();
    for range in list {
        let (first, last) = range.split_once('-').unwrap_or((range, range));
        let number = |text: &str| -> Result<usize> {
            text.parse()
                .map_err(|error| format!("--cpus {range:?}: {error}").into())
        };
        cpus.extend(number(first)?..=number(last)?);
    }
    Ok(cpus)
}

/// The plan the options give; what is not given is what
/// `synthetic_pretrain.py` takes, at the sizes of its 5M model.
fn plan(options: &mut Options) -> Result<Plan> {
    let defaults = Settings::default();
    let clip: f64 = options.take("clip", 0.0)?;
    Ok(Plan {
        model: Config {
            vocab: options.take("vocab", 256)?,
            width: options.take("width", 256)?,
            heads: options.take("heads", 4)?,
            layers: options.take("layers", 6)?,
            bins: options.take("bins", 0)?,
        },
        block: options.take("block", 1412)?,
        batch: options.take("batch", 27)?,
        steps: options.take("steps", 1000)?,
        every: options.take("every", 100)?,
        holdout: options.take("holdout", 108)?,
        optimizer: Settings {
            rate: options.take("rate", defaults.rate)?,
            warmup: options.take("warmup", defaults.warmup)?,
            decay: options.take("decay", defaults.decay)?,
            ..defaults
        },
        clip: (clip > 0.0).then_some(clip),
        least_context: options.take("least-context", 16)?,
        domain_windows: options.take("domain-windows", 4096)?,
        domain_quantile: options.take("domain-quantile", 0.999)?,
        seed: options.take("seed", 11)?,
    })
}

fn print(row: &Row, plan: &Plan, last: &mut (f64, f64)) {
    // The rate since the row before: operations and tokens a second.
    let (flops, seconds) = (row.flops - last.0, row.seconds - last.1);
    let tokens = flops / plan.flops() * (plan.block * plan.batch) as f64;
    *last = (row.flops, row.seconds);
    println!(
        "step {:>7}  {:.3e} FLOPs  train {:.4}  holdout {:.4}  norm {:.3}  {:>8.0} s  {:.3e} FLOP/s  {:.0} tokens/s",
        row.step,
        row.flops,
        row.train_loss,
        row.holdout_loss,
        row.norm,
        row.seconds,
        flops / seconds,
        tokens / seconds
    );
}

/// Runs `forecast train`.
///
/// # Errors
///
/// If an option is wrong, or the run fails.
pub fn train(mut options: Options) -> Result<()> {
    let output: String = options.take("output", String::new())?;
    if output.is_empty() {
        return Err(format!("--output is needed\n{USAGE}").into());
    }
    let root: PathBuf = options.take("root", PathBuf::from("datasets/tempopfn-rust"))?;
    let sources = sources(&mut options)?;
    let plan = plan(&mut options)?;
    let crew = match options.take_list("cpus") {
        Some(list) => Pinned::on(cpus(&list)?),
        None => Pinned::cores()?,
    };
    let workers = options.take("workers", crew.size())?;
    let crew = crew.first(workers);
    let threads = threads(&mut options)?;
    options.finish()?;
    let store = RunFiles::new(&output)?;
    println!(
        "{} parameters, {} workers on processors {:?}",
        plan.model.parameters(),
        crew.size(),
        crew.cpus()
    );
    let ports = Ports {
        crew: &crew,
        executor: &threads,
        corpus: &Shards::new(root),
        store: &store,
        clock: &Stopwatch::start(),
    };
    let mut last = (0.0, 0.0);
    pretrain::run(&ports, &plan, &sources, Gemm::new, |row| {
        print(row, &plan, &mut last)
    })
}
