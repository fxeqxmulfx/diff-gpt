//! `forecast`: the command line of the forecasting pipeline. It parses the
//! arguments and wires the infrastructure to the use cases.

mod args;
mod corpus;

use std::process::ExitCode;

use diff_gpt_forecast_application::Result;

use crate::args::Options;

fn run(mut arguments: impl Iterator<Item = String>) -> Result<()> {
    let command = (arguments.next(), arguments.next());
    let command = (command.0.as_deref(), command.1.as_deref());
    let options = |arguments| Options::parse(arguments);
    match command {
        (Some("corpus"), Some("check")) => corpus::check(options(arguments)?),
        (Some("corpus"), Some("generate")) => corpus::generate(options(arguments)?),
        _ => Err(format!("usage:\n{}", corpus::USAGE).into()),
    }
}

fn main() -> ExitCode {
    match run(std::env::args().skip(1)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("forecast: {error}");
            ExitCode::FAILURE
        }
    }
}
