//! `forecast`: the command line of the forecasting pipeline. It parses the
//! arguments and wires the infrastructure to the use cases.

mod args;
mod corpus;
mod train;

use std::process::ExitCode;

use diff_gpt_forecast_application::Result;

use crate::args::Options;

fn run(arguments: &[String]) -> Result<()> {
    let words: Vec<&str> = arguments.iter().map(String::as_str).collect();
    let options = |rest: &[&str]| Options::parse(rest.iter().map(|word| (*word).to_owned()));
    match words.as_slice() {
        ["corpus", "check", rest @ ..] => corpus::check(options(rest)?),
        ["corpus", "generate", rest @ ..] => corpus::generate(options(rest)?),
        ["train", rest @ ..] => train::train(options(rest)?),
        _ => Err(format!("usage:\n{}\n{}", corpus::USAGE, train::USAGE).into()),
    }
}

fn main() -> ExitCode {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    match run(&arguments) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("forecast: {error}");
            ExitCode::FAILURE
        }
    }
}
