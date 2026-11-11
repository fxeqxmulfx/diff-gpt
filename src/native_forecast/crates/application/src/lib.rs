//! The use cases of the forecasting pipeline. They reach files, threads and
//! kernels only through the ports declared here, which the infrastructure
//! implements; nothing here starts a thread or opens a file.
//!
//! - [`ports`]: the ports;
//! - [`corpus`]: making the synthetic corpus and checking it against
//!   TempoPFN's;
//! - [`training`]: steps of the optimizer on batches shared out among
//!   workers;
//! - [`windows`]: the windows of a corpus as sequences to train on;
//! - [`pretrain`]: a pretraining run, from its plan to its history.

#![forbid(unsafe_code)]

pub mod corpus;
pub mod ports;
pub mod pretrain;
pub mod training;
pub mod windows;

/// The error of a use case: what a port reported, or a broken expectation.
pub type Error = Box<dyn core::error::Error + Send + Sync>;

/// The result of a use case.
pub type Result<T> = core::result::Result<T, Error>;
