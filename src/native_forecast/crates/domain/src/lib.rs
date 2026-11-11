//! The domain of the forecasting pipeline: pure computation without files,
//! threads or clocks.
//!
//! - [`random`]: the keyed random streams every source draws from;
//! - [`numeric`]: small numerical tools;
//! - [`calendar`]: civil dates and TempoPFN's sampling frequencies;
//! - [`codec`]: the tokens of a series, in the units of its context;
//! - [`linalg`]: dot products and Cholesky factorizations;
//! - [`model`]: the transformer on one sequence, forward and backward, and
//!   its optimizer;
//! - [`single`]: the Cholesky factorization in single precision, with its
//!   failures;
//! - [`eigen`]: the eigenvectors of a symmetric matrix;
//! - [`synth`]: the synthetic sources, ports of TempoPFN's generators;
//! - [`table`]: series of one length, as the corpus stores them;
//! - [`stats`]: features of a series and the comparison of two corpora.

#![forbid(unsafe_code)]

pub mod calendar;
pub mod codec;
pub mod eigen;
pub mod linalg;
pub mod model;
pub mod numeric;
pub mod random;
pub mod single;
pub mod stats;
pub mod synth;
pub mod table;
