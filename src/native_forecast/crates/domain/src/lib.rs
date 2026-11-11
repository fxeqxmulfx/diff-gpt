//! The domain of the forecasting pipeline: pure computation without files,
//! threads or clocks.
//!
//! - [`random`]: the keyed random streams every source draws from;
//! - [`numeric`]: small numerical tools;
//! - [`synth`]: the synthetic sources, ports of TempoPFN's generators;
//! - [`table`]: series of one length, as the corpus stores them;
//! - [`stats`]: features of a series and the comparison of two corpora.

#![forbid(unsafe_code)]

pub mod numeric;
pub mod random;
pub mod stats;
pub mod synth;
pub mod table;
