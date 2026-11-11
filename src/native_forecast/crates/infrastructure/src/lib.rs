//! The infrastructure of the forecasting pipeline: what stands behind the
//! ports of the use cases.
//!
//! - [`npy`]: tables of series as `.npy` files;
//! - [`corpus_files`]: the reference corpus and the corpus of this workspace
//!   on disk;
//! - [`threads`]: the executor of parallel jobs;
//! - [`kernels`]: the x86 kernels of the training engine, in Kernite;
//! - [`gemm`]: the matrix product around them, behind the model's port.

pub mod corpus_files;
pub mod gemm;
pub mod kernels;
pub mod npy;
pub mod threads;
