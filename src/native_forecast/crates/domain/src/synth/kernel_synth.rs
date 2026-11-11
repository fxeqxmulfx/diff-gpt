//! `kernel`: KernelSynth (Ansari et al., Chronos, 2024): a draw from a
//! Gaussian process whose kernel is a random sum or product of one to five
//! kernels of the bank.
//!
//! Port of TempoPFN's `kernel_synth/kernel_synth.py`, sampled as diff-gpt's
//! stand-in samples it (see [`super::kernels`]).

use crate::random::Rng;
use crate::synth::kernels::{bank, grid, random_composite};

/// The parameters of the source.
#[derive(Clone, Copy, Debug)]
pub struct KernelSynth {
    /// The greatest number of kernels in a composite.
    pub max_kernels: i64,
}

impl Default for KernelSynth {
    fn default() -> Self {
        Self { max_kernels: 5 }
    }
}

impl KernelSynth {
    /// One series of `length` steps.
    pub fn series(&self, rng: &mut Rng, length: usize) -> Vec<f64> {
        let x = grid(length);
        let count = rng.integer(1, self.max_kernels + 1) as usize;
        random_composite(rng, &bank(length), &x, count).sample(&x, rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn series_are_finite_and_varied() {
        let source = KernelSynth::default();
        let mut constant = 0;
        for seed in 0..60 {
            let values = source.series(&mut Rng::from_key(&[seed]), 256);
            assert_eq!(values.len(), 256);
            assert!(values.iter().all(|value| value.is_finite()));
            if values.iter().all(|&value| (value - values[0]).abs() < 1e-9) {
                constant += 1;
            }
        }
        // Only the constant kernel alone, or with itself, gives a constant.
        assert!(constant <= 6, "{constant} constant series of 60");
    }
}
