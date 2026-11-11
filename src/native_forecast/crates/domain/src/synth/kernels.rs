//! The bank of covariance kernels that KernelSynth and CauKer compose, and
//! draws from the Gaussian process of a composite.
//!
//! Port of the kernel bank of TempoPFN's `kernel_synth.py` and
//! `cauker_generator.py` (scikit-learn kernels with their default length
//! scales) and of the sampling of diff-gpt's stand-in for them
//! (`src/diff_gpt/infrastructure/corpus/tempopfn/…/kernel_synth.py`), which
//! made the reference corpus: on the uniform grid every kernel of the bank
//! but the dot product depends on the lag alone, so a sum or product of them
//! is `Σ_p diag(x^p) T_p diag(x^p)` with Toeplitz `T_p`, built from the lags;
//! the Gaussian is drawn through a pivoted Cholesky factorization that stops
//! at the numerical rank, without jitter.

use core::f64::consts::PI;

use crate::linalg::PivotedFactor;
use crate::numeric::linspace;
use crate::random::Rng;

/// A kernel of the bank.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Base {
    /// `exp(−2 sin²(π d / periodicity))`: scikit-learn's `ExpSineSquared`.
    Periodic {
        /// The period, in units of the grid.
        periodicity: f64,
    },
    /// `σ₀² + x x′`: `DotProduct`.
    Dot {
        /// σ₀.
        sigma: f64,
    },
    /// `exp(−d² / (2 ℓ²))`: `RBF`.
    Rbf {
        /// ℓ.
        length_scale: f64,
    },
    /// `(1 + d² / (2 α))^(−α)`: `RationalQuadratic` of length scale 1.
    RationalQuadratic {
        /// α.
        alpha: f64,
    },
    /// `noise` on the diagonal: `WhiteKernel`.
    White {
        /// The variance of the noise.
        noise: f64,
    },
    /// A constant: `ConstantKernel`.
    Constant {
        /// The constant.
        value: f64,
    },
}

/// The 33 kernels of the bank for series of `length` steps on the grid
/// `linspace(0, 1, length)`; the periods are in steps over `length`.
#[must_use]
pub fn bank(length: usize) -> Vec<Base> {
    const PERIODS: [f64; 21] = [
        24.0, 48.0, 96.0, 168.0, 336.0, 672.0, 7.0, 14.0, 30.0, 60.0, 365.0, 730.0, 4.0, 26.0,
        52.0, 4.0, 6.0, 12.0, 4.0, 40.0, 10.0,
    ];
    let mut kernels: Vec<Base> = PERIODS
        .iter()
        .map(|period| Base::Periodic {
            periodicity: period / length as f64,
        })
        .collect();
    kernels.extend([0.0, 1.0, 10.0].map(|sigma| Base::Dot { sigma }));
    kernels.extend([0.1, 1.0, 10.0].map(|length_scale| Base::Rbf { length_scale }));
    kernels.extend([0.1, 1.0, 10.0].map(|alpha| Base::RationalQuadratic { alpha }));
    kernels.extend([0.1, 1.0].map(|noise| Base::White { noise }));
    kernels.push(Base::Constant { value: 1.0 });
    kernels
}

/// A sum of products of kernels of the bank on a grid `x`:
/// `K[i][j] = Σ_p (x_i x_j)^p · terms[p][|i − j|]`.
#[derive(Clone, Debug, PartialEq)]
pub struct Composite {
    /// The lags of the Toeplitz factor of each power; empty where absent.
    terms: Vec<Vec<f64>>,
}

impl Composite {
    /// The composite of one kernel on a grid of the given `lags` from its
    /// first point.
    #[must_use]
    pub fn base(kernel: Base, lags: &[f64]) -> Self {
        let stationary =
            |value: &dyn Fn(f64) -> f64| vec![lags.iter().map(|&d| value(d)).collect()];
        let terms = match kernel {
            Base::Periodic { periodicity } => {
                stationary(&|d| (-2.0 * (PI * d / periodicity).sin().powi(2)).exp())
            }
            Base::Rbf { length_scale } => {
                stationary(&|d| (-0.5 * (d / length_scale).powi(2)).exp())
            }
            Base::RationalQuadratic { alpha } => {
                stationary(&|d| (1.0 + d * d / (2.0 * alpha)).powf(-alpha))
            }
            Base::Constant { value } => stationary(&|_| value),
            Base::White { noise } => {
                let mut white = vec![0.0; lags.len()];
                if let Some(first) = white.first_mut() {
                    *first = noise;
                }
                vec![white]
            }
            Base::Dot { sigma } => vec![vec![sigma * sigma; lags.len()], vec![1.0; lags.len()]],
        };
        Self { terms }
    }

    /// Adds `lags` to the term of `power`.
    fn accumulate(&mut self, power: usize, lags: &[f64]) {
        if self.terms.len() <= power {
            self.terms.resize(power + 1, Vec::new());
        }
        let term = &mut self.terms[power];
        if term.is_empty() {
            term.extend_from_slice(lags);
        } else {
            for (sum, lag) in term.iter_mut().zip(lags) {
                *sum += lag;
            }
        }
    }

    /// The sum of two composites.
    #[must_use]
    pub fn plus(mut self, other: &Self) -> Self {
        for (power, lags) in other.terms.iter().enumerate() {
            if !lags.is_empty() {
                self.accumulate(power, lags);
            }
        }
        self
    }

    /// The product of two composites, entry by entry.
    #[must_use]
    pub fn times(&self, other: &Self) -> Self {
        let mut product = Self { terms: Vec::new() };
        for (p, left) in self.terms.iter().enumerate() {
            for (q, right) in other.terms.iter().enumerate() {
                if !left.is_empty() && !right.is_empty() {
                    let lags: Vec<f64> = left.iter().zip(right).map(|(a, b)| a * b).collect();
                    product.accumulate(p + q, &lags);
                }
            }
        }
        product
    }

    /// The entry `(row, column)` of the matrix on the grid `x`.
    #[must_use]
    pub fn entry(&self, x: &[f64], row: usize, column: usize) -> f64 {
        let lag = row.abs_diff(column);
        let scale = x[row] * x[column];
        let mut power = 1.0;
        let mut total = 0.0;
        for lags in &self.terms {
            if !lags.is_empty() {
                total += power * lags[lag];
            }
            power *= scale;
        }
        total
    }

    /// A draw from the Gaussian of mean 0 whose covariance is this composite
    /// on the grid `x`.
    #[must_use]
    pub fn sample(&self, x: &[f64], rng: &mut Rng) -> Vec<f64> {
        let factor = PivotedFactor::new(x.len(), |row, column| self.entry(x, row, column));
        let z: Vec<f64> = (0..factor.rank()).map(|_| rng.normal()).collect();
        factor.apply(&z)
    }
}

/// A random composite: `count` kernels of the bank, drawn with replacement
/// and folded from the left with `+` or `×` at random.
///
/// # Panics
///
/// If `count` is 0 or the bank is empty.
#[must_use]
pub fn random_composite(rng: &mut Rng, bank: &[Base], lags: &[f64], count: usize) -> Composite {
    let mut composite = Composite::base(*rng.choose(bank), lags);
    for _ in 1..count {
        let next = Composite::base(*rng.choose(bank), lags);
        composite = if rng.chance(0.5) {
            composite.plus(&next)
        } else {
            composite.times(&next)
        };
    }
    composite
}

/// The grid of a series of `length` steps: `linspace(0, 1, length)`. Its
/// points are also the lags from its first point.
#[must_use]
pub fn grid(length: usize) -> Vec<f64> {
    linspace(0.0, 1.0, length)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::{deviation, mean};

    #[test]
    fn the_bank_has_the_kernels_of_the_source() {
        let kernels = bank(2048);
        assert_eq!(kernels.len(), 33);
        assert_eq!(
            kernels[0],
            Base::Periodic {
                periodicity: 24.0 / 2048.0
            }
        );
        assert_eq!(kernels[21], Base::Dot { sigma: 0.0 });
        assert_eq!(kernels[32], Base::Constant { value: 1.0 });
    }

    #[test]
    fn entries_follow_the_formulas_of_scikit_learn() {
        let x = grid(5);
        let at = |kernel, row, column| Composite::base(kernel, &x).entry(&x, row, column);
        assert!(
            (at(Base::Rbf { length_scale: 0.1 }, 0, 2) - (-0.5 * 25.0_f64).exp()).abs() < 1e-15
        );
        assert!((at(Base::RationalQuadratic { alpha: 1.0 }, 1, 3) - 1.0 / 1.125).abs() < 1e-15);
        assert!((at(Base::Periodic { periodicity: 0.5 }, 0, 2) - 1.0).abs() < 1e-12);
        let quarter = (-2.0_f64).exp();
        assert!((at(Base::Periodic { periodicity: 0.5 }, 0, 1) - quarter).abs() < 1e-12);
        assert_eq!(at(Base::Dot { sigma: 10.0 }, 2, 4), 100.5);
        assert_eq!(at(Base::White { noise: 0.1 }, 3, 3), 0.1);
        assert_eq!(at(Base::White { noise: 0.1 }, 3, 2), 0.0);
        assert_eq!(at(Base::Constant { value: 1.0 }, 0, 4), 1.0);
    }

    #[test]
    fn sums_and_products_act_entry_by_entry() {
        let x = grid(6);
        let dot = Composite::base(Base::Dot { sigma: 1.0 }, &x);
        let rbf = Composite::base(Base::Rbf { length_scale: 1.0 }, &x);
        let white = Composite::base(Base::White { noise: 1.0 }, &x);
        let sum = dot.clone().plus(&rbf);
        let product = dot.times(&dot).times(&rbf).plus(&white);
        for row in 0..6 {
            for column in 0..6 {
                let (d, r) = (dot.entry(&x, row, column), rbf.entry(&x, row, column));
                assert!((sum.entry(&x, row, column) - (d + r)).abs() < 1e-15);
                let expected = d * d * r + white.entry(&x, row, column);
                assert!((product.entry(&x, row, column) - expected).abs() < 1e-15);
            }
        }
    }

    #[test]
    fn draws_have_the_shape_of_their_kernel() {
        let x = grid(512);
        let mut rng = Rng::from_key(&[12]);
        // A constant kernel: constant series of unit variance across draws.
        let constant = Composite::base(Base::Constant { value: 1.0 }, &x);
        let levels: Vec<f64> = (0..2000)
            .map(|_| {
                let draw = constant.sample(&x, &mut rng);
                assert!(deviation(&draw) < 1e-12);
                draw[0]
            })
            .collect();
        assert!(mean(&levels).abs() < 0.08 && (deviation(&levels) - 1.0).abs() < 0.06);
        // White noise: independent values of the set variance.
        let white = Composite::base(Base::White { noise: 0.1 }, &x).sample(&x, &mut rng);
        assert!((deviation(&white) - 0.1_f64.sqrt()).abs() < 0.03);
        // A periodic kernel of 64 steps: the draw repeats every 64 steps.
        let periodic = Composite::base(
            Base::Periodic {
                periodicity: 64.0 / 511.0,
            },
            &x,
        );
        let draw = periodic.sample(&x, &mut rng);
        assert!((0..448).all(|step| (draw[step] - draw[step + 64]).abs() < 1e-5));
        assert!(deviation(&draw) > 0.05);
        // A line through the origin from the dot product of σ₀ = 0.
        let line = Composite::base(Base::Dot { sigma: 0.0 }, &x).sample(&x, &mut rng);
        assert!((0..512).all(|step| (line[step] - line[511] * x[step]).abs() < 1e-9));
    }

    #[test]
    fn a_random_composite_is_one_of_its_folds() {
        let x = grid(16);
        let kernels = bank(16);
        let mut rng = Rng::from_key(&[14]);
        for _ in 0..200 {
            let composite = random_composite(&mut rng, &kernels, &x, 5);
            assert!((0..16).all(|row| composite.entry(&x, row, row) >= 0.0));
            assert!(
                composite
                    .sample(&x, &mut rng)
                    .iter()
                    .all(|value| value.is_finite())
            );
        }
    }
}
