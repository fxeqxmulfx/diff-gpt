//! TempoPFN's Gaussian-process prior (`src/synthetic_generation/gp_prior`):
//! a draw from a process with a random composite kernel, through gpytorch.
//!
//! What the Python does, and this port with it:
//!
//! - one to six kernels, each Matérn, linear or periodic with the weights
//!   1.5, 1 and 5, are combined from the left by `+` or `×` at random;
//! - `create_kernel` gets its arguments one place off: the periods of the
//!   frequency never reach it, so a period is always drawn around one of the
//!   means 3, 5, 7, 14, 20, 21, 24, 30, 60, 90, 120, then 512, 612, … up to
//!   half the length, with the deviation `mean^0.6`, and the draw of
//!   `exact_freqs` has no effect;
//! - the linear kernel ignores the lengthscale and the two parameters drawn
//!   for its prior; a `ScaleKernel` multiplies by ln 2;
//! - the mean is zero or a line of standard normal slope and level;
//! - the series is drawn in float32 with `cholesky_jitter(10⁻³)`, where most
//!   kernel matrices need a jitter of 10⁻³ to 10⁻¹ to be factored: the
//!   module [`super::gpytorch`] follows that arithmetic. A `NotPSDError`
//!   repeats the draw with ten times the jitter, twice at most;
//! - one series in ten with a periodic kernel gets a train of bumps at its
//!   longest period, lined up with the first maximum of the series:
//!   multiplied in if the bumps are positive, added if they are negative.

use super::bumps::{Placement, bumps};
use super::gpytorch::{self, Base, Kernel, Leaf, Smoothness};
use crate::linalg::PivotedFactor;
use crate::random::Rng;

/// The parameters of the prior; the defaults are TempoPFN's
/// `GPGeneratorParams`.
#[derive(Clone, Debug)]
pub struct GaussianProcess {
    /// The most kernels in a composite.
    pub max_kernels: usize,
    /// The weights of the Matérn, linear and periodic kernels.
    pub bank: [f64; 3],
    /// The longest mean period, as a part of the length.
    pub max_period_ratio: f64,
    /// The share of series with bumps at their period.
    pub spike_ratio: f64,
    /// `cholesky_jitter` of the first attempt: the "low" noise level.
    pub jitter: f64,
    /// The attempts, each with ten times the jitter of the one before.
    pub attempts: i32,
}

impl Default for GaussianProcess {
    fn default() -> Self {
        Self {
            max_kernels: 6,
            bank: [1.5, 1.0, 5.0],
            max_period_ratio: 0.5,
            spike_ratio: 0.1,
            jitter: 1.0e-3,
            attempts: 3,
        }
    }
}

/// `torch.linspace(0, 1, length)` in float32: from the start in the first
/// half, from the end in the second.
#[must_use]
pub fn grid(length: usize) -> Vec<f32> {
    let last = length.saturating_sub(1);
    let step = 1.0 / last.max(1) as f32;
    (0..length)
        .map(|index| {
            if index < length / 2 {
                step * index as f32
            } else {
                1.0 - step * (last - index) as f32
            }
        })
        .collect()
}

impl GaussianProcess {
    /// A series of `length` steps.
    #[must_use]
    pub fn series(&self, rng: &mut Rng, length: usize) -> Vec<f64> {
        // `exact_freqs`: drawn, and without effect.
        let _ = rng.uniform();
        let count = rng.integer(1, self.max_kernels as i64 + 1);
        let mut kernel = Kernel::Leaf(self.leaf(rng, length));
        for _ in 1..count {
            let next = Kernel::Leaf(self.leaf(rng, length));
            kernel = if rng.index(2) == 0 {
                kernel.plus(next)
            } else {
                kernel.times(next)
            };
        }
        // `LinearMean` draws its weight and its bias; `ConstantMean` is 0.
        let (slope, level) = if rng.index(2) == 0 {
            (rng.normal() as f32, rng.normal() as f32)
        } else {
            (0.0, 0.0)
        };
        // The level of the noise is drawn although it is given: Python
        // evaluates the default of `dict.get` before the lookup.
        let _ = rng.weighted(&[0.1, 0.2, 0.7]);

        let grid = grid(length);
        let drawn = (0..self.attempts).find_map(|attempt| {
            let jitter = (self.jitter * 10.0_f64.powi(attempt)).max(1.0e-4);
            gpytorch::sample(&kernel, &grid, jitter, rng).ok()
        });
        let drawn = drawn.unwrap_or_else(|| fallback(rng, &grid));
        let mut series: Vec<f32> = drawn
            .iter()
            .zip(&grid)
            .map(|(value, point)| value + (point * slope + level))
            .collect();

        if rng.uniform() < self.spike_ratio {
            let longest =
                kernel
                    .periods()
                    .into_iter()
                    .fold(None, |longest: Option<f32>, period| {
                        Some(longest.map_or(period, |longest| longest.max(period)))
                    });
            if let Some(longest) = longest {
                let period = f64::from(longest * length as f32).round() as usize;
                let placement = [Placement::Regular, Placement::Patchy][rng.weighted(&[0.3, 0.7])];
                let train = bumps(rng, length, Some(placement), Some(period));
                return with_bumps(&series, &train, period);
            }
        }
        series.drain(..).map(f64::from).collect()
    }

    /// `create_kernel`: one kernel with its parameters.
    fn leaf(&self, rng: &mut Rng, length: usize) -> Leaf {
        let kind = rng.weighted(&self.bank);
        let scaled = rng.index(2) == 0;
        let lengthscale = rng.uniform_in(0.1, 5.0) as f32;
        let base = match kind {
            0 => Base::Matern {
                smoothness: [
                    Smoothness::Half,
                    Smoothness::ThreeHalves,
                    Smoothness::FiveHalves,
                ][rng.index(3)],
                lengthscale,
            },
            1 => {
                // The parameters of a prior that nothing evaluates.
                let _ = rng.uniform_in(1.0, 6.0);
                let _ = rng.uniform_in(0.1, 1.0);
                Base::Linear
            }
            _ => Base::Periodic {
                period: (self.period(rng, length) as f64 / length as f64) as f32,
                lengthscale,
            },
        };
        Leaf { base, scaled }
    }

    /// `custom_gaussian_sample`: a period in steps, drawn around one of the
    /// means with the deviation `mean^0.6`; a draw below 1 is folded to the
    /// integer above its absolute value.
    fn period(&self, rng: &mut Rng, length: usize) -> usize {
        const SHORT: [usize; 11] = [3, 5, 7, 14, 20, 21, 24, 30, 60, 90, 120];
        let longest = (self.max_period_ratio * length as f64) as usize;
        let mut means = SHORT.to_vec();
        if longest > 200 {
            let from = if 120 < longest / 2 { longest / 2 } else { 220 };
            means.extend((from..longest).step_by(100));
        } else if 240 < longest {
            means.extend([longest / 2, longest]);
        } else if 120 < longest {
            means.push(longest);
        }
        means.retain(|mean| *mean <= longest);
        // TempoPFN fails on a series too short for any mean; a period of one
        // step stands in there.
        let mean = means
            .get(rng.index(means.len().max(1)))
            .map_or(1.0, |mean| *mean as f64);
        let drawn = rng.normal_with(mean, mean.powf(0.6));
        let drawn = if drawn < 1.0 {
            drawn.abs().ceil()
        } else {
            drawn
        };
        (drawn as usize).max(1)
    }
}

/// The series with the train of bumps rolled so that a bump stands one
/// period after the first maximum of the first period.
fn with_bumps(series: &[f32], train: &[f64], period: usize) -> Vec<f64> {
    let length = series.len();
    let shift = if period > 0 && period <= length {
        let mut top = 0;
        for (index, value) in series[..period].iter().enumerate() {
            if *value > series[top] {
                top = index;
            }
        }
        period - top
    } else {
        0
    };
    let negative = train.iter().copied().fold(f64::NEG_INFINITY, f64::max) < 0.0;
    series
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let bump = train[(index + shift) % length];
            if negative {
                f64::from(*value) + bump + 1.0
            } else {
                f64::from(*value) * bump
            }
        })
        .collect()
}

/// What TempoPFN draws when every attempt has failed: a process of gpytorch's
/// default `RBFKernel`, of lengthscale ln 2. gpytorch draws it through a
/// Lanczos root of 2048 points; this is the exact draw. No kernel of the
/// prior comes here: the third attempt fails only on an eigenvalue below
/// −10, and the rounding errors of these matrices stay below 2.
fn fallback(rng: &mut Rng, grid: &[f32]) -> Vec<f32> {
    let lengthscale = core::f64::consts::LN_2;
    let factor = PivotedFactor::new(grid.len(), |row, column| {
        let distance = f64::from(grid[row] - grid[column]) / lengthscale;
        (-0.5 * distance * distance).exp()
    });
    let noise: Vec<f64> = (0..factor.rank()).map(|_| rng.normal()).collect();
    factor
        .apply(&noise)
        .into_iter()
        .map(|value| value as f32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_grid_is_torch_s_linspace() {
        let grid = grid(2048);
        assert_eq!(grid[0], 0.0);
        assert_eq!(grid[2047], 1.0);
        let step = 1.0_f32 / 2047.0;
        assert_eq!(grid[1000], step * 1000.0);
        assert_eq!(grid[1024], 1.0 - step * 1023.0);
        assert!(grid.windows(2).all(|pair| pair[0] < pair[1]));
        assert_eq!(super::grid(1), [1.0]);
    }

    #[test]
    fn the_periods_are_drawn_around_tempopfn_s_means() {
        let prior = GaussianProcess::default();
        let mut rng = Rng::from_key(&[1]);
        let draws = 40_000;
        let periods: Vec<usize> = (0..draws).map(|_| prior.period(&mut rng, 2048)).collect();
        assert!(periods.iter().all(|period| (1..1400).contains(period)));
        // Seventeen means, six of them from 512 to 1012: 6/17 of the periods
        // are long, and the rest stay below 200.
        let long = periods.iter().filter(|period| **period > 300).count();
        assert!(
            (long as f64 / draws as f64 - 6.0 / 17.0).abs() < 0.01,
            "{long}"
        );
        // A period of at most 3 steps is a draw below 4: 70% of those around
        // 3, 35% around 5, 17% around 7, 3% around the longer means.
        let short = periods.iter().filter(|period| **period <= 3).count();
        assert!(
            (short as f64 / draws as f64 - 1.25 / 17.0).abs() < 0.01,
            "{short}"
        );
        // A short series has the means up to half its length, and that.
        let mut rng = Rng::from_key(&[2]);
        let periods: Vec<usize> = (0..2000).map(|_| prior.period(&mut rng, 256)).collect();
        assert!(periods.iter().all(|period| (1..200).contains(period)));
        assert!(periods.iter().any(|period| *period > 122));
    }

    #[test]
    fn the_kernels_come_with_the_weights_of_the_bank() {
        let prior = GaussianProcess::default();
        let mut rng = Rng::from_key(&[3]);
        let draws = 20_000;
        let (mut matern, mut linear, mut periodic, mut scaled) = (0, 0, 0, 0);
        for _ in 0..draws {
            let leaf = prior.leaf(&mut rng, 2048);
            scaled += usize::from(leaf.scaled);
            match leaf.base {
                Base::Matern { lengthscale, .. } => {
                    matern += 1;
                    assert!((0.1..=5.0).contains(&lengthscale));
                }
                Base::Linear => linear += 1,
                Base::Periodic { period, .. } => {
                    periodic += 1;
                    assert!(period >= 1.0 / 2048.0);
                }
            }
        }
        let share = |count: usize| count as f64 / f64::from(draws);
        assert!((share(matern) - 0.2).abs() < 0.01);
        assert!((share(linear) - 2.0 / 15.0).abs() < 0.01);
        assert!((share(periodic) - 2.0 / 3.0).abs() < 0.012);
        assert!((share(scaled) - 0.5).abs() < 0.012);
    }

    #[test]
    fn bumps_multiply_when_positive_and_add_when_negative() {
        let series = [0.5_f32, 2.0, 1.0, 0.25, 3.0, 1.5];
        // The first maximum of the first period of 3 is at 1: the train is
        // rolled by 3 − 1 = 2 steps to the left.
        let train = [1.0, 1.0, 4.0, 1.0, 1.0, 2.0];
        assert_eq!(
            with_bumps(&series, &train, 3),
            [2.0, 2.0, 1.0, 0.5, 3.0, 1.5]
        );
        let train = [-1.0, -1.0, -4.0, -1.0, -1.0, -2.0];
        assert_eq!(
            with_bumps(&series, &train, 3),
            [-2.5, 2.0, 1.0, -0.75, 3.0, 1.5]
        );
        // A period longer than the series leaves the train in place.
        assert_eq!(
            with_bumps(&series, &[2.0, 1.0, 1.0, 1.0, 1.0, 1.0], 7),
            [1.0, 2.0, 1.0, 0.25, 3.0, 1.5]
        );
    }

    #[test]
    fn the_fallback_is_a_smooth_process_of_unit_variance() {
        let mut rng = Rng::from_key(&[4]);
        let grid = grid(128);
        let (mut variance, mut step) = (0.0, 0.0);
        for _ in 0..300 {
            let series = fallback(&mut rng, &grid);
            variance += f64::from(series[64]).powi(2) / 300.0;
            step += f64::from(series[65] - series[64]).powi(2) / 300.0;
        }
        assert!((variance - 1.0).abs() < 0.25, "variance {variance}");
        // E (Δy)² = 2 (1 − k(Δx)) ≈ (Δx / ln 2)².
        let expected = (1.0 / 127.0 / core::f64::consts::LN_2).powi(2);
        assert!((step / expected - 1.0).abs() < 0.3, "step {step}");
    }

    #[test]
    fn series_are_finite_and_differ() {
        let prior = GaussianProcess::default();
        let mut rng = Rng::from_key(&[5]);
        let first = prior.series(&mut rng, 256);
        let second = prior.series(&mut rng, 256);
        assert_eq!(first.len(), 256);
        assert!(first.iter().chain(&second).all(|value| value.is_finite()));
        assert_ne!(first, second);
        assert_eq!(prior.series(&mut Rng::from_key(&[5]), 256), first);
    }
}
