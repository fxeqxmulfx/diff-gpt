//! `ou_process`: an Ornstein-Uhlenbeck process that switches between a fast
//! regime about a low mean and a slow, more volatile one about a high mean,
//! its parameters moved by trends, structural changes and seasonal terms.
//!
//! Port of TempoPFN's `ornstein_uhlenbeck_process/ou_generator.py` with the
//! defaults of `OrnsteinUhlenbeckProcessGeneratorParams`: Euler steps of
//! `dt = 0.01`, no long memory. A trend may take the reversion speed or the
//! volatility below zero, as in the source; nothing is clamped.

use core::f64::consts::TAU;

use crate::random::Rng;

/// The distributions of the parameters of a regime.
#[derive(Clone, Debug)]
pub struct RegimePrior {
    /// The range of the reversion speed.
    pub theta: (f64, f64),
    /// The mean and deviation of the long-run mean.
    pub mu: (f64, f64),
    /// The mean and deviation of the logarithm of the volatility.
    pub log_sigma: (f64, f64),
}

/// The parameters of the source.
#[derive(Clone, Debug)]
pub struct Ou {
    /// The time step of the integration.
    pub dt: f64,
    /// The two regimes.
    pub regimes: [RegimePrior; 2],
    /// The mean and deviation of the first value.
    pub start: (f64, f64),
    /// The range of the probability of staying in a regime for a step.
    pub stay: (f64, f64),
    /// The probabilities of a trend in the mean, the speed and the volatility.
    pub trend_probability: [f64; 3],
    /// The probabilities of seasonal terms in the mean and in the volatility.
    pub seasonality_probability: [f64; 2],
    /// The periods a seasonal term takes, in time units.
    pub seasonal_periods: Vec<f64>,
    /// The range of the level of the series.
    pub level: (f64, f64),
    /// The range of the scale of the series.
    pub scale: (f64, f64),
    /// The range of the deviation of the measurement noise.
    pub measurement_noise: (f64, f64),
    /// The least number of steps between a structural change and an end.
    pub min_segment: f64,
}

impl Default for Ou {
    fn default() -> Self {
        Self {
            dt: 0.01,
            regimes: [
                RegimePrior {
                    theta: (1.0, 5.0),
                    mu: (-2.0, 1.0),
                    log_sigma: (0.3_f64.ln(), 0.3),
                },
                RegimePrior {
                    theta: (0.05, 0.5),
                    mu: (2.0, 1.0),
                    log_sigma: (1.5_f64.ln(), 0.5),
                },
            ],
            start: (0.0, 2.0),
            stay: (0.85, 0.999),
            trend_probability: [0.7, 0.2, 0.3],
            seasonality_probability: [0.6, 0.3],
            seasonal_periods: vec![7.0, 30.0, 90.0, 365.25, 182.625],
            level: (-100.0, 100.0),
            scale: (0.1, 50.0),
            measurement_noise: (0.0, 0.1),
            min_segment: 200.0,
        }
    }
}

/// A smooth change of a parameter over the time of the series.
#[derive(Clone, Debug)]
enum Trend {
    None,
    Linear {
        slope: f64,
    },
    Exponential {
        rate: f64,
        asymptote: f64,
    },
    Logistic {
        growth: f64,
        capacity: f64,
        midpoint: f64,
    },
    Sinusoidal {
        amplitude: f64,
        period: f64,
        phase: f64,
    },
    /// Continuous, linear between its breakpoints; `starts[k]` is its value at
    /// `breakpoints[k]`.
    Piecewise {
        breakpoints: Vec<f64>,
        slopes: Vec<f64>,
        starts: Vec<f64>,
    },
    Polynomial {
        coefficients: Vec<f64>,
    },
}

impl Trend {
    /// A trend of a random kind over `[0, total]`.
    fn sample(rng: &mut Rng, total: f64) -> Self {
        match rng.weighted(&[0.3, 0.15, 0.15, 0.1, 0.15, 0.1, 0.05]) {
            0 => Self::None,
            1 => Self::Linear {
                slope: rng.uniform_in(-0.01, 0.01),
            },
            2 => Self::Exponential {
                rate: rng.uniform_in(-0.005, 0.005),
                asymptote: rng.uniform_in(-5.0, 5.0),
            },
            3 => Self::Logistic {
                growth: rng.uniform_in(0.01, 0.1),
                capacity: rng.uniform_in(5.0, 20.0),
                midpoint: rng.uniform_in(0.3, 0.7) * total,
            },
            4 => Self::Sinusoidal {
                amplitude: rng.uniform_in(1.0, 5.0),
                period: rng.uniform_in(0.1, 0.5) * total,
                phase: rng.angle(),
            },
            5 => {
                let segments = rng.integer(2, 5) as usize;
                let mut inner: Vec<f64> =
                    (1..segments).map(|_| rng.uniform_in(0.0, total)).collect();
                inner.sort_by(f64::total_cmp);
                let mut breakpoints = vec![0.0];
                breakpoints.extend(inner);
                breakpoints.push(total);
                let slopes: Vec<f64> = (0..segments).map(|_| rng.uniform_in(-0.02, 0.02)).collect();
                let mut starts = vec![0.0];
                for segment in 0..segments {
                    let span = breakpoints[segment + 1] - breakpoints[segment];
                    starts.push(starts[segment] + slopes[segment] * span);
                }
                Self::Piecewise {
                    breakpoints,
                    slopes,
                    starts,
                }
            }
            _ => Self::Polynomial {
                // The source draws the degree among the integers of [2, 3).
                coefficients: (0..3).map(|_| rng.uniform_in(-1e-6, 1e-6)).collect(),
            },
        }
    }

    fn at(&self, t: f64) -> f64 {
        match self {
            Self::None => 0.0,
            Self::Linear { slope } => slope * t,
            Self::Exponential { rate, asymptote } => asymptote * (1.0 - (-rate * t).exp()),
            Self::Logistic {
                growth,
                capacity,
                midpoint,
            } => capacity / (1.0 + (-growth * (t - midpoint)).exp()),
            Self::Sinusoidal {
                amplitude,
                period,
                phase,
            } => amplitude * (TAU * t / period + phase).sin(),
            Self::Piecewise {
                breakpoints,
                slopes,
                starts,
            } => {
                // The segment of t: the number of breakpoints after 0 that lie
                // before t, as `searchsorted` counts them.
                let before = breakpoints[1..].iter().filter(|&&point| point < t).count();
                let segment = before.min(slopes.len() - 1);
                starts[segment] + slopes[segment] * (t - breakpoints[segment])
            }
            Self::Polynomial { coefficients } => coefficients
                .iter()
                .rev()
                .fold(0.0, |value, coefficient| value * t + coefficient),
        }
    }
}

/// A trend with structural changes: upward shifts, each of which jumps by
/// half of its magnitude at its time and then settles.
#[derive(Clone, Debug)]
struct Drift {
    trend: Trend,
    changes: Vec<(f64, f64)>,
    width: f64,
}

impl Drift {
    fn sample(rng: &mut Rng, total: f64, min_segment: f64) -> Self {
        let trend = Trend::sample(rng, total);
        let mut count = rng.integer(0, 3) as usize;
        if count > 0 && count as f64 * min_segment >= total {
            count = ((total / min_segment) as usize).saturating_sub(1).max(1);
        }
        let mut times: Vec<f64> = (0..count)
            .map(|_| rng.uniform_in(min_segment, total - min_segment))
            .collect();
        times.sort_by(f64::total_cmp);
        let changes = times
            .into_iter()
            .map(|time| (time, rng.uniform_in(1.0, 5.0)))
            .collect();
        Self {
            trend,
            changes,
            width: 0.1 * min_segment,
        }
    }

    fn at(&self, t: f64) -> f64 {
        let shifts: f64 = self
            .changes
            .iter()
            .filter(|&&(time, _)| t >= time)
            .map(|&(time, magnitude)| magnitude / (1.0 + (-10.0 * (t - time) / self.width).exp()))
            .sum();
        self.trend.at(t) + shifts
    }
}

/// A seasonal term of a parameter.
#[derive(Clone, Debug)]
struct Seasonal {
    period: f64,
    amplitude: f64,
    phase: f64,
    amplitude_trend: f64,
}

impl Seasonal {
    fn at(&self, t: f64) -> f64 {
        (self.amplitude + self.amplitude_trend * t) * (TAU * t / self.period + self.phase).sin()
    }
}

/// The parameters of a regime of one series.
#[derive(Clone, Debug)]
struct Regime {
    theta: f64,
    mu: f64,
    sigma: f64,
    start: f64,
    /// The trends of the mean, the speed and the volatility.
    drifts: [Option<Drift>; 3],
    /// The seasonal terms of the mean and of the volatility.
    seasons: [Vec<Seasonal>; 2],
}

impl Regime {
    /// The mean, the speed and the volatility at time `t`.
    fn at(&self, t: f64) -> (f64, f64, f64) {
        let drift = |index: usize| self.drifts[index].as_ref().map_or(0.0, |d| d.at(t));
        let season = |index: usize| self.seasons[index].iter().map(|s| s.at(t)).sum::<f64>();
        (
            self.mu + drift(0) + season(0),
            self.theta + drift(1),
            self.sigma + drift(2) + season(1),
        )
    }
}

impl Ou {
    /// One series of `length` steps.
    pub fn series(&self, rng: &mut Rng, length: usize) -> Vec<f64> {
        let total = self.dt * length.saturating_sub(1) as f64;
        let regimes = self.regimes(rng, total);
        let stay = [
            rng.uniform_in(self.stay.0, self.stay.1),
            rng.uniform_in(self.stay.0, self.stay.1),
        ];
        let mut regime = rng.index(2);
        let mut value = regimes[regime].start;
        let mut path = Vec::with_capacity(length);
        let root = self.dt.sqrt();
        for step in 0..length {
            path.push(value);
            let (mu, theta, sigma) = regimes[regime].at(self.dt * step as f64);
            value += theta * (mu - value) * self.dt + sigma * rng.normal_with(0.0, root);
            if !rng.chance(stay[regime]) {
                regime = 1 - regime;
            }
        }
        let level = rng.uniform_in(self.level.0, self.level.1);
        let scale = rng.uniform_in(self.scale.0, self.scale.1);
        let noise = rng.uniform_in(self.measurement_noise.0, self.measurement_noise.1);
        path.into_iter()
            .map(|value| level + scale * value + rng.normal_with(0.0, noise))
            .collect()
    }

    /// The two regimes of a series that lasts `total` time units.
    fn regimes(&self, rng: &mut Rng, total: f64) -> [Regime; 2] {
        let mut regimes = self.regimes.clone().map(|prior| Regime {
            theta: rng.uniform_in(prior.theta.0, prior.theta.1),
            mu: rng.normal_with(prior.mu.0, prior.mu.1),
            sigma: rng.lognormal(prior.log_sigma.0, prior.log_sigma.1),
            start: rng.normal_with(self.start.0, self.start.1),
            drifts: [None, None, None],
            seasons: [Vec::new(), Vec::new()],
        });
        // Both regimes get seasonal terms of a parameter, or neither does.
        for (parameter, &probability) in self.seasonality_probability.iter().enumerate() {
            if rng.chance(probability) {
                for regime in &mut regimes {
                    regime.seasons[parameter] = self.seasonal_terms(rng);
                }
            }
        }
        for (parameter, &probability) in self.trend_probability.iter().enumerate() {
            for regime in &mut regimes {
                if rng.chance(probability) {
                    let min_segment = self.min_segment * self.dt;
                    regime.drifts[parameter] = Some(Drift::sample(rng, total, min_segment));
                }
            }
        }
        regimes
    }

    /// One or two seasonal terms with distinct periods.
    fn seasonal_terms(&self, rng: &mut Rng) -> Vec<Seasonal> {
        let count = (rng.integer(1, 3) as usize).min(self.seasonal_periods.len());
        rng.distinct(self.seasonal_periods.len(), count)
            .into_iter()
            .map(|pick| {
                let period = self.seasonal_periods[pick] * (1.0 + rng.uniform_in(-0.05, 0.05));
                let amplitude_trend = rng.uniform_in(-0.001, 0.001);
                Seasonal {
                    period,
                    amplitude: rng.uniform_in(0.5, 3.0),
                    phase: rng.angle(),
                    amplitude_trend,
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::{deviation, mean};

    #[test]
    fn a_plain_process_reverts_to_its_mean_with_its_stationary_spread() {
        // One regime, no trend, no season: a discretized OU process with
        // stationary deviation sigma / sqrt(2 theta) about mu.
        let prior = RegimePrior {
            theta: (4.0, 4.0),
            mu: (3.0, 0.0),
            log_sigma: (0.0, 0.0),
        };
        let source = Ou {
            regimes: [prior.clone(), prior],
            trend_probability: [0.0; 3],
            seasonality_probability: [0.0; 2],
            level: (0.0, 0.0),
            scale: (1.0, 1.0),
            measurement_noise: (0.0, 0.0),
            ..Ou::default()
        };
        let values = source.series(&mut Rng::from_key(&[6]), 400_000);
        let settled = &values[2000..];
        assert!((mean(settled) - 3.0).abs() < 0.03);
        assert!((deviation(settled) - (1.0_f64 / 8.0).sqrt()).abs() < 0.02);
    }

    #[test]
    fn a_piecewise_trend_is_continuous_and_follows_its_slopes() {
        let trend = Trend::Piecewise {
            breakpoints: vec![0.0, 2.0, 5.0, 10.0],
            slopes: vec![1.0, -1.0, 0.5],
            starts: vec![0.0, 2.0, -1.0, 1.5],
        };
        for (t, expected) in [
            (0.0, 0.0),
            (1.0, 1.0),
            (2.0, 2.0),
            (4.0, 0.0),
            (5.0, -1.0),
            (9.0, 1.0),
        ] {
            assert!((trend.at(t) - expected).abs() < 1e-12, "at {t}");
        }
        let polynomial = Trend::Polynomial {
            coefficients: vec![1.0, 2.0, 3.0],
        };
        assert_eq!(polynomial.at(2.0), 1.0 + 4.0 + 12.0);
    }

    #[test]
    fn a_structural_change_jumps_by_half_and_settles() {
        let drift = Drift {
            trend: Trend::None,
            changes: vec![(5.0, 4.0)],
            width: 0.2,
        };
        assert_eq!(drift.at(4.99), 0.0);
        assert_eq!(drift.at(5.0), 2.0);
        assert!((drift.at(6.0) - 4.0).abs() < 1e-9);
    }

    #[test]
    fn the_default_series_are_finite() {
        let source = Ou::default();
        for seed in 0..300 {
            let values = source.series(&mut Rng::from_key(&[seed]), 2048);
            assert_eq!(values.len(), 2048);
            assert!(values.iter().all(|v| v.is_finite()), "seed {seed}");
        }
    }
}
