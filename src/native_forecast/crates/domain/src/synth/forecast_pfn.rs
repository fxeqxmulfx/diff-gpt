//! `forecast_pfn`: the prior of ForecastPFN (Dooley et al., 2023) as TempoPFN
//! extends it: a linear and an exponential trend plus a product of seasonal
//! components read off the calendar, with multiplicative Weibull noise; two
//! such series blended by a logistic transition; then, at random, a time
//! warp, a rescaling of segments, a damping, bumps, or bumps alone.
//!
//! Port of TempoPFN's `forecast_pfn_prior/` with the parameters of its
//! driver. Three behaviours of the source are kept because they shape the
//! corpus:
//!
//! - the driver asks for limits of 500 on the spread and the values, which
//!   the wrapper never passes on: the generator's own 300 apply;
//! - a series outside the limits, or of a frequency the generator cannot
//!   make, is drawn again with a frequency taken uniformly among the eleven
//!   "supported" ones, so the frequencies of the corpus are not those first
//!   drawn; of the eleven, months and quarters fail under pandas 3 (their
//!   aliases `M` and `Q` are gone) and 2048 years overflow its timestamps,
//!   and each such failure is one more redraw;
//! - the trend counts whole days from the first step, so within a day it is
//!   a staircase.

use core::f64::consts::TAU;

use crate::calendar::{DAY, Frequency, fields};
use crate::numeric::linspace;
use crate::random::Rng;
use crate::synth::bumps::bumps;

/// The parameters of the source.
#[derive(Clone, Debug)]
pub struct ForecastPfn {
    /// The probabilities of noise of a low and of a moderate scale.
    pub noise_scale: (f64, f64),
    /// The probability that the harmonics of a component fade with their order.
    pub harmonic_scale_ratio: f64,
    /// The probability and strength of a time warp.
    pub time_warp: (f64, f64),
    /// The probability of rescaling segments, and the range of the factors.
    pub magnitude_scale: (f64, (f64, f64)),
    /// The probability of a damping.
    pub damping: f64,
    /// The probability of added or multiplied bumps.
    pub bumps: f64,
    /// The probability that the series is bumps alone.
    pub pure_bumps: f64,
    /// The greatest spread and the greatest absolute value of a series.
    pub limits: (f64, f64),
    /// The number of redraws before a series outside the limits is kept.
    pub retries: u64,
    /// The probability that a chunk mixes some of its series, and the
    /// greatest number of series mixed.
    pub mixup: (f64, i64),
}

impl Default for ForecastPfn {
    fn default() -> Self {
        Self {
            noise_scale: (0.6, 0.3),
            harmonic_scale_ratio: 0.5,
            time_warp: (0.1, 0.05),
            magnitude_scale: (0.2, (0.9, 1.1)),
            damping: 0.1,
            bumps: 0.15,
            pure_bumps: 0.02,
            limits: (300.0, 300.0),
            retries: 100,
            mixup: (0.1, 4),
        }
    }
}

/// The frequencies the generator redraws among; it can make all but the
/// last three.
const REDRAWN: [Frequency; 11] = [
    Frequency::S,
    Frequency::T1,
    Frequency::T5,
    Frequency::T10,
    Frequency::T15,
    Frequency::H,
    Frequency::D,
    Frequency::W,
    Frequency::M,
    Frequency::Q,
    Frequency::A,
];

/// The scales of the seasonal components of a series.
#[derive(Clone, Copy, Debug, Default)]
struct Seasonality {
    year: f64,
    month: f64,
    week: f64,
    hour: f64,
    minute: f64,
}

impl Seasonality {
    /// The scales for a frequency; `None` for the frequencies the generator
    /// does not know.
    fn sample(rng: &mut Rng, frequency: Frequency) -> Option<Self> {
        let mut scales = Self::default();
        match frequency {
            Frequency::S | Frequency::T1 | Frequency::T5 | Frequency::T10 | Frequency::T15 => {
                scales.minute = rng.uniform();
                scales.hour = rng.uniform_in(0.0, 0.2);
            }
            Frequency::H => {
                scales.minute = rng.uniform_in(0.0, 0.2);
                scales.hour = rng.uniform();
            }
            Frequency::D => {
                scales.week = rng.uniform();
                scales.month = rng.uniform_in(0.0, 0.2);
            }
            Frequency::W => {
                scales.month = rng.uniform_in(0.0, 0.3);
                scales.year = rng.uniform_in(0.0, 0.3);
            }
            Frequency::M | Frequency::Q | Frequency::A => {}
            Frequency::T30 => return None,
        }
        Some(scales)
    }
}

/// The calendar of a series: what its components read.
struct Calendar {
    /// The whole days from the first step to each step.
    days: Vec<f64>,
    month: Vec<f64>,
    day: Vec<f64>,
    weekday: Vec<f64>,
    hour: Vec<f64>,
    minute: Vec<f64>,
}

impl Calendar {
    fn new(start: i64, step: i64, length: usize) -> Self {
        let stamps: Vec<_> = (0..length as i64)
            .map(|index| fields(start + index * step))
            .collect();
        let column = |field: fn(&crate::calendar::Fields) -> i64| -> Vec<f64> {
            stamps.iter().map(|stamp| field(stamp) as f64).collect()
        };
        Self {
            days: (0..length as i64)
                .map(|index| (index * step / DAY) as f64)
                .collect(),
            month: column(|stamp| stamp.month),
            day: column(|stamp| stamp.day),
            weekday: column(|stamp| stamp.weekday),
            hour: column(|stamp| stamp.hour),
            minute: column(|stamp| stamp.minute),
        }
    }
}

impl ForecastPfn {
    /// A chunk of `count` series of `length` steps: each with its frequency
    /// and start, some mixed with each other.
    pub fn chunk(&self, rng: &mut Rng, length: usize, count: usize) -> Vec<Vec<f64>> {
        let mut series: Vec<Vec<f64>> = (0..count as u64)
            .map(|index| {
                let mut rng = rng.fork(index);
                let frequency = Frequency::select(&mut rng, length);
                let start = frequency.start(&mut rng, length);
                self.series(&mut rng, length, start, frequency)
            })
            .collect();
        if count >= 2 && rng.chance(self.mixup.0) {
            let mixed = rng.integer(2, self.mixup.1 + 1) as usize;
            let picks = rng.distinct(count, mixed.min(count));
            let originals: Vec<Vec<f64>> = picks.iter().map(|&pick| series[pick].clone()).collect();
            for &pick in &picks {
                let mut weights: Vec<f64> = picks.iter().map(|_| rng.uniform()).collect();
                let total: f64 = weights.iter().sum();
                weights.iter_mut().for_each(|weight| *weight /= total);
                for (step, value) in series[pick].iter_mut().enumerate() {
                    *value = originals
                        .iter()
                        .zip(&weights)
                        .map(|(original, weight)| weight * original[step])
                        .sum();
                }
            }
        }
        series
    }

    /// One series: drawn again, with another frequency, until it is within
    /// the limits; after the last redraw the series is kept as it is. (Were
    /// every draw to fail, which takes a hundred unlucky frequencies in a
    /// row, the series would be zeros, where the source drops it.)
    pub fn series(
        &self,
        rng: &mut Rng,
        length: usize,
        start: i64,
        frequency: Frequency,
    ) -> Vec<f64> {
        let mut last = None;
        for attempt in 0..=self.retries {
            let mut rng = rng.fork(attempt);
            let chosen = if attempt == 0 {
                frequency
            } else {
                let candidates: Vec<Frequency> = if REDRAWN.contains(&frequency) {
                    REDRAWN
                        .into_iter()
                        .filter(|&other| other != frequency)
                        .collect()
                } else {
                    REDRAWN.to_vec()
                };
                *rng.choose(&candidates)
            };
            let Some(values) = self.attempt(&mut rng, length, start, chosen) else {
                continue;
            };
            if self.acceptable(&values) {
                return values;
            }
            last = Some(values);
        }
        last.unwrap_or_else(|| vec![0.0; length])
    }

    fn acceptable(&self, values: &[f64]) -> bool {
        let low = values.iter().copied().fold(f64::INFINITY, f64::min);
        let high = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        values.iter().all(|value| value.is_finite())
            && high - low <= self.limits.0
            && low.abs() <= self.limits.1
            && high.abs() <= self.limits.1
    }

    /// One draw of a series of the given frequency; `None` where the source
    /// raises: for a frequency it does not know, or cannot lay on a calendar.
    fn attempt(
        &self,
        rng: &mut Rng,
        length: usize,
        start: i64,
        frequency: Frequency,
    ) -> Option<Vec<f64>> {
        let scales = Seasonality::sample(rng, frequency)?;
        let step = frequency.step()?;
        let calendar = Calendar::new(frequency.rollback(start), step, length);
        let timescale = frequency.days_per_period();
        let first = self.component(rng, &calendar, scales, timescale);
        let second = self.component(rng, &calendar, scales, timescale);

        // The logistic transition from the second series to the first, which
        // is at 0.1 a fifth of the way and at 0.9 four fifths of the way.
        let middle = 0.5 * length as f64;
        let rate = (0.1_f64 / 0.9).ln() / (0.2 * length as f64 - middle);
        let mut values: Vec<f64> = (0..length)
            .map(|index| {
                let weight = 1.0 / (1.0 + (-rate * ((index + 1) as f64 - middle)).exp());
                weight * first[index] + (1.0 - weight) * second[index]
            })
            .collect();

        if rng.chance(self.time_warp.0) {
            values = time_warp(rng, &values, self.time_warp.1);
        }
        if rng.chance(self.magnitude_scale.0) {
            let (low, high) = self.magnitude_scale.1;
            // One to three segments: up to two bounds inside the series.
            let inner = rng.integer(0, 3) as usize;
            let mut bounds = rng.distinct(length, inner);
            bounds.sort_unstable();
            bounds.insert(0, 0);
            bounds.push(length);
            for pair in bounds.windows(2) {
                let factor = rng.uniform_in(low, high);
                values[pair[0]..pair[1]]
                    .iter_mut()
                    .for_each(|value| *value *= factor);
            }
        }
        if rng.chance(self.damping) {
            for (value, factor) in values.iter_mut().zip(damping(rng, length)) {
                *value *= factor;
            }
        }
        if rng.chance(self.bumps) {
            // The source passes the bumps through single precision.
            let train: Vec<f64> = bumps(rng, length, None, None)
                .into_iter()
                .map(|bump| f64::from(bump as f32))
                .collect();
            let negated = train.iter().all(|&bump| bump < 0.0);
            for (value, bump) in values.iter_mut().zip(train) {
                *value = if negated {
                    *value * bump
                } else {
                    *value + bump + 1.0
                };
            }
        }
        if rng.chance(self.pure_bumps) {
            values = bumps(rng, length, None, None)
                .into_iter()
                .map(|bump| f64::from(bump as f32))
                .collect();
        }
        Some(values)
    }

    /// One of the two series of an attempt: trend plus seasonal product,
    /// times noise.
    fn component(
        &self,
        rng: &mut Rng,
        calendar: &Calendar,
        scales: Seasonality,
        timescale: f64,
    ) -> Vec<f64> {
        let length = calendar.days.len();
        let slope = rng.normal_with(0.0, 0.01);
        let base = exponential_base(rng, length, timescale);
        let slope_offset = rng.uniform_in(-0.1, 0.5);
        let base_offset = rng.uniform_in(-0.1, 0.5);
        let shape = rng.uniform_in(1.0, 5.0);
        let noise_scale = {
            let kind = rng.uniform();
            if kind <= self.noise_scale.0 {
                rng.uniform_in(0.0, 0.1)
            } else if kind <= self.noise_scale.0 + self.noise_scale.1 {
                rng.uniform_in(0.2, 0.5)
            } else {
                rng.uniform_in(0.7, 0.9)
            }
        };

        let last_day = calendar.days.last().copied().unwrap_or(0.0);
        // The trend is kept in single precision, as in the source.
        let trend: Vec<f64> = calendar
            .days
            .iter()
            .map(|&day| {
                let linear = (1.0 + (day - slope_offset * last_day) * slope) as f32;
                let growth = base.powf(day - base_offset * last_day);
                f64::from((f64::from(linear) * growth) as f32)
            })
            .collect();

        let fading = rng.chance(self.harmonic_scale_ratio);
        let mut seasonal = vec![1.0; length];
        let parts: [(f64, &[f64], usize, f64); 5] = [
            (scales.minute, &calendar.minute, 10, 60.0),
            (scales.hour, &calendar.hour, 10, 24.0),
            (scales.year, &calendar.month, 6, 12.0),
            (scales.month, &calendar.day, 10, 30.5),
            (scales.week, &calendar.weekday, 4, 7.0),
        ];
        for (scale, feature, harmonics, period) in parts {
            if scale != 0.0 {
                let wave = harmonic_wave(rng, feature, harmonics, period, fading);
                for (product, value) in seasonal.iter_mut().zip(wave) {
                    *product *= 1.0 + scale * value;
                }
            }
        }

        // Weibull noise of median 1.
        let spread = 1.0 / 2.0_f64.ln().powf(1.0 / shape);
        trend
            .iter()
            .zip(&seasonal)
            .map(|(trend, seasonal)| {
                let noise = spread * rng.weibull(shape);
                (trend + seasonal) * (1.0 + noise_scale * (noise - 1.0))
            })
            .collect()
    }
}

/// The base of the exponential trend: near 1, bounded so that the trend
/// stays within a factor of 10 over the series, and never above 1.01.
fn exponential_base(rng: &mut Rng, length: usize, timescale: f64) -> f64 {
    let raw = rng.normal_with(1.0, 0.005 / timescale);
    let exponent = 1.1 * length as f64 * timescale;
    let bounded = if raw > 1.0 {
        raw.min(10.0_f64.powf(1.0 / exponent))
    } else if raw < 1.0 {
        raw.max(0.1_f64.powf(1.0 / exponent))
    } else {
        raw
    };
    bounded.clamp(0.0001, 1.01)
}

/// A random periodic wave of unit energy over the harmonics `1..=harmonics`
/// of `period`, read at `feature`.
fn harmonic_wave(
    rng: &mut Rng,
    feature: &[f64],
    harmonics: usize,
    period: f64,
    fading: bool,
) -> Vec<f64> {
    let mut coefficients: Vec<(f64, f64)> = (1..=harmonics)
        .map(|harmonic| {
            let deviation = if fading { 1.0 / harmonic as f64 } else { 1.0 };
            (
                rng.normal_with(0.0, deviation),
                rng.normal_with(0.0, deviation),
            )
        })
        .collect();
    let norm = coefficients
        .iter()
        .map(|(sine, cosine)| sine * sine + cosine * cosine)
        .sum::<f64>()
        .sqrt();
    for (sine, cosine) in &mut coefficients {
        *sine /= norm;
        *cosine /= norm;
    }
    feature
        .iter()
        .map(|&at| {
            coefficients
                .iter()
                .enumerate()
                .map(|(index, (sine, cosine))| {
                    let angle = TAU * (index + 1) as f64 * at / period;
                    sine * angle.sin() + cosine * angle.cos()
                })
                .sum()
        })
        .collect()
}

/// NumPy's `interp` at `at` on the increasing knots `positions`.
fn interpolate(at: f64, positions: &[f64], values: &[f64]) -> f64 {
    let last = positions.len() - 1;
    if at <= positions[0] {
        return values[0];
    }
    if at >= positions[last] {
        return values[last];
    }
    let right = positions.partition_point(|&position| position <= at);
    let (x0, x1) = (positions[right - 1], positions[right]);
    values[right - 1] + (values[right] - values[right - 1]) * (at - x0) / (x1 - x0)
}

/// Reads `values` at randomly displaced times: knots every 20 steps move by
/// a normal offset of deviation `strength · length`, the ends stay.
fn time_warp(rng: &mut Rng, values: &[f64], strength: f64) -> Vec<f64> {
    let length = values.len();
    let knots = (length / 20).max(3);
    let positions = linspace(0.0, (length - 1) as f64, knots);
    let mut targets: Vec<f64> = positions
        .iter()
        .map(|&position| position + rng.normal_with(0.0, strength * length as f64))
        .collect();
    targets[0] = positions[0];
    targets[knots - 1] = positions[knots - 1];
    let indices: Vec<f64> = (0..length).map(|index| index as f64).collect();
    indices
        .iter()
        .map(|&index| {
            let warped = interpolate(index, &positions, &targets).clamp(0.0, (length - 1) as f64);
            interpolate(warped, &indices, values)
        })
        .collect()
}

/// A piecewise constant factor over `length` steps: TempoPFN's
/// `_generate_damping`.
fn damping(rng: &mut Rng, length: usize) -> Vec<f64> {
    let alternating = |rng: &mut Rng, count: usize| -> Vec<f64> {
        (0..count)
            .map(|index| {
                if index % 2 == 0 {
                    rng.uniform_in(0.4, 0.8)
                } else {
                    rng.uniform_in(1.0, 2.0)
                }
            })
            .collect()
    };
    let (breaks, factors): (Vec<f64>, Vec<f64>) = match rng.weighted(&[0.4, 0.5, 0.1]) {
        0 => {
            let steps = rng.integer(3, 7) as usize;
            let grid = linspace(0.0, length as f64, steps + 2);
            (grid[1..=steps].to_vec(), alternating(rng, steps + 1))
        }
        1 => {
            let lengths = [
                rng.integer(1, (length / 2) as i64) as usize,
                rng.integer(1, (length / 2) as i64) as usize,
            ];
            let mut breaks = Vec::new();
            let mut time = 0;
            'filled: while time < length {
                for step in lengths {
                    time += step;
                    if time > length {
                        break 'filled;
                    }
                    breaks.push(time as f64);
                }
            }
            let count = breaks.len();
            (breaks, alternating(rng, count + 1))
        }
        _ => {
            let steps = rng.integer(1, 3) as usize;
            let tail = (length as f64 * 0.1) as usize;
            let mut breaks: Vec<f64> = rng
                .distinct(length - tail, steps)
                .into_iter()
                .map(|place| place as f64)
                .collect();
            breaks.sort_by(f64::total_cmp);
            let factors = (0..=steps).map(|_| rng.uniform_in(0.1, 2.0)).collect();
            (breaks, factors)
        }
    };
    (0..length)
        .map(|index| {
            let passed = breaks.partition_point(|&point| point <= index as f64);
            factors[passed]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calendar::days_from_civil;

    #[test]
    fn interpolation_follows_numpy() {
        let positions = [0.0, 1.0, 3.0];
        let values = [10.0, 20.0, 0.0];
        for (at, expected) in [
            (-1.0, 10.0),
            (0.0, 10.0),
            (0.5, 15.0),
            (1.0, 20.0),
            (2.0, 10.0),
            (9.0, 0.0),
        ] {
            assert_eq!(interpolate(at, &positions, &values), expected, "at {at}");
        }
    }

    #[test]
    fn a_time_warp_keeps_the_ends_and_the_range() {
        let mut rng = Rng::from_key(&[1]);
        let values: Vec<f64> = (0..2048).map(|index| index as f64).collect();
        let warped = time_warp(&mut rng, &values, 0.05);
        assert_eq!((warped[0], warped[2047]), (0.0, 2047.0));
        assert!(warped.iter().all(|&value| (0.0..=2047.0).contains(&value)));
        assert!(
            warped
                .iter()
                .zip(&values)
                .any(|(a, b)| (a - b).abs() > 20.0)
        );
    }

    #[test]
    fn a_damping_is_piecewise_constant_within_its_ranges() {
        let mut rng = Rng::from_key(&[2]);
        for _ in 0..500 {
            let factors = damping(&mut rng, 2048);
            assert_eq!(factors.len(), 2048);
            assert!(factors.iter().all(|&factor| (0.1..=2.0).contains(&factor)));
            let changes = factors.windows(2).filter(|pair| pair[0] != pair[1]).count();
            assert!(changes <= 2047, "{changes} changes");
        }
    }

    #[test]
    fn the_exponential_base_is_bounded_by_the_length() {
        let mut rng = Rng::from_key(&[3]);
        for _ in 0..2000 {
            let hourly = exponential_base(&mut rng, 2048, 1.0 / 24.0);
            assert!((0.975..=1.01).contains(&hourly), "{hourly}");
            let seconds = exponential_base(&mut rng, 2048, 1.0 / 86_400.0);
            assert!((0.0001..=1.01).contains(&seconds));
        }
    }

    #[test]
    fn a_wave_has_unit_energy_and_the_period_of_its_feature() {
        let mut rng = Rng::from_key(&[4]);
        let hours: Vec<f64> = (0..240).map(|index| (index % 24) as f64).collect();
        let wave = harmonic_wave(&mut rng, &hours, 10, 24.0, true);
        assert!((0..216).all(|index| (wave[index] - wave[index + 24]).abs() < 1e-12));
        // Over a whole period the harmonics are orthogonal: the mean square
        // is half the sum of the squared coefficients, which is 1.
        let energy: f64 = wave[..24].iter().map(|value| value * value).sum::<f64>() / 24.0;
        assert!((energy - 0.5).abs() < 1e-9, "{energy}");
    }

    #[test]
    fn series_stay_within_the_limits_whatever_the_first_frequency() {
        let source = ForecastPfn::default();
        let start = days_from_civil(2001, 5, 17) * DAY + 12_345_678_901_234;
        for (seed, frequency) in [Frequency::H, Frequency::D, Frequency::W, Frequency::T30]
            .into_iter()
            .enumerate()
        {
            for index in 0..40 {
                let mut rng = Rng::from_key(&[seed as u64, index]);
                let values = source.series(&mut rng, 512, start, frequency);
                assert_eq!(values.len(), 512);
                let low = values.iter().copied().fold(f64::INFINITY, f64::min);
                let high = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                assert!(high - low <= 300.0 && low >= -300.0 && high <= 300.0);
            }
        }
    }

    #[test]
    fn unknown_frequencies_give_no_series_and_chunks_have_their_size() {
        let source = ForecastPfn::default();
        let mut rng = Rng::from_key(&[7]);
        for frequency in [Frequency::T30, Frequency::M, Frequency::Q, Frequency::A] {
            assert!(source.attempt(&mut rng, 256, 0, frequency).is_none());
        }
        assert!(source.attempt(&mut rng, 256, 0, Frequency::T10).is_some());
        let chunk = source.chunk(&mut rng, 256, 64);
        assert_eq!(chunk.len(), 64);
        assert!(chunk.iter().all(|series| series.len() == 256));
    }
}
