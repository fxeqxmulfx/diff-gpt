//! Features of a series and the comparison of two samples of series by them.
//!
//! A port of a generator has another random stream than its original, so it
//! is checked in distribution: every feature of its series must have the
//! distribution it has in the original's series, by the two-sample
//! Kolmogorov-Smirnov distance.

use crate::numeric::{Complex, fft};

/// The names of the features, in the order of [`features`].
pub const FEATURES: [&str; 18] = [
    "mean",
    "deviation",
    "range",
    "roughness",
    "acf1",
    "acf8",
    "acf64",
    "flat",
    "skewness",
    "kurtosis",
    "step_kurtosis",
    "trend",
    "halves",
    "peak_bin",
    "peak_share",
    "high_share",
    "turns",
    "max_step",
];

/// The features of one series.
pub type Features = [f64; FEATURES.len()];

fn ratio(numerator: f64, denominator: f64) -> f64 {
    if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    }
}

/// The mean and the central moments of orders 2, 3 and 4.
fn moments(values: &[f64]) -> [f64; 4] {
    let count = values.len().max(1) as f64;
    let mean = values.iter().sum::<f64>() / count;
    let mut central = [0.0; 3];
    for value in values {
        let centered = value - mean;
        let square = centered * centered;
        central[0] += square;
        central[1] += square * centered;
        central[2] += square * square;
    }
    [
        mean,
        central[0] / count,
        central[1] / count,
        central[2] / count,
    ]
}

/// The features of `series`, in the order of [`FEATURES`]:
///
/// - the mean, the deviation and the range;
/// - `roughness`: the mean absolute step over the deviation;
/// - the autocorrelations at the lags 1, 8 and 64;
/// - `flat`: the fraction of steps that are exactly 0;
/// - the skewness and the excess kurtosis of the values, and the excess
///   kurtosis of the steps;
/// - `trend`: the correlation of the values with time;
/// - `halves`: the deviation of the first half over the sum of those of both;
/// - `peak_bin`, `peak_share`: the strongest Fourier frequency, in cycles per
///   series (of the longest power-of-two prefix), and its share of the power;
/// - `high_share`: the share of the power above a sixteenth of that prefix;
/// - `turns`: the fraction of consecutive steps of opposite signs;
/// - `max_step`: the greatest absolute step over the range.
///
/// A feature whose denominator vanishes is 0.
#[must_use]
pub fn features(series: &[f32]) -> Features {
    let values: Vec<f64> = series.iter().map(|&value| f64::from(value)).collect();
    let count = values.len();
    let steps: Vec<f64> = values.windows(2).map(|pair| pair[1] - pair[0]).collect();
    let [mean, variance, third, fourth] = moments(&values);
    let deviation = variance.sqrt();
    let low = values.iter().copied().fold(f64::INFINITY, f64::min);
    let high = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = if count == 0 { 0.0 } else { high - low };
    let step_count = steps.len().max(1) as f64;
    let mean_step = steps.iter().map(|step| step.abs()).sum::<f64>() / step_count;
    let autocorrelation = |lag: usize| {
        let sum: f64 = values
            .iter()
            .zip(values.iter().skip(lag))
            .map(|(a, b)| (a - mean) * (b - mean))
            .sum();
        ratio(sum / count.max(1) as f64, variance)
    };
    let [_, step_variance, _, step_fourth] = moments(&steps);

    let middle = (count.max(1) - 1) as f64 / 2.0;
    let time_variance = (count * count).saturating_sub(1) as f64 / 12.0;
    let covariance = values
        .iter()
        .enumerate()
        .map(|(step, value)| (step as f64 - middle) * (value - mean))
        .sum::<f64>()
        / count.max(1) as f64;
    let half = count / 2;
    let first = moments(&values[..half])[1].sqrt();
    let second = moments(&values[half..])[1].sqrt();

    let prefix = if count == 0 { 0 } else { 1 << count.ilog2() };
    let mut spectrum: Vec<Complex> = values[..prefix].iter().map(|v| (v - mean, 0.0)).collect();
    fft(&mut spectrum, false);
    let power: Vec<f64> = spectrum[..prefix / 2 + usize::from(prefix > 0)]
        .iter()
        .skip(1)
        .map(|(re, im)| re * re + im * im)
        .collect();
    let total: f64 = power.iter().sum();
    let (peak, &peak_power) = power
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap_or((0, &0.0));
    let high_power: f64 = power.iter().skip(prefix / 16).sum();

    [
        mean,
        deviation,
        range,
        ratio(mean_step, deviation),
        autocorrelation(1),
        autocorrelation(8),
        autocorrelation(64),
        steps.iter().filter(|&&step| step == 0.0).count() as f64 / step_count,
        ratio(third, variance * deviation),
        ratio(fourth, variance * variance) - if variance > 0.0 { 3.0 } else { 0.0 },
        ratio(step_fourth, step_variance * step_variance)
            - if step_variance > 0.0 { 3.0 } else { 0.0 },
        ratio(covariance, (time_variance * variance).sqrt()),
        if first + second > 0.0 {
            first / (first + second)
        } else {
            0.5
        },
        if total > 0.0 { (peak + 1) as f64 } else { 0.0 },
        ratio(peak_power, total),
        ratio(high_power, total),
        steps
            .windows(2)
            .filter(|pair| pair[0] * pair[1] < 0.0)
            .count() as f64
            / steps.len().saturating_sub(1).max(1) as f64,
        ratio(
            steps.iter().fold(0.0_f64, |m, step| m.max(step.abs())),
            range,
        ),
    ]
}

/// The Kolmogorov-Smirnov distance of two samples: the greatest difference
/// of their empirical distribution functions. Sorts both.
#[must_use]
pub fn ks_distance(first: &mut [f64], second: &mut [f64]) -> f64 {
    first.sort_by(f64::total_cmp);
    second.sort_by(f64::total_cmp);
    let (mut i, mut j, mut distance) = (0, 0, 0.0_f64);
    while i < first.len() && j < second.len() {
        let value = first[i].min(second[j]);
        while i < first.len() && first[i] <= value {
            i += 1;
        }
        while j < second.len() && second[j] <= value {
            j += 1;
        }
        let gap = i as f64 / first.len() as f64 - j as f64 / second.len() as f64;
        distance = distance.max(gap.abs());
    }
    distance
}

/// The distance that two samples of these sizes from one distribution exceed
/// with probability `level`, by the asymptotic bound.
#[must_use]
pub fn ks_critical(first: usize, second: usize, level: f64) -> f64 {
    let (n, m) = (first as f64, second as f64);
    (-(level / 2.0).ln() / 2.0 * (n + m) / (n * m)).sqrt()
}

/// The quantile `q` of a sorted sample, the nearest rank below.
#[must_use]
pub fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    sorted[((sorted.len() - 1) as f64 * q) as usize]
}

/// How one feature is distributed in two samples of series.
#[derive(Clone, Debug)]
pub struct FeatureComparison {
    /// The name of the feature.
    pub feature: &'static str,
    /// The Kolmogorov-Smirnov distance of its two samples.
    pub distance: f64,
    /// Its quantiles 0.05, 0.5 and 0.95 in the reference.
    pub reference: [f64; 3],
    /// Its quantiles 0.05, 0.5 and 0.95 in the candidate.
    pub candidate: [f64; 3],
}

/// Compares the features of a reference sample with those of a candidate.
#[must_use]
pub fn compare(reference: &[Features], candidate: &[Features]) -> Vec<FeatureComparison> {
    FEATURES
        .iter()
        .enumerate()
        .map(|(index, &feature)| {
            let mut first: Vec<f64> = reference.iter().map(|row| row[index]).collect();
            let mut second: Vec<f64> = candidate.iter().map(|row| row[index]).collect();
            let distance = ks_distance(&mut first, &mut second);
            let quantiles = |sorted: &[f64]| [0.05, 0.5, 0.95].map(|q| quantile(sorted, q));
            FeatureComparison {
                feature,
                distance,
                reference: quantiles(&first),
                candidate: quantiles(&second),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::Rng;

    fn feature(series: &[f32], name: &str) -> f64 {
        let index = FEATURES.iter().position(|&f| f == name).expect("a feature");
        features(series)[index]
    }

    #[test]
    fn a_sinusoid_has_its_features() {
        let series: Vec<f32> = (0..2048)
            .map(|t| 3.0 + 2.0 * (core::f32::consts::TAU * t as f32 / 128.0).sin())
            .collect();
        assert!((feature(&series, "mean") - 3.0).abs() < 1e-4);
        assert!((feature(&series, "deviation") - 2.0_f64.sqrt()).abs() < 1e-4);
        assert!((feature(&series, "range") - 4.0).abs() < 1e-4);
        assert_eq!(feature(&series, "peak_bin"), 16.0);
        assert!(feature(&series, "peak_share") > 0.999);
        assert!(feature(&series, "high_share") < 1e-6);
        assert!((feature(&series, "acf64") + 1.0).abs() < 0.05);
        assert!((feature(&series, "kurtosis") + 1.5).abs() < 0.01);
        assert!(feature(&series, "trend").abs() < 0.05);
        assert!((feature(&series, "halves") - 0.5).abs() < 1e-3);
        // Two turns in each of the 16 periods.
        assert!((feature(&series, "turns") - 32.0 / 2046.0).abs() < 2e-3);
    }

    #[test]
    fn degenerate_series_have_finite_features() {
        for series in [vec![], vec![1.0_f32], vec![2.0; 7], vec![0.0, 1.0]] {
            assert!(features(&series).iter().all(|value| value.is_finite()));
        }
        let ramp: Vec<f32> = (0..100).map(|t| t as f32).collect();
        assert!((feature(&ramp, "trend") - 1.0).abs() < 1e-9);
        assert_eq!(feature(&ramp, "flat"), 0.0);
        assert_eq!(feature(&[5.0; 10], "flat"), 1.0);
    }

    #[test]
    fn the_distance_separates_distributions_and_handles_ties() {
        let mut rng = Rng::from_key(&[31]);
        let mut draw = |shift: f64, count: usize| -> Vec<f64> {
            (0..count).map(|_| rng.normal() + shift).collect()
        };
        let (mut a, mut b, mut c) = (draw(0.0, 4000), draw(0.0, 6000), draw(0.3, 6000));
        let critical = ks_critical(4000, 6000, 1e-3);
        assert!((critical - 1.949 * (10_000.0_f64 / 24e6).sqrt()).abs() < 1e-4);
        assert!(ks_distance(&mut a, &mut b) < critical);
        // A shift of 0.3 moves the distribution function by 0.119 at most.
        assert!((ks_distance(&mut a, &mut c) - 0.119).abs() < 0.03);
        let mut ties = [0.0, 0.0, 1.0, 1.0];
        assert_eq!(ks_distance(&mut ties, &mut [0.0, 1.0]), 0.0);
        assert_eq!(ks_distance(&mut ties, &mut [1.0, 1.0]), 0.5);
        assert_eq!(ks_distance(&mut [0.0], &mut [2.0, 3.0]), 1.0);
    }

    #[test]
    fn comparison_reports_every_feature() {
        let mut rng = Rng::from_key(&[5]);
        let mut sample = |scale: f64| -> Vec<Features> {
            (0..300)
                .map(|_| {
                    let series: Vec<f32> =
                        (0..256).map(|_| (scale * rng.normal()) as f32).collect();
                    features(&series)
                })
                .collect()
        };
        let (white, same, wide) = (sample(1.0), sample(1.0), sample(3.0));
        let critical = ks_critical(300, 300, 1e-3);
        assert!(compare(&white, &same).iter().all(|c| c.distance < critical));
        let report = compare(&white, &wide);
        assert_eq!(report.len(), FEATURES.len());
        let of = |name: &str| {
            report
                .iter()
                .find(|c| c.feature == name)
                .expect("a feature")
        };
        assert!(of("deviation").distance > 0.99);
        assert!(of("acf1").distance < critical);
        assert!(of("deviation").candidate[1] > 2.0 * of("deviation").reference[1]);
    }
}
