//! `anomaly`: a constant level with one-step spikes of one sign at a fixed
//! period, to which a quarter of the series add clusters of nearby spikes and
//! another quarter spikes at random places.
//!
//! Port of TempoPFN's `anomalies/anomaly_generator.py` with the defaults of
//! `AnomalyGeneratorParams`, whose magnitude pattern is the correlated one.

use crate::numeric::round_half_even;
use crate::random::Rng;

/// The parameters of the source.
#[derive(Clone, Debug)]
pub struct Anomaly {
    /// The range of the level.
    pub level: (f64, f64),
    /// The probability that the spikes point up.
    pub up_probability: f64,
    /// The range of the period of the spikes, the upper end excluded.
    pub period: (i64, i64),
    /// The probability of a series with clusters.
    pub cluster_probability: f64,
    /// The probability of a series with spikes at random places.
    pub random_probability: f64,
    /// The fraction of the periodic spikes that get a cluster.
    pub cluster_fraction: f64,
    /// The range of the spikes added by a cluster, the upper end excluded.
    pub cluster_spikes: (i64, i64),
    /// The range of their offsets from the periodic spike, the upper end excluded.
    pub cluster_offset: (i64, i64),
    /// The number of random spikes, as a fraction of the periodic ones.
    pub random_fraction: f64,
    /// The range of the typical magnitude of the spikes of a series.
    pub magnitude: (f64, f64),
    /// The weight of the previous magnitude in the next one.
    pub magnitude_correlation: f64,
    /// The deviation of the innovation of a magnitude, as a fraction of the typical one.
    pub magnitude_innovation: f64,
    /// The deviation of the noise of a magnitude, as a fraction of the typical one.
    pub magnitude_noise: f64,
    /// The bounds of a magnitude, as fractions of the typical one.
    pub magnitude_bounds: (f64, f64),
}

impl Default for Anomaly {
    fn default() -> Self {
        Self {
            level: (-100.0, 100.0),
            up_probability: 0.5,
            period: (100, 300),
            cluster_probability: 0.25,
            random_probability: 0.25,
            cluster_fraction: 0.3,
            cluster_spikes: (1, 4),
            cluster_offset: (-10, 11),
            random_fraction: 0.3,
            magnitude: (10.0, 50.0),
            magnitude_correlation: 0.7,
            magnitude_innovation: 0.2,
            magnitude_noise: 0.1,
            magnitude_bounds: (0.1, 3.0),
        }
    }
}

impl Anomaly {
    /// One series of `length` steps.
    pub fn series(&self, rng: &mut Rng, length: usize) -> Vec<f64> {
        let mut values = vec![rng.uniform_in(self.level.0, self.level.1); length];
        let sign = if rng.chance(self.up_probability) {
            1.0
        } else {
            -1.0
        };
        let places = self.places(rng, length);
        if places.is_empty() {
            return values;
        }
        for (place, magnitude) in places.iter().zip(self.magnitudes(rng, places.len())) {
            values[*place] += sign * magnitude;
        }
        values
    }

    /// The places of the spikes, in the order that their magnitudes follow:
    /// the periodic ones in time, each with its cluster, then the random ones.
    fn places(&self, rng: &mut Rng, length: usize) -> Vec<usize> {
        let period = rng.integer(self.period.0, self.period.1) as usize;
        let base: Vec<usize> = (period / 2..length).step_by(period).collect();
        if base.is_empty() {
            return base;
        }
        let mut events: Vec<Vec<usize>> = base.iter().map(|&place| vec![place]).collect();
        let kind = rng.uniform();
        if kind < self.cluster_probability {
            let wanted = round_half_even(self.cluster_fraction * base.len() as f64) as usize;
            for index in rng.distinct(base.len(), wanted.clamp(1, base.len())) {
                let added = rng.integer(self.cluster_spikes.0, self.cluster_spikes.1);
                for _ in 0..added {
                    let offset = rng.integer(self.cluster_offset.0, self.cluster_offset.1);
                    let place = base[index] as i64 + offset;
                    if (0..length as i64).contains(&place) {
                        events[index].push(place as usize);
                    }
                }
                events[index].sort_unstable();
                events[index].dedup();
            }
        } else if kind < self.cluster_probability + self.random_probability {
            let wanted = round_half_even(self.random_fraction * base.len() as f64) as usize;
            let free: Vec<usize> = (0..length)
                .filter(|place| place % period != period / 2)
                .collect();
            for pick in rng.distinct(free.len(), wanted.min(free.len())) {
                events.push(vec![free[pick]]);
            }
        }
        events.concat()
    }

    /// The magnitudes of `count` spikes: an autoregression about the typical
    /// magnitude of the series, with noise, kept within its bounds.
    fn magnitudes(&self, rng: &mut Rng, count: usize) -> Vec<f64> {
        let typical = rng.uniform_in(self.magnitude.0, self.magnitude.1);
        let mut magnitudes = Vec::with_capacity(count);
        let mut previous = typical;
        for index in 0..count {
            if index > 0 {
                previous = self.magnitude_correlation * previous
                    + (1.0 - self.magnitude_correlation) * typical
                    + rng.normal_with(0.0, self.magnitude_innovation * typical);
            }
            magnitudes.push(previous);
        }
        let (low, high) = self.magnitude_bounds;
        for magnitude in &mut magnitudes {
            *magnitude = (*magnitude + rng.normal_with(0.0, self.magnitude_noise * typical))
                .clamp(low * typical, high * typical);
        }
        magnitudes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spikes_are_periodic_one_step_long_and_of_one_sign() {
        let source = Anomaly {
            cluster_probability: 0.0,
            random_probability: 0.0,
            ..Anomaly::default()
        };
        for seed in 0..200 {
            let values = source.series(&mut Rng::from_key(&[seed]), 2048);
            let level = values[0];
            let spikes: Vec<usize> = (0..2048).filter(|&t| values[t] != level).collect();
            let period = spikes[1] - spikes[0];
            assert!((100..300).contains(&period) && spikes[0] == period / 2);
            assert!(spikes.windows(2).all(|pair| pair[1] - pair[0] == period));
            assert_eq!(spikes.len(), (2048 - period / 2).div_ceil(period));
            let up = values[spikes[0]] > level;
            for &t in &spikes {
                let height = (values[t] - level).abs();
                assert!((values[t] > level) == up && (1.0..=150.0).contains(&height));
            }
        }
    }

    #[test]
    fn a_quarter_of_the_series_have_clusters_and_a_quarter_random_spikes() {
        let source = Anomaly::default();
        let mut rng = Rng::from_key(&[21]);
        let (mut clustered, mut random) = (0, 0);
        let trials = 4000;
        for _ in 0..trials {
            let places = source.places(&mut rng, 2048);
            let mut sorted = places.clone();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(sorted.len(), places.len(), "a place holds one spike");
            let near = places
                .windows(2)
                .any(|pair| pair[1].abs_diff(pair[0]) <= 20);
            if !places.is_sorted() {
                random += 1;
            } else if near {
                clustered += 1;
            }
        }
        // A cluster can be empty when every offset drawn is 0, and the random
        // places can fall in order: both are rare.
        assert!((f64::from(clustered) / f64::from(trials) - 0.25).abs() < 0.03);
        assert!((f64::from(random) / f64::from(trials) - 0.25).abs() < 0.03);
    }

    #[test]
    fn magnitudes_stay_within_their_bounds() {
        let source = Anomaly::default();
        let mut rng = Rng::from_key(&[8]);
        for _ in 0..300 {
            let magnitudes = source.magnitudes(&mut rng, 30);
            let low = magnitudes.iter().copied().fold(f64::INFINITY, f64::min);
            let high = magnitudes.iter().copied().fold(0.0, f64::max);
            assert!(low >= 1.0 && high <= 150.0 && high / low <= 30.0 + 1e-9);
        }
    }
}
