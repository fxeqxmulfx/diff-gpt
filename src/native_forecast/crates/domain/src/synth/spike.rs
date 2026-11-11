//! `spike`: a constant baseline with a few triangular or flat-topped spikes
//! of one sign and, half of the time, low-pass coloured noise.
//!
//! Port of TempoPFN's `spikes/spikes_generator.py` with the defaults of
//! `SpikesGeneratorParams`. The source draws a shape name for each series
//! whose direction it then ignores: only whether a spike has a plateau, and
//! the sign drawn for the series, reach the values.

use crate::numeric::{Complex, deviation, fft, fft_frequency, round_half_even};
use crate::random::Rng;

/// The parameters of the source.
#[derive(Clone, Debug)]
pub struct Spike {
    /// The range of the number of spikes of a burst, the upper end excluded.
    pub count_burst: (i64, i64),
    /// The range of the number of evenly spread spikes, the upper end excluded.
    pub count_spread: (i64, i64),
    /// The range of the height of the spikes of a series.
    pub amplitude: (f64, f64),
    /// The range of the angle of their flanks, in degrees.
    pub angle: (f64, f64),
    /// The probability that the spikes come in one burst.
    pub burst_probability: f64,
    /// The range of the length of a plateau, the upper end excluded.
    pub plateau: (i64, i64),
    /// The range of the baseline.
    pub baseline: (f64, f64),
    /// The range of the width of a burst, as a fraction of the series.
    pub burst_width: (f64, f64),
    /// The margin before the first spread spike, as a fraction of their spacing.
    pub edge_margin_ratio: f64,
    /// The probability that the spikes rise above the baseline.
    pub above_probability: f64,
    /// The weights of series with pointed spikes, flat-topped ones, or both.
    pub kinds: [f64; 3],
    /// The least width of a spike.
    pub min_width: usize,
    /// The greatest width of a spike.
    pub max_width: usize,
    /// The least gap between two spikes.
    pub min_margin: usize,
    /// The deviation of the noise.
    pub noise_deviation: f64,
    /// The probability that the series has noise.
    pub noise_probability: f64,
    /// The exponent of the power law of the noise spectrum: 2 is brown noise.
    pub noise_exponent: f64,
    /// The frequency above which the noise is damped, in cycles per step.
    pub noise_cutoff: f64,
}

impl Default for Spike {
    fn default() -> Self {
        Self {
            count_burst: (2, 4),
            count_spread: (4, 7),
            amplitude: (50.0, 300.0),
            angle: (70.0, 85.0),
            burst_probability: 0.05,
            plateau: (30, 50),
            baseline: (-200.0, 200.0),
            burst_width: (0.1, 0.25),
            edge_margin_ratio: 0.2,
            above_probability: 0.5,
            kinds: [0.4, 0.3, 0.3],
            min_width: 30,
            max_width: 100,
            min_margin: 10,
            noise_deviation: 2.0,
            noise_probability: 0.5,
            noise_exponent: 2.0,
            noise_cutoff: 0.1,
        }
    }
}

impl Spike {
    /// One series of `length` steps.
    ///
    /// # Panics
    ///
    /// If the series has noise and `length` is not a power of two.
    pub fn series(&self, rng: &mut Rng, length: usize) -> Vec<f64> {
        let mut values = vec![rng.uniform_in(self.baseline.0, self.baseline.1); length];
        let kind = rng.weighted(&self.kinds);
        let burst = rng.chance(self.burst_probability);
        let noisy = rng.chance(self.noise_probability);
        let (low, high) = if burst {
            self.count_burst
        } else {
            self.count_spread
        };
        let count = rng.integer(low, high) as usize;
        let amplitude = rng.uniform_in(self.amplitude.0, self.amplitude.1);
        let angle = rng.uniform_in(self.angle.0, self.angle.1);
        let above = rng.chance(self.above_probability);

        for position in self.positions(rng, count, burst, length) {
            let flat_topped = match kind {
                0 => false,
                1 => true,
                _ => rng.chance(0.5),
            };
            let spike = self.spike(rng, amplitude, angle, flat_topped, above);
            inject(&mut values, &spike, position);
        }
        if noisy && self.noise_deviation > 0.0 {
            for (value, noise) in values.iter_mut().zip(self.noise(rng, length)) {
                *value += noise;
            }
        }
        values
    }

    /// The least distance between the centres of two spikes.
    const fn separation(&self) -> usize {
        self.max_width + self.min_margin
    }

    /// The centres of the spikes.
    fn positions(&self, rng: &mut Rng, count: usize, burst: bool, length: usize) -> Vec<usize> {
        if count == 0 {
            return Vec::new();
        }
        let margin = self.max_width;
        let usable = length.saturating_sub(2 * margin);
        let count = count.min((usable / self.separation()).max(1));
        let positions = if burst {
            let drawn = rng.uniform_in(self.burst_width.0, self.burst_width.1) * length as f64;
            let width = (count * self.separation()).max(drawn as usize).min(usable);
            let last_start = (length - width - margin + 1).max(margin + 1);
            let start = rng.integer(margin as i64, last_start as i64) as usize;
            self.burst_positions(rng, count, start, start + width)
        } else {
            self.spread_positions(count, margin, length - margin)
        };
        self.separated(positions, margin, length)
    }

    /// Evenly spaced centres in `[start, end]`, the margins at both ends a
    /// fraction of the spacing.
    fn spread_positions(&self, count: usize, start: usize, end: usize) -> Vec<usize> {
        if count <= 1 {
            return vec![usize::midpoint(start, end)];
        }
        let spacing = (end - start) as f64 / ((count - 1) as f64 + 2.0 * self.edge_margin_ratio);
        if spacing < self.separation() as f64 {
            return self.packed_positions(count, start, end);
        }
        let mut positions: Vec<usize> = (0..count)
            .map(|index| {
                let place = start as f64 + (self.edge_margin_ratio + index as f64) * spacing;
                (round_half_even(place) as usize).clamp(start, end)
            })
            .collect();
        positions.sort_unstable();
        positions
    }

    /// Jittered centres from `start` to `end`.
    fn burst_positions(&self, rng: &mut Rng, count: usize, start: usize, end: usize) -> Vec<usize> {
        if count <= 1 {
            return vec![usize::midpoint(start, end)];
        }
        if end - start < (count - 1) * self.separation() {
            return self.packed_positions(count, start, end);
        }
        let interval = (end - start) as f64 / (count - 1) as f64;
        let jitter = (interval * 0.2).min(self.separation() as f64 * 0.3);
        (0..count)
            .map(|index| {
                let place =
                    start as f64 + index as f64 * interval + rng.uniform_in(-jitter, jitter);
                (place as usize).clamp(start, end)
            })
            .collect()
    }

    /// Centres at the least distance from `start` on, as many as fit.
    fn packed_positions(&self, count: usize, start: usize, end: usize) -> Vec<usize> {
        (0..count)
            .map(|index| start + index * self.separation())
            .take_while(|&position| position <= end)
            .collect()
    }

    /// Moves the centres apart to the least distance and drops those that no
    /// longer fit.
    fn separated(&self, mut positions: Vec<usize>, margin: usize, length: usize) -> Vec<usize> {
        if positions.len() <= 1 {
            return positions;
        }
        positions.sort_unstable();
        let mut kept = vec![positions[0].max(margin)];
        for &position in &positions[1..] {
            let next = position.max(kept[kept.len() - 1] + self.separation());
            if next + margin > length {
                break;
            }
            kept.push(next);
        }
        kept
    }

    /// One spike: a rise, a plateau if it is flat-topped, and a fall.
    fn spike(
        &self,
        rng: &mut Rng,
        amplitude: f64,
        angle: f64,
        flat_topped: bool,
        above: bool,
    ) -> Vec<f64> {
        let slope = angle.to_radians().tan();
        let rise = (round_half_even(amplitude / slope) as usize)
            .clamp(self.min_width / 2, self.max_width / 2);
        let plateau = if flat_topped {
            rng.integer(self.plateau.0, self.plateau.1) as usize
        } else {
            0
        };
        let height = if above { amplitude } else { -amplitude };
        // `linspace(…, endpoint=False)`: the rise stops one step short of the
        // height and the fall one step short of the baseline.
        let step = height / rise as f64;
        let mut spike: Vec<f64> = (0..rise).map(|index| index as f64 * step).collect();
        spike.extend(core::iter::repeat_n(height, plateau));
        spike.extend((0..rise).map(|index| height - index as f64 * step));
        spike
    }

    /// Noise with the spectrum `1/f^(exponent/2)` in amplitude, damped above
    /// the cutoff, scaled to the deviation of the source.
    fn noise(&self, rng: &mut Rng, length: usize) -> Vec<f64> {
        let mut spectrum: Vec<Complex> = (0..length).map(|_| (rng.normal(), 0.0)).collect();
        fft(&mut spectrum, false);
        for (bin, value) in spectrum.iter_mut().enumerate() {
            // The source gives the mean the gain of the lowest frequency.
            let frequency = fft_frequency(bin.max(1), length).abs();
            let mut gain = frequency.powf(-self.noise_exponent / 2.0);
            if frequency > self.noise_cutoff {
                gain *= (-((frequency - self.noise_cutoff) / self.noise_cutoff).powi(2)).exp();
            }
            *value = (value.0 * gain, value.1 * gain);
        }
        fft(&mut spectrum, true);
        let noise: Vec<f64> = spectrum.into_iter().map(|(re, _)| re).collect();
        let scale = self.noise_deviation / deviation(&noise);
        noise.into_iter().map(|value| value * scale).collect()
    }
}

/// Adds `spike` to `values`, its middle at `position`, cut at the ends.
fn inject(values: &mut [f64], spike: &[f64], position: usize) {
    let half = spike.len() / 2;
    let start = position.saturating_sub(half);
    let end = (position + half + spike.len() % 2).min(values.len());
    let skipped = half.saturating_sub(position);
    if end > start {
        for (value, bump) in values[start..end].iter_mut().zip(&spike[skipped..]) {
            *value += bump;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spread_spikes_are_evenly_spaced_and_of_one_sign() {
        let source = Spike {
            burst_probability: 0.0,
            noise_probability: 0.0,
            ..Spike::default()
        };
        for seed in 0..200 {
            let values = source.series(&mut Rng::from_key(&[seed]), 2048);
            let baseline = values[0];
            let above = values.iter().any(|&v| v > baseline + 1e-9);
            let below = values.iter().any(|&v| v < baseline - 1e-9);
            assert!(above != below, "spikes of both signs or none");
            let height = values
                .iter()
                .fold(0.0_f64, |m, v| m.max((v - baseline).abs()));
            assert!((50.0..=300.0).contains(&height));
            // Each spike is one run of steps off the baseline.
            let runs = values
                .windows(2)
                .filter(|pair| pair[0] == baseline && pair[1] != baseline)
                .count();
            assert!((4..=6).contains(&runs), "{runs} spikes");
        }
    }

    #[test]
    fn positions_keep_their_distance() {
        let source = Spike::default();
        let mut rng = Rng::from_key(&[9]);
        for _ in 0..2000 {
            let burst = rng.chance(0.5);
            let count = if burst {
                rng.integer(2, 4)
            } else {
                rng.integer(4, 7)
            } as usize;
            let positions = source.positions(&mut rng, count, burst, 2048);
            assert!(!positions.is_empty() && positions.len() <= count);
            assert!(positions.iter().all(|&p| (100..=1948).contains(&p)));
            assert!(positions.windows(2).all(|pair| pair[1] - pair[0] >= 110));
        }
    }

    #[test]
    fn a_spike_has_the_shape_of_the_source() {
        let source = Spike::default();
        let mut rng = Rng::from_key(&[2]);
        // tan(75°) = 3.732: a rise of round(100 / 3.732) = 27 steps.
        let pointed = source.spike(&mut rng, 100.0, 75.0, false, true);
        assert_eq!(pointed.len(), 54);
        assert_eq!(pointed[0], 0.0);
        assert_eq!(pointed[27], 100.0);
        assert!((pointed[53] - 100.0 / 27.0).abs() < 1e-9);
        let flat = source.spike(&mut rng, 100.0, 75.0, true, false);
        assert!((54 + 30..54 + 50).contains(&flat.len()));
        assert!(flat.iter().filter(|&&v| v == -100.0).count() >= 31);
        // A steep, low spike is widened to the least width.
        assert_eq!(source.spike(&mut rng, 50.0, 85.0, false, true).len(), 30);
    }

    #[test]
    fn noise_is_low_pass_with_the_set_deviation() {
        let source = Spike::default();
        let noise = source.noise(&mut Rng::from_key(&[4]), 2048);
        assert!((deviation(&noise) - 2.0).abs() < 1e-9);
        let steps: Vec<f64> = noise.windows(2).map(|pair| pair[1] - pair[0]).collect();
        // Brown noise moves little from one step to the next.
        assert!(deviation(&steps) < 0.2 * deviation(&noise));
    }

    #[test]
    fn injection_is_cut_at_the_ends() {
        let mut values = vec![0.0; 6];
        inject(&mut values, &[1.0, 2.0, 3.0, 4.0, 5.0], 1);
        assert_eq!(values, [2.0, 3.0, 4.0, 5.0, 0.0, 0.0]);
        let mut values = vec![0.0; 6];
        inject(&mut values, &[1.0, 2.0, 3.0, 4.0], 5);
        assert_eq!(values, [0.0, 0.0, 0.0, 1.0, 2.0, 3.0]);
    }
}
