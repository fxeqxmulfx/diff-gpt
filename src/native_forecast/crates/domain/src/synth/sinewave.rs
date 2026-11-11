//! `sinewave`: a sum of one to three sinusoids with slow amplitude and
//! frequency modulation, a linear trend, a level and, mostly, white noise.
//!
//! Port of TempoPFN's `sine_waves/sine_wave_generator.py` with the defaults of
//! `SineWaveGeneratorParams`. With both modulations enabled, as they are, the
//! source computes an amplitude-modulated sinusoid, replaces it with the
//! frequency-modulated one and modulates the amplitude of that with a second
//! period and phase; only the surviving operations are done here.

use core::f64::consts::TAU;

use crate::random::Rng;

/// The parameters of the source.
#[derive(Clone, Debug)]
pub struct SineWave {
    /// The least and greatest number of sinusoids.
    pub components: (i64, i64),
    /// The range of a sinusoid's period, in steps.
    pub period: (f64, f64),
    /// The range of a sinusoid's amplitude.
    pub amplitude: (f64, f64),
    /// The range of the slope of the trend, per step.
    pub trend_slope: (f64, f64),
    /// The range of the level.
    pub level: (f64, f64),
    /// The probability that the series has noise.
    pub noise_probability: f64,
    /// The range of the noise deviation, as a fraction of the summed amplitudes.
    pub noise_level: (f64, f64),
    /// The depth of the amplitude modulation.
    pub amplitude_modulation: f64,
    /// The depth of the frequency modulation.
    pub frequency_modulation: f64,
}

impl Default for SineWave {
    fn default() -> Self {
        Self {
            components: (1, 3),
            period: (10.0, 200.0),
            amplitude: (0.5, 3.0),
            trend_slope: (-0.01, 0.01),
            level: (0.0, 2.0),
            noise_probability: 0.7,
            noise_level: (0.05, 0.2),
            amplitude_modulation: 0.1,
            frequency_modulation: 0.05,
        }
    }
}

impl SineWave {
    /// One series of `length` steps.
    pub fn series(&self, rng: &mut Rng, length: usize) -> Vec<f64> {
        let mut values = vec![0.0; length];
        let mut total_amplitude = 0.0;
        for _ in 0..rng.integer(self.components.0, self.components.1 + 1) {
            let period = rng.uniform_in(self.period.0, self.period.1);
            let amplitude = rng.uniform_in(self.amplitude.0, self.amplitude.1);
            let phase = rng.angle();
            total_amplitude += amplitude;

            let frequency_period = period * rng.uniform_in(8.0, 15.0);
            let frequency_phase = rng.angle();
            let amplitude_period = period * rng.uniform_in(5.0, 10.0);
            let amplitude_phase = rng.angle();
            let mut accumulated = 0.0;
            for (step, value) in values.iter_mut().enumerate() {
                let t = step as f64;
                let deviation = self.frequency_modulation
                    * (TAU * t / frequency_period + frequency_phase).sin();
                accumulated += TAU / period * (1.0 + deviation);
                let envelope = 1.0
                    + self.amplitude_modulation
                        * (TAU * t / amplitude_period + amplitude_phase).sin();
                *value += amplitude * (accumulated + phase).sin() * envelope;
            }
        }

        let slope = rng.uniform_in(self.trend_slope.0, self.trend_slope.1);
        let level = rng.uniform_in(self.level.0, self.level.1);
        for (step, value) in values.iter_mut().enumerate() {
            *value += slope * step as f64 + level;
        }
        if rng.chance(self.noise_probability) {
            let deviation =
                rng.uniform_in(self.noise_level.0, self.noise_level.1) * total_amplitude;
            for value in &mut values {
                *value += rng.normal_with(0.0, deviation);
            }
        }
        values
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_plain_sinusoid_keeps_its_amplitude_and_period() {
        let plain = SineWave {
            components: (1, 1),
            period: (64.0, 64.0),
            amplitude: (2.0, 2.0),
            trend_slope: (0.0, 0.0),
            level: (0.0, 0.0),
            noise_probability: 0.0,
            amplitude_modulation: 0.0,
            frequency_modulation: 0.0,
            ..SineWave::default()
        };
        let values = plain.series(&mut Rng::from_key(&[1]), 512);
        let peak = values.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!((peak - 2.0).abs() < 0.02);
        for step in 0..448 {
            assert!((values[step] - values[step + 64]).abs() < 1e-9);
        }
    }

    #[test]
    fn the_default_series_stay_within_their_bounds() {
        let source = SineWave::default();
        for seed in 0..200 {
            let values = source.series(&mut Rng::from_key(&[seed]), 2048);
            assert_eq!(values.len(), 2048);
            // Three sinusoids of amplitude 3.3, the trend, the level and noise
            // of at most 0.2 of the amplitudes.
            let bound = 3.0 * 3.3 + 0.01 * 2048.0 + 2.0 + 9.0 * 0.2 * 6.0;
            assert!(values.iter().all(|v| v.is_finite() && v.abs() < bound));
        }
    }
}
