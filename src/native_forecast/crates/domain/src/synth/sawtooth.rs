//! `sawtooth`: a sawtooth wave, rising or falling, with a slight linear trend
//! and a faint slow sinusoid.
//!
//! Port of TempoPFN's `sawtooth/sawtooth_generator.py` with the defaults of
//! `SawToothGeneratorParams`.

use core::f64::consts::TAU;

use crate::random::Rng;

/// The parameters of the source.
#[derive(Clone, Debug)]
pub struct Sawtooth {
    /// The range of the number of teeth in the series, not an integer.
    pub periods: (f64, f64),
    /// The range of the amplitude.
    pub amplitude: (f64, f64),
    /// The range of the phase, as a fraction of a tooth.
    pub phase: (f64, f64),
    /// The range of the slope of the trend, per step.
    pub trend_slope: (f64, f64),
    /// The range of the amplitude of the slow sinusoid.
    pub seasonality_amplitude: (f64, f64),
    /// The range of the number of cycles of the slow sinusoid in the series.
    pub seasonality_cycles: (f64, f64),
}

impl Default for Sawtooth {
    fn default() -> Self {
        Self {
            periods: (2.0, 7.0),
            amplitude: (0.5, 3.0),
            phase: (0.0, 1.0),
            trend_slope: (-0.001, 0.001),
            seasonality_amplitude: (0.0, 0.02),
            seasonality_cycles: (2.0, 4.0),
        }
    }
}

impl Sawtooth {
    /// One series of `length` steps.
    pub fn series(&self, rng: &mut Rng, length: usize) -> Vec<f64> {
        let amplitude = rng.uniform_in(self.amplitude.0, self.amplitude.1);
        let phase = rng.uniform_in(self.phase.0, self.phase.1);
        let tooth = length as f64 / rng.uniform_in(self.periods.0, self.periods.1);
        let season =
            length as f64 / rng.uniform_in(self.seasonality_cycles.0, self.seasonality_cycles.1);
        let falling = rng.chance(0.5);
        let slope = rng.uniform_in(self.trend_slope.0, self.trend_slope.1);
        let season_amplitude =
            rng.uniform_in(self.seasonality_amplitude.0, self.seasonality_amplitude.1);
        (0..length)
            .map(|step| {
                let t = step as f64;
                let cycles = t / tooth + phase;
                let ramp = cycles - cycles.floor();
                let wave = amplitude * if falling { 1.0 - ramp } else { ramp };
                wave + slope * t + season_amplitude * (TAU * t / season).sin()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_teeth_have_the_drawn_height_and_count() {
        let source = Sawtooth {
            trend_slope: (0.0, 0.0),
            seasonality_amplitude: (0.0, 0.0),
            ..Sawtooth::default()
        };
        for seed in 0..100 {
            let values = source.series(&mut Rng::from_key(&[seed]), 2048);
            let low = values.iter().copied().fold(f64::INFINITY, f64::min);
            let high = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            assert!(low >= 0.0 && high <= 3.0 && high - low > 0.45);
            // A tooth ends where the series jumps against its slope.
            let jumps = values
                .windows(2)
                .filter(|pair| (pair[1] - pair[0]).abs() > 0.5 * (high - low))
                .count();
            assert!((1..=7).contains(&jumps), "{jumps} teeth");
        }
    }
}
