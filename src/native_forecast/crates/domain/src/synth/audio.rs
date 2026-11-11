//! TempoPFN's four audio priors (`src/synthetic_generation/audio_generators`):
//! graphs of pyo oscillators, noises, envelopes and filters, of which the
//! first 46 ms are kept (see [`super::pyo`]). The parameters are those of
//! `generator_params.py`; each is drawn uniformly in its range.
//!
//! In so short a window a metronome of a period above 46 ms fires once, at
//! the first sample, and an oscillator below 20 Hz is a piece of its ramp:
//! the names of the parts are TempoPFN's, whatever is left of their meaning.

use super::pyo::{Biquad, Brown, Expseg, Follower, Lfo, Metro, Sine, record, white};
use crate::random::Rng;

/// A range of a uniform draw.
type Range = (f64, f64);

/// A draw in the range.
fn within(rng: &mut Rng, range: Range) -> f64 {
    rng.uniform_in(range.0, range.1)
}

/// `FinancialVolatilityAudioGenerator`: a slow saw, brown noise whose level
/// follows a slow sine, and a decaying sine at the start.
#[derive(Clone, Debug)]
pub struct FinancialVolatility {
    /// The frequency of the trend, in hertz.
    pub trend_frequency: Range,
    /// The amplitude of the trend.
    pub trend_mul: Range,
    /// The frequency of the sine that the volatility follows.
    pub carrier_frequency: Range,
    /// The cutoff of the follower.
    pub follower_frequency: Range,
    /// The range the follower's output is mapped to, from ±1.
    pub volatility: Range,
    /// The period of the jumps, in seconds.
    pub jump_time: Range,
    /// The height of a jump's envelope at its start.
    pub jump_start: Range,
    /// The time a jump's envelope takes to fall to 0, in seconds.
    pub jump_decay: Range,
    /// The frequency of the sine of a jump.
    pub jump_frequency: Range,
    /// The probability that the jumps go up.
    pub jump_up: f64,
}

impl Default for FinancialVolatility {
    fn default() -> Self {
        Self {
            trend_frequency: (0.1, 0.5),
            trend_mul: (0.2, 0.5),
            carrier_frequency: (1.0, 5.0),
            follower_frequency: (1.0, 4.0),
            volatility: (0.1, 0.8),
            jump_time: (0.3, 1.0),
            jump_start: (0.5, 1.0),
            jump_decay: (0.05, 0.2),
            jump_frequency: (20.0, 80.0),
            jump_up: 0.5,
        }
    }
}

impl FinancialVolatility {
    /// A series of `length` steps.
    #[must_use]
    pub fn series(&self, rng: &mut Rng, length: usize) -> Vec<f64> {
        let trend_frequency = within(rng, self.trend_frequency);
        let mut trend = Lfo::new(trend_frequency, false, within(rng, self.trend_mul));
        let mut carrier = Sine::new(within(rng, self.carrier_frequency));
        let mut follower = Follower::new(within(rng, self.follower_frequency));
        // `range(min, max)` of a signal taken to lie in ±1.
        let level_mul = ((self.volatility.1 - self.volatility.0) * 0.5) as f32;
        let level_add = ((self.volatility.1 + self.volatility.0) * 0.5) as f32;
        let mut noise = Brown::default();
        let mut trigger = Metro::new(within(rng, self.jump_time));
        let start = within(rng, self.jump_start);
        let mut envelope = Expseg::new(&[(0.0, start), (within(rng, self.jump_decay), 0.0)]);
        let mut jump = Sine::new(within(rng, self.jump_frequency));
        let direction: f32 = if rng.uniform() < self.jump_up {
            1.0
        } else {
            -1.0
        };
        record(length, || {
            let level = level_mul * follower.step(carrier.step()) + level_add;
            let market = level * noise.step(rng);
            let jumps = (direction * envelope.step(trigger.step())) * jump.step();
            (trend.step() + market) + jumps
        })
    }
}

/// `MultiScaleFractalAudioGenerator`: brown noise through three to six
/// low-passes of falling cutoffs, summed with falling weights.
#[derive(Clone, Debug)]
pub struct MultiScaleFractal {
    /// The amplitude of the noise.
    pub base_mul: Range,
    /// The least and the most scales.
    pub scales: (i64, i64),
    /// The cutoff of the first scale, in hertz; scale `i` has half of it `i`
    /// times.
    pub scale_frequency: Range,
    /// The Q of a low-pass.
    pub q: Range,
    /// The attenuation of a scale, raised to its index.
    pub attenuation: Range,
}

impl Default for MultiScaleFractal {
    fn default() -> Self {
        Self {
            base_mul: (0.3, 0.8),
            scales: (3, 6),
            scale_frequency: (20.0, 2000.0),
            q: (0.5, 3.0),
            attenuation: (0.5, 0.8),
        }
    }
}

impl MultiScaleFractal {
    /// A series of `length` steps.
    #[must_use]
    pub fn series(&self, rng: &mut Rng, length: usize) -> Vec<f64> {
        let base_mul = within(rng, self.base_mul) as f32;
        let mut base = Brown::default();
        let count = rng.integer(self.scales.0, self.scales.1 + 1);
        let mut scales: Vec<(Biquad, f32)> = (0..count as i32)
            .map(|index| {
                let frequency = within(rng, self.scale_frequency) * 0.5_f64.powi(index);
                let q = within(rng, self.q);
                let weight = within(rng, self.attenuation).powi(index);
                (Biquad::new(frequency, q), weight as f32)
            })
            .collect();
        record(length, || {
            let noise = base_mul * base.step(rng);
            scales.iter_mut().fold(0.0, |total, (filter, weight)| {
                *weight * filter.step(noise) + total
            })
        })
    }
}

/// `NetworkTopologyAudioGenerator`: a slow saw, bursts of white noise, a
/// negative level that recovers, a fast falling saw, and brown noise under
/// an envelope.
#[derive(Clone, Debug)]
pub struct NetworkTopology {
    /// The frequency of the traffic, in hertz.
    pub traffic_frequency: Range,
    /// The amplitude of the traffic.
    pub traffic_mul: Range,
    /// The rate of the bursts, in hertz.
    pub burst_rate: Range,
    /// The time a burst takes to fall from 0.8 to 0, in seconds.
    pub burst_duration: Range,
    /// The amplitude of the bursts.
    pub burst_mul: Range,
    /// The period of the congestion, in seconds.
    pub congestion_period: Range,
    /// The level of a congestion at its start; negative.
    pub congestion_depth: Range,
    /// The time a congestion takes to rise to 0, in seconds.
    pub congestion_release: Range,
    /// The frequency of the overhead, in hertz.
    pub overhead_frequency: Range,
    /// The amplitude of the overhead.
    pub overhead_mul: Range,
    /// The period of the attacks, in seconds.
    pub attack_period: Range,
    /// The envelope of an attack: seconds and values.
    pub attack_envelope: [(f64, f64); 3],
    /// The amplitude of the attacks.
    pub attack_mul: Range,
}

impl Default for NetworkTopology {
    fn default() -> Self {
        Self {
            traffic_frequency: (0.2, 1.0),
            traffic_mul: (0.2, 0.5),
            burst_rate: (3.0, 12.0),
            burst_duration: (0.02, 0.1),
            burst_mul: (0.2, 0.6),
            congestion_period: (1.0, 3.0),
            congestion_depth: (-0.6, -0.2),
            congestion_release: (0.3, 0.8),
            overhead_frequency: (20.0, 50.0),
            overhead_mul: (0.05, 0.15),
            attack_period: (2.0, 5.0),
            attack_envelope: [(0.0, 1.2), (0.1, 0.8), (0.8, 0.0)],
            attack_mul: (0.4, 0.8),
        }
    }
}

impl NetworkTopology {
    /// A series of `length` steps.
    #[must_use]
    pub fn series(&self, rng: &mut Rng, length: usize) -> Vec<f64> {
        let traffic_frequency = within(rng, self.traffic_frequency);
        let mut traffic = Lfo::new(traffic_frequency, false, within(rng, self.traffic_mul));
        let mut burst_trigger = Metro::new(1.0 / within(rng, self.burst_rate));
        let mut burst = Expseg::new(&[(0.0, 0.8), (within(rng, self.burst_duration), 0.0)]);
        let burst_mul = within(rng, self.burst_mul) as f32;
        let mut congestion_trigger = Metro::new(within(rng, self.congestion_period));
        let depth = within(rng, self.congestion_depth);
        let mut congestion =
            Expseg::new(&[(0.0, depth), (within(rng, self.congestion_release), 0.0)]);
        let overhead_frequency = within(rng, self.overhead_frequency);
        let mut overhead = Lfo::new(overhead_frequency, true, within(rng, self.overhead_mul));
        let mut attack_trigger = Metro::new(within(rng, self.attack_period));
        let mut attack = Expseg::new(&self.attack_envelope);
        let attack_mul = within(rng, self.attack_mul) as f32;
        let mut attack_noise = Brown::default();
        record(length, || {
            let bursts = (burst_mul * burst.step(burst_trigger.step())) * white(rng);
            let congested = congestion.step(congestion_trigger.step());
            let attacks =
                (attack_mul * attack.step(attack_trigger.step())) * attack_noise.step(rng);
            (((traffic.step() + bursts) + congested) + overhead.step()) + attacks
        })
    }
}

/// `StochasticRhythmAudioGenerator`: three to five sines, each under an
/// envelope that a metronome at a multiple of one tempo starts anew.
#[derive(Clone, Debug)]
pub struct StochasticRhythm {
    /// The tempo, in hertz.
    pub tempo: Range,
    /// The least and the most layers.
    pub layers: (i64, i64),
    /// The multiples of the tempo a layer may beat at.
    pub subdivisions: [f64; 6],
    /// The time an envelope takes to fall from 1 to 0.8, in seconds.
    pub attack: Range,
    /// The time at which an envelope reaches 0, in seconds.
    pub decay: Range,
    /// The frequency of a layer's sine, in hertz.
    pub tone_frequency: Range,
    /// The amplitude of a layer.
    pub tone_mul: Range,
}

impl Default for StochasticRhythm {
    fn default() -> Self {
        Self {
            tempo: (2.0, 8.0),
            layers: (3, 5),
            subdivisions: [1.0, 2.0, 3.0, 4.0, 6.0, 8.0],
            attack: (0.001, 0.01),
            decay: (0.05, 0.3),
            tone_frequency: (50.0, 800.0),
            tone_mul: (0.2, 0.5),
        }
    }
}

impl StochasticRhythm {
    /// A series of `length` steps.
    #[must_use]
    pub fn series(&self, rng: &mut Rng, length: usize) -> Vec<f64> {
        let tempo = within(rng, self.tempo);
        let count = rng.integer(self.layers.0, self.layers.1 + 1);
        let mut layers: Vec<(Metro, Expseg, Sine, f32)> = (0..count)
            .map(|_| {
                let rhythm = tempo * *rng.choose(&self.subdivisions);
                let attack = within(rng, self.attack);
                let decay = within(rng, self.decay);
                let envelope = Expseg::new(&[(0.0, 1.0), (attack, 0.8), (decay, 0.0)]);
                let tone = Sine::new(within(rng, self.tone_frequency));
                let mul = within(rng, self.tone_mul) as f32;
                (Metro::new(1.0 / rhythm), envelope, tone, mul)
            })
            .collect();
        record(length, || {
            layers
                .iter_mut()
                .fold(0.0, |total, (trigger, envelope, tone, mul)| {
                    (*mul * envelope.step(trigger.step())) * tone.step() + total
                })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The checks every audio series passes: it starts at 0 under the
    /// fade-in, and its greatest absolute value is 1.
    fn normalized(series: &[f64]) {
        assert_eq!(series[0], 0.0);
        let top = series
            .iter()
            .fold(0.0_f64, |top, value| top.max(value.abs()));
        assert!((top - 1.0).abs() < 1e-12, "the greatest value is {top}");
    }

    #[test]
    fn the_market_has_its_trend_its_noise_and_its_jump() {
        let mut rng = Rng::from_key(&[1]);
        let prior = FinancialVolatility::default();
        let mut negative = 0;
        for _ in 0..200 {
            let series = prior.series(&mut rng, 2048);
            normalized(&series);
            // The saw falls from 0 along its hyperbolic tangent.
            negative += usize::from(series[1024..].iter().sum::<f64>() < 0.0);
        }
        assert!(negative > 120, "{negative} of 200 series end below 0");
    }

    #[test]
    fn the_fractal_noise_is_smooth() {
        let mut rng = Rng::from_key(&[2]);
        let prior = MultiScaleFractal::default();
        for _ in 0..100 {
            let series = prior.series(&mut rng, 2048);
            normalized(&series);
            // Low-passed brown noise: the steps are small next to the range.
            let step = series
                .windows(2)
                .fold(0.0_f64, |top, pair| top.max((pair[1] - pair[0]).abs()));
            assert!(step < 0.2, "a step of {step}");
        }
    }

    #[test]
    fn the_network_carries_a_fast_saw_and_starts_congested() {
        let mut rng = Rng::from_key(&[3]);
        let prior = NetworkTopology::default();
        let mut below = 0;
        for _ in 0..200 {
            let series = prior.series(&mut rng, 2048);
            normalized(&series);
            // The congestion holds the level negative over the window.
            below += usize::from(series[441..].iter().sum::<f64>() < 0.0);
        }
        assert!(below > 150, "{below} of 200 series lie below 0");
    }

    #[test]
    fn the_rhythm_oscillates_under_its_envelopes() {
        let mut rng = Rng::from_key(&[4]);
        let prior = StochasticRhythm::default();
        for _ in 0..100 {
            let series = prior.series(&mut rng, 2048);
            normalized(&series);
            let turns = series
                .windows(3)
                .filter(|w| (w[1] - w[0]) * (w[2] - w[1]) < 0.0)
                .count();
            // Sines of 50 to 800 Hz over 46 ms: 2 to 37 periods each.
            assert!((3..400).contains(&turns), "{turns} turns");
        }
    }
}
