//! The objects of pyo 1.1.0 (<https://github.com/belangeo/pyo>, LGPL-3.0)
//! that TempoPFN's audio priors wire together, each a sample at a time in
//! the float32 of pyo's default build, and the recording that TempoPFN keeps.
//!
//! pyo renders a graph in buffers, every object in the order of its creation,
//! so an object reads the samples its inputs made in the same buffer: for
//! these feed-forward graphs that is the same as one sample of every object
//! in turn. The noises draw from pyo's one generator, uniform numbers here
//! from the stream of the series.
//!
//! TempoPFN renders two seconds at 44.1 kHz and keeps the first 2048 samples,
//! 46 ms: a slow oscillator is a ramp there, and the recorder's fade-in of
//! 10 ms covers the first 441 samples.

use crate::random::Rng;

/// The sampling rate of the offline server.
pub const RATE: f64 = 44_100.0;

/// pyo's `MYFLT` rate.
const RATE_SINGLE: f32 = 44_100.0;

/// `LFO` of type 0 (a rising saw) or 1 (a falling one) at the default
/// sharpness 0.5: the ramp less a hyperbolic tangent that rounds its jump.
#[derive(Clone, Debug)]
pub struct Lfo {
    position: f32,
    increment: f32,
    harmonics: f32,
    falling: bool,
    mul: f32,
}

impl Lfo {
    /// The oscillator of that frequency in hertz, scaled by `mul`.
    #[must_use]
    pub fn new(frequency: f64, falling: bool, mul: f64) -> Self {
        let frequency = (frequency as f32).clamp(0.00001, RATE_SINGLE * 0.25);
        let most = (RATE_SINGLE * 0.25 / frequency).floor();
        Self {
            position: 0.0,
            increment: frequency / RATE_SINGLE,
            harmonics: (0.5_f32 * 46.0 + 4.0).min(most),
            falling,
            mul: mul as f32,
        }
    }

    /// The sample of this step.
    pub fn step(&mut self) -> f32 {
        let mut pointer = self.position + 0.5;
        if pointer >= 1.0 {
            pointer -= 1.0;
        }
        let pointer = pointer * 2.0 - 1.0;
        let value = pointer - (self.harmonics * pointer).tanh() / self.harmonics.tanh();
        self.position += self.increment;
        if self.position < 0.0 {
            self.position += 1.0;
        } else if self.position >= 1.0 {
            self.position -= 1.0;
        }
        self.mul * if self.falling { -value } else { value }
    }
}

/// `Sine`: a table of 512 steps of one period, read with linear
/// interpolation from the phase 0.
#[derive(Clone, Debug)]
pub struct Sine {
    position: f32,
    increment: f32,
}

impl Sine {
    /// The oscillator of that frequency in hertz.
    #[must_use]
    pub fn new(frequency: f64) -> Self {
        Self {
            position: 0.0,
            increment: frequency as f32 * 512.0 / RATE_SINGLE,
        }
    }

    /// The entry of pyo's table.
    fn table(index: usize) -> f32 {
        (core::f64::consts::TAU * (index % 512) as f64 / 512.0).sin() as f32
    }

    /// The sample of this step.
    pub fn step(&mut self) -> f32 {
        if self.position >= 512.0 {
            self.position -= (self.position / 512.0).floor() * 512.0;
        }
        let whole = self.position as usize;
        let (low, high) = (Self::table(whole), Self::table(whole + 1));
        let value = low + (high - low) * (self.position - whole as f32);
        self.position += self.increment;
        value
    }
}

/// `Noise`: uniform in ±0.99.
pub fn white(rng: &mut Rng) -> f32 {
    rng.uniform() as f32 * 1.98 - 0.99
}

/// `BrownNoise`: white noise through a one-pole low-pass at 20 Hz, times 20.
#[derive(Clone, Debug)]
pub struct Brown {
    state: f32,
    coefficient: f32,
}

impl Default for Brown {
    fn default() -> Self {
        let b = (2.0 - f64::from((core::f32::consts::TAU * 20.0 / RATE_SINGLE).cos())) as f32;
        Self {
            state: 0.0,
            coefficient: (f64::from(b) - f64::from((b * b - 1.0).sqrt())) as f32,
        }
    }
}

impl Brown {
    /// The sample of this step.
    pub fn step(&mut self, rng: &mut Rng) -> f32 {
        let noise = white(rng);
        self.state = noise + (self.state - noise) * self.coefficient;
        self.state * 20.0
    }
}

/// `Follower`: the absolute value of its input through a one-pole low-pass.
#[derive(Clone, Debug)]
pub struct Follower {
    follow: f32,
    factor: f32,
}

impl Follower {
    /// The follower with that cutoff in hertz.
    #[must_use]
    pub fn new(frequency: f64) -> Self {
        let frequency = (frequency as f32).max(0.0);
        Self {
            follow: 0.0,
            factor: (-core::f32::consts::TAU * frequency / RATE_SINGLE).exp(),
        }
    }

    /// The sample of this step for that input.
    pub fn step(&mut self, input: f32) -> f32 {
        let level = input.abs();
        self.follow = level + self.factor * (self.follow - level);
        self.follow
    }
}

/// `Metro`: a trigger at the first sample, then one sample after every
/// `time` seconds.
#[derive(Clone, Debug)]
pub struct Metro {
    time: f64,
    current: f64,
    armed: bool,
}

impl Metro {
    /// The metronome of that period in seconds.
    #[must_use]
    pub const fn new(time: f64) -> Self {
        Self {
            time,
            current: 0.0,
            armed: true,
        }
    }

    /// Whether this step is a trigger.
    pub fn step(&mut self) -> bool {
        let fired = if self.current >= self.time {
            self.current -= self.time;
            self.armed = true;
            false
        } else if self.armed {
            self.armed = false;
            true
        } else {
            false
        };
        self.current += 1.0 / RATE;
        fired
    }
}

/// `TrigExpseg` with the default exponent 10 and `inverse`: an envelope
/// through the points `(seconds, value)`, started anew by each trigger. A
/// falling segment follows `1 − (1 − t)¹⁰`, steep at its start; a rising one
/// `t¹⁰`, steep at its end.
#[derive(Clone, Debug)]
pub struct Expseg {
    points: Vec<(f32, f32)>,
    running: bool,
    time: f64,
    value: f64,
    which: usize,
    range: f32,
    pointer: f64,
    increment: f64,
}

impl Expseg {
    /// The exponent of the segments.
    const EXPONENT: f64 = 10.0;

    /// The envelope through those points, of which there are at least two,
    /// the first at time 0.
    #[must_use]
    pub fn new(points: &[(f64, f64)]) -> Self {
        Self {
            points: points
                .iter()
                .map(|(time, value)| (*time as f32, *value as f32))
                .collect(),
            running: false,
            time: 0.0,
            value: 0.0,
            which: 0,
            range: 0.0,
            pointer: 0.0,
            increment: 0.0,
        }
    }

    /// The sample of this step; `trigger` starts the envelope anew.
    pub fn step(&mut self, trigger: bool) -> f32 {
        let Some((last_time, last_value)) = self.points.last().copied() else {
            return 0.0;
        };
        if trigger {
            self.time = 0.0;
            self.value = f64::from(self.points[0].1);
            self.which = 0;
            self.running = true;
        }
        if !self.running {
            return self.value as f32;
        }
        if self.time >= f64::from(self.points[self.which].0) {
            self.which += 1;
            if self.which == self.points.len() {
                self.running = false;
                self.value = f64::from(last_value);
            } else {
                let (from, to) = (self.points[self.which - 1], self.points[self.which]);
                self.range = to.1 - from.1;
                let steps = f64::from(to.0 - from.0) * RATE;
                self.increment = if steps <= 0.0 { 1.0 } else { 1.0 / steps };
                self.pointer = 0.0;
            }
        }
        // pyo also comes here on the sample that ends the last segment, if
        // its time is the last time exactly.
        if self.time <= f64::from(last_time) {
            self.pointer = self.pointer.min(1.0);
            let scale = if self.range < 0.0 {
                1.0 - (1.0 - self.pointer).powf(Self::EXPONENT)
            } else {
                self.pointer.powf(Self::EXPONENT)
            } as f32;
            let from = self.points[self.which.saturating_sub(1)].1;
            self.value = f64::from(scale * self.range + from);
            self.pointer += self.increment;
        }
        self.time += f64::from(1.0 / RATE_SINGLE);
        self.value as f32
    }
}

/// `Biquad` of type 0: the low-pass of the audio EQ cookbook, its memories
/// set to the first input.
#[derive(Clone, Debug)]
pub struct Biquad {
    started: bool,
    inputs: [f32; 2],
    outputs: [f32; 2],
    forward: [f32; 3],
    backward: [f32; 2],
    gain: f32,
}

impl Biquad {
    /// The low-pass of that cutoff in hertz and that Q.
    #[must_use]
    pub fn new(frequency: f64, q: f64) -> Self {
        let frequency = (frequency as f32).clamp(1.0, RATE_SINGLE * 0.5);
        let q = (q as f32).max(0.1);
        let angle = frequency * (core::f32::consts::TAU / RATE_SINGLE);
        let (cosine, alpha) = (angle.cos(), angle.sin() / (2.0 * q));
        Self {
            started: false,
            inputs: [0.0; 2],
            outputs: [0.0; 2],
            forward: [(1.0 - cosine) / 2.0, 1.0 - cosine, (1.0 - cosine) / 2.0],
            backward: [-2.0 * cosine, 1.0 - alpha],
            gain: 1.0 / (1.0 + alpha),
        }
    }

    /// The sample of this step for that input.
    pub fn step(&mut self, input: f32) -> f32 {
        if !self.started {
            self.inputs = [input; 2];
            self.outputs = [input; 2];
            self.started = true;
        }
        let value = (self.forward[0] * input
            + self.forward[1] * self.inputs[0]
            + self.forward[2] * self.inputs[1]
            - self.backward[0] * self.outputs[0]
            - self.backward[1] * self.outputs[1])
            * self.gain;
        self.outputs = [value, self.outputs[0]];
        self.inputs = [input, self.inputs[0]];
        value
    }
}

/// What TempoPFN keeps of a rendering: `length` samples of the graph through
/// `TableRec(…, fadetime=0.01)`, which fades in over 441 samples, divided by
/// their greatest absolute value unless that is 0.
#[must_use]
pub fn record(length: usize, mut graph: impl FnMut() -> f32) -> Vec<f64> {
    let fade = (0.01 * RATE).floor() as f32;
    let samples: Vec<f32> = (0..length)
        .map(|index| {
            let gain = if (index as f32) < fade {
                index as f32 / fade
            } else {
                1.0
            };
            graph() * gain
        })
        .collect();
    let greatest = samples
        .iter()
        .fold(0.0_f64, |top, value| top.max(f64::from(value.abs())));
    samples
        .into_iter()
        .map(|value| {
            if greatest > 0.0 {
                f64::from(value) / greatest
            } else {
                f64::from(value)
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_slow_saw_starts_at_zero_and_leaves_along_the_tangent() {
        // One period of 441 samples: the ramp from 0, less tanh(27 x).
        let mut rising = Lfo::new(100.0, false, 1.0);
        let mut falling = Lfo::new(100.0, true, 0.5);
        let samples: Vec<f32> = (0..441).map(|_| rising.step()).collect();
        assert_eq!(samples[0], 0.0);
        for (index, sample) in samples.iter().enumerate().take(220) {
            let pointer = 2.0 * index as f64 / 441.0;
            let expected = pointer - (27.0 * pointer).tanh();
            assert!(
                (f64::from(*sample) - expected).abs() < 1e-4,
                "sample {index}"
            );
        }
        // The jump is in the middle of the period: from −1 + x up to 1 − x.
        assert!(samples[110] < -0.4 && samples[330] > 0.4);
        assert!((f64::from(falling.step())).abs() < 1e-9);
        assert!((f64::from(falling.step()) + 0.5 * f64::from(samples[1])).abs() < 1e-7);
    }

    #[test]
    fn the_sine_is_a_sine() {
        let mut sine = Sine::new(441.0);
        for index in 0..1000 {
            let expected = (core::f64::consts::TAU * f64::from(index) / 100.0).sin();
            assert!(
                (f64::from(sine.step()) - expected).abs() < 2e-4,
                "sample {index}"
            );
        }
    }

    #[test]
    fn the_noises_have_their_spread() {
        let mut rng = Rng::from_key(&[1]);
        let draws = 200_000;
        let mut square = 0.0;
        for _ in 0..draws {
            let value = f64::from(white(&mut rng));
            assert!((-0.99..0.99).contains(&value));
            square += value * value / f64::from(draws);
        }
        assert!((square - 0.99 * 0.99 / 3.0).abs() < 0.005);
        // The low-pass leaves the deviation σ √((1 − c) / (1 + c)) · 20.
        let mut brown = Brown::default();
        let c = f64::from(brown.coefficient);
        assert!((c - 0.99715).abs() < 2e-5, "coefficient {c}");
        let mut square = 0.0;
        for index in 0..2_000_000 {
            let value = f64::from(brown.step(&mut rng));
            if index >= 5000 {
                square += value * value / 1_995_000.0;
            }
        }
        let expected = 0.99 * 0.99 / 3.0 * (1.0 - c) / (1.0 + c) * 400.0;
        assert!(
            (square / expected - 1.0).abs() < 0.1,
            "{square} against {expected}"
        );
    }

    #[test]
    fn the_follower_rises_to_the_level_of_its_input() {
        let mut follower = Follower::new(100.0);
        let mut last = 0.0;
        for _ in 0..4410 {
            last = follower.step(-0.5);
        }
        assert!((last - 0.5).abs() < 1e-3);
        // One time constant: 1 − 1/e of the way.
        let mut follower = Follower::new(100.0);
        let steps = (RATE / (core::f64::consts::TAU * 100.0)).round() as usize;
        for _ in 0..steps {
            last = follower.step(1.0);
        }
        assert!((f64::from(last) - (1.0 - (-1.0_f64).exp())).abs() < 0.01);
    }

    #[test]
    fn the_metronome_fires_at_once_and_one_sample_after_each_period() {
        let mut metro = Metro::new(0.01);
        let fired: Vec<usize> = (0..1400).filter(|_| metro.step()).collect();
        assert_eq!(fired.len(), 4);
        assert_eq!(fired[0], 0);
        for (index, sample) in fired.iter().enumerate().skip(1) {
            assert!(sample.abs_diff(441 * index + 1) <= 1, "{fired:?}");
        }
    }

    #[test]
    fn the_envelope_falls_steeply_and_holds_its_last_value() {
        let mut envelope = Expseg::new(&[(0.0, 1.0), (0.01, 0.0)]);
        assert_eq!(envelope.step(false), 0.0);
        let samples: Vec<f32> = (0..600).map(|index| envelope.step(index == 0)).collect();
        assert_eq!(samples[0], 1.0);
        for (index, sample) in samples.iter().enumerate().take(441) {
            let expected = (1.0 - index as f64 / 441.0).powi(10);
            assert!(
                (f64::from(*sample) - expected).abs() < 1e-4,
                "sample {index}"
            );
        }
        assert!(samples[445..].iter().all(|sample| *sample == 0.0));
        // A rising segment waits, then rises; a trigger starts it anew.
        let mut envelope = Expseg::new(&[(0.0, -0.5), (0.01, 0.0)]);
        let samples: Vec<f32> = (0..500)
            .map(|index| envelope.step(index % 300 == 0))
            .collect();
        assert_eq!(samples[0], -0.5);
        assert!((f64::from(samples[220]) + 0.5 * (1.0 - 0.5_f64.powi(10))).abs() < 0.01);
        assert_eq!(samples[300], -0.5);
        // Three points: 1 down to 0.8, then to 0.
        let mut envelope = Expseg::new(&[(0.0, 1.0), (0.002, 0.8), (0.01, 0.0)]);
        let samples: Vec<f32> = (0..500).map(|index| envelope.step(index == 0)).collect();
        assert!(samples[40] > 0.8 && samples[87] > 0.799 && samples[95] < 0.8);
        assert!(samples.windows(2).all(|pair| pair[1] <= pair[0]));
        assert_eq!(samples[460], 0.0);
    }

    #[test]
    fn the_low_pass_keeps_a_level_and_damps_a_fast_wave() {
        let mut filter = Biquad::new(200.0, 0.7);
        assert!((0..100).all(|_| (filter.step(0.3) - 0.3).abs() < 1e-5));
        let mut filter = Biquad::new(200.0, 0.7);
        let mut top = 0.0_f32;
        for index in 0..4000 {
            let input = (core::f64::consts::TAU * 2000.0 * f64::from(index) / RATE).sin() as f32;
            let output = filter.step(input);
            if index > 2000 {
                top = top.max(output.abs());
            }
        }
        // Two octaves and more above the cutoff: 12 dB per octave.
        assert!(top < 0.02, "the wave passes with {top}");
    }

    #[test]
    fn the_recording_fades_in_and_is_normalized() {
        let recorded = record(1000, || 2.0);
        assert_eq!(recorded[0], 0.0);
        assert!((recorded[220] - 220.0 / 441.0).abs() < 1e-6);
        assert!(recorded[441..].iter().all(|value| *value == 1.0));
        assert!(record(10, || 0.0).iter().all(|value| *value == 0.0));
    }
}
