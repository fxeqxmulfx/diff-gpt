//! `step`: a piecewise constant series made of 10 to 100 stretches, each with
//! its own pattern of level changes, with faint noise and two slow sinusoids.
//!
//! Port of TempoPFN's `steps/step_generator.py` with the defaults of
//! `StepGeneratorParams`: no smoothing between stretches, no trend and no
//! injected anomalies, which the defaults disable.

use core::f64::consts::TAU;

use crate::numeric::linspace;
use crate::random::Rng;

/// How the level changes inside a stretch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pattern {
    /// A few small changes in the middle half.
    Stable,
    /// Evenly spread rises.
    GradualIncrease,
    /// Evenly spread falls.
    GradualDecrease,
    /// Rises in the first third, then smaller falls, all decaying.
    SpikeUp,
    /// Falls in the first third, then smaller rises, all decaying.
    SpikeDown,
    /// Evenly spread changes of alternating sign.
    Oscillating,
    /// Changes of either sign at random places.
    RandomWalk,
}

/// A kind of stretch.
#[derive(Clone, Debug)]
pub struct Stretch {
    /// Its pattern.
    pub pattern: Pattern,
    /// The range of its length, both ends included.
    pub length: (i64, i64),
    /// The range of its number of level changes, both ends included.
    pub changes: (i64, i64),
    /// The range of a change.
    pub step_size: (f64, f64),
    /// The range of the drift of the level over the stretch; none if both are 0.
    pub drift: (f64, f64),
    /// The factor by which each next change is smaller.
    pub decay: f64,
    /// The weight of this kind among the kinds.
    pub weight: f64,
}

const fn stretch(
    pattern: Pattern,
    length: (i64, i64),
    changes: (i64, i64),
    step_size: (f64, f64),
    weight: f64,
) -> Stretch {
    Stretch {
        pattern,
        length,
        changes,
        step_size,
        drift: (0.0, 0.0),
        decay: 1.0,
        weight,
    }
}

/// The parameters of the source.
#[derive(Clone, Debug)]
pub struct Step {
    /// The kinds of stretches.
    pub stretches: Vec<Stretch>,
    /// The range of the number of stretches, both ends included.
    pub count: (i64, i64),
    /// The range of the first level.
    pub level: (f64, f64),
    /// The range of the deviation of the noise.
    pub noise: (f64, f64),
    /// The greatest amplitudes of the sinusoids of periods 288 and 2016.
    pub seasonality: (f64, f64),
    /// The range of the final scale.
    pub scale: (f64, f64),
    /// The greatest jump of the level between two stretches.
    pub max_jump: f64,
}

impl Default for Step {
    fn default() -> Self {
        use Pattern::{
            GradualDecrease, GradualIncrease, Oscillating, RandomWalk, SpikeDown, SpikeUp, Stable,
        };
        Self {
            stretches: vec![
                stretch(Stable, (200, 600), (0, 3), (-1.0, 1.0), 0.8),
                Stretch {
                    drift: (0.0, 0.1),
                    ..stretch(GradualIncrease, (300, 700), (5, 15), (1.0, 5.0), 0.6)
                },
                Stretch {
                    drift: (-0.1, 0.0),
                    ..stretch(GradualDecrease, (300, 700), (5, 15), (-5.0, -1.0), 0.6)
                },
                Stretch {
                    decay: 0.7,
                    ..stretch(SpikeUp, (200, 500), (3, 8), (3.0, 10.0), 0.4)
                },
                Stretch {
                    decay: 0.7,
                    ..stretch(SpikeDown, (200, 500), (3, 8), (-10.0, -3.0), 0.4)
                },
                stretch(Oscillating, (400, 800), (8, 20), (-4.0, 4.0), 0.3),
                stretch(RandomWalk, (100, 400), (5, 20), (-3.0, 3.0), 0.2),
            ],
            count: (10, 100),
            level: (5.0, 15.0),
            noise: (0.001, 0.01),
            seasonality: (0.8, 0.7),
            scale: (0.1, 10.0),
            max_jump: 5.0,
        }
    }
}

/// `linspace(start, stop, count)` truncated to integers, as `astype(int)`.
fn spread(start: usize, stop: usize, count: usize) -> Vec<usize> {
    linspace(start as f64, stop as f64, count)
        .into_iter()
        .map(|point| point as usize)
        .collect()
}

impl Step {
    /// One series of `length` steps.
    pub fn series(&self, rng: &mut Rng, length: usize) -> Vec<f64> {
        let plan = self.plan(rng, length);
        let mut level = rng.uniform_in(self.level.0, self.level.1);
        let mut values: Vec<f64> = Vec::with_capacity(length);
        for (kind, stretch_length) in plan {
            let mut part = Self::stretch(rng, kind, stretch_length, level);
            if let (false, Some(&first)) = (values.is_empty(), part.first()) {
                let jump = first - level;
                if jump.abs() > self.max_jump {
                    let excess = jump - jump.signum() * self.max_jump;
                    for value in &mut part {
                        *value -= excess;
                    }
                }
            }
            if let Some(&last) = part.last() {
                level = last;
            }
            values.extend(part);
        }

        let noise = rng.uniform_in(self.noise.0, self.noise.1);
        for value in &mut values {
            *value += rng.normal_with(0.0, noise);
        }
        for (amplitude, period) in [(self.seasonality.0, 288.0), (self.seasonality.1, 2016.0)] {
            if amplitude > 0.0 {
                let amplitude = rng.uniform_in(0.0, amplitude);
                for (step, value) in values.iter_mut().enumerate() {
                    *value += amplitude * (TAU * step as f64 / period).sin();
                }
            }
        }
        let scale = rng.uniform_in(self.scale.0, self.scale.1);
        for value in &mut values {
            *value *= scale;
        }
        values
    }

    /// The kinds and lengths of the stretches of a series; the lengths sum to
    /// `length`.
    fn plan(&self, rng: &mut Rng, length: usize) -> Vec<(&Stretch, usize)> {
        let weights: Vec<f64> = self.stretches.iter().map(|kind| kind.weight).collect();
        let count = rng.integer(self.count.0, self.count.1 + 1);
        let mut remaining = length as i64;
        (0..count)
            .map(|index| {
                let kind = &self.stretches[rng.weighted(&weights)];
                let left = count - index;
                let stretch_length = if left == 1 {
                    remaining
                } else {
                    let low = kind.length.0.min(remaining / left);
                    let high = kind.length.1.min(remaining - (left - 1) * 50).max(low);
                    rng.integer(low, high + 1)
                };
                remaining -= stretch_length;
                (kind, stretch_length as usize)
            })
            .collect()
    }

    /// The sorted places of the level changes of a stretch, all below `length`.
    fn changepoints(rng: &mut Rng, kind: &Stretch, length: usize) -> Vec<usize> {
        let count = rng.integer(kind.changes.0, kind.changes.1 + 1) as usize;
        if count == 0 || length == 0 {
            return Vec::new();
        }
        let mut places = match kind.pattern {
            Pattern::Stable => {
                Self::scattered(rng, length / 4, 3 * length / 4, count.min(length / 2))
            }
            Pattern::GradualIncrease | Pattern::GradualDecrease => {
                let spacing = (length / (count * 2)).max(1) as i64;
                spread(length / 10, 9 * length / 10, count)
                    .into_iter()
                    .map(|place| {
                        let moved = place as i64 + rng.integer(-spacing, spacing + 1);
                        moved.clamp(0, length as i64 - 1) as usize
                    })
                    .collect()
            }
            Pattern::SpikeUp | Pattern::SpikeDown => {
                let third = length / 3;
                let early = (count / 2).max(1);
                let mut places = spread(length / 20, third, early);
                places.extend(spread(third + 1, 9 * length / 10, count - early));
                places
            }
            Pattern::Oscillating => spread(length / 10, 9 * length / 10, count),
            Pattern::RandomWalk => {
                Self::scattered(rng, length / 10, 9 * length / 10, count.min(length / 2))
            }
        };
        places.sort_unstable();
        places
    }

    /// `count` distinct places of `[start, stop)`, or all of them if fewer.
    fn scattered(rng: &mut Rng, start: usize, stop: usize, count: usize) -> Vec<usize> {
        let span = stop.saturating_sub(start);
        rng.distinct(span, count.min(span))
            .into_iter()
            .map(|offset| start + offset)
            .collect()
    }

    /// The changes of the level at the `count` changepoints of a stretch.
    fn step_sizes(rng: &mut Rng, kind: &Stretch, count: usize) -> Vec<f64> {
        let mut sizes: Vec<f64> = (0..count)
            .map(|_| rng.uniform_in(kind.step_size.0, kind.step_size.1))
            .collect();
        let half = count / 2;
        for (index, size) in sizes.iter_mut().enumerate() {
            let decay = kind.decay.powi(index as i32);
            *size = match kind.pattern {
                Pattern::Stable => *size * 0.1,
                Pattern::GradualIncrease => size.abs() * decay,
                Pattern::GradualDecrease => -size.abs() * decay,
                Pattern::SpikeUp if index < half => size.abs() * decay,
                Pattern::SpikeUp => -size.abs() * 0.5 * decay,
                Pattern::SpikeDown if index < half => -size.abs() * decay,
                Pattern::SpikeDown => size.abs() * 0.5 * decay,
                Pattern::Oscillating if index % 2 == 1 => -size.abs(),
                Pattern::Oscillating => size.abs(),
                Pattern::RandomWalk => *size,
            };
        }
        sizes
    }

    /// One stretch of `length` steps that starts at `level`.
    fn stretch(rng: &mut Rng, kind: &Stretch, length: usize, level: f64) -> Vec<f64> {
        let places = Self::changepoints(rng, kind, length);
        let sizes = Self::step_sizes(rng, kind, places.len());
        let mut values = vec![level; length];
        let mut current = level;
        for (place, size) in places.into_iter().zip(sizes) {
            current += size;
            for value in values.iter_mut().skip(place) {
                *value = current;
            }
        }
        if kind.drift != (0.0, 0.0) {
            let drift = rng.uniform_in(kind.drift.0, kind.drift.1);
            for (value, ramp) in values.iter_mut().zip(linspace(0.0, drift, length)) {
                *value += ramp;
            }
        }
        values
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stretches_tile_the_series() {
        let source = Step::default();
        let mut rng = Rng::from_key(&[3]);
        for _ in 0..500 {
            let plan = source.plan(&mut rng, 2048);
            assert!((10..=100).contains(&plan.len()));
            assert!(plan.iter().all(|&(_, length)| length >= 20));
            assert_eq!(plan.iter().map(|&(_, length)| length).sum::<usize>(), 2048);
        }
    }

    #[test]
    fn changes_follow_their_pattern() {
        let source = Step::default();
        let mut rng = Rng::from_key(&[5]);
        for kind in &source.stretches {
            for length in [20, 57, 300, 700] {
                let places = Step::changepoints(&mut rng, kind, length);
                assert!(places.is_sorted());
                assert!(places.iter().all(|&place| place < length));
                let sizes = Step::step_sizes(&mut rng, kind, 8);
                match kind.pattern {
                    Pattern::GradualIncrease => assert!(sizes.iter().all(|&s| s > 0.0)),
                    Pattern::GradualDecrease => assert!(sizes.iter().all(|&s| s < 0.0)),
                    Pattern::SpikeUp => {
                        assert!(sizes[..4].iter().all(|&s| s > 0.0));
                        assert!(sizes[4..].iter().all(|&s| s < 0.0));
                        assert!(sizes[7].abs() < 0.5 * 10.0 * 0.7_f64.powi(7) + 1e-12);
                    }
                    Pattern::Oscillating => {
                        assert!(
                            sizes
                                .iter()
                                .enumerate()
                                .all(|(i, &s)| (s < 0.0) == (i % 2 == 1))
                        );
                    }
                    Pattern::Stable => assert!(sizes.iter().all(|&s| s.abs() <= 0.1)),
                    Pattern::SpikeDown | Pattern::RandomWalk => {}
                }
            }
        }
    }

    #[test]
    fn a_series_is_mostly_flat() {
        let source = Step {
            noise: (0.0, 0.0),
            seasonality: (0.0, 0.0),
            ..Step::default()
        };
        let mut flat: Vec<usize> = (0..51)
            .map(|seed| {
                let values = source.series(&mut Rng::from_key(&[seed]), 2048);
                assert_eq!(values.len(), 2048);
                assert!(values.iter().all(|v| v.is_finite()));
                values.windows(2).filter(|pair| pair[0] == pair[1]).count()
            })
            .collect();
        flat.sort_unstable();
        // Only the stretches that drift, about a third of them, are not flat.
        assert!(flat[25] > 1024, "median of {} flat steps", flat[25]);
    }
}
