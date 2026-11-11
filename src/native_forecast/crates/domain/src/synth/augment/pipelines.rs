//! The two offline augmentations, each making one series at a time from
//! series of the base sources, with the settings of the `main` of their
//! scripts: `offline_per_sample_iid_augmentations.py` ([`PerSample`]) and
//! `offline_temp_batch_augmentations.py` ([`TempBatch`]).
//!
//! Both keep a series only if it differs enough from the series it was made
//! from: by 5% of the interquartile range of that series, on average.

use super::convolution::random_convolution;
use super::mixup;
use super::scalers::Scaler;
use super::series::{Edits, add_noise};
use super::{Drawn, PROPORTIONS, Pool, mean, quantile, sorted, source};
use crate::random::Rng;
use crate::synth::Source;

/// The probability of a mixture, wherever one may be made.
const MIXUP: f64 = 0.5;

/// The probability with which random convolutions are considered at all, and
/// then that with which a series goes through them.
const CONVOLUTION: f64 = 0.3;

/// The least change, in interquartile ranges of the original, that makes an
/// augmented series worth keeping (`--change-threshold`).
const THRESHOLD: f64 = 0.05;

/// The mean absolute difference of two series, as PyTorch's float32 mean.
fn difference(original: &[f32], augmented: &[f32]) -> f64 {
    let differences: Vec<f32> = original
        .iter()
        .zip(augmented)
        .map(|(before, after)| (after - before).abs())
        .collect();
    f64::from(mean(&differences))
}

/// The interquartile range and the range of a series.
fn ranges(values: &[f32]) -> (f64, f64) {
    let sorted = sorted(values);
    let quartiles = quantile(&sorted, 0.75) - quantile(&sorted, 0.25);
    let range = sorted[sorted.len() - 1] - sorted[0];
    (f64::from(quartiles), f64::from(range))
}

fn finite(values: &[f32]) -> bool {
    values.iter().all(|value| value.is_finite())
}

fn double(values: &[f32]) -> Vec<f64> {
    values.iter().map(|value| f64::from(*value)).collect()
}

/// The augmentation of one series at a time.
///
/// A series of a base source is scaled (half of the time: by one of the four
/// scalers), mixed with series of up to nine other sources (half of the
/// time), edited, put through random convolutions (9% of the time) and
/// mixed again (half of the time). It is kept if it has changed enough,
/// else another series is taken.
#[derive(Clone, Copy, Debug)]
pub struct PerSample {
    /// The most series that are tried. TempoPFN tries without end; so many
    /// failures in a row do not happen with series that vary.
    pub tries: usize,
}

impl Default for PerSample {
    fn default() -> Self {
        Self { tries: 1000 }
    }
}

impl PerSample {
    /// The edits of this augmentation: all of them.
    const EDITS: Edits = Edits {
        calendar: true,
        quantization: true,
    };

    /// An augmented series. (A candidate with a value that is not finite,
    /// which this arithmetic does not make of finite series, is dropped like
    /// one that has not changed enough.)
    pub fn series(&self, rng: &mut Rng, pool: &mut dyn Pool) -> Vec<f64> {
        let mut last = Vec::new();
        for _ in 0..self.tries.max(1) {
            let base = pool.draw(source(rng), rng);
            let scaler = if rng.chance(0.5) {
                None
            } else {
                Some(*rng.choose(&Scaler::ALL))
            };
            let mut values = base.values.clone();
            if let Some(scaler) = scaler {
                scaler.scale(&mut values);
            }
            if rng.chance(MIXUP) {
                values = Self::mixed(rng, pool, values, scaler);
            }
            Self::EDITS.apply(rng, &mut values, base.frequency, base.start);
            // TempoPFN puts the series in a batch with seven others and takes
            // it back: the others do not touch it.
            if rng.chance(CONVOLUTION) && rng.chance(CONVOLUTION) {
                values = random_convolution(rng, &values);
            }
            if rng.chance(MIXUP) {
                values = Self::mixed(rng, pool, values, scaler);
            }
            // Against the series as it was before its scaling.
            let (quartiles, range) = ranges(&base.values);
            let scale = quartiles.max(1.0e-6 * range).max(1.0e-8);
            let enough = difference(&base.values, &values) / scale >= THRESHOLD;
            last = values;
            if enough && finite(&last) {
                break;
            }
        }
        double(&last)
    }

    /// The series mixed, with one set of weights, with a series of each of
    /// one to nine sources, drawn without replacement by their weights and
    /// scaled as the series was.
    fn mixed(
        rng: &mut Rng,
        pool: &mut dyn Pool,
        series: Vec<f32>,
        scaler: Option<Scaler>,
    ) -> Vec<f32> {
        let weights = PROPORTIONS.map(|(_, weight)| weight);
        let others = mixup::count(rng) - 1;
        let others = rng.distinct_weighted(&weights, others);
        let mut sources = vec![series];
        for other in others {
            let mut values = pool.draw(PROPORTIONS[other].0, rng).values;
            if let Some(scaler) = scaler {
                scaler.scale(&mut values);
            }
            sources.push(values);
        }
        let concentration = mixup::concentration(rng);
        let views: Vec<&[f32]> = sources.iter().map(Vec::as_slice).collect();
        mixup::fixed(rng, &views, concentration)
    }
}

/// The augmentation of a batch of which one series is kept.
///
/// 64 distinct series of the base sources are each scaled (by one of the
/// four scalers or not at all); some are replaced by mixtures of the others;
/// all are edited; the batch may take noise, a scaling and random
/// convolutions; some are replaced by mixtures again. One of the series that
/// have changed enough is kept, or the one that has changed most.
#[derive(Clone, Copy, Debug)]
pub struct TempBatch {
    /// The number of series of the batch (`--mixed-batch-size`).
    pub size: usize,
}

impl Default for TempBatch {
    fn default() -> Self {
        Self { size: 64 }
    }
}

impl TempBatch {
    /// The edits of this augmentation: neither calendar nor quantization.
    const EDITS: Edits = Edits {
        calendar: false,
        quantization: false,
    };

    /// An augmented series.
    ///
    /// # Panics
    ///
    /// If the batch is to have no series.
    pub fn series(&self, rng: &mut Rng, pool: &mut dyn Pool) -> Vec<f64> {
        assert!(self.size > 0, "a batch has at least one series");
        let batch = self.batch(rng, pool);
        let originals: Vec<&[f32]> = batch.iter().map(|drawn| drawn.values.as_slice()).collect();
        let mut values: Vec<Vec<f32>> = originals.iter().map(|series| series.to_vec()).collect();

        mixup::batch(rng, &mut values, MIXUP);
        for (series, drawn) in values.iter_mut().zip(&batch) {
            Self::EDITS.apply(rng, series, drawn.frequency, drawn.start);
        }
        if rng.chance(0.1) {
            let spread = 0.01 * deviation_of_all(&values);
            for series in &mut values {
                add_noise(rng, series, spread);
            }
        }
        if rng.chance(0.5) {
            let factor = rng.uniform_in(0.95, 1.05) as f32;
            values
                .iter_mut()
                .flatten()
                .for_each(|value| *value *= factor);
        }
        if rng.chance(CONVOLUTION) {
            for series in &mut values {
                if rng.chance(CONVOLUTION) {
                    *series = random_convolution(rng, series);
                }
            }
        }
        mixup::batch(rng, &mut values, MIXUP);

        // Against the series as they were after their scaling.
        let scores: Vec<f64> = originals
            .iter()
            .zip(&values)
            .map(|(original, series)| {
                let quartiles = ranges(original).0;
                let scale = if quartiles > 0.0 { quartiles } else { 1.0e-6 };
                if finite(series) {
                    difference(original, series) / scale
                } else {
                    f64::NEG_INFINITY
                }
            })
            .collect();
        let enough: Vec<usize> = (0..scores.len())
            .filter(|&index| scores[index] >= THRESHOLD)
            .collect();
        let kept = if enough.is_empty() {
            // The first of the series that changed most.
            (0..scores.len()).fold(0, |best, index| {
                if scores[index] > scores[best] {
                    index
                } else {
                    best
                }
            })
        } else {
            *rng.choose(&enough)
        };
        double(&values[kept])
    }

    /// The series of the batch, each from a source drawn by its weight: no
    /// series twice, none with a value that is not a number, each scaled by
    /// one of the four scalers or, one time in five, not at all.
    fn batch(&self, rng: &mut Rng, pool: &mut dyn Pool) -> Vec<Drawn> {
        let mut batch: Vec<Drawn> = Vec::with_capacity(self.size);
        let mut taken: Vec<(Source, usize)> = Vec::with_capacity(self.size);
        // TempoPFN gives up after 200 fetches per series, of 50 tries each.
        for _ in 0..self.size * 200 * 50 {
            if batch.len() == self.size {
                break;
            }
            let source = source(rng);
            let mut drawn = pool.draw(source, rng);
            let scaler = rng.index(Scaler::ALL.len() + 1);
            if drawn.values.iter().any(|value| value.is_nan())
                || taken.contains(&(source, drawn.place))
            {
                continue;
            }
            if let Some(scaler) = Scaler::ALL.get(scaler) {
                scaler.scale(&mut drawn.values);
            }
            taken.push((source, drawn.place));
            batch.push(drawn);
        }
        assert!(!batch.is_empty(), "the pool holds a series without NaN");
        batch
    }
}

/// The standard deviation of all the values of a batch, as `torch.std`.
fn deviation_of_all(batch: &[Vec<f32>]) -> f32 {
    let count = batch.iter().map(Vec::len).sum::<usize>() as f64;
    let total: f64 = batch.iter().flatten().map(|value| f64::from(*value)).sum();
    let center = total / count;
    let squares: f64 = batch
        .iter()
        .flatten()
        .map(|value| (f64::from(*value) - center).powi(2))
        .sum();
    (squares / (count - 1.0)).sqrt() as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calendar::Frequency;

    /// A pool of sines, a few of each source: series `place` of a source has
    /// its own frequency and level.
    struct Sines {
        length: usize,
        each: usize,
        draws: usize,
    }

    impl Pool for Sines {
        fn draw(&mut self, source: Source, rng: &mut Rng) -> Drawn {
            self.draws += 1;
            let place = rng.index(self.each);
            let pitch = 0.05 + 0.01 * source.id() as f32 + 0.003 * place as f32;
            Drawn {
                values: (0..self.length)
                    .map(|step| place as f32 + (pitch * step as f32).sin())
                    .collect(),
                frequency: Frequency::D,
                // The pipelines do not read the start: it names the source.
                start: source.id() as i64,
                place,
            }
        }
    }

    #[test]
    fn the_change_is_measured_in_interquartile_ranges() {
        let original: Vec<f32> = (0..101).map(|step| step as f32).collect();
        let shifted: Vec<f32> = original.iter().map(|value| value + 5.0).collect();
        assert_eq!(ranges(&original), (50.0, 100.0));
        assert_eq!(difference(&original, &shifted), 5.0);
        assert_eq!(difference(&original, &original), 0.0);
        let batch = vec![vec![1.0_f32, 2.0], vec![3.0, 4.0]];
        assert!((f64::from(deviation_of_all(&batch)) - (5.0_f64 / 3.0).sqrt()).abs() < 1e-6);
    }

    #[test]
    fn a_per_sample_series_differs_from_every_base_series() {
        let mut rng = Rng::from_key(&[1]);
        let mut pool = Sines {
            length: 256,
            each: 40,
            draws: 0,
        };
        let augmentation = PerSample::default();
        let mut series = Vec::new();
        for _ in 0..60 {
            let made = augmentation.series(&mut rng, &mut pool);
            assert_eq!(made.len(), 256);
            assert!(made.iter().all(|value| value.is_finite()));
            series.push(made);
        }
        // One base series a try, and those of the mixtures: about five.
        assert!((60..60 * 12).contains(&pool.draws), "{} draws", pool.draws);
        series.dedup();
        assert_eq!(series.len(), 60);
        // The same streams give the same series.
        let again = |key: u64| {
            let mut pool = Sines {
                length: 256,
                each: 40,
                draws: 0,
            };
            augmentation.series(&mut Rng::from_key(&[key]), &mut pool)
        };
        assert_eq!(again(7), again(7));
        assert_ne!(again(7), again(8));
    }

    #[test]
    fn a_temp_batch_takes_distinct_series_and_keeps_one() {
        let mut rng = Rng::from_key(&[2]);
        let augmentation = TempBatch::default();
        let mut pool = Sines {
            length: 256,
            each: 40,
            draws: 0,
        };
        let batch = augmentation.batch(&mut rng, &mut pool);
        assert_eq!(batch.len(), 64);
        let mut taken: Vec<(i64, usize)> = batch
            .iter()
            .map(|drawn| (drawn.start, drawn.place))
            .collect();
        taken.sort_unstable();
        taken.dedup();
        assert_eq!(taken.len(), 64, "no series twice");

        for _ in 0..20 {
            let made = augmentation.series(&mut rng, &mut pool);
            assert_eq!(made.len(), 256);
            assert!(made.iter().all(|value| value.is_finite()));
        }
        // A pool of fewer series than the batch asks for gives what it has.
        let mut small = Sines {
            length: 64,
            each: 2,
            draws: 0,
        };
        let few = TempBatch { size: 64 }.batch(&mut rng, &mut small);
        assert_eq!(few.len(), 2 * PROPORTIONS.len());
        assert!(TempBatch { size: 64 }.series(&mut rng, &mut small).len() == 64);
    }
}
