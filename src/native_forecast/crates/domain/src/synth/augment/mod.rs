//! TempoPFN's two offline augmentations of the series of its 14 generators
//! (`src/synthetic_generation/augmentations`, with the augmenters and the
//! scalers of `src/data`): the last two sources of its training mixture.
//!
//! - [`pipelines::PerSample`]: one series is scaled, mixed with series of
//!   other sources, put through two to five kinds of edits, and kept if it
//!   has changed enough;
//! - [`pipelines::TempBatch`]: 64 series are scaled, mixed among themselves
//!   before and after the same edits, and one of those that changed enough
//!   is kept.
//!
//! The series come from a [`Pool`]: TempoPFN reads them from the files of the
//! corpus, in turn; here a pool hands out the series of a source at random.
//! The arithmetic is PyTorch's float32.

pub mod convolution;
pub mod mixup;
pub mod pipelines;
pub mod scalers;
pub mod series;

use super::Source;
use crate::calendar::Frequency;
use crate::random::Rng;

/// A series of a base source, with the calendar TempoPFN stores beside it.
#[derive(Clone, Debug, PartialEq)]
pub struct Drawn {
    /// The values.
    pub values: Vec<f32>,
    /// The sampling frequency.
    pub frequency: Frequency,
    /// The first timestamp, in nanoseconds since 1970.
    pub start: i64,
    /// What tells this series from the others of its source in the pool.
    pub place: usize,
}

/// A supply of series of the base sources.
pub trait Pool {
    /// A series of `source`, taken at random.
    fn draw(&mut self, source: Source, rng: &mut Rng) -> Drawn;
}

/// The weights with which both augmentations take their series from the
/// base sources: the `generator_proportions` of their `main`.
pub const PROPORTIONS: [(Source, f64); 14] = [
    (Source::ForecastPfn, 1.0),
    (Source::Gp, 1.0),
    (Source::Kernel, 1.0),
    (Source::Sinewave, 1.0),
    (Source::Sawtooth, 1.0),
    (Source::Step, 0.1),
    (Source::Anomaly, 1.0),
    (Source::Spike, 1.0),
    (Source::CaukerUnivariate, 2.0),
    (Source::OuProcess, 1.0),
    (Source::AudioFinancialVolatility, 0.1),
    (Source::AudioMultiScaleFractal, 0.1),
    (Source::AudioNetworkTopology, 0.5),
    (Source::AudioStochasticRhythm, 1.0),
];

/// A source, drawn with those weights.
pub(crate) fn source(rng: &mut Rng) -> Source {
    let weights = PROPORTIONS.map(|(_, weight)| weight);
    PROPORTIONS[rng.weighted(&weights)].0
}

/// A pool that makes its series with the generators themselves, a chunk of a
/// source at a time: slow for the Gaussian processes, and enough for tests
/// and small corpora.
#[derive(Clone, Debug)]
pub struct Fresh {
    length: usize,
    stock: Vec<(Source, Vec<Drawn>)>,
    made: usize,
}

impl Fresh {
    /// A pool of series of `length` steps.
    #[must_use]
    pub const fn new(length: usize) -> Self {
        Self {
            length,
            stock: Vec::new(),
            made: 0,
        }
    }
}

impl Pool for Fresh {
    fn draw(&mut self, source: Source, rng: &mut Rng) -> Drawn {
        let place = match self
            .stock
            .iter()
            .position(|(stocked, _)| *stocked == source)
        {
            Some(place) => place,
            None => {
                self.stock.push((source, Vec::new()));
                self.stock.len() - 1
            }
        };
        let stock = &mut self.stock[place].1;
        loop {
            if let Some(drawn) = stock.pop() {
                return drawn;
            }
            let mut stream = rng.fork(self.made as u64);
            let calendars = source.calendars(&stream, self.length);
            let chunk = source.chunk(&mut stream, self.length);
            for (values, (frequency, start)) in chunk.into_iter().zip(calendars) {
                self.made += 1;
                stock.push(Drawn {
                    values: values.into_iter().map(|value| value as f32).collect(),
                    frequency,
                    start,
                    place: self.made,
                });
            }
        }
    }
}

/// The mean, as `torch.mean` of a float32 tensor.
pub(crate) fn mean(values: &[f32]) -> f32 {
    let total: f64 = values.iter().map(|value| f64::from(*value)).sum();
    (total / values.len() as f64) as f32
}

/// The standard deviation with the divisor `n − 1`, as `torch.std`.
pub(crate) fn deviation(values: &[f32]) -> f32 {
    let center = f64::from(mean(values));
    let total: f64 = values
        .iter()
        .map(|value| (f64::from(*value) - center).powi(2))
        .sum();
    (total / (values.len() as f64 - 1.0)).sqrt() as f32
}

/// The values in increasing order.
pub(crate) fn sorted(values: &[f32]) -> Vec<f32> {
    let mut sorted = values.to_vec();
    sorted.sort_by(f32::total_cmp);
    sorted
}

/// `torch.quantile` of sorted values: linear between the two nearest.
pub(crate) fn quantile(sorted: &[f32], level: f32) -> f32 {
    let Some(last) = sorted.len().checked_sub(1) else {
        return f32::NAN;
    };
    let position = level * last as f32;
    let below = position.floor() as usize;
    let above = (below + 1).min(last);
    let part = position - below as f32;
    sorted[below] + (sorted[above] - sorted[below]) * part
}

/// `torch.median` of sorted values: the lower of the two middle ones.
pub(crate) fn median(sorted: &[f32]) -> f32 {
    sorted
        .get(sorted.len().saturating_sub(1) / 2)
        .copied()
        .unwrap_or(f32::NAN)
}

/// The least and the greatest value.
pub(crate) fn extent(values: &[f32]) -> (f32, f32) {
    values
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(low, high), value| {
            (low.min(*value), high.max(*value))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_statistics_are_torch_s() {
        let values = [3.0_f32, 1.0, 4.0, 1.0, 5.0, 9.0];
        assert!((mean(&values) - 23.0 / 6.0).abs() < 1e-6);
        assert!((f64::from(deviation(&values)) - 8.966_666_666_f64.sqrt()).abs() < 1e-6);
        let sorted = sorted(&values);
        assert_eq!(sorted, [1.0, 1.0, 3.0, 4.0, 5.0, 9.0]);
        assert_eq!(median(&sorted), 3.0);
        assert_eq!(quantile(&sorted, 0.5), 3.5);
        assert_eq!(quantile(&sorted, 0.25), 1.5);
        assert_eq!(quantile(&sorted, 1.0), 9.0);
        assert_eq!(extent(&values), (1.0, 9.0));
        assert!(deviation(&[2.0]).is_nan());
    }

    #[test]
    fn the_sources_are_drawn_with_their_weights() {
        let mut rng = Rng::from_key(&[1]);
        let draws = 59_000;
        let mut counts = [0_usize; 14];
        for _ in 0..draws {
            let drawn = source(&mut rng);
            counts[PROPORTIONS
                .iter()
                .position(|(source, _)| *source == drawn)
                .expect("a base source")] += 1;
        }
        for ((_, weight), count) in PROPORTIONS.iter().zip(counts) {
            let expected = weight / 11.8 * f64::from(draws);
            assert!((count as f64 - expected).abs() < 5.0 * expected.sqrt() + 5.0);
        }
    }

    #[test]
    fn a_fresh_pool_hands_out_distinct_series_with_their_calendar() {
        let mut rng = Rng::from_key(&[2]);
        let mut pool = Fresh::new(256);
        let first = pool.draw(Source::Sinewave, &mut rng);
        let second = pool.draw(Source::Sinewave, &mut rng);
        let other = pool.draw(Source::ForecastPfn, &mut rng);
        assert_eq!(first.values.len(), 256);
        assert_ne!(first.values, second.values);
        assert_ne!(first.place, second.place);
        assert_ne!(other.place, first.place);
        assert!(other.values.iter().all(|value| value.is_finite()));
        // More than a chunk of one source.
        let places: Vec<usize> = (0..100)
            .map(|_| pool.draw(Source::Step, &mut rng).place)
            .collect();
        let mut distinct = places.clone();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(distinct.len(), 100);
    }
}
