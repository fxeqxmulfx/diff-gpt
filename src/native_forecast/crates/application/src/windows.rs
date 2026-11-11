//! The windows of a corpus that a model trains on: `DiffCorpusDataLoader`
//! and `corpus_domain` of `src/diff_gpt/application/data_loader.py`.
//!
//! A window is some consecutive values of one series: its source drawn by
//! the weights of TempoPFN's training mixture ([`Source::weight`]), then a
//! series and an offset, uniformly. It is put in the units of its first `c`
//! values, `c` uniform between a least context and the whole window, as a
//! forecast's context gives the units of what follows it: the values after
//! `c` play the future, which need not keep to the context's scale.

use diff_gpt_forecast_domain::codec::{self, Codec};
use diff_gpt_forecast_domain::random::Rng;
use diff_gpt_forecast_domain::synth::Source;
use diff_gpt_forecast_domain::table::Table;

use crate::Result;
use crate::ports::{Executor, SeriesReader, Split};
use crate::training::{Example, Examples};

/// The series of some sources, in memory.
#[derive(Clone, Debug)]
pub struct Corpus {
    tables: Vec<(Source, Table)>,
}

impl Corpus {
    /// The corpus of these tables.
    #[must_use]
    pub const fn of(tables: Vec<(Source, Table)>) -> Self {
        Self { tables }
    }

    /// Every series of `sources` in `split`, read through `reader`.
    ///
    /// # Errors
    ///
    /// If a source cannot be read or has no series.
    pub fn load<E: Executor, R: SeriesReader>(
        executor: &E,
        reader: &R,
        split: Split,
        sources: &[Source],
    ) -> Result<Self> {
        let read = executor.map(sources.len(), |index| {
            let source = sources[index];
            let count = reader.series(split, source)?;
            if count == 0 {
                return Err(format!("{} has no {} series", source.name(), split.name()).into());
            }
            Ok((source, reader.read(split, source, 0, count)?))
        });
        Ok(Self {
            tables: read.into_iter().collect::<Result<_>>()?,
        })
    }

    /// The sources and their series.
    #[must_use]
    pub fn tables(&self) -> &[(Source, Table)] {
        &self.tables
    }

    /// The number of values in the corpus.
    #[must_use]
    pub fn values(&self) -> usize {
        let tables = self.tables.iter();
        tables.map(|(_, table)| table.values().len()).sum()
    }
}

/// The most windows drawn in place of one that holds a missing value.
const TRIES: usize = 1000;

/// Draws windows of a corpus.
#[derive(Clone, Debug)]
pub struct Sampler<'a> {
    corpus: &'a Corpus,
    weights: Vec<f64>,
    rows: usize,
    least: usize,
}

impl<'a> Sampler<'a> {
    /// The sampler of windows of `rows` values whose context has `least`
    /// values at least.
    ///
    /// # Errors
    ///
    /// If the corpus is empty, a source has series shorter than a window,
    /// or the least context is not within the window.
    pub fn new(corpus: &'a Corpus, rows: usize, least: usize) -> Result<Self> {
        if corpus.tables.is_empty() {
            return Err("a corpus without sources has no window".into());
        }
        if !(1..=rows).contains(&least) {
            return Err(format!("a context of {least} values in a window of {rows}").into());
        }
        for (source, table) in &corpus.tables {
            if table.length() < rows || table.rows() == 0 {
                return Err(format!(
                    "{} has {} series of {} values, and a window has {rows}",
                    source.name(),
                    table.rows(),
                    table.length()
                )
                .into());
            }
        }
        Ok(Self {
            corpus,
            weights: corpus
                .tables
                .iter()
                .map(|(source, _)| source.weight())
                .collect(),
            rows,
            least,
        })
    }

    /// The number of values of a window.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Draws a window into `window` and returns the place of its source in
    /// the corpus and the length of its context. A window with a value that
    /// is not finite is drawn again.
    ///
    /// # Panics
    ///
    /// If [`TRIES`] windows in a row hold such a value.
    pub fn draw(&self, rng: &mut Rng, window: &mut Vec<f64>) -> (usize, usize) {
        for _ in 0..TRIES {
            let place = rng.weighted(&self.weights);
            let table = &self.corpus.tables[place].1;
            let series = table.row(rng.index(table.rows()));
            let offset = rng.index(table.length() - self.rows + 1);
            let values = &series[offset..offset + self.rows];
            if values.iter().all(|value| value.is_finite()) {
                window.clear();
                window.extend(values.iter().map(|value| f64::from(*value)));
                let context = rng.integer(self.least as i64, self.rows as i64 + 1) as usize;
                return (place, context);
            }
        }
        panic!("{TRIES} windows in a row hold a value that is not finite");
    }

    /// The domain of the steps of the corpus: the quantile `quantile` of
    /// the absolute steps, in the units of their contexts, of `windows`
    /// windows drawn from the stream of `seed`.
    #[must_use]
    pub fn domain(&self, windows: usize, quantile: f64, seed: u64) -> f64 {
        let mut rng = Rng::from_key(&[seed]);
        let mut window = Vec::with_capacity(self.rows);
        let mut steps = Vec::with_capacity(windows * (self.rows - 1));
        for _ in 0..windows {
            let (_, context) = self.draw(&mut rng, &mut window);
            codec::steps_of(&window, context, &mut steps);
        }
        codec::domain_of(&mut steps, quantile)
    }
}

/// The windows of a corpus as sequences to train on: a window of
/// `block + 2` values is `block + 1` tokens, of which the model reads all
/// but the last and predicts all but the first.
#[derive(Clone, Debug)]
pub struct Windows<'a> {
    sampler: Sampler<'a>,
    codec: Codec,
    seed: u64,
}

impl<'a> Windows<'a> {
    /// The windows of `block` positions of `corpus`, in the tokens of
    /// `codec`, whose contexts have `least` values at least; example `i` of
    /// batch `b` is drawn from the stream of `[seed, b, i]`.
    ///
    /// # Errors
    ///
    /// As [`Sampler::new`].
    pub fn new(
        corpus: &'a Corpus,
        codec: Codec,
        block: usize,
        least: usize,
        seed: u64,
    ) -> Result<Self> {
        Ok(Self {
            sampler: Sampler::new(corpus, block + 2, least)?,
            codec,
            seed,
        })
    }

    /// The sampler of the windows.
    #[must_use]
    pub const fn sampler(&self) -> &Sampler<'a> {
        &self.sampler
    }
}

impl Examples for Windows<'_> {
    fn fill(&self, batch: u64, index: usize, example: &mut Example) {
        let mut rng = Rng::from_key(&[self.seed, batch, index as u64]);
        let mut window = Vec::with_capacity(self.sampler.rows);
        let (_, context) = self.sampler.draw(&mut rng, &mut window);
        let encoded = self.codec.encode(&window, context);
        let block = encoded.tokens.len() - 1;
        example.tokens.clear();
        example.tokens.extend_from_slice(&encoded.tokens[..block]);
        example.targets.clear();
        example.targets.extend_from_slice(&encoded.tokens[1..]);
        example.bins.clear();
        if !encoded.bins.is_empty() {
            example.bins.extend_from_slice(&encoded.bins[..block]);
        }
        example.weights.clear();
        example.weights.resize(block, 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two sources: ramps of slope 1 and of slope 100, the second with a
    /// missing value in one series.
    fn corpus() -> Corpus {
        let mut slow = Table::new(64);
        let mut fast = Table::new(64);
        for series in 0..8 {
            let ramp = |slope: f64| -> Vec<f64> {
                (0..64)
                    .map(|step| f64::from(series) + slope * f64::from(step))
                    .collect()
            };
            slow.push(&ramp(1.0));
            let mut steep = ramp(100.0);
            if series == 3 {
                steep[20] = f64::NAN;
            }
            fast.push(&steep);
        }
        Corpus::of(vec![(Source::Sinewave, slow), (Source::OuProcess, fast)])
    }

    #[test]
    fn windows_are_whole_pieces_of_series_drawn_by_the_weights() {
        let corpus = corpus();
        assert_eq!(corpus.values(), 2 * 8 * 64);
        let sampler = Sampler::new(&corpus, 12, 4).expect("valid");
        let mut rng = Rng::from_key(&[1]);
        let mut window = Vec::new();
        let (mut fast, mut contexts) = (0, [0_usize; 13]);
        for _ in 0..4000 {
            let (place, context) = sampler.draw(&mut rng, &mut window);
            assert_eq!(window.len(), 12);
            let slope = if place == 0 { 1.0 } else { 100.0 };
            assert!(window.windows(2).all(|pair| pair[1] - pair[0] == slope));
            fast += place;
            contexts[context] += 1;
        }
        // OU weighs 3 to the sine's 1.
        assert!((2900..3100).contains(&fast), "{fast} of the fast source");
        // The context is uniform from 4 to the whole window.
        assert!(contexts[..4].iter().all(|count| *count == 0));
        assert!(
            contexts[4..].iter().all(|count| (350..550).contains(count)),
            "{contexts:?}"
        );
    }

    #[test]
    fn a_ramp_is_one_token_in_the_units_of_its_context() {
        let corpus = corpus();
        let codec = Codec::new(64, 4.0, 16);
        let windows = Windows::new(&corpus, codec, 10, 4, 5).expect("valid");
        let mut example = Example::default();
        for index in 0..50 {
            windows.fill(3, index, &mut example);
            assert_eq!(example.tokens.len(), 10);
            assert_eq!((example.targets.len(), example.bins.len()), (10, 10));
            assert_eq!(example.weights, [1.0; 10]);
            // A step of one unit is 31 / 4 levels, rounded on the grid from
            // the first value: the tokens are 32 + 7 or 32 + 8, whatever
            // the slope of the ramp.
            assert!(example.tokens.iter().all(|token| (39..=40).contains(token)));
            assert_eq!(example.tokens[1..], example.targets[..9]);
            // The level rises through the bins.
            assert!(example.bins.windows(2).all(|pair| pair[0] <= pair[1]));
            assert!(example.bins[0] < example.bins[9]);
        }
        // An example is a function of its batch and its index.
        let batch = |batch: u64| -> Vec<Example> {
            let mut examples = vec![Example::default(); 20];
            let slots = examples.iter_mut().enumerate();
            slots.for_each(|(index, example)| windows.fill(batch, index, example));
            examples
        };
        assert_eq!(batch(3), batch(3));
        assert_ne!(batch(3), batch(4));
    }

    #[test]
    fn the_domain_of_ramps_is_a_unit() {
        let corpus = corpus();
        let sampler = Sampler::new(&corpus, 12, 2).expect("valid");
        let domain = sampler.domain(200, 0.999, 9);
        assert!((domain - 1.0).abs() < 1e-9, "{domain}");
    }

    #[test]
    fn windows_longer_than_the_series_are_refused() {
        let corpus = corpus();
        assert!(Sampler::new(&corpus, 65, 4).is_err());
        assert!(Sampler::new(&corpus, 12, 13).is_err());
        assert!(Sampler::new(&corpus, 12, 0).is_err());
        assert!(Sampler::new(&Corpus::of(Vec::new()), 12, 4).is_err());
        assert!(Windows::new(&corpus, Codec::new(64, 4.0, 0), 62, 4, 1).is_ok());
        assert!(Windows::new(&corpus, Codec::new(64, 4.0, 0), 63, 4, 1).is_err());
    }
}
