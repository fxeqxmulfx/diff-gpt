//! The synthetic corpus: its shards, and the check of a source against the
//! corpus that TempoPFN's Python generators made.
//!
//! Chunk `c` of a source of a split is drawn from the random stream of the
//! key `[VERSION, split, source, c]`, so a series does not depend on the size
//! of the shards, on the other sources or on the threads that make it. The
//! two augmenting sources make their series from those of the generators: a
//! chunk of theirs depends, besides, on how many series of each generator
//! the corpus is to have ([`Base`]).

use diff_gpt_forecast_domain::calendar::Frequency;
use diff_gpt_forecast_domain::random::Rng;
use diff_gpt_forecast_domain::stats::{self, FeatureComparison, Features};
use diff_gpt_forecast_domain::synth::augment::{Drawn, PROPORTIONS, Pool};
use diff_gpt_forecast_domain::synth::{CHUNK, Source};
use diff_gpt_forecast_domain::table::Table;

use crate::Result;
use crate::ports::{Executor, SeriesReader, SeriesWriter, Split};

/// The version of the corpus: the first word of the key of every stream.
/// A change to a source that changes its series takes a new version.
pub const VERSION: u64 = 1;

/// The stream of chunk `chunk` of `source` in `split`.
fn stream(split: Split, source: Source, chunk: usize) -> Rng {
    Rng::from_key(&[VERSION, split.id(), source.id(), chunk as u64])
}

/// The series of chunk `chunk` of `source` in `split`. An augmenting source
/// takes its base series from `base`; without one it makes them on the spot,
/// slowly, and they are other series than a corpus holds.
#[must_use]
pub fn chunk(
    split: Split,
    source: Source,
    chunk: usize,
    length: usize,
    base: Option<&Base>,
) -> Vec<Vec<f64>> {
    let mut rng = stream(split, source, chunk);
    match base {
        Some(base) => source.chunk_from(&mut rng, length, &mut base.pool()),
        None => source.chunk(&mut rng, length),
    }
}

/// The first series of every generator in a split, held in memory with the
/// calendar of each: what the augmenting sources make their series from.
#[derive(Clone, Debug)]
pub struct Base {
    stocks: Vec<Stock>,
}

/// The series of one generator, with the frequency and the first timestamp
/// of each.
#[derive(Clone, Debug)]
struct Stock {
    source: Source,
    series: Table,
    calendars: Vec<(Frequency, i64)>,
}

impl Base {
    /// Reads from `corpus` the first `series` series of every generator in
    /// `split`, which must be series of `length` steps.
    ///
    /// # Errors
    ///
    /// If the corpus holds fewer series of a generator, or series of another
    /// length, or cannot be read.
    pub fn load<E: Executor, R: SeriesReader>(
        executor: &E,
        corpus: &R,
        split: Split,
        series: usize,
        length: usize,
    ) -> Result<Self> {
        let mut stocks = Vec::with_capacity(PROPORTIONS.len());
        for (source, _) in PROPORTIONS {
            let held = corpus.series(split, source)?;
            if held < series || series == 0 {
                return Err(format!(
                    "{}/{} holds {held} series and the augmented sources need {series}: \
                     generate the generators first",
                    split.name(),
                    source.name()
                )
                .into());
            }
            let table = corpus.read(split, source, 0, series)?;
            if table.length() != length {
                return Err(format!(
                    "{}/{} holds series of {} steps, not {length}",
                    split.name(),
                    source.name(),
                    table.length()
                )
                .into());
            }
            let calendars = executor
                .map(series.div_ceil(CHUNK), |index| {
                    source.calendars(&stream(split, source, index), length)
                })
                .into_iter()
                .flatten()
                .take(series)
                .collect();
            stocks.push(Stock {
                source,
                series: table,
                calendars,
            });
        }
        Ok(Self { stocks })
    }

    /// A pool that hands out these series, each as likely as another of its
    /// source. (TempoPFN reads its files in turn: every series as often.)
    #[must_use]
    pub fn pool(&self) -> impl Pool + '_ {
        Rows { base: self }
    }
}

/// The pool of a [`Base`].
struct Rows<'a> {
    base: &'a Base,
}

impl Pool for Rows<'_> {
    fn draw(&mut self, source: Source, rng: &mut Rng) -> Drawn {
        let stock = self
            .base
            .stocks
            .iter()
            .find(|stock| stock.source == source)
            .expect("the base holds every generator");
        let place = rng.index(stock.series.rows());
        let (frequency, start) = stock.calendars[place];
        Drawn {
            values: stock.series.row(place).to_vec(),
            frequency,
            start,
            place,
        }
    }
}

/// The size of a corpus to make.
#[derive(Clone, Copy, Debug)]
pub struct Shape {
    /// The length of a series.
    pub length: usize,
    /// The number of chunks of [`CHUNK`] series in a shard.
    pub chunks_per_shard: usize,
    /// The number of shards of each source.
    pub shards: usize,
}

impl Shape {
    /// The number of series of each source.
    #[must_use]
    pub const fn series(&self) -> usize {
        self.shards * self.chunks_per_shard * CHUNK
    }
}

/// Writes the shards of `sources` in `split` that `corpus` does not hold yet,
/// and returns how many it wrote. The generators come first: an augmenting
/// source then reads from `corpus` as many series of each generator as it is
/// to have itself.
///
/// # Errors
///
/// If a shard cannot be written, or an augmenting source lacks series of a
/// generator.
pub fn generate<E: Executor, C: SeriesWriter + SeriesReader>(
    executor: &E,
    corpus: &C,
    split: Split,
    sources: &[Source],
    shape: Shape,
    mut progress: impl FnMut(Source, usize),
) -> Result<usize> {
    let mut written = 0;
    let mut base: Option<Base> = None;
    let (augmenting, generators): (Vec<Source>, Vec<Source>) =
        sources.iter().partition(|source| source.augments());
    for source in generators.into_iter().chain(augmenting) {
        for shard in 0..shape.shards {
            if corpus.has(split, source, shard) {
                continue;
            }
            if source.augments() && base.is_none() {
                let series = shape.series();
                base = Some(Base::load(executor, corpus, split, series, shape.length)?);
            }
            let first = shard * shape.chunks_per_shard;
            let chunks = executor.map(shape.chunks_per_shard, |index| {
                chunk(split, source, first + index, shape.length, base.as_ref())
            });
            let mut table = Table::new(shape.length);
            for series in chunks.iter().flatten() {
                table.push(series);
            }
            corpus.write(split, source, shard, &table)?;
            written += 1;
            progress(source, shard);
        }
    }
    Ok(written)
}

/// How the series of a source compare with those of the reference corpus.
#[derive(Clone, Debug)]
pub struct SourceCheck {
    /// The source.
    pub source: Source,
    /// The number of reference series compared.
    pub reference_series: usize,
    /// The number of series of this crate compared.
    pub candidate_series: usize,
    /// The distance that two samples of these sizes from one distribution
    /// exceed with the probability asked for.
    pub critical: f64,
    /// The comparison of each feature.
    pub features: Vec<FeatureComparison>,
}

impl SourceCheck {
    /// The features whose distance exceeds the critical one.
    pub fn failures(&self) -> impl Iterator<Item = &FeatureComparison> {
        self.features
            .iter()
            .filter(|feature| feature.distance > self.critical)
    }
}

/// Compares the first `series` series of `source` in `split`, as this crate
/// makes them, with as many of `reference`, feature by feature; `level` is
/// the probability with which a feature of equal distributions fails. An
/// augmenting source takes its base series from `base`.
///
/// # Errors
///
/// If the reference cannot be read, or an augmenting source has no base.
pub fn check<E: Executor, R: SeriesReader>(
    executor: &E,
    reference: &R,
    split: Split,
    source: Source,
    series: usize,
    level: f64,
    base: Option<&Base>,
) -> Result<SourceCheck> {
    if source.augments() && base.is_none() {
        return Err(format!("{} needs the series of the generators", source.name()).into());
    }
    // The test takes independent series: of the series that share a model,
    // only the first is compared, in the reference and in the candidate.
    let independent = |row: usize| (row % CHUNK).is_multiple_of(source.group());
    let available = reference.series(split, source)?;
    let table = reference.read(split, source, 0, series.min(available))?;
    let rows: Vec<usize> = (0..table.rows()).filter(|&row| independent(row)).collect();
    let reference_features: Vec<Features> =
        executor.map(rows.len(), |index| stats::features(table.row(rows[index])));

    let chunks = series.div_ceil(CHUNK);
    let candidate_features: Vec<Features> = executor
        .map(chunks, |index| {
            chunk(split, source, index, table.length(), base)
                .iter()
                .enumerate()
                .filter(|&(place, _)| index * CHUNK + place < series && independent(place))
                .map(|(_, values)| {
                    let single: Vec<f32> = values.iter().map(|&value| value as f32).collect();
                    stats::features(&single)
                })
                .collect::<Vec<Features>>()
        })
        .into_iter()
        .flatten()
        .collect();

    Ok(SourceCheck {
        source,
        reference_series: reference_features.len(),
        candidate_series: candidate_features.len(),
        critical: stats::ks_critical(reference_features.len(), candidate_features.len(), level),
        features: stats::compare(&reference_features, &candidate_features),
    })
}

/// Checks each of `sources` as [`check`] does, reporting each as it is done.
/// The augmenting sources make their series as [`generate`] does for a
/// corpus of `series` series a source, rounded up to whole chunks: from as
/// many series of each generator, read from `corpus`.
///
/// # Errors
///
/// If the reference cannot be read, or `corpus` lacks series of a generator
/// that an augmenting source needs.
#[expect(
    clippy::too_many_arguments,
    reason = "the parts of a check: where from, what and how strictly"
)]
pub fn check_all<E: Executor, R: SeriesReader, C: SeriesReader>(
    executor: &E,
    reference: &R,
    corpus: &C,
    split: Split,
    sources: &[Source],
    series: usize,
    level: f64,
    mut report: impl FnMut(&SourceCheck),
) -> Result<Vec<SourceCheck>> {
    let mut base: Option<Base> = None;
    let mut checks = Vec::with_capacity(sources.len());
    for &source in sources {
        if source.augments() && base.is_none() {
            let length = reference.read(split, source, 0, 1)?.length();
            let rows = series.div_ceil(CHUNK) * CHUNK;
            base = Some(Base::load(executor, corpus, split, rows, length)?);
        }
        let check = check(
            executor,
            reference,
            split,
            source,
            series,
            level,
            base.as_ref(),
        )?;
        report(&check);
        checks.push(check);
    }
    Ok(checks)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Mutex;

    use super::*;
    use crate::ports::Serial;

    /// A corpus held in memory: a real, if small, implementation of both ports.
    #[derive(Default)]
    struct Memory {
        shards: Mutex<BTreeMap<(u64, u64, usize), Table>>,
    }

    impl SeriesWriter for Memory {
        fn has(&self, split: Split, source: Source, shard: usize) -> bool {
            let shards = self.shards.lock().expect("no writer panicked");
            shards.contains_key(&(split.id(), source.id(), shard))
        }

        fn write(&self, split: Split, source: Source, shard: usize, series: &Table) -> Result<()> {
            let mut shards = self.shards.lock().expect("no writer panicked");
            shards.insert((split.id(), source.id(), shard), series.clone());
            Ok(())
        }
    }

    impl SeriesReader for Memory {
        fn series(&self, split: Split, source: Source) -> Result<usize> {
            let shards = self.shards.lock().expect("no writer panicked");
            Ok(shards
                .iter()
                .filter(|(key, _)| (key.0, key.1) == (split.id(), source.id()))
                .map(|(_, table)| table.rows())
                .sum())
        }

        fn read(&self, split: Split, source: Source, first: usize, count: usize) -> Result<Table> {
            let shards = self.shards.lock().expect("no writer panicked");
            let mut all: Option<Table> = None;
            for (key, table) in shards.iter() {
                if (key.0, key.1) == (split.id(), source.id()) {
                    all.get_or_insert_with(|| Table::new(table.length()))
                        .append(table);
                }
            }
            let all = all.ok_or("no such source")?;
            if first + count > all.rows() {
                return Err("not that many series".into());
            }
            let length = all.length();
            let values = all.values()[first * length..(first + count) * length].to_vec();
            Ok(Table::from_values(length, values))
        }
    }

    const SHAPE: Shape = Shape {
        length: 256,
        chunks_per_shard: 2,
        shards: 3,
    };

    #[test]
    fn generation_fills_the_missing_shards_only() {
        let memory = Memory::default();
        let sources = [Source::Sawtooth, Source::Anomaly];
        let mut seen = Vec::new();
        let written = generate(
            &Serial,
            &memory,
            Split::Train,
            &sources,
            SHAPE,
            |source, shard| {
                seen.push((source, shard));
            },
        )
        .expect("memory does not fail");
        assert_eq!(written, 6);
        assert_eq!(seen.len(), 6);
        assert_eq!(
            memory.series(Split::Train, Source::Anomaly).expect("held"),
            6 * CHUNK
        );
        let again = generate(&Serial, &memory, Split::Train, &sources, SHAPE, |_, _| {})
            .expect("memory does not fail");
        assert_eq!(again, 0);
    }

    #[test]
    fn a_series_does_not_depend_on_the_shard_size() {
        let (small, large) = (Memory::default(), Memory::default());
        let sources = [Source::Sinewave];
        let coarse = Shape {
            chunks_per_shard: 6,
            shards: 1,
            ..SHAPE
        };
        generate(&Serial, &small, Split::Holdout, &sources, SHAPE, |_, _| {}).expect("written");
        generate(&Serial, &large, Split::Holdout, &sources, coarse, |_, _| {}).expect("written");
        let count = 6 * CHUNK;
        assert_eq!(
            small
                .read(Split::Holdout, Source::Sinewave, 0, count)
                .expect("held"),
            large
                .read(Split::Holdout, Source::Sinewave, 0, count)
                .expect("held"),
        );
        let train = chunk(Split::Train, Source::Sinewave, 0, 256, None);
        assert_ne!(train, chunk(Split::Holdout, Source::Sinewave, 0, 256, None));
    }

    #[test]
    fn a_source_passes_against_its_own_other_split_and_fails_against_another() {
        let memory = Memory::default();
        let shape = Shape {
            length: 512,
            chunks_per_shard: 8,
            shards: 1,
        };
        generate(
            &Serial,
            &memory,
            Split::Holdout,
            &[Source::Sawtooth],
            shape,
            |_, _| {},
        )
        .expect("written");
        let same = check(
            &Serial,
            &memory,
            Split::Holdout,
            Source::Sawtooth,
            512,
            1e-4,
            None,
        )
        .expect("checked");
        // The same split gives the very same series: every distance is 0.
        assert!(same.features.iter().all(|feature| feature.distance == 0.0));
        assert_eq!((same.reference_series, same.candidate_series), (512, 512));

        // The reference holds sawtooth series under the name of another split
        // of the candidate: another stream of the same distribution.
        let mirrored = Memory::default();
        let train = memory
            .read(Split::Holdout, Source::Sawtooth, 0, 512)
            .expect("held");
        mirrored
            .write(Split::Train, Source::Sawtooth, 0, &train)
            .expect("written");
        let other = check(
            &Serial,
            &mirrored,
            Split::Train,
            Source::Sawtooth,
            512,
            1e-4,
            None,
        )
        .expect("checked");
        assert_eq!(other.failures().count(), 0, "{:?}", other.failures().next());
        assert!(other.features.iter().any(|feature| feature.distance > 0.0));

        // Sine waves under the name of the sawtooth source fail the check.
        let wrong = Memory::default();
        let sines = chunk(Split::Train, Source::Sinewave, 0, 512, None);
        let mut table = Table::new(512);
        for _ in 0..8 {
            for series in &sines {
                table.push(series);
            }
        }
        wrong
            .write(Split::Train, Source::Sawtooth, 0, &table)
            .expect("written");
        let failed = check(
            &Serial,
            &wrong,
            Split::Train,
            Source::Sawtooth,
            512,
            1e-4,
            None,
        )
        .expect("checked");
        assert!(failed.failures().count() > 3);
    }

    #[test]
    fn the_augmenting_sources_are_made_last_from_the_generators_of_the_corpus() {
        let memory = Memory::default();
        let shape = Shape {
            length: 256,
            chunks_per_shard: 1,
            shards: 2,
        };
        // Asked for first, the augmenting sources are still written last.
        let sources: Vec<Source> = Source::ALL.iter().rev().copied().collect();
        let mut order = Vec::new();
        generate(
            &Serial,
            &memory,
            Split::Train,
            &sources,
            shape,
            |source, _| {
                order.push(source);
            },
        )
        .expect("written");
        assert_eq!(order.len(), 32);
        assert!(order[..28].iter().all(|source| !source.augments()));
        assert!(order[28..].iter().all(|source| source.augments()));

        // The corpus holds what the check makes: the same base, the same
        // streams.
        let series = shape.series();
        let mut reported = 0;
        let checks = check_all(
            &Serial,
            &memory,
            &memory,
            Split::Train,
            &[Source::AugmentedPerSample, Source::AugmentedTempBatch],
            series,
            1e-4,
            |_| reported += 1,
        )
        .expect("checked");
        assert_eq!((checks.len(), reported), (2, 2));
        for check in &checks {
            assert_eq!(check.candidate_series, series);
            assert!(check.features.iter().all(|feature| feature.distance == 0.0));
        }

        // An augmented series is none of the generators' series.
        let base = Base::load(&Serial, &memory, Split::Train, series, 256).expect("loaded");
        let augmented = memory
            .read(Split::Train, Source::AugmentedTempBatch, 0, series)
            .expect("held");
        let mut pool = base.pool();
        let drawn = pool.draw(Source::Sawtooth, &mut Rng::from_key(&[5]));
        assert_eq!(drawn.values.len(), 256);
        assert!(drawn.place < series);
        assert!(augmented.iter().all(|row| row != drawn.values.as_slice()));

        // Without the generators there is nothing to augment.
        let alone = check(
            &Serial,
            &memory,
            Split::Train,
            Source::AugmentedPerSample,
            series,
            1e-4,
            None,
        );
        assert!(alone.is_err());
        let empty = Memory::default();
        let failed = generate(
            &Serial,
            &empty,
            Split::Train,
            &[Source::AugmentedTempBatch],
            shape,
            |_, _| {},
        );
        let message = failed.expect_err("no generator").to_string();
        assert!(
            message.contains("generate the generators first"),
            "{message}"
        );
        // Series of another length are refused.
        assert!(Base::load(&Serial, &memory, Split::Train, series, 512).is_err());
        assert!(Base::load(&Serial, &memory, Split::Train, series + 1, 256).is_err());
    }
}
