//! The synthetic corpus: its shards, and the check of a source against the
//! corpus that TempoPFN's Python generators made.
//!
//! Chunk `c` of a source of a split is drawn from the random stream of the
//! key `[VERSION, split, source, c]`, so a series does not depend on the size
//! of the shards, on the other sources or on the threads that make it.

use diff_gpt_forecast_domain::random::Rng;
use diff_gpt_forecast_domain::stats::{self, FeatureComparison, Features};
use diff_gpt_forecast_domain::synth::{CHUNK, Source};
use diff_gpt_forecast_domain::table::Table;

use crate::Result;
use crate::ports::{Executor, SeriesReader, SeriesWriter, Split};

/// The version of the corpus: the first word of the key of every stream.
/// A change to a source that changes its series takes a new version.
pub const VERSION: u64 = 1;

/// The series of chunk `chunk` of `source` in `split`.
#[must_use]
pub fn chunk(split: Split, source: Source, chunk: usize, length: usize) -> Vec<Vec<f64>> {
    let mut rng = Rng::from_key(&[VERSION, split.id(), source.id(), chunk as u64]);
    source.chunk(&mut rng, length)
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

/// Writes the shards of `sources` in `split` that `writer` does not hold yet,
/// and returns how many it wrote.
///
/// # Errors
///
/// If a shard cannot be written.
pub fn generate<E: Executor, W: SeriesWriter>(
    executor: &E,
    writer: &W,
    split: Split,
    sources: &[Source],
    shape: Shape,
    mut progress: impl FnMut(Source, usize),
) -> Result<usize> {
    let mut written = 0;
    for &source in sources {
        for shard in 0..shape.shards {
            if writer.has(split, source, shard) {
                continue;
            }
            let first = shard * shape.chunks_per_shard;
            let chunks = executor.map(shape.chunks_per_shard, |index| {
                chunk(split, source, first + index, shape.length)
            });
            let mut table = Table::new(shape.length);
            for series in chunks.iter().flatten() {
                table.push(series);
            }
            writer.write(split, source, shard, &table)?;
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
/// the probability with which a feature of equal distributions fails.
///
/// # Errors
///
/// If the reference cannot be read.
pub fn check<E: Executor, R: SeriesReader>(
    executor: &E,
    reference: &R,
    split: Split,
    source: Source,
    series: usize,
    level: f64,
) -> Result<SourceCheck> {
    let available = reference.series(split, source)?;
    let table = reference.read(split, source, 0, series.min(available))?;
    let reference_features: Vec<Features> =
        executor.map(table.rows(), |row| stats::features(table.row(row)));

    let chunks = series.div_ceil(CHUNK);
    let candidate_features: Vec<Features> = executor
        .map(chunks, |index| {
            chunk(split, source, index, table.length())
                .iter()
                .map(|values| {
                    let single: Vec<f32> = values.iter().map(|&value| value as f32).collect();
                    stats::features(&single)
                })
                .collect::<Vec<Features>>()
        })
        .into_iter()
        .flatten()
        .take(series)
        .collect();

    Ok(SourceCheck {
        source,
        reference_series: reference_features.len(),
        candidate_series: candidate_features.len(),
        critical: stats::ks_critical(reference_features.len(), candidate_features.len(), level),
        features: stats::compare(&reference_features, &candidate_features),
    })
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
        let train = chunk(Split::Train, Source::Sinewave, 0, 256);
        assert_ne!(train, chunk(Split::Holdout, Source::Sinewave, 0, 256));
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
        )
        .expect("checked");
        assert_eq!(other.failures().count(), 0, "{:?}", other.failures().next());
        assert!(other.features.iter().any(|feature| feature.distance > 0.0));

        // Sine waves under the name of the sawtooth source fail the check.
        let wrong = Memory::default();
        let sines = chunk(Split::Train, Source::Sinewave, 0, 512);
        let mut table = Table::new(512);
        for _ in 0..8 {
            for series in &sines {
                table.push(series);
            }
        }
        wrong
            .write(Split::Train, Source::Sawtooth, 0, &table)
            .expect("written");
        let failed =
            check(&Serial, &wrong, Split::Train, Source::Sawtooth, 512, 1e-4).expect("checked");
        assert!(failed.failures().count() > 3);
    }
}
