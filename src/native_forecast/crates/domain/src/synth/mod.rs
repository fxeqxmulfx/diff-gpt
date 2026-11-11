//! The synthetic sources: Rust ports of the generators of TempoPFN
//! (<https://github.com/automl/TempoPFN>, Apache-2.0, commit `5969ec6`), each
//! with the parameters that TempoPFN's `continuous_generation.py` gives it.
//!
//! A source makes its series in chunks of [`CHUNK`], as TempoPFN's driver asks
//! for them, since some sources couple the series of a chunk.

pub mod anomaly;
pub mod ou;
pub mod sawtooth;
pub mod sinewave;
pub mod spike;
pub mod step;

use crate::random::Rng;

/// The length of the series of the corpus.
pub const LENGTH: usize = 2048;

/// The number of series a source makes at a time.
pub const CHUNK: usize = 64;

/// A source of the corpus.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Source {
    /// Sawtooth waves.
    Sawtooth,
    /// Sums of modulated sinusoids.
    Sinewave,
    /// Piecewise constant series.
    Step,
    /// Periodic one-step spikes on a constant level.
    Anomaly,
    /// A few wide spikes on a constant baseline.
    Spike,
    /// Regime-switching Ornstein-Uhlenbeck processes.
    OuProcess,
}

impl Source {
    /// Every source, in the order of TempoPFN's `configs/example.yaml`.
    pub const ALL: [Self; 6] = [
        Self::Sawtooth,
        Self::Sinewave,
        Self::Step,
        Self::Anomaly,
        Self::Spike,
        Self::OuProcess,
    ];

    /// The name of the source in TempoPFN and in the corpus.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Sawtooth => "sawtooth",
            Self::Sinewave => "sinewave",
            Self::Step => "step",
            Self::Anomaly => "anomaly",
            Self::Spike => "spike",
            Self::OuProcess => "ou_process",
        }
    }

    /// The source of that name.
    #[must_use]
    pub fn named(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|source| source.name() == name)
    }

    /// The place of the source among the 16 of TempoPFN's
    /// `configs/example.yaml`, which keys its random streams: a source keeps
    /// its series when others are added.
    #[must_use]
    pub const fn id(self) -> u64 {
        match self {
            Self::Sawtooth => 3,
            Self::Sinewave => 4,
            Self::Step => 5,
            Self::Anomaly => 6,
            Self::Spike => 7,
            Self::OuProcess => 9,
        }
    }

    /// The weight of the source in TempoPFN's training mixture, the
    /// `generator_proportions` of `configs/example.yaml`.
    #[must_use]
    pub const fn weight(self) -> f64 {
        match self {
            Self::Sawtooth | Self::Sinewave | Self::Step | Self::Anomaly | Self::Spike => 1.0,
            Self::OuProcess => 3.0,
        }
    }

    /// A chunk of [`CHUNK`] series of `length` steps.
    #[must_use]
    pub fn chunk(self, rng: &mut Rng, length: usize) -> Vec<Vec<f64>> {
        let one = |rng: &mut Rng| match self {
            Self::Sawtooth => sawtooth::Sawtooth::default().series(rng, length),
            Self::Sinewave => sinewave::SineWave::default().series(rng, length),
            Self::Step => step::Step::default().series(rng, length),
            Self::Anomaly => anomaly::Anomaly::default().series(rng, length),
            Self::Spike => spike::Spike::default().series(rng, length),
            Self::OuProcess => ou::Ou::default().series(rng, length),
        };
        (0..CHUNK as u64)
            .map(|index| one(&mut rng.fork(index)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_and_ids_identify_the_sources() {
        for source in Source::ALL {
            assert_eq!(Source::named(source.name()), Some(source));
        }
        let mut ids: Vec<u64> = Source::ALL.iter().map(|source| source.id()).collect();
        assert!(ids.is_sorted(), "the sources are in TempoPFN's order");
        ids.dedup();
        assert_eq!(ids.len(), Source::ALL.len());
        assert_eq!(Source::named("no such source"), None);
    }

    #[test]
    fn a_chunk_is_reproducible_and_finite() {
        for source in Source::ALL {
            let chunk = source.chunk(&mut Rng::from_key(&[1, source.id()]), LENGTH);
            assert_eq!(chunk.len(), CHUNK);
            assert!(chunk.iter().all(|series| series.len() == LENGTH));
            assert!(chunk.iter().flatten().all(|value| value.is_finite()));
            assert_eq!(
                chunk,
                source.chunk(&mut Rng::from_key(&[1, source.id()]), LENGTH)
            );
            assert_ne!(chunk[0], chunk[1]);
        }
    }
}
