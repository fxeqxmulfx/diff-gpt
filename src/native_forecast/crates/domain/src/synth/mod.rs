//! The synthetic sources: Rust ports of the generators of TempoPFN
//! (<https://github.com/automl/TempoPFN>, Apache-2.0, commit `5969ec6`), each
//! with the parameters that TempoPFN's `continuous_generation.py` gives it,
//! and of the two offline augmentations of their series ([`augment`]).
//!
//! A source makes its series in chunks of [`CHUNK`], as TempoPFN's driver asks
//! for them, since some sources couple the series of a chunk.

pub mod anomaly;
pub mod audio;
pub mod augment;
pub mod bumps;
pub mod cauker;
pub mod forecast_pfn;
pub mod gp;
pub mod gpytorch;
pub mod kernel_synth;
pub mod kernels;
pub mod ou;
pub mod pyo;
pub mod sawtooth;
pub mod sinewave;
pub mod spike;
pub mod step;

use crate::calendar::Frequency;
use crate::random::Rng;

/// The length of the series of the corpus.
pub const LENGTH: usize = 2048;

/// The number of series a source makes at a time.
pub const CHUNK: usize = 64;

/// The word that keys the stream of the calendars of a chunk, apart from the
/// streams of its series.
const CALENDAR: u64 = u64::MAX;

/// A source of the corpus.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Source {
    /// The prior of ForecastPFN: trends and calendar seasonalities.
    ForecastPfn,
    /// Draws from Gaussian processes of composite kernels, with the rounding
    /// of gpytorch in float32.
    Gp,
    /// KernelSynth: draws from Gaussian processes of composite kernels.
    Kernel,
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
    /// CauKer: the nodes of random structural causal models.
    CaukerUnivariate,
    /// Regime-switching Ornstein-Uhlenbeck processes.
    OuProcess,
    /// pyo: a slow saw, noise of a varying level and a decaying sine.
    AudioFinancialVolatility,
    /// pyo: brown noise through a bank of low-passes.
    AudioMultiScaleFractal,
    /// pyo: saws, bursts of noise and envelopes.
    AudioNetworkTopology,
    /// pyo: sines under envelopes that metronomes restart.
    AudioStochasticRhythm,
    /// Series of the generators above, scaled, mixed and edited one by one.
    AugmentedPerSample,
    /// Series of the generators above, scaled, mixed and edited in batches of
    /// which one series is kept.
    AugmentedTempBatch,
}

impl Source {
    /// Every source, in the order of TempoPFN's `configs/example.yaml`.
    pub const ALL: [Self; 16] = [
        Self::ForecastPfn,
        Self::Gp,
        Self::Kernel,
        Self::Sawtooth,
        Self::Sinewave,
        Self::Step,
        Self::Anomaly,
        Self::Spike,
        Self::CaukerUnivariate,
        Self::OuProcess,
        Self::AudioFinancialVolatility,
        Self::AudioMultiScaleFractal,
        Self::AudioNetworkTopology,
        Self::AudioStochasticRhythm,
        Self::AugmentedPerSample,
        Self::AugmentedTempBatch,
    ];

    /// The name of the source in TempoPFN and in the corpus. The augmented
    /// sources carry the length of the series of TempoPFN's corpus.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::ForecastPfn => "forecast_pfn",
            Self::Gp => "gp",
            Self::Kernel => "kernel",
            Self::CaukerUnivariate => "cauker_univariate",
            Self::Sawtooth => "sawtooth",
            Self::Sinewave => "sinewave",
            Self::Step => "step",
            Self::Anomaly => "anomaly",
            Self::Spike => "spike",
            Self::OuProcess => "ou_process",
            Self::AudioFinancialVolatility => "audio_financial_volatility",
            Self::AudioMultiScaleFractal => "audio_multi_scale_fractal",
            Self::AudioNetworkTopology => "audio_network_topology",
            Self::AudioStochasticRhythm => "audio_stochastic_rhythm",
            Self::AugmentedPerSample => "augmented_per_sample_2048",
            Self::AugmentedTempBatch => "augmented_temp_batch_2048",
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
            Self::ForecastPfn => 0,
            Self::Gp => 1,
            Self::Kernel => 2,
            Self::CaukerUnivariate => 8,
            Self::Sawtooth => 3,
            Self::Sinewave => 4,
            Self::Step => 5,
            Self::Anomaly => 6,
            Self::Spike => 7,
            Self::OuProcess => 9,
            Self::AudioFinancialVolatility => 10,
            Self::AudioMultiScaleFractal => 11,
            Self::AudioNetworkTopology => 12,
            Self::AudioStochasticRhythm => 13,
            Self::AugmentedPerSample => 14,
            Self::AugmentedTempBatch => 15,
        }
    }

    /// The weight of the source in TempoPFN's training mixture, the
    /// `generator_proportions` of `configs/example.yaml`.
    #[must_use]
    pub const fn weight(self) -> f64 {
        match self {
            Self::ForecastPfn
            | Self::Gp
            | Self::Kernel
            | Self::Sawtooth
            | Self::Sinewave
            | Self::Step
            | Self::Anomaly
            | Self::Spike
            | Self::CaukerUnivariate => 1.0,
            Self::OuProcess => 3.0,
            Self::AudioFinancialVolatility | Self::AudioMultiScaleFractal => 0.1,
            Self::AudioNetworkTopology | Self::AudioStochasticRhythm => 0.5,
            Self::AugmentedPerSample | Self::AugmentedTempBatch => 2.0,
        }
    }

    /// The number of consecutive series of a chunk that come from one random
    /// model and so depend on each other: 1 where the series are independent.
    #[must_use]
    pub const fn group(self) -> usize {
        match self {
            Self::CaukerUnivariate => cauker::CauKer::NODES,
            _ => 1,
        }
    }

    /// Whether the source makes its series from those of the other sources:
    /// it then takes them from an [`augment::Pool`].
    #[must_use]
    pub const fn augments(self) -> bool {
        matches!(self, Self::AugmentedPerSample | Self::AugmentedTempBatch)
    }

    /// A chunk of [`CHUNK`] series of `length` steps. An augmenting source
    /// has its base series made on the spot ([`augment::Fresh`]), which is
    /// slow: see [`Self::chunk_from`].
    #[must_use]
    pub fn chunk(self, rng: &mut Rng, length: usize) -> Vec<Vec<f64>> {
        self.chunk_from(rng, length, &mut augment::Fresh::new(length))
    }

    /// A chunk of [`CHUNK`] series of `length` steps; an augmenting source
    /// takes its base series, which must have that length, from `pool`.
    #[must_use]
    pub fn chunk_from(
        self,
        rng: &mut Rng,
        length: usize,
        pool: &mut dyn augment::Pool,
    ) -> Vec<Vec<f64>> {
        // Two sources couple the series of a chunk.
        match self {
            Self::CaukerUnivariate => return cauker::CauKer::corpus().chunk(rng, length, CHUNK),
            Self::ForecastPfn => {
                return forecast_pfn::ForecastPfn::default().chunk(rng, length, CHUNK);
            }
            _ => {}
        }
        let mut one = |rng: &mut Rng| match self {
            Self::Gp => gp::GaussianProcess::default().series(rng, length),
            Self::Kernel => kernel_synth::KernelSynth::default().series(rng, length),
            Self::CaukerUnivariate | Self::ForecastPfn => unreachable!("made above"),
            Self::Sawtooth => sawtooth::Sawtooth::default().series(rng, length),
            Self::Sinewave => sinewave::SineWave::default().series(rng, length),
            Self::Step => step::Step::default().series(rng, length),
            Self::Anomaly => anomaly::Anomaly::default().series(rng, length),
            Self::Spike => spike::Spike::default().series(rng, length),
            Self::OuProcess => ou::Ou::default().series(rng, length),
            Self::AudioFinancialVolatility => {
                audio::FinancialVolatility::default().series(rng, length)
            }
            Self::AudioMultiScaleFractal => audio::MultiScaleFractal::default().series(rng, length),
            Self::AudioNetworkTopology => audio::NetworkTopology::default().series(rng, length),
            Self::AudioStochasticRhythm => audio::StochasticRhythm::default().series(rng, length),
            Self::AugmentedPerSample => augment::pipelines::PerSample::default().series(rng, pool),
            Self::AugmentedTempBatch => augment::pipelines::TempBatch::default().series(rng, pool),
        };
        (0..CHUNK as u64)
            .map(|index| one(&mut rng.fork(index)))
            .collect()
    }

    /// The sampling frequency and the first timestamp of each series of the
    /// chunk that [`Self::chunk`] draws from a stream in the state of `rng`:
    /// TempoPFN's driver stores them beside the series, and the calendar edit
    /// of an augmentation reads them. Those of ForecastPFN are the ones its
    /// series were made with; the others are drawn alike, apart from the
    /// series.
    #[must_use]
    pub fn calendars(self, rng: &Rng, length: usize) -> Vec<(Frequency, i64)> {
        let mut rng = rng.clone();
        if self == Self::ForecastPfn {
            return forecast_pfn::ForecastPfn::calendars(&mut rng, length, CHUNK);
        }
        let mut rng = Rng::from_key(&[rng.next_u64(), rng.next_u64(), CALENDAR]);
        (0..CHUNK)
            .map(|_| {
                let frequency = Frequency::select(&mut rng, length);
                (frequency, frequency.start(&mut rng, length))
            })
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
        let augmenting = Source::ALL.iter().filter(|source| source.augments());
        assert_eq!(augmenting.count(), 2);
    }

    #[test]
    fn a_chunk_is_reproducible_and_finite() {
        // Shorter series than the corpus has, to keep the test quick.
        let length = LENGTH / 8;
        for source in Source::ALL {
            let chunk = source.chunk(&mut Rng::from_key(&[1, source.id()]), length);
            assert_eq!(chunk.len(), CHUNK);
            assert!(chunk.iter().all(|series| series.len() == length));
            assert!(chunk.iter().flatten().all(|value| value.is_finite()));
            assert_eq!(
                chunk,
                source.chunk(&mut Rng::from_key(&[1, source.id()]), length)
            );
            assert_ne!(chunk[0], chunk[1]);
        }
    }

    #[test]
    fn the_calendars_of_a_chunk_are_reproducible_and_apart_from_its_series() {
        for source in [Source::ForecastPfn, Source::Sinewave] {
            let rng = Rng::from_key(&[2, source.id()]);
            let calendars = source.calendars(&rng, LENGTH);
            assert_eq!(calendars.len(), CHUNK);
            assert_eq!(calendars, source.calendars(&rng, LENGTH));
            // Hours to weeks suit 2048 steps; a series ends before 2200.
            for (frequency, start) in &calendars {
                assert!(frequency.step().is_some(), "{frequency:?}");
                assert!(
                    *start < crate::calendar::days_from_civil(2200, 1, 1) * crate::calendar::DAY
                );
            }
            let mut distinct: Vec<i64> = calendars.iter().map(|(_, start)| *start).collect();
            distinct.sort_unstable();
            distinct.dedup();
            assert_eq!(distinct.len(), CHUNK);
        }
        // ForecastPFN's are the first draws of the streams of its series.
        let mut rng = Rng::from_key(&[3]);
        let calendars = Source::ForecastPfn.calendars(&rng, 512);
        let mut first = rng.fork(0);
        let frequency = Frequency::select(&mut first, 512);
        assert_eq!(calendars[0], (frequency, frequency.start(&mut first, 512)));
    }
}
