//! The tokens of a series: DiffGPT's difference tokens on a level grid
//! (`src/diff_gpt/domain/encoder_decoder.py`, order 1, the grid anchored at
//! the first value as its "mash" tokenizer anchors it), in the units of the
//! window's context (`src/diff_gpt/domain/normalization.py`,
//! "mean_abs_step").
//!
//! A window is put in units: its values less the mean of its context, over
//! the context's mean absolute step. A grid of `s` levels a unit starts at
//! the first value, and the token at time `t` is the step of the level,
//! `q_t = L_t − L_{t−1} + V/2`, where `s = (V − 2) / (2D)` gives a step of
//! the domain `D` units the token `V − 1`.
//!
//! Two changes from the Python codec:
//! - A step outside the vocabulary is not lost. The level a token leads to
//!   is the level the decoder will hold, and the next token is the step
//!   from that level to the next value: after a jump the tokens stay at the
//!   edge of the vocabulary until the level has caught up, and the decoded
//!   series is again within half a level of the series. In Python a clipped
//!   step offsets every later value.
//! - Beside each token goes, if asked for, the bin of the level it leads
//!   to: where the value lies among the context's values, in deviations of
//!   the context from its mean. A model that reads steps alone has to add
//!   them up to know where the series is; the bin tells it.

/// The units of a window, read off its context.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Units {
    /// The mean of the observed values.
    pub location: f64,
    /// The mean absolute step between observed neighbours; if that is 0 or
    /// there is none, the mean absolute value; if that is 0 too, 1.
    pub scale: f64,
    /// The deviation of the observed values from their mean; `scale` if it
    /// is 0.
    pub spread: f64,
}

impl Units {
    /// The units of a context. A value that is not finite is missing: it
    /// counts in no mean, and a step needs both of its values.
    #[must_use]
    pub fn of(context: &[f64]) -> Self {
        let seen = || context.iter().copied().filter(|value| value.is_finite());
        let count = seen().count() as f64;
        if count == 0.0 {
            return Self {
                location: 0.0,
                scale: 1.0,
                spread: 1.0,
            };
        }
        let location = seen().sum::<f64>() / count;
        let steps = || {
            let pairs = context.windows(2);
            pairs.filter(|pair| pair[0].is_finite() && pair[1].is_finite())
        };
        let step = steps().map(|pair| (pair[1] - pair[0]).abs()).sum::<f64>()
            / (steps().count() as f64).max(1.0);
        let size = seen().map(f64::abs).sum::<f64>() / count;
        let scale = [step, size, 1.0]
            .into_iter()
            .find(|scale| *scale > 0.0 && scale.is_finite())
            .unwrap_or(1.0);
        let variance = seen().map(|value| (value - location).powi(2)).sum::<f64>() / count;
        let spread = variance.sqrt();
        Self {
            location,
            scale,
            spread: if spread > 0.0 && spread.is_finite() {
                spread
            } else {
                scale
            },
        }
    }
}

/// The number of level bins a unit of `asinh` takes: with [`BIN_KNEE`], the
/// bins are an eighth of a deviation wide at the context's mean.
const BIN_RATE: f64 = 16.0;

/// The deviations from the mean at which the bins start to widen.
const BIN_KNEE: f64 = 2.0;

/// The grid of the tokens.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Codec {
    /// The number of tokens, `V`.
    pub vocab: usize,
    /// The levels of the grid in a unit, `s`.
    pub per_unit: f64,
    /// The number of level bins; 0 if the tokens go alone.
    pub bins: usize,
}

impl Codec {
    /// The codec of `vocab` tokens whose last token is a step of `domain`
    /// units: `s = (V − 2) / (2D)`.
    ///
    /// # Panics
    ///
    /// If there are fewer than 4 tokens, or the domain is not positive.
    #[must_use]
    pub fn new(vocab: usize, domain: f64, bins: usize) -> Self {
        assert!(vocab >= 4, "a vocabulary of {vocab} tokens");
        assert!(domain > 0.0 && domain.is_finite(), "a domain of {domain}");
        Self {
            vocab,
            per_unit: (vocab - 2) as f64 / (2.0 * domain),
            bins,
        }
    }

    /// The token of a step of no level.
    #[must_use]
    pub const fn zero(&self) -> u32 {
        (self.vocab / 2) as u32
    }

    /// The domain `D`: the step, in units, of the last token.
    #[must_use]
    pub fn domain(&self) -> f64 {
        (self.vocab - 2) as f64 / (2.0 * self.per_unit)
    }

    /// The tokens of a window whose first `context` values give its units:
    /// a token, and a bin if the codec has bins, for each value after the
    /// first.
    ///
    /// # Panics
    ///
    /// If the window is empty or `context` is not within it.
    #[must_use]
    pub fn encode(&self, window: &[f64], context: usize) -> Encoded {
        assert!(
            (1..=window.len()).contains(&context),
            "a context of {context} values in a window of {}",
            window.len()
        );
        let mut track = Track::start(Units::of(&window[..context]), window);
        let mut tokens = Vec::with_capacity(window.len() - 1);
        let mut bins = Vec::with_capacity(if self.bins > 0 { window.len() - 1 } else { 0 });
        for value in &window[1..] {
            tokens.push(track.encode(self, *value));
            if self.bins > 0 {
                bins.push(track.bin(self));
            }
        }
        Encoded {
            tokens,
            bins,
            track,
        }
    }
}

/// A window in tokens.
#[derive(Clone, Debug, PartialEq)]
pub struct Encoded {
    /// The token of each value after the first.
    pub tokens: Vec<u32>,
    /// The level bin each token leads to; empty if the codec has no bins.
    pub bins: Vec<u32>,
    /// Where the tokens end: what a forecast goes on from.
    pub track: Track,
}

/// The state of a series in tokens: its units, the anchor of its grid and
/// the level the tokens so far lead to.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Track {
    units: Units,
    /// The first value, in units.
    anchor: f64,
    /// The level: grid steps above the anchor.
    level: i64,
}

impl Track {
    /// The track of a window in `units`, at its first observed value; at
    /// the location of the units if it has none.
    #[must_use]
    pub fn start(units: Units, window: &[f64]) -> Self {
        let first = window.iter().copied().find(|value| value.is_finite());
        Self {
            units,
            anchor: first.map_or(0.0, |first| (first - units.location) / units.scale),
            level: 0,
        }
    }

    /// The units of the track.
    #[must_use]
    pub const fn units(&self) -> &Units {
        &self.units
    }

    /// Moves to the next value and returns its token: the step from the
    /// level the track holds to the level nearest the value, or the token
    /// nearest that step. A missing value is a step of no level.
    pub fn encode(&mut self, codec: &Codec, value: f64) -> u32 {
        if !value.is_finite() {
            return codec.zero();
        }
        let units = (value - self.units.location) / self.units.scale - self.anchor;
        // Far beyond what a series reaches, and within the integers.
        let wanted = (units * codec.per_unit + 0.5)
            .floor()
            .clamp(-1.0e15, 1.0e15) as i64;
        let zero = i64::from(codec.zero());
        let step = (wanted - self.level).clamp(-zero, codec.vocab as i64 - 1 - zero);
        self.level += step;
        (step + zero) as u32
    }

    /// Takes a token and returns the value it leads to.
    ///
    /// # Panics
    ///
    /// If the token is outside the vocabulary.
    pub fn decode(&mut self, codec: &Codec, token: u32) -> f64 {
        assert!(
            (token as usize) < codec.vocab,
            "token {token} of {}",
            codec.vocab
        );
        self.level += i64::from(token) - i64::from(codec.zero());
        self.value(codec)
    }

    /// The value the tokens so far lead to.
    #[must_use]
    pub fn value(&self, codec: &Codec) -> f64 {
        let units = self.anchor + self.level as f64 / codec.per_unit;
        units * self.units.scale + self.units.location
    }

    /// The bin of the level the track holds: with `z` the value's distance
    /// from the context's mean in deviations of the context, the bin of
    /// `asinh(z / 2)`, sixteen to a unit and the mean in the middle, kept
    /// within the bins.
    ///
    /// # Panics
    ///
    /// If the codec has no bins.
    #[must_use]
    pub fn bin(&self, codec: &Codec) -> u32 {
        assert!(codec.bins > 0, "a codec without bins");
        let above = (self.anchor + self.level as f64 / codec.per_unit) * self.units.scale;
        let z = above / self.units.spread;
        let place = (codec.bins / 2) as f64 + BIN_RATE * (z / BIN_KNEE).asinh();
        place.floor().clamp(0.0, (codec.bins - 1) as f64) as u32
    }
}

/// The domain that holds the share `quantile` of the absolute steps, in
/// units, of windows of a corpus: `corpus_domain` of
/// `src/diff_gpt/application/data_loader.py`. The steps are sorted in place.
///
/// # Panics
///
/// If there is no step.
pub fn domain_of(steps: &mut [f64], quantile: f64) -> f64 {
    assert!(!steps.is_empty(), "no step to take a domain from");
    steps.sort_by(f64::total_cmp);
    let place = (quantile.clamp(0.0, 1.0) * (steps.len() - 1) as f64).round() as usize;
    steps[place]
}

/// The absolute steps of a window in the units of its first `context`
/// values, added to `steps`: what [`domain_of`] reads.
pub fn steps_of(window: &[f64], context: usize, steps: &mut Vec<f64>) {
    let units = Units::of(&window[..context]);
    let pairs = window.windows(2);
    let seen = pairs.filter(|pair| pair[0].is_finite() && pair[1].is_finite());
    steps.extend(seen.map(|pair| (pair[1] - pair[0]).abs() / units.scale));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::Rng;

    #[test]
    fn units_are_the_mean_the_mean_step_and_the_deviation() {
        let units = Units::of(&[1.0, 3.0, 2.0, 6.0]);
        assert_eq!(units.location, 3.0);
        assert!((units.scale - 7.0 / 3.0).abs() < 1e-12);
        assert!((units.spread - 3.5_f64.sqrt()).abs() < 1e-12);
        // A missing value counts in no mean, and takes its two steps along.
        let gap = Units::of(&[1.0, f64::NAN, 2.0, 6.0]);
        assert_eq!((gap.location, gap.scale), (3.0, 4.0));
        // A constant has no step: its size is its unit, and 1 if it is 0.
        let flat = Units::of(&[-5.0, -5.0, -5.0]);
        assert_eq!((flat.location, flat.scale, flat.spread), (-5.0, 5.0, 5.0));
        let zero = Units::of(&[0.0, 0.0]);
        assert_eq!((zero.location, zero.scale, zero.spread), (0.0, 1.0, 1.0));
        let none = Units::of(&[f64::NAN]);
        assert_eq!((none.location, none.scale, none.spread), (0.0, 1.0, 1.0));
        let one = Units::of(&[7.0]);
        assert_eq!((one.location, one.scale, one.spread), (7.0, 7.0, 7.0));
    }

    /// A random walk with a slow wave, of steps about one unit.
    fn walk(count: usize, seed: u64) -> Vec<f64> {
        let mut rng = Rng::from_key(&[seed]);
        let mut level = 40.0;
        (0..count)
            .map(|step| {
                level += rng.normal() * 0.3;
                level + 2.0 * (step as f64 * 0.21).sin()
            })
            .collect()
    }

    #[test]
    fn tokens_decode_to_within_half_a_level() {
        let codec = Codec::new(256, 12.0, 0);
        assert!((codec.domain() - 12.0).abs() < 1e-12);
        let series = walk(400, 1);
        let encoded = codec.encode(&series, 100);
        assert_eq!((encoded.tokens.len(), encoded.bins.len()), (399, 0));
        let units = Units::of(&series[..100]);
        let half = 0.5 / codec.per_unit * units.scale;
        let mut track = Track::start(units, &series);
        assert_eq!(track.value(&codec), series[0]);
        for (token, value) in encoded.tokens.iter().zip(&series[1..]) {
            let decoded = track.decode(&codec, *token);
            assert!(
                (decoded - value).abs() <= half * (1.0 + 1e-9),
                "{decoded} for {value}"
            );
        }
        assert_eq!(track, encoded.track);
        // The steps of the series are around the middle of the vocabulary.
        let zero = codec.zero();
        assert_eq!(zero, 128);
        let mean = encoded.tokens.iter().map(|t| f64::from(*t)).sum::<f64>() / 399.0;
        assert!((mean - 128.0).abs() < 2.0, "mean token {mean}");
        assert!(encoded.tokens.iter().any(|token| *token != zero));
    }

    #[test]
    fn tokens_do_not_depend_on_the_units_of_the_series() {
        let codec = Codec::new(256, 12.0, 64);
        let series = walk(300, 2);
        let moved: Vec<f64> = series.iter().map(|value| 3.7e4 * value - 1.2e6).collect();
        let (plain, scaled) = (codec.encode(&series, 80), codec.encode(&moved, 80));
        assert_eq!(plain.tokens, scaled.tokens);
        assert_eq!(plain.bins, scaled.bins);
    }

    #[test]
    fn a_jump_beyond_the_vocabulary_is_caught_up() {
        let codec = Codec::new(64, 4.0, 0);
        // Steps of a unit, then a jump of 30 units, 7.5 domains.
        let mut series: Vec<f64> = (0..40).map(|step| f64::from(step % 2)).collect();
        series.extend((0..40).map(|step| 30.0 + f64::from(step % 2)));
        let encoded = codec.encode(&series, 40);
        let units = Units::of(&series[..40]);
        assert_eq!(units.scale, 1.0);
        let half = 0.5 / codec.per_unit;
        let mut track = Track::start(units, &series);
        let mut late = 0;
        for (index, (token, value)) in encoded.tokens.iter().zip(&series[1..]).enumerate() {
            let decoded = track.decode(&codec, *token);
            if (decoded - value).abs() > half * (1.0 + 1e-9) {
                late += 1;
                // Only right after the jump, and at the top token meanwhile.
                assert!((39..47).contains(&index), "off at {index}");
                assert_eq!(*token, 63);
            }
        }
        // 30 units are 232.5 levels, and the top token is 31: eight tokens.
        assert_eq!(late, 7);
        // A value far outside any series does not overflow the level.
        let mut far = Track::start(units, &series);
        assert_eq!(far.encode(&codec, 1.0e300), 63);
        assert_eq!(far.encode(&codec, -1.0e300), 0);
        assert_eq!(far.encode(&codec, f64::NAN), 32);
    }

    #[test]
    fn bins_follow_the_level_in_deviations_of_the_context() {
        // A fine grid: 254 levels a unit.
        let codec = Codec::new(256, 0.5, 128);
        // A context of mean 10 and deviation 2, in steps of 4.
        let context = [8.0, 12.0, 8.0, 12.0];
        let units = Units::of(&context);
        assert_eq!(
            (units.location, units.scale, units.spread),
            (10.0, 4.0, 2.0)
        );
        let bin_at = |value: f64, arrives: bool| {
            let mut track = Track::start(units, &context);
            // The top token is a step of half a unit: the level needs time.
            for _ in 0..400 {
                track.encode(&codec, value);
            }
            assert!(
                !arrives || (track.value(&codec) - value).abs() < 0.01,
                "at {value}"
            );
            track.bin(&codec)
        };
        // The mean is the first bin of the upper half; a deviation above it
        // is asinh(1/2) · 16 = 7.7 bins higher.
        assert_eq!((bin_at(10.1, true), bin_at(9.9, true)), (64, 63));
        assert_eq!((bin_at(12.05, true), bin_at(7.95, true)), (64 + 7, 63 - 7));
        // The bins widen: 10.5 deviations are asinh(5.25) · 16 = 37.8 bins.
        assert_eq!(bin_at(31.0, true), 64 + 37);
        // And end, 103 deviations from the mean.
        assert_eq!((bin_at(1.0e6, false), bin_at(-1.0e6, false)), (127, 0));
        let encoded = codec.encode(&[8.0, 12.0, 8.0, 12.0, 10.0], 4);
        assert_eq!(encoded.bins.len(), 4);
    }

    #[test]
    fn the_domain_holds_a_share_of_the_steps() {
        let mut steps = Vec::new();
        // Steps of 0, 1, …, 100 in a context of mean step 2.
        let window: Vec<f64> = (0..=101).map(|n| f64::from(n * (n - 1)) / 2.0).collect();
        steps_of(&window, 5, &mut steps);
        assert_eq!(steps.len(), 101);
        assert_eq!(Units::of(&window[..5]).scale, 1.5);
        assert!((domain_of(&mut steps, 0.9) - 90.0 / 1.5).abs() < 1e-9);
        assert!((domain_of(&mut steps, 1.0) - 100.0 / 1.5).abs() < 1e-9);
    }
}
