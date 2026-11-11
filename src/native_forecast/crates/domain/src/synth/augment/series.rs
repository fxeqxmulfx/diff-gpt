//! The edits of one series: `UnivariateOfflineAugmentor.apply_per_series_only`
//! of TempoPFN's `offline_per_sample_iid_augmentations.py`.
//!
//! Two to five of six kinds of edit are drawn without replacement and made in
//! a fixed order, one edit of each kind drawn; then the series may be scaled
//! by a few percent and take a little noise.

use super::convolution::differential;
use super::{deviation, extent, mean, sorted};
use crate::calendar::{Frequency, fields, month_end};
use crate::random::Rng;

/// The weights of the six kinds: invariances, structure, seasonality,
/// artifacts, analytic, discrete.
const KINDS: [f64; 6] = [0.6, 0.6, 0.5, 0.3, 0.4, 0.6];

/// The edits that the two augmentations do not share.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Edits {
    /// Whether the seasonality may be weekend dips, month-end bumps and
    /// impulses on the calendar of the series.
    pub calendar: bool,
    /// Whether the discrete edit may be a quantization.
    pub quantization: bool,
}

impl Edits {
    /// Edits `values`, a series of the given frequency and first timestamp.
    pub fn apply(self, rng: &mut Rng, values: &mut Vec<f32>, frequency: Frequency, start: i64) {
        let count = rng.integer(2, 6) as usize;
        let kinds = rng.distinct_weighted(&KINDS, count);
        let has = |kind: usize| kinds.contains(&kind);
        if has(0) {
            if rng.chance(0.5) {
                values.reverse();
            } else {
                values.iter_mut().for_each(|value| *value = -*value);
            }
        }
        if has(1) {
            if rng.chance(0.5) {
                regime_change(rng, values);
            } else {
                shock_recovery(rng, values);
            }
        }
        if has(2) {
            if self.calendar && rng.chance(0.5) {
                calendar(rng, values, frequency, start);
            } else {
                amplitude(rng, values);
            }
        }
        if has(3) {
            resample(rng, values);
        }
        if has(4) {
            differential(rng, values);
        }
        if has(5) {
            if self.quantization && rng.chance(0.6) {
                quantize(rng, values);
            } else {
                censor(rng, values);
            }
        }
        if rng.chance(0.5) {
            let factor = rng.uniform_in(0.95, 1.05) as f32;
            values.iter_mut().for_each(|value| *value *= factor);
        }
        if rng.chance(0.1) {
            let spread = 0.01 * deviation(values);
            add_noise(rng, values, spread);
        }
    }
}

/// Adds white noise of the given deviation, if it is positive and finite.
pub(crate) fn add_noise(rng: &mut Rng, values: &mut [f32], deviation: f32) {
    if deviation.is_finite() && deviation > 0.0 {
        for value in values {
            *value += rng.normal() as f32 * deviation;
        }
    }
}

/// The deviation of a series, or 1 if it has none.
fn spread(values: &[f32]) -> f32 {
    let spread = deviation(values);
    if spread.is_finite() && spread != 0.0 {
        spread
    } else {
        1.0
    }
}

/// Stretches the values about their mean by `factor`, then adds `shift`.
fn stretch(values: &mut [f32], factor: f32, shift: f32) {
    let center = mean(values);
    for value in values {
        *value = (*value - center) * factor + center + shift;
    }
}

/// One to three change points, each segment stretched about its mean by a
/// factor in `[0.8, 1.25)` and shifted by a normal of 0.15 deviations of the
/// series.
fn regime_change(rng: &mut Rng, values: &mut [f32]) {
    let length = values.len();
    let shortest = (length / 32).max(8);
    let mut changes = rng.integer(1, 4) as usize;
    if length <= (changes + 1) * shortest {
        changes = (length / (2 * shortest)).saturating_sub(1).max(1);
    }
    let Some(positions) = length.checked_sub(2 * shortest).filter(|&count| count > 0) else {
        return;
    };
    let mut bounds: Vec<usize> = rng
        .distinct(positions, changes.min(positions))
        .into_iter()
        .map(|position| position + shortest)
        .collect();
    bounds.extend([0, length]);
    bounds.sort_unstable();
    let spread = f64::from(spread(values));
    let edits: Vec<(f32, f32)> = bounds
        .windows(2)
        .map(|_| {
            let factor = rng.uniform_in(0.8, 1.25) as f32;
            (factor, rng.normal_with(0.0, 0.15 * spread) as f32)
        })
        .collect();
    for (segment, (factor, shift)) in bounds.windows(2).zip(edits) {
        stretch(&mut values[segment[0]..segment[1]], factor, shift);
    }
}

/// A level shift of 0.5 to 2 deviations, of either sign, that decays
/// exponentially from a random time on, with a time constant of 3% to 25% of
/// the length.
fn shock_recovery(rng: &mut Rng, values: &mut [f32]) {
    let length = values.len();
    let (earliest, latest) = ((length / 16).max(1), (length - length / 16).max(2));
    if earliest >= latest {
        return;
    }
    let moment = rng.integer(earliest as i64, latest as i64) as f32;
    let mut size = rng.uniform_in(0.5, 2.0) * f64::from(spread(values));
    if rng.chance(0.5) {
        size = -size;
    }
    let constant = (rng.uniform_in(0.03, 0.25) * length as f64).max(1.0) as f32;
    for (step, value) in values.iter_mut().enumerate() {
        let decay = (-(step as f32 - moment).max(0.0) / constant).exp();
        *value += size as f32 * decay;
    }
}

/// Factors on the calendar of the series, applied to its deviations from its
/// mean: one below 1 on the weekends of a series sampled daily or faster,
/// one above 1 on the last days of the months, and one or two impulses.
fn calendar(rng: &mut Rng, values: &mut [f32], frequency: Frequency, start: i64) {
    let length = values.len();
    let dates = frequency.dates(start, length);
    let mut factors = vec![1.0_f32; length];
    let mut scale = |marked: &dyn Fn(i64) -> bool, low: f64, high: f64| {
        if dates.iter().any(|&date| marked(date)) {
            let factor = rng.uniform_in(low, high) as f32;
            for (slot, &date) in factors.iter_mut().zip(&dates) {
                if marked(date) {
                    *slot *= factor;
                }
            }
        }
    };
    if matches!(
        frequency,
        Frequency::H
            | Frequency::D
            | Frequency::S
            | Frequency::T1
            | Frequency::T5
            | Frequency::T10
            | Frequency::T15
            | Frequency::T30
    ) {
        scale(&|date| fields(date).weekday >= 5, 0.7, 0.95);
    }
    scale(&month_end, 1.05, 1.3);
    let impulses: Vec<usize> = (0..rng.integer(1, 3)).map(|_| rng.index(length)).collect();
    for place in impulses {
        factors[place] *= rng.uniform_in(0.8, 1.4) as f32;
    }
    let center = mean(values);
    for (value, factor) in values.iter_mut().zip(factors) {
        *value = (*value - center) * factor + center;
    }
}

/// A window of a sixteenth to half of the series, stretched about its mean
/// by a factor in `[0.5, 1.8)`.
fn amplitude(rng: &mut Rng, values: &mut [f32]) {
    let length = values.len() as i64;
    let shortest = (length / 16).max(8);
    let longest = (length / 2).max(shortest + 1);
    let window = rng.integer(shortest, longest + 1);
    let first = rng.integer(0, (length - window).max(1));
    let segment = first as usize..((first + window).min(length)) as usize;
    let factor = rng.uniform_in(0.5, 1.8) as f32;
    stretch(&mut values[segment], factor, 0.0);
}

/// How a series is drawn again from every few of its values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Redraw {
    /// Straight lines between the samples (`np.interp`).
    Lines,
    /// Each sample held until the next.
    Hold,
    /// Lines, then a moving average over `max(3, length / 128)` steps that
    /// takes zeros beyond the ends (`np.convolve(…, mode="same")`).
    Smooth,
}

/// The series drawn again from its values at `offset`, `offset + factor`, …;
/// as it is if there are fewer than three of them.
fn redrawn(values: &[f32], factor: usize, offset: usize, redraw: Redraw) -> Vec<f32> {
    let length = values.len();
    let samples = (offset..length).step_by(factor).count();
    if samples < 3 {
        return values.to_vec();
    }
    let (first, last) = (offset, offset + (samples - 1) * factor);
    // The sample at or before a step; beyond the ends, the nearest one.
    let below = |step: usize| first + (step.clamp(first, last) - first) / factor * factor;
    let lines = || -> Vec<f64> {
        (0..length)
            .map(|step| {
                let below = below(step);
                let above = (below + factor).min(last);
                let (left, right) = (f64::from(values[below]), f64::from(values[above]));
                let part = (step.clamp(first, last) - below) as f64 / factor as f64;
                left + (right - left) * part
            })
            .collect()
    };
    match redraw {
        Redraw::Lines => lines().into_iter().map(|value| value as f32).collect(),
        Redraw::Hold => (0..length).map(|step| values[below(step)]).collect(),
        Redraw::Smooth => {
            let drawn = lines();
            let window = (length / 128).max(3);
            let (before, after) = (window / 2, (window - 1) / 2);
            (0..length)
                .map(|step| {
                    let range = step.saturating_sub(before)..=(step + after).min(length - 1);
                    (drawn[range].iter().sum::<f64>() / window as f64) as f32
                })
                .collect()
        }
    }
}

/// The series sampled every 2 to 8 steps and drawn again from the samples:
/// by lines half of the time, by holding a fifth, smoothed the rest.
fn resample(rng: &mut Rng, values: &mut Vec<f32>) {
    let most = (values.len() / 32).clamp(2, 8) as i64;
    let factor = rng.integer(2, most + 1) as usize;
    let offset = rng.index(factor);
    let redraw = [Redraw::Lines, Redraw::Hold, Redraw::Smooth][rng.weighted(&[0.5, 0.2, 0.3])];
    *values = redrawn(values, factor, offset, redraw);
}

/// `CensorAugmenter`: the series is cut from above at one of its values,
/// from below at another, or left as it is, each one time in three.
fn censor(rng: &mut Rng, values: &mut [f32]) {
    let mode = rng.index(3);
    let (one, other) = (rng.uniform() as f32, rng.uniform() as f32);
    let order = sorted(values);
    let last = order.len() - 1;
    let level = |share: f32| order[((share * last as f32) as usize).min(last)];
    let (floor, ceiling) = (level(one.min(other)), level(one.max(other)));
    match mode {
        1 => values
            .iter_mut()
            .for_each(|value| *value = value.min(ceiling)),
        2 => values
            .iter_mut()
            .for_each(|value| *value = value.max(floor)),
        _ => {}
    }
}

/// `QuantizationAugmenter`: with probability 0.4, each value moves to the
/// nearest of 5 to 15 levels: the least and the greatest value, and points
/// drawn uniformly between them (TempoPFN draws them from a scrambled Sobol
/// sequence, whose points are uniform one by one).
fn quantize(rng: &mut Rng, values: &mut [f32]) {
    if !rng.chance(0.4) {
        return;
    }
    let count = rng.integer(5, 16) as usize;
    let (low, high) = extent(values);
    let mut levels = vec![low, high];
    levels.extend((2..count).map(|_| low + (high - low) * rng.uniform() as f32));
    levels.sort_by(f32::total_cmp);
    if high == low {
        return;
    }
    for value in values {
        let mut nearest = levels[0];
        for &level in &levels[1..] {
            if (*value - level).abs() < (*value - nearest).abs() {
                nearest = level;
            }
        }
        *value = nearest;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calendar::{DAY, days_from_civil};

    fn wave(length: usize) -> Vec<f32> {
        (0..length)
            .map(|step| (step as f32 * 0.21).sin() + 0.002 * step as f32)
            .collect()
    }

    #[test]
    fn a_regime_change_stretches_whole_segments() {
        let mut rng = Rng::from_key(&[1]);
        // A rising series: its steps are 0.7 to 1.3.
        let original: Vec<f32> = (0..512)
            .map(|step| step as f32 + 0.3 * (step as f32).sin())
            .collect();
        for _ in 0..50 {
            let mut series = original.clone();
            regime_change(&mut rng, &mut series);
            // Within a segment the edit is affine: the steps keep their
            // ratio to those of the original, the factor of the segment.
            let ratios: Vec<f32> = (0..511)
                .map(|step| {
                    (series[step + 1] - series[step]) / (original[step + 1] - original[step])
                })
                .collect();
            let factors = ratios
                .iter()
                .filter(|ratio| (0.79..1.26).contains(*ratio))
                .count();
            assert!(factors >= 508, "{factors} steps within a segment");
            // A change point disturbs one ratio: a break before and after.
            let breaks = ratios
                .windows(2)
                .filter(|pair| (pair[0] - pair[1]).abs() > 0.02)
                .count();
            assert!((1..=6).contains(&breaks), "{breaks} breaks");
        }
        // A series too short for a change point is left alone.
        let mut short = wave(16);
        regime_change(&mut rng, &mut short);
        assert_eq!(short, wave(16));
    }

    #[test]
    fn a_shock_shifts_the_start_and_decays() {
        let mut rng = Rng::from_key(&[2]);
        let original = vec![0.0_f32; 640];
        for _ in 0..50 {
            let mut series = original.clone();
            shock_recovery(&mut rng, &mut series);
            // A flat series has no deviation: the size is 0.5 to 2.
            let size = series[0];
            assert!((0.5..2.0).contains(&size.abs()), "size {size}");
            let moment = series
                .iter()
                .position(|value| *value != size)
                .expect("decays");
            assert!((40..=600).contains(&moment), "from {moment}");
            assert!(
                series[moment..]
                    .windows(2)
                    .all(|pair| pair[1].abs() <= pair[0].abs())
            );
            assert!(series[639].abs() < size.abs());
        }
    }

    #[test]
    fn the_calendar_marks_weekends_and_month_ends() {
        let mut rng = Rng::from_key(&[3]);
        // Thursday 2024-02-01: daily steps, deviations of 1 from a mean of 0
        // would hide the mean, so the series alternates around 10.
        let start = days_from_civil(2024, 2, 1) * DAY;
        let original: Vec<f32> = (0..60)
            .map(|step| if step % 2 == 0 { 11.0 } else { 9.0 })
            .collect();
        let mut dipped = 0;
        for _ in 0..40 {
            let mut series = original.clone();
            calendar(&mut rng, &mut series, Frequency::D, start);
            let factor =
                |step: usize| f64::from(series[step] - 10.0) / f64::from(original[step] - 10.0);
            // Saturday the 3rd and Sunday the 4th share the weekend factor.
            let weekend = factor(2);
            if (0.7..0.95).contains(&weekend) {
                dipped += 1;
                assert!((factor(9) - weekend).abs() < 1e-5 || (factor(3) - weekend).abs() < 1e-5);
            }
            // The 29th of February ends the month: an ordinary weekday.
            let end = factor(28);
            assert!(
                (1.05 * 0.8..1.3 * 1.4).contains(&end),
                "month end factor {end}"
            );
            let changed = (0..60)
                .filter(|&step| (factor(step) - 1.0).abs() > 1e-6)
                .count();
            // 17 weekend days, 2 month ends, 1 or 2 impulses.
            assert!((19..=21).contains(&changed), "{changed} changed");
        }
        assert!(dipped >= 36, "weekend dips in {dipped} of 40");
        // A weekly series has no weekend dips.
        let mut weekly = original.clone();
        calendar(&mut rng, &mut weekly, Frequency::W, start);
        let changed = (0..60)
            .filter(|&step| weekly[step] != original[step])
            .count();
        assert!(changed <= 16, "{changed} changed");
    }

    #[test]
    fn an_amplitude_window_keeps_the_rest() {
        let mut rng = Rng::from_key(&[4]);
        let original = wave(320);
        for _ in 0..50 {
            let mut series = original.clone();
            amplitude(&mut rng, &mut series);
            let changed: Vec<usize> = (0..320)
                .filter(|&step| series[step] != original[step])
                .collect();
            let (first, last) = (changed[0], changed[changed.len() - 1]);
            assert!(last - first < 161, "a window of {}", last - first + 1);
            assert!(changed.len() + 4 > 20, "{} changed", changed.len());
        }
    }

    #[test]
    fn resampling_keeps_the_samples_or_smooths_them() {
        let mut rng = Rng::from_key(&[5]);
        let original = wave(256);
        let (mut exact, mut held, mut smooth) = (0, 0, 0);
        for _ in 0..300 {
            let mut series = original.clone();
            resample(&mut rng, &mut series);
            let kept = (0..256)
                .filter(|&step| series[step] == original[step])
                .count();
            let flat = series.windows(2).filter(|pair| pair[0] == pair[1]).count();
            if flat > 100 {
                held += 1;
            } else if kept >= 256 / 8 {
                exact += 1;
            } else {
                smooth += 1;
            }
            assert!(series.iter().all(|value| value.is_finite()));
        }
        assert!((120..180).contains(&exact), "lines {exact}");
        assert!((35..90).contains(&held), "held {held}");
        assert!((60..125).contains(&smooth), "smoothed {smooth}");
    }

    #[test]
    fn the_lines_the_holds_and_the_moving_average_are_numpy_s() {
        // The squares, sampled at 1, 4 and 7: 1, 16 and 49.
        let squares: Vec<f32> = (0..10).map(|step| (step * step) as f32).collect();
        let lines = [1.0, 1.0, 6.0, 11.0, 16.0, 27.0, 38.0, 49.0, 49.0, 49.0];
        assert_eq!(redrawn(&squares, 3, 1, Redraw::Lines), lines);
        assert_eq!(
            redrawn(&squares, 3, 1, Redraw::Hold),
            [1.0, 1.0, 1.0, 1.0, 16.0, 16.0, 16.0, 49.0, 49.0, 49.0]
        );
        // A window of three: a step before, the step, a step after.
        let smooth = redrawn(&squares, 3, 1, Redraw::Smooth);
        let expected = [
            2.0 / 3.0,
            8.0 / 3.0,
            6.0,
            11.0,
            18.0,
            27.0,
            38.0,
            136.0 / 3.0,
            49.0,
            98.0 / 3.0,
        ];
        for (value, expected) in smooth.iter().zip(expected) {
            assert!((value - expected).abs() < 1e-5, "{value} and {expected}");
        }
        // A window of 4, of 512 values: two steps before, one after.
        let ramp: Vec<f32> = (0..512).map(|step| step as f32).collect();
        let smooth = redrawn(&ramp, 2, 0, Redraw::Smooth);
        assert!((smooth[100] - 99.5).abs() < 1e-4 && (smooth[0] - 0.25).abs() < 1e-6);
        // Fewer than three samples: the series stays.
        assert_eq!(redrawn(&squares, 5, 4, Redraw::Lines), squares);
    }

    #[test]
    fn censoring_cuts_at_values_of_the_series() {
        let mut rng = Rng::from_key(&[7]);
        let original = wave(200);
        let (mut above, mut below, mut neither) = (0, 0, 0);
        for _ in 0..300 {
            let mut series = original.clone();
            censor(&mut rng, &mut series);
            let (low, high) = extent(&series);
            let (old_low, old_high) = extent(&original);
            assert!(original.contains(&low) && original.contains(&high));
            if series == original {
                neither += 1;
            } else if high < old_high {
                assert_eq!(low, old_low);
                above += 1;
            } else {
                assert!(low > old_low);
                below += 1;
            }
        }
        assert!(
            above > 60 && below > 60 && neither > 70,
            "{above} {below} {neither}"
        );
    }

    #[test]
    fn quantization_leaves_five_to_fifteen_levels() {
        let mut rng = Rng::from_key(&[8]);
        let original = wave(600);
        let mut quantized = 0;
        for _ in 0..300 {
            let mut series = original.clone();
            quantize(&mut rng, &mut series);
            if series == original {
                continue;
            }
            quantized += 1;
            let mut levels = sorted(&series);
            levels.dedup();
            assert!((2..=15).contains(&levels.len()), "{} levels", levels.len());
            assert_eq!(extent(&series), extent(&original));
        }
        assert!((90..150).contains(&quantized), "quantized {quantized}");
        let mut flat = vec![1.0_f32; 30];
        for _ in 0..20 {
            quantize(&mut rng, &mut flat);
        }
        assert_eq!(flat, vec![1.0_f32; 30]);
    }

    #[test]
    fn the_edits_change_most_series_and_keep_them_finite() {
        let mut rng = Rng::from_key(&[9]);
        let original = wave(512);
        let start = days_from_civil(2001, 5, 17) * DAY;
        for edits in [
            Edits {
                calendar: true,
                quantization: true,
            },
            Edits {
                calendar: false,
                quantization: false,
            },
        ] {
            let mut changed = 0;
            for _ in 0..300 {
                let mut series = original.clone();
                edits.apply(&mut rng, &mut series, Frequency::H, start);
                assert_eq!(series.len(), 512);
                assert!(series.iter().all(|value| value.is_finite()));
                changed += usize::from(series != original);
            }
            assert!(changed > 280, "{changed} of 300 changed");
        }
    }
}
