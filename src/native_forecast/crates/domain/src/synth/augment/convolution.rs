//! Convolutions of a series with small kernels, as PyTorch's `Conv1d` with
//! `padding="same"` computes them, and the two augmenters of TempoPFN's
//! `src/data/augmentations.py` that are built on them:
//! `DifferentialAugmenter` ([`differential`]) and `RandomConvAugmenter`
//! ([`random_convolution`]).

use super::{extent, mean};
use crate::random::Rng;

/// How a series continues beyond its ends: the `padding_mode` of `Conv1d`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Padding {
    /// Mirrored about its first and its last value, neither repeated.
    Reflect,
    /// Its first and its last value, repeated.
    Replicate,
    /// Itself again, as on a circle.
    Circular,
}

impl Padding {
    /// The three paddings, in the order `RandomConvAugmenter` lists them.
    pub const ALL: [Self; 3] = [Self::Reflect, Self::Replicate, Self::Circular];

    /// The place, in a series of `length` values, of what stands at
    /// `position`, which may be before the series or after it.
    fn place(self, position: i64, length: usize) -> usize {
        let last = length as i64 - 1;
        let place = match self {
            Self::Replicate => position.clamp(0, last),
            Self::Circular => position.rem_euclid(length as i64),
            Self::Reflect if last == 0 => 0,
            Self::Reflect => {
                let folded = position.rem_euclid(2 * last);
                if folded > last {
                    2 * last - folded
                } else {
                    folded
                }
            }
        };
        place as usize
    }
}

/// The cross-correlation of a series with `kernel`, its taps `dilation`
/// steps apart, plus `bias`: a `Conv1d` of one channel that keeps the length
/// (`padding="same"`).
///
/// # Panics
///
/// If the series is empty.
#[must_use]
pub fn correlate(
    values: &[f32],
    kernel: &[f32],
    dilation: usize,
    padding: Padding,
    bias: f32,
) -> Vec<f32> {
    assert!(!values.is_empty(), "a series has at least one value");
    let reach = dilation * kernel.len().saturating_sub(1);
    let before = (reach / 2) as i64;
    let padded: Vec<f32> = (0..(values.len() + reach) as i64)
        .map(|position| values[padding.place(position - before, values.len())])
        .collect();
    (0..values.len())
        .map(|step| {
            let taps = padded[step..].iter().step_by(dilation.max(1));
            bias + kernel
                .iter()
                .zip(taps)
                .map(|(weight, value)| weight * value)
                .sum::<f32>()
        })
        .collect()
}

/// The bell `exp(−(x/σ)²/2)` at the `size` integers centred on 0.
fn bell(size: usize, sigma: f32) -> Vec<f32> {
    let half = (size / 2) as i64;
    (-half..=half)
        .map(|offset| (-0.5 * (offset as f32 / sigma).powi(2)).exp())
        .collect()
}

/// `processed`, moved and stretched onto the range of `original`.
fn onto_range(processed: &[f32], original: &[f32]) -> Vec<f32> {
    let (low, high) = extent(original);
    let (processed_low, processed_high) = extent(processed);
    let width = processed_high - processed_low + 1.0e-8;
    processed
        .iter()
        .map(|value| (value - processed_low) / width * (high - low) + low)
        .collect()
}

/// The running sum, kept in double precision as PyTorch's `cumsum` keeps it.
fn running_sum<'a>(values: impl Iterator<Item = &'a f32>) -> Vec<f32> {
    let mut total = 0.0_f64;
    values
        .map(|value| {
            total += f64::from(*value);
            total as f32
        })
        .collect()
}

/// The probability with which [`differential`] changes a series.
const DIFFERENTIAL: f64 = 0.4;

/// `DifferentialAugmenter`: with probability 0.4, the series is replaced by
/// one of six of its relatives, on its own range: the series smoothed by a
/// Gaussian window, the first to the fourth difference of the smoothed
/// series, or the running sum of the series from the left or from the right.
/// A flat series stays as it is.
pub fn differential(rng: &mut Rng, values: &mut Vec<f32>) {
    if !rng.chance(DIFFERENTIAL) {
        return;
    }
    // A size among 5, …, 51 made odd: 5 is half as likely as the others.
    let size = (rng.integer(5, 52) / 2 * 2 + 1) as usize;
    let sigma = rng.uniform_in(2.0, 20.0) as f32;
    let relative = rng.index(6);
    let from_right = rng.chance(0.5);

    let (low, high) = extent(values);
    if high - low < 1.0e-8 {
        return;
    }
    let smooth = |values: &[f32]| {
        let mut window = bell(size, sigma);
        let total: f32 = window.iter().sum();
        window.iter_mut().for_each(|weight| *weight /= total);
        correlate(values, &window, 1, Padding::Reflect, 0.0)
    };
    let difference =
        |values: &[f32], stencil: &[f32]| correlate(values, stencil, 1, Padding::Reflect, 0.0);
    let processed = match relative {
        0 => smooth(values),
        1 => difference(&smooth(values), &[-1.0, 0.0, 1.0]),
        2 => difference(&smooth(values), &[1.0, -2.0, 1.0]),
        3 if from_right => {
            let mut sums = running_sum(values.iter().rev());
            sums.reverse();
            sums
        }
        3 => running_sum(values.iter()),
        4 => difference(&smooth(values), &[-1.0, 2.0, 0.0, -2.0, 1.0]),
        _ => difference(&smooth(values), &[1.0, -4.0, 6.0, -4.0, 1.0]),
    };
    *values = onto_range(&processed, values);
}

/// A kernel of `RandomConvAugmenter`, of `size` taps: a Gaussian window,
/// white noise, a random parabola or a noisy difference `[−1, 0, 1]` padded
/// with zeros; four times out of five its magnitudes sum to 1.
fn random_kernel(rng: &mut Rng, size: usize) -> Vec<f32> {
    let mut kernel: Vec<f32> = match rng.index(4) {
        0 => bell(size, rng.uniform_in(0.5, 5.0) as f32),
        1 => (0..size).map(|_| rng.normal() as f32).collect(),
        2 => {
            let [square, slope, level] = [(); 3].map(|()| rng.normal() as f32);
            (0..size)
                .map(|tap| {
                    let x = -1.0 + 2.0 * tap as f32 / (size.max(2) - 1) as f32;
                    square * x * x + slope * x + level
                })
                .collect()
        }
        _ => {
            let mut kernel = vec![0.0_f32; size.max(3)];
            let first = (kernel.len() - 3) / 2;
            for (tap, base) in [-1.0_f32, 0.0, 1.0].into_iter().enumerate() {
                kernel[first + tap] = base + 0.1 * rng.normal() as f32;
            }
            kernel
        }
    };
    if rng.chance(0.8) {
        let total = kernel.iter().map(|weight| weight.abs()).sum::<f32>() + 1.0e-8;
        kernel.iter_mut().for_each(|weight| *weight /= total);
    }
    kernel
}

/// `RandomConvAugmenter` on one series: one to three convolutions in a row,
/// each with a random kernel of 3 to 31 taps, 1 to 8 steps apart, a bias and
/// a padding, with nothing, a rectifier or a hyperbolic tangent between two
/// of them; the result takes the range of the series, or its mean if the
/// result is flat.
#[must_use]
pub fn random_convolution(rng: &mut Rng, values: &[f32]) -> Vec<f32> {
    let layers = rng.integer(1, 4);
    let mut processed = values.to_vec();
    for layer in 0..layers {
        let size = (rng.integer(1, 16) * 2 + 1) as usize;
        let dilation = rng.integer(1, 9) as usize;
        let bias = rng.uniform_in(-0.5, 0.5) as f32;
        let padding = *rng.choose(&Padding::ALL);
        let kernel = random_kernel(rng, size);
        processed = correlate(&processed, &kernel, dilation, padding, bias);
        if layer + 1 < layers {
            match rng.index(3) {
                1 => processed
                    .iter_mut()
                    .for_each(|value| *value = value.max(0.0)),
                2 => processed.iter_mut().for_each(|value| *value = value.tanh()),
                _ => {}
            }
        }
    }
    let (low, high) = extent(&processed);
    if high - low < 1.0e-8 {
        vec![mean(values); values.len()]
    } else {
        onto_range(&processed, values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wave(length: usize) -> Vec<f32> {
        (0..length)
            .map(|step| (step as f32 * 0.37).sin() + 0.01 * step as f32)
            .collect()
    }

    #[test]
    fn the_paddings_continue_a_series_as_pytorch_does() {
        let at = |padding: Padding| {
            (-3..8)
                .map(|position| padding.place(position, 5))
                .collect::<Vec<usize>>()
        };
        assert_eq!(at(Padding::Reflect), [3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1]);
        assert_eq!(at(Padding::Replicate), [0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4]);
        assert_eq!(at(Padding::Circular), [2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2]);
        assert_eq!(Padding::Reflect.place(-7, 1), 0);
    }

    #[test]
    fn a_correlation_keeps_the_length_and_reads_the_taps_in_order() {
        let series = [1.0_f32, 2.0, 4.0, 8.0, 16.0];
        // A difference: next less previous, the ends mirrored or repeated.
        let stencil = [-1.0_f32, 0.0, 1.0];
        assert_eq!(
            correlate(&series, &stencil, 1, Padding::Reflect, 0.0),
            [0.0, 3.0, 6.0, 12.0, 0.0]
        );
        assert_eq!(
            correlate(&series, &stencil, 1, Padding::Replicate, 0.5),
            [1.5, 3.5, 6.5, 12.5, 8.5]
        );
        // Two steps apart on a circle: x[i + 2] − x[i − 2].
        assert_eq!(
            correlate(&series, &stencil, 2, Padding::Circular, 0.0),
            [-4.0, -8.0, 15.0, -1.0, -2.0]
        );
        // One tap is a scaling.
        assert_eq!(
            correlate(&series, &[2.0], 3, Padding::Reflect, 1.0),
            [3.0, 5.0, 9.0, 17.0, 33.0]
        );
    }

    #[test]
    fn a_differential_keeps_the_range_and_changes_two_series_in_five() {
        let mut rng = Rng::from_key(&[1]);
        let original = wave(400);
        let (low, high) = extent(&original);
        let trials = 2000;
        let mut changed = 0;
        for _ in 0..trials {
            let mut series = original.clone();
            differential(&mut rng, &mut series);
            assert_eq!(series.len(), original.len());
            if series != original {
                changed += 1;
                let (new_low, new_high) = extent(&series);
                assert!((new_low - low).abs() < 1e-4 && (new_high - high).abs() < 1e-4);
            }
        }
        let share = f64::from(changed) / f64::from(trials);
        assert!((share - 0.4).abs() < 0.04, "changed {share}");
        // A flat series has no range to keep.
        let mut flat = vec![2.5_f32; 64];
        for _ in 0..50 {
            differential(&mut rng, &mut flat);
        }
        assert_eq!(flat, vec![2.5_f32; 64]);
    }

    #[test]
    fn the_running_sums_go_both_ways() {
        let series = [1.0_f32, 2.0, 3.0];
        assert_eq!(running_sum(series.iter()), [1.0, 3.0, 6.0]);
        assert_eq!(running_sum(series.iter().rev()), [3.0, 5.0, 6.0]);
    }

    #[test]
    fn the_kernels_have_their_sizes_and_are_mostly_normalized() {
        let mut rng = Rng::from_key(&[2]);
        let trials = 4000;
        let mut normalized = 0;
        for _ in 0..trials {
            let size = (rng.integer(1, 16) * 2 + 1) as usize;
            let kernel = random_kernel(&mut rng, size);
            assert_eq!(kernel.len(), size);
            assert!(kernel.iter().all(|weight| weight.is_finite()));
            let total: f32 = kernel.iter().map(|weight| weight.abs()).sum();
            if (total - 1.0).abs() < 1e-4 {
                normalized += 1;
            }
        }
        let share = f64::from(normalized) / f64::from(trials);
        assert!((share - 0.8).abs() < 0.03, "normalized {share}");
    }

    #[test]
    fn a_random_convolution_keeps_the_range_or_gives_the_mean() {
        let mut rng = Rng::from_key(&[3]);
        let original = wave(300);
        let (low, high) = extent(&original);
        let mut whole = 0;
        for _ in 0..300 {
            let series = random_convolution(&mut rng, &original);
            assert_eq!(series.len(), original.len());
            assert!(series.iter().all(|value| value.is_finite()));
            let (new_low, new_high) = extent(&series);
            if new_high > new_low {
                // From the least value up; a result of a tiny range, as
                // after a saturated tangent, does not reach the greatest.
                assert!((new_low - low).abs() < 1e-4 && new_high < high + 1e-3);
                whole += usize::from((new_high - high).abs() < 1e-3);
            } else {
                assert!((new_low - mean(&original)).abs() < 1e-6);
            }
        }
        assert!(whole > 250, "{whole} of 300 took the whole range");
        // A flat series comes back flat, at its level.
        let flat = random_convolution(&mut rng, &[1.5_f32; 40]);
        assert!(flat.iter().all(|value| (value - 1.5).abs() < 1e-6));
    }
}
