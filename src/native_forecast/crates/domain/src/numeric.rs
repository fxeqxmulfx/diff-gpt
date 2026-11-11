//! Small numerical tools shared by the sources: NumPy's `linspace` and
//! rounding, and a radix-2 Fourier transform.

use core::f64::consts::TAU;

/// `count` points from `start` to `stop`, both included, computed as NumPy's
/// `linspace` computes them: `start + k·step`, the last one set to `stop`.
#[must_use]
pub fn linspace(start: f64, stop: f64, count: usize) -> Vec<f64> {
    match count {
        0 => Vec::new(),
        1 => vec![start],
        _ => {
            let step = (stop - start) / (count - 1) as f64;
            let mut points: Vec<f64> = (0..count).map(|k| k as f64 * step + start).collect();
            points[count - 1] = stop;
            points
        }
    }
}

/// Rounds half to even, as Python's `round` and NumPy's `round` do.
#[must_use]
pub fn round_half_even(value: f64) -> f64 {
    value.round_ties_even()
}

/// A complex number as its real and imaginary parts.
pub type Complex = (f64, f64);

/// The discrete Fourier transform of `values` in place, or its inverse
/// (divided by the length), by the radix-2 algorithm.
///
/// # Panics
///
/// If the length is not a power of two.
pub fn fft(values: &mut [Complex], inverse: bool) {
    let count = values.len();
    if count < 2 {
        return;
    }
    assert!(
        count.is_power_of_two(),
        "radix-2 transform of {count} points"
    );
    let shift = usize::BITS - count.trailing_zeros();
    for index in 0..count {
        let mirror = index.reverse_bits() >> shift;
        if index < mirror {
            values.swap(index, mirror);
        }
    }
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut half = 1;
    while half < count {
        let step = sign * TAU / (2 * half) as f64;
        for start in (0..count).step_by(2 * half) {
            for offset in 0..half {
                let (sin, cos) = (step * offset as f64).sin_cos();
                let (re, im) = values[start + offset + half];
                let odd = (re * cos - im * sin, re * sin + im * cos);
                let even = values[start + offset];
                values[start + offset] = (even.0 + odd.0, even.1 + odd.1);
                values[start + offset + half] = (even.0 - odd.0, even.1 - odd.1);
            }
        }
        half *= 2;
    }
    if inverse {
        let scale = 1.0 / count as f64;
        for value in values.iter_mut() {
            *value = (value.0 * scale, value.1 * scale);
        }
    }
}

/// The frequency of bin `index` of a transform of `count` points, in cycles
/// per sample, as NumPy's `fftfreq` gives it.
#[must_use]
pub fn fft_frequency(index: usize, count: usize) -> f64 {
    let signed = if index < count.div_ceil(2) {
        index as f64
    } else {
        index as f64 - count as f64
    };
    signed / count as f64
}

/// The mean of `values`; 0 for none.
#[must_use]
pub fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

/// The population standard deviation of `values`, as NumPy's `std`.
#[must_use]
pub fn deviation(values: &[f64]) -> f64 {
    let center = mean(values);
    mean(
        &values
            .iter()
            .map(|v| (v - center).powi(2))
            .collect::<Vec<_>>(),
    )
    .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linspace_hits_both_ends() {
        assert_eq!(linspace(2.0, 18.0, 9)[4], 10.0);
        assert_eq!(linspace(1.0, 6.0, 3), [1.0, 3.5, 6.0]);
        assert_eq!(linspace(4.0, 9.0, 1), [4.0]);
        assert!(linspace(0.0, 1.0, 0).is_empty());
    }

    #[test]
    fn rounding_takes_ties_to_even() {
        assert_eq!(round_half_even(0.5), 0.0);
        assert_eq!(round_half_even(1.5), 2.0);
        assert_eq!(round_half_even(2.5), 2.0);
        assert_eq!(round_half_even(-0.5), 0.0);
        assert_eq!(round_half_even(2.4), 2.0);
    }

    #[test]
    fn transform_finds_a_tone_and_inverts() {
        let count = 64;
        let signal: Vec<Complex> = (0..count)
            .map(|t| ((TAU * 5.0 * t as f64 / count as f64).cos(), 0.0))
            .collect();
        let mut spectrum = signal.clone();
        fft(&mut spectrum, false);
        for (bin, &(re, im)) in spectrum.iter().enumerate() {
            let expected = if bin == 5 || bin == count - 5 {
                count as f64 / 2.0
            } else {
                0.0
            };
            assert!((re - expected).abs() < 1e-9 && im.abs() < 1e-9, "bin {bin}");
        }
        fft(&mut spectrum, true);
        for (back, original) in spectrum.iter().zip(&signal) {
            assert!((back.0 - original.0).abs() < 1e-12 && back.1.abs() < 1e-12);
        }
    }

    #[test]
    fn frequencies_follow_numpy() {
        assert_eq!(fft_frequency(0, 8), 0.0);
        assert_eq!(fft_frequency(3, 8), 0.375);
        assert_eq!(fft_frequency(4, 8), -0.5);
        assert_eq!(fft_frequency(7, 8), -0.125);
    }

    #[test]
    fn moments_of_a_small_sample() {
        assert_eq!(mean(&[1.0, 2.0, 3.0]), 2.0);
        assert!((deviation(&[1.0, 2.0, 3.0]) - (2.0_f64 / 3.0).sqrt()).abs() < 1e-15);
    }
}
