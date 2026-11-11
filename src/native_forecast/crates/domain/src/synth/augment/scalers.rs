//! The scalers of TempoPFN's `src/data/scalers.py`, on one series.

use super::{extent, mean, median, quantile, sorted};

/// A way to bring a series to a standard level and spread.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scaler {
    /// `RobustScaler`: less the median, over the interquartile range, cut at
    /// ±50.
    Robust,
    /// `MinMaxScaler`: onto ±1.
    MinMax,
    /// `MedianScaler`: less the median.
    Median,
    /// `MeanScaler`: less the mean.
    Mean,
}

impl Scaler {
    /// The four scalers, in the order both augmentations list them.
    pub const ALL: [Self; 4] = [Self::Robust, Self::MinMax, Self::Median, Self::Mean];

    /// Scales the series in place with its own statistics.
    pub fn scale(self, values: &mut [f32]) {
        let finite: Vec<f32> = values
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .collect();
        match self {
            Self::Robust => {
                // Without a finite value the statistics stay at 0 and 1.
                let (center, spread) = if finite.is_empty() {
                    (0.0, 1.0)
                } else {
                    let sorted = sorted(&finite);
                    let spread = if sorted.len() > 1 {
                        (quantile(&sorted, 0.75) - quantile(&sorted, 0.25)).max(1.0e-3)
                    } else {
                        1.0e-3
                    };
                    (median(&sorted), spread)
                };
                let divisor = (spread + 1.0e-6).max(1.0e-3);
                for value in values {
                    *value = ((*value - center) / divisor).clamp(-50.0, 50.0);
                }
            }
            Self::MinMax => {
                let (low, mut high) = extent(values);
                if (high - low).abs() < 1.0e-8 {
                    high = low + 1.0;
                }
                for value in values {
                    *value = (*value - low) / (high - low + 1.0e-8) * 2.0 - 1.0;
                }
            }
            Self::Median => {
                let center = if finite.is_empty() {
                    0.0
                } else {
                    median(&sorted(&finite))
                };
                for value in values {
                    *value -= center;
                }
            }
            Self::Mean => {
                let center = if finite.is_empty() {
                    0.0
                } else {
                    mean(&finite)
                };
                for value in values {
                    *value -= center;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_scaler_centres_as_it_says() {
        let series = [2.0_f32, 4.0, 4.0, 10.0, 20.0];
        let scaled = |scaler: Scaler| {
            let mut values = series;
            scaler.scale(&mut values);
            values
        };
        assert_eq!(scaled(Scaler::Mean), [-6.0, -4.0, -4.0, 2.0, 12.0]);
        assert_eq!(scaled(Scaler::Median), [-2.0, 0.0, 0.0, 6.0, 16.0]);
        let robust = scaled(Scaler::Robust);
        // Quartiles 4 and 10: the interquartile range is 6.
        assert!((robust[4] - 16.0 / 6.000_001).abs() < 1e-5 && robust[1] == 0.0);
        let unit = scaled(Scaler::MinMax);
        assert!((unit[0] + 1.0).abs() < 1e-6 && (unit[4] - 1.0).abs() < 1e-6);
        assert!((unit[3] - (8.0 / 18.0 * 2.0 - 1.0)).abs() < 1e-6);
    }

    #[test]
    fn flat_and_wild_series_stay_finite() {
        let mut flat = [3.0_f32; 8];
        Scaler::MinMax.scale(&mut flat);
        assert_eq!(flat, [-1.0; 8]);
        let mut flat = [3.0_f32; 8];
        Scaler::Robust.scale(&mut flat);
        assert_eq!(flat, [0.0; 8]);
        let mut wild = [0.0_f32, 0.0, 0.0, 0.0, 1.0e9];
        Scaler::Robust.scale(&mut wild);
        assert_eq!(wild[4], 50.0);
    }
}
