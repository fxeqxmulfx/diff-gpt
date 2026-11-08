//! Explicit positional axes and inference-only context extension.
//! No target values or learned-parameter readback enter these host choices.

use crate::Result;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Axes {
    #[default]
    Sequence,
    Axial {
        features: usize,
    },
}

impl Axes {
    pub fn validate(self, head_dim: usize) -> Result<()> {
        if head_dim < 4 || !head_dim.is_multiple_of(4) {
            return Err("positional head dimension must be a positive multiple of four".into());
        }
        if matches!(self, Self::Axial { features: 0 }) {
            return Err("axial positions require at least one feature".into());
        }
        Ok(())
    }

    pub fn time_extent(self, tokens: usize) -> usize {
        match self {
            Self::Sequence => tokens,
            Self::Axial { features } => tokens.div_ceil(features),
        }
    }

    pub fn time_bands(self, head_dim: usize) -> usize {
        match self {
            Self::Sequence => head_dim / 2,
            Self::Axial { .. } => head_dim / 4,
        }
    }

    pub fn base_frequencies(self, head_dim: usize) -> Vec<f32> {
        let per_axis = self.time_bands(head_dim);
        (0..head_dim / 2)
            .map(|j| 1.0 / 10000f32.powf((j % per_axis) as f32 / per_axis as f32))
            .collect()
    }

    pub fn base_tables(self, length: usize, head_dim: usize) -> (Vec<f32>, Vec<f32>) {
        if self == Self::Sequence {
            // Preserve the selected model's operation order and all table bits.
            return crate::rotary::base_tables(length, head_dim, false);
        }
        let Self::Axial { features } = self else {
            unreachable!()
        };
        let bands = head_dim / 2;
        let per_axis = bands / 2;
        let mut angles = vec![0.; length * bands];
        let mut rope = vec![0.; 2 * angles.len()];
        for pos in 0..length {
            for band in 0..bands {
                let coordinate = if band < per_axis {
                    pos / features
                } else {
                    pos % features
                };
                let angle =
                    coordinate as f32 / 10000f32.powf((band % per_axis) as f32 / per_axis as f32);
                let i = pos * bands + band;
                angles[i] = angle;
                rope[i] = angle.cos();
                rope[angles.len() + i] = angle.sin();
            }
        }
        (angles, rope)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Extension {
    pub training_extent: usize,
    pub document_extent: usize,
}

impl Extension {
    pub fn new(training_extent: usize, document_extent: usize) -> Result<Self> {
        if training_extent == 0 || document_extent == 0 {
            return Err("YaRN extents must be positive actual time/token extents".into());
        }
        Ok(Self {
            training_extent,
            document_extent,
        })
    }

    pub fn extended(self) -> bool {
        self.document_extent > self.training_extent
    }

    pub fn inverse_scale(self) -> f32 {
        if self.extended() {
            self.training_extent as f32 / self.document_extent as f32
        } else {
            1.
        }
    }

    pub fn logit_gain(self) -> f32 {
        if !self.extended() {
            return 1.;
        }
        let amplitude = 1. + 0.1 * (self.document_extent as f32 / self.training_extent as f32).ln();
        amplitude * amplitude
    }
}

/// Independent scalar reference, used only in CPU checks and validation.
/// `learned` is the actual trained frequency, not the original RoPE base.
pub fn reference_frequency(learned: f64, context: Extension) -> f64 {
    let u = (context.training_extent as f64 / context.document_extent as f64).min(1.);
    let rotations = context.training_extent as f64 * learned / std::f64::consts::TAU;
    let w = ((rotations - 1.) / 31.).clamp(0., 1.);
    learned * (u + w * (1. - u))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axes_keep_sequence_bits_and_split_time_from_feature() -> Result<()> {
        for d in [16, 32, 128] {
            for length in [1, 2, 7, 31, 129] {
                assert_eq!(
                    Axes::Sequence.base_tables(length, d),
                    crate::rotary::base_tables(length, d, false)
                );
                for features in [1, 2, 3, 5] {
                    let axes = Axes::Axial { features };
                    axes.validate(d)?;
                    let (angles, rotations) = axes.base_tables(length, d);
                    let m = d / 4;
                    for p in 0..length {
                        for j in 0..d / 2 {
                            let coord = if j < m { p / features } else { p % features };
                            let reference = coord as f64 / 10000f64.powf((j % m) as f64 / m as f64);
                            let i = p * (d / 2) + j;
                            assert!(
                                (angles[i] as f64 - reference).abs()
                                    <= 2e-6 * reference.abs().max(1.)
                            );
                            let energy = rotations[i] as f64 * rotations[i] as f64
                                + rotations[angles.len() + i] as f64
                                    * rotations[angles.len() + i] as f64;
                            assert!((energy - 1.).abs() < 2e-7);
                            if features == 1 && j >= m {
                                assert_eq!(angles[i].to_bits(), 0);
                            }
                        }
                    }
                }
            }
        }
        assert!(Axes::Axial { features: 0 }.validate(32).is_err());
        assert!(Axes::Sequence.validate(6).is_err());
        Ok(())
    }

    #[test]
    fn yarn_uses_learned_rotation_counts_and_is_exactly_identity_without_extension() -> Result<()> {
        assert!(Extension::new(0, 4).is_err());
        assert!(Extension::new(4, 0).is_err());
        for train in [2, 3, 16, 128, 2048] {
            for doc in [1, train, 2 * train, 4 * train, 16 * train] {
                let context = Extension::new(train, doc)?;
                assert!(context.logit_gain() >= 1.);
                for band in 0..64 {
                    for alpha in [-8f64, -1., 0., 1., 8.] {
                        let learned = 10000f64.powf(-(band as f64) / 64.) * alpha.exp();
                        let result = reference_frequency(learned, context);
                        let lower = learned * (train as f64 / doc as f64).min(1.);
                        assert!(result >= lower * (1. - 2e-16) && result <= learned * (1. + 2e-16));
                        if doc <= train {
                            assert_eq!(result.to_bits(), learned.to_bits());
                            assert_eq!(context.logit_gain().to_bits(), 1f32.to_bits());
                        }
                    }
                }
            }
        }
        let c = Extension::new(128, 512)?;
        let lo = std::f64::consts::TAU / 128.;
        let hi = 32. * lo;
        assert!((reference_frequency(lo, c) / lo - 0.25).abs() < 1e-15);
        assert!((reference_frequency(hi, c) / hi - 1.).abs() < 1e-15);
        let middle = 16.5 * lo;
        assert!((reference_frequency(middle, c) / middle - 0.625).abs() < 1e-15);
        Ok(())
    }

    #[test]
    fn full_score_temperature_is_not_query_key_amplitude_scaling() -> Result<()> {
        let g = Extension::new(16, 128)?.logit_gain() as f64;
        let (re, im, beta, d) = (0.3, 0.8, 0.6, 16.);
        let intended = g * (re + beta * im * im / d);
        let amplitude_scaled = g * re + g * g * beta * im * im / d;
        assert!((intended - amplitude_scaled).abs() > 0.01);
        Ok(())
    }
}
