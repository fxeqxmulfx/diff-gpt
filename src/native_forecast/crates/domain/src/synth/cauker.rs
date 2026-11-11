//! `cauker_univariate`: CauKer (Xie et al., 2025): the nodes of a random
//! structural causal model. The roots of a random directed acyclic graph are
//! draws from Gaussian processes of composite kernels with random means, and
//! every other node is a random activation of a random affine map of its
//! parents. Each node is one series.
//!
//! Port of diff-gpt's stand-in for TempoPFN's `cauker/cauker_generator.py`,
//! which made the reference corpus: the roots of a model have independent
//! noises, as in the original CauKer. TempoPFN's driver takes the 6 nodes of
//! a model of 6 nodes and at most 3 parents as 6 univariate series.

use crate::random::Rng;
use crate::synth::kernels::{bank, grid, random_composite};

/// The parameters of the source.
#[derive(Clone, Copy, Debug)]
pub struct CauKer {
    /// The number of nodes of a model.
    pub nodes: usize,
    /// The greatest number of parents of a node.
    pub max_parents: usize,
    /// The greatest number of kernels in the composite of a root.
    pub max_kernels: i64,
}

impl CauKer {
    /// The number of nodes of a model of the corpus.
    pub const NODES: usize = 6;

    /// The models of the corpus, as TempoPFN's driver sets them.
    #[must_use]
    pub const fn corpus() -> Self {
        Self {
            nodes: Self::NODES,
            max_parents: 3,
            max_kernels: 7,
        }
    }
}

/// One of the four means of the library, on the grid `x`.
fn mean(rng: &mut Rng, x: &[f64]) -> Vec<f64> {
    match rng.index(4) {
        0 => vec![0.0; x.len()],
        1 => {
            let (a, b) = (rng.uniform_in(-1.0, 1.0), rng.uniform_in(-1.0, 1.0));
            x.iter().map(|&point| a * point + b).collect()
        }
        2 => {
            let (a, b) = (rng.uniform_in(0.5, 1.5), rng.uniform_in(0.5, 1.5));
            x.iter().map(|&point| a * (b * point).exp()).collect()
        }
        _ => {
            let mut spikes = vec![0.0; x.len()];
            for _ in 0..rng.integer(1, 6) {
                spikes[rng.index(x.len())] += rng.uniform_in(-5.0, 5.0);
            }
            spikes
        }
    }
}

/// An activation of the library, with its parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Activation {
    Linear {
        slope: f64,
        offset: f64,
    },
    Relu,
    Sigmoid,
    Sin,
    /// NumPy's `mod`: the remainder has the sign of the modulus.
    Modulo {
        modulus: f64,
    },
    LeakyRelu {
        slope: f64,
    },
}

impl Activation {
    /// One of the six activations, uniformly, with random parameters.
    fn sample(rng: &mut Rng) -> Self {
        match rng.index(6) {
            0 => Self::Linear {
                slope: rng.uniform_in(0.5, 2.0),
                offset: rng.uniform_in(-1.0, 1.0),
            },
            1 => Self::Relu,
            2 => Self::Sigmoid,
            3 => Self::Sin,
            4 => Self::Modulo {
                modulus: rng.uniform_in(1.0, 5.0),
            },
            _ => Self::LeakyRelu {
                slope: rng.uniform_in(0.01, 0.3),
            },
        }
    }

    fn apply(self, value: f64) -> f64 {
        match self {
            Self::Linear { slope, offset } => slope * value + offset,
            Self::Relu => value.max(0.0),
            Self::Sigmoid => 1.0 / (1.0 + (-value).exp()),
            Self::Sin => value.sin(),
            Self::Modulo { modulus } => value.rem_euclid(modulus),
            Self::LeakyRelu { slope } if value <= 0.0 => slope * value,
            Self::LeakyRelu { .. } => value,
        }
    }
}

impl CauKer {
    /// The series of the nodes of one random model, in random order.
    pub fn model(&self, rng: &mut Rng, length: usize) -> Vec<Vec<f64>> {
        let x = grid(length);
        let kernels = bank(length);
        // Node i takes its parents among the nodes before it, so the nodes
        // are already in a topological order.
        let mut nodes: Vec<Vec<f64>> = Vec::with_capacity(self.nodes);
        for node in 0..self.nodes {
            let parents = rng.integer(0, node.min(self.max_parents) as i64 + 1) as usize;
            let series = if parents == 0 {
                let count = rng.integer(1, self.max_kernels + 1) as usize;
                let kernel = random_composite(rng, &kernels, &x, count);
                let (first, second) = (mean(rng, &x), mean(rng, &x));
                let product = rng.chance(0.5);
                let noise = kernel.sample(&x, rng);
                (0..length)
                    .map(|step| {
                        let mean = if product {
                            first[step] * second[step]
                        } else {
                            first[step] + second[step]
                        };
                        mean + noise[step]
                    })
                    .collect()
            } else {
                let mut input = vec![0.0; length];
                for parent in rng.distinct(node, parents) {
                    let weight = rng.normal();
                    for (sum, value) in input.iter_mut().zip(&nodes[parent]) {
                        *sum += weight * value;
                    }
                }
                let bias = rng.normal();
                let activation = Activation::sample(rng);
                input
                    .into_iter()
                    .map(|value| activation.apply(value + bias))
                    .collect()
            };
            nodes.push(series);
        }
        rng.shuffle(&mut nodes);
        nodes
    }

    /// `count` series: the nodes of as many models as it takes, the last
    /// model cut short, as TempoPFN's driver cuts its chunk.
    pub fn chunk(&self, rng: &mut Rng, length: usize, count: usize) -> Vec<Vec<f64>> {
        let mut series = Vec::with_capacity(count + self.nodes);
        let mut model = 0;
        while series.len() < count {
            series.extend(self.model(&mut rng.fork(model), length));
            model += 1;
        }
        series.truncate(count);
        series
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_model_gives_a_series_for_each_node() {
        let source = CauKer::corpus();
        for seed in 0..40 {
            let nodes = source.model(&mut Rng::from_key(&[seed]), 128);
            assert_eq!(nodes.len(), 6);
            assert!(nodes.iter().all(|series| series.len() == 128));
            assert!(nodes.iter().flatten().all(|value| !value.is_nan()));
        }
        let chunk = source.chunk(&mut Rng::from_key(&[99]), 64, 64);
        assert_eq!(chunk.len(), 64);
    }

    #[test]
    fn activations_follow_numpy() {
        let modulo = Activation::Modulo { modulus: 2.5 };
        assert!((modulo.apply(-7.3) - 0.2).abs() < 1e-12);
        assert!((modulo.apply(9.1) - 1.6).abs() < 1e-12);
        let leaky = Activation::LeakyRelu { slope: 0.1 };
        assert_eq!((leaky.apply(-2.0), leaky.apply(3.0)), (-0.2, 3.0));
        assert_eq!(
            (Activation::Relu.apply(-2.0), Activation::Relu.apply(3.0)),
            (0.0, 3.0)
        );
        assert_eq!(Activation::Sigmoid.apply(0.0), 0.5);
        let linear = Activation::Linear {
            slope: 2.0,
            offset: -1.0,
        };
        assert_eq!(linear.apply(3.0), 5.0);
        let mut rng = Rng::from_key(&[5]);
        let mut kinds = [0; 6];
        for _ in 0..600 {
            kinds[match Activation::sample(&mut rng) {
                Activation::Linear { .. } => 0,
                Activation::Relu => 1,
                Activation::Sigmoid => 2,
                Activation::Sin => 3,
                Activation::Modulo { .. } => 4,
                Activation::LeakyRelu { .. } => 5,
            }] += 1;
        }
        assert!(kinds.iter().all(|&count| count > 60), "{kinds:?}");
    }

    #[test]
    fn means_are_of_the_four_kinds() {
        let x = grid(100);
        let mut rng = Rng::from_key(&[6]);
        let (mut zero, mut sparse, mut dense) = (0, 0, 0);
        for _ in 0..400 {
            let values = mean(&mut rng, &x);
            let nonzero = values.iter().filter(|&&value| value != 0.0).count();
            match nonzero {
                0 => zero += 1,
                1..=5 => sparse += 1,
                _ => dense += 1,
            }
        }
        assert!(
            zero > 60 && sparse > 60 && dense > 150,
            "{zero} {sparse} {dense}"
        );
    }
}
