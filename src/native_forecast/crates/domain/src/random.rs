//! Random numbers: one generator, seeded by a key, and the samplers the
//! synthetic sources draw from.
//!
//! The stream is xoshiro256++ (Blackman and Vigna, 2019), its state filled by
//! SplitMix64 from a key of words such as `[version, split, source, index]`, so
//! that every series of the corpus has its own stream whichever thread makes
//! it. The streams are not NumPy's: the port of a source is checked against its
//! Python original in distribution, never draw by draw.

use core::f64::consts::TAU;

const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;

/// One step of SplitMix64's output function.
const fn mix(word: u64) -> u64 {
    let mut z = word;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A stream of random numbers.
#[derive(Clone, Debug)]
pub struct Rng {
    state: [u64; 4],
}

impl Rng {
    /// The stream of `key`: equal keys give equal streams, and keys that differ
    /// in a word or in their length give unrelated ones.
    #[must_use]
    pub fn from_key(key: &[u64]) -> Self {
        let mut hash = mix(GOLDEN ^ key.len() as u64);
        for &word in key {
            hash = mix(hash.wrapping_add(GOLDEN) ^ mix(word.wrapping_add(GOLDEN)));
        }
        let mut state = [0_u64; 4];
        for slot in &mut state {
            hash = hash.wrapping_add(GOLDEN);
            *slot = mix(hash);
        }
        if state == [0; 4] {
            state[0] = GOLDEN;
        }
        Self { state }
    }

    /// A stream of its own for a part of the work, named by `tag`.
    #[must_use]
    pub fn fork(&mut self, tag: u64) -> Self {
        Self::from_key(&[self.next_u64(), self.next_u64(), tag])
    }

    /// The next 64 bits.
    pub const fn next_u64(&mut self) -> u64 {
        let [s0, s1, s2, s3] = self.state;
        let result = s0.wrapping_add(s3).rotate_left(23).wrapping_add(s0);
        let t = s1 << 17;
        let s2 = s2 ^ s0;
        let s3 = s3 ^ s1;
        let s1 = s1 ^ s2;
        let s0 = s0 ^ s3;
        self.state = [s0, s1, s2 ^ t, s3.rotate_left(45)];
        result
    }

    /// Uniform in `[0, 1)`, with 53 random bits.
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1_u64 << 53) as f64)
    }

    /// Uniform in `[low, high)`.
    pub fn uniform_in(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.uniform()
    }

    /// Uniform among `0, …, bound − 1`, without bias (Lemire, 2019).
    ///
    /// # Panics
    ///
    /// If `bound` is 0.
    pub fn below(&mut self, bound: u64) -> u64 {
        assert!(bound > 0, "an empty range has no element to draw");
        let mut product = u128::from(self.next_u64()) * u128::from(bound);
        if (product as u64) < bound {
            let threshold = bound.wrapping_neg() % bound;
            while (product as u64) < threshold {
                product = u128::from(self.next_u64()) * u128::from(bound);
            }
        }
        (product >> 64) as u64
    }

    /// Uniform among the integers of `[low, high)`, as NumPy's `integers`.
    ///
    /// # Panics
    ///
    /// If the range is empty.
    pub fn integer(&mut self, low: i64, high: i64) -> i64 {
        assert!(low < high, "empty range {low}..{high}");
        // The offset may exceed `i64::MAX`; the sum never leaves the range.
        low.wrapping_add(self.below(high.abs_diff(low)) as i64)
    }

    /// An index below `bound`.
    pub fn index(&mut self, bound: usize) -> usize {
        self.below(bound as u64) as usize
    }

    /// True with probability `probability`.
    pub fn chance(&mut self, probability: f64) -> bool {
        self.uniform() < probability
    }

    /// Standard normal, by Marsaglia's polar method; the second value of a
    /// pair is dropped, so that a stream has no hidden state.
    pub fn normal(&mut self) -> f64 {
        loop {
            let u = self.uniform_in(-1.0, 1.0);
            let v = self.uniform_in(-1.0, 1.0);
            let s = u * u + v * v;
            if s < 1.0 && s > 0.0 {
                return u * (-2.0 * s.ln() / s).sqrt();
            }
        }
    }

    /// Normal with the given mean and standard deviation.
    pub fn normal_with(&mut self, mean: f64, deviation: f64) -> f64 {
        mean + deviation * self.normal()
    }

    /// Exponential with mean 1.
    pub fn exponential(&mut self) -> f64 {
        -(1.0 - self.uniform()).ln()
    }

    /// Log-normal: the exponential of a normal with the given mean and deviation.
    pub fn lognormal(&mut self, mean: f64, deviation: f64) -> f64 {
        self.normal_with(mean, deviation).exp()
    }

    /// Weibull with the given shape and scale 1.
    pub fn weibull(&mut self, shape: f64) -> f64 {
        self.exponential().powf(1.0 / shape)
    }

    /// Gamma with the given shape and scale 1 (Marsaglia and Tsang, 2000).
    ///
    /// # Panics
    ///
    /// If `shape` is not positive.
    pub fn gamma(&mut self, shape: f64) -> f64 {
        assert!(shape > 0.0, "gamma shape {shape}");
        if shape < 1.0 {
            let boost = (1.0 - self.uniform()).powf(1.0 / shape);
            return self.gamma(shape + 1.0) * boost;
        }
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x = self.normal();
            let v = 1.0 + c * x;
            if v <= 0.0 {
                continue;
            }
            let v = v * v * v;
            let u = 1.0 - self.uniform();
            if u.ln() < 0.5 * x * x + d - d * v + d * v.ln() {
                return d * v;
            }
        }
    }

    /// Beta with the given shapes.
    pub fn beta(&mut self, a: f64, b: f64) -> f64 {
        let x = self.gamma(a);
        let y = self.gamma(b);
        x / (x + y)
    }

    /// A point of the simplex, Dirichlet with the concentrations `alpha`.
    pub fn dirichlet(&mut self, alpha: &[f64]) -> Vec<f64> {
        let mut point: Vec<f64> = alpha.iter().map(|&a| self.gamma(a)).collect();
        let total: f64 = point.iter().sum();
        if total > 0.0 {
            for value in &mut point {
                *value /= total;
            }
        } else {
            let winner = self.index(point.len());
            point[winner] = 1.0;
        }
        point
    }

    /// Uniform angle in `[0, 2π)`.
    pub fn angle(&mut self) -> f64 {
        self.uniform_in(0.0, TAU)
    }

    /// An index drawn with probability proportional to its weight.
    ///
    /// # Panics
    ///
    /// If no weight is positive.
    pub fn weighted(&mut self, weights: &[f64]) -> usize {
        let total: f64 = weights.iter().sum();
        assert!(total > 0.0, "no positive weight among {weights:?}");
        let mut target = self.uniform() * total;
        let mut last = 0;
        for (index, &weight) in weights.iter().enumerate() {
            if weight <= 0.0 {
                continue;
            }
            last = index;
            if target < weight {
                return index;
            }
            target -= weight;
        }
        last
    }

    /// A uniform choice among `items`.
    ///
    /// # Panics
    ///
    /// If `items` is empty.
    pub fn choose<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[self.index(items.len())]
    }

    /// Shuffles `items` uniformly (Fisher and Yates).
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for last in (1..items.len()).rev() {
            items.swap(last, self.index(last + 1));
        }
    }

    /// `count` distinct indices below `bound`, in random order.
    ///
    /// # Panics
    ///
    /// If `count` exceeds `bound`.
    pub fn distinct(&mut self, bound: usize, count: usize) -> Vec<usize> {
        assert!(count <= bound, "{count} distinct indices below {bound}");
        let mut pool: Vec<usize> = (0..bound).collect();
        for slot in 0..count {
            let pick = slot + self.index(bound - slot);
            pool.swap(slot, pick);
        }
        pool.truncate(count);
        pool
    }

    /// `count` distinct indices, each next one drawn with probability
    /// proportional to its weight among those left, as NumPy's
    /// `choice(…, replace=False, p=…)` draws them in distribution.
    ///
    /// # Panics
    ///
    /// If fewer than `count` weights are positive.
    pub fn distinct_weighted(&mut self, weights: &[f64], count: usize) -> Vec<usize> {
        let mut left = weights.to_vec();
        let mut picked = Vec::with_capacity(count);
        for _ in 0..count {
            let index = self.weighted(&left);
            left[index] = 0.0;
            picked.push(index);
        }
        picked
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn moments(draw: impl FnMut() -> f64) -> (f64, f64) {
        let count = 200_000;
        let values: Vec<f64> = core::iter::repeat_with(draw).take(count).collect();
        let mean = values.iter().sum::<f64>() / count as f64;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / count as f64;
        (mean, variance)
    }

    #[test]
    fn xoshiro_matches_its_reference_vector() {
        // The first outputs of xoshiro256++ from the state (1, 2, 3, 4), as the
        // authors' C code gives them.
        let mut rng = Rng {
            state: [1, 2, 3, 4],
        };
        let expected = [
            41_943_041_u64,
            58_720_359,
            3_588_806_011_781_223,
            3_591_011_842_654_386,
        ];
        for value in expected {
            assert_eq!(rng.next_u64(), value);
        }
    }

    #[test]
    fn keys_separate_streams() {
        let first = Rng::from_key(&[1, 2]).next_u64();
        assert_eq!(first, Rng::from_key(&[1, 2]).next_u64());
        assert_ne!(first, Rng::from_key(&[2, 1]).next_u64());
        assert_ne!(first, Rng::from_key(&[1, 2, 0]).next_u64());
        assert_ne!(
            Rng::from_key(&[]).next_u64(),
            Rng::from_key(&[0]).next_u64()
        );
    }

    #[test]
    fn uniform_and_integers_stay_in_range() {
        let mut rng = Rng::from_key(&[7]);
        let mut seen = [false; 5];
        for _ in 0..10_000 {
            let u = rng.uniform();
            assert!((0.0..1.0).contains(&u));
            let k = rng.integer(-2, 3);
            assert!((-2..3).contains(&k));
            seen[(k + 2) as usize] = true;
        }
        assert_eq!(seen, [true; 5]);
    }

    #[test]
    fn samplers_have_their_moments() {
        let mut rng = Rng::from_key(&[11]);
        let (mean, variance) = moments(|| rng.normal());
        assert!(mean.abs() < 0.01 && (variance - 1.0).abs() < 0.02);
        let (mean, variance) = moments(|| rng.exponential());
        assert!((mean - 1.0).abs() < 0.01 && (variance - 1.0).abs() < 0.03);
        for shape in [0.3, 1.0, 2.5, 9.0] {
            let (mean, variance) = moments(|| rng.gamma(shape));
            assert!((mean - shape).abs() < 0.03 * shape.max(1.0), "{shape}");
            assert!((variance - shape).abs() < 0.06 * shape.max(1.0), "{shape}");
        }
        let (mean, _) = moments(|| rng.beta(2.0, 6.0));
        assert!((mean - 0.25).abs() < 0.005);
        // The median of a Weibull of shape k and scale 1 is ln(2)^(1/k).
        let mut draws: Vec<f64> = (0..100_001).map(|_| rng.weibull(3.0)).collect();
        draws.sort_by(f64::total_cmp);
        assert!((draws[50_000] - 2.0_f64.ln().powf(1.0 / 3.0)).abs() < 0.01);
    }

    #[test]
    fn weighted_choice_follows_the_weights() {
        let mut rng = Rng::from_key(&[13]);
        let weights = [1.0, 0.0, 3.0];
        let mut counts = [0_u32; 3];
        for _ in 0..40_000 {
            counts[rng.weighted(&weights)] += 1;
        }
        assert_eq!(counts[1], 0);
        assert!((f64::from(counts[2]) / 40_000.0 - 0.75).abs() < 0.01);
    }

    #[test]
    fn distinct_indices_are_distinct() {
        let mut rng = Rng::from_key(&[17]);
        for _ in 0..200 {
            let mut picked = rng.distinct(9, 5);
            picked.sort_unstable();
            picked.dedup();
            assert_eq!(picked.len(), 5);
            let mut picked = rng.distinct_weighted(&[1.0, 2.0, 0.0, 4.0], 3);
            picked.sort_unstable();
            assert_eq!(picked, [0, 1, 3]);
        }
        let mut items: Vec<u32> = (0..50).collect();
        rng.shuffle(&mut items);
        items.sort_unstable();
        assert!(items.iter().copied().eq(0..50));
        let point = rng.dirichlet(&[0.5, 1.0, 2.0]);
        assert!((point.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }
}
