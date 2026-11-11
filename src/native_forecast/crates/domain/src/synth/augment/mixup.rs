//! Mixtures of series: `MixUpAugmenter` of TempoPFN's
//! `src/data/augmentations.py`. The weights of a mixture are a point of the
//! simplex, drawn from a symmetric Dirichlet distribution whose
//! concentration is itself drawn, log-uniformly between 0.1 (one series
//! dominates) and 5 (the series weigh about the same).

use crate::random::Rng;

/// The most series a mixture combines (`max_n_series_to_combine`).
pub const MOST: usize = 10;

/// The share of the mixtures of a batch whose weights drift in time
/// (`p_time_dependent`).
const DRIFTING: f64 = 0.5;

/// A concentration: `_sample_alpha`.
pub fn concentration(rng: &mut Rng) -> f64 {
    10.0_f64.powf(rng.uniform_in(0.1_f64.log10(), 5.0_f64.log10()))
}

/// The number of series of a mixture: uniform among 2, …, [`MOST`].
pub fn count(rng: &mut Rng) -> usize {
    rng.integer(2, MOST as i64 + 1) as usize
}

/// Weights for `count` series, in single precision.
fn weights(rng: &mut Rng, count: usize, concentration: f64) -> Vec<f32> {
    rng.dirichlet(&vec![concentration; count])
        .into_iter()
        .map(|weight| weight as f32)
        .collect()
}

/// The mixture of `sources` with one set of weights: `_static_mix`.
///
/// # Panics
///
/// If there is no source, or the sources differ in length.
#[must_use]
pub fn fixed(rng: &mut Rng, sources: &[&[f32]], concentration: f64) -> Vec<f32> {
    let weights = weights(rng, sources.len(), concentration);
    let mut mixed = vec![0.0_f32; sources[0].len()];
    for (source, weight) in sources.iter().zip(weights) {
        assert_eq!(source.len(), mixed.len(), "a source of another length");
        for (total, value) in mixed.iter_mut().zip(*source) {
            *total += value * weight;
        }
    }
    mixed
}

/// The mixture of `sources` whose weights go in a straight line from one
/// point of the simplex at the first step to another at the last:
/// `_simplex_path_mix`.
///
/// # Panics
///
/// If there is no source, or the sources differ in length.
#[must_use]
pub fn drifting(rng: &mut Rng, sources: &[&[f32]], concentration: f64) -> Vec<f32> {
    let first = weights(rng, sources.len(), concentration);
    let last = weights(rng, sources.len(), concentration);
    let length = sources[0].len();
    let mut mixed = vec![0.0_f32; length];
    for ((source, first), last) in sources.iter().zip(first).zip(last) {
        assert_eq!(source.len(), length, "a source of another length");
        for (step, (total, value)) in mixed.iter_mut().zip(*source).enumerate() {
            let part = step as f32 / (length.max(2) - 1) as f32;
            *total += value * (first * (1.0 - part) + last * part);
        }
    }
    mixed
}

/// `MixUpAugmenter.transform`: each series of the batch is replaced, with
/// the given probability, by a mixture of 2 to [`MOST`] of the other series
/// as they were, with fixed or drifting weights. A batch of [`MOST`] series
/// or fewer stays as it is.
pub fn batch(rng: &mut Rng, batch: &mut [Vec<f32>], probability: f64) {
    if batch.len() <= MOST {
        return;
    }
    let mut mixtures = Vec::new();
    for target in 0..batch.len() {
        if !rng.chance(probability) {
            continue;
        }
        let count = count(rng);
        let sources: Vec<&[f32]> = rng
            .distinct(batch.len() - 1, count)
            .into_iter()
            .map(|other| batch[other + usize::from(other >= target)].as_slice())
            .collect();
        let concentration = concentration(rng);
        let mixed = if rng.chance(DRIFTING) {
            drifting(rng, &sources, concentration)
        } else {
            fixed(rng, &sources, concentration)
        };
        mixtures.push((target, mixed));
    }
    for (target, mixed) in mixtures {
        batch[target] = mixed;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_concentrations_are_log_uniform_between_a_tenth_and_five() {
        let mut rng = Rng::from_key(&[1]);
        let draws: Vec<f64> = (0..4000).map(|_| concentration(&mut rng)).collect();
        assert!(draws.iter().all(|draw| (0.1..5.0).contains(draw)));
        // The median of a log-uniform draw is the geometric mean of the ends.
        let below = draws.iter().filter(|draw| **draw < 0.5_f64.sqrt()).count();
        assert!((1850..2150).contains(&below), "{below} below the median");
        let counts: Vec<usize> = (0..900).map(|_| count(&mut rng)).collect();
        assert!(counts.iter().all(|count| (2..=MOST).contains(count)));
        assert!(counts.contains(&2) && counts.contains(&MOST));
    }

    #[test]
    fn a_fixed_mixture_is_a_convex_combination() {
        let mut rng = Rng::from_key(&[2]);
        let ones = vec![1.0_f32; 50];
        let ramp: Vec<f32> = (0..50).map(|step| step as f32).collect();
        for _ in 0..100 {
            let mixed = fixed(&mut rng, &[&ones, &ramp], 1.0);
            // w + (1 − w) t: the first value is the weight of the ones.
            let weight = mixed[0];
            assert!((0.0..=1.0).contains(&weight));
            for (step, value) in mixed.iter().enumerate() {
                let expected = weight + (1.0 - weight) * step as f32;
                assert!((value - expected).abs() < 1e-3, "{value} and {expected}");
            }
        }
    }

    #[test]
    fn drifting_weights_go_from_one_point_to_another() {
        let mut rng = Rng::from_key(&[3]);
        let ones = vec![1.0_f32; 101];
        let zeros = vec![0.0_f32; 101];
        let mut moved = 0;
        for _ in 0..100 {
            // The mixture is the weight of the ones: a straight line.
            let mixed = drifting(&mut rng, &[&ones, &zeros], 0.5);
            let (first, last) = (mixed[0], mixed[100]);
            assert!((0.0..=1.0).contains(&first) && (0.0..=1.0).contains(&last));
            assert!((mixed[50] - (first + last) / 2.0).abs() < 1e-5);
            assert!((mixed[25] - (0.75 * first + 0.25 * last)).abs() < 1e-5);
            moved += usize::from((first - last).abs() > 0.05);
        }
        assert!(moved > 70, "{moved} of 100 moved");
    }

    #[test]
    fn a_batch_mixes_the_other_series_as_they_were() {
        let mut rng = Rng::from_key(&[4]);
        // Series `i` is the indicator of step `i`: a mixture shows its
        // sources and their weights.
        let original: Vec<Vec<f32>> = (0..24)
            .map(|series| {
                (0..24)
                    .map(|step| f32::from(u8::from(step == series)))
                    .collect()
            })
            .collect();
        let (mut replaced, mut total) = (0, 0);
        for _ in 0..50 {
            let mut mixed = original.clone();
            batch(&mut rng, &mut mixed, 0.5);
            for (target, series) in mixed.iter().enumerate() {
                total += 1;
                if *series == original[target] {
                    continue;
                }
                replaced += 1;
                assert_eq!(series[target], 0.0, "a series is not mixed into itself");
                let sources = series.iter().filter(|weight| **weight > 0.0).count();
                assert!((1..=MOST).contains(&sources), "{sources} sources");
            }
        }
        let share = f64::from(replaced) / f64::from(total);
        assert!((share - 0.5).abs() < 0.05, "replaced {share}");
        // A small batch is left alone.
        let mut small = original[..MOST].to_vec();
        batch(&mut rng, &mut small, 1.0);
        assert_eq!(small, original[..MOST]);
    }
}
