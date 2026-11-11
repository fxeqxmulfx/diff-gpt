//! Trains of short triangular bumps on a level of 1, of either sign:
//! TempoPFN's `generate_spikes` (`src/synthetic_generation/utils.py`), which
//! its ForecastPFN and Gaussian-process priors add to or multiply into their
//! series.

use crate::random::Rng;

/// How the bumps are placed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Placement {
    /// At a fixed interval.
    Regular,
    /// At a fixed interval, in patches: some kept, the rest of a patch left
    /// out.
    Patchy,
    /// At random places.
    Random,
}

/// A series of `size` steps: 1 everywhere but for bumps of one to three
/// steps up and as many down, the whole negated with probability 0.3.
/// `placement` and `interval` are drawn where they are not given.
pub fn bumps(
    rng: &mut Rng,
    size: usize,
    placement: Option<Placement>,
    interval: Option<usize>,
) -> Vec<f64> {
    let build_up = if size < 120 {
        1
    } else if size < 250 {
        [2, 1][rng.weighted(&[0.3, 0.7])]
    } else {
        [3, 2, 1][rng.weighted(&[0.15, 0.45, 0.4])]
    };
    let duration = 2 * build_up;
    let mut placement = placement.unwrap_or_else(|| {
        [Placement::Regular, Placement::Patchy, Placement::Random][rng.weighted(&[0.4, 0.5, 0.1])]
    });
    if placement == Placement::Patchy && size < 64 {
        placement = Placement::Regular;
    }

    let mut patch = (0, 0);
    let places: Vec<usize> = if placement == Placement::Random {
        let most = (size / (duration * 3)).clamp(6, 20);
        let count = rng.integer(4, most as i64) as usize;
        let mut places: Vec<usize> = rng
            .distinct(
                size.saturating_sub(duration),
                count.min(size.saturating_sub(duration)),
            )
            .into_iter()
            .map(|offset| offset + duration)
            .collect();
        places.sort_unstable();
        places
    } else {
        let interval = interval.unwrap_or_else(|| {
            let low = (duration as f64 / 0.15).ceil() as i64;
            let high = (duration as f64 / 0.05).ceil() as i64;
            rng.integer(low, high) as usize
        });
        let count = size.div_ceil(interval);
        if placement == Placement::Patchy {
            let size_bound = ((count as f64 * 0.7).max(3.0)) as i64;
            let patch_size = rng.integer(2, size_bound);
            let kept = rng.integer((patch_size as f64 * 0.4).ceil() as i64, patch_size);
            patch = (patch_size, kept);
        }
        (interval..size).step_by(interval).collect()
    };

    // One draw decides for all the bumps of a regular train whether they are
    // low or high; a random train decides for each.
    let shared = (placement != Placement::Random).then(|| rng.uniform());
    let mut values = vec![0.0; size];
    let mut in_patch = 0;
    for place in places {
        let peak = place as i64 - 1;
        if placement == Placement::Patchy {
            if in_patch >= patch.0 {
                in_patch = 0;
            }
            in_patch += 1;
            if in_patch > patch.1 {
                continue;
            }
        }
        let low = shared.unwrap_or_else(|| rng.uniform()) < 0.7;
        let rate = if low {
            rng.uniform_in(0.5, 2.0)
        } else {
            rng.uniform_in(2.5, 5.0)
        };
        let first = peak - build_up as i64 + 1;
        for step in 0..build_up as i64 {
            if (0..size as i64).contains(&(first + step)) {
                values[(first + step) as usize] = rate * (step + 1) as f64;
            }
        }
        for step in 1..build_up as i64 {
            if peak + step < size as i64 {
                // A negative index counts from the end, as in NumPy.
                let mirrored = (peak - step).rem_euclid(size as i64) as usize;
                values[(peak + step) as usize] = values[mirrored];
            }
        }
    }
    let sign = if rng.chance(0.7) { 1.0 } else { -1.0 };
    values
        .into_iter()
        .map(|value| sign * (value + 1.0))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_regular_train_has_a_bump_at_each_interval() {
        let mut rng = Rng::from_key(&[1]);
        for _ in 0..200 {
            let values = bumps(&mut rng, 2048, Some(Placement::Regular), Some(100));
            let sign = values[0].signum();
            assert_eq!(values[0], sign);
            // The peaks are one step before the multiples of the interval.
            for place in (100..2048).step_by(100) {
                let peak = sign * values[place - 1] - 1.0;
                assert!((0.5..=15.0).contains(&peak), "peak {peak}");
                assert!(sign * values[place - 1] >= sign * values[place]);
            }
            let off: usize = values.iter().filter(|&&value| value != sign).count();
            assert!((20..=100).contains(&off), "{off} steps off the level");
        }
    }

    #[test]
    fn patches_leave_bumps_out_and_random_trains_have_few() {
        let mut rng = Rng::from_key(&[2]);
        let count = |values: &[f64]| {
            let level = values[0];
            values
                .windows(2)
                .filter(|pair| pair[0] == level && pair[1] != level)
                .count()
        };
        for _ in 0..200 {
            let patchy = bumps(&mut rng, 2048, Some(Placement::Patchy), Some(50));
            let kept = count(&patchy);
            assert!((14..40).contains(&kept), "{kept} of 40 kept");
            let random = bumps(&mut rng, 2048, Some(Placement::Random), None);
            assert!((1..20).contains(&count(&random)));
        }
    }

    #[test]
    fn drawn_trains_are_of_both_signs_and_bounded() {
        let mut rng = Rng::from_key(&[3]);
        let mut negative = 0;
        for _ in 0..1000 {
            let values = bumps(&mut rng, 2048, None, None);
            assert_eq!(values.len(), 2048);
            let sign = values[0].signum();
            assert!(
                values
                    .iter()
                    .all(|&value| (1.0..=16.0).contains(&(sign * value)))
            );
            negative += usize::from(sign < 0.0);
        }
        assert!((240..360).contains(&negative), "{negative} negated of 1000");
        assert_eq!(
            bumps(&mut rng, 30, Some(Placement::Patchy), Some(7)).len(),
            30
        );
    }
}
