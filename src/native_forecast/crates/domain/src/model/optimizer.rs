//! Schedule-free AdamW (Defazio et al., "The Road Less Scheduled",
//! arXiv:2405.15682), as `ScheduleFreeAdamW` of
//! `src/diff_gpt/domain/optimizer/schedule_free.py` computes it.
//!
//! The optimizer keeps two points: `z`, which takes the plain steps, and
//! `y`, where the gradients are evaluated. The point to evaluate the model
//! at, `x`, is a weighted average of the `z`, and `y` lies between the two:
//! `y = (1 − β₁) z + β₁ x`. Only `y` and `z` are stored.
//!
//! One step with the gradient `g` at `y`, the step count `t` from 1:
//! - `lr_t = lr · min(1, t / warmup)`, and `c_t = lr_t² / Σ lr_i²`, the
//!   weight of this step in the average;
//! - `v = β₂ v + (1 − β₂) g²`, and `u = g / (√(v / (1 − β₂^t)) + ε) + λ y`;
//! - `y ← y + c_t (z − y) + lr_t (β₁ (1 − c_t) − 1) u`;
//! - `z ← z − lr_t u`.

/// The settings of the optimizer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Settings {
    /// The step size after the warmup.
    pub rate: f64,
    /// The steps over which the step size rises linearly from 0.
    pub warmup: u64,
    /// The interpolation between `z` and `x` that gives `y`.
    pub beta1: f64,
    /// The decay of the average of the squared gradients.
    pub beta2: f64,
    /// The weight decay, on `y`.
    pub decay: f64,
    /// What is added to the root of the averaged squares.
    pub epsilon: f64,
}

impl Default for Settings {
    /// The settings of `synthetic_pretrain.py`: rate 0.01, warmup 1000,
    /// betas 0.9 and 0.95, weight decay 0.03.
    fn default() -> Self {
        Self {
            rate: 0.01,
            warmup: 1000,
            beta1: 0.9,
            beta2: 0.95,
            decay: 0.03,
            epsilon: 1e-8,
        }
    }
}

/// The numbers of the optimizer that are not per parameter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Schedule {
    settings: Settings,
    /// The steps taken.
    steps: u64,
    /// The sum of the squared step sizes so far.
    total: f64,
}

/// The factors of one step, the same for every parameter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Factors {
    /// `lr_t`.
    rate: f32,
    /// `c_t`.
    share: f32,
    /// `lr_t (β₁ (1 − c_t) − 1)`.
    pull: f32,
    /// `β₂`.
    beta2: f32,
    /// `1 / (1 − β₂^t)`.
    correction: f32,
    /// `λ`.
    decay: f32,
    /// `ε`.
    epsilon: f32,
}

impl Schedule {
    /// The schedule before its first step.
    #[must_use]
    pub const fn new(settings: Settings) -> Self {
        Self {
            settings,
            steps: 0,
            total: 0.0,
        }
    }

    /// The schedule after `steps` steps whose squared step sizes sum to
    /// `total`: what a saved run goes on from.
    #[must_use]
    pub const fn resumed(settings: Settings, steps: u64, total: f64) -> Self {
        Self {
            settings,
            steps,
            total,
        }
    }

    /// The settings.
    #[must_use]
    pub const fn settings(&self) -> &Settings {
        &self.settings
    }

    /// The steps taken.
    #[must_use]
    pub const fn steps(&self) -> u64 {
        self.steps
    }

    /// The sum of the squared step sizes so far.
    #[must_use]
    pub const fn total(&self) -> f64 {
        self.total
    }

    /// Counts a step and returns its factors.
    pub fn advance(&mut self) -> Factors {
        let settings = self.settings;
        self.steps += 1;
        let warm = (self.steps as f64 / settings.warmup.max(1) as f64).min(1.0);
        let rate = settings.rate * warm;
        self.total += rate * rate;
        let share = if self.total > 0.0 {
            rate * rate / self.total
        } else {
            0.0
        };
        Factors {
            rate: rate as f32,
            share: share as f32,
            pull: (rate * (settings.beta1 * (1.0 - share) - 1.0)) as f32,
            beta2: settings.beta2 as f32,
            correction: (1.0 / (1.0 - settings.beta2.powf(self.steps as f64))) as f32,
            decay: settings.decay as f32,
            epsilon: settings.epsilon as f32,
        }
    }
}

impl Factors {
    /// Takes the step on a range of parameters: `y` where the gradient
    /// `gradient` was evaluated, `z` and the averaged squares `squares`.
    ///
    /// # Panics
    ///
    /// If the four ranges differ in length.
    pub fn apply(&self, y: &mut [f32], z: &mut [f32], squares: &mut [f32], gradient: &[f32]) {
        let count = y.len();
        assert_eq!(
            (z.len(), squares.len(), gradient.len()),
            (count, count, count)
        );
        let keep = 1.0 - self.beta2;
        for (((y, z), v), g) in y.iter_mut().zip(z).zip(squares).zip(gradient) {
            *v = self.beta2 * *v + keep * g * g;
            let update = g / ((*v * self.correction).sqrt() + self.epsilon) + self.decay * *y;
            *y += self.share * (*z - *y) + self.pull * update;
            *z -= self.rate * update;
        }
    }
}

/// The point to evaluate the model at, into `x`: `y + (z − y)(1 − 1/β₁)`.
///
/// # Panics
///
/// If the three ranges differ in length.
pub fn averaged(y: &[f32], z: &[f32], beta1: f64, x: &mut [f32]) {
    assert_eq!((z.len(), x.len()), (y.len(), y.len()));
    let back = (1.0 - 1.0 / beta1) as f32;
    for ((x, y), z) in x.iter_mut().zip(y).zip(z) {
        *x = y + (z - y) * back;
    }
}

/// Scales a gradient down to the norm `most` if its norm `norm` is above
/// it, and returns the factor.
pub fn clip(gradient: &mut [f32], norm: f64, most: f64) -> f64 {
    if norm <= most || norm == 0.0 {
        return 1.0;
    }
    let factor = most / norm;
    gradient.iter_mut().for_each(|g| *g *= factor as f32);
    factor
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The optimizer as the paper states it, in double precision, with `x`
    /// kept: `y = (1 − β₁) z + β₁ x`, a step of `z`, `x` the average.
    struct Reference {
        settings: Settings,
        x: Vec<f64>,
        z: Vec<f64>,
        v: Vec<f64>,
        steps: u64,
        total: f64,
    }

    impl Reference {
        fn y(&self) -> Vec<f64> {
            let beta1 = self.settings.beta1;
            let pairs = self.x.iter().zip(&self.z);
            pairs.map(|(x, z)| (1.0 - beta1) * z + beta1 * x).collect()
        }

        fn step(&mut self, gradient: &[f64]) {
            let settings = self.settings;
            let y = self.y();
            self.steps += 1;
            let rate = settings.rate * (self.steps as f64 / settings.warmup as f64).min(1.0);
            self.total += rate * rate;
            let share = rate * rate / self.total;
            let correction = 1.0 - settings.beta2.powi(self.steps as i32);
            for index in 0..y.len() {
                let g = gradient[index];
                self.v[index] = settings.beta2 * self.v[index] + (1.0 - settings.beta2) * g * g;
                let update = g / ((self.v[index] / correction).sqrt() + settings.epsilon)
                    + settings.decay * y[index];
                self.z[index] -= rate * update;
                self.x[index] = (1.0 - share) * self.x[index] + share * self.z[index];
            }
        }
    }

    /// The gradient of `Σ (w − target)² / 2` with a part that changes by
    /// step.
    fn gradient(y: &[f64], step: u64) -> Vec<f64> {
        y.iter()
            .enumerate()
            .map(|(index, w)| w - (index as f64 - 1.5) + 0.3 * ((step + index as u64) as f64).sin())
            .collect()
    }

    #[test]
    fn the_packed_steps_are_those_of_the_paper() {
        let settings = Settings {
            rate: 0.05,
            warmup: 5,
            ..Settings::default()
        };
        let start = [0.7_f64, -0.2, 1.4, 0.0];
        let mut reference = Reference {
            settings,
            x: start.to_vec(),
            z: start.to_vec(),
            v: vec![0.0; 4],
            steps: 0,
            total: 0.0,
        };
        let mut schedule = Schedule::new(settings);
        let mut y: Vec<f32> = start.iter().map(|w| *w as f32).collect();
        let (mut z, mut squares) = (y.clone(), vec![0.0_f32; 4]);
        for step in 0..40 {
            let at: Vec<f64> = y.iter().map(|w| f64::from(*w)).collect();
            let g = gradient(&at, step);
            reference.step(&gradient(&reference.y(), step));
            let single: Vec<f32> = g.iter().map(|g| *g as f32).collect();
            schedule
                .advance()
                .apply(&mut y, &mut z, &mut squares, &single);
            let mut x = vec![0.0_f32; 4];
            averaged(&y, &z, settings.beta1, &mut x);
            for index in 0..4 {
                assert!(
                    (f64::from(y[index]) - reference.y()[index]).abs() < 2e-5,
                    "y, step {step}"
                );
                assert!(
                    (f64::from(z[index]) - reference.z[index]).abs() < 2e-5,
                    "z, step {step}"
                );
                assert!(
                    (f64::from(x[index]) - reference.x[index]).abs() < 2e-5,
                    "x, step {step}"
                );
            }
        }
        assert_eq!(schedule.steps(), 40);
        // Five warmup steps of 0.01, …, 0.05, then 35 of 0.05.
        let total = 0.05_f64.powi(2) * (0.04 + 0.16 + 0.36 + 0.64 + 36.0);
        assert!((schedule.total() - total).abs() < 1e-12);
        let resumed = Schedule::resumed(settings, schedule.steps(), schedule.total());
        assert_eq!(resumed, schedule);
        // The averaged point has moved toward the targets.
        let mut x = vec![0.0_f32; 4];
        averaged(&y, &z, settings.beta1, &mut x);
        let distance = |point: &[f64]| -> f64 {
            let terms = point.iter().enumerate();
            terms
                .map(|(index, w)| (w - (index as f64 - 1.5)).powi(2))
                .sum()
        };
        let end: Vec<f64> = x.iter().map(|w| f64::from(*w)).collect();
        assert!(distance(&end) < 0.5 * distance(&start));
    }

    #[test]
    fn clipping_scales_a_long_gradient_only() {
        let mut gradient = [3.0_f32, 4.0];
        assert_eq!(clip(&mut gradient, 5.0, 10.0), 1.0);
        assert_eq!(gradient, [3.0, 4.0]);
        assert!((clip(&mut gradient, 5.0, 1.0) - 0.2).abs() < 1e-12);
        assert!((gradient[0] - 0.6).abs() < 1e-6 && (gradient[1] - 0.8).abs() < 1e-6);
        assert_eq!(clip(&mut [0.0, 0.0], 0.0, 1.0), 1.0);
    }
}
