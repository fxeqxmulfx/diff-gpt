//! Shared user-selected WSD schedule, relative to each frozen task budget.
use crate::Result;
pub const PEAK: f64 = 0.001;
#[derive(Clone, Copy, Debug)]
pub struct Schedule {
    pub budget: usize,
    pub warmup: usize,
    pub cooldown_start: usize,
}
impl Schedule {
    pub fn new(budget: usize) -> Result<Self> {
        if ![20_000, 30_000, 100_000].contains(&budget) {
            return Err("undeclared task update budget".into());
        }
        Ok(Self {
            budget,
            warmup: budget * 2 / 100,
            cooldown_start: budget * 80 / 100,
        })
    }
    pub fn rate(self, completed: usize) -> f32 {
        let t = completed.min(self.budget - 1);
        if t < self.warmup {
            return (PEAK * t as f64 / self.warmup as f64) as f32;
        }
        let decay = if t < self.cooldown_start {
            0.
        } else {
            let u =
                (t - self.cooldown_start) as f64 / (self.budget - 1 - self.cooldown_start) as f64;
            0.5 * (1. - (std::f64::consts::PI * u).cos())
        };
        (PEAK + (1e-8 - PEAK) * decay) as f32
    }
}
