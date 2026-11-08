//! User-selected 2% warmup, 78% stable peak, 20% cosine decay.
use crate::Result;
pub const BUDGET: usize = 100_000;
pub const COOLDOWN_START: usize = BUDGET - BUDGET * 20 / 100;
pub const WARMUP: usize = BUDGET * 2 / 100;
pub const ARM: &str = "adamw-wsd-cosine20-warmup2000";
#[derive(Clone, Copy, Debug)]
pub struct Schedule {
    pub warmup: usize,
}
impl Schedule {
    pub fn parse(label: &str) -> Result<Self> {
        if label != ARM {
            return Err("unknown frozen schedule arm".into());
        }
        Ok(Self { warmup: WARMUP })
    }
    pub fn rate(self, peak: f64, completed: usize) -> f32 {
        let t = completed.min(BUDGET - 1);
        if t < self.warmup {
            return (peak * t as f64 / self.warmup as f64) as f32;
        }
        let decay = if t < COOLDOWN_START {
            0.
        } else {
            let u = (t - COOLDOWN_START) as f64 / (BUDGET - 1 - COOLDOWN_START) as f64;
            0.5 * (1. - (std::f64::consts::PI * u).cos())
        };
        (peak + (1e-8 - peak) * decay) as f32
    }
    pub fn name(self) -> &'static str {
        "wsd-cosine20"
    }
}
