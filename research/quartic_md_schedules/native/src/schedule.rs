//! Paper2606.25971v2 §4.1, schedule/warmup ablations and Appendix B.1.
//! Endpoints and warmup discretization below are explicit local choices.
use crate::Result;
pub const BUDGET: usize = 100_000;
pub const COOLDOWN_START: usize = 80_000;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    AdamW,
    AdamMD,
    AdamMDW,
    MuonMD,
}
impl Family {
    pub fn name(self) -> &'static str {
        match self {
            Self::AdamW => "adamw",
            Self::AdamMD => "adammd",
            Self::AdamMDW => "adammdw",
            Self::MuonMD => "muonmd",
        }
    }
    pub fn peak(self) -> f64 {
        if self == Self::MuonMD { 0.01 } else { 0.002 }
    }
    pub fn decay(self) -> f32 {
        if matches!(self, Self::AdamW | Self::AdamMDW) {
            0.1
        } else {
            0.
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
    Linear,
    WsdSqrt20,
    WsdCuberoot20,
    WsdFourthroot20,
}
#[derive(Clone, Copy, Debug)]
pub struct Schedule {
    pub family: Family,
    pub shape: Shape,
    pub warmup: usize,
}
pub const BASELINE_ARMS: [&str; 2] = ["adamw-linear", "muonmd-linear"];
pub const ARMS: [&str; 16] = [
    "adammd-linear",
    "adammdw-linear",
    "adamw-linear-warmup1000",
    "muonmd-linear-warmup1000",
    "adammd-linear-warmup1000",
    "adammdw-linear-warmup1000",
    "adamw-wsd-sqrt20",
    "muonmd-wsd-sqrt20",
    "adammd-wsd-sqrt20",
    "adammdw-wsd-sqrt20",
    "adamw-wsd-sqrt20-warmup1000",
    "muonmd-wsd-sqrt20-warmup1000",
    "adammd-wsd-sqrt20-warmup1000",
    "adammdw-wsd-sqrt20-warmup1000",
    "muonmd-wsd-cuberoot20",
    "muonmd-wsd-fourthroot20",
];
impl Schedule {
    pub fn parse(label: &str) -> Result<Self> {
        if !ARMS.contains(&label) && !BASELINE_ARMS.contains(&label) {
            return Err("unknown frozen schedule arm".into());
        }
        let family = if label.starts_with("adammdw-") {
            Family::AdamMDW
        } else if label.starts_with("adammd-") {
            Family::AdamMD
        } else if label.starts_with("muonmd-") {
            Family::MuonMD
        } else {
            Family::AdamW
        };
        Ok(Self {
            family,
            shape: if label.contains("wsd-sqrt20") {
                Shape::WsdSqrt20
            } else if label.contains("wsd-cuberoot20") {
                Shape::WsdCuberoot20
            } else if label.contains("wsd-fourthroot20") {
                Shape::WsdFourthroot20
            } else {
                Shape::Linear
            },
            warmup: if label.ends_with("warmup1000") {
                1000
            } else {
                0
            },
        })
    }
    pub fn rate(self, peak: f64, completed: usize) -> f32 {
        let t = completed.min(BUDGET - 1);
        if t < self.warmup {
            return (peak * t as f64 / self.warmup as f64) as f32;
        }
        let decay = match self.shape {
            Shape::Linear => (t - self.warmup) as f64 / (BUDGET - 1 - self.warmup) as f64,
            Shape::WsdSqrt20 | Shape::WsdCuberoot20 | Shape::WsdFourthroot20 => {
                if t < COOLDOWN_START {
                    0.
                } else {
                    let u = (t - COOLDOWN_START) as f64 / (BUDGET - 1 - COOLDOWN_START) as f64;
                    match self.shape {
                        Shape::WsdCuberoot20 => u.cbrt(),
                        Shape::WsdFourthroot20 => u.sqrt().sqrt(),
                        _ => u.sqrt(),
                    }
                }
            }
        };
        (peak + (1e-8 - peak) * decay) as f32
    }
    pub fn name(self) -> &'static str {
        match self.shape {
            Shape::Linear => "linear",
            Shape::WsdSqrt20 => "wsd-sqrt20",
            Shape::WsdCuberoot20 => "wsd-cuberoot20",
            Shape::WsdFourthroot20 => "wsd-fourthroot20",
        }
    }
}
