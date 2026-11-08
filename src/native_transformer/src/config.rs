use crate::{Config, Result, layout, position::Axes, schedule::Schedule};
use serde_json::{Value, json};

/// Positional layout: generic 1D main or DiffGPT's required axial 2D layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Positions {
    /// Learned 1D frequencies, shared by all heads and layers; the default main.
    LeRoPE,
    /// Half the complex pairs encode time and half feature, flattened row-major.
    AxialLeRoPE { features: usize },
}
impl Positions {
    pub(crate) fn axes(self) -> Axes {
        match self {
            Self::LeRoPE => Axes::Sequence,
            Self::AxialLeRoPE { features } => Axes::Axial { features },
        }
    }
}

/// Model dimensions and the fixed AdamW/WSD training budget.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelConfig {
    /// Real hidden width: 64, 128 or 256.
    pub width: usize,
    /// Number of attention heads; real per-head width must be 16, 32 or 128.
    pub heads: usize,
    /// Number of transformer blocks.
    pub layers: usize,
    /// Number of discrete tokens.
    pub vocab: usize,
    /// Maximum examples per forward/backward call.
    pub max_batch: usize,
    /// Maximum tokens per example, including generated prefix tokens.
    pub max_length: usize,
    /// Supported frozen WSD budget: 5000, 20000, 30000 or 100000 updates.
    pub budget: usize,
    /// Generic 1D main positions or DiffGPT's axial time-feature positions.
    pub positions: Positions,
    /// Actual maximum input-token length in training, never allocated capacity.
    /// Required by the DiffGPT interface for inference-only YaRN.
    pub training_input_length: Option<usize>,
}
impl ModelConfig {
    /// The common D128/H4/L1 main geometry; specify capacities for the task.
    pub fn main(vocab: usize, max_batch: usize, max_length: usize, budget: usize) -> Self {
        Self {
            width: 128,
            heads: 4,
            layers: 1,
            vocab,
            max_batch,
            max_length,
            budget,
            positions: Positions::LeRoPE,
            training_input_length: None,
        }
    }
    /// D128/H1/L1 DiffGPT defaults: axial time-feature LeRoPE and a real context reference.
    pub fn diffgpt(
        vocab: usize,
        features: usize,
        max_batch: usize,
        max_length: usize,
        budget: usize,
        training_input_length: usize,
    ) -> Result<Self> {
        let mut out = Self::main(vocab, max_batch, max_length, budget);
        out.heads = 1;
        out.positions = Positions::AxialLeRoPE { features };
        out.training_input_length = Some(training_input_length);
        out.validate()?;
        Ok(out)
    }
    /// Actual training extent in the units used by this positional layout.
    pub fn training_extent(&self) -> Option<usize> {
        self.training_input_length
            .map(|n| self.positions.axes().time_extent(n))
    }
    /// Validate native index ranges before constructing parameters or GPU buffers.
    pub fn validate(&self) -> Result<()> {
        Schedule::new(self.budget)?;
        if ![64, 128, 256].contains(&self.width)
            || self.heads == 0
            || !self.width.is_multiple_of(self.heads)
            || ![16, 32, 128].contains(&(self.width / self.heads))
            || self.layers == 0
            || self.vocab < 2
            || self.max_batch == 0
            || self.max_length == 0
        {
            return Err("unsupported model dimensions or empty capacity".into());
        }
        if matches!(self.positions, Positions::AxialLeRoPE { features: 0 }) {
            return Err("axial positions need at least one feature".into());
        }
        if self
            .training_input_length
            .is_some_and(|n| n == 0 || n > self.max_length)
        {
            return Err("actual training input length is zero or exceeds capacity".into());
        }
        let checked = |values: &[usize]| -> Result<usize> {
            let n = values
                .iter()
                .try_fold(1usize, |a, b| a.checked_mul(*b))
                .ok_or("model size overflow")?;
            if n > u32::MAX as usize / 4 {
                return Err("model exceeds native index range".into());
            }
            Ok(n)
        };
        let r = checked(&[self.max_batch, self.max_length])?;
        checked(&[r, self.heads, self.max_length])?;
        checked(&[r, self.width, 4])?;
        checked(&[r, layout::hidden(self.width, ""), 2])?;
        checked(&[r, self.vocab])?;
        checked(&[self.layers, self.width, self.width, 16])?;
        checked(&[self.vocab, self.width])?;
        let count = layout::total_count(&self.native_unchecked());
        checked(&[count, 3])?;
        Ok(())
    }
    pub(crate) fn native_unchecked(&self) -> Config {
        Config {
            width: self.width,
            heads: self.heads,
            layers: self.layers,
            vocab: self.vocab,
            max_batch: self.max_batch,
            max_length: self.max_length,
            normalized: true,
            ssmax: true,
            gain: self.heads as f32 / self.width as f32,
            lr: 0.001,
            decay: 0.1,
            warmup: self.budget * 2 / 100,
            architecture: "complex-phase-score-lerope".into(),
            phase_optimizer: false,
        }
    }
    /// Expanded FP32 parameter count, including duplicated complex matrix entries.
    pub fn parameter_count(&self) -> Result<usize> {
        self.validate()?;
        Ok(layout::total_count(&self.native_unchecked()))
    }
    /// Number of independent optimized real coordinates.
    pub fn independent_parameter_count(&self) -> Result<usize> {
        self.validate()?;
        Ok(layout::independent_count(&self.native_unchecked()))
    }
    /// JSON representation used in portable native checkpoints and the CLI.
    pub fn to_json(&self) -> Value {
        let positions = match self.positions {
            Positions::LeRoPE => json!({"kind":"lerope"}),
            Positions::AxialLeRoPE { features } => {
                json!({"kind":"axial_lerope","features":features})
            }
        };
        json!({"width":self.width,"heads":self.heads,"layers":self.layers,"vocab":self.vocab,
            "max_batch":self.max_batch,"max_length":self.max_length,"budget":self.budget,"positions":positions,
            "training_input_length":self.training_input_length})
    }
    /// Parse and validate configuration without accessing CUDA.
    pub fn from_json(value: &Value) -> Result<Self> {
        let n = |k: &str| -> Result<usize> {
            Ok(value[k]
                .as_u64()
                .ok_or_else(|| format!("missing integer {k}"))?
                .try_into()?)
        };
        let positions = match value["positions"]["kind"].as_str() {
            Some("lerope") => Positions::LeRoPE,
            Some("axial_lerope") => Positions::AxialLeRoPE {
                features: value["positions"]["features"]
                    .as_u64()
                    .ok_or("missing feature count")?
                    .try_into()?,
            },
            _ => return Err("unknown positional layout".into()),
        };
        let out = Self {
            width: n("width")?,
            heads: n("heads")?,
            layers: n("layers")?,
            vocab: n("vocab")?,
            max_batch: n("max_batch")?,
            max_length: n("max_length")?,
            budget: n("budget")?,
            positions,
            training_input_length: if value["training_input_length"].is_null() {
                None
            } else {
                Some(n("training_input_length")?)
            },
        };
        out.validate()?;
        Ok(out)
    }
}
