use crate::{
    Checkpoint, Forecast, Model, ModelConfig, Positions, Result, Step,
    codec::{Difference, Domain, Matrix, Tokenizer},
};

/// DiffGPT's production interface: axial 2D LeRoPE with inference-only YaRN.
/// One-dimensional models remain available through [`Model`], outside this interface.
pub struct DiffGpt {
    model: Model,
}
fn validate(config: &ModelConfig) -> Result<()> {
    config.validate()?;
    if !matches!(config.positions, Positions::AxialLeRoPE { .. }) {
        return Err("DiffGPT requires2Daxial LeRoPE; use the generic Model API for1D".into());
    }
    if config.training_input_length.is_none() {
        return Err("DiffGPT needs the actual maximum training input length for YaRN".into());
    }
    Ok(())
}
impl DiffGpt {
    /// Create the D128/H1/L1 2D default. `training_input_length` comes from actual
    /// training inputs, including any teacher-forced feature tokens.
    pub fn new(
        vocab: usize,
        features: usize,
        max_batch: usize,
        max_length: usize,
        budget: usize,
        training_input_length: usize,
        seed: u32,
    ) -> Result<Self> {
        Self::with_config(
            ModelConfig::diffgpt(
                vocab,
                features,
                max_batch,
                max_length,
                budget,
                training_input_length,
            )?,
            seed,
        )
    }
    /// Construct a chosen width/depth while enforcing DiffGPT's 2D-only policy.
    pub fn with_config(config: ModelConfig, seed: u32) -> Result<Self> {
        validate(&config)?;
        Ok(Self {
            model: Model::new(config, seed)?,
        })
    }
    /// Load the saved axes/context without reinterpreting 1D checkpoint weights.
    pub fn from_checkpoint(checkpoint: &Checkpoint) -> Result<Self> {
        validate(checkpoint.config())?;
        Ok(Self {
            model: Model::from_checkpoint(checkpoint)?,
        })
    }
    /// Configuration including feature count and the actual training input length.
    pub fn config(&self) -> &ModelConfig {
        self.model.config()
    }
    /// One ordinary training update; YaRN is never applied during training.
    pub fn train_step(
        &mut self,
        tokens: &[u32],
        targets: &[u32],
        batch: usize,
        length: usize,
    ) -> Result<Step> {
        self.model.train_step(tokens, targets, batch, length)
    }
    /// Forecast with time-axis YaRN only when the declared rollout exceeds training.
    /// Feature-axis frequencies remain unchanged. All three encoders are supported.
    pub fn forecast(
        &mut self,
        observed: &Matrix<f64>,
        domains: &[Domain],
        operators: &[Difference],
        tokenizer: Tokenizer,
        future_rows: usize,
    ) -> Result<Forecast> {
        let training_extent = self
            .config()
            .training_extent()
            .ok_or("missing training extent")?;
        self.model.forecast(
            observed,
            domains,
            operators,
            tokenizer,
            future_rows,
            Some(training_extent),
        )
    }
    /// Preserve weights, moments, axes and real context reference in one checkpoint.
    pub fn checkpoint(&self) -> Result<Checkpoint> {
        self.model.checkpoint()
    }
    /// Completed optimizer updates.
    pub fn completed(&self) -> usize {
        self.model.completed()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn diffgpt_requires_2d_and_a_real_training_reference() -> Result<()> {
        let cfg = ModelConfig::diffgpt(512, 2, 512, 17, 5000, 5)?;
        validate(&cfg)?;
        assert_eq!(cfg.training_extent(), Some(3));
        assert_eq!(ModelConfig::from_json(&cfg.to_json())?, cfg);
        let mut one = cfg.clone();
        one.positions = Positions::LeRoPE;
        assert_eq!(one.training_extent(), Some(5));
        one.validate()?;
        assert!(validate(&one).is_err());
        let mut missing = cfg;
        missing.training_input_length = None;
        assert!(validate(&missing).is_err());
        assert!(ModelConfig::diffgpt(512, 2, 512, 17, 5000, 18).is_err());
        Ok(())
    }
}
