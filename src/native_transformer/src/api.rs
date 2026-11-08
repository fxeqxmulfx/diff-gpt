use crate::{
    Checkpoint, Config, Engine, IGNORE_TARGET, ModelConfig, Positions, Result,
    codec::{self, Difference, Domain, Matrix, Tokenizer},
    layout,
    optimizer::Optimizer,
    position::Extension,
    random::Random,
    schedule::Schedule,
};

/// Host-only update counters. A training step does not copy GPU metrics back.
#[derive(Clone, Copy, Debug)]
pub struct Step {
    /// Completed optimizer updates.
    pub completed: usize,
    /// Number of training examples presented, including repeated examples.
    pub examples: u64,
    /// Learning rate used by this update.
    pub learning_rate: f32,
}
/// Explicit evaluation output, copied to the host only when requested.
pub struct Evaluation {
    /// Argmax token for every input position, in row-major batch order.
    pub predictions: Vec<u32>,
    /// Loss averaged over supervised targets; `None` when every target is masked.
    pub mean_loss: Option<f64>,
    /// Number of supervised targets.
    pub targets: usize,
    /// Number of correct supervised argmax predictions.
    pub correct: usize,
}
/// Deterministic DiffGPT forecast using the actual last-observation anchor.
pub struct Forecast {
    /// Future raw values only, without the observed prefix.
    pub values: Matrix<f64>,
    /// Generated future tokens in time-major, feature-minor order.
    pub tokens: Matrix<i64>,
    /// Context tokens clipped by the explicitly selected level grid.
    pub clipped_context_tokens: usize,
}
/// Safe reusable owner of the model, optimizer state and update counters.
pub struct Model {
    pub(crate) config: ModelConfig,
    pub(crate) engine: Engine,
    optimizer: Optimizer,
    pub(crate) completed: usize,
    pub(crate) examples: u64,
}
impl Model {
    /// Initialize the selected architecture with the frozen native RNG recipe.
    pub fn new(config: ModelConfig, seed: u32) -> Result<Self> {
        config.validate()?;
        if seed > u32::MAX - 20000 {
            return Err("initialization seed overflows router RNG offset".into());
        }
        let cfg: Config = config.native_unchecked();
        let initial = layout::initialize_model(&cfg, seed);
        let mut engine = Engine::new(cfg, &initial)?;
        engine.set_positional_axes(config.positions.axes())?;
        let optimizer = Optimizer::new(&engine, config.budget)?;
        Ok(Self {
            config,
            engine,
            optimizer,
            completed: 0,
            examples: 0,
        })
    }
    /// Restore validated weights, both moments, positional layout and WSD step.
    pub fn from_checkpoint(checkpoint: &Checkpoint) -> Result<Self> {
        checkpoint.validate()?;
        let config = checkpoint.config().clone();
        let values = checkpoint.values();
        let n = config.parameter_count()?;
        let mut engine = Engine::new(config.native_unchecked(), &values[..n])?;
        engine.set_positional_axes(config.positions.axes())?;
        engine.first_moment.write(&values[n..2 * n])?;
        engine.second_moment.write(&values[2 * n..])?;
        let optimizer = Optimizer::new(&engine, config.budget)?;
        Ok(Self {
            config,
            engine,
            optimizer,
            completed: checkpoint.completed(),
            examples: checkpoint.examples(),
        })
    }
    /// Configuration fixed for this instance.
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }
    /// Completed optimizer updates.
    pub fn completed(&self) -> usize {
        self.completed
    }
    /// Total training examples processed.
    pub fn examples(&self) -> u64 {
        self.examples
    }
    /// Name reported by the CUDA driver.
    pub fn device_name(&self) -> String {
        self.engine.cuda.name().to_owned()
    }
    fn batch(&self, tokens: &[u32], targets: &[u32], batch: usize, length: usize) -> Result<()> {
        if batch == 0
            || batch > self.config.max_batch
            || length == 0
            || length > self.config.max_length
            || batch.checked_mul(length) != Some(tokens.len())
            || targets.len() != tokens.len()
            || tokens.iter().any(|&t| t as usize >= self.config.vocab)
            || targets
                .iter()
                .any(|&t| t != IGNORE_TARGET && t as usize >= self.config.vocab)
        {
            return Err("invalid batch dimensions, token or target".into());
        }
        Ok(())
    }
    /// One forward/backward/AdamW update; no disk writes or host metric readback.
    pub fn train_step(
        &mut self,
        tokens: &[u32],
        targets: &[u32],
        batch: usize,
        length: usize,
    ) -> Result<Step> {
        self.batch(tokens, targets, batch, length)?;
        if self.completed >= self.config.budget || targets.iter().all(|&t| t == IGNORE_TARGET) {
            return Err("training budget exhausted or no supervised target".into());
        }
        let examples = self
            .examples
            .checked_add(batch as u64)
            .ok_or("example counter overflow")?;
        self.engine.set_inference_extension(None)?;
        self.engine.forward(tokens, targets, batch, length)?;
        self.engine.backward(batch, length)?;
        self.optimizer.step(&self.engine, self.completed)?;
        let learning_rate = Schedule::new(self.config.budget)?.rate(self.completed);
        self.completed += 1;
        self.examples = examples;
        if let Some(previous) = self.config.training_input_length {
            self.config.training_input_length = Some(previous.max(length));
        }
        Ok(Step {
            completed: self.completed,
            examples,
            learning_rate,
        })
    }
    /// Evaluate with native positions. This does not update weights or moments.
    pub fn evaluate(
        &mut self,
        tokens: &[u32],
        targets: &[u32],
        batch: usize,
        length: usize,
    ) -> Result<Evaluation> {
        self.batch(tokens, targets, batch, length)?;
        self.engine.set_inference_extension(None)?;
        self.engine.forward(tokens, targets, batch, length)?;
        let mut predictions = self.engine.predictions.read()?;
        predictions.truncate(tokens.len());
        let losses = self.engine.losses(tokens.len())?;
        let count = targets.iter().filter(|&&t| t != IGNORE_TARGET).count();
        let correct = predictions
            .iter()
            .zip(targets)
            .filter(|(p, t)| **t != IGNORE_TARGET && *p == *t)
            .count();
        let mean_loss = if count == 0 {
            None
        } else {
            Some(
                losses
                    .iter()
                    .zip(targets)
                    .filter(|(_, t)| **t != IGNORE_TARGET)
                    .map(|(v, _)| *v as f64)
                    .sum::<f64>()
                    / count as f64,
            )
        };
        Ok(Evaluation {
            predictions,
            mean_loss,
            targets: count,
            correct,
        })
    }
    /// Generate argmax tokens using only the prompt and earlier predictions.
    ///
    /// `yarn_training_extent` enables experimental inference-only YaRN. It is
    /// measured in tokens for 1D, time rows for axial positions. The declared
    /// document extent stays fixed for the whole rollout. The default is `None`.
    pub fn generate(
        &mut self,
        prompt: &[u32],
        count: usize,
        yarn_training_extent: Option<usize>,
    ) -> Result<Vec<u32>> {
        let extent = prompt
            .len()
            .checked_add(count.saturating_sub(1))
            .ok_or("generation size overflow")?;
        if prompt.is_empty()
            || extent > self.config.max_length
            || prompt.iter().any(|&t| t as usize >= self.config.vocab)
        {
            return Err("invalid or over-capacity generation prompt".into());
        }
        let extension = yarn_training_extent
            .map(|train| Extension::new(train, self.config.positions.axes().time_extent(extent)))
            .transpose()?;
        self.engine.set_inference_extension(extension)?;
        let result = (|| {
            let mut context = prompt.to_vec();
            for _ in 0..count {
                self.engine.forward(
                    &context,
                    &vec![IGNORE_TARGET; context.len()],
                    1,
                    context.len(),
                )?;
                let token = self.engine.predictions.read()?[context.len() - 1];
                context.push(token);
            }
            Ok(context[prompt.len()..].to_vec())
        })();
        self.engine.set_inference_extension(None)?;
        result
    }
    /// Forecast raw future rows with MASH, LGT or MASH-half and actual re-anchoring.
    /// The vocabulary and feature order must match the trained model.
    pub fn forecast(
        &mut self,
        observed: &Matrix<f64>,
        domains: &[Domain],
        operators: &[Difference],
        tokenizer: Tokenizer,
        future_rows: usize,
        yarn_training_extent: Option<usize>,
    ) -> Result<Forecast> {
        let features = observed.columns();
        if let Positions::AxialLeRoPE { features: expected } = self.config.positions
            && expected != features
        {
            return Err("forecast features differ from axial layout".into());
        }
        let encoded = codec::encode(
            observed,
            self.config.vocab as i64,
            domains,
            operators,
            tokenizer,
        )?;
        let prompt: Vec<u32> = encoded.tokens.values().iter().map(|&t| t as u32).collect();
        let count = future_rows
            .checked_mul(features)
            .ok_or("forecast size overflow")?;
        let future = self.generate(&prompt, count, yarn_training_extent)?;
        let tokens = Matrix::new(features, future.iter().map(|&t| t as i64).collect())?;
        let degree = operators
            .iter()
            .map(Difference::degree)
            .max()
            .ok_or("missing difference operators")?;
        let prefix = Matrix::new(
            features,
            observed.values()[(observed.rows() - degree) * features..].to_vec(),
        )?;
        let decoded = codec::decode(
            &prefix,
            &encoded.grid,
            &tokens,
            self.config.vocab as i64,
            operators,
            tokenizer,
        )?;
        let values = Matrix::new(features, decoded.values()[degree * features..].to_vec())?;
        Ok(Forecast {
            values,
            tokens,
            clipped_context_tokens: encoded.clipped_tokens,
        })
    }
    /// Explicitly copy model and optimizer state into a validated CPU checkpoint.
    pub fn checkpoint(&self) -> Result<Checkpoint> {
        self.engine.cuda.sync()?;
        let bytes = self
            .engine
            .parameters
            .read()?
            .iter()
            .chain(&self.engine.first_moment.read()?)
            .chain(&self.engine.second_moment.read()?)
            .flat_map(|v| v.to_le_bytes())
            .collect();
        Checkpoint::from_raw(self.config.clone(), self.completed, self.examples, bytes)
    }
}

/// Stateful native epoch shuffling, separate from model/optimizer checkpoints.
/// The caller owns dataset identity and restores its data position on resume.
pub struct EpochSampler {
    rng: Random,
    epochs: u64,
}
impl EpochSampler {
    /// Seed the same native `seed+10000` stream as the fixed-suite research sampler.
    pub fn new(seed: u32) -> Result<Self> {
        if seed > u32::MAX - 10000 {
            return Err("sampler seed overflow".into());
        }
        Ok(Self {
            rng: Random::new(seed + 10000),
            epochs: 0,
        })
    }
    /// Shuffle every example once, without reconstructing earlier epochs.
    pub fn next_epoch(&mut self, examples: usize) -> Result<Vec<usize>> {
        if examples == 0 || examples > u32::MAX as usize {
            return Err("invalid sampler size".into());
        }
        let next = self.epochs.checked_add(1).ok_or("epoch counter overflow")?;
        let mut order = (0..examples).collect::<Vec<_>>();
        self.rng.shuffle(&mut order);
        self.epochs = next;
        Ok(order)
    }
    /// Epochs whose permutations have been requested, including a partially consumed one.
    pub fn epochs(&self) -> u64 {
        self.epochs
    }
}
