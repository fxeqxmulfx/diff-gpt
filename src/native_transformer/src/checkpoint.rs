use crate::{MODEL_ID, ModelConfig, PRECISION, Result, hash, layout};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{BufReader, BufWriter, Read, Write},
    path::Path,
};
const MAGIC: &[u8; 8] = b"DGPKERN1";

/// Validated CPU snapshot of weights, both moments and optimizer progress.
pub struct Checkpoint {
    config: ModelConfig,
    completed: usize,
    examples: u64,
    state: Vec<u8>,
}
impl Checkpoint {
    /// Import research-compatible little-endian FP32 weights, first and second moments.
    /// The caller supplies the geometry and progress from its trusted provenance record.
    pub fn from_raw(
        config: ModelConfig,
        completed: usize,
        examples: u64,
        state: Vec<u8>,
    ) -> Result<Self> {
        let out = Self {
            config,
            completed,
            examples,
            state,
        };
        out.validate()?;
        Ok(out)
    }
    /// Configuration of the saved model.
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }
    /// Completed updates, used as the next WSD/AdamW step index on restoration.
    pub fn completed(&self) -> usize {
        self.completed
    }
    /// Number of training examples processed.
    pub fn examples(&self) -> u64 {
        self.examples
    }
    /// Exact legacy-compatible state bytes for parity checks or export.
    pub fn raw_state(&self) -> &[u8] {
        &self.state
    }
    /// SHA256 of the raw FP32 state, independent of checkpoint-container metadata.
    pub fn state_sha256(&self) -> String {
        hash(&self.state)
    }
    pub(crate) fn values(&self) -> Vec<f32> {
        self.state
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect()
    }
    /// Check dimensions, complex matrix ties, finite moments and progress on the CPU.
    pub fn validate(&self) -> Result<()> {
        let count = self.config.parameter_count()?;
        if self.completed > self.config.budget
            || self.state.len() != count * 12
            || self.completed == 0 && self.examples != 0
            || self.completed > 0 && self.examples < self.completed as u64
        {
            return Err("invalid checkpoint progress or state length".into());
        }
        let values = self.values();
        layout::validate_state(
            &self.config.native_unchecked(),
            &values[..count],
            &values[count..2 * count],
            &values[2 * count..],
        )
    }
    /// Atomic buffered save. All integrity fields and state are checked before writing.
    /// A rename publishes the complete file; no partially written checkpoint is exposed.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        self.validate()?;
        let path = path.as_ref();
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let name = path
            .file_name()
            .ok_or("checkpoint needs a file name")?
            .to_str()
            .ok_or("non-UTF8 checkpoint filename")?;
        let pending = parent.join(format!(".{name}.{}.pending", std::process::id()));
        let header = serde_json::to_vec(
            &json!({"schema":1,"architecture":MODEL_ID,"precision":PRECISION,"kernel_sha256":crate::kernel_sha256(),
            "config":self.config.to_json(),"completed":self.completed,"examples":self.examples,
            "state_bytes":self.state.len(),"state_sha256":self.state_sha256()}),
        )?;
        let mut digest = Sha256::new();
        digest.update(&header);
        digest.update(&self.state);
        let checksum = digest.finalize();
        let file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&pending)?;
        let result = (|| -> Result<()> {
            let mut file = BufWriter::with_capacity(512 * 1024, file);
            file.write_all(MAGIC)?;
            file.write_all(&(header.len() as u64).to_le_bytes())?;
            file.write_all(&header)?;
            file.write_all(&self.state)?;
            file.write_all(&checksum)?;
            file.flush()?;
            file.get_ref().sync_all()?;
            fs::rename(&pending, path)?;
            File::open(parent)?.sync_all()?;
            Ok(())
        })();
        if result.is_err() {
            let _ = fs::remove_file(pending);
        }
        result
    }
    /// Read and validate a versioned checkpoint without accessing CUDA.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        let file_size = file.metadata()?.len();
        let mut input = BufReader::new(file);
        let mut magic = [0; 8];
        input.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err("unknown checkpoint format".into());
        }
        let mut length = [0; 8];
        input.read_exact(&mut length)?;
        let header_len = usize::try_from(u64::from_le_bytes(length))?;
        if header_len == 0 || header_len > 16 * 1024 {
            return Err("invalid checkpoint header size".into());
        }
        let mut header = vec![0; header_len];
        input.read_exact(&mut header)?;
        let meta: Value = serde_json::from_slice(&header)?;
        if meta["schema"] != 1
            || meta["architecture"] != MODEL_ID
            || meta["precision"] != PRECISION
            || meta["kernel_sha256"] != crate::kernel_sha256()
        {
            return Err("checkpoint architecture, format or precision differs".into());
        }
        let config = ModelConfig::from_json(&meta["config"])?;
        let bytes = config
            .parameter_count()?
            .checked_mul(12)
            .ok_or("state size overflow")?;
        if meta["state_bytes"] != bytes || file_size != (16 + header_len + bytes + 32) as u64 {
            return Err("checkpoint is truncated, overlong or has wrong state geometry".into());
        }
        let mut state = vec![0; bytes];
        input.read_exact(&mut state)?;
        let mut checksum = [0; 32];
        input.read_exact(&mut checksum)?;
        let mut digest = Sha256::new();
        digest.update(header);
        digest.update(&state);
        if digest.finalize().as_slice() != checksum || meta["state_sha256"] != hash(&state) {
            return Err("checkpoint checksum mismatch".into());
        }
        Self::from_raw(
            config,
            meta["completed"]
                .as_u64()
                .ok_or("missing completed")?
                .try_into()?,
            meta["examples"].as_u64().ok_or("missing examples")?,
            state,
        )
    }
}

/// A save schedule based on examples seen, with no filesystem polling.
/// Save initialization, desired accuracy milestones, final state and explicit
/// stops separately. Periodic checkpoints must be at least 500 epochs apart.
pub struct CheckpointCadence {
    stride: u64,
    next: u64,
}
impl CheckpointCadence {
    /// Construct a cadence. `training_examples` is the actual train split size.
    pub fn new(training_examples: usize, every_epochs: usize) -> Result<Self> {
        if training_examples == 0 || every_epochs < 500 {
            return Err("checkpoint cadence requires a train set and at least500epochs".into());
        }
        let stride = (training_examples as u64)
            .checked_mul(every_epochs as u64)
            .ok_or("checkpoint cadence overflow")?;
        Ok(Self {
            stride,
            next: stride,
        })
    }
    /// Whether a completed update has reached the next periodic save boundary.
    pub fn due(&self, examples: u64) -> bool {
        examples >= self.next
    }
    /// Advance only after a successful save; supports resumed example counters.
    pub fn saved(&mut self, examples: u64) {
        self.next = (examples / self.stride)
            .saturating_add(1)
            .saturating_mul(self.stride);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn checkpoint_rejects_corruption_and_cadence_uses_actual_epochs() -> Result<()> {
        let cfg = ModelConfig::main(11, 3, 4, 20_000);
        let mut values = layout::initialize_model(&cfg.native_unchecked(), 1);
        values.resize(values.len() * 3, 0.);
        let cp = Checkpoint::from_raw(
            cfg,
            0,
            0,
            values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        )?;
        let dir =
            std::env::temp_dir().join(format!("gpt-kernite-checkpoint-{}", std::process::id()));
        fs::create_dir(&dir)?;
        let file = dir.join("model.ckpt");
        cp.save(&file)?;
        assert_eq!(Checkpoint::load(&file)?.raw_state(), cp.raw_state());
        let mut corrupt = fs::read(&file)?;
        let i = corrupt.len() - 40;
        corrupt[i] ^= 1;
        fs::write(&file, corrupt)?;
        assert!(Checkpoint::load(&file).is_err());
        let mut cadence = CheckpointCadence::new(4704, 500)?;
        assert!(!cadence.due(4704 * 500 - 1));
        assert!(cadence.due(4704 * 500));
        cadence.saved(4704 * 501);
        assert!(!cadence.due(4704 * 999));
        assert!(cadence.due(4704 * 1000));
        assert!(CheckpointCadence::new(4704, 499).is_err());
        fs::remove_dir_all(dir)?;
        Ok(())
    }
}
