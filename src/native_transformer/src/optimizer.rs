//! Fixed current-main AdamW/WSD recipe. Matrix ties contract before clipping.
use crate::{Result, engine::Engine, layout, optimizer_kernels::kernels, schedule::Schedule};
use kernite_cuda::{Buffer, Grid};

pub struct Optimizer {
    schedule: Schedule,
    kernels: kernels::Kernels,
    map: Buffer<u32>,
    gradient: Buffer<f32>,
    parts: Buffer<f32>,
    norm: Buffer<f32>,
    frequency: Option<FrequencyGroup>,
}
struct FrequencyGroup {
    map: Buffer<u32>,
    gradient: Buffer<f32>,
    parts: Buffer<f32>,
    norm: Buffer<f32>,
}
impl Optimizer {
    pub fn new(e: &Engine, budget: usize) -> Result<Self> {
        let schedule = Schedule::new(budget)?;
        if !(e.config.architecture == "complex-linear" || layout::is_mhc(&e.config))
            || e.config.lr != crate::schedule::PEAK
            || e.config.decay != 0.1
            || e.config.warmup != schedule.warmup
            || e.config.phase_optimizer
        {
            return Err("only the declared paired WSD recipe is supported".into());
        }
        layout::validate(&e.config, &e.parameters.read()?)?;
        let owners = Self::owners(&e.config)?;
        Ok(Self {
            schedule,
            kernels: kernels::Kernels::load(&e.cuda)?,
            map: e.cuda.upload(owners.as_flattened())?,
            gradient: e.cuda.alloc(2 * owners.len())?,
            parts: e.cuda.alloc((2 * owners.len()).div_ceil(256))?,
            norm: e.cuda.alloc(1)?,
            frequency: if layout::learned_rotary(&e.config) {
                let map = Self::frequency_owners(&e.config);
                Some(FrequencyGroup {
                    map: e.cuda.upload(map.as_flattened())?,
                    gradient: e.cuda.alloc(2 * map.len())?,
                    parts: e.cuda.alloc((2 * map.len()).div_ceil(256))?,
                    norm: e.cuda.alloc(1)?,
                })
            } else {
                None
            },
        })
    }
    pub fn owners(cfg: &crate::Config) -> Result<Vec<[u32; 5]>> {
        let mut owners = layout::pairs(cfg);
        let base = layout::base_count(cfg);
        let total = layout::frequency_offset(cfg);
        // kind=1 means two independent real coordinates, with no duplicate
        // writes or shared variance. Pairing may cross a router boundary.
        for at in (base..total).step_by(2) {
            if at + 1 < total {
                owners.push([at as u32, (at + 1) as u32, 0, 0, 1]);
            } else {
                // One real owner, with a zero scratch slot and no second write.
                owners.push([at as u32, at as u32, 0, 0, 2]);
            }
        }
        let owned: usize = owners.iter().map(|m| if m[4] == 2 { 1 } else { 2 }).sum();
        if owned + layout::frequency_count(cfg) != layout::independent_count(cfg) {
            return Err("independent-coordinate coverage mismatch".into());
        }
        Ok(owners)
    }
    pub fn frequency_owners(cfg: &crate::Config) -> Vec<[u32; 5]> {
        (layout::frequency_offset(cfg)..layout::total_count(cfg))
            .step_by(2)
            .map(|at| [at as u32, (at + 1) as u32, 0, 0, 1])
            .collect()
    }
    pub fn step(&mut self, e: &Engine, completed: usize) -> Result<()> {
        if e.inference_training_forbidden() {
            return Err(
                "optimizer updates are forbidden while inference-only YaRN is configured".into(),
            );
        }
        let count = self.gradient.len() / 2;
        let grid = Grid::new(count.div_ceil(256) as u32, 256);
        let t = (completed + 1) as f64;
        // SAFETY: map and independent coordinates use the validated model layout;
        // norm parts cover every coordinate. Operations share one CUDA stream.
        unsafe {
            self.kernels.contract(
                grid,
                self.gradient.ptr(),
                e.gradients.ptr(),
                self.map.ptr(),
                count as u32,
            )?;
            self.kernels.norm_partial(
                Grid::new(self.parts.len() as u32, 32),
                self.parts.ptr(),
                self.gradient.ptr(),
                self.gradient.len() as u32,
            )?;
            self.kernels.norm_finish(
                Grid::new(1, 32),
                self.norm.ptr(),
                self.parts.ptr(),
                self.parts.len() as u32,
            )?;
            self.kernels.adam_coordinates(
                grid,
                e.parameters.ptr(),
                e.first_moment.ptr(),
                e.second_moment.ptr(),
                self.gradient.ptr(),
                self.map.ptr(),
                self.norm.ptr(),
                count as u32,
                self.schedule.rate(completed),
                0.1,
                (1. - 0.9f64.powf(t)) as f32,
                (1. - 0.95f64.powf(t)).sqrt() as f32,
            )?;
            if let Some(frequency) = &self.frequency {
                let count = frequency.gradient.len() / 2;
                let grid = Grid::new(count.div_ceil(256) as u32, 256);
                self.kernels.contract(
                    grid,
                    frequency.gradient.ptr(),
                    e.gradients.ptr(),
                    frequency.map.ptr(),
                    count as u32,
                )?;
                self.kernels.norm_partial(
                    Grid::new(frequency.parts.len() as u32, 32),
                    frequency.parts.ptr(),
                    frequency.gradient.ptr(),
                    frequency.gradient.len() as u32,
                )?;
                self.kernels.norm_finish(
                    Grid::new(1, 32),
                    frequency.norm.ptr(),
                    frequency.parts.ptr(),
                    frequency.parts.len() as u32,
                )?;
                self.kernels.adam_coordinates(
                    grid,
                    e.parameters.ptr(),
                    e.first_moment.ptr(),
                    e.second_moment.ptr(),
                    frequency.gradient.ptr(),
                    frequency.map.ptr(),
                    frequency.norm.ptr(),
                    count as u32,
                    self.schedule.rate(completed),
                    0.,
                    (1. - 0.9f64.powf(t)) as f32,
                    (1. - 0.95f64.powf(t)).sqrt() as f32,
                )?;
            }
        }
        Ok(())
    }
    #[cfg(test)]
    fn validate(e: &Engine, data: &[f32]) -> Result<()> {
        let n = e.parameters.len();
        if data.len() != 3 * n || data.iter().any(|x| !x.is_finite()) {
            return Err("invalid optimizer state".into());
        }
        layout::validate_state(&e.config, &data[..n], &data[n..2 * n], &data[2 * n..])
    }
    #[cfg(test)]
    pub fn save(&self, e: &Engine) -> Result<Vec<f32>> {
        let mut out = e.parameters.read()?;
        out.extend(e.first_moment.read()?);
        out.extend(e.second_moment.read()?);
        Self::validate(e, &out)?;
        Ok(out)
    }
    #[cfg(test)]
    pub fn restore(&mut self, e: &mut Engine, data: &[f32]) -> Result<()> {
        Self::validate(e, data)?;
        let n = e.parameters.len();
        e.parameters.write(&data[..n])?;
        e.first_moment.write(&data[n..2 * n])?;
        e.second_moment.write(&data[2 * n..])?;
        Ok(())
    }
}
