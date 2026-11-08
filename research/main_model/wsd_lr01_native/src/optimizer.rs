//! Fixed current-main AdamW/WSD recipe. Matrix ties contract before clipping.
pub use crate::schedule::BUDGET;
use crate::{Result, engine::Engine, layout, optimizer_kernels::kernels, schedule::Schedule};
use kernite_cuda::{Buffer, Grid};

pub struct Optimizer {
    pub arm: String,
    schedule: Schedule,
    kernels: kernels::Kernels,
    map: Buffer<u32>,
    gradient: Buffer<f32>,
    parts: Buffer<f64>,
    norm: Buffer<f32>,
}
impl Optimizer {
    pub fn new(e: &Engine, initial: &[f32], arm: &str) -> Result<Self> {
        if arm != crate::schedule::ARM || e.config.architecture != "complex-linear" {
            return Err("only the declared main-model WSD arm is supported".into());
        }
        layout::validate(&e.config, initial)?;
        let owners = layout::pairs(&e.config);
        Ok(Self {
            arm: arm.into(),
            schedule: Schedule::parse(arm)?,
            kernels: kernels::Kernels::load(&e.cuda)?,
            map: e.cuda.upload(owners.as_flattened())?,
            gradient: e.cuda.alloc(2 * owners.len())?,
            parts: e.cuda.alloc((2 * owners.len()).div_ceil(256))?,
            norm: e.cuda.alloc(1)?,
        })
    }
    pub fn step(&mut self, e: &Engine, completed: usize) -> Result<()> {
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
                self.schedule.rate(0.01, completed),
                0.1,
                (1. - 0.9f64.powf(t)) as f32,
                (1. - 0.95f64.powf(t)).sqrt() as f32,
            )?;
        }
        Ok(())
    }
    fn validate(e: &Engine, data: &[f32]) -> Result<()> {
        let n = e.parameters.len();
        if data.len() != 3 * n || data.iter().any(|x| !x.is_finite()) {
            return Err("invalid optimizer state".into());
        }
        layout::validate_state(&e.config, &data[..n], &data[n..2 * n], &data[2 * n..])
    }
    pub fn save(&self, e: &Engine) -> Result<Vec<f32>> {
        let mut out = e.parameters.read()?;
        out.extend(e.first_moment.read()?);
        out.extend(e.second_moment.read()?);
        Self::validate(e, &out)?;
        Ok(out)
    }
    pub fn restore(&mut self, e: &mut Engine, data: &[f32]) -> Result<()> {
        Self::validate(e, data)?;
        let n = e.parameters.len();
        e.parameters.write(&data[..n])?;
        e.first_moment.write(&data[n..2 * n])?;
        e.second_moment.write(&data[2 * n..])?;
        Ok(())
    }
}
