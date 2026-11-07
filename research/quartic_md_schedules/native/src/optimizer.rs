//! Extend the frozen31d0a10 optimizer with native AdamMD, physical MDW and schedules.
use crate::{
    Result, adam_md_ops, base_ops,
    engine::Engine,
    gemm, layout, md_ops,
    schedule::{Family, Schedule},
};
use kernite_cuda::{Buffer, Cuda, Grid, Ptr};

pub use crate::schedule::BUDGET;
fn grid(n: usize) -> Grid {
    Grid::new(n.div_ceil(256) as u32, 256)
}
pub struct Ops {
    pub k: md_ops::kernels::Kernels,
    pub adam: adam_md_ops::kernels::Kernels,
    gemm: gemm::kernels::Kernels,
    base: base_ops::kernels::Kernels,
    parts: Buffer<f64>,
}
impl Ops {
    pub fn new(cuda: &Cuda, max_values: usize) -> Result<Self> {
        Ok(Self {
            k: md_ops::kernels::Kernels::load(cuda)?,
            adam: adam_md_ops::kernels::Kernels::load(cuda)?,
            gemm: gemm::kernels::Kernels::load(cuda)?,
            base: base_ops::kernels::Kernels::load(cuda)?,
            parts: cuda.alloc(max_values.div_ceil(256))?,
        })
    }
    pub fn norm(&self, x: &Buffer<f32>, out: &Buffer<f32>) -> Result<()> {
        let n = x.len().div_ceil(256);
        assert!(n <= self.parts.len());
        // SAFETY: allocated input/output lengths match these launch bounds. Same CUDA stream.
        unsafe {
            self.k.norm_partial(
                Grid::new(n as u32, 32),
                self.parts.ptr(),
                x.ptr(),
                x.len() as u32,
            )?;
            self.k
                .norm_finish(Grid::new(1, 32), out.ptr(), self.parts.ptr(), n as u32)?;
        }
        Ok(())
    }
    fn mm(
        &self,
        out: Ptr<f32>,
        a: Ptr<f32>,
        b: Ptr<f32>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        let g = Grid::new((m.div_ceil(128) * n.div_ceil(128)) as u32, 256);
        // SAFETY: all callers allocate dense matrices of the stated sizes, with aligned bases.
        unsafe {
            if n.is_multiple_of(4) && k.is_multiple_of(4) {
                self.gemm
                    .sgemm::<true>(g, out, a, b, m as u32, n as u32, k as u32)?;
            } else {
                self.gemm
                    .sgemm::<false>(g, out, a, b, m as u32, n as u32, k as u32)?;
            }
        }
        Ok(())
    }
}
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub offset: usize,
    pub radius: f32,
    pub momentum: Buffer<f32>,
    pub variance: Buffer<f32>,
    pub raw: Buffer<f32>,
    pub gains: Buffer<f32>,
    pub gm: Buffer<f32>,
    pub gv: Buffer<f32>,
    next: Buffer<f32>,
    direction: Buffer<f32>,
    pub nu: Buffer<f32>,
    signal: Buffer<f32>,
    pub x: Buffer<f32>,
    xt: Buffer<f32>,
    y: Buffer<f32>,
    a: Buffer<f32>,
    b: Buffer<f32>,
    pub candidate: Buffer<f32>,
    norm: Buffer<f32>,
}
impl Matrix {
    pub fn new(cuda: &Cuda, rows: usize, cols: usize, offset: usize, radius: f32) -> Result<Self> {
        let count = rows * cols;
        let gains = rows + cols;
        Ok(Self {
            rows,
            cols,
            offset,
            radius,
            momentum: cuda.alloc(2 * count)?,
            variance: cuda.alloc(2 * count)?,
            raw: cuda.upload(&vec![1f32.exp_m1().ln(); gains])?,
            gains: cuda.upload(&vec![1.; gains])?,
            gm: cuda.alloc(gains)?,
            gv: cuda.alloc(gains)?,
            next: cuda.alloc(gains)?,
            direction: cuda.alloc(2 * count)?,
            nu: cuda.alloc(2 * count)?,
            signal: cuda.alloc(count)?,
            x: cuda.alloc(4 * count)?,
            xt: cuda.alloc(4 * count)?,
            y: cuda.alloc(4 * count)?,
            a: cuda.alloc(4 * rows.min(cols).pow(2))?,
            b: cuda.alloc(4 * rows.min(cols).pow(2))?,
            candidate: cuda.alloc(2 * count)?,
            norm: cuda.alloc(1)?,
        })
    }
    /// Expanded real GEMM implements XX^H; transpose here is the real expansion of adjoint.
    pub fn polar(&self, ops: &Ops) -> Result<()> {
        let (r, c) = (self.rows.min(self.cols), self.rows.max(self.cols));
        let (rr, cc) = (2 * r, 2 * c);
        ops.norm(&self.nu, &self.norm)?;
        // SAFETY: real expansions have exactly4*rows*cols entries; Gram buffers have4*min².
        unsafe {
            if self.rows > self.cols {
                ops.k.pack::<true>(
                    grid(r * c),
                    self.x.ptr(),
                    self.nu.ptr(),
                    self.norm.ptr(),
                    r as u32,
                    c as u32,
                )?;
            } else {
                ops.k.pack::<false>(
                    grid(r * c),
                    self.x.ptr(),
                    self.nu.ptr(),
                    self.norm.ptr(),
                    r as u32,
                    c as u32,
                )?;
            }
            for _ in 0..5 {
                ops.base.transpose(
                    grid(rr * cc),
                    self.xt.ptr(),
                    self.x.ptr(),
                    rr as u32,
                    cc as u32,
                )?;
                ops.mm(self.a.ptr(), self.x.ptr(), self.xt.ptr(), rr, rr, cc)?;
                ops.mm(self.b.ptr(), self.a.ptr(), self.a.ptr(), rr, rr, rr)?;
                ops.k
                    .polynomial(grid(rr * rr), self.b.ptr(), self.a.ptr(), (rr * rr) as u32)?;
                ops.mm(self.y.ptr(), self.b.ptr(), self.x.ptr(), rr, cc, rr)?;
                ops.k
                    .ns_finish(grid(r * c), self.x.ptr(), self.y.ptr(), r as u32, c as u32)?;
            }
        }
        Ok(())
    }
    fn step(&mut self, ops: &Ops, update: &Update<'_>) -> Result<()> {
        let Update {
            engine: e,
            map,
            g,
            gradient_norm,
            completed,
            schedule,
        } = update;
        let (r, c) = (self.rows.min(self.cols), self.rows.max(self.cols));
        let count = self.rows * self.cols;
        let (bc1, bc2) = bias(*completed);
        let lr = schedule.rate(schedule.family.peak(), *completed);
        // SAFETY: constructor partitions the owner map into four nonoverlapping complex matrices.
        // Every factor gradient is prepared before the next gain buffer replaces the old one.
        unsafe {
            if schedule.family == Family::MuonMD {
                ops.k.prepare(
                    grid(count),
                    self.direction.ptr(),
                    self.momentum.ptr(),
                    self.nu.ptr(),
                    self.signal.ptr(),
                    e.parameters.ptr(),
                    g.ptr(),
                    map.ptr(),
                    self.gains.ptr(),
                    gradient_norm.ptr(),
                    self.offset as u32,
                    self.rows as u32,
                    self.cols as u32,
                )?;
            } else {
                ops.adam.prepare_adam(
                    grid(count),
                    self.candidate.ptr(),
                    self.momentum.ptr(),
                    self.variance.ptr(),
                    self.signal.ptr(),
                    e.parameters.ptr(),
                    g.ptr(),
                    map.ptr(),
                    self.gains.ptr(),
                    gradient_norm.ptr(),
                    self.offset as u32,
                    self.rows as u32,
                    self.cols as u32,
                    lr,
                    bc1,
                    bc2,
                )?;
            }
            ops.k.gains(
                Grid::new((self.rows + self.cols) as u32, 32),
                self.raw.ptr(),
                self.gm.ptr(),
                self.gv.ptr(),
                self.next.ptr(),
                self.gains.ptr(),
                self.signal.ptr(),
                self.rows as u32,
                self.cols as u32,
                lr,
                bc1,
                bc2,
            )?;
        }
        // A zero warmup rate advances histories but leaves physical weights and cached gains exact.
        if lr == 0. {
            return Ok(());
        }
        if schedule.family == Family::MuonMD {
            self.polar(ops)?;
            unsafe {
                let scaled = lr * self.radius / (r as f32).sqrt();
                if self.rows > self.cols {
                    ops.k.candidate::<true>(
                        grid(count),
                        self.candidate.ptr(),
                        self.direction.ptr(),
                        self.x.ptr(),
                        r as u32,
                        c as u32,
                        scaled,
                    )?;
                } else {
                    ops.k.candidate::<false>(
                        grid(count),
                        self.candidate.ptr(),
                        self.direction.ptr(),
                        self.x.ptr(),
                        r as u32,
                        c as u32,
                        scaled,
                    )?;
                }
            }
        }
        ops.norm(&self.candidate, &self.norm)?;
        if schedule.family == Family::AdamMDW {
            unsafe {
                ops.adam.fuse_decay(
                    grid(count),
                    e.parameters.ptr(),
                    self.candidate.ptr(),
                    self.norm.ptr(),
                    self.next.ptr(),
                    map.ptr(),
                    self.offset as u32,
                    self.rows as u32,
                    self.cols as u32,
                    self.radius,
                    lr * schedule.family.decay(),
                )?;
            }
            ops.norm(&self.candidate, &self.norm)?;
            unsafe {
                ops.adam.repair_rows(
                    grid(self.rows),
                    self.raw.ptr(),
                    self.next.ptr(),
                    self.norm.ptr(),
                    self.rows as u32,
                    self.radius,
                )?;
            }
        } else {
            unsafe {
                ops.k.fuse(
                    grid(count),
                    e.parameters.ptr(),
                    self.candidate.ptr(),
                    self.norm.ptr(),
                    self.next.ptr(),
                    map.ptr(),
                    self.offset as u32,
                    self.rows as u32,
                    self.cols as u32,
                    self.radius,
                )?;
            }
        }
        std::mem::swap(&mut self.gains, &mut self.next);
        Ok(())
    }
    fn buffers(&self) -> [&Buffer<f32>; 6] {
        [
            &self.momentum,
            &self.variance,
            &self.raw,
            &self.gains,
            &self.gm,
            &self.gv,
        ]
    }
}
fn bias(completed: usize) -> (f32, f32) {
    let t = (completed + 1) as f64;
    (
        (1. - 0.9f64.powf(t)) as f32,
        (1. - 0.99f64.powf(t)).sqrt() as f32,
    )
}
struct Update<'a> {
    engine: &'a Engine,
    map: &'a Buffer<u32>,
    g: &'a Buffer<f32>,
    gradient_norm: &'a Buffer<f32>,
    completed: usize,
    schedule: Schedule,
}
pub struct Optimizer {
    pub arm: String,
    pub schedule: Schedule,
    ops: Ops,
    map: Buffer<u32>,
    pub gradient: Buffer<f32>,
    norm: Buffer<f32>,
    pub matrices: Vec<Matrix>,
    embedding: usize,
}
impl Optimizer {
    pub fn new(e: &Engine, initial: &[f32], arm: &str) -> Result<Self> {
        let schedule = Schedule::parse(arm)?;
        layout::validate(&e.config, initial)?;
        let owners = layout::pairs(&e.config);
        let (w, h) = (e.config.width, layout::hidden(e.config.width, ""));
        let embedding = e.config.vocab * w / 2;
        let mut matrices = Vec::new();
        let mut offset = embedding;
        if schedule.family != Family::AdamW {
            for _ in 0..e.config.layers {
                for (rows, cols) in [
                    (3 * w / 2, w / 2),
                    (w / 2, w / 2),
                    (h, w / 2),
                    (w / 2, h / 2),
                ] {
                    let radius = owners[offset..offset + rows * cols]
                        .iter()
                        .map(|p| {
                            let (a, b) =
                                (initial[p[0] as usize] as f64, initial[p[1] as usize] as f64);
                            a * a + b * b
                        })
                        .sum::<f64>()
                        .sqrt() as f32;
                    matrices.push(Matrix::new(&e.cuda, rows, cols, offset, radius)?);
                    offset += rows * cols;
                }
            }
            assert_eq!(offset, owners.len());
        }
        Ok(Self {
            arm: arm.into(),
            schedule,
            ops: Ops::new(&e.cuda, 2 * owners.len())?,
            map: e.cuda.upload(owners.as_flattened())?,
            gradient: e.cuda.alloc(2 * owners.len())?,
            norm: e.cuda.alloc(1)?,
            matrices,
            embedding,
        })
    }
    pub fn step(&mut self, e: &Engine, completed: usize) -> Result<()> {
        let count = self.gradient.len() / 2;
        unsafe {
            self.ops.k.contract(
                grid(count),
                self.gradient.ptr(),
                e.gradients.ptr(),
                self.map.ptr(),
                count as u32,
            )?;
        }
        self.ops.norm(&self.gradient, &self.norm)?;
        let adam_count = if self.matrices.is_empty() {
            count
        } else {
            self.embedding
        };
        let (bc1, bc2) = bias(completed);
        unsafe {
            self.ops.k.adam_coordinates(
                grid(adam_count),
                e.parameters.ptr(),
                e.first_moment.ptr(),
                e.second_moment.ptr(),
                self.gradient.ptr(),
                self.map.ptr(),
                self.norm.ptr(),
                adam_count as u32,
                self.schedule.rate(0.002, completed),
                self.schedule.family.decay(),
                bc1,
                bc2,
            )?;
        }
        let update = Update {
            engine: e,
            map: &self.map,
            g: &self.gradient,
            gradient_norm: &self.norm,
            completed,
            schedule: self.schedule,
        };
        for matrix in &mut self.matrices {
            matrix.step(&self.ops, &update)?;
        }
        Ok(())
    }
    pub fn save(&self, e: &Engine) -> Result<Vec<f32>> {
        let mut data = Vec::new();
        for b in [&e.parameters, &e.first_moment, &e.second_moment] {
            data.extend(b.read()?);
        }
        for matrix in &self.matrices {
            for b in matrix.buffers() {
                data.extend(b.read()?);
            }
        }
        self.validate(e, &data)?;
        Ok(data)
    }
    fn validate(&self, e: &Engine, data: &[f32]) -> Result<()> {
        let n = e.parameters.len();
        let expected = 3 * n
            + self
                .matrices
                .iter()
                .flat_map(Matrix::buffers)
                .map(Buffer::len)
                .sum::<usize>();
        if data.len() != expected || !data.iter().all(|x| x.is_finite()) {
            return Err("optimizer state size/nonfinite error".into());
        }
        layout::validate_state(&e.config, &data[..n], &data[n..2 * n], &data[2 * n..3 * n])?;
        let owners = layout::pairs(&e.config);
        let mut at = 3 * n;
        for mat in &self.matrices {
            let ng = mat.rows + mat.cols;
            at += mat.momentum.len();
            if data[at..at + mat.variance.len()].iter().any(|x| *x < 0.) {
                return Err("negative direction variance".into());
            }
            at += mat.variance.len();
            let raw = &data[at..at + ng];
            at += ng;
            let gains = &data[at..at + ng];
            at += 2 * ng;
            let variance = &data[at..at + ng];
            at += ng;
            if gains.iter().any(|x| *x <= 0.) || variance.iter().any(|x| *x < 0.) {
                return Err("nonpositive gain or negative variance".into());
            }
            for (&a, &g) in raw.iter().zip(gains) {
                let softplus = (a as f64).max(0.) + (-(a as f64).abs()).exp().ln_1p();
                if (g as f64 - softplus).abs() > 2e-6 * softplus.max(1.) {
                    return Err("raw/materialized gain mismatch".into());
                }
            }
            let mut sq = 0.;
            for (i, pair) in owners[mat.offset..mat.offset + mat.rows * mat.cols]
                .iter()
                .enumerate()
            {
                let scale = gains[i / mat.cols] as f64 * gains[mat.rows + i % mat.cols] as f64;
                let a = data[pair[0] as usize] as f64 / scale;
                let b = data[pair[1] as usize] as f64 / scale;
                sq += a * a + b * b;
            }
            if sq > 0. && (sq.sqrt() / mat.radius as f64 - 1.).abs() > 3e-5 {
                return Err("direction sphere invariant failed".into());
            }
        }
        Ok(())
    }
    pub fn restore(&mut self, e: &mut Engine, data: &[f32]) -> Result<()> {
        self.validate(e, data)?;
        let mut at = 0;
        for b in [&mut e.parameters, &mut e.first_moment, &mut e.second_moment] {
            b.write(&data[at..at + b.len()])?;
            at += b.len();
        }
        for m in &mut self.matrices {
            for b in [
                &mut m.momentum,
                &mut m.variance,
                &mut m.raw,
                &mut m.gains,
                &mut m.gm,
                &mut m.gv,
            ] {
                b.write(&data[at..at + b.len()])?;
                at += b.len();
            }
        }
        Ok(())
    }
}
