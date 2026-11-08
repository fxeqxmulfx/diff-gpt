use crate::pipeline;
use kernite_cuda::{Buffer, Cuda, Grid, Ptr};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub fn shape(m: usize, n: usize, k: usize) -> (usize, usize, usize, usize) {
    let np = n.div_ceil(8) * 8;
    let kp = k.div_ceil(8) * 8;
    let tiles = m.div_ceil(64) * n.div_ceil(64);
    // Geometry-only policy, fixed before timing/training; never selects a dtype.
    let wanted = if k >= 512 {
        32usize.div_ceil(tiles).min(16)
    } else {
        1
    };
    let chunk = if wanted == 1 {
        kp
    } else {
        kp.div_ceil(wanted).div_ceil(32) * 32
    };
    (np, kp, chunk, kp.div_ceil(chunk))
}

pub struct Pipeline {
    kernels: pipeline::kernels::Kernels,
    a: Buffer<u16>,
    b: Buffer<u16>,
    partial: Buffer<f32>,
}
impl Pipeline {
    pub fn multiply_tf32(
        &self,
        kernels: &crate::tensor::kernels::Kernels,
        out: Ptr<f32>,
        a: Ptr<f32>,
        b: Ptr<f32>,
        dims: (usize, usize, usize),
    ) -> Result<()> {
        let (m, n, k) = dims;
        let (_, _, chunk, _) = shape(m, n, k);
        let splits = k.div_ceil(chunk);
        if splits * m * n > self.partial.len() {
            return Err("TF32 workspace capacity exceeded".into());
        }
        unsafe {
            kernels.gemm::<0>(
                Grid::new(
                    (m.div_ceil(crate::tensor::kernels::BM)
                        * n.div_ceil(crate::tensor::kernels::BN)
                        * splits) as u32,
                    crate::tensor::kernels::THREADS as u32,
                ),
                self.partial.ptr(),
                a,
                b,
                m as u32,
                n as u32,
                k as u32,
                chunk as u32,
            )?;
            self.kernels.finish(
                Grid::new((m * n).div_ceil(256) as u32, 256),
                out,
                self.partial.ptr(),
                m as u32,
                n as u32,
                n as u32,
                splits as u32,
            )?;
        }
        Ok(())
    }
    pub fn new(cuda: &Cuda, a: usize, b: usize, partial: usize) -> Result<Self> {
        Ok(Self {
            kernels: pipeline::kernels::Kernels::load(cuda)?,
            a: cuda.alloc(a)?,
            b: cuda.alloc(b)?,
            partial: cuda.alloc(partial)?,
        })
    }
    pub fn multiply<const BF16: bool>(
        &self,
        out: Ptr<f32>,
        a: Ptr<f32>,
        b: Ptr<f32>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        let (np, kp, chunk, splits) = shape(m, n, k);
        if m * kp > self.a.len() || kp * np > self.b.len() || splits * m * np > self.partial.len() {
            return Err("tensor workspace capacity exceeded".into());
        }
        let g = |count: usize| Grid::new(count.div_ceil(256) as u32, 256);
        // All buffers are disjoint and live across ordered launches in one CUDA context.
        unsafe {
            self.kernels.pack::<BF16>(
                g(m * kp),
                self.a.ptr(),
                a,
                m as u32,
                k as u32,
                m as u32,
                kp as u32,
            )?;
            self.kernels.pack::<BF16>(
                g(kp * np),
                self.b.ptr(),
                b,
                k as u32,
                n as u32,
                kp as u32,
                np as u32,
            )?;
            self.kernels.hgemm::<BF16>(
                Grid::new(
                    (m.div_ceil(pipeline::kernels::BM)
                        * np.div_ceil(pipeline::kernels::BN)
                        * splits) as u32,
                    pipeline::kernels::THREADS as u32,
                )
                .with_shared((pipeline::kernels::STAGES * pipeline::kernels::STAGE) as u32),
                self.partial.ptr(),
                self.a.ptr(),
                self.b.ptr(),
                m as u32,
                np as u32,
                kp as u32,
                chunk as u32,
            )?;
            self.kernels.finish(
                g(m * n),
                out,
                self.partial.ptr(),
                m as u32,
                n as u32,
                np as u32,
                splits as u32,
            )?;
        }
        Ok(())
    }
}
