//! Host orchestration over preallocated GPU buffers. No Python, libtorch, autograd engine,
//! model-side per-update allocation, or host readback in a training update.

use crate::{attention, base_ops as ops, complex, gemm, layout, sequence_ops};
use kernite_cuda::{Buffer, Cuda, Grid, Ptr};
use std::error::Error;

pub type Result<T> = std::result::Result<T, Box<dyn Error>>;
type F = Buffer<f32>;
type P = Ptr<f32>;

#[derive(Clone)]
pub struct Config {
    pub width: usize,
    pub vocab: usize,
    pub heads: usize,
    pub max_batch: usize,
    pub max_length: usize,
    pub layers: usize,
    pub normalized: bool,
    pub ssmax: bool,
    pub gain: f32,
    pub lr: f64,
    pub decay: f32,
    pub warmup: usize,
    pub architecture: String,
    pub phase_optimizer: bool,
}

struct Layer {
    offset: usize,
    n1: F,
    qkv: F,
    qnorm: F,
    qkr: F,
    dqnorm: F,
    dqkr: F,
    prob: F,
    ds: F,
    delta: F,
    gd: F,
    dv: F,
    att: F,
    ap: F,
    an: F,
    mid: F,
    n2: F,
    fc: F,
    hidden: F,
    fp: F,
    fnorm: F,
    out: F,
    dn1: F,
    dqkv: F,
    datt: F,
    dap: F,
    dmid: F,
    dn2: F,
    dfc: F,
    dhidden: F,
    dfp: F,
    dx: F,
    branch: F,
    flags: Buffer<u32>,
    head_ok: Buffer<u32>,
}

impl Layer {
    fn new(cuda: &Cuda, cfg: &Config, hidden: usize, offset: usize) -> Result<Self> {
        let w = cfg.width;
        let r = cfg.max_batch * cfg.max_length;
        let scores = cfg.max_batch * cfg.heads * cfg.max_length * cfg.max_length;
        Ok(Self {
            offset,
            n1: cuda.alloc(r * w)?,
            qkv: cuda.alloc(r * 3 * w)?,
            qnorm: cuda.alloc(r * 3 * w)?,
            qkr: cuda.alloc(r * 3 * w)?,
            dqnorm: cuda.alloc(r * 3 * w)?,
            dqkr: cuda.alloc(r * 3 * w)?,
            prob: cuda.alloc(scores)?,
            ds: cuda.alloc(scores)?,
            delta: cuda.alloc(r * w)?,
            gd: cuda.alloc(r * w)?,
            dv: cuda.alloc(r * w)?,
            att: cuda.alloc(r * w)?,
            ap: cuda.alloc(r * w)?,
            an: cuda.alloc(r * w)?,
            mid: cuda.alloc(r * w)?,
            n2: cuda.alloc(r * w)?,
            fc: cuda.alloc(r * 2 * hidden)?,
            hidden: cuda.alloc(r * hidden)?,
            fp: cuda.alloc(r * w)?,
            fnorm: cuda.alloc(r * w)?,
            out: cuda.alloc(r * w)?,
            dn1: cuda.alloc(r * w)?,
            dqkv: cuda.alloc(r * 3 * w)?,
            datt: cuda.alloc(r * w)?,
            dap: cuda.alloc(r * w)?,
            dmid: cuda.alloc(r * w)?,
            dn2: cuda.alloc(r * w)?,
            dfc: cuda.alloc(r * 2 * hidden)?,
            dhidden: cuda.alloc(r * hidden)?,
            dfp: cuda.alloc(r * w)?,
            dx: cuda.alloc(r * w)?,
            branch: cuda.alloc(r * w)?,
            flags: cuda.alloc(r)?,
            head_ok: cuda.alloc(cfg.max_batch * cfg.heads)?,
        })
    }
}

pub struct Engine {
    pub cuda: Cuda,
    pub config: Config,
    ops: ops::kernels::Kernels,
    att: attention::kernels::Kernels,
    seq: sequence_ops::kernels::Kernels,
    stable_loss: Option<crate::stable_loss::kernels::Kernels>,
    gemm: gemm::kernels::Kernels,
    complex: complex::kernels::Kernels,
    owners: Buffer<u32>,
    owner_count: usize,
    pub parameters: F,
    pub gradients: F,
    pub first_moment: F,
    pub second_moment: F,
    rope: F,
    tokens: Buffer<u32>,
    labels: Buffer<u32>,
    flags: Buffer<u32>,
    embed: F,
    embed_norm: F,
    shared: F,
    last: F,
    dlast: F,
    dshared: F,
    dx: F,
    dembed: F,
    pub logits: F,
    pub dlogits: F,
    losses: F,
    pub predictions: Buffer<u32>,
    scratch: F,
    layers: Vec<Layer>,
    host_tokens: Vec<u32>,
    host_labels: Vec<u32>,
    host_rope: Vec<f32>,
    rope_length: usize,
}

fn grid(count: usize) -> Grid {
    Grid::new(count.div_ceil(256) as u32, 256)
}
fn rows_grid(rows: usize) -> Grid {
    Grid::new(rows.div_ceil(4) as u32, 128)
}

impl Engine {
    pub fn new(config: Config, parameters: &[f32]) -> Result<Self> {
        let w = config.width;
        if ![64, 128, 256].contains(&w)
            || config.layers == 0
            || config.heads == 0
            || !w.is_multiple_of(config.heads)
            || ![16, 32, 128].contains(&(w / config.heads))
            || config.max_batch == 0
            || config.max_length == 0
        {
            return Err("unsupported model shape".into());
        }
        if !config.normalized
            || !config.ssmax
            || (config.gain - (config.heads as f32 / w as f32)).abs() > f32::EPSILON
            || ![
                "real",
                "real-product",
                "complex-linear",
                "complex-hermitian",
            ]
            .contains(&config.architecture.as_str())
        {
            return Err(
                "sequence backend requires the frozen Peri-RMS, SSMax 1/d architecture".into(),
            );
        }
        let r = config.max_batch * config.max_length;
        let hidden = layout::hidden(w, &config.architecture);
        let per_layer = 4 * w * w + 3 * w * hidden;
        if parameters.len() != config.vocab * w + config.layers * per_layer {
            return Err("parameter layout does not match the native model".into());
        }
        layout::validate(&config, parameters)?;
        if config.phase_optimizer && !layout::is_complex(&config.architecture) {
            return Err("phase AdamW requires independent complex parameters".into());
        }
        let cuda = Cuda::new()?;
        let ops = ops::kernels::Kernels::load(&cuda)?;
        let att = attention::kernels::Kernels::load(&cuda)?;
        let gemm = gemm::kernels::Kernels::load(&cuda)?;
        let complex = complex::kernels::Kernels::load(&cuda)?;
        let owners = if layout::is_complex(&config.architecture) {
            layout::pairs(&config)
        } else {
            vec![]
        };
        let owner_count = owners.len();
        let owner_data: Vec<u32> = owners.into_iter().flatten().collect();
        let mut layers = Vec::new();
        for l in 0..config.layers {
            layers.push(Layer::new(
                &cuda,
                &config,
                hidden,
                config.vocab * w + l * per_layer,
            )?);
        }
        Ok(Self {
            ops,
            seq: sequence_ops::kernels::Kernels::load(&cuda)?,
            stable_loss: None,
            att,
            gemm,
            complex,
            owners: if owner_data.is_empty() {
                cuda.alloc(1)?
            } else {
                cuda.upload(&owner_data)?
            },
            owner_count,
            parameters: cuda.upload(parameters)?,
            gradients: cuda.alloc(parameters.len())?,
            first_moment: cuda.alloc(parameters.len())?,
            second_moment: cuda.alloc(parameters.len())?,
            rope: cuda.alloc(config.max_length * (w / config.heads))?,
            tokens: cuda.alloc(r)?,
            labels: cuda.alloc(r)?,
            flags: cuda.alloc(r)?,
            embed: cuda.alloc(r * w)?,
            embed_norm: cuda.alloc(r * w)?,
            shared: cuda.alloc(r * w)?,
            last: cuda.alloc(r * w)?,
            dlast: cuda.alloc(r * w)?,
            dshared: cuda.alloc(r * w)?,
            dx: cuda.alloc(r * w)?,
            dembed: cuda.alloc(r * w)?,
            logits: cuda.alloc(r * config.vocab)?,
            dlogits: cuda.alloc(r * config.vocab)?,
            losses: cuda.alloc(r)?,
            predictions: cuda.alloc(r)?,
            scratch: cuda.alloc(
                (r * 2 * hidden)
                    .max(2 * hidden * w)
                    .max(3 * w * w)
                    .max(r * config.vocab)
                    .max(config.vocab * w),
            )?,
            layers,
            host_tokens: vec![0; r],
            host_labels: vec![u32::MAX; r],
            host_rope: vec![0.; config.max_length * (w / config.heads)],
            rope_length: 0,
            cuda,
            config,
        })
    }

    // SAFETY for kernel calls below: construction checks shape/parameter layout and allocates
    // every buffer at the configured maximum batch and length. Entry points enforce those bounds. Kernels
    // all launch in order on the same CUDA context's default stream; these buffers outlive
    // every launch. Each operation below is assigned disjoint input/output ranges, except
    // the documented elementwise accumulation into its own element.
    fn mm(&self, out: P, a: P, b: P, m: usize, n: usize, k: usize) -> Result<()> {
        let g = Grid::new(
            (m.div_ceil(gemm::kernels::BM) * n.div_ceil(gemm::kernels::BN)) as u32,
            gemm::kernels::THREADS as u32,
        );
        unsafe {
            if k.is_multiple_of(4) && n.is_multiple_of(4) {
                self.gemm
                    .sgemm::<true>(g, out, a, b, m as u32, n as u32, k as u32)?;
            } else {
                self.gemm
                    .sgemm::<false>(g, out, a, b, m as u32, n as u32, k as u32)?;
            }
        }
        Ok(())
    }

    fn transpose(&self, x: P, rows: usize, cols: usize) -> Result<()> {
        unsafe {
            self.ops.transpose(
                grid(rows * cols),
                self.scratch.ptr(),
                x,
                rows as u32,
                cols as u32,
            )?;
        }
        Ok(())
    }

    fn linear(&self, out: P, x: P, offset: usize, rows: usize, n: usize, k: usize) -> Result<()> {
        self.transpose(self.parameters.at(offset), n, k)?;
        self.mm(out, x, self.scratch.ptr(), rows, n, k)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "explicit dimensions and pointers mirror Linear backward"
    )]
    fn linear_back(
        &self,
        dx: P,
        dy: P,
        x: P,
        offset: usize,
        rows: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        self.mm(dx, dy, self.parameters.at(offset), rows, k, n)?;
        self.transpose(dy, rows, n)?;
        self.mm(self.gradients.at(offset), self.scratch.ptr(), x, n, k, rows)
    }

    fn rms(&self, y: P, x: P, rows: usize) -> Result<()> {
        unsafe {
            if self.config.width == 64 {
                self.ops.rms::<64>(rows_grid(rows), y, x, rows as u32)?;
            } else if self.config.width == 128 {
                self.ops.rms::<128>(rows_grid(rows), y, x, rows as u32)?;
            } else {
                self.ops.rms::<256>(rows_grid(rows), y, x, rows as u32)?;
            }
        }
        Ok(())
    }

    fn rms_back(&self, dx: P, x: P, dy: P, rows: usize) -> Result<()> {
        unsafe {
            if self.config.width == 64 {
                self.ops
                    .rms_bwd::<64>(rows_grid(rows), dx, x, dy, rows as u32)?;
            } else if self.config.width == 128 {
                self.ops
                    .rms_bwd::<128>(rows_grid(rows), dx, x, dy, rows as u32)?;
            } else {
                self.ops
                    .rms_bwd::<256>(rows_grid(rows), dx, x, dy, rows as u32)?;
            }
        }
        Ok(())
    }

    fn scale(&self) -> f32 {
        if self.config.normalized {
            (2.0 * self.config.layers as f64).sqrt().recip() as f32
        } else {
            1.0
        }
    }
    fn attention_forward(&self, layer: &Layer, batch: usize, length: usize) -> Result<()> {
        let w = self.config.width;
        let h = self.config.heads;
        let r = batch * length;
        macro_rules! dispatch {
            ($d:literal) => {
                // SAFETY: layer buffers cover the maximum batch/length; all launches
                // use exact head grids and execute on the same CUDA stream.
                unsafe {
                    self.att.norm::<$d>(
                        Grid::new((r * 3 * h) as u32, 32),
                        layer.qnorm.ptr(),
                        layer.qkv.ptr(),
                        h as u32,
                    )?;
                    self.att.rotate::<false>(
                        grid(r * 3 * w),
                        layer.qkr.ptr(),
                        layer.qnorm.ptr(),
                        self.rope.ptr(),
                        $d,
                        h as u32,
                        length as u32,
                        (r * 3 * w) as u32,
                    )?;
                    self.att.forward::<$d>(
                        Grid::new((r * h) as u32, 32),
                        layer.att.ptr(),
                        layer.prob.ptr(),
                        layer.delta.ptr(),
                        layer.head_ok.ptr(),
                        layer.qkr.ptr(),
                        h as u32,
                        length as u32,
                        u32::from(self.config.architecture == "complex-hermitian"),
                    )?;
                }
            };
        }
        match w / h {
            16 => dispatch!(16),
            32 => dispatch!(32),
            128 => dispatch!(128),
            _ => return Err("unsupported head width".into()),
        };
        Ok(())
    }
    fn attention_backward(&self, layer: &Layer, batch: usize, length: usize) -> Result<()> {
        let w = self.config.width;
        let h = self.config.heads;
        let r = batch * length;
        macro_rules! dispatch {
            ($d:literal) => {
                // SAFETY: every head owns its gradient rows; kernels perform complete
                // deterministic reductions without atomic accumulation.
                unsafe {
                    self.att.query_backward::<$d>(
                        Grid::new((r * h) as u32, 32),
                        layer.dqkr.ptr(),
                        layer.ds.ptr(),
                        layer.gd.ptr(),
                        layer.dv.ptr(),
                        layer.datt.ptr(),
                        layer.delta.ptr(),
                        layer.prob.ptr(),
                        layer.qkr.ptr(),
                        h as u32,
                        length as u32,
                        u32::from(self.config.architecture == "complex-hermitian"),
                    )?;
                    self.att.key_backward::<$d>(
                        Grid::new((r * h) as u32, 32),
                        layer.dqkr.ptr(),
                        layer.ds.ptr(),
                        layer.gd.ptr(),
                        layer.dv.ptr(),
                        layer.prob.ptr(),
                        layer.qkr.ptr(),
                        h as u32,
                        length as u32,
                    )?;
                    self.att.rotate::<true>(
                        grid(r * 3 * w),
                        layer.dqnorm.ptr(),
                        layer.dqkr.ptr(),
                        self.rope.ptr(),
                        $d,
                        h as u32,
                        length as u32,
                        (r * 3 * w) as u32,
                    )?;
                    self.att.norm_backward::<$d>(
                        Grid::new((r * 3 * h) as u32, 32),
                        layer.dqkv.ptr(),
                        layer.qkv.ptr(),
                        layer.dqnorm.ptr(),
                        h as u32,
                    )?;
                }
            };
        }
        match w / h {
            16 => dispatch!(16),
            32 => dispatch!(32),
            128 => dispatch!(128),
            _ => return Err("unsupported head width".into()),
        };
        Ok(())
    }
    /// Opt in to the corrected GPU loss; existing checkpoint replay defaults
    /// to the frozen original loss until this is explicitly called.
    pub fn enable_stable_loss(&mut self) -> Result<()> {
        if self.stable_loss.is_none() {
            self.stable_loss = Some(crate::stable_loss::kernels::Kernels::load(&self.cuda)?);
        }
        Ok(())
    }

    pub fn forward(
        &mut self,
        tokens: &[u32],
        labels: &[u32],
        batch: usize,
        length: usize,
    ) -> Result<()> {
        if batch == 0
            || batch > self.config.max_batch
            || length == 0
            || length > self.config.max_length
            || tokens.len() != batch * length
            || labels.len() != tokens.len()
            || tokens.iter().any(|&t| t as usize >= self.config.vocab)
        {
            return Err("invalid sequence batch".into());
        }
        let r = batch * length;
        let w = self.config.width;
        let h = layout::hidden(w, "");
        let count = (r * w) as u32;
        self.host_tokens[..r].copy_from_slice(tokens);
        self.tokens.write(&self.host_tokens)?;
        self.host_labels[..r].copy_from_slice(labels);
        self.labels.write(&self.host_labels)?;
        if self.rope_length != length {
            let d = w / self.config.heads;
            for sine in 0..2 {
                for pos in 0..length {
                    for c in 0..d / 2 {
                        let angle = pos as f32 / 10000f32.powf(2. * c as f32 / d as f32);
                        self.host_rope[sine * length * d / 2 + pos * d / 2 + c] =
                            if sine == 1 { angle.sin() } else { angle.cos() };
                    }
                }
            }
            self.rope.write(&self.host_rope)?;
            self.rope_length = length;
        }
        unsafe {
            self.seq.prefix(
                grid(r),
                self.flags.ptr(),
                self.tokens.ptr(),
                length as u32,
                r as u32,
            )?;
            self.ops.embed(
                grid(r * w),
                self.embed.ptr(),
                self.parameters.ptr(),
                self.tokens.ptr(),
                w as u32,
                count,
            )?;
        }
        let mut x = self.embed.ptr();
        if self.config.normalized {
            self.rms(self.embed_norm.ptr(), x, r)?;
            x = self.embed_norm.ptr();
        }
        let mut flags = self.flags.ptr();
        for layer in &self.layers {
            let o = layer.offset;
            self.rms(layer.n1.ptr(), x, r)?;
            self.linear(layer.qkv.ptr(), layer.n1.ptr(), o, r, 3 * w, w)?;
            self.attention_forward(layer, batch, length)?;
            unsafe {
                self.seq.guard(
                    grid(r),
                    layer.flags.ptr(),
                    flags,
                    layer.head_ok.ptr(),
                    length as u32,
                    self.config.heads as u32,
                    r as u32,
                )?;
            }
            flags = layer.flags.ptr();
            self.linear(layer.ap.ptr(), layer.att.ptr(), o + 3 * w * w, r, w, w)?;
            let mut branch = layer.ap.ptr();
            if self.config.normalized {
                self.rms(layer.an.ptr(), branch, r)?;
                branch = layer.an.ptr();
            }
            unsafe {
                self.seq.residual::<true>(
                    grid(r * w),
                    layer.mid.ptr(),
                    x,
                    branch,
                    flags,
                    w as u32,
                    length as u32,
                    count,
                    self.scale(),
                )?;
            }
            self.rms(layer.n2.ptr(), layer.mid.ptr(), r)?;
            self.linear(layer.fc.ptr(), layer.n2.ptr(), o + 4 * w * w, r, 2 * h, w)?;
            unsafe {
                if self.config.architecture != "real" {
                    self.complex.product::<true>(
                        grid(r * h / 2),
                        layer.hidden.ptr(),
                        layer.fc.ptr(),
                        (h / 2) as u32,
                        (r * h / 2) as u32,
                    )?;
                } else {
                    self.ops.swiglu(
                        grid(r * h),
                        layer.hidden.ptr(),
                        layer.fc.ptr(),
                        h as u32,
                        (r * h) as u32,
                    )?;
                }
            }
            self.linear(
                layer.fp.ptr(),
                layer.hidden.ptr(),
                o + 4 * w * w + 2 * h * w,
                r,
                w,
                h,
            )?;
            branch = layer.fp.ptr();
            if self.config.normalized {
                self.rms(layer.fnorm.ptr(), branch, r)?;
                branch = layer.fnorm.ptr();
            }
            unsafe {
                self.seq.residual::<false>(
                    grid(r * w),
                    layer.out.ptr(),
                    layer.mid.ptr(),
                    branch,
                    flags,
                    w as u32,
                    length as u32,
                    count,
                    self.scale(),
                )?;
            }
            x = layer.out.ptr();
        }
        unsafe {
            self.seq.share::<false>(
                grid(r * w),
                self.shared.ptr(),
                x,
                flags,
                w as u32,
                length as u32,
                count,
            )?;
        }
        self.rms(self.last.ptr(), self.shared.ptr(), r)?;
        self.linear(
            self.logits.ptr(),
            self.last.ptr(),
            0,
            r,
            self.config.vocab,
            w,
        )?;
        let normalizer = labels
            .iter()
            .filter(|&&v| (v as usize) < self.config.vocab)
            .count()
            .max(1) as f32;
        unsafe {
            if let Some(stable) = &self.stable_loss {
                stable.loss(
                    rows_grid(r),
                    self.dlogits.ptr(),
                    self.losses.ptr(),
                    self.predictions.ptr(),
                    self.logits.ptr(),
                    self.labels.ptr(),
                    self.config.vocab as u32,
                    r as u32,
                    normalizer,
                )?;
            } else {
                self.seq.loss(
                    rows_grid(r),
                    self.dlogits.ptr(),
                    self.losses.ptr(),
                    self.predictions.ptr(),
                    self.logits.ptr(),
                    self.labels.ptr(),
                    self.config.vocab as u32,
                    r as u32,
                    normalizer,
                )?;
            }
        }
        Ok(())
    }

    pub fn backward(&self, batch: usize, length: usize) -> Result<()> {
        if batch == 0
            || batch > self.config.max_batch
            || length == 0
            || length > self.config.max_length
        {
            return Err("batch or length outside allocated bounds".into());
        }
        let (w, r) = (self.config.width, batch * length);
        let h = layout::hidden(w, &self.config.architecture);
        let count = (r * w) as u32;
        self.linear_back(
            self.dlast.ptr(),
            self.dlogits.ptr(),
            self.last.ptr(),
            0,
            r,
            self.config.vocab,
            w,
        )?;
        self.rms_back(self.dshared.ptr(), self.shared.ptr(), self.dlast.ptr(), r)?;
        unsafe {
            self.seq.share::<true>(
                grid(r * w),
                self.dx.ptr(),
                self.dshared.ptr(),
                self.layers.last().unwrap().flags.ptr(),
                w as u32,
                length as u32,
                count,
            )?;
        }
        let mut incoming = self.dx.ptr();
        for (i, layer) in self.layers.iter().enumerate().rev() {
            let o = layer.offset;
            let x = if i > 0 {
                self.layers[i - 1].out.ptr()
            } else if self.config.normalized {
                self.embed_norm.ptr()
            } else {
                self.embed.ptr()
            };
            unsafe {
                self.seq.share::<true>(
                    grid(r * w),
                    layer.branch.ptr(),
                    incoming,
                    layer.flags.ptr(),
                    w as u32,
                    length as u32,
                    count,
                )?;
                self.seq.scale(
                    grid(r * w),
                    layer.branch.ptr(),
                    layer.branch.ptr(),
                    count,
                    self.scale(),
                )?;
            }
            let mut grad = layer.branch.ptr();
            if self.config.normalized {
                self.rms_back(layer.dfp.ptr(), layer.fp.ptr(), grad, r)?;
                grad = layer.dfp.ptr();
            }
            self.linear_back(
                layer.dhidden.ptr(),
                grad,
                layer.hidden.ptr(),
                o + 4 * w * w + 2 * h * w,
                r,
                w,
                h,
            )?;
            unsafe {
                if self.config.architecture != "real" {
                    self.complex.product_bwd::<true>(
                        grid(r * h / 2),
                        layer.dfc.ptr(),
                        layer.fc.ptr(),
                        layer.dhidden.ptr(),
                        (h / 2) as u32,
                        (r * h / 2) as u32,
                    )?;
                } else {
                    self.ops.swiglu_bwd(
                        grid(r * h),
                        layer.dfc.ptr(),
                        layer.fc.ptr(),
                        layer.dhidden.ptr(),
                        h as u32,
                        (r * h) as u32,
                    )?;
                }
            }
            self.linear_back(
                layer.dn2.ptr(),
                layer.dfc.ptr(),
                layer.n2.ptr(),
                o + 4 * w * w,
                r,
                2 * h,
                w,
            )?;
            self.rms_back(layer.dmid.ptr(), layer.mid.ptr(), layer.dn2.ptr(), r)?;
            unsafe {
                self.ops.add(
                    grid(r * w),
                    layer.dmid.ptr(),
                    incoming,
                    layer.dmid.ptr(),
                    count,
                )?;
                self.seq.attention_grad(
                    grid(r * w),
                    layer.branch.ptr(),
                    layer.dmid.ptr(),
                    layer.flags.ptr(),
                    w as u32,
                    length as u32,
                    count,
                    self.scale(),
                )?;
            }
            grad = layer.branch.ptr();
            if self.config.normalized {
                self.rms_back(layer.dap.ptr(), layer.ap.ptr(), grad, r)?;
                grad = layer.dap.ptr();
            }
            self.linear_back(
                layer.datt.ptr(),
                grad,
                layer.att.ptr(),
                o + 3 * w * w,
                r,
                w,
                w,
            )?;
            self.attention_backward(layer, batch, length)?;
            self.linear_back(
                layer.dn1.ptr(),
                layer.dqkv.ptr(),
                layer.n1.ptr(),
                o,
                r,
                3 * w,
                w,
            )?;
            self.rms_back(layer.dx.ptr(), x, layer.dn1.ptr(), r)?;
            unsafe {
                self.ops.add(
                    grid(r * w),
                    layer.dx.ptr(),
                    layer.dx.ptr(),
                    layer.dmid.ptr(),
                    count,
                )?;
            }
            incoming = layer.dx.ptr();
        }
        if self.config.normalized {
            self.rms_back(self.dembed.ptr(), self.embed.ptr(), incoming, r)?;
            incoming = self.dembed.ptr();
        }
        unsafe {
            self.ops.embed_bwd(
                grid(self.config.vocab * w),
                self.gradients.ptr(),
                incoming,
                self.tokens.ptr(),
                w as u32,
                self.config.vocab as u32,
                r as u32,
            )?;
        }
        Ok(())
    }

    pub fn update(&self, completed: usize) -> Result<()> {
        let lr = self.config.lr
            * if self.config.warmup > 0 {
                (completed as f64 / self.config.warmup as f64).min(1.0)
            } else {
                1.0
            };
        let step = (completed + 1) as f64;
        unsafe {
            if self.owner_count > 0 {
                // Contract the expanded real gradient BEFORE updating moments.
                // Updating four real blocks independently would break C-linearity.
                macro_rules! update_complex {
                    ($phase:literal) => {
                        self.complex.adamw::<$phase>(
                            grid(self.owner_count),
                            self.parameters.ptr(),
                            self.first_moment.ptr(),
                            self.second_moment.ptr(),
                            self.gradients.ptr(),
                            self.owners.ptr(),
                            self.owner_count as u32,
                            lr as f32,
                            self.config.decay,
                            (1.0 - 0.9f64.powf(step)) as f32,
                            (1.0 - 0.98f64.powf(step)).sqrt() as f32,
                        )?
                    };
                }
                if self.config.phase_optimizer {
                    update_complex!(true);
                } else {
                    update_complex!(false);
                }
                return Ok(());
            }
            self.ops.adamw(
                grid(self.parameters.len()),
                self.parameters.ptr(),
                self.first_moment.ptr(),
                self.second_moment.ptr(),
                self.gradients.ptr(),
                self.parameters.len() as u32,
                lr as f32,
                self.config.decay,
                (1.0 - 0.9f64.powf(step)) as f32,
                (1.0 - 0.98f64.powf(step)).sqrt() as f32,
            )?;
        }
        Ok(())
    }

    pub fn losses(&self, rows: usize) -> Result<Vec<f32>> {
        Ok(self.losses.read()?[..rows].to_vec())
    }

    /// Read-only host copies after a diagnostic forward/backward (two-token scope).
    pub fn diagnostic_snapshot(
        &self,
        rows: usize,
    ) -> Result<std::collections::BTreeMap<String, Vec<f32>>> {
        let mut out = std::collections::BTreeMap::new();
        let w = self.config.width;
        let h = layout::hidden(w, "");
        for (name, buffer) in [
            ("embedding", &self.embed),
            ("embedding_normalized", &self.embed_norm),
            ("readout_input", &self.last),
        ] {
            out.insert(name.to_owned(), buffer.read()?[..rows * w].to_vec());
        }
        for (i, l) in self.layers.iter().enumerate() {
            for (name, buffer, cols) in [
                ("qkv", &l.qkv, 3 * w),
                ("attention", &l.att, w),
                ("attention_projected", &l.ap, w),
                ("mid", &l.mid, w),
                ("ffn_gate_value", &l.fc, 2 * h),
                ("ffn_hidden", &l.hidden, h),
                ("ffn_projected", &l.fp, w),
                ("output", &l.out, w),
                ("grad_qkv", &l.dqkv, 3 * w),
                ("grad_attention", &l.datt, w),
                ("grad_ffn_gate_value", &l.dfc, 2 * h),
                ("grad_ffn_hidden", &l.dhidden, h),
            ] {
                out.insert(
                    format!("layer{i}.{name}"),
                    buffer.read()?[..rows * cols].to_vec(),
                );
            }
            out.insert(
                format!("layer{i}.probabilities"),
                l.prob.read()?[..rows * self.config.heads * 2].to_vec(),
            );
        }
        Ok(out)
    }
}
