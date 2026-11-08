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
    routing: Option<[Router; 2]>,
    offset: usize,
    n1: F,
    qkv: F,
    qnorm: F,
    qkr: F,
    dqnorm: F,
    dqkr: F,
    prob: F,
    centered_coefficients: Buffer<f32>,
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
    fn new(cuda: &Cuda, cfg: &Config, hidden: usize, offset: usize, index: usize) -> Result<Self> {
        let w = cfg.width;
        let r = cfg.max_batch * cfg.max_length;
        let scores = cfg.max_batch * cfg.heads * cfg.max_length * cfg.max_length;
        Ok(Self {
            routing: if layout::is_mhc(cfg) {
                Some([
                    Router::new(cuda, cfg, 2 * index)?,
                    Router::new(cuda, cfg, 2 * index + 1)?,
                ])
            } else {
                None
            },
            offset,
            n1: cuda.alloc(r * w)?,
            qkv: cuda.alloc(r * 3 * w)?,
            qnorm: cuda.alloc(r * 3 * w)?,
            qkr: cuda.alloc(r * 3 * w)?,
            dqnorm: cuda.alloc(r * 3 * w)?,
            dqkr: cuda.alloc(r * 3 * w)?,
            prob: cuda.alloc(scores)?,
            centered_coefficients: cuda.alloc(1)?,
            ds: cuda.alloc(scores)?,
            delta: cuda.alloc(r * w)?,
            gd: cuda.alloc(r * w)?,
            dv: cuda.alloc(r * w)?,
            att: cuda.alloc(r * w)?,
            ap: cuda.alloc(r * w)?,
            an: cuda.alloc(r * w)?,
            mid: cuda.alloc(r * w * if layout::is_mhc(cfg) { 4 } else { 1 })?,
            n2: cuda.alloc(r * w)?,
            fc: cuda.alloc(r * 2 * hidden)?,
            hidden: cuda.alloc(r * hidden)?,
            fp: cuda.alloc(r * w)?,
            fnorm: cuda.alloc(r * w)?,
            out: cuda.alloc(r * w * if layout::is_mhc(cfg) { 4 } else { 1 })?,
            dn1: cuda.alloc(r * w)?,
            dqkv: cuda.alloc(r * 3 * w)?,
            datt: cuda.alloc(r * w)?,
            dap: cuda.alloc(r * w)?,
            dmid: cuda.alloc(r * w * if layout::is_mhc(cfg) { 4 } else { 1 })?,
            dn2: cuda.alloc(r * w)?,
            dfc: cuda.alloc(r * 2 * hidden)?,
            dhidden: cuda.alloc(r * hidden)?,
            dfp: cuda.alloc(r * w)?,
            dx: cuda.alloc(r * w * if layout::is_mhc(cfg) { 4 } else { 1 })?,
            branch: cuda.alloc(r * w)?,
            flags: cuda.alloc(r)?,
            head_ok: cuda.alloc(cfg.max_batch * cfg.heads)?,
        })
    }
}

pub struct Engine {
    mhc: Option<crate::mhc::kernels::Kernels>,
    streams: F,
    stream_grad: F,
    pub cuda: Cuda,
    pub config: Config,
    ops: ops::kernels::Kernels,
    transpose_tile: crate::tiled_transpose::kernels::Kernels,
    ordinary_forward: crate::ordinary_forward::kernels::Kernels,
    embedding_backward: crate::embedding_backward::kernels::Kernels,
    att: attention::kernels::Kernels,
    seq: sequence_ops::kernels::Kernels,
    stable_loss: Option<crate::stable_loss::kernels::Kernels>,
    stable_attention: Option<crate::stable_attention::kernels::Kernels>,
    stable_chain: Option<crate::stable_chain::kernels::Kernels>,
    plain_product: Option<crate::plain_product::kernels::Kernels>,
    ordinary_attention: Option<crate::ordinary_attention::kernels::Kernels>,
    centered_attention: Option<crate::centered_attention::kernels::Kernels>,
    gemm: gemm::kernels::Kernels,
    tensor: crate::tensor::kernels::Kernels,
    pipeline: crate::pipeline_host::Pipeline,
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
                "complex-mhc",
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
        if parameters.len() != layout::total_count(&config) {
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
                l,
            )?);
        }
        let mut engine = Self {
            mhc: if layout::is_mhc(&config) {
                Some(crate::mhc::kernels::Kernels::load(&cuda)?)
            } else {
                None
            },
            streams: cuda.alloc(if layout::is_mhc(&config) {
                r * 4 * w
            } else {
                1
            })?,
            stream_grad: cuda.alloc(if layout::is_mhc(&config) {
                r * 4 * w
            } else {
                1
            })?,
            ops,
            transpose_tile: crate::tiled_transpose::kernels::Kernels::load(&cuda)?,
            ordinary_forward: crate::ordinary_forward::kernels::Kernels::load(&cuda)?,
            embedding_backward: crate::embedding_backward::kernels::Kernels::load(&cuda)?,
            seq: sequence_ops::kernels::Kernels::load(&cuda)?,
            stable_loss: None,
            stable_attention: None,
            stable_chain: None,
            plain_product: None,
            ordinary_attention: None,
            centered_attention: None,
            att,
            gemm,
            tensor: crate::tensor::kernels::Kernels::load(&cuda)?,
            pipeline: {
                let columns = (4 * w).max(2 * hidden).max(config.vocab).div_ceil(8) * 8;
                let rows = r.div_ceil(8) * 8;
                let operand = rows.max(columns) * columns;
                let partial = (rows * columns).max(16 * columns * columns);
                crate::pipeline_host::Pipeline::new(&cuda, operand, operand, partial)?
            },
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
        };
        engine.enable_stable_loss()?;
        engine.enable_stable_attention()?;
        engine.enable_plain_product()?;
        engine.enable_ordinary_attention()?;
        Ok(engine)
    }

    // SAFETY for kernel calls below: construction checks shape/parameter layout and allocates
    // every buffer at the configured maximum batch and length. Entry points enforce those bounds. Kernels
    // all launch in order on the same CUDA context's default stream; these buffers outlive
    // every launch. Each operation below is assigned disjoint input/output ranges, except
    // the documented elementwise accumulation into its own element.
    fn mm_backward(&self, out: P, a: P, b: P, m: usize, n: usize, k: usize) -> Result<()> {
        if crate::precision::TF32_BACKWARD {
            return self
                .pipeline
                .multiply_tf32(&self.tensor, out, a, b, (m, n, k));
        }
        let g = Grid::new(
            (m.div_ceil(gemm::kernels::BM) * n.div_ceil(gemm::kernels::BN)) as u32,
            gemm::kernels::THREADS as u32,
        );
        unsafe {
            if k.is_multiple_of(4)
                && n.is_multiple_of(4)
                && out.addr().is_multiple_of(16)
                && a.addr().is_multiple_of(16)
                && b.addr().is_multiple_of(16)
            {
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
            if crate::placement::TILED_TRANSPOSE {
                self.transpose_tile.transpose(
                    Grid::new((rows.div_ceil(32) * cols.div_ceil(32)) as u32, 256),
                    self.scratch.ptr(),
                    x,
                    rows as u32,
                    cols as u32,
                )?;
                return Ok(());
            }
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
        self.pipeline
            .multiply::<{ crate::precision::BF16 }>(out, x, self.scratch.ptr(), rows, n, k)
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
        self.mm_backward(dx, dy, self.parameters.at(offset), rows, k, n)?;
        self.transpose(dy, rows, n)?;
        self.mm_backward(self.gradients.at(offset), self.scratch.ptr(), x, n, k, rows)
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
                        Grid::new(
                            ((r * 3 * h) / (if $d == 16 { 2usize } else { 1usize })).div_ceil(4)
                                as u32,
                            128,
                        ),
                        layer.qnorm.ptr(),
                        layer.qkv.ptr(),
                        h as u32,
                        (r * 3 * h) as u32,
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
                    if crate::placement::FUSED_ATTENTION {
                        self.ordinary_forward.forward::<$d>(
                            Grid::new(
                                ((r * h) / if $d == 16 { 2 } else { 1 }).div_ceil(4) as u32,
                                128,
                            ),
                            layer.att.ptr(),
                            layer.prob.ptr(),
                            layer.qkr.ptr(),
                            h as u32,
                            length as u32,
                            (r * h) as u32,
                        )?;
                    } else {
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
                        if let Some(centered) = &self.centered_attention {
                            centered.coefficients::<$d>(
                                Grid::new((r * h) as u32, 32),
                                layer.centered_coefficients.ptr(),
                                layer.prob.ptr(),
                                layer.qkr.ptr(),
                                h as u32,
                                length as u32,
                            )?;
                            centered.values::<$d>(
                                Grid::new((r * h) as u32, 32),
                                layer.att.ptr(),
                                layer.centered_coefficients.ptr(),
                                layer.qkr.ptr(),
                                h as u32,
                                length as u32,
                            )?;
                        } else if let Some(ordinary) = &self.ordinary_attention {
                            ordinary.values::<$d>(
                                Grid::new((r * h) as u32, 32),
                                layer.att.ptr(),
                                layer.prob.ptr(),
                                layer.qkr.ptr(),
                                h as u32,
                                length as u32,
                            )?;
                        }
                    }
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
                    macro_rules! query {
                        ($kernel:expr) => {
                            $kernel.query_backward::<$d>(
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
                        };
                    }
                    if let Some(ordinary) = &self.ordinary_attention {
                        ordinary.query_backward::<$d>(
                            Grid::new(
                                ((r * h) / if $d == 16 { 2 } else { 1 }).div_ceil(4) as u32,
                                128,
                            ),
                            layer.dqkr.ptr(),
                            layer.ds.ptr(),
                            layer.gd.ptr(),
                            layer.dv.ptr(),
                            layer.datt.ptr(),
                            layer.prob.ptr(),
                            layer.qkr.ptr(),
                            h as u32,
                            length as u32,
                            (r * h) as u32,
                        )?;
                    } else if let Some(att) = &self.stable_attention {
                        query!(att);
                    } else {
                        query!(self.att);
                    }
                    if let Some(chain) = &self.stable_chain
                        && self.ordinary_attention.is_none()
                    {
                        chain.query_backward(
                            Grid::new(r as u32, 32),
                            layer.dqkr.ptr(),
                            layer.ds.ptr(),
                            layer.qkr.ptr(),
                            layer.prob.ptr(),
                            self.parameters.at(layer.offset + 3 * w * w),
                            layer.branch.ptr(),
                            length as u32,
                            u32::from(self.config.architecture == "complex-hermitian"),
                        )?;
                    }
                    if let Some(centered) = &self.centered_attention {
                        centered.key_backward::<$d>(
                            Grid::new((r * h) as u32, 32),
                            layer.dqkr.ptr(),
                            layer.ds.ptr(),
                            layer.gd.ptr(),
                            layer.centered_coefficients.ptr(),
                            layer.qkr.ptr(),
                            h as u32,
                            length as u32,
                        )?;
                    } else {
                        self.att.key_backward::<$d>(
                            Grid::new(
                                ((r * h) / (if $d == 16 { 2usize } else { 1usize })).div_ceil(4)
                                    as u32,
                                128,
                            ),
                            layer.dqkr.ptr(),
                            layer.ds.ptr(),
                            layer.gd.ptr(),
                            layer.dv.ptr(),
                            layer.prob.ptr(),
                            layer.qkr.ptr(),
                            h as u32,
                            length as u32,
                            (r * h) as u32,
                        )?;
                    }
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
                        Grid::new(((r * 3 * h) / if $d == 16 { 2 } else { 1 }) as u32, 32),
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

    pub fn enable_plain_product(&mut self) -> Result<()> {
        if !layout::is_complex(&self.config.architecture) {
            return Err("plain product requires the complex projection layout".into());
        }
        self.plain_product = Some(crate::plain_product::kernels::Kernels::load(&self.cuda)?);
        Ok(())
    }

    pub fn uses_plain_product(&self) -> bool {
        self.plain_product.is_some()
    }

    /// Ordinary causal values use every first/repeated position. XSA prefix
    /// flags stay zero, so both forward sharing and adjoint masks are disabled.
    pub fn enable_ordinary_attention(&mut self) -> Result<()> {
        self.centered_attention = None;
        self.ordinary_attention = Some(crate::ordinary_attention::kernels::Kernels::load(
            &self.cuda,
        )?);
        let zeros = vec![0u32; self.config.max_batch * self.config.max_length];
        for layer in &mut self.layers {
            layer.flags.write(&zeros)?;
        }
        Ok(())
    }

    pub fn enable_centered_attention(&mut self) -> Result<()> {
        Err("centering is disabled for this comparison".into())
    }

    pub fn enable_stable_attention(&mut self) -> Result<()> {
        self.stable_attention = Some(crate::stable_attention::kernels::Kernels::load(&self.cuda)?);
        if self.config.heads == 1 && self.config.width == 128 {
            self.stable_chain = Some(crate::stable_chain::kernels::Kernels::load(&self.cuda)?);
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
        if self.mhc.is_some() {
            self.forward_mhc(x, batch, length)?;
        } else {
            let mut flags = self.flags.ptr();
            for layer in &self.layers {
                let o = layer.offset;
                self.rms(layer.n1.ptr(), x, r)?;
                self.linear(layer.qkv.ptr(), layer.n1.ptr(), o, r, 3 * w, w)?;
                self.attention_forward(layer, batch, length)?;
                if self.centered_attention.is_some() {
                    // Exact constant-prefix cancellation does not need XSA's value-norm guard.
                    unsafe {
                        self.seq.prefix(
                            grid(r),
                            layer.flags.ptr(),
                            self.tokens.ptr(),
                            length as u32,
                            r as u32,
                        )?;
                    }
                } else if self.ordinary_attention.is_none() {
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
                }
                flags = layer.flags.ptr();
                self.linear(layer.ap.ptr(), layer.att.ptr(), o + 3 * w * w, r, w, w)?;
                let mut branch = layer.ap.ptr();
                if self.config.normalized {
                    self.rms(layer.an.ptr(), branch, r)?;
                    branch = layer.an.ptr();
                }
                unsafe {
                    if self.ordinary_attention.is_some() && self.centered_attention.is_none() {
                        // The ordinary residual includes the first position as well.
                        self.seq.residual::<false>(
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
                    } else {
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
                }
                self.rms(layer.n2.ptr(), layer.mid.ptr(), r)?;
                self.linear(layer.fc.ptr(), layer.n2.ptr(), o + 4 * w * w, r, 2 * h, w)?;
                unsafe {
                    if let Some(plain) = &self.plain_product {
                        plain.forward(
                            grid(r * h / 2),
                            layer.hidden.ptr(),
                            layer.fc.ptr(),
                            (h / 2) as u32,
                            (r * h / 2) as u32,
                        )?;
                    } else if self.config.architecture != "real" {
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
        if self.mhc.is_some() {
            return self.backward_mhc(batch, length);
        }
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
                if let Some(plain) = &self.plain_product {
                    plain.backward(
                        grid(r * h / 2),
                        layer.dfc.ptr(),
                        layer.fc.ptr(),
                        layer.dhidden.ptr(),
                        (h / 2) as u32,
                        (r * h / 2) as u32,
                    )?;
                } else if self.config.architecture != "real" {
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
                if self.ordinary_attention.is_some() && self.centered_attention.is_none() {
                    self.seq.scale(
                        grid(r * w),
                        layer.branch.ptr(),
                        layer.dmid.ptr(),
                        count,
                        self.scale(),
                    )?;
                } else {
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
            self.embedding_backward.accumulate(
                Grid::new(self.config.vocab as u32, w as u32),
                self.gradients.ptr(),
                incoming,
                self.tokens.ptr(),
                w as u32,
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
                if self.mhc.is_some() {
                    let offset = layout::base_count(&self.config);
                    let count = self.parameters.len() - offset;
                    self.ops.adamw(
                        grid(count),
                        self.parameters.at(offset),
                        self.first_moment.at(offset),
                        self.second_moment.at(offset),
                        self.gradients.at(offset),
                        count as u32,
                        lr as f32,
                        self.config.decay,
                        (1.0 - 0.9f64.powf(step)) as f32,
                        (1.0 - 0.98f64.powf(step)).sqrt() as f32,
                    )?;
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
                ("qkv_rotated", &l.qkr, 3 * w),
                ("qkv_normalized", &l.qnorm, 3 * w),
                ("attention", &l.att, w),
                ("attention_projected", &l.ap, w),
                ("mid", &l.mid, w),
                ("ffn_gate_value", &l.fc, 2 * h),
                ("ffn_hidden", &l.hidden, h),
                ("ffn_projected", &l.fp, w),
                ("output", &l.out, w),
                ("grad_qkv", &l.dqkv, 3 * w),
                ("grad_qkv_rotated", &l.dqkr, 3 * w),
                ("grad_qkv_normalized", &l.dqnorm, 3 * w),
                ("grad_attention", &l.datt, w),
                ("grad_attention_projected", &l.dap, w),
                ("grad_attention_normalized", &l.branch, w),
                ("grad_mid", &l.dmid, w),
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

include!("engine_mhc.rs");
