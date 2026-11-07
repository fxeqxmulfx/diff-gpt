//! Host orchestration over preallocated GPU buffers. No Python, libtorch, autograd engine,
//! per-update allocation, or host readback in a training update.

use crate::{attention, complex, gemm, layout, ops};
use kernite_cuda::{Buffer, Cuda, Grid, Ptr};
use std::error::Error;

pub type Result<T> = std::result::Result<T, Box<dyn Error>>;
type F = Buffer<f32>;
type P = Ptr<f32>;

#[derive(Clone)]
pub struct Config {
    pub width: usize,
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
    fn new(cuda: &Cuda, w: usize, hidden: usize, offset: usize) -> Result<Self> {
        let r = 512;
        Ok(Self {
            offset,
            n1: cuda.alloc(r * w)?,
            qkv: cuda.alloc(r * 3 * w)?,
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
            flags: cuda.alloc(256)?,
            head_ok: cuda.alloc(256 * w / 32)?,
        })
    }
}

pub struct Engine {
    pub cuda: Cuda,
    pub config: Config,
    ops: ops::kernels::Kernels,
    att: attention::kernels::Kernels,
    gemm: gemm::kernels::Kernels,
    complex: complex::kernels::Kernels,
    pair_map: Buffer<u32>,
    pub parameters: F,
    pub gradients: F,
    pub first_moment: F,
    pub second_moment: F,
    rope: F,
    pub train: Buffer<u32>,
    pub validation: Buffer<u32>,
    pub order: Buffer<u32>,
    pub identity: Buffer<u32>,
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
    hits: Buffer<u32>,
    scratch: F,
    layers: Vec<Layer>,
}

fn grid(count: usize) -> Grid {
    Grid::new(count.div_ceil(256) as u32, 256)
}
fn rows_grid(rows: usize) -> Grid {
    Grid::new(rows.div_ceil(4) as u32, 128)
}

impl Engine {
    pub fn new(
        config: Config,
        parameters: &[f32],
        rope: &[f32],
        train: &[u32],
        validation: &[u32],
        order: &[u32],
    ) -> Result<Self> {
        let w = config.width;
        if ![128, 256].contains(&w) || ![1, 2].contains(&config.layers) || rope.len() != 64 {
            return Err("unsupported model shape".into());
        }
        let hidden = layout::hidden(w, &config.architecture);
        let per_layer = 4 * w * w + 3 * w * hidden;
        if parameters.len() != 97 * w + config.layers * per_layer {
            return Err("parameter layout does not match the native model".into());
        }
        let cuda = Cuda::new()?;
        let ops = ops::kernels::Kernels::load(&cuda)?;
        let att = attention::kernels::Kernels::load(&cuda)?;
        let gemm = gemm::kernels::Kernels::load(&cuda)?;
        let complex = complex::kernels::Kernels::load(&cuda)?;
        let pair_map = if config.architecture == "real" {
            vec![0u32; 5]
        } else {
            layout::pairs(w, config.layers, hidden)
                .into_iter()
                .flatten()
                .collect()
        };
        let mut layers = Vec::new();
        for l in 0..config.layers {
            layers.push(Layer::new(&cuda, w, hidden, 97 * w + l * per_layer)?);
        }
        Ok(Self {
            ops,
            att,
            gemm,
            complex,
            pair_map: cuda.upload(&pair_map)?,
            parameters: cuda.upload(parameters)?,
            gradients: cuda.alloc(parameters.len())?,
            first_moment: cuda.alloc(parameters.len())?,
            second_moment: cuda.alloc(parameters.len())?,
            rope: cuda.upload(rope)?,
            train: cuda.upload(train)?,
            validation: cuda.upload(validation)?,
            order: cuda.upload(order)?,
            identity: cuda.upload(&(0..4705).collect::<Vec<u32>>())?,
            tokens: cuda.alloc(512)?,
            labels: cuda.alloc(256)?,
            flags: cuda.alloc(256)?,
            embed: cuda.alloc(512 * w)?,
            embed_norm: cuda.alloc(512 * w)?,
            shared: cuda.alloc(512 * w)?,
            last: cuda.alloc(512 * w)?,
            dlast: cuda.alloc(512 * w)?,
            dshared: cuda.alloc(512 * w)?,
            dx: cuda.alloc(512 * w)?,
            dembed: cuda.alloc(512 * w)?,
            logits: cuda.alloc(512 * 97)?,
            dlogits: cuda.alloc(512 * 97)?,
            losses: cuda.alloc(256)?,
            hits: cuda.alloc(256)?,
            scratch: cuda.alloc((512 * 2 * hidden).max(2 * hidden * w).max(3 * w * w))?,
            layers,
            cuda,
            config,
        })
    }

    // SAFETY for kernel calls below: construction checks shape/parameter layout and allocates
    // every buffer at the maximum batch, 256. Public entry points enforce that bound. Kernels
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
            if self.config.width == 128 {
                self.ops.rms::<128>(rows_grid(rows), y, x, rows as u32)?;
            } else {
                self.ops.rms::<256>(rows_grid(rows), y, x, rows as u32)?;
            }
        }
        Ok(())
    }

    fn rms_back(&self, dx: P, x: P, dy: P, rows: usize) -> Result<()> {
        unsafe {
            if self.config.width == 128 {
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
    fn qscale(&self) -> f32 {
        if self.config.ssmax {
            std::f32::consts::LN_2
        } else {
            1.0
        }
    }

    pub fn forward(
        &self,
        validation: bool,
        offset: usize,
        batch: usize,
        ordered: bool,
    ) -> Result<()> {
        if batch == 0 || batch > 256 {
            return Err("batch must be in 1..=256".into());
        }
        let data = if validation {
            &self.validation
        } else {
            &self.train
        };
        let order = if ordered { &self.order } else { &self.identity };
        if offset + batch > order.len() {
            return Err("batch exceeds the frozen order".into());
        }
        let (w, r) = (self.config.width, 2 * batch);
        let h = layout::hidden(w, &self.config.architecture);
        let count = (r * w) as u32;
        unsafe {
            self.ops.gather(
                grid(batch),
                self.tokens.ptr(),
                self.labels.ptr(),
                self.flags.ptr(),
                data.ptr(),
                order.at(offset),
                batch as u32,
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
            unsafe {
                self.att.forward(
                    Grid::new((batch * w / 32) as u32, 32),
                    layer.att.ptr(),
                    layer.head_ok.ptr(),
                    layer.qkv.ptr(),
                    self.rope.ptr(),
                    w as u32,
                    (w / 32) as u32,
                    self.qscale(),
                    self.config.gain,
                )?;
                self.ops.guard(
                    grid(batch),
                    layer.flags.ptr(),
                    flags,
                    layer.head_ok.ptr(),
                    (w / 32) as u32,
                    batch as u32,
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
                self.ops.residual::<true>(
                    grid(r * w),
                    layer.mid.ptr(),
                    x,
                    branch,
                    flags,
                    w as u32,
                    count,
                    self.scale(),
                )?;
            }
            self.rms(layer.n2.ptr(), layer.mid.ptr(), r)?;
            self.linear(layer.fc.ptr(), layer.n2.ptr(), o + 4 * w * w, r, 2 * h, w)?;
            unsafe {
                if self.config.architecture != "real" {
                    if self.config.architecture == "complex-product" {
                        self.complex.product::<true>(
                            grid(r * h / 2),
                            layer.hidden.ptr(),
                            layer.fc.ptr(),
                            (h / 2) as u32,
                            (r * h / 2) as u32,
                        )?;
                    } else {
                        self.complex.product::<false>(
                            grid(r * h / 2),
                            layer.hidden.ptr(),
                            layer.fc.ptr(),
                            (h / 2) as u32,
                            (r * h / 2) as u32,
                        )?;
                    }
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
                self.ops.residual::<false>(
                    grid(r * w),
                    layer.out.ptr(),
                    layer.mid.ptr(),
                    branch,
                    flags,
                    w as u32,
                    count,
                    self.scale(),
                )?;
            }
            x = layer.out.ptr();
        }
        unsafe {
            self.ops
                .share::<false>(grid(r * w), self.shared.ptr(), x, flags, w as u32, count)?;
        }
        self.rms(self.last.ptr(), self.shared.ptr(), r)?;
        self.linear(self.logits.ptr(), self.last.ptr(), 0, r, 97, w)?;
        unsafe {
            self.ops.loss(
                rows_grid(batch),
                self.dlogits.ptr(),
                self.losses.ptr(),
                self.hits.ptr(),
                self.logits.ptr(),
                self.labels.ptr(),
                batch as u32,
            )?;
        }
        Ok(())
    }

    pub fn backward(&self, batch: usize) -> Result<()> {
        if batch == 0 || batch > 256 {
            return Err("batch must be in 1..=256".into());
        }
        let (w, r) = (self.config.width, 2 * batch);
        let h = layout::hidden(w, &self.config.architecture);
        let count = (r * w) as u32;
        self.linear_back(
            self.dlast.ptr(),
            self.dlogits.ptr(),
            self.last.ptr(),
            0,
            r,
            97,
            w,
        )?;
        self.rms_back(self.dshared.ptr(), self.shared.ptr(), self.dlast.ptr(), r)?;
        unsafe {
            self.ops.share::<true>(
                grid(r * w),
                self.dx.ptr(),
                self.dshared.ptr(),
                self.layers.last().unwrap().flags.ptr(),
                w as u32,
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
                self.ops.residual_bwd::<false>(
                    grid(r * w),
                    layer.branch.ptr(),
                    incoming,
                    layer.flags.ptr(),
                    w as u32,
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
                    if self.config.architecture == "complex-product" {
                        self.complex.product_bwd::<true>(
                            grid(r * h / 2),
                            layer.dfc.ptr(),
                            layer.fc.ptr(),
                            layer.dhidden.ptr(),
                            (h / 2) as u32,
                            (r * h / 2) as u32,
                        )?;
                    } else {
                        self.complex.product_bwd::<false>(
                            grid(r * h / 2),
                            layer.dfc.ptr(),
                            layer.fc.ptr(),
                            layer.dhidden.ptr(),
                            (h / 2) as u32,
                            (r * h / 2) as u32,
                        )?;
                    }
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
                self.ops.residual_bwd::<true>(
                    grid(r * w),
                    layer.branch.ptr(),
                    layer.dmid.ptr(),
                    layer.flags.ptr(),
                    w as u32,
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
            unsafe {
                self.att.backward(
                    Grid::new((batch * w / 32) as u32, 32),
                    layer.dqkv.ptr(),
                    layer.datt.ptr(),
                    layer.qkv.ptr(),
                    self.rope.ptr(),
                    w as u32,
                    (w / 32) as u32,
                    self.qscale(),
                    self.config.gain,
                )?;
            }
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
                grid(97 * w),
                self.gradients.ptr(),
                incoming,
                self.tokens.ptr(),
                w as u32,
                97,
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
        if self.config.architecture != "real" {
            let count = self.pair_map.len() / 5;
            // SAFETY: layout::pairs partitions all stored parameters; each owner
            // reads its gradient and writes only its own real expansion/state.
            unsafe {
                if self.config.phase_optimizer {
                    self.complex.adamw::<true>(
                        grid(count),
                        self.parameters.ptr(),
                        self.first_moment.ptr(),
                        self.second_moment.ptr(),
                        self.gradients.ptr(),
                        self.pair_map.ptr(),
                        count as u32,
                        lr as f32,
                        self.config.decay,
                        (1.0 - 0.9f64.powf(step)) as f32,
                        (1.0 - 0.98f64.powf(step)).sqrt() as f32,
                    )?;
                } else {
                    self.complex.adamw::<false>(
                        grid(count),
                        self.parameters.ptr(),
                        self.first_moment.ptr(),
                        self.second_moment.ptr(),
                        self.gradients.ptr(),
                        self.pair_map.ptr(),
                        count as u32,
                        lr as f32,
                        self.config.decay,
                        (1.0 - 0.9f64.powf(step)) as f32,
                        (1.0 - 0.98f64.powf(step)).sqrt() as f32,
                    )?;
                }
            }
            return Ok(());
        }
        unsafe {
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

    pub fn metrics(&self, batch: usize) -> Result<(f64, usize)> {
        let loss = self.losses.read()?;
        let hits = self.hits.read()?;
        if loss[..batch].iter().any(|x| !x.is_finite()) {
            return Err("nonfinite loss".into());
        }
        Ok((
            loss[..batch].iter().map(|&x| x as f64).sum(),
            hits[..batch].iter().map(|&x| x as usize).sum(),
        ))
    }

    pub fn debug(&self, dir: &std::path::Path, batch: usize) -> Result<()> {
        let rows = 2 * batch;
        for (name, buf, n) in [
            ("dlogits", &self.dlogits, rows * 97),
            ("last", &self.last, rows * self.config.width),
            ("dlast", &self.dlast, rows * self.config.width),
            ("dshared", &self.dshared, rows * self.config.width),
        ] {
            crate::write_floats(&dir.join(format!("debug-{name}.f32")), &buf.read()?[..n])?;
        }
        for (i, layer) in self.layers.iter().enumerate() {
            for (name, buf, n) in [
                ("n1", &layer.n1, rows * self.config.width),
                ("qkv", &layer.qkv, rows * 3 * self.config.width),
                ("dqkv", &layer.dqkv, rows * 3 * self.config.width),
                ("dmid", &layer.dmid, rows * self.config.width),
                ("dn2", &layer.dn2, rows * self.config.width),
                ("dfp", &layer.dfp, rows * self.config.width),
                ("datt", &layer.datt, rows * self.config.width),
            ] {
                crate::write_floats(
                    &dir.join(format!("debug-layer{i}-{name}.f32")),
                    &buf.read()?[..n],
                )?;
            }
        }
        Ok(())
    }

    pub fn evaluate(&self, validation: bool) -> Result<serde_json::Value> {
        let n = if validation {
            self.validation.len() / 3
        } else {
            self.train.len() / 3
        };
        let (mut loss, mut hits) = (0.0, 0usize);
        for start in (0..n).step_by(256) {
            let b = 256.min(n - start);
            self.forward(validation, start, b, false)?;
            let (l, h) = self.metrics(b)?;
            loss += l;
            hits += h;
        }
        Ok(
            serde_json::json!({"loss":loss/n as f64,"exact":hits,"examples":n,"sequence_accuracy":hits as f64/n as f64}),
        )
    }
}
