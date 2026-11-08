// Included inside engine: buffer ownership and complete routed forward/backward.
struct Router {
    offset: usize,
    normalized: F,
    projected: F,
    logits: F,
    gates: F,
    read: Buffer<f64>,
    dread: Buffer<f64>,
    dgates: Buffer<f64>,
    dlogits: F,
    dprojected: F,
    dnormalized: F,
    dinput: F,
}
impl Router {
    fn new(cuda: &Cuda, cfg: &Config, sublayer: usize) -> Result<Self> {
        let r = cfg.max_batch * cfg.max_length;
        Ok(Self {
            offset: layout::router_offset(cfg, sublayer),
            normalized: cuda.alloc(r * 4 * cfg.width)?,
            projected: cuda.alloc(r * 24)?,
            logits: cuda.alloc(r * 24)?,
            gates: cuda.alloc(r * 24)?,
            read: cuda.alloc(r * cfg.width)?,
            dread: cuda.alloc(r * cfg.width)?,
            dgates: cuda.alloc(r * 24)?,
            dlogits: cuda.alloc(r * 24)?,
            dprojected: cuda.alloc(r * 24)?,
            dnormalized: cuda.alloc(r * 4 * cfg.width)?,
            dinput: cuda.alloc(r * 4 * cfg.width)?,
        })
    }
}
impl Engine {
    fn route(&self, router: &Router, x: P, rows: usize) -> Result<()> {
        let k = self.mhc.as_ref().unwrap();
        let w = self.config.width;
        unsafe {
            k.rms::<false>(
                Grid::new(rows as u32, 32),
                router.normalized.ptr(),
                x,
                x,
                (4 * w) as u32,
                rows as u32,
            )?;
        }
        self.linear(
            router.projected.ptr(),
            router.normalized.ptr(),
            router.offset,
            rows,
            24,
            4 * w,
        )?;
        unsafe {
            k.logits(
                grid(rows * 24),
                router.logits.ptr(),
                router.projected.ptr(),
                self.parameters.at(router.offset),
                w as u32,
                (rows * 24) as u32,
            )?;
            k.sinkhorn::<false>(
                Grid::new(rows.div_ceil(2) as u32, 32),
                router.gates.ptr(),
                router.logits.ptr(),
                router.dgates.ptr(),
                rows as u32,
            )?;
            k.read(
                grid(rows * w),
                router.read.ptr(),
                x,
                router.gates.ptr(),
                w as u32,
                (rows * w) as u32,
            )?;
        }
        Ok(())
    }
    fn route_backward(
        &self,
        router: &Router,
        x: P,
        branch: P,
        dy: P,
        dx: P,
        rows: usize,
    ) -> Result<()> {
        let k = self.mhc.as_ref().unwrap();
        let w = self.config.width;
        unsafe {
            k.gate_grad(
                Grid::new((rows * 24) as u32, 32),
                router.dgates.ptr(),
                x,
                branch,
                dy,
                router.dread.ptr(),
                w as u32,
                rows as u32,
                self.scale(),
            )?;
            k.sinkhorn::<true>(
                Grid::new(rows.div_ceil(2) as u32, 32),
                router.dlogits.ptr(),
                router.logits.ptr(),
                router.dgates.ptr(),
                rows as u32,
            )?;
            k.parameter_grad(
                Grid::new(27, 32),
                self.gradients.at(router.offset),
                router.dlogits.ptr(),
                router.projected.ptr(),
                w as u32,
                rows as u32,
            )?;
            k.projection_grad(
                grid(rows * 24),
                router.dprojected.ptr(),
                router.dlogits.ptr(),
                self.parameters.at(router.offset),
                w as u32,
                (rows * 24) as u32,
            )?;
        }
        self.linear_back(
            router.dnormalized.ptr(),
            router.dprojected.ptr(),
            router.normalized.ptr(),
            router.offset,
            rows,
            24,
            4 * w,
        )?;
        unsafe {
            k.rms::<true>(
                Grid::new(rows as u32, 32),
                router.dinput.ptr(),
                x,
                router.dnormalized.ptr(),
                (4 * w) as u32,
                rows as u32,
            )?;
            k.read_backward(
                grid(rows * w),
                dx,
                router.dread.ptr(),
                router.gates.ptr(),
                w as u32,
                (rows * w) as u32,
            )?;
            self.ops.add(
                grid(rows * 4 * w),
                dx,
                dx,
                router.dinput.ptr(),
                (rows * 4 * w) as u32,
            )?;
        }
        Ok(())
    }
    fn forward_mhc(&self, embedding: P, batch: usize, length: usize) -> Result<()> {
        let (w, r) = (self.config.width, batch * length);
        let h = layout::hidden(w, "");
        let k = self.mhc.as_ref().unwrap();
        unsafe {
            k.expand::<false>(
                grid(r * w),
                self.streams.ptr(),
                embedding,
                w as u32,
                (r * w) as u32,
            )?;
        }
        let mut x = self.streams.ptr();
        for layer in &self.layers {
            let routers = layer.routing.as_ref().unwrap();
            let (a, f) = (&routers[0], &routers[1]);
            let o = layer.offset;
            self.route(a, x, r)?;
            unsafe {
                k.read_rms::<false>(
                    Grid::new(r as u32, 32),
                    layer.n1.ptr(),
                    a.dread.ptr(),
                    a.read.ptr(),
                    layer.dn1.ptr(),
                    w as u32,
                    r as u32,
                )?;
            }
            self.linear(layer.qkv.ptr(), layer.n1.ptr(), o, r, 3 * w, w)?;
            self.attention_forward(layer, batch, length)?;
            self.linear(layer.ap.ptr(), layer.att.ptr(), o + 3 * w * w, r, w, w)?;
            self.rms(layer.an.ptr(), layer.ap.ptr(), r)?;
            unsafe {
                k.write(
                    grid(r * 4 * w),
                    layer.mid.ptr(),
                    x,
                    layer.an.ptr(),
                    a.gates.ptr(),
                    w as u32,
                    (r * 4 * w) as u32,
                    self.scale(),
                )?;
            }
            self.route(f, layer.mid.ptr(), r)?;
            unsafe {
                k.read_rms::<false>(
                    Grid::new(r as u32, 32),
                    layer.n2.ptr(),
                    f.dread.ptr(),
                    f.read.ptr(),
                    layer.dn2.ptr(),
                    w as u32,
                    r as u32,
                )?;
            }
            self.linear(layer.fc.ptr(), layer.n2.ptr(), o + 4 * w * w, r, 2 * h, w)?;
            unsafe {
                self.plain_product.as_ref().unwrap().forward(
                    grid(r * h / 2),
                    layer.hidden.ptr(),
                    layer.fc.ptr(),
                    (h / 2) as u32,
                    (r * h / 2) as u32,
                )?;
            }
            self.linear(
                layer.fp.ptr(),
                layer.hidden.ptr(),
                o + 4 * w * w + 2 * h * w,
                r,
                w,
                h,
            )?;
            self.rms(layer.fnorm.ptr(), layer.fp.ptr(), r)?;
            unsafe {
                k.write(
                    grid(r * 4 * w),
                    layer.out.ptr(),
                    layer.mid.ptr(),
                    layer.fnorm.ptr(),
                    f.gates.ptr(),
                    w as u32,
                    (r * 4 * w) as u32,
                    self.scale(),
                )?;
            }
            x = layer.out.ptr();
        }
        unsafe {
            k.collapse::<false>(grid(r * w), self.shared.ptr(), x, w as u32, (r * w) as u32)?;
        }
        Ok(())
    }
    fn backward_mhc(&self, batch: usize, length: usize) -> Result<()> {
        let (w, r) = (self.config.width, batch * length);
        let h = layout::hidden(w, "");
        let k = self.mhc.as_ref().unwrap();
        unsafe {
            k.collapse::<true>(
                grid(r * w),
                self.stream_grad.ptr(),
                self.dshared.ptr(),
                w as u32,
                (r * w) as u32,
            )?;
        }
        let mut incoming = self.stream_grad.ptr();
        for (i, layer) in self.layers.iter().enumerate().rev() {
            let routers = layer.routing.as_ref().unwrap();
            let (a, f) = (&routers[0], &routers[1]);
            let o = layer.offset;
            let x = if i > 0 {
                self.layers[i - 1].out.ptr()
            } else {
                self.streams.ptr()
            };
            unsafe {
                k.write_backward(
                    grid(r * w),
                    layer.dmid.ptr(),
                    layer.branch.ptr(),
                    incoming,
                    f.gates.ptr(),
                    w as u32,
                    (r * w) as u32,
                    self.scale(),
                )?;
            }
            self.rms_back(layer.dfp.ptr(), layer.fp.ptr(), layer.branch.ptr(), r)?;
            self.linear_back(
                layer.dhidden.ptr(),
                layer.dfp.ptr(),
                layer.hidden.ptr(),
                o + 4 * w * w + 2 * h * w,
                r,
                w,
                h,
            )?;
            unsafe {
                self.plain_product.as_ref().unwrap().backward(
                    grid(r * h / 2),
                    layer.dfc.ptr(),
                    layer.fc.ptr(),
                    layer.dhidden.ptr(),
                    (h / 2) as u32,
                    (r * h / 2) as u32,
                )?;
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
            unsafe {
                k.read_rms::<true>(
                    Grid::new(r as u32, 32),
                    layer.n2.ptr(),
                    f.dread.ptr(),
                    f.read.ptr(),
                    layer.dn2.ptr(),
                    w as u32,
                    r as u32,
                )?;
            }
            self.route_backward(
                f,
                layer.mid.ptr(),
                layer.fnorm.ptr(),
                incoming,
                layer.dmid.ptr(),
                r,
            )?;
            unsafe {
                k.write_backward(
                    grid(r * w),
                    layer.dx.ptr(),
                    layer.branch.ptr(),
                    layer.dmid.ptr(),
                    a.gates.ptr(),
                    w as u32,
                    (r * w) as u32,
                    self.scale(),
                )?;
            }
            self.rms_back(layer.dap.ptr(), layer.ap.ptr(), layer.branch.ptr(), r)?;
            self.linear_back(
                layer.datt.ptr(),
                layer.dap.ptr(),
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
            unsafe {
                k.read_rms::<true>(
                    Grid::new(r as u32, 32),
                    layer.n1.ptr(),
                    a.dread.ptr(),
                    a.read.ptr(),
                    layer.dn1.ptr(),
                    w as u32,
                    r as u32,
                )?;
            }
            self.route_backward(a, x, layer.an.ptr(), layer.dmid.ptr(), layer.dx.ptr(), r)?;
            incoming = layer.dx.ptr();
        }
        unsafe {
            k.expand::<true>(
                grid(r * w),
                self.dx.ptr(),
                incoming,
                w as u32,
                (r * w) as u32,
            )?;
        }
        self.rms_back(self.dembed.ptr(), self.embed.ptr(), self.dx.ptr(), r)?;
        unsafe {
            self.ops.embed_bwd(
                grid(self.config.vocab * w),
                self.gradients.ptr(),
                self.dembed.ptr(),
                self.tokens.ptr(),
                w as u32,
                self.config.vocab as u32,
                r as u32,
            )?;
        }
        Ok(())
    }
}
