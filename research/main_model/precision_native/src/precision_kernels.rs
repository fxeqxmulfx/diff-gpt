//! Fixed representations only. No data-dependent precision selection.
#[kernite_macros::kernels]
pub mod kernels {
    fn quant<const MODE: usize>(v: f32) {
        if MODE == 2 || MODE == 5 {
            let h: f16;
            cvt.rn.f16.f32(h, v);
            cvt.f32.f16(v, h);
        }
        if MODE == 3 || MODE == 6 {
            let h: bf16;
            cvt.rn.bf16.f32(h, v);
            cvt.f32.bf16(v, h);
        }
        if MODE == 4 || MODE == 7 {
            let h: tf32;
            cvt.rna.tf32.f32(h, v);
            mov.b32(v, h);
        }
    }
    fn round_partial<const MODE: usize>(v: f32) {
        if MODE >= 5 {
            quant::<MODE>(v);
        }
    }
    fn shuffle64(v: f64, shift: u32) {
        let (a, b, c, d): (u32, u32, u32, u32);
        mov.b64((a, b), v);
        shfl.sync.bfly.b32(c, a, shift, 31, 0xffffffff);
        shfl.sync.bfly.b32(d, b, shift, 31, 0xffffffff);
        mov.b64(v, (c, d));
    }
    // Accumulate TwoSum's local addition residual and the incoming residual.
    // The low component's own additions still round and need an error bound.
    // The offline screen retains overflow/underflow cases; no runtime fallback.
    fn compensated_add(hi: f32, lo: f32, x: f32, error: f32) {
        let (t, v, z, r): (f32, f32, f32, f32);
        add.rn.f32(t, hi, x);
        sub.rn.f32(v, t, hi);
        sub.rn.f32(z, t, v);
        sub.rn.f32(z, hi, z);
        sub.rn.f32(r, x, v);
        add.rn.f32(z, z, r);
        add.rn.f32(z, z, error);
        add.rn.f32(lo, lo, z);
        mov.f32(hi, t);
    }
    #[instances((16,0),(16,1),(16,2),(16,3),(16,4),(16,5),(16,6),(16,7),
                (32,0),(32,1),(32,2),(32,3),(32,4),(32,5),(32,6),(32,7),
                (128,0),(128,1),(128,2),(128,3),(128,4),(128,5),(128,6),(128,7),
                (16,8),(32,8),(128,8))]
    pub unsafe extern "ptx-kernel" fn dot<const D: usize, const MODE: usize>(
        out: *mut f64,
        a: *const f32,
        b: *const f32,
    ) {
        let (row, lane, col, at): (u32, u32, u32, u32);
        let ptr: u64;
        mov.u32(row, ctaid.x);
        mov.u32(lane, tid.x);
        let (x, y, product, s, other): (f32, f32, f32, f32, f32);
        let (low, residual, other_low): (f32, f32, f32);
        let (xd, yd, pd, sd, od): (f64, f64, f64, f64, f64);
        mov.f32(s, 0.0);
        mov.f32(low, 0.0);
        mov.f64(sd, 0.0f64);
        for j in 0..(D + 31) / 32 {
            add.u32(col, lane, j * 32);
            mov.f32(x, 0.0);
            mov.f32(y, 0.0);
            let valid: pred;
            setp.lt.u32(valid, col, D);
            if valid {
                mad.lo.u32(at, row, D, col);
                mad.wide.u32(ptr, at, 4, a);
                ld.global.f32(x, [ptr]);
                mad.wide.u32(ptr, at, 4, b);
                ld.global.f32(y, [ptr]);
            }
            if MODE == 0 {
                cvt.f64.f32(xd, x);
                cvt.f64.f32(yd, y);
                mul.rn.f64(pd, xd, yd);
                add.rn.f64(sd, sd, pd);
            } else {
                quant::<MODE>(x);
                quant::<MODE>(y);
                mul.rn.f32(product, x, y);
                if MODE == 8 {
                    neg.f32(residual, product);
                    fma.rn.f32(residual, x, y, residual);
                    compensated_add(s, low, product, residual);
                } else {
                    round_partial::<MODE>(product);
                    add.rn.f32(s, s, product);
                    round_partial::<MODE>(s);
                }
            }
        }
        for j in 0..2 {
            if MODE == 0 {
                mov.f64(od, sd);
                let shift: u32;
                mov.u32(shift, 1 << j);
                shuffle64(od, shift);
                add.rn.f64(sd, sd, od);
            } else {
                shfl.sync.bfly.b32(other, s, 1 << j, 31, 0xffffffff);
                if MODE == 8 {
                    shfl.sync.bfly.b32(other_low, low, 1 << j, 31, 0xffffffff);
                    compensated_add(s, low, other, other_low);
                } else {
                    add.rn.f32(s, s, other);
                    round_partial::<MODE>(s);
                }
            }
        }
        if MODE != 0 {
            cvt.f64.f32(sd, s);
        }
        if MODE == 8 {
            cvt.f64.f32(od, low);
            add.rn.f64(sd, sd, od);
        }
        for j in 2..5 {
            mov.f64(od, sd);
            let shift: u32;
            mov.u32(shift, 1 << j);
            shuffle64(od, shift);
            add.rn.f64(sd, sd, od);
        }
        let lead: pred;
        setp.eq.u32(lead, lane, 0);
        if lead {
            mad.wide.u32(ptr, row, 8, out);
            st.global.f64([ptr], sd);
        }
    }
}
