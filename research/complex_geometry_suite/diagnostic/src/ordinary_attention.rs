//! Ordinary values and adjoint using the frozen causal SSMax probabilities.
//! The frozen probability kernel's unused XSA output is overwritten. Keeping
//! score arithmetic identical isolates removal of the value projection.
#[kernite_macros::kernels]
pub mod kernels {
    fn load32(v: f32, p: u64, i: u32) {
        let a: u64;
        mad.wide.u32(a, i, 4, p);
        ld.global.f32(v, [a]);
    }
    fn store32(p: u64, i: u32, v: f32) {
        let a: u64;
        mad.wide.u32(a, i, 4, p);
        st.global.f32([a], v);
    }
    fn load(v: f64, p: u64, i: u32) {
        let f: f32;
        load32(f, p, i);
        cvt.f64.f32(v, f);
    }
    fn store(p: u64, i: u32, v: f64) {
        let f: f32;
        cvt.rn.f32.f64(f, v);
        store32(p, i, f);
    }
    fn sum(v: f64) {
        let (a, b, c, d): (u32, u32, u32, u32);
        let x: f64;
        for j in 0..5 {
            mov.b64((a, b), v);
            shfl.sync.bfly.b32(c, a, 1 << j, 31, 0xffffffff);
            shfl.sync.bfly.b32(d, b, 1 << j, 31, 0xffffffff);
            mov.b64(x, (c, d));
            add.rn.f64(v, v, x);
        }
    }
    #[instances(16, 32, 128)]
    pub unsafe extern "ptx-kernel" fn values<const D: usize>(
        out: *mut f32,
        prob: *const f32,
        qkv: *const f32,
        heads: u32,
        length: u32,
    ) {
        let (id, lane, b, t, h, w, base, anchor, scorebase, key, at, col, outbase): (
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
        );
        mov.u32(id, ctaid.x);
        mov.u32(lane, tid.x);
        rem.u32(h, id, heads);
        div.u32(t, id, heads);
        div.u32(b, t, length);
        rem.u32(t, t, length);
        mul.lo.u32(w, heads, D);
        mul.lo.u32(base, b, length);
        mul.lo.u32(base, base, 3);
        mul.lo.u32(base, base, w);
        mad.lo.u32(anchor, h, D, base);
        mad.lo.u32(anchor, w, 2, anchor);
        mad.lo.u32(scorebase, b, heads, h);
        mad.lo.u32(scorebase, scorebase, length, t);
        mul.lo.u32(scorebase, scorebase, length);
        mad.lo.u32(outbase, b, length, t);
        mul.lo.u32(outbase, outbase, w);
        mad.lo.u32(outbase, h, D, outbase);
        let (first, avg): ([f32; (D + 31) / 32], [f32; (D + 31) / 32]);
        let (p, v, z): (f32, f32, f32);
        for j in 0..(D + 31) / 32 {
            add.u32(col, lane, 32 * j);
            mov.f32(first[j], 0.0);
            mov.f32(avg[j], 0.0);
            let active: pred;
            setp.lt.u32(active, col, D);
            if active {
                add.u32(at, anchor, col);
                load32(first[j], qkv, at);
            }
        }
        mov.u32(key, 0);
        label(mix);
        add.u32(at, scorebase, key);
        load32(p, prob, at);
        for j in 0..(D + 31) / 32 {
            add.u32(col, lane, 32 * j);
            let active: pred;
            setp.lt.u32(active, col, D);
            if active {
                mul.lo.u32(at, key, 3);
                mad.lo.u32(at, at, w, anchor);
                add.u32(at, at, col);
                load32(v, qkv, at);
                sub.rn.f32(v, v, first[j]);
                mul.rn.f32(z, p, v);
                add.rn.f32(avg[j], avg[j], z);
            }
        }
        add.u32(key, key, 1);
        let more: pred;
        setp.le.u32(more, key, t);
        if more {
            bra(mix);
        }
        for j in 0..(D + 31) / 32 {
            add.u32(col, lane, 32 * j);
            let active: pred;
            setp.lt.u32(active, col, D);
            if active {
                add.rn.f32(z, first[j], avg[j]);
                add.u32(at, outbase, col);
                store32(out, at, z);
            }
        }
    }
    #[instances(16, 32, 128)]
    pub unsafe extern "ptx-kernel" fn query_backward<const D: usize>(
        dq: *mut f32,
        ds: *mut f32,
        gd: *mut f32,
        dv: *mut f32,
        dy: *const f32,
        prob: *const f32,
        qkv: *const f32,
        heads: u32,
        length: u32,
    ) {
        let (id, lane, b, t, h, w, base, qbase, anchor, scorebase, key, at, col, outbase): (
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
        );
        mov.u32(id, ctaid.x);
        mov.u32(lane, tid.x);
        rem.u32(h, id, heads);
        div.u32(t, id, heads);
        div.u32(b, t, length);
        rem.u32(t, t, length);
        mul.lo.u32(w, heads, D);
        mul.lo.u32(base, b, length);
        mul.lo.u32(base, base, 3);
        mul.lo.u32(base, base, w);
        mul.lo.u32(qbase, t, 3);
        mad.lo.u32(qbase, qbase, w, base);
        mad.lo.u32(qbase, h, D, qbase);
        mad.lo.u32(anchor, h, D, base);
        mad.lo.u32(anchor, w, 2, anchor);
        mad.lo.u32(scorebase, b, heads, h);
        mad.lo.u32(scorebase, scorebase, length, t);
        mul.lo.u32(scorebase, scorebase, length);
        mad.lo.u32(outbase, b, length, t);
        mul.lo.u32(outbase, outbase, w);
        mad.lo.u32(outbase, h, D, outbase);
        let (first, gradient, dquery): (
            [f64; (D + 31) / 32],
            [f64; (D + 31) / 32],
            [f64; (D + 31) / 32],
        );
        let (s, z, p, total, value, gain, dsval): (f64, f64, f64, f64, f64, f64, f64);
        for j in 0..(D + 31) / 32 {
            add.u32(col, lane, 32 * j);
            mov.f64(first[j], 0.0f64);
            mov.f64(gradient[j], 0.0f64);
            mov.f64(dquery[j], 0.0f64);
            let active: pred;
            setp.lt.u32(active, col, D);
            if active {
                add.u32(at, anchor, col);
                load(first[j], qkv, at);
                add.u32(at, outbase, col);
                load(gradient[j], dy, at);
                store(gd, at, gradient[j]);
                mov.f64(z, 0.0f64);
                store(dv, at, z);
            }
        }
        add.u32(at, t, 1);
        let gain32: f32;
        cvt.rn.f32.u32(gain32, at);
        lg2.approx.f32(gain32, gain32);
        mul.rn.f32(gain32, gain32, 0.6931471805599453 / (D as f32));
        cvt.f64.f32(gain, gain32);
        mov.f64(total, 0.0f64);
        // Two passes avoid rounding the intermediate dP dot products to FP32.
        for pass in 0..2 {
            mov.u32(key, 0);
            label(keys);
            mov.f64(s, 0.0f64);
            for j in 0..(D + 31) / 32 {
                add.u32(col, lane, 32 * j);
                let active: pred;
                setp.lt.u32(active, col, D);
                if active {
                    mul.lo.u32(at, key, 3);
                    mad.lo.u32(at, at, w, anchor);
                    add.u32(at, at, col);
                    load(value, qkv, at);
                    sub.rn.f64(value, value, first[j]);
                    mul.rn.f64(z, gradient[j], value);
                    add.rn.f64(s, s, z);
                }
            }
            sum(s);
            add.u32(at, scorebase, key);
            load(p, prob, at);
            if pass == 0 {
                mul.rn.f64(z, p, s);
                add.rn.f64(total, total, z);
            } else {
                sub.rn.f64(s, s, total);
                mul.rn.f64(s, s, p);
                mul.rn.f64(dsval, s, gain);
                let lead: pred;
                setp.eq.u32(lead, lane, 0);
                if lead {
                    store(ds, at, dsval);
                }
                for j in 0..(D + 31) / 32 {
                    add.u32(col, lane, 32 * j);
                    let active: pred;
                    setp.lt.u32(active, col, D);
                    if active {
                        mul.lo.u32(at, key, 3);
                        add.u32(at, at, 1);
                        mad.lo.u32(at, at, w, base);
                        mad.lo.u32(at, h, D, at);
                        add.u32(at, at, col);
                        load(value, qkv, at);
                        mul.rn.f64(z, dsval, value);
                        add.rn.f64(dquery[j], dquery[j], z);
                    }
                }
            }
            add.u32(key, key, 1);
            let more: pred;
            setp.le.u32(more, key, t);
            if more {
                bra(keys);
            }
        }
        for j in 0..(D + 31) / 32 {
            add.u32(col, lane, 32 * j);
            let active: pred;
            setp.lt.u32(active, col, D);
            if active {
                add.u32(at, qbase, col);
                store(dq, at, dquery[j]);
            }
        }
    }
}
