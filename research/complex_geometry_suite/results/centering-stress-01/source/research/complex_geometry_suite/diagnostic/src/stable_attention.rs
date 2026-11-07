//! Opt-in XSA adjoint with FP64 geometry and dot-product accumulation.
//! FP32 activations/probabilities and outputs retain the original interface.
//! This reduces cancellation inside XSA; upstream FP32 boundaries remain.
#[kernite_macros::kernels]
pub mod kernels {
    fn load(v: f64, p: u64, i: u32) {
        let a: u64;
        let f: f32;
        mad.wide.u32(a, i, 4, p);
        ld.global.f32(f, [a]);
        cvt.f64.f32(v, f);
    }
    fn store(p: u64, i: u32, v: f64) {
        let a: u64;
        let f: f32;
        cvt.rn.f32.f64(f, v);
        mad.wide.u32(a, i, 4, p);
        st.global.f32([a], f);
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
    fn load_j<const D: usize>(v: f64, p: u64, base: u32, col: u32) {
        let (partner, other, at): (u32, u32, u32);
        let negative: f64;
        let real: pred;
        setp.lt.u32(real, col, D / 2);
        add.u32(partner, col, D / 2);
        sub.u32(other, col, D / 2);
        selp.u32(partner, partner, other, real);
        add.u32(at, base, partner);
        load(v, p, at);
        neg.f64(negative, v);
        selp.f64(v, negative, v, real);
    }
    #[instances(16, 32, 128)]
    pub unsafe extern "ptx-kernel" fn query_backward<const D: usize>(
        dq: *mut f32,
        ds: *mut f32,
        gd: *mut f32,
        dv: *mut f32,
        dy: *const f32,
        delta: *const f32,
        prob: *const f32,
        qkv: *const f32,
        heads: u32,
        length: u32,
        hermitian: u32,
    ) {
        let (id, lane, b, t, h, w, base, qbase, vbase, anchor, scorebase, key, at, col, outbase): (
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
        mad.lo.u32(vbase, w, 2, qbase);
        mad.lo.u32(anchor, h, D, base);
        mad.lo.u32(anchor, w, 2, anchor);
        mad.lo.u32(scorebase, b, heads, h);
        mad.lo.u32(scorebase, scorebase, length, t);
        mul.lo.u32(scorebase, scorebase, length);
        mad.lo.u32(outbase, b, length, t);
        mul.lo.u32(outbase, outbase, w);
        mad.lo.u32(outbase, h, D, outbase);
        let (own, change, gradient, first, dchange, dquery, jown, jgradient, jchange): (
            [f64; (D + 31) / 32],
            [f64; (D + 31) / 32],
            [f64; (D + 31) / 32],
            [f64; (D + 31) / 32],
            [f64; (D + 31) / 32],
            [f64; (D + 31) / 32],
            [f64; (D + 31) / 32],
            [f64; (D + 31) / 32],
            [f64; (D + 31) / 32],
        );
        let (norm, den, a, beta, z, u, gv, rem, gain, p, s, total, kval, dsval): (
            f64,
            f64,
            f64,
            f64,
            f64,
            f64,
            f64,
            f64,
            f64,
            f64,
            f64,
            f64,
            f64,
            f64,
        );
        mov.f64(norm, 0.0f64);
        let complex_mode: pred;
        setp.ne.u32(complex_mode, hermitian, 0);
        let first_query: pred;
        setp.eq.u32(first_query, t, 0);
        for j in 0..(D + 31) / 32 {
            add.u32(col, lane, 32 * j);
            mov.f64(own[j], 0.0f64);
            mov.f64(change[j], 0.0f64);
            mov.f64(gradient[j], 0.0f64);
            mov.f64(first[j], 0.0f64);
            mov.f64(dquery[j], 0.0f64);
            mov.f64(jown[j], 0.0f64);
            mov.f64(jgradient[j], 0.0f64);
            mov.f64(jchange[j], 0.0f64);
            let active: pred;
            setp.lt.u32(active, col, D);
            if active {
                add.u32(at, vbase, col);
                load(own[j], qkv, at);
                add.u32(at, anchor, col);
                load(first[j], qkv, at);
                add.u32(at, outbase, col);
                load(change[j], delta, at);
                load(gradient[j], dy, at);
            }
            let complex_active: pred;
            and.pred(complex_active, active, complex_mode);
            if complex_active {
                load_j::<D>(jown[j], qkv, vbase, col);
                load_j::<D>(jgradient[j], dy, outbase, col);
                load_j::<D>(jchange[j], delta, outbase, col);
            }
            if first_query {
                mov.f64(gradient[j], 0.0f64);
                mov.f64(jgradient[j], 0.0f64);
            }
            mul.rn.f64(z, own[j], own[j]);
            add.rn.f64(norm, norm, z);
        }
        sum(norm);
        sqrt.rn.f64(norm, norm);
        max.f64(den, norm, 1e-12f64);
        mov.f64(a, 0.0f64);
        mov.f64(beta, 0.0f64);
        mov.f64(gv, 0.0f64);
        let (c, eta): (f64, f64);
        mov.f64(c, 0.0f64);
        mov.f64(eta, 0.0f64);
        for j in 0..(D + 31) / 32 {
            div.rn.f64(u, own[j], den);
            mul.rn.f64(z, gradient[j], u);
            add.rn.f64(a, a, z);
            mul.rn.f64(z, change[j], u);
            add.rn.f64(beta, beta, z);
            mul.rn.f64(z, gradient[j], own[j]);
            add.rn.f64(gv, gv, z);
            div.rn.f64(u, jown[j], den);
            mul.rn.f64(z, gradient[j], u);
            add.rn.f64(c, c, z);
            mul.rn.f64(z, change[j], u);
            add.rn.f64(eta, eta, z);
        }
        sum(a);
        sum(beta);
        sum(gv);
        sum(c);
        sum(eta);
        let above: pred;
        setp.ge.f64(above, norm, 1e-12f64);
        mov.f64(rem, 0.0f64);
        if !above {
            div.rn.f64(rem, norm, den);
            mul.rn.f64(rem, rem, rem);
            sub.rn.f64(rem, 1.0f64, rem);
        }
        for j in 0..(D + 31) / 32 {
            div.rn.f64(u, own[j], den);
            mul.rn.f64(z, a, u);
            sub.rn.f64(dchange[j], gradient[j], z);
            div.rn.f64(z, jown[j], den);
            mul.rn.f64(z, c, z);
            sub.rn.f64(dchange[j], dchange[j], z);
            mul.rn.f64(s, a, change[j]);
            mul.rn.f64(z, beta, gradient[j]);
            add.rn.f64(s, s, z);
            neg.f64(s, s);
            mul.rn.f64(z, eta, jgradient[j]);
            add.rn.f64(s, s, z);
            mul.rn.f64(z, c, jchange[j]);
            add.rn.f64(s, s, z);
            if above {
                mul.rn.f64(z, a, beta);
                let extra: f64;
                mul.rn.f64(extra, c, eta);
                add.rn.f64(z, z, extra);
                mul.rn.f64(z, z, 2.0f64);
                mul.rn.f64(z, z, u);
                add.rn.f64(s, s, z);
                div.rn.f64(s, s, den);
            } else {
                div.rn.f64(s, s, den);
                mul.rn.f64(z, rem, gradient[j]);
                add.rn.f64(s, s, z);
                mul.rn.f64(z, gv, own[j]);
                mul.rn.f64(z, z, 2.0f64);
                div.rn.f64(z, z, den);
                div.rn.f64(z, z, den);
                sub.rn.f64(s, s, z);
            }
            sub.rn.f64(s, s, dchange[j]);
            add.u32(col, lane, 32 * j);
            let active: pred;
            setp.lt.u32(active, col, D);
            if active {
                add.u32(at, outbase, col);
                store(gd, at, dchange[j]);
                store(dv, at, s);
            }
        }
        add.u32(at, t, 1);
        let gain32: f32;
        cvt.rn.f32.u32(gain32, at);
        lg2.approx.f32(gain32, gain32);
        mul.rn.f32(gain32, gain32, 0.6931471805599453 / (D as f32));
        cvt.f64.f32(gain, gain32);
        mov.f64(total, 0.0f64);
        mov.u32(key, 0);
        label(dprob);
        mov.f64(s, 0.0f64);
        for j in 0..(D + 31) / 32 {
            add.u32(col, lane, 32 * j);
            mov.f64(kval, 0.0f64);
            let active: pred;
            setp.lt.u32(active, col, D);
            if active {
                mul.lo.u32(at, key, 3);
                add.u32(at, at, 2);
                mad.lo.u32(at, at, w, base);
                mad.lo.u32(at, h, D, at);
                add.u32(at, at, col);
                load(kval, qkv, at);
            }
            sub.rn.f64(kval, kval, first[j]);
            mul.rn.f64(z, dchange[j], kval);
            add.rn.f64(s, s, z);
        }
        sum(s);
        add.u32(at, scorebase, key);
        load(p, prob, at);
        mul.rn.f64(z, p, s);
        add.rn.f64(total, total, z);
        let lead: pred;
        setp.eq.u32(lead, lane, 0);
        if lead {
            store(ds, at, s);
        }
        add.u32(key, key, 1);
        let more: pred;
        setp.le.u32(more, key, t);
        if more {
            bra(dprob);
        }
        bar.warp.sync(0xffffffff);
        mov.u32(key, 0);
        label(dscore);
        add.u32(at, scorebase, key);
        load(s, ds, at);
        load(p, prob, at);
        sub.rn.f64(s, s, total);
        mul.rn.f64(s, s, p);
        mul.rn.f64(dsval, s, gain);
        bar.warp.sync(0xffffffff);
        if lead {
            store(ds, at, dsval);
        }
        for j in 0..(D + 31) / 32 {
            add.u32(col, lane, 32 * j);
            mov.f64(kval, 0.0f64);
            let active: pred;
            setp.lt.u32(active, col, D);
            if active {
                mul.lo.u32(at, key, 3);
                add.u32(at, at, 1);
                mad.lo.u32(at, at, w, base);
                mad.lo.u32(at, h, D, at);
                add.u32(at, at, col);
                load(kval, qkv, at);
            }
            mul.rn.f64(z, dsval, kval);
            add.rn.f64(dquery[j], dquery[j], z);
        }
        add.u32(key, key, 1);
        setp.le.u32(more, key, t);
        if more {
            bra(dscore);
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
