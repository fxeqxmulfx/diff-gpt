//! Opt-in XSA adjoint with FP32 geometry and dot-product accumulation.
//! FP32 activations/probabilities and outputs retain the original interface.
//! This reduces cancellation inside XSA; upstream FP32 boundaries remain.
#[kernite_macros::kernels]
pub mod kernels {
    fn load(v: f32, p: u64, i: u32) {
        let a: u64;
        let f: f32;
        mad.wide.u32(a, i, 4, p);
        ld.global.f32(f, [a]);
        mov.f32(v, f);
    }
    fn store(p: u64, i: u32, v: f32) {
        let a: u64;
        let f: f32;
        mov.f32(f, v);
        mad.wide.u32(a, i, 4, p);
        st.global.f32([a], f);
    }
    fn sum(v: f32) {
        let x: f32;
        for j in 0..5 {
            shfl.sync.bfly.b32(x, v, 1 << j, 31, 0xffffffff);
            add.rn.f32(v, v, x);
        }
    }
    fn load_j<const D: usize>(v: f32, p: u64, base: u32, col: u32) {
        let (partner, other, at): (u32, u32, u32);
        let negative: f32;
        let real: pred;
        setp.lt.u32(real, col, D / 2);
        add.u32(partner, col, D / 2);
        sub.u32(other, col, D / 2);
        selp.u32(partner, partner, other, real);
        add.u32(at, base, partner);
        load(v, p, at);
        neg.f32(negative, v);
        selp.f32(v, negative, v, real);
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
            [f32; (D + 31) / 32],
            [f32; (D + 31) / 32],
            [f32; (D + 31) / 32],
            [f32; (D + 31) / 32],
            [f32; (D + 31) / 32],
            [f32; (D + 31) / 32],
            [f32; (D + 31) / 32],
            [f32; (D + 31) / 32],
            [f32; (D + 31) / 32],
        );
        let (norm, den, a, beta, z, u, gv, rem, gain, p, s, total, kval, dsval): (
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
        );
        mov.f32(norm, 0.0f32);
        let complex_mode: pred;
        setp.ne.u32(complex_mode, hermitian, 0);
        let first_query: pred;
        setp.eq.u32(first_query, t, 0);
        for j in 0..(D + 31) / 32 {
            add.u32(col, lane, 32 * j);
            mov.f32(own[j], 0.0f32);
            mov.f32(change[j], 0.0f32);
            mov.f32(gradient[j], 0.0f32);
            mov.f32(first[j], 0.0f32);
            mov.f32(dquery[j], 0.0f32);
            mov.f32(jown[j], 0.0f32);
            mov.f32(jgradient[j], 0.0f32);
            mov.f32(jchange[j], 0.0f32);
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
                mov.f32(gradient[j], 0.0f32);
                mov.f32(jgradient[j], 0.0f32);
            }
            mul.rn.f32(z, own[j], own[j]);
            add.rn.f32(norm, norm, z);
        }
        sum(norm);
        sqrt.rn.f32(norm, norm);
        max.f32(den, norm, 1e-12f32);
        mov.f32(a, 0.0f32);
        mov.f32(beta, 0.0f32);
        mov.f32(gv, 0.0f32);
        let (c, eta): (f32, f32);
        mov.f32(c, 0.0f32);
        mov.f32(eta, 0.0f32);
        for j in 0..(D + 31) / 32 {
            div.rn.f32(u, own[j], den);
            mul.rn.f32(z, gradient[j], u);
            add.rn.f32(a, a, z);
            mul.rn.f32(z, change[j], u);
            add.rn.f32(beta, beta, z);
            mul.rn.f32(z, gradient[j], own[j]);
            add.rn.f32(gv, gv, z);
            div.rn.f32(u, jown[j], den);
            mul.rn.f32(z, gradient[j], u);
            add.rn.f32(c, c, z);
            mul.rn.f32(z, change[j], u);
            add.rn.f32(eta, eta, z);
        }
        sum(a);
        sum(beta);
        sum(gv);
        sum(c);
        sum(eta);
        let above: pred;
        setp.ge.f32(above, norm, 1e-12f32);
        mov.f32(rem, 0.0f32);
        if !above {
            div.rn.f32(rem, norm, den);
            mul.rn.f32(rem, rem, rem);
            sub.rn.f32(rem, 1.0f32, rem);
        }
        for j in 0..(D + 31) / 32 {
            div.rn.f32(u, own[j], den);
            mul.rn.f32(z, a, u);
            sub.rn.f32(dchange[j], gradient[j], z);
            div.rn.f32(z, jown[j], den);
            mul.rn.f32(z, c, z);
            sub.rn.f32(dchange[j], dchange[j], z);
            mul.rn.f32(s, a, change[j]);
            mul.rn.f32(z, beta, gradient[j]);
            add.rn.f32(s, s, z);
            neg.f32(s, s);
            mul.rn.f32(z, eta, jgradient[j]);
            add.rn.f32(s, s, z);
            mul.rn.f32(z, c, jchange[j]);
            add.rn.f32(s, s, z);
            if above {
                mul.rn.f32(z, a, beta);
                let extra: f32;
                mul.rn.f32(extra, c, eta);
                add.rn.f32(z, z, extra);
                mul.rn.f32(z, z, 2.0f32);
                mul.rn.f32(z, z, u);
                add.rn.f32(s, s, z);
                div.rn.f32(s, s, den);
            } else {
                div.rn.f32(s, s, den);
                mul.rn.f32(z, rem, gradient[j]);
                add.rn.f32(s, s, z);
                mul.rn.f32(z, gv, own[j]);
                mul.rn.f32(z, z, 2.0f32);
                div.rn.f32(z, z, den);
                div.rn.f32(z, z, den);
                sub.rn.f32(s, s, z);
            }
            sub.rn.f32(s, s, dchange[j]);
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
        mov.f32(gain, gain32);
        mov.f32(total, 0.0f32);
        mov.u32(key, 0);
        label(dprob);
        mov.f32(s, 0.0f32);
        for j in 0..(D + 31) / 32 {
            add.u32(col, lane, 32 * j);
            mov.f32(kval, 0.0f32);
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
            sub.rn.f32(kval, kval, first[j]);
            mul.rn.f32(z, dchange[j], kval);
            add.rn.f32(s, s, z);
        }
        sum(s);
        add.u32(at, scorebase, key);
        load(p, prob, at);
        mul.rn.f32(z, p, s);
        add.rn.f32(total, total, z);
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
        sub.rn.f32(s, s, total);
        mul.rn.f32(s, s, p);
        mul.rn.f32(dsval, s, gain);
        bar.warp.sync(0xffffffff);
        if lead {
            store(ds, at, dsval);
        }
        for j in 0..(D + 31) / 32 {
            add.u32(col, lane, 32 * j);
            mov.f32(kval, 0.0f32);
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
            mul.rn.f32(z, dsval, kval);
            add.rn.f32(dquery[j], dquery[j], z);
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
