//! Native complex-coordinate AdamMD and the physical MDW correction.
//! Sources: paper2606.25971v2 Appendix A; Transformer.MDW.Defs and TrainingStorage.
#[kernite_macros::kernels]
pub mod kernels {
    fn load(x: f32, p: u64, i: u32) {
        let q: u64;
        mad.wide.u32(q, i, 4, p);
        ld.global.f32(x, [q]);
    }
    fn put(p: u64, i: u32, x: f32) {
        let q: u64;
        mad.wide.u32(q, i, 4, p);
        st.global.f32([q], x);
    }
    fn integer(x: u32, p: u64, i: u32) {
        let q: u64;
        mad.wide.u32(q, i, 4, p);
        ld.global.u32(x, [q]);
    }
    fn index(i: u32) {
        let (b, t, n): (u32, u32, u32);
        mov.u32(b, ctaid.x);
        mov.u32(t, tid.x);
        mov.u32(n, ntid.x);
        mad.lo.u32(i, b, n, t);
    }
    fn owners(ar: u32, bi: u32, br: u32, ai: u32, map: u64, i: u32) {
        let at: u32;
        mul.lo.u32(at, i, 5);
        integer(ar, map, at);
        add.u32(at, at, 1);
        integer(bi, map, at);
        add.u32(at, at, 1);
        integer(br, map, at);
        add.u32(at, at, 1);
        integer(ai, map, at);
    }
    fn adam(p: f32, m: f32, v: f32, g: f32, lr: f32, bc1: f32, bc2: f32) {
        let (t, z): (f32, f32);
        sub.rn.f32(t, g, m);
        fma.rn.f32(m, 0.1, t, m);
        mul.rn.f32(t, g, g);
        mul.rn.f32(v, v, 0.99);
        fma.rn.f32(v, 0.01, t, v);
        sqrt.rn.f32(z, v);
        div.rn.f32(z, z, bc2);
        add.rn.f32(z, z, 1e-8);
        div.rn.f32(t, lr, bc1);
        mul.rn.f32(t, t, m);
        div.rn.f32(t, t, z);
        sub.rn.f32(p, p, t);
    }
    pub unsafe extern "ptx-kernel" fn prepare_adam(
        candidate: *mut f32,
        moment: *mut f32,
        variance: *mut f32,
        signal: *mut f32,
        p: *const f32,
        g: *const f32,
        map: *const u32,
        gains: *const f32,
        norm: *const f32,
        offset: u32,
        rows: u32,
        cols: u32,
        lr: f32,
        bc1: f32,
        bc2: f32,
    ) {
        let (i, count, r, c, at, owner, ar, bi, br, ai): (
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
        index(i);
        mul.lo.u32(count, rows, cols);
        let past: pred;
        setp.ge.u32(past, i, count);
        if past {
            ret();
        }
        div.u32(r, i, cols);
        rem.u32(c, i, cols);
        add.u32(c, c, rows);
        let (rg, cg, scale, a, b, ga, gb, ma, mb, va, vb, t, s): (
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
        load(rg, gains, r);
        load(cg, gains, c);
        mul.rn.f32(rg, rg, cg);
        add.u32(owner, offset, i);
        owners(ar, bi, br, ai, map, owner);
        load(a, p, ar);
        load(b, p, bi);
        div.rn.f32(a, a, rg);
        div.rn.f32(b, b, rg);
        mul.lo.u32(at, owner, 2);
        load(ga, g, at);
        add.u32(at, at, 1);
        load(gb, g, at);
        ld.global.f32(scale, [norm]);
        max.f32(scale, scale, 1.0);
        div.rn.f32(ga, ga, scale);
        div.rn.f32(gb, gb, scale);
        mul.rn.f32(s, ga, a);
        mul.rn.f32(t, gb, b);
        add.rn.f32(s, s, t);
        put(signal, i, s);
        mul.rn.f32(ga, ga, rg);
        mul.rn.f32(gb, gb, rg);
        mul.lo.u32(at, i, 2);
        load(ma, moment, at);
        load(va, variance, at);
        add.u32(at, at, 1);
        load(mb, moment, at);
        load(vb, variance, at);
        adam(a, ma, va, ga, lr, bc1, bc2);
        adam(b, mb, vb, gb, lr, bc1, bc2);
        mul.lo.u32(at, i, 2);
        put(moment, at, ma);
        put(variance, at, va);
        put(candidate, at, a);
        add.u32(at, at, 1);
        put(moment, at, mb);
        put(variance, at, vb);
        put(candidate, at, b);
    }
    /// First form the MD proposal, then subtract rate*decay times the OLD physical weight.
    /// Scratch receives the direction recovered with the proposed gains, for storage repair.
    pub unsafe extern "ptx-kernel" fn fuse_decay(
        p: *mut f32,
        candidate: *mut f32,
        norm: *const f32,
        gains: *const f32,
        map: *const u32,
        offset: u32,
        rows: u32,
        cols: u32,
        radius: f32,
        decay_rate: f32,
    ) {
        let (i, count, r, c, at, owner, ar, bi, br, ai): (
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
        index(i);
        mul.lo.u32(count, rows, cols);
        let past: pred;
        setp.ge.u32(past, i, count);
        if past {
            ret();
        }
        div.u32(r, i, cols);
        rem.u32(c, i, cols);
        add.u32(c, c, rows);
        let (a, b, s, rg, cg, pa, pb, negative, t): (f32, f32, f32, f32, f32, f32, f32, f32, f32);
        ld.global.f32(s, [norm]);
        setp.eq.f32(past, s, 0.0);
        selp.f32(s, 1.0, s, past);
        div.rn.f32(s, radius, s);
        load(rg, gains, r);
        load(cg, gains, c);
        mul.rn.f32(s, s, rg);
        mul.rn.f32(s, s, cg);
        mul.lo.u32(at, i, 2);
        load(a, candidate, at);
        add.u32(at, at, 1);
        load(b, candidate, at);
        mul.rn.f32(a, a, s);
        mul.rn.f32(b, b, s);
        add.u32(owner, offset, i);
        owners(ar, bi, br, ai, map, owner);
        load(pa, p, ar);
        load(pb, p, bi);
        mul.rn.f32(t, decay_rate, pa);
        sub.rn.f32(a, a, t);
        mul.rn.f32(t, decay_rate, pb);
        sub.rn.f32(b, b, t);
        put(p, ar, a);
        put(p, ai, a);
        put(p, bi, b);
        neg.f32(negative, b);
        put(p, br, negative);
        mul.rn.f32(rg, rg, cg);
        div.rn.f32(a, a, rg);
        div.rn.f32(b, b, rg);
        mul.lo.u32(at, i, 2);
        put(candidate, at, a);
        add.u32(at, at, 1);
        put(candidate, at, b);
    }
    /// Leave physical weights, column gains and every adaptive history untouched.
    pub unsafe extern "ptx-kernel" fn repair_rows(
        raw: *mut f32,
        gains: *mut f32,
        norm: *const f32,
        rows: u32,
        radius: f32,
    ) {
        let i: u32;
        index(i);
        let past: pred;
        setp.ge.u32(past, i, rows);
        if past {
            ret();
        }
        let (s, g, z, t): (f32, f32, f32, f32);
        ld.global.f32(s, [norm]);
        setp.eq.f32(past, s, 0.0);
        if past {
            ret();
        }
        div.rn.f32(s, s, radius);
        load(g, gains, i);
        mul.rn.f32(g, g, s);
        put(gains, i, g);
        // inverse_softplus(g)=g+log(1-exp(-g)); small g uses its stable series.
        mul.rn.f32(z, g, -1.4426950408889634);
        ex2.approx.f32(z, z);
        sub.rn.f32(z, 1.0, z);
        lg2.approx.f32(z, z);
        mul.rn.f32(z, z, 0.6931471805599453);
        add.rn.f32(z, z, g);
        setp.lt.f32(past, g, 0.01);
        if past {
            lg2.approx.f32(z, g);
            mul.rn.f32(z, z, 0.6931471805599453);
            mul.rn.f32(t, g, g);
            mul.rn.f32(t, t, 0.041666666666666664);
            fma.rn.f32(t, 0.5, g, t);
            add.rn.f32(z, z, t);
        }
        put(raw, i, z);
    }
}
