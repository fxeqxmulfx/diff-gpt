//! Complex MuonMD kernels. See PLAN.txt for the complex extension of paper2606.25971.
//! Independent coordinates and real Hermitian derivatives; no real-block optimizer.
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
    fn sum64(x: f64) {
        let (a, b, c, d): (u32, u32, u32, u32);
        let y: f64;
        for j in 0..5 {
            mov.b64((a, b), x);
            shfl.sync.bfly.b32(c, a, 1 << j, 31, 0xffffffff);
            shfl.sync.bfly.b32(d, b, 1 << j, 31, 0xffffffff);
            mov.b64(y, (c, d));
            add.rn.f64(x, x, y);
        }
    }
    fn owners(ar: u32, bi: u32, br: u32, ai: u32, kind: u32, map: u64, i: u32) {
        let at: u32;
        mul.lo.u32(at, i, 5);
        integer(ar, map, at);
        add.u32(at, at, 1);
        integer(bi, map, at);
        add.u32(at, at, 1);
        integer(br, map, at);
        add.u32(at, at, 1);
        integer(ai, map, at);
        add.u32(at, at, 1);
        integer(kind, map, at);
    }
    pub unsafe extern "ptx-kernel" fn contract(
        g: *mut f32,
        expanded: *const f32,
        map: *const u32,
        count: u32,
    ) {
        let (i, ar, bi, br, ai, kind, at): (u32, u32, u32, u32, u32, u32, u32);
        index(i);
        let past: pred;
        setp.ge.u32(past, i, count);
        if past {
            ret();
        }
        owners(ar, bi, br, ai, kind, map, i);
        let (a, b, t): (f32, f32, f32);
        load(a, expanded, ar);
        load(b, expanded, bi);
        let matrix: pred;
        setp.eq.u32(matrix, kind, 0);
        if matrix {
            load(t, expanded, ai);
            add.rn.f32(a, a, t);
            load(t, expanded, br);
            sub.rn.f32(b, b, t);
        }
        mul.lo.u32(at, i, 2);
        put(g, at, a);
        add.u32(at, at, 1);
        put(g, at, b);
    }
    /// One warp per256 real values, FP64 accumulation of squares.
    pub unsafe extern "ptx-kernel" fn norm_partial(out: *mut f64, x: *const f32, count: u32) {
        let (b, t, i): (u32, u32, u32);
        mov.u32(b, ctaid.x);
        mov.u32(t, tid.x);
        mad.lo.u32(i, b, 256, t);
        let (s, d): (f64, f64);
        let v: f32;
        mov.f64(s, 0f64);
        let valid: pred;
        for j in 0..8 {
            mov.f32(v, 0.0);
            setp.lt.u32(valid, i, count);
            if valid {
                load(v, x, i);
            }
            cvt.f64.f32(d, v);
            fma.rn.f64(s, d, d, s);
            add.u32(i, i, 32);
        }
        sum64(s);
        setp.eq.u32(valid, t, 0);
        let p: u64;
        mad.wide.u32(p, b, 8, out);
        if valid {
            st.global.f64([p], s);
        }
    }
    pub unsafe extern "ptx-kernel" fn norm_finish(out: *mut f32, parts: *const f64, count: u32) {
        let (i, t): (u32, u32);
        mov.u32(t, tid.x);
        mov.u32(i, t);
        let (s, d): (f64, f64);
        mov.f64(s, 0f64);
        let past: pred;
        let p: u64;
        label(next);
        setp.ge.u32(past, i, count);
        if past {
            bra(done);
        }
        mad.wide.u32(p, i, 8, parts);
        ld.global.f64(d, [p]);
        add.rn.f64(s, s, d);
        add.u32(i, i, 32);
        bra(next);
        label(done);
        sum64(s);
        sqrt.rn.f64(s, s);
        let v: f32;
        cvt.rn.f32.f64(v, s);
        setp.eq.u32(past, t, 0);
        if past {
            st.global.f32([out], v);
        }
    }
    fn adam(p: f32, m: f32, v: f32, g: f32, lr: f32, decay: f32, bc1: f32, bc2: f32) {
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
        mul.rn.f32(z, lr, decay);
        sub.rn.f32(z, 1.0, z);
        mul.rn.f32(p, p, z);
        sub.rn.f32(p, p, t);
    }
    pub unsafe extern "ptx-kernel" fn adam_coordinates(
        p: *mut f32,
        m: *mut f32,
        v: *mut f32,
        g: *const f32,
        map: *const u32,
        norm: *const f32,
        count: u32,
        lr: f32,
        decay: f32,
        bc1: f32,
        bc2: f32,
    ) {
        let (i, ar, bi, br, ai, kind, at): (u32, u32, u32, u32, u32, u32, u32);
        index(i);
        let past: pred;
        setp.ge.u32(past, i, count);
        if past {
            ret();
        }
        owners(ar, bi, br, ai, kind, map, i);
        let (ga, gb, pa, pb, ma, mb, va, vb, scale): (f32, f32, f32, f32, f32, f32, f32, f32, f32);
        ld.global.f32(scale, [norm]);
        max.f32(scale, scale, 1.0);
        mul.lo.u32(at, i, 2);
        load(ga, g, at);
        add.u32(at, at, 1);
        load(gb, g, at);
        div.rn.f32(ga, ga, scale);
        div.rn.f32(gb, gb, scale);
        load(pa, p, ar);
        load(pb, p, bi);
        load(ma, m, ar);
        load(mb, m, bi);
        load(va, v, ar);
        load(vb, v, bi);
        adam(pa, ma, va, ga, lr, decay, bc1, bc2);
        adam(pb, mb, vb, gb, lr, decay, bc1, bc2);
        put(p, ar, pa);
        put(p, bi, pb);
        put(m, ar, ma);
        put(m, bi, mb);
        put(v, ar, va);
        put(v, bi, vb);
        setp.eq.u32(past, kind, 0);
        if past {
            neg.f32(pb, pb);
            neg.f32(mb, mb);
            put(p, ai, pa);
            put(p, br, pb);
            put(m, ai, ma);
            put(m, br, mb);
            put(v, ai, va);
            put(v, br, vb);
        }
    }
    /// All gain signals and direction gradients are evaluated at the old factors.
    pub unsafe extern "ptx-kernel" fn prepare(
        d: *mut f32,
        moment: *mut f32,
        nu: *mut f32,
        signal: *mut f32,
        p: *const f32,
        g: *const f32,
        map: *const u32,
        gains: *const f32,
        norm: *const f32,
        offset: u32,
        rows: u32,
        cols: u32,
    ) {
        let (i, count, r, c, at, owner, ar, bi, br, ai, kind): (
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
        let (rg, cg, scale, a, b, ga, gb, ma, mb, t, s): (
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
        owners(ar, bi, br, ai, kind, map, owner);
        load(a, p, ar);
        load(b, p, bi);
        div.rn.f32(a, a, rg);
        div.rn.f32(b, b, rg);
        mul.lo.u32(at, i, 2);
        put(d, at, a);
        add.u32(at, at, 1);
        put(d, at, b);
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
        add.u32(at, at, 1);
        load(mb, moment, at);
        fma.rn.f32(ma, 0.95, ma, ga);
        fma.rn.f32(mb, 0.95, mb, gb);
        mul.lo.u32(at, i, 2);
        put(moment, at, ma);
        add.u32(at, at, 1);
        put(moment, at, mb);
        fma.rn.f32(ga, 0.95, ma, ga);
        fma.rn.f32(gb, 0.95, mb, gb);
        mul.lo.u32(at, i, 2);
        put(nu, at, ga);
        add.u32(at, at, 1);
        put(nu, at, gb);
    }
    /// Stable sigmoid and softplus; no artificial gain floor.
    pub unsafe extern "ptx-kernel" fn gains(
        raw: *mut f32,
        m: *mut f32,
        v: *mut f32,
        next: *mut f32,
        old: *const f32,
        signal: *const f32,
        rows: u32,
        cols: u32,
        lr: f32,
        bc1: f32,
        bc2: f32,
    ) {
        let (q, t, i, len, at, opposite): (u32, u32, u32, u32, u32, u32);
        mov.u32(q, ctaid.x);
        mov.u32(t, tid.x);
        mov.u32(i, t);
        let row: pred;
        setp.lt.u32(row, q, rows);
        selp.u32(len, cols, rows, row);
        let (s, h): (f64, f64);
        let (a, b, z, g, pa, ma, va): (f32, f32, f32, f32, f32, f32, f32);
        mov.f64(s, 0f64);
        let past: pred;
        label(next_entry);
        setp.ge.u32(past, i, len);
        if past {
            bra(reduce);
        }
        mad.lo.u32(at, q, cols, i);
        add.u32(opposite, rows, i);
        if !row {
            sub.u32(at, q, rows);
            mad.lo.u32(at, i, cols, at);
            mov.u32(opposite, i);
        }
        load(a, signal, at);
        load(b, old, opposite);
        mul.rn.f32(a, a, b);
        cvt.f64.f32(h, a);
        add.rn.f64(s, s, h);
        add.u32(i, i, 32);
        bra(next_entry);
        label(reduce);
        sum64(s);
        setp.ne.u32(past, t, 0);
        if past {
            ret();
        }
        load(pa, raw, q);
        load(ma, m, q);
        load(va, v, q);
        cvt.rn.f32.f64(g, s);
        abs.f32(z, pa);
        mul.rn.f32(z, z, -1.4426950408889634);
        ex2.approx.f32(z, z);
        add.rn.f32(b, z, 1.0);
        div.rn.f32(a, 1.0, b);
        div.rn.f32(z, z, b);
        setp.ge.f32(past, pa, 0.0);
        selp.f32(a, a, z, past);
        mul.rn.f32(g, g, a);
        mov.f32(z, 0.0);
        adam(pa, ma, va, g, lr, z, bc1, bc2);
        put(raw, q, pa);
        put(m, q, ma);
        put(v, q, va);
        abs.f32(z, pa);
        mul.rn.f32(z, z, -1.4426950408889634);
        ex2.approx.f32(z, z);
        // log1p(z) through its cubic series avoids cancellation for negative raw gains.
        setp.lt.f32(past, z, 0.0001);
        mul.rn.f32(b, z, 0.3333333333333333);
        sub.rn.f32(b, b, 0.5);
        fma.rn.f32(b, b, z, 1.0);
        mul.rn.f32(b, b, z);
        add.rn.f32(z, z, 1.0);
        lg2.approx.f32(z, z);
        mul.rn.f32(z, z, 0.6931471805599453);
        selp.f32(z, b, z, past);
        max.f32(a, pa, 0.0);
        add.rn.f32(z, z, a);
        put(next, q, z);
    }
    fn expand(x: u64, i: u32, rows: u32, cols: u32, a: f32, b: f32) {
        let (r, c, at, stride, bottom): (u32, u32, u32, u32, u32);
        let neg: f32;
        div.u32(r, i, cols);
        rem.u32(c, i, cols);
        mul.lo.u32(stride, cols, 2);
        mad.lo.u32(at, r, stride, c);
        put(x, at, a);
        add.u32(bottom, r, rows);
        mad.lo.u32(bottom, bottom, stride, c);
        put(x, bottom, b);
        add.u32(at, at, cols);
        neg.f32(neg, b);
        put(x, at, neg);
        add.u32(bottom, bottom, cols);
        put(x, bottom, a);
    }
    #[instances(false, true)]
    pub unsafe extern "ptx-kernel" fn pack<const TALL: bool>(
        x: *mut f32,
        nu: *const f32,
        norm: *const f32,
        rows: u32,
        cols: u32,
    ) {
        let (i, count, r, c, at): (u32, u32, u32, u32, u32);
        index(i);
        mul.lo.u32(count, rows, cols);
        let past: pred;
        setp.ge.u32(past, i, count);
        if past {
            ret();
        }
        mov.u32(at, i);
        if TALL {
            div.u32(r, i, cols);
            rem.u32(c, i, cols);
            mad.lo.u32(at, c, rows, r);
        }
        mul.lo.u32(at, at, 2);
        let (a, b, s): (f32, f32, f32);
        load(a, nu, at);
        add.u32(at, at, 1);
        load(b, nu, at);
        if TALL {
            neg.f32(b, b);
        }
        ld.global.f32(s, [norm]);
        setp.eq.f32(past, s, 0.0);
        selp.f32(s, 1.0, s, past);
        div.rn.f32(a, a, s);
        div.rn.f32(b, b, s);
        expand(x, i, rows, cols, a, b);
    }
    pub unsafe extern "ptx-kernel" fn polynomial(b: *mut f32, a: *const f32, count: u32) {
        let i: u32;
        index(i);
        let past: pred;
        setp.ge.u32(past, i, count);
        if past {
            ret();
        }
        let (x, y): (f32, f32);
        load(x, a, i);
        load(y, b, i);
        mul.rn.f32(x, x, -4.775);
        fma.rn.f32(y, 2.0315, y, x);
        put(b, i, y);
    }
    pub unsafe extern "ptx-kernel" fn ns_finish(x: *mut f32, y: *const f32, rows: u32, cols: u32) {
        let (i, count, r, c, at, stride): (u32, u32, u32, u32, u32, u32);
        index(i);
        mul.lo.u32(count, rows, cols);
        let past: pred;
        setp.ge.u32(past, i, count);
        if past {
            ret();
        }
        div.u32(r, i, cols);
        rem.u32(c, i, cols);
        mul.lo.u32(stride, cols, 2);
        mad.lo.u32(at, r, stride, c);
        let (a, b, t): (f32, f32, f32);
        load(a, x, at);
        load(t, y, at);
        fma.rn.f32(a, 3.4445, a, t);
        add.u32(r, r, rows);
        mad.lo.u32(at, r, stride, c);
        load(b, x, at);
        load(t, y, at);
        fma.rn.f32(b, 3.4445, b, t);
        expand(x, i, rows, cols, a, b);
    }
    #[instances(false, true)]
    pub unsafe extern "ptx-kernel" fn candidate<const TALL: bool>(
        out: *mut f32,
        d: *const f32,
        x: *const f32,
        rows: u32,
        cols: u32,
        rate: f32,
    ) {
        let (i, count, r, c, at, stride): (u32, u32, u32, u32, u32, u32);
        index(i);
        mul.lo.u32(count, rows, cols);
        let past: pred;
        setp.ge.u32(past, i, count);
        if past {
            ret();
        }
        if TALL {
            div.u32(c, i, rows);
            rem.u32(r, i, rows);
        }
        if !TALL {
            div.u32(r, i, cols);
            rem.u32(c, i, cols);
        }
        mul.lo.u32(stride, cols, 2);
        mad.lo.u32(at, r, stride, c);
        let (a, b, da, db): (f32, f32, f32, f32);
        load(a, x, at);
        add.u32(r, r, rows);
        mad.lo.u32(at, r, stride, c);
        load(b, x, at);
        if TALL {
            neg.f32(b, b);
        }
        mul.lo.u32(at, i, 2);
        load(da, d, at);
        add.u32(at, at, 1);
        load(db, d, at);
        mul.rn.f32(a, a, rate);
        mul.rn.f32(b, b, rate);
        sub.rn.f32(da, da, a);
        sub.rn.f32(db, db, b);
        mul.lo.u32(at, i, 2);
        put(out, at, da);
        add.u32(at, at, 1);
        put(out, at, db);
    }
    pub unsafe extern "ptx-kernel" fn fuse(
        p: *mut f32,
        candidate: *const f32,
        norm: *const f32,
        gains: *const f32,
        map: *const u32,
        offset: u32,
        rows: u32,
        cols: u32,
        radius: f32,
    ) {
        let (i, count, r, c, at, owner, ar, bi, br, ai, kind): (
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
        let (a, b, s, rg, cg, neg): (f32, f32, f32, f32, f32, f32);
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
        owners(ar, bi, br, ai, kind, map, owner);
        put(p, ar, a);
        put(p, ai, a);
        put(p, bi, b);
        neg.f32(neg, b);
        put(p, br, neg);
    }
}
