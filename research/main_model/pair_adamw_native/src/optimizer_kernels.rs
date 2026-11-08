//! Contracted/clipped coordinate AdamW, adapted from quartic_muonmd/md_ops.rs.
//! User-selected beta1=.9, beta2=.95; all complex ties are repaired exactly.
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
    fn sum64(v: f32) {
        let x: f32;
        for j in 0..5 {
            shfl.sync.bfly.b32(x, v, 1 << j, 31, 0xffffffff);
            add.rn.f32(v, v, x);
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
    /// One warp per256 real values, FP32 accumulation of squares.
    pub unsafe extern "ptx-kernel" fn norm_partial(out: *mut f32, x: *const f32, count: u32) {
        let (b, t, i): (u32, u32, u32);
        mov.u32(b, ctaid.x);
        mov.u32(t, tid.x);
        mad.lo.u32(i, b, 256, t);
        let (s, d): (f32, f32);
        let v: f32;
        mov.f32(s, 0f32);
        let valid: pred;
        for j in 0..8 {
            mov.f32(v, 0.0);
            setp.lt.u32(valid, i, count);
            if valid {
                load(v, x, i);
            }
            mov.f32(d, v);
            fma.rn.f32(s, d, d, s);
            add.u32(i, i, 32);
        }
        sum64(s);
        setp.eq.u32(valid, t, 0);
        let p: u64;
        mad.wide.u32(p, b, 4, out);
        if valid {
            st.global.f32([p], s);
        }
    }
    pub unsafe extern "ptx-kernel" fn norm_finish(out: *mut f32, parts: *const f32, count: u32) {
        let (i, t): (u32, u32);
        mov.u32(t, tid.x);
        mov.u32(i, t);
        let (s, d): (f32, f32);
        mov.f32(s, 0f32);
        let past: pred;
        let p: u64;
        label(next);
        setp.ge.u32(past, i, count);
        if past {
            bra(done);
        }
        mad.wide.u32(p, i, 4, parts);
        ld.global.f32(d, [p]);
        add.rn.f32(s, s, d);
        add.u32(i, i, 32);
        bra(next);
        label(done);
        sum64(s);
        sqrt.rn.f32(s, s);
        let v: f32;
        mov.f32(v, s);
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
        mul.rn.f32(v, v, 0.95);
        fma.rn.f32(v, 0.05, t, v);
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
    fn adam_pair(
        pa: f32,
        pb: f32,
        ma: f32,
        mb: f32,
        va: f32,
        vb: f32,
        ga: f32,
        gb: f32,
        lr: f32,
        decay: f32,
        bc1: f32,
        bc2: f32,
    ) {
        let (a, b, z, step, shrink): (f32, f32, f32, f32, f32);
        sub.rn.f32(a, ga, ma);
        fma.rn.f32(ma, 0.1, a, ma);
        sub.rn.f32(b, gb, mb);
        fma.rn.f32(mb, 0.1, b, mb);
        mul.rn.f32(a, ga, ga);
        fma.rn.f32(a, gb, gb, a);
        mul.rn.f32(a, a, 0.5);
        mul.rn.f32(va, va, 0.95);
        fma.rn.f32(va, 0.05, a, va);
        mov.f32(vb, va);
        sqrt.rn.f32(z, va);
        div.rn.f32(z, z, bc2);
        add.rn.f32(z, z, 1e-8);
        div.rn.f32(step, lr, bc1);
        mul.rn.f32(a, step, ma);
        div.rn.f32(a, a, z);
        mul.rn.f32(b, step, mb);
        div.rn.f32(b, b, z);
        mul.rn.f32(shrink, lr, decay);
        sub.rn.f32(shrink, 1.0, shrink);
        mul.rn.f32(pa, pa, shrink);
        sub.rn.f32(pa, pa, a);
        mul.rn.f32(pb, pb, shrink);
        sub.rn.f32(pb, pb, b);
    }
    /// Owners before complex_count are true complex pairs. The remaining
    /// packed router coordinates retain independent real AdamW variances.
    pub unsafe extern "ptx-kernel" fn adam_pair_coordinates(
        p: *mut f32,
        m: *mut f32,
        v: *mut f32,
        g: *const f32,
        map: *const u32,
        norm: *const f32,
        count: u32,
        complex_count: u32,
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
        let shared: pred;
        setp.lt.u32(shared, i, complex_count);
        if shared {
            adam_pair(pa, pb, ma, mb, va, vb, ga, gb, lr, decay, bc1, bc2);
        } else {
            adam(pa, ma, va, ga, lr, decay, bc1, bc2);
            adam(pb, mb, vb, gb, lr, decay, bc1, bc2);
        }
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
}
