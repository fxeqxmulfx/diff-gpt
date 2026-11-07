//! Kernite PTX: phase-adding products and AdamW in independent complex coordinates.
//! Radial gate is g*sigmoid(sqrt(|g|²+eps)); this is not analytic complex SiLU.

#[kernite_macros::kernels]
pub mod kernels {
    fn load(x: f32, base: u64, i: u32) {
        let p: u64;
        mad.wide.u32(p, i, 4, base);
        ld.global.f32(x, [p]);
    }
    fn store(base: u64, i: u32, x: f32) {
        let p: u64;
        mad.wide.u32(p, i, 4, base);
        st.global.f32([p], x);
    }
    fn load_u(x: u32, base: u64, i: u32) {
        let p: u64;
        mad.wide.u32(p, i, 4, base);
        ld.global.u32(x, [p]);
    }
    fn index(i: u32) {
        let (b, t, n): (u32, u32, u32);
        mov.u32(b, ctaid.x);
        mov.u32(t, tid.x);
        mov.u32(n, ntid.x);
        mad.lo.u32(i, b, n, t);
    }
    fn gate(s: f32, radius: f32, a: f32, b: f32) {
        let t: f32;
        mul.rn.f32(radius, a, a);
        mul.rn.f32(t, b, b);
        add.rn.f32(radius, radius, t);
        add.rn.f32(radius, radius, 0.00000011920928955078125);
        sqrt.rn.f32(radius, radius);
        mul.rn.f32(s, radius, -1.4426950408889634);
        ex2.approx.f32(s, s);
        add.rn.f32(s, s, 1.0);
        div.rn.f32(s, 1.0, s);
    }
    #[instances(false, true)]
    pub unsafe extern "ptx-kernel" fn product<const CROSS: bool>(
        y: *mut f32,
        x: *const f32,
        pairs: u32,
        count: u32,
    ) {
        let (i, row, c, at, at2, out): (u32, u32, u32, u32, u32, u32);
        index(i);
        let past: pred;
        setp.ge.u32(past, i, count);
        if past {
            ret();
        }
        div.u32(row, i, pairs);
        rem.u32(c, i, pairs);
        mul.lo.u32(at, row, pairs);
        mul.lo.u32(at, at, 4);
        add.u32(at, at, c);
        let (a, b, u, v, s, radius, re, im, temp): (f32, f32, f32, f32, f32, f32, f32, f32, f32);
        load(a, x, at);
        add.u32(at2, at, pairs);
        load(b, x, at2);
        add.u32(at2, at2, pairs);
        load(u, x, at2);
        add.u32(at2, at2, pairs);
        load(v, x, at2);
        gate(s, radius, a, b);
        mul.rn.f32(re, a, u);
        mul.rn.f32(im, b, v);
        if CROSS {
            sub.rn.f32(re, re, im);
            mul.rn.f32(im, a, v);
            mul.rn.f32(temp, b, u);
            add.rn.f32(im, im, temp);
            mul.rn.f32(s, s, 0.7071067811865476);
        }
        mul.rn.f32(re, re, s);
        mul.rn.f32(im, im, s);
        mul.lo.u32(out, row, pairs);
        mul.lo.u32(out, out, 2);
        add.u32(out, out, c);
        store(y, out, re);
        add.u32(out, out, pairs);
        store(y, out, im);
    }

    #[instances(false, true)]
    pub unsafe extern "ptx-kernel" fn product_bwd<const CROSS: bool>(
        dx: *mut f32,
        x: *const f32,
        dy: *const f32,
        pairs: u32,
        count: u32,
    ) {
        let (i, row, c, at, at2, out): (u32, u32, u32, u32, u32, u32);
        index(i);
        let past: pred;
        setp.ge.u32(past, i, count);
        if past {
            ret();
        }
        div.u32(row, i, pairs);
        rem.u32(c, i, pairs);
        mul.lo.u32(at, row, pairs);
        mul.lo.u32(at, at, 4);
        add.u32(at, at, c);
        let (a, b, u, v, s, radius, gr, gi, da, db, du, dv, ds, t, re, im): (
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
            f32,
            f32,
        );
        load(a, x, at);
        add.u32(at2, at, pairs);
        load(b, x, at2);
        add.u32(at2, at2, pairs);
        load(u, x, at2);
        add.u32(at2, at2, pairs);
        load(v, x, at2);
        mul.lo.u32(out, row, pairs);
        mul.lo.u32(out, out, 2);
        add.u32(out, out, c);
        load(gr, dy, out);
        add.u32(out, out, pairs);
        load(gi, dy, out);
        gate(s, radius, a, b);
        if CROSS {
            mul.rn.f32(gr, gr, 0.7071067811865476);
            mul.rn.f32(gi, gi, 0.7071067811865476);
        }
        mul.rn.f32(da, gr, u);
        mul.rn.f32(db, gi, v);
        mul.rn.f32(du, gr, a);
        mul.rn.f32(dv, gi, b);
        mul.rn.f32(re, a, u);
        mul.rn.f32(im, b, v);
        if CROSS {
            mul.rn.f32(t, gi, v);
            add.rn.f32(da, da, t);
            mul.rn.f32(db, gi, u);
            mul.rn.f32(t, gr, v);
            sub.rn.f32(db, db, t);
            mul.rn.f32(t, gi, b);
            add.rn.f32(du, du, t);
            mul.rn.f32(dv, gi, a);
            mul.rn.f32(t, gr, b);
            sub.rn.f32(dv, dv, t);
            sub.rn.f32(re, re, im);
            mul.rn.f32(im, a, v);
            mul.rn.f32(t, b, u);
            add.rn.f32(im, im, t);
        }
        mul.rn.f32(ds, gr, re);
        mul.rn.f32(t, gi, im);
        add.rn.f32(ds, ds, t);
        sub.rn.f32(t, 1.0, s);
        mul.rn.f32(ds, ds, t);
        mul.rn.f32(ds, ds, s);
        div.rn.f32(ds, ds, radius);
        mul.rn.f32(da, da, s);
        mul.rn.f32(t, ds, a);
        add.rn.f32(da, da, t);
        mul.rn.f32(db, db, s);
        mul.rn.f32(t, ds, b);
        add.rn.f32(db, db, t);
        mul.rn.f32(du, du, s);
        mul.rn.f32(dv, dv, s);
        store(dx, at, da);
        add.u32(at, at, pairs);
        store(dx, at, db);
        add.u32(at, at, pairs);
        store(dx, at, du);
        add.u32(at, at, pairs);
        store(dx, at, dv);
    }

    fn moment(
        pi: f32,
        mi: f32,
        vi: f32,
        gi: f32,
        square: f32,
        lr: f32,
        decay: f32,
        bc1: f32,
        bc2root: f32,
    ) {
        let (z, t): (f32, f32);
        sub.rn.f32(z, gi, mi);
        fma.rn.f32(mi, 0.1, z, mi);
        mul.rn.f32(vi, vi, 0.98);
        fma.rn.f32(vi, 0.02, square, vi);
        sqrt.rn.f32(z, vi);
        div.rn.f32(z, z, bc2root);
        add.rn.f32(z, z, 1e-8);
        div.rn.f32(t, lr, bc1);
        mul.rn.f32(t, t, mi);
        div.rn.f32(t, t, z);
        mul.rn.f32(z, lr, decay);
        sub.rn.f32(z, 1.0, z);
        mul.rn.f32(pi, pi, z);
        sub.rn.f32(pi, pi, t);
    }

    /// Each owner updates one independent complex parameter, writing its exact real expansion.
    #[instances(false, true)]
    pub unsafe extern "ptx-kernel" fn adamw<const PHASE: bool>(
        p: *mut f32,
        m: *mut f32,
        v: *mut f32,
        g: *const f32,
        map: *const u32,
        count: u32,
        lr: f32,
        decay: f32,
        bc1: f32,
        bc2root: f32,
    ) {
        let (i, at, ar, bi, br, ai, kind): (u32, u32, u32, u32, u32, u32, u32);
        index(i);
        let past: pred;
        setp.ge.u32(past, i, count);
        if past {
            ret();
        }
        mul.lo.u32(at, i, 5);
        load_u(ar, map, at);
        add.u32(at, at, 1);
        load_u(bi, map, at);
        add.u32(at, at, 1);
        load_u(br, map, at);
        add.u32(at, at, 1);
        load_u(ai, map, at);
        add.u32(at, at, 1);
        load_u(kind, map, at);
        let (ga, gb, pa, pb, ma, mb, va, vb, sa, sb, t): (
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
        load(ga, g, ar);
        load(gb, g, bi);
        let matrix: pred;
        setp.eq.u32(matrix, kind, 0);
        if matrix {
            load(t, g, ai);
            add.rn.f32(ga, ga, t);
            load(t, g, br);
            sub.rn.f32(gb, gb, t);
        }
        mul.rn.f32(sa, ga, ga);
        mul.rn.f32(sb, gb, gb);
        if PHASE {
            add.rn.f32(sa, sa, sb);
            mul.rn.f32(sa, sa, 0.5);
            mov.f32(sb, sa);
        }
        load(pa, p, ar);
        load(pb, p, bi);
        load(ma, m, ar);
        load(mb, m, bi);
        load(va, v, ar);
        load(vb, v, bi);
        moment(pa, ma, va, ga, sa, lr, decay, bc1, bc2root);
        moment(pb, mb, vb, gb, sb, lr, decay, bc1, bc2root);
        store(p, ar, pa);
        store(p, bi, pb);
        store(m, ar, ma);
        store(m, bi, mb);
        store(v, ar, va);
        store(v, bi, vb);
        if matrix {
            neg.f32(pb, pb);
            neg.f32(mb, mb);
            store(p, ai, pa);
            store(p, br, pb);
            store(m, ai, ma);
            store(m, br, mb);
            store(v, ai, va);
            store(v, br, vb);
        }
    }
}
