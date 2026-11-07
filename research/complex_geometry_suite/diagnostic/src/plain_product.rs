//! Plain complex multiplication in the existing half-split hidden layout.
//! The only mathematical change from the radial FFN is omitting its sigmoid gate.
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
    fn index(i: u32) {
        let (b, t, n): (u32, u32, u32);
        mov.u32(b, ctaid.x);
        mov.u32(t, tid.x);
        mov.u32(n, ntid.x);
        mad.lo.u32(i, b, n, t);
    }
    pub unsafe extern "ptx-kernel" fn forward(y: *mut f32, x: *const f32, pairs: u32, count: u32) {
        let (i, row, c, at, j, out): (u32, u32, u32, u32, u32, u32);
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
        let (a, b, u, v, re, im, t): (f32, f32, f32, f32, f32, f32, f32);
        load(a, x, at);
        add.u32(j, at, pairs);
        load(b, x, j);
        add.u32(j, j, pairs);
        load(u, x, j);
        add.u32(j, j, pairs);
        load(v, x, j);
        mul.rn.f32(re, a, u);
        mul.rn.f32(t, b, v);
        sub.rn.f32(re, re, t);
        mul.rn.f32(im, a, v);
        mul.rn.f32(t, b, u);
        add.rn.f32(im, im, t);
        mul.rn.f32(re, re, 0.7071067811865476);
        mul.rn.f32(im, im, 0.7071067811865476);
        mul.lo.u32(out, row, pairs);
        mul.lo.u32(out, out, 2);
        add.u32(out, out, c);
        store(y, out, re);
        add.u32(out, out, pairs);
        store(y, out, im);
    }
    pub unsafe extern "ptx-kernel" fn backward(
        dx: *mut f32,
        x: *const f32,
        dy: *const f32,
        pairs: u32,
        count: u32,
    ) {
        let (i, row, c, at, j, out): (u32, u32, u32, u32, u32, u32);
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
        let (a, b, u, v, gr, gi, da, db, du, dv, t): (
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
        add.u32(j, at, pairs);
        load(b, x, j);
        add.u32(j, j, pairs);
        load(u, x, j);
        add.u32(j, j, pairs);
        load(v, x, j);
        mul.lo.u32(out, row, pairs);
        mul.lo.u32(out, out, 2);
        add.u32(out, out, c);
        load(gr, dy, out);
        add.u32(out, out, pairs);
        load(gi, dy, out);
        mul.rn.f32(gr, gr, 0.7071067811865476);
        mul.rn.f32(gi, gi, 0.7071067811865476);
        mul.rn.f32(da, gr, u);
        mul.rn.f32(t, gi, v);
        add.rn.f32(da, da, t);
        mul.rn.f32(db, gi, u);
        mul.rn.f32(t, gr, v);
        sub.rn.f32(db, db, t);
        mul.rn.f32(du, gr, a);
        mul.rn.f32(t, gi, b);
        add.rn.f32(du, du, t);
        mul.rn.f32(dv, gi, a);
        mul.rn.f32(t, gr, b);
        sub.rn.f32(dv, dv, t);
        store(dx, at, da);
        add.u32(at, at, pairs);
        store(dx, at, db);
        add.u32(at, at, pairs);
        store(dx, at, du);
        add.u32(at, at, pairs);
        store(dx, at, dv);
    }
}
