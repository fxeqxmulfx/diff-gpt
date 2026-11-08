//! Static FP32 score coefficients and additional Q/K/coordinate adjoints.
//! The ordinary attention adjoint supplies dL/dR (already including ln(N)/d).
#[kernite_macros::kernels]
pub mod kernels {
    fn load(v: f32, p: u64, i: u32) {
        let a: u64;
        mad.wide.u32(a, i, 4, p);
        ld.global.f32(v, [a]);
    }
    fn store(p: u64, i: u32, v: f32) {
        let a: u64;
        mad.wide.u32(a, i, 4, p);
        st.global.f32([a], v);
    }
    fn head_index(id: u32, lane: u32) {
        let warp: u32;
        mov.u32(id, ctaid.x);
        mov.u32(lane, tid.x);
        shr.u32(warp, lane, 5);
        mad.lo.u32(id, id, 4, warp);
        and.b32(lane, lane, 31);
    }
    // d Im(conj(q)*k)/dq = (k_im,-k_re), in the within-head layout.
    fn cross<const D: usize>(v: f32, p: u64, base: u32, col: u32) {
        let (at, other): (u32, u32);
        let real: pred;
        let negative: f32;
        setp.lt.u32(real, col, D / 2);
        add.u32(at, col, D / 2);
        sub.u32(other, col, D / 2);
        selp.u32(at, at, other, real);
        add.u32(at, base, at);
        load(v, p, at);
        neg.f32(negative, v);
        selp.f32(v, v, negative, real);
    }
    pub unsafe extern "ptx-kernel" fn coefficients(
        out: *mut f32,
        parameters: *const f32,
        heads: u32,
    ) {
        let (h, lane, at): (u32, u32, u32);
        mov.u32(h, ctaid.x);
        mov.u32(lane, tid.x);
        mad.lo.u32(h, h, 256, lane);
        let past: pred;
        setp.ge.u32(past, h, heads);
        if past {
            ret();
        }
        let (a, m, r, t, s, u, beta, derivative): (f32, f32, f32, f32, f32, f32, f32, f32);
        load(a, parameters, h);
        abs.f32(m, a);
        max.f32(m, m, 1.0);
        div.rn.f32(r, 1.0, m);
        div.rn.f32(t, a, m);
        mul.rn.f32(s, r, r);
        mul.rn.f32(u, t, t);
        add.rn.f32(s, s, u);
        sqrt.rn.f32(s, s);
        div.rn.f32(u, r, s);
        div.rn.f32(beta, t, s);
        mul.rn.f32(derivative, u, u);
        mul.rn.f32(derivative, derivative, u);
        mul.lo.u32(at, h, 2);
        store(out, at, beta);
        add.u32(at, at, 1);
        store(out, at, derivative);
    }
    #[instances(16, 32, 128)]
    pub unsafe extern "ptx-kernel" fn query_correction<const D: usize>(
        dqkv: *mut f32,
        parts: *mut f32,
        ds: *const f32,
        imaginary: *const f32,
        qkv: *const f32,
        coefficients: *const f32,
        heads: u32,
        length: u32,
        head_count: u32,
    ) {
        let (id, lane, h, b, t, w, base, qbase, scorebase, key, col, at, kbase): (
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
        head_index(id, lane);
        let past: pred;
        setp.ge.u32(past, id, head_count);
        if past {
            ret();
        }
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
        mad.lo.u32(scorebase, b, heads, h);
        mad.lo.u32(scorebase, scorebase, length, t);
        mul.lo.u32(scorebase, scorebase, length);
        let (beta, derivative, im, g, c, v, term, da): (f32, f32, f32, f32, f32, f32, f32, f32);
        mul.lo.u32(at, h, 2);
        load(beta, coefficients, at);
        add.u32(at, at, 1);
        load(derivative, coefficients, at);
        let grad: [f32; (D + 31) / 32];
        for j in 0..(D + 31) / 32 {
            mov.f32(grad[j], 0.0);
        }
        mov.f32(da, 0.0);
        mov.u32(key, 0);
        label(keys);
        add.u32(at, scorebase, key);
        load(im, imaginary, at);
        load(g, ds, at);
        mul.rn.f32(c, im, 1.0 / (D as f32));
        mul.rn.f32(term, g, c);
        mul.rn.f32(term, term, im);
        mul.rn.f32(term, term, derivative);
        add.rn.f32(da, da, term);
        mul.rn.f32(c, c, beta);
        mul.rn.f32(c, c, 2.0);
        mul.rn.f32(c, c, g);
        mul.lo.u32(kbase, key, 3);
        add.u32(kbase, kbase, 1);
        mad.lo.u32(kbase, kbase, w, base);
        mad.lo.u32(kbase, h, D, kbase);
        for j in 0..(D + 31) / 32 {
            add.u32(col, lane, 32 * j);
            let active: pred;
            setp.lt.u32(active, col, D);
            if active {
                cross::<D>(v, qkv, kbase, col);
                mul.rn.f32(term, c, v);
                add.rn.f32(grad[j], grad[j], term);
            }
        }
        add.u32(key, key, 1);
        let more: pred;
        setp.le.u32(more, key, t);
        if more {
            bra(keys);
        }
        for j in 0..(D + 31) / 32 {
            add.u32(col, lane, 32 * j);
            let active: pred;
            setp.lt.u32(active, col, D);
            if active {
                add.u32(at, qbase, col);
                load(v, dqkv, at);
                add.rn.f32(v, v, grad[j]);
                store(dqkv, at, v);
            }
        }
        let lead: pred;
        setp.eq.u32(lead, lane, 0);
        if lead {
            store(parts, id, da);
        }
    }
    #[instances(16, 32, 128)]
    pub unsafe extern "ptx-kernel" fn key_correction<const D: usize>(
        dqkv: *mut f32,
        ds: *const f32,
        imaginary: *const f32,
        qkv: *const f32,
        coefficients: *const f32,
        heads: u32,
        length: u32,
        head_count: u32,
    ) {
        let (id, lane, h, b, key, w, base, kbase, scorebase, query, col, at, qbase): (
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
        head_index(id, lane);
        let past: pred;
        setp.ge.u32(past, id, head_count);
        if past {
            ret();
        }
        rem.u32(h, id, heads);
        div.u32(key, id, heads);
        div.u32(b, key, length);
        rem.u32(key, key, length);
        mul.lo.u32(w, heads, D);
        mul.lo.u32(base, b, length);
        mul.lo.u32(base, base, 3);
        mul.lo.u32(base, base, w);
        mul.lo.u32(kbase, key, 3);
        add.u32(kbase, kbase, 1);
        mad.lo.u32(kbase, kbase, w, base);
        mad.lo.u32(kbase, h, D, kbase);
        mad.lo.u32(scorebase, b, heads, h);
        mul.lo.u32(scorebase, scorebase, length);
        let (beta, im, g, c, v, term): (f32, f32, f32, f32, f32, f32);
        mul.lo.u32(at, h, 2);
        load(beta, coefficients, at);
        let grad: [f32; (D + 31) / 32];
        for j in 0..(D + 31) / 32 {
            mov.f32(grad[j], 0.0);
        }
        mov.u32(query, key);
        label(queries);
        add.u32(at, scorebase, query);
        mad.lo.u32(at, at, length, key);
        load(im, imaginary, at);
        load(g, ds, at);
        mul.rn.f32(c, im, 1.0 / (D as f32));
        mul.rn.f32(c, c, beta);
        mul.rn.f32(c, c, -2.0);
        mul.rn.f32(c, c, g);
        mul.lo.u32(qbase, query, 3);
        mad.lo.u32(qbase, qbase, w, base);
        mad.lo.u32(qbase, h, D, qbase);
        for j in 0..(D + 31) / 32 {
            add.u32(col, lane, 32 * j);
            let active: pred;
            setp.lt.u32(active, col, D);
            if active {
                cross::<D>(v, qkv, qbase, col);
                mul.rn.f32(term, c, v);
                add.rn.f32(grad[j], grad[j], term);
            }
        }
        add.u32(query, query, 1);
        let more: pred;
        setp.lt.u32(more, query, length);
        if more {
            bra(queries);
        }
        for j in 0..(D + 31) / 32 {
            add.u32(col, lane, 32 * j);
            let active: pred;
            setp.lt.u32(active, col, D);
            if active {
                add.u32(at, kbase, col);
                load(v, dqkv, at);
                add.rn.f32(v, v, grad[j]);
                store(dqkv, at, v);
            }
        }
    }
    pub unsafe extern "ptx-kernel" fn coordinate_gradient(
        gradient: *mut f32,
        parts: *const f32,
        heads: u32,
        rows: u32,
    ) {
        let (h, lane, row, at): (u32, u32, u32, u32);
        mov.u32(h, ctaid.x);
        mov.u32(lane, tid.x);
        mad.lo.u32(h, h, 256, lane);
        let past: pred;
        setp.ge.u32(past, h, heads);
        if past {
            ret();
        }
        let (s, v): (f32, f32);
        mov.f32(s, 0.0);
        mov.u32(row, 0);
        label(next);
        mad.lo.u32(at, row, heads, h);
        load(v, parts, at);
        add.rn.f32(s, s, v);
        add.u32(row, row, 1);
        let more: pred;
        setp.lt.u32(more, row, rows);
        if more {
            bra(next);
        }
        store(gradient, h, s);
    }
}
