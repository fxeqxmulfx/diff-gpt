//! Historical Q/K RMS + RoPE + 1/sqrt(D) attention; only real XSA is selected.
//! Hermitian XSA subtracts the real projections onto both v and Jv, where
//! J(re, im) = (-im, re). Its real adjoint differentiates both directions
//! and the common clipped norm; it is not a holomorphic derivative.
//! Independent row and key reductions avoid floating-point atomic updates.
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
    fn sum(v: f32) {
        let x: f32;
        for j in 0..5 {
            shfl.sync.bfly.b32(x, v, 1 << j, 31, 0xffffffff);
            add.rn.f32(v, v, x);
        }
    }
    fn sum64(v: f32) {
        let x: f32;
        for j in 0..5 {
            shfl.sync.bfly.b32(x, v, 1 << j, 31, 0xffffffff);
            add.rn.f32(v, v, x);
        }
    }
    fn norm_sum64<const STEPS: usize>(v: f32) {
        let x: f32;
        for j in 0..STEPS {
            shfl.sync.bfly.b32(x, v, 1 << j, 31, 0xffffffff);
            add.rn.f32(v, v, x);
        }
    }
    fn exp(v: f32) {
        mul.rn.f32(v, v, 1.4426950408889634);
        ex2.approx.f32(v, v);
    }

    // Load Jx in the half-split complex coordinates of one attention head.
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
    pub unsafe extern "ptx-kernel" fn norm<const D: usize>(
        out: *mut f32,
        x: *const f32,
        heads: u32,
    ) {
        let (block, lane, part, at, col): (u32, u32, u32, u32, u32);
        mov.u32(block, ctaid.x);
        mov.u32(lane, tid.x);
        div.u32(part, block, heads);
        rem.u32(part, part, 3);
        let values: [f32; (D + 31) / 32];
        let (s, z): (f32, f32);
        mov.f32(s, 0.0);
        for j in 0..(D + 31) / 32 {
            add.u32(col, lane, 32 * j);
            mov.f32(values[j], 0.0);
            let active: pred;
            setp.lt.u32(active, col, D);
            if active {
                mad.lo.u32(at, block, D, col);
                load(values[j], x, at);
            }
            mul.rn.f32(z, values[j], values[j]);
            add.rn.f32(s, s, z);
        }
        sum(s);
        mul.rn.f32(s, s, 1.0 / (D as f32));
        add.rn.f32(s, s, 0.00000011920928955078125);
        rsqrt.approx.f32(s, s);
        let value: pred;
        setp.eq.u32(value, part, 2);
        if value {
            mov.f32(s, 1.0);
        }
        for j in 0..(D + 31) / 32 {
            add.u32(col, lane, 32 * j);
            let active: pred;
            setp.lt.u32(active, col, D);
            if active {
                mad.lo.u32(at, block, D, col);
                mul.rn.f32(z, values[j], s);
                store(out, at, z);
            }
        }
    }
    #[instances(false, true)]
    pub unsafe extern "ptx-kernel" fn rotate<const BACK: bool>(
        out: *mut f32,
        x: *const f32,
        rope: *const f32,
        head_dim: u32,
        heads: u32,
        length: u32,
        count: u32,
    ) {
        let (i, b, t, n, row, pos, part, c, pair, at, other, half, width): (
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
        mov.u32(b, ctaid.x);
        mov.u32(t, tid.x);
        mov.u32(n, ntid.x);
        mad.lo.u32(i, b, n, t);
        let past: pred;
        setp.ge.u32(past, i, count);
        if past {
            ret();
        }
        mul.lo.u32(width, heads, head_dim);
        mul.lo.u32(n, width, 3);
        div.u32(row, i, n);
        rem.u32(pos, row, length);
        div.u32(part, i, width);
        rem.u32(part, part, 3);
        rem.u32(c, i, head_dim);
        shr.u32(half, head_dim, 1);
        let (v, z, co, si): (f32, f32, f32, f32);
        load(v, x, i);
        let value: pred;
        setp.eq.u32(value, part, 2);
        if value {
            store(out, i, v);
            ret();
        }
        rem.u32(pair, c, half);
        mad.lo.u32(at, pos, half, pair);
        load(co, rope, at);
        mul.lo.u32(n, length, half);
        add.u32(at, at, n);
        load(si, rope, at);
        let low: pred;
        setp.lt.u32(low, c, half);
        add.u32(other, i, half);
        if !low {
            sub.u32(other, i, half);
        }
        load(z, x, other);
        mul.rn.f32(v, v, co);
        mul.rn.f32(z, z, si);
        if BACK {
            if !low {
                neg.f32(z, z);
            }
        } else {
            if low {
                neg.f32(z, z);
            }
        }
        add.rn.f32(v, v, z);
        store(out, i, v);
    }

    #[instances(16, 32, 128)]
    pub unsafe extern "ptx-kernel" fn forward<const D: usize>(
        out: *mut f32,
        prob: *mut f32,
        delta_out: *mut f32,
        head_ok: *mut u32,
        qkv: *const f32,
        heads: u32,
        length: u32,
        hermitian: u32,
    ) {
        let (id, lane, b, t, h, w, base, qbase, vbase, anchor, scorebase, key, at, col): (
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
        let (q, own, first, avg, jown): (
            [f32; (D + 31) / 32],
            [f32; (D + 31) / 32],
            [f32; (D + 31) / 32],
            [f32; (D + 31) / 32],
            [f32; (D + 31) / 32],
        );
        let (norm, s, z, max, total, gain, p, kval): (f32, f32, f32, f32, f32, f32, f32, f32);
        mov.f32(norm, 0.0);
        let complex_mode: pred;
        setp.ne.u32(complex_mode, hermitian, 0);
        add.u32(at, t, 1);
        mov.f32(gain, D as f32);
        rsqrt.approx.f32(gain, gain);
        for j in 0..(D + 31) / 32 {
            add.u32(col, lane, 32 * j);
            mov.f32(q[j], 0.0);
            mov.f32(own[j], 0.0);
            mov.f32(first[j], 0.0);
            mov.f32(avg[j], 0.0);
            mov.f32(jown[j], 0.0);
            let active: pred;
            setp.lt.u32(active, col, D);
            if active {
                add.u32(at, qbase, col);
                load(q[j], qkv, at);
                add.u32(at, vbase, col);
                load(own[j], qkv, at);
                add.u32(at, anchor, col);
                load(first[j], qkv, at);
            }
            let complex_active: pred;
            and.pred(complex_active, active, complex_mode);
            if complex_active {
                load_j::<D>(jown[j], qkv, vbase, col);
            }
            mul.rn.f32(z, own[j], own[j]);
            add.rn.f32(norm, norm, z);
        }
        sum(norm);
        sqrt.rn.f32(norm, norm);
        let first_query: pred;
        setp.eq.u32(first_query, t, 0);
        let write_guard: pred;
        setp.eq.u32(write_guard, lane, 0);
        and.pred(write_guard, write_guard, first_query);
        if write_guard {
            let ok: pred;
            setp.gt.f32(ok, norm, 1e-12);
            selp.u32(at, 1, 0, ok);
            let addr: u64;
            mad.lo.u32(col, b, heads, h);
            mad.wide.u32(addr, col, 4, head_ok);
            st.global.u32([addr], at);
        }
        mov.f32(max, f32::NEG_INFINITY);
        mov.u32(key, 0);
        label(scores);
        mov.f32(s, 0.0);
        for j in 0..(D + 31) / 32 {
            add.u32(col, lane, 32 * j);
            mov.f32(kval, 0.0);
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
            mul.rn.f32(z, q[j], kval);
            add.rn.f32(s, s, z);
        }
        sum(s);
        mul.rn.f32(s, s, gain);
        let lead: pred;
        setp.eq.u32(lead, lane, 0);
        if lead {
            add.u32(at, scorebase, key);
            store(prob, at, s);
        }
        max.f32(max, max, s);
        add.u32(key, key, 1);
        let more: pred;
        setp.le.u32(more, key, t);
        if more {
            bra(scores);
        }
        bar.warp.sync(0xffffffff);
        mov.f32(total, 0.0);
        mov.u32(key, 0);
        label(normalizer);
        add.u32(at, scorebase, key);
        load(p, prob, at);
        sub.rn.f32(p, p, max);
        exp(p);
        add.rn.f32(total, total, p);
        add.u32(key, key, 1);
        setp.le.u32(more, key, t);
        if more {
            bra(normalizer);
        }
        mov.u32(key, 0);
        label(average);
        add.u32(at, scorebase, key);
        load(p, prob, at);
        sub.rn.f32(p, p, max);
        exp(p);
        div.rn.f32(p, p, total);
        bar.warp.sync(0xffffffff);
        if lead {
            store(prob, at, p);
        }
        for j in 0..(D + 31) / 32 {
            add.u32(col, lane, 32 * j);
            mov.f32(kval, 0.0);
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
            mul.rn.f32(kval, kval, p);
            add.rn.f32(avg[j], avg[j], kval);
        }
        add.u32(key, key, 1);
        setp.le.u32(more, key, t);
        if more {
            bra(average);
        }
        let (den, projection, imag_projection, rem, unit): (f32, f32, f32, f32, f32);
        max.f32(den, norm, 1e-12);
        mov.f32(projection, 0.0);
        mov.f32(imag_projection, 0.0);
        for j in 0..(D + 31) / 32 {
            sub.rn.f32(z, own[j], first[j]);
            sub.rn.f32(avg[j], avg[j], z);
            div.rn.f32(unit, own[j], den);
            mul.rn.f32(z, avg[j], unit);
            add.rn.f32(projection, projection, z);
            div.rn.f32(unit, jown[j], den);
            mul.rn.f32(z, avg[j], unit);
            add.rn.f32(imag_projection, imag_projection, z);
        }
        sum(projection);
        sum(imag_projection);
        mov.f32(rem, 0.0);
        let clipped: pred;
        setp.le.f32(clipped, norm, 1e-12);
        if clipped {
            div.rn.f32(rem, norm, den);
            mul.rn.f32(rem, rem, rem);
            sub.rn.f32(rem, 1.0, rem);
        }
        mad.lo.u32(base, b, length, t);
        mad.lo.u32(base, base, w, 0);
        mad.lo.u32(base, h, D, base);
        for j in 0..(D + 31) / 32 {
            add.u32(col, lane, 32 * j);
            let active: pred;
            setp.lt.u32(active, col, D);
            if active {
                add.u32(at, base, col);
                store(delta_out, at, avg[j]);
                div.rn.f32(unit, own[j], den);
                mul.rn.f32(z, projection, unit);
                sub.rn.f32(z, avg[j], z);
                div.rn.f32(unit, jown[j], den);
                mul.rn.f32(s, imag_projection, unit);
                sub.rn.f32(z, z, s);
                mul.rn.f32(s, rem, own[j]);
                add.rn.f32(z, z, s);
                selp.f32(z, 0.0, z, first_query);
                store(out, at, z);
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
        mov.f32(norm, 0.0);
        let complex_mode: pred;
        setp.ne.u32(complex_mode, hermitian, 0);
        let first_query: pred;
        setp.eq.u32(first_query, t, 0);
        for j in 0..(D + 31) / 32 {
            add.u32(col, lane, 32 * j);
            mov.f32(own[j], 0.0);
            mov.f32(change[j], 0.0);
            mov.f32(gradient[j], 0.0);
            mov.f32(first[j], 0.0);
            mov.f32(dquery[j], 0.0);
            mov.f32(jown[j], 0.0);
            mov.f32(jgradient[j], 0.0);
            mov.f32(jchange[j], 0.0);
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
                mov.f32(gradient[j], 0.0);
                mov.f32(jgradient[j], 0.0);
            }
            mul.rn.f32(z, own[j], own[j]);
            add.rn.f32(norm, norm, z);
        }
        sum(norm);
        sqrt.rn.f32(norm, norm);
        max.f32(den, norm, 1e-12);
        mov.f32(a, 0.0);
        mov.f32(beta, 0.0);
        mov.f32(gv, 0.0);
        let (c, eta): (f32, f32);
        mov.f32(c, 0.0);
        mov.f32(eta, 0.0);
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
        setp.ge.f32(above, norm, 1e-12);
        mov.f32(rem, 0.0);
        if !above {
            div.rn.f32(rem, norm, den);
            mul.rn.f32(rem, rem, rem);
            sub.rn.f32(rem, 1.0, rem);
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
                mul.rn.f32(z, z, 2.0);
                mul.rn.f32(z, z, u);
                add.rn.f32(s, s, z);
                div.rn.f32(s, s, den);
            } else {
                div.rn.f32(s, s, den);
                mul.rn.f32(z, rem, gradient[j]);
                add.rn.f32(s, s, z);
                mul.rn.f32(z, gv, own[j]);
                mul.rn.f32(z, z, 2.0);
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
        mov.f32(gain, D as f32);
        rsqrt.approx.f32(gain, gain);
        mov.f32(total, 0.0);
        mov.u32(key, 0);
        label(dprob);
        mov.f32(s, 0.0);
        for j in 0..(D + 31) / 32 {
            add.u32(col, lane, 32 * j);
            mov.f32(kval, 0.0);
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
            mov.f32(kval, 0.0);
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

    #[instances(16, 32, 128)]
    pub unsafe extern "ptx-kernel" fn key_backward<const D: usize>(
        dqkv: *mut f32,
        ds: *const f32,
        gd: *const f32,
        dv: *const f32,
        prob: *const f32,
        qkv: *const f32,
        heads: u32,
        length: u32,
    ) {
        let (id, lane, b, key, h, w, base, outbase, scorebase, t, at, col, qbase): (
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
        div.u32(key, id, heads);
        div.u32(b, key, length);
        rem.u32(key, key, length);
        mul.lo.u32(w, heads, D);
        mul.lo.u32(base, b, length);
        mad.lo.u32(outbase, base, w, 0);
        mad.lo.u32(outbase, h, D, outbase);
        mul.lo.u32(base, base, 3);
        mul.lo.u32(base, base, w);
        mad.lo.u32(base, h, D, base);
        mad.lo.u32(scorebase, b, heads, h);
        mul.lo.u32(scorebase, scorebase, length);
        mul.lo.u32(scorebase, scorebase, length);
        let (dk, dvalue): ([f32; (D + 31) / 32], [f32; (D + 31) / 32]);
        let (s, p, z, v): (f32, f32, f32, f32);
        for j in 0..(D + 31) / 32 {
            mov.f32(dk[j], 0.0);
            mov.f32(dvalue[j], 0.0);
        }
        mov.u32(t, key);
        label(query);
        mad.lo.u32(at, t, length, key);
        add.u32(at, at, scorebase);
        load(s, ds, at);
        load(p, prob, at);
        mul.lo.u32(qbase, t, 3);
        mad.lo.u32(qbase, qbase, w, base);
        for j in 0..(D + 31) / 32 {
            add.u32(col, lane, 32 * j);
            let active: pred;
            setp.lt.u32(active, col, D);
            if active {
                add.u32(at, qbase, col);
                load(v, qkv, at);
                mul.rn.f32(z, s, v);
                add.rn.f32(dk[j], dk[j], z);
                mad.lo.u32(at, t, w, outbase);
                add.u32(at, at, col);
                load(v, gd, at);
                mul.rn.f32(z, p, v);
                add.rn.f32(dvalue[j], dvalue[j], z);
            }
        }
        add.u32(t, t, 1);
        let more: pred;
        setp.lt.u32(more, t, length);
        if more {
            bra(query);
        }
        for j in 0..(D + 31) / 32 {
            add.u32(col, lane, 32 * j);
            let active: pred;
            setp.lt.u32(active, col, D);
            if active {
                mul.lo.u32(at, key, 3);
                add.u32(at, at, 1);
                mad.lo.u32(at, at, w, base);
                add.u32(at, at, col);
                store(dqkv, at, dk[j]);
                add.u32(qbase, at, w);
                mad.lo.u32(at, key, w, outbase);
                add.u32(at, at, col);
                load(v, dv, at);
                add.rn.f32(v, v, dvalue[j]);
                store(dqkv, qbase, v);
            }
        }
    }

    #[instances(16, 32, 128)]
    pub unsafe extern "ptx-kernel" fn norm_backward<const D: usize>(
        dx: *mut f32,
        x: *const f32,
        dy: *const f32,
        heads: u32,
    ) {
        let (block, lane, part, at, col): (u32, u32, u32, u32, u32);
        mov.u32(block, ctaid.x);
        mov.u32(lane, tid.x);
        if D == 16 {
            // Static head-width specialization; no precision decision at runtime.
            shr.u32(part, lane, 4);
            mad.lo.u32(block, block, 2, part);
            and.b32(lane, lane, 15);
        }
        div.u32(part, block, heads);
        rem.u32(part, part, 3);
        let (v, g): ([f32; (D + 31) / 32], [f32; (D + 31) / 32]);
        let (energy, dot, z, inv, den, radial, parallel, tangent): (
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
        );
        let f: f32;
        mov.f32(energy, 0.0f32);
        mov.f32(dot, 0.0f32);
        for j in 0..(D + 31) / 32 {
            add.u32(col, lane, 32 * j);
            mov.f32(v[j], 0.0f32);
            mov.f32(g[j], 0.0f32);
            let active: pred;
            setp.lt.u32(active, col, D);
            if active {
                mad.lo.u32(at, block, D, col);
                load(f, x, at);
                mov.f32(v[j], f);
                load(f, dy, at);
                mov.f32(g[j], f);
            }
            mul.rn.f32(z, v[j], v[j]);
            add.rn.f32(energy, energy, z);
            mul.rn.f32(z, v[j], g[j]);
            add.rn.f32(dot, dot, z);
        }
        if D == 16 {
            norm_sum64::<4>(energy);
            norm_sum64::<4>(dot);
        } else {
            norm_sum64::<5>(energy);
            norm_sum64::<5>(dot);
        }
        mul.rn.f32(energy, energy, 1.0f32 / (D as f32));
        mul.rn.f32(dot, dot, 1.0f32 / (D as f32));
        let positive: pred;
        setp.gt.f32(positive, energy, 0.0f32);
        selp.f32(den, energy, 1.0f32, positive);
        add.rn.f32(inv, energy, 0.00000011920928955078125f32);
        sqrt.rn.f32(inv, inv);
        div.rn.f32(inv, 1.0f32, inv);
        mul.rn.f32(radial, inv, inv);
        mul.rn.f32(radial, radial, inv);
        mul.rn.f32(radial, radial, 0.00000011920928955078125f32);
        for j in 0..(D + 31) / 32 {
            mul.rn.f32(parallel, v[j], dot);
            div.rn.f32(parallel, parallel, den);
            sub.rn.f32(tangent, g[j], parallel);
            mul.rn.f32(tangent, tangent, inv);
            mul.rn.f32(parallel, parallel, radial);
            add.rn.f32(tangent, tangent, parallel);
            let value: pred;
            setp.eq.u32(value, part, 2);
            if value {
                mov.f32(tangent, g[j]);
            }
            mov.f32(f, tangent);
            add.u32(col, lane, 32 * j);
            let active: pred;
            setp.lt.u32(active, col, D);
            if active {
                mad.lo.u32(at, block, D, col);
                store(dx, at, f);
            }
        }
    }
}
