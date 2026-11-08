//! Causal probability centering, with FP32 accumulation and an anchored value sum.
//! The ordinary Q adjoint is reused; this module supplies the centered V adjoint.
//! Near uniform scores use expm1 differences before probability rounding.
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
    fn load(v: f32, p: u64, i: u32) {
        let f: f32;
        load32(f, p, i);
        mov.f32(v, f);
    }
    fn store(p: u64, i: u32, v: f32) {
        let f: f32;
        mov.f32(f, v);
        store32(p, i, f);
    }
    fn load64(v: f32, p: u64, i: u32) {
        let a: u64;
        mad.wide.u32(a, i, 4, p);
        ld.global.f32(v, [a]);
    }
    fn store64(p: u64, i: u32, v: f32) {
        let a: u64;
        mad.wide.u32(a, i, 4, p);
        st.global.f32([a], v);
    }
    fn sum64(v: f32) {
        let x: f32;
        for j in 0..5 {
            shfl.sync.bfly.b32(x, v, 1 << j, 31, 0xffffffff);
            add.rn.f32(v, v, x);
        }
    }
    // Six terms for |x| <= 1e-3; avoids subtracting two rounded numbers near1.
    fn exp_delta(y: f32, x: f32) {
        mov.f32(y, 0.001388888888888889f32);
        mul.rn.f32(y, y, x);
        add.rn.f32(y, y, 0.008333333333333333f32);
        mul.rn.f32(y, y, x);
        add.rn.f32(y, y, 0.041666666666666664f32);
        mul.rn.f32(y, y, x);
        add.rn.f32(y, y, 0.16666666666666666f32);
        mul.rn.f32(y, y, x);
        add.rn.f32(y, y, 0.5f32);
        mul.rn.f32(y, y, x);
        add.rn.f32(y, y, 1.0f32);
        mul.rn.f32(y, y, x);
    }
    #[instances(16, 32, 128)]
    pub unsafe extern "ptx-kernel" fn coefficients<const D: usize>(
        out: *mut f32,
        prob: *const f32,
        qkv: *const f32,
        heads: u32,
        length: u32,
    ) {
        let (id, lane, b, t, h, w, base, qbase, kfirst, rowbase, key, col, at): (
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
        add.u32(kfirst, base, w);
        mad.lo.u32(kfirst, h, D, kfirst);
        mad.lo.u32(rowbase, b, heads, h);
        mad.lo.u32(rowbase, rowbase, length, t);
        mul.lo.u32(rowbase, rowbase, length);
        let (q, k0): ([f32; (D + 31) / 32], [f32; (D + 31) / 32]);
        let (v, z, s, maximum, total, mean, n, gain): (f32, f32, f32, f32, f32, f32, f32, f32);
        let gain32: f32;
        add.u32(at, t, 1);
        cvt.rn.f32.u32(n, at);
        cvt.rn.f32.u32(gain32, at);
        lg2.approx.f32(gain32, gain32);
        mul.rn.f32(gain32, gain32, 0.6931471805599453 / (D as f32));
        mov.f32(gain, gain32);
        for j in 0..(D + 31) / 32 {
            mov.f32(q[j], 0.0f32);
            mov.f32(k0[j], 0.0f32);
            add.u32(col, lane, 32 * j);
            let active: pred;
            setp.lt.u32(active, col, D);
            if active {
                add.u32(at, qbase, col);
                load(q[j], qkv, at);
                add.u32(at, kfirst, col);
                load(k0[j], qkv, at);
            }
        }
        let lead: pred;
        setp.eq.u32(lead, lane, 0);
        mov.f32(maximum, 0.0f32);
        mov.f32(total, 0.0f32);
        mov.u32(key, 0);
        label(scores);
        mov.f32(s, 0.0f32);
        for j in 0..(D + 31) / 32 {
            add.u32(col, lane, 32 * j);
            let active: pred;
            setp.lt.u32(active, col, D);
            if active {
                mul.lo.u32(at, key, 3);
                mad.lo.u32(at, at, w, kfirst);
                add.u32(at, at, col);
                load(v, qkv, at);
                sub.rn.f32(v, v, k0[j]);
                mul.rn.f32(z, v, q[j]);
                add.rn.f32(s, s, z);
            }
        }
        sum64(s);
        mul.rn.f32(s, s, gain);
        abs.f32(z, s);
        max.f32(maximum, maximum, z);
        add.u32(at, rowbase, key);
        if lead {
            store64(out, at, s);
        }
        load(v, prob, at);
        add.rn.f32(total, total, v);
        add.u32(key, key, 1);
        let more_scores: pred;
        setp.le.u32(more_scores, key, t);
        if more_scores {
            bra(scores);
        }
        div.rn.f32(mean, total, n);
        bar.warp.sync(0xffffffff);
        let near: pred;
        setp.le.f32(near, maximum, 0.001f32);
        let far: pred;
        setp.gt.f32(far, maximum, 0.001f32);
        if far {
            bra(center_ready);
        }
        mov.f32(total, 0.0f32);
        mov.u32(key, 0);
        label(delta_sum);
        add.u32(at, rowbase, key);
        load64(s, out, at);
        exp_delta(v, s);
        add.rn.f32(total, total, v);
        add.u32(key, key, 1);
        let more_delta: pred;
        setp.le.u32(more_delta, key, t);
        if more_delta {
            bra(delta_sum);
        }
        div.rn.f32(mean, total, n);
        add.rn.f32(total, total, n);
        label(center_ready);
        mov.u32(key, 0);
        label(center);
        add.u32(at, rowbase, key);
        if near {
            load64(s, out, at);
            exp_delta(v, s);
            sub.rn.f32(v, v, mean);
            div.rn.f32(v, v, total);
        } else {
            load(v, prob, at);
            sub.rn.f32(v, v, mean);
        }
        if lead {
            store64(out, at, v);
        }
        add.u32(key, key, 1);
        let more_center: pred;
        setp.le.u32(more_center, key, t);
        if more_center {
            bra(center);
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
            mov.f32(first[j], 0.0f32);
            mov.f32(avg[j], 0.0f32);
            let active: pred;
            setp.lt.u32(active, col, D);
            if active {
                add.u32(at, anchor, col);
                load(first[j], qkv, at);
            }
        }
        mov.u32(key, 0);
        label(mix);
        add.u32(at, scorebase, key);
        load64(p, prob, at);
        for j in 0..(D + 31) / 32 {
            add.u32(col, lane, 32 * j);
            let active: pred;
            setp.lt.u32(active, col, D);
            if active {
                mul.lo.u32(at, key, 3);
                mad.lo.u32(at, at, w, anchor);
                add.u32(at, at, col);
                load(v, qkv, at);
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
                mov.f32(z, avg[j]);
                add.u32(at, outbase, col);
                store(out, at, z);
            }
        }
    }

    #[instances(16, 32, 128)]
    pub unsafe extern "ptx-kernel" fn key_backward<const D: usize>(
        dq: *mut f32,
        ds: *const f32,
        gd: *const f32,
        prob: *const f32,
        qkv: *const f32,
        heads: u32,
        length: u32,
    ) {
        let (id, lane, b, t, h, w, base, keybase, query, rowbase, col, at): (
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
        mul.lo.u32(keybase, t, 3);
        mad.lo.u32(keybase, keybase, w, base);
        mad.lo.u32(keybase, h, D, keybase);
        let (dk, dv): ([f32; (D + 31) / 32], [f32; (D + 31) / 32]);
        let (p, score, g, z, value): (f32, f32, f32, f32, f32);
        for j in 0..(D + 31) / 32 {
            mov.f32(dk[j], 0.0f32);
            mov.f32(dv[j], 0.0f32);
        }
        mov.u32(query, t);
        label(queries);
        mad.lo.u32(rowbase, b, heads, h);
        mad.lo.u32(rowbase, rowbase, length, query);
        mul.lo.u32(rowbase, rowbase, length);
        add.u32(at, rowbase, t);
        load64(p, prob, at);
        load(score, ds, at);
        for j in 0..(D + 31) / 32 {
            add.u32(col, lane, 32 * j);
            let active: pred;
            setp.lt.u32(active, col, D);
            if active {
                mul.lo.u32(at, query, 3);
                mad.lo.u32(at, at, w, base);
                mad.lo.u32(at, h, D, at);
                add.u32(at, at, col);
                load(value, qkv, at);
                mul.rn.f32(z, score, value);
                add.rn.f32(dk[j], dk[j], z);
                mad.lo.u32(at, b, length, query);
                mul.lo.u32(at, at, w);
                mad.lo.u32(at, h, D, at);
                add.u32(at, at, col);
                load(g, gd, at);
                mul.rn.f32(z, p, g);
                add.rn.f32(dv[j], dv[j], z);
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
                add.u32(at, keybase, w);
                add.u32(at, at, col);
                store(dq, at, dk[j]);
                add.u32(at, at, w);
                store(dq, at, dv[j]);
            }
        }
    }
}
