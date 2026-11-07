//! Causal probability centering, with FP64 accumulation and an anchored value sum.
//! The ordinary Q adjoint is reused; this module supplies the centered V adjoint.
//! A row's stored mean replaces 1/N to preserve exactly uniform FP32 rows.
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
    fn load(v: f64, p: u64, i: u32) {
        let f: f32;
        load32(f, p, i);
        cvt.f64.f32(v, f);
    }
    fn store(p: u64, i: u32, v: f64) {
        let f: f32;
        cvt.rn.f32.f64(f, v);
        store32(p, i, f);
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
        let (first, avg): ([f64; (D + 31) / 32], [f64; (D + 31) / 32]);
        let (p, v, z, mean): (f64, f64, f64, f64);
        for j in 0..(D + 31) / 32 {
            add.u32(col, lane, 32 * j);
            mov.f64(first[j], 0.0f64);
            mov.f64(avg[j], 0.0f64);
            let active: pred;
            setp.lt.u32(active, col, D);
            if active {
                add.u32(at, anchor, col);
                load(first[j], qkv, at);
            }
        }
        mov.f64(mean, 0.0f64);
        mov.u32(key, 0);
        label(prob_mean);
        add.u32(at, scorebase, key);
        load(p, prob, at);
        add.rn.f64(mean, mean, p);
        add.u32(key, key, 1);
        let another: pred;
        setp.le.u32(another, key, t);
        if another {
            bra(prob_mean);
        }
        add.u32(at, t, 1);
        cvt.rn.f64.u32(z, at);
        div.rn.f64(mean, mean, z);
        mov.u32(key, 0);
        label(mix);
        add.u32(at, scorebase, key);
        load(p, prob, at);
        sub.rn.f64(p, p, mean);
        for j in 0..(D + 31) / 32 {
            add.u32(col, lane, 32 * j);
            let active: pred;
            setp.lt.u32(active, col, D);
            if active {
                mul.lo.u32(at, key, 3);
                mad.lo.u32(at, at, w, anchor);
                add.u32(at, at, col);
                load(v, qkv, at);
                sub.rn.f64(v, v, first[j]);
                mul.rn.f64(z, p, v);
                add.rn.f64(avg[j], avg[j], z);
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
                mov.f64(z, avg[j]);
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
        let (id, lane, b, t, h, w, base, keybase, query, rowbase, col, at, jkey): (
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
        mul.lo.u32(keybase, t, 3);
        mad.lo.u32(keybase, keybase, w, base);
        mad.lo.u32(keybase, h, D, keybase);
        let (dk, dv): ([f64; (D + 31) / 32], [f64; (D + 31) / 32]);
        let (mean, p, score, g, z, value): (f64, f64, f64, f64, f64, f64);
        for j in 0..(D + 31) / 32 {
            mov.f64(dk[j], 0.0f64);
            mov.f64(dv[j], 0.0f64);
        }
        mov.u32(query, t);
        label(queries);
        mad.lo.u32(rowbase, b, heads, h);
        mad.lo.u32(rowbase, rowbase, length, query);
        mul.lo.u32(rowbase, rowbase, length);
        mov.f64(mean, 0.0f64);
        mov.u32(jkey, 0);
        label(mean_keys);
        add.u32(at, rowbase, jkey);
        load(p, prob, at);
        add.rn.f64(mean, mean, p);
        add.u32(jkey, jkey, 1);
        let more_keys: pred;
        setp.le.u32(more_keys, jkey, query);
        if more_keys {
            bra(mean_keys);
        }
        add.u32(at, query, 1);
        cvt.rn.f64.u32(z, at);
        div.rn.f64(mean, mean, z);
        add.u32(at, rowbase, t);
        load(p, prob, at);
        sub.rn.f64(p, p, mean);
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
                mul.rn.f64(z, score, value);
                add.rn.f64(dk[j], dk[j], z);
                mad.lo.u32(at, b, length, query);
                mul.lo.u32(at, at, w);
                mad.lo.u32(at, h, D, at);
                add.u32(at, at, col);
                load(g, gd, at);
                mul.rn.f64(z, p, g);
                add.rn.f64(dv[j], dv[j], z);
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
