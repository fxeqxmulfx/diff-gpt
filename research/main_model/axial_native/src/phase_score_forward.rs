//! Squared-imaginary correction to ordinary causal SSMax. The real dot,
//! softmax and anchored value sum retain the selected parent's operation order.
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
    fn sum<const D: usize>(v: f32) {
        let x: f32;
        for j in 0..5 {
            if D != 16 || j < 4 {
                shfl.sync.bfly.b32(x, v, 1 << j, 31, 0xffffffff);
                add.rn.f32(v, v, x);
            }
        }
        // The fifth parent butterfly added the unused half-warp's positive zero.
        if D == 16 {
            add.rn.f32(v, v, 0.0);
        }
    }
    fn exp(v: f32) {
        mul.rn.f32(v, v, 1.4426950408889634);
        ex2.approx.f32(v, v);
    }
    #[instances(16, 32, 128)]
    pub unsafe extern "ptx-kernel" fn forward<const D: usize>(
        out: *mut f32,
        prob: *mut f32,
        qkv: *const f32,
        imaginary: *mut f32,
        coefficients: *const f32,
        heads: u32,
        length: u32,
        head_count: u32,
        logit_gain: f32,
    ) {
        let (id, lane, b, t, h, w, base, qbase, anchor, scorebase, key, at, col, outbase): (
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
        let warp: u32;
        shr.u32(warp, lane, 5);
        mad.lo.u32(id, id, 4, warp);
        and.b32(lane, lane, 31);
        // D16 model head counts are4,8,16: two adjacent heads share a query
        // position, so both half-warps execute the same causal loop lengths.
        if D == 16 {
            let half: u32;
            shr.u32(half, lane, 4);
            mad.lo.u32(id, id, 2, half);
            and.b32(lane, lane, 15);
        }
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
        mad.lo.u32(anchor, h, D, base);
        mad.lo.u32(anchor, w, 2, anchor);
        mad.lo.u32(scorebase, b, heads, h);
        mad.lo.u32(scorebase, scorebase, length, t);
        mul.lo.u32(scorebase, scorebase, length);
        mad.lo.u32(outbase, b, length, t);
        mul.lo.u32(outbase, outbase, w);
        mad.lo.u32(outbase, h, D, outbase);
        let (q, first, avg): (
            [f32; (D + 31) / 32],
            [f32; (D + 31) / 32],
            [f32; (D + 31) / 32],
        );
        let (s, z, max, total, gain, p, v): (f32, f32, f32, f32, f32, f32, f32);
        let (im, cross, beta, correction): (f32, f32, f32, f32);
        mul.lo.u32(at, h, 2);
        load(beta, coefficients, at);
        add.u32(at, t, 1);
        cvt.rn.f32.u32(gain, at);
        lg2.approx.f32(gain, gain);
        mul.rn.f32(gain, gain, 0.6931471805599453 / (D as f32));
        for j in 0..(D + 31) / 32 {
            add.u32(col, lane, 32 * j);
            mov.f32(q[j], 0.0);
            mov.f32(first[j], 0.0);
            mov.f32(avg[j], 0.0);
            let active: pred;
            setp.lt.u32(active, col, D);
            if active {
                add.u32(at, qbase, col);
                load(q[j], qkv, at);
                add.u32(at, anchor, col);
                load(first[j], qkv, at);
            }
        }
        mov.f32(max, f32::NEG_INFINITY);
        mov.u32(key, 0);
        label(scores);
        mov.f32(s, 0.0);
        mov.f32(im, 0.0);
        for j in 0..(D + 31) / 32 {
            add.u32(col, lane, 32 * j);
            mov.f32(v, 0.0);
            let active: pred;
            setp.lt.u32(active, col, D);
            if active {
                mul.lo.u32(at, key, 3);
                add.u32(at, at, 1);
                mad.lo.u32(at, at, w, base);
                mad.lo.u32(at, h, D, at);
                add.u32(at, at, col);
                load(v, qkv, at);
                let (partner, negative): (u32, f32);
                let real: pred;
                setp.lt.u32(real, col, D / 2);
                add.u32(partner, at, D / 2);
                sub.u32(at, at, D / 2);
                selp.u32(at, partner, at, real);
                load(cross, qkv, at);
                neg.f32(negative, cross);
                selp.f32(cross, cross, negative, real);
                mul.rn.f32(z, q[j], cross);
                add.rn.f32(im, im, z);
            }
            mul.rn.f32(z, q[j], v);
            add.rn.f32(s, s, z);
        }
        sum::<D>(s);
        sum::<D>(im);
        mul.rn.f32(correction, im, 1.0 / (D as f32));
        mul.rn.f32(correction, beta, correction);
        mul.rn.f32(correction, correction, im);
        add.rn.f32(s, s, correction);
        mul.rn.f32(s, s, gain);
        // One temperature on the complete real score, after both terms.
        mul.rn.f32(s, s, logit_gain);
        let lead: pred;
        setp.eq.u32(lead, lane, 0);
        if lead {
            add.u32(at, scorebase, key);
            store(prob, at, s);
            store(imaginary, at, im);
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
        label(mix);
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
        setp.le.u32(more, key, t);
        if more {
            bra(mix);
        }
        for j in 0..(D + 31) / 32 {
            add.u32(col, lane, 32 * j);
            let active: pred;
            setp.lt.u32(active, col, D);
            if active {
                add.rn.f32(z, first[j], avg[j]);
                add.u32(at, outbase, col);
                store(out, at, z);
            }
        }
    }
}
