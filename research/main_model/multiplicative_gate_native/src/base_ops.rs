//! GPU operations for the frozen two-token transformer. FP32 activations and parameters;
//! RMS derivatives keep the original separated FP32 radial/tangent calculation.

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
    fn store_u(base: u64, i: u32, x: u32) {
        let p: u64;
        mad.wide.u32(p, i, 4, base);
        st.global.u32([p], x);
    }
    fn index(i: u32) {
        let (b, t, n): (u32, u32, u32);
        mov.u32(b, ctaid.x);
        mov.u32(t, tid.x);
        mov.u32(n, ntid.x);
        mad.lo.u32(i, b, n, t);
    }
    fn sum32(x: f32) {
        let v: f32;
        for j in 0..5 {
            shfl.sync.bfly.b32(v, x, 1 << j, 31, 0xffffffff);
            add.rn.f32(x, x, v);
        }
    }
    fn sum64(v: f32) {
        let x: f32;
        for j in 0..5 {
            shfl.sync.bfly.b32(x, v, 1 << j, 31, 0xffffffff);
            add.rn.f32(v, v, x);
        }
    }
    fn exp(x: f32) {
        mul.rn.f32(x, x, 1.4426950408889634);
        ex2.approx.f32(x, x);
    }

    /// Gather an already frozen batch; one thread per two-token example.
    pub unsafe extern "ptx-kernel" fn gather(
        tokens: *mut u32,
        labels: *mut u32,
        flags: *mut u32,
        data: *const u32,
        order: *const u32,
        batch: u32,
    ) {
        let (b, row, i, a, c, label): (u32, u32, u32, u32, u32, u32);
        index(b);
        let past: pred;
        setp.ge.u32(past, b, batch);
        if past {
            ret();
        }
        load_u(row, order, b);
        mul.lo.u32(i, row, 3);
        load_u(a, data, i);
        add.u32(i, i, 1);
        load_u(c, data, i);
        add.u32(i, i, 1);
        load_u(label, data, i);
        mul.lo.u32(i, b, 2);
        store_u(tokens, i, a);
        add.u32(i, i, 1);
        store_u(tokens, i, c);
        store_u(labels, b, label);
        let eq: pred;
        setp.eq.u32(eq, a, c);
        selp.u32(i, 1, 0, eq);
        store_u(flags, b, i);
    }

    /// Embedding lookup without changing the shared readout parameter.
    pub unsafe extern "ptx-kernel" fn embed(
        y: *mut f32,
        p: *const f32,
        tokens: *const u32,
        width: u32,
        count: u32,
    ) {
        let (i, row, col, tok, at): (u32, u32, u32, u32, u32);
        index(i);
        let past: pred;
        setp.ge.u32(past, i, count);
        if past {
            ret();
        }
        div.u32(row, i, width);
        rem.u32(col, i, width);
        load_u(tok, tokens, row);
        mad.lo.u32(at, tok, width, col);
        let x: f32;
        load(x, p, at);
        store(y, i, x);
    }

    /// Deterministic embedding gradient, added to the tied readout gradient.
    pub unsafe extern "ptx-kernel" fn embed_bwd(
        g: *mut f32,
        dx: *const f32,
        tokens: *const u32,
        width: u32,
        vocab: u32,
        rows: u32,
    ) {
        let (i, count, word, col, row, tok, at): (u32, u32, u32, u32, u32, u32, u32);
        index(i);
        mul.lo.u32(count, width, vocab);
        let past: pred;
        setp.ge.u32(past, i, count);
        if past {
            ret();
        }
        div.u32(word, i, width);
        rem.u32(col, i, width);
        let (s, v): (f32, f32);
        mov.f32(s, 0.0);
        mov.u32(row, 0);
        label(iteration);
        load_u(tok, tokens, row);
        let hit: pred;
        setp.eq.u32(hit, word, tok);
        if hit {
            mad.lo.u32(at, row, width, col);
            load(v, dx, at);
            add.rn.f32(s, s, v);
        }
        add.u32(row, row, 1);
        setp.lt.u32(hit, row, rows);
        if hit {
            bra(iteration);
        }
        load(v, g, i);
        add.rn.f32(s, v, s);
        store(g, i, s);
    }

    /// Transpose for the original row-major Linear weight convention.
    pub unsafe extern "ptx-kernel" fn transpose(y: *mut f32, x: *const f32, rows: u32, cols: u32) {
        let (i, n, r, c, j): (u32, u32, u32, u32, u32);
        index(i);
        mul.lo.u32(n, rows, cols);
        let past: pred;
        setp.ge.u32(past, i, n);
        if past {
            ret();
        }
        div.u32(r, i, cols);
        rem.u32(c, i, cols);
        mad.lo.u32(j, c, rows, r);
        let v: f32;
        load(v, x, i);
        store(y, j, v);
    }

    /// Each warp normalizes one row; epsilon is FP32 machine epsilon.
    #[instances(64, 128, 256)]
    pub unsafe extern "ptx-kernel" fn rms<const W: usize>(y: *mut f32, x: *const f32, rows: u32) {
        let (i, row, lane, at): (u32, u32, u32, u32);
        index(i);
        shr.u32(row, i, 5);
        and.b32(lane, i, 31);
        let past: pred;
        setp.ge.u32(past, row, rows);
        if past {
            ret();
        }
        mad.lo.u32(at, row, W, lane);
        let v: [f32; W / 32];
        let (s, z): (f32, f32);
        mov.f32(s, 0.0);
        for j in 0..W / 32 {
            load(v[j], x, at);
            mul.rn.f32(z, v[j], v[j]);
            add.rn.f32(s, s, z);
            add.u32(at, at, 32);
        }
        sum32(s);
        mul.rn.f32(s, s, 1.0 / (W as f32));
        add.rn.f32(s, s, 0.00000011920928955078125);
        rsqrt.approx.f32(s, s);
        mad.lo.u32(at, row, W, lane);
        for j in 0..W / 32 {
            mul.rn.f32(z, v[j], s);
            store(y, at, z);
            add.u32(at, at, 32);
        }
    }

    /// The separated radial/tangent RMS backward, using FP32 before the final cast.
    #[instances(64, 128, 256)]
    pub unsafe extern "ptx-kernel" fn rms_bwd<const W: usize>(
        dx: *mut f32,
        x: *const f32,
        dy: *const f32,
        rows: u32,
    ) {
        let (i, row, lane, at): (u32, u32, u32, u32);
        index(i);
        shr.u32(row, i, 5);
        and.b32(lane, i, 31);
        let past: pred;
        setp.ge.u32(past, row, rows);
        if past {
            ret();
        }
        mad.lo.u32(at, row, W, lane);
        let (v, g): ([f32; W / 32], [f32; W / 32]);
        let (energy, dot, temp): (f32, f32, f32);
        let z: f32;
        mov.f32(energy, 0.0f32);
        mov.f32(dot, 0.0f32);
        for j in 0..W / 32 {
            load(z, x, at);
            mov.f32(v[j], z);
            load(z, dy, at);
            mov.f32(g[j], z);
            mul.rn.f32(temp, v[j], v[j]);
            add.rn.f32(energy, energy, temp);
            mul.rn.f32(temp, v[j], g[j]);
            add.rn.f32(dot, dot, temp);
            add.u32(at, at, 32);
        }
        sum64(energy);
        sum64(dot);
        mul.rn.f32(energy, energy, 1.0f32 / (W as f32));
        mul.rn.f32(dot, dot, 1.0f32 / (W as f32));
        let (den, inv, radial, parallel, tangent): (f32, f32, f32, f32, f32);
        let positive: pred;
        setp.gt.f32(positive, energy, 0.0f32);
        selp.f32(den, energy, 1.0f32, positive);
        add.rn.f32(inv, energy, 0.00000011920928955078125f32);
        sqrt.rn.f32(inv, inv);
        div.rn.f32(inv, 1.0f32, inv);
        mul.rn.f32(radial, inv, inv);
        mul.rn.f32(radial, radial, inv);
        mul.rn.f32(radial, radial, 0.00000011920928955078125f32);
        mad.lo.u32(at, row, W, lane);
        for j in 0..W / 32 {
            mul.rn.f32(parallel, v[j], dot);
            div.rn.f32(parallel, parallel, den);
            sub.rn.f32(tangent, g[j], parallel);
            mul.rn.f32(tangent, tangent, inv);
            mul.rn.f32(parallel, parallel, radial);
            add.rn.f32(tangent, tangent, parallel);
            mov.f32(z, tangent);
            store(dx, at, z);
            add.u32(at, at, 32);
        }
    }

    /// Fused SwiGLU, with the frozen gate/value column layout.
    pub unsafe extern "ptx-kernel" fn swiglu(y: *mut f32, x: *const f32, hidden: u32, count: u32) {
        let (i, row, col, at): (u32, u32, u32, u32);
        index(i);
        let past: pred;
        setp.ge.u32(past, i, count);
        if past {
            ret();
        }
        div.u32(row, i, hidden);
        rem.u32(col, i, hidden);
        mul.lo.u32(row, row, 2);
        mad.lo.u32(at, row, hidden, col);
        let (gate, value, s): (f32, f32, f32);
        load(gate, x, at);
        add.u32(at, at, hidden);
        load(value, x, at);
        neg.f32(s, gate);
        exp(s);
        add.rn.f32(s, s, 1.0);
        div.rn.f32(s, 1.0, s);
        mul.rn.f32(s, gate, s);
        mul.rn.f32(s, s, value);
        store(y, i, s);
    }

    pub unsafe extern "ptx-kernel" fn swiglu_bwd(
        dx: *mut f32,
        x: *const f32,
        dy: *const f32,
        hidden: u32,
        count: u32,
    ) {
        let (i, row, col, at, next): (u32, u32, u32, u32, u32);
        index(i);
        let past: pred;
        setp.ge.u32(past, i, count);
        if past {
            ret();
        }
        div.u32(row, i, hidden);
        rem.u32(col, i, hidden);
        mul.lo.u32(row, row, 2);
        mad.lo.u32(at, row, hidden, col);
        add.u32(next, at, hidden);
        let (gate, value, s, g, t, dg, dv): (f32, f32, f32, f32, f32, f32, f32);
        load(gate, x, at);
        load(value, x, next);
        load(g, dy, i);
        neg.f32(s, gate);
        exp(s);
        add.rn.f32(s, s, 1.0);
        div.rn.f32(s, 1.0, s);
        mul.rn.f32(dv, gate, s);
        mul.rn.f32(dv, dv, g);
        sub.rn.f32(t, 1.0, s);
        mul.rn.f32(t, t, gate);
        add.rn.f32(t, t, 1.0);
        mul.rn.f32(t, t, s);
        mul.rn.f32(dg, g, value);
        mul.rn.f32(dg, dg, t);
        store(dx, at, dg);
        store(dx, next, dv);
    }

    /// Update the prefix-sharing flag only while every first value head is unclipped.
    pub unsafe extern "ptx-kernel" fn guard(
        out: *mut u32,
        previous: *const u32,
        heads_ok: *const u32,
        heads: u32,
        batch: u32,
    ) {
        let (b, flag, h, at, ok): (u32, u32, u32, u32, u32);
        index(b);
        let past: pred;
        setp.ge.u32(past, b, batch);
        if past {
            ret();
        }
        load_u(flag, previous, b);
        mov.u32(h, 0);
        label(iteration);
        mad.lo.u32(at, b, heads, h);
        load_u(ok, heads_ok, at);
        and.b32(flag, flag, ok);
        add.u32(h, h, 1);
        setp.lt.u32(past, h, heads);
        if past {
            bra(iteration);
        }
        store_u(out, b, flag);
    }

    /// Branch masks and prefix sharing are applied at the same places as PrefixStableGPT.
    #[instances(true, false)]
    pub unsafe extern "ptx-kernel" fn residual<const ATT: bool>(
        out: *mut f32,
        x: *const f32,
        branch: *const f32,
        flags: *const u32,
        width: u32,
        count: u32,
        scale: f32,
    ) {
        let (i, row, b, t, c, flag, at): (u32, u32, u32, u32, u32, u32, u32);
        index(i);
        let past: pred;
        setp.ge.u32(past, i, count);
        if past {
            ret();
        }
        div.u32(row, i, width);
        rem.u32(c, i, width);
        shr.u32(b, row, 1);
        and.b32(t, row, 1);
        load_u(flag, flags, b);
        mov.u32(at, i);
        let skip: pred;
        let (v, r): (f32, f32);
        mov.f32(v, 0.0);
        if ATT {
            or.b32(flag, flag, t);
            // First position and shared second position both have zero attention branch.
            load_u(flag, flags, b);
            setp.eq.u32(skip, t, 0);
            if skip {
                bra(store);
            }
            setp.ne.u32(skip, flag, 0);
            if skip {
                bra(store);
            }
        } else {
            setp.ne.u32(skip, flag, 0);
            if skip {
                mul.lo.u32(at, b, 2);
                mad.lo.u32(at, at, width, c);
            }
        }
        load(v, branch, at);
        mul.rn.f32(v, v, scale);
        label(store);
        load(r, x, i);
        add.rn.f32(v, v, r);
        store(out, i, v);
    }

    #[instances(true, false)]
    pub unsafe extern "ptx-kernel" fn residual_bwd<const ATT: bool>(
        out: *mut f32,
        dy: *const f32,
        flags: *const u32,
        width: u32,
        count: u32,
        scale: f32,
    ) {
        let (i, row, b, t, flag, next): (u32, u32, u32, u32, u32, u32);
        index(i);
        let past: pred;
        setp.ge.u32(past, i, count);
        if past {
            ret();
        }
        div.u32(row, i, width);
        shr.u32(b, row, 1);
        and.b32(t, row, 1);
        load_u(flag, flags, b);
        let (v, other): (f32, f32);
        let yes: pred;
        mov.f32(v, 0.0);
        if ATT {
            setp.eq.u32(yes, t, 0);
            if yes {
                bra(done);
            }
            setp.ne.u32(yes, flag, 0);
            if yes {
                bra(done);
            }
            load(v, dy, i);
        } else {
            setp.eq.u32(yes, flag, 0);
            if yes {
                bra(ordinary);
            }
            setp.ne.u32(yes, t, 0);
            if yes {
                bra(done);
            }
            add.u32(next, i, width);
            load(v, dy, i);
            load(other, dy, next);
            add.rn.f32(v, v, other);
            bra(done);
            label(ordinary);
            load(v, dy, i);
        }
        label(done);
        mul.rn.f32(v, v, scale);
        store(out, i, v);
    }

    #[instances(true, false)]
    pub unsafe extern "ptx-kernel" fn share<const BACK: bool>(
        out: *mut f32,
        x: *const f32,
        flags: *const u32,
        width: u32,
        count: u32,
    ) {
        let (i, row, b, t, flag, at): (u32, u32, u32, u32, u32, u32);
        index(i);
        let past: pred;
        setp.ge.u32(past, i, count);
        if past {
            ret();
        }
        div.u32(row, i, width);
        shr.u32(b, row, 1);
        and.b32(t, row, 1);
        load_u(flag, flags, b);
        let (v, other): (f32, f32);
        let yes: pred;
        mov.f32(v, 0.0);
        mov.u32(at, i);
        setp.eq.u32(yes, flag, 0);
        if yes {
            bra(ordinary);
        }
        if BACK {
            setp.ne.u32(yes, t, 0);
            if yes {
                bra(done);
            }
            load(v, x, i);
            add.u32(at, i, width);
            load(other, x, at);
            add.rn.f32(v, v, other);
            bra(done);
        } else {
            mul.lo.u32(at, t, width);
            sub.u32(at, i, at);
        }
        label(ordinary);
        load(v, x, at);
        label(done);
        store(out, i, v);
    }

    pub unsafe extern "ptx-kernel" fn add(out: *mut f32, a: *const f32, b: *const f32, count: u32) {
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
        add.rn.f32(x, x, y);
        store(out, i, x);
    }

    /// All V classes, supervised at position 1 only. One warp per example.
    #[instances(7, 97, 193)]
    pub unsafe extern "ptx-kernel" fn loss<const V: usize>(
        dlogits: *mut f32,
        losses: *mut f32,
        hits: *mut u32,
        logits: *const f32,
        targets: *const u32,
        batch: u32,
    ) {
        let (i, b, lane, at, target): (u32, u32, u32, u32, u32);
        index(i);
        shr.u32(b, i, 5);
        and.b32(lane, i, 31);
        let past: pred;
        setp.ge.u32(past, b, batch);
        if past {
            ret();
        }
        load_u(target, targets, b);
        let (v, p): ([f32; (V + 31) / 32], [f32; (V + 31) / 32]);
        let (max, peer, total, chosen, n, zero): (f32, f32, f32, f32, f32, f32);
        mov.f32(max, f32::NEG_INFINITY);
        mov.f32(chosen, 0.0);
        mov.f32(zero, 0.0);
        let (cls, best, candidate): (u32, u32, u32);
        mov.u32(best, 0xffffffff);
        for j in 0..(V + 31) / 32 {
            add.u32(cls, lane, 32 * j);
            mov.f32(v[j], f32::NEG_INFINITY);
            setp.lt.u32(past, cls, V);
            if past {
                mad.lo.u32(at, b, 2 * V, cls);
                store(dlogits, at, zero);
                add.u32(at, at, V);
                load(v[j], logits, at);
            }
            max.f32(max, max, v[j]);
        }
        for j in 0..5 {
            shfl.sync.bfly.b32(peer, max, 1 << j, 31, 0xffffffff);
            max.f32(max, max, peer);
        }
        mov.f32(total, 0.0);
        for j in 0..(V + 31) / 32 {
            sub.rn.f32(p[j], v[j], max);
            exp(p[j]);
            add.rn.f32(total, total, p[j]);
            add.u32(cls, lane, 32 * j);
            setp.eq.u32(past, cls, target);
            if past {
                mov.f32(chosen, v[j]);
            }
            setp.eq.f32(past, v[j], max);
            if past {
                min.u32(best, best, cls);
            }
        }
        sum32(total);
        sum32(chosen);
        for j in 0..5 {
            shfl.sync.bfly.b32(candidate, best, 1 << j, 31, 0xffffffff);
            min.u32(best, best, candidate);
        }
        cvt.rn.f32.u32(n, batch);
        for j in 0..(V + 31) / 32 {
            add.u32(cls, lane, 32 * j);
            setp.lt.u32(past, cls, V);
            if past {
                div.rn.f32(p[j], p[j], total);
                let label: f32;
                let is_target: pred;
                setp.eq.u32(is_target, cls, target);
                selp.f32(label, 1.0, 0.0, is_target);
                sub.rn.f32(p[j], p[j], label);
                div.rn.f32(p[j], p[j], n);
                mad.lo.u32(at, b, 2 * V, cls);
                add.u32(at, at, V);
                store(dlogits, at, p[j]);
            }
        }
        setp.eq.u32(past, lane, 0);
        if past {
            lg2.approx.f32(total, total);
            mul.rn.f32(total, total, 0.6931471805599453);
            sub.rn.f32(chosen, max, chosen);
            add.rn.f32(total, total, chosen);
            store(losses, b, total);
            let hit: pred;
            setp.eq.u32(hit, best, target);
            selp.u32(best, 1, 0, hit);
            store_u(hits, b, best);
        }
    }

    /// Native AdamW: all parameters, including the tied embedding, updated once.
    pub unsafe extern "ptx-kernel" fn adamw(
        p: *mut f32,
        m: *mut f32,
        v: *mut f32,
        g: *const f32,
        count: u32,
        lr: f32,
        decay: f32,
        bc1: f32,
        bc2root: f32,
    ) {
        let i: u32;
        index(i);
        let past: pred;
        setp.ge.u32(past, i, count);
        if past {
            ret();
        }
        let (pi, mi, vi, gi, z, s): (f32, f32, f32, f32, f32, f32);
        load(pi, p, i);
        load(mi, m, i);
        load(vi, v, i);
        load(gi, g, i);
        sub.rn.f32(z, gi, mi);
        fma.rn.f32(mi, 0.1, z, mi);
        mul.rn.f32(z, gi, gi);
        mul.rn.f32(vi, vi, 0.98);
        fma.rn.f32(vi, 0.02, z, vi);
        store(m, i, mi);
        store(v, i, vi);
        sqrt.rn.f32(z, vi);
        div.rn.f32(z, z, bc2root);
        add.rn.f32(z, z, 1e-8);
        div.rn.f32(s, lr, bc1);
        mul.rn.f32(mi, mi, s);
        div.rn.f32(mi, mi, z);
        mul.rn.f32(s, lr, decay);
        sub.rn.f32(s, 1.0, s);
        mul.rn.f32(pi, pi, s);
        sub.rn.f32(pi, pi, mi);
        store(p, i, pi);
    }
}
