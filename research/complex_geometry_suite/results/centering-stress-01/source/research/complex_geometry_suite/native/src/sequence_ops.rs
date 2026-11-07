//! Variable-length prefix stabilization and masked cross entropy.
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
    fn load_u(v: u32, p: u64, i: u32) {
        let a: u64;
        mad.wide.u32(a, i, 4, p);
        ld.global.u32(v, [a]);
    }
    fn store_u(p: u64, i: u32, v: u32) {
        let a: u64;
        mad.wide.u32(a, i, 4, p);
        st.global.u32([a], v);
    }
    fn index(i: u32) {
        let (b, t, n): (u32, u32, u32);
        mov.u32(b, ctaid.x);
        mov.u32(t, tid.x);
        mov.u32(n, ntid.x);
        mad.lo.u32(i, b, n, t);
    }
    fn sum(v: f32) {
        let z: f32;
        for j in 0..5 {
            shfl.sync.bfly.b32(z, v, 1 << j, 31, 0xffffffff);
            add.rn.f32(v, v, z);
        }
    }
    fn exp(v: f32) {
        mul.rn.f32(v, v, 1.4426950408889634);
        ex2.approx.f32(v, v);
    }
    pub unsafe extern "ptx-kernel" fn prefix(
        out: *mut u32,
        tokens: *const u32,
        length: u32,
        rows: u32,
    ) {
        let (i, t, base, j, first, token, flag, at): (u32, u32, u32, u32, u32, u32, u32, u32);
        index(i);
        let past: pred;
        setp.ge.u32(past, i, rows);
        if past {
            ret();
        }
        rem.u32(t, i, length);
        sub.u32(base, i, t);
        load_u(first, tokens, base);
        mov.u32(j, 0);
        mov.u32(flag, 1);
        label(scan);
        add.u32(at, base, j);
        load_u(token, tokens, at);
        let different: pred;
        setp.ne.u32(different, token, first);
        if different {
            mov.u32(flag, 0);
        }
        add.u32(j, j, 1);
        let more: pred;
        setp.le.u32(more, j, t);
        if more {
            bra(scan);
        }
        store_u(out, i, flag);
    }
    pub unsafe extern "ptx-kernel" fn guard(
        out: *mut u32,
        prior: *const u32,
        head_ok: *const u32,
        length: u32,
        heads: u32,
        rows: u32,
    ) {
        let (i, b, h, at, flag, ok): (u32, u32, u32, u32, u32, u32);
        index(i);
        let past: pred;
        setp.ge.u32(past, i, rows);
        if past {
            ret();
        }
        div.u32(b, i, length);
        load_u(flag, prior, i);
        mov.u32(h, 0);
        label(scan);
        mad.lo.u32(at, b, heads, h);
        load_u(ok, head_ok, at);
        and.b32(flag, flag, ok);
        add.u32(h, h, 1);
        let more: pred;
        setp.lt.u32(more, h, heads);
        if more {
            bra(scan);
        }
        store_u(out, i, flag);
    }
    #[instances(false, true)]
    pub unsafe extern "ptx-kernel" fn share<const BACK: bool>(
        out: *mut f32,
        x: *const f32,
        flags: *const u32,
        width: u32,
        length: u32,
        count: u32,
    ) {
        let (i, row, t, at, flag, j, pos, first): (u32, u32, u32, u32, u32, u32, u32, u32);
        index(i);
        let past: pred;
        setp.ge.u32(past, i, count);
        if past {
            ret();
        }
        div.u32(row, i, width);
        rem.u32(t, row, length);
        load_u(flag, flags, row);
        let (v, z): (f32, f32);
        mov.f32(v, 0.0);
        let yes: pred;
        setp.ne.u32(yes, flag, 0);
        if BACK {
            let zero: pred;
            setp.eq.u32(zero, t, 0);
            if !zero {
                bra(ordinary);
            }
            load(v, x, i);
            mov.u32(j, 1);
            label(others);
            let more: pred;
            setp.lt.u32(more, j, length);
            if !more {
                bra(done);
            }
            add.u32(pos, row, j);
            load_u(flag, flags, pos);
            setp.ne.u32(yes, flag, 0);
            if yes {
                mad.lo.u32(at, j, width, i);
                load(z, x, at);
                add.rn.f32(v, v, z);
            }
            add.u32(j, j, 1);
            bra(others);
            label(ordinary);
            if !yes {
                load(v, x, i);
            }
        } else {
            mov.u32(at, i);
            if yes {
                mul.lo.u32(first, t, width);
                sub.u32(at, i, first);
            }
            load(v, x, at);
        }
        label(done);
        store(out, i, v);
    }
    #[instances(false, true)]
    pub unsafe extern "ptx-kernel" fn residual<const ATT: bool>(
        out: *mut f32,
        x: *const f32,
        branch: *const f32,
        flags: *const u32,
        width: u32,
        length: u32,
        count: u32,
        scale: f32,
    ) {
        let (i, row, t, at, flag, z): (u32, u32, u32, u32, u32, u32);
        index(i);
        let past: pred;
        setp.ge.u32(past, i, count);
        if past {
            ret();
        }
        div.u32(row, i, width);
        rem.u32(t, row, length);
        load_u(flag, flags, row);
        let shared: pred;
        setp.ne.u32(shared, flag, 0);
        let (v, r): (f32, f32);
        mov.f32(v, 0.0);
        mov.u32(at, i);
        if ATT {
            let first: pred;
            setp.eq.u32(first, t, 0);
            if first {
                bra(write);
            }
            if shared {
                bra(write);
            }
        } else {
            if shared {
                mul.lo.u32(z, t, width);
                sub.u32(at, i, z);
            }
        }
        load(v, branch, at);
        mul.rn.f32(v, v, scale);
        label(write);
        load(r, x, i);
        add.rn.f32(v, v, r);
        store(out, i, v);
    }
    // Attention branches are identically zero at first/shared positions.
    pub unsafe extern "ptx-kernel" fn attention_grad(
        out: *mut f32,
        dy: *const f32,
        flags: *const u32,
        width: u32,
        length: u32,
        count: u32,
        scale: f32,
    ) {
        let (i, row, t, flag): (u32, u32, u32, u32);
        index(i);
        let past: pred;
        setp.ge.u32(past, i, count);
        if past {
            ret();
        }
        div.u32(row, i, width);
        rem.u32(t, row, length);
        load_u(flag, flags, row);
        let v: f32;
        mov.f32(v, 0.0);
        let skip: pred;
        setp.eq.u32(skip, t, 0);
        if skip {
            bra(done);
        }
        setp.ne.u32(skip, flag, 0);
        if skip {
            bra(done);
        }
        load(v, dy, i);
        mul.rn.f32(v, v, scale);
        label(done);
        store(out, i, v);
    }
    pub unsafe extern "ptx-kernel" fn scale(out: *mut f32, x: *const f32, count: u32, gain: f32) {
        let i: u32;
        index(i);
        let past: pred;
        setp.ge.u32(past, i, count);
        if past {
            ret();
        }
        let v: f32;
        load(v, x, i);
        mul.rn.f32(v, v, gain);
        store(out, i, v);
    }
    pub unsafe extern "ptx-kernel" fn loss(
        dlogits: *mut f32,
        losses: *mut f32,
        predictions: *mut u32,
        logits: *const f32,
        targets: *const u32,
        vocab: u32,
        rows: u32,
        normalizer: f32,
    ) {
        let (i, row, lane, cls, at, target, best, peerclass): (
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
        shr.u32(row, i, 5);
        and.b32(lane, i, 31);
        let past: pred;
        setp.ge.u32(past, row, rows);
        if past {
            ret();
        }
        load_u(target, targets, row);
        let (v, max, peer, total, chosen, p, g, label): (f32, f32, f32, f32, f32, f32, f32, f32);
        mov.f32(max, f32::NEG_INFINITY);
        mov.f32(chosen, 0.0);
        mov.u32(best, 0xffffffff);
        mov.u32(cls, lane);
        label(maximum);
        let valid: pred;
        setp.lt.u32(valid, cls, vocab);
        if !valid {
            bra(reduce_max);
        }
        mad.lo.u32(at, row, vocab, cls);
        load(v, logits, at);
        max.f32(max, max, v);
        add.u32(cls, cls, 32);
        bra(maximum);
        label(reduce_max);
        for j in 0..5 {
            shfl.sync.bfly.b32(peer, max, 1 << j, 31, 0xffffffff);
            max.f32(max, max, peer);
        }
        mov.f32(total, 0.0);
        mov.u32(cls, lane);
        label(total_loop);
        setp.lt.u32(valid, cls, vocab);
        if !valid {
            bra(reduce_total);
        }
        mad.lo.u32(at, row, vocab, cls);
        load(v, logits, at);
        sub.rn.f32(p, v, max);
        exp(p);
        add.rn.f32(total, total, p);
        let eq: pred;
        setp.eq.u32(eq, cls, target);
        if eq {
            mov.f32(chosen, v);
        }
        setp.eq.f32(eq, v, max);
        if eq {
            min.u32(best, best, cls);
        }
        add.u32(cls, cls, 32);
        bra(total_loop);
        label(reduce_total);
        sum(total);
        sum(chosen);
        for j in 0..5 {
            shfl.sync.bfly.b32(peerclass, best, 1 << j, 31, 0xffffffff);
            min.u32(best, best, peerclass);
        }
        let supervised: pred;
        setp.lt.u32(supervised, target, vocab);
        mov.u32(cls, lane);
        label(gradient);
        setp.lt.u32(valid, cls, vocab);
        if !valid {
            bra(metrics);
        }
        mad.lo.u32(at, row, vocab, cls);
        mov.f32(g, 0.0);
        if supervised {
            load(v, logits, at);
            sub.rn.f32(p, v, max);
            exp(p);
            div.rn.f32(p, p, total);
            let hit: pred;
            setp.eq.u32(hit, cls, target);
            selp.f32(label, 1.0, 0.0, hit);
            sub.rn.f32(g, p, label);
            div.rn.f32(g, g, normalizer);
        }
        store(dlogits, at, g);
        add.u32(cls, cls, 32);
        bra(gradient);
        label(metrics);
        mov.f32(v, 0.0);
        if supervised {
            lg2.approx.f32(v, total);
            mul.rn.f32(v, v, 0.6931471805599453);
            add.rn.f32(v, v, max);
            sub.rn.f32(v, v, chosen);
        }
        let lead: pred;
        setp.eq.u32(lead, lane, 0);
        if lead {
            store(losses, row, v);
            store_u(predictions, row, best);
        }
    }
}
