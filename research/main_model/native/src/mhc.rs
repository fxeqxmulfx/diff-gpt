//! Four-stream dynamic mHC. Paper 2512.24880v2 section 4.2: twenty
//! column-then-row normalizations, sigmoid reads and twice-sigmoid writes.
//! A stable first log-space cycle is followed by 19 positive-space cycles.
//! All normalizations and the full adjoint use FP64. Backward recomputes
//! the forty normalized matrices in registers; no iteration tape on disk/host.
#[kernite_macros::kernels]
pub mod kernels {
    fn load(v: f64, p: u64, i: u32) {
        let a: u64;
        let f: f32;
        mad.wide.u32(a, i, 4, p);
        ld.global.f32(f, [a]);
        cvt.f64.f32(v, f);
    }
    fn store(p: u64, i: u32, v: f64) {
        let a: u64;
        let f: f32;
        cvt.rn.f32.f64(f, v);
        mad.wide.u32(a, i, 4, p);
        st.global.f32([a], f);
    }
    fn load64(v: f64, p: u64, i: u32) {
        let a: u64;
        mad.wide.u32(a, i, 8, p);
        ld.global.f64(v, [a]);
    }
    fn store64(p: u64, i: u32, v: f64) {
        let a: u64;
        mad.wide.u32(a, i, 8, p);
        st.global.f64([a], v);
    }
    fn index(i: u32) {
        let (b, t, n): (u32, u32, u32);
        mov.u32(b, ctaid.x);
        mov.u32(t, tid.x);
        mov.u32(n, ntid.x);
        mad.lo.u32(i, b, n, t);
    }
    fn shuffle(v: f64, shift: u32) {
        let (a, b, c, d): (u32, u32, u32, u32);
        mov.b64((a, b), v);
        shfl.sync.bfly.b32(c, a, shift, 31, 0xffffffff);
        shfl.sync.bfly.b32(d, b, shift, 31, 0xffffffff);
        mov.b64(v, (c, d));
    }
    fn sum4<const S: usize>(v: f64) {
        let other: f64;
        for j in 0..2 {
            mov.f64(other, v);
            let shift: u32;
            mov.u32(shift, 1 << (S + j));
            shuffle(other, shift);
            add.rn.f64(v, v, other);
        }
    }
    fn max4<const S: usize>(v: f64) {
        let other: f64;
        for j in 0..2 {
            mov.f64(other, v);
            let shift: u32;
            mov.u32(shift, 1 << (S + j));
            shuffle(other, shift);
            max.f64(v, v, other);
        }
    }
    fn sum32(v: f64) {
        let other: f64;
        for j in 0..5 {
            mov.f64(other, v);
            let shift: u32;
            mov.u32(shift, 1 << j);
            shuffle(other, shift);
            add.rn.f64(v, v, other);
        }
    }
    // Range reduction and twelve Taylor terms. Arguments here are <=0.
    // Below -700 the discarded mass is <1e-304, far below FP32 storage.
    fn exp64(y: f64, x: f64) {
        let tiny: pred;
        setp.lt.f64(tiny, x, -700.0f64);
        let bounded: f64;
        max.f64(bounded, x, -700.0f64);
        let (z, r, p, term, nf): (f64, f64, f64, f64, f64);
        let (n, e): (s32, s32);
        let bits: u64;
        mul.rn.f64(z, bounded, 1.4426950408889634f64);
        cvt.rni.s32.f64(n, z);
        cvt.rn.f64.s32(nf, n);
        fma.rn.f64(r, nf, -0.6931471805599453f64, bounded);
        mov.f64(y, 1.0f64);
        mov.f64(term, 1.0f64);
        for k in 1..13 {
            mul.rn.f64(term, term, r);
            div.rn.f64(term, term, k as f64);
            add.rn.f64(y, y, term);
        }
        add.s32(e, n, 1023);
        cvt.u64.s32(bits, e);
        shl.b64(bits, bits, 52);
        mov.b64(p, bits);
        mul.rn.f64(y, y, p);
        selp.f64(y, 0.0f64, y, tiny);
    }
    // Only used for sums in [1,4]. Mantissa reduction keeps |atanh argument|<=1/3.
    fn log64(out: f64, x: f64) {
        let (bits, mb, eb): (u64, u64, u64);
        let e: u32;
        let (m, y, den, square, power, term, ef): (f64, f64, f64, f64, f64, f64, f64);
        mov.b64(bits, x);
        shr.b64(eb, bits, 52);
        cvt.u32.u64(e, eb);
        sub.u32(e, e, 1023);
        cvt.rn.f64.u32(ef, e);
        and.b64(mb, bits, 0x000fffffffffffffu64);
        or.b64(mb, mb, 0x3ff0000000000000u64);
        mov.b64(m, mb);
        sub.rn.f64(y, m, 1.0f64);
        add.rn.f64(den, m, 1.0f64);
        div.rn.f64(y, y, den);
        mul.rn.f64(square, y, y);
        mov.f64(power, y);
        mov.f64(out, 0.0f64);
        for k in 0..12 {
            div.rn.f64(term, power, (2 * k + 1) as f64);
            add.rn.f64(out, out, term);
            mul.rn.f64(power, power, square);
        }
        mul.rn.f64(out, out, 2.0f64);
        fma.rn.f64(out, ef, 0.6931471805599453f64, out);
    }
    fn sigmoid(y: f64, x: f64) {
        let (a, e, den): (f64, f64, f64);
        let positive: pred;
        abs.f64(a, x);
        neg.f64(a, a);
        exp64(e, a);
        add.rn.f64(den, e, 1.0f64);
        setp.ge.f64(positive, x, 0.0f64);
        selp.f64(y, 1.0f64, e, positive);
        div.rn.f64(y, y, den);
    }
    pub unsafe extern "ptx-kernel" fn logits(
        y: *mut f32,
        projected: *const f32,
        params: *const f32,
        width: u32,
        count: u32,
    ) {
        let (i, c, group, at): (u32, u32, u32, u32);
        index(i);
        let past: pred;
        setp.ge.u32(past, i, count);
        if past {
            ret();
        }
        rem.u32(c, i, 24);
        div.u32(group, c, 4);
        min.u32(group, group, 2);
        let (x, a, b): (f64, f64, f64);
        load(x, projected, i);
        mad.lo.u32(at, width, 96, c);
        load(b, params, at);
        mad.lo.u32(at, width, 96, 24);
        add.u32(at, at, group);
        load(a, params, at);
        fma.rn.f64(x, x, a, b);
        store(y, i, x);
    }
    #[instances(false, true)]
    pub unsafe extern "ptx-kernel" fn sinkhorn<const BACK: bool>(
        output: *mut f32,
        logits: *const f32,
        upstream: *const f64,
        rows: u32,
    ) {
        let (row, lane, k, at): (u32, u32, u32, u32);
        mov.u32(row, ctaid.x);
        mov.u32(lane, tid.x);
        and.b32(k, lane, 15);
        let past: pred;
        setp.ge.u32(past, row, rows);
        if past {
            ret();
        }
        let (v, p, g, s, m, ls, tmp, neg): (f64, f64, f64, f64, f64, f64, f64, f64);
        let gate_lane: pred;
        setp.lt.u32(gate_lane, lane, 8);
        if !gate_lane {
            bra(skip_gates);
        }
        mad.lo.u32(at, row, 24, lane);
        load(v, logits, at);
        sigmoid(p, v);
        if BACK {
            // Stable derivative remains nonzero even when the stored sigmoid rounds to one.
            abs.f64(tmp, v);
            neg.f64(tmp, tmp);
            exp64(tmp, tmp);
            add.rn.f64(s, tmp, 1.0f64);
            mul.rn.f64(s, s, s);
            div.rn.f64(p, tmp, s);
            load64(g, upstream, at);
            mul.rn.f64(p, p, g);
        }
        let post: pred;
        setp.ge.u32(post, lane, 4);
        if post {
            mul.rn.f64(p, p, 2.0f64);
        }
        store(output, at, p);
        label(skip_gates);
        mad.lo.u32(at, row, 24, 8);
        add.u32(at, at, k);
        load(v, logits, at);
        let tape: [f64; 40];
        for step in 0..40 {
            if step < 2 {
                // Keep the first cycle in log space so even an entire very
                // small row survives before its normalization. Afterward,
                // row sums are one and column sums >=1/4. Alternation keeps
                // each next denominator in [1/4,4]; ordinary division is safe.
                mov.f64(m, v);
                if step % 2 == 0 {
                    max4::<2>(m);
                } else {
                    max4::<0>(m);
                }
                sub.rn.f64(v, v, m);
                exp64(p, v);
            }
            mov.f64(s, p);
            if step % 2 == 0 {
                sum4::<2>(s);
            } else {
                sum4::<0>(s);
            }
            if step < 2 {
                log64(ls, s);
                sub.rn.f64(v, v, ls);
            }
            div.rn.f64(p, p, s);
            if BACK {
                mov.f64(tape[step], p);
            }
        }
        if BACK {
            load64(g, upstream, at);
            mul.rn.f64(g, g, p);
            for step in 0..40 {
                mov.f64(s, g);
                if step % 2 == 0 {
                    sum4::<0>(s);
                } else {
                    sum4::<2>(s);
                }
                neg.f64(neg, tape[39 - step]);
                fma.rn.f64(g, neg, s, g);
            }
            mov.f64(p, g);
        }
        let active: pred;
        setp.lt.u32(active, lane, 16);
        if active {
            store(output, at, p);
        }
    }
    #[instances(false, true)]
    pub unsafe extern "ptx-kernel" fn rms<const BACK: bool>(
        y: *mut f32,
        x: *const f32,
        dy: *const f32,
        width: u32,
        rows: u32,
    ) {
        let (row, lane, c, at): (u32, u32, u32, u32);
        mov.u32(row, ctaid.x);
        mov.u32(lane, tid.x);
        let past: pred;
        setp.ge.u32(past, row, rows);
        if past {
            ret();
        }
        let (sum, dot, v, g, n, inv, factor, z): (f64, f64, f64, f64, f64, f64, f64, f64);
        mov.f64(sum, 0.0f64);
        mov.f64(dot, 0.0f64);
        mov.u32(c, lane);
        label(rms_reduce);
        let done: pred;
        setp.ge.u32(done, c, width);
        if done {
            bra(rms_reduced);
        }
        mad.lo.u32(at, row, width, c);
        load(v, x, at);
        fma.rn.f64(sum, v, v, sum);
        if BACK {
            load(g, dy, at);
            fma.rn.f64(dot, v, g, dot);
        }
        add.u32(c, c, 32);
        bra(rms_reduce);
        label(rms_reduced);
        sum32(sum);
        sum32(dot);
        cvt.rn.f64.u32(n, width);
        div.rn.f64(sum, sum, n);
        let (den, radial): (f64, f64);
        let positive: pred;
        setp.gt.f64(positive, sum, 0.0f64);
        selp.f64(den, sum, 1.0f64, positive);
        add.rn.f64(sum, sum, 0.00000011920928955078125f64);
        sqrt.rn.f64(inv, sum);
        div.rn.f64(inv, 1.0f64, inv);
        div.rn.f64(factor, dot, n);
        div.rn.f64(factor, factor, den);
        mul.rn.f64(radial, inv, inv);
        mul.rn.f64(radial, radial, inv);
        mul.rn.f64(radial, radial, 0.00000011920928955078125f64);
        mov.u32(c, lane);
        label(rms_store);
        setp.ge.u32(done, c, width);
        if done {
            bra(rms_finished);
        }
        mad.lo.u32(at, row, width, c);
        load(v, x, at);
        if BACK {
            load(g, dy, at);
            mul.rn.f64(z, v, factor);
            sub.rn.f64(v, g, z);
            mul.rn.f64(v, v, inv);
            mul.rn.f64(z, z, radial);
            add.rn.f64(v, v, z);
        } else {
            mul.rn.f64(v, v, inv);
        }
        store(y, at, v);
        add.u32(c, c, 32);
        bra(rms_store);
        label(rms_finished);
    }
    #[instances(false, true)]
    pub unsafe extern "ptx-kernel" fn expand<const BACK: bool>(
        y: *mut f32,
        x: *const f32,
        width: u32,
        count: u32,
    ) {
        let (i, row, c, at): (u32, u32, u32, u32);
        index(i);
        let past: pred;
        setp.ge.u32(past, i, count);
        if past {
            ret();
        }
        div.u32(row, i, width);
        rem.u32(c, i, width);
        let (v, s): (f64, f64);
        if BACK {
            mov.f64(s, 0.0f64);
            for j in 0..4 {
                mad.lo.u32(at, row, 4, j);
                mad.lo.u32(at, at, width, c);
                load(v, x, at);
                add.rn.f64(s, s, v);
            }
            store(y, i, s);
        } else {
            load(v, x, i);
            for j in 0..4 {
                mad.lo.u32(at, row, 4, j);
                mad.lo.u32(at, at, width, c);
                store(y, at, v);
            }
        }
    }
    #[instances(false, true)]
    pub unsafe extern "ptx-kernel" fn collapse<const BACK: bool>(
        y: *mut f32,
        x: *const f32,
        width: u32,
        count: u32,
    ) {
        let (i, row, c, at): (u32, u32, u32, u32);
        index(i);
        let past: pred;
        setp.ge.u32(past, i, count);
        if past {
            ret();
        }
        div.u32(row, i, width);
        rem.u32(c, i, width);
        let (v, s): (f64, f64);
        if BACK {
            load(v, x, i);
            mul.rn.f64(v, v, 0.25f64);
            for j in 0..4 {
                mad.lo.u32(at, row, 4, j);
                mad.lo.u32(at, at, width, c);
                store(y, at, v);
            }
        } else {
            mov.f64(s, 0.0f64);
            for j in 0..4 {
                mad.lo.u32(at, row, 4, j);
                mad.lo.u32(at, at, width, c);
                load(v, x, at);
                add.rn.f64(s, s, v);
            }
            mul.rn.f64(s, s, 0.25f64);
            store(y, i, s);
        }
    }
    pub unsafe extern "ptx-kernel" fn read(
        y: *mut f64,
        x: *const f32,
        gates: *const f32,
        width: u32,
        count: u32,
    ) {
        let (i, row, c, at): (u32, u32, u32, u32);
        index(i);
        let past: pred;
        setp.ge.u32(past, i, count);
        if past {
            ret();
        }
        div.u32(row, i, width);
        rem.u32(c, i, width);
        let (v, g, s): (f64, f64, f64);
        mov.f64(s, 0.0f64);
        for j in 0..4 {
            mad.lo.u32(at, row, 24, j);
            load(g, gates, at);
            mad.lo.u32(at, row, 4, j);
            mad.lo.u32(at, at, width, c);
            load(v, x, at);
            fma.rn.f64(s, v, g, s);
        }
        store64(y, i, s);
    }
    pub unsafe extern "ptx-kernel" fn write(
        y: *mut f32,
        x: *const f32,
        branch: *const f32,
        gates: *const f32,
        width: u32,
        count: u32,
        scale: f32,
    ) {
        let (i, row, c, stream, at, j): (u32, u32, u32, u32, u32, u32);
        index(i);
        let past: pred;
        setp.ge.u32(past, i, count);
        if past {
            ret();
        }
        rem.u32(c, i, width);
        div.u32(row, i, width);
        rem.u32(stream, row, 4);
        div.u32(row, row, 4);
        let (s, v, g, b, sc): (f64, f64, f64, f64, f64);
        cvt.f64.f32(sc, scale);
        mad.lo.u32(at, row, width, c);
        load(b, branch, at);
        mul.rn.f64(b, b, sc);
        mad.lo.u32(at, row, 24, 4);
        add.u32(at, at, stream);
        load(g, gates, at);
        mul.rn.f64(s, g, b);
        for k in 0..4 {
            mad.lo.u32(at, row, 24, 8 + k);
            mad.lo.u32(at, stream, 4, at);
            load(g, gates, at);
            mad.lo.u32(j, row, 4, k);
            mad.lo.u32(j, j, width, c);
            load(v, x, j);
            fma.rn.f64(s, g, v, s);
        }
        store(y, i, s);
    }
    // Skip transpose and the branch derivative. Coefficient gradients are separate reductions.
    pub unsafe extern "ptx-kernel" fn write_backward(
        dx: *mut f32,
        db: *mut f32,
        dy: *const f32,
        gates: *const f32,
        width: u32,
        count: u32,
        scale: f32,
    ) {
        let (i, row, c, at, j): (u32, u32, u32, u32, u32);
        index(i);
        let past: pred;
        setp.ge.u32(past, i, count);
        if past {
            ret();
        }
        div.u32(row, i, width);
        rem.u32(c, i, width);
        let (g, v, s, b, sc): (f64, f64, f64, f64, f64);
        mov.f64(b, 0.0f64);
        cvt.f64.f32(sc, scale);
        for k in 0..4 {
            mov.f64(s, 0.0f64);
            for t in 0..4 {
                mad.lo.u32(at, row, 24, 8 + t * 4 + k);
                load(g, gates, at);
                mad.lo.u32(j, row, 4, t);
                mad.lo.u32(j, j, width, c);
                load(v, dy, j);
                fma.rn.f64(s, g, v, s);
            }
            mad.lo.u32(at, row, 4, k);
            mad.lo.u32(at, at, width, c);
            store(dx, at, s);
            load(v, dy, at);
            mad.lo.u32(at, row, 24, 4 + k);
            load(g, gates, at);
            fma.rn.f64(b, g, v, b);
        }
        mul.rn.f64(b, b, sc);
        store(db, i, b);
    }
    pub unsafe extern "ptx-kernel" fn read_backward(
        dx: *mut f32,
        dr: *const f64,
        gates: *const f32,
        width: u32,
        count: u32,
    ) {
        let (i, row, c, at): (u32, u32, u32, u32);
        index(i);
        let past: pred;
        setp.ge.u32(past, i, count);
        if past {
            ret();
        }
        div.u32(row, i, width);
        rem.u32(c, i, width);
        let (v, g, s): (f64, f64, f64);
        load64(v, dr, i);
        for k in 0..4 {
            mad.lo.u32(at, row, 24, k);
            load(g, gates, at);
            mad.lo.u32(at, row, 4, k);
            mad.lo.u32(at, at, width, c);
            load(s, dx, at);
            fma.rn.f64(s, v, g, s);
            store(dx, at, s);
        }
    }
    pub unsafe extern "ptx-kernel" fn gate_grad(
        dg: *mut f64,
        x: *const f32,
        branch: *const f32,
        dy: *const f32,
        dr: *const f64,
        width: u32,
        rows: u32,
        scale: f32,
    ) {
        let (id, row, gate, lane, c, at, j, si, sj): (u32, u32, u32, u32, u32, u32, u32, u32, u32);
        mov.u32(id, ctaid.x);
        mov.u32(lane, tid.x);
        div.u32(row, id, 24);
        rem.u32(gate, id, 24);
        let past: pred;
        setp.ge.u32(past, row, rows);
        if past {
            ret();
        }
        let (sum, a, b, sc): (f64, f64, f64, f64);
        mov.f64(sum, 0.0f64);
        cvt.f64.f32(sc, scale);
        mov.u32(c, lane);
        label(gate_loop);
        let done: pred;
        setp.ge.u32(done, c, width);
        if done {
            bra(gate_done);
        }
        let (rd, wr): (pred, pred);
        setp.lt.u32(rd, gate, 4);
        setp.lt.u32(wr, gate, 8);
        if !rd {
            bra(gate_not_read);
        }
        mad.lo.u32(at, row, 4, gate);
        mad.lo.u32(at, at, width, c);
        load(a, x, at);
        mad.lo.u32(j, row, width, c);
        load64(b, dr, j);
        bra(gate_accumulate);
        label(gate_not_read);
        if !wr {
            bra(gate_skip);
        }
        sub.u32(si, gate, 4);
        mad.lo.u32(at, row, 4, si);
        mad.lo.u32(at, at, width, c);
        load(a, dy, at);
        mad.lo.u32(j, row, width, c);
        load(b, branch, j);
        mul.rn.f64(b, b, sc);
        bra(gate_accumulate);
        label(gate_skip);
        sub.u32(j, gate, 8);
        div.u32(si, j, 4);
        rem.u32(sj, j, 4);
        mad.lo.u32(at, row, 4, si);
        mad.lo.u32(at, at, width, c);
        load(a, dy, at);
        mad.lo.u32(at, row, 4, sj);
        mad.lo.u32(at, at, width, c);
        load(b, x, at);
        label(gate_accumulate);
        fma.rn.f64(sum, a, b, sum);
        add.u32(c, c, 32);
        bra(gate_loop);
        label(gate_done);
        sum32(sum);
        let first: pred;
        setp.eq.u32(first, lane, 0);
        if first {
            store64(dg, id, sum);
        }
    }
    pub unsafe extern "ptx-kernel" fn projection_grad(
        dp: *mut f32,
        dlog: *const f32,
        params: *const f32,
        width: u32,
        count: u32,
    ) {
        let (i, c, group, at): (u32, u32, u32, u32);
        index(i);
        let past: pred;
        setp.ge.u32(past, i, count);
        if past {
            ret();
        }
        rem.u32(c, i, 24);
        div.u32(group, c, 4);
        min.u32(group, group, 2);
        mad.lo.u32(at, width, 96, 24);
        add.u32(at, at, group);
        let (g, a): (f64, f64);
        load(a, params, at);
        load(g, dlog, i);
        mul.rn.f64(g, g, a);
        store(dp, i, g);
    }
    pub unsafe extern "ptx-kernel" fn parameter_grad(
        dp: *mut f32,
        dlog: *const f32,
        projected: *const f32,
        width: u32,
        rows: u32,
    ) {
        let (idx, lane, row, c, at, group): (u32, u32, u32, u32, u32, u32);
        mov.u32(idx, ctaid.x);
        mov.u32(lane, tid.x);
        let (sum, g, v): (f64, f64, f64);
        mov.f64(sum, 0.0f64);
        mov.u32(row, lane);
        label(param_loop);
        let done: pred;
        setp.ge.u32(done, row, rows);
        if done {
            bra(param_done);
        }
        let bias: pred;
        setp.lt.u32(bias, idx, 24);
        if !bias {
            bra(param_alpha);
        }
        mad.lo.u32(at, row, 24, idx);
        load(g, dlog, at);
        add.rn.f64(sum, sum, g);
        bra(param_next);
        label(param_alpha);
        sub.u32(group, idx, 24);
        for j in 0..24 {
            let matching: pred;
            if j < 4 {
                setp.eq.u32(matching, group, 0);
            } else {
                if j < 8 {
                    setp.eq.u32(matching, group, 1);
                } else {
                    setp.eq.u32(matching, group, 2);
                }
            }
            if matching {
                mad.lo.u32(at, row, 24, j);
                load(g, dlog, at);
                load(v, projected, at);
                fma.rn.f64(sum, g, v, sum);
            }
        }
        label(param_next);
        add.u32(row, row, 32);
        bra(param_loop);
        label(param_done);
        sum32(sum);
        let first: pred;
        setp.eq.u32(first, lane, 0);
        if first {
            mad.lo.u32(at, width, 96, idx);
            store(dp, at, sum);
        }
    }

    #[instances(false, true)]
    pub unsafe extern "ptx-kernel" fn read_rms<const BACK: bool>(
        y: *mut f32,
        dx: *mut f64,
        x: *const f64,
        dy: *const f32,
        width: u32,
        rows: u32,
    ) {
        let (row, lane, c, at): (u32, u32, u32, u32);
        mov.u32(row, ctaid.x);
        mov.u32(lane, tid.x);
        let past: pred;
        setp.ge.u32(past, row, rows);
        if past {
            ret();
        }
        let (energy, dot, v, g, n, inv, den, parallel, tangent, radial): (
            f64,
            f64,
            f64,
            f64,
            f64,
            f64,
            f64,
            f64,
            f64,
            f64,
        );
        mov.f64(energy, 0.0f64);
        mov.f64(dot, 0.0f64);
        mov.u32(c, lane);
        label(reduce);
        let done: pred;
        setp.ge.u32(done, c, width);
        if done {
            bra(reduced);
        }
        mad.lo.u32(at, row, width, c);
        load64(v, x, at);
        fma.rn.f64(energy, v, v, energy);
        if BACK {
            load(g, dy, at);
            fma.rn.f64(dot, v, g, dot);
        }
        add.u32(c, c, 32);
        bra(reduce);
        label(reduced);
        sum32(energy);
        sum32(dot);
        cvt.rn.f64.u32(n, width);
        div.rn.f64(energy, energy, n);
        div.rn.f64(dot, dot, n);
        let positive: pred;
        setp.gt.f64(positive, energy, 0.0f64);
        selp.f64(den, energy, 1.0f64, positive);
        add.rn.f64(inv, energy, 0.00000011920928955078125f64);
        sqrt.rn.f64(inv, inv);
        div.rn.f64(inv, 1.0f64, inv);
        mul.rn.f64(radial, inv, inv);
        mul.rn.f64(radial, radial, inv);
        mul.rn.f64(radial, radial, 0.00000011920928955078125f64);
        mov.u32(c, lane);
        label(write);
        setp.ge.u32(done, c, width);
        if done {
            bra(finished);
        }
        mad.lo.u32(at, row, width, c);
        load64(v, x, at);
        if BACK {
            load(g, dy, at);
            mul.rn.f64(parallel, v, dot);
            div.rn.f64(parallel, parallel, den);
            sub.rn.f64(tangent, g, parallel);
            mul.rn.f64(tangent, tangent, inv);
            mul.rn.f64(parallel, parallel, radial);
            add.rn.f64(tangent, tangent, parallel);
            store64(dx, at, tangent);
        } else {
            mul.rn.f64(v, v, inv);
            store(y, at, v);
        }
        add.u32(c, c, 32);
        bra(write);
        label(finished);
    }
}
