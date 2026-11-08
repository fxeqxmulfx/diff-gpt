//! Four-stream dynamic mHC. Paper 2512.24880v2 section 4.2: twenty
//! column-then-row normalizations, sigmoid reads and twice-sigmoid writes.
//! A stable first log-space cycle is followed by 19 positive-space cycles.
//! All normalizations and the full adjoint use FP32. Backward recomputes
//! the forty normalized matrices in registers; no iteration tape on disk/host.
#[kernite_macros::kernels]
pub mod kernels {
    fn load(v: f32, p: u64, i: u32) {
        let a: u64;
        let f: f32;
        mad.wide.u32(a, i, 4, p);
        ld.global.f32(f, [a]);
        mov.f32(v, f);
    }
    fn store(p: u64, i: u32, v: f32) {
        let a: u64;
        let f: f32;
        mov.f32(f, v);
        mad.wide.u32(a, i, 4, p);
        st.global.f32([a], f);
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
    fn index(i: u32) {
        let (b, t, n): (u32, u32, u32);
        mov.u32(b, ctaid.x);
        mov.u32(t, tid.x);
        mov.u32(n, ntid.x);
        mad.lo.u32(i, b, n, t);
    }
    fn shuffle(v: f32, shift: u32) {
        let x: f32;
        shfl.sync.bfly.b32(x, v, shift, 31, 0xffffffff);
        mov.f32(v, x);
    }
    fn sum4<const S: usize>(v: f32) {
        let other: f32;
        for j in 0..2 {
            mov.f32(other, v);
            let shift: u32;
            mov.u32(shift, 1 << (S + j));
            shuffle(other, shift);
            add.rn.f32(v, v, other);
        }
    }
    fn max4<const S: usize>(v: f32) {
        let other: f32;
        for j in 0..2 {
            mov.f32(other, v);
            let shift: u32;
            mov.u32(shift, 1 << (S + j));
            shuffle(other, shift);
            max.f32(v, v, other);
        }
    }
    fn sum32(v: f32) {
        let other: f32;
        for j in 0..5 {
            mov.f32(other, v);
            let shift: u32;
            mov.u32(shift, 1 << j);
            shuffle(other, shift);
            add.rn.f32(v, v, other);
        }
    }
    // Native fixed FP32 exponential for nonpositive arguments.
    fn exp64(y: f32, x: f32) {
        let z: f32;
        mul.rn.f32(z, x, 1.4426950408889634f32);
        ex2.approx.f32(y, z);
    }
    // Only used for sums in [1,4]. Mantissa reduction keeps |atanh argument|<=1/3.
    fn log64(out: f32, x: f32) {
        lg2.approx.f32(out, x);
        mul.rn.f32(out, out, 0.6931471805599453f32);
    }
    // Error-free sum when finite and representable. Both words stay FP32.
    fn two_sum(s: f32, e: f32, a: f32, b: f32) {
        let (aa, bb, bv, av, br, ar): (f32, f32, f32, f32, f32, f32);
        mov.f32(aa, a);
        mov.f32(bb, b);
        add.rn.f32(s, aa, bb);
        sub.rn.f32(bv, s, aa);
        sub.rn.f32(av, s, bv);
        sub.rn.f32(br, bb, bv);
        sub.rn.f32(ar, aa, av);
        add.rn.f32(e, ar, br);
    }
    fn sigmoid(y: f32, x: f32) {
        let (a, e, den): (f32, f32, f32);
        let positive: pred;
        abs.f32(a, x);
        neg.f32(a, a);
        exp64(e, a);
        add.rn.f32(den, e, 1.0f32);
        setp.ge.f32(positive, x, 0.0f32);
        selp.f32(y, 1.0f32, e, positive);
        div.rn.f32(y, y, den);
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
        let (x, a, b): (f32, f32, f32);
        load(x, projected, i);
        mad.lo.u32(at, width, 96, c);
        load(b, params, at);
        mad.lo.u32(at, width, 96, 24);
        add.u32(at, at, group);
        load(a, params, at);
        fma.rn.f32(x, x, a, b);
        store(y, i, x);
    }
    #[instances(false, true)]
    pub unsafe extern "ptx-kernel" fn sinkhorn<const BACK: bool>(
        output: *mut f32,
        logits: *const f32,
        upstream: *const f32,
        rows: u32,
    ) {
        let (row, lane, k, at): (u32, u32, u32, u32);
        // One independent 4x4 matrix in each half warp. The original kernel
        // computed the matrix twice and discarded lanes16..31. XOR shuffles
        // by1,2,4,8 stay inside their half and preserve every reduction order.
        mov.u32(row, ctaid.x);
        mov.u32(lane, tid.x);
        shr.u32(k, lane, 4);
        mad.lo.u32(row, row, 2, k);
        and.b32(lane, lane, 15);
        mov.u32(k, lane);
        let past: pred;
        setp.ge.u32(past, row, rows);
        if past {
            ret();
        }
        let (v, p, g, s, m, ls, tmp, neg): (f32, f32, f32, f32, f32, f32, f32, f32);
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
            abs.f32(tmp, v);
            neg.f32(tmp, tmp);
            exp64(tmp, tmp);
            add.rn.f32(s, tmp, 1.0f32);
            mul.rn.f32(s, s, s);
            div.rn.f32(p, tmp, s);
            load64(g, upstream, at);
            mul.rn.f32(p, p, g);
        }
        let post: pred;
        setp.ge.u32(post, lane, 4);
        if post {
            mul.rn.f32(p, p, 2.0f32);
        }
        store(output, at, p);
        label(skip_gates);
        mad.lo.u32(at, row, 24, 8);
        add.u32(at, at, k);
        load(v, logits, at);
        let tape: [f32; 40];
        let (low, correction, argument): (f32, f32, f32);
        mov.f32(low, 0.0f32);
        for step in 0..40 {
            if step < 2 {
                // Keep the first cycle in log space so even an entire very
                // small row survives before its normalization. Afterward,
                // row sums are one and column sums >=1/4. Alternation keeps
                // each next denominator in [1/4,4]; ordinary division is safe.
                mov.f32(m, v);
                if step % 2 == 0 {
                    max4::<2>(m);
                } else {
                    max4::<0>(m);
                }
                neg.f32(m, m);
                two_sum(v, correction, v, m);
                add.rn.f32(low, low, correction);
                add.rn.f32(argument, v, low);
                exp64(p, argument);
            }
            mov.f32(s, p);
            if step % 2 == 0 {
                sum4::<2>(s);
            } else {
                sum4::<0>(s);
            }
            if step < 2 {
                log64(ls, s);
                neg.f32(ls, ls);
                two_sum(v, correction, v, ls);
                add.rn.f32(low, low, correction);
            }
            div.rn.f32(p, p, s);
            if BACK {
                mov.f32(tape[step], p);
            }
        }
        if BACK {
            load64(g, upstream, at);
            mul.rn.f32(g, g, p);
            for step in 0..40 {
                mov.f32(s, g);
                if step % 2 == 0 {
                    sum4::<0>(s);
                } else {
                    sum4::<2>(s);
                }
                neg.f32(neg, tape[39 - step]);
                fma.rn.f32(g, neg, s, g);
            }
            mov.f32(p, g);
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
        let warp: u32;
        shr.u32(warp, lane, 5);
        mad.lo.u32(row, row, 4, warp);
        and.b32(lane, lane, 31);
        let past: pred;
        setp.ge.u32(past, row, rows);
        if past {
            ret();
        }
        let (sum, dot, v, g, n, inv, factor, z): (f32, f32, f32, f32, f32, f32, f32, f32);
        let (xscale, gscale, ax, ag, epsilon, ratio): (f32, f32, f32, f32, f32, f32);
        mov.f32(xscale, 0.00034526698300124393f32);
        mov.f32(gscale, 0.00034526698300124393f32);
        mov.u32(c, lane);
        let done: pred;
        label(rms_maximum);
        setp.ge.u32(done, c, width);
        if done {
            bra(rms_maximum_done);
        }
        mad.lo.u32(at, row, width, c);
        load(v, x, at);
        abs.f32(ax, v);
        max.f32(xscale, xscale, ax);
        if BACK {
            load(g, dy, at);
            abs.f32(ag, g);
            max.f32(gscale, gscale, ag);
        }
        add.u32(c, c, 32);
        bra(rms_maximum);
        label(rms_maximum_done);
        for j in 0..5 {
            shfl.sync.bfly.b32(ax, xscale, 1 << j, 31, 0xffffffff);
            max.f32(xscale, xscale, ax);
            if BACK {
                shfl.sync.bfly.b32(ag, gscale, 1 << j, 31, 0xffffffff);
                max.f32(gscale, gscale, ag);
            }
        }
        div.rn.f32(epsilon, 0.00000011920928955078125f32, xscale);
        div.rn.f32(epsilon, epsilon, xscale);
        div.rn.f32(ratio, gscale, xscale);
        mov.f32(sum, 0.0f32);
        mov.f32(dot, 0.0f32);
        mov.u32(c, lane);
        label(rms_reduce);
        setp.ge.u32(done, c, width);
        if done {
            bra(rms_reduced);
        }
        mad.lo.u32(at, row, width, c);
        load(v, x, at);
        div.rn.f32(v, v, xscale);
        fma.rn.f32(sum, v, v, sum);
        if BACK {
            load(g, dy, at);
            div.rn.f32(g, g, gscale);
            fma.rn.f32(dot, v, g, dot);
        }
        add.u32(c, c, 32);
        bra(rms_reduce);
        label(rms_reduced);
        sum32(sum);
        sum32(dot);
        cvt.rn.f32.u32(n, width);
        div.rn.f32(sum, sum, n);
        let (den, radial): (f32, f32);
        let positive: pred;
        setp.gt.f32(positive, sum, 0.0f32);
        selp.f32(den, sum, 1.0f32, positive);
        add.rn.f32(sum, sum, epsilon);
        sqrt.rn.f32(inv, sum);
        div.rn.f32(inv, 1.0f32, inv);
        div.rn.f32(factor, dot, n);
        div.rn.f32(factor, factor, den);
        mul.rn.f32(radial, inv, inv);
        mul.rn.f32(radial, radial, inv);
        mul.rn.f32(radial, radial, epsilon);
        mov.u32(c, lane);
        label(rms_store);
        setp.ge.u32(done, c, width);
        if done {
            bra(rms_finished);
        }
        mad.lo.u32(at, row, width, c);
        load(v, x, at);
        div.rn.f32(v, v, xscale);
        if BACK {
            load(g, dy, at);
            div.rn.f32(g, g, gscale);
            mul.rn.f32(z, v, factor);
            sub.rn.f32(v, g, z);
            mul.rn.f32(v, v, inv);
            mul.rn.f32(z, z, radial);
            add.rn.f32(v, v, z);
            mul.rn.f32(v, v, ratio);
        } else {
            mul.rn.f32(v, v, inv);
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
        let (v, s): (f32, f32);
        if BACK {
            mov.f32(s, 0.0f32);
            for j in 0..4 {
                mad.lo.u32(at, row, 4, j);
                mad.lo.u32(at, at, width, c);
                load(v, x, at);
                add.rn.f32(s, s, v);
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
        let (v, s): (f32, f32);
        if BACK {
            load(v, x, i);
            mul.rn.f32(v, v, 0.25f32);
            for j in 0..4 {
                mad.lo.u32(at, row, 4, j);
                mad.lo.u32(at, at, width, c);
                store(y, at, v);
            }
        } else {
            mov.f32(s, 0.0f32);
            for j in 0..4 {
                mad.lo.u32(at, row, 4, j);
                mad.lo.u32(at, at, width, c);
                load(v, x, at);
                add.rn.f32(s, s, v);
            }
            mul.rn.f32(s, s, 0.25f32);
            store(y, i, s);
        }
    }
    pub unsafe extern "ptx-kernel" fn read(
        y: *mut f32,
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
        let (v, g, s): (f32, f32, f32);
        mov.f32(s, 0.0f32);
        for j in 0..4 {
            mad.lo.u32(at, row, 24, j);
            load(g, gates, at);
            mad.lo.u32(at, row, 4, j);
            mad.lo.u32(at, at, width, c);
            load(v, x, at);
            fma.rn.f32(s, v, g, s);
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
        let (s, v, g, b, sc): (f32, f32, f32, f32, f32);
        mov.f32(sc, scale);
        mad.lo.u32(at, row, width, c);
        load(b, branch, at);
        mul.rn.f32(b, b, sc);
        mad.lo.u32(at, row, 24, 4);
        add.u32(at, at, stream);
        load(g, gates, at);
        mul.rn.f32(s, g, b);
        for k in 0..4 {
            mad.lo.u32(at, row, 24, 8 + k);
            mad.lo.u32(at, stream, 4, at);
            load(g, gates, at);
            mad.lo.u32(j, row, 4, k);
            mad.lo.u32(j, j, width, c);
            load(v, x, j);
            fma.rn.f32(s, g, v, s);
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
        let (g, v, s, b, sc): (f32, f32, f32, f32, f32);
        mov.f32(b, 0.0f32);
        mov.f32(sc, scale);
        for k in 0..4 {
            mov.f32(s, 0.0f32);
            for t in 0..4 {
                mad.lo.u32(at, row, 24, 8 + t * 4 + k);
                load(g, gates, at);
                mad.lo.u32(j, row, 4, t);
                mad.lo.u32(j, j, width, c);
                load(v, dy, j);
                fma.rn.f32(s, g, v, s);
            }
            mad.lo.u32(at, row, 4, k);
            mad.lo.u32(at, at, width, c);
            store(dx, at, s);
            load(v, dy, at);
            mad.lo.u32(at, row, 24, 4 + k);
            load(g, gates, at);
            fma.rn.f32(b, g, v, b);
        }
        mul.rn.f32(b, b, sc);
        store(db, i, b);
    }
    pub unsafe extern "ptx-kernel" fn read_backward(
        dx: *mut f32,
        dr: *const f32,
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
        let (v, g, s): (f32, f32, f32);
        load64(v, dr, i);
        for k in 0..4 {
            mad.lo.u32(at, row, 24, k);
            load(g, gates, at);
            mad.lo.u32(at, row, 4, k);
            mad.lo.u32(at, at, width, c);
            load(s, dx, at);
            fma.rn.f32(s, v, g, s);
            store(dx, at, s);
        }
    }
    pub unsafe extern "ptx-kernel" fn gate_grad(
        dg: *mut f32,
        x: *const f32,
        branch: *const f32,
        dy: *const f32,
        dr: *const f32,
        width: u32,
        rows: u32,
        scale: f32,
    ) {
        let (id, row, gate, lane, c, at, j, si, sj): (u32, u32, u32, u32, u32, u32, u32, u32, u32);
        mov.u32(id, ctaid.x);
        mov.u32(lane, tid.x);
        let warp: u32;
        shr.u32(warp, lane, 5);
        mad.lo.u32(id, id, 4, warp);
        and.b32(lane, lane, 31);
        div.u32(row, id, 24);
        rem.u32(gate, id, 24);
        let past: pred;
        setp.ge.u32(past, row, rows);
        if past {
            ret();
        }
        let (sum, a, b, sc): (f32, f32, f32, f32);
        mov.f32(sum, 0.0f32);
        mov.f32(sc, scale);
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
        mul.rn.f32(b, b, sc);
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
        fma.rn.f32(sum, a, b, sum);
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
        let (g, a): (f32, f32);
        load(a, params, at);
        load(g, dlog, i);
        mul.rn.f32(g, g, a);
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
        let (sum, g, v): (f32, f32, f32);
        mov.f32(sum, 0.0f32);
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
        add.rn.f32(sum, sum, g);
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
                fma.rn.f32(sum, g, v, sum);
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
        dx: *mut f32,
        x: *const f32,
        dy: *const f32,
        width: u32,
        rows: u32,
    ) {
        let (row, lane, c, at): (u32, u32, u32, u32);
        mov.u32(row, ctaid.x);
        mov.u32(lane, tid.x);
        let warp: u32;
        shr.u32(warp, lane, 5);
        mad.lo.u32(row, row, 4, warp);
        and.b32(lane, lane, 31);
        let past: pred;
        setp.ge.u32(past, row, rows);
        if past {
            ret();
        }
        let (energy, dot, v, g, n, inv, den, parallel, tangent, radial): (
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
        mov.f32(energy, 0.0f32);
        mov.f32(dot, 0.0f32);
        mov.u32(c, lane);
        label(reduce);
        let done: pred;
        setp.ge.u32(done, c, width);
        if done {
            bra(reduced);
        }
        mad.lo.u32(at, row, width, c);
        load64(v, x, at);
        fma.rn.f32(energy, v, v, energy);
        if BACK {
            load(g, dy, at);
            fma.rn.f32(dot, v, g, dot);
        }
        add.u32(c, c, 32);
        bra(reduce);
        label(reduced);
        sum32(energy);
        sum32(dot);
        cvt.rn.f32.u32(n, width);
        div.rn.f32(energy, energy, n);
        div.rn.f32(dot, dot, n);
        let positive: pred;
        setp.gt.f32(positive, energy, 0.0f32);
        selp.f32(den, energy, 1.0f32, positive);
        add.rn.f32(inv, energy, 0.00000011920928955078125f32);
        sqrt.rn.f32(inv, inv);
        div.rn.f32(inv, 1.0f32, inv);
        mul.rn.f32(radial, inv, inv);
        mul.rn.f32(radial, radial, inv);
        mul.rn.f32(radial, radial, 0.00000011920928955078125f32);
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
            mul.rn.f32(parallel, v, dot);
            div.rn.f32(parallel, parallel, den);
            sub.rn.f32(tangent, g, parallel);
            mul.rn.f32(tangent, tangent, inv);
            mul.rn.f32(parallel, parallel, radial);
            add.rn.f32(tangent, tangent, parallel);
            store64(dx, at, tangent);
        } else {
            mul.rn.f32(v, v, inv);
            store(y, at, v);
        }
        add.u32(c, c, 32);
        bra(write);
        label(finished);
    }
}
