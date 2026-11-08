//! Shared learned frequencies. FP32 only; no host parameter readback or atomics.
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
    // Range-reduced ninth-order Taylor polynomial on [-ln(2)/2,ln(2)/2].
    // Retain the rounding residual of exp(alpha) so expm1 stays sensitive to
    // tiny alpha and a nearly cancelled base rotation keeps its small phase.
    fn exponential(e: f32, elo: f32, alpha: f32) {
        let (n, r, p, scale, minus_n, tail): (f32, f32, f32, f32, f32, f32);
        mul.rn.f32(n, alpha, 1.4426950408889634);
        cvt.rni.f32.f32(n, n);
        neg.f32(minus_n, n);
        fma.rn.f32(r, minus_n, 0.693145751953125, alpha);
        fma.rn.f32(r, minus_n, 0.0000014286068203094172, r);
        mov.f32(p, 1.0 / 362880.0);
        fma.rn.f32(p, p, r, 1.0 / 40320.0);
        fma.rn.f32(p, p, r, 1.0 / 5040.0);
        fma.rn.f32(p, p, r, 1.0 / 720.0);
        fma.rn.f32(p, p, r, 1.0 / 120.0);
        fma.rn.f32(p, p, r, 1.0 / 24.0);
        fma.rn.f32(p, p, r, 1.0 / 6.0);
        fma.rn.f32(p, p, r, 0.5);
        mul.rn.f32(p, p, r);
        fma.rn.f32(p, p, r, r);
        ex2.approx.f32(scale, n);
        fma.rn.f32(e, scale, p, scale);
        sub.rn.f32(tail, scale, e);
        fma.rn.f32(elo, scale, p, tail);
    }
    pub unsafe extern "ptx-kernel" fn table(
        out: *mut f32,
        theta: *mut f32,
        base: *const f32,
        base_rope: *const f32,
        alpha: *const f32,
        bands: u32,
        count: u32,
    ) {
        let (i, b, t, n, band, at): (u32, u32, u32, u32, u32, u32);
        mov.u32(b, ctaid.x);
        mov.u32(t, tid.x);
        mov.u32(n, ntid.x);
        mad.lo.u32(i, b, n, t);
        let past: pred;
        setp.ge.u32(past, i, count);
        if past {
            ret();
        }
        rem.u32(band, i, bands);
        let (a, angle, e, elo, em1, delta, turns, negative_turns): (
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
        );
        let (back, high, low, delta_low, negative_delta): (f32, f32, f32, f32, f32);
        let (s, c, s0, c0, value, tmp): (f32, f32, f32, f32, f32, f32);
        load(a, alpha, band);
        load(angle, base, i);
        exponential(e, elo, a);
        mul.rn.f32(value, angle, e);
        fma.rn.f32(value, angle, elo, value);
        store(theta, i, value);
        // Error-free TwoSum of e and -1, followed by the exponential tail.
        // The correction is always FP32 arithmetic, with no dtype fallback.
        sub.rn.f32(em1, e, 1.0);
        sub.rn.f32(back, em1, e);
        sub.rn.f32(high, em1, back);
        sub.rn.f32(low, e, high);
        sub.rn.f32(back, -1.0, back);
        add.rn.f32(low, low, back);
        add.rn.f32(low, low, elo);
        mul.rn.f32(delta, angle, em1);
        neg.f32(negative_delta, delta);
        fma.rn.f32(delta_low, angle, em1, negative_delta);
        fma.rn.f32(delta_low, angle, low, delta_low);
        mul.rn.f32(turns, delta, 0.15915494309189535);
        cvt.rni.f32.f32(turns, turns);
        neg.f32(negative_turns, turns);
        fma.rn.f32(delta, negative_turns, 6.28125, delta);
        fma.rn
            .f32(delta, negative_turns, 0.0019353071795864769, delta);
        add.rn.f32(delta, delta, delta_low);
        sin.approx.f32(s, delta);
        cos.approx.f32(c, delta);
        load(c0, base_rope, i);
        add.u32(at, i, count);
        load(s0, base_rope, at);
        mul.rn.f32(value, c0, c);
        mul.rn.f32(tmp, s0, s);
        sub.rn.f32(value, value, tmp);
        store(out, i, value);
        mul.rn.f32(value, s0, c);
        mul.rn.f32(tmp, c0, s);
        add.rn.f32(value, value, tmp);
        store(out, at, value);
    }
    // One warp per shared band. Every Q/K head is visited in a fixed order;
    // host layer ordering serializes accumulation into the same alpha vector.
    pub unsafe extern "ptx-kernel" fn frequency_backward(
        gradient: *mut f32,
        rotated: *const f32,
        dy: *const f32,
        theta: *const f32,
        bands: u32,
        heads: u32,
        length: u32,
        rows: u32,
        reset: u32,
    ) {
        let (band, lane, term, total, twice_heads, row, within, part, head): (
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
        let (width, d, stride, at, imaginary, pos, ti): (u32, u32, u32, u32, u32, u32, u32);
        mov.u32(band, ctaid.x);
        mov.u32(lane, tid.x);
        mov.u32(term, lane);
        mul.lo.u32(d, bands, 2);
        mul.lo.u32(width, d, heads);
        mul.lo.u32(stride, width, 3);
        mul.lo.u32(twice_heads, heads, 2);
        mul.lo.u32(total, rows, twice_heads);
        let (sum, qr, qi, dr, di, angle, value, tmp): (f32, f32, f32, f32, f32, f32, f32, f32);
        mov.f32(sum, 0.0);
        let done: pred;
        label(next);
        setp.ge.u32(done, term, total);
        if done {
            bra(finish);
        }
        div.u32(row, term, twice_heads);
        rem.u32(within, term, twice_heads);
        div.u32(part, within, heads);
        rem.u32(head, within, heads);
        mad.lo.u32(at, row, stride, band);
        mad.lo.u32(at, part, width, at);
        mad.lo.u32(at, head, d, at);
        add.u32(imaginary, at, bands);
        load(qr, rotated, at);
        load(qi, rotated, imaginary);
        load(dr, dy, at);
        load(di, dy, imaginary);
        rem.u32(pos, row, length);
        mad.lo.u32(ti, pos, bands, band);
        load(angle, theta, ti);
        mul.rn.f32(value, qr, di);
        mul.rn.f32(tmp, qi, dr);
        sub.rn.f32(value, value, tmp);
        fma.rn.f32(sum, value, angle, sum);
        add.u32(term, term, 32);
        bra(next);
        label(finish);
        for j in 0..5 {
            shfl.sync.bfly.b32(value, sum, 1 << j, 31, 0xffffffff);
            add.rn.f32(sum, sum, value);
        }
        setp.eq.u32(done, lane, 0);
        mov.f32(tmp, 0.0);
        let accumulate: pred;
        setp.eq.u32(accumulate, reset, 0);
        and.pred(accumulate, accumulate, done);
        if accumulate {
            load(tmp, gradient, band);
        }
        if done {
            add.rn.f32(sum, tmp, sum);
            store(gradient, band, sum);
        }
    }
}

/// Match the selected model's host-FP32 table construction, optionally warped.
pub fn base_tables(length: usize, head_dim: usize, log_positions: bool) -> (Vec<f32>, Vec<f32>) {
    let bands = head_dim / 2;
    let mut angles = vec![0.; length * bands];
    let mut rope = vec![0.; 2 * angles.len()];
    for pos in 0..length {
        for band in 0..bands {
            let mut angle = pos as f32 / 10000f32.powf(2. * band as f32 / head_dim as f32);
            if log_positions {
                angle *= ((pos + 1) as f32).ln();
            }
            let i = pos * bands + band;
            angles[i] = angle;
            rope[i] = angle.cos();
            rope[angles.len() + i] = angle.sin();
        }
    }
    (angles, rope)
}
