//! Elementwise sigmoid after ordinary attention, before output projection/RMS.
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
    fn index(i: u32) {
        let (b, t, n): (u32, u32, u32);
        mov.u32(b, ctaid.x);
        mov.u32(t, tid.x);
        mov.u32(n, ntid.x);
        mad.lo.u32(i, b, n, t);
    }
    pub unsafe extern "ptx-kernel" fn forward(
        out: *mut f32,
        gates: *mut f32,
        slopes: *mut f32,
        tails: *mut f32,
        attention: *const f32,
        logits: *const f32,
        count: u32,
    ) {
        let i: u32;
        index(i);
        let past: pred;
        setp.ge.u32(past, i, count);
        if past {
            ret();
        }
        let (z, a, e, denominator, positive, negative, slope, value): (
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
        );
        let half: f32;
        load(z, logits, i);
        load(a, attention, i);
        abs.f32(half, z);
        mul.rn.f32(half, half, -0.7213475204444817);
        ex2.approx.f32(half, half);
        store(tails, i, half);
        mul.rn.f32(e, half, half);
        add.rn.f32(denominator, e, 1.0);
        div.rn.f32(positive, 1.0, denominator);
        div.rn.f32(negative, e, denominator);
        let nonnegative: pred;
        setp.ge.f32(nonnegative, z, 0.0);
        selp.f32(value, positive, negative, nonnegative);
        store(gates, i, value);
        // Preserve small derivatives even when the stored positive gate rounds to1.
        mul.rn.f32(slope, positive, half);
        mul.rn.f32(slope, slope, slope);
        store(slopes, i, slope);
        // Scale before rounding a possibly subnormal probability to one word.
        mul.rn.f32(negative, a, half);
        mul.rn.f32(negative, negative, half);
        mul.rn.f32(negative, negative, positive);
        mul.rn.f32(value, a, positive);
        selp.f32(value, value, negative, nonnegative);
        store(out, i, value);
    }
    pub unsafe extern "ptx-kernel" fn backward(
        dattention: *mut f32,
        dlogits: *mut f32,
        dy: *const f32,
        attention: *const f32,
        tails: *const f32,
        logits: *const f32,
        count: u32,
    ) {
        let i: u32;
        index(i);
        let past: pred;
        setp.ge.u32(past, i, count);
        if past {
            ret();
        }
        let (g, a, half, inverse, out, other, z): (f32, f32, f32, f32, f32, f32, f32);
        load(g, dy, i);
        load(a, attention, i);
        load(half, tails, i);
        load(z, logits, i);
        mul.rn.f32(inverse, half, half);
        add.rn.f32(inverse, inverse, 1.0);
        div.rn.f32(inverse, 1.0, inverse);
        mul.rn.f32(other, g, half);
        mul.rn.f32(other, other, half);
        mul.rn.f32(other, other, inverse);
        mul.rn.f32(out, g, inverse);
        let nonnegative: pred;
        setp.ge.f32(nonnegative, z, 0.0);
        selp.f32(out, out, other, nonnegative);
        store(dattention, i, out);
        mul.rn.f32(out, a, half);
        mul.rn.f32(out, out, inverse);
        mul.rn.f32(other, g, half);
        mul.rn.f32(other, other, inverse);
        mul.rn.f32(out, out, other);
        store(dlogits, i, out);
    }
}
