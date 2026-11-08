//! Complex gate sigmoid(u)*(1+i*v)/sqrt(1+v*v), after attention, before W_O/RMS.
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
    fn offsets(re: u32, im: u32, i: u32, half: u32) {
        let (group, col): (u32, u32);
        div.u32(group, i, half);
        rem.u32(col, i, half);
        mul.lo.u32(re, group, half);
        add.u32(re, re, re);
        add.u32(re, re, col);
        add.u32(im, re, half);
    }
    fn phase(c: f32, s: f32, v: f32) {
        let (scale, small, large, denominator, temp): (f32, f32, f32, f32, f32);
        abs.f32(scale, v);
        max.f32(scale, scale, 1.0);
        div.rn.f32(small, 1.0, scale);
        div.rn.f32(large, v, scale);
        mul.rn.f32(denominator, small, small);
        mul.rn.f32(temp, large, large);
        add.rn.f32(denominator, denominator, temp);
        sqrt.rn.f32(denominator, denominator);
        div.rn.f32(c, small, denominator);
        div.rn.f32(s, large, denominator);
    }
    fn sigmoid_parts(h: f32, inverse: f32, u: f32) {
        abs.f32(h, u);
        mul.rn.f32(h, h, -0.7213475204444817);
        ex2.approx.f32(h, h);
        mul.rn.f32(inverse, h, h);
        add.rn.f32(inverse, inverse, 1.0);
        div.rn.f32(inverse, 1.0, inverse);
    }
    fn sigmoid_mul(out: f32, x: f32, h: f32, inverse: f32, u: f32) {
        let negative: f32;
        mul.rn.f32(negative, x, h);
        mul.rn.f32(negative, negative, h);
        mul.rn.f32(negative, negative, inverse);
        mul.rn.f32(out, x, inverse);
        let nonnegative: pred;
        setp.ge.f32(nonnegative, u, 0.0);
        selp.f32(out, out, negative, nonnegative);
    }
    pub unsafe extern "ptx-kernel" fn forward(
        out: *mut f32,
        gates: *mut f32,
        slopes: *mut f32,
        tails: *mut f32,
        attention: *const f32,
        logits: *const f32,
        half: u32,
        count: u32,
    ) {
        let (i, re, im): (u32, u32, u32);
        index(i);
        let past: pred;
        setp.ge.u32(past, i, count);
        if past {
            ret();
        }
        offsets(re, im, i, half);
        let (u, v, ar, ai, c, s, h, inverse, yr, yi, temp, value): (
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
        load(u, logits, re);
        load(v, logits, im);
        load(ar, attention, re);
        load(ai, attention, im);
        phase(c, s, v);
        sigmoid_parts(h, inverse, u);
        store(tails, re, h);
        store(tails, im, h);
        sigmoid_mul(value, c, h, inverse, u);
        store(gates, re, value);
        sigmoid_mul(value, s, h, inverse, u);
        store(gates, im, value);
        mul.rn.f32(value, inverse, h);
        mul.rn.f32(value, value, value);
        store(slopes, re, value);
        mul.rn.f32(value, c, c);
        sigmoid_mul(value, value, h, inverse, u);
        store(slopes, im, value);
        sigmoid_mul(ar, ar, h, inverse, u);
        sigmoid_mul(ai, ai, h, inverse, u);
        mul.rn.f32(yr, c, ar);
        mul.rn.f32(temp, s, ai);
        sub.rn.f32(yr, yr, temp);
        mul.rn.f32(yi, s, ar);
        mul.rn.f32(temp, c, ai);
        add.rn.f32(yi, yi, temp);
        store(out, re, yr);
        store(out, im, yi);
    }
    pub unsafe extern "ptx-kernel" fn backward(
        dattention: *mut f32,
        dlogits: *mut f32,
        dy: *const f32,
        attention: *const f32,
        tails: *const f32,
        logits: *const f32,
        half: u32,
        count: u32,
    ) {
        let (i, re, im): (u32, u32, u32);
        index(i);
        let past: pred;
        setp.ge.u32(past, i, count);
        if past {
            ret();
        }
        offsets(re, im, i, half);
        let (
            u,
            v,
            ar,
            ai,
            gr,
            gi,
            c,
            s,
            h,
            inverse,
            qr,
            qi,
            dr,
            di,
            temp,
            left,
            right,
            du,
            dv,
            root,
        ): (
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
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
        );
        // Read both coordinates before writing either in-place input adjoint.
        load(u, logits, re);
        load(v, logits, im);
        load(ar, attention, re);
        load(ai, attention, im);
        load(gr, dy, re);
        load(gi, dy, im);
        load(h, tails, re);
        phase(c, s, v);
        mul.rn.f32(inverse, h, h);
        add.rn.f32(inverse, inverse, 1.0);
        div.rn.f32(inverse, 1.0, inverse);
        sigmoid_mul(dr, gr, h, inverse, u);
        sigmoid_mul(di, gi, h, inverse, u);
        mul.rn.f32(left, c, dr);
        mul.rn.f32(temp, s, di);
        add.rn.f32(left, left, temp);
        mul.rn.f32(right, c, di);
        mul.rn.f32(temp, s, dr);
        sub.rn.f32(right, right, temp);
        store(dattention, re, left);
        store(dattention, im, right);
        mul.rn.f32(qr, c, ar);
        mul.rn.f32(temp, s, ai);
        sub.rn.f32(qr, qr, temp);
        mul.rn.f32(qi, s, ar);
        mul.rn.f32(temp, c, ai);
        add.rn.f32(qi, qi, temp);
        // Split sigmoid'(u) between operands, preserving saturated tails.
        mul.rn.f32(left, qr, h);
        mul.rn.f32(left, left, inverse);
        mul.rn.f32(right, gr, h);
        mul.rn.f32(right, right, inverse);
        mul.rn.f32(du, left, right);
        mul.rn.f32(left, qi, h);
        mul.rn.f32(left, left, inverse);
        mul.rn.f32(right, gi, h);
        mul.rn.f32(right, right, inverse);
        mul.rn.f32(temp, left, right);
        add.rn.f32(du, du, temp);
        // d phase/dv=c^2: compensate operands before forming the tiny square.
        sqrt.rn.f32(root, inverse);
        mul.rn.f32(temp, root, h);
        let nonnegative: pred;
        setp.ge.f32(nonnegative, u, 0.0);
        selp.f32(root, root, temp, nonnegative);
        mul.rn.f32(left, qr, root);
        mul.rn.f32(left, left, c);
        mul.rn.f32(right, gi, root);
        mul.rn.f32(right, right, c);
        mul.rn.f32(dv, left, right);
        mul.rn.f32(left, qi, root);
        mul.rn.f32(left, left, c);
        mul.rn.f32(right, gr, root);
        mul.rn.f32(right, right, c);
        mul.rn.f32(temp, left, right);
        sub.rn.f32(dv, dv, temp);
        store(dlogits, re, du);
        store(dlogits, im, dv);
    }
}
