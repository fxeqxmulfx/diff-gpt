//! Two-key causal Q/K RMS, fixed RoPE, optional SSMax, and centered clipped XSA.
//! One warp handles a 32-dimensional head. No attention atomics are needed.

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
    fn sum(x: f32) {
        let z: f32;
        for j in 0..5 {
            shfl.sync.bfly.b32(z, x, 1 << j, 31, 0xffffffff);
            add.rn.f32(x, x, z);
        }
    }
    fn sum64(x: f64) {
        let (lo, hi, ol, oh): (u32, u32, u32, u32);
        let z: f64;
        for j in 0..5 {
            mov.b64((lo, hi), x);
            shfl.sync.bfly.b32(ol, lo, 1 << j, 31, 0xffffffff);
            shfl.sync.bfly.b32(oh, hi, 1 << j, 31, 0xffffffff);
            mov.b64(z, (ol, oh));
            add.rn.f64(x, x, z);
        }
    }
    fn exp(x: f32) {
        mul.rn.f32(x, x, 1.4426950408889634);
        ex2.approx.f32(x, x);
    }
    fn position(b: u32, h: u32, lane: u32, heads: u32) {
        let id: u32;
        mov.u32(id, ctaid.x);
        div.u32(b, id, heads);
        rem.u32(h, id, heads);
        mov.u32(lane, tid.x);
    }

    fn values(
        qkv: u64,
        rope: u64,
        width: u32,
        b: u32,
        h: u32,
        lane: u32,
        qscale: f32,
        gain: f32,
        raw: [f32; 3],
        rot: [f32; 3],
        v0: f32,
        v1: f32,
        delta: f32,
        unit: f32,
        norm: f32,
        den: f32,
        p0: f32,
        p1: f32,
        projection: f32,
        remainder: f32,
    ) {
        let (base, col, at, partner, pair): (u32, u32, u32, u32, u32);
        mul.lo.u32(base, b, 6);
        mul.lo.u32(base, base, width);
        mad.lo.u32(col, h, 32, lane);
        add.u32(base, base, col);
        and.b32(pair, lane, 15);
        xor.b32(partner, lane, 16);
        let (ss, inv, other, cos, sin, a, c): (f32, f32, f32, f32, f32, f32, f32);
        let lower: pred;
        setp.lt.u32(lower, lane, 16);
        for j in 0..3 {
            let offset = [3, 1, 4][j];
            mad.lo.u32(at, width, offset, base);
            load(raw[j], qkv, at);
            mul.rn.f32(ss, raw[j], raw[j]);
            sum(ss);
            mul.rn.f32(ss, ss, 0.03125);
            add.rn.f32(ss, ss, 0.00000011920928955078125);
            rsqrt.approx.f32(inv, ss);
            mul.rn.f32(rot[j], raw[j], inv);
            shfl.sync.idx.b32(other, rot[j], partner, 31, 0xffffffff);
            let pos = [16, 0, 16][j];
            add.u32(at, pair, pos);
            load(cos, rope, at);
            add.u32(at, at, 32);
            load(sin, rope, at);
            mul.rn.f32(a, rot[j], cos);
            mul.rn.f32(c, other, sin);
            if lower {
                neg.f32(c, c);
            }
            add.rn.f32(rot[j], a, c);
        }
        mul.rn.f32(rot[0], rot[0], qscale);
        mul.rn.f32(p0, rot[0], rot[1]);
        mul.rn.f32(p1, rot[0], rot[2]);
        sum(p0);
        sum(p1);
        mul.rn.f32(p0, p0, gain);
        mul.rn.f32(p1, p1, gain);
        max.f32(ss, p0, p1);
        sub.rn.f32(p0, p0, ss);
        sub.rn.f32(p1, p1, ss);
        exp(p0);
        exp(p1);
        add.rn.f32(ss, p0, p1);
        div.rn.f32(p0, p0, ss);
        div.rn.f32(p1, p1, ss);
        mad.lo.u32(at, width, 2, base);
        load(v0, qkv, at);
        mad.lo.u32(at, width, 5, base);
        load(v1, qkv, at);
        sub.rn.f32(delta, v1, v0);
        mul.rn.f32(a, delta, p1);
        sub.rn.f32(delta, a, delta);
        mul.rn.f32(norm, v1, v1);
        sum(norm);
        sqrt.rn.f32(norm, norm);
        max.f32(den, norm, 1e-12);
        div.rn.f32(unit, v1, den);
        mul.rn.f32(projection, delta, unit);
        sum(projection);
        let above: pred;
        setp.gt.f32(above, norm, 1e-12);
        mov.f32(remainder, 0.0);
        if !above {
            div.rn.f32(remainder, norm, den);
            mul.rn.f32(remainder, remainder, remainder);
            sub.rn.f32(remainder, 1.0, remainder);
        }
    }

    pub unsafe extern "ptx-kernel" fn forward(
        out: *mut f32,
        head_ok: *mut u32,
        qkv: *const f32,
        rope: *const f32,
        width: u32,
        heads: u32,
        qscale: f32,
        gain: f32,
    ) {
        let (b, h, lane): (u32, u32, u32);
        position(b, h, lane, heads);
        let (raw, rot): ([f32; 3], [f32; 3]);
        let (v0, v1, delta, unit, norm, den, p0, p1, proj, rem): (
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
        values(
            qkv, rope, width, b, h, lane, qscale, gain, raw, rot, v0, v1, delta, unit, norm, den,
            p0, p1, proj, rem,
        );
        let (y, z): (f32, f32);
        mul.rn.f32(y, proj, unit);
        sub.rn.f32(y, delta, y);
        mul.rn.f32(z, rem, v1);
        add.rn.f32(y, y, z);
        let (at, row, col): (u32, u32, u32);
        mad.lo.u32(col, h, 32, lane);
        mul.lo.u32(row, b, 2);
        mad.lo.u32(at, row, width, col);
        mov.f32(z, 0.0);
        store(out, at, z);
        add.u32(at, at, width);
        store(out, at, y);
        mul.rn.f32(z, v0, v0);
        sum(z);
        sqrt.rn.f32(z, z);
        let ok: pred;
        setp.eq.u32(ok, lane, 0);
        if ok {
            let valid: pred;
            setp.gt.f32(valid, z, 1e-12);
            selp.u32(col, 1, 0, valid);
            mad.lo.u32(at, b, heads, h);
            let p: u64;
            mad.wide.u32(p, at, 4, head_ok);
            st.global.u32([p], col);
        }
    }

    fn rms_back(out: f32, x: f32, g: f32) {
        let (v, u, e, dot, den, inv, radial, parallel, tangent): (
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
        cvt.f64.f32(v, x);
        cvt.f64.f32(u, g);
        mul.rn.f64(e, v, v);
        mul.rn.f64(dot, v, u);
        sum64(e);
        sum64(dot);
        mul.rn.f64(e, e, 0.03125f64);
        mul.rn.f64(dot, dot, 0.03125f64);
        let positive: pred;
        setp.gt.f64(positive, e, 0.0f64);
        selp.f64(den, e, 1.0f64, positive);
        mul.rn.f64(parallel, v, dot);
        div.rn.f64(parallel, parallel, den);
        add.rn.f64(inv, e, 0.00000011920928955078125f64);
        sqrt.rn.f64(inv, inv);
        div.rn.f64(inv, 1.0f64, inv);
        sub.rn.f64(tangent, u, parallel);
        mul.rn.f64(tangent, tangent, inv);
        mul.rn.f64(radial, inv, inv);
        mul.rn.f64(radial, radial, inv);
        mul.rn.f64(radial, radial, 0.00000011920928955078125f64);
        mul.rn.f64(radial, radial, parallel);
        add.rn.f64(tangent, tangent, radial);
        cvt.rn.f32.f64(out, tangent);
    }

    pub unsafe extern "ptx-kernel" fn backward(
        dqkv: *mut f32,
        dy: *const f32,
        qkv: *const f32,
        rope: *const f32,
        width: u32,
        heads: u32,
        qscale: f32,
        gain: f32,
    ) {
        let (b, h, lane): (u32, u32, u32);
        position(b, h, lane, heads);
        let (raw, rot): ([f32; 3], [f32; 3]);
        let (v0, v1, delta, unit, norm, den, p0, p1, proj, rem): (
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
        values(
            qkv, rope, width, b, h, lane, qscale, gain, raw, rot, v0, v1, delta, unit, norm, den,
            p0, p1, proj, rem,
        );
        let (at, col, row, base): (u32, u32, u32, u32);
        mad.lo.u32(col, h, 32, lane);
        mad.lo.u32(row, b, 2, 1);
        mad.lo.u32(at, row, width, col);
        let (g, gu, gd, ug, vg, dv1, dden, dnorm, ratio, temp, other): (
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
        load(g, dy, at);
        mul.rn.f32(ug, g, unit);
        sum(ug);
        mul.rn.f32(vg, g, v1);
        sum(vg);
        mul.rn.f32(gd, ug, unit);
        sub.rn.f32(gd, g, gd);
        mul.rn.f32(gu, proj, g);
        mul.rn.f32(temp, ug, delta);
        add.rn.f32(gu, gu, temp);
        neg.f32(gu, gu);
        div.rn.f32(dv1, gu, den);
        mul.rn.f32(temp, rem, g);
        add.rn.f32(dv1, dv1, temp);
        mul.rn.f32(dden, gu, unit);
        sum(dden);
        neg.f32(dden, dden);
        div.rn.f32(dden, dden, den);
        mov.f32(dnorm, 0.0);
        let above: pred;
        setp.gt.f32(above, norm, 1e-12);
        if !above {
            div.rn.f32(ratio, norm, den);
            mul.rn.f32(temp, ratio, -2.0);
            mul.rn.f32(temp, temp, vg);
            div.rn.f32(dnorm, temp, den);
            mul.rn.f32(temp, temp, ratio);
            div.rn.f32(temp, temp, den);
            sub.rn.f32(dden, dden, temp);
        }
        setp.ge.f32(above, norm, 1e-12);
        if above {
            add.rn.f32(dnorm, dnorm, dden);
        }
        setp.gt.f32(above, norm, 0.0);
        if above {
            div.rn.f32(temp, v1, norm);
            mul.rn.f32(temp, temp, dnorm);
            add.rn.f32(dv1, dv1, temp);
        }
        let (gp, ds0, ds1, dv0): (f32, f32, f32, f32);
        sub.rn.f32(temp, v1, v0);
        mul.rn.f32(gp, gd, temp);
        sum(gp);
        mul.rn.f32(temp, gp, p1);
        neg.f32(ds0, temp);
        mul.rn.f32(ds0, ds0, p0);
        sub.rn.f32(ds1, gp, temp);
        mul.rn.f32(ds1, ds1, p1);
        mul.rn.f32(ds0, ds0, gain);
        mul.rn.f32(ds1, ds1, gain);
        mul.rn.f32(temp, p1, gd);
        sub.rn.f32(temp, temp, gd);
        neg.f32(dv0, temp);
        add.rn.f32(dv1, dv1, temp);
        mul.lo.u32(base, b, 6);
        mad.lo.u32(base, base, width, col);
        mad.lo.u32(at, width, 2, base);
        store(dqkv, at, dv0);
        mad.lo.u32(at, width, 5, base);
        store(dqkv, at, dv1);
        mov.f32(temp, 0.0);
        store(dqkv, base, temp);
        let grad: [f32; 3];
        mul.rn.f32(grad[0], rot[1], ds0);
        mul.rn.f32(temp, rot[2], ds1);
        add.rn.f32(grad[0], grad[0], temp);
        mul.rn.f32(grad[0], grad[0], qscale);
        mul.rn.f32(grad[1], rot[0], ds0);
        mul.rn.f32(grad[2], rot[0], ds1);
        let (pair, partner): (u32, u32);
        let (cos, sin, result): (f32, f32, f32);
        and.b32(pair, lane, 15);
        xor.b32(partner, lane, 16);
        setp.ge.u32(above, lane, 16);
        for j in 0..3 {
            let pos = [16, 0, 16][j];
            shfl.sync.idx.b32(other, grad[j], partner, 31, 0xffffffff);
            add.u32(at, pair, pos);
            load(cos, rope, at);
            add.u32(at, at, 32);
            load(sin, rope, at);
            mul.rn.f32(temp, other, sin);
            if above {
                neg.f32(temp, temp);
            }
            mul.rn.f32(grad[j], grad[j], cos);
            add.rn.f32(grad[j], grad[j], temp);
            rms_back(result, raw[j], grad[j]);
            let offset = [3, 1, 4][j];
            mad.lo.u32(at, width, offset, base);
            store(dqkv, at, result);
        }
    }
}
