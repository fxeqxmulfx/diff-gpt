//! Exact finite FP32 products encoded as FP64 using unsigned wide multiplication.
//! Every invocation uses this fixed representation; predicates implement the
//! input format's zero/Inf/NaN semantics, not precision selection or fallback.
#[kernite_macros::kernels]
pub mod kernels {
    fn wide_product(out: f64, a: f32, b: f32) {
        let (aa, bb, ea, eb, ma, mb, za, zb, sign, leading, k, shift): (
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
        let (expa, expb, e): (s32, s32, s32);
        let (product, bits, exponent, sign64): (u64, u64, u64, u64);
        let (na, nb, zero, ia, ib, nana, nanb, invalid, one, other): (
            pred,
            pred,
            pred,
            pred,
            pred,
            pred,
            pred,
            pred,
            pred,
            pred,
        );
        mov.b32(aa, a);
        mov.b32(bb, b);
        xor.b32(sign, aa, bb);
        and.b32(sign, sign, 0x80000000);
        cvt.u64.u32(sign64, sign);
        shl.b64(sign64, sign64, 32);
        shr.u32(ea, aa, 23);
        and.b32(ea, ea, 255);
        shr.u32(eb, bb, 23);
        and.b32(eb, eb, 255);
        and.b32(ma, aa, 0x7fffff);
        and.b32(mb, bb, 0x7fffff);
        setp.ne.u32(na, ea, 0);
        setp.ne.u32(nb, eb, 0);
        selp.u32(za, 0x800000, 0, na);
        selp.u32(zb, 0x800000, 0, nb);
        or.b32(ma, ma, za);
        or.b32(mb, mb, zb);
        sub.s32(expa, ea, 150);
        sub.s32(expb, eb, 150);
        selp.s32(expa, expa, -149, na);
        selp.s32(expb, expb, -149, nb);
        mul.wide.u32(product, ma, mb);
        clz.b64(leading, product);
        sub.u32(k, 63, leading);
        sub.u32(shift, 52, k);
        shl.b64(bits, product, shift);
        and.b64(bits, bits, 0x000fffffffffffffu64);
        add.s32(e, expa, expb);
        add.s32(e, e, k);
        add.s32(e, e, 1023);
        cvt.u64.u32(exponent, e);
        shl.b64(exponent, exponent, 52);
        or.b64(bits, bits, exponent);
        or.b64(bits, bits, sign64);
        setp.eq.u64(zero, product, 0);
        selp.u64(bits, sign64, bits, zero);
        // Preserve exceptional input classes instead of concealing invalid data.
        setp.eq.u32(ia, ea, 255);
        setp.eq.u32(ib, eb, 255);
        setp.ne.u32(nana, ma, 0x800000);
        and.pred(nana, nana, ia);
        setp.ne.u32(nanb, mb, 0x800000);
        and.pred(nanb, nanb, ib);
        setp.eq.u32(one, mb, 0);
        and.pred(one, one, ia);
        setp.eq.u32(other, ma, 0);
        and.pred(other, other, ib);
        or.pred(invalid, one, other);
        or.pred(invalid, invalid, nana);
        or.pred(invalid, invalid, nanb);
        or.pred(one, ia, ib);
        or.b64(exponent, sign64, 0x7ff0000000000000u64);
        selp.u64(bits, exponent, bits, one);
        selp.u64(bits, 0x7ff8000000000000u64, bits, invalid);
        mov.b64(out, bits);
    }
    #[instances(false, true)]
    pub unsafe extern "ptx-kernel" fn products<const INTEGER: bool>(
        out: *mut f64,
        a: *const f32,
        b: *const f32,
        n: u32,
    ) {
        let (i, block, lane, size): (u32, u32, u32, u32);
        mov.u32(block, ctaid.x);
        mov.u32(lane, tid.x);
        mov.u32(size, ntid.x);
        mad.lo.u32(i, block, size, lane);
        let past: pred;
        setp.ge.u32(past, i, n);
        if past {
            ret();
        }
        let address: u64;
        let (x, y): (f32, f32);
        let (z, xd, yd): (f64, f64, f64);
        mad.wide.u32(address, i, 4, a);
        ld.global.f32(x, [address]);
        mad.wide.u32(address, i, 4, b);
        ld.global.f32(y, [address]);
        if INTEGER {
            wide_product(z, x, y);
        } else {
            cvt.f64.f32(xd, x);
            cvt.f64.f32(yd, y);
            mul.rn.f64(z, xd, yd);
        }
        mad.wide.u32(address, i, 8, out);
        st.global.f64([address], z);
    }
}
