//! Token-local uncentered 2x2 moment normalization and its complete real adjoint.
//! Scaling prevents input squares from overflowing; every GPU operation is FP32.
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
    fn sum(v: f32) {
        let z: f32;
        for j in 0..5 {
            shfl.sync.bfly.b32(z, v, 1 << j, 31, 0xffffffff);
            add.rn.f32(v, v, z);
        }
    }
    fn maximum(v: f32) {
        let z: f32;
        for j in 0..5 {
            shfl.sync.bfly.b32(z, v, 1 << j, 31, 0xffffffff);
            max.f32(v, v, z);
        }
    }
    // Each warp handles one complete vector; inactive pair lanes still reduce.
    // heads describes the head-local split inside this vector. value_heads is
    // zero for ordinary rows, or the QKV head count for unchanged V passthrough.
    #[instances((16, false), (32, false), (64, false), (128, false), (256, false),
                (16, true), (32, true), (64, true), (128, true), (256, true))]
    pub unsafe extern "ptx-kernel" fn normalize<const W: usize, const BACK: bool>(
        out: *mut f32,
        x: *const f32,
        dy: *const f32,
        rows: u32,
        heads: u32,
        value_heads: u32,
    ) {
        let (row, lane, warp, block, hd, half, k, group, col, at, ai, part): (
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
        mov.u32(block, ctaid.x);
        mov.u32(lane, tid.x);
        shr.u32(warp, lane, 5);
        mad.lo.u32(row, block, 4, warp);
        and.b32(lane, lane, 31);
        let past: pred;
        setp.ge.u32(past, row, rows);
        if past {
            ret();
        }
        div.u32(hd, W, heads);
        shr.u32(half, hd, 1);
        max.u32(part, value_heads, 1);
        div.u32(part, row, part);
        rem.u32(part, part, 3);
        let (is_qkv, is_value, passthrough): (pred, pred, pred);
        setp.ne.u32(is_qkv, value_heads, 0);
        setp.eq.u32(is_value, part, 2);
        and.pred(passthrough, is_qkv, is_value);
        let re: [f32; (W / 2 + 31) / 32];
        let im: [f32; (W / 2 + 31) / 32];
        let gr: [f32; (W / 2 + 31) / 32];
        let gi: [f32; (W / 2 + 31) / 32];
        let (m, z): (f32, f32);
        mov.f32(m, 0.00034526698300124393);
        for j in 0..(W / 2 + 31) / 32 {
            add.u32(k, lane, 32 * j);
            mov.f32(re[j], 0.0);
            mov.f32(im[j], 0.0);
            mov.f32(gr[j], 0.0);
            mov.f32(gi[j], 0.0);
            mov.u32(at, 0);
            mov.u32(ai, 0);
            let active: pred;
            setp.lt.u32(active, k, W / 2);
            if active {
                div.u32(group, k, half);
                rem.u32(col, k, half);
                mad.lo.u32(col, group, hd, col);
                mad.lo.u32(at, row, W, col);
                add.u32(ai, at, half);
                load(re[j], x, at);
                load(im[j], x, ai);
                if BACK {
                    load(gr[j], dy, at);
                    load(gi[j], dy, ai);
                }
            }
            let copy_value: pred;
            and.pred(copy_value, active, passthrough);
            if copy_value {
                if BACK {
                    store(out, at, gr[j]);
                    store(out, ai, gi[j]);
                } else {
                    store(out, at, re[j]);
                    store(out, ai, im[j]);
                }
            }
            abs.f32(z, re[j]);
            max.f32(m, m, z);
            abs.f32(z, im[j]);
            max.f32(m, m, z);
        }
        if passthrough {
            ret();
        }
        maximum(m);
        let (aa, bb, cc, a, b, c, e, s, t, r): (f32, f32, f32, f32, f32, f32, f32, f32, f32, f32);
        mov.f32(aa, 0.0);
        mov.f32(bb, 0.0);
        mov.f32(cc, 0.0);
        for j in 0..(W / 2 + 31) / 32 {
            div.rn.f32(re[j], re[j], m);
            div.rn.f32(im[j], im[j], m);
            mul.rn.f32(z, re[j], re[j]);
            add.rn.f32(aa, aa, z);
            mul.rn.f32(z, re[j], im[j]);
            add.rn.f32(bb, bb, z);
            mul.rn.f32(z, im[j], im[j]);
            add.rn.f32(cc, cc, z);
        }
        sum(aa);
        sum(bb);
        sum(cc);
        mul.rn.f32(aa, aa, 2.0 / (W as f32));
        mul.rn.f32(bb, bb, 2.0 / (W as f32));
        mul.rn.f32(cc, cc, 2.0 / (W as f32));
        // epsilon/m^2. Its underflow at huge m is harmless for nonzero rows:
        // shrinkage itself gives a strictly positive lower eigenvalue there.
        div.rn.f32(e, 0.00034526698300124393, m);
        mul.rn.f32(e, e, e);
        mul.rn.f32(a, aa, 0.96875);
        mul.rn.f32(z, cc, 0.03125);
        add.rn.f32(a, a, z);
        add.rn.f32(a, a, e);
        mul.rn.f32(c, cc, 0.96875);
        mul.rn.f32(z, aa, 0.03125);
        add.rn.f32(c, c, z);
        add.rn.f32(c, c, e);
        mul.rn.f32(b, bb, 0.9375);
        mul.rn.f32(s, a, c);
        mul.rn.f32(z, b, b);
        sub.rn.f32(s, s, z);
        sqrt.rn.f32(s, s);
        add.rn.f32(t, a, c);
        add.rn.f32(z, s, s);
        add.rn.f32(t, t, z);
        sqrt.rn.f32(t, t);
        mul.rn.f32(r, s, t);
        div.rn.f32(r, 1.0, r);
        let (da, db, dc): (f32, f32, f32);
        if BACK {
            let (u, v, cross, f, ls, lt, l): (f32, f32, f32, f32, f32, f32, f32);
            mov.f32(u, 0.0);
            mov.f32(v, 0.0);
            mov.f32(cross, 0.0);
            for j in 0..(W / 2 + 31) / 32 {
                mul.rn.f32(z, gr[j], re[j]);
                add.rn.f32(u, u, z);
                mul.rn.f32(z, gi[j], im[j]);
                add.rn.f32(v, v, z);
                mul.rn.f32(z, gr[j], im[j]);
                add.rn.f32(cross, cross, z);
                mul.rn.f32(z, gi[j], re[j]);
                add.rn.f32(cross, cross, z);
            }
            sum(u);
            sum(v);
            sum(cross);
            add.rn.f32(f, c, s);
            mul.rn.f32(f, f, u);
            add.rn.f32(z, a, s);
            mul.rn.f32(z, z, v);
            add.rn.f32(f, f, z);
            mul.rn.f32(z, b, cross);
            sub.rn.f32(f, f, z);
            mul.rn.f32(l, r, f);
            mul.rn.f32(da, r, v);
            mul.rn.f32(dc, r, u);
            mul.rn.f32(db, r, cross);
            neg.f32(db, db);
            add.rn.f32(ls, u, v);
            mul.rn.f32(ls, ls, r);
            div.rn.f32(z, l, s);
            sub.rn.f32(ls, ls, z);
            div.rn.f32(lt, l, t);
            neg.f32(lt, lt);
            div.rn.f32(z, lt, t);
            add.rn.f32(ls, ls, z);
            mul.rn.f32(z, z, 0.5);
            add.rn.f32(da, da, z);
            add.rn.f32(dc, dc, z);
            div.rn.f32(ls, ls, s);
            mul.rn.f32(z, ls, b);
            sub.rn.f32(db, db, z);
            mul.rn.f32(ls, ls, 0.5);
            mul.rn.f32(z, ls, c);
            add.rn.f32(da, da, z);
            mul.rn.f32(z, ls, a);
            add.rn.f32(dc, dc, z);
            // Pull back trace-preserving shrinkage, then the three moments.
            mul.rn.f32(aa, da, 0.96875);
            mul.rn.f32(z, dc, 0.03125);
            add.rn.f32(aa, aa, z);
            mul.rn.f32(cc, dc, 0.96875);
            mul.rn.f32(z, da, 0.03125);
            add.rn.f32(cc, cc, z);
            mul.rn.f32(bb, db, 0.9375);
            mul.rn.f32(aa, aa, 4.0 / (W as f32));
            mul.rn.f32(cc, cc, 4.0 / (W as f32));
            mul.rn.f32(bb, bb, 2.0 / (W as f32));
        }
        for j in 0..(W / 2 + 31) / 32 {
            add.u32(k, lane, 32 * j);
            let active: pred;
            setp.lt.u32(active, k, W / 2);
            if active {
                div.u32(group, k, half);
                rem.u32(col, k, half);
                mad.lo.u32(col, group, hd, col);
                mad.lo.u32(at, row, W, col);
                add.u32(ai, at, half);
                let (yr, yi, temp): (f32, f32, f32);
                add.rn.f32(yr, c, s);
                add.rn.f32(yi, a, s);
                if BACK {
                    mul.rn.f32(yr, yr, gr[j]);
                    mul.rn.f32(z, b, gi[j]);
                    sub.rn.f32(yr, yr, z);
                    mul.rn.f32(yi, yi, gi[j]);
                    mul.rn.f32(z, b, gr[j]);
                    sub.rn.f32(yi, yi, z);
                    mul.rn.f32(yr, yr, r);
                    mul.rn.f32(yi, yi, r);
                    mul.rn.f32(temp, aa, re[j]);
                    mul.rn.f32(z, bb, im[j]);
                    add.rn.f32(temp, temp, z);
                    add.rn.f32(yr, yr, temp);
                    mul.rn.f32(temp, cc, im[j]);
                    mul.rn.f32(z, bb, re[j]);
                    add.rn.f32(temp, temp, z);
                    add.rn.f32(yi, yi, temp);
                    div.rn.f32(yr, yr, m);
                    div.rn.f32(yi, yi, m);
                } else {
                    mul.rn.f32(yr, yr, re[j]);
                    mul.rn.f32(z, b, im[j]);
                    sub.rn.f32(yr, yr, z);
                    mul.rn.f32(yi, yi, im[j]);
                    mul.rn.f32(z, b, re[j]);
                    sub.rn.f32(yi, yi, z);
                    mul.rn.f32(yr, yr, r);
                    mul.rn.f32(yi, yi, r);
                }
                store(out, at, yr);
                store(out, ai, yi);
            }
        }
    }
}
