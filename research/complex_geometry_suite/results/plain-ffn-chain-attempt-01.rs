//! FP64 input adjoint of RMS(Wx), avoiding a rounded intermediate adjoint.
//! Used for single 128-wide attention heads with nearly radial Q/K derivatives.
#[kernite_macros::kernels]
pub mod kernels {
    fn load(v: f64, p: u64, i: u32) {
        let at: u64;
        let f: f32;
        mad.wide.u32(at, i, 4, p);
        ld.global.f32(f, [at]);
        cvt.f64.f32(v, f);
    }
    fn sum(v: f64) {
        let (a, b, c, d): (u32, u32, u32, u32);
        let x: f64;
        for j in 0..5 {
            mov.b64((a, b), v);
            shfl.sync.bfly.b32(c, a, 1 << j, 31, 0xffffffff);
            shfl.sync.bfly.b32(d, b, 1 << j, 31, 0xffffffff);
            mov.b64(x, (c, d));
            add.rn.f64(v, v, x);
        }
    }
    pub unsafe extern "ptx-kernel" fn input_backward(
        dx: *mut f32,
        x: *const f32,
        weight: *const f32,
        dy: *const f32,
    ) {
        let row_grad: shared<[f64; 128], 16>;
        let (row, lane, base, col, k, at, shared_base, shared_at): (
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
        );
        mov.u32(row, ctaid.x);
        mov.u32(lane, tid.x);
        mul.lo.u32(base, row, 128);
        mov.u32(shared_base, row_grad);
        let (z, g): ([f64; 4], [f64; 4]);
        let (a, b, energy, dot, t, inv, den, parallel, tangent, radial, result): (
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
            f64,
        );
        mov.f64(energy, 0.0f64);
        mov.f64(dot, 0.0f64);
        for j in 0..4 {
            add.u32(col, lane, 32 * j);
            mov.f64(z[j], 0.0f64);
            mov.u32(k, 0);
            label(project);
            add.u32(at, base, k);
            load(a, x, at);
            mad.lo.u32(at, col, 128, k);
            load(b, weight, at);
            fma.rn.f64(z[j], a, b, z[j]);
            add.u32(k, k, 1);
            let more: pred;
            setp.lt.u32(more, k, 128);
            if more {
                bra(project);
            }
            add.u32(at, base, col);
            load(g[j], dy, at);
            fma.rn.f64(energy, z[j], z[j], energy);
            fma.rn.f64(dot, z[j], g[j], dot);
        }
        sum(energy);
        sum(dot);
        mul.rn.f64(energy, energy, 0.0078125f64);
        mul.rn.f64(dot, dot, 0.0078125f64);
        add.rn.f64(inv, energy, 0.00000011920928955078125f64);
        sqrt.rn.f64(inv, inv);
        div.rn.f64(inv, 1.0f64, inv);
        let positive: pred;
        setp.gt.f64(positive, energy, 0.0f64);
        selp.f64(den, energy, 1.0f64, positive);
        mul.rn.f64(radial, inv, inv);
        mul.rn.f64(radial, radial, inv);
        mul.rn.f64(radial, radial, 0.00000011920928955078125f64);
        for j in 0..4 {
            mul.rn.f64(parallel, z[j], dot);
            div.rn.f64(parallel, parallel, den);
            sub.rn.f64(tangent, g[j], parallel);
            mul.rn.f64(tangent, tangent, inv);
            fma.rn.f64(tangent, parallel, radial, tangent);
            add.u32(col, lane, 32 * j);
            mad.lo.u32(shared_at, col, 8, shared_base);
            st.shared.f64([shared_at], tangent);
        }
        bar.sync(0);
        for j in 0..4 {
            add.u32(col, lane, 32 * j);
            mov.f64(result, 0.0f64);
            mov.u32(k, 0);
            label(adjoint);
            mad.lo.u32(shared_at, k, 8, shared_base);
            ld.shared.f64(a, [shared_at]);
            mad.lo.u32(at, k, 128, col);
            load(b, weight, at);
            fma.rn.f64(result, a, b, result);
            add.u32(k, k, 1);
            let again: pred;
            setp.lt.u32(again, k, 128);
            if again {
                bra(adjoint);
            }
            let f: f32;
            let out: u64;
            cvt.rn.f32.f64(f, result);
            add.u32(at, base, col);
            mad.wide.u32(out, at, 4, dx);
            st.global.f32([out], f);
        }
    }
}
