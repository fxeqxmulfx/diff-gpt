//! FP32 Q/K adjoint through XSA -> output projection -> RMS, with no FP32 cut.
//! Recomputing this composed map protects epsilon-scale radial score derivatives.
#[kernite_macros::kernels]
pub mod kernels {
    fn load(v: f32, p: u64, i: u32) {
        let at: u64;
        let f: f32;
        mad.wide.u32(at, i, 4, p);
        ld.global.f32(f, [at]);
        mov.f32(v, f);
    }
    fn store(p: u64, i: u32, v: f32) {
        let at: u64;
        let f: f32;
        mov.f32(f, v);
        mad.wide.u32(at, i, 4, p);
        st.global.f32([at], f);
    }
    fn sum(v: f32) {
        let x: f32;
        for j in 0..5 {
            shfl.sync.bfly.b32(x, v, 1 << j, 31, 0xffffffff);
            add.rn.f32(v, v, x);
        }
    }
    pub unsafe extern "ptx-kernel" fn query_backward(
        dq: *mut f32,
        ds: *mut f32,
        qkv: *const f32,
        prob: *const f32,
        weight: *const f32,
        dy: *const f32,
        length: u32,
        hermitian: u32,
    ) {
        let shared_values: shared<[f32; 128], 16>;
        let (
            row,
            lane,
            batch,
            pos,
            base,
            vbase,
            anchor,
            outbase,
            scorebase,
            col,
            at,
            k,
            key,
            smem,
            sat,
            partner,
            other,
        ): (
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
            u32,
            u32,
            u32,
            u32,
            u32,
        );
        mov.u32(row, ctaid.x);
        mov.u32(lane, tid.x);
        div.u32(batch, row, length);
        rem.u32(pos, row, length);
        mul.lo.u32(base, batch, length);
        mul.lo.u32(base, base, 384);
        mad.lo.u32(vbase, pos, 384, base);
        add.u32(vbase, vbase, 256);
        add.u32(anchor, base, 256);
        mul.lo.u32(outbase, row, 128);
        mul.lo.u32(scorebase, row, length);
        mov.u32(smem, shared_values);
        let (own, jown, first, delta, z, g, ga, dchange, dquery): (
            [f32; 4],
            [f32; 4],
            [f32; 4],
            [f32; 4],
            [f32; 4],
            [f32; 4],
            [f32; 4],
            [f32; 4],
            [f32; 4],
        );
        let (
            a,
            b,
            p,
            norm,
            den,
            beta,
            eta,
            energy,
            dot,
            inv,
            parallel,
            tangent,
            radial,
            total,
            s,
            gain,
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
        );
        mov.f32(norm, 0.0f32);
        for j in 0..4 {
            add.u32(col, lane, 32 * j);
            add.u32(at, vbase, col);
            load(own[j], qkv, at);
            add.u32(at, anchor, col);
            load(first[j], qkv, at);
            let real: pred;
            setp.lt.u32(real, col, 64);
            add.u32(partner, col, 64);
            sub.u32(other, col, 64);
            selp.u32(partner, partner, other, real);
            add.u32(at, vbase, partner);
            load(jown[j], qkv, at);
            neg.f32(a, jown[j]);
            selp.f32(jown[j], a, jown[j], real);
            let complex_mode: pred;
            setp.ne.u32(complex_mode, hermitian, 0);
            selp.f32(jown[j], jown[j], 0.0f32, complex_mode);
            fma.rn.f32(norm, own[j], own[j], norm);
            mov.f32(delta[j], 0.0f32);
            mov.f32(dquery[j], 0.0f32);
        }
        sum(norm);
        max.f32(den, norm, 1e-24f32);
        mov.u32(key, 0);
        label(center);
        add.u32(at, scorebase, key);
        load(p, prob, at);
        for j in 0..4 {
            add.u32(col, lane, 32 * j);
            mad.lo.u32(at, key, 384, base);
            add.u32(at, at, 256);
            add.u32(at, at, col);
            load(a, qkv, at);
            sub.rn.f32(a, a, first[j]);
            fma.rn.f32(delta[j], p, a, delta[j]);
        }
        add.u32(key, key, 1);
        let more: pred;
        setp.le.u32(more, key, pos);
        if more {
            bra(center);
        }
        mov.f32(beta, 0.0f32);
        mov.f32(eta, 0.0f32);
        for j in 0..4 {
            sub.rn.f32(a, first[j], own[j]);
            add.rn.f32(delta[j], delta[j], a);
            fma.rn.f32(beta, delta[j], own[j], beta);
            fma.rn.f32(eta, delta[j], jown[j], eta);
        }
        sum(beta);
        sum(eta);
        div.rn.f32(beta, beta, den);
        div.rn.f32(eta, eta, den);
        for j in 0..4 {
            mul.rn.f32(a, beta, own[j]);
            sub.rn.f32(a, delta[j], a);
            mul.rn.f32(b, eta, jown[j]);
            sub.rn.f32(a, a, b);
            div.rn.f32(b, norm, den);
            sub.rn.f32(b, 1.0f32, b);
            fma.rn.f32(a, b, own[j], a);
            add.u32(col, lane, 32 * j);
            mad.lo.u32(sat, col, 4, smem);
            st.shared.f32([sat], a);
        }
        bar.sync(0);
        mov.f32(energy, 0.0f32);
        mov.f32(dot, 0.0f32);
        for j in 0..4 {
            add.u32(col, lane, 32 * j);
            mov.f32(z[j], 0.0f32);
            mov.u32(k, 0);
            label(project);
            mad.lo.u32(sat, k, 4, smem);
            ld.shared.f32(a, [sat]);
            mad.lo.u32(at, col, 128, k);
            load(b, weight, at);
            fma.rn.f32(z[j], a, b, z[j]);
            add.u32(k, k, 1);
            let next: pred;
            setp.lt.u32(next, k, 128);
            if next {
                bra(project);
            }
            add.u32(at, outbase, col);
            load(g[j], dy, at);
            fma.rn.f32(energy, z[j], z[j], energy);
            fma.rn.f32(dot, z[j], g[j], dot);
        }
        sum(energy);
        sum(dot);
        mul.rn.f32(energy, energy, 0.0078125f32);
        mul.rn.f32(dot, dot, 0.0078125f32);
        add.rn.f32(inv, energy, 0.00000011920928955078125f32);
        sqrt.rn.f32(inv, inv);
        div.rn.f32(inv, 1.0f32, inv);
        let positive: pred;
        setp.gt.f32(positive, energy, 0.0f32);
        selp.f32(b, energy, 1.0f32, positive);
        mul.rn.f32(radial, inv, inv);
        mul.rn.f32(radial, radial, inv);
        mul.rn.f32(radial, radial, 0.00000011920928955078125f32);
        for j in 0..4 {
            mul.rn.f32(parallel, z[j], dot);
            div.rn.f32(parallel, parallel, b);
            sub.rn.f32(tangent, g[j], parallel);
            mul.rn.f32(tangent, tangent, inv);
            fma.rn.f32(tangent, parallel, radial, tangent);
            add.u32(col, lane, 32 * j);
            mad.lo.u32(sat, col, 4, smem);
            st.shared.f32([sat], tangent);
        }
        bar.sync(0);
        mov.f32(beta, 0.0f32);
        mov.f32(eta, 0.0f32);
        for j in 0..4 {
            add.u32(col, lane, 32 * j);
            mov.f32(ga[j], 0.0f32);
            mov.u32(k, 0);
            label(adjoint);
            mad.lo.u32(sat, k, 4, smem);
            ld.shared.f32(a, [sat]);
            mad.lo.u32(at, k, 128, col);
            load(b, weight, at);
            fma.rn.f32(ga[j], a, b, ga[j]);
            add.u32(k, k, 1);
            let next: pred;
            setp.lt.u32(next, k, 128);
            if next {
                bra(adjoint);
            }
            fma.rn.f32(beta, ga[j], own[j], beta);
            fma.rn.f32(eta, ga[j], jown[j], eta);
        }
        sum(beta);
        sum(eta);
        div.rn.f32(beta, beta, den);
        div.rn.f32(eta, eta, den);
        for j in 0..4 {
            mul.rn.f32(a, beta, own[j]);
            sub.rn.f32(dchange[j], ga[j], a);
            mul.rn.f32(a, eta, jown[j]);
            sub.rn.f32(dchange[j], dchange[j], a);
        }
        let gain32: f32;
        add.u32(at, pos, 1);
        cvt.rn.f32.u32(gain32, at);
        lg2.approx.f32(gain32, gain32);
        mul.rn.f32(gain32, gain32, 0.6931471805599453 / 128.0);
        mov.f32(gain, gain32);
        mov.f32(total, 0.0f32);
        mov.u32(key, 0);
        label(dprob);
        mov.f32(s, 0.0f32);
        for j in 0..4 {
            add.u32(col, lane, 32 * j);
            mad.lo.u32(at, key, 384, base);
            add.u32(at, at, 256);
            add.u32(at, at, col);
            load(a, qkv, at);
            sub.rn.f32(a, a, first[j]);
            fma.rn.f32(s, dchange[j], a, s);
        }
        sum(s);
        add.u32(at, scorebase, key);
        load(p, prob, at);
        fma.rn.f32(total, p, s, total);
        let lead: pred;
        setp.eq.u32(lead, lane, 0);
        if lead {
            store(ds, at, s);
        }
        add.u32(key, key, 1);
        let more: pred;
        setp.le.u32(more, key, pos);
        if more {
            bra(dprob);
        }
        bar.sync(0);
        mov.u32(key, 0);
        label(scores);
        add.u32(at, scorebase, key);
        load(s, ds, at);
        load(p, prob, at);
        sub.rn.f32(s, s, total);
        mul.rn.f32(s, s, p);
        mul.rn.f32(s, s, gain);
        let lead: pred;
        setp.eq.u32(lead, lane, 0);
        if lead {
            store(ds, at, s);
        }
        for j in 0..4 {
            add.u32(col, lane, 32 * j);
            mad.lo.u32(at, key, 384, base);
            add.u32(at, at, 128);
            add.u32(at, at, col);
            load(a, qkv, at);
            fma.rn.f32(dquery[j], s, a, dquery[j]);
        }
        add.u32(key, key, 1);
        let more: pred;
        setp.le.u32(more, key, pos);
        if more {
            bra(scores);
        }
        for j in 0..4 {
            add.u32(col, lane, 32 * j);
            mad.lo.u32(at, pos, 384, base);
            add.u32(at, at, col);
            store(dq, at, dquery[j]);
        }
    }
}
