//! Fixed-FP32 phase-conjugated skip mixing. Read/write gates and Sinkhorn stay real.
//! Three real parameters per sublayer, with stream zero fixing the phase gauge.
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
    fn sum32(v: f32) {
        let other: f32;
        for j in 0..5 {
            shfl.sync.bfly.b32(other, v, 1 << j, 31, 0xffffffff);
            add.rn.f32(v, v, other);
        }
    }
    fn partner(other: u32, imaginary: pred, c: u32, width: u32, heads: u32) {
        let (d, half, within, plus, minus): (u32, u32, u32, u32, u32);
        div.u32(d, width, heads);
        shr.u32(half, d, 1);
        rem.u32(within, c, d);
        setp.ge.u32(imaginary, within, half);
        add.u32(plus, c, half);
        sub.u32(minus, c, half);
        selp.u32(other, minus, plus, imaginary);
    }
    fn rotate<const ADJOINT: bool>(
        y: f32,
        x: f32,
        other: f32,
        table: u64,
        edge: u32,
        imaginary: pred,
    ) {
        let at: u32;
        let (c, s, negative, selected, term): (f32, f32, f32, f32, f32);
        mad.lo.u32(at, edge, 2, 12);
        load(c, table, at);
        add.u32(at, at, 1);
        load(s, table, at);
        if ADJOINT {
            neg.f32(s, s);
        }
        neg.f32(negative, s);
        selp.f32(selected, s, negative, imaginary);
        mul.rn.f32(term, c, x);
        fma.rn.f32(y, selected, other, term);
    }
    // Layout: four interleaved unit phases, four phase derivatives, sixteen
    // interleaved edge coefficients. One thread writes this small shared table.
    pub unsafe extern "ptx-kernel" fn table(out: *mut f32, parameters: *const f32) {
        let (lane, at): (u32, u32);
        mov.u32(lane, tid.x);
        let past: pred;
        setp.ne.u32(past, lane, 0);
        if past {
            ret();
        }
        let c: [f32; 4];
        let s: [f32; 4];
        let (a, m, r, t, k, den, beta, cr, si, value): (
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
        mov.f32(c[0], 1.0);
        mov.f32(s[0], 0.0);
        mov.u32(at, 0);
        mov.f32(cr, c[0]);
        store(out, at, cr);
        mov.u32(at, 1);
        mov.f32(si, s[0]);
        store(out, at, si);
        mov.f32(beta, 0.0);
        mov.u32(at, 8);
        store(out, at, beta);
        for j in 0..3 {
            mov.u32(at, j);
            load(a, parameters, at);
            abs.f32(m, a);
            max.f32(m, m, 1.0);
            div.rn.f32(r, 1.0, m);
            div.rn.f32(t, a, m);
            mul.rn.f32(den, t, t);
            fma.rn.f32(den, r, r, den);
            sqrt.rn.f32(k, den);
            div.rn.f32(k, 1.0, k);
            mul.rn.f32(c[j + 1], r, k);
            mul.rn.f32(s[j + 1], t, k);
            mov.u32(at, 2 * (j + 1));
            mov.f32(cr, c[j + 1]);
            store(out, at, cr);
            mov.u32(at, 2 * (j + 1) + 1);
            mov.f32(si, s[j + 1]);
            store(out, at, si);
            mul.rn.f32(beta, c[j + 1], c[j + 1]);
            mov.u32(at, 9 + j);
            store(out, at, beta);
        }
        for i in 0..4 {
            for j in 0..4 {
                if i == j {
                    mov.f32(cr, 1.0);
                    mov.f32(si, 0.0);
                } else {
                    mul.rn.f32(value, s[i], s[j]);
                    fma.rn.f32(cr, c[i], c[j], value);
                    mul.rn.f32(value, c[i], s[j]);
                    neg.f32(value, value);
                    fma.rn.f32(si, s[i], c[j], value);
                }
                mov.u32(at, 12 + 2 * (4 * i + j));
                store(out, at, cr);
                mov.u32(at, 13 + 2 * (4 * i + j));
                store(out, at, si);
            }
        }
    }
    pub unsafe extern "ptx-kernel" fn write(
        y: *mut f32,
        x: *const f32,
        branch: *const f32,
        gates: *const f32,
        phases: *const f32,
        width: u32,
        heads: u32,
        count: u32,
        scale: f32,
    ) {
        let (i, row, c, stream, at, j, other, edge): (u32, u32, u32, u32, u32, u32, u32, u32);
        index(i);
        let (past, imaginary): (pred, pred);
        setp.ge.u32(past, i, count);
        if past {
            ret();
        }
        rem.u32(c, i, width);
        div.u32(row, i, width);
        rem.u32(stream, row, 4);
        div.u32(row, row, 4);
        partner(other, imaginary, c, width, heads);
        let (s, v, vi, rotated, g, b, sc): (f32, f32, f32, f32, f32, f32, f32);
        mov.f32(sc, scale);
        mad.lo.u32(at, row, width, c);
        load(b, branch, at);
        mul.rn.f32(b, b, sc);
        mad.lo.u32(at, row, 24, 4);
        add.u32(at, at, stream);
        load(g, gates, at);
        mul.rn.f32(s, g, b);
        for k in 0..4 {
            mad.lo.u32(edge, stream, 4, k);
            mad.lo.u32(at, row, 24, 8);
            add.u32(at, at, edge);
            load(g, gates, at);
            mad.lo.u32(j, row, 4, k);
            mul.lo.u32(j, j, width);
            add.u32(at, j, c);
            load(v, x, at);
            add.u32(at, j, other);
            load(vi, x, at);
            rotate::<false>(rotated, v, vi, phases, edge, imaginary);
            fma.rn.f32(s, g, rotated, s);
        }
        store(y, i, s);
    }
    pub unsafe extern "ptx-kernel" fn write_backward(
        dx: *mut f32,
        db: *mut f32,
        dy: *const f32,
        gates: *const f32,
        phases: *const f32,
        width: u32,
        heads: u32,
        count: u32,
        scale: f32,
    ) {
        let (i, row, c, at, j, other, edge): (u32, u32, u32, u32, u32, u32, u32);
        index(i);
        let (past, imaginary): (pred, pred);
        setp.ge.u32(past, i, count);
        if past {
            ret();
        }
        div.u32(row, i, width);
        rem.u32(c, i, width);
        partner(other, imaginary, c, width, heads);
        let (g, v, vi, rotated, s, b, sc): (f32, f32, f32, f32, f32, f32, f32);
        mov.f32(b, 0.0);
        mov.f32(sc, scale);
        for k in 0..4 {
            mov.f32(s, 0.0);
            for t in 0..4 {
                mov.u32(edge, 4 * t + k);
                mad.lo.u32(at, row, 24, 8 + t * 4 + k);
                load(g, gates, at);
                mad.lo.u32(j, row, 4, t);
                mul.lo.u32(j, j, width);
                add.u32(at, j, c);
                load(v, dy, at);
                add.u32(at, j, other);
                load(vi, dy, at);
                rotate::<true>(rotated, v, vi, phases, edge, imaginary);
                fma.rn.f32(s, g, rotated, s);
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
    // Overwrite only the 16 skip coefficients after the unchanged real-gate
    // reduction. Its read/write adjoints retain their original operation order.
    pub unsafe extern "ptx-kernel" fn skip_gate_grad(
        dg: *mut f32,
        x: *const f32,
        dy: *const f32,
        phases: *const f32,
        width: u32,
        heads: u32,
        rows: u32,
    ) {
        let (id, row, edge, lane, warp, c, at, j, si, sj, other): (
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
        mov.u32(id, ctaid.x);
        mov.u32(lane, tid.x);
        shr.u32(warp, lane, 5);
        mad.lo.u32(id, id, 4, warp);
        and.b32(lane, lane, 31);
        div.u32(row, id, 16);
        rem.u32(edge, id, 16);
        div.u32(si, edge, 4);
        rem.u32(sj, edge, 4);
        let (past, done, imaginary): (pred, pred, pred);
        setp.ge.u32(past, row, rows);
        if past {
            ret();
        }
        // Each edge phase is shared over every channel. Pull its coefficients
        // outside the dot products instead of rounding a rotation per channel:
        // <dy, phase*x> = c*<dy,x> + s*<dy,i*x>.
        // The first dot retains main's exact operation order; c=1,s=0
        // therefore preserves the original zero-phase adjoint bit for bit.
        let (sum, quadrature, a, b, bi, negative, signed, cr, si_phase): (
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
        mov.f32(sum, 0.0);
        mov.f32(quadrature, 0.0);
        mov.u32(c, lane);
        label(next);
        setp.ge.u32(done, c, width);
        if done {
            bra(finish);
        }
        partner(other, imaginary, c, width, heads);
        mad.lo.u32(at, row, 4, si);
        mad.lo.u32(at, at, width, c);
        load(a, dy, at);
        mad.lo.u32(j, row, 4, sj);
        mul.lo.u32(j, j, width);
        add.u32(at, j, c);
        load(b, x, at);
        add.u32(at, j, other);
        load(bi, x, at);
        neg.f32(negative, bi);
        selp.f32(signed, bi, negative, imaginary);
        fma.rn.f32(sum, a, b, sum);
        fma.rn.f32(quadrature, a, signed, quadrature);
        add.u32(c, c, 32);
        bra(next);
        label(finish);
        sum32(sum);
        sum32(quadrature);
        mad.lo.u32(at, edge, 2, 12);
        load(cr, phases, at);
        add.u32(at, at, 1);
        load(si_phase, phases, at);
        mul.rn.f32(sum, cr, sum);
        fma.rn.f32(sum, si_phase, quadrature, sum);
        let first: pred;
        setp.eq.u32(first, lane, 0);
        if first {
            mad.lo.u32(at, row, 24, 8);
            add.u32(at, at, edge);
            store(dg, at, sum);
        }
    }
    pub unsafe extern "ptx-kernel" fn phase_grad(
        gradient: *mut f32,
        x: *const f32,
        dy: *const f32,
        gates: *const f32,
        phases: *const f32,
        width: u32,
        heads: u32,
        rows: u32,
    ) {
        let (l, lane, term, total, pairs, half, d, row, pair, head, band, c, ci): (
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
        let (at, base, edge, target, source): (u32, u32, u32, u32, u32);
        mov.u32(l, ctaid.x);
        add.u32(l, l, 1);
        mov.u32(lane, tid.x);
        mov.u32(term, lane);
        shr.u32(pairs, width, 1);
        div.u32(d, width, heads);
        shr.u32(half, d, 1);
        mul.lo.u32(total, rows, pairs);
        let (sum, xr, xi, yr, yi, zr, zi, p, tmp, value, beta): (
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
        mov.f32(sum, 0.0);
        let (done, same, real_part, imag_part): (pred, pred, pred, pred);
        setp.eq.u32(real_part, 0, 1);
        setp.eq.u32(imag_part, 0, 0);
        label(next);
        setp.ge.u32(done, term, total);
        if done {
            bra(finish);
        }
        div.u32(row, term, pairs);
        rem.u32(pair, term, pairs);
        div.u32(head, pair, half);
        rem.u32(band, pair, half);
        mad.lo.u32(c, head, d, band);
        add.u32(ci, c, half);
        for other in 0..4 {
            setp.eq.u32(same, l, other);
            if !same {
                for incoming in 0..2 {
                    if incoming == 0 {
                        mov.u32(target, l);
                        mov.u32(source, other);
                    } else {
                        mov.u32(target, other);
                        mov.u32(source, l);
                    }
                    mad.lo.u32(edge, target, 4, source);
                    mad.lo.u32(at, row, 24, 8);
                    add.u32(at, at, edge);
                    load(p, gates, at);
                    mad.lo.u32(base, row, 4, source);
                    mul.lo.u32(base, base, width);
                    add.u32(at, base, c);
                    load(xr, x, at);
                    add.u32(at, base, ci);
                    load(xi, x, at);
                    rotate::<false>(zr, xr, xi, phases, edge, real_part);
                    rotate::<false>(zi, xi, xr, phases, edge, imag_part);
                    mad.lo.u32(base, row, 4, target);
                    mul.lo.u32(base, base, width);
                    add.u32(at, base, c);
                    load(yr, dy, at);
                    add.u32(at, base, ci);
                    load(yi, dy, at);
                    mul.rn.f32(tmp, yr, zi);
                    neg.f32(tmp, tmp);
                    fma.rn.f32(value, yi, zr, tmp);
                    if incoming == 1 {
                        neg.f32(value, value);
                    }
                    fma.rn.f32(sum, p, value, sum);
                }
            }
        }
        add.u32(term, term, 32);
        bra(next);
        label(finish);
        sum32(sum);
        let first: pred;
        setp.eq.u32(first, lane, 0);
        if first {
            add.u32(at, l, 8);
            load(beta, phases, at);
            mul.rn.f32(sum, sum, beta);
            sub.u32(at, l, 1);
            store(gradient, at, sum);
        }
    }
}
