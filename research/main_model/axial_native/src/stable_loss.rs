//! Cancellation-resistant cross entropy for finite FP32 logits.
//! One warp per row; target >= vocab masks both loss and gradient. Argmax
//! retains the smallest index on a tie. Caller provides a positive normalizer.
//! Exponentials remain FP32; non-target accumulation and divisions use FP32.
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
    fn load_u(v: u32, p: u64, i: u32) {
        let a: u64;
        mad.wide.u32(a, i, 4, p);
        ld.global.u32(v, [a]);
    }
    fn store_u(p: u64, i: u32, v: u32) {
        let a: u64;
        mad.wide.u32(a, i, 4, p);
        st.global.u32([a], v);
    }
    fn sum64(v: f32) {
        let x: f32;
        for j in 0..5 {
            shfl.sync.bfly.b32(x, v, 1 << j, 31, 0xffffffff);
            add.rn.f32(v, v, x);
        }
    }
    fn exp(v: f32) {
        mul.rn.f32(v, v, 1.4426950408889634);
        ex2.approx.f32(v, v);
    }
    // ln(x)=2*atanh(y), y=(x-1)/(x+1). Call only for |y|<=1/3.
    // Twelve odd terms have truncation error <1.1e-13, before FP32 rounding.
    fn log_series(out: f32, y: f32) {
        let (square, power, term): (f32, f32, f32);
        mul.rn.f32(square, y, y);
        mov.f32(power, y);
        mov.f32(out, 0.0f32);
        for j in 0..12 {
            div.rn.f32(term, power, (2 * j + 1) as f32);
            add.rn.f32(out, out, term);
            mul.rn.f32(power, power, square);
        }
        mul.rn.f32(out, out, 2.0f32);
    }
    // ln(x), x>=1: extract mantissa in [1,2) and a nonnegative exponent.
    fn log_positive(out: f32, x: f32) {
        lg2.approx.f32(out, x);
        mul.rn.f32(out, out, 0.6931471805599453f32);
    }
    pub unsafe extern "ptx-kernel" fn loss(
        dlogits: *mut f32,
        losses: *mut f32,
        predictions: *mut u32,
        logits: *const f32,
        targets: *const u32,
        vocab: u32,
        rows: u32,
        normalizer: f32,
    ) {
        let (i, b, t, n, row, lane, cls, at, target, best, peerclass): (
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
        mov.u32(b, ctaid.x);
        mov.u32(t, tid.x);
        mov.u32(n, ntid.x);
        mad.lo.u32(i, b, n, t);
        shr.u32(row, i, 5);
        and.b32(lane, i, 31);
        let past: pred;
        setp.ge.u32(past, row, rows);
        if past {
            ret();
        }
        load_u(target, targets, row);
        let supervised: pred;
        setp.lt.u32(supervised, target, vocab);
        let (v, max, peer, p, chosen, g): (f32, f32, f32, f32, f32, f32);
        mov.f32(max, f32::NEG_INFINITY);
        mov.u32(best, 0xffffffff);
        mov.u32(cls, lane);
        label(maximum);
        let valid: pred;
        setp.lt.u32(valid, cls, vocab);
        if !valid {
            bra(reduce_max);
        }
        mad.lo.u32(at, row, vocab, cls);
        load(v, logits, at);
        max.f32(max, max, v);
        add.u32(cls, cls, 32);
        bra(maximum);
        label(reduce_max);
        for j in 0..5 {
            shfl.sync.bfly.b32(peer, max, 1 << j, 31, 0xffffffff);
            max.f32(max, max, peer);
        }
        let (other, target_mass, total, term, result, norm): (f32, f32, f32, f32, f32, f32);
        mov.f32(other, 0.0f32);
        mov.f32(target_mass, 0.0f32);
        mov.f32(chosen, 0.0);
        if supervised {
            mad.lo.u32(at, row, vocab, target);
            load(chosen, logits, at);
            sub.rn.f32(p, chosen, max);
            exp(p);
            mov.f32(target_mass, p);
        }
        mov.u32(cls, lane);
        label(total_loop);
        setp.lt.u32(valid, cls, vocab);
        if !valid {
            bra(reduce_total);
        }
        mad.lo.u32(at, row, vocab, cls);
        load(v, logits, at);
        sub.rn.f32(p, v, max);
        exp(p);
        let eq: pred;
        setp.ne.u32(eq, cls, target);
        if eq {
            mov.f32(term, p);
            add.rn.f32(other, other, term);
        }
        setp.eq.f32(eq, v, max);
        if eq {
            min.u32(best, best, cls);
        }
        add.u32(cls, cls, 32);
        bra(total_loop);
        label(reduce_total);
        sum64(other);
        add.rn.f32(total, other, target_mass);
        mov.f32(norm, normalizer);
        for j in 0..5 {
            shfl.sync.bfly.b32(peerclass, best, 1 << j, 31, 0xffffffff);
            min.u32(best, best, peerclass);
        }
        mov.u32(cls, lane);
        label(gradient);
        setp.lt.u32(valid, cls, vocab);
        if !valid {
            bra(metrics);
        }
        mad.lo.u32(at, row, vocab, cls);
        mov.f32(g, 0.0);
        if !supervised {
            bra(store_gradient);
        }
        let hit: pred;
        setp.eq.u32(hit, cls, target);
        if hit {
            neg.f32(term, other);
        } else {
            load(v, logits, at);
            sub.rn.f32(p, v, max);
            exp(p);
            mov.f32(term, p);
        }
        div.rn.f32(result, term, total);
        div.rn.f32(result, result, norm);
        mov.f32(g, result);
        label(store_gradient);
        store(dlogits, at, g);
        add.u32(cls, cls, 32);
        bra(gradient);
        label(metrics);
        let lead: pred;
        setp.eq.u32(lead, lane, 0);
        if !lead {
            ret();
        }
        mov.f32(v, 0.0);
        if !supervised {
            bra(store_metrics);
        }
        let (y, denom, max64, chosen64, delta): (f32, f32, f32, f32, f32);
        let small: pred;
        setp.le.f32(small, other, 1.0f32);
        setp.eq.f32(eq, chosen, max);
        and.pred(small, small, eq);
        if small {
            add.rn.f32(denom, other, 2.0f32);
            div.rn.f32(y, other, denom);
            log_series(result, y);
        } else {
            log_positive(result, total);
            mov.f32(max64, max);
            mov.f32(chosen64, chosen);
            sub.rn.f32(delta, max64, chosen64);
            add.rn.f32(result, result, delta);
        }
        mov.f32(v, result);
        label(store_metrics);
        store(losses, row, v);
        store_u(predictions, row, best);
    }
}

#[cfg(test)]
mod tests {
    use super::kernels;
    use crate::loss_precision::reference_softmax;
    use kernite_cuda::{Cuda, Grid};

    #[test]
    #[ignore = "requires CUDA"]
    fn gpu_confidence_masks_tails_and_shift_invariance() -> crate::Result<()> {
        let cuda = Cuda::new()?;
        let kernel = kernels::Kernels::load(&cuda)?;
        for vocab in [1, 2, 3, 31, 32, 33, 97, 257] {
            let rows = 13;
            let mut logits = vec![0f32; rows * vocab];
            let mut labels = vec![0u32; rows];
            for r in 0..rows {
                let gold = r % vocab;
                labels[r] = if r == 11 {
                    u32::MAX
                } else if r == 12 {
                    vocab as u32
                } else {
                    gold as u32
                };
                let gap = match r {
                    0 => 0.,
                    1 => 5.,
                    2 => 18.,
                    3 => 20.,
                    4 => 40.,
                    5 => 80.,
                    6 => -2000.,
                    7 => 0.00001,
                    8 | 9 => 20.,
                    _ => 2.,
                };
                let offset = if r == 9 { 1000. } else { 0. };
                for c in 0..vocab {
                    logits[r * vocab + c] = offset + if c == gold { gap } else { 0. };
                }
            }
            let x = cuda.upload(&logits)?;
            let y = cuda.upload(&labels)?;
            let d = cuda.alloc::<f32>(logits.len())?;
            let loss = cuda.alloc::<f32>(rows)?;
            let pred = cuda.alloc::<u32>(rows)?;
            for normalizer in [1f32, 96., 256.] {
                unsafe {
                    kernel.loss(
                        Grid::new(rows.div_ceil(4) as u32, 128),
                        d.ptr(),
                        loss.ptr(),
                        pred.ptr(),
                        x.ptr(),
                        y.ptr(),
                        vocab as u32,
                        rows as u32,
                        normalizer,
                    )?;
                }
                let dg = d.read()?;
                let ls = loss.read()?;
                let pr = pred.read()?;
                for r in 0..rows {
                    let z = &logits[r * vocab..(r + 1) * vocab];
                    let max = z.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    assert_eq!(pr[r] as usize, z.iter().position(|&v| v == max).unwrap());
                    if labels[r] as usize >= vocab {
                        assert_eq!(ls[r], 0.);
                        assert!(dg[r * vocab..(r + 1) * vocab].iter().all(|&v| v == 0.));
                        continue;
                    }
                    let (expected, g) = reference_softmax(z, labels[r] as usize);
                    let loss_error = (ls[r] as f64 - expected).abs();
                    assert!(
                        loss_error <= 1e-5 * expected.abs() + 1e-44,
                        "vocab={vocab},row={r},loss={} expected={expected}",
                        ls[r]
                    );
                    let mut sum = 0.;
                    let mut abs = 0.;
                    for c in 0..vocab {
                        let want = g[c] / normalizer as f64;
                        let got = dg[r * vocab + c] as f64;
                        assert!(
                            (got - want).abs() <= 1e-5 * want.abs() + 1e-44,
                            "vocab={vocab},row={r},class={c},got={got} expected={want}"
                        );
                        if want as f32 != 0. {
                            assert_ne!(got, 0., "lost representable derivative");
                        }
                        sum += got;
                        abs += got.abs();
                    }
                    assert!(
                        sum.abs() <= 2e-7 * abs + 1e-44,
                        "gradient no longer sums to zero"
                    );
                }
                assert_eq!(ls[8], ls[9]);
                // Rows 8/9 place the target in different columns; loss and sorted
                // derivatives must agree after adding the common logit offset.
                let mut a = dg[8 * vocab..9 * vocab].to_vec();
                let mut b = dg[9 * vocab..10 * vocab].to_vec();
                a.sort_by(f32::total_cmp);
                b.sort_by(f32::total_cmp);
                assert_eq!(a, b);
            }
        }
        Ok(())
    }
}
