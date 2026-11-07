// FP32 SGEMM from kernite examples/matmul, a1b6bc2, MIT.
// Copyright retained in KERNITE_LICENSE. No TF32 or reduced-precision products.
/// The kernels.
#[kernite_macros::kernels]
pub mod kernels {
    /// Rows of `c` a block computes.
    pub const BM: usize = 128;
    /// Columns of `c` a block computes.
    pub const BN: usize = 1 << BN_SHIFT;
    /// The power of 2 that `BN` is, so that `sgemm` divides by `BN` with a shift.
    const BN_SHIFT: usize = 7;
    /// Steps of `k` a tile takes.
    pub const BK: usize = 8;
    /// Threads in a block: each computes 8 rows by 8 columns of `c`.
    pub const THREADS: usize = 256;
    /// Floats from one step's row of a tile in shared memory to the next: a row of `BM` and four
    /// more. The 32 lanes of a warp store the floats of 16 rows of `a`, in two halves 4 steps of
    /// `k` apart, and `4 * S` floats are 16 banks along, so that the 32 land in banks of their own.
    const S: usize = BM + 4;
    /// Bytes of a tile in shared memory.
    const TILE: usize = BK * S * 4;

    /// `c = a b`, of `a`, `m` rows of `k` floats, and `b`, `k` rows of `n`, into `c`, `m` rows
    /// of `n`: each element of `c` a chain of fused multiply-adds over `k` in order, from +0.
    /// `m` and `n` are 1 or more. The grid is a row of blocks of `THREADS` threads, one for each
    /// tile of `c`, `BM` rows by `BN` columns: the tiles of a row of tiles, then those of the
    /// next.
    ///
    /// A thread keeps 8 rows by 8 columns of its block's tile in 64 registers: the 4 rows from
    /// `ty * 4` and the 4 from `64 + ty * 4`, the same for the columns by `tx * 4`, where the
    /// threads are 16 `tx` by 16 `ty`, and a warp 8 `tx` by 4 `ty`. For each step of `k` it reads
    /// 8 floats of `a` and 8 of `b` from shared memory, and adds all 64 products to its tile.
    ///
    /// The tiles of `a` and `b` go through shared memory, `BK` steps of `k` at a time, in two
    /// buffers. While a tile is multiplied, the next is read from global memory into registers,
    /// then stored to the other buffer, so that a block waits at one barrier for each tile. A tile
    /// of `a` is stored transposed, a row of `S` floats for each step, so that a thread's 8 floats
    /// of a step are two pieces of 16 bytes.
    ///
    /// The first tile has `k % BK` steps, none if `k` is a multiple of `BK`, and the tiles after
    /// it are whole. So no chain takes a step past `k` on zeros, which would turn a chain at -0
    /// into one at +0, as `fma(0, 0, -0)` is +0. Past the edges of `c` the tiles hold zeros, and
    /// nothing is stored.
    ///
    /// `WIDE` is for an `n` and a `k` that are multiples of 4, and `a`, `b` and `c` that start at
    /// a multiple of 16 bytes: the global loads and stores are then `v4`, 16 bytes each. Without
    /// it they are 4 bytes.
    #[instances(true, false)]
    pub unsafe extern "ptx-kernel" fn sgemm<const WIDE: bool>(
        c: *mut f32,
        a: *const f32,
        b: *const f32,
        m: u32,
        n: u32,
        k: u32,
    ) {
        let a_tiles: shared<[f32; 2 * BK * S], 16>;
        let b_tiles: shared<[f32; 2 * BK * S], 16>;
        let t: u32;
        mov.u32(t, tid.x);
        let (tx, ty): (u32, u32);
        lanes(tx, ty);
        let (a_rd, b_rd): (u32, u32);
        mov.u32(a_rd, a_tiles);
        mad.lo.u32(a_rd, ty, 16, a_rd);
        mov.u32(b_rd, b_tiles);
        mad.lo.u32(b_rd, tx, 16, b_rd);

        note("what this thread loads: 4 floats of a row of a's tile, and 4 of a step's row of b's");
        let (ar, ak, br, bc): (u32, u32, u32, u32);
        shr.u32(ar, t, 1);
        and.b32(ak, t, 1);
        shl.b32(ak, ak, 2);
        shr.u32(br, t, 5);
        and.b32(bc, t, 31);
        shl.b32(bc, bc, 2);
        let (a_wr, b_wr, base): (u32, u32, u32);
        mad.lo.u32(a_wr, ak, S, ar);
        mov.u32(base, a_tiles);
        mad.lo.u32(a_wr, a_wr, 4, base);
        mad.lo.u32(b_wr, br, S, bc);
        mov.u32(base, b_tiles);
        mad.lo.u32(b_wr, b_wr, 4, base);

        let (i0, j0, rows, cols): (u32, u32, u32, u32);
        tile(i0, j0, rows, cols, m, n);
        let (row, col): (u32, u32);
        add.u32(row, i0, ar);
        add.u32(col, j0, bc);
        note("where the first tile of a and of b is, in floats from their starts");
        let (oa, ob): (u32, u32);
        let (pa, pb): (u64, u64);
        mad.lo.u32(oa, row, k, ak);
        mad.lo.u32(ob, br, n, col);

        note("the loads whose row of a, or columns of b, lie in the matrices");
        let (ok_a, ok_b): ([pred; 4], [pred; 4]);
        let u: u32;
        for j in 0..4 {
            if !WIDE || j == 0 {
                setp.lt.u32(ok_a[j], ar, rows);
                add.u32(u, bc, j);
                setp.lt.u32(ok_b[j], u, cols);
            }
        }
        let (va, vb): ([f32; 4], [f32; 4]);
        for j in 0..4 {
            mov.f32(va[j], 0);
            mov.f32(vb[j], 0);
        }
        let acc: [f32; 64];
        for i in 0..64 {
            mov.f32(acc[i], 0);
        }
        let (fa0, fb0, fa1, fb1): ([f32; 8], [f32; 8], [f32; 8], [f32; 8]);

        let steps: u32;
        and.b32(steps, k, BK - 1);
        let more: pred;
        setp.eq.u32(more, steps, 0);
        if more {
            bra(whole);
        }
        note("the first tile is short of BK steps of k, so that the others are whole");
        let (oa_whole, ob_whole): (u32, u32);
        add.u32(oa_whole, oa, steps);
        mad.lo.u32(ob_whole, steps, n, ob);
        let (in_b, in_a): (pred, pred);
        let (part_a, part_b): ([pred; 4], [pred; 4]);
        setp.lt.u32(in_b, br, steps);
        for j in 0..4 {
            if !WIDE || j == 0 {
                and.pred(part_b[j], ok_b[j], in_b);
                add.u32(u, ak, j);
                setp.lt.u32(in_a, u, steps);
                and.pred(part_a[j], ok_a[j], in_a);
            }
        }
        mad.wide.u32(pa, oa, 4, a);
        mad.wide.u32(pb, ob, 4, b);
        load::<WIDE>(va, vb, pa, pb, part_a, part_b);
        store(va, vb, a_wr, b_wr);
        bar.sync(0);
        let (x, y, left): (u32, u32, u32);
        mov.u32(x, a_rd);
        mov.u32(y, b_rd);
        mov.u32(left, steps);
        label(short);
        frags::<0>(fa0, fb0, x, y);
        products(acc, fa0, fb0);
        add.u32(x, x, 4 * S);
        add.u32(y, y, 4 * S);
        sub.u32(left, left, 1);
        setp.ne.u32(more, left, 0);
        if more {
            bra(short);
        }
        bar.sync(0);
        mov.u32(oa, oa_whole);
        mov.u32(ob, ob_whole);

        label(whole);
        note("the whole tiles: the next is loaded while this one is multiplied");
        let nfull: u32;
        shr.u32(nfull, k, 3);
        setp.eq.u32(more, nfull, 0);
        if more {
            bra(done);
        }
        let (stride, go, d): (u32, pred, u32);
        mul.lo.u32(stride, n, BK);
        mad.wide.u32(pa, oa, 4, a);
        mad.wide.u32(pb, ob, 4, b);
        load::<WIDE>(va, vb, pa, pb, ok_a, ok_b);
        store(va, vb, a_wr, b_wr);
        setp.gt.u32(go, nfull, 1);
        if go {
            add.u32(oa, oa, BK);
            add.u32(ob, ob, stride);
        }
        bar.sync(0);
        mov.u32(d, TILE);
        add.u32(a_wr, a_wr, d);
        add.u32(b_wr, b_wr, d);
        mov.u32(left, nfull);
        label(tile);
        mad.wide.u32(pa, oa, 4, a);
        mad.wide.u32(pb, ob, 4, b);
        load::<WIDE>(va, vb, pa, pb, ok_a, ok_b);
        setp.gt.u32(go, left, 2);
        if go {
            add.u32(oa, oa, BK);
            add.u32(ob, ob, stride);
        }
        frags::<0>(fa0, fb0, a_rd, b_rd);
        for s in 0..BK {
            let next = s + 1;
            if s % 2 == 0 {
                if next < BK {
                    frags::<next>(fa1, fb1, a_rd, b_rd);
                }
                products(acc, fa0, fb0);
            } else {
                if next < BK {
                    frags::<next>(fa0, fb0, a_rd, b_rd);
                }
                products(acc, fa1, fb1);
            }
        }
        store(va, vb, a_wr, b_wr);
        add.u32(a_rd, a_rd, d);
        sub.u32(a_wr, a_wr, d);
        add.u32(b_rd, b_rd, d);
        sub.u32(b_wr, b_wr, d);
        sub.u32(d, 0, d);
        bar.sync(0);
        sub.u32(left, left, 1);
        setp.ne.u32(more, left, 0);
        if more {
            bra(tile);
        }

        label(done);
        note(
            "the tile of c: rows ty * 4 + r and 64 + ty * 4 + r, columns tx * 4 + q and 64 + tx * 4 + q",
        );
        lanes(tx, ty);
        tile(i0, j0, rows, cols, m, n);
        let (ci, cj, idx): (u32, u32, u32);
        shl.b32(ci, ty, 2);
        shl.b32(cj, tx, 2);
        for r in 0..8 {
            let (li, ri, ok_r): (u32, u32, pred);
            add.u32(li, ci, r % 4 + 64 * (r / 4));
            setp.lt.u32(ok_r, li, rows);
            add.u32(ri, i0, li);
            for g in 0..2 {
                let (lj, ok): (u32, pred);
                let (cg, pc): (u32, u64);
                add.u32(lj, cj, 64 * g);
                add.u32(cg, j0, lj);
                mad.lo.u32(idx, ri, n, cg);
                mad.wide.u32(pc, idx, 4, c);
                if WIDE {
                    setp.lt.u32(ok, lj, cols);
                    and.pred(ok, ok, ok_r);
                    if ok {
                        st.global.v4.f32(
                            [pc],
                            (
                                acc[8 * r + 4 * g],
                                acc[8 * r + 4 * g + 1],
                                acc[8 * r + 4 * g + 2],
                                acc[8 * r + 4 * g + 3],
                            ),
                        );
                    }
                } else {
                    for q in 0..4 {
                        add.u32(u, lj, q);
                        setp.lt.u32(ok, u, cols);
                        and.pred(ok, ok, ok_r);
                        if ok {
                            st.global.f32([pc + 4 * q], acc[8 * r + 4 * g + q]);
                        }
                    }
                }
            }
        }
    }

    /// This thread's place among the 256: `tx` and `ty`, 16 each, where a warp is 8 `tx` by 4
    /// `ty`. It keeps rows `ty * 4` and `64 + ty * 4` of the tile of `c`, and columns `tx * 4` and
    /// `64 + tx * 4`.
    fn lanes(tx: u32, ty: u32) {
        let (t, hi): (u32, u32);
        mov.u32(t, tid.x);
        and.b32(tx, t, 7);
        and.b32(hi, t, 32);
        shr.u32(hi, hi, 2);
        or.b32(tx, tx, hi);
        shr.u32(ty, t, 3);
        and.b32(ty, ty, 3);
        shr.u32(hi, t, 6);
        shl.b32(hi, hi, 2);
        or.b32(ty, ty, hi);
    }

    /// This block's tile of `c`: its first row `i0` and column `j0`, and how many rows and columns
    /// of it lie in `c`. The tiles are counted along a row, `across` of them, then down. A block's
    /// number is its place along the grid's x. The tile is worked out again where it is wanted
    /// again, as it takes fewer registers than it would to keep.
    fn tile(i0: u32, j0: u32, rows: u32, cols: u32, m: u32, n: u32) {
        let (across, id, bi, bj): (u32, u32, u32, u32);
        // `n` is 1 or more, so `n - 1` does not overflow, as `n + BN - 1` could. The division by
        // `BN` is a shift. `ptxas` 13.4 divides by 128 as by any number, and keeps the result
        // through the main loop, in a register of its own: the loop then runs 0.6% slower at 2048.
        sub.u32(across, n, 1);
        shr.u32(across, across, BN_SHIFT);
        add.u32(across, across, 1);
        mov.u32(id, ctaid.x);
        div.u32(bi, id, across);
        rem.u32(bj, id, across);
        mul.lo.u32(i0, bi, BM);
        mul.lo.u32(j0, bj, BN);
        sub.u32(rows, m, i0);
        sub.u32(cols, n, j0);
    }

    /// Loads this thread's floats of the tile of `a` at `pa`, and of `b` at `pb`, the ones whose
    /// predicates hold.
    fn load<const WIDE: bool>(
        va: [f32; 4],
        vb: [f32; 4],
        pa: u64,
        pb: u64,
        ok_a: [pred; 4],
        ok_b: [pred; 4],
    ) {
        if WIDE {
            if ok_a[0] {
                ld.global.v4.f32((va[0], va[1], va[2], va[3]), [pa]);
            }
            if ok_b[0] {
                ld.global.v4.f32((vb[0], vb[1], vb[2], vb[3]), [pb]);
            }
        } else {
            for j in 0..4 {
                if ok_a[j] {
                    ld.global.f32(va[j], [pa + 4 * j]);
                }
                if ok_b[j] {
                    ld.global.f32(vb[j], [pb + 4 * j]);
                }
            }
        }
    }

    /// Stores what `load` loaded to shared memory: the floats of `a`, a column each, so that
    /// a step's row of the tile is `BM` rows of `a`; and of `b` as they came.
    fn store(va: [f32; 4], vb: [f32; 4], a_wr: u32, b_wr: u32) {
        for j in 0..4 {
            st.shared.f32([a_wr + 4 * S * j], va[j]);
        }
        st.shared.v4.f32([b_wr], (vb[0], vb[1], vb[2], vb[3]));
    }

    /// The floats of step `T` of a tile in shared memory, for this thread's 8 rows and 8
    /// columns: `a` and `b` are its addresses.
    fn frags<const T: usize>(fa: [f32; 8], fb: [f32; 8], a: u32, b: u32) {
        ld.shared
            .v4
            .f32((fa[0], fa[1], fa[2], fa[3]), [a + 4 * S * T]);
        ld.shared
            .v4
            .f32((fa[4], fa[5], fa[6], fa[7]), [a + 4 * S * T + 256]);
        ld.shared
            .v4
            .f32((fb[0], fb[1], fb[2], fb[3]), [b + 4 * S * T]);
        ld.shared
            .v4
            .f32((fb[4], fb[5], fb[6], fb[7]), [b + 4 * S * T + 256]);
    }

    /// Adds the products of `fa` and `fb` to `acc`, each a fused multiply-add.
    fn products(acc: [f32; 64], fa: [f32; 8], fb: [f32; 8]) {
        for i in 0..8 {
            for j in 0..8 {
                fma.rn.f32(acc[8 * i + j], fa[i], fb[j], acc[8 * i + j]);
            }
        }
    }
}
