//! Adapted from Kernite's matmul/tensor.rs (snapshot/hash in tensor-tiles-02).
//! Three-stage asynchronous shared-memory pipeline; fixed FP16/BF16 formats.
#[kernite_macros::kernels]
pub mod kernels {
    /// Rows of `c` a block computes.
    pub const BM: usize = 64;
    /// Columns of `c` a block computes: a multiple of 64.
    pub const BN: usize = 64;
    /// Steps of `k` a tile takes: a multiple of 32.
    pub const BK: usize = 32;
    /// Tiles of `a` and `b` a block holds in shared memory: one is multiplied, the rest are on
    /// their way.
    pub const STAGES: usize = 3;
    /// Threads in a block: 8 warps, 2 down and 4 across the tile of `c`.
    pub const THREADS: usize = 128;
    /// Tile rows of `c` that blocks walk down before they go across, so that the blocks that run
    /// together read few rows of `a` and few columns of `b`: 4 by 4 tiles are the 16 blocks this
    /// GPU runs at once.
    pub const GROUP: usize = 4;
    /// Bytes of a stage in shared memory: a tile of `a`, then a tile of `b`.
    pub const STAGE: usize = (BM + BN) * BK * 2;

    /// Warps along the columns of the tile; the others stack down its rows.
    const WARPS_N: usize = 2;
    /// Rows and columns of the tile a warp keeps.
    const WM: usize = BM / (THREADS / 32 / WARPS_N);
    const WN: usize = BN / WARPS_N;
    /// How many tiles of one `mma`, 16 rows by 8 columns, the warp's tile has down and across.
    const MT: usize = WM / 16;
    const NT: usize = WN / 8;
    /// Steps of 16 of `k`, those of one `mma`, in a tile: an even number, so that the sets of
    /// fragments take turns the same way in each tile.
    const KG: usize = BK / 16;
    /// Registers of a set of fragments: `a`'s, 4 for each of `MT` tiles of 16 rows, and `b`'s, 2
    /// for each of `NT` tiles of 8 columns. A warp has two sets, to load one while it multiplies
    /// the other.
    const FA: usize = 4 * MT;
    const FB: usize = 2 * NT;
    /// Pieces of 16 bytes in a row of a tile of `a`, which has `k` along it, and in a row of `b`.
    const A_PIECES: usize = BK / 8;
    const B_PIECES: usize = BN / 8;
    /// Bytes in a row of a tile of `b`, and in a tile of `a`.
    const B_ROW: usize = B_PIECES * 16;
    const A_BYTES: usize = BM * A_PIECES * 16;
    /// How many pieces of each tile a thread copies; the next is `THREADS` pieces on.
    const A_COPIES: usize = BM * A_PIECES / THREADS;
    const B_COPIES: usize = BK * B_PIECES / THREADS;
    /// The rows of a tile, and the bytes of shared memory, between a piece a thread copies and
    /// its next.
    const A_ROWS: usize = THREADS / A_PIECES;
    const B_ROWS: usize = THREADS / B_PIECES;
    const COPY: usize = THREADS * 16;

    /// `c = a b`, of `a`, `m` rows of `k` halves, and `b`, `k` rows of `n`, into `c`, `m` rows
    /// of `n` floats. `m` and `n` are 1 or more, and `n` and `k` are multiples of 8, so that a
    /// row is whole pieces of 16 bytes, and `a`, `b` and `c` start at a multiple of 16 bytes. The
    /// grid has a block of `THREADS` threads for each tile of `c`, `BM` rows by `BN` columns.
    ///
    /// A warp keeps `WM` rows by `WN` columns of its block's tile in `f32` registers. For each
    /// step of 16 of `k` it loads the rows of `a` and the columns of `b` that its tile needs with
    /// `ldmatrix`, and adds their products to it with `mma`, `MT` by `NT` of them. It loads the
    /// next step's before it multiplies this one's, so that the tensor cores never wait for a
    /// load.
    ///
    /// The tiles of `a` and `b` go through shared memory, `BK` steps of `k` at a time, in
    /// `STAGES` stages: `cp.async` copies the tile `STAGES - 1` ahead while a tile is multiplied,
    /// and a block waits at one barrier for each tile. A row of `a`'s tile is 64 bytes and one
    /// of `b`'s 256, so the 8 rows an `ldmatrix` reads at once, if stored in order, would share
    /// their banks. Instead a piece of 16 bytes is stored in its row at the place its column xor
    /// a number of its row gives, which puts those 8 rows, and the 8 pieces that 8 threads write
    /// with `cp.async`, in banks of their own.
    ///
    /// The last tile of `k` may be short, and the last tiles of `m` and `n` may reach past the
    /// rows of `a` and the columns of `b`. For each piece outside, `cp.async` reads nothing and
    /// copies zeros, so that what is outside adds zero. Past the edges of `c` nothing is stored.
    #[instances((false, false), (false, true), (true, false), (true, true))]
    pub unsafe extern "ptx-kernel" fn hgemm<const BF16: bool, const FINAL: bool>(
        c: *mut f32,
        a: *const u16,
        b: *const u16,
        m: u32,
        n: u32,
        full_k: u32,
        chunk: u32,
    ) {
        // `smem` starts at a multiple of 128 bytes, the width of the 32 banks, which the xors in
        // the doc need.
        let smem: shared<[f16], 128>;
        let (smem_a, smem_b): (u32, u32);
        mov.u32(smem_a, smem);
        add.u32(smem_b, smem_a, A_BYTES);
        let (t, lane, warp): (u32, u32, u32);
        mov.u32(t, tid.x);
        and.b32(lane, t, 31);
        shr.u32(warp, t, 5);
        let (m0, n0): (u32, u32);
        block_tile(m0, n0, m, n);
        let (mt, nt, tiles, block, part, start, k, index): (
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
        );
        add.u32(mt, m, BM - 1);
        div.u32(mt, mt, BM);
        add.u32(nt, n, BN - 1);
        div.u32(nt, nt, BN);
        mul.lo.u32(tiles, mt, nt);
        mov.u32(block, ctaid.x);
        div.u32(part, block, tiles);
        mul.lo.u32(start, part, chunk);
        sub.u32(k, full_k, start);
        min.u32(k, k, chunk);
        mad.wide.u32(a, start, 2, a);
        mul.lo.u32(index, start, n);
        mad.wide.u32(b, index, 2, b);
        mul.lo.u32(index, m, n);
        mul.lo.u32(index, index, part);
        mad.wide.u32(c, index, 4, c);
        let (rows, cols): (u32, u32);
        sub.u32(rows, m, m0);
        sub.u32(cols, n, n0);

        note(
            "what this thread copies: a piece of a's tile and one of b's, then the pieces THREADS on",
        );
        let (ar, aq, ka): (u32, u32, u32);
        div.u32(ar, t, A_PIECES);
        rem.u32(aq, t, A_PIECES);
        mul.lo.u32(ka, aq, 8);
        let (pa, pb): ([u64; A_COPIES], [u64; B_COPIES]);
        let (ok_a, ok_b): ([pred; A_COPIES], pred);
        let idx: u32;
        for j in 0..A_COPIES {
            add.u32(idx, ar, A_ROWS * j);
            setp.lt.u32(ok_a[j], idx, rows);
            add.u32(idx, idx, m0);
            mad.lo.u32(idx, idx, full_k, ka);
            mad.wide.u32(pa[j], idx, 2, a);
        }
        let (br, bq, kb): (u32, u32, u32);
        div.u32(br, t, B_PIECES);
        rem.u32(bq, t, B_PIECES);
        mul.lo.u32(kb, bq, 8);
        setp.lt.u32(ok_b, kb, cols);
        add.u32(kb, kb, n0);
        for j in 0..B_COPIES {
            add.u32(idx, br, B_ROWS * j);
            mad.lo.u32(idx, idx, n, kb);
            mad.wide.u32(pb[j], idx, 2, b);
        }
        let step_b: u64;
        mul.wide.u32(step_b, n, 2 * BK);

        note(
            "where they go in a stage: a piece's column xor a number of its row, so 8 rows meet in no bank",
        );
        let (wa, wb, x): (u32, u32, u32);
        shr.u32(x, t, 3);
        and.b32(x, x, 7);
        xor.b32(wa, t, x);
        mad.lo.u32(wa, wa, 16, smem_a);
        and.b32(x, br, 7);
        xor.b32(x, bq, x);
        mad.lo.u32(wb, br, B_ROW, smem_b);
        mad.lo.u32(wb, x, 16, wb);

        note("where ldmatrix reads: lane l gives the address of row l % 8 of matrix l / 8");
        let (la, lb): ([u32; KG], [u32; NT / 2]);
        let (wm, wn, y): (u32, u32, u32);
        div.u32(wm, warp, WARPS_N);
        rem.u32(wn, warp, WARPS_N);
        shr.u32(x, lane, 1);
        and.b32(x, x, 7);
        and.b32(y, lane, 15);
        mul.lo.u32(y, y, A_PIECES);
        for g in 0..KG {
            shr.u32(la[g], lane, 4);
            add.u32(la[g], la[g], y);
            add.u32(la[g], la[g], 2 * g);
            xor.b32(la[g], la[g], x);
            mad.lo.u32(la[g], wm, WM * A_PIECES, la[g]);
            mad.lo.u32(la[g], la[g], 16, smem_a);
        }
        and.b32(x, lane, 7);
        shr.u32(y, lane, 4);
        mad.lo.u32(y, wn, NT, y);
        let row: u32;
        and.b32(row, lane, 15);
        for j in 0..NT / 2 {
            add.u32(lb[j], y, 2 * j);
            xor.b32(lb[j], lb[j], x);
            mad.lo.u32(lb[j], lb[j], 16, smem_b);
            mad.lo.u32(lb[j], row, B_ROW, lb[j]);
        }

        let acc: [f32; 4 * MT * NT];
        for i in 0..4 * MT * NT {
            mov.f32(acc[i], 0);
        }
        let (fa, fb): ([b32; 2 * FA], [b32; 2 * FB]);

        note("the tiles of k, the last short if k is not a multiple of BK");
        let (left, rest, kleft): (u32, u32, u32);
        div.u32(left, k, BK);
        rem.u32(rest, k, BK);
        let short: pred;
        setp.ne.u32(short, rest, 0);
        if short {
            add.u32(left, left, 1);
        }
        mov.u32(kleft, k);
        let (rd, wr): (u32, u32);
        mov.u32(rd, 0);
        let (sa, sb): (u32, u32);
        for s in 0..STAGES - 1 {
            add.u32(sa, wa, s * STAGE);
            add.u32(sb, wb, s * STAGE);
            load(sa, sb, pa, pb, ok_a, ok_b, ka, br, kleft, step_b);
            cp.r#async.commit_group();
        }
        mov.u32(wr, (STAGES - 1) * STAGE);
        let no_tiles: pred;
        setp.eq.u32(no_tiles, left, 0);
        if no_tiles {
            bra(done);
        }
        cp.r#async.wait_group(STAGES - 2);
        bar.sync(0);
        frags::<0, 0>(fa, fb, la, lb, rd);

        label(tile);
        for g in 0..KG {
            let cur = g % 2;
            let next = (g + 1) % 2;
            let after = g + 1;
            if after < KG {
                frags::<after, next>(fa, fb, la, lb, rd);
            }
            if g == 0 {
                add.u32(sa, wa, wr);
                add.u32(sb, wb, wr);
                load(sa, sb, pa, pb, ok_a, ok_b, ka, br, kleft, step_b);
                cp.r#async.commit_group();
            }
            if after == KG {
                note(
                    "the next tile has landed and no warp reads this one now: its stage is the next to write",
                );
                cp.r#async.wait_group(STAGES - 2);
                bar.sync(0);
                mov.u32(wr, rd);
                add.u32(rd, rd, STAGE);
                let end: pred;
                setp.eq.u32(end, rd, STAGES * STAGE);
                if end {
                    mov.u32(rd, 0);
                }
                frags::<0, next>(fa, fb, la, lb, rd);
            }
            products::<cur, BF16>(acc, fa, fb);
        }
        sub.u32(left, left, 1);
        let again: pred;
        setp.ne.u32(again, left, 0);
        if again {
            bra(tile);
        }

        label(done);
        cp.r#async.wait_all();
        if FINAL {
            // Direct output is enabled only for one partition without N padding.
            for i in 0..4 * MT * NT {
                add.rn.f32(acc[i], 0.0, acc[i]);
            }
        }
        note(
            "this thread's floats of an mma tile: rows lane / 4 and + 8, columns 2 (lane % 4) and + 1",
        );
        // `gid` and `tig` are what the PTX ISA calls `lane / 4` and `lane % 4`; `col` is the first
        // of the two columns a thread has in a tile, 2 `tig`.
        let (gid, tig, col): (u32, u32, u32);
        shr.u32(gid, lane, 2);
        and.b32(tig, lane, 3);
        shl.b32(col, tig, 1);
        let (ci, cj): (u32, u32);
        mad.lo.u32(ci, wm, WM, gid);
        mad.lo.u32(cj, wn, WN, col);
        let ok_c: [pred; NT];
        for nt in 0..NT {
            add.u32(x, cj, 8 * nt);
            setp.lt.u32(ok_c[nt], x, cols);
        }
        let (ok_r, ok): (pred, pred);
        let pc: u64;
        for mt in 0..MT {
            for h in 0..2 {
                add.u32(x, ci, 16 * mt + 8 * h);
                setp.lt.u32(ok_r, x, rows);
                add.u32(idx, x, m0);
                mad.lo.u32(idx, idx, n, n0);
                add.u32(idx, idx, cj);
                mad.wide.u32(pc, idx, 4, c);
                for nt in 0..NT {
                    and.pred(ok, ok_r, ok_c[nt]);
                    let at = 4 * (NT * mt + nt);
                    if ok {
                        st.global
                            .v2
                            .f32([pc + 32 * nt], (acc[at + 2 * h], acc[at + 2 * h + 1]));
                    }
                }
            }
        }
    }

    /// This block's tile of `c`: its first row `m0` and column `n0`. The tiles are counted down
    /// `GROUP` of them, then across, the last group down as many as are left.
    fn block_tile(m0: u32, n0: u32, m: u32, n: u32) {
        let (tiles_m, tiles_n, width, first, size, within, bm, bn): (
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
        );
        // `m` and `n` are 1 or more, so these do not overflow, as `m + BM - 1` could.
        sub.u32(tiles_m, m, 1);
        div.u32(tiles_m, tiles_m, BM);
        add.u32(tiles_m, tiles_m, 1);
        sub.u32(tiles_n, n, 1);
        div.u32(tiles_n, tiles_n, BN);
        add.u32(tiles_n, tiles_n, 1);
        mul.lo.u32(width, tiles_n, GROUP);
        let id: u32;
        mov.u32(id, ctaid.x);
        let tile_count: u32;
        mul.lo.u32(tile_count, tiles_m, tiles_n);
        rem.u32(id, id, tile_count);
        div.u32(first, id, width);
        mul.lo.u32(within, first, width);
        sub.u32(within, id, within);
        mul.lo.u32(first, first, GROUP);
        sub.u32(size, tiles_m, first);
        min.u32(size, size, GROUP);
        rem.u32(bm, within, size);
        add.u32(bm, bm, first);
        div.u32(bn, within, size);
        mul.lo.u32(m0, bm, BM);
        mul.lo.u32(n0, bn, BN);
    }

    /// Copies the next tile of `a` and of `b`, from `pa` and `pb`, to the stage at `sa` and `sb`,
    /// and moves `pa` and `pb` on to the tile after. `kleft` is the steps of `k` left from this
    /// tile on, and `load` leaves it the steps left after it. A piece past the rows of `a`, the
    /// columns of `b` or the `kleft` steps is not read: it is zeros.
    fn load(
        sa: u32,
        sb: u32,
        pa: [u64; A_COPIES],
        pb: [u64; B_COPIES],
        ok_a: [pred; A_COPIES],
        ok_b: pred,
        ka: u32,
        br: u32,
        kleft: u32,
        step_b: u64,
    ) {
        let (inside, ok): (pred, pred);
        setp.lt.u32(inside, ka, kleft);
        for j in 0..A_COPIES {
            and.pred(ok, ok_a[j], inside);
            cp.r#async
                .cg
                .shared
                .global([sa + COPY * j], [pa[j]], 16, !ok);
            add.u64(pa[j], pa[j], 2 * BK);
        }
        let kk: u32;
        for j in 0..B_COPIES {
            add.u32(kk, br, B_ROWS * j);
            setp.lt.u32(inside, kk, kleft);
            and.pred(ok, ok_b, inside);
            cp.r#async
                .cg
                .shared
                .global([sb + COPY * j], [pb[j]], 16, !ok);
            add.u64(pb[j], pb[j], step_b);
        }
        max.u32(kleft, kleft, BK);
        sub.u32(kleft, kleft, BK);
    }

    /// Loads step `G` of the stage at `rd` into set `SET` of `fa` and `fb`: for each of the
    /// warp's `MT` tiles of 16 rows, 16 steps of `a`, and for each two of its `NT` tiles of 8
    /// columns, 16 steps of `b`.
    fn frags<const G: usize, const SET: usize>(
        fa: [b32; 2 * FA],
        fb: [b32; 2 * FB],
        la: [u32; KG],
        lb: [u32; NT / 2],
        rd: u32,
    ) {
        let (x, y): (u32, u32);
        add.u32(x, la[G], rd);
        for mt in 0..MT {
            let at = FA * SET + 4 * mt;
            let down = 16 * A_PIECES * 16 * mt;
            ldmatrix
                .sync
                .aligned
                .m8n8
                .x4
                .shared
                .b16((fa[at], fa[at + 1], fa[at + 2], fa[at + 3]), [x + down]);
        }
        for j in 0..NT / 2 {
            let at = FB * SET + 4 * j;
            let down = 16 * B_ROW * G;
            add.u32(y, lb[j], rd);
            ldmatrix
                .sync
                .aligned
                .m8n8
                .x4
                .trans
                .shared
                .b16((fb[at], fb[at + 1], fb[at + 2], fb[at + 3]), [y + down]);
        }
    }

    /// Adds the products of set `SET` of `fa` and `fb` to `acc`.
    fn products<const SET: usize, const BF16: bool>(
        acc: [f32; 4 * MT * NT],
        fa: [b32; 2 * FA],
        fb: [b32; 2 * FB],
    ) {
        for mt in 0..MT {
            for nt in 0..NT {
                let at = 4 * (NT * mt + nt);
                let (ia, ib) = (FA * SET + 4 * mt, FB * SET + 2 * nt);
                if BF16 {
                    mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32(
                        (acc[at], acc[at + 1], acc[at + 2], acc[at + 3]),
                        (fa[ia], fa[ia + 1], fa[ia + 2], fa[ia + 3]),
                        (fb[ib], fb[ib + 1]),
                        (acc[at], acc[at + 1], acc[at + 2], acc[at + 3]),
                    );
                } else {
                    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32(
                        (acc[at], acc[at + 1], acc[at + 2], acc[at + 3]),
                        (fa[ia], fa[ia + 1], fa[ia + 2], fa[ia + 3]),
                        (fb[ib], fb[ib + 1]),
                        (acc[at], acc[at + 1], acc[at + 2], acc[at + 3]),
                    );
                }
            }
        }
    }

    // Rows and columns are logical input dimensions; stride is rounded up to eight.
    #[instances(false, true)]
    pub unsafe extern "ptx-kernel" fn pack<const BF16: bool>(
        out: *mut u16,
        input: *const f32,
        rows: u32,
        cols: u32,
        padded_rows: u32,
        stride: u32,
    ) {
        let (i, block, t, count, row, col, at): (u32, u32, u32, u32, u32, u32, u32);
        mov.u32(block, ctaid.x);
        mov.u32(t, tid.x);
        mad.lo.u32(i, block, 256, t);
        mul.lo.u32(count, padded_rows, stride);
        let (valid, valid2, done): (pred, pred, pred);
        setp.ge.u32(done, i, count);
        if done {
            ret();
        }
        div.u32(row, i, stride);
        rem.u32(col, i, stride);
        setp.lt.u32(valid, row, rows);
        setp.lt.u32(valid2, col, cols);
        and.pred(valid, valid, valid2);
        let value: f32;
        mov.f32(value, 0.0);
        let ptr: u64;
        if valid {
            mad.lo.u32(at, row, cols, col);
            mad.wide.u32(ptr, at, 4, input);
            ld.global.f32(value, [ptr]);
        }
        let half: b16;
        if BF16 {
            cvt.rn.bf16.f32(half, value);
        } else {
            cvt.rn.f16.f32(half, value);
        }
        mad.wide.u32(ptr, i, 2, out);
        st.global.b16([ptr], half);
    }

    // Fixed FP32 reduction across independent K partitions and removal of N padding.
    pub unsafe extern "ptx-kernel" fn finish(
        out: *mut f32,
        parts: *const f32,
        m: u32,
        n: u32,
        np: u32,
        splits: u32,
    ) {
        let (i, block, t, count, row, col, at, stride, s): (
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
        mov.u32(t, tid.x);
        mad.lo.u32(i, block, 256, t);
        mul.lo.u32(count, m, n);
        let done: pred;
        setp.ge.u32(done, i, count);
        if done {
            ret();
        }
        div.u32(row, i, n);
        rem.u32(col, i, n);
        mad.lo.u32(at, row, np, col);
        mul.lo.u32(stride, m, np);
        let (sum, value): (f32, f32);
        let ptr: u64;
        mov.f32(sum, 0.0);
        mov.u32(s, 0);
        label(next);
        mad.wide.u32(ptr, at, 4, parts);
        ld.global.f32(value, [ptr]);
        add.rn.f32(sum, sum, value);
        add.u32(at, at, stride);
        add.u32(s, s, 1);
        setp.lt.u32(done, s, splits);
        if done {
            bra(next);
        }
        mad.wide.u32(ptr, i, 4, out);
        st.global.f32([ptr], sum);
    }
}
