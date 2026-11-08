//! Static TF32/FP16/BF16 tensor-core GEMM, FP32 accumulations and outputs.
//! 64x64 tiles, four warps, padded shared operands and 32x32 register tiles/warp.
//! NVIDIA PTX ISA 9.7.16.5.7/5.8: <https://docs.nvidia.com/cuda/parallel-thread-execution/>
#[kernite_macros::kernels]
pub mod kernels {
    pub const BM: usize = 64;
    pub const BN: usize = 64;
    pub const BK: usize = 32;
    pub const THREADS: usize = 128;
    const AS: usize = 36;
    const BS: usize = 72;

    fn load(v: f32, p: u64, at: u32) {
        let addr: u64;
        mad.wide.u32(addr, at, 4, p);
        ld.global.f32(v, [addr]);
    }
    fn encode<const FORMAT: usize>(out: b32, lo: f32, hi: f32) {
        if FORMAT == 0 {
            cvt.rna.tf32.f32(out, lo);
        } else if FORMAT == 1 {
            // PTX puts the first input in the upper half of the packed result.
            cvt.rn.f16x2.f32(out, hi, lo);
        } else {
            cvt.rn.bf16x2.f32(out, hi, lo);
        }
    }
    /// FORMAT=0 TF32,1 FP16,2 BF16, all chosen at compile time.
    #[instances((0, false), (0, true), (1, false), (2, false))]
    pub unsafe extern "ptx-kernel" fn gemm<const FORMAT: usize, const FINAL: bool>(
        out: *mut f32,
        a: *const f32,
        b: *const f32,
        m: u32,
        n: u32,
        k: u32,
        chunk: u32,
    ) {
        let sa: shared<[f32; BM * AS], 16>;
        let sb: shared<[f32; BK * BS], 16>;
        let (thread, lane, warp, block, tiles_n, row0, col0, wr, wc, group, small): (
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
        mov.u32(thread, tid.x);
        and.b32(lane, thread, 31);
        shr.u32(warp, thread, 5);
        mov.u32(block, ctaid.x);
        add.u32(tiles_n, n, BN - 1);
        div.u32(tiles_n, tiles_n, BN);
        let (tiles_m, tiles, part, start, end, output_offset): (u32, u32, u32, u32, u32, u32);
        add.u32(tiles_m, m, BM - 1);
        div.u32(tiles_m, tiles_m, BM);
        mul.lo.u32(tiles, tiles_m, tiles_n);
        div.u32(part, block, tiles);
        rem.u32(block, block, tiles);
        mul.lo.u32(start, part, chunk);
        add.u32(end, start, chunk);
        min.u32(end, end, k);
        mul.lo.u32(output_offset, m, n);
        mul.lo.u32(output_offset, output_offset, part);
        mad.wide.u32(out, output_offset, 4, out);
        div.u32(row0, block, tiles_n);
        mul.lo.u32(row0, row0, BM);
        rem.u32(col0, block, tiles_n);
        mul.lo.u32(col0, col0, BN);
        shr.u32(wr, warp, 1);
        mul.lo.u32(wr, wr, 32);
        and.b32(wc, warp, 1);
        mul.lo.u32(wc, wc, 32);
        shr.u32(group, lane, 2);
        and.b32(small, lane, 3);
        let (abase, bbase, addr, at, idx, row, col, rr, cc, tile_k): (
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
        mov.u32(abase, sa);
        mov.u32(bbase, sb);
        let acc: [f32; 32];
        let (af, bf): ([b32; 8], [b32; 8]);
        let (lo, hi): (f32, f32);
        let (valid, valid2, done): (pred, pred, pred);
        for i in 0..32 {
            mov.f32(acc[i], 0.0);
        }
        mov.u32(tile_k, start);
        label(next_tile);
        setp.ge.u32(done, tile_k, end);
        if done {
            bra(finish);
        }
        for j in 0..BM * BK / THREADS {
            add.u32(idx, thread, j * THREADS);
            div.u32(row, idx, BK);
            rem.u32(col, idx, BK);
            add.u32(rr, row0, row);
            add.u32(cc, tile_k, col);
            setp.lt.u32(valid, rr, m);
            setp.lt.u32(valid2, cc, k);
            and.pred(valid, valid, valid2);
            mov.f32(lo, 0.0);
            if valid {
                mad.lo.u32(at, rr, k, cc);
                load(lo, a, at);
            }
            mad.lo.u32(at, row, AS, col);
            mad.lo.u32(addr, at, 4, abase);
            st.shared.f32([addr], lo);
        }
        for j in 0..BK * BN / THREADS {
            add.u32(idx, thread, j * THREADS);
            div.u32(row, idx, BN);
            rem.u32(col, idx, BN);
            add.u32(rr, tile_k, row);
            add.u32(cc, col0, col);
            setp.lt.u32(valid, rr, k);
            setp.lt.u32(valid2, cc, n);
            and.pred(valid, valid, valid2);
            mov.f32(lo, 0.0);
            if valid {
                mad.lo.u32(at, rr, n, cc);
                load(lo, b, at);
            }
            mad.lo.u32(at, row, BS, col);
            mad.lo.u32(addr, at, 4, bbase);
            st.shared.f32([addr], lo);
        }
        bar.sync(0);
        // Kernite expands both the loop and this condition at compile time.
        for kg in 0..4 {
            if FORMAT == 0 || kg < 2 {
                for mt in 0..2 {
                    for j in 0..4 {
                        add.u32(row, wr, group);
                        add.u32(row, row, 16 * mt + 8 * (j % 2));
                        if FORMAT == 0 {
                            add.u32(col, small, 8 * kg + 4 * (j / 2));
                        } else {
                            mad.lo.u32(col, small, 2, 16 * kg + 8 * (j / 2));
                        }
                        mad.lo.u32(at, row, AS, col);
                        mad.lo.u32(addr, at, 4, abase);
                        ld.shared.f32(lo, [addr]);
                        mov.f32(hi, 0.0);
                        if FORMAT != 0 {
                            ld.shared.f32(hi, [addr + 4]);
                        }
                        encode::<FORMAT>(af[mt * 4 + j], lo, hi);
                    }
                }
                for nt in 0..4 {
                    for j in 0..2 {
                        add.u32(col, wc, group);
                        add.u32(col, col, 8 * nt);
                        if FORMAT == 0 {
                            add.u32(row, small, 8 * kg + 4 * j);
                        } else {
                            mad.lo.u32(row, small, 2, 16 * kg + 8 * j);
                        }
                        mad.lo.u32(at, row, BS, col);
                        mad.lo.u32(addr, at, 4, bbase);
                        ld.shared.f32(lo, [addr]);
                        mov.f32(hi, 0.0);
                        if FORMAT != 0 {
                            ld.shared.f32(hi, [addr + 4 * BS]);
                        }
                        encode::<FORMAT>(bf[nt * 2 + j], lo, hi);
                    }
                }
                for mt in 0..2 {
                    for nt in 0..4 {
                        let c = (mt * 4 + nt) * 4;
                        let ia = mt * 4;
                        let ib = nt * 2;
                        if FORMAT == 0 {
                            mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32(
                                (acc[c], acc[c + 1], acc[c + 2], acc[c + 3]),
                                (af[ia], af[ia + 1], af[ia + 2], af[ia + 3]),
                                (bf[ib], bf[ib + 1]),
                                (acc[c], acc[c + 1], acc[c + 2], acc[c + 3]),
                            );
                        } else if FORMAT == 1 {
                            mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32(
                                (acc[c], acc[c + 1], acc[c + 2], acc[c + 3]),
                                (af[ia], af[ia + 1], af[ia + 2], af[ia + 3]),
                                (bf[ib], bf[ib + 1]),
                                (acc[c], acc[c + 1], acc[c + 2], acc[c + 3]),
                            );
                        } else {
                            mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32(
                                (acc[c], acc[c + 1], acc[c + 2], acc[c + 3]),
                                (af[ia], af[ia + 1], af[ia + 2], af[ia + 3]),
                                (bf[ib], bf[ib + 1]),
                                (acc[c], acc[c + 1], acc[c + 2], acc[c + 3]),
                            );
                        }
                    }
                }
            }
        }
        bar.sync(0);
        add.u32(tile_k, tile_k, BK);
        bra(next_tile);
        label(finish);
        if FINAL {
            // Exactly the single-partition finish: +0 plus the MMA result.
            for i in 0..32 {
                add.rn.f32(acc[i], 0.0, acc[i]);
            }
        }
        let ptr: u64;
        for mt in 0..2 {
            for nt in 0..4 {
                for j in 0..4 {
                    add.u32(row, row0, wr);
                    add.u32(row, row, group);
                    add.u32(row, row, 16 * mt + 8 * (j / 2));
                    add.u32(col, col0, wc);
                    mad.lo.u32(col, small, 2, col);
                    add.u32(col, col, 8 * nt + j % 2);
                    setp.lt.u32(valid, row, m);
                    setp.lt.u32(valid2, col, n);
                    and.pred(valid, valid, valid2);
                    if valid {
                        mad.lo.u32(at, row, n, col);
                        mad.wide.u32(ptr, at, 4, out);
                        st.global.f32([ptr], acc[(mt * 4 + nt) * 4 + j]);
                    }
                }
            }
        }
    }
}
