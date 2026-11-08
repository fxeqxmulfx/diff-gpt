//! Bit-preserving 32x32 transpose tile, 256 threads, one padding word per row.
#[kernite_macros::kernels]
pub mod kernels {
    pub unsafe extern "ptx-kernel" fn transpose(
        out: *mut f32,
        input: *const f32,
        rows: u32,
        cols: u32,
    ) {
        let tile: shared<[b32; 32 * 33], 16>;
        let (thread, block, tx, ty, tiles_n, r0, c0, row, col, i, s, base): (
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
        mov.u32(thread, tid.x);
        mov.u32(block, ctaid.x);
        and.b32(tx, thread, 31);
        shr.u32(ty, thread, 5);
        add.u32(tiles_n, cols, 31);
        shr.u32(tiles_n, tiles_n, 5);
        div.u32(r0, block, tiles_n);
        rem.u32(c0, block, tiles_n);
        mul.lo.u32(r0, r0, 32);
        mul.lo.u32(c0, c0, 32);
        mov.u32(base, tile);
        let ptr: u64;
        let bits: b32;
        let (ok, ok2): (pred, pred);
        for j in 0..4 {
            add.u32(row, r0, ty);
            add.u32(row, row, 8 * j);
            add.u32(col, c0, tx);
            setp.lt.u32(ok, row, rows);
            setp.lt.u32(ok2, col, cols);
            and.pred(ok, ok, ok2);
            mov.b32(bits, 0);
            if ok {
                mad.lo.u32(i, row, cols, col);
                mad.wide.u32(ptr, i, 4, input);
                ld.global.b32(bits, [ptr]);
            }
            mad.lo.u32(s, ty, 33, tx);
            add.u32(s, s, 8 * j * 33);
            mad.lo.u32(s, s, 4, base);
            st.shared.b32([s], bits);
        }
        bar.sync(0);
        for j in 0..4 {
            add.u32(row, c0, ty);
            add.u32(row, row, 8 * j);
            add.u32(col, r0, tx);
            setp.lt.u32(ok, row, cols);
            setp.lt.u32(ok2, col, rows);
            and.pred(ok, ok, ok2);
            if ok {
                mad.lo.u32(s, tx, 33, ty);
                add.u32(s, s, 8 * j);
                mad.lo.u32(s, s, 4, base);
                ld.shared.b32(bits, [s]);
                mad.lo.u32(i, row, rows, col);
                mad.wide.u32(ptr, i, 4, out);
                st.global.b32([ptr], bits);
            }
        }
    }
}
