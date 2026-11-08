//! Scan32 token rows per warp, then visit matching rows in ascending order.
//! Each lane owns one embedding coordinate. The FP32 additions and final tied
//! readout-gradient addition are exactly the parent's sequential subsequence.
#[kernite_macros::kernels]
pub mod kernels {
    pub unsafe extern "ptx-kernel" fn accumulate(
        g: *mut f32,
        dx: *const f32,
        tokens: *const u32,
        width: u32,
        rows: u32,
    ) {
        let (word, col, lane, start, row, token, mask, reversed, bit, next, at): (
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
        mov.u32(word, ctaid.x);
        mov.u32(col, tid.x);
        and.b32(lane, col, 31);
        mov.u32(start, 0);
        let (sum, value): (f32, f32);
        mov.f32(sum, 0.0);
        let ptr: u64;
        let (done, valid, hit): (pred, pred, pred);
        label(chunk);
        setp.ge.u32(done, start, rows);
        if done {
            bra(write);
        }
        add.u32(row, start, lane);
        setp.lt.u32(valid, row, rows);
        mov.u32(token, 0);
        if valid {
            mad.wide.u32(ptr, row, 4, tokens);
            ld.global.u32(token, [ptr]);
        }
        setp.eq.u32(hit, token, word);
        and.pred(hit, hit, valid);
        vote.sync.ballot.b32(mask, hit, 0xffffffff);
        label(matches);
        setp.eq.u32(done, mask, 0);
        if done {
            bra(next_chunk);
        }
        brev.b32(reversed, mask);
        clz.b32(bit, reversed);
        add.u32(row, start, bit);
        mad.lo.u32(at, row, width, col);
        mad.wide.u32(ptr, at, 4, dx);
        ld.global.f32(value, [ptr]);
        add.rn.f32(sum, sum, value);
        sub.u32(next, mask, 1);
        and.b32(mask, mask, next);
        bra(matches);
        label(next_chunk);
        add.u32(start, start, 32);
        bra(chunk);
        label(write);
        mad.lo.u32(at, word, width, col);
        mad.wide.u32(ptr, at, 4, g);
        ld.global.f32(value, [ptr]);
        add.rn.f32(sum, value, sum);
        st.global.f32([ptr], sum);
    }
}
