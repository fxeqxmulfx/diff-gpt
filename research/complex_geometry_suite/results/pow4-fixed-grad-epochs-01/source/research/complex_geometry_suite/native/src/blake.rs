//! One-block BLAKE2b for the frozen two-register memory partition.
//! RFC 7693 compression, with an 8-byte digest and personalization.
const IV: [u64; 8] = [
    0x6a09e667f3bcc908,
    0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b,
    0xa54ff53a5f1d36f1,
    0x510e527fade682d1,
    0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b,
    0x5be0cd19137e2179,
];
const SIGMA: [[usize; 16]; 10] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
    [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
];
fn g(v: &mut [u64; 16], a: usize, b: usize, c: usize, d: usize, x: u64, y: u64) {
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(x);
    v[d] = (v[d] ^ v[a]).rotate_right(32);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(24);
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(y);
    v[d] = (v[d] ^ v[a]).rotate_right(16);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(63);
}
pub fn digest8(bytes: &[u8], person: &[u8]) -> [u8; 8] {
    assert!(bytes.len() <= 128 && person.len() <= 16);
    let mut h = IV;
    h[0] ^= 0x01010008;
    let mut p = [0u8; 16];
    p[..person.len()].copy_from_slice(person);
    h[6] ^= u64::from_le_bytes(p[..8].try_into().unwrap());
    h[7] ^= u64::from_le_bytes(p[8..].try_into().unwrap());
    let mut block = [0u8; 128];
    block[..bytes.len()].copy_from_slice(bytes);
    let m: Vec<_> = block
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| u64::from_le_bytes(*c))
        .collect();
    let mut v = [0u64; 16];
    v[..8].copy_from_slice(&h);
    v[8..].copy_from_slice(&IV);
    v[12] ^= bytes.len() as u64;
    v[14] = !v[14];
    for r in 0..12 {
        let s = SIGMA[r % 10];
        for j in 0..4 {
            g(
                &mut v,
                j,
                j + 4,
                j + 8,
                j + 12,
                m[s[2 * j]],
                m[s[2 * j + 1]],
            );
        }
        for j in 0..4 {
            g(
                &mut v,
                j,
                4 + (j + 1) % 4,
                8 + (j + 2) % 4,
                12 + (j + 3) % 4,
                m[s[8 + 2 * j]],
                m[s[9 + 2 * j]],
            );
        }
    }
    (h[0] ^ v[0] ^ v[8]).to_le_bytes()
}
#[test]
fn empty_blake2b8() {
    assert_eq!(
        digest8(b"", b""),
        [0xe4, 0xa6, 0xa0, 0x57, 0x74, 0x79, 0xb2, 0xb4]
    );
}
