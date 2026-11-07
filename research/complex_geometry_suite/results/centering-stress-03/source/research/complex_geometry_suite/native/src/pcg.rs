//! NumPy-compatible SeedSequence/PCG64, implemented in Rust for frozen data.
//! Algorithms and upstream test vectors: see ../THIRD_PARTY.txt.
fn mixed(value: u32, constant: &mut u32) -> u32 {
    let v = value ^ *constant;
    *constant = constant.wrapping_mul(0x931e8875);
    let v = v.wrapping_mul(*constant);
    v ^ (v >> 16)
}
fn seed_words(entropy: &[u32], count: usize) -> Vec<u32> {
    let mut c = 0x43b0d7e5;
    let mut pool = [0u32; 4];
    for (i, p) in pool.iter_mut().enumerate() {
        *p = mixed(entropy.get(i).copied().unwrap_or(0), &mut c);
    }
    for src in 0..4 {
        for dst in 0..4 {
            if src != dst {
                let y = mixed(pool[src], &mut c);
                let v = 0xca01f9ddu32
                    .wrapping_mul(pool[dst])
                    .wrapping_sub(0x4973f715u32.wrapping_mul(y));
                pool[dst] = v ^ (v >> 16);
            }
        }
    }
    for &x in entropy.iter().skip(4) {
        for p in &mut pool {
            let y = mixed(x, &mut c);
            let v = 0xca01f9ddu32
                .wrapping_mul(*p)
                .wrapping_sub(0x4973f715u32.wrapping_mul(y));
            *p = v ^ (v >> 16);
        }
    }
    let mut c = 0x8b51f9ddu32;
    (0..count)
        .map(|i| {
            let v = pool[i % 4] ^ c;
            c = c.wrapping_mul(0x58f38ded);
            let v = v.wrapping_mul(c);
            v ^ (v >> 16)
        })
        .collect()
}
pub struct Pcg {
    state: u128,
    inc: u128,
    buffer: Option<u32>,
}
impl Pcg {
    pub fn new(seed: &[u32]) -> Self {
        let words = seed_words(seed, 8);
        let u = |i| words[i] as u128 | ((words[i + 1] as u128) << 32);
        let initial = (u(0) << 64) | u(2);
        let inc = (((u(4) << 64) | u(6)) << 1) | 1;
        let mut r = Self {
            state: 0,
            inc,
            buffer: None,
        };
        r.step();
        r.state = r.state.wrapping_add(initial);
        r.step();
        r
    }
    fn step(&mut self) {
        self.state = self
            .state
            .wrapping_mul((2549297995355413924u128 << 64) | 4865540595714422341)
            .wrapping_add(self.inc);
    }
    pub fn next64(&mut self) -> u64 {
        self.step();
        ((self.state >> 64) as u64 ^ self.state as u64).rotate_right((self.state >> 122) as u32)
    }
    fn next32(&mut self) -> u32 {
        if let Some(v) = self.buffer.take() {
            v
        } else {
            let v = self.next64();
            self.buffer = Some((v >> 32) as u32);
            v as u32
        }
    }
    pub fn below(&mut self, n: u32) -> u32 {
        assert!(n > 0);
        if n == 1 {
            return 0;
        }
        let threshold = n.wrapping_neg() % n;
        loop {
            let m = self.next32() as u64 * n as u64;
            if m as u32 >= threshold {
                return (m >> 32) as u32;
            }
        }
    }
    pub fn shuffle<T>(&mut self, values: &mut [T]) {
        for i in (1..values.len()).rev() {
            let max = i as u32;
            let mask = u32::MAX >> max.leading_zeros();
            let j = loop {
                let j = self.next32() & mask;
                if j <= max {
                    break j as usize;
                }
            };
            values.swap(i, j);
        }
    }
}
#[test]
fn upstream_seed_and_pcg_vectors() {
    assert_eq!(
        seed_words(&[42], 4),
        [3444837047, 2669555309, 2046530742, 3581440988]
    );
    assert_eq!(
        seed_words(&[3735928559, 195939070, 229505742, 305419896], 4),
        [3914649087, 576849849, 3593928901, 2229911004]
    );
    let mut r = Pcg::new(&[0xdeadbeaf]);
    assert_eq!(
        (0..4).map(|_| r.next64()).collect::<Vec<_>>(),
        [
            0x60d24054e17a0698,
            0xd5e79d89856e4f12,
            0xd254972fe64bd782,
            0xf1e3072a53c72571
        ]
    );
}
