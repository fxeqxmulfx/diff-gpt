//! Python's `random`, as far as microgpt draws on it: the same Mersenne Twister, seeded the
//! same way, so that the weights start where Python's do and the names come in its order.

/// The words of the twister's state.
const N: usize = 624;
/// The offset of the word each is mixed with.
const M: usize = 397;

/// Python's `random.Random`.
pub struct Random {
    state: [u32; N],
    index: usize,
    /// The second of the pair of normal deviates `gauss` draws at a time, until it is asked for.
    gauss_next: Option<f64>,
}

impl Random {
    pub fn words(seed: &[u32]) -> Self {
        let mut out = Self::new(0);
        out.init_by_array(seed);
        out
    }
    pub fn torch_seed(seed: u32) -> Self {
        let mut out = Self::new(0);
        out.init_genrand(seed);
        out
    }
    pub fn sample(&mut self, n: usize, k: usize) -> Vec<usize> {
        let mut set_size = 21;
        if k > 5 {
            let mut v = 4;
            while v < 3 * k {
                v *= 4;
            }
            set_size += v;
        }
        let mut result = Vec::new();
        if n <= set_size {
            let mut pool: Vec<_> = (0..n).collect();
            for i in 0..k {
                let j = self.below((n - i) as u32) as usize;
                result.push(pool[j]);
                pool[j] = pool[n - i - 1];
            }
        } else {
            for _ in 0..k {
                loop {
                    let j = self.below(n as u32) as usize;
                    if !result.contains(&j) {
                        result.push(j);
                        break;
                    }
                }
            }
        }
        result
    }
    /// `random.seed(seed)`, for a seed below 2^32.
    pub fn new(seed: u32) -> Random {
        let mut r = Random {
            state: [0; N],
            index: N,
            gauss_next: None,
        };
        r.init_by_array(&[seed]);
        r
    }

    fn init_genrand(&mut self, seed: u32) {
        self.state[0] = seed;
        for i in 1..N {
            let prev = self.state[i - 1];
            self.state[i] = 1812433253u32
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(i as u32);
        }
        self.index = N;
    }

    fn init_by_array(&mut self, key: &[u32]) {
        self.init_genrand(19650218);
        let mt = &mut self.state;
        let (mut i, mut j) = (1, 0);
        for _ in 0..N.max(key.len()) {
            let prev = mt[i - 1];
            mt[i] = (mt[i] ^ (prev ^ (prev >> 30)).wrapping_mul(1664525))
                .wrapping_add(key[j])
                .wrapping_add(j as u32);
            (i, j) = (i + 1, j + 1);
            if i >= N {
                mt[0] = mt[N - 1];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
        }
        for _ in 0..N - 1 {
            let prev = mt[i - 1];
            mt[i] = (mt[i] ^ (prev ^ (prev >> 30)).wrapping_mul(1566083941)).wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                mt[0] = mt[N - 1];
                i = 1;
            }
        }
        mt[0] = 0x8000_0000;
    }

    /// The next 32 bits.
    pub fn next_u32(&mut self) -> u32 {
        if self.index >= N {
            let mt = &mut self.state;
            for k in 0..N {
                let y = (mt[k] & 0x8000_0000) | (mt[(k + 1) % N] & 0x7fff_ffff);
                mt[k] = mt[(k + M) % N] ^ (y >> 1) ^ if y & 1 == 1 { 0x9908_b0df } else { 0 };
            }
            self.index = 0;
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    /// `random.random()`: a float in [0, 1) of 53 random bits.
    pub fn random(&mut self) -> f64 {
        let (a, b) = (self.next_u32() >> 5, self.next_u32() >> 6);
        (a as f64 * 67108864.0 + b as f64) * (1.0 / 9007199254740992.0)
    }

    /// `random.getrandbits(k)`, for `k` from 1 to 32.
    pub fn bits(&mut self, k: u32) -> u32 {
        assert!(
            (1..=32).contains(&k),
            "kernite's port draws 1 to 32 bits, not {k}"
        );
        self.next_u32() >> (32 - k)
    }

    /// `random._randbelow(n)`: below `n`, by drawing as many bits as `n` has until one is.
    pub fn below(&mut self, n: u32) -> u32 {
        let k = u32::BITS - n.leading_zeros();
        loop {
            let r = self.bits(k);
            if r < n {
                return r;
            }
        }
    }

    /// `random.shuffle(xs)`.
    pub fn shuffle<T>(&mut self, xs: &mut [T]) {
        for i in (1..xs.len()).rev() {
            let j = self.below(i as u32 + 1) as usize;
            xs.swap(i, j);
        }
    }

    /// `random.gauss(mu, sigma)`: by the pair of deviates the Box-Muller transform makes of two
    /// uniform ones, the second kept for the next call.
    pub fn gauss(&mut self, mu: f64, sigma: f64) -> f64 {
        let z = match self.gauss_next.take() {
            Some(z) => z,
            None => {
                let x2pi = self.random() * std::f64::consts::TAU;
                let g2rad = (-2.0 * (1.0 - self.random()).ln()).sqrt();
                self.gauss_next = Some(x2pi.sin() * g2rad);
                x2pi.cos() * g2rad
            }
        };
        mu + z * sigma
    }
}
