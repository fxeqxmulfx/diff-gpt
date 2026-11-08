//! Native version-4 DiffGPT level-grid tokenization for finite time series.
//! Exact CPU rationals select scales and levels; binary64 rounds only decoding
//! and the reference's fractional prefix seeds. No Python or CUDA is needed.
#![forbid(unsafe_code)]

use num_bigint::BigInt;
use num_rational::BigRational as Rational;
use num_traits::{Signed, ToPrimitive, Zero};

pub type Result<T> = std::result::Result<T, String>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tokenizer {
    Lgt,
    Mash,
    MashHalf,
}

#[derive(Clone, Debug)]
pub struct Difference {
    lags: Vec<usize>,
    coefficients: Vec<BigInt>,
}

impl Difference {
    pub fn new(mut lags: Vec<usize>) -> Result<Self> {
        if lags.contains(&0) || lags.len() > 61 {
            return Err("difference factors need positive lags and at most 61 factors".into());
        }
        lags.sort_unstable();
        let mut coefficients = vec![BigInt::from(1)];
        for &lag in &lags {
            let length = coefficients
                .len()
                .checked_add(lag)
                .ok_or("degree overflow")?;
            let mut next = vec![BigInt::zero(); length];
            for (i, v) in coefficients.iter().enumerate() {
                next[i] += v;
                next[i + lag] -= v;
            }
            coefficients = next;
        }
        Ok(Self { lags, coefficients })
    }

    pub fn order(order: usize) -> Result<Self> {
        if order > 61 {
            return Err("at most 61 difference factors are supported".into());
        }
        Self::new(vec![1; order])
    }

    pub fn degree(&self) -> usize {
        self.coefficients.len() - 1
    }

    pub fn lags(&self) -> &[usize] {
        &self.lags
    }

    fn headroom(&self) -> BigInt {
        if self.lags.is_empty() {
            BigInt::zero()
        } else {
            self.coefficients.iter().map(Signed::abs).sum()
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Domain {
    pub low: f64,
    pub high: f64,
}

impl Domain {
    pub fn symmetric(bound: f64) -> Self {
        Self {
            low: -bound,
            high: bound,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Grid {
    pub scale: Vec<f64>,
    pub zero: Vec<i64>,
}

/// Row-major storage with an explicit feature count, including empty prefixes.
#[derive(Clone, Debug, PartialEq)]
pub struct Matrix<T> {
    columns: usize,
    values: Vec<T>,
}

impl<T> Matrix<T> {
    pub fn new(columns: usize, values: Vec<T>) -> Result<Self> {
        if columns == 0 || !values.len().is_multiple_of(columns) {
            return Err("matrix needs positive columns and complete rows".into());
        }
        Ok(Self { columns, values })
    }
    pub fn rows(&self) -> usize {
        self.values.len() / self.columns
    }
    pub fn columns(&self) -> usize {
        self.columns
    }
    pub fn values(&self) -> &[T] {
        &self.values
    }
}

#[derive(Clone, Debug)]
pub struct Encoded {
    pub prefix: Matrix<f64>,
    pub grid: Grid,
    pub tokens: Matrix<i64>,
    /// Explicit counterpart of the reference's TokenClippingWarning count.
    pub clipped_tokens: usize,
}

fn rational(x: f64) -> Result<Rational> {
    Rational::from_float(x).ok_or_else(|| "nonfinite binary64 value".into())
}
fn integer(x: i64) -> Rational {
    Rational::from_integer(x.into())
}
fn floor(q: &Rational) -> BigInt {
    q.floor().to_integer()
}
fn ceil(q: &Rational) -> BigInt {
    q.ceil().to_integer()
}
fn half() -> Rational {
    Rational::new(1.into(), 2.into())
}

fn vocabulary(vocab: i64) -> Result<()> {
    if !(4..(1_i64 << 62)).contains(&vocab) || vocab % 2 != 0 {
        Err("vocabulary must be even, >=4 and <2^62".into())
    } else {
        Ok(())
    }
}

/// Greatest nonnegative finite binary64 <= q. Bit order is monotone here.
/// An exact binary search avoids assuming a rational-to-float rounding mode.
fn below(q: &Rational) -> Result<f64> {
    if q.is_negative() {
        return Err("negative scale endpoint".into());
    }
    let mut low = 0_u64;
    let mut high = f64::MAX.to_bits();
    while low < high {
        let mid = low + (high - low).div_ceil(2);
        if rational(f64::from_bits(mid))? <= *q {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    Ok(f64::from_bits(low))
}

fn fits(
    s: f64,
    lo: &Rational,
    hi: &Rational,
    room: i64,
    order0: bool,
    shift: &Rational,
) -> Result<bool> {
    let scale = rational(s)?;
    Ok(if order0 {
        floor(&(&scale * hi)) - floor(&(&scale * lo)) < BigInt::from(room)
    } else {
        ceil(&(&scale * hi - shift)) - floor(&(&scale * lo - shift)) <= BigInt::from(room)
    })
}

fn max_scale(
    lo: &Rational,
    hi: &Rational,
    room: i64,
    order0: bool,
    shift: &Rational,
) -> Result<f64> {
    let width = hi - lo;
    let pivot = integer(room) * lo / &width;
    let mut best = below(&(integer(room - 1) / &width))?;
    for n in [floor(&(&pivot - shift)), ceil(&(&pivot - shift))] {
        let p = Rational::from_integer(n) + shift;
        let mut ends = Vec::new();
        if lo.is_negative() {
            ends.push(&p / lo);
        }
        if hi.is_positive() {
            ends.push((&p + integer(room)) / hi);
        }
        let end = below(&ends.into_iter().min().ok_or("empty scale interval")?)?;
        for s in [end, if end == 0. { 0. } else { end.next_down() }] {
            if s > 0. && fits(s, lo, hi, room, order0, shift)? {
                best = best.max(s);
                break;
            }
        }
    }
    if !best.is_finite() || best <= 0. {
        return Err("domain has no positive finite scale".into());
    }
    Ok(best)
}

pub fn level_grid(
    vocab: i64,
    domains: &[Domain],
    operators: &[Difference],
    tokenizer: Tokenizer,
) -> Result<Grid> {
    vocabulary(vocab)?;
    if domains.len() != operators.len() || domains.is_empty() {
        return Err("one nonempty domain and operator per column is required".into());
    }
    let mut grid = Grid {
        scale: Vec::new(),
        zero: Vec::new(),
    };
    let point_width = rational(2. * f64::EPSILON)?;
    for (domain, op) in domains.iter().zip(operators) {
        let lo = rational(domain.low)?;
        let hi = rational(domain.high)?;
        if lo > hi {
            return Err("domain low exceeds high".into());
        }
        let headroom = op
            .headroom()
            .to_i64()
            .ok_or("headroom exceeds supported vocabulary")?;
        if vocab <= headroom {
            return Err("vocabulary does not leave room for quantization".into());
        }
        let order0 = op.lags.is_empty();
        let shifted = tokenizer == Tokenizer::MashHalf && !order0;
        let shift = if shifted { half() } else { Rational::zero() };
        let room = vocab - headroom + i64::from(shifted);
        let bound = (-&lo).max(hi.clone());
        let s = if &hi - &lo > point_width {
            max_scale(&lo, &hi, room, order0, &shift)?
        } else if bound > &point_width * half() {
            max_scale(&(-&bound), &bound, room, order0, &shift)?
        } else {
            (1_u64 << 20) as f64
        };
        let scale = rational(s)?;
        let (least, most) = if order0 {
            (
                -floor(&(&scale * &lo)),
                BigInt::from(vocab - 1) - floor(&(&scale * &hi)),
            )
        } else {
            (
                BigInt::from(headroom / 2 - i64::from(shifted)) - floor(&(&scale * &lo - &shift)),
                BigInt::from(vocab - headroom / 2) - ceil(&(&scale * &hi - &shift)),
            )
        };
        let middle = BigInt::from(vocab / 2) - floor(&(&scale * (&lo + &hi) * half() + half()));
        let z = middle.max(least).min(most);
        if z.abs() >= BigInt::from(1_u64 << 62) {
            return Err("zero point exceeds 2^62".into());
        }
        grid.scale.push(s);
        grid.zero.push(z.to_i64().ok_or("zero point overflow")?);
    }
    Ok(grid)
}

fn level(x: f64, scale: f64, fraction: f64, factors: usize) -> Result<i64> {
    if factors > 61 || !(0. ..1.).contains(&fraction) || !scale.is_finite() || scale <= 0. {
        return Err("invalid level arguments".into());
    }
    let bits = 53.min(62 - factors);
    let product = scale * x;
    if !x.is_finite() || !product.is_finite() || product.abs() >= (1_u64 << bits) as f64 {
        return Err("level grid exceeds its exact binary64/int64 range".into());
    }
    floor(&(rational(scale)? * rational(x)? - rational(fraction)? + half()))
        .to_i64()
        .ok_or_else(|| "level overflow".into())
}

mod codec;
pub use codec::{decode, encode};

#[cfg(test)]
mod tests;
