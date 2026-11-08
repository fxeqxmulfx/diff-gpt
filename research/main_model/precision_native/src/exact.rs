//! Exact dyadic dot oracle and exact rational acceptance comparisons.
use num_bigint::BigInt;
use num_traits::{Signed, ToPrimitive, Zero};

fn parts(v: f32) -> (BigInt, usize) {
    assert!(v.is_finite());
    let bits = v.to_bits();
    let e = (bits >> 23) & 255;
    let m = if e == 0 {
        bits & 0x7fffff
    } else {
        (bits & 0x7fffff) | 0x800000
    };
    let n = BigInt::from(m);
    // Value is signed significand * 2^(shift-149).
    (
        if bits >> 31 == 0 { n } else { -n },
        if e == 0 { 0 } else { (e - 1) as usize },
    )
}
pub fn dot(a: &[f32], b: &[f32]) -> BigInt {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(BigInt::zero(), |sum, (&x, &y)| {
        let (xm, xe) = parts(x);
        let (ym, ye) = parts(y);
        sum + ((xm * ym) << (xe + ye))
    })
}
fn f64_units(v: f64) -> BigInt {
    assert!(v.is_finite());
    let bits = v.to_bits();
    let e = (bits >> 52) & 2047;
    let m = if e == 0 {
        bits & 0xfffffffffffff
    } else {
        (bits & 0xfffffffffffff) | 0x10000000000000
    };
    let n = BigInt::from(m) << if e == 0 { 0 } else { (e - 1) as usize };
    if bits >> 63 == 0 { n } else { -n }
}
// Dot units are2^-298; FP64 comparison units are2^-1074.
pub fn assess(exact: &BigInt, actual: f64) -> (bool, bool, f64, f64) {
    if !actual.is_finite() {
        return (false, false, f64::INFINITY, f64::INFINITY);
    }
    let target = exact << 776usize;
    let error = (f64_units(actual) - &target).abs();
    let one = BigInt::from(1u32) << 1074usize;
    let forward_scale = target.abs().max(one.clone());
    let forward = &error * 50000u32 <= forward_scale;
    let backward = &error * 2000u32 * 1_000_000_000_000u64
        <= target.abs() * 1_000_000_000_000u64 + one * 2000u32;
    // Convert smaller scaled integers for display, without overflow at2^1074.
    let display_error = (&error >> 776usize).to_f64().unwrap_or(f64::INFINITY) * 2f64.powi(-298);
    let display_exact = exact.to_f64().unwrap() * 2f64.powi(-298);
    (forward, backward, display_error, display_exact)
}

#[test]
fn exact_cancellation_subnormal_and_rational_budgets() {
    let n = dot(&[1., 1. / 4096., -1.], &[1., 1., 1.]);
    assert_eq!(assess(&n, 1. / 4096.), (true, true, 0., 1. / 4096.));
    assert!(!assess(&n, 0.).0);
    assert!(!assess(&n, 0.).1);
    assert_eq!(
        dot(&[f32::from_bits(1)], &[f32::from_bits(1)]),
        BigInt::from(1)
    );
    let zero = dot(&[0., -0., 1., -1.], &[1., 1., 1., 1.]);
    assert!(zero.is_zero());
    assert!(assess(&zero, -0.).0);
    assert!(!assess(&zero, 2e-12).1);
    assert!(assess(&zero, 0.5e-12).1);
    assert!(!assess(&zero, f64::NAN).0);
}
