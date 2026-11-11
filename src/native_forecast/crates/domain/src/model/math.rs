//! The arithmetic of the model beside the matrix products: the exponential,
//! the normalizations, the rotary positions, the causal softmax, the SwiGLU
//! gate and the exclusive projection, each with the map that takes a
//! gradient back through it.

use super::ROPE_BASE;

/// The epsilon under the root of an RMS normalization: that of float32, as
/// PyTorch's `rms_norm` takes when none is given, and as `GPT` sets every
/// site (`use_native_rms_epsilon`).
pub const RMS_EPSILON: f32 = f32::EPSILON;

/// The least norm a value is divided by in the exclusive projection:
/// `XSA_EPS` of `centered_xsa.py`.
pub const UNIT_EPSILON: f32 = 1.0e-12;

/// The number of partial sums a sum is taken in, which lets it run in one
/// vector register.
const LANES: usize = 8;

/// The sum of the products of two vectors, in [`LANES`] partial sums.
#[must_use]
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let (a_chunks, a_rest) = a.as_chunks::<LANES>();
    let (b_chunks, b_rest) = b.as_chunks::<LANES>();
    let mut lanes = [0.0_f32; LANES];
    for (a, b) in a_chunks.iter().zip(b_chunks) {
        for ((lane, a), b) in lanes.iter_mut().zip(a).zip(b) {
            *lane += a * b;
        }
    }
    let rest: f32 = a_rest.iter().zip(b_rest).map(|(a, b)| a * b).sum();
    lanes.iter().sum::<f32>() + rest
}

/// The sum of a vector, in [`LANES`] partial sums.
#[must_use]
#[inline]
pub fn sum(values: &[f32]) -> f32 {
    let (chunks, rest) = values.as_chunks::<LANES>();
    let mut lanes = [0.0_f32; LANES];
    for chunk in chunks {
        for (lane, value) in lanes.iter_mut().zip(chunk) {
            *lane += value;
        }
    }
    lanes.iter().sum::<f32>() + rest.iter().sum::<f32>()
}

/// The greatest float of a vector; −∞ for none.
#[must_use]
#[inline]
pub fn highest(values: &[f32]) -> f32 {
    let (chunks, rest) = values.as_chunks::<LANES>();
    let mut lanes = [f32::NEG_INFINITY; LANES];
    for chunk in chunks {
        for (lane, value) in lanes.iter_mut().zip(chunk) {
            *lane = if *value > *lane { *value } else { *lane };
        }
    }
    let lanes = lanes.iter().chain(rest);
    lanes.fold(f32::NEG_INFINITY, |a, b| a.max(*b))
}

/// `e^x` to a few units in the last place, in arithmetic that a compiler
/// turns into vector instructions: `2^n · e^r` with `n = round(x / ln 2)`
/// and a polynomial of degree 6 in `r`. An `x` below −87 counts as −87, and
/// one above 88 as 88.
#[must_use]
#[inline]
pub fn exp(x: f32) -> f32 {
    const LN_2_HIGH: f32 = 0.693_145_75;
    const LN_2_LOW: f32 = 1.428_606_8e-6;
    let x = x.clamp(-87.0, 88.0);
    let n = (x * core::f32::consts::LOG2_E).round_ties_even();
    let r = n.mul_add(-LN_2_LOW, n.mul_add(-LN_2_HIGH, x));
    let p = r.mul_add(1.0 / 720.0, 1.0 / 120.0);
    let p = r.mul_add(p, 1.0 / 24.0);
    let p = r.mul_add(p, 1.0 / 6.0);
    let p = r.mul_add(p, 0.5);
    let p = r.mul_add(p, 1.0);
    let p = r.mul_add(p, 1.0);
    p * f32::from_bits(((n as i32 + 127) as u32) << 23)
}

/// The logistic function `1 / (1 + e^(−x))`.
#[must_use]
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + exp(-x))
}

/// Normalizes a vector to unit RMS in place and returns what it was
/// multiplied by, `1 / √(mean x² + ε)`.
#[inline]
pub fn normalize(vector: &mut [f32]) -> f32 {
    let energy = dot(vector, vector) / vector.len() as f32;
    let inverse = 1.0 / (energy + RMS_EPSILON).sqrt();
    vector.iter_mut().for_each(|value| *value *= inverse);
    inverse
}

/// Normalizes each row of `width` floats of `x` to unit RMS, into `y`, and
/// keeps in `inverse` what each row was multiplied by.
pub fn rms(x: &[f32], width: usize, y: &mut [f32], inverse: &mut [f32]) {
    for ((row, out), inverse) in x.chunks(width).zip(y.chunks_mut(width)).zip(inverse) {
        out.copy_from_slice(row);
        *inverse = normalize(out);
    }
}

/// The gradient at the input of [`rms`], added to `dx`, from the gradient
/// `dy` at its output `y`: `(dy − y · mean(dy ⊙ y)) / √(mean x² + ε)`.
pub fn rms_back(dy: &[f32], y: &[f32], inverse: &[f32], width: usize, dx: &mut [f32]) {
    let rows = dy.chunks(width).zip(y.chunks(width)).zip(inverse);
    for (((dy, y), inverse), dx) in rows.zip(dx.chunks_mut(width)) {
        let along = dot(dy, y) / width as f32;
        for ((dx, dy), y) in dx.iter_mut().zip(dy).zip(y) {
            *dx += (dy - y * along) * inverse;
        }
    }
}

/// The cosines and the sines of the rotary angles: for each of `positions`
/// positions, `head / 2` of each, pair `i` turning by
/// `position · base^(−2i / head)` (`rope_tables` of `rope.py`).
#[must_use]
pub fn rope_tables(head: usize, positions: usize) -> (Vec<f32>, Vec<f32>) {
    let pairs = head / 2;
    let mut cos = Vec::with_capacity(positions * pairs);
    let mut sin = Vec::with_capacity(positions * pairs);
    for position in 0..positions {
        for pair in 0..pairs {
            let rate = ROPE_BASE.powf(-((2 * pair) as f64) / head as f64);
            let angle = position as f64 * rate;
            cos.push(angle.cos() as f32);
            sin.push(angle.sin() as f32);
        }
    }
    (cos, sin)
}

/// Rotates the pairs `(i, i + head/2)` of a vector by the angles of its
/// position, or back by them if `back`: the map that takes a gradient
/// through the rotation is the rotation back.
#[inline]
pub fn rotate(vector: &mut [f32], cos: &[f32], sin: &[f32], back: bool) {
    let sign = if back { -1.0 } else { 1.0 };
    let (first, second) = vector.split_at_mut(cos.len());
    for (((a, b), cos), sin) in first.iter_mut().zip(second).zip(cos).zip(sin) {
        let sin = sign * *sin;
        (*a, *b) = (*a * cos - *b * sin, *a * sin + *b * cos);
    }
}

/// The gradient at a vector that was normalized ([`normalize`]) and then
/// rotated ([`rotate`]) into `y`, set in `dx`, from the gradient `dy` at
/// `y`. A rotation keeps inner products, so the part of `dy` along `y` is
/// taken off before the turn back.
#[inline]
pub fn normalize_back(
    dy: &[f32],
    y: &[f32],
    inverse: f32,
    cos: &[f32],
    sin: &[f32],
    dx: &mut [f32],
) {
    let along = dot(dy, y) / y.len() as f32;
    for ((dx, dy), y) in dx.iter_mut().zip(dy).zip(y) {
        *dx = (dy - y * along) * inverse;
    }
    rotate(dx, cos, sin, true);
}

/// The causal softmax of the scores of one head times `scale`, in place:
/// row `t` of `length` floats becomes the weights of positions `0..=t`, and
/// zeros after.
pub fn causal_softmax(scores: &mut [f32], length: usize, scale: f32) {
    for (row, scores) in scores.chunks_mut(length).enumerate() {
        let (seen, hidden) = scores.split_at_mut(row + 1);
        let top = highest(seen);
        seen.iter_mut()
            .for_each(|score| *score = exp((*score - top) * scale));
        let share = 1.0 / sum(seen);
        seen.iter_mut().for_each(|weight| *weight *= share);
        hidden.fill(0.0);
    }
}

/// The gradient at the scores, in place of the gradient `dp` at the weights
/// `p` of [`causal_softmax`]: `scale · p ⊙ (dp − Σ p ⊙ dp)` over the
/// positions a row sees, and zeros after.
pub fn causal_softmax_back(dp: &mut [f32], p: &[f32], length: usize, scale: f32) {
    for (row, (dp, p)) in dp.chunks_mut(length).zip(p.chunks(length)).enumerate() {
        let (seen, hidden) = dp.split_at_mut(row + 1);
        let mean = dot(seen, &p[..=row]);
        for (d, p) in seen.iter_mut().zip(p) {
            *d = scale * p * (*d - mean);
        }
        hidden.fill(0.0);
    }
}

/// The softmax of a row of logits, in place, and the logarithm of the
/// weight of `target`.
pub fn softmax(logits: &mut [f32], target: usize) -> f32 {
    let top = highest(logits);
    let at_target = logits[target] - top;
    logits
        .iter_mut()
        .for_each(|logit| *logit = exp(*logit - top));
    let total = sum(logits);
    let share = 1.0 / total;
    logits.iter_mut().for_each(|weight| *weight *= share);
    at_target - total.ln()
}

/// SwiGLU's gate: `silu(gate) ⊙ value` for the two halves of each row of
/// `2 · hidden` floats of `expanded`, into `gated`.
pub fn swiglu(expanded: &[f32], hidden: usize, gated: &mut [f32]) {
    for (row, gated) in expanded.chunks(2 * hidden).zip(gated.chunks_mut(hidden)) {
        let (gate, value) = row.split_at(hidden);
        for ((out, gate), value) in gated.iter_mut().zip(gate).zip(value) {
            *out = gate * sigmoid(*gate) * value;
        }
    }
}

/// The gradient at the input of [`swiglu`], set in `d_expanded`, from the
/// gradient `d_gated` at its output.
pub fn swiglu_back(d_gated: &[f32], expanded: &[f32], hidden: usize, d_expanded: &mut [f32]) {
    let rows = expanded.chunks(2 * hidden).zip(d_gated.chunks(hidden));
    for ((row, d_gated), d_row) in rows.zip(d_expanded.chunks_mut(2 * hidden)) {
        let (gate, value) = row.split_at(hidden);
        let (d_gate, d_value) = d_row.split_at_mut(hidden);
        let inputs = gate.iter().zip(value).zip(d_gated);
        for (((gate, value), d_gated), (d_gate, d_value)) in
            inputs.zip(d_gate.iter_mut().zip(d_value))
        {
            let open = sigmoid(*gate);
            *d_value = d_gated * gate * open;
            // silu'(x) = σ(x) (1 + x (1 − σ(x))).
            *d_gate = d_gated * value * open * (1.0 + gate * (1.0 - open));
        }
    }
}

/// The exclusive projection of one head at one position, into `out`: the
/// attended vector less its component along the token's own value, as
/// `centered_projection` of `centered_xsa.py` evaluates it. With
/// `δ = attended − own` and `u = own / max(|own|, ε)`, it is
/// `δ − (δ·u) u + r · own`, where `r` is 0 for an own value above ε and
/// `1 − (|own| / ε)²` below. Returns `|own|` and `δ·u`.
#[inline]
pub fn exclude(attended: &[f32], own: &[f32], out: &mut [f32]) -> (f32, f32) {
    let norm = dot(own, own).sqrt();
    let divisor = norm.max(UNIT_EPSILON);
    // δ first, so that an attended vector that is the own value gives zero.
    for ((out, a), v) in out.iter_mut().zip(attended).zip(own) {
        *out = a - v;
    }
    let along = dot(out, own) / divisor;
    let back = remainder(norm) - along / divisor;
    for (out, v) in out.iter_mut().zip(own) {
        *out += back * v;
    }
    (norm, along)
}

/// What the clipped unit vector leaves of an own value of this norm.
#[inline]
fn remainder(norm: f32) -> f32 {
    if norm > UNIT_EPSILON {
        0.0
    } else {
        1.0 - (norm / UNIT_EPSILON).powi(2)
    }
}

/// The gradients at the inputs of [`exclude`], `d_attended` set and `d_own`
/// added to, from the gradient `d_out` at its output; `kept` is what
/// [`exclude`] returned.
#[inline]
pub fn exclude_back(
    d_out: &[f32],
    attended: &[f32],
    own: &[f32],
    kept: (f32, f32),
    d_attended: &mut [f32],
    d_own: &mut [f32],
) {
    let (norm, along) = kept;
    let divisor = norm.max(UNIT_EPSILON);
    let remainder = remainder(norm);
    let clipped = norm <= UNIT_EPSILON;
    // The component of the gradient along u.
    let d_along = dot(d_out, own) / divisor;
    let rows = d_out.iter().zip(attended).zip(own);
    for (((d_out, a), v), (d_attended, d_own)) in rows.zip(d_attended.iter_mut().zip(d_own)) {
        let unit = v / divisor;
        // At δ: the projection is its own transpose.
        let at_delta = d_out - d_along * unit;
        // At u: −((d·u) δ + (δ·u) d), whose component along u is −2 (δ·u) (d·u).
        let at_unit = -(d_along * (a - v) + along * d_out);
        let through_unit = if clipped {
            // u = own / ε, and the remainder r · own with r = 1 − |own|² / ε².
            at_unit / divisor + remainder * d_out - 2.0 * unit * d_along
        } else {
            // u = own / |own|: the part of the gradient across u, by |own|.
            (at_unit + 2.0 * along * d_along * unit) / norm
        };
        *d_attended = at_delta;
        *d_own += through_unit - at_delta;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_exponential_is_accurate_and_bounded() {
        let mut worst = 0.0_f64;
        for step in -8700..=8800 {
            let x = step as f32 / 100.0;
            let exact = f64::from(x).exp();
            worst = worst.max(((f64::from(exp(x)) - exact) / exact).abs());
        }
        assert!(worst < 3e-7, "relative error {worst}");
        assert_eq!(exp(0.0), 1.0);
        assert!(exp(-1000.0) > 0.0 && exp(-1000.0) < 1e-37);
        assert!(exp(1000.0).is_finite());
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-7 && sigmoid(-200.0) >= 0.0);
        assert!((sigmoid(40.0) - 1.0).abs() < 1e-7);
    }

    #[test]
    fn sums_in_lanes_are_the_sums() {
        let values: Vec<f32> = (0..37).map(|index| (index as f32 * 0.7).sin()).collect();
        let plain: f32 = values.iter().sum();
        assert!((sum(&values) - plain).abs() < 1e-5);
        let squares: f32 = values.iter().map(|v| v * v).sum();
        assert!((dot(&values, &values) - squares).abs() < 1e-5);
        let top = values.iter().fold(f32::NEG_INFINITY, |a, b| a.max(*b));
        assert_eq!(highest(&values), top);
        assert_eq!(highest(&values[..3]), values[1].max(values[2]));
        assert_eq!(
            (sum(&[]), dot(&[], &[]), highest(&[])),
            (0.0, 0.0, f32::NEG_INFINITY)
        );
    }

    /// The derivative of `loss` along `direction` at `x`, by central
    /// differences.
    fn slope(x: &[f32], direction: &[f32], loss: impl Fn(&[f32]) -> f64) -> f64 {
        let step = 1e-3_f32;
        let moved = |sign: f32| -> Vec<f32> {
            x.iter()
                .zip(direction)
                .map(|(x, d)| x + sign * step * d)
                .collect()
        };
        (loss(&moved(1.0)) - loss(&moved(-1.0))) / (2.0 * f64::from(step))
    }

    fn floats(count: usize, seed: u64) -> Vec<f32> {
        let mut rng = crate::random::Rng::from_key(&[seed]);
        (0..count).map(|_| rng.normal() as f32).collect()
    }

    fn inner(a: &[f32], b: &[f32]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(a, b)| f64::from(*a) * f64::from(*b))
            .sum()
    }

    #[test]
    fn the_rms_normalization_and_its_gradient_agree() {
        let (x, weights, direction) = (floats(24, 1), floats(24, 2), floats(24, 3));
        let forward = |x: &[f32]| {
            let (mut y, mut inverse) = (vec![0.0; 24], vec![0.0; 3]);
            rms(x, 8, &mut y, &mut inverse);
            (y, inverse)
        };
        let (y, inverse) = forward(&x);
        for row in y.chunks(8) {
            assert!((dot(row, row) / 8.0 - 1.0).abs() < 1e-5);
        }
        let mut dx = vec![0.0; 24];
        rms_back(&weights, &y, &inverse, 8, &mut dx);
        let numeric = slope(&x, &direction, |x| inner(&forward(x).0, &weights));
        assert!((inner(&dx, &direction) - numeric).abs() < 2e-3, "{numeric}");
    }

    #[test]
    fn a_rotation_keeps_norms_and_turns_back() {
        let (cos, sin) = rope_tables(8, 5);
        assert_eq!(cos.len(), 20);
        // Position 0 is not turned; pair 0 turns by a radian a position.
        assert!(cos[..4].iter().all(|c| *c == 1.0) && sin[..4].iter().all(|s| *s == 0.0));
        assert!((cos[4] - 1.0_f32.cos()).abs() < 1e-6 && (sin[8] - 2.0_f32.sin()).abs() < 1e-6);
        // The last pair turns by 10⁻³ of that: base^(−6/8) is 10⁻³.
        assert!((sin[7] - 1.0e-3).abs() < 1e-7);
        let original = floats(8, 4);
        let mut turned = original.clone();
        rotate(&mut turned, &cos[12..16], &sin[12..16], false);
        assert!((inner(&turned, &turned) - inner(&original, &original)).abs() < 1e-5);
        assert!(
            turned
                .iter()
                .zip(&original)
                .any(|(a, b)| (a - b).abs() > 0.1)
        );
        rotate(&mut turned, &cos[12..16], &sin[12..16], true);
        assert!(
            turned
                .iter()
                .zip(&original)
                .all(|(a, b)| (a - b).abs() < 1e-6)
        );
    }

    #[test]
    fn a_normalized_rotated_vector_and_its_gradient_agree() {
        let (cos, sin) = rope_tables(8, 4);
        let (cos, sin) = (&cos[12..16], &sin[12..16]);
        let (x, weights, direction) = (floats(8, 21), floats(8, 22), floats(8, 23));
        let forward = |x: &[f32]| {
            let mut y = x.to_vec();
            let inverse = normalize(&mut y);
            rotate(&mut y, cos, sin, false);
            (y, inverse)
        };
        let (y, inverse) = forward(&x);
        let mut dx = vec![0.0; 8];
        normalize_back(&weights, &y, inverse, cos, sin, &mut dx);
        let numeric = slope(&x, &direction, |x| inner(&forward(x).0, &weights));
        assert!((inner(&dx, &direction) - numeric).abs() < 2e-3, "{numeric}");
    }

    #[test]
    fn the_causal_softmax_and_its_gradient_agree() {
        let (length, scale) = (5, 1.7);
        let (scores, weights, direction) = (floats(25, 5), floats(25, 6), floats(25, 7));
        let forward = |scores: &[f32]| {
            let mut p = scores.to_vec();
            causal_softmax(&mut p, length, scale);
            p
        };
        let p = forward(&scores);
        for (row, weights) in p.chunks(length).enumerate() {
            assert!((weights[..=row].iter().sum::<f32>() - 1.0).abs() < 1e-6);
            assert!(weights[row + 1..].iter().all(|w| *w == 0.0));
        }
        // Two scores a unit apart weigh e^scale to one.
        assert!((p[5 + 1] / p[5] - ((scores[6] - scores[5]) * scale).exp()).abs() < 1e-4);
        let mut gradient = weights.clone();
        causal_softmax_back(&mut gradient, &p, length, scale);
        let numeric = slope(&scores, &direction, |s| inner(&forward(s), &weights));
        assert!(
            (inner(&gradient, &direction) - numeric).abs() < 1e-3,
            "{numeric}"
        );
        // The row softmax gives the logarithm of the target's weight.
        let mut row = scores[..5].to_vec();
        let log = softmax(&mut row, 3);
        assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!((log - row[3].ln()).abs() < 1e-5);
    }

    #[test]
    fn the_gate_and_its_gradient_agree() {
        let hidden = 6;
        let (expanded, weights, direction) = (floats(24, 8), floats(12, 9), floats(24, 10));
        let forward = |expanded: &[f32]| {
            let mut gated = vec![0.0; 12];
            swiglu(expanded, hidden, &mut gated);
            gated
        };
        // silu(1) · 2 = 2 / (1 + 1/e).
        let mut one = [0.0];
        swiglu(&[1.0, 2.0], 1, &mut one);
        assert!((one[0] - 2.0 / (1.0 + (-1.0_f32).exp())).abs() < 1e-6);
        let mut gradient = vec![0.0; 24];
        swiglu_back(&weights, &expanded, hidden, &mut gradient);
        let numeric = slope(&expanded, &direction, |e| inner(&forward(e), &weights));
        assert!(
            (inner(&gradient, &direction) - numeric).abs() < 1e-3,
            "{numeric}"
        );
    }

    #[test]
    fn the_exclusive_projection_and_its_gradient_agree() {
        let width = 6;
        let (attended, own, weights) = (floats(width, 11), floats(width, 12), floats(width, 13));
        let (direction_a, direction_v) = (floats(width, 14), floats(width, 15));
        let forward = |attended: &[f32], own: &[f32]| {
            let mut out = vec![0.0; width];
            let kept = exclude(attended, own, &mut out);
            (out, kept)
        };
        let (out, kept) = forward(&attended, &own);
        // Nothing of the own value is left, the rest of the attended vector
        // is, and a value attended alone goes.
        assert!(inner(&out, &own).abs() < 1e-5);
        let along = inner(&attended, &own) / inner(&own, &own);
        for ((out, a), v) in out.iter().zip(&attended).zip(&own) {
            assert!((f64::from(*out) - (f64::from(*a) - along * f64::from(*v))).abs() < 1e-5);
        }
        assert!(forward(&own, &own).0.iter().all(|v| *v == 0.0));
        let (mut d_attended, mut d_own) = (vec![0.0; width], vec![0.0; width]);
        exclude_back(&weights, &attended, &own, kept, &mut d_attended, &mut d_own);
        let by_attended = slope(&attended, &direction_a, |a| {
            inner(&forward(a, &own).0, &weights)
        });
        let by_own = slope(&own, &direction_v, |v| {
            inner(&forward(&attended, v).0, &weights)
        });
        assert!((inner(&d_attended, &direction_a) - by_attended).abs() < 1e-3);
        assert!(
            (inner(&d_own, &direction_v) - by_own).abs() < 1e-3,
            "{by_own}"
        );
    }

    #[test]
    fn a_value_under_the_epsilon_is_not_projected_out() {
        // A zero own value has no direction: the attended vector stays, and
        // nothing depends on the own value to first order.
        let (attended, weights) = (floats(6, 16), floats(6, 17));
        let own = vec![0.0_f32; 6];
        let mut out = vec![0.0; 6];
        let kept = exclude(&attended, &own, &mut out);
        assert_eq!((out.as_slice(), kept), (attended.as_slice(), (0.0, 0.0)));
        let (mut d_attended, mut d_own) = (vec![0.0; 6], vec![0.0; 6]);
        exclude_back(&weights, &attended, &own, kept, &mut d_attended, &mut d_own);
        assert_eq!(d_attended, weights);
        assert!(d_own.iter().all(|d| *d == 0.0));
        // Half the epsilon: the remainder keeps three quarters of the value.
        let small = [0.5e-12_f32, 0.0, 0.0, 0.0, 0.0, 0.0];
        let kept = exclude(&[0.0; 6], &small, &mut out);
        assert!((kept.0 - 0.5e-12).abs() < 1e-18);
        // δ = −own: −own + (own·u) u + r own = own (−1 + ¼ + ¾) = 0.
        assert!(out[0].abs() < 1e-19, "{}", out[0]);
    }
}
