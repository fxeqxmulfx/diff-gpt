//! The eigen-decomposition `A = V Λ Vᵀ` of a real symmetric matrix in double
//! precision: what gpytorch falls back to (`torch.linalg.eigh` on the matrix
//! in float64) when no jitter makes a covariance matrix positive definite.
//!
//! The matrix is brought to tridiagonal form by Householder reflectors, a
//! panel of them at a time as in LAPACK's `dsytrd`, and the tridiagonal
//! matrix is diagonalized by the implicit QL iteration (EISPACK's `tql2`).
//! `V` is kept as its factors, the reflectors and the plane rotations, which
//! [`SymmetricEigen::combine`] applies to a vector: a draw from a normal law
//! needs one product with `V`, not `V` itself.

use crate::linalg::dot;

/// The number of reflectors gathered before the rest of the matrix is
/// updated.
const PANEL: usize = 32;

/// The number of columns updated at a time: the parts of a panel's vectors
/// that an update reads, `2 · PANEL · TILE` numbers, stay in the second-level
/// cache.
const TILE: usize = 256;

/// The most iterations spent on one eigenvalue; the iteration converges
/// cubically, in two or three.
const ITERATIONS: usize = 60;

/// A plane rotation of the coordinates `place` and `place + 1`.
#[derive(Clone, Copy, Debug)]
struct Rotation {
    place: u32,
    cosine: f64,
    sine: f64,
}

/// The eigenvalues of a symmetric matrix, and its eigenvectors as a product
/// of reflectors and rotations.
#[derive(Clone, Debug)]
pub struct SymmetricEigen {
    size: usize,
    /// The eigenvalues, in the order of the columns of `V`; not sorted.
    values: Vec<f64>,
    /// Row `k` holds, from column `k + 2` on, the vector of the reflector of
    /// step `k` after its leading 1 at `k + 1`.
    reflectors: Vec<f64>,
    /// The factors τ of the reflectors `I − τ v vᵀ`.
    scales: Vec<f64>,
    /// The rotations of the QL iteration, in the order they were made.
    rotations: Vec<Rotation>,
}

impl SymmetricEigen {
    /// Decomposes the symmetric matrix of `size` rows whose entry
    /// `(row, column)`, for `column ≤ row`, is `entry(row, column)`.
    #[must_use]
    pub fn new(size: usize, entry: impl Fn(usize, usize) -> f64) -> Self {
        // The upper triangle by rows, which is the lower one by columns.
        let mut matrix = vec![0.0; size * size];
        for row in 0..size {
            for column in row..size {
                matrix[row * size + column] = entry(column, row);
            }
        }
        let mut diagonal = vec![0.0; size];
        let mut off = vec![0.0; size];
        let mut scales = vec![0.0; size];
        tridiagonalize(&mut matrix, size, &mut diagonal, &mut off, &mut scales);
        let rotations = diagonalize(&mut diagonal, &mut off);
        Self {
            size,
            values: diagonal,
            reflectors: matrix,
            scales,
            rotations,
        }
    }

    /// The eigenvalues, in the order of the columns of `V`; not sorted.
    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// `V w`: the combination of the eigenvectors with the weights `w`.
    #[must_use]
    pub fn combine(&self, weights: &[f64]) -> Vec<f64> {
        debug_assert_eq!(weights.len(), self.size);
        let size = self.size;
        let mut vector = weights.to_vec();
        // V = H₀ H₁ ⋯ G₁ G₂ ⋯: the last factor acts first.
        for rotation in self.rotations.iter().rev() {
            let place = rotation.place as usize;
            let (lower, upper) = (vector[place], vector[place + 1]);
            vector[place] = rotation.cosine.mul_add(lower, rotation.sine * upper);
            vector[place + 1] = rotation.cosine.mul_add(upper, -rotation.sine * lower);
        }
        for step in (0..size.saturating_sub(1)).rev() {
            let scale = self.scales[step];
            if scale == 0.0 {
                continue;
            }
            let tail = &self.reflectors[step * size + step + 2..(step + 1) * size];
            let (head, rest) = vector.split_at_mut(step + 2);
            let projection = scale * (head[step + 1] + dot(tail, rest));
            head[step + 1] -= projection;
            for (value, entry) in rest.iter_mut().zip(tail) {
                *value = (-projection).mul_add(*entry, *value);
            }
        }
        vector
    }
}

/// Reduces the symmetric matrix, given by its upper triangle in rows of
/// `size` entries, to the tridiagonal `Qᵀ A Q` with `Q = H₀ H₁ ⋯`, where
/// `H_k = I − τ_k v_k v_kᵀ` and `v_k` is zero up to `k`, 1 at `k + 1`. On
/// return `diagonal` and `off` (the entry between `k` and `k + 1`) hold the
/// tridiagonal matrix, `scales` the τ, and row `k` of the matrix the rest of
/// `v_k` from column `k + 2` on.
fn tridiagonalize(
    matrix: &mut [f64],
    size: usize,
    diagonal: &mut [f64],
    off: &mut [f64],
    scales: &mut [f64],
) {
    let steps = size.saturating_sub(1);
    // The vectors v and w of the pending update A − Σ (v wᵀ + w vᵀ), one
    // after the other, and the same by rows for the update of the rest.
    let mut v_columns = vec![0.0_f64; PANEL * size];
    let mut w_columns = vec![0.0_f64; PANEL * size];
    let mut v_rows = vec![0.0_f64; size * PANEL];
    let mut w_rows = vec![0.0_f64; size * PANEL];
    let mut first = 0;
    while first < steps {
        let width = PANEL.min(steps - first);
        for pending in 0..width {
            let step = first + pending;
            let (v_done, v_rest) = v_columns.split_at_mut(pending * size);
            let (w_done, w_rest) = w_columns.split_at_mut(pending * size);
            let (v, w) = (&mut v_rest[..size], &mut w_rest[..size]);
            {
                // The row of the step, from its diagonal on, as the pending
                // updates leave it; then its reflector.
                let row = &mut matrix[step * size + step..(step + 1) * size];
                for (earlier_v, earlier_w) in
                    v_done.chunks_exact(size).zip(w_done.chunks_exact(size))
                {
                    let (at_v, at_w) = (earlier_v[step], earlier_w[step]);
                    for ((value, x), y) in row
                        .iter_mut()
                        .zip(&earlier_v[step..])
                        .zip(&earlier_w[step..])
                    {
                        *value = (-at_v).mul_add(*y, (-at_w).mul_add(*x, *value));
                    }
                }
                diagonal[step] = row[0];
                let alpha = row[1];
                let norm = dot(&row[2..], &row[2..]).sqrt();
                if norm == 0.0 {
                    off[step] = alpha;
                    scales[step] = 0.0;
                } else {
                    let beta = -alpha.hypot(norm).copysign(alpha);
                    for value in &mut row[2..] {
                        *value /= alpha - beta;
                    }
                    off[step] = beta;
                    scales[step] = (beta - alpha) / beta;
                }
                v[..=step].fill(0.0);
                v[step + 1] = 1.0;
                v[step + 2..].copy_from_slice(&row[2..]);
            }
            // w = τ (A v) − ½ τ² (vᵀ A v) v, with A the matrix under the
            // pending updates: the stored rows after the step, then the
            // corrections.
            w.fill(0.0);
            for row in step + 1..size {
                let stored = &matrix[row * size + row..(row + 1) * size];
                let (head, tail) = w.split_at_mut(row + 1);
                head[row] += stored[0].mul_add(v[row], dot(&stored[1..], &v[row + 1..]));
                let factor = v[row];
                for (value, entry) in tail.iter_mut().zip(&stored[1..]) {
                    *value = factor.mul_add(*entry, *value);
                }
            }
            for (earlier_v, earlier_w) in v_done.chunks_exact(size).zip(w_done.chunks_exact(size)) {
                let with_w = dot(&earlier_w[step + 1..], &v[step + 1..]);
                let with_v = dot(&earlier_v[step + 1..], &v[step + 1..]);
                for ((value, x), y) in w[step + 1..]
                    .iter_mut()
                    .zip(&earlier_v[step + 1..])
                    .zip(&earlier_w[step + 1..])
                {
                    *value = (-with_w).mul_add(*x, (-with_v).mul_add(*y, *value));
                }
            }
            let scale = scales[step];
            for value in &mut w[step + 1..] {
                *value *= scale;
            }
            let half = -0.5 * scale * dot(&w[step + 1..], &v[step + 1..]);
            for (value, x) in w[step + 1..].iter_mut().zip(&v[step + 1..]) {
                *value = half.mul_add(*x, *value);
            }
        }
        // The update of the rows after the panel, a tile of columns at a time.
        let start = first + width;
        for row in start..size {
            for pending in 0..width {
                v_rows[row * PANEL + pending] = v_columns[pending * size + row];
                w_rows[row * PANEL + pending] = w_columns[pending * size + row];
            }
        }
        for tile in (start..size).step_by(TILE) {
            let end = (tile + TILE).min(size);
            for row in start..end {
                let from = row.max(tile);
                let piece = &mut matrix[row * size + from..row * size + end];
                for pending in 0..width {
                    let at_v = v_rows[row * PANEL + pending];
                    let at_w = w_rows[row * PANEL + pending];
                    let v = &v_columns[pending * size + from..pending * size + end];
                    let w = &w_columns[pending * size + from..pending * size + end];
                    for ((value, x), y) in piece.iter_mut().zip(v).zip(w) {
                        *value = (-at_v).mul_add(*y, (-at_w).mul_add(*x, *value));
                    }
                }
            }
        }
        first = start;
    }
    if size > 0 {
        diagonal[size - 1] = matrix[size * size - 1];
    }
}

/// Diagonalizes the symmetric tridiagonal matrix of diagonal `diagonal` and
/// of entries `off[k]` between `k` and `k + 1` by the implicit QL iteration
/// with Wilkinson's shift. On return `diagonal` holds the eigenvalues, and
/// the rotations returned, as `Z = G₁ G₂ ⋯` in the order they were made, are
/// the eigenvectors.
fn diagonalize(diagonal: &mut [f64], off: &mut [f64]) -> Vec<Rotation> {
    let size = diagonal.len();
    let mut rotations = Vec::new();
    let norm = diagonal
        .iter()
        .zip(off.iter())
        .map(|(d, e)| d.abs() + e.abs())
        .fold(0.0, f64::max);
    // An off-diagonal entry below this is zero for a matrix of this norm.
    let floor = norm * f64::EPSILON / 1024.0;
    for lowest in 0..size {
        for _ in 0..ITERATIONS {
            // The block that starts at `lowest` ends at the first negligible
            // off-diagonal entry.
            let mut end = lowest;
            while end + 1 < size {
                let scale = diagonal[end].abs() + diagonal[end + 1].abs();
                if off[end].abs() <= f64::EPSILON * scale || off[end].abs() <= floor {
                    break;
                }
                end += 1;
            }
            if end == lowest {
                break;
            }
            let mut g = (diagonal[lowest + 1] - diagonal[lowest]) / (2.0 * off[lowest]);
            let mut r = g.hypot(1.0);
            g = diagonal[end] - diagonal[lowest] + off[lowest] / (g + r.copysign(g));
            let (mut sine, mut cosine, mut p) = (1.0, 1.0, 0.0);
            let mut vanished = false;
            for place in (lowest..end).rev() {
                let f = sine * off[place];
                let b = cosine * off[place];
                r = f.hypot(g);
                off[place + 1] = r;
                if r == 0.0 {
                    diagonal[place + 1] -= p;
                    off[end] = 0.0;
                    vanished = true;
                    break;
                }
                sine = f / r;
                cosine = g / r;
                g = diagonal[place + 1] - p;
                r = (diagonal[place] - g).mul_add(sine, 2.0 * cosine * b);
                p = sine * r;
                diagonal[place + 1] = g + p;
                g = cosine.mul_add(r, -b);
                rotations.push(Rotation {
                    place: place as u32,
                    cosine,
                    sine,
                });
            }
            if vanished {
                continue;
            }
            diagonal[lowest] -= p;
            off[lowest] = g;
            off[end] = 0.0;
        }
    }
    rotations
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::Rng;

    /// The eigenvectors as rows: row `k` is `V eₖ`.
    fn vectors(eigen: &SymmetricEigen, size: usize) -> Vec<Vec<f64>> {
        (0..size)
            .map(|k| {
                let mut unit = vec![0.0; size];
                unit[k] = 1.0;
                eigen.combine(&unit)
            })
            .collect()
    }

    #[test]
    fn the_factors_give_the_matrix_back() {
        let mut rng = Rng::from_key(&[1]);
        for size in [1, 2, 3, 10, 33, 70, 150] {
            let mut matrix = vec![0.0; size * size];
            for row in 0..size {
                for column in 0..=row {
                    let value = rng.normal();
                    matrix[row * size + column] = value;
                    matrix[column * size + row] = value;
                }
            }
            let eigen = SymmetricEigen::new(size, |row, column| matrix[row * size + column]);
            let vectors = vectors(&eigen, size);
            let scale = size as f64;
            for first in 0..size {
                for second in 0..=first {
                    // Orthonormal eigenvectors.
                    let product = dot(&vectors[first], &vectors[second]);
                    let unit = f64::from(u8::from(first == second));
                    assert!((product - unit).abs() < 1e-12 * scale, "size {size}");
                    // V Λ Vᵀ is the matrix.
                    let back: f64 = (0..size)
                        .map(|k| eigen.values()[k] * vectors[k][first] * vectors[k][second])
                        .sum();
                    let error = (back - matrix[first * size + second]).abs();
                    assert!(error < 1e-12 * scale, "size {size}: error {error}");
                }
            }
        }
    }

    #[test]
    fn a_matrix_of_low_rank_has_its_rank_of_eigenvalues() {
        let mut rng = Rng::from_key(&[2]);
        let size = 90;
        let factors: Vec<Vec<f64>> = (0..3)
            .map(|_| (0..size).map(|_| rng.normal()).collect())
            .collect();
        let signs = [2.0, 1.0, -1.5];
        let entry = |row: usize, column: usize| -> f64 {
            (0..3)
                .map(|k| signs[k] * factors[k][row] * factors[k][column])
                .sum()
        };
        let eigen = SymmetricEigen::new(size, entry);
        let mut values = eigen.values().to_vec();
        values.sort_by(f64::total_cmp);
        assert!(values[0] < -10.0, "one negative eigenvalue: {}", values[0]);
        assert!(values[1].abs() < 1e-10 && values[size - 3].abs() < 1e-10);
        assert!(values[size - 2] > 10.0 && values[size - 1] > values[size - 2]);
        let trace: f64 = (0..size).map(|k| entry(k, k)).sum();
        assert!((values.iter().sum::<f64>() - trace).abs() < 1e-9 * trace.abs().max(1.0));
    }

    #[test]
    fn a_diagonal_matrix_needs_no_rotation() {
        let eigen = SymmetricEigen::new(
            5,
            |row, column| {
                if row == column { row as f64 - 2.0 } else { 0.0 }
            },
        );
        assert_eq!(eigen.values(), [-2.0, -1.0, 0.0, 1.0, 2.0]);
        assert_eq!(
            eigen.combine(&[1.0, 2.0, 3.0, 4.0, 5.0]),
            [1.0, 2.0, 3.0, 4.0, 5.0]
        );
        assert!(SymmetricEigen::new(0, |_, _| 0.0).values().is_empty());
    }
}
