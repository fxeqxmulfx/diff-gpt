//! Single-precision linear algebra on the lower triangle of a symmetric
//! matrix: what PyTorch's float32 tensors get from LAPACK's `spotrf`, whose
//! failures on matrices that rounding has made indefinite shape the series of
//! TempoPFN's Gaussian-process prior (see [`crate::synth::gpytorch`]).

/// The dot product of two slices of one length in single precision,
/// accumulated with fused multiply-adds in 64 lanes, then in 8, then one by
/// one. The result does not depend on the vector width of the machine.
#[must_use]
pub fn dot(first: &[f32], second: &[f32]) -> f32 {
    debug_assert_eq!(first.len(), second.len());
    let (first_wide, first_rest) = first.as_chunks::<64>();
    let (second_wide, second_rest) = second.as_chunks::<64>();
    let mut wide = [0.0_f32; 64];
    for (a, b) in first_wide.iter().zip(second_wide) {
        for lane in 0..64 {
            wide[lane] = a[lane].mul_add(b[lane], wide[lane]);
        }
    }
    let mut lanes = [0.0_f32; 8];
    for (index, lane) in wide.iter().enumerate() {
        lanes[index % 8] += lane;
    }
    let (first_lanes, first_rest) = first_rest.as_chunks::<8>();
    let (second_lanes, second_rest) = second_rest.as_chunks::<8>();
    for (a, b) in first_lanes.iter().zip(second_lanes) {
        for lane in 0..8 {
            lanes[lane] = a[lane].mul_add(b[lane], lanes[lane]);
        }
    }
    let mut total = ((lanes[0] + lanes[4]) + (lanes[2] + lanes[6]))
        + ((lanes[1] + lanes[5]) + (lanes[3] + lanes[7]));
    for (a, b) in first_rest.iter().zip(second_rest) {
        total = a.mul_add(*b, total);
    }
    total
}

/// The number of rows that are factored together: their part of the matrix,
/// 128 kB at 2048 columns, stays in the second-level cache while the rows
/// above stream past once.
const ROWS: usize = 16;

/// The place of the first entry of a row in the packed triangle.
const fn start(row: usize) -> usize {
    row * (row + 1) / 2
}

/// The lower triangle of a square matrix in single precision, packed by
/// rows: row `i` has `i + 1` entries. It stands for a symmetric matrix, or
/// for a lower-triangular factor.
#[derive(Clone, Debug, PartialEq)]
pub struct Lower {
    size: usize,
    values: Vec<f32>,
}

impl Lower {
    /// The triangle of `size` rows whose entry `(row, column)`, for
    /// `column ≤ row`, is `entry(row, column)`.
    #[must_use]
    pub fn from_fn(size: usize, mut entry: impl FnMut(usize, usize) -> f32) -> Self {
        let mut values = Vec::with_capacity(start(size));
        for row in 0..size {
            for column in 0..=row {
                values.push(entry(row, column));
            }
        }
        Self { size, values }
    }

    /// The number of rows.
    #[must_use]
    pub const fn size(&self) -> usize {
        self.size
    }

    /// Row `row`: its entries in the columns `0..=row`.
    #[must_use]
    pub fn row(&self, row: usize) -> &[f32] {
        &self.values[start(row)..=start(row) + row]
    }

    /// The entries, row after row.
    #[must_use]
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// The entries, row after row, to change in place.
    #[must_use]
    pub fn values_mut(&mut self) -> &mut [f32] {
        &mut self.values
    }

    /// The diagonal, to change in place.
    pub fn diagonal_mut(&mut self) -> impl Iterator<Item = &mut f32> {
        let mut rest = self.values.as_mut_slice();
        (0..self.size).map_while(move |row| {
            let (head, tail) = core::mem::take(&mut rest).split_at_mut_checked(row + 1)?;
            rest = tail;
            head.last_mut()
        })
    }

    /// Replaces the symmetric matrix by its Cholesky factor `L`, lower
    /// triangular with `L Lᵀ` the matrix.
    ///
    /// # Errors
    ///
    /// The row of the first pivot that is not positive, as LAPACK's `spotrf`
    /// reports it; the triangle then holds a partial factor.
    pub fn cholesky(&mut self) -> Result<(), usize> {
        let size = self.size;
        for first in (0..size).step_by(ROWS) {
            let last = (first + ROWS).min(size);
            let (above, rest) = self.values.split_at_mut(start(first));
            let block = &mut rest[..start(last) - start(first)];
            // The columns left of the block: each row of the block against
            // the finished rows above.
            for column in 0..first {
                let finished = &above[start(column)..=start(column) + column];
                let (shared, pivot) = finished.split_at(column);
                let mut remaining = &mut *block;
                for row in first..last {
                    let (current, next) = remaining.split_at_mut(row + 1);
                    current[column] =
                        (current[column] - dot(&current[..column], shared)) / pivot[0];
                    remaining = next;
                }
            }
            // The columns of the block.
            for column in first..last {
                let offset = start(column) - start(first);
                let (upper, lower) = block.split_at_mut(offset + column + 1);
                let current = &mut upper[offset..];
                let pivot = current[column] - dot(&current[..column], &current[..column]);
                if pivot.is_nan() || pivot <= 0.0 {
                    return Err(column);
                }
                let pivot = pivot.sqrt();
                current[column] = pivot;
                let shared = &current[..column];
                let mut remaining = lower;
                for row in column + 1..last {
                    let (below, next) = remaining.split_at_mut(row + 1);
                    below[column] = (below[column] - dot(&below[..column], shared)) / pivot;
                    remaining = next;
                }
            }
        }
        Ok(())
    }

    /// The lower triangle of `L Lᵀ` for this lower-triangular `L`.
    #[must_use]
    pub fn gram(&self) -> Self {
        let mut values = vec![0.0; self.values.len()];
        for first in (0..self.size).step_by(ROWS) {
            let last = (first + ROWS).min(self.size);
            for column in 0..last {
                let shared = self.row(column);
                for row in first.max(column)..last {
                    values[start(row) + column] = dot(&self.row(row)[..=column], shared);
                }
            }
        }
        Self {
            size: self.size,
            values,
        }
    }

    /// `L z` for this lower-triangular `L`.
    #[must_use]
    pub fn apply(&self, vector: &[f32]) -> Vec<f32> {
        debug_assert_eq!(vector.len(), self.size);
        (0..self.size)
            .map(|row| dot(self.row(row), &vector[..=row]))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::Rng;

    /// A random symmetric positive definite matrix: `B Bᵀ + shift I` for a
    /// random lower-triangular `B`.
    fn positive(rng: &mut Rng, size: usize, shift: f32) -> Lower {
        let factor = Lower::from_fn(size, |_, _| rng.normal() as f32);
        let mut matrix = factor.gram();
        for value in matrix.diagonal_mut() {
            *value += shift;
        }
        matrix
    }

    #[test]
    fn the_dot_product_matches_the_plain_sum() {
        let mut rng = Rng::from_key(&[1]);
        for length in [0, 1, 7, 8, 9, 63, 64, 65, 200, 1000] {
            let first: Vec<f32> = (0..length).map(|_| rng.normal() as f32).collect();
            let second: Vec<f32> = (0..length).map(|_| rng.normal() as f32).collect();
            let exact: f64 = first
                .iter()
                .zip(&second)
                .map(|(a, b)| f64::from(*a) * f64::from(*b))
                .sum();
            let scale: f64 = first
                .iter()
                .zip(&second)
                .map(|(a, b)| f64::from(a.abs()) * f64::from(b.abs()))
                .sum();
            let error = (f64::from(dot(&first, &second)) - exact).abs();
            assert!(error <= 1e-6 * scale + 1e-12, "length {length}: {error}");
        }
    }

    #[test]
    fn the_factor_of_a_positive_definite_matrix_gives_it_back() {
        let mut rng = Rng::from_key(&[2]);
        for size in [1, 2, 5, 16, 17, 40, 100] {
            let matrix = positive(&mut rng, size, 1.0);
            let mut factor = matrix.clone();
            assert_eq!(factor.cholesky(), Ok(()));
            assert!((0..size).all(|row| factor.row(row)[row] > 0.0));
            let back = factor.gram();
            let scale = matrix.values().iter().fold(0.0_f32, |a, b| a.max(b.abs()));
            for (a, b) in matrix.values().iter().zip(back.values()) {
                assert!((a - b).abs() <= 1e-4 * scale, "size {size}: {a} and {b}");
            }
        }
    }

    #[test]
    fn the_factorization_stops_at_the_first_pivot_that_is_not_positive() {
        let mut rng = Rng::from_key(&[3]);
        // A matrix of rank 3: the fourth pivot is rounding noise or zero.
        let thin = |row: usize, column: usize| {
            if column < 3 {
                1.0 + (row * 3 + column) as f32
            } else {
                0.0
            }
        };
        let factor = Lower::from_fn(40, |row, column| {
            if column <= row {
                thin(row, column)
            } else {
                0.0
            }
        });
        let mut singular = factor.gram();
        // Make the rest of the diagonal clearly negative past the rank.
        for (row, value) in singular.diagonal_mut().enumerate() {
            if row >= 20 {
                *value -= 1.0e6;
            }
        }
        let failed = singular
            .clone()
            .cholesky()
            .expect_err("the matrix is indefinite");
        assert!((3..=20).contains(&failed), "failed at {failed}");
        // A zero matrix fails at once; with a jitter it does not.
        let mut zero = Lower::from_fn(8, |_, _| 0.0);
        assert_eq!(zero.clone().cholesky(), Err(0));
        for value in zero.diagonal_mut() {
            *value += 1.0e-3;
        }
        assert_eq!(zero.cholesky(), Ok(()));
        // A not-a-number pivot fails as well.
        let mut broken = positive(&mut rng, 6, 1.0);
        broken.values_mut()[start(4) + 4] = f32::NAN;
        assert_eq!(broken.cholesky(), Err(4));
    }

    #[test]
    fn the_factor_applied_to_a_vector_is_the_matrix_product() {
        let mut rng = Rng::from_key(&[4]);
        let factor = Lower::from_fn(33, |_, _| rng.normal() as f32);
        let vector: Vec<f32> = (0..33).map(|_| rng.normal() as f32).collect();
        let product = factor.apply(&vector);
        for (row, value) in product.iter().enumerate() {
            let exact: f32 = (0..=row)
                .map(|column| factor.row(row)[column] * vector[column])
                .sum();
            assert!((value - exact).abs() <= 1e-4 * (1.0 + exact.abs()));
        }
        assert_eq!(factor.clone().diagonal_mut().count(), 33);
    }
}
