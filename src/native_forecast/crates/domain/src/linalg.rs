//! Dense linear algebra for the Gaussian-process sources: dot products that
//! the compiler vectorizes, and blocked Cholesky factorizations.

// Without FMA in the target, `mul_add` is a library call for every product:
// fifty times slower. Cargo reads `.cargo/config.toml`, which enables it, only
// when it runs inside this workspace.
#[cfg(not(target_feature = "fma"))]
compile_error!("run cargo inside src/native_forecast, whose .cargo/config.toml enables FMA");

/// The dot product of two slices of one length, accumulated with fused
/// multiply-adds in sixteen lanes: the compiler turns the lanes into four
/// AVX2 registers, and the result does not depend on the vector width.
#[must_use]
pub fn dot(first: &[f64], second: &[f64]) -> f64 {
    debug_assert_eq!(first.len(), second.len());
    let (first_lanes, first_rest) = first.as_chunks::<16>();
    let (second_lanes, second_rest) = second.as_chunks::<16>();
    let mut lanes = [0.0_f64; 16];
    for (a, b) in first_lanes.iter().zip(second_lanes) {
        for lane in 0..16 {
            lanes[lane] = a[lane].mul_add(b[lane], lanes[lane]);
        }
    }
    let mut quarter = [0.0_f64; 4];
    for (index, lane) in lanes.iter().enumerate() {
        quarter[index % 4] += lane;
    }
    let mut total = (quarter[0] + quarter[2]) + (quarter[1] + quarter[3]);
    for (a, b) in first_rest.iter().zip(second_rest) {
        total = a.mul_add(*b, total);
    }
    total
}

/// The number of columns factored between two updates of the rest of the
/// matrix.
const BLOCK: usize = 64;

/// The number of rows in a tile of the update of the rest of the matrix: two
/// tiles of the panel, `2 · TILE · BLOCK` numbers, fit the first-level cache.
const TILE: usize = 24;

/// The pivoted Cholesky factor of a positive semidefinite matrix `K` of
/// numerical rank `rank`: `K ≈ P L Lᵀ Pᵀ` with `L` lower trapezoidal of
/// `rank` columns and `P` a permutation.
#[derive(Clone, Debug)]
pub struct PivotedFactor {
    size: usize,
    rank: usize,
    /// `L` in row-major order with rows of `size` entries, of which row `i`
    /// uses the first `min(i + 1, rank)`.
    factor: Vec<f64>,
    /// The row of `K` that row `i` of `L` stands for.
    order: Vec<usize>,
}

impl PivotedFactor {
    /// Factors the symmetric matrix of `size` rows whose entry `(i, j)`, for
    /// `j ≤ i`, is `entry(i, j)`, as LAPACK's blocked `dpstrf` does: at each
    /// step the greatest remaining diagonal entry is the pivot, and the
    /// factorization stops when that is no more than
    /// `size · ε · max(diagonal)`, with ε the rounding unit.
    #[must_use]
    pub fn new(size: usize, entry: impl Fn(usize, usize) -> f64) -> Self {
        let mut a = vec![0.0; size * size];
        for row in 0..size {
            for column in 0..=row {
                a[row * size + column] = entry(row, column);
            }
        }
        let greatest = (0..size).map(|i| a[i * size + i]).fold(0.0_f64, f64::max);
        let stop = size as f64 * (f64::EPSILON / 2.0) * greatest;
        let mut order: Vec<usize> = (0..size).collect();
        // The squares of the entries of each row in the columns of the block.
        let mut squares = vec![0.0; size];
        let mut pivot_row = [0.0; BLOCK];
        let mut panel = Vec::new();
        let mut rank = size;

        'blocks: for start in (0..size).step_by(BLOCK) {
            let end = (start + BLOCK).min(size);
            squares[start..].fill(0.0);
            for column in start..end {
                let (mut pivot, mut residual) = (column, f64::NEG_INFINITY);
                for row in column..size {
                    if column > start {
                        let previous = a[row * size + column - 1];
                        squares[row] += previous * previous;
                    }
                    let left = a[row * size + row] - squares[row];
                    if left > residual || left.is_nan() {
                        (pivot, residual) = (row, left);
                    }
                }
                // Written so that a NaN pivot stops the factorization too.
                if !(residual > stop && residual > 0.0) {
                    rank = column;
                    break 'blocks;
                }
                if pivot != column {
                    swap_symmetric(&mut a, size, column, pivot);
                    squares.swap(column, pivot);
                    order.swap(column, pivot);
                }
                let root = residual.sqrt();
                a[column * size + column] = root;
                let width = column - start;
                pivot_row[..width]
                    .copy_from_slice(&a[column * size + start..column * size + column]);
                for row in column + 1..size {
                    let values = &mut a[row * size + start..=row * size + column];
                    let explained = dot(&values[..width], &pivot_row[..width]);
                    values[width] = (values[width] - explained) / root;
                }
            }
            // The rest of the matrix loses what the block explains of it.
            let width = end - start;
            panel.clear();
            for row in end..size {
                panel.extend_from_slice(&a[row * size + start..row * size + end]);
            }
            // In tiles of rows whose parts of the panel stay in the cache.
            let rest = size - end;
            for tile_row in (0..rest).step_by(TILE) {
                for tile_column in (0..=tile_row).step_by(TILE) {
                    for row in tile_row..(tile_row + TILE).min(rest) {
                        let own = &panel[row * width..(row + 1) * width];
                        let last = (tile_column + TILE).min(row + 1);
                        let base = (end + row) * size + end;
                        let targets = &mut a[base + tile_column..base + last];
                        let others = panel[tile_column * width..last * width].chunks_exact(width);
                        for (target, other) in targets.iter_mut().zip(others) {
                            *target -= dot(own, other);
                        }
                    }
                }
            }
        }
        Self {
            size,
            rank,
            factor: a,
            order,
        }
    }

    /// The numerical rank found.
    #[must_use]
    pub const fn rank(&self) -> usize {
        self.rank
    }

    /// Row `row` of `L`.
    fn row(&self, row: usize) -> &[f64] {
        &self.factor[row * self.size..row * self.size + (row + 1).min(self.rank)]
    }

    /// `P L z` for a vector `z` of `rank` entries: a draw from the Gaussian of
    /// covariance `K` when `z` is standard normal.
    ///
    /// # Panics
    ///
    /// If `z` has not `rank` entries.
    #[must_use]
    pub fn apply(&self, z: &[f64]) -> Vec<f64> {
        assert_eq!(z.len(), self.rank, "one normal for each column");
        let mut out = vec![0.0; self.size];
        for (row, &original) in self.order.iter().enumerate() {
            let values = self.row(row);
            out[original] = dot(values, &z[..values.len()]);
        }
        out
    }

    /// The matrix `P L Lᵀ Pᵀ`, row after row: what the factor stands for.
    #[must_use]
    pub fn product(&self) -> Vec<f64> {
        let mut out = vec![0.0; self.size * self.size];
        for (row, &original) in self.order.iter().enumerate() {
            for (other, &original_other) in self.order.iter().enumerate() {
                let (first, second) = (self.row(row), self.row(other));
                let shared = first.len().min(second.len());
                out[original * self.size + original_other] =
                    dot(&first[..shared], &second[..shared]);
            }
        }
        out
    }
}

/// Exchanges the rows and columns `first < second` of a symmetric matrix
/// stored in its lower triangle, but for the diagonal entry of `first`, which
/// the caller overwrites.
fn swap_symmetric(a: &mut [f64], size: usize, first: usize, second: usize) {
    a[second * size + second] = a[first * size + first];
    for column in 0..first {
        a.swap(first * size + column, second * size + column);
    }
    for row in second + 1..size {
        a.swap(row * size + first, row * size + second);
    }
    for between in first + 1..second {
        a.swap(between * size + first, second * size + between);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::Rng;

    fn factor_of(matrix: &[Vec<f64>]) -> PivotedFactor {
        PivotedFactor::new(matrix.len(), |row, column| matrix[row][column])
    }

    /// The greatest difference between an entry of `P L Lᵀ Pᵀ` and of `matrix`.
    fn reproduction_error(factor: &PivotedFactor, matrix: &[Vec<f64>]) -> f64 {
        let product = factor.product();
        let flat = matrix.iter().flatten();
        product
            .iter()
            .zip(flat)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max)
    }

    /// The Gram matrix of `size` random vectors of `dimension` entries.
    fn gram(rng: &mut Rng, size: usize, dimension: usize) -> Vec<Vec<f64>> {
        let basis: Vec<Vec<f64>> = (0..size)
            .map(|_| (0..dimension).map(|_| rng.normal()).collect())
            .collect();
        (0..size)
            .map(|i| (0..size).map(|j| dot(&basis[i], &basis[j])).collect())
            .collect()
    }

    #[test]
    fn the_dot_product_matches_the_plain_sum() {
        let mut rng = Rng::from_key(&[1]);
        for length in [0, 1, 7, 15, 16, 17, 64, 100] {
            let a: Vec<f64> = (0..length).map(|_| rng.normal()).collect();
            let b: Vec<f64> = (0..length).map(|_| rng.normal()).collect();
            let plain: f64 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
            assert!((dot(&a, &b) - plain).abs() < 1e-12, "length {length}");
        }
    }

    #[test]
    fn full_rank_matrices_are_reproduced_across_blocks() {
        let mut rng = Rng::from_key(&[2]);
        for size in [1, 40, 64, 65, 150] {
            let matrix = gram(&mut rng, size, size + 3);
            let factor = factor_of(&matrix);
            assert_eq!(factor.rank(), size);
            assert!(reproduction_error(&factor, &matrix) < 1e-9, "size {size}");
        }
    }

    #[test]
    fn the_factorization_stops_at_the_rank() {
        let mut rng = Rng::from_key(&[3]);
        for (size, rank) in [(50, 3), (200, 64), (200, 70), (130, 129)] {
            let matrix = gram(&mut rng, size, rank);
            let factor = factor_of(&matrix);
            assert_eq!(factor.rank(), rank, "size {size}");
            assert!(reproduction_error(&factor, &matrix) < 1e-9, "size {size}");
        }
        let constant = vec![vec![2.0; 10]; 10];
        let factor = factor_of(&constant);
        assert_eq!(factor.rank(), 1);
        let line = factor.apply(&[3.0]);
        assert!(
            line.iter()
                .all(|value| (value - 3.0 * 2.0_f64.sqrt()).abs() < 1e-14)
        );
        assert_eq!(factor_of(&vec![vec![0.0; 4]; 4]).rank(), 0);
        assert_eq!(factor_of(&[]).rank(), 0);
    }

    #[test]
    fn draws_have_the_covariance() {
        let matrix = vec![
            vec![4.0, 1.0, 0.5],
            vec![1.0, 2.0, -0.3],
            vec![0.5, -0.3, 1.0],
        ];
        let factor = factor_of(&matrix);
        let mut rng = Rng::from_key(&[4]);
        let draws = 200_000;
        let mut sums = [[0.0; 3]; 3];
        for _ in 0..draws {
            let z: Vec<f64> = (0..3).map(|_| rng.normal()).collect();
            let sample = factor.apply(&z);
            for (i, row) in sums.iter_mut().enumerate() {
                for (j, sum) in row.iter_mut().enumerate() {
                    *sum += sample[i] * sample[j];
                }
            }
        }
        for (sum, expected) in sums.iter().flatten().zip(matrix.iter().flatten()) {
            assert!((sum / f64::from(draws) - expected).abs() < 0.04);
        }
    }
}
