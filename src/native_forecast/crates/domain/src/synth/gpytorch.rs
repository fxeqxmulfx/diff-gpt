//! What gpytorch 1.15.2 and linear_operator 0.6.1 compute, in float32, when
//! TempoPFN's Gaussian-process prior draws a zero-mean process with a
//! composite kernel on a grid: `model(x).sample()` under
//! `max_cholesky_size(len(x))`.
//!
//! The numerics are part of the prior. A kernel matrix of 2048 points is
//! numerically singular, and in float32 it is not even positive
//! semidefinite: gpytorch takes the distances between the points from
//! `‖a‖² − 2 a·b + ‖b‖²`, whose rounding errors are large next to the small
//! distances of a short period. So the Cholesky factorization fails, and
//! `psd_safe_cholesky` adds to the diagonal a jitter of 1, 10, then 100 times
//! `cholesky_jitter` (10⁻³ here) until it succeeds: white noise of deviation
//! 0.03, 0.1 or 0.3 in the drawn series. When no jitter is enough, the root
//! comes from an eigen-decomposition with the negative eigenvalues cut.
//!
//! This module follows the same steps in the same arithmetic:
//!
//! - the kernels (`kernels/{matern,periodic,linear,scale}_kernel.py`,
//!   `functions/matern_covariance.py`, `sq_dist` and `dist` of
//!   `kernels/kernel.py`), operation by operation in float32;
//! - the classes of linear_operator that an `AdditiveKernel` and a
//!   `ProductKernel` build: a dense matrix, the root of a `LinearKernel`, a
//!   sum, a product. They decide where a factorization is taken: a product of
//!   two operators that are not dense replaces each by the Gram matrix of its
//!   root (`MulLinearOperator`), and a root added to a single operator first
//!   asks for the inverse root of that operator (`add_low_rank`), which
//!   raises when the Cholesky factorization fails with every jitter;
//! - the draw: `root_decomposition` of the whole matrix, by Cholesky with
//!   the jitters, else by `eigh` in float64.
//!
//! Only the lower triangle of a matrix is computed: LAPACK reads no other.
//! The streams of random numbers and the order of the sums inside the
//! factorizations are not PyTorch's, so a series is not reproduced bit for
//! bit; its law is.

use crate::eigen::SymmetricEigen;
use crate::random::Rng;
use crate::single::{Lower, dot};

/// softplus(0) = ln 2: the output scale of a `ScaleKernel` and the variance
/// of a `LinearKernel`, whose raw parameters TempoPFN leaves at 0.
const UNIT: f32 = core::f32::consts::LN_2;

/// The number of jitters `psd_safe_cholesky` tries (`cholesky_max_tries`).
const TRIES: i32 = 3;

/// The smoothness ν of a Matérn kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Smoothness {
    /// ν = 1/2: the exponential kernel.
    Half,
    /// ν = 3/2.
    ThreeHalves,
    /// ν = 5/2.
    FiveHalves,
}

/// A kernel of gpytorch at the parameters TempoPFN sets.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Base {
    /// `LinearKernel`: `ln 2 · x x'`.
    Linear,
    /// `MaternKernel` of that lengthscale.
    Matern {
        /// ν.
        smoothness: Smoothness,
        /// The lengthscale, which divides the distances.
        lengthscale: f32,
    },
    /// `PeriodicKernel`: `exp(−2 sin²(π |x − x'| / period) / lengthscale)`.
    Periodic {
        /// The period, in the unit of the grid.
        period: f32,
        /// The lengthscale, which gpytorch does not square here.
        lengthscale: f32,
    },
}

/// A kernel, alone or under a `ScaleKernel` of output scale ln 2.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Leaf {
    /// The kernel.
    pub base: Base,
    /// Whether a `ScaleKernel` wraps it.
    pub scaled: bool,
}

/// A composite kernel, as gpytorch's `+` and `*` build it: a sum of sums is
/// one sum, a product of products one product.
#[derive(Clone, Debug, PartialEq)]
pub enum Kernel {
    /// One kernel.
    Leaf(Leaf),
    /// `AdditiveKernel`: the first term, then the others.
    Sum(Box<Self>, Vec<Self>),
    /// `ProductKernel`: the first factor, then the others.
    Product(Box<Self>, Vec<Self>),
}

impl Kernel {
    /// `self + other`.
    #[must_use]
    pub fn plus(self, other: Self) -> Self {
        let (first, mut terms) = match self {
            Self::Sum(first, terms) => (first, terms),
            single => (Box::new(single), Vec::new()),
        };
        match other {
            Self::Sum(next, others) => {
                terms.push(*next);
                terms.extend(others);
            }
            single => terms.push(single),
        }
        Self::Sum(first, terms)
    }

    /// `self * other`.
    #[must_use]
    pub fn times(self, other: Self) -> Self {
        let (first, mut factors) = match self {
            Self::Product(first, factors) => (first, factors),
            single => (Box::new(single), Vec::new()),
        };
        match other {
            Self::Product(next, others) => {
                factors.push(*next);
                factors.extend(others);
            }
            single => factors.push(single),
        }
        Self::Product(first, factors)
    }

    /// The periods of the periodic kernels, in the order of the tree.
    #[must_use]
    pub fn periods(&self) -> Vec<f32> {
        match self {
            Self::Leaf(leaf) => match leaf.base {
                Base::Periodic { period, .. } => vec![period],
                Base::Linear | Base::Matern { .. } => Vec::new(),
            },
            Self::Sum(first, rest) | Self::Product(first, rest) => core::iter::once(&**first)
                .chain(rest)
                .flat_map(Self::periods)
                .collect(),
        }
    }
}

/// The Cholesky factorization of an operator failed with every jitter where
/// linear_operator has no other way (`root_inv_decomposition` in
/// `add_low_rank`): a `NotPSDError`, which TempoPFN answers by trying again
/// with ten times the jitter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NotPositiveDefinite;

/// A draw of the zero-mean process of that kernel at the points of `grid`,
/// with `jitter` for `cholesky_jitter`: `MultivariateNormal.sample()`.
///
/// # Errors
///
/// [`NotPositiveDefinite`], where linear_operator raises `NotPSDError`.
pub fn sample(
    kernel: &Kernel,
    grid: &[f32],
    jitter: f64,
    rng: &mut Rng,
) -> Result<Vec<f32>, NotPositiveDefinite> {
    let matrix = evaluate(kernel, grid, jitter)?.dense();
    Ok(draw(&matrix, jitter, rng))
}

/// The class of linear_operator that a kernel evaluates to, with its dense
/// value. Only the class and the value bear on what follows.
#[derive(Clone, Debug)]
enum Operator {
    /// `DenseLinearOperator`.
    Dense(Lower),
    /// `LowRankRootLinearOperator` of a root of one column: `r rᵀ`.
    Root(Vec<f32>),
    /// `SumLinearOperator`: its terms added from the left.
    Sum(Lower),
    /// `MulLinearOperator`: the product, entry by entry, of two Gram
    /// matrices.
    Mul(Lower),
}

impl Operator {
    /// `to_dense()`.
    fn dense(self) -> Lower {
        match self {
            Self::Dense(matrix) | Self::Sum(matrix) | Self::Mul(matrix) => matrix,
            Self::Root(root) => outer(&root),
        }
    }

    /// `root_decomposition().to_dense()`: what a `MulLinearOperator` keeps
    /// of a factor. A root is its own; any other operator is replaced by the
    /// Gram matrix of its Cholesky factor, jitter included, or of its
    /// eigen-root.
    fn rooted(self, jitter: f64) -> Lower {
        match self {
            Self::Root(root) => outer(&root),
            Self::Dense(matrix) | Self::Sum(matrix) | Self::Mul(matrix) => ladder(&matrix, jitter)
                .map_or_else(|| positive_part(&matrix), |factor| factor.gram()),
        }
    }
}

/// `r rᵀ`, as `torch.matmul` of a column and a row gives it.
fn outer(root: &[f32]) -> Lower {
    Lower::from_fn(root.len(), |row, column| root[row] * root[column])
}

/// The kernel evaluated on the grid, as `Kernel.__call__` evaluates it under
/// `lazily_evaluate_kernels(False)`.
fn evaluate(kernel: &Kernel, grid: &[f32], jitter: f64) -> Result<Operator, NotPositiveDefinite> {
    match kernel {
        Kernel::Leaf(leaf) => Ok(leaf.operator(grid)),
        Kernel::Sum(first, rest) => {
            // `AdditiveKernel.forward`: a zero operator plus each term.
            let mut total = evaluate(first, grid, jitter)?;
            for term in rest {
                let next = evaluate(term, grid, jitter)?;
                total = add(total, next, jitter)?;
            }
            Ok(total)
        }
        Kernel::Product(first, rest) => {
            let mut total = evaluate(first, grid, jitter)?;
            for factor in rest {
                let next = evaluate(factor, grid, jitter)?;
                total = multiply(total, next, jitter);
            }
            Ok(total)
        }
    }
}

/// `total + next` (`__add__` of the class of `total`).
fn add(total: Operator, next: Operator, jitter: f64) -> Result<Operator, NotPositiveDefinite> {
    Ok(match (total, next) {
        (Operator::Dense(mut matrix), Operator::Dense(other)) => {
            accumulate(&mut matrix, &other);
            Operator::Dense(matrix)
        }
        (Operator::Sum(mut matrix), next) => {
            accumulate(&mut matrix, &next.dense());
            Operator::Sum(matrix)
        }
        (total, next) => {
            // `LinearOperator.__add__` hands a root to `add_low_rank`, which
            // updates the root and the inverse root of `total` and so takes
            // its Cholesky factor: with no fallback for the inverse. The sum
            // itself is a plain `SumLinearOperator`; its cached roots are
            // lost when it is next evaluated. A root `r rᵀ` plus a jitter
            // always has a factor, so only the other classes are factored.
            if let (Operator::Dense(matrix) | Operator::Mul(matrix), Operator::Root(_)) =
                (&total, &next)
                && ladder(matrix, jitter).is_none()
            {
                return Err(NotPositiveDefinite);
            }
            let mut matrix = total.dense();
            accumulate(&mut matrix, &next.dense());
            Operator::Sum(matrix)
        }
    })
}

/// `total * next` (`LinearOperator._mul_matrix`): dense if either is, else a
/// `MulLinearOperator` of their roots.
fn multiply(total: Operator, next: Operator, jitter: f64) -> Operator {
    match (total, next) {
        (Operator::Dense(mut matrix), other) | (other, Operator::Dense(mut matrix)) => {
            scale(&mut matrix, &other.dense());
            Operator::Dense(matrix)
        }
        (total, next) => {
            let mut matrix = total.rooted(jitter);
            scale(&mut matrix, &next.rooted(jitter));
            Operator::Mul(matrix)
        }
    }
}

/// `matrix += other`, entry by entry.
fn accumulate(matrix: &mut Lower, other: &Lower) {
    for (value, term) in matrix.values_mut().iter_mut().zip(other.values()) {
        *value += term;
    }
}

/// `matrix *= other`, entry by entry.
fn scale(matrix: &mut Lower, other: &Lower) {
    for (value, factor) in matrix.values_mut().iter_mut().zip(other.values()) {
        *value *= factor;
    }
}

/// `psd_safe_cholesky`: the Cholesky factor of the matrix, or of the matrix
/// with `jitter`, then 10, then 100 times `jitter` on its diagonal, the first
/// that exists. The jitters are added one on the other in float32, as the
/// differences that Python computes in float64.
fn ladder(matrix: &Lower, jitter: f64) -> Option<Lower> {
    let mut factor = matrix.clone();
    if factor.cholesky().is_ok() {
        return Some(factor);
    }
    let mut shifted = matrix.clone();
    let mut previous = 0.0;
    for tries in 0..TRIES {
        let next = jitter * 10.0_f64.powi(tries);
        let step = (next - previous) as f32;
        for value in shifted.diagonal_mut() {
            *value += step;
        }
        previous = next;
        let mut factor = shifted.clone();
        if factor.cholesky().is_ok() {
            return Some(factor);
        }
    }
    None
}

/// The eigen-decomposition of the matrix in float64, as `_symeig` takes it.
fn eigen(matrix: &Lower) -> SymmetricEigen {
    SymmetricEigen::new(matrix.size(), |row, column| {
        f64::from(matrix.row(row)[column])
    })
}

/// The square root of an eigenvalue cut at zero, in float32 as the root of
/// `root_decomposition(method="symeig")` has it.
fn cut(value: f64) -> f32 {
    (value.max(0.0) as f32).sqrt()
}

/// The Gram matrix of the eigen-root `V √max(Λ, 0)`, itself rounded to
/// float32: the matrix without its negative eigenvalues.
fn positive_part(matrix: &Lower) -> Lower {
    let size = matrix.size();
    let eigen = eigen(matrix);
    // The root by rows.
    let mut root = vec![0.0_f32; size * size];
    let mut unit = vec![0.0; size];
    for (column, value) in eigen.values().iter().enumerate() {
        let weight = cut(*value);
        if weight == 0.0 {
            continue;
        }
        unit[column] = 1.0;
        for (row, entry) in eigen.combine(&unit).into_iter().enumerate() {
            root[row * size + column] = entry as f32 * weight;
        }
        unit[column] = 0.0;
    }
    Lower::from_fn(size, |row, column| {
        dot(
            &root[row * size..(row + 1) * size],
            &root[column * size..(column + 1) * size],
        )
    })
}

/// A draw from the normal law of that covariance matrix:
/// `zero_mean_mvn_samples`, the root times a standard normal vector.
fn draw(matrix: &Lower, jitter: f64, rng: &mut Rng) -> Vec<f32> {
    let noise: Vec<f32> = (0..matrix.size()).map(|_| rng.normal() as f32).collect();
    if let Some(factor) = ladder(matrix, jitter) {
        return factor.apply(&noise);
    }
    let eigen = eigen(matrix);
    let weights: Vec<f64> = eigen
        .values()
        .iter()
        .zip(&noise)
        .map(|(value, normal)| f64::from(cut(*value) * normal))
        .collect();
    eigen
        .combine(&weights)
        .into_iter()
        .map(|value| value as f32)
        .collect()
}

impl Leaf {
    /// `forward` of the kernel with both arguments the grid.
    fn operator(self, grid: &[f32]) -> Operator {
        let mut matrix = match self.base {
            Base::Linear => {
                // The root `x √variance`; a `ScaleKernel` multiplies the
                // root by `√outputscale`.
                let deviation = UNIT.sqrt();
                let mut root: Vec<f32> = grid.iter().map(|point| point * deviation).collect();
                if self.scaled {
                    for value in &mut root {
                        *value *= deviation;
                    }
                }
                return Operator::Root(root);
            }
            Base::Matern {
                smoothness,
                lengthscale,
            } => matern(grid, smoothness, lengthscale),
            Base::Periodic {
                period,
                lengthscale,
            } => periodic(grid, period, lengthscale),
        };
        if self.scaled {
            for value in matrix.values_mut() {
                *value *= UNIT;
            }
        }
        Operator::Dense(matrix)
    }
}

/// The values less their mean, in float32.
fn centered(values: &[f32]) -> Vec<f32> {
    let total: f64 = values.iter().map(|value| f64::from(*value)).sum();
    let mean = (total / values.len() as f64) as f32;
    values.iter().map(|value| value - mean).collect()
}

/// `dist(x, x)` of gpytorch for points on a line: `sq_dist` centres the
/// points, takes each squared distance as the product of the rows
/// `(−2a, a², 1)` and `(b, 1, b²)`, the three terms added in this order,
/// puts 0 on the diagonal and cuts the negative values; `dist` takes the
/// square root of that, cut at 10⁻³⁰.
fn distances(points: &[f32]) -> Lower {
    const FLOOR: f32 = 1.0e-30;
    let points = centered(points);
    let squares: Vec<f32> = points.iter().map(|point| point * point).collect();
    Lower::from_fn(points.len(), |row, column| {
        let square = if row == column {
            0.0
        } else {
            let product = -2.0 * points[row] * points[column];
            ((product + squares[row]) + squares[column]).max(0.0)
        };
        square.max(FLOOR).sqrt()
    })
}

/// `MaternCovariance`: the points centred and divided by the lengthscale,
/// their distances `d` times `√(2ν)`, and the kernel of that.
fn matern(grid: &[f32], smoothness: Smoothness, lengthscale: f32) -> Lower {
    let points: Vec<f32> = centered(grid)
        .into_iter()
        .map(|point| point / lengthscale)
        .collect();
    let mut matrix = distances(&points);
    let factor = match smoothness {
        Smoothness::Half => 1.0,
        Smoothness::ThreeHalves => 3.0_f64.sqrt() as f32,
        Smoothness::FiveHalves => 5.0_f64.sqrt() as f32,
    };
    for value in matrix.values_mut() {
        let scaled = *value * factor;
        let decay = (-scaled).exp();
        *value = match smoothness {
            Smoothness::Half => decay,
            Smoothness::ThreeHalves => (scaled + 1.0) * decay,
            Smoothness::FiveHalves => decay * ((scaled + 1.0) + (scaled * scaled) / 3.0),
        };
    }
    matrix
}

/// `PeriodicKernel.forward`: the points divided by `period / π`, their
/// distances `d`, and `exp(−2 sin²(d) / lengthscale)`.
fn periodic(grid: &[f32], period: f32, lengthscale: f32) -> Lower {
    let ratio = period / core::f32::consts::PI;
    let points: Vec<f32> = grid.iter().map(|point| point / ratio).collect();
    let mut matrix = distances(&points);
    for value in matrix.values_mut() {
        let sine = value.sin();
        *value = ((sine * sine) / lengthscale * -2.0).exp();
    }
    matrix
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `torch.linspace(0, 1, size)` closely enough for these tests.
    fn grid(size: usize) -> Vec<f32> {
        (0..size)
            .map(|index| index as f32 / (size - 1) as f32)
            .collect()
    }

    fn leaf(base: Base) -> Kernel {
        Kernel::Leaf(Leaf {
            base,
            scaled: false,
        })
    }

    const SMOOTH: Base = Base::Periodic {
        period: 0.5,
        lengthscale: 2.0,
    };
    const ROUGH: Base = Base::Matern {
        smoothness: Smoothness::Half,
        lengthscale: 0.1,
    };

    /// The deviation of the part of a series that no line through its
    /// neighbours explains: `(y₀ − 2y₁ + y₂) / √6` is white noise of the
    /// deviation of the noise of `y`, plus the curvature of `y`.
    fn roughness(series: &[f32]) -> f64 {
        let total: f64 = series
            .windows(3)
            .map(|w| f64::from(w[0] - 2.0 * w[1] + w[2]).powi(2) / 6.0)
            .sum();
        (total / (series.len() - 2) as f64).sqrt()
    }

    #[test]
    fn sums_and_products_flatten_as_gpytorch_does() {
        let (a, b, c) = (leaf(Base::Linear), leaf(SMOOTH), leaf(ROUGH));
        let sum = a.clone().plus(b.clone()).plus(c.clone());
        assert_eq!(
            sum,
            Kernel::Sum(Box::new(a.clone()), vec![b.clone(), c.clone()])
        );
        let product = sum.clone().times(a.clone()).times(b.clone());
        assert_eq!(
            product,
            Kernel::Product(Box::new(sum.clone()), vec![a.clone(), b.clone()])
        );
        let nested = product.clone().plus(c.clone().plus(a.clone()));
        assert_eq!(nested, Kernel::Sum(Box::new(product), vec![c, a]));
        assert_eq!(nested.periods(), [0.5, 0.5]);
    }

    #[test]
    fn the_kernels_are_gpytorch_s_on_the_grid() {
        let grid = grid(65);
        let at = |base: Base, row: usize, column: usize| {
            let Operator::Dense(matrix) = (Leaf {
                base,
                scaled: false,
            })
            .operator(&grid) else {
                panic!("a dense kernel");
            };
            f64::from(matrix.row(row)[column])
        };
        // One on the diagonal, and the closed forms off it.
        let distance = 16.0 / 64.0_f64;
        for (smoothness, value) in [
            (Smoothness::Half, (-distance / 0.5).exp()),
            (Smoothness::ThreeHalves, {
                let s = 3.0_f64.sqrt() * distance / 0.5;
                (1.0 + s) * (-s).exp()
            }),
            (Smoothness::FiveHalves, {
                let s = 5.0_f64.sqrt() * distance / 0.5;
                (1.0 + s + s * s / 3.0) * (-s).exp()
            }),
        ] {
            let base = Base::Matern {
                smoothness,
                lengthscale: 0.5,
            };
            assert!((at(base, 40, 40) - 1.0).abs() < 1e-6);
            assert!((at(base, 40, 24) - value).abs() < 1e-5, "{smoothness:?}");
        }
        let base = Base::Periodic {
            period: 0.75,
            lengthscale: 1.5,
        };
        let sine = (core::f64::consts::PI * distance / 0.75).sin();
        assert!((at(base, 9, 9) - 1.0).abs() < 1e-6);
        assert!((at(base, 25, 9) - (-2.0 * sine * sine / 1.5).exp()).abs() < 1e-5);
        // A scale multiplies by ln 2; the linear kernel is a root.
        let scaled = Leaf { base, scaled: true };
        let Operator::Dense(matrix) = scaled.operator(&grid) else {
            panic!("a dense kernel");
        };
        assert!((f64::from(matrix.row(9)[9]) - core::f64::consts::LN_2).abs() < 1e-6);
        let linear = Leaf {
            base: Base::Linear,
            scaled: true,
        };
        let matrix = linear.operator(&grid).dense();
        let expected = core::f64::consts::LN_2.powi(2) * (32.0 / 64.0) * (48.0 / 64.0);
        assert!((f64::from(matrix.row(48)[32]) - expected).abs() < 1e-6);
    }

    #[test]
    fn the_classes_follow_linear_operator() {
        let grid = grid(48);
        let class = |kernel: &Kernel| match evaluate(kernel, &grid, 1e-3) {
            Ok(Operator::Dense(_)) => "dense",
            Ok(Operator::Root(_)) => "root",
            Ok(Operator::Sum(_)) => "sum",
            Ok(Operator::Mul(_)) => "mul",
            Err(NotPositiveDefinite) => "error",
        };
        let (line, smooth, rough) = (leaf(Base::Linear), leaf(SMOOTH), leaf(ROUGH));
        assert_eq!(class(&line), "root");
        assert_eq!(class(&smooth.clone().plus(rough.clone())), "dense");
        assert_eq!(class(&smooth.clone().plus(line.clone())), "sum");
        assert_eq!(class(&line.clone().plus(line.clone())), "sum");
        assert_eq!(class(&smooth.clone().times(line.clone())), "dense");
        assert_eq!(class(&line.clone().times(line.clone())), "mul");
        assert_eq!(
            class(&line.clone().times(line.clone()).plus(rough.clone())),
            "sum"
        );
        let mixed = smooth.plus(line.clone()).times(line);
        assert_eq!(class(&mixed), "mul");
        assert_eq!(class(&mixed.times(rough)), "dense");
    }

    #[test]
    fn a_product_of_roots_takes_the_gram_matrix_of_a_factor_with_its_jitter() {
        let grid = grid(48);
        let (line, smooth) = (leaf(Base::Linear), leaf(SMOOTH));
        // (smooth + line) × line: the sum is singular, so its factor carries
        // the jitter 10⁻³ on the diagonal, times the line's x² ln 2.
        let sum = evaluate(&smooth.clone().plus(line.clone()), &grid, 1e-3)
            .expect("no inverse root is taken of a singular matrix here")
            .dense();
        assert!(sum.clone().cholesky().is_err(), "the sum is singular");
        let product = evaluate(&smooth.plus(line.clone()).times(line), &grid, 1e-3)
            .expect("the product raises nothing")
            .dense();
        for row in [5, 20, 47] {
            let square = f64::from(grid[row]).powi(2) * core::f64::consts::LN_2;
            let diagonal = (f64::from(sum.row(row)[row]) + 1e-3) * square;
            assert!((f64::from(product.row(row)[row]) - diagonal).abs() < 2e-5);
            let other = f64::from(sum.row(row)[2])
                * f64::from(grid[row] * grid[2])
                * core::f64::consts::LN_2;
            assert!((f64::from(product.row(row)[2]) - other).abs() < 2e-5);
        }
    }

    #[test]
    fn a_root_added_to_an_indefinite_operator_raises() {
        let size = 12;
        let indefinite = |shift: f32| {
            Operator::Dense(Lower::from_fn(size, |row, column| {
                if row == column { shift } else { 0.0 }
            }))
        };
        let root = || Operator::Root(vec![1.0; size]);
        // −0.05 is mended by the jitter 0.1, the third of the ladder; −0.5
        // by none.
        assert!(matches!(
            add(indefinite(-0.05), root(), 1e-3),
            Ok(Operator::Sum(_))
        ));
        assert!(matches!(
            add(indefinite(-0.5), root(), 1e-3),
            Err(NotPositiveDefinite)
        ));
        // Ten times the jitter mends it; and no factor is asked of a sum, or
        // when a dense matrix is added.
        assert!(matches!(
            add(indefinite(-0.5), root(), 1e-2),
            Ok(Operator::Sum(_))
        ));
        let Operator::Dense(matrix) = indefinite(-0.5) else {
            panic!("a dense operator");
        };
        assert!(matches!(
            add(Operator::Sum(matrix), root(), 1e-3),
            Ok(Operator::Sum(_))
        ));
        assert!(matches!(
            add(root(), indefinite(-0.5), 1e-3),
            Ok(Operator::Sum(_))
        ));
        assert!(matches!(
            add(indefinite(-0.5), indefinite(-0.5), 1e-3),
            Ok(Operator::Dense(_))
        ));
    }

    #[test]
    fn the_ladder_adds_the_least_jitter_that_gives_a_factor() {
        let diagonal =
            |shift: f32| Lower::from_fn(6, |row, column| if row == column { shift } else { 0.0 });
        let pivot = |shift: f32, jitter: f64| {
            ladder(&diagonal(shift), jitter).map(|factor| f64::from(factor.row(3)[3]).powi(2))
        };
        assert!((pivot(1.0, 1e-3).expect("positive") - 1.0).abs() < 1e-6);
        assert!((pivot(0.0, 1e-3).expect("first jitter") - 1e-3).abs() < 1e-8);
        assert!((pivot(-0.005, 1e-3).expect("second jitter") - 0.005).abs() < 1e-7);
        assert!((pivot(-0.05, 1e-3).expect("third jitter") - 0.05).abs() < 1e-7);
        assert_eq!(pivot(-0.5, 1e-3), None);
        assert!((pivot(-0.5, 1e-2).expect("third jitter") - 0.5).abs() < 1e-6);
    }

    #[test]
    fn a_smooth_process_is_drawn_with_the_noise_of_the_first_jitter() {
        let grid = grid(256);
        let mut rng = Rng::from_key(&[5]);
        let mut total = 0.0;
        for _ in 0..20 {
            let series = sample(&leaf(SMOOTH), &grid, 1e-3, &mut rng).expect("no root is added");
            total += roughness(&series).powi(2) / 20.0;
        }
        // The kernel is smooth: the roughness is the jitter's, 10⁻³.
        assert!((total / 1e-3 - 1.0).abs() < 0.1, "noise variance {total}");
    }

    #[test]
    fn a_rough_process_needs_no_jitter_and_has_the_kernel_s_covariance() {
        let grid = grid(64);
        let mut rng = Rng::from_key(&[6]);
        let Operator::Dense(matrix) = evaluate(&leaf(ROUGH), &grid, 1e-3).expect("a leaf") else {
            panic!("a dense kernel");
        };
        assert!(matrix.clone().cholesky().is_ok());
        let (mut variance, mut covariance) = (0.0, 0.0);
        let draws = 4000;
        for _ in 0..draws {
            let series = sample(&leaf(ROUGH), &grid, 1e-3, &mut rng).expect("a leaf");
            variance += f64::from(series[30]).powi(2) / f64::from(draws);
            covariance += f64::from(series[30] * series[36]) / f64::from(draws);
        }
        assert!((variance - 1.0).abs() < 0.08, "variance {variance}");
        let expected = (-(6.0 / 63.0) / 0.1_f64).exp();
        assert!(
            (covariance - expected).abs() < 0.06,
            "covariance {covariance}"
        );
    }

    #[test]
    fn a_line_is_drawn_through_the_origin_with_the_first_jitter() {
        let grid = grid(256);
        let mut rng = Rng::from_key(&[7]);
        let (mut slopes, mut noise) = (0.0, 0.0);
        let draws = 300;
        for _ in 0..draws {
            let series = sample(&leaf(Base::Linear), &grid, 1e-3, &mut rng).expect("a leaf");
            slopes += f64::from(series[255]).powi(2) / f64::from(draws);
            noise += roughness(&series).powi(2) / f64::from(draws);
        }
        assert!(
            (slopes / core::f64::consts::LN_2 - 1.0).abs() < 0.25,
            "{slopes}"
        );
        assert!((noise / 1e-3 - 1.0).abs() < 0.1, "noise variance {noise}");
    }

    #[test]
    fn an_indefinite_matrix_is_drawn_from_its_positive_part() {
        // A matrix with the eigenvalues 3, 1 and −2 on orthogonal vectors,
        // and zeros: no jitter of the ladder mends −2.
        let size = 24;
        let mut rng = Rng::from_key(&[8]);
        let vectors: Vec<Vec<f64>> = (0..3)
            .map(|k| {
                (0..size)
                    .map(|i| {
                        let angle = core::f64::consts::PI * ((k + 1) * (2 * i + 1)) as f64
                            / (2 * size) as f64;
                        angle.cos() * (2.0 / size as f64).sqrt()
                    })
                    .collect()
            })
            .collect();
        let values = [3.0, 1.0, -2.0];
        let entry = |row: usize, column: usize, values: &[f64; 3]| -> f64 {
            (0..3)
                .map(|k| values[k] * vectors[k][row] * vectors[k][column])
                .sum()
        };
        let matrix = Lower::from_fn(size, |row, column| entry(row, column, &values) as f32);
        assert!(ladder(&matrix, 1e-3).is_none());
        let positive = positive_part(&matrix);
        for row in 0..size {
            for column in 0..=row {
                let expected = entry(row, column, &[3.0, 1.0, 0.0]);
                assert!((f64::from(positive.row(row)[column]) - expected).abs() < 1e-5);
            }
        }
        // The draws have the covariance of the positive part.
        let draws = 6000;
        let mut covariance = vec![0.0; 3];
        for _ in 0..draws {
            let series = draw(&matrix, 1e-3, &mut rng);
            for (k, total) in covariance.iter_mut().enumerate() {
                let along: f64 = series
                    .iter()
                    .zip(&vectors[k])
                    .map(|(y, v)| f64::from(*y) * v)
                    .sum();
                *total += along * along / f64::from(draws);
            }
        }
        assert!((covariance[0] - 3.0).abs() < 0.25, "{covariance:?}");
        assert!((covariance[1] - 1.0).abs() < 0.1, "{covariance:?}");
        assert!(covariance[2] < 1e-6, "{covariance:?}");
    }
}
