//! The matrix product in single precision on one thread, `C = A B` or
//! `C += A B`, blocked as GotoBLAS and BLIS block it, so that each operand is
//! read from the cache that holds it: the right operand is packed in strips
//! of [`KC`] rows by [`NC`] columns, the left in blocks of [`MC`] rows by
//! [`KC`] columns, and the micro-kernel of [`crate::kernels`] multiplies a
//! sliver of each into a tile of the product.
//!
//! The engine runs one product per thread at a time, on as many threads as
//! there are cores: a thread keeps a [`Gemm`], which owns the packed blocks.

use diff_gpt_forecast_domain::model::{Matrix, Product};

use crate::kernels::{self, MR, NR};

/// Steps of the inner dimension a block takes.
const KC: usize = 256;
/// Rows of the left operand packed at a time: with [`KC`] columns, 172 kB of
/// the 256 kB of a core's second-level cache.
const MC: usize = 168;
/// Columns of the right operand packed at a time: with [`KC`] rows, 2 MB, a
/// core's share of the third-level cache.
const NC: usize = 2048;
/// Floats in a line of the cache.
const LINE: usize = 16;

/// The line a matrix lacks past its last row or column.
static ZEROS: [f32; KC] = [0.0; KC];

/// Weaves lines of floats of one length: `to` takes the first float of
/// each line, then the second of each, and so on. A matrix is either rows
/// in lines or, as a transpose, columns in lines ([`Matrix`]); a sliver
/// needs the floats of its step together.
fn weave<const LINES: usize>(lines: [&[f32]; LINES], to: &mut [f32]) {
    for (step, to) in to.as_chunks_mut::<LINES>().0.iter_mut().enumerate() {
        for (slot, line) in to.iter_mut().zip(&lines) {
            *slot = line[step];
        }
    }
}

/// The working memory of the products of one thread.
#[derive(Clone, Debug)]
pub struct Gemm {
    left: Vec<f32>,
    right: Vec<f32>,
    /// The first float of `right` that starts a line of the cache.
    start: usize,
}

impl Default for Gemm {
    fn default() -> Self {
        Self::new()
    }
}

impl Gemm {
    /// The working memory, a little over 2 MB.
    ///
    /// # Panics
    ///
    /// If the CPU lacks AVX2 or FMA.
    #[must_use]
    pub fn new() -> Self {
        assert!(kernels::supported(), "the kernels need AVX2 and FMA");
        let right = vec![0.0_f32; KC * NC.next_multiple_of(NR) + LINE];
        let start = right.as_ptr().align_offset(4 * LINE).min(LINE);
        Self {
            left: vec![0.0; MC.next_multiple_of(MR) * KC],
            right,
            start,
        }
    }
}

impl Product for Gemm {
    fn multiply_spaced(
        &mut self,
        c: &mut [f32],
        stride: usize,
        a: Matrix<'_>,
        b: Matrix<'_>,
        keep: bool,
    ) {
        let (m, k, n) = (a.rows(), a.columns(), b.columns());
        assert_eq!(b.rows(), k, "the inner dimensions differ");
        if m == 0 || n == 0 {
            return;
        }
        assert!(
            stride >= n,
            "rows of {n} floats do not fit a stride of {stride}"
        );
        assert!(c.len() >= (m - 1) * stride + n, "the product ends early");
        if k == 0 {
            if !keep {
                c.chunks_mut(stride)
                    .take(m)
                    .for_each(|row| row[..n].fill(0.0));
            }
            return;
        }
        for first_column in (0..n).step_by(NC) {
            let columns = NC.min(n - first_column);
            for first_step in (0..k).step_by(KC) {
                let steps = KC.min(k - first_step);
                self.pack_right(b, first_step, steps, first_column, columns);
                // The second block of steps goes on from the first.
                let keep = keep || first_step > 0;
                for first_row in (0..m).step_by(MC) {
                    let rows = MC.min(m - first_row);
                    self.pack_left(a, first_row, rows, first_step, steps);
                    self.block(
                        c,
                        stride,
                        (first_row, rows),
                        (first_column, columns),
                        steps,
                        keep,
                    );
                }
            }
        }
    }
}

impl Gemm {
    /// The packed blocks multiplied into their part of `c`, rows of `n`
    /// floats: the rows and the columns are each a first one and a count.
    fn block(
        &self,
        c: &mut [f32],
        n: usize,
        (first_row, rows): (usize, usize),
        (first_column, columns): (usize, usize),
        steps: usize,
        keep: bool,
    ) {
        let right = &self.right[self.start..];
        for (strip, right) in right
            .chunks(NR * steps)
            .take(columns.div_ceil(NR))
            .enumerate()
        {
            let column = first_column + strip * NR;
            let width = NR.min(first_column + columns - column);
            for (sliver, left) in self
                .left
                .chunks(MR * steps)
                .take(rows.div_ceil(MR))
                .enumerate()
            {
                let row = first_row + sliver * MR;
                let height = MR.min(first_row + rows - row);
                if height == MR && width == NR {
                    kernels::tile(&mut c[row * n + column..], n, left, right, steps, keep);
                    continue;
                }
                // A tile over the edge of `c`: made apart, and what lies in
                // `c` moved there.
                let mut edge = [0.0_f32; MR * NR];
                kernels::tile(&mut edge, NR, left, right, steps, false);
                for (line, made) in edge.chunks(NR).take(height).enumerate() {
                    let to = &mut c[(row + line) * n + column..][..width];
                    for (slot, value) in to.iter_mut().zip(made) {
                        *slot = if keep { *slot + value } else { *value };
                    }
                }
            }
        }
    }

    /// Packs `steps` columns of `rows` rows of `a` into slivers of [`MR`]
    /// rows, a sliver's steps one after another: zeros past the last row.
    fn pack_left(
        &mut self,
        a: Matrix<'_>,
        first_row: usize,
        rows: usize,
        first_step: usize,
        steps: usize,
    ) {
        let ((row_step, column_step), data) = (a.steps(), a.data());
        let slivers = self.left.chunks_mut(MR * steps).take(rows.div_ceil(MR));
        for (sliver, packed) in slivers.enumerate() {
            let row = first_row + sliver * MR;
            let height = MR.min(first_row + rows - row);
            if row_step != 1 {
                // Each row is a line of the floats: the lines are woven.
                let lines: [&[f32]; MR] = core::array::from_fn(|line| {
                    if line < height {
                        &data[(row + line) * row_step + first_step * column_step..][..steps]
                    } else {
                        &ZEROS[..steps]
                    }
                });
                weave(lines, packed);
                continue;
            }
            // The rows lie side by side, as in a transpose: a step of the
            // sliver is a piece of one line, and the floats are read in the
            // order they lie in.
            let first = row + first_step * column_step;
            if height == MR {
                for (step, to) in packed.as_chunks_mut::<MR>().0.iter_mut().enumerate() {
                    let from = &data[first + step * column_step..][..MR];
                    let from: &[f32; MR] = from.try_into().expect("a sliver's rows");
                    to.copy_from_slice(from);
                }
            } else {
                for (step, to) in packed.as_chunks_mut::<MR>().0.iter_mut().enumerate() {
                    let (taken, rest) = to.split_at_mut(height);
                    taken.copy_from_slice(&data[first + step * column_step..][..height]);
                    rest.fill(0.0);
                }
            }
        }
    }

    /// Packs `steps` rows of `columns` columns of `b` into slivers of [`NR`]
    /// columns, a sliver's steps one after another: zeros past the last
    /// column.
    fn pack_right(
        &mut self,
        b: Matrix<'_>,
        first_step: usize,
        steps: usize,
        first_column: usize,
        columns: usize,
    ) {
        let ((row_step, column_step), data) = (b.steps(), b.data());
        let right = &mut self.right[self.start..];
        let slivers = right.chunks_mut(NR * steps).take(columns.div_ceil(NR));
        for (sliver, packed) in slivers.enumerate() {
            let column = first_column + sliver * NR;
            let width = NR.min(first_column + columns - column);
            if column_step != 1 {
                // Each column is a line of the floats, as in a transpose.
                let lines: [&[f32]; NR] = core::array::from_fn(|line| {
                    if line < width {
                        &data[(column + line) * column_step + first_step * row_step..][..steps]
                    } else {
                        &ZEROS[..steps]
                    }
                });
                weave(lines, packed);
                continue;
            }
            // Each row is a line: a step of the sliver is a piece of one.
            let first = first_step * row_step + column;
            if width == NR {
                for (step, to) in packed.as_chunks_mut::<NR>().0.iter_mut().enumerate() {
                    let from = &data[first + step * row_step..][..NR];
                    let from: &[f32; NR] = from.try_into().expect("a sliver's columns");
                    to.copy_from_slice(from);
                }
            } else {
                for (step, to) in packed.as_chunks_mut::<NR>().0.iter_mut().enumerate() {
                    let (taken, rest) = to.split_at_mut(width);
                    taken.copy_from_slice(&data[first + step * row_step..][..width]);
                    rest.fill(0.0);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use diff_gpt_forecast_domain::model::sequence::Workspace;
    use diff_gpt_forecast_domain::model::{Config, Plain};
    use diff_gpt_forecast_domain::random::Rng;

    use super::*;

    /// Pseudo-random floats in [-1, 1).
    fn floats(count: usize, seed: u32) -> Vec<f32> {
        let mut state = seed.wrapping_mul(2_654_435_761).max(1);
        (0..count)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state >> 8) as f32 / (1 << 23) as f32 - 1.0
            })
            .collect()
    }

    /// The product by the definition, in double precision.
    fn reference(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f64> {
        let mut c = vec![0.0_f64; m * n];
        for i in 0..m {
            for l in 0..k {
                let left = f64::from(a[i * k + l]);
                for j in 0..n {
                    c[i * n + j] += left * f64::from(b[l * n + j]);
                }
            }
        }
        c
    }

    fn transpose(x: &[f32], rows: usize, columns: usize) -> Vec<f32> {
        let mut t = vec![0.0; x.len()];
        for i in 0..rows {
            for j in 0..columns {
                t[j * rows + i] = x[i * columns + j];
            }
        }
        t
    }

    #[test]
    fn products_of_every_edge_match_the_definition() {
        let mut gemm = Gemm::new();
        let shapes = [
            (1, 1, 1),
            (5, 3, 7),
            (6, 4, 16),
            (7, 5, 17),
            (13, 300, 33),
            (200, 17, 50),
            (170, 260, 40),
            (12, 2, 2100),
        ];
        for (index, &(m, k, n)) in shapes.iter().enumerate() {
            let a = floats(m * k, index as u32 + 1);
            let b = floats(k * n, index as u32 + 100);
            let exact = reference(&a, &b, m, k, n);
            let close = |c: &[f32], offset: f64| {
                c.iter().zip(&exact).all(|(got, want)| {
                    (f64::from(*got) - want - offset).abs() < 1e-4 * (1.0 + want.abs())
                })
            };
            let mut c = vec![0.5_f32; m * n];
            gemm.multiply(&mut c, Matrix::new(&a, m, k), Matrix::new(&b, k, n), false);
            assert!(close(&c, 0.0), "{m} by {k} by {n}");
            // Added to what is there.
            gemm.multiply(&mut c, Matrix::new(&a, m, k), Matrix::new(&b, k, n), true);
            let twice: Vec<f32> = c.iter().map(|value| value / 2.0).collect();
            assert!(close(&twice, 0.0), "{m} by {k} by {n}, kept");
            // Either operand through its transpose.
            let (at, bt) = (transpose(&a, m, k), transpose(&b, k, n));
            let mut c = vec![0.0_f32; m * n];
            gemm.multiply(
                &mut c,
                Matrix::new(&at, k, m).transposed(),
                Matrix::new(&bt, n, k).transposed(),
                false,
            );
            assert!(close(&c, 0.0), "{m} by {k} by {n}, transposed");
            let mut c = vec![1.0_f32; m * n];
            gemm.multiply(
                &mut c,
                Matrix::new(&a, m, k),
                Matrix::new(&bt, n, k).transposed(),
                true,
            );
            assert!(
                close(&c, 1.0),
                "{m} by {k} by {n}, right transposed and kept"
            );
        }
    }

    #[test]
    fn empty_products_leave_or_clear_the_result() {
        let mut gemm = Gemm::new();
        let mut c = vec![3.0_f32; 6];
        gemm.multiply(&mut c, Matrix::new(&[], 2, 0), Matrix::new(&[], 0, 3), true);
        assert_eq!(c, [3.0; 6]);
        gemm.multiply(
            &mut c,
            Matrix::new(&[], 2, 0),
            Matrix::new(&[], 0, 3),
            false,
        );
        assert_eq!(c, [0.0; 6]);
        let mut none: Vec<f32> = Vec::new();
        gemm.multiply(
            &mut none,
            Matrix::new(&[], 0, 4),
            Matrix::new(&[0.0; 12], 4, 3),
            false,
        );
        assert!(none.is_empty());
    }

    #[test]
    fn blocks_of_wider_matrices_multiply_into_spaced_rows() {
        let mut gemm = Gemm::new();
        for (m, k, n) in [(7, 5, 17), (50, 20, 33), (9, 300, 4)] {
            // The operands are blocks of matrices four columns wider, and
            // the product goes into rows three floats wider.
            let (wide_a, wide_b) = (floats((m + 1) * (k + 4), 7), floats((n + 1) * (k + 4), 8));
            let a = Matrix::spaced(&wide_a[2..], m, k, k + 4);
            // The right operand through the transpose of a block.
            let b = Matrix::spaced(&wide_b[3..], n, k, k + 4).transposed();
            let stride = n + 3;
            let mut exact = vec![-2.0_f32; m * stride];
            let mut made = exact.clone();
            Plain.multiply_spaced(&mut exact, stride, a, b, false);
            gemm.multiply_spaced(&mut made, stride, a, b, false);
            for (index, (got, want)) in made.iter().zip(&exact).enumerate() {
                assert!(
                    (got - want).abs() < 1e-4 * (1.0 + want.abs()),
                    "{m} by {k} by {n}"
                );
                // The floats between the rows stay.
                assert!(index % stride < n || *got == -2.0);
            }
            Plain.multiply_spaced(&mut exact, stride, a, b, true);
            gemm.multiply_spaced(&mut made, stride, a, b, true);
            let close = made.iter().zip(&exact);
            assert!(
                close
                    .clone()
                    .all(|(got, want)| (got - want).abs() < 1e-4 * (1.0 + want.abs()))
            );
        }
    }

    #[test]
    fn the_model_runs_on_the_kernels_as_by_the_definition() {
        let config = Config {
            vocab: 40,
            width: 32,
            heads: 4,
            layers: 2,
            bins: 0,
        };
        let mut rng = Rng::from_key(&[5]);
        let parameters: Vec<f32> = (0..config.parameters())
            .map(|_| rng.normal_with(0.0, 0.2) as f32)
            .collect();
        let tokens: Vec<u32> = (0..110).map(|_| rng.index(config.vocab) as u32).collect();
        let targets: Vec<u32> = tokens.iter().skip(1).chain(&tokens[..1]).copied().collect();
        let weights = vec![1.0 / 110.0_f32; 110];
        let mut workspace = Workspace::new(config, 110);
        let mut run = |product: &mut dyn FnMut(&mut Workspace, &mut [f32]) -> f64| {
            let mut gradient = vec![0.0_f32; config.parameters()];
            let loss = product(&mut workspace, &mut gradient);
            (loss, gradient)
        };
        let (plain_loss, plain) = run(&mut |workspace, gradient| {
            workspace.forward(&mut Plain, &parameters, &tokens, &[]);
            let loss = workspace.loss(&targets, &weights);
            workspace.backward(&mut Plain, &parameters, gradient);
            loss
        });
        let mut gemm = Gemm::new();
        let (fast_loss, fast) = run(&mut |workspace, gradient| {
            workspace.forward(&mut gemm, &parameters, &tokens, &[]);
            let loss = workspace.loss(&targets, &weights);
            workspace.backward(&mut gemm, &parameters, gradient);
            loss
        });
        assert!(
            (plain_loss - fast_loss).abs() < 1e-5,
            "{plain_loss} and {fast_loss}"
        );
        assert!(plain_loss > 2.0, "a loss of {plain_loss} on random weights");
        let scale = plain.iter().fold(0.0_f32, |a, b| a.max(b.abs()));
        assert!(scale > 1e-3);
        for (fast, plain) in fast.iter().zip(&plain) {
            assert!((fast - plain).abs() < 1e-4 * scale, "{fast} and {plain}");
        }
    }

    /// A product that keeps the time and the work of its kinds of products.
    struct Timed {
        inner: Gemm,
        /// Seconds and operations of the products of weights whose left
        /// operand is read along its rows, of those whose left operand is
        /// a transpose, and of the bands of attention.
        kinds: [(f64, f64); 3],
    }

    impl Product for Timed {
        fn multiply_spaced(
            &mut self,
            c: &mut [f32],
            stride: usize,
            a: Matrix<'_>,
            b: Matrix<'_>,
            keep: bool,
        ) {
            // A head of the model measured is 64 wide, and no product of
            // its weights has that width.
            let kind = if a.columns() == 64 || b.columns() == 64 {
                2
            } else {
                usize::from(a.steps().0 == 1)
            };
            let start = std::time::Instant::now();
            self.inner.multiply_spaced(c, stride, a, b, keep);
            self.kinds[kind].0 += start.elapsed().as_secs_f64();
            self.kinds[kind].1 += (2 * a.rows() * a.columns() * b.columns()) as f64;
        }
    }

    /// Where the time of a training sequence goes on one thread:
    /// `cargo test --release -p diff-gpt-forecast-infrastructure sequence_speed -- --ignored --nocapture`.
    #[test]
    #[ignore = "a measurement, not a check"]
    fn sequence_speed() {
        let config = Config {
            vocab: 256,
            width: 256,
            heads: 4,
            layers: 6,
            bins: 0,
        };
        let length = 1412;
        let mut rng = Rng::from_key(&[1]);
        let parameters = config.initial(&mut rng);
        let tokens: Vec<u32> = (0..length)
            .map(|_| rng.index(config.vocab) as u32)
            .collect();
        let weights = vec![1.0_f32; length];
        let mut workspace = Workspace::new(config, length);
        let mut gradient = vec![0.0_f32; config.parameters()];
        let mut product = Timed {
            inner: Gemm::new(),
            kinds: [(0.0, 0.0); 3],
        };
        for turn in 0..3 {
            product.kinds = [(0.0, 0.0); 3];
            let start = std::time::Instant::now();
            workspace.forward(&mut product, &parameters, &tokens, &[]);
            let forward = start.elapsed().as_secs_f64();
            let in_forward = product.kinds;
            workspace.loss(&tokens, &weights);
            let lost = start.elapsed().as_secs_f64();
            workspace.backward(&mut product, &parameters, &mut gradient);
            let all = start.elapsed().as_secs_f64();
            let estimate = 2.0 * (config.products(length) * length) as f64;
            println!(
                "turn {turn}: {all:.3} s, {:.1} GFLOPS by the estimate; forward {forward:.3} s, loss {:.3} s, backward {:.3} s",
                estimate / all / 1e9,
                lost - forward,
                all - lost
            );
            let names = ["rows", "transposed", "bands"];
            let mut inside = 0.0;
            for (kind, name) in names.iter().enumerate() {
                let (seconds, flops) = product.kinds[kind];
                inside += seconds;
                println!(
                    "  {name:<10} {seconds:.3} s  {:6.1} GFLOPS  ({:.3} s of it forward)",
                    flops / seconds / 1e9,
                    in_forward[kind].0
                );
            }
            println!("  outside the products: {:.3} s", all - inside);
        }
    }

    /// The speed of each product of one attention head of 64 channels at
    /// 1412 positions, band by band as the model makes them:
    /// `cargo test --release -p diff-gpt-forecast-infrastructure band_speed -- --ignored --nocapture`.
    #[test]
    #[ignore = "a measurement, not a check"]
    fn band_speed() {
        let (t, dh, d3, band) = (1412_usize, 64_usize, 768_usize, 96_usize);
        let mut gemm = Gemm::new();
        let qkv = floats(t * d3, 1);
        let mut weights = floats(t * t, 2);
        let narrow = floats(t * dh, 3);
        let mut out = vec![0.0_f32; t * dh];
        let bands: Vec<(usize, usize)> = (0..t)
            .step_by(band)
            .map(|start| (start, (start + band).min(t)))
            .collect();
        let work = bands
            .iter()
            .map(|(start, end)| 2 * (end - start) * end * dh)
            .sum::<usize>() as f64;
        let mut time = |name: &str, run: &mut dyn FnMut(&mut Gemm)| {
            let mut best = f64::INFINITY;
            for _ in 0..5 {
                let start = std::time::Instant::now();
                run(&mut gemm);
                best = best.min(start.elapsed().as_secs_f64());
            }
            println!(
                "{name:<28} {:6.2} ms  {:5.1} GFLOPS",
                best * 1e3,
                work / best / 1e9
            );
        };
        time("scores: Q K^T", &mut |gemm| {
            for &(start, end) in &bands {
                gemm.multiply_spaced(
                    &mut weights[start * t + start..],
                    t,
                    Matrix::spaced(&qkv[start * d3..], t - start, dh, d3),
                    Matrix::spaced(&qkv[start * d3 + 256..], end - start, dh, d3).transposed(),
                    false,
                );
            }
        });
        time("at the weights: dA V^T", &mut |gemm| {
            for &(start, end) in &bands {
                gemm.multiply_spaced(
                    &mut weights[start * t + start..],
                    t,
                    Matrix::new(&narrow[start * dh..t * dh], t - start, dh),
                    Matrix::spaced(&qkv[start * d3 + 512..], end - start, dh, d3).transposed(),
                    false,
                );
            }
        });
        let weights = floats(t * t, 4);
        time("attended: P V", &mut |gemm| {
            for &(start, end) in &bands {
                gemm.multiply(
                    &mut out[start * dh..end * dh],
                    Matrix::spaced(&weights[start * t..], end - start, end, t),
                    Matrix::spaced(&qkv[512..], end, dh, d3),
                    false,
                );
            }
        });
        time("at the values: P^T dA", &mut |gemm| {
            for &(start, end) in &bands {
                gemm.multiply(
                    &mut out[start * dh..end * dh],
                    Matrix::spaced(&weights[start * t + start..], t - start, end - start, t)
                        .transposed(),
                    Matrix::new(&narrow[start * dh..t * dh], t - start, dh),
                    false,
                );
            }
        });
        time("at the keys: dS^T Q", &mut |gemm| {
            for &(start, end) in &bands {
                gemm.multiply(
                    &mut out[start * dh..end * dh],
                    Matrix::spaced(&weights[start * t + start..], t - start, end - start, t)
                        .transposed(),
                    Matrix::spaced(&qkv[start * d3..], t - start, dh, d3),
                    false,
                );
            }
        });
    }

    /// The speed of a product on one thread:
    /// `cargo test --release -p diff-gpt-forecast-infrastructure speed -- --ignored --nocapture`.
    #[test]
    #[ignore = "a measurement, not a check"]
    fn speed() {
        let mut gemm = Gemm::new();
        for (m, k, n) in [
            (512, 256, 1366),
            (512, 683, 256),
            (512, 64, 512),
            (512, 512, 64),
            (2048, 256, 1024),
        ] {
            let (a, b) = (floats(m * k, 1), floats(k * n, 2));
            let mut c = vec![0.0_f32; m * n];
            let turns = (4.0e9 / (2 * m * k * n) as f64).ceil() as usize;
            let mut best = f64::INFINITY;
            for _ in 0..5 {
                let start = std::time::Instant::now();
                for _ in 0..turns {
                    gemm.multiply(&mut c, Matrix::new(&a, m, k), Matrix::new(&b, k, n), false);
                }
                best = best.min(start.elapsed().as_secs_f64() / turns as f64);
            }
            println!(
                "{m:>5} x {k:>4} x {n:>5}: {:6.1} GFLOPS",
                (2 * m * k * n) as f64 / best / 1e9
            );
        }
    }
}
