//! The model: DiffGPT's plain GPT (`src/diff_gpt/domain/model/gpt.py`, class
//! `GPT`), a decoder-only transformer over tokens.
//!
//! - The token embedding is tied to the readout; no positions are added.
//! - A block adds two branches to the residual stream, each reading it
//!   through a parameter-free RMS normalization: causal attention, then a
//!   SwiGLU feed-forward network of hidden width `(8d + 1) / 3`.
//! - Attention: queries and keys are normalized to unit RMS per head, rotated
//!   by their position (RoPE, base 10⁴, pairs `i` and `i + d_h/2`), and
//!   scored by `q·k / √d_h`, which is the cosine at the fixed temperature
//!   `√d_h`; the attended value loses its component along the token's own
//!   value (exclusive self-attention), then is projected.
//! - The logits are the final normalized state times the embedding.
//! - Every matrix is drawn from a normal distribution of deviation 0.02, and
//!   none has a bias.
//!
//! One addition, off unless asked for: a second embedding, of the bin of
//! the level a token leads to ([`crate::codec`]), added to the token's at
//! the input. It is not tied to the readout, which predicts tokens alone.
//!
//! Here the model is its arithmetic on one sequence, forward and backward
//! ([`sequence`]), over a port for the matrix product ([`Product`]), which
//! the infrastructure fills with kernels. The domain's own product,
//! [`Plain`], is the definition the kernels are checked against.

pub mod math;
pub mod optimizer;
pub mod sequence;

use crate::random::Rng;

/// The deviation of the initial weights.
pub const INITIAL: f64 = 0.02;

/// The base of the rotary positions.
pub const ROPE_BASE: f64 = 10_000.0;

/// The sizes of a model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Config {
    /// The number of tokens.
    pub vocab: usize,
    /// The width of the residual stream.
    pub width: usize,
    /// The number of attention heads of a block.
    pub heads: usize,
    /// The number of blocks.
    pub layers: usize,
    /// The number of level bins with an embedding of their own, added to
    /// the token's at the input; 0 for a model of tokens alone.
    pub bins: usize,
}

/// Where the matrices of a block lie among the parameters: each is rows of
/// its output width, a row for each input channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Block {
    /// The queries, keys and values: `width` rows of `3 · width`.
    pub qkv: usize,
    /// The projection of the attended values: `width` rows of `width`.
    pub projection: usize,
    /// The gates and the values of the feed-forward network: `width` rows of
    /// `2 · hidden`.
    pub expansion: usize,
    /// The output of the feed-forward network: `hidden` rows of `width`.
    pub contraction: usize,
}

impl Config {
    /// Checks the sizes.
    ///
    /// # Errors
    ///
    /// If a size is 0, the heads do not divide the width, or a head has an
    /// odd width, which RoPE cannot rotate in pairs.
    pub fn check(&self) -> Result<(), String> {
        if self.vocab == 0 || self.width == 0 || self.heads == 0 || self.layers == 0 {
            return Err(format!("a size of {self:?} is 0"));
        }
        if !self.width.is_multiple_of(self.heads) || !self.head().is_multiple_of(2) {
            return Err(format!(
                "{} heads do not split the width {} into even parts",
                self.heads, self.width
            ));
        }
        Ok(())
    }

    /// The width of a head.
    #[must_use]
    pub const fn head(&self) -> usize {
        self.width / self.heads
    }

    /// The hidden width of the feed-forward network: two thirds of `4 · width`.
    #[must_use]
    pub const fn hidden(&self) -> usize {
        (8 * self.width + 1) / 3
    }

    /// The number of parameters of the embedding, which comes first.
    #[must_use]
    pub const fn embedding(&self) -> usize {
        self.vocab * self.width
    }

    /// The number of parameters of a block.
    #[must_use]
    pub const fn block_size(&self) -> usize {
        4 * self.width * self.width + 3 * self.hidden() * self.width
    }

    /// Where the matrices of block `layer` lie.
    #[must_use]
    pub const fn block(&self, layer: usize) -> Block {
        let (width, hidden) = (self.width, self.hidden());
        let qkv = self.embedding() + layer * self.block_size();
        let projection = qkv + 3 * width * width;
        let expansion = projection + width * width;
        Block {
            qkv,
            projection,
            expansion,
            contraction: expansion + 2 * hidden * width,
        }
    }

    /// Where the embedding of the level bins lies: after the blocks,
    /// `bins` rows of `width`.
    #[must_use]
    pub const fn bin_embedding(&self) -> usize {
        self.embedding() + self.layers * self.block_size()
    }

    /// The number of parameters.
    #[must_use]
    pub const fn parameters(&self) -> usize {
        self.bin_embedding() + self.bins * self.width
    }

    /// Fresh parameters: independent normal draws of deviation [`INITIAL`].
    #[must_use]
    pub fn initial(&self, rng: &mut Rng) -> Vec<f32> {
        (0..self.parameters())
            .map(|_| rng.normal_with(0.0, INITIAL) as f32)
            .collect()
    }

    /// The multiply-adds of the matrix products of one token in training,
    /// forward and backward, in a sequence of `length` tokens: three times
    /// those of the forward pass, which are one per parameter of a matrix,
    /// the embedding counted once as the readout, and `2 · length · width`
    /// per block for the scores and the attended values.
    #[must_use]
    pub const fn products(&self, length: usize) -> usize {
        3 * (self.bin_embedding() + self.layers * 2 * length * self.width)
    }
}

/// A matrix read where it lies: `rows` rows of `columns` floats, a row's
/// floats `column_step` apart and a column's `row_step` apart.
#[derive(Clone, Copy, Debug)]
pub struct Matrix<'a> {
    data: &'a [f32],
    rows: usize,
    columns: usize,
    row_step: usize,
    column_step: usize,
}

impl<'a> Matrix<'a> {
    /// The matrix of `rows` rows of `columns` floats stored row after row.
    ///
    /// # Panics
    ///
    /// If `data` is not that many floats.
    #[must_use]
    pub fn new(data: &'a [f32], rows: usize, columns: usize) -> Self {
        assert_eq!(data.len(), rows * columns, "not {rows} rows of {columns}");
        Self::spaced(data, rows, columns, columns)
    }

    /// The matrix whose row `i` is the `columns` floats of `data` from
    /// `i · step` on: a block of columns of a wider matrix.
    ///
    /// # Panics
    ///
    /// If `data` ends before the last row does.
    #[must_use]
    pub fn spaced(data: &'a [f32], rows: usize, columns: usize, step: usize) -> Self {
        let needed = if rows == 0 {
            0
        } else {
            (rows - 1) * step + columns
        };
        assert!(
            data.len() >= needed,
            "{} floats for {rows} rows",
            data.len()
        );
        Self {
            data,
            rows,
            columns,
            row_step: step,
            column_step: 1,
        }
    }

    /// The transpose, read from the same floats.
    #[must_use]
    pub const fn transposed(self) -> Self {
        Self {
            data: self.data,
            rows: self.columns,
            columns: self.rows,
            row_step: self.column_step,
            column_step: self.row_step,
        }
    }

    /// The number of rows.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// The number of columns.
    #[must_use]
    pub const fn columns(&self) -> usize {
        self.columns
    }

    /// The floats the matrix is read from: element `(i, j)` is float
    /// `i · row_step + j · column_step`.
    #[must_use]
    pub const fn data(&self) -> &'a [f32] {
        self.data
    }

    /// How far apart the floats of a column lie, and those of a row:
    /// `(row_step, column_step)`.
    #[must_use]
    pub const fn steps(&self) -> (usize, usize) {
        (self.row_step, self.column_step)
    }

    /// Element `(row, column)`.
    #[must_use]
    pub fn at(&self, row: usize, column: usize) -> f32 {
        self.data[row * self.row_step + column * self.column_step]
    }
}

/// The matrix product in single precision: the one operation of the model
/// that kernels replace.
pub trait Product {
    /// `c = a b`, or `c += a b` if `keep`, where row `i` of the product is
    /// the floats of `c` from `i · stride` on; the floats between the rows
    /// stay as they are.
    ///
    /// # Panics
    ///
    /// If the shapes do not agree, or `c` ends before the last row does.
    fn multiply_spaced(
        &mut self,
        c: &mut [f32],
        stride: usize,
        a: Matrix<'_>,
        b: Matrix<'_>,
        keep: bool,
    );

    /// `c = a b`, or `c += a b` if `keep`, where `c` is the rows of the
    /// product one after another.
    ///
    /// # Panics
    ///
    /// If the shapes do not agree.
    fn multiply(&mut self, c: &mut [f32], a: Matrix<'_>, b: Matrix<'_>, keep: bool) {
        assert_eq!(
            c.len(),
            a.rows() * b.columns(),
            "the product has another size"
        );
        self.multiply_spaced(c, b.columns(), a, b, keep);
    }
}

/// The product by its definition: slow, and what the kernels are held to.
#[derive(Clone, Copy, Debug, Default)]
pub struct Plain;

impl Product for Plain {
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
        assert!(
            stride >= n,
            "rows of {n} floats do not fit a stride of {stride}"
        );
        assert!(
            m == 0 || c.len() >= (m - 1) * stride + n,
            "the product ends early"
        );
        for i in 0..m {
            let row = &mut c[i * stride..][..n];
            if !keep {
                row.fill(0.0);
            }
            for l in 0..k {
                let left = a.at(i, l);
                for (j, slot) in row.iter_mut().enumerate() {
                    *slot += left * b.at(l, j);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_layout_tiles_the_parameters() {
        let config = Config {
            vocab: 50,
            width: 24,
            heads: 3,
            layers: 2,
            bins: 0,
        };
        config.check().expect("valid");
        assert_eq!((config.head(), config.hidden()), (8, 64));
        let first = config.block(0);
        assert_eq!(first.qkv, 50 * 24);
        assert_eq!(first.projection - first.qkv, 24 * 72);
        assert_eq!(first.expansion - first.projection, 24 * 24);
        assert_eq!(first.contraction - first.expansion, 24 * 128);
        assert_eq!(config.block(1).qkv - first.contraction, 64 * 24);
        assert_eq!(config.parameters(), config.block(2).qkv);
        let binned = Config { bins: 10, ..config };
        assert_eq!(binned.bin_embedding(), config.parameters());
        assert_eq!(binned.parameters(), config.parameters() + 240);
        assert_eq!(binned.products(7), config.products(7));
        assert_eq!(
            config.products(7),
            3 * (config.parameters() + 2 * 2 * 7 * 24)
        );
        let weights = config.initial(&mut Rng::from_key(&[1]));
        assert_eq!(weights.len(), config.parameters());
        let spread = (weights.iter().map(|w| f64::from(*w).powi(2)).sum::<f64>()
            / weights.len() as f64)
            .sqrt();
        assert!((spread - INITIAL).abs() < 1e-3, "deviation {spread}");
        for wrong in [
            Config { heads: 5, ..config },
            Config {
                width: 18,
                heads: 6,
                ..config
            },
            Config {
                layers: 0,
                ..config
            },
        ] {
            assert!(wrong.check().is_err(), "{wrong:?}");
        }
    }

    #[test]
    fn the_plain_product_reads_transposes_and_blocks_in_place() {
        // [[1, 2, 3], [4, 5, 6]] by its transpose.
        let a = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let matrix = Matrix::new(&a, 2, 3);
        assert_eq!((matrix.at(1, 0), matrix.transposed().at(0, 1)), (4.0, 4.0));
        let mut c = [0.0_f32; 4];
        Plain.multiply(&mut c, matrix, matrix.transposed(), false);
        assert_eq!(c, [14.0, 32.0, 32.0, 77.0]);
        Plain.multiply(&mut c, matrix, matrix.transposed(), true);
        assert_eq!(c, [28.0, 64.0, 64.0, 154.0]);
        let mut gram = [0.0_f32; 9];
        Plain.multiply(&mut gram, matrix.transposed(), matrix, false);
        assert_eq!(gram, [17.0, 22.0, 27.0, 22.0, 29.0, 36.0, 27.0, 36.0, 45.0]);
        // The block [[2, 3], [5, 6]] by itself, into rows three floats apart.
        let block = Matrix::spaced(&a[1..], 2, 2, 3);
        assert_eq!((block.at(0, 0), block.at(1, 1)), (2.0, 6.0));
        let mut wide = [-1.0_f32; 5];
        Plain.multiply_spaced(&mut wide, 3, block, block, false);
        assert_eq!(wide, [19.0, 24.0, -1.0, 40.0, 51.0]);
        Plain.multiply_spaced(&mut wide, 3, block.transposed(), block, true);
        assert_eq!(
            wide,
            [19.0 + 29.0, 24.0 + 36.0, -1.0, 40.0 + 36.0, 51.0 + 45.0]
        );
    }
}
