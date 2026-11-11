//! The model on one sequence: the forward pass, the cross-entropy of its
//! logits, and the backward pass that adds the gradient of the loss to a
//! vector laid out as the parameters.
//!
//! A [`Workspace`] holds every array of the two passes, so that neither
//! allocates: a thread that trains keeps one and runs sequence after
//! sequence through it.
//!
//! What `GPT` of `gpt.py` adds to the plain function, kept here:
//! - the attention of the first position gives exactly zero
//!   (`FirstZeroQKAttention`): a token that attends to itself alone has
//!   nothing left once its own value is projected out;
//! - so do the positions that repeat the first token from the start
//!   (`PrefixStableBlock`), whose states are that of the first, as long as
//!   no head's value at the first position is under the epsilon of the
//!   projection.
//!
//! Left out, since they change the rounding and not the function: the
//! double-precision backward pass of the RMS normalizations
//! (`rms_norm.py`), and the values centered at the first position before
//! they are averaged (`CenteredQKAttention`); the projection itself is
//! centered at the own value, as there.

use super::math::{
    UNIT_EPSILON, causal_softmax, causal_softmax_back, dot, exclude, exclude_back, normalize,
    normalize_back, rms, rms_back, rope_tables, rotate, softmax, swiglu, swiglu_back,
};
use super::{Config, Matrix, Product};

/// The rows of scores and of attention weights a product makes at once.
/// Row `t` needs its first `t + 1` columns only, so a product over a band
/// of rows stops at the band's last column, and the bands together do
/// little more than half the work of the square.
const BAND: usize = 48;

/// What the backward pass needs of a block's forward pass.
struct Kept {
    /// The normalized input of the attention: a row of `width` a position.
    norm1: Vec<f32>,
    /// What each row of the input was multiplied by.
    inverse1: Vec<f32>,
    /// A row of `3 · width` a position: the queries and the keys of the
    /// heads, normalized and rotated, then the values.
    qkv: Vec<f32>,
    /// What the query and the key of each position and head were multiplied
    /// by.
    inverse_qk: Vec<f32>,
    /// The attention weights: for each head, a row of `length` a position.
    weights: Vec<f32>,
    /// The attended values: for each head, a row of its width a position.
    attended: Vec<f32>,
    /// What [`exclude`] returned for each head and position.
    along: Vec<(f32, f32)>,
    /// The attended values less their own: a row of `width` a position.
    excluded: Vec<f32>,
    /// The positions from the start whose attention gives zero.
    stable: usize,
    /// The normalized input of the feed-forward network.
    norm2: Vec<f32>,
    /// What each row of that input was multiplied by.
    inverse2: Vec<f32>,
    /// The gates and the values: a row of `2 · hidden` a position.
    expanded: Vec<f32>,
    /// The gated values: a row of `hidden` a position.
    gated: Vec<f32>,
}

impl Kept {
    fn new(config: &Config, capacity: usize) -> Self {
        let (width, heads, hidden) = (config.width, config.heads, config.hidden());
        Self {
            norm1: vec![0.0; capacity * width],
            inverse1: vec![0.0; capacity],
            qkv: vec![0.0; capacity * 3 * width],
            inverse_qk: vec![0.0; capacity * heads * 2],
            weights: vec![0.0; heads * capacity * capacity],
            attended: vec![0.0; capacity * width],
            along: vec![(0.0, 0.0); capacity * heads],
            excluded: vec![0.0; capacity * width],
            stable: 0,
            norm2: vec![0.0; capacity * width],
            inverse2: vec![0.0; capacity],
            expanded: vec![0.0; capacity * 2 * hidden],
            gated: vec![0.0; capacity * hidden],
        }
    }
}

/// The gradients the backward pass holds at one time.
struct Back {
    /// At the residual stream.
    stream: Vec<f32>,
    /// At a normalized state.
    norm: Vec<f32>,
    /// At the queries, the keys and the values of a block.
    qkv: Vec<f32>,
    /// At the attended values less their own.
    excluded: Vec<f32>,
    /// At the gates and the values of a feed-forward network.
    expanded: Vec<f32>,
    /// At its gated values.
    gated: Vec<f32>,
    /// At the attention weights of one head, then at its scores.
    weights: Vec<f32>,
    /// At the attended values of one head.
    attended: Vec<f32>,
    /// At the rotated queries of one head.
    queries: Vec<f32>,
    /// At the rotated keys of one head.
    keys: Vec<f32>,
    /// At the values of one head, through the weights.
    values: Vec<f32>,
}

impl Back {
    fn new(config: &Config, capacity: usize) -> Self {
        let (width, head, hidden) = (config.width, config.head(), config.hidden());
        Self {
            stream: vec![0.0; capacity * width],
            norm: vec![0.0; capacity * width],
            qkv: vec![0.0; capacity * 3 * width],
            excluded: vec![0.0; capacity * width],
            expanded: vec![0.0; capacity * 2 * hidden],
            gated: vec![0.0; capacity * hidden],
            weights: vec![0.0; capacity * capacity],
            attended: vec![0.0; capacity * head],
            queries: vec![0.0; capacity * head],
            keys: vec![0.0; capacity * head],
            values: vec![0.0; capacity * head],
        }
    }
}

/// The arrays of the forward and the backward pass of sequences of up to
/// a number of tokens.
pub struct Workspace {
    config: Config,
    capacity: usize,
    cos: Vec<f32>,
    sin: Vec<f32>,
    /// The tokens of the last forward pass.
    tokens: Vec<u32>,
    /// The residual stream: a row of `width` a position.
    stream: Vec<f32>,
    blocks: Vec<Kept>,
    /// The normalized last state, and what each row was multiplied by.
    norm: Vec<f32>,
    inverse: Vec<f32>,
    /// A row of `vocab` a position: the logits, then the gradient at them.
    logits: Vec<f32>,
    back: Back,
}

/// The ranges of rows `start..end` that tile `0..length` in bands.
fn bands(length: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..length)
        .step_by(BAND)
        .map(move |start| (start, (start + BAND).min(length)))
}

impl Workspace {
    /// A workspace for sequences of up to `capacity` tokens.
    ///
    /// # Panics
    ///
    /// If the sizes of `config` are not valid.
    #[must_use]
    pub fn new(config: Config, capacity: usize) -> Self {
        config.check().expect("valid sizes");
        let (cos, sin) = rope_tables(config.head(), capacity);
        Self {
            config,
            capacity,
            cos,
            sin,
            tokens: Vec::with_capacity(capacity),
            stream: vec![0.0; capacity * config.width],
            blocks: (0..config.layers)
                .map(|_| Kept::new(&config, capacity))
                .collect(),
            norm: vec![0.0; capacity * config.width],
            inverse: vec![0.0; capacity],
            logits: vec![0.0; capacity * config.vocab],
            back: Back::new(&config, capacity),
        }
    }

    /// The sizes of the model.
    #[must_use]
    pub const fn config(&self) -> &Config {
        &self.config
    }

    /// The most tokens a sequence may have.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// The logits of the last forward pass, a row of `vocab` for each
    /// token; after [`Workspace::loss`], the gradient of the loss at them.
    #[must_use]
    pub fn logits(&self) -> &[f32] {
        &self.logits[..self.tokens.len() * self.config.vocab]
    }

    /// Runs the model on `tokens`.
    ///
    /// # Panics
    ///
    /// If `parameters` are not those of the model, there is no token or
    /// more than the capacity, or a token is outside the vocabulary.
    pub fn forward<P: Product>(&mut self, product: &mut P, parameters: &[f32], tokens: &[u32]) {
        let config = self.config;
        assert_eq!(parameters.len(), config.parameters(), "another model");
        assert!(
            !tokens.is_empty() && tokens.len() <= self.capacity,
            "{} tokens in a workspace of {}",
            tokens.len(),
            self.capacity
        );
        let (t, d, heads, dh) = (tokens.len(), config.width, config.heads, config.head());
        let (hidden, half) = (config.hidden(), config.head() / 2);
        self.tokens.clear();
        self.tokens.extend_from_slice(tokens);
        let stream = &mut self.stream[..t * d];
        for (row, token) in stream.chunks_mut(d).zip(tokens) {
            let token = *token as usize;
            assert!(token < config.vocab, "token {token} of {}", config.vocab);
            row.copy_from_slice(&parameters[token * d..][..d]);
        }
        let scale = 1.0 / (dh as f32).sqrt();
        let mut stable = tokens
            .iter()
            .take_while(|token| **token == tokens[0])
            .count();
        for (layer, kept) in self.blocks.iter_mut().enumerate() {
            let at = config.block(layer);
            rms(stream, d, &mut kept.norm1[..t * d], &mut kept.inverse1[..t]);
            product.multiply(
                &mut kept.qkv[..t * 3 * d],
                Matrix::new(&kept.norm1[..t * d], t, d),
                Matrix::new(&parameters[at.qkv..][..3 * d * d], d, 3 * d),
                false,
            );
            for (position, row) in kept.qkv[..t * 3 * d].chunks_mut(3 * d).enumerate() {
                let cos = &self.cos[position * half..][..half];
                let sin = &self.sin[position * half..][..half];
                for (head, inverse) in kept.inverse_qk[position * heads * 2..][..heads * 2]
                    .chunks_mut(2)
                    .enumerate()
                {
                    for (part, inverse) in inverse.iter_mut().enumerate() {
                        let vector = &mut row[part * d + head * dh..][..dh];
                        *inverse = normalize(vector);
                        rotate(vector, cos, sin, false);
                    }
                }
            }
            let first = &kept.qkv[2 * d..3 * d];
            if first
                .chunks(dh)
                .any(|own| dot(own, own).sqrt() <= UNIT_EPSILON)
            {
                stable = 1;
            }
            kept.stable = stable;
            for head in 0..heads {
                let weights = &mut kept.weights[head * t * t..][..t * t];
                for (start, end) in bands(t) {
                    product.multiply_spaced(
                        &mut weights[start * t..],
                        t,
                        Matrix::spaced(
                            &kept.qkv[start * 3 * d + head * dh..],
                            end - start,
                            dh,
                            3 * d,
                        ),
                        Matrix::spaced(&kept.qkv[d + head * dh..], end, dh, 3 * d).transposed(),
                        false,
                    );
                }
                causal_softmax(weights, t, scale);
                let attended = &mut kept.attended[head * t * dh..][..t * dh];
                for (start, end) in bands(t) {
                    product.multiply(
                        &mut attended[start * dh..end * dh],
                        Matrix::spaced(&weights[start * t..], end - start, end, t),
                        Matrix::spaced(&kept.qkv[2 * d + head * dh..], end, dh, 3 * d),
                        false,
                    );
                }
                for position in 0..t {
                    let out = &mut kept.excluded[position * d + head * dh..][..dh];
                    if position < stable {
                        out.fill(0.0);
                        continue;
                    }
                    let own = &kept.qkv[position * 3 * d + 2 * d + head * dh..][..dh];
                    kept.along[head * t + position] =
                        exclude(&attended[position * dh..][..dh], own, out);
                }
            }
            product.multiply(
                stream,
                Matrix::new(&kept.excluded[..t * d], t, d),
                Matrix::new(&parameters[at.projection..][..d * d], d, d),
                true,
            );
            rms(stream, d, &mut kept.norm2[..t * d], &mut kept.inverse2[..t]);
            product.multiply(
                &mut kept.expanded[..t * 2 * hidden],
                Matrix::new(&kept.norm2[..t * d], t, d),
                Matrix::new(&parameters[at.expansion..][..2 * hidden * d], d, 2 * hidden),
                false,
            );
            swiglu(
                &kept.expanded[..t * 2 * hidden],
                hidden,
                &mut kept.gated[..t * hidden],
            );
            product.multiply(
                stream,
                Matrix::new(&kept.gated[..t * hidden], t, hidden),
                Matrix::new(&parameters[at.contraction..][..hidden * d], hidden, d),
                true,
            );
        }
        rms(stream, d, &mut self.norm[..t * d], &mut self.inverse[..t]);
        product.multiply(
            &mut self.logits[..t * config.vocab],
            Matrix::new(&self.norm[..t * d], t, d),
            Matrix::new(&parameters[..config.embedding()], config.vocab, d).transposed(),
            false,
        );
    }

    /// The loss of the last forward pass: the sum over the positions of
    /// `weights[t]` times the cross-entropy of the logits of position `t`
    /// against `targets[t]`. The logits are replaced by the gradient of the
    /// loss at them, which [`Workspace::backward`] starts from. A position of
    /// weight 0 is not looked at: its target may be any number.
    ///
    /// # Panics
    ///
    /// If there is not a target and a weight for each token of the pass.
    pub fn loss(&mut self, targets: &[u32], weights: &[f32]) -> f64 {
        let (t, vocab) = (self.tokens.len(), self.config.vocab);
        assert_eq!(
            (targets.len(), weights.len()),
            (t, t),
            "not a target a token"
        );
        let mut total = 0.0_f64;
        let rows = self.logits[..t * vocab].chunks_mut(vocab);
        for ((row, target), weight) in rows.zip(targets).zip(weights) {
            if *weight == 0.0 {
                row.fill(0.0);
                continue;
            }
            let target = *target as usize;
            assert!(target < vocab, "target {target} of {vocab}");
            total -= f64::from(*weight) * f64::from(softmax(row, target));
            row[target] -= 1.0;
            row.iter_mut().for_each(|share| *share *= weight);
        }
        total
    }

    /// Adds to `gradient` the gradient at the parameters of the loss that
    /// [`Workspace::loss`] last returned.
    ///
    /// # Panics
    ///
    /// If `parameters` or `gradient` are not those of the model.
    pub fn backward<P: Product>(
        &mut self,
        product: &mut P,
        parameters: &[f32],
        gradient: &mut [f32],
    ) {
        let config = self.config;
        assert_eq!(parameters.len(), config.parameters(), "another model");
        assert_eq!(gradient.len(), config.parameters(), "another model");
        let (t, d, heads, dh) = (self.tokens.len(), config.width, config.heads, config.head());
        let (hidden, half, vocab) = (config.hidden(), config.head() / 2, config.vocab);
        let scale = 1.0 / (dh as f32).sqrt();
        let back = &mut self.back;
        let at_logits = Matrix::new(&self.logits[..t * vocab], t, vocab);
        product.multiply(
            &mut gradient[..vocab * d],
            at_logits.transposed(),
            Matrix::new(&self.norm[..t * d], t, d),
            true,
        );
        product.multiply(
            &mut back.norm[..t * d],
            at_logits,
            Matrix::new(&parameters[..vocab * d], vocab, d),
            false,
        );
        let at_stream = &mut back.stream[..t * d];
        at_stream.fill(0.0);
        rms_back(
            &back.norm[..t * d],
            &self.norm[..t * d],
            &self.inverse[..t],
            d,
            at_stream,
        );
        for (layer, kept) in self.blocks.iter().enumerate().rev() {
            let at = config.block(layer);
            // The feed-forward network.
            product.multiply(
                &mut gradient[at.contraction..][..hidden * d],
                Matrix::new(&kept.gated[..t * hidden], t, hidden).transposed(),
                Matrix::new(at_stream, t, d),
                true,
            );
            product.multiply(
                &mut back.gated[..t * hidden],
                Matrix::new(at_stream, t, d),
                Matrix::new(&parameters[at.contraction..][..hidden * d], hidden, d).transposed(),
                false,
            );
            swiglu_back(
                &back.gated[..t * hidden],
                &kept.expanded[..t * 2 * hidden],
                hidden,
                &mut back.expanded[..t * 2 * hidden],
            );
            let at_expanded = Matrix::new(&back.expanded[..t * 2 * hidden], t, 2 * hidden);
            product.multiply(
                &mut gradient[at.expansion..][..2 * hidden * d],
                Matrix::new(&kept.norm2[..t * d], t, d).transposed(),
                at_expanded,
                true,
            );
            product.multiply(
                &mut back.norm[..t * d],
                at_expanded,
                Matrix::new(&parameters[at.expansion..][..2 * hidden * d], d, 2 * hidden)
                    .transposed(),
                false,
            );
            rms_back(
                &back.norm[..t * d],
                &kept.norm2[..t * d],
                &kept.inverse2[..t],
                d,
                at_stream,
            );
            // The attention.
            product.multiply(
                &mut gradient[at.projection..][..d * d],
                Matrix::new(&kept.excluded[..t * d], t, d).transposed(),
                Matrix::new(at_stream, t, d),
                true,
            );
            product.multiply(
                &mut back.excluded[..t * d],
                Matrix::new(at_stream, t, d),
                Matrix::new(&parameters[at.projection..][..d * d], d, d).transposed(),
                false,
            );
            back.excluded[..kept.stable * d].fill(0.0);
            back.qkv[..t * 3 * d].fill(0.0);
            for head in 0..heads {
                let weights = &kept.weights[head * t * t..][..t * t];
                let attended = &kept.attended[head * t * dh..][..t * dh];
                let values = Matrix::spaced(&kept.qkv[2 * d + head * dh..], t, dh, 3 * d);
                for position in 0..t {
                    exclude_back(
                        &back.excluded[position * d + head * dh..][..dh],
                        &attended[position * dh..][..dh],
                        &kept.qkv[position * 3 * d + 2 * d + head * dh..][..dh],
                        kept.along[head * t + position],
                        &mut back.attended[position * dh..][..dh],
                        &mut back.qkv[position * 3 * d + 2 * d + head * dh..][..dh],
                    );
                }
                back.attended[..kept.stable * dh].fill(0.0);
                for (start, end) in bands(t) {
                    // At the weights of the band: its attended gradients by
                    // the values it sees.
                    product.multiply_spaced(
                        &mut back.weights[start * t..],
                        t,
                        Matrix::new(&back.attended[start * dh..end * dh], end - start, dh),
                        Matrix::spaced(values.data(), end, dh, 3 * d).transposed(),
                        false,
                    );
                    // At the values of the band: the attended gradients of
                    // the positions that see them, by their weights.
                    product.multiply(
                        &mut back.values[start * dh..end * dh],
                        Matrix::spaced(&weights[start * t + start..], t - start, end - start, t)
                            .transposed(),
                        Matrix::new(&back.attended[start * dh..t * dh], t - start, dh),
                        false,
                    );
                }
                causal_softmax_back(&mut back.weights[..t * t], weights, t, scale);
                for (start, end) in bands(t) {
                    product.multiply(
                        &mut back.queries[start * dh..end * dh],
                        Matrix::spaced(&back.weights[start * t..], end - start, end, t),
                        Matrix::spaced(&kept.qkv[d + head * dh..], end, dh, 3 * d),
                        false,
                    );
                    product.multiply(
                        &mut back.keys[start * dh..end * dh],
                        Matrix::spaced(
                            &back.weights[start * t + start..],
                            t - start,
                            end - start,
                            t,
                        )
                        .transposed(),
                        Matrix::spaced(
                            &kept.qkv[start * 3 * d + head * dh..],
                            t - start,
                            dh,
                            3 * d,
                        ),
                        false,
                    );
                }
                for position in 0..t {
                    let cos = &self.cos[position * half..][..half];
                    let sin = &self.sin[position * half..][..half];
                    let rotated = &kept.qkv[position * 3 * d..][..3 * d];
                    let row = &mut back.qkv[position * 3 * d..][..3 * d];
                    let inverse = &kept.inverse_qk[(position * heads + head) * 2..][..2];
                    normalize_back(
                        &back.queries[position * dh..][..dh],
                        &rotated[head * dh..][..dh],
                        inverse[0],
                        cos,
                        sin,
                        &mut row[head * dh..][..dh],
                    );
                    normalize_back(
                        &back.keys[position * dh..][..dh],
                        &rotated[d + head * dh..][..dh],
                        inverse[1],
                        cos,
                        sin,
                        &mut row[d + head * dh..][..dh],
                    );
                    let through = &back.values[position * dh..][..dh];
                    for (slot, value) in row[2 * d + head * dh..][..dh].iter_mut().zip(through) {
                        *slot += value;
                    }
                }
            }
            let at_qkv = Matrix::new(&back.qkv[..t * 3 * d], t, 3 * d);
            product.multiply(
                &mut gradient[at.qkv..][..3 * d * d],
                Matrix::new(&kept.norm1[..t * d], t, d).transposed(),
                at_qkv,
                true,
            );
            product.multiply(
                &mut back.norm[..t * d],
                at_qkv,
                Matrix::new(&parameters[at.qkv..][..3 * d * d], d, 3 * d).transposed(),
                false,
            );
            rms_back(
                &back.norm[..t * d],
                &kept.norm1[..t * d],
                &kept.inverse1[..t],
                d,
                at_stream,
            );
        }
        for (row, token) in at_stream.chunks(d).zip(&self.tokens) {
            let slots = &mut gradient[*token as usize * d..][..d];
            for (slot, value) in slots.iter_mut().zip(row) {
                *slot += value;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::Plain;
    use super::*;
    use crate::random::Rng;

    const CONFIG: Config = Config {
        vocab: 11,
        width: 8,
        heads: 2,
        layers: 2,
    };

    /// Weights wide enough for every part of the model to bend.
    fn weights(seed: u64) -> Vec<f32> {
        let mut rng = Rng::from_key(&[seed]);
        (0..CONFIG.parameters())
            .map(|_| rng.normal_with(0.0, 0.4) as f32)
            .collect()
    }

    fn tokens(count: usize, seed: u64) -> Vec<u32> {
        let mut rng = Rng::from_key(&[seed]);
        (0..count).map(|_| rng.index(CONFIG.vocab) as u32).collect()
    }

    /// The loss of `tokens` with each position's next token as its target
    /// and weights that differ by position.
    fn loss(workspace: &mut Workspace, parameters: &[f32], tokens: &[u32]) -> f64 {
        let targets: Vec<u32> = tokens.iter().skip(1).chain(&tokens[..1]).copied().collect();
        let weights: Vec<f32> = (0..tokens.len())
            .map(|position| 0.5 + (position % 3) as f32 * 0.25)
            .collect();
        workspace.forward(&mut Plain, parameters, tokens);
        workspace.loss(&targets, &weights)
    }

    /// Checks the gradient of the loss of `tokens` along a random direction
    /// within each matrix of the model against central differences.
    fn check(tokens: &[u32], seed: u64) {
        let parameters = weights(seed);
        let mut workspace = Workspace::new(CONFIG, tokens.len());
        let mut gradient = vec![0.0_f32; parameters.len()];
        loss(&mut workspace, &parameters, tokens);
        workspace.backward(&mut Plain, &parameters, &mut gradient);
        let mut edges = vec![0, CONFIG.embedding()];
        for layer in 0..CONFIG.layers {
            let block = CONFIG.block(layer);
            edges.extend([block.projection, block.expansion, block.contraction]);
            edges.push(CONFIG.block(layer + 1).qkv);
        }
        let mut rng = Rng::from_key(&[seed, 99]);
        for (matrix, range) in edges.windows(2).enumerate() {
            let mut direction = vec![0.0_f32; parameters.len()];
            direction[range[0]..range[1]]
                .iter_mut()
                .for_each(|slot| *slot = rng.normal() as f32);
            let step = 1e-3_f32;
            let moved = |sign: f32| -> Vec<f32> {
                let moves = parameters.iter().zip(&direction);
                moves.map(|(p, d)| p + sign * step * d).collect()
            };
            let up = loss(&mut workspace, &moved(1.0), tokens);
            let down = loss(&mut workspace, &moved(-1.0), tokens);
            let numeric = (up - down) / (2.0 * f64::from(step));
            let exact: f64 = gradient
                .iter()
                .zip(&direction)
                .map(|(g, d)| f64::from(*g) * f64::from(*d))
                .sum();
            assert!(
                (exact - numeric).abs() < 0.01 * numeric.abs() + 0.01,
                "matrix {matrix} of {tokens:?}: {exact} against {numeric}"
            );
            assert!(exact.abs() > 1e-3, "matrix {matrix} has no gradient");
        }
    }

    #[test]
    fn the_gradient_agrees_with_differences_in_every_matrix() {
        check(&tokens(9, 1), 1);
        // More positions than a band holds.
        check(&tokens(BAND + 13, 2), 2);
    }

    #[test]
    fn the_gradient_agrees_on_a_repeated_first_token() {
        let mut repeated = tokens(10, 3);
        repeated[..4].fill(7);
        repeated[4] = 2;
        check(&repeated, 3);
    }

    #[test]
    fn a_position_does_not_see_the_tokens_after_it() {
        let parameters = weights(4);
        let mut workspace = Workspace::new(CONFIG, 60);
        let mut sequence = tokens(60, 4);
        workspace.forward(&mut Plain, &parameters, &sequence);
        let whole = workspace.logits().to_vec();
        assert_eq!(whole.len(), 60 * CONFIG.vocab);
        // The logits tell positions apart.
        assert!(whole[..CONFIG.vocab] != whole[CONFIG.vocab..2 * CONFIG.vocab]);
        // The same tokens without the last ten, and with others after them.
        workspace.forward(&mut Plain, &parameters, &sequence[..50]);
        assert_eq!(workspace.logits(), &whole[..50 * CONFIG.vocab]);
        sequence[50..]
            .iter_mut()
            .for_each(|token| *token = (*token + 1) % 11);
        workspace.forward(&mut Plain, &parameters, &sequence);
        assert_eq!(
            &workspace.logits()[..50 * CONFIG.vocab],
            &whole[..50 * CONFIG.vocab]
        );
        assert!(workspace.logits()[50 * CONFIG.vocab..] != whole[50 * CONFIG.vocab..]);
    }

    #[test]
    fn a_repeated_first_token_keeps_the_state_of_the_first_position() {
        let parameters = weights(5);
        let mut workspace = Workspace::new(CONFIG, 12);
        let mut sequence = vec![3_u32; 12];
        sequence[8..].copy_from_slice(&[1, 3, 3, 9]);
        workspace.forward(&mut Plain, &parameters, &sequence);
        let logits = workspace.logits().to_vec();
        let (first, vocab) = (&logits[..CONFIG.vocab], CONFIG.vocab);
        for position in 1..8 {
            assert_eq!(
                &logits[position * vocab..][..vocab],
                first,
                "position {position}"
            );
        }
        // After another token the same one reads differently.
        assert!(&logits[9 * vocab..][..vocab] != first);
        // The first position is the model without attention: its logits are
        // those of a sequence of one token.
        workspace.forward(&mut Plain, &parameters, &sequence[..1]);
        assert_eq!(workspace.logits(), first);
    }

    #[test]
    fn a_weightless_position_gives_no_loss_and_no_gradient() {
        let parameters = weights(6);
        let sequence = tokens(6, 6);
        let mut workspace = Workspace::new(CONFIG, 6);
        workspace.forward(&mut Plain, &parameters, &sequence);
        // The cross-entropy of uniform weights on one position is that of
        // its row alone.
        let row: Vec<f64> = workspace.logits()[2 * 11..3 * 11]
            .iter()
            .map(|logit| f64::from(*logit))
            .collect();
        let total: f64 = row.iter().map(|logit| logit.exp()).sum();
        let expected = total.ln() - row[4];
        let mut weights = [0.0_f32; 6];
        weights[2] = 1.0;
        // A weightless target may be outside the vocabulary.
        let loss = workspace.loss(&[99, 99, 4, 99, 99, 99], &weights);
        assert!((loss - expected).abs() < 1e-5, "{loss} against {expected}");
        let at_logits = workspace.logits();
        assert!(at_logits[..2 * 11].iter().all(|g| *g == 0.0));
        assert!(at_logits[3 * 11..].iter().all(|g| *g == 0.0));
        assert!(at_logits[2 * 11..3 * 11].iter().sum::<f32>().abs() < 1e-6);
        // Nothing after the one position reaches the parameters: the
        // gradient is that of the sequence cut after it.
        let mut gradient = vec![0.0_f32; parameters.len()];
        workspace.backward(&mut Plain, &parameters, &mut gradient);
        let mut cut = vec![0.0_f32; parameters.len()];
        workspace.forward(&mut Plain, &parameters, &sequence[..3]);
        let again = workspace.loss(&[99, 99, 4], &[0.0, 0.0, 1.0]);
        workspace.backward(&mut Plain, &parameters, &mut cut);
        assert!((again - loss).abs() < 1e-6);
        assert!(gradient.iter().any(|g| g.abs() > 1e-2));
        for (whole, cut) in gradient.iter().zip(&cut) {
            assert!((whole - cut).abs() < 1e-5, "{whole} and {cut}");
        }
    }
}
