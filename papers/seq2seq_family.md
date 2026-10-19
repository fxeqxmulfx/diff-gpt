# What is a family of sequence-to-sequence functions?

In the context of Yun et al. (2020), "Are Transformers universal approximators
of sequence-to-sequence functions?" (arXiv:1912.10077v2).

## Definition

A **sequence-to-sequence function** is a map

$$f:\mathbb{R}^{d\times n}\to\mathbb{R}^{d\times n}.$$

It takes a **sequence** of $n$ tokens, each a $d$-dimensional embedding vector,
and returns **another sequence** of the same shape $d\times n$.

## How it works

The input is a matrix $\mathbf{X}\in\mathbb{R}^{d\times n}$:

- $n$ **columns** represent the $n$ tokens in the sequence;
- each column $\mathbf{X}_{:,j}\in\mathbb{R}^{d}$ is the embedding of token $j$.

$$
\mathbf{X}=\big[\,\underbrace{\mathbf{X}_{:,1}}_{\text{token 1}},\
\underbrace{\mathbf{X}_{:,2}}_{\text{token 2}},\ \dots,\
\underbrace{\mathbf{X}_{:,n}}_{\text{token }n}\,\big].
$$

The output $f(\mathbf{X})$ is another $d\times n$ matrix: a new sequence of
embeddings. For example, BERT maps input token embeddings to contextual
representations of the same tokens. This fixed-length mathematical setting
does not directly describe translation between sequences of different lengths.

## Why call it a function rather than a layer?

Each transformer block (attention followed by a feed-forward layer) is already
a sequence-to-sequence function $\mathbb{R}^{d\times n}\to\mathbb{R}^{d\times n}$.
A transformer is a **composition** of these blocks and therefore another
sequence-to-sequence function. The paper introduces a **family**, or class, of
functions that transformers can represent:

$$
\mathcal{T}^{h,m,r}\;\overset{\text{def}}{=}\;\{\,g:\mathbb{R}^{d\times n}\to
\mathbb{R}^{d\times n}\mid g\text{ is a composition of blocks }t^{h,m,r}\,\},
$$

where $h$ is the number of attention heads, $m$ is the head size, and $r$ is
the hidden width of the feed-forward layer.

A **family of functions** is the set obtained by varying all weights and the
depth (number of blocks), while keeping $(h,m,r)$ fixed. It describes the
transformer's expressive capacity.

## Why this matters for the main result

The paper compares two families:

| Notation | Function class |
|---|---|
| $\mathcal{T}^{h,m,r}$ | Functions representable by the transformer |
| $\mathcal{F}_{\rm PE}$ | Continuous permutation-equivariant functions with compact support |
| $\mathcal{F}_{\rm CD}$ | Continuous functions on a compact domain, without the equivariance restriction |

**Universal approximation** means that for every target function $f$ in
$\mathcal{F}_{\rm PE}$, or in $\mathcal{F}_{\rm CD}$ when positional encodings
are available, some transformer $g$ approximates it arbitrarily closely in
the paper's $L^p$ distance: $\mathsf{d}_p(f,g)\le\epsilon$.

The sequence length $n$ is fixed in each such statement. The approximating
weights and depth can depend on $n$, $f$, and $\epsilon$.

Thus, a family of sequence-to-sequence functions is the language used to ask
precisely how rich a set of sequence-to-sequence maps a transformer can express.
