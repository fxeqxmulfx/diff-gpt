"""Isolated model mechanisms that preserve the public GPT's learned tensors."""

from typing import TYPE_CHECKING

from diff_gpt.domain.model.mini_block import RMSNormEmbedding, RMSNormProjection
from diff_gpt.domain.model.rms_norm import RMSNorm

if TYPE_CHECKING:
    from diff_gpt.domain.model.gpt import GPT


def install_peri_rms(model: GPT) -> None:
    """
    Normalize both residual branch outputs without drawing new parameters.

    Source: arXiv:2510.22026v2, Table 1, Peri-LN update; corresponding scheme
    Transformer.Normalization.Scheme.peri in the local transformer project.
    Adaptation: parameter-free RMS with the existing repaired FP64 backward,
    applied to both attention and SwiGLU branches. This candidate retains the
    benchmark's embedding, unit residual coefficients, initialization and all
    attention details; it does not import the older complete scaled recipe.
    """
    for block in model.blocks:
        for branch in (block.sa, block.ffwd):
            projection = RMSNormProjection(branch.c_proj)
            projection.norm = RMSNorm(projection.out_features)
            branch.c_proj = projection


def install_embedding_rms(model: GPT) -> None:
    """
    Normalize the embedding while preserving its shared readout parameter.

    Source: the embedding/final-normalization row and forward equations in
    transformer/MODEL_ARCHITECTURE.md at 8890a985. This isolated follow-up adds
    only embedding RMS to the branch-normalized candidate. The older recipe's
    depth scaling and optimizer are not imported. Use the existing repaired
    RMS backward and native dtype epsilon at this additional normalization.
    """
    embedding = RMSNormEmbedding(model.token_embedding_table)
    embedding.norm = RMSNorm(model.n_embd)
    model.token_embedding_table = embedding
