"""Isolated model mechanisms that preserve the public GPT's learned tensors."""

from typing import TYPE_CHECKING

from diff_gpt.domain.model.mini_block import RMSNormProjection
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
