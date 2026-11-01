"""
Numerically centered evaluation of the existing clipped XSA projection.

Source: mha.py at b98c665b and the companion Lean theorem
Transformer.SequenceModel.Attention.centeredXSA_eq. Centering changes the
floating-point evaluation, not the real function. The below-epsilon own
remainder is essential. Q/K normalization and attention weights are unchanged.
"""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

XSA_EPS = 1e-12


def centered_projection(
    centered_average: Tensor,
    own_delta: Tensor,
    own: Tensor,
    eps: float = XSA_EPS,
) -> Tensor:
    """
    Remove the own direction after averaging values relative to an anchor.

    The same anchor must define centered_average and own_delta=own-anchor.
    Normalized attention then gives delta=original_average-own. For own
    values above eps the own projection is exactly zero, without subtracting
    two nonzero vectors. Below eps its original clipped remainder is retained.
    """
    if eps <= 0:
        msg = "XSA clipping epsilon must be positive"
        raise ValueError(msg)
    norm = torch.linalg.vector_norm(own, dim=-1, keepdim=True)
    denominator = norm.clamp_min(eps)
    unit = own / denominator
    delta = centered_average - own_delta
    # Zero the unused numerator before squaring, avoiding overflow. At equality
    # retain clamp_min's own backward convention, as F.normalize does.
    above = norm > eps
    ratio = torch.where(above, torch.zeros_like(norm), norm) / denominator
    remainder = torch.where(above, torch.zeros_like(norm), 1 - ratio.square())
    return delta - (delta * unit).sum(-1, keepdim=True) * unit + remainder * own
