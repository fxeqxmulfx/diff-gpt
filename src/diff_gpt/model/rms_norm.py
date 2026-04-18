import torch
from torch.nn import functional as F


def rms_norm(x: torch.Tensor) -> torch.Tensor:
    """Purely functional rmsnorm with no learnable params."""
    return F.rms_norm(x, (x.size(-1),))
