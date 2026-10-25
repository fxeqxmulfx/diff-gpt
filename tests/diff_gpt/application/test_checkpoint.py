"""
Tests for the encoder settings that DiffGPT.save writes and DiffGPT.load checks.

Weights decode right only with the token scheme, tokenizer, order and domain
they were trained with. `save` wrote the weights alone, so `load` took them
into a DiffGPT with other settings, which then decoded silently wrong.
"""

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from diff_gpt.application.diff_gpt import DiffGPT
from diff_gpt.domain.encoder_decoder import ENCODER_VERSION, Difference
from diff_gpt.domain.model.gpt import GPT

if TYPE_CHECKING:
    from pathlib import Path

DOMAIN = np.array([1.0, 2.0])
SETTINGS = {
    "order_of_derivative": 1,
    "domain_of_definition": DOMAIN,
    "tokenizer": "mash",
}


def _diffgpt(seed: int, **settings: object) -> DiffGPT:
    """Return a DiffGPT with seeded weights and SETTINGS updated by `settings`."""
    torch.manual_seed(seed)
    gpt = GPT(vocab_size=64, n_embd=16, block_size=32, n_head=2, n_layer=1)
    return DiffGPT(model=gpt, **(SETTINGS | settings))


def _weights(model: DiffGPT) -> dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in model.model.state_dict().items()}


def _same(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_load_takes_weights_saved_with_the_same_settings(tmp_path: Path) -> None:
    """Check that the weights load where the settings match, also per column."""
    path = str(tmp_path / "model.bin")
    saved = _diffgpt(0)
    saved.save(path)
    for settings in (
        {},
        {"order_of_derivative": np.array([1, 1])},
        {"order_of_derivative": [Difference(1), 1]},
    ):
        twin = _diffgpt(1, **settings)
        twin.load(path)
        assert _same(_weights(saved), _weights(twin))
        assert not twin.model.training


def test_load_compares_the_lags_of_seasonal_operators(tmp_path: Path) -> None:
    """
    Check that a seasonal order is saved as its lags and compared by them.

    The factors commute, so Difference(24, 1) is Difference(1, 24), but the
    seasonal difference alone is another operator.
    """
    path = str(tmp_path / "model.bin")
    _diffgpt(0, order_of_derivative=Difference(1, 24)).save(path)
    _diffgpt(1, order_of_derivative=[Difference(24, 1)] * 2).load(path)
    other = _diffgpt(1, order_of_derivative=Difference(24))
    saved = r"order_of_derivative=\[\[1, 24\], \[1, 24\]\]"
    with pytest.raises(ValueError, match=saved):
        other.load(path)


def test_load_compares_a_domain_of_ranges_by_its_pairs(tmp_path: Path) -> None:
    """Check that ranges [lo, hi], shape (F, 2), are saved as pairs and compared."""
    path = str(tmp_path / "model.bin")
    ranges = np.array([[-1.0, 1.0], [0.0, 2.0]])
    saved = _diffgpt(0, domain_of_definition=ranges)
    saved.save(path)
    encoder = torch.load(path, weights_only=True)["encoder"]
    assert encoder["domain_of_definition"] == ranges.tolist()
    twin = _diffgpt(1, domain_of_definition=ranges.copy())
    twin.load(path)
    assert _same(_weights(saved), _weights(twin))
    other = _diffgpt(1, domain_of_definition=np.array([[-1.0, 1.0], [0.0, 3.0]]))
    with pytest.raises(ValueError, match="holds weights for domain_of_definition="):
        other.load(path)


@pytest.mark.parametrize(
    ("settings", "name"),
    [
        ({"tokenizer": "lgt"}, "tokenizer"),
        ({"tokenizer": "mash-half"}, "tokenizer"),
        ({"order_of_derivative": 2}, "order_of_derivative"),
        ({"domain_of_definition": np.array([1.0, 3.0])}, "domain_of_definition"),
    ],
)
def test_load_refuses_weights_saved_with_other_settings(
    tmp_path: Path,
    settings: dict[str, object],
    name: str,
) -> None:
    """Check that load raises on another setting and keeps the weights."""
    path = str(tmp_path / "model.bin")
    _diffgpt(0).save(path)
    other = _diffgpt(1, **settings)
    initial = _weights(other)
    with pytest.raises(ValueError, match=f"holds weights for {name}="):
        other.load(path)
    assert _same(initial, _weights(other))


def test_load_takes_a_bare_state_dict_unchecked(tmp_path: Path) -> None:
    """Check that a bare state dict, as train_numa writes it, loads unchecked."""
    path = str(tmp_path / "gpt.pt")
    saved = _diffgpt(0)
    torch.save(saved.model.state_dict(), path)
    other = _diffgpt(1, tokenizer="lgt")
    other.load(path)
    assert _same(_weights(saved), _weights(other))


@pytest.mark.parametrize("version", [ENCODER_VERSION - 1, None])
def test_load_refuses_weights_of_another_token_scheme(
    tmp_path: Path,
    version: int | None,
) -> None:
    """
    Check that load refuses a checkpoint of an older encoder, or of none.

    The tokens of version 1 differ from the current ones for the same
    settings, and checkpoints before version 2 record no version at all.
    """
    path = str(tmp_path / "model.bin")
    _diffgpt(0).save(path)
    checkpoint = torch.load(path, weights_only=True)
    if version is None:
        del checkpoint["encoder"]["encoder_version"]
    else:
        checkpoint["encoder"]["encoder_version"] = version
    torch.save(checkpoint, path)
    other = _diffgpt(1)
    initial = _weights(other)
    with pytest.raises(ValueError, match="holds weights for encoder_version="):
        other.load(path)
    assert _same(initial, _weights(other))
