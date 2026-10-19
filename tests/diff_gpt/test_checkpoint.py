# Copyright 2025 Mikhail Borisov
"""
Tests for the encoder settings that DiffGPT.save writes and DiffGPT.load checks.

Weights decode right only with the tokenizer, order, domain and use_decimal
they were trained with. `save` wrote the weights alone, so `load` took them
into a DiffGPT with other settings, which then decoded silently wrong.
"""

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from diff_gpt.diff_gpt import DiffGPT
from diff_gpt.encoder_decoder import np_to_decimal
from diff_gpt.model.gpt import GPT

if TYPE_CHECKING:
    from pathlib import Path

DOMAIN = np.array([1.0, 2.0])
SETTINGS = {
    "order_of_derivative": 1,
    "domain_of_definition": DOMAIN,
    "use_decimal": False,
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
    for settings in ({}, {"order_of_derivative": np.array([1, 1])}):
        twin = _diffgpt(1, **settings)
        twin.load(path)
        assert _same(_weights(saved), _weights(twin))
        assert not twin.model.training


@pytest.mark.parametrize(
    ("settings", "name"),
    [
        ({"tokenizer": "lgt"}, "tokenizer"),
        ({"order_of_derivative": 2}, "order_of_derivative"),
        ({"domain_of_definition": np.array([1.0, 3.0])}, "domain_of_definition"),
        (
            {"use_decimal": True, "domain_of_definition": np_to_decimal(DOMAIN)},
            "use_decimal",
        ),
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
