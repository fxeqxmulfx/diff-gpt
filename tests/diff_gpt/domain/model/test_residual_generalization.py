"""
Preserve both independently trained scaled-residual generalization endpoints.

Frozen lab task145 supplies weights; modular arithmetic supplies labels. These
CPU tests preserve learned held-out behavior, not a promise about retraining
speed. See tests/fixtures/residual_generalization/README.md for provenance.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from diff_gpt.domain.model.normalized_gpt import NormalizedGPT as GPT

FIXTURES = Path(__file__).resolve().parents[3] / "fixtures/residual_generalization"


@torch.no_grad()
def correct_count(model: GPT, rows: torch.Tensor) -> int:
    """Exhaust the held-out split in fixed CPU batches."""
    return sum(
        int((model(part[:, :-1])[0][:, -1].argmax(-1) == part[:, -1]).sum())
        for part in rows.split(512)
    )


@pytest.mark.parametrize("seed", [11, 17])
def test_scaled_residual_preserves_learned_unseen_division(seed: int) -> None:
    """Retain both seeds' scores and reject disabling the learned branch balance."""
    torch.set_num_threads(1)
    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    case = next(
        c for c in manifest["cases"] if c["label"] == f"mechanism-s{seed}-scaled"
    )
    path = FIXTURES / case["file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == case["sha256"]
    configuration = dict(case["configuration"])
    assert configuration.pop("mixing") == "scaled"
    # Frozen fixtures describe their historical constructor, not today's API.
    for historical in ("centered_xsa", "rope_layers"):
        configuration.pop(historical, None)
    model = GPT(**configuration).float().eval()
    with np.load(path, allow_pickle=False) as archive:
        rows = torch.from_numpy(archive["test_rows"]).long()
        training = torch.from_numpy(archive["train_inputs"]).long()
        model.load_state_dict(
            {
                name: torch.from_numpy(archive[name])
                for name in archive.files
                if name not in {"test_rows", "train_inputs"}
            }
        )
    prime = case["configuration"]["vocab_size"]
    assert len(rows) == case["test_examples"]
    assert all(
        a * pow(b, -1, prime) % prime == target for a, b, target in rows.tolist()
    )
    test_inputs = set((rows[:, 0] * prime + rows[:, 1]).tolist())
    train_inputs = set((training[:, 0] * prime + training[:, 1]).tolist())
    assert len(test_inputs) == len(rows)
    assert test_inputs.isdisjoint(train_inputs)
    correct = correct_count(model, rows)
    # Permit one rounded argmax to differ across supported CPU kernels.
    assert correct >= case["test_correct"] - 1
    assert correct / len(rows) >= case["minimum_accuracy"]
    for block in model.blocks:
        block.residual_scale = 1.0
    assert correct_count(model, rows) / len(rows) < case["minimum_accuracy"]
