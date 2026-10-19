# Copyright 2025 Mikhail Borisov
"""Tests for the weights train() hands back with the schedule-free optimizer."""

from typing import TYPE_CHECKING

import torch

import diff_gpt.train as tr
from diff_gpt.data_loader import TokenSequenceDataLoader
from diff_gpt.model.gpt import GPT

if TYPE_CHECKING:
    import pytest


def test_train_returns_weights_at_schedule_free_x_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Check that train() without a val split hands back the weights at the x point.

    Without a val split nothing restores a snapshot, so the weights handed
    back must be the averaged x point (optimizer in eval mode), not the y point.
    """
    captured: list[tr.AdamWScheduleFree] = []

    class Spy(tr.AdamWScheduleFree):
        def eval(self) -> None:
            # train() switches its optimizer to eval, so this sees the instance.
            captured.append(self)
            super().eval()

    monkeypatch.setattr(tr, "AdamWScheduleFree", Spy)
    torch.manual_seed(0)
    tokens = torch.randint(0, 64, (2000,)).tolist()
    model = GPT(
        vocab_size=64,
        n_embd=16,
        block_size=8,
        n_head=2,
        n_layer=1,
        use_checkpoint=False,
    )
    loader = TokenSequenceDataLoader(
        tokens,
        block_size=8,
        batch_size=4,
        device="cpu",
        train_part=1.0,
    )
    tr.train(
        model,
        loader,
        max_iters=6,
        eval_interval=3,
        eval_iters=1,
        use_tqdm=False,
        use_early_stopping=False,
        save_best=True,
    )
    opt = captured[0]
    assert opt.param_groups[0]["train_mode"] is False
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt.eval()  # already at x: a second eval() must be a no-op
    for n, p in model.named_parameters():
        assert torch.equal(before[n], p)
