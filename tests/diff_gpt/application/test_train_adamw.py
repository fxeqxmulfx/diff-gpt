"""Preserve native AdamW as the GPTMini training default and its warmup."""

from unittest.mock import patch

import torch
from torch.testing import assert_close

from diff_gpt.application.data_loader import TokenSequenceDataLoader
from diff_gpt.application.train import train
from diff_gpt.domain.model.gpt import GPT


def test_default_optimizer_matches_native_adamw_with_historical_warmup() -> None:
    """Every update agrees with independent native AdamW, including zero LR."""
    torch.manual_seed(7)
    model = GPT(64, 16, block_size=8, n_layer=1, n_head=2, use_checkpoint=False)
    reference = GPT(64, 16, block_size=8, n_layer=1, n_head=2, use_checkpoint=False)
    reference.load_state_dict(model.state_dict())
    tokens = torch.arange(9).unsqueeze(0).repeat(2, 1)
    loader = TokenSequenceDataLoader(
        list(range(64)) * 4, block_size=8, batch_size=2, device="cpu", train_part=1.0
    )
    optimizer = torch.optim.AdamW(
        reference.parameters(), lr=3e-4, betas=(0.9, 0.98), weight_decay=0.1
    )
    for step in range(4):
        optimizer.zero_grad(set_to_none=True)
        reference(tokens[:, :-1], tokens[:, 1:])[1].backward()
        optimizer.param_groups[0]["lr"] = 3e-4 * min(1, step / 2)
        optimizer.step()
    with patch.object(
        TokenSequenceDataLoader,
        "get_batch",
        return_value=(tokens[:, :-1], tokens[:, 1:]),
    ):
        train(
            model,
            loader,
            max_iters=4,
            warmup=2,
            eval_interval=2,
            eval_iters=1,
            use_tqdm=False,
        )
    assert_close(model.state_dict(), reference.state_dict(), rtol=0, atol=0)
