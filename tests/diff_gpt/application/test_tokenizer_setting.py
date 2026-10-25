"""
Tests for the tokenizer that DiffGPT and the DataFrame loader keep.

Both took the process default (DIFF_GPT_TOKENIZER) at import, so a model
trained with "mash" decoded silently wrong in a process whose default was
"lgt". The tokenizer is now a setting of each.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from diff_gpt.application.data_loader import DiffDataFrameDataLoader
from diff_gpt.application.diff_gpt import DiffGPT
from diff_gpt.domain.encoder_decoder import encode, get_domain_of_definition
from diff_gpt.domain.model.gpt import GPT

V = 64
ORDER = 2
TOKENIZERS = ("lgt", "mash", "mash-half")


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "a": np.cumsum(rng.normal(size=60)),
            "b": np.cumsum(rng.normal(size=60)),
        },
    )


def _whole_series_loader(
    df: pd.DataFrame,
    dod: np.ndarray,
    tokenizer: str,
) -> DiffDataFrameDataLoader:
    """Return a loader whose only window is the whole encoded series."""
    n_tokens = (len(df) - ORDER) * df.shape[1]
    return DiffDataFrameDataLoader(
        dfs=[df],
        block_size=n_tokens - 1,
        batch_size=1,
        vocab_size=V,
        order_of_derivative=ORDER,
        domain_of_definition=dod,
        device="cpu",
        train_part=1.0,
        tokenizer=tokenizer,
    )


@pytest.mark.parametrize("tokenizer", TOKENIZERS)
def test_dataframe_loader_encodes_with_its_tokenizer(tokenizer: str) -> None:
    """Check that the DataFrame loader encodes with the tokenizer it was given."""
    df = _frame()
    inp = df.to_numpy(dtype=np.float64)
    dod = get_domain_of_definition(inp, ORDER)
    tokens = {
        tok: encode(inp, V, dod, ORDER, tokenizer=tok)[2].reshape(-1).tolist()
        for tok in TOKENIZERS
    }
    assert tokens["lgt"] != tokens["mash"]
    assert tokens["mash-half"] != tokens["mash"]
    x, y = _whole_series_loader(df, dod, tokenizer).get_batch("train")
    assert x[0].tolist() == tokens[tokenizer][:-1]
    assert y[0].tolist() == tokens[tokenizer][1:]


def test_diffgpt_keeps_its_tokenizer_and_refuses_a_loader_with_another() -> None:
    """
    Check that DiffGPT keeps its tokenizer and won't train on another's tokens.

    Training must stop before the first update, leaving the weights as they
    were.
    """
    df = _frame()
    dod = get_domain_of_definition(
        df.to_numpy(dtype=np.float64),
        ORDER,
    )
    torch.manual_seed(0)
    model = DiffGPT(
        model=GPT(vocab_size=V, n_embd=16, block_size=32, n_head=2, n_layer=1),
        order_of_derivative=ORDER,
        domain_of_definition=dod,
        tokenizer="mash",
    )
    assert model.tokenizer == "mash"
    initial = {k: v.clone() for k, v in model.model.state_dict().items()}
    with pytest.raises(ValueError, match="loader encodes with tokenizer 'lgt'"):
        model.train(
            loader=_whole_series_loader(df, dod, "lgt"),
            max_iters=1,
            use_tqdm=False,
        )
    for k, v in model.model.state_dict().items():
        assert torch.equal(initial[k], v)
    with pytest.raises(ValueError, match="unknown tokenizer"):
        DiffGPT(
            model=model.model,
            order_of_derivative=ORDER,
            domain_of_definition=dod,
            tokenizer="nope",
        )
