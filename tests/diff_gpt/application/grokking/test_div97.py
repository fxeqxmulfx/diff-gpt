"""Check the grokking protocol against independent browser fixtures and the codec."""

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from benchmarks.grokking.div97 import (
    BATCH,
    DOMAIN,
    MIN_PLATEAU,
    TOTAL,
    TRAIN,
    VOCAB,
    Criterion,
    EpochSampler,
    P,
    RunConfig,
    digest,
    division_rows,
    domain,
    make_tensors,
    run_experiment,
    token_table,
    verify_checkpoint,
    vocabulary,
)
from diff_gpt.domain.encoder_decoder import decode, encode
from diff_gpt.domain.model.gpt import GPT

if TYPE_CHECKING:
    from pathlib import Path

# Generated independently with dist/model.js in the website, not this port.
BROWSER_ROWS = {
    11: (
        "af1518d3d6d9f078a11a6cfcdd0ed6318c2386ea41a9b6d9826d4064e1c1b01e",
        "fabbebaf7ab9a960308e775938abe42b8f5ced24da85f69f9890c078467c94ec",
    ),
    17: (
        "0ae63eaf37fd8c32ecf9ccff2790788fd3db7de4d2da6c253094ba1f19362bc8",
        "3bc9435f860665c066037288312ef09fbc6399e6bc1ff1fdd970be9242f72044",
    ),
    23: (
        "bfc8141de1293be4f60493ae24169a4b691cf46c1c86d9be3ecdd3e32fee08da",
        "08af43e6607898c94c01940c819b357b201ee74bdc019c776c0be6f4e50fc90e",
    ),
}
BROWSER_EPOCHS = (
    "a02e44fa563e7db95ae1d6f56221e4c105b59d47b1b1ad4986d17d73f0f1b37d",
    "6019a14e3e30c2730836faa0177ccd580ab560a0994102c8c8cbe43aaad92f9b",
    "74c8ca69f285eee54a64f40768110ef96a86a2f32b8d170f976a6f478bdd2645",
)
IGNORE_INDEX = -100
PROBES = 3


@pytest.mark.parametrize("seed", BROWSER_ROWS)
def test_rows_are_the_complete_task_with_the_browser_split(seed: int) -> None:
    """Every equation occurs once, the halves are disjoint and hashes match JS."""
    rows = division_rows(seed)
    assert rows.shape == (TOTAL, 3)
    assert np.unique(rows[:, :2], axis=0).shape[0] == TOTAL
    assert np.all(rows[:, 1] != 0)
    np.testing.assert_array_equal(rows[:, 1] * rows[:, 2] % P, rows[:, 0])
    assert (digest(rows[:TRAIN]), digest(rows[TRAIN:])) == BROWSER_ROWS[seed]


def test_epoch_sampler_matches_the_browser_without_replacement() -> None:
    """Three complete epochs include the exact tail and match JS permutations."""
    sampler = EpochSampler(11)
    for expected in BROWSER_EPOCHS:
        batches = list(sampler.batches())
        assert [len(batch) for batch in batches] == [BATCH] * 9 + [48]
        order = np.concatenate(batches)
        np.testing.assert_array_equal(np.sort(order), np.arange(TRAIN))
        assert digest(order) == expected


def test_zero_order_retains_every_residue_and_is_prefix_independent() -> None:
    """Encoding complete equations equals lookups, including a=b and a=0."""
    table, metadata = token_table()
    assert len(np.unique(table)) == P
    assert metadata["rounded_decoding_exact"]
    for values in ([0.0, 7.0, 0.0], [42.0, 42.0, 1.0], [1.0, 96.0, 96.0]):
        raw = np.array(values).reshape(-1, 1)
        prefix, _, tokens = encode(raw, VOCAB, DOMAIN, 0, tokenizer="lgt")
        assert prefix.shape == (0, 1)
        np.testing.assert_array_equal(tokens[:, 0], table[np.array(values, dtype=int)])


def test_answer_is_absent_from_context_and_only_it_is_supervised() -> None:
    """Changing labels cannot change the inputs; masked GPT loss scores only q."""
    table, _ = token_table()
    rows = division_rows(11)[:3]
    x, y = make_tensors(rows, table, "cpu")
    changed = rows.copy()
    changed[:, 2] = (changed[:, 2] + 1) % P
    changed_x, changed_y = make_tensors(changed, table, "cpu")
    assert torch.equal(x, changed_x)
    assert not torch.equal(y, changed_y)
    assert torch.all(y[:, 0] == IGNORE_INDEX)
    model = GPT(vocab_size=256, n_embd=8, n_head=1, n_layer=1, block_size=3)
    logits, loss = model(x, y)
    torch.testing.assert_close(loss, F.cross_entropy(logits[:, -1], y[:, -1]))


@pytest.mark.parametrize(("rise", "delayed"), [(250, True), (251, False)])
def test_grokking_uses_inclusive_plateau_and_first_rise(
    rise: int, *, delayed: bool
) -> None:
    """The website accepts 25 plateau epochs and at most 250 rise epochs."""
    criterion = Criterion()
    for epoch in range(26):
        criterion.observe(epoch, 1.0, 0.01)
    for epoch in range(26, 25 + rise):
        criterion.observe(epoch, 1.0, 0.5)
    criterion.observe(25 + rise, 1.0, 0.9)
    # A subsequent dip must not replace the first rise with a faster one.
    criterion.observe(26 + rise, 1.0, 0.01)
    criterion.observe(27 + rise, 1.0, 1.0)
    assert criterion.plateau_epochs == MIN_PLATEAU
    assert criterion.rise_epochs == rise
    assert criterion.delayed is delayed


def test_generalization_without_memorization_is_not_delayed_grokking() -> None:
    """A fast crossing alone cannot satisfy the memorization requirement."""
    criterion = Criterion()
    criterion.observe(0, 0.01, 0.01)
    criterion.observe(1, 1.0, 1.0)
    assert criterion.first99 == 1
    assert not criterion.delayed


@pytest.mark.parametrize("optimizer", ["adamw", "schedule-free"])
@pytest.mark.parametrize("order", [0, 1, 2])
def test_saved_weights_replay_full_metrics_and_public_diffgpt(
    tmp_path: Path,
    optimizer: str,
    order: int,
) -> None:
    """Both optimizer modes save the scored state and public predict agrees."""
    directory = tmp_path / optimizer
    config = RunConfig(width=8, heads=1, epochs=1, optimizer=optimizer, order=order)
    result = run_experiment(config, directory)
    verified = verify_checkpoint(directory)
    assert verified["epoch"] == result["epoch"] == 1
    assert verified["metrics"] == result["metrics"]
    assert len(verified["diffgpt_predict"]) == PROBES
    assert (
        json.loads((directory / "result.json").read_text())["config"]["optimizer"]
        == optimizer
    )


@pytest.mark.parametrize("order", [0, 1])
def test_full_public_verification_keeps_the_original_answers(
    tmp_path: Path, order: int
) -> None:
    """Every test forecast is scored; the query replay still decodes against q."""
    directory = tmp_path / "run"
    run_experiment(RunConfig(width=8, heads=1, epochs=1, order=order), directory)
    verified = verify_checkpoint(directory, full_predict=True)
    public = verified["diffgpt_predict_many"]
    assert public["test_rows"] == TRAIN
    assert 0 <= public["correct"] <= TRAIN
    assert set(verified["query_metrics"]) == {"train", "test"}
    if order:
        assert "decoded_accuracy" in verified["query_metrics"]["test"]


def test_a_checkpoint_of_an_earlier_model_is_rejected(tmp_path: Path) -> None:
    """Parameter-free norms load into any GPT; only the saved marker differs."""
    directory = tmp_path / "run"
    run_experiment(RunConfig(width=8, heads=1, epochs=1), directory)
    checkpoint = torch.load(directory / "checkpoint.pt", weights_only=True)
    assert checkpoint.pop("normalization") == "peri_rms_qk_rms"
    torch.save(checkpoint, directory / "checkpoint.pt")
    with pytest.raises(ValueError, match="Peri/QK-RMSNorm"):
        verify_checkpoint(directory)


@pytest.mark.parametrize("order", [1, 2])
def test_positive_order_is_the_actual_codec_and_keeps_all_operands(order: int) -> None:
    """Zero prefixes are fixed; cached differences equal independently encoded rows."""
    rows = division_rows(11)
    table, metadata = token_table(order)
    assert metadata["scale"] == [1.0]
    assert metadata["max_decoding_error"] == 0.0
    inputs, targets = make_tensors(rows, table, "cpu", order)
    assert torch.unique(inputs, dim=0).shape[0] == TOTAL
    for index, row in enumerate(rows[:32]):
        raw = np.concatenate((np.zeros(order), row)).reshape(-1, 1)
        prefix, scale, tokens = encode(
            raw,
            vocabulary(order),
            domain(order),
            order,
            tokenizer="lgt",
        )
        np.testing.assert_array_equal(tokens[:2, 0], inputs[index].numpy())
        assert tokens[-1, 0] == targets[index, -1].item()
        decoded = decode(
            prefix, scale, tokens, vocabulary(order), order, tokenizer="lgt"
        )
        np.testing.assert_array_equal(decoded, raw)


def test_without_initial_level_first_order_inputs_are_ambiguous() -> None:
    """Equal operands lose their absolute value, but need distinct output deltas."""
    contexts = []
    outputs = []
    for a in (1, 2):
        _, _, tokens = encode(
            np.array([[a], [a], [1]], dtype=np.float64),
            vocabulary(1),
            domain(1),
            1,
            tokenizer="lgt",
        )
        contexts.append(tokens[0, 0])
        outputs.append(tokens[1, 0])
    assert contexts[0] == contexts[1]
    assert outputs[0] != outputs[1]
