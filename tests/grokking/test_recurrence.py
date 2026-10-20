"""Validate temporal differences, disjoint trajectories and resumable training."""

from copy import deepcopy
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from benchmarks.grokking.div97 import domain, vocabulary
from benchmarks.grokking.recurrence import (
    BATCH,
    DEFAULT_BETA1,
    Config,
    P,
    batch_slices,
    build,
    dataset,
    encode_windows,
    future_grid,
    grid_domain,
    grid_vocabulary,
    run,
    tensors,
    train_batch,
    trajectory_orbits,
    verify,
    windows,
)
from benchmarks.grokking.recurrence_readouts import (
    AliasSampler,
    FieldSampler,
    MeanSampler,
    QuantizedFieldSampler,
    RangeSampler,
)
from diff_gpt.encoder_decoder import decode, encode
from diff_gpt.optimizer.schedule_free import ScheduleFreeAdamW
from diff_gpt.sampler.temperature import TemperatureSampler

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("task", ["division", "fibonacci"])
def test_training_and_test_have_disjoint_complete_orbits(task: str) -> None:
    """No state, or future/past shift of that state, crosses the split."""
    config = Config(task=task)
    raw, train, metadata = dataset(config)
    orbits = trajectory_orbits(task)
    assert not set(metadata["train_orbits"]) & set(metadata["test_orbits"])
    states = {state for orbit in orbits for state in orbit}
    expected = (P - 1) ** 2 if task == "division" else P**2 - 1
    assert len(states) == sum(map(len, orbits)) == len(raw) == expected
    train_states = set(map(tuple, raw[:train, :2]))
    test_states = set(map(tuple, raw[train:, :2]))
    assert not train_states & test_states
    assert train_states | test_states == states
    assert metadata["optimal_full_table_token_accuracy"] == 1.0
    assert not metadata["synthetic_prefix"]


def test_division_is_between_successive_rows_of_one_column() -> None:
    """The entire context follows the recurrence, without inserted zeros."""
    config = Config()
    raw = windows([(6, 4)], config)
    np.testing.assert_array_equal(raw[0, :5], [6, 4, 50, 35, 43])
    np.testing.assert_array_equal(raw[:, 2:] * raw[:, 1:-1] % P, raw[:, :-2])


@pytest.mark.parametrize(
    ("supervise_from", "expected"),
    [
        (0, [-100, -100, -100, 158]),
        (1, [174, 113, 136, 158]),
        (3, [-100, -100, 136, 158]),
    ],
)
def test_prefix_supervision_uses_true_next_tokens_without_future_inputs(
    supervise_from: int,
    expected: list[int],
) -> None:
    """Intermediate targets follow actual rows; the held-out future stays absent."""
    config = Config(context=4, supervise_from=supervise_from)
    raw = windows([(6, 4)], config)
    np.testing.assert_array_equal(raw[0], [6, 4, 50, 35, 43, 73])
    x, y = tensors(raw, config, "cpu")
    assert x.tolist() == [[126, 174, 113, 136]]
    assert y.tolist() == [expected]
    changed = raw.copy()
    changed[:, -1] = 74
    changed_x, changed_y = tensors(changed, config, "cpu")
    torch.testing.assert_close(x, changed_x, atol=0, rtol=0)
    torch.testing.assert_close(y[:, :-1], changed_y[:, :-1], atol=0, rtol=0)
    assert changed_y[:, -1].tolist() == [159]


def test_prefix_target_cannot_be_read_from_a_later_input_position() -> None:
    """Causal attention hides a target that is teacher-forced later in the window."""
    config = Config(width=8, layers=2, heads=1, context=4, supervise_from=1)
    x, y = tensors(windows([(6, 4)], config), config, "cpu")
    assert y[0, 2] == x[0, 3]
    model = build(config, "cpu").eval()
    changed = x.clone()
    changed[:, -1] = (changed[:, -1] + 7) % vocabulary(1)
    with torch.inference_mode():
        before = model(x)[0][:, :-1]
        after = model(changed)[0][:, :-1]
    torch.testing.assert_close(before, after, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("task", ["division", "fibonacci"])
@pytest.mark.parametrize("order", [1, 2])
def test_differences_are_down_rows_and_agree_with_the_actual_codec(
    task: str,
    order: int,
) -> None:
    """Cached integer differences and actual encode/decode agree end to end."""
    config = Config(task=task, order=order)
    raw, _, _ = dataset(config)
    x, y = tensors(raw, config, "cpu")
    for index, row in enumerate(raw[:16]):
        prefix, scale, tokens = encode(
            row[:, None].astype(np.float64),
            vocabulary(order),
            domain(order),
            order,
            tokenizer="lgt",
        )
        np.testing.assert_array_equal(tokens[:-1, 0], x[index].numpy())
        assert tokens[-1, 0] == y[index, -1].item()
        decoded = decode(
            prefix, scale, tokens, vocabulary(order), order, tokenizer="lgt"
        )
        np.testing.assert_array_equal(decoded[:, 0], row)
    # Replacing a future row changes only its supervised derivative token.
    changed = raw.copy()
    changed[:, -1] = (changed[:, -1] + 1) % P
    changed_x, changed_y = tensors(changed, config, "cpu")
    assert torch.equal(changed_x, x)
    assert not torch.equal(changed_y, y)


@pytest.mark.parametrize(
    ("readout", "batching", "batch_size", "micro_batch_size", "beta1"),
    [
        ("argmax", "tail", 512, 0, DEFAULT_BETA1),
        ("field", "tail", 512, 0, DEFAULT_BETA1),
        ("argmax", "balanced", 512, 0, DEFAULT_BETA1),
        ("argmax", "tail", 8192, 512, DEFAULT_BETA1),
        ("argmax", "balanced", 512, 0, 0.5),
    ],
)
def test_resume_reproduces_uninterrupted_weights_and_optimizer(
    tmp_path: Path,
    readout: str,
    batching: str,
    batch_size: int,
    micro_batch_size: int,
    beta1: float,
) -> None:
    """Restored query, sampler and averaging iterates match an unbroken run."""
    config = Config(
        width=8,
        layers=1,
        heads=1,
        context=4,
        epochs=1,
        readout=readout,
        batching=batching,
        batch_size=batch_size,
        micro_batch_size=micro_batch_size,
        beta1=beta1,
    )
    split = tmp_path / "split"
    whole = tmp_path / "whole"
    run(config, split, "cpu")
    if (
        beta1 == DEFAULT_BETA1
        and readout == "argmax"
        and batching == "tail"
        and batch_size == BATCH
    ):
        # Existing long runs predate the beta1 and restart config fields.
        legacy = torch.load(split / "checkpoint.pt", weights_only=True)
        legacy["config"].pop("beta1")
        legacy["config"].pop("restart_step")
        legacy["config"].pop("restart_warmup")
        torch.save(legacy, split / "checkpoint.pt")
        assert "-b1" not in config.name()
    if beta1 != DEFAULT_BETA1:
        assert f"-b1{beta1:g}" in config.name()
    longer = replace(config, epochs=2)
    resumed = run(longer, split, "cpu", resume=True)
    uninterrupted = run(longer, whole, "cpu")
    assert resumed["metrics"] == uninterrupted["metrics"]
    first = torch.load(split / "checkpoint.pt", weights_only=True)
    second = torch.load(whole / "checkpoint.pt", weights_only=True)
    assert first["optimizer"]["param_groups"][0]["betas"][0] == beta1
    for key, value in first["model"].items():
        torch.testing.assert_close(value, second["model"][key], atol=0, rtol=0)
    for key, state in first["optimizer"]["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(
                value, second["optimizer"]["state"][key][name], atol=0, rtol=0
            )
    assert first["sampler_state"] == second["sampler_state"]
    assert first["sampler_order"] == second["sampler_order"]
    assert first["optimizer"]["param_groups"] == second["optimizer"]["param_groups"]
    updates = (first["train_rows"] + batch_size - 1) // batch_size
    assert first["optimizer"]["param_groups"][0]["step"] == longer.epochs * updates
    checked = verify(split, "cpu", full=readout == "field")
    assert checked["metrics"] == resumed["metrics"]


def test_resume_leaves_a_run_that_reached_its_epochs(tmp_path: Path) -> None:
    """Resuming without more epochs would train nothing and replace the result."""
    config = Config(width=8, layers=1, heads=1, context=4, epochs=1)
    directory = tmp_path / "run"
    run(config, directory, "cpu")
    files = {path.name: path.read_bytes() for path in directory.iterdir()}
    with pytest.raises(ValueError, match="already reached epoch 1"):
        run(config, directory, "cpu", resume=True)
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == files


def test_balanced_batches_keep_every_sample_and_avoid_the_six_sample_tail() -> None:
    """Batch boundaries change without resampling, dropping rows or extra updates."""
    order = list(reversed(range(4614)))
    ordinary = [order[part] for part in batch_slices(len(order), "tail")]
    balanced = [order[part] for part in batch_slices(len(order), "balanced")]
    assert [len(batch) for batch in ordinary] == [512] * 9 + [6]
    assert sorted(len(batch) for batch in balanced) == [461] * 6 + [462] * 4
    assert [index for batch in balanced for index in batch] == order


@pytest.mark.parametrize("supervise_from", [0, 1, 3])
def test_accumulation_matches_a_full_batch_gradient_and_single_optimizer_update(
    supervise_from: int,
) -> None:
    """An uneven final microbatch has its true weight in the global mean."""
    config = Config(
        width=8, layers=1, heads=1, context=4, supervise_from=supervise_from
    )
    raw = windows([(6, 4), (5, 2), (7, 3), (96, 1), (1, 96), (31, 17), (4, 9)], config)
    x, y = tensors(raw, config, "cpu")
    indices = torch.tensor([6, 0, 3, 1, 5, 2, 4])
    torch.manual_seed(2026)
    whole = build(config, "cpu")
    accumulated = deepcopy(whole)
    optimizers = [
        ScheduleFreeAdamW(
            model.parameters(), lr=0.01, weight_decay=1, betas=(0.9, 0.98)
        )
        for model in (whole, accumulated)
    ]
    train_batch(whole, optimizers[0], x, y, indices)
    train_batch(accumulated, optimizers[1], x, y, indices, micro_batch_size=3)
    for first, second in zip(whole.parameters(), accumulated.parameters(), strict=True):
        torch.testing.assert_close(first.grad, second.grad, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(first, second, atol=2e-6, rtol=2e-5)
        for key in ("base", "evaluation", "second"):
            torch.testing.assert_close(
                optimizers[0].state[first][key],
                optimizers[1].state[second][key],
                atol=2e-6,
                rtol=2e-5,
            )
    assert [optimizer.param_groups[0]["step"] for optimizer in optimizers] == [1, 1]


def test_field_readout_preserves_training_weights_and_optimizer(tmp_path: Path) -> None:
    """An inference-only change cannot consume the training RNG or modify weights."""
    config = Config(task="fibonacci", width=8, heads=1, layers=1, context=2, epochs=1)
    saved = []
    for readout in ("argmax", "field"):
        directory = tmp_path / readout
        run(replace(config, readout=readout), directory, "cpu")
        saved.append(torch.load(directory / "checkpoint.pt", weights_only=True))
    first, second = saved
    for key, value in first["model"].items():
        torch.testing.assert_close(value, second["model"][key], atol=0, rtol=0)
    for key, state in first["optimizer"]["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(
                value, second["optimizer"]["state"][key][name], atol=0, rtol=0
            )
    assert first["sampler_state"] == second["sampler_state"]
    assert first["sampler_order"] == second["sampler_order"]


@pytest.mark.parametrize("tokenizer", ["lgt", "mash"])
@pytest.mark.parametrize("order", [1, 2])
def test_nonunit_grid_replays_actual_encoding_and_reanchored_decoder(
    tokenizer: str, order: int
) -> None:
    """Quantized tokens and one-step decoder bases match the public codec."""
    config = Config(
        task="fibonacci",
        tokenizer=tokenizer,
        order=order,
        grid_domain=96 * 2 ** (order - 1),
        vocab_size=512 * 2 ** (order - 1),
    )
    raw = windows([(6, 4), (96, 1), (0, 96), (31, 72)], config)
    tokens, scale = encode_windows(raw, config)
    bases = future_grid(raw[:, :-1], config)
    for index, row in enumerate(raw):
        _, actual_scale, actual_tokens = encode(
            row[:, None].astype(float),
            grid_vocabulary(config),
            grid_domain(config),
            config.order,
            tokenizer=tokenizer,
        )
        np.testing.assert_array_equal(tokens[index], actual_tokens[:, 0])
        assert scale[index] == actual_scale[0]
        actual = decode(
            row[-1 - order : -1, None].astype(float),
            actual_scale,
            actual_tokens[-1:],
            grid_vocabulary(config),
            config.order,
            tokenizer=tokenizer,
        )[-1, 0]
        predicted = (
            bases[index, 0]
            + (tokens[index, -1] - grid_vocabulary(config) // 2) * bases[index, 1]
        )
        np.testing.assert_allclose(predicted, actual, atol=1e-12, rtol=0)
    # Decoder bases cannot depend on a row that has not been observed yet.
    changed = raw.copy()
    changed[:, -1] = (changed[:, -1] + 17) % P
    np.testing.assert_array_equal(future_grid(changed[:, :-1], config), bases)
    x, _ = tensors(raw, config, "cpu")
    changed_x, _ = tensors(changed, config, "cpu")
    assert torch.equal(x, changed_x)


def test_range_sampler_uses_only_observed_rows_and_known_support() -> None:
    """The support mask is independent of the recurrence and its unknown answer."""
    config = Config(task="fibonacci", order=2)
    past = torch.tensor([[96, 0], [1, 96]])
    sampler = RangeSampler(past, config, TemperatureSampler(temperature=0))
    logits = torch.zeros(2, vocabulary(config.order))
    logits[:, 1] = 100  # This would decode outside the known range.
    result = sampler(logits)
    assert result.tolist() == [[352], [65]]  # Both decode to the lower bound, zero.
    assert sampler.allowed.sum(-1).tolist() == [97, 97]


def test_field_sampler_combines_alias_mass_and_returns_an_actual_derivative() -> None:
    """A range mask differs from summing the model's cyclic hypotheses."""
    config = Config(task="fibonacci", order=1)
    past = torch.tensor([[96], [0]])
    logits = torch.full((2, 256), -100.0)
    # Decoded hypotheses 99 and 2 share a canonical residue.
    logits[0, 131] = 2
    logits[0, 34] = 1
    # Two equivalent hypotheses beat a larger single-bin competitor.
    logits[1, 129] = 1
    logits[1, 226] = 1
    logits[1, 130] = 1.5
    assert FieldSampler(past, config)(logits).tolist() == [[34], [129]]


def test_mean_readout_can_interpolate_an_unseen_ordinal_bin() -> None:
    """Symmetric hypotheses select their midpoint even when that bin has no mass."""
    logits = torch.tensor(
        [
            [0.0, -torch.inf, 0.0, -torch.inf, -torch.inf],
            [0.0, -torch.inf, -torch.inf, -torch.inf, 0.0],
        ]
    )
    assert MeanSampler()(logits).tolist() == [[1], [2]]


@pytest.mark.parametrize("tokenizer", ["lgt", "mash"])
@pytest.mark.parametrize("order", [1, 2])
def test_quantized_field_pools_aliases_and_emits_the_actual_canonical_token(
    tokenizer: str, order: int
) -> None:
    """Independent public decoding checks residue pooling on both non-unit codecs."""
    config = Config(
        task="fibonacci",
        order=order,
        context=2,
        tokenizer=tokenizer,
        grid_domain=96 * 2 ** (order - 1),
        vocab_size=512 * 2 ** (order - 1),
    )
    past = np.array([[4, 96, 3, 95], [96, 1, 0, 96]])[:, -(order + 2) :]
    vocab = grid_vocabulary(config)
    _, scale, _ = encode(
        past.T.astype(float),
        vocab,
        np.repeat(grid_domain(config), len(past)),
        order,
        tokenizer=tokenizer,
    )
    decoded = decode(
        np.repeat(past[:, -order:].T.astype(float), vocab, axis=1),
        np.repeat(scale, vocab),
        np.tile(np.arange(vocab), len(past))[None, :],
        vocab,
        order,
        tokenizer=tokenizer,
    )[-1].reshape(len(past), vocab)
    values = np.rint(decoded).astype(int)
    logits = torch.full((len(past), vocab), -100.0)
    wanted, competitor = 2, 3
    for row in range(len(past)):
        aliases = np.flatnonzero(values[row] % P == wanted)
        assert values[row, aliases[0]] != values[row, aliases[-1]]
        logits[row, aliases[[0, -1]]] = 1.0
        logits[row, np.flatnonzero(values[row] == competitor)[0]] = 1.5
    result = QuantizedFieldSampler(torch.as_tensor(past), config)(logits)
    for row, token in zip(past, result[:, 0].tolist(), strict=True):
        _, _, actual = encode(
            np.append(row, wanted)[:, None].astype(float),
            vocab,
            grid_domain(config),
            order,
            tokenizer=tokenizer,
        )
        assert token == actual[-1, 0]


def test_alias_readout_is_invariant_to_observed_integer_representatives() -> None:
    """Equivalent observed field values give the same forecast without a target."""
    config = Config(task="fibonacci", width=8, heads=1, layers=1, context=2)
    torch.manual_seed(2026)
    model = build(config, "cpu").eval()
    past = torch.tensor([[5, 9, 14]])
    alternate = torch.tensor([[5, 106, 14]])  # Same observed residues modulo 97.
    before = {key: value.clone() for key, value in model.state_dict().items()}
    outputs = []
    for observed in (past, alternate):
        readout = AliasSampler(observed, config, model)
        tokens, _ = encode_windows(observed.numpy(), config)
        with torch.inference_mode():
            logits = model(torch.as_tensor(tokens))[0][:, -1]
            outputs.append(readout(logits))
        assert readout.prompts.min() >= vocabulary(1) // 2 - (P - 1)
        assert readout.prompts.max() <= vocabulary(1) // 2 + (P - 1)
    torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], atol=0, rtol=0)
