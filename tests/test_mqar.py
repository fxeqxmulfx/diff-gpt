"""Check the MQAR benchmark against the lab's splits, its contract and its runs."""

import json
import random
from typing import TYPE_CHECKING, Any

import pytest
import torch
from torch.nn import functional as F

from benchmarks.mqar import (
    IGNORE,
    MQAR,
    RunConfig,
    build_split,
    evaluate,
    fingerprint,
    labels,
    run,
    uniform_draws,
)
from diff_gpt.domain.model.rope import rope_tables

if TYPE_CHECKING:
    from pathlib import Path

# Generated with the lab's generator, not this port: in the lab's venv,
# build_split(generator(Problem(MQAR(**controls), length, min_length)),
#             split, seed, count).fingerprint
LAB_SPLITS = [
    (
        {"symbols": 64, "pairs": 8, "queries": 4, "alpha": 0.1},
        (64, 64, "train", 0, 3),
        "3c5d6848ab2a814ee050e56a34f6e611dbd0e0ff86bc624e269314c85665fde2",
    ),
    (
        {"symbols": 64, "pairs": 8, "queries": 4, "alpha": 0.1},
        (64, None, "train", 0, 16),
        "10b4c4f3c3bffc1489775fa9f1c1a3d4012c8f7dd33adf91d40dae0c39bb966b",
    ),
    (
        {"symbols": 64, "pairs": 8, "queries": 4, "alpha": 0.1},
        (128, None, "test", 0, 16),
        "e04dec2c99a2e7c93d7119ffa0a82877d453c36e451c0d383185224259d93796",
    ),
    (
        {"symbols": 64, "pairs": 8, "queries": 8, "alpha": 0.1},
        (64, None, "validation", 1, 16),
        "39fc0fddfede189df136104933ca3035043985b498b5a83f2da59b6eb050be10",
    ),
    (
        {
            "symbols": 32,
            "pairs": 8,
            "queries": 4,
            "query_gap": 5,
            "alpha": 0.5,
            "overwrites": 3,
        },
        (48, 30, "train", 2, 16),
        "c62044c079cefd0816324c86daa54e7d21a93271b0139210e99ae73e76a1c686",
    ),
    (
        {},
        (40, None, "test", 0, 16),
        "7baacfffb249a7662bc24ccf03e0b317042829fdf7d19155ecbdb9fb46613e3a",
    ),
]
# The first example of the lab's first split above, and of its rewrite split.
LAB_FIRST = [
    1, 86, 140, 52, 149, 93, 129, 83, 150, 96, 157, 61, 137, 41, 136, 82,
    129, 161, 118, 135, 152, 41, 109, 103, 123, 104, 145, 123, 162, 143, 163, 61,
    83, 136, 153, 139, 122, 140, 105, 105, 156, 162, 116, 147, 138, 135, 132, 132,
    137, 112, 139, 134, 151, 103, 126, 135, 93, 105, 145, 153, 157, 103, 100, 158,
]  # fmt: skip
LAB_REWRITES = [
    1, 39, 83, 50, 71, 52, 88, 52, 73, 48, 73, 38, 98, 52, 92, 48,
    81, 78, 96, 98, 71, 70, 78, 95, 52, 77, 77, 87, 80, 83, 98, 74,
    85, 78, 99, 48, 38, 89, 39, 97, 72, 70, 90, 77,
]  # fmt: skip
TINY = {
    "symbols": 8,
    "pairs": 2,
    "queries": 2,
    "length": 12,
    "ood": (16, 24),
    "train": 64,
    "test": 8,
    "width": 16,
    "heads": 2,
    "layers": 1,
    "warmup": 2,
    "batch": 8,
    "steps": 4,
    "evaluate_every": 2,
}


@pytest.mark.parametrize(("controls", "problem", "expected"), LAB_SPLITS)
def test_splits_are_the_labs(
    controls: dict[str, Any], problem: tuple[Any, ...], expected: str
) -> None:
    """The port draws the lab's examples in the lab's order, so the hash agrees."""
    task = MQAR(**controls)
    length, min_length, split, seed, count = problem
    examples = build_split(task, length, split, seed, count, min_length)
    assert fingerprint(task, length, split, seed, examples, min_length) == expected


@pytest.mark.parametrize(
    ("controls", "problem", "tokens", "answers"),
    [
        (
            {"symbols": 64, "pairs": 8, "queries": 4, "alpha": 0.1},
            (64, 64, "train", 0),
            LAB_FIRST,
            {21: 136, 31: 137, 32: 150, 56: 129},
        ),
        (
            {
                "symbols": 32,
                "pairs": 8,
                "queries": 4,
                "query_gap": 5,
                "alpha": 0.5,
                "overwrites": 3,
            },
            (48, 30, "train", 2),
            LAB_REWRITES,
            {24: 92, 35: 81, 36: 98, 38: 83},
        ),
    ],
)
def test_queries_are_answered_with_the_latest_write(
    controls: dict[str, Any],
    problem: tuple[Any, ...],
    tokens: list[int],
    answers: dict[int, int],
) -> None:
    """Each query is supervised with the last value written for its key."""
    task = MQAR(**controls)
    length, min_length, split, seed = problem
    [(drawn, targets)] = build_split(task, length, split, seed, 1, min_length)
    latest = {}
    for position in range(1, task.end, 2):
        latest[tokens[position]] = tokens[position + 1]
    assert drawn == tokens
    assert {p: t for p, t in enumerate(targets) if t != IGNORE} == answers
    assert all(latest[tokens[p]] == value for p, value in answers.items())


@pytest.mark.parametrize("width", [1, 5, 64, 100, 1025])
def test_uniform_draws_are_randrange_draws(width: int) -> None:
    """The draws are CPython's randrange draws and leave the same state."""
    ours, reference = random.Random(width), random.Random(width)  # noqa: S311
    draws = uniform_draws(ours, 7, 7 + width, 200)
    assert draws == [reference.randrange(7, 7 + width) for _ in range(200)]
    assert ours.getstate() == reference.getstate()


def broken(tokens: list[int], changes: dict[int, int]) -> list[int]:
    """Return a copy of `tokens` with `changes` written over it."""
    result = list(tokens)
    for position, token in changes.items():
        result[position] = token
    return result


@pytest.mark.parametrize(
    ("controls", "tokens", "message"),
    [
        ({}, broken(LAB_FIRST, {0: 2}), "incomplete MQAR prefix"),
        ({}, LAB_FIRST[:16], "incomplete MQAR prefix"),
        ({}, broken(LAB_FIRST, {1: 140}), "invalid MQAR write"),
        ({}, broken(LAB_FIRST, {2: 86}), "invalid MQAR write"),
        ({}, broken(LAB_FIRST, {3: 86}), "rewrite must change"),
        ({}, broken(LAB_FIRST, {21: 99}), "one distinct prior key"),
        ({}, broken(LAB_FIRST, {31: 41}), "one distinct prior key"),
        ({}, broken(LAB_FIRST, {40: 0}), "fillers must be value tokens"),
        ({}, broken(LAB_FIRST, {56: 158}), "wrong number of supervised queries"),
        (
            {"query_gap": 5},
            broken(LAB_FIRST, {17: 41, 21: 118}),
            "one distinct prior key",
        ),
        ({"overwrites": 1}, LAB_FIRST, "pairs - overwrites distinct keys"),
    ],
)
def test_labels_reject_inputs_outside_the_task(
    controls: dict[str, Any], tokens: list[int], message: str
) -> None:
    """The oracle refuses inputs that break the writes, queries or fillers."""
    task = MQAR(**{"symbols": 64, "pairs": 8, "queries": 4, **controls})
    with pytest.raises(ValueError, match=message):
        labels(task, tokens)


def test_labels_reject_a_rewrite_to_the_current_value() -> None:
    """A rewrite that keeps the value is no rewrite: the task never draws one."""
    task = MQAR(symbols=32, pairs=8, queries=4, query_gap=5, alpha=0.5, overwrites=3)
    with pytest.raises(ValueError, match="rewrite must change"):
        labels(task, broken(LAB_REWRITES, {8: 88}))


@pytest.mark.parametrize(
    "controls",
    [
        {"overwrites": -1},
        {"overwrites": 8},
        {"queries": 0},
        {"queries": 9},
        {"queries": 8, "overwrites": 1},
        {"symbols": 4},
        {"symbols": 1, "pairs": 2, "queries": 1, "overwrites": 1},
        {"query_gap": -1},
        {"alpha": -0.5},
        {"alpha": float("nan")},
    ],
)
def test_task_rejects_invalid_controls(controls: dict[str, Any]) -> None:
    """The task refuses the controls the lab's MQAR refuses."""
    with pytest.raises(ValueError, match=r"require|rewrite|must"):
        MQAR(**controls)


@pytest.mark.parametrize(
    "settings",
    [
        {"length": 6},
        {"ood": (12,)},
        {"ood": (16, 16)},
        {"steps": 0},
        {"test": 0},
        {"rope": ""},
        {"rope": "10"},
        {"rope": "2"},
        {"rope_base": 0.0},
    ],
)
def test_run_rejects_invalid_settings(settings: dict[str, Any]) -> None:
    """A run needs room for the queries, longer test lengths and a RoPE layout."""
    with pytest.raises(
        ValueError, match=r"MQAR needs|ood lengths|must be positive|rope must"
    ):
        RunConfig(**{**TINY, **settings})


def test_rope_settings_build_their_gpt_and_name_their_run() -> None:
    """The RoPE layers and base reach the GPT; only other values change the name."""
    config = RunConfig(**{**TINY, "layers": 2, "rope": "10", "rope_base": 500.0})
    model = config.model(config.length)
    assert [block.sa.use_rope for block in model.blocks] == [True, False]
    cos, sin = rope_tables(model.blocks[0].sa.head_dim, config.length, 500.0)
    assert torch.equal(model.rope_cos, cos)
    assert torch.equal(model.rope_sin, sin)
    assert "-l2-rope10-base500-" in config.name
    assert "rope" not in RunConfig(**TINY).name
    assert "base" not in RunConfig(**TINY).name


def test_evaluate_scores_each_query_and_each_example() -> None:
    """Batched scores equal the scores of the examples one by one."""
    config = RunConfig(**TINY)
    torch.manual_seed(0)
    model = config.model(config.length).eval()
    examples = build_split(config.task, config.length, "test", 0, 8)
    tokens = torch.tensor([x for x, _ in examples])
    targets = torch.tensor([y for _, y in examples])
    scores = evaluate(model, tokens, targets)
    hits, losses, exact = [], [], []
    for x, y in zip(tokens, targets, strict=True):
        logits, _ = model(x[None])
        supervised = y != IGNORE
        row = logits[0, supervised]
        losses += F.cross_entropy(row, y[supervised], reduction="none").tolist()
        correct = (row.argmax(-1) == y[supervised]).tolist()
        hits += correct
        exact.append(all(correct))
    assert scores["loss"] == pytest.approx(sum(losses) / len(losses), rel=1e-6)
    assert scores["token_accuracy"] == sum(hits) / len(hits)
    assert scores["sequence_accuracy"] == sum(exact) / len(exact)


def test_run_saves_its_weights_and_rescores_them(tmp_path: Path) -> None:
    """A saved run is scored again at new lengths, never retrained."""
    config = RunConfig(**TINY)
    first = run(config, tmp_path, "cpu", log=False)
    directory = tmp_path / config.name
    assert set(first["test"]) == {"12", "16", "24"}
    assert [row["step"] for row in first["history"]] == [2, 4]
    assert first["train_flops"] == config.model(12).estimate_flops() * 4 * 8 * 12

    rescored = run(
        RunConfig(**{**TINY, "ood": (20,), "test": 4}), tmp_path, "cpu", log=False
    )
    assert set(rescored["test"]) == {"12", "20"}
    assert rescored["history"] == first["history"]
    assert rescored["train_seconds"] == first["train_seconds"]
    assert set(rescored["fingerprints"]) == {"train", "test", "test/length-20"}

    state = torch.load(directory / "model.pt", weights_only=True)
    model = config.model(12)
    model.load_state_dict(state)
    model.eval()
    examples = build_split(config.task, 12, "test", config.data_seed, 4)
    tokens, targets = (torch.tensor(column) for column in zip(*examples, strict=True))
    assert evaluate(model, tokens, targets) == rescored["test"]["12"]

    with pytest.raises(ValueError, match="trained with other settings"):
        run(RunConfig(**{**TINY, "evaluate_every": 1}), tmp_path, "cpu", log=False)


def test_a_run_saved_before_new_settings_is_rescored(tmp_path: Path) -> None:
    """Settings added after a run was trained take their defaults when it is read."""
    config = RunConfig(**TINY)
    first = run(config, tmp_path, "cpu", log=False)
    result_path = tmp_path / config.name / "result.json"
    saved = json.loads(result_path.read_text())
    for key in ("rope", "rope_base"):
        del saved["training"][key]
    result_path.write_text(json.dumps(saved))
    assert run(config, tmp_path, "cpu", log=False)["test"] == first["test"]
