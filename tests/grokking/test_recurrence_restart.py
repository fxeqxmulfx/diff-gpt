"""Check repeat warmup without resetting trained weights or optimizer history."""

import csv
import hashlib
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest
import torch

from benchmarks.grokking.recurrence import Config, run, update_learning_rate
from benchmarks.grokking.recurrence_restart import prepare_fork
from diff_gpt.optimizer.schedule_free import ScheduleFreeAdamW

if TYPE_CHECKING:
    from pathlib import Path


def assert_same(left: object, right: object) -> None:
    """Compare complete nested checkpoint state, including every tensor bit."""
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key, value in left.items():
            assert_same(value, right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_same(a, b)
    else:
        assert left == right


def test_repeat_warmup_changes_only_rate_before_each_optimizer_update() -> None:
    """The ramp uses local time while moments, global time and averaging survive."""
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    optimizer = ScheduleFreeAdamW([parameter], lr=0.01, betas=(0.9, 0.98))
    for _ in range(2):
        parameter.grad = torch.tensor([0.2, -0.1])
        optimizer.step()
    config = Config(learning_rate=0.03, restart_step=2, restart_warmup=4)
    before = {key: value.clone() for key, value in optimizer.state[parameter].items()}
    previous_total = optimizer.param_groups[0]["total"]
    update_learning_rate(config, optimizer)
    assert_same(before, optimizer.state[parameter])
    assert optimizer.param_groups[0]["total"] == previous_total
    assert optimizer.param_groups[0]["step"] == config.restart_step
    expected_rates = [0.0075, 0.015, 0.0225, 0.03, 0.03]
    for number, expected in enumerate(expected_rates, start=1):
        update_learning_rate(config, optimizer)
        group = optimizer.param_groups[0]
        assert group["lr"] == pytest.approx(expected)
        parameter.grad = torch.tensor([0.2, -0.1])
        optimizer.step()
        assert group["step"] == config.restart_step + number
        previous_total += expected**2 * (1 - group["betas"][1] ** group["step"])
        assert group["total"] == pytest.approx(previous_total)


def test_fork_and_mid_warmup_resume_preserve_every_training_state(
    tmp_path: Path,
) -> None:
    """A real MASH fork and a mid-ramp resume match an uninterrupted branch."""
    original = Config(
        width=8,
        heads=1,
        layers=1,
        epochs=1,
        tokenizer="mash",
        vocab_size=512,
        grid_domain=96,
    )
    parent = tmp_path / "parent"
    run(original, parent, "cpu")
    source = parent / "checkpoint.pt"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    initial = torch.load(source, weights_only=True)
    config = replace(
        original,
        learning_rate=0.03,
        epochs=2,
        restart_step=initial["optimizer"]["param_groups"][0]["step"],
        restart_warmup=15,
    )
    split, whole = tmp_path / "split", tmp_path / "whole"
    provenance = prepare_fork(source, split, config)
    prepare_fork(source, whole, config)
    assert provenance["parent_sha256"] == digest
    assert not provenance["optimizer_and_sampler_reset"]
    forked = torch.load(split / "checkpoint.pt", weights_only=True)
    for key in ("model", "optimizer", "raw", "sampler_state", "sampler_order"):
        assert_same(forked[key], initial[key])
    run(config, split, "cpu", resume=True)
    middle = torch.load(split / "checkpoint.pt", weights_only=True)
    group = middle["optimizer"]["param_groups"][0]
    elapsed_steps = group["step"] - config.restart_step
    assert 0 < elapsed_steps < config.restart_warmup
    assert group["lr"] == pytest.approx(
        config.learning_rate * elapsed_steps / config.restart_warmup
    )
    longer = replace(config, epochs=3)
    run(longer, split, "cpu", resume=True)
    run(longer, whole, "cpu", resume=True)
    first = torch.load(split / "checkpoint.pt", weights_only=True)
    second = torch.load(whole / "checkpoint.pt", weights_only=True)
    for key in (
        "model",
        "optimizer",
        "sampler_state",
        "sampler_order",
        "metrics",
        "query_metrics",
    ):
        assert_same(first[key], second[key])
    group = first["optimizer"]["param_groups"][0]
    assert group["lr"] == config.learning_rate
    assert group["total"] > initial["optimizer"]["param_groups"][0]["total"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    with (split / "history.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["epoch"]) for row in rows] == list(range(longer.epochs + 1))
    assert float(rows[-1]["learning_rate"]) == config.learning_rate
    assert float(rows[-1]["averaging_coefficient"]) > 0
    with pytest.raises(ValueError, match="only"):
        prepare_fork(source, tmp_path / "invalid", replace(config, context=3))
    assert not (tmp_path / "invalid").exists()
