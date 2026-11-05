"""Three-seed coverage, real epoch tails, and first-crossing benchmark integrity."""

import json
from dataclasses import asdict
from typing import TYPE_CHECKING

import pytest

from research.current_grokking import data
from research.current_grokking.protocol import Case
from research.current_grokking.run import EpochSampler
from research.early99 import run
from research.early99.protocol import SEEDS, exposure, first99, selected, specification

if TYPE_CHECKING:
    from pathlib import Path

SHORT_TASKS = 25
LONG_TASKS = 10
TOTAL_TASKS = 35
THREE_SEEDS = 3
CROSSING_UPDATES = 5
MEDIAN_EPOCHS = 2


def test_exact_requested_task_set_tiers_and_three_seeds() -> None:
    """All named nonconstant cases are kept; the three trivial controls stay out."""
    arithmetic = {
        "arithmetic-" + name
        for name in (
            "add",
            "sub",
            "mul",
            "div",
            "abs-diff",
            "abs-diff-blind",
            "perm-add",
            "half-sum",
            "sum-square",
            "sum-cube",
            "sum-pow4",
            "square-times",
            "square-sum",
            "cube-plus",
            "affine-product",
        )
    }
    arithmetic.add("division-p193")
    arithmetic |= {
        f"power-division-p{prime}-k{power}"
        for prime in (97, 193)
        for power in (2, 3, 5)
    }
    others = {
        "suite-boolean-and-shift",
        "suite-boolean-and-random",
        "gpt-probe-key-value-binding",
        "basis-depth-easy",
        "basis-depth-hard",
        "gpt-probe-branch-amplitudes",
        "gpt-probe-linear-transport",
        "basis-parity16",
        "basis-parity20",
        "basis-parity24",
        "basis-recall-easy",
        "recurrence-fibonacci-k1-c2-lgt",
        "associative-divisor-n2-l1-w0",
    }
    short, long = selected("short"), selected("long")
    assert len(short) == SHORT_TASKS
    assert len(long) == LONG_TASKS
    assert {r["name"] for r in selected()} == arithmetic | others
    assert not {r["name"] for r in short} & {r["name"] for r in long}
    assert SEEDS == (0, 1, 2)
    assert (
        len([(row["name"], seed) for row in selected() for seed in SEEDS])
        == TOTAL_TASKS * THREE_SEEDS
    )
    for row in selected():
        case = specification(row)
        counts = exposure(0, row["train_examples"], case.batch)
        assert case.metric == "sequence_accuracy"
        assert case.target == pytest.approx(0.99)
        assert case.after_target == 0
        assert case.observe <= counts["updates_per_epoch"]


@pytest.mark.parametrize(("size", "batch"), [(10, 4), (2, 8), (16, 4)])
def test_epoch_count_matches_actual_sampler_short_tails(size: int, batch: int) -> None:
    """Exposure agrees with the real sampler, including size < batch and tails."""
    sampler = EpochSampler(size, batch, wrap=False, seed=1)
    seen = 0
    for updates in range(18):
        counts = exposure(updates, size, batch)
        assert counts["examples_seen"] == seen
        assert counts["epochs"] == pytest.approx(seen / size)
        parts, _ = sampler.next()
        seen += sum(count for _, _, count in parts)
    assert exposure(5, 10, 4)["epochs"] == pytest.approx(1.8)


def test_real_arithmetic_table_epoch_includes_short_tail() -> None:
    """One real division-table pass exposes each training row exactly once."""
    case = Case("epoch-oracle", "arithmetic", {"operation": "div", "prime": 7}, batch=8)
    task = data.arithmetic(case, "cpu")
    size = len(task.splits["train"])
    sampler = EpochSampler(size, case.batch, wrap=False, seed=1)
    steps = exposure(0, size, case.batch)["updates_per_epoch"]
    actual = sum(sum(count for _, _, count in sampler.next()[0]) for _ in range(steps))
    assert actual == exposure(steps, size, case.batch)["examples_seen"] == size


def test_first99_accepts_one_crossing_and_rejects_token_only_accuracy() -> None:
    """A later decline does not erase success; 100% tokens cannot replace exactness."""
    history = [
        {
            "step": 0,
            "seconds": 1,
            "validation": {"sequence_accuracy": 0.98, "token_accuracy": 1},
        },
        {"step": 5, "seconds": 2, "validation": {"sequence_accuracy": 0.99}},
        {"step": 10, "seconds": 3, "validation": {"sequence_accuracy": 0.2}},
    ]
    measured = first99(history, 10, 4)
    assert measured is not None
    assert measured["updates"] == CROSSING_UPDATES
    assert measured["epochs"] == pytest.approx(1.8)
    assert measured["previous_observed_updates"] == 0
    assert first99(history[:1], 10, 4) is None


def test_three_seed_report_keeps_misses_and_requires_full_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed seed must not disappear from task medians or aggregate scores."""
    monkeypatch.setattr(
        run, "selected", lambda _: [{"name": "fixture", "tier": "short"}]
    )
    for seed in SEEDS:
        directory = tmp_path / "fixture" / f"seed{seed}"
        directory.mkdir(parents=True)
        (directory / "early99_result.json").write_text(
            json.dumps(
                {
                    "case": "fixture",
                    "tier": "short",
                    "seed": seed,
                    "status": "passed",
                    "first99_epochs": seed + 1,
                }
            )
        )
    complete = run.report(tmp_path, "short")
    assert complete["all_passed"]
    assert complete["tasks"][0]["median_first99_epochs"] == MEDIAN_EPOCHS
    assert complete["tasks"][0]["worst_first99_epochs"] == THREE_SEEDS
    path = tmp_path / "fixture/seed2/early99_result.json"
    missed = json.loads(path.read_text())
    missed.update(status="budget_exhausted", first99_epochs=None)
    path.write_text(json.dumps(missed))
    incomplete = run.report(tmp_path, "short")
    assert incomplete["complete"]
    assert not incomplete["all_passed"]
    assert incomplete["planned_runs"] == THREE_SEEDS
    assert incomplete["passed_runs"] == THREE_SEEDS - 1
    assert incomplete["tasks"][0]["median_first99_epochs"] is None
    assert incomplete["geometric_mean_first99_epochs"] is None


def test_plan_preserves_sources_and_refuses_tampered_archive(tmp_path: Path) -> None:
    """A comparison cannot silently change its frozen task definitions."""
    run.prepare_output(tmp_path)
    run.prepare_output(tmp_path)
    path = tmp_path / "suite_source/suite.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="Archived benchmark source changed"):
        run.prepare_output(tmp_path)


def test_summarize_checks_actual_data_and_keeps_first_crossing() -> None:
    """Executed data must match the manifest before epoch scores are accepted."""
    case = Case("fixture", "arithmetic", batch=4)
    row = {
        "case": asdict(case),
        "tier": "short",
        "train_examples": 10,
        "data_fingerprints": {"train": "fixed-train", "validation": "fixed-heldout"},
    }
    history = [
        {"step": 0, "seconds": 0, "validation": {"sequence_accuracy": 0.1}},
        {"step": 5, "seconds": 1, "validation": {"sequence_accuracy": 0.99}},
        {"step": 10, "seconds": 2, "validation": {"sequence_accuracy": 0.5}},
    ]
    result = {
        "metadata": {
            "case": asdict(case),
            "seed": 1,
            "data_fingerprints": dict(row["data_fingerprints"]),
            "initial_parameters_sha256": "fixture",
        },
        "endpoint": {
            "train": {"examples": 10},
            "validation": history[-1]["validation"],
        },
        "completed": 10,
        "first_target": 5,
        "seconds": 2,
    }
    measured = run.summarize(row, 1, result, history)
    assert measured["status"] == "passed"
    assert measured["first99_epochs"] == pytest.approx(1.8)
    result["metadata"]["data_fingerprints"]["train"] = "changed"
    with pytest.raises(ValueError, match="split fingerprints"):
        run.summarize(row, 1, result, history)
