"""Compare actual example exposure, including bounded replayed work from journals."""

from __future__ import annotations

import csv
import json
import math
import statistics
from typing import TYPE_CHECKING

from research.early99.experiments.compare import paired_cell as canonical_pair
from research.early99.protocol import SEEDS, selected
from research.early99_execution.journal import verify_saved_journal
from research.early99_reporting.round2_audit import MIN_IMPROVED_SCREEN_TASKS, require

if TYPE_CHECKING:
    from pathlib import Path


def finite(value: float) -> float | None:
    """Encode an unbounded endpoint explicitly rather than emitting invalid JSON."""
    return value if math.isfinite(value) else None


def paired_cell(row: dict, seed: int, baseline: Path, candidate: Path) -> dict:
    """Retain original pairing checks and add every replayed training example."""
    result = canonical_pair(row, seed, baseline, candidate)
    for prefix, root in [("baseline", baseline), ("candidate", candidate)]:
        cell = root / row["name"] / f"seed{seed}"
        first = result[prefix + "_first99_epochs"]
        censor = result[prefix + "_censor_epochs"]
        lower = upper = 0.0
        if (cell / "result.json").exists():
            saved = json.loads((cell / "result.json").read_text())
            metadata = saved["metadata"]
            if "execution_journal" in saved:
                verified = verify_saved_journal(
                    cell,
                    saved["completed"],
                    row["train_examples"],
                    row["case"]["batch"],
                    metadata.get("resumptions", []),
                    saved["execution_journal"],
                )
                lower = verified["extra_examples_lower"] / row["train_examples"]
                upper = verified["extra_examples_upper"] / row["train_examples"]
            elif metadata.get("resumptions"):
                # Preserve old long controls: their durable execution ledger did
                # not yet exist. Their canonical exposure is only a lower bound.
                upper = math.inf
        result[prefix + "_canonical_first99_epochs"] = first
        result[prefix + "_canonical_censor_epochs"] = censor
        result[prefix + "_replay_epochs_lower"] = lower
        result[prefix + "_replay_epochs_upper"] = finite(upper)
        for metric, canonical in [("first99", first), ("censor", censor)]:
            low = canonical + lower if canonical is not None else None
            high = finite(canonical + upper) if canonical is not None else None
            result[f"{prefix}_{metric}_epochs_lower"] = low
            result[f"{prefix}_{metric}_epochs_upper"] = high
            result[f"{prefix}_{metric}_epochs"] = low if low == high else None
    paired = result["baseline_status"] == result["candidate_status"] == "passed"
    lower = upper = None
    if paired:
        base_low = result["baseline_first99_epochs_lower"]
        base_high = result["baseline_first99_epochs_upper"]
        new_low = result["candidate_first99_epochs_lower"]
        new_high = result["candidate_first99_epochs_upper"]
        lower = new_low / base_high if base_high else 0.0
        upper = new_high / base_low if new_high is not None and base_low else None
    result["epoch_ratio_lower"] = lower
    result["epoch_ratio_upper"] = upper
    result["epoch_ratio"] = lower if lower == upper else None
    return result


def task_metrics(name: str, rows: list[dict]) -> dict:
    """Bound all-seed medians and worst ratios; omit no unsuccessful seed."""
    require(len(rows) == len(SEEDS), "Incomplete three-seed group")
    output = {"case": name, "tier": rows[0]["tier"]}
    for prefix in ["baseline", "candidate"]:
        passed = sum(row[prefix + "_status"] == "passed" for row in rows)
        output[prefix + "_passed_seeds"] = passed
        for aggregate, function in [("median", statistics.median), ("worst", max)]:
            low = (
                function(row[prefix + "_first99_epochs_lower"] for row in rows)
                if passed == len(rows)
                else None
            )
            high = (
                function(
                    row[prefix + "_first99_epochs_upper"]
                    if row[prefix + "_first99_epochs_upper"] is not None
                    else math.inf
                    for row in rows
                )
                if passed == len(rows)
                else None
            )
            high = finite(high) if high is not None else None
            key = f"{prefix}_{aggregate}_first99_epochs"
            output[key + "_lower"] = low
            output[key + "_upper"] = high
            output[key] = low if low == high else None
    complete = all(row["epoch_ratio_lower"] is not None for row in rows)
    for aggregate, function in [("median", statistics.median), ("worst", max)]:
        low = function(row["epoch_ratio_lower"] for row in rows) if complete else None
        high = (
            function(
                row["epoch_ratio_upper"]
                if row["epoch_ratio_upper"] is not None
                else math.inf
                for row in rows
            )
            if complete
            else None
        )
        high = finite(high) if high is not None else None
        key = f"{aggregate}_paired_epoch_ratio"
        output[key + "_lower"] = low
        output[key + "_upper"] = high
        output[key] = low if low == high else None
    return output


def screen_decision(rows: list[dict], tasks: list[dict]) -> tuple[int, int, bool]:
    """Require the declared gate to hold or fail for the entire exposure interval."""
    passed = sum(row["candidate_status"] == "passed" for row in rows)
    improved = sum(
        t["median_paired_epoch_ratio_upper"] is not None
        and t["median_paired_epoch_ratio_upper"] < 1
        for t in tasks
    )
    possible = sum(
        t["median_paired_epoch_ratio_lower"] is not None
        and t["median_paired_epoch_ratio_lower"] < 1
        for t in tasks
    )
    lower = passed == len(rows) and improved >= MIN_IMPROVED_SCREEN_TASKS
    upper = passed == len(rows) and possible >= MIN_IMPROVED_SCREEN_TASKS
    require(
        lower == upper,
        "Replay uncertainty changes promotion; independent repetition required",
    )
    return passed, improved, lower


def geometric_mean_bounds(
    rows: list[dict], lower: str, upper: str
) -> tuple[float | None, float | None]:
    """Require all cells and positive values; never compute a successful-only mean."""
    lows, highs = [r[lower] for r in rows], [r[upper] for r in rows]
    if any(value is None or value <= 0 for value in lows):
        return None, None
    low = math.exp(statistics.mean(math.log(value) for value in lows))
    high = (
        math.exp(statistics.mean(math.log(value) for value in highs))
        if all(value is not None and value > 0 for value in highs)
        else None
    )
    return low, high


def comparison(baseline: Path, candidate: Path, tier: str, *, output: Path) -> dict:
    """Write all cells with exact values or explicit replay-exposure bounds."""
    rows, tasks = [], []
    for task in selected(tier):
        group = [paired_cell(task, seed, baseline, candidate) for seed in SEEDS]
        rows.extend(group)
        tasks.append(task_metrics(task["name"], group))
    bounds = geometric_mean_bounds(rows, "epoch_ratio_lower", "epoch_ratio_upper")
    result = {
        "tier": tier,
        "baseline_directory": str(baseline),
        "candidate_directory": str(candidate),
        "expected_cells": len(rows),
        "baseline_passed": sum(r["baseline_status"] == "passed" for r in rows),
        "candidate_passed": sum(r["candidate_status"] == "passed" for r in rows),
        "all_paired_passed": all(
            r["baseline_status"] == r["candidate_status"] == "passed" for r in rows
        ),
        "geometric_mean_epoch_ratio": bounds[0] if bounds[0] == bounds[1] else None,
        "geometric_mean_epoch_ratio_lower": bounds[0],
        "geometric_mean_epoch_ratio_upper": bounds[1],
        "paired_earlier": sum(
            r["epoch_ratio_upper"] is not None and r["epoch_ratio_upper"] < 1
            for r in rows
        ),
        "paired_tied": sum(r["epoch_ratio"] == 1 for r in rows),
        "paired_later": sum(
            r["epoch_ratio_lower"] is not None and r["epoch_ratio_lower"] > 1
            for r in rows
        ),
        "failures_and_pending": [
            r
            for r in rows
            if r["baseline_status"] != "passed" or r["candidate_status"] != "passed"
        ],
        "epoch_metric": (
            "Actual training examples including replay. Canonical trajectory "
            "columns retained explicitly; uncertain values have lower/upper bounds."
        ),
        "runs": rows,
        "tasks": tasks,
    }
    for prefix in ["baseline", "candidate"]:
        low, high = geometric_mean_bounds(
            rows, prefix + "_first99_epochs_lower", prefix + "_first99_epochs_upper"
        )
        result[prefix + "_geometric_mean_epochs_lower"] = low
        result[prefix + "_geometric_mean_epochs_upper"] = high
        result[prefix + "_geometric_mean_epochs"] = low if low == high else None
    output.mkdir(parents=True, exist_ok=True)
    (output / f"comparison-{tier}.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    for suffix, records in [("", rows), ("-tasks", tasks)]:
        with (output / f"comparison-{tier}{suffix}.csv").open(
            "w", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(records[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(records)
    return result
