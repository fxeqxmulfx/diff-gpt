"""Audited paired epoch ratios with complete seed coverage and visible censoring."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import statistics
from pathlib import Path

from research.current_grokking.run import write
from research.early99.protocol import SEEDS, selected
from research.early99.run import DEFAULT_OUTPUT, report, summarize


def read_cell(output: Path, row: dict, seed: int) -> tuple[dict, dict | None]:
    """Recompute each measurement from its raw result and complete history."""
    directory = output / row["name"] / f"seed{seed}"
    path = directory / "result.json"
    if not path.exists():
        error = directory / "error.json"
        return {
            "status": json.loads(error.read_text())["kind"]
            if error.exists()
            else "pending"
        }, None
    result = json.loads(path.read_text())
    history = [
        json.loads(line)
        for line in (directory / "history.jsonl").read_text().splitlines()
    ]
    completed = result["completed"]
    interval = row["case"]["observe"]
    expected = list(range(0, completed + 1, interval))
    if completed % interval:
        expected.append(completed)
    if [r["step"] for r in history] != expected:
        message = f"Observation cadence changed in {directory}"
        raise ValueError(message)
    cell = summarize(row, seed, result, history)
    if cell["status"] == "budget_exhausted" and completed != row["case"]["max_updates"]:
        message = f"Unfinished cell cannot be called budget-censored: {directory}"
        raise ValueError(message)
    if not result["optimizer_steps_verified"] or not result["parameters_finite"]:
        message = f"Unverified optimizer or parameters: {directory}"
        raise ValueError(message)
    return cell, result["metadata"]


def paired_cell(row: dict, seed: int, baseline: Path, candidate: Path) -> dict:
    """Audit the fixed comparison contract before computing one paired ratio."""
    base, base_metadata = read_cell(baseline, row, seed)
    new, new_metadata = read_cell(candidate, row, seed)
    if base_metadata is not None and new_metadata is not None:
        for key in (
            "case",
            "seed",
            "batch_seed",
            "device",
            "torch",
            "compiled_loss",
            "data_fingerprints",
            "data_metadata",
            "observe_updates",
            "model_source_reference",
            "sources",
        ):
            if base_metadata[key] != new_metadata[key]:
                message = f"Unpaired {key}: {row['name']} seed {seed}"
                raise ValueError(message)
        # These experiments change only the optimizer's rate, so both model
        # initialization and capacity must also match, not merely be reported.
        for key in ("initial_parameters_sha256", "parameters", "model_class"):
            if base_metadata[key] != new_metadata[key]:
                message = f"Optimizer-only experiment changed {key}"
                raise ValueError(message)
    result = {"case": row["name"], "tier": row["tier"], "seed": seed}
    for prefix, cell, metadata in (
        ("baseline", base, base_metadata),
        ("candidate", new, new_metadata),
    ):
        result[prefix + "_status"] = cell["status"]
        for field in (
            "first99_epochs",
            "first99_updates",
            "seconds_last_run_segment",
            "resumed",
            "final_heldout_accuracy",
        ):
            result[prefix + "_" + field] = cell.get(field)
        crossing = cell.get("first99") or {}
        result[prefix + "_previous_observed_epochs"] = crossing.get(
            "previous_observed_epochs"
        )
        result[prefix + "_first99_seconds_including_setup"] = crossing.get(
            "recorded_seconds"
        )
        result[prefix + "_censor_epochs"] = (
            cell.get("completed", {}).get("epochs")
            if cell["status"] == "budget_exhausted"
            else None
        )
        result[prefix + "_parameters"] = metadata["parameters"] if metadata else None
    paired = base["status"] == new["status"] == "passed"
    result["epoch_ratio"] = (
        new["first99_epochs"] / base["first99_epochs"]
        if paired and base["first99_epochs"] > 0
        else None
    )
    result["seconds_ratio"] = (
        new["seconds_last_run_segment"] / base["seconds_last_run_segment"]
        if paired
        and not base["resumed"]
        and not new["resumed"]
        and base["seconds_last_run_segment"] > 0
        else None
    )
    return result


def comparison(
    baseline: Path, candidate: Path, tier: str, *, output: Path | None = None
) -> dict:
    """Require all expected paired cells for the aggregate geometric mean."""
    rows, tasks = [], []
    for task in selected(tier):
        cells = [paired_cell(task, seed, baseline, candidate) for seed in SEEDS]
        rows.extend(cells)
        summary = {"case": task["name"], "tier": tier}
        for prefix in ("baseline", "candidate"):
            values = [
                c[prefix + "_first99_epochs"]
                for c in cells
                if c[prefix + "_status"] == "passed"
            ]
            summary[prefix + "_passed_seeds"] = len(values)
            summary[prefix + "_median_first99_epochs"] = (
                statistics.median(values) if len(values) == len(SEEDS) else None
            )
            summary[prefix + "_worst_first99_epochs"] = (
                max(values) if len(values) == len(SEEDS) else None
            )
        ratios = [c["epoch_ratio"] for c in cells]
        complete = all(r is not None for r in ratios)
        summary["median_paired_epoch_ratio"] = (
            statistics.median(ratios) if complete else None
        )
        summary["worst_paired_epoch_ratio"] = max(ratios) if complete else None
        tasks.append(summary)
    ratios = [r["epoch_ratio"] for r in rows]
    complete = all(r is not None and r > 0 for r in ratios)
    result = {
        "tier": tier,
        "baseline_directory": str(baseline),
        "candidate_directory": str(candidate),
        "expected_cells": len(rows),
        "baseline_passed": sum(r["baseline_status"] == "passed" for r in rows),
        "candidate_passed": sum(r["candidate_status"] == "passed" for r in rows),
        "all_paired_passed": complete,
        "geometric_mean_epoch_ratio": math.exp(
            statistics.mean(math.log(r) for r in ratios)
        )
        if complete
        else None,
        "paired_earlier": sum(r is not None and r < 1 for r in ratios),
        "paired_tied": sum(r == 1 for r in ratios),
        "paired_later": sum(r is not None and r > 1 for r in ratios),
        "failures_and_pending": [
            r
            for r in rows
            if r["baseline_status"] != "passed" or r["candidate_status"] != "passed"
        ],
        "interpretation": (
            "Ratios are candidate/baseline; below one is earlier. All expected seeds "
            "are retained. Aggregate geometric mean requires every cell to pass in "
            "both runs. Budget exhaustion is right-censored; pending/errors are not "
            "budget misses. Per-task statistics require all three seeds. Resumed "
            "clocks are separate segments; their runtime ratios are omitted."
        ),
        "tasks": tasks,
        "runs": rows,
    }
    output = candidate if output is None else output
    output.mkdir(parents=True, exist_ok=True)
    write(output / f"comparison-{tier}.json", result)
    for suffix, records in (("", rows), ("-tasks", tasks)):
        buffer = io.StringIO()
        writer = csv.DictWriter(
            buffer, fieldnames=list(records[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(records)
        (output / f"comparison-{tier}{suffix}.csv").write_text(buffer.getvalue())
    return result


def main() -> None:
    """Write full paired JSON, cell CSV and three-seed task CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, help="Separate report destination for multiple controls"
    )
    parser.add_argument("--tier", choices=("short", "long"), default="short")
    args = parser.parse_args()
    report(args.baseline, args.tier)
    report(args.candidate, args.tier)
    result = comparison(args.baseline, args.candidate, args.tier, output=args.output)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "expected_cells",
                    "baseline_passed",
                    "candidate_passed",
                    "geometric_mean_epoch_ratio",
                    "paired_earlier",
                    "paired_tied",
                    "paired_later",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
