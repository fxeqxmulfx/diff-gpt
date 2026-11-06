"""Compare coverage before speed, retaining all capacity interventions and censors."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

from research.cubic_capacity.factories import ROOT, STUDY, plan
from research.cubic_capacity.readiness import sha256
from research.early99_optim_reporting.metrics import finite, geometric_mean_bounds
from research.early99_reporting.round2_audit import (
    read,
    require,
    verify_csv,
    verify_hash,
)

RESULTS = STUDY / "results"
SEEDS = {0, 1, 2}


def complete_seeds(rows: list[dict]) -> None:
    """Do not silently drop, repeat or substitute any declared seed."""
    require(
        len(rows) == len(SEEDS) and {r["seed"] for r in rows} == SEEDS,
        "Incomplete or duplicated seed group",
    )


def group_metrics(rows: list[dict]) -> dict:
    """Require all three seeds to pass before computing first-crossing statistics."""
    complete_seeds(rows)
    require(len({(r["size"], r["arm"]) for r in rows}) == 1, "Mixed size/arm group")
    passed = sum(r["status"] == "passed" for r in rows)
    require(
        all(r["status"] in {"passed", "budget_exhausted"} for r in rows),
        "Unfinished seed",
    )
    result = {
        key: rows[0][key]
        for key in [
            "size",
            "arm",
            "layers",
            "width",
            "heads",
            "head_width",
            "parameters",
            "peak_learning_rate",
        ]
    }
    result.update(
        passed=passed, expected=len(SEEDS), budget_exhausted=len(rows) - passed
    )
    for label, aggregate in [("median", statistics.median), ("worst", max)]:
        low = (
            aggregate(r["first99_epochs_lower"] for r in rows)
            if passed == len(SEEDS)
            else None
        )
        high = (
            finite(
                aggregate(
                    r["first99_epochs_upper"]
                    if r["first99_epochs_upper"] is not None
                    else math.inf
                    for r in rows
                )
            )
            if passed == len(SEEDS)
            else None
        )
        result[f"{label}_first99_epochs_lower"] = low
        result[f"{label}_first99_epochs_upper"] = high
        result[f"{label}_first99_epochs"] = low if low == high else None
    low, high = geometric_mean_bounds(
        rows, "first99_epochs_lower", "first99_epochs_upper"
    )
    result.update(
        geometric_mean_first99_epochs_lower=low,
        geometric_mean_first99_epochs_upper=high,
        geometric_mean_first99_epochs=low if low == high else None,
    )
    result.update(
        total_updates=sum(r["completed_updates"] for r in rows),
        total_actual_epochs_lower=sum(r["actual_epochs_lower"] for r in rows),
        total_actual_epochs_upper=sum(r["actual_epochs_upper"] for r in rows)
        if all(r["actual_epochs_upper"] is not None for r in rows)
        else None,
        seconds_sum_last_run_segments=sum(r["seconds_last_run_segment"] for r in rows),
        resumptions=sum(r["resumptions"] for r in rows),
    )
    return result


def paired_ratio(base: dict, new: dict, kind: str) -> dict:
    """Compare paired seeds while explicitly retaining the capacity intervention."""
    require(base["seed"] == new["seed"], "Unpaired seed")
    same_size = base["size"] == new["size"]
    if same_size:
        require(
            base["parameters"] == new["parameters"]
            and base["initial_parameters_sha256"] == new["initial_parameters_sha256"],
            "Same-size initialization changed",
        )
    result = {
        "kind": kind,
        "seed": base["seed"],
        "baseline_size": base["size"],
        "baseline_arm": base["arm"],
        "candidate_size": new["size"],
        "candidate_arm": new["arm"],
        "same_size_initialization": same_size,
    }
    for prefix, cell in [("baseline", base), ("candidate", new)]:
        for key in [
            "status",
            "parameters",
            "first99_epochs_lower",
            "first99_epochs_upper",
            "canonical_epochs",
            "actual_epochs_lower",
            "actual_epochs_upper",
            "seconds_last_run_segment",
        ]:
            result[f"{prefix}_{key}"] = cell[key]
    low = high = None
    if base["status"] == new["status"] == "passed":
        denominator_low, denominator_high = (
            base["first99_epochs_lower"],
            base["first99_epochs_upper"],
        )
        if denominator_low > 0:
            low = (
                new["first99_epochs_lower"] / denominator_high
                if denominator_high
                else 0.0
            )
            high = (
                new["first99_epochs_upper"] / denominator_low
                if new["first99_epochs_upper"] is not None
                else None
            )
    result.update(
        epoch_ratio_lower=low,
        epoch_ratio_upper=high,
        epoch_ratio=low if low == high else None,
    )
    result["parameter_ratio"] = new["parameters"] / base["parameters"]
    # Historical controls predate the per-update journal. Their timings remain
    # visible, but their overhead differs from this newly journaled experiment.
    result["seconds_ratio"] = (
        new["seconds_last_run_segment"] / base["seconds_last_run_segment"]
        if not base["reused"]
        and not new["reused"]
        and not base["resumptions"]
        and not new["resumptions"]
        and base["status"] == new["status"] == "passed"
        else None
    )
    return result


def contrast_metrics(rows: list[dict]) -> dict:
    """Retain recovered/newly censored seeds, and omit incomplete ratio means."""
    complete_seeds(rows)
    result = {
        key: rows[0][key]
        for key in [
            "kind",
            "baseline_size",
            "baseline_arm",
            "candidate_size",
            "candidate_arm",
            "parameter_ratio",
        ]
    }
    require(
        all(all(row[key] == value for key, value in result.items()) for row in rows),
        "Mixed contrast group",
    )
    low, high = geometric_mean_bounds(rows, "epoch_ratio_lower", "epoch_ratio_upper")
    result.update(
        baseline_passed=sum(r["baseline_status"] == "passed" for r in rows),
        candidate_passed=sum(r["candidate_status"] == "passed" for r in rows),
        recovered=sum(
            r["baseline_status"] == "budget_exhausted"
            and r["candidate_status"] == "passed"
            for r in rows
        ),
        newly_censored=sum(
            r["baseline_status"] == "passed"
            and r["candidate_status"] == "budget_exhausted"
            for r in rows
        ),
        both_censored=sum(
            r["baseline_status"] == r["candidate_status"] == "budget_exhausted"
            for r in rows
        ),
        geometric_mean_epoch_ratio_lower=low,
        geometric_mean_epoch_ratio_upper=high,
        geometric_mean_epoch_ratio=low if low == high else None,
        paired_earlier=sum(
            r["epoch_ratio_upper"] is not None and r["epoch_ratio_upper"] < 1
            for r in rows
        ),
        paired_tied=sum(r["epoch_ratio"] == 1 for r in rows),
        paired_later=sum(
            r["epoch_ratio_lower"] is not None and r["epoch_ratio_lower"] > 1
            for r in rows
        ),
    )
    return result


def report(audited: dict, search: dict) -> dict:
    """Reconstruct all 36 cells and all 51 declared comparison rows."""
    rows = audited["rows"]
    expected = {
        (shape["name"], arm["name"], seed)
        for shape in search["shapes"]
        for arm in search["arms"]
        for seed in search["seeds"]
    }
    lookup = {(row["size"], row["arm"], row["seed"]): row for row in rows}
    require(
        audited["complete"] and len(lookup) == len(rows) and set(lookup) == expected,
        "Missing or duplicate study cells",
    )
    groups = [
        group_metrics(
            [lookup[shape["name"], arm["name"], seed] for seed in search["seeds"]]
        )
        for shape in search["shapes"]
        for arm in search["arms"]
    ]
    pairs, contrasts = [], []
    definitions = [
        ("architecture", (shape["name"], control), (shape["name"], "normalized_lr2"))
        for shape in search["shapes"]
        for control in ["original_lr1", "original_lr2"]
    ]
    definitions.extend(
        ("capacity", ("current", arm["name"]), (shape["name"], arm["name"]))
        for shape in search["shapes"]
        if not shape["reuse_existing"]
        for arm in search["arms"]
    )
    for kind, baseline, candidate in definitions:
        group = [
            paired_ratio(lookup[*baseline, seed], lookup[*candidate, seed], kind)
            for seed in search["seeds"]
        ]
        pairs.extend(group)
        contrasts.append(contrast_metrics(group))
    return {
        "complete": True,
        "new_training_cells": audited["new_training_cells"],
        "reused_controls": audited["reused_controls"],
        "scope": search["scope"],
        "fixed75_selection_changed": False,
        "new_long_suite_authorized_by_this_study": False,
        "timing_limits": (
            "Recorded trainer times include setup and scoring. Historical "
            "current-size controls predate the new durable per-update journal; "
            "runtime ratios against them are omitted. Even within the new runs, "
            "cache and setup costs make time descriptive rather than a controlled "
            "throughput benchmark."
        ),
        "comparison_rules": (
            "All three seeds remain visible. Coverage is assessed first; a size/arm "
            "median or geometric mean requires 3/3 passes. Ratios below one mean "
            "earlier crossing. Actual exposure includes replay with explicit "
            "lower/upper bounds. Size changes preserve data and sample order, but "
            "necessarily change parameter counts and initialized tensors; widening "
            "adds heads at fixed head width 32. The normalized residual coefficient "
            "remains 1/sqrt(2L)."
        ),
        "groups": groups,
        "comparisons": contrasts,
        "paired_cells": pairs,
        "cells": rows,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    """Use portable line endings and verify every saved value."""
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    verify_csv(path, rows)


def main() -> None:
    """Require the completed audit and write a reproducible descriptive comparison."""
    audited = read(RESULTS / "complete-audit.json")
    verify_hash(Path(__file__).with_name("audit.py"), audited["auditor_sha256"])
    verify_hash(STUDY / "plan.json", audited["plan_sha256"])
    verify_hash(RESULTS / "validation.json", audited["validation_sha256"])
    for row in audited["rows"]:
        for path, digest in row["artifacts_sha256"].items():
            verify_hash(ROOT / path, digest)
    result = report(audited, plan())
    result["complete_audit_sha256"] = sha256(RESULTS / "complete-audit.json")
    result["reporter_sha256"] = sha256(Path(__file__))
    (RESULTS / "comparison.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    for name in ["groups", "comparisons", "paired_cells", "cells"]:
        rows = [
            {k: v for k, v in row.items() if k != "artifacts_sha256"}
            for row in result[name]
        ]
        write_csv(RESULTS / f"{name}.csv", rows)
    print(json.dumps(result["groups"], indent=2))


if __name__ == "__main__":
    main()
