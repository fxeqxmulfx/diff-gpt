"""Describe embedding/Peri interactions on the complete fixed screen, without tuning."""

from __future__ import annotations

import itertools
import json
import math
import statistics
from pathlib import Path

from research.early99.protocol import SEEDS, selected
from research.early99_core_ablations.factories import RATE_MULTIPLIER
from research.early99_core_ablations_reporting.run_screens import (
    RESULTS,
    ROOT,
    RUNS,
    STUDY,
    write_json,
)
from research.early99_dimension_gain.readiness import sha256
from research.early99_optim_reporting.metrics import paired_cell
from research.early99_reporting.round2_audit import read, require, verify_hash


def interaction_bounds(costs: dict) -> tuple[float | None, float | None]:
    """Bound T(both)*T(base)/(T(embedding)*T(peri)); retain any unresolved cell."""
    require(set(costs) == {"base", "embedding", "peri", "both"}, "Incomplete factorial")
    if any(c["status"] != "passed" for c in costs.values()):
        return None, None
    if any(c["lower"] is None or c["lower"] <= 0 for c in costs.values()):
        return None, None
    for cost in costs.values():
        require(
            cost["upper"] is None or cost["upper"] >= cost["lower"], "Invalid interval"
        )
    numerator = costs["both"], costs["base"]
    denominator = costs["embedding"], costs["peri"]
    low = (
        numerator[0]["lower"]
        * numerator[1]["lower"]
        / (denominator[0]["upper"] * denominator[1]["upper"])
        if all(c["upper"] is not None for c in denominator)
        else 0.0
    )
    high = (
        numerator[0]["upper"]
        * numerator[1]["upper"]
        / (denominator[0]["lower"] * denominator[1]["lower"])
        if all(c["upper"] is not None for c in numerator)
        else None
    )
    return low, high


def cost(row: dict, side: str) -> dict:
    """Retain crossing and censor bounds even when an interaction is undefined."""
    return {
        "status": row[f"{side}_status"],
        "lower": row[f"{side}_first99_epochs_lower"],
        "upper": row[f"{side}_first99_epochs_upper"],
        "censor_lower": row[f"{side}_censor_epochs_lower"],
        "censor_upper": row[f"{side}_censor_epochs_upper"],
    }


def main() -> None:
    """Use the existing combined control only after every declared screen is audited."""
    plan = read(STUDY / "plan.json")
    integrity = read(RESULTS / "screen-integrity.json")
    require(
        integrity["all_three_declared_screens_complete"], "Finish all screens first"
    )
    for path, digest in integrity["evidence_sha256"].items():
        verify_hash(ROOT / path, digest)
    roots = {
        "base": ROOT / plan["screen_control"],
        "embedding": RUNS / "core-embedding-rms-lr2",
        "peri": RUNS / "core-scaled-peri-rms-lr2",
        "both": RUNS / "peri-rms-scaled-lr2",
    }
    combined = read(roots["both"] / "plan.json")["configuration"]
    require(
        combined["model_variant"] == "peri-rms-embedding-scaled"
        and combined["learning_rate_multiplier"] == RATE_MULTIPLIER
        and combined["schedule"] == "constant",
        "Existing combined control is not the declared two-factor composition",
    )
    lookup = {r["name"]: r for r in selected("short")}
    rows, groups = [], []
    for task in plan["screen_tasks"]:
        group = []
        for seed in SEEDS:
            pairs = {
                name: paired_cell(lookup[task], seed, roots["base"], root)
                for name, root in roots.items()
                if name != "base"
            }
            bases = [cost(pair, "baseline") for pair in pairs.values()]
            require(all(a == b for a, b in itertools.pairwise(bases)), "Base differs")
            costs = {
                "base": bases[0],
                **{name: cost(pair, "candidate") for name, pair in pairs.items()},
            }
            low, high = interaction_bounds(costs)
            row = {
                "case": task,
                "seed": seed,
                "costs": costs,
                "interaction_ratio_lower": low,
                "interaction_ratio_upper": high,
                "interaction_ratio": low if low == high else None,
            }
            rows.append(row)
            group.append(row)
        low = (
            statistics.median(r["interaction_ratio_lower"] for r in group)
            if all(r["interaction_ratio_lower"] is not None for r in group)
            else None
        )
        high = (
            statistics.median(
                r["interaction_ratio_upper"]
                if r["interaction_ratio_upper"] is not None
                else math.inf
                for r in group
            )
            if low is not None
            else None
        )
        if high is not None and not math.isfinite(high):
            high = None
        groups.append(
            {
                "case": task,
                "all_three_seeds_solved_by_all_four": all(
                    c["status"] == "passed" for r in group for c in r["costs"].values()
                ),
                "median_interaction_ratio_lower": low,
                "median_interaction_ratio_upper": high,
                "median_interaction_ratio": low if low == high else None,
            }
        )
    result = {
        "kind": (
            "Exploratory descriptive interaction; "
            "no new training or selection criterion"
        ),
        "formula": "T(both)*T(base)/(T(embedding)*T(scaled Peri))",
        "interpretation": (
            "Below1 means the combined addition is faster than multiplying "
            "the two individually measured epoch ratios. This is not a claim "
            "about long-suite causality, unseen tasks, or learning guarantees. "
            "Costs are first-observed crossings, with replay bounds; a failed "
            "or zero-epoch cell has no interaction ratio."
        ),
        "suite_conditioned_on_original_seed1_success": True,
        "selection_uses_this_statistic": False,
        "geometric_mean": None,
        "runs": rows,
        "tasks": groups,
        "source_sha256": sha256(Path(__file__)),
        "screen_integrity_sha256": sha256(RESULTS / "screen-integrity.json"),
        "plans_sha256": {
            str((root / "plan.json").relative_to(ROOT)): sha256(root / "plan.json")
            for root in roots.values()
        },
    }
    write_json(RESULTS / "embedding-peri-interaction.json", result)
    print(json.dumps(groups, indent=2))


if __name__ == "__main__":
    main()
