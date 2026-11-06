"""Export complete recorded key/value curves for a retrospective four-arm view."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from research.early99.protocol import SEEDS, TARGET, exposure, first99, selected
from research.early99_core_ablations_reporting.run_screens import (
    RESULTS,
    ROOT,
    RUNS,
    STUDY,
    write_json,
)
from research.early99_dimension_gain.readiness import sha256
from research.early99_optim_reporting.metrics import paired_cell
from research.early99_reporting.round2_audit import read, require, verify_csv

CASE = "gpt-probe-key-value-binding"
ARMS = {
    "original": "adamw-lr2",
    "embedding": "core-embedding-rms-lr2",
    "peri": "core-scaled-peri-rms-lr2",
    "both": "peri-rms-scaled-lr2",
}


def main() -> None:
    """Keep every observation and censor; do not change the screen or its gate."""
    task = next(row for row in selected("short") if row["name"] == CASE)
    sources = {
        Path(__file__),
        STUDY / "plan.json",
        ROOT / "research/early99/suite.json",
        ROOT / "research/early99/protocol.py",
        ROOT / "research/early99/experiments/compare.py",
        ROOT / "research/early99_optim_reporting/metrics.py",
        ROOT / "research/early99_execution/journal.py",
    }
    rows, runs = [], []
    for arm, candidate in ARMS.items():
        root = RUNS / candidate
        sources.add(root / "plan.json")
        for seed in SEEDS:
            cell = root / CASE / f"seed{seed}"
            pair = paired_cell(task, seed, RUNS / ARMS["original"], root)
            require(
                pair["candidate_replay_epochs_lower"]
                == pair["candidate_replay_epochs_upper"]
                == 0,
                "This trajectory figure requires zero replay exposure",
            )
            result = read(cell / "early99_result.json")
            history = [
                json.loads(line)
                for line in (cell / "history.jsonl").read_text().splitlines()
            ]
            require(
                result["status"] == pair["candidate_status"]
                and result["status"] in {"passed", "budget_exhausted"},
                "Only completed, independently reconstructed cells can be plotted",
            )
            require(
                result["first99"]
                == first99(history, task["train_examples"], task["case"]["batch"]),
                "Stored crossing differs from the complete history",
            )
            require(
                result["completed"]
                == exposure(
                    history[-1]["step"], task["train_examples"], task["case"]["batch"]
                ),
                "Final exposure differs from the complete history",
            )
            for name in ("history.jsonl", "early99_result.json", "result.json"):
                sources.add(cell / name)
            if (cell / "execution-journal.json").exists():
                sources.add(cell / "execution-journal.json")
            observations = []
            for observation in history:
                for split, expected in (
                    ("train", task["train_examples"]),
                    ("validation", task["heldout_examples"]),
                ):
                    score = observation[split]
                    accuracy = score["sequence_accuracy"]
                    require(
                        score["example_count"] == expected
                        and 0 <= score["exact"] <= expected
                        and math.isfinite(accuracy)
                        and accuracy == score["exact"] / expected,
                        "Curve must use exact accuracy on the complete split",
                    )
                point = {
                    "arm": arm,
                    "seed": seed,
                    "updates": observation["step"],
                    "epochs": exposure(
                        observation["step"],
                        task["train_examples"],
                        task["case"]["batch"],
                    )["epochs"],
                    "train_accuracy": observation["train"]["sequence_accuracy"],
                    "heldout_accuracy": observation["validation"]["sequence_accuracy"],
                }
                observations.append(point)
            rows.extend(observations)
            train_crossing = next(
                (p["epochs"] for p in observations if p["train_accuracy"] >= TARGET),
                None,
            )
            runs.append(
                {
                    "arm": arm,
                    "candidate": candidate,
                    "seed": seed,
                    "status": result["status"],
                    "observations": len(observations),
                    "first_train99_epochs": train_crossing,
                    "first_heldout99_epochs": result["first99_epochs"],
                    "final_epochs": observations[-1]["epochs"],
                    "final_train_accuracy": observations[-1]["train_accuracy"],
                    "final_heldout_accuracy": observations[-1]["heldout_accuracy"],
                    "best_heldout_accuracy": max(
                        p["heldout_accuracy"] for p in observations
                    ),
                    "replay_epochs_lower": pair["candidate_replay_epochs_lower"],
                    "replay_epochs_upper": pair["candidate_replay_epochs_upper"],
                }
            )
    csv_path = RESULTS / "key-value-curves.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    verify_csv(csv_path, rows)
    report = {
        "kind": "Retrospective descriptive learning curves; no new training",
        "case": CASE,
        "task_selection": (
            "Chosen after observing all three scaled-Peri key/value misses"
        ),
        "scope": "One task, all three seeds and all four existing factorial arms",
        "selection_uses_this_report": False,
        "causal_explanation_of_long_failures": False,
        "suite_conditioned_on_original_seed1_success": True,
        "all_observations_retained_without_smoothing": True,
        "expected_runs": len(ARMS) * len(SEEDS),
        "observations": len(rows),
        "target": TARGET,
        "epoch_metric": "Actual presented examples / train size; verified zero replay",
        "runs": runs,
        "csv_sha256": sha256(csv_path),
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sorted(sources)},
    }
    write_json(RESULTS / "key-value-curves.json", report)
    print(json.dumps({"observations": len(rows), "runs": runs}, indent=2))


if __name__ == "__main__":
    main()
