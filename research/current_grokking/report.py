"""Collect complete, bounded-negative, pending and execution-error outcomes."""

import argparse
import csv
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

from research.current_grokking.protocol import ROOT, Case
from research.current_grokking.run import metric, write


def collect(root: Path) -> dict:
    """Report the whole predeclared matrix, including every unfinished cell."""
    plan = json.loads((root / "plan.json").read_text())
    rows, families = [], defaultdict(Counter)
    for case in plan["cases"]:
        for seed in plan["seeds"]:
            directory = root / case["name"] / f"seed{seed}"
            row = {
                "case": case["name"],
                "kind": case["kind"],
                "seed": seed,
                "metric": case["metric"],
                "target": case["target"],
                "max_updates": case["max_updates"],
            }
            if (directory / "result.json").exists():
                result = json.loads((directory / "result.json").read_text())
                row.update(
                    {
                        key: result.get(key)
                        for key in (
                            "status",
                            "completed",
                            "first_fit",
                            "first_train99",
                            "first_target",
                            "classification",
                            "final_target_met",
                            "seconds",
                        )
                    }
                )
                if row["first_fit"] is None:
                    row["first_fit"] = row["first_train99"]
                # The serialized case is the exact protocol used by the metric helper.
                spec = Case(**case)
                row["final_validation"] = metric(spec, result["endpoint"]["validation"])
                row["final_train"] = metric(spec, result["endpoint"]["train"])
                row["final_test"] = result["final_test"]
                row["result_path"] = str(directory / "result.json")
                row["audit_available"] = (directory / "audit.json").exists()
                row["model_source_reference"] = result["metadata"].get(
                    "model_source_reference"
                )
            elif (directory / "status.json").exists():
                status = json.loads((directory / "status.json").read_text())
                row.update(status="in_progress", completed=status["step"])
                if list(directory.glob("error-*.json")):
                    row["status"] = "interrupted_or_execution_error"
            elif list(directory.glob("error-*.json")):
                row["status"] = "execution_error"
            elif directory.exists():
                row["status"] = "preparing"
            else:
                row["status"] = "pending"
            families[case["kind"]]["planned"] += 1
            families[case["kind"]][row["status"]] += 1
            if row["status"] == "passed":
                retained = bool(row["final_target_met"])
                families[case["kind"]][
                    "retained_target" if retained else "transient_crossing"
                ] += 1
            if row.get("classification") == "delayed_generalization":
                families[case["kind"]]["delayed_generalization"] += 1
            rows.append(row)
    counts = Counter(row["status"] for row in rows)
    return {
        "model": plan["model"],
        "optimizer": plan["optimizer"],
        "planned_runs": len(rows),
        "counts": dict(counts),
        "observed_target_crossings": counts["passed"],
        "retained_targets": sum(bool(row.get("final_target_met")) for row in rows),
        "transient_crossings": sum(
            row["status"] == "passed" and not row["final_target_met"] for row in rows
        ),
        "families": {key: dict(value) for key, value in families.items()},
        "complete": all(
            row["status"] in {"passed", "budget_exhausted"} for row in rows
        ),
        "interpretation": (
            "Passed means the declared validation target was observed at least once; "
            "final_target_met records retention. Budget exhaustion is a bounded miss. "
            "Delayed generalization requires two fitted-train/near-baseline heldout "
            "observations before target. One model seed is an initial screen. "
            "Controls, forecasting, language modeling and new supplied-feature GPT "
            "probes have task-specific targets, not interchangeable grokking claims."
        ),
        "runs": rows,
    }


def main() -> None:
    """Write JSON/CSV coverage without discarding pending or negative results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "runs/current-grokking")
    args = parser.parse_args()
    report = collect(args.root)
    write(args.root / "summary.json", report)
    fields = [
        "case",
        "kind",
        "seed",
        "status",
        "metric",
        "target",
        "max_updates",
        "completed",
        "first_train99",
        "first_fit",
        "first_target",
        "classification",
        "final_target_met",
        "final_train",
        "final_validation",
        "seconds",
        "audit_available",
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(report["runs"])
    (args.root / "summary.csv").write_text(buffer.getvalue())
    print(
        json.dumps({key: report[key] for key in ("planned_runs", "counts", "complete")})
    )


if __name__ == "__main__":
    main()
