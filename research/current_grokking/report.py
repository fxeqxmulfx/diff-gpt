"""Collect complete, bounded-negative, pending and execution-error outcomes."""

import argparse
import csv
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

from research.current_grokking.protocol import ROOT, Case
from research.current_grokking.run import metric, write

HISTORY_ACCURACY_TARGET = 0.99
FIVE_MINUTES_SECONDS = 300


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
                **{
                    key: case[key]
                    for key in ("width", "layers", "heads", "batch", "learning_rate")
                },
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
                row["parameters"] = result["metadata"]["parameters"]
                row["resumed"] = bool(result["metadata"].get("resumptions"))
                history_path = directory / "history.jsonl"
                observations = [
                    json.loads(line)
                    for line in history_path.read_text().splitlines()
                ]
                accuracies = [
                    (
                        value,
                        value["validation"].get(
                            "sequence_accuracy", value["validation"].get("accuracy")
                        ),
                    )
                    for value in observations
                ]
                crossing99 = next(
                    (
                        (value, accuracy)
                        for value, accuracy in accuracies
                        if accuracy is not None and accuracy >= HISTORY_ACCURACY_TARGET
                    ),
                    None,
                )
                if crossing99 is not None:
                    row["first_heldout99"] = crossing99[0]["step"]
                    row["first_heldout99_accuracy"] = crossing99[1]
                    if not row["resumed"]:
                        row["first_heldout99_seconds"] = crossing99[0]["seconds"]
                if row["first_target"] is not None and not row["resumed"]:
                    crossing = next(
                        value
                        for value in observations
                        if value["step"] == row["first_target"]
                    )
                    row["first_target_seconds"] = crossing["seconds"]
                    row["initial_observation_seconds"] = observations[0]["seconds"]
                    row["seconds_after_initial_observation"] = (
                        crossing["seconds"] - observations[0]["seconds"]
                    )
                for split in ("train", "validation"):
                    measured = result["endpoint"][split]
                    row[split + "_examples"] = measured.get(
                        "examples", measured.get("example_count")
                    )
                    for key in (
                        "loss",
                        "token_accuracy",
                        "balanced_accuracy",
                        "final_answer_accuracy",
                        "decoded_accuracy",
                        "native_token_accuracy",
                        "numeric_output_coverage",
                    ):
                        if key in measured:
                            row[split + "_" + key] = measured[key]
                row["final_test"] = result["final_test"]
                if "final_extra" in result:
                    extra = result["final_extra"]
                    row["final_extra"] = extra
                    if extra.get("applicable"):
                        row["forecast_origins"] = extra["origins"]
                        for model_name, scores in extra.get("scores", {}).items():
                            row["forecast_" + model_name + "_mse"] = scores[
                                "train_normalized"
                            ]["mse"]
                row["result_path"] = str(directory / "result.json")
                row["audit_available"] = (directory / "audit.json").exists()
                if row["audit_available"]:
                    replay = json.loads((directory / "audit.json").read_text())
                    values = replay["replayed"]["validation"]
                    row["cpu_primary_metric"] = metric(spec, values)
                    row["endpoint_target_decision_reproduced"] = (
                        row["cpu_primary_metric"] >= spec.target
                    ) == row["final_target_met"]
                    if (
                        "correct" in values
                        and "correct" in result["endpoint"]["validation"]
                    ):
                        row["cpu_correct_count_delta"] = (
                            values["correct"]
                            - result["endpoint"]["validation"]["correct"]
                        )
                row["model_source_reference"] = result["metadata"].get(
                    "model_source_reference"
                )
            elif (directory / "skipped.json").exists():
                skipped = json.loads((directory / "skipped.json").read_text())
                row.update(
                    status="skipped_at_user_request",
                    completed=skipped.get("completed", 0),
                    skip_reason=skipped["reason"],
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
        "observed_heldout99_crossings": sum(
            row.get("first_heldout99") is not None for row in rows
        ),
        "heldout99_within_five_minutes": sum(
            row.get("first_heldout99_seconds", float("inf")) <= FIVE_MINUTES_SECONDS
            for row in rows
        ),
        "heldout99_definition": (
            "Any observed primary held-out sequence/classification accuracy >= 0.99; "
            "no delay or retention requirement. This split is validation where the "
            "benchmark uses ongoing held-out evaluation."
        ),
        "retained_targets": sum(bool(row.get("final_target_met")) for row in rows),
        "transient_crossings": sum(
            row["status"] == "passed" and not row["final_target_met"] for row in rows
        ),
        "audited_endpoints": sum(bool(row.get("audit_available")) for row in rows),
        "endpoint_target_decision_mismatches": [
            row["case"]
            for row in rows
            if row.get("endpoint_target_decision_reproduced") is False
        ],
        "families": {key: dict(value) for key, value in families.items()},
        "complete": all(
            row["status"] in {"passed", "budget_exhausted", "skipped_at_user_request"}
            for row in rows
        ),
        "interpretation": (
            "Passed means the declared validation target was observed at least once; "
            "final_target_met records retention. Budget exhaustion is a bounded miss. "
            "Delayed generalization requires two fitted-train/near-baseline heldout "
            "observations before target. One model seed is an initial screen. "
            "Controls, forecasting, language modeling and new supplied-feature GPT "
            "probes have task-specific targets, not interchangeable grokking claims."
        ),
        "timing_note": (
            "First-target times include setup and initial evaluation. The interval "
            "after initial evaluation also includes scoring and checkpoint overhead. "
            "These timing fields are omitted for resumed runs because their clocks "
            "reset; seconds for a resumed result describes its last segment."
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
        "width",
        "layers",
        "heads",
        "parameters",
        "batch",
        "learning_rate",
        "completed",
        "first_train99",
        "first_fit",
        "first_target",
        "first_heldout99",
        "first_heldout99_accuracy",
        "first_heldout99_seconds",
        "classification",
        "final_target_met",
        "final_train",
        "final_validation",
        "train_examples",
        "validation_examples",
        "train_loss",
        "validation_loss",
        "validation_token_accuracy",
        "validation_balanced_accuracy",
        "validation_final_answer_accuracy",
        "validation_decoded_accuracy",
        "validation_native_token_accuracy",
        "validation_numeric_output_coverage",
        "seconds",
        "resumed",
        "first_target_seconds",
        "initial_observation_seconds",
        "seconds_after_initial_observation",
        "audit_available",
        "cpu_primary_metric",
        "endpoint_target_decision_reproduced",
        "cpu_correct_count_delta",
        "forecast_origins",
        "forecast_current_gpt_mse",
        "forecast_persistence_mse",
        "forecast_seasonal_naive_mse",
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
