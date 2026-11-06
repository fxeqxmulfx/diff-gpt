"""Reconstruct every screen comparison and verify the supplemental evidence."""

from __future__ import annotations

import json
import re
from pathlib import Path

from research.early99.experiments.compare import paired_cell
from research.early99.protocol import SEEDS, selected
from research.early99_dimension_gain.readiness import (
    require,
    sha256,
    verify_training_ready,
)
from research.early99_dimension_gain_reporting.run_full import promoted
from research.early99_dimension_gain_reporting.run_screens import (
    RESULTS,
    ROOT,
    RUNS,
    STUDY,
    read,
    screen_decision,
    task_metrics,
    verify_csv,
    write_json,
)

UNCHANGED_WIDTH = 16


def main() -> None:
    """Verify complete evidence without changing a run, screen or frozen choice."""
    verify_training_ready()
    plan = read(STUDY / "plan.json")
    name = plan["configurations"][0]["name"]
    require(promoted(name, plan), "The declared promotion gate was not met")
    tasks = {r["name"]: r for r in selected("short")}
    controls = {
        "parent": ROOT / plan["screen_control"],
        "lr2": RUNS / "adamw-lr2",
        "baseline": ROOT / "runs/early99/baseline",
        "noqk": RUNS / "peri-rms-scaled-ssmax-noqk-lr2",
    }
    hashes, summaries = {}, {}
    for label, control in controls.items():
        path = RESULTS / f"screen-{name}-vs-{label}.json"
        record = read(path)
        rows, task_rows = [], []
        for task in plan["screen_tasks"]:
            group = [
                paired_cell(tasks[task], seed, control, RUNS / name) for seed in SEEDS
            ]
            rows.extend(group)
            task_rows.append(task_metrics(task, group))
        require(
            rows == record["runs"] and task_rows == record["tasks"],
            "Comparison changed",
        )
        passed, improved, gate = screen_decision(rows, task_rows)
        require(
            (passed, improved) == (record["passed"], record["improved_task_medians"]),
            "Metrics changed",
        )
        require(
            record["advance"] == (gate if label == "parent" else None),
            "Control policy changed",
        )
        require(
            record["geometric_mean_epoch_ratio"] is None,
            "Screen has a forbidden aggregate",
        )
        require(
            record["search_plan_sha256"] == sha256(STUDY / "plan.json"),
            "Search plan changed",
        )
        require(
            record["candidate_plan_sha256"] == sha256(RUNS / name / "plan.json"),
            "Candidate plan changed",
        )
        csv_path, task_path = (
            path.with_suffix(".csv"),
            path.with_name(path.stem + "-tasks.csv"),
        )
        verify_csv(csv_path, rows)
        verify_csv(task_path, task_rows)
        for file in (path, csv_path, task_path):
            hashes[str(file.relative_to(ROOT))] = sha256(file)
        summaries[label] = {
            "passed": passed,
            "improved_task_medians": improved,
            "earlier": sum(r["epoch_ratio"] < 1 for r in rows),
            "tied": sum(r["epoch_ratio"] == 1 for r in rows),
            "later": sum(r["epoch_ratio"] > 1 for r in rows),
            "worst_ratio": max(r["epoch_ratio"] for r in rows),
        }
    proof = read(RESULTS / "guarantees.json")
    output = ROOT / proof["output"]
    require(sha256(output) == proof["output_sha256"], "Lean output changed")
    text = output.read_text()
    require(
        not any(s in text for s in ("warning:", "error:", "sorryAx")),
        "Unchecked proof output",
    )
    axioms = {
        n: a.split(", ")
        for n, a in re.findall(
            r"'(Early99DimensionGain\.[^']+)' depends on axioms: \[([^\]]*)\]", text
        )
    }
    require(axioms == proof["axioms"], "Printed axioms do not match the proof record")
    require(
        all(
            set(a) <= {"propext", "Classical.choice", "Quot.sound"}
            for a in axioms.values()
        ),
        "Extra proof axiom",
    )
    for source, expected in proof["external_source_sha256"].items():
        require(
            sha256(Path(proof["external_project"]) / source) == expected,
            "External proof dependency changed",
        )
    stress = read(RESULTS / "score-stress.json")
    probability = max(r["maximum_probability_absolute_error"] for r in stress["rows"])
    gradient = max(
        max(r["query_relative_gradient_error"], r["key_relative_gradient_error"])
        for r in stress["rows"]
    )
    require(
        probability == stress["maximum_probability_absolute_error"]
        and gradient == stress["maximum_relative_gradient_error"],
        "Stress maxima changed",
    )
    require(
        probability <= plan["score_stress"]["maximum_probability_absolute_error"]
        and gradient <= plan["score_stress"]["maximum_relative_gradient_error"],
        "Stress tolerance exceeded",
    )
    probe = read(RESULTS / "width16-probe.json")
    require(
        probe["source_sha256"] == sha256(Path(__file__).with_name("width16_probe.py")),
        "Probe source changed",
    )
    expected_probes = {
        t
        for t in plan["screen_tasks"]
        if tasks[t]["case"]["width"] // tasks[t]["case"]["heads"] == UNCHANGED_WIDTH
    }
    require(
        {r["case"] for r in probe["rows"]} == expected_probes
        and len(probe["rows"]) == len(expected_probes),
        "Incomplete equal-gain probe",
    )
    for row in probe["rows"]:
        require(
            row["gain"]["quarter"] == row["gain"]["inverse"], "Unequal control gains"
        )
        require(
            row["eager"]["parent_vs_candidate_gradients"]["unequal_elements"] == 0,
            "Eager control gradients differ",
        )
    for file in (
        RESULTS / "audit-screen.json",
        RESULTS / "guarantees.json",
        RESULTS / "score-stress.json",
        RESULTS / "width16-probe.json",
    ):
        hashes[str(file.relative_to(ROOT))] = sha256(file)
    result = {
        "ready_for_full_short": True,
        "comparisons": summaries,
        "evidence_sha256": hashes,
        "source_sha256": sha256(Path(__file__)),
        "equal_gain_caveat": (
            "Eager gradients agree in all three seed-0 probes, but compiled and "
            "repeated compiled gradients show small differences. Width-16 learning "
            "trajectories are not bitwise controls. All three width-32 task medians "
            "improve, so promotion does not rely on a width-16 gain."
        ),
    }
    write_json(RESULTS / "screen-integrity.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "evidence_sha256"}))


if __name__ == "__main__":
    main()
