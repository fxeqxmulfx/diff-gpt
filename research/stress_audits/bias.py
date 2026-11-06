"""Replay every new residual-bias checkpoint and publish complete paired results."""

from __future__ import annotations

import csv
import itertools
import json
import shutil
from pathlib import Path

import torch

from research.stress_audits.bias_cell import audit_cell
from research.transformer_stress.common import ROOT, clean, require, write_json
from research.transformer_stress_bias.run import (
    OUTPUT,
    PLAN,
    cell_directory,
    control_rows,
    key,
    prepare,
    read,
    settings_for,
    sha256,
    summarize,
)
from research.transformer_stress_parity.run import data_digest


def paired(rows: list[dict], plan: dict) -> list[dict]:
    """Retain every new cell and its same-activation/teacher/seed control."""
    controls = {key(r): r for r in rows if r["bias_mode"] == plan["control_bias_mode"]}
    result = []
    for row in rows:
        if row["bias_mode"] == plan["control_bias_mode"]:
            continue
        reference = controls[key(dict(row, bias_mode=plan["control_bias_mode"]))]
        first, prior = (
            row["first_threshold_updates"],
            reference["first_threshold_updates"],
        )
        result.append(
            {
                **{
                    k: row[k]
                    for k in (
                        "activation_variant",
                        "bias_mode",
                        "teacher",
                        "branches",
                        "strength",
                        "seed",
                    )
                },
                "control_status": reference["status"],
                "candidate_status": row["status"],
                "control_first_threshold_updates": prior,
                "candidate_first_threshold_updates": first,
                "paired_first_update_ratio": first / prior
                if first is not None and prior is not None
                else None,
                "control_final_train_relative_mse": reference[
                    "final_train_relative_mse"
                ],
                "candidate_final_train_relative_mse": row["final_train_relative_mse"],
                "control_parameters": reference["parameters"],
                "candidate_parameters": row["parameters"],
                "added_bias_parameters": row["added_bias_parameters"],
                "control_final_heldout_relative_mse": reference[
                    "final_heldout_relative_mse"
                ],
                "candidate_final_heldout_relative_mse": row[
                    "final_heldout_relative_mse"
                ],
            }
        )
    require(len(result) == plan["expected_new_cells"], "Missing paired comparisons")
    return result


def main() -> None:
    """Verify all 96 new checkpoints before publishing the full 144-row study."""
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    require(
        (OUTPUT / "summary.json").is_file()
        and read(OUTPUT / "summary.json")["complete"],
        "Finish all declared cells before the final audit",
    )
    plan = read(PLAN)
    manifest = prepare(plan)
    original = manifest["parent_settings"]
    data = {
        "original_odd": torch.load(OUTPUT / "data.pt", weights_only=True),
        "mixed": torch.load(OUTPUT / "mixed-data.pt", weights_only=True),
    }
    require(
        {name: data_digest(value) for name, value in data.items()}
        == read(OUTPUT / "data.json"),
        "Data fingerprints changed",
    )
    rows = control_rows(plan)
    require(len(rows) == plan["reused_control_cells"], "Missing parent rows")
    checkpoint_hashes = {}
    for activation, mode, teacher, depth, strength, seed in itertools.product(
        plan["activation_variants"],
        plan["bias_modes"],
        plan["teachers"],
        original["branches"],
        original["teacher_strengths"],
        original["seeds"],
    ):
        settings = settings_for(plan, original, activation, mode)
        directory = cell_directory(activation, mode, teacher, depth, strength, seed)
        result = audit_cell(directory, settings, data[teacher], plan)
        require(
            result["status"] == "numerical_failure"
            or result["completed_updates"] == settings["updates"],
            "Incomplete original budget",
        )
        require(
            (result["branches"], result["strength"], result["seed"])
            == (depth, strength, seed),
            "Wrong checkpoint identity",
        )
        rows.append(
            dict(
                result,
                activation_variant=activation,
                bias_mode=mode,
                weight_decay=plan["weight_decay"],
                teacher=teacher,
            )
        )
        checkpoint_hashes[str((directory / "checkpoint.pt").relative_to(ROOT))] = (
            sha256(directory / "checkpoint.pt")
        )
    summary = summarize(rows, plan, original)
    require(summary["complete"], "Incomplete scientific grid")
    require(
        clean(summary) == read(OUTPUT / "summary.json"),
        "Summary differs from raw records",
    )
    pairs = paired(rows, plan)
    audit = {
        "source_files_verified": len(manifest["source_hashes"]),
        "new_cells": plan["expected_new_cells"],
        "reused_control_cells": plan["reused_control_cells"],
        "all_new_final_full_split_scores_reproduced": True,
        "all_initial_states_capacity_optimizer_sampler_histories_verified": True,
        "both_original_data_files_unchanged": True,
        "complete_grid_selection_and_pairs_reconstructed": True,
        "checkpoint_sha256": checkpoint_hashes,
        "auditor_sha256": sha256(Path(__file__)),
        "inherited_auditor_sha256": sha256(Path(__file__).with_name("bias_cell.py")),
        "control_audit_sha256": sha256(
            ROOT / "research/transformer_stress_decay/results/audit.json"
        ),
    }
    write_json(OUTPUT / "paired.json", pairs)
    write_json(OUTPUT / "audit.json", audit)
    results = PLAN.parent / "results"
    results.mkdir(exist_ok=True)
    for name in (
        "manifest.json",
        "data.json",
        "summary.json",
        "paired.json",
        "audit.json",
    ):
        shutil.copyfile(OUTPUT / name, results / name)
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with (results / "cells.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: json.dumps(v) if isinstance(v, (list, dict)) else v
                    for k, v in row.items()
                }
            )
    print(json.dumps({k: v for k, v in audit.items() if k != "checkpoint_sha256"}))
    print(json.dumps(summary["arms"]))


if __name__ == "__main__":
    main()
