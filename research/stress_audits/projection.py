"""Replay every new projection checkpoint and publish all paired outcomes."""

from __future__ import annotations

import csv
import itertools
import json
import shutil
from pathlib import Path

import torch

from research.stress_audits.projection_cell import audit_cell
from research.transformer_stress.common import ROOT, clean, require, write_json
from research.transformer_stress_parity.run import data_digest
from research.transformer_stress_projection.run import (
    OUTPUT,
    PLAN,
    RESULTS,
    cell_directory,
    control_rows,
    key,
    prepare,
    read,
    settings_for,
    sha256,
    summarize,
)


def paired(rows: list[dict], plan: dict) -> list[dict]:
    """Compare every new cell to its identical teacher/depth/strength/seed control."""
    mode = plan["control_projection_mode"]
    controls = {key(r): r for r in rows if r["projection_mode"] == mode}
    result = []
    for row in rows:
        if row["projection_mode"] == mode:
            continue
        parent = controls[key(dict(row, projection_mode=mode))]
        first, prior = row["first_threshold_updates"], parent["first_threshold_updates"]
        result.append(
            {
                **{
                    k: row[k]
                    for k in (
                        "projection_mode",
                        "teacher",
                        "branches",
                        "strength",
                        "seed",
                    )
                },
                "control_status": parent["status"],
                "candidate_status": row["status"],
                "control_first_threshold_updates": prior,
                "candidate_first_threshold_updates": first,
                "paired_first_update_ratio": first / prior
                if first is not None and prior is not None
                else None,
                **{
                    f"{label}_{name}": source[name]
                    for label, source in (("control", parent), ("candidate", row))
                    for name in (
                        "final_train_relative_mse",
                        "final_heldout_relative_mse",
                        "parameters",
                    )
                },
            }
        )
    require(len(result) == plan["expected_new_cells"], "Missing paired comparisons")
    require(
        all(r["candidate_parameters"] == r["control_parameters"] for r in result),
        "Changed paired capacity",
    )
    return result


def main() -> None:
    """Verify all new checkpoints before publishing the complete 48-cell grid."""
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    require(
        (OUTPUT / "summary.json").is_file()
        and read(OUTPUT / "summary.json")["complete"],
        "Finish the declared grid first",
    )
    plan = read(PLAN)
    manifest = prepare(plan)
    original = manifest["parent_settings"]
    data = {
        "original_odd": torch.load(OUTPUT / "data.pt", weights_only=True),
        "mixed": torch.load(OUTPUT / "mixed-data.pt", weights_only=True),
    }
    require(
        {n: data_digest(v) for n, v in data.items()} == read(OUTPUT / "data.json"),
        "Data fingerprints changed",
    )
    rows = control_rows(plan)
    require(len(rows) == plan["reused_control_cells"], "Missing controls")
    checkpoint_hashes, history_hashes = {}, {}
    for mode, teacher, depth, strength, seed in itertools.product(
        plan["projection_modes"],
        plan["teachers"],
        original["branches"],
        original["teacher_strengths"],
        original["seeds"],
    ):
        directory = cell_directory(mode, teacher, depth, strength, seed)
        settings = settings_for(plan, original, mode)
        result = audit_cell(directory, settings, data[teacher], plan)
        require(
            (result["branches"], result["strength"], result["seed"])
            == (depth, strength, seed),
            "Wrong cell identity",
        )
        rows.append(
            dict(
                result,
                projection_mode=mode,
                activation_variant=plan["activation_variant"],
                bias_mode=plan["bias_mode"],
                weight_decay=plan["weight_decay"],
                teacher=teacher,
            )
        )
        checkpoint_hashes[str((directory / "checkpoint.pt").relative_to(ROOT))] = (
            sha256(directory / "checkpoint.pt")
        )
        history_hashes[str((directory / "history.json").relative_to(ROOT))] = sha256(
            directory / "history.json"
        )
    summary = summarize(rows, plan, original)
    require(summary["complete"], "Incomplete grid")
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
        "history_sha256": history_hashes,
        "auditor_sha256": sha256(Path(__file__)),
        "cell_auditor_sha256": sha256(Path(__file__).with_name("projection_cell.py")),
        "control_audit_sha256": sha256(
            ROOT / "research/transformer_stress_bias/results/audit.json"
        ),
    }
    write_json(OUTPUT / "paired.json", pairs)
    write_json(OUTPUT / "audit.json", audit)
    for name in (
        "manifest.json",
        "data.json",
        "summary.json",
        "paired.json",
        "audit.json",
    ):
        shutil.copyfile(OUTPUT / name, RESULTS / name)
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with (RESULTS / "cells.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: json.dumps(v) if isinstance(v, (list, dict)) else v
                    for k, v in row.items()
                }
            )
    print(
        json.dumps(
            {
                k: v
                for k, v in audit.items()
                if k not in {"checkpoint_sha256", "history_sha256"}
            }
        )
    )
    print(json.dumps(summary["arms"]))


if __name__ == "__main__":
    main()
