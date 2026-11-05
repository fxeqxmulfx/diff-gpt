"""Audit the branch-only epsilon study without changing earlier audit sources."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path

import torch

from research.transformer_stress.common import ROOT, digest, finite, require, write_json
from research.transformer_stress.residual import evaluate
from research.transformer_stress_epsilon.models import SoftPeriNetwork


def main() -> None:
    """Verify every declared epsilon, original control and fixed-budget checkpoint."""
    torch.set_num_threads(1)
    output = ROOT / "runs/transformer-stress/epsilon-v1"
    manifest = json.loads((output / "manifest.json").read_text())
    plan, original = manifest["plan"], manifest["parent_settings"]
    for name, expected in manifest["source_hashes"].items():
        for path in (ROOT / name, output / "sources" / name):
            require(
                hashlib.sha256(path.read_bytes()).hexdigest() == expected,
                f"Changed source: {path}",
            )
    splits = torch.load(output / "data.pt", weights_only=True)
    flattened = {
        "rotation": splits["rotation"],
        **{
            f"{split}_{key}": value
            for split in ("train", "heldout")
            for key, value in splits[split].items()
        },
    }
    require(
        digest(flattened) == json.loads((output / "data.json").read_text())["sha256"],
        "Changed data",
    )
    control_path = ROOT / "runs/transformer-stress/rates-v1/summary.json"
    require(
        hashlib.sha256(control_path.read_bytes()).hexdigest()
        == manifest["control_summary_sha256"],
        "Changed control summary",
    )
    controls = [
        {"branch_epsilon": plan["control_epsilon"], **row}
        for row in json.loads(control_path.read_text())["cells"]
        if row["variant"] == plan["variant"]
        and row["learning_rate"] == plan["learning_rate"]
    ]
    rows = []
    for epsilon, depth, strength, seed in itertools.product(
        plan["new_branch_epsilons"],
        original["branches"],
        original["teacher_strengths"],
        original["seeds"],
    ):
        directory = output / f"eps{epsilon}" / f"d{depth}-a{strength}-seed{seed}"
        result = json.loads((directory / "result.json").read_text())
        history = json.loads((directory / "history.json").read_text())
        checkpoint = torch.load(directory / "checkpoint.pt", weights_only=False)
        settings = dict(
            original, learning_rate=plan["learning_rate"], branch_epsilon=epsilon
        )
        require(checkpoint["settings"] == settings, "Undeclared settings change")
        require(checkpoint["result"] == result, "Checkpoint result mismatch")
        require(result["completed_updates"] == settings["updates"], "Budget changed")
        require(
            [row["updates"] for row in history]
            == list(range(0, settings["updates"] + 1, settings["observe_updates"])),
            "Cadence changed",
        )
        first = next(
            (
                row["updates"]
                for row in history
                if row["heldout"]["relative_mse"] <= settings["target_relative_mse"]
            ),
            None,
        )
        require(first == result["first_threshold_updates"], "Incorrect crossing")
        require(
            result["status"] == ("passed" if first is not None else "budget_miss"),
            "Status mismatch",
        )
        torch.manual_seed(seed)
        model = SoftPeriNetwork(settings, depth, plan["variant"])
        require(
            digest(dict(model.branches.named_parameters()))
            == result["common_initialization_sha256"],
            "Changed initial matrices",
        )
        model.load_state_dict(checkpoint["model"], strict=True)
        require(all(finite(p) for p in model.parameters()), "Nonfinite model state")
        require(
            sum(p.numel() for p in model.parameters()) == result["parameters"],
            "Parameter count mismatch",
        )
        for split in ("train", "heldout"):
            actual = evaluate(model, splits[split], strength)["relative_mse"]
            require(
                math.isclose(actual, history[-1][split]["relative_mse"], rel_tol=1e-12),
                "Full-split score mismatch",
            )
        for state in checkpoint["optimizer"]["state"].values():
            require(
                state["step"].item() == settings["updates"], "Optimizer step mismatch"
            )
            require(
                all(finite(x) for x in state.values() if isinstance(x, torch.Tensor)),
                "Nonfinite optimizer state",
            )
        for group in checkpoint["optimizer"]["param_groups"]:
            require(
                group["lr"] == settings["learning_rate"]
                and group["betas"] == tuple(settings["betas"])
                and group["weight_decay"] == settings["weight_decay"],
                "Optimizer configuration mismatch",
            )
        rng = torch.Generator().manual_seed(seed)
        for _ in range(
            math.ceil(
                settings["updates"] / (settings["train_examples"] // settings["batch"])
            )
        ):
            torch.randperm(settings["train_examples"], generator=rng)
        require(
            torch.equal(rng.get_state(), checkpoint["rng_state"]), "Sampler changed"
        )
        rows.append({"branch_epsilon": epsilon, **result})
    summary = json.loads((output / "summary.json").read_text())
    require(
        summary["complete"] and summary["cells"] == controls + rows,
        "Changed or missing comparison rows",
    )
    require(len(rows) == plan["expected_new_cells"], "Missing scientific cells")
    audit = {
        "new_cells": len(rows),
        "reused_control_cells": len(controls),
        "passed_new_cells": sum(row["status"] == "passed" for row in rows),
        "budget_misses": sum(row["status"] == "budget_miss" for row in rows),
        "source_files_verified": len(manifest["source_hashes"]),
        "all_sources_data_training_states_and_comparison_rows_verified": True,
        "final_full_split_metrics_reproduced": True,
        "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    write_json(output / "audit.json", audit)
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
