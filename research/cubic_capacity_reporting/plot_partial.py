"""Plot all seeds from one audited completed contrast, retaining first crossings."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "research/cubic_capacity/results"
EXPECTED_CELLS = 6


def digest(path: Path) -> str:
    """Check exact input bytes before drawing a partial learning-curve figure."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:  # noqa: FBT001 -- Assertion-style guard.
    """Refuse incomplete or replay-ambiguous trajectories rather than hiding them."""
    if not condition:
        raise ValueError(message)


def epochs(updates: int, count: int, batch: int) -> float:
    """Use exact presented examples, including the short final training batch."""
    per_epoch = math.ceil(count / batch)
    full, remainder = divmod(updates, per_epoch)
    return (full * count + min(remainder * batch, count)) / count


def main() -> None:
    """Render every seed and both train/held-out curves from the audited raw data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    source = RESULTS / f"partial-group-{args.name}.json"
    result = json.loads(source.read_text())
    require(
        result["complete_group"] and not result["complete_study"], "Wrong report scope"
    )
    cells = result["cells"]
    require(
        len(cells) == EXPECTED_CELLS
        and len({(r["size"], r["seed"]) for r in cells}) == EXPECTED_CELLS,
        "Incomplete six-cell contrast",
    )
    require({r["seed"] for r in cells} == {0, 1, 2}, "Changed seed coverage")
    for path, expected in result["source_and_evidence_sha256"].items():
        require(
            digest(ROOT / path) == expected, f"Changed report source/evidence: {path}"
        )
    for row in cells:
        require(
            row["resumptions"] == 0
            and row["replay_examples_lower"] == 0
            and row["replay_examples_upper"] == 0,
            "A resumed trajectory needs explicit replay plotting; do not hide it",
        )
        for path, expected in row["artifacts_sha256"].items():
            require(
                digest(ROOT / path) == expected, f"Changed audited artifact: {path}"
            )
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.7), sharex=True, sharey=True)
    task = result["task"]
    for row in cells:
        history = [
            json.loads(line)
            for line in (ROOT / row["directory"] / "history.jsonl")
            .read_text()
            .splitlines()
        ]
        x = [
            epochs(h["step"], task["train_examples"], task["case"]["batch"])
            for h in history
        ]
        require(
            math.isclose(x[-1], row["actual_epochs_lower"], rel_tol=0, abs_tol=1e-10),
            "Plotted exposure differs from the audited work",
        )
        require(
            row["actual_epochs_lower"] == row["actual_epochs_upper"],
            "Uncertain exposure",
        )
        axis = axes[row["seed"]]
        color = "#2368a2" if row["reused"] else "#cf5b35"
        label = f"{row['layers']} layer" + ("s" if row["layers"] != 1 else "")
        label += f", width {row['width']}"
        heldout = [h["validation"]["sequence_accuracy"] for h in history]
        train = [h["train"]["sequence_accuracy"] for h in history]
        axis.plot(x, heldout, color=color, linewidth=1.3, label=f"{label}: held-out")
        axis.plot(
            x,
            train,
            color=color,
            linewidth=0.9,
            linestyle="--",
            alpha=0.5,
            label=f"{label}: train",
        )
        marker = "*" if row["status"] == "passed" else "X"
        axis.scatter(
            [x[-1]],
            [heldout[-1]],
            color=color,
            marker=marker,
            s=100 if marker == "*" else 55,
            zorder=5,
        )
    for seed, axis in enumerate(axes):
        axis.axhline(0.99, color="#555555", linestyle=":", linewidth=1)
        axis.set_title(f"Seed {seed}")
        axis.set_ylim(-0.02, 1.025)
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.grid(alpha=0.18)
    axes[0].set_ylabel("Exact accuracy on the complete split")
    figure.supxlabel(
        "Training epochs (actual examples / training-set size)", y=0.09, fontsize=11
    )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    group = result["groups"][1]
    arm_label = {
        "original_lr1": "original model, LR 0.001",
        "original_lr2": "original model, LR 0.002",
        "normalized_lr2": "normalized inverse-width model, LR 0.002",
    }[group["arm"]]
    figure.suptitle(
        f"Cubic-sum: {arm_label} — capacity comparison",
        fontsize=14,
        y=0.99,
    )
    figure.text(
        0.5,
        0.02,
        "Stars mark first observed 99%; X marks budget exhaustion. "
        "One completed contrast; the full 27-run capacity study remains open.",
        ha="center",
        fontsize=9,
    )
    figure.subplots_adjust(top=0.72, bottom=0.18, left=0.07, right=0.99, wspace=0.12)
    outputs = []
    for suffix in ["png", "svg"]:
        target = RESULTS / f"partial-group-{args.name}.{suffix}"
        figure.savefig(target, dpi=160, facecolor="white")
        if suffix == "svg":
            # Matplotlib leaves trailing spaces in multiline path attributes.
            target.write_text(
                "\n".join(line.rstrip() for line in target.read_text().splitlines())
                + "\n"
            )
        outputs.append(target)
    plt.close(figure)
    record = {
        "complete_group": True,
        "complete_study": False,
        "plotted_cells": len(cells),
        "plotted_seeds": [0, 1, 2],
        "source_sha256": {
            str(p.relative_to(ROOT)): digest(p) for p in [source, Path(__file__)]
        },
        "figures_sha256": {str(p.relative_to(ROOT)): digest(p) for p in outputs},
        "scope": result["limits"],
    }
    (RESULTS / f"partial-group-{args.name}-figure.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                k: record[k]
                for k in ["complete_group", "complete_study", "plotted_cells"]
            }
        )
    )


if __name__ == "__main__":
    main()
