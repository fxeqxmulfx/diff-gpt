"""Plot every seed's crossing or budget censor and final train/held-out accuracy."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "research/cubic_capacity/results"
SIZES = ["current", "depth", "width", "depth_width"]
ARMS = ["original_lr1", "original_lr2", "normalized_lr2"]
TITLES = ["Original · LR 0.001", "Original · LR 0.002", "Normalized · LR 0.002"]
LABELS = ["L1 · W128\n209k", "L2 · W128\n405k", "L1 · W256\n812k", "L2 · W256\n1,598k"]
COLORS = ["#2563eb", "#b45309", "#0f766e"]


def digest(path: Path) -> str:
    """Bind the rendered figure to the full completed comparison."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Keep all 36 outcomes visible without a successful-only average."""
    path = RESULTS / "comparison.json"
    report = json.loads(path.read_text())
    if not report["complete"] or len(report["cells"]) != 36:  # noqa: PLR2004
        message = "The figure requires every declared cell"
        raise ValueError(message)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.4), sharex=True, sharey="row")
    for column, (arm, title, color) in enumerate(
        zip(ARMS, TITLES, COLORS, strict=True)
    ):
        top, bottom = axes[:, column]
        for row in [r for r in report["cells"] if r["arm"] == arm]:
            x = SIZES.index(row["size"]) + (row["seed"] - 1) * 0.14
            passed = row["status"] == "passed"
            low, high = row["actual_epochs_lower"], row["actual_epochs_upper"]
            top.scatter(
                x, low, color=color, marker="o" if passed else "^", s=36, zorder=3
            )
            if high is None or high != low:
                top.annotate(
                    "",
                    (x, high if high is not None else low * 1.1),
                    (x, low),
                    arrowprops={"arrowstyle": "->", "color": color},
                )
            bottom.plot(
                [x, x],
                [
                    100 * row["final_train_accuracy"],
                    100 * row["final_heldout_accuracy"],
                ],
                color=color,
                alpha=0.25,
                lw=1,
            )
            bottom.scatter(
                x,
                100 * row["final_train_accuracy"],
                facecolors="none",
                edgecolors=color,
                marker="s",
                s=30,
                zorder=3,
            )
            bottom.scatter(
                x,
                100 * row["final_heldout_accuracy"],
                color=color,
                marker="o",
                s=30,
                zorder=3,
            )
        top.set_title(title, fontsize=12, pad=10)
        top.set_ylim(bottom=0)
        bottom.set_ylim(-4, 105)
        bottom.axhline(99, color="#666666", ls=":", lw=0.9)
        for ax in (top, bottom):
            ax.set_xticks(range(len(SIZES)), LABELS, fontsize=10)
            ax.set_xlim(-0.45, 3.45)
            ax.grid(axis="y", alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
        bottom.set_xlabel("Layers · width / unique parameters", fontsize=10, labelpad=8)
    axes[0, 0].set_ylabel("Epochs to first 99% or budget censor")
    axes[1, 0].set_ylabel("Final exact accuracy (%)")
    axes[0, 0].scatter(
        [], [], c="#444444", s=36, marker="o", label="First held-out 99%"
    )
    axes[0, 0].scatter([], [], c="#444444", s=36, marker="^", label="Budget exhausted")
    axes[0, 0].legend(loc="best", fontsize=9, frameon=False)
    axes[1, 0].scatter(
        [], [], edgecolors="#444444", facecolors="none", s=30, marker="s", label="Train"
    )
    axes[1, 0].scatter([], [], c="#444444", s=30, label="Held-out")
    axes[1, 0].legend(loc="best", fontsize=9, frameon=False)
    fig.suptitle(
        "Cubic-sum: does depth or width recover generalization?", fontsize=16, y=0.99
    )
    fig.text(
        0.5,
        0.025,
        "Three paired seeds per group · fixed head width 32 · all failures retained\n"
        "One retrospectively selected task; current-size results are reused. "
        "This does not select a benchmark winner.",
        ha="center",
        fontsize=10,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.075, 1, 0.96))
    outputs = []
    for extension in ["png", "svg"]:
        output = RESULTS / f"capacity-comparison.{extension}"
        fig.savefig(
            output,
            dpi=180,
            metadata={"Creator": "cubic_capacity_reporting.plot"}
            if extension == "svg"
            else {},
        )
        outputs.append(output)
    plt.close(fig)
    record = {
        "comparison_sha256": digest(path),
        "plotter_sha256": digest(Path(__file__)),
        "cells": len(report["cells"]),
        "figures_sha256": {output.name: digest(output) for output in outputs},
    }
    (RESULTS / "figure.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
