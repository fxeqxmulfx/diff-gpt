"""Plot every paired parameter observation and all attention/residual summaries."""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

STUDY = Path(__file__).parent
RESULTS = STUDY / "results"
GROUPS = [
    ("initial", "original"),
    ("initial", "normalized"),
    ("final", "original_lr1"),
    ("final", "original_lr2"),
    ("final", "normalized_lr2"),
]
LABELS = [
    "Old\ninitial",
    "Normalized\ninitial",
    "Old LR1\nfinal",
    "Old LR2\nfinal",
    "Normalized\nfinal",
]
PARAMETERS = [
    ("blocks.0.sa.c_attn.weight.q", "Q"),
    ("blocks.0.sa.c_attn.weight.k", "K"),
    ("blocks.0.sa.c_attn.weight.v", "V"),
    ("blocks.0.sa.c_proj.weight", "Attention\noutput"),
    ("blocks.0.ffwd.c_fc.weight.gate", "SwiGLU\ngate"),
    ("blocks.0.ffwd.c_fc.weight.value", "SwiGLU\nvalue"),
    ("blocks.0.ffwd.c_proj.weight", "FFN\noutput"),
    ("token_embedding_table.weight", "Tied\nweight"),
]


def digest(path: Path) -> str:
    """Bind each figure to its exact inputs and plotting source."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def points(axis: plt.Axes, index: int, values: list, color: str) -> None:
    """Show every observation with deterministic display-only horizontal offsets."""
    xs = [
        index + 0.28 * (position / max(len(values) - 1, 1) - 0.5)
        for position in range(len(values))
    ]
    axis.scatter(xs, values, s=13, color=color, alpha=0.5, edgecolors="none")
    axis.scatter(
        [index],
        [statistics.median(values)],
        marker="D",
        s=32,
        color=color,
        edgecolors="black",
        linewidths=0.5,
        zorder=3,
    )


def main() -> None:
    """Render a descriptive four-panel figure without filtering seeds or batches."""
    data = json.loads((RESULTS / "measurements.json").read_text())
    summary = json.loads((RESULTS / "summary.json").read_text())
    if summary["measurements_sha256"] != digest(RESULTS / "measurements.json"):
        message = "Summary belongs to another measurement artifact"
        raise ValueError(message)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "transformer-path-diagnostic",
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    for axis, stage, control, title in [
        (axes[0, 0], "initial", "original", "Matched initialization"),
        (
            axes[0, 1],
            "final",
            "original_lr2",
            "Final checkpoints: same LR, different training histories",
        ),
    ]:
        for index, (parameter, _) in enumerate(PARAMETERS):
            values = [
                r["ratio"]
                for r in summary["paired_parameter_unit_probe_gradients"]
                if r["stage"] == stage
                and r["control"] == control
                and r["parameter"] == parameter
            ]
            points(axis, index, values, "#287c8e")
        axis.set_yscale("log")
        axis.axhline(1, color="black", linewidth=1, linestyle="--")
        axis.set_xticks(range(len(PARAMETERS)), [label for _, label in PARAMETERS])
        axis.set_ylabel("Normalized / old gradient RMS\n(shared unit logit probe)")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    colors = ["#737373", "#62ad9e", "#4e79a7", "#9c75b4", "#d46a36"]
    for index, (stage, arm) in enumerate(GROUPS):
        rows = [r for r in data["rows"] if r["stage"] == stage and r["arm"] == arm]
        attention = [
            a["normalized_entropy_mean_nonsingleton"]
            for r in rows
            for a in r["attention"]
        ]
        feedforward = [
            r["objectives"]["unit_logit_probe"]["residual_paths"][1]["gradient_paths"][
                "second_to_first_l2"
            ]
            for r in rows
        ]
        points(axes[1, 0], index, attention, colors[index])
        points(axes[1, 1], index, feedforward, colors[index])
    axes[1, 0].set_ylabel(
        "Attention entropy / log(visible keys)\n1 means uniform attention"
    )
    axes[1, 0].set_ylim(0, 1.04)
    axes[1, 0].set_title("Every head, seed and training batch")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_ylabel(
        "FFN branch / identity gradient L2\n(shared unit logit probe)"
    )
    axes[1, 1].set_title("Residual gradient contributions")
    for axis in axes[1]:
        axis.set_xticks(range(len(GROUPS)), LABELS)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle(
        "Cubic-sum: gradients remain active, but their balance changes", fontsize=15
    )
    figure.supxlabel(
        "Dots retain every observation; diamonds mark medians. "
        "Three seeds, four training batches.\n"
        "Final checkpoints have different ages and confidence; "
        "these are descriptive VJPs, not causal or worst-case bounds.",
        fontsize=9,
    )
    outputs = []
    for extension in ["png", "svg"]:
        path = RESULTS / f"gradient-paths.{extension}"
        figure.savefig(
            path, dpi=170, metadata={"Date": None} if extension == "svg" else None
        )
        if extension == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines())
                + "\n"
            )
        outputs.append(path)
    plt.close(figure)
    record = {
        "source_sha256": digest(Path(__file__)),
        "inputs_sha256": {
            p.name: digest(p)
            for p in [RESULTS / "measurements.json", RESULTS / "summary.json"]
        },
        "outputs_sha256": {p.name: digest(p) for p in outputs},
        "parameter_observations_per_panel": len(PARAMETERS) * 12,
        "attention_observations": len(data["rows"]) * 4,
        "residual_observations": len(data["rows"]),
        "no_filtered_seeds_or_batches": True,
    }
    (RESULTS / "figure.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
