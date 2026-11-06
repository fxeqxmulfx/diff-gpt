"""Render every initial-scale observation using the existing system plotting tools."""

import hashlib
import json
import math
import statistics
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
from matplotlib import pyplot as plt

STUDY = Path(__file__).parent
RESULTS = STUDY / "results"
TASKS = {
    "suite-boolean-and-random": ("Boolean AND", "#0072B2"),
    "power-division-p193-k2": ("Division p193", "#E69F00"),
    "power-division-p97-k2": ("Division p97", "#009E73"),
    "gpt-probe-key-value-binding": ("Key/value binding", "#CC79A7"),
    "arithmetic-square-sum": ("Square-sum", "#D55E00"),
    "gpt-probe-linear-transport": ("Linear transport", "#56B4E9"),
}
ARMS = {
    "original": "Original",
    "embedding_rms": "Embedding\nRMS",
    "scaled_peri_rms": "Scaled\nPeri-RMS",
    "both": "Both",
}


def digest(path: Path) -> str:
    """Bind the figure to its exact committed measurements."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Plot 18 observations per arm, with no missing or censored points removed."""
    source = RESULTS / "initial-scales.json"
    data = json.loads(source.read_text())
    rows = {(r["case"], r["seed"], r["arm"]): r for r in data["rows"]}
    expected = {
        (task, seed, arm) for task in TASKS for seed in range(3) for arm in ARMS
    }
    if set(rows) != expected or len(data["rows"]) != len(expected):
        message = "Figure requires every unique declared initial-scale observation"
        raise ValueError(message)
    plt.rcParams.update({"svg.hashsalt": "core-initial-scale", "font.size": 10})
    figure, axes = plt.subplots(1, 2, figsize=(11, 5.6), sharey=True)
    for axis, branch in zip(axes, ("attention", "feedforward"), strict=True):
        axis.set_title("Attention branch" if branch == "attention" else "SwiGLU branch")
        axis.axhline(1, color="#777777", linestyle="--", linewidth=0.8, zorder=1)
        for index, arm in enumerate(ARMS):
            values = []
            for task_index, (task, (label, color)) in enumerate(TASKS.items()):
                for seed in range(3):
                    ratio = rows[task, seed, arm][branch]["global_rms_ratio"]
                    if not math.isfinite(ratio) or ratio <= 0:
                        message = "Log plot cannot represent a missing or zero ratio"
                        raise ValueError(message)
                    values.append(ratio)
                    offset = (task_index * 3 + seed - 8.5) * 0.025
                    axis.scatter(
                        index + offset,
                        ratio,
                        color=color,
                        s=24,
                        label=label if index == 0 and seed == 0 else None,
                        zorder=3,
                    )
            median = statistics.median(values)
            axis.plot([index - 0.26, index + 0.26], [median, median], color="#222222")
        axis.set_yscale("log")
        axis.set_ylim(0.001, 100)
        axis.set_xlim(-0.5, len(ARMS) - 0.5)
        axis.set_xticks(range(len(ARMS)), ARMS.values())
        axis.grid(axis="y", color="#e4e4e4", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Branch RMS / residual-state RMS (log scale)")
    figure.suptitle("Initial branch magnitude relative to its residual", fontsize=15)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.09), ncol=3
    )
    figure.text(
        0.5,
        0.035,
        "First block; first 32 training examples; "
        "6 tasks x 3 seeds; all token positions.\n"
        "Black bars: medians across 18 initializations. Dashed line: equal RMS. "
        "This measures initialization only.",
        ha="center",
        fontsize=9,
    )
    figure.subplots_adjust(bottom=0.31, top=0.84, left=0.09, right=0.98, wspace=0.12)
    figure.savefig(RESULTS / "initial-scales.svg", metadata={"Date": None})
    figure.savefig(RESULTS / "initial-scales.png", dpi=180)
    plt.close(figure)
    record = {
        "observations": len(rows),
        "source_sha256": {
            source.name: digest(source),
            "plot.py": digest(Path(__file__)),
        },
        "figure_sha256": {
            path.name: digest(path)
            for path in (RESULTS / "initial-scales.svg", RESULTS / "initial-scales.png")
        },
        "matplotlib": mpl.__version__,
        "command": (
            "MPLCONFIGDIR=/tmp/gpt-core-initial-scale-mpl "
            "/usr/bin/python3 research/core_initial_scale/plot.py"
        ),
        "training_environment_changed": False,
    }
    (RESULTS / "figure.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
