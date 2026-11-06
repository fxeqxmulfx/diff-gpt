"""Render all exported learning observations using existing system Matplotlib."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "research/early99_core_ablations/results"
ARMS = {
    "original": ("Original LR x2", "#0072B2"),
    "embedding": ("Embedding RMS only", "#E69F00"),
    "peri": ("Scaled Peri-RMS only", "#D55E00"),
    "both": ("Both", "#009E73"),
}


def digest(path: Path) -> str:
    """Pin exactly the exported observations and rendering sources."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:  # noqa: FBT001
    """Reject partial or changed input rather than silently dropping a curve."""
    if not condition:
        raise ValueError(message)


def main() -> None:
    """Display zero-epoch observations, every later point and all budget misses."""
    data_path = RESULTS / "key-value-curves.json"
    csv_path = RESULTS / "key-value-curves.csv"
    data = json.loads(data_path.read_text())
    require(digest(csv_path) == data["csv_sha256"], "Curve CSV has changed")
    for path, expected in data["source_sha256"].items():
        require(digest(ROOT / path) == expected, f"Curve source changed: {path}")
    with csv_path.open(newline="") as handle:
        points = list(csv.DictReader(handle))
    expected = {(arm, seed) for arm in ARMS for seed in range(3)}
    summaries = {(run["arm"], run["seed"]): run for run in data["runs"]}
    require(
        set(summaries) == expected and len(data["runs"]) == len(expected),
        "Incomplete runs",
    )
    require(len(points) == data["observations"], "Incomplete observations")
    require(
        {(point["arm"], int(point["seed"])) for point in points} == expected,
        "Unexpected or missing curve",
    )
    plt.rcParams.update({"svg.hashsalt": "core-key-value-curves", "font.size": 10})
    figure, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)
    for seed in range(3):
        axes[0, seed].set_title(f"Seed {seed}")
        for arm, (_, color) in ARMS.items():
            curve = [p for p in points if p["arm"] == arm and int(p["seed"]) == seed]
            summary = summaries[arm, seed]
            require(len(curve) == summary["observations"], "Missing curve points")
            epochs = [float(p["epochs"]) for p in curve]
            require(epochs == sorted(set(epochs)) and epochs[0] == 0, "Invalid epochs")
            for row, metric in enumerate(("train_accuracy", "heldout_accuracy")):
                accuracy = [float(p[metric]) for p in curve]
                require(all(0 <= value <= 1 for value in accuracy), "Invalid accuracy")
                values = [100 * value for value in accuracy]
                axis = axes[row, seed]
                axis.plot(epochs, values, color=color, linewidth=1.3)
                marker = "x" if summary["status"] == "budget_exhausted" else "o"
                axis.scatter(epochs[-1], values[-1], color=color, marker=marker, s=28)
        for axis in axes[:, seed]:
            axis.axhline(
                data["target"] * 100, color="#555555", linestyle=":", linewidth=0.8
            )
            axis.set_xscale("symlog", linthresh=1, linscale=0.4)
            axis.set_xlim(0, 600)
            axis.set_ylim(-2, 104)
            axis.set_xticks([0, 1, 10, 100, 468.75], ["0", "1", "10", "100", "469"])
            axis.grid(color="#e8e8e8", linewidth=0.6)
            axis.spines[["top", "right"]].set_visible(False)
        axes[1, seed].set_xlabel("Training epochs (log scale above 1)")
    axes[0, 0].set_ylabel("Training exact accuracy (%)")
    axes[1, 0].set_ylabel("Held-out exact accuracy (%)")
    figure.suptitle(
        "Key/value binding: training and held-out trajectories", fontsize=15
    )
    figure.legend(
        [Line2D([0], [0], color=color, linewidth=2) for _, color in ARMS.values()],
        [label for label, _ in ARMS.values()],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.105),
        ncol=4,
    )
    figure.text(
        0.5,
        0.028,
        "Retrospective view of a task chosen after the Peri-only misses. "
        "Every recorded observation; no smoothing.\n"
        "Curves stop at the first 99% held-out crossing or the fixed budget. "
        "Circles: passes; crosses: budget misses; dotted line: 99%.",
        ha="center",
        fontsize=9,
    )
    figure.subplots_adjust(
        bottom=0.24, top=0.88, left=0.07, right=0.99, hspace=0.15, wspace=0.09
    )
    outputs = [RESULTS / f"key-value-curves.{suffix}" for suffix in ("svg", "png")]
    figure.savefig(outputs[0], metadata={"Date": None})
    outputs[0].write_text(
        "\n".join(line.rstrip() for line in outputs[0].read_text().splitlines()) + "\n"
    )
    figure.savefig(outputs[1], dpi=180)
    plt.close(figure)
    record = {
        "runs": len(summaries),
        "observations": len(points),
        "source_sha256": {
            str(path.relative_to(ROOT)): digest(path)
            for path in (Path(__file__), data_path, csv_path)
        },
        "figure_sha256": {path.name: digest(path) for path in outputs},
        "matplotlib": mpl.__version__,
        "command": (
            "MPLCONFIGDIR=/tmp/gpt-core-curves-mpl /usr/bin/python3 "
            "research/early99_core_ablations_reporting/plot_key_value.py"
        ),
        "training_environment_changed": False,
    }
    (RESULTS / "key-value-curves-figure.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
