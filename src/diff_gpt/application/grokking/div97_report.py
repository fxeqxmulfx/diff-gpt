"""
Summarize complete DiffGPT runs and optionally replay the browser's criterion.

Use --plot with matplotlib installed to export standalone scientific figures.
The summary keeps censored runs and distinguishes averaged and query weights.
"""

from __future__ import annotations

import csv
import hashlib
import json
from typing import TYPE_CHECKING, Any

from diff_gpt.application.grokking.div97 import (
    ROOT,
    TARGET_ACCURACY,
    Criterion,
)

if TYPE_CHECKING:
    import argparse
    from collections.abc import Sequence
    from pathlib import Path

    from diff_gpt.application.ports import BrowserReplay


def history(directory: Path) -> list[dict[str, float]]:
    """Read numeric observations, leaving absent query columns absent."""
    with (directory / "history.csv").open(newline="") as handle:
        return [
            {key: float(value) for key, value in row.items() if value}
            for row in csv.DictReader(handle)
        ]


def classify(rows: Sequence[dict[str, float]], prefix: str = "") -> dict[str, Any]:
    """Replay all epochs and retain the exact low-test plateau interval."""
    rule = Criterion()
    longest = (None, None)
    first_train99 = None
    train_key, test_key = f"{prefix}train_accuracy", f"{prefix}test_accuracy"
    for row in rows:
        epoch = int(row["epoch"])
        if first_train99 is None and row[train_key] >= TARGET_ACCURACY:
            first_train99 = epoch
        previous = rule.plateau_epochs
        rule.observe(epoch, row[train_key], row[test_key])
        if rule.plateau_epochs > previous:
            longest = (rule.plateau_start, epoch)
    after = (
        [row[test_key] for row in rows if row["epoch"] >= rule.first99]
        if rule.first99 is not None
        else []
    )
    return {
        "first_train99": first_train99,
        "first99": rule.first99,
        "first90": rule.first90,
        "plateau_epochs": rule.plateau_epochs,
        "plateau_start": longest[0],
        "plateau_end": longest[1],
        "rise_epochs": rule.rise_epochs,
        "delayed": rule.delayed,
        "after99_observations": len(after),
        "minimum_test_after99": min(after) if after else None,
        "below99_after_first_crossing": sum(v < TARGET_ACCURACY for v in after),
    }


def summarize(directory: Path) -> dict[str, Any]:
    """Verify the complete curve against the online classifier and saved metrics."""
    result = json.loads((directory / "result.json").read_text())
    metadata = json.loads((directory / "metadata.json").read_text())
    verification = json.loads((directory / "verification.json").read_text())
    rows = history(directory)
    if [row["epoch"] for row in rows] != list(range(result["epoch"] + 1)):
        msg = f"missing epoch observation in {directory}"
        raise ValueError(msg)
    criterion = classify(rows)
    for key in ("first99", "first90", "plateau_epochs", "rise_epochs", "delayed"):
        if criterion[key] != result["criterion"][key]:
            msg = f"online and replayed {key} disagree at {directory}"
            raise ValueError(msg)
    if verification["metrics"] != result["metrics"]:
        msg = f"checkpoint replay does not exactly match saved metrics: {directory}"
        raise ValueError(msg)
    return {
        "directory": str(directory.resolve().relative_to(ROOT)),
        "config": result["config"],
        "parameters": metadata["parameters"],
        "epoch": result["epoch"],
        "metrics": result["metrics"],
        "criterion": criterion,
        "query_criterion": classify(rows, "query_")
        if "query_test_accuracy" in rows[0]
        else None,
        "checkpoint_replayed": True,
        "query_checkpoint_replayed": "query_metrics" in verification,
        "public_diffgpt_test": verification.get("diffgpt_predict_many"),
        "sha256": {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in (
                "checkpoint.pt",
                "history.csv",
                "metadata.json",
                "verification.json",
            )
        },
        "source_sha256": metadata["source_sha256"],
    }


def replay_browser(
    summaries: list[dict[str, Any]],
    module: Path,
    *,
    node: BrowserReplay,
) -> dict[str, Any]:
    """Compare both trajectories to the actual browser implementation in Node."""
    entries = []
    expected = []
    for item in summaries:
        rows = history(ROOT / item["directory"])
        for prefix, key in (("", "criterion"), ("query_", "query_criterion")):
            if item[key] is None:
                continue
            name = item["directory"] + "/" + (prefix or "evaluation")
            entries.append(
                {
                    "name": name,
                    "rows": [
                        {
                            "epoch": row["epoch"],
                            "trainAccuracy": row[f"{prefix}train_accuracy"],
                            "heldoutAccuracy": row[f"{prefix}test_accuracy"],
                        }
                        for row in rows
                    ],
                }
            )
            expected.append(
                {
                    "name": name,
                    **{
                        field: item[key][field]
                        for field in (
                            "first99",
                            "first90",
                            "plateau_epochs",
                            "rise_epochs",
                            "delayed",
                        )
                    },
                }
            )
    actual = node(entries, module)
    if actual != expected:
        msg = "browser ShortGrokking replay disagrees with the Python classifier"
        raise ValueError(msg)
    return {
        "module": str(module.resolve()),
        "sha256": hashlib.sha256(module.read_bytes()).hexdigest(),
        "trajectories": len(actual),
        "matched": True,
    }


def plot_runs(summaries: list[dict[str, Any]], destination: Path) -> None:
    """Export accuracy and loss for the confirmation runs, including the control."""
    import matplotlib as mpl  # noqa: PLC0415

    mpl.use("Agg")
    from matplotlib import pyplot as plt  # noqa: PLC0415
    from matplotlib.ticker import PercentFormatter  # noqa: PLC0415

    figure, axes = plt.subplots(
        len(summaries), 2, figsize=(12, 3 * len(summaries)), squeeze=False
    )
    for item, (accuracy_axis, loss_axis) in zip(summaries, axes, strict=True):
        rows = history(ROOT / item["directory"])
        epochs = [row["epoch"] for row in rows]
        config = item["config"]
        title = f"Seed {config['seed']} · {config['labels']}"
        for split, color in (("train", "#4976bc"), ("test", "#c04f29")):
            accuracy_axis.plot(
                epochs, [r[f"{split}_accuracy"] for r in rows], color=color, label=split
            )
            loss_axis.semilogy(
                epochs,
                [max(r[f"{split}_loss"], 1e-8) for r in rows],
                color=color,
                label=split,
            )
        if "query_test_accuracy" in rows[0]:
            accuracy_axis.plot(
                epochs,
                [r["query_test_accuracy"] for r in rows],
                color="#c04f29",
                alpha=0.35,
                linewidth=0.8,
                label="test (query)",
            )
        criterion = item["criterion"]
        if criterion["plateau_start"] is not None:
            for axis in (accuracy_axis, loss_axis):
                axis.axvspan(
                    criterion["plateau_start"],
                    criterion["plateau_end"],
                    color="#87949c",
                    alpha=0.18,
                )
        if criterion["first99"] is not None:
            accuracy_axis.axvline(
                criterion["first99"], color="#777777", linewidth=0.8, linestyle=":"
            )
            title += f" · first test ≥99%: {criterion['first99']}"
        accuracy_axis.set(title=title, ylabel="Accuracy", ylim=(-0.03, 1.03))
        accuracy_axis.yaxis.set_major_formatter(PercentFormatter(1))
        loss_axis.set(title="Full-split cross-entropy", ylabel="Loss (nats)")
        for axis in (accuracy_axis, loss_axis):
            axis.set_xlabel("Epoch (10 optimizer updates)")
            axis.grid(alpha=0.2)
            axis.legend(loc="best", fontsize=8)
    figure.suptitle(
        "DiffGPT k=0 · modular division mod 97 · W128 / 1 block / 4 heads\n"
        "ScheduleFreeAdamW: lr=0.01, weight decay=1; shaded: memorization plateau",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    for suffix in (".svg", ".png"):
        figure.savefig(destination.with_suffix(suffix), dpi=170)
    plt.close(figure)


def report(args: argparse.Namespace, *, node: BrowserReplay) -> None:
    """Summarize the runs, replay them in the browser module if asked, and plot."""
    directories = sorted(
        {path.parent for root in args.runs for path in root.glob("*/verification.json")}
    )
    summaries = [summarize(directory) for directory in directories]
    report = {"runs": summaries}
    if args.browser_module:
        report["browser_criterion_replay"] = replay_browser(
            summaries, args.browser_module, node=node
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.plot:
        plot_runs(
            [item for item in summaries if "confirm" in item["directory"]], args.output
        )
    print(
        json.dumps(
            {
                "runs": len(summaries),
                "output": str(args.output),
                "delayed": sum(item["criterion"]["delayed"] for item in summaries),
                "browser_replay": report.get("browser_criterion_replay"),
            },
            indent=2,
        )
    )
