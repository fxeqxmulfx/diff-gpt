"""Summarize active and completed temporal experiments without hiding plateaus."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from diff_gpt.application.grokking.div97 import write_json
from diff_gpt.application.grokking.div97_report import classify, history, replay_browser
from diff_gpt.application.grokking.recurrence import (
    BATCH,
    DEFAULT_BETA1,
    ROOT,
    TARGET_ACCURACY,
    Config,
)

if TYPE_CHECKING:
    import argparse
    from pathlib import Path

    from diff_gpt.application.ports import BrowserReplay

DIVISION_PERIOD = 48


def summarize(directory: Path) -> dict[str, Any]:
    """Freeze the curve at the atomically recorded status epoch."""
    status = json.loads((directory / "status.json").read_text())
    metadata = json.loads((directory / "metadata.json").read_text())
    rows = [row for row in history(directory) if row["epoch"] <= status["epoch"]]
    if [row["epoch"] for row in rows] != list(range(status["epoch"] + 1)):
        msg = f"missing epoch observations: {directory}"
        raise ValueError(msg)
    result_path = directory / "result.json"
    completed = (
        result_path.exists()
        and json.loads(result_path.read_text())["epoch"] == status["epoch"]
    )
    verification_path = directory / "verification.json"
    verification = (
        json.loads(verification_path.read_text())
        if verification_path.exists()
        else None
    )
    failure_path = directory / "failure.json"
    deferred_path = directory / "deferred.json"
    criterion = classify(rows)
    stable99 = 0
    for row in reversed(rows):
        if min(row["test_accuracy"], row["test_decoded_accuracy"]) < TARGET_ACCURACY:
            break
        stable99 += 1
    if any(
        criterion[key] != status["criterion"][key]
        for key in ("first99", "first90", "plateau_epochs", "rise_epochs", "delayed")
    ):
        msg = f"online criterion and replay differ: {directory}"
        raise ValueError(msg)
    return {
        "directory": str(directory.resolve().relative_to(ROOT)),
        **status,
        "completed": completed,
        "failure": json.loads(failure_path.read_text())
        if failure_path.exists()
        else None,
        "deferred": json.loads(deferred_path.read_text())
        if deferred_path.exists()
        else None,
        "parameters": metadata["parameters"],
        "updates_per_epoch": metadata["updates_per_epoch"],
        "optimizer": metadata["optimizer"],
        "source_sha256": metadata["source_sha256"],
        "restart": metadata.get("restart"),
        "data": metadata["data"],
        "criterion": criterion,
        "consecutive_epochs_at_99": stable99,
        "argmax_criterion": classify(
            [
                {
                    **row,
                    "train_accuracy": row["train_argmax_accuracy"],
                    "test_accuracy": row["test_argmax_accuracy"],
                }
                for row in rows
            ]
        )
        if "test_argmax_accuracy" in rows[0]
        else None,
        "query_criterion": classify(rows, "query_"),
        "verification": verification,
        "public_full_verification": bool(
            verification
            and verification["epoch"] == status["epoch"]
            and verification["public_rows"] == metadata["data"]["test_rows"]
            and round(verification["public_accuracy"] * verification["public_rows"])
            == round(
                status["metrics"]["test"]["decoded_accuracy"]
                * verification["public_rows"]
            )
        ),
        "curve": rows,
    }


def compare_tokenizers(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare paired encoders at identical epoch budgets on identical raw data."""
    groups: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for run in runs:
        settings = asdict(Config(**run["config"]))
        tokenizer = settings.pop("tokenizer")
        settings.pop("epochs")
        key = json.dumps(settings, sort_keys=True)
        groups.setdefault(key, {}).setdefault(tokenizer, []).append(run)
    comparisons = []
    for key, tokenizers in groups.items():
        for lgt in tokenizers.get("lgt", []):
            for mash in tokenizers.get("mash", []):
                if any(
                    lgt["data"][field] != mash["data"][field]
                    for field in ("raw_sha256", "train_rows", "test_rows")
                ):
                    msg = "paired tokenizer experiments have different raw data"
                    raise ValueError(msg)
                if (
                    lgt["parameters"] != mash["parameters"]
                    or lgt["optimizer"] != mash["optimizer"]
                ):
                    msg = "paired tokenizers have different models or optimizers"
                    raise ValueError(msg)
                core = [
                    name
                    for name in lgt["source_sha256"]
                    if name.startswith("src/") and "/grokking/" not in name
                ]
                if any(
                    lgt["source_sha256"][name] != mash["source_sha256"].get(name)
                    for name in core
                ):
                    msg = "paired tokenizers use different core source versions"
                    raise ValueError(msg)
                curves = {
                    "lgt": {int(row["epoch"]): row for row in lgt["curve"]},
                    "mash": {int(row["epoch"]): row for row in mash["curve"]},
                }
                end = min(lgt["epoch"], mash["epoch"])
                epochs = sorted(
                    {100, 250, 500, 750, 900, 1000, 1500, 2000, 3000, 5000, 10000, end}
                )
                rows = []
                for epoch in epochs:
                    if epoch > end:
                        continue
                    rows.append(
                        {
                            "epoch": epoch,
                            **{
                                tokenizer: {
                                    "token_accuracy": curve[epoch]["test_accuracy"],
                                    "decoded_accuracy": curve[epoch][
                                        "test_decoded_accuracy"
                                    ],
                                    "loss": curve[epoch]["test_loss"],
                                }
                                for tokenizer, curve in curves.items()
                            },
                        }
                    )
                comparisons.append(
                    {
                        "settings_except_tokenizer_and_budget": json.loads(key),
                        "lgt_directory": lgt["directory"],
                        "mash_directory": mash["directory"],
                        "identical_raw_data": True,
                        "identical_core_source": True,
                        "matched_through_epoch": end,
                        "measurements": rows,
                    }
                )
    return comparisons


def plot(runs: list[dict[str, Any]], destination: Path) -> None:
    """Export standalone accuracy and loss figures with equal treatment of runs."""
    import matplotlib as mpl  # noqa: PLC0415

    mpl.use("Agg")
    from matplotlib import pyplot as plt  # noqa: PLC0415
    from matplotlib.ticker import PercentFormatter  # noqa: PLC0415

    groups: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        c = run["config"]
        key = (
            "Division: full-period context"
            if c["task"] == "division" and c["context"] >= DIVISION_PERIOD
            else "Division: short context"
            if c["task"] == "division"
            else "Fibonacci: non-unit grid"
            if c.get("grid_domain") or c.get("vocab_size")
            else "Fibonacci: integer grid"
        )
        groups.setdefault(key, []).append(run)
    figure, axes = plt.subplots(
        len(groups), 2, figsize=(13, 3.5 * len(groups)), squeeze=False
    )
    for (title, group), (acc, loss) in zip(sorted(groups.items()), axes, strict=True):
        rates = {
            (run["config"]["learning_rate"], run["config"]["weight_decay"])
            for run in group
        }
        for run in group:
            c = run["config"]
            label = (
                f"k={c['order']}, C={c['context']}, W={c['width']}, "
                f"L={c['layers']}, H={c['heads']}, s={c['seed']}"
            )
            if c.get("tokenizer") or c.get("grid_domain"):
                label += f", {c.get('tokenizer', 'lgt').upper()}"
            if c.get("grid_domain") or c.get("vocab_size"):
                label += (
                    f", V={run['data']['vocab_size']}, D={run['data']['domain'][0]:g}"
                )
            if c.get("readout", "argmax") != "argmax":
                label += f", {c['readout']}"
            if len(rates) > 1:
                label += f", lr={c['learning_rate']:g}, wd={c['weight_decay']:g}"
            if c.get("beta1", DEFAULT_BETA1) != DEFAULT_BETA1:
                label += f", beta1={c['beta1']:g}"
            if c.get("restart_step", 0):
                restart_epoch = c["restart_step"] / run["updates_per_epoch"]
                label += (
                    f", restart at epoch {restart_epoch:g}"
                    f", warmup {c['restart_warmup']} updates"
                )
            if c.get("supervise_from", 0):
                label += f", supervise C>={c['supervise_from']}"
            if c.get("batching", "tail") != "tail":
                label += f", {c['batching']} batches"
            if c.get("batch_size", BATCH) != BATCH:
                label += f", batch={c['batch_size']}"
            if c.get("micro_batch_size", 0):
                label += f", micro={c['micro_batch_size']}"
            if run.get("deferred"):
                label += ", preserved checkpoint"
            rows = run["curve"]
            epochs = [row["epoch"] for row in rows]
            (line,) = acc.plot(
                epochs, [row["test_accuracy"] for row in rows], label=label
            )
            color = line.get_color()
            acc.plot(
                epochs,
                [row["train_accuracy"] for row in rows],
                color=color,
                alpha=0.2,
                linewidth=0.8,
            )
            if "test_argmax_accuracy" in rows[0]:
                acc.plot(
                    epochs,
                    [row["test_argmax_accuracy"] for row in rows],
                    color=color,
                    linestyle="--",
                    alpha=0.65,
                    linewidth=0.8,
                    label="same weights, native argmax",
                )
            loss.semilogy(
                epochs,
                [max(row["test_loss"], 1e-8) for row in rows],
                color=color,
                label=label,
            )
            loss.semilogy(
                epochs,
                [max(row["train_loss"], 1e-8) for row in rows],
                color=color,
                alpha=0.2,
                linewidth=0.8,
            )
        acc.axhline(0.99, color="#888888", linewidth=0.8, linestyle=":")
        acc.set(title=title, ylabel="Exact token accuracy", ylim=(-0.02, 1.03))
        acc.yaxis.set_major_formatter(PercentFormatter(1))
        loss.set(title="Token cross-entropy before readout", ylabel="Loss (nats)")
        for axis in (acc, loss):
            axis.set_xlabel("Epoch (complete training pass)")
            axis.grid(alpha=0.2)
            axis.legend(fontsize=7, loc="best")
    figure.suptitle(
        "DiffGPT recurrences mod 97 · differences down rows · disjoint whole orbits\n"
        "ScheduleFreeAdamW; solid: test, dashed: native argmax, faint: train",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    for suffix in (".png", ".svg"):
        figure.savefig(destination.with_suffix(suffix), dpi=160)
    plt.close(figure)


def report(args: argparse.Namespace, *, node: BrowserReplay) -> None:
    """Summarize the runs and their tokenizer pairs, then write the report."""
    directories = sorted(
        {path.parent for root in args.roots for path in root.glob("*/status.json")}
    )
    runs = [summarize(directory) for directory in directories]
    if args.confirmed_only:
        runs = [
            run
            for run in runs
            if run["public_full_verification"]
            and run["consecutive_epochs_at_99"] >= run["config"]["after99"] + 1
        ]
    comparisons = compare_tokenizers(runs)
    report = {"runs": runs, "tokenizer_comparisons": comparisons}
    if args.browser_module:
        report["browser_criterion_replay"] = replay_browser(
            [run for run in runs if run["completed"]], args.browser_module, node=node
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    write_json(
        args.output.with_name(args.output.stem + "-tokenizers.json"), comparisons
    )
    if args.plot:
        plot(runs, args.output)
    print(
        json.dumps(
            {
                "runs": len(runs),
                "completed": sum(run["completed"] for run in runs),
                "tokenizer_pairs": len(comparisons),
                "public_full_verified": sum(
                    run["public_full_verification"] for run in runs
                ),
                "output": str(args.output),
            }
        ),
        flush=True,
    )
