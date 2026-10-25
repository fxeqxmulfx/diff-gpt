"""
Summarize fast-first runs, target costs, seed spread and fair selectors.

Empirical weather-table ceilings are diagnostics, not population Bayes floors.
Synthetic heldout skill uses exact expected token CE. Unknown-floor datasets
are selected on validation alone. A validation-best checkpoint selected from
the full trajectory has the same training FLOP budget as the final checkpoint;
the first-patience early-stop checkpoint is also reported with its smaller cost.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.grokking.div97 import ROOT, Criterion, write_json
from benchmarks.grokking.series import skill
from benchmarks.grokking.series_data import FORMULAS, WEATHER, _read_weather
from benchmarks.grokking.series_oracle import conditional_reference
from benchmarks.grokking.series_shortcuts import formula_shortcuts
from benchmarks.grokking.series_suite import FAST_EPOCHS, quality
from benchmarks.grokking.series_weather_audit import formula_comparators
from diff_gpt.domain.encoder_decoder import TokenClippingWarning, encode

CONFIRMATION_SEEDS = 3
FORMULA_OPERANDS = 2
GAP_PHASE_EPOCHS = 25
GAP_GROWTH_NATS = 0.01
SHORTCUT_SKILL_TARGET = 0.9


def _reference_skill(loss: float, reference: dict[str, Any]) -> float | None:
    if reference.get("input_independent_control"):
        return None
    return skill(loss, reference["baseline_loss"], reference["floor_loss"])


def audit_references(
    directory: Path, metadata: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Reevaluate legacy formula curves against the full recorded equation table."""
    config = metadata["config"]["data"]
    references = json.loads(json.dumps(metadata["references"]))
    if not config["dataset"].startswith("weather_formula"):
        return references, None
    path = Path(config["source"]) if config.get("source") else WEATHER
    frame, provenance = _read_weather(path)
    if provenance["source_sha256"] != metadata["data"]["source_sha256"]:
        msg = "cannot audit formula references after a source-data change"
        raise ValueError(msg)
    names = FORMULAS[config["formula"]]
    values = frame.loc[:, list(names)].to_numpy()
    recorded = metadata["data"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", TokenClippingWarning)
        _, _, tokens = encode(
            (values - np.asarray(recorded["mean"])) / np.asarray(recorded["std"]),
            config["vocab_size"],
            np.asarray(recorded["domain"]),
            0,
            tokenizer=config["tokenizer"],
        )
    if config["random_labels"] and "target_label_sha256" in recorded:
        tokens[:, -1] = np.random.default_rng(config["seed"] ^ 0x1ABE1).permutation(
            tokens[:, -1]
        )
    changes = {}
    with np.load(directory / "frozen_splits.npz", allow_pickle=False) as frozen:
        for name, columns in references.items():
            contexts, targets = frozen[f"{name}__x"], frozen[f"{name}__y"][:, -1]
            if contexts.shape[1] != FORMULA_OPERANDS:
                bits = recorded["bits_per_input"]
                contexts = contexts.reshape(len(contexts), 2, bits) @ (
                    2 ** np.arange(bits - 1, -1, -1)
                )
            if config["random_labels"]:
                # Legacy controls shuffled within each frozen split. Their
                # known permutation prior is still independent of each input.
                population = (
                    tokens[:, -1] if "target_label_sha256" in recorded else targets
                )
                prior = np.bincount(population, minlength=config["vocab_size"]) / len(
                    population
                )
                corrected = {
                    "floor_loss": float(
                        -np.sum(prior * np.log(np.maximum(prior, 1e-300)))
                    ),
                    "ceiling_accuracy": float(prior.max()),
                    "oracle_realized_loss": float(-np.log(prior[targets]).mean()),
                    "oracle_realized_accuracy": float(
                        (targets == prior.argmax()).mean()
                    ),
                }
            else:
                corrected = conditional_reference(
                    tokens[:, :2],
                    tokens[:, -1],
                    contexts,
                    targets,
                    config["vocab_size"],
                )
            column = config["formula"]
            changes[name] = {
                "recorded_floor_loss": columns[column]["floor_loss"],
                "full_population_reference": corrected,
            }
            columns[column].update(corrected)
            if config["random_labels"]:
                columns[column]["input_independent_control"] = True
    audit = {
        "method": (
            "full-table conditional entropy; known permutation marginal "
            "for random-label controls"
        ),
        "source_sha256": provenance["source_sha256"],
        "changes": changes,
        "original_curves_preserved": True,
        "input_rule_skill": (
            "undefined for input-independent random labels; "
            "the population marginal already attains their Bayes floor"
            if config["random_labels"]
            else "unigram-relative conditional CE skill"
        ),
    }
    write_json(directory / "reference-audit.json", audit)
    return references, audit


def audit_curve(
    directory: Path, references: dict[str, Any], budget: int
) -> dict[str, Any]:
    """Preserve measured losses and recompute skill and the plateau classifier."""
    with (directory / "history.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = [row for row in reader if int(row["epoch"]) <= budget]
    column = next(iter(references["train"]))
    criteria = {}
    for prefix, mode in (("", "averaged"), ("query_", "query")):
        criterion = Criterion()
        for row in rows:
            values = {}
            for name, columns in references.items():
                ref = columns[column]
                value = _reference_skill(
                    float(row[f"{prefix}{name}:{column}:loss"]),
                    ref,
                )
                row[f"{prefix}{name}:{column}:skill"] = value
                row[f"{prefix}{name}_skill"] = value
                values[name] = value
            if all(values[name] is not None for name in ("train", "test")):
                criterion.observe(
                    int(row["epoch"]),
                    min(1.0, max(0.0, values["train"])),
                    min(1.0, max(0.0, values["test"])),
                )
        criteria[mode] = {column: asdict(criterion)}
    with (directory / "history-audited.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return criteria


def signature(config: dict[str, Any], source: dict[str, str] | None = None) -> str:
    """Group identical scientific settings across initialization/split seeds."""
    values = json.loads(json.dumps(config))
    values.pop("epochs", None)
    values["data"].pop("seed", None)
    if source is not None:
        values["source_sha256"] = source
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()[:12]


def spread(values: list[float]) -> dict[str, float | int]:
    """Report all seed values with mean, sample deviation and conservative range."""
    return {
        "seeds": len(values),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "range": float(np.ptp(values)),
        "min": min(values),
        "max": max(values),
    }


def _column_costs(
    directory: Path,
    metadata: dict[str, Any],
    *,
    anchor: bool,
    budget: int,
    references: dict[str, Any],
    formula: bool,
) -> dict[str, Any]:
    with (directory / "history.csv").open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if int(row["epoch"]) <= budget]
    result = {}
    for column in metadata["references"]["test"]:
        result[column] = {}
        for mode in ("", "query_"):
            key = f"{mode}test:{column}:{'accuracy' if anchor else 'skill'}"
            result[column][mode or "averaged"] = {}
            for target in (0.9, 0.99):
                ref = references["test"][column]
                crossing = (
                    next(
                        (
                            row
                            for row in rows
                            if (
                                _reference_skill(
                                    float(row[f"{mode}test:{column}:loss"]),
                                    ref,
                                )
                                or -np.inf
                            )
                            >= target
                        ),
                        None,
                    )
                    if formula
                    else next(
                        (row for row in rows if row[key] and float(row[key]) >= target),
                        None,
                    )
                )
                result[column][mode or "averaged"][str(target)] = (
                    {
                        name: float(crossing[name])
                        for name in ("epoch", "seconds", "training_flops")
                    }
                    if crossing
                    else None
                )
    return result


def summarize_run(directory: Path) -> dict[str, Any]:
    """Record final metrics and both validation-selection comparators."""
    result = json.loads((directory / "result.json").read_text())
    metadata = json.loads((directory / "metadata.json").read_text())
    references, audit = audit_references(directory, metadata)
    shortcuts = formula_shortcuts(directory, metadata)
    for entry in shortcuts.get("operands", {}).values():
        for name, measured_shortcut in entry["training_fitted_conditional"].items():
            ref = next(iter(references[name].values()))
            measured_shortcut["skill"] = _reference_skill(
                measured_shortcut["loss"], ref
            )
    anchor = result["config"]["data"]["dataset"] in {"div97", "fibonacci"}
    measured = quality(directory, budget=result["epoch"])
    costs = _column_costs(
        directory,
        metadata,
        anchor=anchor,
        budget=result["epoch"],
        references=references,
        formula=audit is not None,
    )
    if audit is not None:
        for metrics in (result["metrics"], result["query_metrics"]):
            for name, split in metrics.items():
                for column, values in split["columns"].items():
                    ref = references[name][column]
                    values["skill"] = _reference_skill(values["loss"], ref)
                split["skill"] = next(iter(split["columns"].values()))["skill"]
        measured["crossings"] = costs[result["config"]["data"]["formula"]]["averaged"]
        result["criteria"] = audit_curve(directory, references, result["epoch"])
    selector = result["validation_selection"]
    with (directory / "history.csv").open(newline="") as handle:
        available = list(csv.DictReader(handle))
    history_end = max(int(row["epoch"]) for row in available)
    rows = [row for row in available if int(row["epoch"]) <= result["epoch"]]
    gap_growth = {}
    early_minima = {}
    for column in result["metrics"]["test"]["columns"]:
        gaps = np.array(
            [
                float(row[f"test:{column}:loss"]) - float(row[f"train:{column}:loss"])
                for row in rows
            ]
        )
        gap_growth[column] = bool(
            len(gaps) > GAP_PHASE_EPOCHS
            and np.any(
                gaps[GAP_PHASE_EPOCHS:] - gaps[:-GAP_PHASE_EPOCHS] > GAP_GROWTH_NATS
            )
        )
        stop = selector["early_stop_epoch"]
        early_minima[column] = (
            min(
                float(row[f"test:{column}:loss"])
                for row in rows
                if int(row["epoch"]) <= stop
            )
            if stop is not None
            else None
        )
    comparisons = {}
    if selector["best_metrics"] is not None:
        comparisons["full_budget_validation_best"] = {
            "selected_epoch": selector["best_epoch"],
            "training_flops_consumed": result["training_flops"],
            "metrics": selector["best_metrics"],
            "selection_uses_test": False,
        }
    if selector["early_stop_metrics"] is not None:
        comparisons["first_patience_early_stop"] = {
            "stop_epoch": selector["early_stop_epoch"],
            "selected_epoch": selector["early_stop_selected_epoch"],
            "training_flops_consumed": selector["early_stop_epoch"]
            * metadata["training_flops_per_epoch"],
            "metrics": selector["early_stop_metrics"],
            "selection_uses_test": False,
        }
    return {
        "directory": str(directory),
        "signature": signature(result["config"], metadata["source_sha256"]),
        "config": result["config"],
        "epoch": result["epoch"],
        "history_end_epoch": history_end,
        "unfinalized_extension": history_end > result["epoch"],
        "seconds": result["seconds"],
        "training_flops": result["training_flops"],
        "quality": measured,
        "per_column_target_costs": costs,
        "references": references,
        "reference_audit": audit,
        "shortcut_baselines": shortcuts,
        "same_row_formula_comparators": formula_comparators(directory, metadata),
        "metrics": result["metrics"],
        "query_metrics": result["query_metrics"],
        "criteria": result["criteria"],
        "comparisons": comparisons,
        "gap_growth_phase": gap_growth,
        "early_window_test_minima_diagnostic": early_minima,
        "forecast": result.get("forecast"),
        "rollouts": {
            path.stem: json.loads(path.read_text())
            for path in directory.glob("rollout-h*-w*.json")
        },
        "data": metadata["data"],
        "normalization": metadata["normalization"],
        "source_sha256": metadata["source_sha256"],
        "verified": (directory / "verification.json").exists(),
    }


def summarize_group(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep partial confirmations visible and require three seeds for conclusions."""
    # Prefer each seed's longest completed trajectory when a screen was extended.
    seeds = {}
    for run in sorted(runs, key=lambda value: value["epoch"]):
        seeds[run["config"]["data"]["seed"]] = run
    runs = [seeds[seed] for seed in sorted(seeds)]
    result = {
        "signature": runs[0]["signature"],
        "config": runs[0]["config"],
        "seed_ids": sorted(seeds),
        "confirmed_on_three_seeds": len(seeds) >= CONFIRMATION_SEEDS,
        "all_verified": all(run["verified"] for run in runs),
        "final_test_loss": spread([run["metrics"]["test"]["loss"] for run in runs]),
        "final_test_accuracy": spread(
            [run["metrics"]["test"]["accuracy"] for run in runs]
        ),
        "training_flops": spread([run["training_flops"] for run in runs]),
        "optimized_training_rows": spread(
            [run["data"]["split_rows"]["train"] for run in runs]
        ),
        "columns": {},
    }
    if not runs[0]["quality"]["known_floor"]:
        crossings = [
            run["quality"]["crossings"][str(run["quality"]["targets"][0])]
            for run in runs
        ]
        result["validation_target"] = {
            "definition": "10% validation CE improvement over training-fitted unigram",
            "reached_seeds": sum(item is not None for item in crossings),
            "confirmed_on_three_seeds": len(seeds) >= CONFIRMATION_SEEDS
            and all(item is not None for item in crossings),
            "confirmed_fast_on_three_seeds": len(seeds) >= CONFIRMATION_SEEDS
            and all(
                item is not None and item["epoch"] <= FAST_EPOCHS for item in crossings
            ),
            "per_seed_crossings": dict(zip(sorted(seeds), crossings, strict=True)),
        }
    rollout_key = next(
        (key for key in runs[0]["rollouts"] if key.endswith("-w8")), None
    )
    if runs[0]["shortcut_baselines"]:
        result["formula_shortcut_controls"] = {}
        for operand in runs[0]["shortcut_baselines"]["operands"]:
            measured = [
                run["shortcut_baselines"]["operands"][operand][
                    "training_fitted_conditional"
                ]["test"]
                for run in runs
            ]
            result["formula_shortcut_controls"][operand] = {
                "test_loss": spread([item["loss"] for item in measured]),
                "model_final_minus_shortcut_loss": spread(
                    [
                        run["metrics"]["test"]["loss"] - item["loss"]
                        for run, item in zip(runs, measured, strict=True)
                    ]
                ),
                "single_operand_reaches_0_9_on_all_three_seeds": len(seeds)
                >= CONFIRMATION_SEEDS
                and all(
                    item["skill"] is not None and item["skill"] >= SHORTCUT_SKILL_TARGET
                    for item in measured
                ),
                "single_operand_target_pass_counts": {
                    str(target): sum(
                        item["skill"] is not None and item["skill"] >= target
                        for item in measured
                    )
                    for target in (0.9, 0.99)
                },
            }
    if rollout_key and all(rollout_key in run["rollouts"] for run in runs):
        rollouts = [run["rollouts"][rollout_key] for run in runs]
        names = set.intersection(*(set(item["scores"]) for item in rollouts))
        result["free_running_forecasts"] = {
            "horizon_rows": rollouts[0]["horizon_rows"],
            "model_context_rows": runs[0]["config"]["data"]["context"]
            + runs[0]["config"]["data"]["order"],
            "seasonal_history_rows": runs[0]["data"]["seasonal_period"],
            "origins_per_seed": [len(item["selected_origins"]) for item in rollouts],
            "train_normalized": {
                name: {
                    metric: spread(
                        [
                            item["scores"][name]["train_normalized"][metric]
                            for item in rollouts
                        ]
                    )
                    for metric in ("mse", "mae")
                }
                for name in sorted(names)
            },
        }
    for column in runs[0]["metrics"]["test"]["columns"]:
        final = [run["metrics"]["test"]["columns"][column]["loss"] for run in runs]
        early = [
            run["comparisons"]
            .get("first_patience_early_stop", {})
            .get("metrics", {})
            .get("test", {})
            .get("columns", {})
            .get(column, {})
            .get("loss")
            for run in runs
        ]
        entry = {
            "final_test_loss": spread(final),
            "delayed_count": sum(
                run["criteria"]["averaged"][column]["delayed"] for run in runs
            ),
            "target_costs": {},
        }
        for target in ("0.9", "0.99"):
            costs = [
                run["per_column_target_costs"][column]["averaged"][target]
                for run in runs
            ]
            entry["target_costs"][target] = {
                "reached_seeds": sum(value is not None for value in costs),
                "confirmed_all_three_seeds": len(seeds) >= CONFIRMATION_SEEDS
                and all(value is not None for value in costs),
                "fast_on_all_three_seeds": len(seeds) >= CONFIRMATION_SEEDS
                and all(
                    value is not None and value["epoch"] <= FAST_EPOCHS
                    for value in costs
                ),
                "epochs": spread(
                    [value["epoch"] for value in costs if value is not None]
                )
                if any(value is not None for value in costs)
                else None,
                "training_flops": spread(
                    [value["training_flops"] for value in costs if value is not None]
                )
                if any(value is not None for value in costs)
                else None,
                "seconds": spread(
                    [value["seconds"] for value in costs if value is not None]
                )
                if any(value is not None for value in costs)
                else None,
            }
        if all(value is not None for value in early):
            early_stats = spread(early)
            difference = early_stats["mean"] - entry["final_test_loss"]["mean"]
            minimum_stats = spread(
                [run["early_window_test_minima_diagnostic"][column] for run in runs]
            )
            late_gain = minimum_stats["mean"] - entry["final_test_loss"]["mean"]
            entry.update(
                {
                    "early_stop_test_loss": early_stats,
                    "late_improvement": difference,
                    "late_improvement_exceeds_seed_range": len(seeds)
                    >= CONFIRMATION_SEEDS
                    and difference
                    > max(early_stats["range"], entry["final_test_loss"]["range"]),
                    "gap_growth_on_all_seeds": all(
                        run["gap_growth_phase"][column] for run in runs
                    ),
                    "early_window_test_minimum": minimum_stats,
                    "improvement_over_early_window_minimum": late_gain,
                }
            )
            entry["late_generalization_confirmed"] = (
                len(seeds) >= CONFIRMATION_SEEDS
                and late_gain
                > max(minimum_stats["range"], entry["final_test_loss"]["range"])
                and entry["gap_growth_on_all_seeds"]
            )
        result["columns"][column] = entry
    return result


def legacy_anchors(root: Path) -> list[dict[str, Any]]:
    """Include fresh original-runner anchors without reclassifying old runs."""
    values = []
    for family in ("anchors-div97", "anchors-fibonacci"):
        for path in sorted((root / family).glob("*/result.json")):
            result = json.loads(path.read_text())
            metadata = json.loads((path.parent / "metadata.json").read_text())
            config = result["config"]
            context = config.get("context", 2)
            block = context + 1
            count = metadata.get(
                "train_rows", metadata.get("data", {}).get("train_rows")
            )
            per_epoch = (
                (
                    6 * metadata["parameters"]
                    + 12 * config["layers"] * config["width"] * block
                )
                * context
                * count
            )
            with (path.parent / "history.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            crossings = {}
            for target, key in (("0.9", "first90"), ("0.99", "first99")):
                epoch = result["criterion"][key]
                row = next(
                    (value for value in rows if int(value["epoch"]) == epoch), None
                )
                crossings[target] = (
                    {
                        "epoch": epoch,
                        "training_flops": epoch * per_epoch,
                        "seconds": float(row["seconds"]),
                    }
                    if row
                    else None
                )
            values.append(
                {
                    "directory": str(path.parent),
                    "family": family,
                    "config": config,
                    "metrics": result["metrics"],
                    "criterion": result["criterion"],
                    "target_costs": crossings,
                    "normalization": metadata["normalization"],
                    "source_sha256": metadata["source_sha256"],
                    "verified": (path.parent / "verification.json").exists(),
                }
            )
    return values


def anchor_comparison(
    root: Path, current: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Compare recorded historical and fresh anchors at their 99% crossings."""
    lookup = {
        (item["family"], item["config"]["seed"]): item
        for item in current
        if not item["config"].get("random_labels")
        and item["config"].get("labels") != "random"
    }
    result = []
    for family, directory in (
        ("anchors-div97", "grokking-div97-confirm"),
        ("anchors-fibonacci", "grokking-div97-rows-field"),
    ):
        for path in sorted((root.parent / directory).glob("*/result.json")):
            historical = json.loads(path.read_text())
            epoch = historical["criterion"]["first99"]
            config = historical["config"]
            fresh = lookup.get((family, config["seed"]))
            if epoch is None or fresh is None:
                continue
            metadata = json.loads((path.parent / "metadata.json").read_text())
            context = config.get("context", 2)
            count = metadata.get(
                "train_rows", metadata.get("data", {}).get("train_rows")
            )
            per_epoch = (
                (
                    6 * metadata["parameters"]
                    + 12 * config["layers"] * config["width"] * (context + 1)
                )
                * context
                * count
            )
            with (path.parent / "history.csv").open(newline="") as handle:
                row = next(
                    row for row in csv.DictReader(handle) if int(row["epoch"]) == epoch
                )
            before = {
                "epoch": epoch,
                "training_flops": per_epoch * epoch,
                "seconds": float(row["seconds"]),
            }
            after = fresh["target_costs"]["0.99"]
            result.append(
                {
                    "family": family,
                    "seed": config["seed"],
                    "historical_directory": str(path.parent),
                    "current_directory": fresh["directory"],
                    "historical": before,
                    "current": after,
                    "current_to_historical_ratios": {
                        key: after[key] / before[key] for key in before
                    }
                    if after is not None
                    else None,
                    "historical_source_sha256": metadata["source_sha256"],
                    "current_source_sha256": fresh["source_sha256"],
                    "interpretation": (
                        "normalization and optimizer implementation both changed; "
                        "this comparison does not isolate either"
                    ),
                }
            )
    return result


def report(
    output: Path, *, root: Path | None = None, include: list[Path] | None = None
) -> dict[str, Any]:
    """Write a machine-readable report with explicit population limitations."""
    roots = [output, *(include or [])]
    paths = sorted(
        {path for directory in roots for path in directory.glob("*/*/result.json")}
    )
    historical, runs = [], []
    for path in paths:
        measured = summarize_run(path.parent)
        superseded = path.parent.parent.parent.name == "fast-first" and measured[
            "config"
        ]["data"]["dataset"] in {"div97", "fibonacci"}
        if superseded or measured["unfinalized_extension"]:
            historical.append(measured)
        else:
            runs.append(measured)
    by_signature = {}
    for run in runs:
        by_signature.setdefault(run["signature"], []).append(run)
    value = {
        "priority": "fast generalization first; delayed only after failed fast screens",
        "definitions": {
            "target_cost": (
                "first heldout target crossing; training FLOPs exclude evaluation"
            ),
            "formula_floor": (
                "full source-table conditional entropy over each frozen split's "
                "inputs; diagnostic, "
                "not a population Bayes floor"
            ),
            "synthetic_floor": (
                "exact transition entropy; heldout skill uses expected CE"
            ),
            "unknown_floor_target": (
                "10% lower validation CE than a training-fitted unigram"
            ),
            "full_budget_baseline": (
                "best validation checkpoint after identical total training FLOPs"
            ),
            "early_stop_baseline": (
                "first-patience checkpoint, with its own smaller consumed FLOPs"
            ),
            "coverage": (
                "evaluate only explicitly reported training-vocabulary populations"
            ),
            "forecast_protocol": (
                "internal chronological holdouts; not official M4 challenge test scores"
            ),
        },
        "anchors": legacy_anchors(root or output.parent),
        "runs": runs,
        "historical_superseded_runs": historical,
        "groups": [summarize_group(group) for group in by_signature.values()],
        "selections": {
            f"{directory.name}/{path.stem}": json.loads(path.read_text())
            for directory in roots
            for path in sorted(directory.glob("selection-*.json"))
        },
        "stages_complete": [
            f"{directory.name}/{path.stem}"
            for directory in roots
            for path in directory.glob("stage-*-complete.json")
            if json.loads(path.read_text()).get("complete")
        ],
        "stage_errors": {
            f"{directory.name}/{path.stem}": json.loads(path.read_text())
            for directory in roots
            for path in directory.glob("stage-*-error.json")
            if directory.name != "fast-first"
        },
        "historical_stage_errors": {
            f"{directory.name}/{path.stem}": json.loads(path.read_text())
            for directory in roots
            for path in directory.glob("stage-*-error.json")
            if directory.name == "fast-first"
        },
    }
    value["historical_anchor_comparison"] = anchor_comparison(
        root or output.parent, value["anchors"]
    )
    ranking = []
    validation_ranking = []
    for group in value["groups"]:
        if not group["confirmed_on_three_seeds"]:
            continue
        config = group["config"]
        data = config["data"]
        practical = group.get("validation_target")
        if practical and practical["confirmed_on_three_seeds"]:
            costs = list(practical["per_seed_crossings"].values())
            validation_ranking.append(
                {
                    "dataset": data["dataset"],
                    "order": data["order"],
                    "context": data["context"],
                    "shuffled_time": data["shuffled_time"],
                    "training_rows_mean": group["optimized_training_rows"]["mean"],
                    "width": config["width"],
                    "heads": config["heads"],
                    "target_metric": "validation_ce_over_training_unigram",
                    "target": 0.9,
                    "target_direction": "lower",
                    "optimizer_state": "averaged",
                    "fast_on_all_three_seeds": practical[
                        "confirmed_fast_on_three_seeds"
                    ],
                    "mean_epoch": float(np.mean([item["epoch"] for item in costs])),
                    "mean_training_flops": float(
                        np.mean([item["training_flops"] for item in costs])
                    ),
                    "mean_seconds": float(np.mean([item["seconds"] for item in costs])),
                    "signature": group["signature"],
                }
            )
        for column, measured in group["columns"].items():
            for target, costs in measured["target_costs"].items():
                if not costs["confirmed_all_three_seeds"]:
                    continue
                ranking.append(
                    {
                        "dataset": data["dataset"],
                        "column": column,
                        "order": data["order"],
                        "noise": data["noise"],
                        "width": config["width"],
                        "heads": config["heads"],
                        "target": target,
                        "target_metric": "heldout_accuracy"
                        if data["dataset"] in {"div97", "fibonacci"}
                        else "expected_token_ce_skill"
                        if data["dataset"]
                        in {"ar", "seasonal_ar", "gaussian_ar", "gaussian_seasonal_ar"}
                        else "full_table_reference_ce_skill",
                        "optimizer_state": "averaged",
                        "fast_on_all_three_seeds": costs["fast_on_all_three_seeds"],
                        "mean_epoch": costs["epochs"]["mean"],
                        "training_rows_mean": group["optimized_training_rows"]["mean"],
                        "single_operand_also_reaches_target": any(
                            item["single_operand_target_pass_counts"][target]
                            >= CONFIRMATION_SEEDS
                            for item in group.get(
                                "formula_shortcut_controls", {}
                            ).values()
                        ),
                        "mean_training_flops": costs["training_flops"]["mean"],
                        "mean_seconds": costs["seconds"]["mean"],
                        "signature": group["signature"],
                    }
                )
    ranking.sort(
        key=lambda item: (
            item["dataset"],
            item["noise"],
            item["target"],
            item["mean_training_flops"],
        )
    )
    value["confirmed_target_rankings"] = ranking
    if ranking:
        with (output / "confirmed-target-rankings.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(ranking[0]))
            writer.writeheader()
            writer.writerows(ranking)
    validation_ranking.sort(
        key=lambda item: (item["dataset"], item["order"], item["mean_training_flops"])
    )
    value["confirmed_validation_rankings"] = validation_ranking
    if validation_ranking:
        with (output / "confirmed-validation-rankings.csv").open(
            "w", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(validation_ranking[0]))
            writer.writeheader()
            writer.writerows(validation_ranking)
    write_json(output / "summary.json", value)
    return value


def main() -> None:
    """Summarize completed stages while another process trains later stages."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=ROOT / "runs/grokking-series/fast-first-current",
    )
    parser.add_argument("--include", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    value = report(args.output, include=args.include)
    print(
        json.dumps(
            {
                "completed_runs": len(value["runs"]),
                "three_seed_groups": sum(
                    group["confirmed_on_three_seeds"] for group in value["groups"]
                ),
                "stages_complete": value["stages_complete"],
                "summary": str(args.output / "summary.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
