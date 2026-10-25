"""
Execute the fast-first experiment loop and retain every measured plan.

Screens use one seed and 250 epochs. Selection prefers the lowest training
FLOPs to 99% skill, then 90% skill. Unknown-floor tasks use an explicit 10%
improvement over the training-fitted unigram on validation. Only an unsuccessful
screen triggers a longer run; every selected setting is confirmed on 3 seeds.
"""

from __future__ import annotations

import csv
import json
import math
import traceback
from dataclasses import asdict, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from diff_gpt.application.grokking import series as series_runner
from diff_gpt.application.grokking.div97 import write_json
from diff_gpt.application.grokking.series import Config, run, verify
from diff_gpt.application.grokking.series_data import DataConfig, dataset

if TYPE_CHECKING:
    import argparse
    from collections.abc import Sequence
    from pathlib import Path

SEEDS = (11, 17, 23)
FAST_EPOCHS = 250
FALLBACK_EPOCHS = 1500
KNOWN_TARGETS = (0.99, 0.9)
VALIDATION_REDUCTION = 0.1
NOISE_SCREEN_DECAY = 3.0
NOISE_SCREEN_RATE = 0.03
FIBONACCI_NOISE_FALLBACK = {0.01: 4000, 0.05: 4000, 0.2: 1500}


def observations(directory: Path, budget: int | None = None) -> list[dict[str, str]]:
    """Read durable measurements without tuning on rows outside a screen."""
    with (directory / "history.csv").open(newline="") as handle:
        return [
            row
            for row in csv.DictReader(handle)
            if budget is None or int(row["epoch"]) <= budget
        ]


def quality(directory: Path, budget: int | None = None) -> dict[str, Any]:
    """Measure target crossings and the best held-out quality at fixed compute."""
    metadata = json.loads((directory / "metadata.json").read_text())
    rows = observations(directory, budget)
    config = metadata["config"]
    known = all(
        ref["floor_loss"] is not None for ref in metadata["references"]["test"].values()
    )
    anchor = config["data"]["dataset"] in {"div97", "fibonacci"}
    target_key = "test_accuracy" if anchor else "test_skill"
    selection = (
        "test_bayes_loss"
        if config["data"]["dataset"]
        in {"ar", "seasonal_ar", "gaussian_ar", "gaussian_seasonal_ar"}
        else "test_loss"
        if known
        else "validation_loss"
    )
    references = metadata["references"]["validation" if not known else "test"]
    baseline = float(np.mean([ref["baseline_loss"] for ref in references.values()]))
    targets = KNOWN_TARGETS if known else ((1 - VALIDATION_REDUCTION) * baseline,)
    crossings = {}
    for target in targets:
        candidates = [
            row
            for row in rows
            if (
                float(row[target_key] or "-inf") >= target
                if known
                else float(row[selection]) <= target
            )
        ]
        crossings[str(target)] = (
            {
                key: float(candidates[0][key])
                for key in ("epoch", "seconds", "training_flops")
            }
            if candidates
            else None
        )
    best = min(rows, key=lambda row: float(row[selection]))
    return {
        "known_floor": known,
        "target_metric": target_key if known else selection,
        "target_direction": "higher" if known else "lower",
        "targets": list(targets),
        "crossings": crossings,
        "best_selection_loss": float(best[selection]),
        "best_epoch": int(best["epoch"]),
        "best_training_flops": float(best["training_flops"]),
        "selection_split": "test" if known else "validation",
        "baseline_loss": baseline,
    }


def selection_key(measured: dict[str, Any]) -> tuple[float, float, float]:
    """Prefer fast target attainment; use final quality only after compute ties."""
    for index, target in enumerate(measured["targets"]):
        if (crossing := measured["crossings"][str(target)]) is not None:
            return (
                float(index),
                crossing["training_flops"],
                measured["best_selection_loss"],
            )
    return math.inf, measured["best_selection_loss"], measured["best_training_flops"]


def execute(config: Config, output: Path, stage: str, device: str) -> Path:
    """Reuse complete runs, or resume exactly from their last durable state."""
    directory = output / stage / config.name()
    write_json(
        output / "loop-status.json",
        {
            "stage": stage,
            "run": str(directory),
            "config": asdict(config),
            "state": "running",
        },
    )
    result = directory / "result.json"
    if result.exists():
        previous = json.loads(result.read_text())
        if previous["epoch"] >= config.epochs or (
            previous["epoch"] < config.epochs
            and all(
                value["first99"] is not None
                for value in previous["criteria"]["averaged"].values()
            )
        ):
            metadata = json.loads((directory / "metadata.json").read_text())
            if metadata["source_sha256"] != series_runner.source_hashes():
                msg = (
                    "completed run has other source; "
                    "use its frozen code or a new output"
                )
                raise ValueError(msg)
            if metadata["data"]["hashes"] != dataset(config.data).fingerprint():
                msg = "completed run has other data; use a new output directory"
                raise ValueError(msg)
            if not (directory / "verification.json").exists():
                verify(directory, device)
            print(f"Reused: {directory}", flush=True)
            return directory
        archive = directory / f"result-budget-{previous['epoch']}.json"
        if not archive.exists():
            archive.write_bytes(result.read_bytes())
    run(config, directory, device, resume=directory.exists())
    verify(directory, device)
    return directory


def screen(
    configs: Sequence[Config],
    output: Path,
    stage: str,
    device: str,
    *,
    fallback: int | None = FALLBACK_EPOCHS,
) -> Config:
    """Screen single changes first, then extend only a selected failed screen."""
    write_json(output / f"plan-{stage}.json", [asdict(config) for config in configs])
    measured = []
    for config in configs:
        directory = execute(config, output, stage, device)
        measured.append(
            {
                "directory": str(directory),
                "config": asdict(config),
                **quality(directory, FAST_EPOCHS),
            }
        )
        write_json(output / f"screen-{stage}.json", measured)
    selected_index = min(
        range(len(measured)), key=lambda index: selection_key(measured[index])
    )
    selected = configs[selected_index]
    success = math.isfinite(selection_key(measured[selected_index])[0])
    decision = {
        "screen_budget_epochs": FAST_EPOCHS,
        "fast_target_reached": success,
        "selected": asdict(selected),
        "measurement": measured[selected_index],
        "selection": "lowest training FLOPs to target, then held-out loss",
        "fallback_required": not success,
    }
    decision["fallback_deferred"] = not success and fallback is None
    if not success and fallback is not None:
        selected = replace(selected, epochs=fallback)
        execute(selected, output, stage, device)
    decision["selected"] = asdict(selected)
    write_json(output / f"selection-{stage}.json", decision)
    return selected


def confirm(config: Config, output: Path, stage: str, device: str) -> None:
    """Confirm the selected representation/optimizer on three GPU seeds."""
    configs = [replace(config, data=replace(config.data, seed=seed)) for seed in SEEDS]
    write_json(
        output / f"plan-{stage}-confirm.json", [asdict(item) for item in configs]
    )
    for item in configs:
        execute(item, output, stage, device)


def formula(output: Path, device: str) -> Config:
    """Compare capacity, optimization, data amount and quantization resolution."""
    base = Config(epochs=FAST_EPOCHS)
    variants = [
        base,
        replace(base, width=64),
        replace(base, width=32),
        replace(base, heads=1),
        replace(base, learning_rate=0.03),
        replace(base, learning_rate=0.003),
        replace(base, weight_decay=0.1),
        replace(base, weight_decay=3.0),
        replace(base, beta1=0.5),
        replace(base, data=replace(base.data, max_train=8192)),
        replace(base, data=replace(base.data, max_train=0)),
        replace(base, data=replace(base.data, domain_multiplier=2.0)),
        replace(base, data=replace(base.data, domain_multiplier=4.0)),
    ]
    selected = screen(variants, output, "formula-fast", device)
    confirm(selected, output, "formula-fast", device)
    execute(
        replace(
            selected,
            data=replace(selected.data, random_labels=True),
            epochs=FAST_EPOCHS,
        ),
        output,
        "formula-fast",
        device,
    )
    for name in ("VPdef", "H2OC", "sh"):
        other = replace(
            selected, data=replace(selected.data, formula=name), epochs=FAST_EPOCHS
        )
        other = screen([other], output, f"formula-{name}", device)
        confirm(other, output, f"formula-{name}", device)
    return selected


def formula_refined(output: Path, device: str) -> None:
    """Use all available equations with smaller models before extending training."""
    selections = []
    for name in ("VPact", "VPdef", "H2OC", "sh"):
        base = Config(
            data=DataConfig(formula=name, max_train=0), width=64, epochs=FAST_EPOCHS
        )
        variants = [
            base,
            replace(base, width=32),
            replace(base, batch_size=4096),
            replace(base, width=32, batch_size=4096),
        ]
        if name == "VPact":
            variants.append(replace(base, learning_rate=0.003))
        stage = f"formula-{name}-fast"
        selected = screen(variants, output, stage, device, fallback=None)
        selections.append((stage, selected))
    for stage, candidate in selections:
        selected = candidate
        decision = json.loads((output / f"selection-{stage}.json").read_text())
        if decision["fallback_required"]:
            selected = replace(candidate, epochs=FALLBACK_EPOCHS)
            execute(selected, output, stage, device)
            decision.update(selected=asdict(selected), fallback_deferred=False)
            write_json(output / f"selection-{stage}.json", decision)
        confirm(selected, output, stage, device)
        confirm(
            replace(
                selected,
                epochs=FAST_EPOCHS,
                data=replace(selected.data, random_labels=True),
            ),
            output,
            stage,
            device,
        )


def noise(output: Path, device: str) -> None:
    """Use clean tests and fixed corrupted supervision at each noise fraction."""
    for name, order, heads in (("div97", 0, 4), ("fibonacci", 1, 1)):
        base = Config(
            data=DataConfig(dataset=name, order=order, vocab_size=256),
            heads=heads,
            epochs=FAST_EPOCHS,
            after99=100,
        )
        variants = (
            [base]
            if name == "div97"
            else [
                base,
                replace(base, width=64),
                replace(base, learning_rate=0.03),
                replace(base, weight_decay=3.0),
                replace(base, weight_decay=0.3),
                replace(base, beta1=0.5),
            ]
        )
        selected = screen(
            variants,
            output,
            f"{name}-clean-fast",
            device,
            fallback=10000 if name == "fibonacci" else FALLBACK_EPOCHS,
        )
        confirm(selected, output, f"{name}-clean-fast", device)
        for fraction in (0.01, 0.05, 0.2):
            noisy = replace(
                selected,
                data=replace(selected.data, noise=fraction),
                epochs=FAST_EPOCHS,
            )
            candidates = [
                noisy,
                replace(
                    noisy,
                    weight_decay=NOISE_SCREEN_DECAY
                    if noisy.weight_decay != NOISE_SCREEN_DECAY
                    else 1.0,
                ),
                replace(
                    noisy,
                    learning_rate=NOISE_SCREEN_RATE
                    if noisy.learning_rate != NOISE_SCREEN_RATE
                    else 0.01,
                ),
            ]
            stage = f"{name}-noise-{fraction:g}"
            chosen = screen(
                candidates,
                output,
                stage,
                device,
                fallback=FIBONACCI_NOISE_FALLBACK[fraction]
                if name == "fibonacci"
                else FALLBACK_EPOCHS,
            )
            confirm(chosen, output, stage, device)


def temporal(output: Path, device: str) -> None:
    """Compare weather columns under the two codec orders."""
    for order in (0, 1):
        base = Config(
            data=DataConfig(
                dataset="weather", order=order, context=2, max_train=512, max_eval=256
            ),
            batch_size=64,
            micro_batch_size=64,
            epochs=FAST_EPOCHS,
        )
        variants = [
            base,
            replace(base, width=64),
            replace(base, heads=1),
            replace(base, learning_rate=0.003),
            replace(base, data=replace(base.data, max_train=2048)),
        ]
        stage = f"weather-k{order}-fast"
        selected = screen(variants, output, stage, device)
        confirm(selected, output, stage, device)
        measured = _fast_confirmations(selected, output, stage)
        if not all(math.isfinite(selection_key(item)[0]) for item in measured):
            candidates = json.loads((output / f"screen-{stage}.json").read_text())
            for item in sorted(candidates, key=selection_key):
                alternative = series_runner.config_from_dict(item["config"])
                if alternative == selected or not math.isfinite(selection_key(item)[0]):
                    continue
                confirm(alternative, output, stage, device)
                measurements = _fast_confirmations(alternative, output, stage)
                if all(
                    math.isfinite(selection_key(value)[0]) for value in measurements
                ):
                    write_json(
                        output / f"selection-weather-k{order}-robust.json",
                        {
                            "selected": asdict(alternative),
                            "reason": (
                                "initial selection missed a confirmation; use the "
                                "next reproducible fast screened setting"
                            ),
                            "measurements": measurements,
                            "confirmed_fast_on_three_seeds": True,
                        },
                    )
                    break


def _fast_confirmations(
    config: Config, output: Path, stage: str
) -> list[dict[str, Any]]:
    """Test the declared short budget on every confirmation, including extensions."""
    return [
        {
            "seed": seed,
            **quality(
                output
                / stage
                / replace(config, data=replace(config.data, seed=seed)).name(),
                FAST_EPOCHS,
            ),
        }
        for seed in SEEDS
    ]


def temporal_late(output: Path, device: str) -> None:
    """Retain the fast baseline while testing late formula-column improvement."""
    for order in (0, 1):
        stage = f"weather-k{order}-fast"
        path = output / f"selection-{stage}.json"
        robust = output / f"selection-weather-k{order}-robust.json"
        if robust.exists():
            path = robust
        values = json.loads(path.read_text())["selected"]
        base = series_runner.config_from_dict(values)
        configs = [
            replace(base, epochs=FALLBACK_EPOCHS, data=replace(base.data, seed=seed))
            for seed in SEEDS
        ]
        write_json(
            output / f"plan-weather-k{order}-late-confirm.json",
            [asdict(config) for config in configs],
        )
        for config in configs:
            execute(config, output, stage, device)


def synthetic(output: Path, device: str) -> None:
    """Measure exact expected test CE as well as sampled-label CE."""
    for name, context in (("ar", 1), ("seasonal_ar", 24)):
        base = Config(
            data=DataConfig(dataset=name, context=context, max_train=256, max_eval=256),
            batch_size=128,
            epochs=FAST_EPOCHS,
        )
        variants = [
            base,
            replace(base, width=64),
            replace(base, heads=1),
            replace(base, data=replace(base.data, max_train=2048)),
            replace(base, weight_decay=0.1),
        ]
        if name == "seasonal_ar":
            variants.extend((replace(base, rope_base=10.0), replace(base, layers=2)))
        selected = screen(variants, output, f"{name}-fast", device)
        confirm(selected, output, f"{name}-fast", device)


def forecasting(output: Path, device: str) -> None:
    """Retain validation-selected baselines at the identical training budget."""
    for name, context in (("weather", 8), ("etth1", 24), ("m4_hourly", 24)):
        base = Config(
            data=DataConfig(dataset=name, context=context, max_train=256, max_eval=128),
            batch_size=32,
            micro_batch_size=32,
            epochs=FAST_EPOCHS,
        )
        variants = [
            base,
            replace(base, width=64),
            replace(base, heads=1),
            replace(base, rope_base=10.0),
            replace(base, learning_rate=0.003),
            replace(base, data=replace(base.data, max_train=1024)),
            replace(base, data=replace(base.data, order=1)),
            replace(base, width=64, data=replace(base.data, order=1)),
            replace(base, data=replace(base.data, domain_multiplier=2.0)),
        ]
        stage = f"forecast-{name}-fast"
        selected = screen(variants, output, stage, device)
        confirm(selected, output, stage, device)
        control = replace(selected, data=replace(selected.data, shuffled_time=True))
        confirm(control, output, stage, device)


def gaussian(output: Path, device: str) -> None:
    """Check the same protocol on linear continuous AR with a known token law."""
    selections = []
    for name, context in (("gaussian_ar", 1), ("gaussian_seasonal_ar", 24)):
        base = Config(
            data=DataConfig(dataset=name, context=context, max_train=256, max_eval=256),
            width=64,
            batch_size=128,
            epochs=FAST_EPOCHS,
        )
        variants = [
            base,
            replace(base, width=128),
            replace(base, heads=1),
            replace(base, data=replace(base.data, max_train=2048)),
            replace(base, weight_decay=0.1),
            replace(base, learning_rate=0.003),
        ]
        if context > 1:
            variants.extend((replace(base, rope_base=10.0), replace(base, layers=2)))
        stage = f"{name}-fast"
        selected = screen(variants, output, stage, device, fallback=None)
        selections.append((stage, selected))
    for stage, candidate in selections:
        selected = candidate
        decision = json.loads((output / f"selection-{stage}.json").read_text())
        if decision["fallback_required"]:
            selected = replace(selected, epochs=FALLBACK_EPOCHS)
            execute(selected, output, stage, device)
            decision.update(selected=asdict(selected), fallback_deferred=False)
            write_json(output / f"selection-{stage}.json", decision)
        confirm(selected, output, stage, device)


def run_stages(args: argparse.Namespace) -> None:
    """Run the stages that `args` name and write the suite's summary."""
    args.output.mkdir(parents=True, exist_ok=True)
    actions = {
        "formula": formula,
        "formula_refined": formula_refined,
        "noise": noise,
        "temporal": temporal,
        "temporal_late": temporal_late,
        "synthetic": synthetic,
        "gaussian": gaussian,
        "forecasting": forecasting,
    }
    for stage in args.stages:
        try:
            actions[stage](args.output, args.device)
            write_json(args.output / f"stage-{stage}-complete.json", {"complete": True})
        except Exception:
            write_json(
                args.output / f"stage-{stage}-error.json",
                {"traceback": traceback.format_exc()},
            )
            raise
    write_json(
        args.output / "loop-status.json", {"state": "complete", "stages": args.stages}
    )
