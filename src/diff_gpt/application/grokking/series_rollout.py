"""
Free-running multi-row forecasts for completed real-series experiments.

Each generated row becomes the next input. The public DiffGPT predictor is
called one row at a time on a rolling context, so its existing context limit
is respected. Original observations after the forecast origin are used only
for scoring and the explicit training-vocabulary guard. Compare final weights,
the full-budget validation-best weights, persistence and seasonal naive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from benchmarks.grokking.div97 import ROOT, load_checkpoint, write_json
from benchmarks.grokking.series import build, config_from_dict
from benchmarks.grokking.series_data import _raw_series
from diff_gpt.application.diff_gpt import DiffGPT
from diff_gpt.domain.encoder_decoder import TokenClippingWarning, encode
from diff_gpt.domain.sampler.temperature import TemperatureSampler

HORIZONS = {"weather": 144, "etth1": 24, "m4_hourly": 48}


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scores(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    error = prediction - truth
    return {"mse": float(np.mean(error**2)), "mae": float(np.mean(np.abs(error)))}


@torch.no_grad()
def rollout(directory: Path, device: str, *, maximum: int = 8) -> dict[str, Any]:
    """Evaluate fixed eligible origins using the recorded training-only codec."""
    if maximum < 1:
        msg = "maximum forecast origins must be positive"
        raise ValueError(msg)
    metadata = json.loads((directory / "metadata.json").read_text())
    config = config_from_dict(metadata["config"])
    if config.data.dataset not in HORIZONS:
        msg = "multi-row rollout is defined for weather, ETTh1 and M4 Hourly"
        raise ValueError(msg)
    for relative, expected in metadata["source_sha256"].items():
        if _hash(ROOT / relative) != expected:
            msg = "training source changed; run the scorer from the frozen source"
            raise ValueError(msg)
    horizon = HORIZONS[config.data.dataset]
    files = {
        "final": directory / "checkpoint.pt",
        "validation_best_same_budget": directory / "best-validation.pt",
    }
    if (directory / "early-stopping.pt").exists():
        files["first_patience_early_stop"] = directory / "early-stopping.pt"
    checkpoint_hashes = {name: _hash(path) for name, path in files.items()}
    destination = directory / f"rollout-h{horizon}-w{maximum}.json"
    if destination.exists():
        previous = json.loads(destination.read_text())
        if previous["checkpoint_sha256"] == checkpoint_hashes and previous[
            "scorer_sha256"
        ] == _hash(Path(__file__)):
            return previous
    raw, columns, source, period = _raw_series(config.data)
    if source.get("source_sha256") != metadata["data"].get("source_sha256"):
        msg = "forecast source data differs from the recorded training run"
        raise ValueError(msg)
    if config.data.shuffled_time:
        rng = np.random.default_rng(config.data.seed ^ 0x5E71E5)
        for values in raw:
            end = int(len(values) * config.data.train_fraction)
            val_end = int(len(values) * (config.data.train_fraction + 1) / 2)
            for first, stop in ((0, end), (end, val_end), (val_end, len(values))):
                values[first:stop] = values[first:stop][rng.permutation(stop - first)]
    features = len(columns)
    with np.load(directory / "frozen_splits.npz", allow_pickle=False) as frozen:
        ids = frozen["test__ids"].copy()
        x = frozen["train__x"].copy()
        y = frozen["train__y"].copy()
        column = frozen["train__column"].copy()
    seen = [
        set(x[:, index::features].ravel()) | set(y[column == index])
        for index in range(features)
    ]
    fits = metadata["data"]["fit"]
    selected, excluded = [], {"horizon_past_end": 0, "unseen_training_token": 0}
    rng = np.random.default_rng(config.data.seed ^ 0xA0FE)
    for index in rng.permutation(len(ids)):
        series_id, first, origin = map(int, ids[index])
        values = raw[series_id]
        if origin + horizon > len(values):
            excluded["horizon_past_end"] += 1
            continue
        fit = fits[series_id]
        mean, std, domain = (
            np.asarray(fit[name]) for name in ("mean", "std", "domain")
        )
        normalized = (values[first : origin + horizon] - mean) / std
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", TokenClippingWarning)
            _, _, tokens = encode(
                normalized,
                config.data.vocab_size,
                domain,
                config.data.order,
                tokenizer=config.data.tokenizer,
            )
        if any(
            not set(tokens[:, feature]) <= seen[feature] for feature in range(features)
        ):
            excluded["unseen_training_token"] += 1
            continue
        clipped = sum(
            int(str(item.message).split()[0])
            for item in caught
            if isinstance(item.message, TokenClippingWarning)
        )
        selected.append(
            {
                "series": series_id,
                "first": first,
                "origin": origin,
                "mean": mean,
                "std": std,
                "domain": domain,
                "truth": values[origin : origin + horizon].copy(),
                "context": values[first:origin].copy(),
                "clipped_actual_tokens": clipped,
            }
        )
        if len(selected) >= maximum:
            break
    result = {
        "horizon_rows": horizon,
        "readout": "autoregressive token argmax via public DiffGPT.predict_many",
        "teacher_forced_future": False,
        "checkpoint_sha256": checkpoint_hashes,
        "scorer_sha256": _hash(Path(__file__)),
        "selected_origins": [
            {
                key: value
                for key, value in row.items()
                if key in {"series", "first", "origin", "clipped_actual_tokens"}
            }
            for row in selected
        ],
        "excluded_origins_before_limit": excluded,
        "all_actual_context_and_future_tokens_seen_in_training": True,
        "protocol": (
            "internal chronological test segments; rolling model context; "
            "seasonal baseline uses only observations before origin"
        ),
        "scores": {},
    }
    if not selected:
        result["limitation"] = (
            "no complete forecast horizon satisfied training-vocabulary coverage"
        )
        write_json(destination, result)
        return result
    truth = np.stack([item["truth"] for item in selected])
    means = np.stack([item["mean"] for item in selected])[:, None]
    stds = np.stack([item["std"] for item in selected])[:, None]
    normalized_truth = (truth - means) / stds
    persistence = np.stack(
        [np.repeat(item["context"][-1:], horizon, axis=0) for item in selected]
    )
    seasonal = np.stack(
        [
            raw[item["series"]][item["origin"] - period + np.arange(horizon) % period]
            for item in selected
        ]
    )

    def record(name: str, prediction: np.ndarray) -> None:
        normalized_prediction = (prediction - means) / stds
        result["scores"][name] = {
            "raw": _scores(prediction, truth),
            "train_normalized": _scores(normalized_prediction, normalized_truth),
            "per_column_train_normalized": {
                column: _scores(
                    normalized_prediction[:, :, index], normalized_truth[:, :, index]
                )
                for index, column in enumerate(columns)
            },
        }

    record("persistence", persistence)
    record("seasonal_naive", seasonal)
    for name, path in files.items():
        saved = load_checkpoint(path, device)
        model = build(config, x.shape[1], device)
        model.load_state_dict(saved["model"])
        model.eval()
        predictions = np.empty_like(truth)
        groups = {}
        for index, item in enumerate(selected):
            groups.setdefault(item["series"], []).append(index)
        for members in groups.values():
            item = selected[members[0]]
            wrapper = DiffGPT(
                model,
                config.data.order,
                item["domain"],
                tokenizer=config.data.tokenizer,
            )
            frames = [
                pd.DataFrame(
                    (selected[index]["context"] - selected[index]["mean"])
                    / selected[index]["std"],
                    columns=columns,
                )
                for index in members
            ]
            context_rows = frames[0].shape[0]
            for step in range(horizon):
                forecasts = wrapper.predict_many(
                    frames,
                    max_new_points=1,
                    sampler=TemperatureSampler(0),
                    batch_size=maximum,
                )
                for local, index in enumerate(members):
                    normalized = forecasts[local].iloc[-1].to_numpy()
                    predictions[index, step] = (
                        normalized * selected[index]["std"] + selected[index]["mean"]
                    )
                frames = [
                    forecast.iloc[-context_rows:].reset_index(drop=True)
                    for forecast in forecasts
                ]
        record(name, predictions)
        result["scores"][name]["selected_epoch"] = saved["epoch"]
    dependency_paths = [
        Path(__file__),
        ROOT / "src/diff_gpt/application/diff_gpt.py",
        ROOT / "src/diff_gpt/domain/model/engine.py",
        ROOT / "src/diff_gpt/domain/model/kv_cache.py",
        ROOT / "src/diff_gpt/domain/sampler/temperature.py",
    ]
    result["forecast_source_sha256"] = {
        str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path): _hash(path)
        for path in dependency_paths
    }
    write_json(destination, result)
    return result


def main() -> None:
    """Score completed forecast families after GPU training has finished."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output", type=Path, nargs="?", default=ROOT / "runs/grokking-series/fast-first"
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--maximum", type=int, default=8)
    parser.add_argument("--selected-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for path in sorted(args.output.glob("forecast-*/*/result.json")):
        if args.selected_only:
            choice = json.loads(
                (args.output / f"selection-{path.parent.parent.name}.json").read_text()
            )["selected"]
            current = json.loads(path.read_text())["config"]
            for item in (choice, current):
                item.pop("epochs", None)
                item["data"].pop("seed", None)
                item["data"].pop("shuffled_time", None)
            if choice != current:
                continue
        result = rollout(path.parent, args.device, maximum=args.maximum)
        print(
            json.dumps(
                {
                    "run": str(path.parent),
                    "horizon": result["horizon_rows"],
                    "origins": len(result["selected_origins"]),
                    "scores": {
                        name: value["train_normalized"]
                        for name, value in result["scores"].items()
                    },
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
