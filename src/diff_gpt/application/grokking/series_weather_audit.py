"""
Audit restricted-input formula ceilings inside the multivariate weather runs.

These references condition only on the formula's two same-row operands. The
full GPT sees additional columns and temporal context, so these are diagnostic
comparators, not Bayes floors for its complete prompt. Positive-order native
tokens also require integration; a level-token ceiling cannot be substituted.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.grokking.div97 import write_json
from benchmarks.grokking.series_data import FORMULAS, WEATHER, _read_weather
from benchmarks.grokking.series_oracle import conditional_reference
from diff_gpt.domain.encoder_decoder import TokenClippingWarning, encode


def formula_comparators(directory: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    """Tabulate the full weather table on the recorded train-fitted level grid."""
    config = metadata["config"]["data"]
    if config["dataset"] != "weather":
        return {}
    if config["order"] != 0:
        return {
            "native_token_floor": None,
            "reason": (
                "level equations do not determine native derivative-token labels; "
                "the positive-order multivariate Bayes floor remains unknown"
            ),
        }
    path = Path(config["source"]) if config["source"] else WEATHER
    frame, provenance = _read_weather(path)
    if provenance["source_sha256"] != metadata["data"]["source_sha256"]:
        msg = "weather formula audit source differs from training"
        raise ValueError(msg)
    columns = frame.columns.tolist()
    fit = metadata["data"]["fit"][0]
    values = frame.to_numpy().copy()
    if config["shuffled_time"]:
        end = int(len(values) * config["train_fraction"])
        validation_end = int(len(values) * (config["train_fraction"] + 1) / 2)
        rng = np.random.default_rng(config["seed"] ^ 0x5E71E5)
        for first, stop in (
            (0, end),
            (end, validation_end),
            (validation_end, len(values)),
        ):
            values[first:stop] = values[first:stop][rng.permutation(stop - first)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", TokenClippingWarning)
        _, _, tokens = encode(
            (values - np.asarray(fit["mean"])) / np.asarray(fit["std"]),
            config["vocab_size"],
            np.asarray(fit["domain"]),
            0,
            tokenizer=config["tokenizer"],
        )
    result = {
        "native_full_prompt_bayes_floor": None,
        "method": (
            "full source-table conditional entropy given the two same-row "
            "formula operands on the recorded level grid"
        ),
        "limitation": (
            "additional prompt values can lower this reference CE; "
            "it is not the Bayes floor of the complete multivariate prompt"
        ),
        "columns": {},
    }
    with np.load(directory / "frozen_splits.npz", allow_pickle=False) as frozen:
        for label, names in FORMULAS.items():
            if any(name not in columns for name in names):
                continue
            indexes = [columns.index(name) for name in names]
            inputs = tokens[:, indexes[:2]]
            targets = tokens[:, indexes[-1]]
            result["columns"][names[-1]] = {
                "formula": label,
                "operands": list(names[:2]),
                "splits": {
                    name: conditional_reference(
                        inputs,
                        targets,
                        inputs[frozen[f"{name}__ids"][:, 2]],
                        targets[frozen[f"{name}__ids"][:, 2]],
                        config["vocab_size"],
                    )
                    for name in ("train", "validation", "test")
                },
            }
    write_json(directory / "same-row-formula-comparators.json", result)
    return result


def main() -> None:
    """Audit every completed multivariate weather run in an output directory."""
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    for path in args.output.glob("*/*/metadata.json"):
        metadata = json.loads(path.read_text())
        if metadata["config"]["data"]["dataset"] == "weather":
            formula_comparators(path.parent, metadata)


if __name__ == "__main__":
    main()
