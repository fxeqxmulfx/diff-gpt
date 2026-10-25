"""Audit independent N=2 recall on fresh held-out memories, without training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from benchmarks.grokking.div97 import digest, write_json
from benchmarks.grokking.mqar_div97 import (
    VALUE,
    Frames,
    Task,
    build,
    config_from_record,
    copy_metrics,
    copy_queries,
    score,
    stream,
)
from diff_gpt.domain.optimizer.schedule_free import ScheduleFreeAdamW

if TYPE_CHECKING:
    from benchmarks.grokking.mqar_div97 import RunConfig

AUDIT_STREAM = 0x434F5059
LOADS = 2
FRAMES = 4096
VARIANTS = (
    "original",
    "flip_first_query",
    "flip_second_query",
    "both_queries_first",
    "both_queries_second",
    "swap_loads",
)


def variant(frames: Frames, config: RunConfig, kind: str) -> Frames:
    """Change valid inputs, then recompute their targets and sources from memory."""
    if not config.independent_copy or config.loads != LOADS or config.overwrites:
        msg = "copy audit requires independent N=2 copying without overwrites"
        raise ValueError(msg)
    raw = frames.raw.copy()
    if kind in {"flip_first_query", "flip_second_query"}:
        column = int(kind == "flip_second_query")
        raw[:, -1, column] = np.where(
            raw[:, -1, column] == raw[:, 0, 0], raw[:, 1, 0], raw[:, 0, 0]
        )
    elif kind == "both_queries_first":
        raw[:, -1, 1] = raw[:, -1, 0]
    elif kind == "both_queries_second":
        raw[:, -1, 0] = raw[:, -1, 1]
    elif kind == "swap_loads":
        raw[:, :LOADS] = raw[:, :LOADS][:, ::-1].copy()
    elif kind != "original":
        raise ValueError(kind)
    matches = raw[:, -1, :2, None] == raw[:, None, :LOADS, 0]
    if not matches.any(axis=-1).all():
        msg = "an audit query is not in memory"
        raise ValueError(msg)
    sources = matches.argmax(axis=-1)
    operands = np.take_along_axis(raw[:, :LOADS, VALUE], sources, axis=1)
    raw[:, -1, VALUE] = operands[:, int(config.labels == "divisor")]
    return Frames(
        raw,
        operands[:, None],
        sources[:, None],
        np.full((len(raw), 1, LOADS), -1, dtype=np.int64),
        np.ones((len(raw), 1), dtype=np.int64),
    )


@torch.no_grad()
def audit(directory: Path, device: str = "cpu") -> dict[str, Any]:
    """Score both Schedule-Free states and counterfactual changes on fresh data."""
    metadata = json.loads((directory / "metadata.json").read_text())
    config = config_from_record(metadata)
    task = Task(config)
    fresh = copy_queries(FRAMES, config, "test", stream(config, AUDIT_STREAM))
    variants = {kind: variant(fresh, config, kind) for kind in VARIANTS}
    checkpoint = torch.load(
        directory / "checkpoint.pt", map_location=device, weights_only=True
    )
    if config_from_record(checkpoint) != config:
        msg = "checkpoint and metadata configurations differ"
        raise ValueError(msg)
    model = build(config, device).eval()
    model.load_state_dict(checkpoint["model"])
    optimizer = ScheduleFreeAdamW(model.parameters())
    optimizer.load_state_dict(checkpoint["optimizer"])
    other = 1 - int(config.labels == "divisor")
    shortcut_source = 1 - fresh.sources[:, 0, other]
    shortcut = fresh.raw[np.arange(FRAMES), shortcut_source, VALUE]
    result = {
        "data_protocol": config.data_protocol,
        "epoch": checkpoint["epoch"],
        "frames": FRAMES,
        "contexts": FRAMES // LOADS**2,
        "stream": AUDIT_STREAM,
        "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "fresh_test_sha256": digest(fresh.raw),
        "semantics": (
            "fresh memories from the test partition; all four queries per memory; "
            "all changed queries remain loaded; targets recomputed from memory"
        ),
        "complement_shortcut_accuracy": float(
            (shortcut == fresh.raw[:, -1, VALUE]).mean()
        ),
        "states": {},
    }
    relevant_flip = (
        "flip_second_query" if config.labels == "divisor" else "flip_first_query"
    )
    for state in ("evaluation", "query"):
        if state == "query":
            optimizer.train()
        measured = {}
        for kind, frames in variants.items():
            metrics, predicted = score(model, *task.tensors(frames, device))
            levels = task.codec.levels[predicted[:, 0].cpu().numpy()]
            correct = levels == frames.raw[:, -1, VALUE]
            measured[kind] = dict(metrics)
            if kind == "original":
                baseline, baseline_correct = levels, correct
                measured[kind].update(copy_metrics(frames, levels, config))
            else:
                changed = levels != baseline
                measured[kind]["prediction_changed_fraction"] = float(changed.mean())
                both_correct = baseline_correct & correct
                if kind == relevant_flip:
                    different = fresh.raw[:, -1, VALUE] != frames.raw[:, -1, VALUE]
                    measured[kind]["correct_response_change"] = float(
                        (both_correct & changed)[different].mean()
                    )
                elif kind in {"swap_loads", "flip_first_query", "flip_second_query"}:
                    measured[kind]["both_correct_invariant"] = float(
                        (both_correct & ~changed).mean()
                    )
        result["states"][state] = measured
    write_json(directory / "copy-audit.json", result)
    return result


def main() -> None:
    """Audit complete run directories; never update or transfer trained weights."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for directory in args.directories:
        print(json.dumps({"run": str(directory), **audit(directory, args.device)}))


if __name__ == "__main__":
    main()
