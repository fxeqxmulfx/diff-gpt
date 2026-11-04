"""Independently check latest-write answers and obsolete-value predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from diff_gpt.domain.model.gpt import GPT
from research.current_grokking import data
from research.current_grokking.protocol import ROOT, Case
from research.current_grokking.run import verify_model_reference, write
from research.current_grokking.train import Logits

if TYPE_CHECKING:
    from research.current_grokking.interfaces import Task


@torch.no_grad()
def latest_values(model: Logits, task: Task, name: str) -> dict:
    """Parse writes directly from inputs, without using the generator's labels."""
    specification = task.spec.task
    rows = task.splits[name].rows
    counts = dict.fromkeys(
        (
            "queries_checked",
            "queries_correct",
            "rewritten_queries",
            "rewritten_correct",
            "unchanged_queries",
            "unchanged_correct",
            "superseded_value_predictions",
        ),
        0,
    )
    for start in range(0, len(rows), 64):
        chunk = rows[start : start + 64]
        predicted = model(chunk[:, 0]).argmax(-1)
        for example, guesses in zip(chunk, predicted, strict=True):
            tokens, targets = example[:2].tolist()
            predicted_values = guesses.tolist()
            memory, prior, rewritten = {}, {}, set()
            for position in range(1, 1 + 2 * specification.pairs, 2):
                key, value = tokens[position : position + 2]
                if key in memory:
                    prior.setdefault(key, set()).add(memory[key])
                    rewritten.add(key)
                memory[key] = value
            for position, target in enumerate(targets):
                if target < 0:
                    continue
                key, guess = tokens[position], predicted_values[position]
                if target != memory[key]:
                    message = (
                        "Held-out MQAR target is not its key's latest written value"
                    )
                    raise AssertionError(message)
                correct = int(guess == memory[key])
                counts["queries_checked"] += 1
                counts["queries_correct"] += correct
                group = "rewritten" if key in rewritten else "unchanged"
                counts[group + "_queries"] += 1
                counts[group + "_correct"] += correct
                counts["superseded_value_predictions"] += int(
                    not correct and guess in prior.get(key, ())
                )
    counts["all_targets_match_independent_latest_write_oracle"] = True
    return counts


def audit(directory: Path) -> dict:
    """Load a completed current-GPT checkpoint and inspect its held-out queries."""
    torch.set_num_threads(2)
    verify_model_reference()
    checkpoint = torch.load(
        directory / "checkpoint.pt", weights_only=True, map_location="cpu"
    )
    metadata = checkpoint["metadata"]
    case = Case(**metadata["case"])
    task = data.build(case, "cpu")
    if task.fingerprints != metadata["data_fingerprints"]:
        message = "Recall-audit inputs differ from the actual training cohort"
        raise ValueError(message)
    model = Logits(
        GPT(
            task.vocab,
            case.width,
            block_size=task.context,
            n_head=case.heads,
            n_layer=case.layers,
            use_checkpoint=False,
            pad_vocab_size_to=1,
        )
    ).eval()
    model.model.load_state_dict(checkpoint["model"])
    names = [task.selection, *(name for name in task.splits if name.startswith("test"))]
    result = {
        "case": case.name,
        "seed": metadata["seed"],
        "device": "cpu",
        "checkpoint_sha256": hashlib.sha256(
            (directory / "checkpoint.pt").read_bytes()
        ).hexdigest(),
        "oracle": "Sequential dictionary updates parsed from input key/value writes",
        "splits": {name: latest_values(model, task, name) for name in names},
    }
    write(directory / "latest_value_audit.json", result)
    return result


def main() -> None:
    """Audit one finished associative-recall or binding task."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--root", type=Path, default=ROOT / "runs/current-grokking")
    args = parser.parse_args()
    print(json.dumps(audit(args.root / args.case / "seed1")), flush=True)


if __name__ == "__main__":
    main()
