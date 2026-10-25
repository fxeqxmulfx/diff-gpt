"""Probe whether a temporal DiffGPT checkpoint learned to copy a past period."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from benchmarks.grokking.div97 import load_checkpoint, write_json
from benchmarks.grokking.recurrence import Config, build, tensors


@torch.inference_mode()
def probe(directory: Path, device: str, samples: int) -> dict[str, object]:
    """Use unseen-orbit contexts, input interventions and synthetic copy controls."""
    saved = load_checkpoint(directory / "checkpoint.pt", device)
    config = Config(**saved["config"])
    if config.grid_domain or config.vocab_size:
        msg = "this copy probe requires the exact integer grid"
        raise ValueError(msg)
    model = build(config, device)
    model.load_state_dict(saved["model"])
    model.eval()
    raw = saved["raw"][saved["train_rows"] : saved["train_rows"] + samples]
    x, y = tensors(raw.cpu().numpy(), config, device)

    def predict(inputs: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                model(inputs[first : first + 128])[0][:, -1].argmax(-1)
                for first in range(0, len(inputs), 128)
            ]
        )

    def accuracy(actual: torch.Tensor, expected: torch.Tensor) -> float:
        return float((actual == expected).double().mean())

    baseline = predict(x)
    interventions = {}
    for position in sorted(
        {0, config.context // 4, config.context // 2, config.context - 1}
    ):
        changed = x.clone()
        changed[:, position] = x[:, position].roll(1)
        moved = changed[:, position] != x[:, position]
        predicted = predict(changed)
        interventions[str(position)] = {
            "changed_rows": int(moved.sum()),
            "prediction_copies_first": accuracy(predicted[moved], changed[moved, 0]),
            "prediction_matches_original_answer": accuracy(
                predicted[moved], y[moved, -1]
            ),
            "prediction_matches_unperturbed": accuracy(
                predicted[moved], baseline[moved]
            ),
        }
    # A distinct synthetic diagnostic, not an accuracy claim on the recurrence.
    rng = torch.Generator(device=device).manual_seed(2026)
    pool = x.unique()
    synthetic = pool[torch.randint(len(pool), x.shape, generator=rng, device=device)]
    shorter = {}
    for context in sorted({2, 3, 4, 8, 16, 24, config.context - 1, config.context}):
        if context > config.context:
            continue
        short = replace(config, context=context)
        inputs, targets = tensors(
            raw[:, : short.order + context + 1].cpu().numpy(), short, device
        )
        shorter[str(context)] = accuracy(predict(inputs), targets[:, -1])
    result = {
        "checkpoint_epoch": saved["epoch"],
        "rows": len(x),
        "natural_recurrence_accuracy": accuracy(baseline, y[:, -1]),
        "true_next_token_equals_first_observed_token": accuracy(y[:, -1], x[:, 0]),
        "input_interventions": interventions,
        "synthetic_independent_tokens_copy_accuracy": accuracy(
            predict(synthetic), synthetic[:, 0]
        ),
        "shorter_context_transfer_accuracy": shorter,
        "interpretation": (
            "Interventions and synthetic inputs probe the copy mechanism; "
            "they are separate from the natural-recurrence test score."
        ),
    }
    write_json(directory / f"period-probe-epoch{saved['epoch']}.json", result)
    return result


def main() -> None:
    """Run reproducible controls without altering a checkpoint or sampler."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--samples", type=int, default=1024)
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(probe(args.directory, args.device, args.samples), flush=True)


if __name__ == "__main__":
    main()
