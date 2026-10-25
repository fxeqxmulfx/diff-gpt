"""The command line of diff_gpt.application.mqar."""

import argparse
import json
from pathlib import Path

import torch

from diff_gpt.application import mqar
from diff_gpt.application.mqar import RunConfig, run
from diff_gpt.application.paths import ROOT


def main() -> None:
    """Train the configured runs, one per seed, and print their scores."""
    defaults = RunConfig()
    parser = argparse.ArgumentParser(description=mqar.__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/mqar")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[defaults.seed])
    parser.add_argument("--ood", type=int, nargs="*", default=list(defaults.ood))
    for item in RunConfig.__dataclass_fields__.values():
        if item.name in {"seed", "ood"}:
            continue
        parser.add_argument(
            "--" + item.name.replace("_", "-"),
            type=type(getattr(defaults, item.name)),
            default=getattr(defaults, item.name),
        )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    # Full FP32 matmuls: TF32 rounds their inputs to 10 bits of mantissa.
    torch.set_float32_matmul_precision("highest")
    settings = {
        name: getattr(args, name)
        for name in RunConfig.__dataclass_fields__
        if name not in {"seed", "ood"}
    }
    for seed in args.seeds:
        config = RunConfig(**settings, ood=tuple(args.ood), seed=seed)
        result = run(config, args.output, args.device, log=not args.quiet)
        scores = {
            key: round(value["token_accuracy"], 4)
            for key, value in result["test"].items()
        }
        print(
            json.dumps({"name": result["name"], "token_accuracy": scores}), flush=True
        )


if __name__ == "__main__":
    main()
