"""The command line of diff_gpt.application.grokking.mqar_div97."""

import argparse
import json
from dataclasses import fields
from pathlib import Path

import torch

from diff_gpt.application.grokking import mqar_div97
from diff_gpt.application.grokking.div97 import ROOT
from diff_gpt.application.grokking.mqar_div97 import (
    CONTEXTS,
    LABELS,
    RunConfig,
    run_plan,
    verify_checkpoint,
)
from diff_gpt.application.grokking.mqar_div97_diagnostics import DiagnosticConfig
from diff_gpt.infrastructure import provenance


def _arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=mqar_div97.__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "runs/grokking-div97-mqar"
    )
    parser.add_argument("--plan", type=Path)
    parser.add_argument(
        "--verify", type=Path, help="Directory containing a saved checkpoint"
    )
    parser.add_argument(
        "--full-predict",
        action="store_true",
        help="Verify every test frame through DiffGPT.predict",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--diagnostics-every",
        type=int,
        default=0,
        help="Extended probes for both optimizer states every N epochs; 0 disables",
    )
    parser.add_argument("--diagnostics-frames", type=int, default=512)
    parser.add_argument(
        "--snapshot-every",
        type=int,
        default=10,
        help="With diagnostics, save both weight states every N epochs; 0 disables",
    )
    types = {"int": int, "float": float, "str": str}
    choices = {"labels": LABELS, "contexts": CONTEXTS}
    for field in fields(RunConfig):
        flag = "--" + field.name.replace("_", "-")
        if field.type == "bool":
            parser.add_argument(
                flag, action=argparse.BooleanOptionalAction, default=field.default
            )
        else:
            parser.add_argument(
                flag,
                type=types[field.type],
                default=field.default,
                choices=choices.get(field.name),
            )
    return parser


def main() -> None:
    """Run one experiment, a resumable JSON plan, or checkpoint verification."""
    args = _arguments().parse_args()
    logging = (
        DiagnosticConfig(
            every=args.diagnostics_every,
            frames=args.diagnostics_frames,
            snapshot_every=args.snapshot_every,
        )
        if args.diagnostics_every
        else None
    )
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("highest")
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if args.verify:
        print(
            json.dumps(
                verify_checkpoint(
                    args.verify, args.device, full_predict=args.full_predict
                ),
                indent=2,
            )
        )
        return
    if args.plan:
        configs = [RunConfig(**config) for config in json.loads(args.plan.read_text())]
    else:
        configs = [
            RunConfig(
                **{field.name: getattr(args, field.name) for field in fields(RunConfig)}
            )
        ]
    run_plan(
        configs,
        args.output,
        device=args.device,
        log_every=args.log_every,
        logging=logging,
        git_head=provenance.git_head,
    )


if __name__ == "__main__":
    main()
