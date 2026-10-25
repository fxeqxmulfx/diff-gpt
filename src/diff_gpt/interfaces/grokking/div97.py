"""The command line of diff_gpt.application.grokking.div97."""

import argparse
import json
from pathlib import Path

import torch

from diff_gpt.application.grokking import div97
from diff_gpt.application.grokking.div97 import RunConfig, run_plan, verify_checkpoint
from diff_gpt.application.paths import ROOT
from diff_gpt.infrastructure import provenance


def main() -> None:
    """Run one experiment, a resumable JSON plan, or checkpoint verification."""
    parser = argparse.ArgumentParser(description=div97.__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/grokking-div97")
    parser.add_argument("--plan", type=Path)
    parser.add_argument(
        "--verify", type=Path, help="Directory containing a saved checkpoint"
    )
    parser.add_argument(
        "--full-predict",
        action="store_true",
        help="Verify every test row through DiffGPT.predict_many",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--optimizer", choices=["adamw", "schedule-free"], default="schedule-free"
    )
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1.0)
    parser.add_argument("--beta2", type=float, default=0.98)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--after99", type=int, default=25)
    parser.add_argument("--labels", choices=["division", "random"], default="division")
    parser.add_argument("--order", type=int, default=0)
    args = parser.parse_args()
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
                **{name: getattr(args, name) for name in RunConfig.__dataclass_fields__}
            )
        ]
    run_plan(
        configs,
        args.output,
        device=args.device,
        log_every=args.log_every,
        git_head=provenance.git_head,
    )


if __name__ == "__main__":
    main()
