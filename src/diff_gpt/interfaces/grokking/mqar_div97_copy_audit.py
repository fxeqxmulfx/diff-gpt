"""The command line of diff_gpt.application.grokking.mqar_div97_copy_audit."""

import argparse
import json
from pathlib import Path

import torch

from diff_gpt.application.grokking import mqar_div97_copy_audit
from diff_gpt.application.grokking.mqar_div97_copy_audit import audit


def main() -> None:
    """Audit complete run directories; never update or transfer trained weights."""
    parser = argparse.ArgumentParser(description=mqar_div97_copy_audit.__doc__)
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
