"""The command line of diff_gpt.application.diff_bench."""

import torch

from diff_gpt.application.diff_bench import bench


def main() -> None:
    """Train three models on sin/cos and print their mean val_loss, time and MAE."""
    torch.set_float32_matmul_precision("high")
    bench()


if __name__ == "__main__":
    main()
