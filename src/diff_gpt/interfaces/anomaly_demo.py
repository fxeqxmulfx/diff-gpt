"""The command line of diff_gpt.application.anomaly_demo."""

import os

import torch

from diff_gpt.application.anomaly_demo import Settings, demo


def read_settings() -> Settings:
    """Read the settings of the use case from the environment."""
    return Settings(
        n_inject=int(os.environ.get("ANOMALY_N_INJECT", "5")),
        max_iters=int(os.environ.get("ANOMALY_MAX_ITERS", "2000")),
        top_k=int(os.environ.get("ANOMALY_TOP_K", "10")),
    )


def main() -> None:
    """
    Train on clean ETTh1 OT, score a spiked test window and print the top-K anomalies.

    Settings come from ANOMALY_N_INJECT (spikes to inject), ANOMALY_MAX_ITERS
    (training steps) and ANOMALY_TOP_K (positions to report); the recall of the
    injected spikes in the top-K is printed last.
    """
    torch.set_float32_matmul_precision("high")
    demo(read_settings())


if __name__ == "__main__":
    main()
