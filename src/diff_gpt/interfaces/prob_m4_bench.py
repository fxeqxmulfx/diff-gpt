"""The command line of diff_gpt.application.prob_m4_bench."""

import os

import torch

from diff_gpt.application.prob_m4_bench import Settings, bench


def read_settings() -> Settings:
    """Read the settings of the use case from the environment."""
    return Settings(
        freq=os.environ.get("M4_FREQ", "Hourly"),
        n_series=int(os.environ.get("M4_N_SERIES", "50")),
        num_samples=int(os.environ.get("M4_NUM_SAMPLES", "100")),
        temperature=float(os.environ.get("M4_TEMPERATURE", "1.0")),
        train_part=float(os.environ.get("M4_TRAIN_PART", "0.8")),
        target_bs=int(os.environ.get("M4_BLOCK_SIZE", "256")),
        vocab_size=int(os.environ.get("M4_VOCAB_SIZE", "256")),
        n_embd=int(os.environ.get("M4_N_EMBD", "64")),
        n_head=int(os.environ.get("M4_N_HEAD", "4")),
        n_layer=int(os.environ.get("M4_N_LAYER", "2")),
        max_iters=int(os.environ.get("M4_MAX_ITERS", "20000")),
        eval_interval=int(os.environ.get("M4_EVAL_INTERVAL", "500")),
        top_p_env=os.environ.get("M4_TOP_P"),
    )


def main() -> None:
    """
    Train one global diff-gpt on M4 and print point and probabilistic scores.

    Each series gets M4_NUM_SAMPLES trajectories sampled at M4_TEMPERATURE
    (with nucleus top-p when M4_TOP_P is set); their per-step median is the
    point forecast.
    """
    torch.set_float32_matmul_precision("high")
    bench(read_settings())


if __name__ == "__main__":
    main()
