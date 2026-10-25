"""The command line of diff_gpt.application.m4_bench."""

import os

import torch

from diff_gpt.application.m4_bench import DataSettings, Settings, bench


def read_data_settings() -> DataSettings:
    """Read the settings of prepare from the environment."""
    lags = os.environ.get("M4_LAGS")
    return DataSettings(
        freq=os.environ.get("M4_FREQ", "Hourly"),
        n_series=int(os.environ.get("M4_N_SERIES", "50")),
        min_len=int(os.environ.get("M4_MIN_LEN", "0")),
        train_part=float(os.environ.get("M4_TRAIN_PART", "0.8")),
        target_bs=int(os.environ.get("M4_BLOCK_SIZE", "256")),
        lags=lags,
        order=None if lags else int(os.environ.get("M4_ORDER", "1")),
        dq=os.environ.get("M4_DOMAIN_QUANTILE"),
    )


def read_settings() -> Settings:
    """Read the settings of the use case from the environment."""
    return Settings(
        data=read_data_settings(),
        seed=int(os.environ.get("M4_SEED", "0")),
        vocab_size=int(os.environ.get("M4_VOCAB_SIZE", "256")),
        n_embd=int(os.environ.get("M4_N_EMBD", "64")),
        n_layer=int(os.environ.get("M4_N_LAYER", "2")),
        n_head=int(os.environ.get("M4_N_HEAD", "4")),
        compile_mode=(
            os.environ.get("M4_COMPILE_MODE", "max-autotune")
            if torch.cuda.is_available()
            else None
        ),
        max_iters=int(os.environ.get("M4_MAX_ITERS", "20000")),
        eval_interval=int(os.environ.get("M4_EVAL_INTERVAL", "500")),
        train_series=int(os.environ.get("M4_TRAIN_SERIES", "0")),
        save_path=os.environ.get("M4_SAVE"),
        sampler_names=os.environ.get("M4_SAMPLERS", "argmax").split(","),
    )


def main() -> None:
    """
    Train one global diff-gpt on the picked M4 series and print its scores.

    Settings come from the M4_* environment variables (see the module
    docstring); sMAPE and MASE are printed for each readout in M4_SAMPLERS.
    """
    # Full FP32 matmuls: TF32 rounds their inputs to 10 bits of mantissa.
    torch.set_float32_matmul_precision("highest")
    bench(read_settings())


if __name__ == "__main__":
    main()
