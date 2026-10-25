"""The command line of diff_gpt.application.ett_bench."""

import os

import torch

from diff_gpt.application.ett_bench import Settings, bench


def read_settings() -> Settings:
    """Read the settings of the use case from the environment."""
    return Settings(
        compile_model=torch.cuda.is_available()
        and os.environ.get("ETT_COMPILE", "1") == "1",
        dataset=os.environ.get("ETT_DATASET", "ETTh1"),
        seq_len=int(os.environ.get("ETT_SEQ_LEN", "96")),
        pred_lens=[
            int(x) for x in os.environ.get("ETT_PRED_LENS", "96,192,336,720").split(",")
        ],
        stride_env=os.environ.get("ETT_STRIDE"),
        max_iters=int(os.environ.get("ETT_MAX_ITERS", "2000")),
        n_embd=int(os.environ.get("ETT_N_EMBD", "128")),
        n_layer=int(os.environ.get("ETT_N_LAYER", "2")),
        n_head=int(os.environ.get("ETT_N_HEAD", "4")),
        use_checkpoint=os.environ.get("ETT_USE_CHECKPOINT", "0") == "1",
        batch_size_override=os.environ.get("ETT_BATCH_SIZE"),
        seed=int(os.environ.get("ETT_SEED", "0")),
        vocab_size=int(os.environ.get("ETT_VOCAB_SIZE", "64")),
        readouts=os.environ.get("ETT_READOUTS", "argmax").split(","),
        num_samples=int(os.environ.get("ETT_NUM_SAMPLES", "50")),
        save_prefix=os.environ.get("ETT_SAVE"),
        align_windows=os.environ.get("ETT_ALIGN_WINDOWS", "0") == "1",
    )


def main() -> None:
    """
    Train and evaluate one model per horizon in ETT_PRED_LENS.

    All settings come from ETT_* environment variables; prints the per-readout
    results of every horizon as it finishes and a summary table at the end.
    """
    torch.set_float32_matmul_precision("high")
    bench(read_settings())


if __name__ == "__main__":
    main()
