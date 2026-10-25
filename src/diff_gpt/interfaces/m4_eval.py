"""The command line of diff_gpt.application.m4_eval."""

import os
import sys

from diff_gpt.application.m4_eval import Settings, evaluate
from diff_gpt.interfaces.m4_bench import read_data_settings


def read_settings() -> Settings:
    """Read the settings of the use case from the environment."""
    readouts = os.environ.get("M4_EVAL_READOUTS")
    return Settings(
        data=read_data_settings(),
        vocab_size=int(os.environ.get("M4_VOCAB_SIZE", "256")),
        n_embd=int(os.environ.get("M4_N_EMBD", "64")),
        n_head=int(os.environ.get("M4_N_HEAD", "4")),
        n_layer=int(os.environ.get("M4_N_LAYER", "2")),
        device=os.environ.get("M4_EVAL_DEVICE", "cpu"),
        eval_readouts=readouts,
        n_samples=int(os.environ.get("M4_NUM_SAMPLES", "100")) if readouts else 0,
        eval_summary_only=os.environ.get("M4_EVAL_SUMMARY_ONLY"),
    )


def main() -> None:
    """
    Evaluate the checkpoints given as arguments and print diagnostics.

    Prints a median-path summary per checkpoint. Then, if M4_EVAL_READOUTS is
    set, prints Bayes readouts and proper scores from M4_NUM_SAMPLES
    trajectories; else, unless M4_EVAL_SUMMARY_ONLY is set, prints paired
    per-series comparisons of every checkpoint against the first one.
    M4_EVAL_DEVICE selects the device (default cpu).
    """
    paths = sys.argv[1:]
    if not paths:
        msg = "need at least one checkpoint path"
        raise SystemExit(msg)
    evaluate(paths, read_settings())


if __name__ == "__main__":
    main()
