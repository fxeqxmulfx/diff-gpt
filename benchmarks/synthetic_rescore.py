r"""
Score a synthetic_pretrain checkpoint again, on fresh holdout windows.

While it trains, synthetic_pretrain scores fixed holdout windows: --score-windows
per source, --horizon points after a context that fills the rest of the block.
This scores one of its checkpoints on other windows of the holdout split, of
any context C and horizon H that fit the block, C - p + H tokens with p the
rows of the difference prefix:
- --per-source windows of every source, drawn with --window-seed (a seed
  other than the run's --score-seed gives fresh windows);
- --num-samples samples of each at the run's temperature, or --temperature;
- the scores of synthetic_pretrain's `forecast_scores`: the MASE of the
  median and GIFT-Eval's CRPS (the weighted quantile loss), each relative to
  the naive forecast and combined over the sources by a geometric mean;
  --score-samples also scores fewer samples, from disjoint groups of the
  ones drawn.

The checkpoint is read straight onto --device, whichever device saved it, so
a CPU process that sees no GPU rescores a GPU run. Settings are chosen on the
holdout alone: no GIFT-Eval or TIME data enters this script. The holdout is
the run's corpus, and a manifest that changed since the run is refused.
--output, a JSON file that must not exist yet, gets the scores of every
source, the settings and the decoding time.

Usage:
    uv run --no-sync python -m benchmarks.synthetic_rescore \
        --run runs/synthetic-pretrain/pilot --context 512 --horizon 720 \
        --output runs/synthetic-pretrain/pilot/rescore-c512-h720.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from benchmarks.synthetic_pretrain import forecast_scores, holdout_windows, load_split
from diff_gpt.diff_gpt import DiffGPT
from diff_gpt.encoder_decoder import prefix_rows
from diff_gpt.model.gpt import GPT
from diff_gpt.normalization import predict_samples_normalized
from diff_gpt.sampler.temperature import TemperatureSampler

ROOT = Path(__file__).resolve().parents[1]


def load_checkpoint(
    run: Path,
    checkpoint: str,
    device: str,
) -> tuple[DiffGPT, dict[str, Any]]:
    """
    Return the run's DiffGPT holding the weights of `checkpoint`, and its config.

    The weights load onto `device` in eval mode. As DiffGPT.load, a file
    whose encoder settings differ from the run's raises ValueError.
    """
    config = json.loads((run / "config.json").read_text())
    args = config["args"]
    gpt = GPT(
        vocab_size=args["vocab"],
        n_embd=args["width"],
        block_size=args["block_size"],
        n_head=args["heads"],
        n_layer=args["layers"],
    ).to(device)
    model = DiffGPT(
        model=gpt,
        order_of_derivative=args["order"],
        domain_of_definition=np.array(config["domain_of_definition"]),
        tokenizer=config["tokenizer"],
    )
    # DiffGPT.load reads tensors onto the device that saved them, which a
    # process without that GPU cannot: map them here, with load's own check.
    saved = torch.load(run / checkpoint, map_location=device, weights_only=True)
    current = model._encoder_settings()  # noqa: SLF001 - DiffGPT.load's check
    differing = [
        name for name, value in current.items() if saved["encoder"].get(name) != value
    ]
    if differing:
        msg = f"{run / checkpoint} holds other encoder settings: {differing}"
        raise ValueError(msg)
    gpt.load_state_dict(saved["model"])
    gpt.eval()
    return model, config


def rescore(
    model: DiffGPT,
    config: dict[str, Any],
    *,
    per_source: int,
    window_seed: int,
    context: int,
    horizon: int,
    num_samples: int,
    score_samples: list[int],
    temperature: float,
    batch_size: int,
    sample_seed: int,
    dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    """
    Draw and score the forecasts of fresh windows of the run's holdout split.

    Returns `forecast_scores` of all the samples ("scores") and of each size
    in `score_samples` ("fewer_samples"), the number of windows, and the
    seconds and generated tokens per second of the draw.
    """
    args = config["args"]
    holdout = load_split(Path(args["corpus"]), "holdout")
    names = list(holdout)
    windows = holdout_windows(holdout, per_source, context, horizon, window_seed)
    began = time.monotonic()
    samples = predict_samples_normalized(
        model,
        [c[:, None] for c in windows.contexts],
        max_new_points=horizon,
        num_samples=num_samples,
        method=args["normalization"],
        sampler=TemperatureSampler(temperature=temperature),
        batch_size=batch_size,
        seed=sample_seed,
        dtype=dtype,
    )
    seconds = time.monotonic() - began
    drawn = np.stack(samples)[..., 0]
    return {
        "scores": forecast_scores(drawn, windows, names),
        "fewer_samples": {
            str(size): forecast_scores(drawn, windows, names, group=size)
            for size in score_samples
        },
        "windows": len(windows.sources),
        "decode_seconds": seconds,
        "tokens_per_second": drawn.size / seconds,
    }


def _git_head() -> str:
    return subprocess.check_output(  # noqa: S603 - fixed git argv, no shell.
        [shutil.which("git") or "/usr/bin/git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run", type=Path, required=True, help="the run's --output")
    parser.add_argument("--checkpoint", default="last.bin", help="a file of --run")
    parser.add_argument("--output", type=Path, required=True, help="a new JSON file")
    parser.add_argument("--per-source", type=int, default=64)
    parser.add_argument("--window-seed", type=int, default=1)
    parser.add_argument(
        "--context",
        type=int,
        help="context rows; by default the rest of the block after --horizon",
    )
    parser.add_argument("--horizon", type=int, help="by default the run's --horizon")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument(
        "--score-samples",
        type=int,
        nargs="*",
        default=[],
        help="also score forecasts of each of these numbers of samples, at most "
        "--num-samples, from disjoint groups of the ones drawn",
    )
    parser.add_argument("--temperature", type=float, help="by default the run's")
    parser.add_argument("--batch-size", type=int, default=1000, help="decode rows")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Rescore the checkpoint as the arguments say and write the JSON record."""
    parser = _parser()
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"{args.output} exists")
    config = json.loads((args.run / "config.json").read_text())
    run_args = config["args"]
    block_size = run_args["block_size"]
    prefix = prefix_rows(run_args["order"])
    horizon = run_args["horizon"] if args.horizon is None else args.horizon
    context = block_size + prefix - horizon if args.context is None else args.context
    if horizon < 1 or context <= prefix or context - prefix + horizon > block_size:
        parser.error(
            f"a context of {context} rows and {horizon} points need {prefix} < C, "
            f"1 <= H and C - {prefix} + H <= {block_size}, the block",
        )
    if any(not 1 <= size <= args.num_samples for size in args.score_samples):
        parser.error(f"--score-samples must lie in [1, {args.num_samples}]")
    manifest = Path(run_args["corpus"]).parent / "manifest.json"
    digest = None
    if manifest.exists():
        digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if config["manifest_sha256"] not in {None, digest}:
        parser.error(f"{manifest} is not the corpus manifest the run trained with")
    model, config = load_checkpoint(args.run, args.checkpoint, args.device)
    temperature = args.temperature
    if temperature is None:
        temperature = run_args["temperature"]
    result = rescore(
        model,
        config,
        per_source=args.per_source,
        window_seed=args.window_seed,
        context=context,
        horizon=horizon,
        num_samples=args.num_samples,
        score_samples=args.score_samples,
        temperature=temperature,
        batch_size=args.batch_size,
        sample_seed=args.sample_seed,
        dtype=None if args.dtype == "float32" else getattr(torch, args.dtype),
    )
    record = {
        "run": str(args.run),
        "checkpoint": args.checkpoint,
        "context": context,
        "horizon": horizon,
        "per_source": args.per_source,
        "window_seed": args.window_seed,
        "num_samples": args.num_samples,
        "sample_seed": args.sample_seed,
        "temperature": temperature,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "device": args.device,
        **result,
        "git_head": _git_head(),
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "ld_preload": os.environ.get("LD_PRELOAD"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=1) + "\n")
    scores = result["scores"]
    print(
        f"{args.run / args.checkpoint}: C {context}, H {horizon}, "
        f"{result['windows']} windows, S {args.num_samples}: "
        f"mase {scores['mase']:.4f}, crps {scores['crps']:.4f}, "
        f"{result['decode_seconds']:.1f} s",
    )


if __name__ == "__main__":
    main()
