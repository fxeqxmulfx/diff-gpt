"""
Quick torch.profiler run over the text-training inner loop. Reports where
time is spent per forward/backward/optimizer step so we can see the impact
of AttnRes (block_attn_res einsums + extra RMSNorms) on end-to-end cost.

Usage (from repo root):
    uv run --no-sync python -m benchmarks.profile_text
"""

from pathlib import Path

import torch
from torch.profiler import profile, record_function, ProfilerActivity

from diff_gpt.model.gpt import GPT
from diff_gpt.optimizer.adamw_schedulefree import AdamWScheduleFree
from tests.diff_gpt.test_text import CharEncoder


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "tinyshakespeare.txt"
WARMUP_ITERS = 5
PROFILE_ITERS = 20


def main() -> None:
    import os

    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")
    # Default to CPU to match the test_text regression environment; flip with
    # PROFILE_DEVICE=cuda to see GPU hotspots.
    device = os.environ.get("PROFILE_DEVICE", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    encoder = CharEncoder(text, pad_to=64)
    encoded = encoder.encode(text)
    data = torch.tensor(encoded, dtype=torch.long)

    # Same config as test_text — benchmark scale.
    model = GPT(
        vocab_size=encoder.vocab_size,
        n_embd=64,
        block_size=32,
        n_head=4,
        n_layer=4,
        use_checkpoint=False,
    ).to(device)
    optimizer = AdamWScheduleFree(
        model.parameters(), lr=1e-2, betas=(0.9, 0.95), weight_decay=0.1
    )
    optimizer.train()
    model.train()
    block_size = model.block_size
    batch_size = 16

    def get_batch() -> tuple[torch.Tensor, torch.Tensor]:
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    # Warmup so compile/JIT effects don't dominate.
    for _ in range(WARMUP_ITERS):
        xb, yb = get_batch()
        _, loss = model(xb, yb)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=False,
        with_stack=False,
        with_modules=True,
    ) as prof:
        for _ in range(PROFILE_ITERS):
            with record_function("step"):
                xb, yb = get_batch()
                with record_function("forward"):
                    _, loss = model(xb, yb)
                with record_function("backward"):
                    loss.backward()
                with record_function("opt_step"):
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

    sort_self = "self_cuda_time_total" if device == "cuda" else "self_cpu_time_total"
    sort_total = "cuda_time_total" if device == "cuda" else "cpu_time_total"
    print("=" * 30, "Per-op by self time (top 20)", "=" * 30)
    print(
        prof.key_averages().table(sort_by=sort_self, row_limit=20)
    )
    print("=" * 30, "Per-region by total time (top 80)", "=" * 30)
    print(
        prof.key_averages().table(
            sort_by=sort_total,
            row_limit=80,
            max_name_column_width=60,
        )
    )


if __name__ == "__main__":
    main()
