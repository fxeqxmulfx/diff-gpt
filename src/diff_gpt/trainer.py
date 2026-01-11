import numpy as np
import torch
import gc
import sys
from tqdm import tqdm

from diff_gpt.data_loader import DiffSQLiteDataLoader
from diff_gpt.model.gpt import BaseGPT, GPT
from diff_gpt.optimizer.adamw_schedulefree import AdamWScheduleFree


def get_hardware_constraints(device_type: str) -> float:
    if device_type == "cpu":
        return 32.0
    props = torch.cuda.get_device_properties(0)
    return props.total_memory / (1024**3)


def auto_tune_params(
    model: BaseGPT,
    vram_gb: float,
    use_checkpoint: bool,
) -> tuple[int, int, int]:
    params = sum(p.numel() for p in model.parameters())
    L = model.block_size
    H = model.n_embd
    N = model.n_layer
    reserved = 1.5
    available_mem = (vram_gb - reserved) * (1024**3)
    static_mem = params * 20
    ATTN_HEURISTIC_FACTOR = 14
    if use_checkpoint:
        dynamic_mem_per_seq = (2 * N + ATTN_HEURISTIC_FACTOR) * H * L
    else:
        dynamic_mem_per_seq = ATTN_HEURISTIC_FACTOR * N * H * L
    remaining = available_mem - static_mem
    if remaining <= 0:
        print("Error: Model too big for GPU!")
        sys.exit(1)
    mini_batch_size = int(remaining / dynamic_mem_per_seq)
    mini_batch_size = max(1, min(32, mini_batch_size))
    target_tokens = max(32_000, int(params * 0.003))
    target_seqs = target_tokens // L
    grad_accum_size = max(1, target_seqs // mini_batch_size)
    tokens_per_step = mini_batch_size * grad_accum_size * L
    return mini_batch_size, grad_accum_size, tokens_per_step


def train(
    db_path: str,
    vocab_size: int,
    order_of_derivative: int,
    domain_of_definition: np.ndarray,
    model_path: str,
    block_size: int = 512,
    n_layer: int = 8,
    n_head: int = 8,
    n_embd: int = 512,
    use_decimal: bool = False,
    lr: float = 1e-3,
    weight_decay: float = 0.1,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_checkpoint = True
    model = GPT(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        use_checkpoint=use_checkpoint,
    ).to(device)
    vram = get_hardware_constraints(device)
    mini_batch_size, grad_accum_size, tokens_per_step = auto_tune_params(
        model=model,
        vram_gb=vram,
        use_checkpoint=use_checkpoint,
    )
    if mini_batch_size == 1 and grad_accum_size < 1:
        print("Warning: Very constrained memory, training may be slow")
    print(f"--- Config ({vram:.1f}GB GPU) ---")
    print(f"Micro-Batch: {mini_batch_size} | Grad Accum: {grad_accum_size}")
    loader = DiffSQLiteDataLoader(
        block_size=model.block_size,
        batch_size=mini_batch_size,
        vocab_size=vocab_size,
        order_of_derivative=order_of_derivative,
        domain_of_definition=domain_of_definition,
        use_decimal=use_decimal,
        device=device,
        database=db_path,
    )
    total_tokens = sum(loader.tables_tokens_len.values())
    max_iters = max(1000, int(total_tokens / tokens_per_step))
    print(f"Max Iters: {max_iters}")
    optimizer = AdamWScheduleFree(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    model.train()
    optimizer.train()
    pbar = tqdm(range(max_iters))
    step = 0
    try:
        for step in pbar:
            step_completed = False
            while not step_completed:
                try:
                    optimizer.zero_grad()
                    accum_loss = 0.0
                    for _ in range(grad_accum_size):
                        X, Y = loader.get_batch()
                        X, Y = X.contiguous(), Y.contiguous()
                        _, loss = model(X, Y)
                        loss = loss / grad_accum_size
                        loss.backward()
                        accum_loss += loss.item()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    step_completed = True
                    pbar.set_description(
                        f"Loss: {accum_loss:.4f} | MBS: {mini_batch_size}"
                    )
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        optimizer.zero_grad()
                        torch.cuda.empty_cache()
                        gc.collect()
                        if mini_batch_size <= 1:
                            print("Error: Batch size is 1 and still OOM. Cannot train.")
                            raise e
                        old_mbs = mini_batch_size
                        mini_batch_size = max(1, mini_batch_size // 2)
                        grad_accum_size *= 2
                        loader.batch_size = mini_batch_size
                        print(
                            f"\n[OOM] Restarting step. Reduced MBS: {old_mbs} -> {mini_batch_size}"
                        )
                    else:
                        raise e
            if vram < 6 and step % 50 == 0:
                gc.collect()
                torch.cuda.empty_cache()
    finally:
        loader.close()
        model.eval()
        optimizer.eval()
        if step > 0:
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")
