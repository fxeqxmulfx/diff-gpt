"""
Decimal-precision variant of diff_bench.py.

Same sin/cos reference, but the encoder/decoder run in arbitrary-precision
Decimal arithmetic (use_decimal=True). Slower; used to stress-test the
tokenization numerics without float64's 53-bit cumsum floor. The GPT side
still trains in float32 — the Decimal work is in encode/decode around it.
"""

import numpy as np
import pandas as pd
import torch

from diff_gpt.data_loader import DiffDataFrameDataLoader
from diff_gpt.diff_gpt import DiffGPT
from diff_gpt.encoder_decoder import get_domain_of_definition, np_to_decimal
from diff_gpt.model.gpt import GPT


def mae(x: np.ndarray, y: np.ndarray) -> float:
    result = np.mean(np.abs(x - y), axis=0)
    return float(np.mean(result))


# ADD diff encoder decoder: val_loss=0.18 train_time=27.0 MAE=0.0061
# FIX diff encoder decoder: val_loss=0.19 train_time=27.0 MAE=0.0041
# UPD use round instead floor: val_loss=0.19 train_time=26.67 MAE=0.0016
# FIX shift in diff encoder decoder: val_loss=0.19 train_time=28.0 MAE=0.0016
# CLR train: val_loss=0.19 train_time=33.67 MAE=0.0024


def main() -> None:
    torch.set_float32_matmul_precision("high")
    n_embd = 64
    block_size = 32
    n_head = 4
    n_layer = 4
    learning_rate = 1e-2
    betas = (0.9, 0.95)
    weight_decay = 0.1
    max_iters = 5_000
    eval_interval = 5_000
    eval_iters = 200
    batch_size = 16
    rope_theta = 10000
    vocab_size = 256
    swiglu_alpha = 1.702
    swiglu_limit = 7.0
    train_part = 0.9
    use_checkpoint = True
    use_tqdm = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    idx = np.arange(1_000_000, dtype=np.float64)
    inp = pd.DataFrame(
        {
            "sin": np.sin(idx),
            "cos": np.cos(idx),
        }
    )
    domain_of_definition = get_domain_of_definition(
        np_to_decimal(inp.to_numpy(dtype=np.float64)),
        order_of_derivative=1,
        use_decimal=True,
    )
    n = 3
    val_loss_array = torch.zeros((n,))
    train_time_array = torch.zeros((n,))
    mae_loss_array = torch.zeros((n,))
    for i in range(n):
        base_model = GPT(
            vocab_size=vocab_size,
            n_embd=n_embd,
            block_size=block_size,
            n_head=n_head,
            n_layer=n_layer,
            rope_theta=rope_theta,
            swiglu_alpha=swiglu_alpha,
            swiglu_limit=swiglu_limit,
            use_checkpoint=use_checkpoint,
        )
        base_model.to(device=device).compile(mode="max-autotune-no-cudagraphs")
        model = DiffGPT(
            base_model,
            order_of_derivative=1,
            domain_of_definition=domain_of_definition,
            use_decimal=True,
        )
        loader = DiffDataFrameDataLoader(
            dfs=[inp],
            block_size=base_model.block_size,
            batch_size=batch_size,
            vocab_size=vocab_size,
            order_of_derivative=1,
            domain_of_definition=domain_of_definition,
            use_decimal=True,
            device=device,
            train_part=train_part,
        )
        val_loss, train_time = model.train(
            loader=loader,
            learning_rate=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
            max_iters=max_iters,
            eval_interval=eval_interval,
            eval_iters=eval_iters,
            use_tqdm=use_tqdm,
        )
        result_df = model.predict(
            df=inp[-16:-8],
            max_new_points=8,
            sampler=None,
        )
        val_loss_array[i] = val_loss
        train_time_array[i] = train_time
        mae_loss_array[i] = mae(
            np_to_decimal(inp[-8:].to_numpy()), result_df[-8:].to_numpy()
        )
    val_loss = float(torch.mean(val_loss_array))
    train_time = float(torch.mean(train_time_array))
    mae_loss = float(torch.mean(mae_loss_array))
    print(
        f"val_loss={round(val_loss, 2)} train_time={round(train_time, 2)} MAE={round(mae_loss, 4)}"
    )


if __name__ == "__main__":
    main()
