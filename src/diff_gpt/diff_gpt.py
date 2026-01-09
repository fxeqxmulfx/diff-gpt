import numpy as np
import pandas as pd
import torch

from diff_gpt.model.gpt import BaseGPT
from diff_gpt.sampler.sampler import Sampler
from diff_gpt.train import train
from diff_gpt.encoder_decoder import decode, encode, np_to_decimal


class DiffGPT:
    __slots__ = (
        "model",
        "order_of_derivative",
        "domain_of_definition",
        "use_decimal",
    )

    def __init__(
        self,
        model: BaseGPT,
        order_of_derivative: int,
        domain_of_definition: np.ndarray,
        use_decimal: bool,
    ) -> None:
        self.model = model
        self.order_of_derivative = order_of_derivative
        self.domain_of_definition = domain_of_definition
        self.use_decimal = use_decimal

    def train(
        self,
        df: pd.DataFrame,
        learning_rate: float = 1e-2,
        betas: tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.1,
        max_iters: int = 5_000,
        eval_interval: int = 5_000,
        eval_iters: int = 200,
        batch_size: int = 16,
        train_part: float = 0.9,
        use_tqdm: bool = True,
    ) -> tuple[float, int]:
        inp = df.to_numpy(dtype=np.float64)
        use_decimal = self.use_decimal
        if use_decimal:
            inp = np_to_decimal(inp)
        order_of_derivative = self.order_of_derivative
        domain_of_definition = self.domain_of_definition
        _, _, encoded_data = encode(
            inp=inp,
            vocab_size=self.model.vocab_size,
            domain_of_definition=domain_of_definition,
            order_of_derivative=order_of_derivative,
            use_decimal=use_decimal,
        )
        encoded_data = tuple(encoded_data.reshape(-1).tolist())
        val_loss, train_time_s = train(
            mut_model=self.model,
            encoded_data=encoded_data,
            learning_rate=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
            max_iters=max_iters,
            eval_interval=eval_interval,
            eval_iters=eval_iters,
            batch_size=batch_size,
            train_part=train_part,
            use_tqdm=use_tqdm,
        )
        result = val_loss, train_time_s
        return result

    @torch.inference_mode()
    def predict(
        self,
        df: pd.DataFrame,
        max_new_points: int,
        sampler: Sampler | None,
    ) -> pd.DataFrame:
        columns = df.shape[1]
        assert (df.shape[0] + max_new_points - 1) * columns <= self.model.block_size
        vocab_size = self.model.vocab_size
        device = self.model.device_type
        use_decimal = self.use_decimal
        inp = df.to_numpy(dtype=np.float64)
        if use_decimal:
            inp = np_to_decimal(inp)
        domain_of_definition = self.domain_of_definition
        order_of_derivative = self.order_of_derivative
        start, scale, encoded_data = encode(
            inp=inp,
            vocab_size=vocab_size,
            domain_of_definition=domain_of_definition,
            order_of_derivative=order_of_derivative,
            use_decimal=use_decimal,
        )
        encoded_data = np.expand_dims(encoded_data.reshape(-1), axis=0)
        context = torch.from_numpy(encoded_data).to(device=device)
        generated = (
            self.model.generate(
                context,
                max_new_tokens=max_new_points * columns,
                sampler=sampler,
            )
            .cpu()
            .numpy()[0]
            .reshape(-1, columns)
        )
        decoded = decode(
            start=start,
            scale=scale,
            inp=generated,
            vocab_size=vocab_size,
            order_of_derivative=order_of_derivative,
            use_decimal=use_decimal,
        )
        result = pd.DataFrame(decoded, columns=df.columns)
        return result

    def save(self, path: str = "diff-gpt.bin") -> None:
        torch.save(self.model.state_dict(), path)

    def load(self, path: str = "diff-gpt.bin") -> None:
        self.model.load_state_dict(torch.load(path, weights_only=True))
        self.model.eval()
