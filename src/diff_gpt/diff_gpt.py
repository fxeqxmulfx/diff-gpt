import numpy as np
import pandas as pd
import torch

from diff_gpt.model.gpt import BaseGPT
from diff_gpt.sampler.sampler import Sampler
from diff_gpt.train import DataLoader, train
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
        loader: DataLoader,
        learning_rate: float = 1e-2,
        betas: tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.1,
        max_iters: int = 5_000,
        eval_interval: int = 5_000,
        eval_iters: int = 200,
        use_tqdm: bool = True,
        use_early_stopping: bool = True,
        grad_accum_steps: int = 1,
        grad_clip_norm: float | None = None,
    ) -> tuple[float, int]:
        """
        Train the underlying GPT against batches produced by `loader`.

        The loader owns the data-source details (SQLite streaming, in-memory
        DataFrames, a flat token sequence) and handles any encoding, splits,
        and loss masking. DiffGPT only drives the optimization loop and
        holds the encoder metadata used by `predict`.
        """
        return train(
            mut_model=self.model,
            loader=loader,
            learning_rate=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
            max_iters=max_iters,
            eval_interval=eval_interval,
            eval_iters=eval_iters,
            use_tqdm=use_tqdm,
            use_early_stopping=use_early_stopping,
            grad_accum_steps=grad_accum_steps,
            grad_clip_norm=grad_clip_norm,
        )

    @torch.inference_mode()
    def predict(
        self,
        df: pd.DataFrame,
        max_new_points: int | None = None,
        future_covariates_df: pd.DataFrame | None = None,
        target_columns: "list[str] | list[int] | None" = None,
        sampler: Sampler | None = None,
    ) -> pd.DataFrame:
        """
        Forecast future rows from an observed context.

        Two modes:
          1. Autoregressive (default): generate ALL columns of
             `max_new_points` rows by sampling each token. `max_new_points`
             is required; `future_covariates_df` and `target_columns` must
             be None.
          2. TimeXer-style teacher-forced: pass `future_covariates_df` and
             `target_columns`. Non-target columns in the future are
             teacher-forced from `future_covariates_df`; only target
             columns are sampled. `max_new_points` is derived from
             `len(future_covariates_df)` (or must match it if given).

        Returns a DataFrame with (`len(df)` + pred_len) rows and the same
        columns as `df`.
        """
        if future_covariates_df is None:
            assert target_columns is None, (
                "target_columns is only meaningful with future_covariates_df"
            )
            assert max_new_points is not None, (
                "max_new_points is required when future_covariates_df is None"
            )
            return self._predict_autoregressive(
                df=df, max_new_points=max_new_points, sampler=sampler
            )
        # Teacher-forced path.
        assert target_columns is not None and len(target_columns) > 0, (
            "target_columns must be non-empty when future_covariates_df is given"
        )
        pred_len = len(future_covariates_df)
        assert max_new_points is None or max_new_points == pred_len, (
            f"max_new_points ({max_new_points}) must equal "
            f"len(future_covariates_df) ({pred_len})"
        )
        return self._predict_teacher_forced(
            context_df=df,
            future_covariates_df=future_covariates_df,
            target_columns=target_columns,
            sampler=sampler,
        )

    def _predict_autoregressive(
        self,
        df: pd.DataFrame,
        max_new_points: int,
        sampler: Sampler | None,
    ) -> pd.DataFrame:
        columns = df.shape[1]
        total_positions = (df.shape[0] + max_new_points - 1) * columns
        assert total_positions <= self.model.block_size, (
            f"prompt rows ({df.shape[0]}) + new points ({max_new_points}) with "
            f"{columns} columns require {total_positions} context positions, "
            f"but block_size is {self.model.block_size}"
        )
        vocab_size = self.model.vocab_size
        device = self.model.device_type
        inp = df.to_numpy(dtype=np.float64)
        if self.use_decimal:
            inp = np_to_decimal(inp)
        start, scale, encoded_data = encode(
            inp=inp,
            vocab_size=vocab_size,
            domain_of_definition=self.domain_of_definition,
            order_of_derivative=self.order_of_derivative,
            use_decimal=self.use_decimal,
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
            order_of_derivative=self.order_of_derivative,
            use_decimal=self.use_decimal,
        )
        return pd.DataFrame(decoded, columns=df.columns)

    def _predict_teacher_forced(
        self,
        context_df: pd.DataFrame,
        future_covariates_df: pd.DataFrame,
        target_columns: "list[str] | list[int]",
        sampler: Sampler | None,
    ) -> pd.DataFrame:
        columns_list = list(context_df.columns)
        assert list(future_covariates_df.columns) == columns_list, (
            "context and future must have the same columns"
        )
        n_features = len(columns_list)

        # Resolve names / ints → index set.
        target_set: set[int] = set()
        for c in target_columns:
            if isinstance(c, str):
                assert c in columns_list, (
                    f"target column {c!r} not found in {columns_list}"
                )
                target_set.add(columns_list.index(c))
            else:
                target_set.add(int(c))

        ctx_len = len(context_df)
        pred_len = len(future_covariates_df)
        order = self.order_of_derivative
        device = self.model.device_type

        total_tokens = (ctx_len + pred_len - order) * n_features
        assert total_tokens <= self.model.block_size, (
            f"ctx={ctx_len} pred={pred_len} cols={n_features} (order={order}) "
            f"need {total_tokens} tokens but block_size={self.model.block_size}"
        )

        # Build the full concatenated input; zero out target columns in the
        # future portion so their encoded tokens are placeholders that we'll
        # overwrite during sampling.
        full_df = pd.concat(
            [context_df, future_covariates_df], ignore_index=True
        ).copy()
        for c in target_set:
            full_df.iloc[ctx_len:, c] = 0.0

        inp = full_df.to_numpy(dtype=np.float64)
        if self.use_decimal:
            inp = np_to_decimal(inp)
        start, scale, encoded_full = encode(
            inp=inp,
            vocab_size=self.model.vocab_size,
            domain_of_definition=self.domain_of_definition,
            order_of_derivative=order,
            use_decimal=self.use_decimal,
        )
        encoded_flat = encoded_full.reshape(-1).astype(np.int64).copy()

        # First (ctx_len - order) token rows depend only on the observed
        # history, so they're teacher-forced directly.
        ctx_token_count = (ctx_len - order) * n_features
        tokens = torch.tensor(
            encoded_flat[:ctx_token_count], dtype=torch.int64, device=device
        ).unsqueeze(0)

        # Walk the remaining positions: teacher-force runs of covariate
        # tokens, then sample at each target-column position.
        p = ctx_token_count
        while p < total_tokens:
            next_target = None
            for q in range(p, total_tokens):
                if (q % n_features) in target_set:
                    next_target = q
                    break
            if next_target is None:
                if p < total_tokens:
                    tail = torch.tensor(
                        encoded_flat[p:total_tokens],
                        dtype=torch.int64,
                        device=device,
                    ).unsqueeze(0)
                    tokens = torch.cat([tokens, tail], dim=1)
                break
            if next_target > p:
                run = torch.tensor(
                    encoded_flat[p:next_target], dtype=torch.int64, device=device
                ).unsqueeze(0)
                tokens = torch.cat([tokens, run], dim=1)
            logits, _ = self.model.forward(tokens)
            next_logit = logits[:, -1]
            if sampler is not None:
                next_id = sampler(next_logit)
            else:
                probs = torch.softmax(next_logit, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            token_int = int(next_id.item())
            encoded_flat[next_target] = token_int
            tokens = torch.cat(
                [
                    tokens,
                    torch.tensor([[token_int]], dtype=torch.int64, device=device),
                ],
                dim=1,
            )
            p = next_target + 1

        encoded_final = encoded_flat.reshape(-1, n_features)
        decoded = decode(
            start=start,
            scale=scale,
            inp=encoded_final,
            vocab_size=self.model.vocab_size,
            order_of_derivative=order,
            use_decimal=self.use_decimal,
        )
        return pd.DataFrame(decoded, columns=columns_list)

    def save(self, path: str = "diff-gpt.bin") -> None:
        torch.save(self.model.state_dict(), path)

    def load(self, path: str = "diff-gpt.bin") -> None:
        self.model.load_state_dict(torch.load(path, weights_only=True))
        self.model.eval()
