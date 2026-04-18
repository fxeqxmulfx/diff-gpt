import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from diff_gpt.model.engine import Engine
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
        order_of_derivative: int | np.ndarray,
        domain_of_definition: np.ndarray,
        use_decimal: bool,
    ) -> None:
        """
        `order_of_derivative`: scalar for uniform k across columns, or a
        length-F array for per-column k.
        """
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
        save_best: bool = True,
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
            save_best=save_best,
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

    def _generate_future_samples(
        self,
        df: pd.DataFrame,
        max_new_points: int,
        sampler: Sampler | None,
        num_samples: int,
    ) -> np.ndarray:
        """Generate `num_samples` parallel future trajectories.

        Returns ndarray of shape (num_samples, max_new_points, n_features)
        containing only the predicted future rows (no context).

        Use `sampler=TemperatureSampler(temperature=0.0)` for argmax (mode);
        `temperature=1.0` for proper samples from the learned distribution.
        """
        assert num_samples >= 1
        columns = df.shape[1]
        order = self.order_of_derivative
        if isinstance(order, (int, np.integer)):
            max_order = int(order)
        else:
            order_arr = np.asarray(order)
            assert order_arr.shape == (columns,), (
                f"order_of_derivative shape {order_arr.shape} must match columns {columns}"
            )
            max_order = int(order_arr.max())
        assert max_order <= df.shape[0], (
            f"max order {max_order} > context length {df.shape[0]}: "
            f"not enough history to compute the k-th derivative prefix"
        )
        total_positions = (df.shape[0] + max_new_points - max_order) * columns
        assert total_positions <= self.model.block_size, (
            f"prompt rows ({df.shape[0]}) + new points ({max_new_points}) with "
            f"{columns} columns (max_order={max_order}) require {total_positions} "
            f"context positions, but block_size is {self.model.block_size}"
        )
        vocab_size = self.model.vocab_size
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
        context_tokens = encoded_data.reshape(-1).astype(np.int64).tolist()
        # Engine.generate_batch natively expands one prompt into num_samples
        # parallel trajectories (KV-cache-accelerated).
        engine = Engine(model=self.model)
        results, _ = engine.generate_batch(
            tokens=context_tokens,
            num_samples=num_samples,
            max_tokens=max_new_points * columns,
            sampler=sampler,
        )
        futures = np.empty(
            (num_samples, max_new_points, columns), dtype=np.float64
        )
        for s in range(num_samples):
            generated = np.asarray(results[s], dtype=np.int64).reshape(-1, columns)
            decoded = decode(
                start=start,
                scale=scale,
                inp=generated,
                vocab_size=vocab_size,
                order_of_derivative=self.order_of_derivative,
                use_decimal=self.use_decimal,
            )
            futures[s] = np.asarray(
                decoded[df.shape[0] : df.shape[0] + max_new_points],
                dtype=np.float64,
            )
        return futures

    def _predict_autoregressive(
        self,
        df: pd.DataFrame,
        max_new_points: int,
        sampler: Sampler | None,
    ) -> pd.DataFrame:
        future = self._generate_future_samples(
            df=df,
            max_new_points=max_new_points,
            sampler=sampler,
            num_samples=1,
        )[0]
        full = np.concatenate([df.to_numpy(dtype=np.float64), future], axis=0)
        return pd.DataFrame(full, columns=df.columns)

    @torch.inference_mode()
    def predict_samples(
        self,
        df: pd.DataFrame,
        max_new_points: int,
        num_samples: int,
        sampler: Sampler | None = None,
    ) -> np.ndarray:
        """Probabilistic forecasting: return `num_samples` future trajectories.

        Each trajectory is an independent sample from the model's learned
        joint distribution over the `max_new_points`-step future. Use a
        temperature-1 sampler (or nucleus top-p) to get proper samples;
        temperature-0 collapses all trajectories to the mode and is only
        useful as a smoke test.

        Returns ndarray of shape (num_samples, max_new_points, n_features).
        """
        return self._generate_future_samples(
            df=df,
            max_new_points=max_new_points,
            sampler=sampler,
            num_samples=num_samples,
        )

    @torch.inference_mode()
    def anomaly_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-(time, channel) anomaly scores via conditional negative
        log-likelihood.

        Encodes the signal to tokens, runs a single forward pass, and
        returns −log p(token_t | ctx) at every position — a strictly
        proper score under the model's learned distribution. Higher
        values mean "more surprising given the preceding context" and
        are the natural calibrated anomaly signal.

        Shape: (T, F), same columns and index as `df`. Positions that
        cannot be scored are NaN: the first `max_k` rows are the start
        prefix (no derivative available), and the very first encoded
        token has no preceding context to condition on.
        """
        columns = df.shape[1]
        T = df.shape[0]
        order = self.order_of_derivative
        if isinstance(order, (int, np.integer)):
            max_order = int(order)
        else:
            order_arr = np.asarray(order)
            assert order_arr.shape == (columns,), (
                f"order_of_derivative shape {order_arr.shape} must match columns {columns}"
            )
            max_order = int(order_arr.max())
        assert max_order <= T, (
            f"max order {max_order} > signal length {T}"
        )

        inp = df.to_numpy(dtype=np.float64)
        if self.use_decimal:
            inp = np_to_decimal(inp)
        _, _, encoded = encode(
            inp=inp,
            vocab_size=self.model.vocab_size,
            domain_of_definition=self.domain_of_definition,
            order_of_derivative=self.order_of_derivative,
            use_decimal=self.use_decimal,
        )
        N = encoded.size
        assert N <= self.model.block_size, (
            f"signal has {N} tokens but block_size is {self.model.block_size}; "
            f"chunk the input manually for longer sequences"
        )
        device = torch.device(self.model.device_type)
        tokens = torch.as_tensor(
            encoded.reshape(-1).astype(np.int64), dtype=torch.int64, device=device
        ).unsqueeze(0)  # (1, N)
        was_training = self.model.training
        self.model.eval()
        logits, _ = self.model.forward(tokens)  # (1, N, V)
        if was_training:
            self.model.train()
        # logits[i] predicts the token at position i+1.
        log_probs = F.log_softmax(logits[0, :-1], dim=-1)  # (N-1, V)
        nll = -log_probs.gather(
            -1, tokens[0, 1:].unsqueeze(-1)
        ).squeeze(-1)  # (N-1,)
        nll_np = nll.detach().cpu().numpy().astype(np.float64)

        flat = np.full(N, np.nan, dtype=np.float64)
        flat[1:] = nll_np
        scores_encoded = flat.reshape(encoded.shape[0], columns)

        out = np.full((T, columns), np.nan, dtype=np.float64)
        out[max_order:] = scores_encoded
        return pd.DataFrame(out, columns=df.columns, index=df.index)

    @torch.inference_mode()
    def predict_quantiles(
        self,
        df: pd.DataFrame,
        max_new_points: int,
        quantiles: "list[float] | tuple[float, ...]" = (0.1, 0.5, 0.9),
        num_samples: int = 100,
        sampler: Sampler | None = None,
    ) -> dict[float, pd.DataFrame]:
        """Empirical quantile forecasts from `num_samples` Monte-Carlo trajectories.

        Returns a dict mapping each requested quantile α ∈ (0, 1) to a
        DataFrame of shape (max_new_points, n_features) containing the
        per-step empirical α-quantile across the samples. Defaults produce
        the standard P10/P50/P90 bands for probabilistic forecasting.
        """
        for q in quantiles:
            assert 0.0 < q < 1.0, f"quantile {q} must be in (0, 1)"
        futures = self._generate_future_samples(
            df=df,
            max_new_points=max_new_points,
            sampler=sampler,
            num_samples=num_samples,
        )
        # futures: (N, H, F). Reduce across samples axis.
        out: dict[float, pd.DataFrame] = {}
        for q in quantiles:
            q_vals = np.quantile(futures, q, axis=0)  # (H, F)
            out[float(q)] = pd.DataFrame(q_vals, columns=df.columns)
        return out

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
        # With per-column k the encoder aligns all columns to the shortest
        # valid window, which is (T - max_k). Treat `order` as scalar or
        # take its max for the token-count calculus.
        if isinstance(order, (int, np.integer)):
            max_order = int(order)
        else:
            order_arr = np.asarray(order)
            assert order_arr.shape == (n_features,), (
                f"order_of_derivative shape {order_arr.shape} must match n_features {n_features}"
            )
            max_order = int(order_arr.max())
        assert max_order <= ctx_len, (
            f"max order {max_order} > context length {ctx_len}: "
            f"not enough history to compute the k-th derivative prefix"
        )

        total_tokens = (ctx_len + pred_len - max_order) * n_features
        assert total_tokens <= self.model.block_size, (
            f"ctx={ctx_len} pred={pred_len} cols={n_features} (max_order={max_order}) "
            f"need {total_tokens} tokens but block_size={self.model.block_size}"
        )

        # Build the full concatenated input; zero out target columns in the
        # future portion so their encoded tokens are placeholders. Covariate
        # entries in encoded_full are correct (both endpoints known); target
        # entries are bogus but we'll overwrite them during generation.
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

        # First (ctx_len - max_order) token rows depend only on the observed
        # history — that's the prefill prompt. The remaining positions are
        # driven step-by-step via a force_schedule: None at target-column
        # positions (sample), the known covariate token otherwise.
        ctx_token_count = (ctx_len - max_order) * n_features
        prompt_tokens = encoded_flat[:ctx_token_count].tolist()
        force_schedule: list[int | None] = [
            None if (p % n_features) in target_set else int(encoded_flat[p])
            for p in range(ctx_token_count, total_tokens)
        ]
        max_tokens = total_tokens - ctx_token_count

        engine = Engine(model=self.model)
        results, _ = engine.generate_batch(
            tokens=prompt_tokens,
            num_samples=1,
            max_tokens=max_tokens,
            sampler=sampler,
            force_schedule=force_schedule,
        )
        encoded_final = np.asarray(results[0], dtype=np.int64).reshape(
            -1, n_features
        )
        decoded = decode(
            start=start,
            scale=scale,
            inp=encoded_final,
            vocab_size=self.model.vocab_size,
            order_of_derivative=self.order_of_derivative,
            use_decimal=self.use_decimal,
        )
        return pd.DataFrame(decoded, columns=columns_list)

    def save(self, path: str = "diff-gpt.bin") -> None:
        torch.save(self.model.state_dict(), path)

    def load(self, path: str = "diff-gpt.bin") -> None:
        self.model.load_state_dict(torch.load(path, weights_only=True))
        self.model.eval()
