# Copyright 2025 Mikhail Borisov
"""
`DiffGPT`: a time-series forecaster built from a plain GPT and the tokenizer.

DataFrames are encoded into derivative tokens (`encoder_decoder.encode`), the
GPT continues the token stream, and the generated tokens are decoded back into
values; `anomaly_scores` reads the token likelihoods instead.
"""

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from diff_gpt.encoder_decoder import decode, encode, np_to_decimal
from diff_gpt.metrics.point import bayes_point
from diff_gpt.metrics.probabilistic import crps_empirical, empirical_coverage
from diff_gpt.model.engine import Engine
from diff_gpt.numa import pin_to_node
from diff_gpt.sampler.temperature import TemperatureSampler
from diff_gpt.train import train
from diff_gpt.train_ddp import train_numa as _train_numa

if TYPE_CHECKING:
    from collections.abc import Callable

    from diff_gpt.model.gpt import BaseGPT
    from diff_gpt.sampler.sampler import Sampler
    from diff_gpt.train import DataLoader

# CRPS and empirical coverage cannot be estimated from fewer samples.
_MIN_CALIBRATION_SAMPLES = 2


class DiffGPT:
    """
    GPT forecaster over derivative-encoded time series.

    Holds the model together with the encoder settings it was trained with
    (`order_of_derivative`, `domain_of_definition`, `use_decimal`), so that
    every prediction encodes the context and decodes the output the same way.
    """

    __slots__ = (
        "domain_of_definition",
        "model",
        "order_of_derivative",
        "use_decimal",
    )

    def __init__(
        self,
        model: BaseGPT,
        order_of_derivative: int | np.ndarray,
        domain_of_definition: np.ndarray,
        *,
        use_decimal: bool,
    ) -> None:
        """
        Wrap `model` with the encoder settings used for training and prediction.

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
        *,
        use_tqdm: bool = True,
        use_early_stopping: bool = True,
        grad_accum_steps: int = 1,
        grad_clip_norm: float | None = None,
        save_best: bool = True,
        pin_node: int | None = None,
    ) -> tuple[float, int]:
        """
        Train the underlying GPT against batches produced by `loader`.

        The loader owns the data-source details (SQLite streaming, in-memory
        DataFrames, a flat token sequence) and handles any encoding, splits,
        and loss masking. DiffGPT only drives the optimization loop and
        holds the encoder metadata used by `predict`.

        `pin_node`: if set, pin this process to the given NUMA node's CPUs
        (via `numa.pin_to_node`) before training. Cheap way to cut
        cross-socket memory traffic on CPU training. For multi-node DDP
        across all NUMA nodes, use `diff_gpt.train_ddp.train_numa` instead.
        """
        if pin_node is not None:
            pin_to_node(pin_node)
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

    @staticmethod
    def train_numa(
        model_factory: Callable[[int], BaseGPT],
        loader_factory: Callable[[int, int], DataLoader],
        train_kwargs: dict | None = None,
        nodes: list[int] | None = None,
        checkpoint_path: str | None = None,
    ) -> tuple[float, int]:
        """
        Spawn one worker per NUMA node and train the inner GPT across nodes via DDP.

        Thin pass-through to `train_ddp.train_numa` — DDP wraps the bare
        `BaseGPT`, not the DiffGPT shell (which is numpy-side encoder/decoder
        metadata, not an nn.Module).

        Both factories are called *inside* each worker after that worker
        has been pinned to its NUMA node, so every allocation lands on
        local memory. Seed `loader_factory` by `rank` or all workers see
        identical batches and DDP collapses to no-speedup replication.

        Rebuild the DiffGPT around the returned weights in the caller:

            val_loss, _ = DiffGPT.train_numa(
                model_factory=build_gpt,
                loader_factory=build_loader,
                checkpoint_path="best.pt",
            )
            gpt = build_gpt(0)
            gpt.load_state_dict(torch.load("best.pt"))
            model = DiffGPT(model=gpt, order_of_derivative=...,
                            domain_of_definition=..., use_decimal=...)
        """
        return _train_numa(
            model_factory=model_factory,
            loader_factory=loader_factory,
            train_kwargs=train_kwargs,
            nodes=nodes,
            checkpoint_path=checkpoint_path,
        )

    @torch.inference_mode()
    def predict(
        self,
        df: pd.DataFrame,
        max_new_points: int | None = None,
        future_covariates_df: pd.DataFrame | None = None,
        target_columns: list[str] | list[int] | None = None,
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
            if target_columns is not None:
                msg = "target_columns is only meaningful with future_covariates_df"
                raise ValueError(msg)
            if max_new_points is None:
                msg = "max_new_points is required when future_covariates_df is None"
                raise ValueError(msg)
            return self._predict_autoregressive(
                df=df,
                max_new_points=max_new_points,
                sampler=sampler,
            )
        # Teacher-forced path.
        if target_columns is None or len(target_columns) == 0:
            msg = "target_columns must be non-empty when future_covariates_df is given"
            raise ValueError(msg)
        pred_len = len(future_covariates_df)
        if max_new_points is not None and max_new_points != pred_len:
            msg = (
                f"max_new_points ({max_new_points}) must equal "
                f"len(future_covariates_df) ({pred_len})"
            )
            raise ValueError(msg)
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
        seed: int = 42,
    ) -> np.ndarray:
        """
        Generate `num_samples` parallel future trajectories.

        Returns ndarray of shape (num_samples, max_new_points, n_features)
        containing only the predicted future rows (no context).

        Use `sampler=TemperatureSampler(temperature=0.0)` for argmax (mode);
        `temperature=1.0` for proper samples from the learned distribution.
        """
        if num_samples < 1:
            msg = f"num_samples must be >= 1, got {num_samples}"
            raise ValueError(msg)
        columns = df.shape[1]
        order = self.order_of_derivative
        if isinstance(order, (int, np.integer)):
            max_order = int(order)
        else:
            order_arr = np.asarray(order)
            if order_arr.shape != (columns,):
                msg = (
                    f"order_of_derivative shape {order_arr.shape} must match "
                    f"columns {columns}"
                )
                raise ValueError(msg)
            max_order = int(order_arr.max())
        if max_order > df.shape[0]:
            msg = (
                f"max order {max_order} > context length {df.shape[0]}: "
                "not enough history to compute the k-th derivative prefix"
            )
            raise ValueError(msg)
        total_positions = (df.shape[0] + max_new_points - max_order) * columns
        if total_positions > self.model.block_size:
            msg = (
                f"prompt rows ({df.shape[0]}) + new points ({max_new_points}) with "
                f"{columns} columns (max_order={max_order}) require {total_positions} "
                f"context positions, but block_size is {self.model.block_size}"
            )
            raise ValueError(msg)
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
            seed=seed,
        )
        futures = np.empty((num_samples, max_new_points, columns), dtype=np.float64)
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
        seed: int = 42,
    ) -> np.ndarray:
        """
        Probabilistic forecasting: return `num_samples` future trajectories.

        Each trajectory is an independent sample from the model's learned
        joint distribution over the `max_new_points`-step future. Use a
        temperature-1 sampler (or nucleus top-p) to get proper samples;
        temperature-0 collapses all trajectories to the mode and is only
        useful as a smoke test.

        `seed` seeds the generator: repeated calls with the same seed return the
        same trajectories, so independent Monte-Carlo batches (a second CRPS
        estimate, a bootstrap) need different seeds.

        Returns ndarray of shape (num_samples, max_new_points, n_features).
        """
        return self._generate_future_samples(
            df=df,
            max_new_points=max_new_points,
            sampler=sampler,
            num_samples=num_samples,
            seed=seed,
        )

    @torch.inference_mode()
    def anomaly_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return per-(time, channel) anomaly scores: conditional negative log-likelihood.

        Encodes the signal to tokens, runs a single forward pass, and
        returns -log p(token_t | ctx) at every position — a strictly
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
            if order_arr.shape != (columns,):
                msg = (
                    f"order_of_derivative shape {order_arr.shape} must match "
                    f"columns {columns}"
                )
                raise ValueError(msg)
            max_order = int(order_arr.max())
        if max_order > T:
            msg = f"max order {max_order} > signal length {T}"
            raise ValueError(msg)

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
        if self.model.block_size < N:
            msg = (
                f"signal has {N} tokens but block_size is {self.model.block_size}; "
                "chunk the input manually for longer sequences"
            )
            raise ValueError(msg)
        device = torch.device(self.model.device_type)
        tokens = torch.as_tensor(
            encoded.reshape(-1).astype(np.int64),
            dtype=torch.int64,
            device=device,
        ).unsqueeze(0)  # (1, N)
        was_training = self.model.training
        self.model.eval()
        logits, _ = self.model.forward(tokens)  # (1, N, V)
        if was_training:
            self.model.train()
        # logits[i] predicts the token at position i+1.
        log_probs = F.log_softmax(logits[0, :-1], dim=-1)  # (N-1, V)
        nll = -log_probs.gather(-1, tokens[0, 1:].unsqueeze(-1)).squeeze(-1)  # (N-1,)
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
        quantiles: list[float] | tuple[float, ...] = (0.1, 0.5, 0.9),
        num_samples: int = 100,
        sampler: Sampler | None = None,
        seed: int = 42,
    ) -> dict[float, pd.DataFrame]:
        """
        Empirical quantile forecasts from `num_samples` Monte-Carlo trajectories.

        Returns a dict mapping each requested quantile alpha ∈ (0, 1) to a
        DataFrame of shape (max_new_points, n_features) containing the
        per-step empirical alpha-quantile across the samples. Defaults produce
        the standard P10/P50/P90 bands for probabilistic forecasting.
        """
        for q in quantiles:
            if not 0.0 < q < 1.0:
                msg = f"quantile {q} must be in (0, 1)"
                raise ValueError(msg)
        futures = self._generate_future_samples(
            df=df,
            max_new_points=max_new_points,
            sampler=sampler,
            num_samples=num_samples,
            seed=seed,
        )
        # futures: (N, H, F). Reduce across samples axis.
        out: dict[float, pd.DataFrame] = {}
        for q in quantiles:
            q_vals = np.quantile(futures, q, axis=0)  # (H, F)
            out[float(q)] = pd.DataFrame(q_vals, columns=df.columns)
        return out

    @torch.inference_mode()
    def calibrate_temperature(
        self,
        windows: list[tuple[pd.DataFrame, np.ndarray]],
        max_new_points: int,
        temperatures: tuple[float, ...] = (0.4, 0.5, 0.6, 0.7, 0.85, 1.0),
        num_samples: int = 30,
        criterion: str = "crps",
        level: float = 0.8,
        seed: int = 42,
    ) -> tuple[float, dict[float, float]]:
        """
        Pick the free-running sampling temperature on held-out windows.

        Cross-entropy calibrates each *one-step* distribution, but a free run
        feeds its own samples back, so per-step errors compound and the joint
        forecast is over-dispersed: on ETTh1 (672-token rollouts) temperature 1
        gives 0.95 coverage of a nominal 0.80 band and a CRPS twice the value
        at temperature 0.5. Temperature is the one knob that rescales the joint
        spread without retraining.

        `windows`: (context DataFrame, truth array of shape (H, F)) pairs from
        a validation split — never the test data being reported.
        `criterion`: "crps" minimises the mean empirical CRPS (a proper score);
        "coverage" minimises |empirical coverage - `level`|.

        Returns the best temperature and the score of every candidate.
        """
        if len(windows) == 0:
            msg = "need at least one calibration window"
            raise ValueError(msg)
        if criterion not in ("crps", "coverage"):
            msg = f"unknown criterion {criterion!r}"
            raise ValueError(msg)
        if num_samples < _MIN_CALIBRATION_SAMPLES:
            msg = "CRPS and coverage need at least 2 samples"
            raise ValueError(msg)
        scores: dict[float, float] = {}
        for t in temperatures:
            if t <= 0:
                msg = "temperature must be positive for sampling"
                raise ValueError(msg)
            vals = []
            for ctx, raw_truth in windows:
                truth = np.asarray(raw_truth, dtype=np.float64)
                samples = self._generate_future_samples(
                    df=ctx,
                    max_new_points=max_new_points,
                    sampler=TemperatureSampler(temperature=t),
                    num_samples=num_samples,
                    seed=seed,  # same draws for every candidate: a paired comparison
                )
                if criterion == "crps":
                    vals.append(crps_empirical(samples, truth))
                else:
                    vals.append(
                        abs(empirical_coverage(samples, truth, level=level) - level),
                    )
            scores[float(t)] = float(np.mean(vals))
        best = min(scores, key=lambda k: scores[k])
        return best, scores

    @torch.inference_mode()
    def predict_point(
        self,
        df: pd.DataFrame,
        max_new_points: int,
        metric: str = "mae",
        num_samples: int = 100,
        sampler: Sampler | None = None,
        naive_scale: float | np.ndarray | None = None,
        smape_ref: float | None = None,
        mase_ref: float | None = None,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Bayes-optimal point forecast for `metric` from Monte-Carlo trajectories.

        Draws `num_samples` trajectories (temperature 1 unless `sampler` says
        otherwise) and returns, per horizon and column, the point minimising
        the expected loss: marginal median for "mae", mean for "mse", the
        expected-sMAPE minimiser for "smape", and M4's OWA minimiser for "owa"
        (needs `naive_scale`, `smape_ref`, `mase_ref`; see `metrics.bayes_point`).
        Unlike feeding argmax / median tokens back into the model, this is the
        marginal decision at each horizon and integrates over intermediate
        uncertainty. Returns a DataFrame of shape (max_new_points, n_features).
        """
        if sampler is None:
            sampler = TemperatureSampler(temperature=1.0)
        futures = self._generate_future_samples(
            df=df,
            max_new_points=max_new_points,
            sampler=sampler,
            num_samples=num_samples,
            seed=seed,
        )
        point = bayes_point(
            futures,
            metric,
            naive_scale=naive_scale,
            smape_ref=smape_ref,
            mase_ref=mase_ref,
        )
        return pd.DataFrame(point, columns=df.columns)

    def _predict_teacher_forced(
        self,
        context_df: pd.DataFrame,
        future_covariates_df: pd.DataFrame,
        target_columns: list[str] | list[int],
        sampler: Sampler | None,
    ) -> pd.DataFrame:
        columns_list = list(context_df.columns)
        if list(future_covariates_df.columns) != columns_list:
            msg = "context and future must have the same columns"
            raise ValueError(msg)
        n_features = len(columns_list)

        # Resolve names / ints → index set.
        target_set: set[int] = set()
        for c in target_columns:
            if isinstance(c, str):
                if c not in columns_list:
                    msg = f"target column {c!r} not found in {columns_list}"
                    raise ValueError(msg)
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
            if order_arr.shape != (n_features,):
                msg = (
                    f"order_of_derivative shape {order_arr.shape} must match "
                    f"n_features {n_features}"
                )
                raise ValueError(msg)
            max_order = int(order_arr.max())
        if max_order > ctx_len:
            msg = (
                f"max order {max_order} > context length {ctx_len}: "
                "not enough history to compute the k-th derivative prefix"
            )
            raise ValueError(msg)

        total_tokens = (ctx_len + pred_len - max_order) * n_features
        if total_tokens > self.model.block_size:
            msg = (
                f"ctx={ctx_len} pred={pred_len} cols={n_features} "
                f"(max_order={max_order}) need {total_tokens} tokens but "
                f"block_size={self.model.block_size}"
            )
            raise ValueError(msg)

        # Build the full concatenated input; zero out target columns in the
        # future portion so their encoded tokens are placeholders. Covariate
        # entries in encoded_full are correct (both endpoints known); target
        # entries are bogus but we'll overwrite them during generation.
        full_df = pd.concat(
            [context_df, future_covariates_df],
            ignore_index=True,
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
        encoded_final = np.asarray(results[0], dtype=np.int64).reshape(-1, n_features)
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
        """Save the model's state dict to `path`."""
        torch.save(self.model.state_dict(), path)

    def load(self, path: str = "diff-gpt.bin") -> None:
        """Load the model's state dict from `path` and switch it to eval mode."""
        self.model.load_state_dict(torch.load(path, weights_only=True))
        self.model.eval()
