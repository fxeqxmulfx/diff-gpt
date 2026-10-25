"""
Data loaders that sample (x, y) token windows for `train`.

`DiffDataFrameDataLoader` encodes in-memory DataFrames, `DiffCorpusDataLoader`
draws normalized windows from a corpus of many series, and
`TokenSequenceDataLoader` serves an already-tokenized sequence.
"""

import warnings
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch

from diff_gpt.domain.encoder_decoder import (
    Difference,
    TokenClippingWarning,
    encode,
    fill_missing,
    get_domain_of_definition,
    observed_tokens,
    prefix_rows,
    resolve_tokenizer,
)
from diff_gpt.domain.normalization import NORMALIZATIONS, normalize, window_scale

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import pandas as pd

    from diff_gpt.domain.encoder_decoder import Order


def _fill(inp: np.ndarray) -> np.ndarray:
    """
    Fill the NaN of `inp` for the encoder: `fill_missing`, and 0 in a column of NaN.

    Every token of such a column reads a NaN, so no loss sees its values.
    """
    return fill_missing(np.where(np.isnan(inp).all(axis=0), 0.0, inp))


class TokenSequenceDataLoader:
    """
    Trivial loader over a flat sequence of already-tokenized integers.

    Use when the token sequence is produced by an external encoder (e.g.
    character-level tokenization) and no derivative encoding is needed.
    """

    __slots__ = (
        "_train",
        "_val",
        "batch_size",
        "block_size",
        "device",
        "rng",
    )

    def __init__(
        self,
        tokens: tuple[int, ...] | list[int],
        block_size: int,
        batch_size: int,
        device: str,
        train_part: float = 0.9,
        rng: torch.Generator | None = None,
    ) -> None:
        """Split `tokens` at the `train_part` fraction into train and val parts."""
        data = tuple(tokens)
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.rng = rng if rng is not None else torch.Generator().manual_seed(42)
        n = int(len(data) * train_part)
        self._train = data[:n]
        self._val = data[n:]

    def has_val(self) -> bool:
        """Return True if the val part is longer than `block_size`."""
        return len(self._val) > self.block_size

    def get_batch(self, split: str = "train") -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a batch of windows from `split` as (inputs, next-token targets)."""
        if split == "train":
            data = self._train
        elif split == "val":
            data = self._val
        else:
            msg = f"unknown split {split!r}; expected 'train' or 'val'"
            raise ValueError(msg)
        if len(data) <= self.block_size:
            msg = f"data length ({len(data)}) must be > block_size ({self.block_size})"
            raise ValueError(msg)
        ix = torch.randint(
            0,
            len(data) - self.block_size,
            (self.batch_size,),
            generator=self.rng,
        ).tolist()
        x = torch.stack(
            [
                torch.tensor(data[i : i + self.block_size], dtype=torch.int64)
                for i in ix
            ],
        )
        y = torch.stack(
            [
                torch.tensor(data[i + 1 : i + self.block_size + 1], dtype=torch.int64)
                for i in ix
            ],
        )
        return x.to(device=self.device), y.to(device=self.device)


def _mask_targets(
    y: torch.Tensor,
    block_size: int,
    n_features: int,
    target_columns: Sequence[int] | None,
    phase: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Mask (with -100) every position of `y` that is not a target column.

    Position j of row i holds the column `(phase[i] + j) % n_features`, where
    `phase[i]` is the column of that row's first target token. Windows are cut
    at arbitrary token offsets, so the phase differs per row; assuming phase 0
    aligned the mask with the real target column in only 1/F of the windows.
    """
    if target_columns is None:
        return y
    targets = tuple(target_columns)
    if len(targets) == 0:
        msg = "target_columns must be non-empty when provided"
        raise ValueError(msg)
    if not all(0 <= c < n_features for c in targets):
        msg = f"target_columns {targets} out of range for n_features {n_features}"
        raise ValueError(msg)
    if phase is None:
        phase = torch.zeros(y.shape[0], dtype=torch.long)
    col = (
        phase.to(device=y.device, dtype=torch.long).unsqueeze(1)
        + torch.arange(block_size, device=y.device).unsqueeze(0)
    ) % n_features
    tgt = torch.tensor(targets, dtype=torch.long, device=y.device)
    keep = (col.unsqueeze(-1) == tgt).any(dim=-1)  # (B, block_size)
    return y.masked_fill(~keep, -100)


class _Windows(NamedTuple):
    """The windows of one split: its series in one tensor and every window start."""

    tokens: torch.Tensor  # the series of the split, one after another
    # Whether each token reads no NaN (`observed_tokens`); None where all do.
    observed: torch.Tensor | None
    starts: torch.Tensor  # the offset in `tokens` of every window
    phases: torch.Tensor  # the column of each window's first target token
    longest: int  # the length of the longest series of the split


def _windows(
    serieses: list[tuple[np.ndarray, np.ndarray]],
    block_size: int,
    step: int,
    n_features: int,
    device: str,
) -> _Windows:
    """
    Index the windows of `block_size + 1` tokens that start at a multiple of `step`.

    `serieses` holds the tokens of each series and whether each reads no
    NaN. The series longer than `block_size` go into one tensor, one after
    another, and the starts run over them in order, so that no window
    crosses into the next series and a uniform start is a uniform window.
    """
    eligible = [s for s in serieses if len(s[0]) > block_size]
    lengths = np.array([len(tokens) for tokens, _ in eligible], dtype=np.int64)
    counts = (lengths - block_size - 1) // step + 1
    # Each start's offset within its own series.
    first = np.cumsum(counts) - counts
    local = (np.arange(counts.sum(), dtype=np.int64) - np.repeat(first, counts)) * step
    starts = np.repeat(np.cumsum(lengths) - lengths, counts) + local
    tokens = np.zeros(0, dtype=np.int64)
    observed = np.zeros(0, dtype=bool)
    if eligible:
        tokens = np.concatenate([t for t, _ in eligible])
        observed = np.concatenate([o for _, o in eligible])
    return _Windows(
        tokens=torch.from_numpy(tokens).to(device),
        observed=None if observed.all() else torch.from_numpy(observed).to(device),
        starts=torch.from_numpy(starts).to(device),
        phases=torch.from_numpy((local + 1) % n_features).to(device),
        longest=max((len(tokens) for tokens, _ in serieses), default=0),
    )


class DiffDataFrameDataLoader:
    """
    In-memory multi-series data loader driven by pandas DataFrames.

    Each DataFrame is encoded once at construction time using the same
    (vocab_size, domain_of_definition, order_of_derivative, tokenizer). Every
    series is then split chronologically at `train_part`, rounded down to a
    whole row, and `get_batch('train'|'val')` samples windows uniformly across
    all series belonging to the requested split (weighted by number of valid
    window starts). The tokens of each split live on `device` in one tensor,
    so a batch is one gather there.

    If `target_columns` is set, only positions that belong to those columns
    contribute to the loss (TimeXer-style exogenous forecasting) — covariate
    positions in `y` are set to -100 (F.cross_entropy's ignore_index).

    NaN marks a missing value: each series is encoded with its NaN filled
    (`encoder_decoder.fill_missing`), and the targets whose tokens read a
    NaN are set to -100 too (`encoder_decoder.observed_tokens`).

    `align_windows`: cut windows only at token offsets that are multiples of
    `n_features`, so every window starts with column 0 exactly like the
    context at inference (`DiffGPT.predict`). Default False keeps the
    arbitrary-phase sampling.
    """

    __slots__ = (
        "_train_windows",
        "_val_windows",
        "align_windows",
        "batch_size",
        "block_size",
        "device",
        "n_features",
        "rng",
        "target_columns",
        "tokenizer",
    )

    def __init__(
        self,
        dfs: list[pd.DataFrame],
        block_size: int,
        batch_size: int,
        vocab_size: int,
        order_of_derivative: Order,
        domain_of_definition: np.ndarray,
        *,
        device: str,
        train_part: float = 0.9,
        target_columns: Sequence[int] | None = None,
        rng: torch.Generator | None = None,
        align_windows: bool = False,
        tokenizer: str | None = None,
    ) -> None:
        """
        Encode every DataFrame once and split each series at `train_part`.

        `tokenizer`: "lgt", "mash" or "mash-half", as for `encode`; None picks
        `encoder_decoder.DEFAULT_TOKENIZER`. Use the `DiffGPT`'s tokenizer.
        """
        if len(dfs) == 0:
            msg = "dfs must be non-empty"
            raise ValueError(msg)
        n_features = dfs[0].shape[1]
        if any(df.shape[1] != n_features for df in dfs):
            msg = "all dataframes must have the same number of columns"
            raise ValueError(msg)
        self.block_size = block_size
        self.batch_size = batch_size
        self.n_features = n_features
        self.target_columns = (
            tuple(target_columns) if target_columns is not None else None
        )
        self.align_windows = align_windows
        self.tokenizer = resolve_tokenizer(tokenizer)
        self.device = device
        self.rng = rng if rng is not None else torch.Generator().manual_seed(42)
        # Encode each DataFrame into a flat token array, then split.
        train_serieses: list[tuple[np.ndarray, np.ndarray]] = []
        val_serieses: list[tuple[np.ndarray, np.ndarray]] = []
        for df in dfs:
            inp = df.to_numpy(dtype=np.float64)
            _, _, encoded = encode(
                inp=_fill(inp),
                vocab_size=vocab_size,
                domain_of_definition=domain_of_definition,
                order_of_derivative=order_of_derivative,
                tokenizer=self.tokenizer,
            )
            flat = encoded.reshape(-1)
            observed = observed_tokens(inp, order_of_derivative).reshape(-1)
            # Split on a row boundary, so that the val slice starts with column
            # 0 like the train slice. `get_batch` takes a position's column from
            # its offset in the slice: with a mid-row split the val target mask
            # would keep a covariate, and aligned val windows would carry
            # another phase than the inference context.
            n = int(len(flat) * train_part)
            n -= n % n_features
            train_serieses.append((flat[:n], observed[:n]))
            val_serieses.append((flat[n:], observed[n:]))
        step = n_features if align_windows else 1
        self._train_windows = _windows(
            train_serieses,
            block_size,
            step,
            n_features,
            device,
        )
        self._val_windows = _windows(val_serieses, block_size, step, n_features, device)

    def has_val(self) -> bool:
        """Return True if any val series is longer than `block_size`."""
        return self._val_windows.starts.numel() > 0

    def get_batch(self, split: str = "train") -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a batch of windows from `split` as (inputs, next-token targets).

        Positions of `y` outside `target_columns`, or whose tokens read a NaN,
        are masked with -100.
        """
        if split == "train":
            windows = self._train_windows
        elif split == "val":
            windows = self._val_windows
        else:
            msg = f"unknown split {split!r}; expected 'train' or 'val'"
            raise ValueError(msg)
        n_starts = windows.starts.numel()
        if n_starts == 0:
            msg = (
                f"No {split!r} series long enough: block_size={self.block_size}, "
                f"max_series_len={windows.longest}"
            )
            raise ValueError(msg)
        picks = torch.randint(0, n_starts, (self.batch_size,), generator=self.rng)
        if picks.device != windows.starts.device:
            # From pinned memory the copy does not wait for the queued steps.
            picks = picks.pin_memory().to(windows.starts.device, non_blocking=True)
        index = windows.starts[picks].unsqueeze(1) + torch.arange(
            self.block_size,
            device=windows.starts.device,
        )
        x = windows.tokens[index]
        # y is x shifted by one token; phase is the column of y[:, 0].
        y = windows.tokens[index + 1]
        if windows.observed is not None:
            y = y.masked_fill(~windows.observed[index + 1], -100)
        y = _mask_targets(
            y,
            self.block_size,
            self.n_features,
            self.target_columns,
            windows.phases[picks],
        )
        return x, y


class _Sources(NamedTuple):
    """The series of a split: each source's (N, L) array and its probability."""

    names: tuple[str, ...]
    series: tuple[np.ndarray, ...]
    probabilities: np.ndarray


def _sources(
    series: Mapping[str, np.ndarray],
    weights: Mapping[str, float] | None,
    rows: int,
) -> _Sources:
    """
    Check the sources and normalize their weights to probabilities.

    Every source needs a series and series of at least `rows` rows; missing
    or negative weights, or weights that sum to 0, raise ValueError.
    """
    if not series:
        msg = "the corpus needs at least one source"
        raise ValueError(msg)
    names = tuple(series)
    for name in names:
        shape = series[name].shape
        if len(shape) != 2 or shape[0] < 1 or shape[1] < rows:  # noqa: PLR2004
            msg = (
                f"source {name!r} has shape {shape}, but a window needs series "
                f"of at least {rows} rows, in an array of shape (N, L)"
            )
            raise ValueError(msg)
    if weights is None:
        raw = np.ones(len(names))
    else:
        missing = [name for name in names if name not in weights]
        if missing:
            msg = f"weights has no entry for sources {missing}"
            raise ValueError(msg)
        raw = np.array([weights[name] for name in names], dtype=np.float64)
    if (raw < 0).any() or raw.sum() <= 0:
        msg = f"weights must be non-negative with a positive sum, got {raw.tolist()}"
        raise ValueError(msg)
    return _Sources(
        names=names,
        series=tuple(series[name] for name in names),
        probabilities=raw / raw.sum(),
    )


def _draw_windows(
    sources: _Sources,
    count: int,
    rows: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Draw `count` windows of `rows` rows: the (rows, count) float64 columns.

    Each draws its source by the probabilities, then a series and an offset
    uniformly.
    """
    picks = rng.choice(len(sources.names), size=count, p=sources.probabilities)
    shapes = np.array([series.shape for series in sources.series])[picks]
    indices = rng.integers(shapes[:, 0])
    offsets = rng.integers(shapes[:, 1] - rows + 1)
    windows = np.empty((rows, count), dtype=np.float64)
    for b, (source, i, offset) in enumerate(zip(picks, indices, offsets, strict=True)):
        windows[:, b] = sources.series[source][i, offset : offset + rows]
    return windows


def _normalize_windows(
    windows: np.ndarray,
    normalization: str,
    min_context: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Divide each window by the statistics of its first c rows.

    c is uniform in [`min_context`, rows], drawn per window, and the
    statistics are those of `window_scale` with `normalization`.
    """
    rows, count = windows.shape
    lengths = rng.integers(min_context, rows + 1, size=count)
    # The rows after a prefix count as missing, so one call reads every prefix.
    prefix = np.where(np.arange(rows)[:, None] < lengths, windows, np.nan)
    return normalize(windows, window_scale(prefix, normalization))


def _missing_runs(
    rows: int,
    rng: np.random.Generator,
    max_fraction: float,
    max_run: int,
) -> np.ndarray:
    """
    Return a mask of runs of missing rows for one window.

    The runs cover a fraction of the rows uniform in [0, `max_fraction`];
    their lengths are geometric, with a mean log-uniform in [1, `max_run`].
    """
    mask = np.zeros(rows, dtype=bool)
    target = rng.uniform(0.0, max_fraction) * rows
    mean_run = float(np.exp(rng.uniform(0.0, np.log(max_run))))
    covered = 0
    while covered < target:
        start = int(rng.integers(rows))
        stop = start + int(rng.geometric(1.0 / mean_run))
        covered += int(np.count_nonzero(~mask[start:stop]))
        mask[start:stop] = True
    return mask


def corpus_domain(
    series: Mapping[str, np.ndarray],
    block_size: int,
    count: int,
    order_of_derivative: int | Difference,
    *,
    quantile: float | None = None,
    symmetric: bool = True,
    weights: Mapping[str, float] | None = None,
    normalization: str = "mean_abs_step",
    min_context: int = 16,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Return one domain for the steps of a corpus in the units of its windows.

    Draws `count` windows from `series` and divides them as
    `DiffCorpusDataLoader` does with the same `block_size`, `weights`,
    `normalization` and `min_context`, and bounds their steps c(B) x with
    `get_domain_of_definition(..., quantile=quantile, symmetric=symmetric)`:
    shape (1,) for [-D, D], or (1, 2) for [lo, hi]. No step spans two
    windows. `rng` defaults to a generator seeded with 0.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    p = prefix_rows(order_of_derivative)
    rows = block_size + 1 + p
    windows = _normalize_windows(
        _draw_windows(_sources(series, weights, rows), count, rows, rng),
        normalization,
        min_context,
        rng,
    )
    # One column, with p rows of NaN after each window: a step that would
    # span two windows reads a NaN, and steps that read one are left out.
    padded = np.vstack([windows, np.full((p, count), np.nan)])
    return get_domain_of_definition(
        padded.reshape(-1, 1, order="F"),
        order_of_derivative,
        quantile=quantile,
        symmetric=symmetric,
    )


class DiffCorpusDataLoader:
    """
    Windows of a corpus of univariate series, each in the units of its context.

    `train`, and `val` if given, map a source name to its series: an array of
    shape (N, L) holding N series of length L, such as a memory-mapped .npy
    of one generator. A window is `block_size + 1` tokens, the
    block_size + 1 + p rows of one series that encode to them, where
    p = `prefix_rows(order_of_derivative)`. Each window draws its source
    with probability proportional to `weights`, which serves both splits and
    so must hold every source of either (equal by default), then a series
    and an offset, uniformly.

    A window is normalized as a forecast's context is: by `window_scale`
    with `normalization`, of its first c rows, where c is uniform in
    [`min_context`, rows]. The rows after c play the future, which need not
    keep to the context's scale. Every window is encoded with the one
    `domain_of_definition`, shape (1,) or (1, 2), in these units
    (`corpus_domain`), so steps beyond it clip by design; the loader
    silences their `TokenClippingWarning`.

    Missing values: with probability `missing_prob` a window loses runs of
    values covering a fraction of it uniform in [0, `max_missing_fraction`],
    in runs of geometric length whose mean is log-uniform in
    [1, `max_missing_run`]. The window's scale reads only the values left; as
    in the other loaders, the input fills the gaps (`fill_missing`) and the
    targets whose tokens read a missing value are -100.
    """

    __slots__ = (
        "_rows",
        "_train",
        "_val",
        "batch_size",
        "block_size",
        "device",
        "domain_of_definition",
        "max_missing_fraction",
        "max_missing_run",
        "min_context",
        "missing_prob",
        "normalization",
        "order_of_derivative",
        "rng",
        "tokenizer",
        "vocab_size",
    )

    def __init__(
        self,
        train: Mapping[str, np.ndarray],
        block_size: int,
        batch_size: int,
        vocab_size: int,
        order_of_derivative: int | Difference,
        domain_of_definition: np.ndarray,
        *,
        device: str,
        val: Mapping[str, np.ndarray] | None = None,
        weights: Mapping[str, float] | None = None,
        normalization: str = "mean_abs_step",
        min_context: int = 16,
        missing_prob: float = 0.0,
        max_missing_fraction: float = 0.5,
        max_missing_run: int = 64,
        rng: np.random.Generator | None = None,
        tokenizer: str | None = None,
    ) -> None:
        """
        Check the sources and the settings; the windows are drawn per batch.

        `tokenizer`: "lgt", "mash" or "mash-half", as for `encode`; None picks
        `encoder_decoder.DEFAULT_TOKENIZER`. Use the `DiffGPT`'s tokenizer.
        `rng` defaults to a generator seeded with 42; both splits draw from it.
        """
        if not isinstance(order_of_derivative, (int, np.integer, Difference)):
            msg = (
                "the corpus is univariate: order_of_derivative must be one "
                f"order, got {order_of_derivative!r}"
            )
            raise TypeError(msg)
        domain = np.asarray(domain_of_definition, dtype=np.float64)
        if domain.shape not in ((1,), (1, 2)):
            msg = (
                "domain_of_definition must have shape (1,) or (1, 2), "
                f"got {domain.shape}"
            )
            raise ValueError(msg)
        if normalization not in NORMALIZATIONS:
            msg = (
                f"unknown normalization {normalization!r}; "
                f"expected one of {NORMALIZATIONS}"
            )
            raise ValueError(msg)
        rows = block_size + 1 + prefix_rows(order_of_derivative)
        if not 1 <= min_context <= rows:
            msg = f"min_context must lie in [1, {rows}], got {min_context}"
            raise ValueError(msg)
        if not 0.0 <= missing_prob <= 1.0 or not 0.0 <= max_missing_fraction < 1.0:
            msg = (
                "missing_prob must lie in [0, 1] and max_missing_fraction in "
                f"[0, 1), got {missing_prob} and {max_missing_fraction}"
            )
            raise ValueError(msg)
        if max_missing_run < 1:
            msg = f"max_missing_run must be >= 1, got {max_missing_run}"
            raise ValueError(msg)
        self.block_size = block_size
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.order_of_derivative = order_of_derivative
        self.domain_of_definition = domain
        self.device = device
        self.normalization = normalization
        self.min_context = min_context
        self.missing_prob = missing_prob
        self.max_missing_fraction = max_missing_fraction
        self.max_missing_run = max_missing_run
        self.rng = rng if rng is not None else np.random.default_rng(42)
        self.tokenizer = resolve_tokenizer(tokenizer)
        self._rows = rows
        self._train = _sources(train, weights, rows)
        self._val = None if val is None else _sources(val, weights, rows)

    def has_val(self) -> bool:
        """Return True if the loader was given `val` sources."""
        return self._val is not None

    def get_batch(self, split: str = "train") -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a batch of windows from `split` as (inputs, next-token targets).

        Targets whose tokens read a missing value are masked with -100.
        """
        if split == "train":
            sources = self._train
        elif split == "val":
            if self._val is None:
                msg = "no val split available: pass val sources"
                raise RuntimeError(msg)
            sources = self._val
        else:
            msg = f"unknown split {split!r}; expected 'train' or 'val'"
            raise ValueError(msg)
        rng = self.rng
        count = self.batch_size
        windows = _draw_windows(sources, count, self._rows, rng)
        if self.missing_prob > 0:
            for b in np.flatnonzero(rng.random(count) < self.missing_prob):
                lost = _missing_runs(
                    self._rows,
                    rng,
                    self.max_missing_fraction,
                    self.max_missing_run,
                )
                windows[lost, b] = np.nan
        normalized = _normalize_windows(
            windows,
            self.normalization,
            self.min_context,
            rng,
        )
        with warnings.catch_warnings():
            # The corpus domain is a quantile: some steps clip by design.
            warnings.simplefilter("ignore", TokenClippingWarning)
            _, _, tokens = encode(
                inp=_fill(normalized),
                vocab_size=self.vocab_size,
                domain_of_definition=np.repeat(
                    self.domain_of_definition,
                    count,
                    axis=0,
                ),
                order_of_derivative=self.order_of_derivative,
                tokenizer=self.tokenizer,
            )
        observed = observed_tokens(normalized, self.order_of_derivative)
        x = torch.from_numpy(np.ascontiguousarray(tokens[:-1].T))
        y = torch.from_numpy(np.ascontiguousarray(tokens[1:].T))
        y = y.masked_fill(~torch.from_numpy(np.ascontiguousarray(observed[1:].T)), -100)
        return x.to(device=self.device), y.to(device=self.device)
