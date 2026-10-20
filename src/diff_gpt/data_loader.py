"""
Data loaders that sample (x, y) token windows for `train`.

`DiffSQLiteDataLoader` streams series from SQLite, `DiffDataFrameDataLoader`
encodes in-memory DataFrames, and `TokenSequenceDataLoader` serves an
already-tokenized sequence.
"""

import sqlite3
from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
import torch

from diff_gpt.encoder_decoder import encode, prefix_rows, resolve_tokenizer

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import pandas as pd

    from diff_gpt.encoder_decoder import Order


def _quote_ident(name: str) -> str:
    """Safely quote a SQLite identifier (table or column name)."""
    return '"' + name.replace('"', '""') + '"'


class DiffSQLiteDataLoader:
    """
    Reads uniformly-sampled time series from a SQLite database.

    Contract: each table has an integer `dt` primary key whose values are
    evenly spaced — the step is reconstructed as floor((max-min) / (rows-1)).
    Non-uniform or non-integer `dt` will cause `get_data` to query wrong row
    ranges and is not supported.

    If `train_part` < 1.0, each table is split chronologically at
    floor(n_tokens * train_part); `get_batch('train')` samples windows from
    the prefix and `get_batch('val')` samples from the suffix. A split draws
    only from tables whose part is longer than `block_size`.
    """

    __slots__ = (
        "_max_order",
        "_train_tables",
        "_val_tables",
        "batch_size",
        "block_size",
        "connection",
        "cursor",
        "device",
        "domain_of_definition",
        "encode_data",
        "order_of_derivative",
        "rng",
        "tables",
        "tables_columns_count",
        "tables_dt_diff",
        "tables_len",
        "tables_range",
        "tables_tokens_len",
        "tables_train_tokens_len",
        "tokenizer",
        "train_part",
        "vocab_size",
    )

    def __init__(
        self,
        block_size: int,
        batch_size: int,
        vocab_size: int,
        order_of_derivative: Order,
        domain_of_definition: np.ndarray | Mapping[str, np.ndarray],
        *,
        device: str,
        database: str,
        encode_data: Callable = encode,
        rng: torch.Generator | None = None,
        train_part: float = 1.0,
        tokenizer: str | None = None,
    ) -> None:
        """
        Open `database` and read the dt range and size of every table.

        Only tables with more than `block_size + 1` tokens are sampled from.
        `domain_of_definition` serves every table, or maps each sampled table
        to its own (tables of different widths need their own). `encode_data`
        turns raw rows into tokens (`encode` by default) with `tokenizer`
        ("lgt" or "mash"; None picks `encoder_decoder.DEFAULT_TOKENIZER`).
        `rng` defaults to a fresh generator seeded with 42.
        """
        self.block_size = block_size
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.order_of_derivative = order_of_derivative
        # The raw rows in front of the tokens, the largest degree, for the
        # row and token offsets; the encoder handles per-column orders.
        self._max_order = prefix_rows(order_of_derivative)
        self.domain_of_definition = domain_of_definition
        self.encode_data = encode_data
        self.tokenizer = resolve_tokenizer(tokenizer)
        self.device = device
        # Construct a fresh generator per instance. Using a mutable default here
        # would share state across loaders in the same process.
        self.rng = rng if rng is not None else torch.Generator().manual_seed(42)
        self.connection = sqlite3.connect(database)
        self.cursor = self.connection.cursor()
        tables = tuple(
            row[0]
            for row in self.cursor.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name != 'sqlite_sequence';",
            ).fetchall()
        )
        # SQLite cannot bind a table name as a parameter, so the queries below
        # interpolate it. That is safe: the names come from sqlite_master and
        # `_quote_ident` quotes them.
        tables_range = {
            table: self.cursor.execute(
                f"SELECT MIN(dt), MAX(dt) FROM {_quote_ident(table)}",  # noqa: S608
            ).fetchone()
            for table in tables
        }
        self.tables_range = tables_range
        tables_columns_count = {
            table: self.cursor.execute(
                "SELECT ncol FROM pragma_table_list WHERE name = ?",
                (table,),
            ).fetchone()[0]
            for table in tables
        }
        self.tables_columns_count = tables_columns_count
        tables_row_count = {
            table: self.cursor.execute(
                f"SELECT count(*) FROM {_quote_ident(table)};",  # noqa: S608
            ).fetchone()[0]
            for table in tables
        }
        tables_dt_diff = {}
        for table in tables:
            count = tables_row_count[table]
            if count > 1:
                t_min, t_max = tables_range[table]
                tables_dt_diff[table] = (t_max - t_min) // (count - 1)
            else:
                tables_dt_diff[table] = 0
        self.tables_dt_diff = tables_dt_diff
        tables_tokens_len = {
            table: (tables_row_count[table] - self._max_order)
            * (tables_columns_count[table] - 1)
            for table in tables
        }
        self.tables_tokens_len = tables_tokens_len
        tables = tuple(
            filter(lambda table: tables_tokens_len[table] > block_size + 1, tables),
        )
        self.tables = tables
        self.tables_len = len(tables)
        if isinstance(domain_of_definition, Mapping):
            missing = [t for t in tables if t not in domain_of_definition]
            if missing:
                msg = f"domain_of_definition has no entry for tables {missing}"
                raise ValueError(msg)
        # Train/val split (per-table tail holdout).
        if not 0.0 < train_part <= 1.0:
            msg = "train_part must be in (0, 1]"
            raise ValueError(msg)
        self.train_part = train_part
        self.tables_train_tokens_len = {
            t: int(tables_tokens_len[t] * train_part) for t in tables
        }
        # A table contributes to a split iff its part has room for a window.
        self._train_tables = tuple(
            t for t in tables if self.tables_train_tokens_len[t] > block_size
        )
        self._val_tables = tuple(
            t
            for t in tables
            if tables_tokens_len[t] - self.tables_train_tokens_len[t] > block_size
        )

    def close(self) -> None:
        """Close the cursor and the database connection."""
        self.cursor.close()
        self.connection.close()

    def _domain(self, table: str) -> np.ndarray:
        """Return the domain of definition that `table` is encoded with."""
        domain = self.domain_of_definition
        return domain[table] if isinstance(domain, Mapping) else domain

    def get_data_len(self, table: str) -> int:
        """Return the number of tokens in `table`."""
        return self.tables_tokens_len[table]

    def get_data(self, table: str, start: int, end: int) -> tuple[int, ...]:
        """
        Return the tokens at flat positions [start, end) of `table`.

        Only the rows covering that range, plus the derivative prefix, are read
        and encoded. If the range runs past the end of the table the result is
        shorter than `end - start`; it is empty when no rows match.
        """
        n_columns = self.tables_columns_count[table]
        n_features = n_columns - 1
        effective_start = start + (self._max_order * n_features)
        effective_end = end + (self._max_order * n_features)
        start_row_idx = effective_start // n_features
        end_row_idx = (effective_end - 1) // n_features
        raw_start_row = start_row_idx - self._max_order
        raw_end_row = end_row_idx
        min_dt = self.tables_range[table][0]
        dt_step = self.tables_dt_diff[table]
        t_start = min_dt + (raw_start_row * dt_step)
        t_end = min_dt + (raw_end_row * dt_step)
        # A table name cannot be bound as a parameter. `table` is a known table
        # (looked up in `tables_columns_count` above) and `_quote_ident` quotes
        # it; the dt bounds are bound.
        query = (
            f"SELECT * FROM {_quote_ident(table)} "  # noqa: S608
            "WHERE dt BETWEEN ? AND ? ORDER BY dt ASC"
        )
        self.cursor.execute(query, (t_start, t_end))
        rows = self.cursor.fetchall()
        if not rows:
            return ()
        arr = np.array(rows, dtype=np.float64)[:, 1:]
        _, _, encoded = self.encode_data(
            inp=arr,
            vocab_size=self.vocab_size,
            domain_of_definition=self._domain(table),
            order_of_derivative=self.order_of_derivative,
            tokenizer=self.tokenizer,
        )
        encoded = encoded.flatten()
        local_offset = effective_start % n_features
        req_length = end - start
        if len(encoded) < local_offset + req_length:
            result = encoded[local_offset:]
        else:
            result = encoded[local_offset : local_offset + req_length]
        return tuple(result.tolist())

    def has_val(self) -> bool:
        """Return True if some table's held-out suffix is longer than `block_size`."""
        return self.train_part < 1.0 and len(self._val_tables) > 0

    def get_batch(self, split: str = "train") -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a batch of windows from `split` as (inputs, next-token targets)."""
        block_size = self.block_size
        if split == "train":
            if not self._train_tables:
                msg = (
                    f"No 'train' table long enough: block_size={block_size}, "
                    "and no table's train part is longer"
                )
                raise ValueError(msg)
            tables = self._train_tables
        elif split == "val":
            if not self.has_val():
                msg = (
                    "no val split available: train_part=1.0 or no table's "
                    "suffix is longer than block_size"
                )
                raise RuntimeError(msg)
            tables = self._val_tables
        else:
            msg = f"unknown split {split!r}; expected 'train' or 'val'"
            raise ValueError(msg)
        tables_len = len(tables)
        batch_size = self.batch_size
        x = torch.zeros(size=(batch_size, block_size), dtype=torch.int64)
        y = torch.zeros(size=(batch_size, block_size), dtype=torch.int64)
        rng = self.rng
        i = 0
        while i < batch_size:
            table_idx = int(
                torch.randint(
                    low=0,
                    high=tables_len,
                    size=(1,),
                    generator=rng,
                ),
            )
            table = tables[table_idx]
            train_end = self.tables_train_tokens_len[table]
            # Every table of `tables` has room for a window: hi > lo.
            if split == "train":
                lo, hi = 0, train_end - block_size
            else:
                total = self.tables_tokens_len[table]
                lo, hi = train_end, total - block_size
            data_idx = int(
                torch.randint(
                    low=lo,
                    high=hi,
                    size=(1,),
                    generator=rng,
                ),
            )
            temp = torch.tensor(
                self.get_data(
                    table=table,
                    start=data_idx,
                    end=data_idx + block_size + 1,
                ),
                dtype=torch.int64,
            )
            if temp.size(0) != block_size + 1:
                continue
            x[i] = temp[:-1]
            y[i] = temp[1:]
            i += 1
        device = self.device
        x, y = x.to(device=device), y.to(device=device)
        return x, y


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
        phase.to(torch.long).unsqueeze(1) + torch.arange(block_size).unsqueeze(0)
    ) % n_features
    tgt = torch.tensor(targets, dtype=torch.long)
    keep = (col.unsqueeze(-1) == tgt).any(dim=-1)  # (B, block_size)
    return y.masked_fill(~keep, -100)


class DiffDataFrameDataLoader:
    """
    In-memory multi-series data loader driven by pandas DataFrames.

    Each DataFrame is encoded once at construction time using the same
    (vocab_size, domain_of_definition, order_of_derivative, tokenizer). Every
    series is then split chronologically at `train_part`, rounded down to a
    whole row, and `get_batch('train'|'val')` samples windows uniformly across
    all series belonging to the requested split (weighted by number of valid
    window starts).

    If `target_columns` is set, only positions that belong to those columns
    contribute to the loss (TimeXer-style exogenous forecasting) — covariate
    positions in `y` are set to -100 (F.cross_entropy's ignore_index).

    `align_windows`: cut windows only at token offsets that are multiples of
    `n_features`, so every window starts with column 0 exactly like the
    context at inference (`DiffGPT.predict`). Default False keeps the
    arbitrary-phase sampling.
    """

    __slots__ = (
        "_train_serieses",
        "_val_serieses",
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

        `tokenizer`: "lgt" or "mash", as for `encode`; None picks
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
        # Encode each DataFrame into a flat token tuple, then split.
        self._train_serieses: list[tuple[int, ...]] = []
        self._val_serieses: list[tuple[int, ...]] = []
        for df in dfs:
            _, _, encoded = encode(
                inp=df.to_numpy(dtype=np.float64),
                vocab_size=vocab_size,
                domain_of_definition=domain_of_definition,
                order_of_derivative=order_of_derivative,
                tokenizer=self.tokenizer,
            )
            flat = tuple(encoded.reshape(-1).tolist())
            # Split on a row boundary, so that the val slice starts with column
            # 0 like the train slice. `get_batch` takes a position's column from
            # its offset in the slice: with a mid-row split the val target mask
            # would keep a covariate, and aligned val windows would carry
            # another phase than the inference context.
            n = int(len(flat) * train_part)
            n -= n % n_features
            self._train_serieses.append(flat[:n])
            self._val_serieses.append(flat[n:])

    def has_val(self) -> bool:
        """Return True if any val series is longer than `block_size`."""
        return any(len(s) > self.block_size for s in self._val_serieses)

    def get_batch(self, split: str = "train") -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a batch of windows from `split` as (inputs, next-token targets).

        Positions of `y` outside `target_columns` are masked with -100.
        """
        if split == "train":
            pool = self._train_serieses
        elif split == "val":
            pool = self._val_serieses
        else:
            msg = f"unknown split {split!r}; expected 'train' or 'val'"
            raise ValueError(msg)
        block_size = self.block_size
        eligible = [s for s in pool if len(s) > block_size]
        if not eligible:
            max_len = max((len(s) for s in pool), default=0)
            msg = (
                f"No {split!r} series long enough: block_size={block_size}, "
                f"max_series_len={max_len}"
            )
            raise ValueError(msg)
        F = self.n_features
        step = F if self.align_windows else 1
        # Valid window starts: local + block_size + 1 <= len(s), stepping by `step`.
        starts_per = [(len(s) - block_size - 1) // step + 1 for s in eligible]
        total_starts = sum(starts_per)
        cum = np.cumsum(starts_per)
        batch_size = self.batch_size
        x = torch.zeros((batch_size, block_size), dtype=torch.int64)
        y = torch.zeros((batch_size, block_size), dtype=torch.int64)
        phase = torch.zeros(batch_size, dtype=torch.long)
        picks = torch.randint(
            0,
            total_starts,
            (batch_size,),
            generator=self.rng,
        ).tolist()
        for i, r in enumerate(picks):
            sidx = int(np.searchsorted(cum, r, side="right"))
            local = (r - (int(cum[sidx - 1]) if sidx > 0 else 0)) * step
            s = eligible[sidx]
            x[i] = torch.tensor(s[local : local + block_size], dtype=torch.int64)
            y[i] = torch.tensor(
                s[local + 1 : local + block_size + 1],
                dtype=torch.int64,
            )
            phase[i] = (local + 1) % F  # column of y[i, 0]
        y = _mask_targets(y, block_size, F, self.target_columns, phase)
        return x.to(device=self.device), y.to(device=self.device)
