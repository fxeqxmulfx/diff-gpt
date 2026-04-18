import sqlite3
from typing import Callable, Sequence
import numpy as np
import pandas as pd
import torch

from diff_gpt.encoder_decoder import encode, np_to_decimal


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
    the prefix and `get_batch('val')` samples from the suffix.
    """

    __slots__ = (
        "block_size",
        "batch_size",
        "vocab_size",
        "order_of_derivative",
        "domain_of_definition",
        "use_decimal",
        "encode_data",
        "device",
        "connection",
        "rng",
        "cursor",
        "tables",
        "tables_len",
        "tables_range",
        "tables_columns_count",
        "tables_dt_diff",
        "tables_tokens_len",
        "train_part",
        "tables_train_tokens_len",
        "_val_tables",
    )

    def __init__(
        self,
        block_size: int,
        batch_size: int,
        vocab_size: int,
        order_of_derivative: int,
        domain_of_definition: np.ndarray,
        use_decimal: bool,
        device: str,
        database: str,
        encode_data: Callable = encode,
        rng: torch.Generator | None = None,
        train_part: float = 1.0,
    ) -> None:
        self.block_size = block_size
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.order_of_derivative = order_of_derivative
        self.domain_of_definition = domain_of_definition
        self.use_decimal = use_decimal
        self.encode_data = encode_data
        self.device = device
        # Construct a fresh generator per instance. Using a mutable default here
        # would share state across loaders in the same process.
        self.rng = rng if rng is not None else torch.Generator().manual_seed(42)
        self.connection = sqlite3.connect(database)
        self.cursor = self.connection.cursor()
        tables = tuple(
            row[0]
            for row in self.cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence';"
            ).fetchall()
        )
        tables_range = dict(
            (
                table,
                self.cursor.execute(
                    f"SELECT MIN(dt), MAX(dt) FROM {_quote_ident(table)}"
                ).fetchone(),
            )
            for table in tables
        )
        self.tables_range = tables_range
        tables_columns_count = dict(
            (
                table,
                self.cursor.execute(
                    "SELECT ncol FROM pragma_table_list WHERE name = ?", (table,)
                ).fetchone()[0],
            )
            for table in tables
        )
        self.tables_columns_count = tables_columns_count
        tables_row_count = dict(
            (
                table,
                self.cursor.execute(
                    f"SELECT count(*) FROM {_quote_ident(table)};"
                ).fetchone()[0],
            )
            for table in tables
        )
        tables_dt_diff = {}
        for table in tables:
            count = tables_row_count[table]
            if count > 1:
                t_min, t_max = tables_range[table]
                tables_dt_diff[table] = (t_max - t_min) // (count - 1)
            else:
                tables_dt_diff[table] = 0
        self.tables_dt_diff = tables_dt_diff
        tables_tokens_len = dict(
            (
                table,
                (tables_row_count[table] - order_of_derivative)
                * (tables_columns_count[table] - 1),
            )
            for table in tables
        )
        self.tables_tokens_len = tables_tokens_len
        tables = tuple(
            filter(lambda table: tables_tokens_len[table] > block_size + 1, tables)
        )
        self.tables = tables
        self.tables_len = len(tables)
        # Train/val split (per-table tail holdout).
        assert 0.0 < train_part <= 1.0, "train_part must be in (0, 1]"
        self.train_part = train_part
        self.tables_train_tokens_len = {
            t: int(tables_tokens_len[t] * train_part) for t in tables
        }
        # A table contributes to val iff its suffix has room for a window.
        self._val_tables = tuple(
            t
            for t in tables
            if tables_tokens_len[t] - self.tables_train_tokens_len[t] > block_size
        )

    def close(self) -> None:
        self.cursor.close()
        self.connection.close()

    def get_data_len(self, table: str) -> int:
        return self.tables_tokens_len[table]

    def get_data(self, table: str, start: int, end: int) -> tuple[int, ...]:
        n_columns = self.tables_columns_count[table]
        n_features = n_columns - 1
        effective_start = start + (self.order_of_derivative * n_features)
        effective_end = end + (self.order_of_derivative * n_features)
        start_row_idx = effective_start // n_features
        end_row_idx = (effective_end - 1) // n_features
        raw_start_row = start_row_idx - self.order_of_derivative
        raw_end_row = end_row_idx
        min_dt = self.tables_range[table][0]
        dt_step = self.tables_dt_diff[table]
        t_start = min_dt + (raw_start_row * dt_step)
        t_end = min_dt + (raw_end_row * dt_step)
        query = (
            f"SELECT * FROM {_quote_ident(table)} "
            "WHERE dt BETWEEN ? AND ? ORDER BY dt ASC"
        )
        self.cursor.execute(query, (t_start, t_end))
        rows = self.cursor.fetchall()
        if not rows:
            return tuple()
        arr = np.array(rows, dtype=np.float64)
        arr = arr[:, 1:]
        encode_data = self.encode_data
        _, _, encoded = encode_data(
            inp=arr,
            vocab_size=self.vocab_size,
            domain_of_definition=self.domain_of_definition,
            order_of_derivative=self.order_of_derivative,
            use_decimal=self.use_decimal,
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
        return self.train_part < 1.0 and len(self._val_tables) > 0

    def get_batch(self, split: str = "train") -> tuple[torch.Tensor, torch.Tensor]:
        block_size = self.block_size
        if split == "train":
            tables = self.tables
        elif split == "val":
            if not self.has_val():
                raise RuntimeError(
                    "no val split available: train_part=1.0 or no table's "
                    "suffix is longer than block_size"
                )
            tables = self._val_tables
        else:
            raise ValueError(f"unknown split {split!r}; expected 'train' or 'val'")
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
                )
            )
            table = tables[table_idx]
            train_end = self.tables_train_tokens_len[table]
            if split == "train":
                lo, hi = 0, train_end - block_size
            else:
                total = self.tables_tokens_len[table]
                lo, hi = train_end, total - block_size
            if hi <= lo:
                continue  # this table has no room in the requested split
            data_idx = int(
                torch.randint(
                    low=lo,
                    high=hi,
                    size=(1,),
                    generator=rng,
                )
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
        "block_size",
        "batch_size",
        "device",
        "rng",
        "_train",
        "_val",
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
        data = tuple(tokens)
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.rng = rng if rng is not None else torch.Generator().manual_seed(42)
        n = int(len(data) * train_part)
        self._train = data[:n]
        self._val = data[n:]

    def has_val(self) -> bool:
        return len(self._val) > self.block_size

    def get_batch(self, split: str = "train") -> tuple[torch.Tensor, torch.Tensor]:
        if split == "train":
            data = self._train
        elif split == "val":
            data = self._val
        else:
            raise ValueError(f"unknown split {split!r}; expected 'train' or 'val'")
        if len(data) <= self.block_size:
            raise ValueError(
                f"data length ({len(data)}) must be > block_size ({self.block_size})"
            )
        ix = torch.randint(
            0, len(data) - self.block_size, (self.batch_size,), generator=self.rng
        ).tolist()
        x = torch.stack(
            [
                torch.tensor(data[i : i + self.block_size], dtype=torch.int64)
                for i in ix
            ]
        )
        y = torch.stack(
            [
                torch.tensor(
                    data[i + 1 : i + self.block_size + 1], dtype=torch.int64
                )
                for i in ix
            ]
        )
        return x.to(device=self.device), y.to(device=self.device)


def _mask_targets(
    y: torch.Tensor,
    block_size: int,
    n_features: int,
    target_columns: Sequence[int] | None,
) -> torch.Tensor:
    if target_columns is None:
        return y
    targets = tuple(target_columns)
    assert len(targets) > 0, "target_columns must be non-empty when provided"
    assert all(0 <= c < n_features for c in targets), (
        f"target_columns {targets} out of range for n_features {n_features}"
    )
    col = torch.arange(block_size) % n_features
    if len(targets) == 1:
        keep = col == targets[0]
    else:
        tgt = torch.tensor(targets, dtype=torch.long)
        keep = (col.unsqueeze(-1) == tgt).any(dim=-1)
    return y.masked_fill(~keep.unsqueeze(0), -100)


class DiffDataFrameDataLoader:
    """
    In-memory multi-series data loader driven by pandas DataFrames.

    Each DataFrame is encoded once at construction time using the same
    (vocab_size, domain_of_definition, order_of_derivative). Every series is
    then split chronologically at `train_part`, and `get_batch('train'|'val')`
    samples windows uniformly across all series belonging to the requested
    split (weighted by number of valid window starts).

    If `target_columns` is set, only positions that belong to those columns
    contribute to the loss (TimeXer-style exogenous forecasting) — covariate
    positions in `y` are set to -100 (F.cross_entropy's ignore_index).
    """

    __slots__ = (
        "block_size",
        "batch_size",
        "n_features",
        "target_columns",
        "device",
        "rng",
        "_train_serieses",
        "_val_serieses",
    )

    def __init__(
        self,
        dfs: list[pd.DataFrame],
        block_size: int,
        batch_size: int,
        vocab_size: int,
        order_of_derivative: "int | np.ndarray",
        domain_of_definition: np.ndarray,
        use_decimal: bool,
        device: str,
        train_part: float = 0.9,
        target_columns: Sequence[int] | None = None,
        rng: torch.Generator | None = None,
    ) -> None:
        assert len(dfs) > 0, "dfs must be non-empty"
        n_features = dfs[0].shape[1]
        assert all(df.shape[1] == n_features for df in dfs), (
            "all dataframes must have the same number of columns"
        )
        self.block_size = block_size
        self.batch_size = batch_size
        self.n_features = n_features
        self.target_columns = tuple(target_columns) if target_columns is not None else None
        self.device = device
        self.rng = rng if rng is not None else torch.Generator().manual_seed(42)
        # Encode each DataFrame into a flat token tuple, then split.
        self._train_serieses: list[tuple[int, ...]] = []
        self._val_serieses: list[tuple[int, ...]] = []
        for df in dfs:
            inp = df.to_numpy(dtype=np.float64)
            if use_decimal:
                inp = np_to_decimal(inp)
            _, _, encoded = encode(
                inp=inp,
                vocab_size=vocab_size,
                domain_of_definition=domain_of_definition,
                order_of_derivative=order_of_derivative,
                use_decimal=use_decimal,
            )
            flat = tuple(encoded.reshape(-1).tolist())
            n = int(len(flat) * train_part)
            self._train_serieses.append(flat[:n])
            self._val_serieses.append(flat[n:])

    def has_val(self) -> bool:
        return any(len(s) > self.block_size for s in self._val_serieses)

    def get_batch(self, split: str = "train") -> tuple[torch.Tensor, torch.Tensor]:
        if split == "train":
            pool = self._train_serieses
        elif split == "val":
            pool = self._val_serieses
        else:
            raise ValueError(f"unknown split {split!r}; expected 'train' or 'val'")
        block_size = self.block_size
        eligible = [s for s in pool if len(s) > block_size]
        if not eligible:
            max_len = max((len(s) for s in pool), default=0)
            raise ValueError(
                f"No {split!r} series long enough: block_size={block_size}, "
                f"max_series_len={max_len}"
            )
        starts_per = [len(s) - block_size for s in eligible]
        total_starts = sum(starts_per)
        cum = np.cumsum(starts_per)
        batch_size = self.batch_size
        x = torch.zeros((batch_size, block_size), dtype=torch.int64)
        y = torch.zeros((batch_size, block_size), dtype=torch.int64)
        picks = torch.randint(
            0, total_starts, (batch_size,), generator=self.rng
        ).tolist()
        for i, r in enumerate(picks):
            sidx = int(np.searchsorted(cum, r, side="right"))
            local = r - (int(cum[sidx - 1]) if sidx > 0 else 0)
            s = eligible[sidx]
            x[i] = torch.tensor(s[local : local + block_size], dtype=torch.int64)
            y[i] = torch.tensor(
                s[local + 1 : local + block_size + 1], dtype=torch.int64
            )
        y = _mask_targets(y, block_size, self.n_features, self.target_columns)
        return x.to(device=self.device), y.to(device=self.device)
