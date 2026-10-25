"""
A data loader that samples (x, y) token windows for `train` from SQLite.

`DiffSQLiteDataLoader` streams series from SQLite.
"""

import sqlite3
from collections.abc import Mapping
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch

from diff_gpt.application.data_loader import _fill
from diff_gpt.domain.encoder_decoder import (
    encode,
    observed_tokens,
    prefix_rows,
    resolve_tokenizer,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from diff_gpt.domain.encoder_decoder import Order


def _quote_ident(name: str) -> str:
    """Safely quote a SQLite identifier (table or column name)."""
    return '"' + name.replace('"', '""') + '"'


class Tokens(NamedTuple):
    """Tokens of a table and, for each, whether it reads no NaN (`observed_tokens`)."""

    values: tuple[int, ...]
    observed: tuple[bool, ...]


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

    NULL values, missing ones, are filled within each window that is read
    (`encoder_decoder.fill_missing`), and the targets whose tokens read one
    are masked with -100 (`encoder_decoder.observed_tokens`).
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
        ("lgt", "mash" or "mash-half"; None picks
        `encoder_decoder.DEFAULT_TOKENIZER`).
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

    def get_data(self, table: str, start: int, end: int) -> Tokens:
        """
        Return the tokens at flat positions [start, end) of `table`.

        Only the rows covering that range, plus the derivative prefix, are read
        and encoded, with their NaN filled; `observed` tells which tokens read
        none. If the range runs past the end of the table the result is
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
            return Tokens(values=(), observed=())
        arr = np.array(rows, dtype=np.float64)[:, 1:]
        _, _, encoded = self.encode_data(
            inp=_fill(arr),
            vocab_size=self.vocab_size,
            domain_of_definition=self._domain(table),
            order_of_derivative=self.order_of_derivative,
            tokenizer=self.tokenizer,
        )
        encoded = encoded.flatten()
        observed = observed_tokens(arr, self.order_of_derivative).reshape(-1)
        local_offset = effective_start % n_features
        part = slice(local_offset, local_offset + end - start)
        return Tokens(
            values=tuple(encoded[part].tolist()),
            observed=tuple(observed[part].tolist()),
        )

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
            tokens = self.get_data(
                table=table,
                start=data_idx,
                end=data_idx + block_size + 1,
            )
            if len(tokens.values) != block_size + 1:
                continue
            temp = torch.tensor(tokens.values, dtype=torch.int64)
            observed = torch.tensor(tokens.observed[1:], dtype=torch.bool)
            x[i] = temp[:-1]
            y[i] = temp[1:].masked_fill(~observed, -100)
            i += 1
        device = self.device
        x, y = x.to(device=device), y.to(device=device)
        return x, y
