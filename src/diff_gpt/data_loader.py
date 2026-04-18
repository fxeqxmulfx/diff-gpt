import sqlite3
from typing import Callable
import numpy as np
import torch

from diff_gpt.encoder_decoder import encode


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

    def get_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        block_size = self.block_size
        batch_size = self.batch_size
        tables = self.tables
        tables_len = self.tables_len
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
            data_len = self.get_data_len(table)
            data_idx = int(
                torch.randint(
                    low=0,
                    high=data_len - block_size,
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
