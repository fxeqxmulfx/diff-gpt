import sqlite3
import pytest
import torch
import numpy as np
from typing import Tuple

from diff_gpt.data_loader import DiffSQLiteDataLoader


def pass_through_encode(
    inp: np.ndarray,
    vocab_size: int,
    domain_of_definition: np.ndarray,
    order_of_derivative: int,
    use_decimal: bool,
) -> Tuple[None, None, torch.Tensor]:
    """
    A dummy encoder that replaces diff_gpt.encoder_decoder.encode.
    It simply performs the derivative slicing (removing first N rows)
    and flattens the result so we can verify the exact numbers retrieved from DB.
    """
    # Simulate data loss due to derivative order
    if len(inp) <= order_of_derivative:
        return None, None, torch.tensor([], dtype=torch.int64)

    # Slice off the history rows needed for calculation
    valid_data = inp[order_of_derivative:]

    # Flatten and return as Int64 tensor for the dataloader
    # We cast input floats to ints just for easier assertion equality
    flattened = torch.tensor(valid_data.flatten(), dtype=torch.int64)
    return None, None, flattened


@pytest.fixture
def linear_db_path(tmp_path):
    """
    Creates a real SQLite file with strictly linear, consistent DT.
    """
    db_file = tmp_path / "test_linear.db"
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Table: 100 rows, DT starts at 1000, Step is 10.
    # Columns: dt (PK), col1, col2. (2 features)
    cursor.execute(
        "CREATE TABLE sensor_data (dt INTEGER PRIMARY KEY, col1 REAL, col2 REAL)"
    )

    data = []
    t_start = 1000
    t_step = 10
    n_rows = 100

    for i in range(n_rows):
        dt = t_start + (i * t_step)
        # Values identify the row index clearly
        # Row 0: 0, 0
        # Row 5: 5, 10
        val1 = float(i)
        val2 = float(i * 2)
        data.append((dt, val1, val2))

    cursor.executemany("INSERT INTO sensor_data VALUES (?, ?, ?)", data)
    conn.commit()
    conn.close()

    return str(db_file)


def test_initialization_math(linear_db_path):
    """
    Verifies that __init__ correctly calculates the step size (dt_diff)
    and total available tokens based on the DB statistics.
    """
    order = 1

    loader = DiffSQLiteDataLoader(
        block_size=10,
        batch_size=1,
        vocab_size=100,
        order_of_derivative=order,
        domain_of_definition=np.array([-1, 1]),
        use_decimal=False,
        device="cpu",
        database=linear_db_path,
        encode_data=pass_through_encode,
    )

    table = "sensor_data"

    # 1. Check Range: 1000 to (1000 + 99*10) = 1990
    assert loader.tables_range[table] == (1000, 1990)

    # 2. Check DT Diff: (1990 - 1000) // (100 - 1) = 990 // 99 = 10
    assert loader.tables_dt_diff[table] == 10

    # 3. Check Token Length
    # Total Rows = 100. Order = 1. Valid Rows = 99.
    # Features per row = 2.
    # Total tokens = 99 * 2 = 198.
    assert loader.tables_tokens_len[table] == 198

    loader.close()


def test_data_retrieval_order_0(linear_db_path):
    """
    Test direct retrieval (Order 0).
    Requesting specific tokens should map to specific rows perfectly.
    """
    loader = DiffSQLiteDataLoader(
        block_size=4,
        batch_size=1,
        vocab_size=100,
        order_of_derivative=0,
        domain_of_definition=np.array([-1, 1]),
        use_decimal=False,
        device="cpu",
        database=linear_db_path,
        encode_data=pass_through_encode,
    )

    # Row 5 in DB: dt=1050, col1=5.0, col2=10.0
    # Features = 2.
    # Token Index start for Row 5 = 5 * 2 = 10.

    start_idx = 10
    end_idx = 12  # Fetch 2 tokens (exactly Row 5)

    data = loader.get_data("sensor_data", start_idx, end_idx)

    assert len(data) == 2
    assert data[0] == 5
    assert data[1] == 10

    loader.close()


def test_data_retrieval_order_1(linear_db_path):
    """
    Test derivative logic.
    If Order=1, requesting index 0 means we need the FIRST valid derivative.
    This corresponds to Row 1 (calculated using Row 0).
    """
    loader = DiffSQLiteDataLoader(
        block_size=4,
        batch_size=1,
        vocab_size=100,
        order_of_derivative=1,
        domain_of_definition=np.array([-1, 1]),
        use_decimal=False,
        device="cpu",
        database=linear_db_path,
        encode_data=pass_through_encode,
    )

    # We request the first 2 available tokens.
    # Logic:
    # 1. effective_start = 0 + (1 * 2) = 2 (Skip Row 0 features)
    # 2. start_row_idx = 2 // 2 = 1
    # 3. raw_start_row = 1 - 1 = 0 (Fetch starting from Row 0 to compute Row 1)
    # 4. Encoder receives [Row 0, Row 1...].
    # 5. Encoder strips Row 0 (order 1).
    # 6. Result starts at Row 1 values.

    # Row 1 values: 1.0, 2.0
    data = loader.get_data("sensor_data", 0, 2)

    assert data[0] == 1
    assert data[1] == 2

    loader.close()


def test_offset_slicing(linear_db_path):
    """
    Test requesting a stream of tokens that starts in the *middle* of a row.
    """
    loader = DiffSQLiteDataLoader(
        block_size=4,
        batch_size=1,
        vocab_size=100,
        order_of_derivative=0,
        domain_of_definition=np.array([-1, 1]),
        use_decimal=False,
        device="cpu",
        database=linear_db_path,
        encode_data=pass_through_encode,
    )

    # Row 5: [5, 10] (Indices 10, 11)
    # Row 6: [6, 12] (Indices 12, 13)

    # Request indices 11 to 13.
    # Should get: [10, 6]
    # (The second value of Row 5, and first value of Row 6)

    data = loader.get_data("sensor_data", 11, 13)

    assert len(data) == 2
    assert data[0] == 10  # Row 5 col 2
    assert data[1] == 6  # Row 6 col 1

    loader.close()


def test_batch_generation(linear_db_path):
    """
    Test that get_batch returns tensors of correct shape and type
    without errors, using the real DB.
    """
    batch_size = 2
    block_size = 5

    loader = DiffSQLiteDataLoader(
        block_size=block_size,
        batch_size=batch_size,
        vocab_size=100,
        order_of_derivative=0,
        domain_of_definition=np.array([-1, 1]),
        use_decimal=False,
        device="cpu",
        database=linear_db_path,
        encode_data=pass_through_encode,
    )

    x, y = loader.get_batch()

    # Check Shapes
    assert x.shape == (batch_size, block_size)
    assert y.shape == (batch_size, block_size)

    # Check Type
    assert x.dtype == torch.int64
    assert y.dtype == torch.int64

    # Logic check:
    # Since get_data fetches (block_size + 1) -> temp
    # x = temp[:-1]
    # y = temp[1:]
    # Therefore, x[i, 1] must equal y[i, 0]
    assert x[0, 1] == y[0, 0]
    assert x[1, 1] == y[1, 0]

    loader.close()


def test_exact_sql_boundaries(linear_db_path):
    """
    Strictly verify the SQL 'BETWEEN' math.
    Ensure we don't miss a row due to integer math or off-by-one errors.
    """
    loader = DiffSQLiteDataLoader(
        block_size=2,
        batch_size=1,
        vocab_size=100,
        order_of_derivative=0,
        domain_of_definition=np.array([-1, 1]),
        use_decimal=False,
        device="cpu",
        database=linear_db_path,
        encode_data=pass_through_encode,
    )

    # Let's target the very last row. Row 99.
    # dt = 1000 + 990 = 1990.
    # Row 99 values: [99, 198].
    # Token indices: 99 * 2 = 198.

    start_idx = 198
    end_idx = 200

    data = loader.get_data("sensor_data", start_idx, end_idx)

    # If the SQL query calculated t_end short by even 1 unit,
    # BETWEEN would fail to include the last row (if step is large).
    # But since t_end is calculated exactly matching the PK, it must work.
    assert len(data) == 2
    assert data[0] == 99
    assert data[1] == 198

    loader.close()
