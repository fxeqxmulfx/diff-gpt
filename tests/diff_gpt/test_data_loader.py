import sqlite3
import pytest
import torch
import numpy as np
from typing import Tuple

from diff_gpt.data_loader import DiffSQLiteDataLoader
from diff_gpt.encoder_decoder import encode


# =============================================================================
# 1. HELPER: PASS-THROUGH ENCODER
# =============================================================================
def pass_through_encode(
    inp: np.ndarray,
    vocab_size: int,
    domain_of_definition: np.ndarray,
    order_of_derivative: int,
    use_decimal: bool,
) -> Tuple[None, None, torch.Tensor]:
    """
    A dummy encoder that injects into the class to bypass ML logic.
    It simulates derivative data loss (stripping first N rows)
    and flattens the result so we can verify exact numbers from the DB.
    """
    # Simulate data loss due to derivative order
    if len(inp) <= order_of_derivative:
        return None, None, torch.tensor([], dtype=torch.int64)

    # Slice off the history rows needed for calculation
    valid_data = inp[order_of_derivative:]

    # Flatten and return as Int64 tensor
    return None, None, torch.tensor(valid_data.flatten(), dtype=torch.int64)


# =============================================================================
# 2. FIXTURE: LINEAR DB (Simple)
# =============================================================================
@pytest.fixture
def linear_db_path(tmp_path):
    """
    Creates a DB with 1 table, strictly linear DT, standard 2 features.
    Used for math verification.
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
        # Values identify the row index: Row i -> [i, i*2]
        val1 = float(i)
        val2 = float(i * 2)
        data.append((dt, val1, val2))

    cursor.executemany("INSERT INTO sensor_data VALUES (?, ?, ?)", data)
    conn.commit()
    conn.close()

    return str(db_file)


# =============================================================================
# 3. FIXTURE: COMPLEX DB (Multi-table, Multi-shape)
# =============================================================================
@pytest.fixture
def complex_db_path(tmp_path):
    """
    Creates a DB with multiple tables of different sizes and column counts.
    Used for filtering logic and wide-table tests.
    """
    db_file = tmp_path / "test_complex.db"
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # 1. 'tiny': Too small for most block sizes (3 rows)
    cursor.execute("CREATE TABLE tiny (dt INTEGER PRIMARY KEY, val REAL)")
    cursor.executemany(
        "INSERT INTO tiny VALUES (?, ?)", [(1, 10.0), (2, 20.0), (3, 30.0)]
    )

    # 2. 'standard': 100 rows, 2 features (dt, f1, f2)
    cursor.execute("CREATE TABLE standard (dt INTEGER PRIMARY KEY, f1 REAL, f2 REAL)")
    std_data = [(1000 + i * 10, float(i), float(i * 2)) for i in range(100)]
    cursor.executemany("INSERT INTO standard VALUES (?, ?, ?)", std_data)

    # 3. 'wide': 20 rows, 4 features (dt, c1, c2, c3, c4)
    cursor.execute(
        "CREATE TABLE wide (dt INTEGER PRIMARY KEY, c1 REAL, c2 REAL, c3 REAL, c4 REAL)"
    )
    wide_data = []
    for i in range(20):
        # Val logic: row_idx + 0.1 * column_position
        # e.g., Row 0 -> [0.1, 0.2, 0.3, 0.4]
        row = (5000 + i * 100, i + 0.1, i + 0.2, i + 0.3, i + 0.4)
        wide_data.append(row)
    cursor.executemany("INSERT INTO wide VALUES (?, ?, ?, ?, ?)", wide_data)

    conn.commit()
    conn.close()
    return str(db_file)


# =============================================================================
# 4. TESTS: MATH & RETRIEVAL LOGIC
# =============================================================================


def test_initialization_metadata(linear_db_path):
    """Verifies internal calculations of range, step size, and token counts."""
    loader = DiffSQLiteDataLoader(
        block_size=10,
        batch_size=1,
        vocab_size=100,
        order_of_derivative=1,
        domain_of_definition=np.array([-1, 1]),
        use_decimal=False,
        device="cpu",
        database=linear_db_path,
        encode_data=pass_through_encode,
    )

    table = "sensor_data"
    # Range: 1000 to 1990
    assert loader.tables_range[table] == (1000, 1990)
    # Step: (1990-1000) / 99 = 10
    assert loader.tables_dt_diff[table] == 10
    # Tokens: 100 rows - 1 (order) = 99 valid rows. 2 features. 99*2 = 198.
    assert loader.tables_tokens_len[table] == 198
    loader.close()


def test_retrieval_order_0(linear_db_path):
    """Test direct value retrieval (Order 0)."""
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

    # Row 5 (dt=1050): [5.0, 10.0].
    # Start Index = 5 * 2 = 10.
    data = loader.get_data("sensor_data", 10, 12)
    assert len(data) == 2
    assert data[0] == 5
    assert data[1] == 10
    loader.close()


def test_retrieval_order_1_offset(linear_db_path):
    """Test that Order=1 correctly skips the first raw row."""
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

    # Request first valid tokens (Index 0).
    # Should correspond to Row 1 (calculated from Row 0).
    # Row 1 values: [1.0, 2.0]
    data = loader.get_data("sensor_data", 0, 2)
    assert data[0] == 1
    assert data[1] == 2
    loader.close()


def test_sub_row_slicing(linear_db_path):
    """Test requesting data that starts in the middle of a row."""
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
    # Request indices 11 to 13 -> [10, 6]
    data = loader.get_data("sensor_data", 11, 13)
    assert tuple(data) == (10, 6)
    loader.close()


def test_cross_row_slicing(linear_db_path):
    """Test requesting data that spans across two rows."""
    loader = DiffSQLiteDataLoader(
        block_size=1,
        batch_size=1,
        vocab_size=100,
        order_of_derivative=0,
        domain_of_definition=np.array([-1, 1]),
        use_decimal=False,
        device="cpu",
        database=linear_db_path,
        encode_data=pass_through_encode,
    )

    # Row 1 last val (2.0) -> Index 3
    # Row 2 first val (2.0) -> Index 4
    data = loader.get_data("sensor_data", 3, 5)
    assert tuple(data) == (2, 2)
    loader.close()


def test_exact_boundary_end(linear_db_path):
    """Test fetching the absolute last tokens in the table."""
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

    # Last row is 99. Indices 198, 199.
    data = loader.get_data("sensor_data", 198, 200)
    assert len(data) == 2
    assert data[0] == 99
    assert data[1] == 198
    loader.close()


# =============================================================================
# 5. TESTS: COMPLEX SCENARIOS & FILTERING
# =============================================================================


def test_table_filtering(complex_db_path):
    """Ensure small tables are ignored."""
    loader = DiffSQLiteDataLoader(
        block_size=5,
        batch_size=1,
        vocab_size=100,
        order_of_derivative=0,
        domain_of_definition=np.array([-1, 1]),
        use_decimal=False,
        device="cpu",
        database=complex_db_path,
        encode_data=pass_through_encode,
    )

    # 'tiny' (3 tokens) < 5+1. Should be removed.
    assert "tiny" not in loader.tables
    assert "standard" in loader.tables
    assert "wide" in loader.tables
    loader.close()


def test_wide_table_flattening(complex_db_path):
    """Test logic on tables with >2 columns."""
    loader = DiffSQLiteDataLoader(
        block_size=4,
        batch_size=1,
        vocab_size=100,
        order_of_derivative=0,
        domain_of_definition=np.array([-1, 1]),
        use_decimal=False,
        device="cpu",
        database=complex_db_path,
        encode_data=pass_through_encode,
    )

    # 'wide' table: Row 0 -> [0.1, 0.2, 0.3, 0.4] (Indices 0,1,2,3)
    #               Row 1 -> [1.1, 1.2, 1.3, 1.4] (Indices 4,5,6,7)

    # Get 0.3, 0.4, 1.1, 1.2 (Indices 2 to 6)
    # Cast to int by dummy encoder -> 0, 0, 1, 1
    data = loader.get_data("wide", 2, 6)
    assert tuple(data) == (0, 0, 1, 1)
    loader.close()


def test_get_batch_structure(complex_db_path):
    """Verify batch shapes, types, and device placement."""
    block_size = 5
    batch_size = 4
    loader = DiffSQLiteDataLoader(
        block_size=block_size,
        batch_size=batch_size,
        vocab_size=100,
        order_of_derivative=0,
        domain_of_definition=np.array([-1, 1]),
        use_decimal=False,
        device="cpu",
        database=complex_db_path,
        encode_data=pass_through_encode,
    )

    x, y = loader.get_batch()

    assert x.shape == (batch_size, block_size)
    assert y.shape == (batch_size, block_size)
    assert x.dtype == torch.int64
    assert y.dtype == torch.int64
    # Consistency check: y[i, 0] should be x[i, 1]
    assert x[0, 1] == y[0, 0]
    loader.close()


def test_stress_random_access(complex_db_path):
    """Run a loop to check for stability in random access."""
    loader = DiffSQLiteDataLoader(
        block_size=10,
        batch_size=16,
        vocab_size=100,
        order_of_derivative=1,
        domain_of_definition=np.array([-1, 1]),
        use_decimal=False,
        device="cpu",
        database=complex_db_path,
        encode_data=pass_through_encode,
    )

    for _ in range(20):
        x, y = loader.get_batch()
        assert x.shape == (16, 10)

    loader.close()


# =============================================================================
# 6. REGRESSION: real encoder with multi-feature + order_of_derivative > 0
# =============================================================================


@pytest.fixture
def wide_only_db_path(tmp_path):
    """DB with a single 4-feature table, used for real-encoder tests."""
    db_file = tmp_path / "wide_only.db"
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute(
        "CREATE TABLE wide (dt INTEGER PRIMARY KEY, c1 REAL, c2 REAL, c3 REAL, c4 REAL)"
    )
    wide_data = [
        (100 + i * 10, i + 0.1, i + 0.2, i + 0.3, i + 0.4) for i in range(200)
    ]
    cursor.executemany("INSERT INTO wide VALUES (?, ?, ?, ?, ?)", wide_data)
    conn.commit()
    conn.close()
    return str(db_file)


def test_get_data_with_real_encoder_multi_feature(wide_only_db_path):
    """
    Regression test for the `encoded.flatten()` fix in get_data.
    The real `encode` returns a 2D array (n_rows - order, n_features);
    without flattening, slicing by a 1D offset is wrong.
    """
    block_size = 8
    vocab_size = 256
    loader = DiffSQLiteDataLoader(
        block_size=block_size,
        batch_size=1,
        vocab_size=vocab_size,
        order_of_derivative=1,
        domain_of_definition=np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float64),
        use_decimal=False,
        device="cpu",
        database=wide_only_db_path,
        encode_data=encode,
    )

    data = loader.get_data("wide", 0, block_size + 1)
    assert len(data) == block_size + 1
    assert all(isinstance(v, int) for v in data)
    assert all(0 <= v < vocab_size for v in data)
    loader.close()


def test_get_batch_with_real_encoder(wide_only_db_path):
    """End-to-end sanity with the real encoder."""
    block_size = 8
    batch_size = 4
    loader = DiffSQLiteDataLoader(
        block_size=block_size,
        batch_size=batch_size,
        vocab_size=256,
        order_of_derivative=1,
        domain_of_definition=np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float64),
        use_decimal=False,
        device="cpu",
        database=wide_only_db_path,
        encode_data=encode,
    )
    for _ in range(10):
        x, y = loader.get_batch()
        assert x.shape == (batch_size, block_size)
        assert y.shape == (batch_size, block_size)
        assert x.dtype == torch.int64
    loader.close()


# =============================================================================
# 7. REGRESSION: get_batch does not recurse and tolerates short reads
# =============================================================================


def test_default_rng_is_not_shared_between_instances(linear_db_path):
    """
    Regression: the default `rng` parameter used to be evaluated once at import time
    (mutable default argument), causing two loaders to draw from the same stream.
    """
    a = DiffSQLiteDataLoader(
        block_size=4,
        batch_size=2,
        vocab_size=100,
        order_of_derivative=0,
        domain_of_definition=np.array([-1, 1]),
        use_decimal=False,
        device="cpu",
        database=linear_db_path,
        encode_data=pass_through_encode,
    )
    b = DiffSQLiteDataLoader(
        block_size=4,
        batch_size=2,
        vocab_size=100,
        order_of_derivative=0,
        domain_of_definition=np.array([-1, 1]),
        use_decimal=False,
        device="cpu",
        database=linear_db_path,
        encode_data=pass_through_encode,
    )
    # Different instances, but each seeded with 42 internally → both must
    # produce the same first batch, proving they own independent generators.
    xa, _ = a.get_batch()
    xb, _ = b.get_batch()
    assert torch.equal(xa, xb)
    # And the second draw of `a` should differ from its first — i.e. its state
    # advances independently of `b`.
    xa2, _ = a.get_batch()
    assert not torch.equal(xa, xa2)
    a.close()
    b.close()


def test_sql_identifier_quoting_for_weird_table_name(tmp_path):
    """Tables whose names contain punctuation must not break the loader."""
    db_file = tmp_path / "weird.db"
    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()
    weird_name = 'weird"name'
    quoted = '"' + weird_name.replace('"', '""') + '"'
    cursor.execute(f"CREATE TABLE {quoted} (dt INTEGER PRIMARY KEY, v REAL)")
    cursor.executemany(
        f"INSERT INTO {quoted} VALUES (?, ?)",
        [(i, float(i)) for i in range(50)],
    )
    conn.commit()
    conn.close()

    loader = DiffSQLiteDataLoader(
        block_size=4,
        batch_size=1,
        vocab_size=100,
        order_of_derivative=0,
        domain_of_definition=np.array([-1.0]),
        use_decimal=False,
        device="cpu",
        database=str(db_file),
        encode_data=pass_through_encode,
    )
    assert weird_name in loader.tables
    data = loader.get_data(weird_name, 0, 3)
    assert len(data) == 3
    loader.close()


def test_get_batch_tolerates_short_reads_without_recursion(complex_db_path):
    """
    Previously, get_batch called itself recursively when a short read occurred,
    which could overflow the stack. It now loops instead — verify by forcing
    short reads for the first few attempts and ensuring we still get a full batch.
    """
    block_size = 5
    batch_size = 4
    call_counter = {"n": 0}

    class FlakyLoader(DiffSQLiteDataLoader):
        def get_data(self, table, start, end):
            call_counter["n"] += 1
            if call_counter["n"] <= 3:
                return tuple()
            return super().get_data(table, start, end)

    loader = FlakyLoader(
        block_size=block_size,
        batch_size=batch_size,
        vocab_size=100,
        order_of_derivative=0,
        domain_of_definition=np.array([-1, 1]),
        use_decimal=False,
        device="cpu",
        database=complex_db_path,
        encode_data=pass_through_encode,
    )

    x, y = loader.get_batch()
    assert x.shape == (batch_size, block_size)
    assert y.shape == (batch_size, block_size)
    # Loop must have re-tried past the forced short reads
    assert call_counter["n"] > batch_size
    loader.close()
