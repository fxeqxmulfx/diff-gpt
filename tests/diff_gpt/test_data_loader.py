"""Tests for the SQLite data loader (DiffSQLiteDataLoader)."""

import sqlite3
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from diff_gpt.data_loader import DiffSQLiteDataLoader
from diff_gpt.encoder_decoder import Difference, encode

if TYPE_CHECKING:
    from pathlib import Path

# The linear DB (`linear_db_path`) has 100 rows; row i has dt = 1000 + 10 * i
# and the feature values [i, 2 * i].
LINEAR_DT_STEP = 10
# Tokens at order 1: 99 valid rows (100 - 1) x 2 features.
LINEAR_TOKENS_AT_ORDER_1 = 198
# Tokens in one row of the linear DB, one per feature.
TOKENS_PER_ROW = 2
# Values of the linear DB rows that the retrieval tests read back.
ROW_1_COL_1 = 2
ROW_5_COL_0 = 5
ROW_5_COL_1 = 10
LAST_ROW_COL_0 = 99
LAST_ROW_COL_1 = 198
# train_part=0.7 cuts the linear DB at token 140, so train values stay <= 138
# (row 69, col 1) and val values stay >= 70 (row 70, col 0).
TRAIN_MAX_VALUE = 138
VAL_MIN_VALUE = 70
# Tokens read back from the table with a punctuated name.
WEIRD_TABLE_READ_LEN = 3
# Leading get_data calls that FlakyLoader turns into short reads.
FORCED_SHORT_READS = 3


# =============================================================================
# 1. HELPER: PASS-THROUGH ENCODER
# =============================================================================
def pass_through_encode(
    inp: np.ndarray,
    vocab_size: int,
    domain_of_definition: np.ndarray,
    order_of_derivative: int,
    *,
    tokenizer: str,
) -> tuple[None, None, torch.Tensor]:
    """
    Return the raw values as int64 tokens instead of encoding them.

    A dummy encoder that injects into the class to bypass ML logic. It simulates
    derivative data loss (stripping first N rows) and flattens the result so we
    can verify exact numbers from the DB.
    """
    # Unused: these parameters only mirror encode()'s signature.
    del vocab_size, domain_of_definition, tokenizer
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
def linear_db_path(tmp_path: Path) -> str:
    """
    Create a DB with 1 table, strictly linear DT, standard 2 features.

    Used for math verification.
    """
    db_file = tmp_path / "test_linear.db"
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Table: 100 rows, DT starts at 1000, Step is 10.
    # Columns: dt (PK), col1, col2. (2 features)
    cursor.execute(
        "CREATE TABLE sensor_data (dt INTEGER PRIMARY KEY, col1 REAL, col2 REAL)",
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
def complex_db_path(tmp_path: Path) -> str:
    """
    Create a DB with multiple tables of different sizes and column counts.

    Used for filtering logic and wide-table tests.
    """
    db_file = tmp_path / "test_complex.db"
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # 1. 'tiny': Too small for most block sizes (3 rows)
    cursor.execute("CREATE TABLE tiny (dt INTEGER PRIMARY KEY, val REAL)")
    cursor.executemany(
        "INSERT INTO tiny VALUES (?, ?)",
        [(1, 10.0), (2, 20.0), (3, 30.0)],
    )

    # 2. 'standard': 100 rows, 2 features (dt, f1, f2)
    cursor.execute("CREATE TABLE standard (dt INTEGER PRIMARY KEY, f1 REAL, f2 REAL)")
    std_data = [(1000 + i * 10, float(i), float(i * 2)) for i in range(100)]
    cursor.executemany("INSERT INTO standard VALUES (?, ?, ?)", std_data)

    # 3. 'wide': 20 rows, 4 features (dt, c1, c2, c3, c4)
    cursor.execute(
        "CREATE TABLE wide (dt INTEGER PRIMARY KEY, "
        "c1 REAL, c2 REAL, c3 REAL, c4 REAL)",
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


def test_initialization_metadata(linear_db_path: str) -> None:
    """Check the internal calculations of range, step size, and token counts."""
    loader = DiffSQLiteDataLoader(
        block_size=10,
        batch_size=1,
        vocab_size=100,
        order_of_derivative=1,
        domain_of_definition=np.array([-1, 1]),
        device="cpu",
        database=linear_db_path,
        encode_data=pass_through_encode,
    )

    table = "sensor_data"
    # Range: 1000 to 1990
    assert loader.tables_range[table] == (1000, 1990)
    # Step between rows: (1990 - 1000) / 99 = 10
    assert loader.tables_dt_diff[table] == LINEAR_DT_STEP
    # Tokens: 100 rows - 1 (order) = 99 valid rows. 2 features. 99*2 = 198.
    assert loader.tables_tokens_len[table] == LINEAR_TOKENS_AT_ORDER_1
    loader.close()


def test_retrieval_order_0(linear_db_path: str) -> None:
    """Test direct value retrieval (Order 0)."""
    loader = DiffSQLiteDataLoader(
        block_size=4,
        batch_size=1,
        vocab_size=100,
        order_of_derivative=0,
        domain_of_definition=np.array([-1, 1]),
        device="cpu",
        database=linear_db_path,
        encode_data=pass_through_encode,
    )

    # Row 5 (dt=1050): [5.0, 10.0].
    # Start Index = 5 * 2 = 10.
    data = loader.get_data("sensor_data", 10, 12)
    assert len(data) == TOKENS_PER_ROW
    assert data[0] == ROW_5_COL_0
    assert data[1] == ROW_5_COL_1
    loader.close()


def test_retrieval_order_1_offset(linear_db_path: str) -> None:
    """Test that Order=1 correctly skips the first raw row."""
    loader = DiffSQLiteDataLoader(
        block_size=4,
        batch_size=1,
        vocab_size=100,
        order_of_derivative=1,
        domain_of_definition=np.array([-1, 1]),
        device="cpu",
        database=linear_db_path,
        encode_data=pass_through_encode,
    )

    # Request first valid tokens (Index 0).
    # Should correspond to Row 1 (calculated from Row 0).
    # Row 1 values: [1.0, 2.0]
    data = loader.get_data("sensor_data", 0, 2)
    assert data[0] == 1
    assert data[1] == ROW_1_COL_1
    loader.close()


def test_sub_row_slicing(linear_db_path: str) -> None:
    """Test requesting data that starts in the middle of a row."""
    loader = DiffSQLiteDataLoader(
        block_size=4,
        batch_size=1,
        vocab_size=100,
        order_of_derivative=0,
        domain_of_definition=np.array([-1, 1]),
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


def test_cross_row_slicing(linear_db_path: str) -> None:
    """Test requesting data that spans across two rows."""
    loader = DiffSQLiteDataLoader(
        block_size=1,
        batch_size=1,
        vocab_size=100,
        order_of_derivative=0,
        domain_of_definition=np.array([-1, 1]),
        device="cpu",
        database=linear_db_path,
        encode_data=pass_through_encode,
    )

    # Row 1 last val (2.0) -> Index 3
    # Row 2 first val (2.0) -> Index 4
    data = loader.get_data("sensor_data", 3, 5)
    assert tuple(data) == (2, 2)
    loader.close()


def test_exact_boundary_end(linear_db_path: str) -> None:
    """Test fetching the absolute last tokens in the table."""
    loader = DiffSQLiteDataLoader(
        block_size=2,
        batch_size=1,
        vocab_size=100,
        order_of_derivative=0,
        domain_of_definition=np.array([-1, 1]),
        device="cpu",
        database=linear_db_path,
        encode_data=pass_through_encode,
    )

    # Last row is 99. Indices 198, 199.
    data = loader.get_data("sensor_data", 198, 200)
    assert len(data) == TOKENS_PER_ROW
    assert data[0] == LAST_ROW_COL_0
    assert data[1] == LAST_ROW_COL_1
    loader.close()


# =============================================================================
# 5. TESTS: COMPLEX SCENARIOS & FILTERING
# =============================================================================


def test_table_filtering(complex_db_path: str) -> None:
    """Ensure small tables are ignored."""
    loader = DiffSQLiteDataLoader(
        block_size=5,
        batch_size=1,
        vocab_size=100,
        order_of_derivative=0,
        domain_of_definition=np.array([-1, 1]),
        device="cpu",
        database=complex_db_path,
        encode_data=pass_through_encode,
    )

    # 'tiny' (3 tokens) < 5+1. Should be removed.
    assert "tiny" not in loader.tables
    assert "standard" in loader.tables
    assert "wide" in loader.tables
    loader.close()


def test_wide_table_flattening(complex_db_path: str) -> None:
    """Test logic on tables with >2 columns."""
    loader = DiffSQLiteDataLoader(
        block_size=4,
        batch_size=1,
        vocab_size=100,
        order_of_derivative=0,
        domain_of_definition=np.array([-1, 1]),
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


def test_get_batch_structure(complex_db_path: str) -> None:
    """Verify batch shapes, types, and device placement."""
    block_size = 5
    batch_size = 4
    loader = DiffSQLiteDataLoader(
        block_size=block_size,
        batch_size=batch_size,
        vocab_size=100,
        order_of_derivative=0,
        domain_of_definition=np.array([-1, 1]),
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


def test_split_defaults_to_train_only(linear_db_path: str) -> None:
    """Without train_part, the SQLite loader exposes no val split."""
    loader = DiffSQLiteDataLoader(
        block_size=10,
        batch_size=2,
        vocab_size=100,
        order_of_derivative=0,
        domain_of_definition=np.array([-1, 1]),
        device="cpu",
        database=linear_db_path,
        encode_data=pass_through_encode,
    )
    assert loader.has_val() is False
    with pytest.raises(RuntimeError, match="no val split"):
        loader.get_batch("val")
    loader.close()


def test_unknown_split_raises(linear_db_path: str) -> None:
    """Check that get_batch raises ValueError for an unknown split."""
    loader = DiffSQLiteDataLoader(
        block_size=10,
        batch_size=2,
        vocab_size=100,
        order_of_derivative=0,
        domain_of_definition=np.array([-1, 1]),
        device="cpu",
        database=linear_db_path,
        encode_data=pass_through_encode,
    )
    with pytest.raises(ValueError, match="unknown split"):
        loader.get_batch("test")
    loader.close()


def test_sqlite_train_val_split_disjoint(linear_db_path: str) -> None:
    """
    Check that train_part=0.7 adds a val split disjoint from the train split.

    With train_part=0.7 on the linear DB (100 rows x 2 cols = 200 tokens),
    train samples come from tokens [0, 140) and val from [140, 200).
    Because values are row-indexed (row i → [i, 2i]), we can check that
    train tokens always fall below the boundary value and val always above.
    """
    torch.manual_seed(0)
    block_size = 20
    loader = DiffSQLiteDataLoader(
        block_size=block_size,
        batch_size=8,
        vocab_size=100,
        order_of_derivative=0,
        domain_of_definition=np.array([-1, 1]),
        device="cpu",
        database=linear_db_path,
        encode_data=pass_through_encode,
        train_part=0.7,
    )
    assert loader.has_val() is True
    # Train window start range is [0, 140 - 20) → last accessible token index 139
    # maps to row 69 col 1 = value 138. So train values ≤ 138.
    # Val window start range is [140, 200 - 20) → first token index 140 maps
    # to row 70 col 0 = value 70; so val values ≥ 70.
    for _ in range(15):
        x_tr, _ = loader.get_batch("train")
        assert x_tr.max().item() <= TRAIN_MAX_VALUE
        x_val, _ = loader.get_batch("val")
        assert x_val.min().item() >= VAL_MIN_VALUE
    loader.close()


def test_stress_random_access(complex_db_path: str) -> None:
    """Run a loop to check for stability in random access."""
    loader = DiffSQLiteDataLoader(
        block_size=10,
        batch_size=16,
        vocab_size=100,
        order_of_derivative=1,
        domain_of_definition=np.array([-1, 1]),
        device="cpu",
        database=complex_db_path,
        encode_data=pass_through_encode,
    )

    for _ in range(20):
        x, _y = loader.get_batch()
        assert x.shape == (16, 10)

    loader.close()


# =============================================================================
# 6. REGRESSION: real encoder with multi-feature + order_of_derivative > 0
# =============================================================================


@pytest.fixture
def wide_only_db_path(tmp_path: Path) -> str:
    """DB with a single 4-feature table, used for real-encoder tests."""
    db_file = tmp_path / "wide_only.db"
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute(
        "CREATE TABLE wide (dt INTEGER PRIMARY KEY, "
        "c1 REAL, c2 REAL, c3 REAL, c4 REAL)",
    )
    wide_data = [(100 + i * 10, i + 0.1, i + 0.2, i + 0.3, i + 0.4) for i in range(200)]
    cursor.executemany("INSERT INTO wide VALUES (?, ?, ?, ?, ?)", wide_data)
    conn.commit()
    conn.close()
    return str(db_file)


def test_get_data_with_real_encoder_multi_feature(wide_only_db_path: str) -> None:
    """
    Check that get_data with the real encoder returns block_size + 1 in-vocab ints.

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
        device="cpu",
        database=wide_only_db_path,
        encode_data=encode,
    )

    data = loader.get_data("wide", 0, block_size + 1)
    assert len(data) == block_size + 1
    assert all(isinstance(v, int) for v in data)
    assert all(0 <= v < vocab_size for v in data)
    loader.close()


def test_get_batch_with_real_encoder(wide_only_db_path: str) -> None:
    """End-to-end sanity with the real encoder."""
    block_size = 8
    batch_size = 4
    loader = DiffSQLiteDataLoader(
        block_size=block_size,
        batch_size=batch_size,
        vocab_size=256,
        order_of_derivative=1,
        domain_of_definition=np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float64),
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


def test_default_rng_is_not_shared_between_instances(linear_db_path: str) -> None:
    """
    Check that loaders with the default rng each own an independent generator.

    Regression: the default `rng` parameter used to be evaluated once at import time
    (mutable default argument), causing two loaders to draw from the same stream.
    """
    a = DiffSQLiteDataLoader(
        block_size=4,
        batch_size=2,
        vocab_size=100,
        order_of_derivative=0,
        domain_of_definition=np.array([-1, 1]),
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


def test_sql_identifier_quoting_for_weird_table_name(tmp_path: Path) -> None:
    """Tables whose names contain punctuation must not break the loader."""
    db_file = tmp_path / "weird.db"
    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()
    weird_name = 'weird"name'
    quoted = '"' + weird_name.replace('"', '""') + '"'
    cursor.execute(f"CREATE TABLE {quoted} (dt INTEGER PRIMARY KEY, v REAL)")
    cursor.executemany(
        # The table name is a test constant, not user input.
        f"INSERT INTO {quoted} VALUES (?, ?)",  # noqa: S608
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
        device="cpu",
        database=str(db_file),
        encode_data=pass_through_encode,
    )
    assert weird_name in loader.tables
    data = loader.get_data(weird_name, 0, WEIRD_TABLE_READ_LEN)
    assert len(data) == WEIRD_TABLE_READ_LEN
    loader.close()


def test_get_batch_tolerates_short_reads_without_recursion(
    complex_db_path: str,
) -> None:
    """
    Check that get_batch retries past short reads and still returns a full batch.

    Previously, get_batch called itself recursively when a short read occurred,
    which could overflow the stack. It now loops instead — verify by forcing
    short reads for the first few attempts and ensuring we still get a full batch.
    """
    block_size = 5
    batch_size = 4
    call_counter = {"n": 0}

    class FlakyLoader(DiffSQLiteDataLoader):
        def get_data(self, table: str, start: int, end: int) -> tuple[int, ...]:
            call_counter["n"] += 1
            if call_counter["n"] <= FORCED_SHORT_READS:
                return ()
            return super().get_data(table, start, end)

    loader = FlakyLoader(
        block_size=block_size,
        batch_size=batch_size,
        vocab_size=100,
        order_of_derivative=0,
        domain_of_definition=np.array([-1, 1]),
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


# =============================================================================
# 8. Splits without room, the real encoder, per-table domains, the tokenizer
# =============================================================================


def test_get_batch_raises_when_no_train_part_has_room(linear_db_path: str) -> None:
    """
    Check that get_batch('train') raises when no table's train part fits a window.

    The table has 198 tokens, more than block_size + 1, so it is sampled from,
    but train_part=0.7 leaves it a 138-token train part, shorter than
    block_size. get_batch used to redraw forever.
    """
    loader = DiffSQLiteDataLoader(
        block_size=150,
        batch_size=2,
        vocab_size=100,
        order_of_derivative=1,
        domain_of_definition=np.array([-1, 1]),
        device="cpu",
        database=linear_db_path,
        encode_data=pass_through_encode,
        train_part=0.7,
    )
    assert "sensor_data" in loader.tables
    assert loader.has_val() is False
    with pytest.raises(ValueError, match="No 'train' table long enough"):
        loader.get_batch("train")
    loader.close()


def _wide_rows() -> np.ndarray:
    """Return the feature rows of the `wide_only_db_path` table."""
    i = np.arange(200, dtype=np.float64)[:, None]
    return i + np.array([0.1, 0.2, 0.3, 0.4])


def test_loader_returns_the_tokens_of_the_real_encoder(wide_only_db_path: str) -> None:
    """Check that the loader hands the rows to `encode` and returns its tokens."""
    block_size = 8
    domain = np.array([2.0, 2.0, 2.0, 2.0])
    loader = DiffSQLiteDataLoader(
        block_size=block_size,
        batch_size=2,
        vocab_size=256,
        order_of_derivative=1,
        domain_of_definition=domain,
        device="cpu",
        database=wide_only_db_path,
    )
    _, _, expected = encode(_wide_rows(), 256, domain, 1)
    data = loader.get_data("wide", 0, block_size + 1)
    assert data == tuple(expected.reshape(-1)[: block_size + 1].tolist())
    x, _ = loader.get_batch("train")
    assert x.shape == (2, block_size)
    loader.close()


def test_tables_of_different_widths_take_a_domain_each(complex_db_path: str) -> None:
    """
    Check that a mapping gives every table its own domain_of_definition.

    One domain array served every table, so with the real encoder a 2-column
    and a 4-column table could not be loaded together.
    """
    domains = {
        "standard": np.array([100.0, 200.0]),
        "wide": np.array([20.0, 20.0, 20.0, 20.0]),
    }
    loader = DiffSQLiteDataLoader(
        block_size=5,
        batch_size=16,
        vocab_size=256,
        order_of_derivative=0,
        domain_of_definition=domains,
        device="cpu",
        database=complex_db_path,
    )
    assert set(loader.tables) == {"standard", "wide"}
    for table, rows in (
        ("standard", np.array([[i, 2 * i] for i in range(100)], dtype=np.float64)),
        ("wide", np.arange(20.0)[:, None] + np.array([0.1, 0.2, 0.3, 0.4])),
    ):
        _, _, expected = encode(rows, 256, domains[table], 0)
        assert loader.get_data(table, 0, 6) == tuple(
            expected.reshape(-1)[:6].tolist(),
        )
    x, _ = loader.get_batch("train")
    assert x.shape == (16, 5)
    loader.close()


def test_a_domain_mapping_must_cover_every_sampled_table(complex_db_path: str) -> None:
    """Check that a domain mapping without a sampled table raises ValueError."""
    with pytest.raises(ValueError, match=r"no entry for tables \['wide'\]"):
        DiffSQLiteDataLoader(
            block_size=5,
            batch_size=1,
            vocab_size=256,
            order_of_derivative=0,
            domain_of_definition={"standard": np.array([100.0, 200.0])},
            device="cpu",
            database=complex_db_path,
        )


@pytest.mark.parametrize("tokenizer", ["lgt", "mash"])
def test_sqlite_loader_encodes_with_its_tokenizer(
    wide_only_db_path: str,
    tokenizer: str,
) -> None:
    """Check that the loader encodes with the tokenizer it was given."""
    # The rows are ramps of slope 1, here 50.4 levels per row: off the
    # integers, so LGT's rounding varies along them, while the MASH grid moves
    # with them and gives V/2 throughout. (A domain of 2 gives 63 levels per
    # row, and both tokenizers V/2.)
    domain = np.array([2.5, 2.5, 2.5, 2.5])
    rows = _wide_rows()
    tokens = {
        tok: encode(rows, 256, domain, 2, tokenizer=tok)[2] for tok in ("lgt", "mash")
    }
    assert not np.array_equal(tokens["lgt"], tokens["mash"])
    loader = DiffSQLiteDataLoader(
        block_size=8,
        batch_size=1,
        vocab_size=256,
        order_of_derivative=2,
        domain_of_definition=domain,
        device="cpu",
        database=wide_only_db_path,
        tokenizer=tokenizer,
    )
    n_tokens = loader.get_data_len("wide")
    assert loader.get_data("wide", 0, n_tokens) == tuple(
        tokens[tokenizer].reshape(-1).tolist(),
    )
    loader.close()


# A seasonal operator: 25 raw rows in front of the tokens.
SEASONAL = Difference(1, 24)
SEASONAL_PREFIX = 25


def test_sqlite_loader_offsets_follow_the_degree_of_the_operator(
    tmp_path: Path,
) -> None:
    """
    Check that a seasonal operator shifts the token offsets by its 25 rows.

    LGT tokens depend only on the rows they difference, so every range the
    loader encodes on its own matches the encoding of the whole table.
    """
    rows = np.cumsum(np.random.default_rng(0).normal(size=(80, 2)), axis=0)
    db_file = tmp_path / "seasonal.db"
    conn = sqlite3.connect(db_file)
    conn.execute("CREATE TABLE hourly (dt INTEGER PRIMARY KEY, a REAL, b REAL)")
    conn.executemany(
        "INSERT INTO hourly VALUES (?, ?, ?)",
        [(3600 * i, float(a), float(b)) for i, (a, b) in enumerate(rows)],
    )
    conn.commit()
    conn.close()
    domain = np.array([10.0, 10.0])
    loader = DiffSQLiteDataLoader(
        block_size=8,
        batch_size=1,
        vocab_size=256,
        order_of_derivative=SEASONAL,
        domain_of_definition=domain,
        device="cpu",
        database=str(db_file),
        tokenizer="lgt",
    )
    n_tokens = (len(rows) - SEASONAL_PREFIX) * rows.shape[1]
    assert loader.get_data_len("hourly") == n_tokens
    _, _, tokens = encode(rows, 256, domain, SEASONAL, tokenizer="lgt")
    flat = tokens.reshape(-1).tolist()
    for start, end in ((0, 9), (5, 30), (n_tokens - 7, n_tokens)):
        assert loader.get_data("hourly", start, end) == tuple(flat[start:end])
    loader.close()
