"""
Σ-Δ (MASH) encoder tests.

Thresholds below are empirical maxima of the prefix-anchored Σ-Δ path, so the
tokenizer is pinned to "mash" explicitly; the level-grid encoder (the package
default) is covered by test_lgt.py.
"""

from functools import partial

import numpy as np
import pytest

from diff_gpt.encoder_decoder import decode as _decode
from diff_gpt.encoder_decoder import encode as _encode
from diff_gpt.encoder_decoder import (
    get_domain_of_definition,
    np_to_decimal,
)

encode = partial(_encode, tokenizer="mash")
decode = partial(_decode, tokenizer="mash")

# Lowest derivative order that reserves 2^k levels of headroom in the
# vocabulary (MASH cascade, when V > 2^k + 2).
MIN_HEADROOM_ORDER = 2
# Empirical maxima of the round-trip error.
# Order 0 stays within Δ/2 = D / (V - 2) ≈ 1/254 (README Theorem 1).
SIN_COS_ORDER_0_MAX_ERR = 0.003937002946875687
SIN_COS_ORDER_1_MAX_ERR = 0.003775003594016138
SIN_COS_ORDER_2_MAX_ERR = 0.000225
SIN_COS_ORDER_3_MAX_ERR = 1.35e-5
FINE_SIN_COS_ORDER_3_MAX_ERR = 2.05337387573312e-06
SMALL_CONST_MAX_ERR = 2.783148670569062e-30


@pytest.mark.slow
def test_diff_encoder_decoder_sin_cos() -> None:
    """Check that the order-0 round trip of Decimal sin/cos stays within its bound."""
    vocab_size = 256
    idx = np.arange(1_000_000, dtype=np.float64)
    inp = np.stack(
        (
            np.sin(idx),
            np.cos(idx),
        ),
        axis=1,
    )
    inp = np_to_decimal(inp)
    order_of_derivative = 0
    use_decimal = True
    domain_of_definition = get_domain_of_definition(
        inp=inp,
        order_of_derivative=order_of_derivative,
        use_decimal=use_decimal,
    )
    start, scale, encoded_inp = encode(
        inp=inp,
        vocab_size=vocab_size,
        domain_of_definition=domain_of_definition,
        order_of_derivative=order_of_derivative,
        use_decimal=use_decimal,
    )
    assert np.all((encoded_inp >= 0) & (encoded_inp < vocab_size))
    decoded_inp = decode(
        start=start,
        scale=scale,
        inp=encoded_inp,
        vocab_size=vocab_size,
        order_of_derivative=order_of_derivative,
        use_decimal=use_decimal,
    )
    assert float(np.max(np.abs(decoded_inp - inp))) <= SIN_COS_ORDER_0_MAX_ERR


@pytest.mark.slow
def test_diff_encoder_decoder_sin_cos_derivative() -> None:
    """Check that the order-1 round trip of Decimal sin/cos stays within its bound."""
    vocab_size = 256
    idx = np.arange(1_000_000, dtype=np.float64)
    inp = np.stack(
        (
            np.sin(idx),
            np.cos(idx),
        ),
        axis=1,
    )
    inp = np_to_decimal(inp)
    order_of_derivative = 1
    use_decimal = True
    domain_of_definition = get_domain_of_definition(
        inp=inp,
        order_of_derivative=order_of_derivative,
        use_decimal=use_decimal,
    )
    start, scale, encoded_inp = encode(
        inp=inp,
        vocab_size=vocab_size,
        domain_of_definition=domain_of_definition,
        order_of_derivative=order_of_derivative,
        use_decimal=use_decimal,
    )
    assert np.all((encoded_inp >= 0) & (encoded_inp < vocab_size))
    decoded_inp = decode(
        start=start,
        scale=scale,
        inp=encoded_inp,
        vocab_size=vocab_size,
        order_of_derivative=order_of_derivative,
        use_decimal=use_decimal,
    )
    assert float(np.max(np.abs(decoded_inp - inp))) <= SIN_COS_ORDER_1_MAX_ERR


@pytest.mark.slow
def test_diff_encoder_decoder_sin_cos_second_derivative() -> None:
    """Check that the order-2 round trip of Decimal sin/cos stays within its bound."""
    vocab_size = 4096
    idx = np.arange(1_000_000, dtype=np.float64)
    inp = np.stack(
        (
            np.sin(idx),
            np.cos(idx),
        ),
        axis=1,
    )
    inp = np_to_decimal(inp)
    order_of_derivative = 2
    use_decimal = True
    domain_of_definition = get_domain_of_definition(
        inp=inp,
        order_of_derivative=order_of_derivative,
        use_decimal=use_decimal,
    )
    start, scale, encoded_inp = encode(
        inp=inp,
        vocab_size=vocab_size,
        domain_of_definition=domain_of_definition,
        order_of_derivative=order_of_derivative,
        use_decimal=use_decimal,
    )
    assert np.all((encoded_inp >= 0) & (encoded_inp < vocab_size))
    decoded_inp = decode(
        start=start,
        scale=scale,
        inp=encoded_inp,
        vocab_size=vocab_size,
        order_of_derivative=order_of_derivative,
        use_decimal=use_decimal,
    )
    assert float(np.max(np.abs(decoded_inp - inp))) <= SIN_COS_ORDER_2_MAX_ERR


@pytest.mark.slow
def test_diff_encoder_decoder_sin_cos_third_derivative() -> None:
    """Check that the order-3 round trip of Decimal sin/cos stays within its bound."""
    vocab_size = 65536
    idx = np.arange(1_000_000, dtype=np.float64)
    inp = np.stack(
        (
            np.sin(idx),
            np.cos(idx),
        ),
        axis=1,
    )
    inp = np_to_decimal(inp)
    order_of_derivative = 3
    use_decimal = True
    domain_of_definition = get_domain_of_definition(
        inp=inp,
        order_of_derivative=order_of_derivative,
        use_decimal=use_decimal,
    )
    start, scale, encoded_inp = encode(
        inp=inp,
        vocab_size=vocab_size,
        domain_of_definition=domain_of_definition,
        order_of_derivative=order_of_derivative,
        use_decimal=use_decimal,
    )
    assert np.all((encoded_inp >= 0) & (encoded_inp < vocab_size))
    decoded_inp = decode(
        start=start,
        scale=scale,
        inp=encoded_inp,
        vocab_size=vocab_size,
        order_of_derivative=order_of_derivative,
        use_decimal=use_decimal,
    )
    assert float(np.max(np.abs(decoded_inp - inp))) <= SIN_COS_ORDER_3_MAX_ERR


@pytest.mark.slow
def test_diff_encoder_decoder_sin_cos_third_derivative_2() -> None:
    """Check the order-3 bound on one finely sampled sin/cos period."""
    vocab_size = 256
    idx = np.arange(0, 2 * np.pi, 0.000001, dtype=np.float64)
    inp = np.stack(
        (
            np.sin(idx),
            np.cos(idx),
        ),
        axis=1,
    )
    inp = np_to_decimal(inp)
    order_of_derivative = 3
    use_decimal = True
    domain_of_definition = get_domain_of_definition(
        inp=inp,
        order_of_derivative=order_of_derivative,
        use_decimal=use_decimal,
    )
    start, scale, encoded_inp = encode(
        inp=inp,
        vocab_size=vocab_size,
        domain_of_definition=domain_of_definition,
        order_of_derivative=order_of_derivative,
        use_decimal=use_decimal,
    )
    assert np.all((encoded_inp >= 0) & (encoded_inp < vocab_size))
    decoded_inp = decode(
        start=start,
        scale=scale,
        inp=encoded_inp,
        vocab_size=vocab_size,
        order_of_derivative=order_of_derivative,
        use_decimal=use_decimal,
    )
    assert float(np.max(np.abs(decoded_inp - inp))) <= FINE_SIN_COS_ORDER_3_MAX_ERR


@pytest.mark.slow
def test_diff_encoder_sin_cos_different_lenght() -> None:
    """Check that encoding a prefix keeps the start, scale and leading tokens."""
    vocab_size = 256
    idx = np.arange(1_000_000, dtype=np.float64)
    inp = np.stack(
        (
            np.sin(idx),
            np.cos(idx),
        ),
        axis=1,
    )
    inp = np_to_decimal(inp)
    order_of_derivative = 1
    use_decimal = True
    domain_of_definition = get_domain_of_definition(
        inp=inp,
        order_of_derivative=order_of_derivative,
        use_decimal=use_decimal,
    )
    start, scale, encoded_inp = encode(
        inp=inp,
        vocab_size=vocab_size,
        domain_of_definition=domain_of_definition,
        order_of_derivative=order_of_derivative,
        use_decimal=use_decimal,
    )
    assert np.all((encoded_inp >= 0) & (encoded_inp < vocab_size))
    start_2, scale_2, encoded_inp_2 = encode(
        inp=inp[:16],
        vocab_size=vocab_size,
        domain_of_definition=domain_of_definition,
        order_of_derivative=order_of_derivative,
        use_decimal=use_decimal,
    )
    assert np.all((encoded_inp_2 >= 0) & (encoded_inp_2 < vocab_size))
    assert np.all(start == start_2)
    assert np.all(scale == scale_2)
    assert np.all(encoded_inp[:8] == encoded_inp_2[:8])


@pytest.mark.slow
def test_diff_encoder_decoder_lin() -> None:
    """Check that order 1 restores linear series exactly with 4 tokens."""
    vocab_size = 4
    idx = np.arange(1_000_000, dtype=np.float64)
    inp = np.stack(
        (
            idx,
            np.flip(idx, (0,)),
        ),
        axis=1,
    )
    inp = np_to_decimal(inp)
    order_of_derivative = 1
    use_decimal = True
    domain_of_definition = get_domain_of_definition(
        inp=inp,
        order_of_derivative=order_of_derivative,
        use_decimal=use_decimal,
    )
    start, scale, encoded_inp = encode(
        inp=inp,
        vocab_size=vocab_size,
        domain_of_definition=domain_of_definition,
        order_of_derivative=order_of_derivative,
        use_decimal=use_decimal,
    )
    assert np.all((encoded_inp >= 0) & (encoded_inp < vocab_size))
    decoded_inp = decode(
        start=start,
        scale=scale,
        inp=encoded_inp,
        vocab_size=vocab_size,
        order_of_derivative=order_of_derivative,
        use_decimal=use_decimal,
    )
    assert float(np.max(np.abs(decoded_inp - inp))) == 0


@pytest.mark.slow
def test_diff_encoder_decoder_const() -> None:
    """Check that order 1 restores a constant series exactly."""
    vocab_size = 4
    inp = np.ones(1_000_000, dtype=np.float64).reshape(-1, 1)
    inp = np_to_decimal(inp)
    order_of_derivative = 1
    use_decimal = True
    domain_of_definition = get_domain_of_definition(
        inp=inp,
        order_of_derivative=order_of_derivative,
        use_decimal=use_decimal,
    )
    start, scale, encoded_inp = encode(
        inp=inp,
        vocab_size=vocab_size,
        domain_of_definition=domain_of_definition,
        order_of_derivative=order_of_derivative,
        use_decimal=use_decimal,
    )
    assert np.all((encoded_inp >= 0) & (encoded_inp < vocab_size))
    decoded_inp = decode(
        start=start,
        scale=scale,
        inp=encoded_inp,
        vocab_size=vocab_size,
        order_of_derivative=order_of_derivative,
        use_decimal=use_decimal,
    )
    assert float(np.max(np.abs(decoded_inp - inp))) == 0


@pytest.mark.slow
def test_diff_encoder_decoder_small_const() -> None:
    """Check that the order-1 round trip of a small constant stays within its bound."""
    vocab_size = 4
    inp = (np.ones(1_000_000, dtype=np.float64) * 0.01).reshape(-1, 1)
    inp = np_to_decimal(inp)
    order_of_derivative = 1
    use_decimal = True
    domain_of_definition = get_domain_of_definition(
        inp=inp,
        order_of_derivative=order_of_derivative,
        use_decimal=use_decimal,
    )
    start, scale, encoded_inp = encode(
        inp=inp,
        vocab_size=vocab_size,
        domain_of_definition=domain_of_definition,
        order_of_derivative=order_of_derivative,
        use_decimal=use_decimal,
    )
    assert np.all((encoded_inp >= 0) & (encoded_inp < vocab_size))
    decoded_inp = decode(
        start=start,
        scale=scale,
        inp=encoded_inp,
        vocab_size=vocab_size,
        order_of_derivative=order_of_derivative,
        use_decimal=use_decimal,
    )
    assert float(np.max(np.abs(decoded_inp - inp))) <= SMALL_CONST_MAX_ERR


@pytest.mark.slow
def test_diff_encoder_decoder_mix() -> None:
    """Check the order-1 bound on mixed sin/cos, linear and constant columns."""
    vocab_size = 256
    const = np.ones(1_000_000, dtype=np.float64)
    small_const = np.ones(1_000_000, dtype=np.float64) * 0.01
    idx = np.arange(1_000_000, dtype=np.float64)
    inp = np.stack(
        (
            np.sin(idx),
            np.cos(idx),
            idx,
            np.flip(idx, (0,)),
            const,
            small_const,
        ),
        axis=1,
    )
    inp = np_to_decimal(inp)
    order_of_derivative = 1
    use_decimal = True
    domain_of_definition = get_domain_of_definition(
        inp=inp,
        order_of_derivative=order_of_derivative,
        use_decimal=use_decimal,
    )
    start, scale, encoded_inp = encode(
        inp=inp,
        vocab_size=vocab_size,
        domain_of_definition=domain_of_definition,
        order_of_derivative=order_of_derivative,
        use_decimal=use_decimal,
    )
    assert np.all((encoded_inp >= 0) & (encoded_inp < vocab_size))
    decoded_inp = decode(
        start=start,
        scale=scale,
        inp=encoded_inp,
        vocab_size=vocab_size,
        order_of_derivative=1,
        use_decimal=use_decimal,
    )
    # The sin/cos columns set the maximum, as in the order-1 sin/cos test.
    assert float(np.max(np.abs(decoded_inp - inp))) <= SIN_COS_ORDER_1_MAX_ERR


def test_kth_order_mash_bound_is_tight_and_uniform_in_T() -> None:
    """
    Check the tight MASH k-th-order Σ-Δ bound up to k = 4 in float64.

    Reconstruction error ≤ small constant times Δ/2 up to k = 4 in IEEE
    float64. Beyond that, the k-fold cumsum in the decoder loses mantissa bits
    faster than the noise-shaping recovers them, so the tight bound only holds
    numerically up to k = 4; the theoretical bound still holds at higher k but
    would need extended precision (long double / Decimal) to observe it.
    """
    V = 4096
    T = 5_000
    # Use full-magnitude sin/cos so higher derivatives aren't negligible.
    idx = np.arange(T, dtype=np.float64)
    inp = np.stack((np.sin(idx), np.cos(idx)), axis=1)
    for k in range(1, 5):
        dod = get_domain_of_definition(
            inp=inp,
            order_of_derivative=k,
            use_decimal=False,
        )
        start, scale, enc = encode(
            inp=inp,
            vocab_size=V,
            domain_of_definition=dod,
            order_of_derivative=k,
            use_decimal=False,
        )
        dec = decode(
            start=start,
            scale=scale,
            inp=enc,
            vocab_size=V,
            order_of_derivative=k,
            use_decimal=False,
        )
        max_err = float(np.max(np.abs(dec - inp)))
        bin_half = float(np.max(dod)) / (V - 2)
        # Allow 2x slack for boundary transient / clipping edge effects.
        assert max_err <= 2 * bin_half, (
            f"k={k}: max_err={max_err:.4e} exceeds 2·Δ/2={2 * bin_half:.4e}"
        )


def test_k2_float64_sigma_delta_bound_is_uniform_in_T() -> None:
    """
    Check that the k=2 Σ-Δ reconstruction error stays bounded as T grows.

    The old 1st-order carry-based encoder produced O(T·Δ) drift at order=2;
    the new k-th-order EFM keeps the bound at a small constant times Δ.
    """
    V = 256
    errors = []
    for T in (2_000, 10_000, 50_000):
        idx = np.arange(T, dtype=np.float64) * 0.01
        inp = np.stack((np.sin(idx), np.cos(idx)), axis=1)
        dod = get_domain_of_definition(
            inp=inp,
            order_of_derivative=2,
            use_decimal=False,
        )
        start, scale, enc = encode(
            inp=inp,
            vocab_size=V,
            domain_of_definition=dod,
            order_of_derivative=2,
            use_decimal=False,
        )
        dec = decode(
            start=start,
            scale=scale,
            inp=enc,
            vocab_size=V,
            order_of_derivative=2,
            use_decimal=False,
        )
        errors.append(float(np.max(np.abs(dec - inp))))
    # Must not grow with T — allow a 2x slack for numerical boundary
    # transients but fail if the old O(T) drift is back.
    assert errors[2] < 3 * errors[0], (
        f"error grew with T: {errors} — noise-shaping is not working"
    )


def test_k2_float64_beats_k1_recon_when_signal_is_smooth() -> None:
    """
    Check that k=2 reconstructs a smooth signal better than k=1.

    For a smooth signal (small 2nd derivative relative to 1st), encoding the
    2nd derivative gives tighter bin width → smaller reconstruction error than
    k=1 at the same vocab size.
    """
    V = 256
    T = 5_000
    idx = np.arange(T, dtype=np.float64) * 0.001  # very slow → smooth
    inp = np.stack((np.sin(idx),), axis=1)

    def round_trip_err(k: int) -> float:
        dod = get_domain_of_definition(
            inp=inp,
            order_of_derivative=k,
            use_decimal=False,
        )
        start, scale, enc = encode(
            inp=inp,
            vocab_size=V,
            domain_of_definition=dod,
            order_of_derivative=k,
            use_decimal=False,
        )
        dec = decode(
            start=start,
            scale=scale,
            inp=enc,
            vocab_size=V,
            order_of_derivative=k,
            use_decimal=False,
        )
        return float(np.max(np.abs(dec - inp)))

    err_k1 = round_trip_err(1)
    err_k2 = round_trip_err(2)
    assert err_k2 < err_k1, (
        f"k=2 err ({err_k2}) should beat k=1 err ({err_k1}) on smooth signal"
    )


def test_per_column_order_of_derivative_round_trip() -> None:
    """
    Check the round trip with a derivative order per column.

    Mixed per-column k: column A with k=0 (raw), column B with k=1, column C
    with k=2. Each column is tokenized with its own derivative order;
    encode/decode must round-trip all three within the tight bound given by
    each column's own (D_c, V, k_c).
    """
    V = 256
    T = 5_000
    idx = np.arange(T, dtype=np.float64)
    inp = np.stack(
        (
            # k=0 target: roughly constant series with small drift.
            0.1 * idx,
            # k=1 target: sinusoid.
            np.sin(idx * 0.1),
            # k=2 target: quadratic — 2nd diff is constant.
            0.001 * idx**2,
        ),
        axis=1,
    )
    order = np.array([0, 1, 2], dtype=np.int64)
    dod = get_domain_of_definition(
        inp=inp,
        order_of_derivative=order,
        use_decimal=False,
    )
    start, scale, enc = encode(
        inp=inp,
        vocab_size=V,
        domain_of_definition=dod,
        order_of_derivative=order,
        use_decimal=False,
    )
    assert np.all((enc >= 0) & (enc < V))
    dec = decode(
        start=start,
        scale=scale,
        inp=enc,
        vocab_size=V,
        order_of_derivative=order,
        use_decimal=False,
    )
    # Each column's reconstruction should be bounded by its own Δ/2.
    for c in range(3):
        err_c = float(np.max(np.abs(dec[:, c] - inp[:, c])))
        k_c = int(order[c])
        # Accommodate per-column scale (MASH shrinks bin width for k≥2).
        headroom = 2**k_c if k_c >= MIN_HEADROOM_ORDER and 2**k_c + 2 < V else 0
        bin_half = float(dod[c]) / (V - 2 - headroom)
        assert err_c <= 2 * bin_half, (
            f"col {c} (k={k_c}): err={err_c:.4e} > 2·Δ/2={2 * bin_half:.4e}"
        )


def test_per_column_order_decimal_round_trip() -> None:
    """Decimal path must also support per-column k without precision loss."""
    V = 256
    T = 500
    idx = np.arange(T, dtype=np.float64)
    inp = np.stack(
        (np.sin(idx * 0.1), np.cos(idx * 0.05), 0.001 * idx**2),
        axis=1,
    )
    inp_dec = np_to_decimal(inp)
    order = np.array([0, 1, 2], dtype=np.int64)
    dod = get_domain_of_definition(
        inp=inp_dec,
        order_of_derivative=order,
        use_decimal=True,
    )
    start, scale, enc = encode(
        inp=inp_dec,
        vocab_size=V,
        domain_of_definition=dod,
        order_of_derivative=order,
        use_decimal=True,
    )
    assert np.all((enc >= 0) & (enc < V))
    dec = decode(
        start=start,
        scale=scale,
        inp=enc,
        vocab_size=V,
        order_of_derivative=order,
        use_decimal=True,
    )
    # Per-column Δ/2 (Decimal is exact; no float cumsum floor).
    for c in range(3):
        k_c = int(order[c])
        headroom = 2**k_c if k_c >= MIN_HEADROOM_ORDER and 2**k_c + 2 < V else 0
        bin_half = float(dod[c]) / (V - 2 - headroom)
        col_err = float(
            np.max(np.abs(np.array([float(v) for v in dec[:, c]]) - inp[:, c])),
        )
        assert col_err <= 2 * bin_half, (
            f"col {c} (k={k_c}): Decimal err {col_err} > 2·Δ/2={2 * bin_half}"
        )


def test_per_column_order_scalar_fallback_equivalence() -> None:
    """
    Check that a uniform per-column order matches the scalar order.

    Passing order as a length-F array of identical values must produce the
    same reconstruction as passing the scalar, to within float64 precision —
    confirms the array path correctly handles the uniform-k case.
    """
    V = 256
    T = 2_000
    idx = np.arange(T, dtype=np.float64)
    inp = np.stack((np.sin(idx * 0.1), np.cos(idx * 0.1)), axis=1)
    k_scalar = 1
    k_array = np.array([k_scalar, k_scalar], dtype=np.int64)
    # Scalar path
    dod_s = get_domain_of_definition(
        inp=inp,
        order_of_derivative=k_scalar,
        use_decimal=False,
    )
    _, _, enc_s = encode(
        inp=inp,
        vocab_size=V,
        domain_of_definition=dod_s,
        order_of_derivative=k_scalar,
        use_decimal=False,
    )
    # Array path with uniform values
    dod_a = get_domain_of_definition(
        inp=inp,
        order_of_derivative=k_array,
        use_decimal=False,
    )
    _, _, enc_a = encode(
        inp=inp,
        vocab_size=V,
        domain_of_definition=dod_a,
        order_of_derivative=k_array,
        use_decimal=False,
    )
    # Tokens must be identical (or very nearly — same arithmetic).
    assert np.array_equal(enc_s, enc_a), (
        "uniform-order array path must match scalar path"
    )


@pytest.mark.slow
def test_diff_encoder_decoder_square() -> None:
    """Check that order 2 restores the square series exactly with 4 tokens."""
    vocab_size = 4
    idx = np.arange(1_000_000, dtype=np.float64)
    inp = np.stack(
        (np.square(idx),),
        axis=1,
    )
    inp = np_to_decimal(inp)
    order_of_derivative = 2
    use_decimal = True
    domain_of_definition = get_domain_of_definition(
        inp=inp,
        order_of_derivative=order_of_derivative,
        use_decimal=use_decimal,
    )
    start, scale, encoded_inp = encode(
        inp=inp,
        vocab_size=vocab_size,
        domain_of_definition=domain_of_definition,
        order_of_derivative=order_of_derivative,
        use_decimal=use_decimal,
    )
    assert np.all((encoded_inp >= 0) & (encoded_inp < vocab_size))
    decoded_inp = decode(
        start=start,
        scale=scale,
        inp=encoded_inp,
        vocab_size=vocab_size,
        order_of_derivative=order_of_derivative,
        use_decimal=use_decimal,
    )
    assert float(np.max(np.abs(decoded_inp - inp))) == 0
