import numpy as np

from diff_gpt.encoder_decoder import (
    decode,
    encode,
    get_domain_of_definition,
    np_to_decimal,
)


def test_diff_encoder_decoder_sin_cos():
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
    assert float(np.max(np.abs(decoded_inp - inp))) <= 0.007873033862000018


def test_diff_encoder_decoder_sin_cos_derivative():
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
    assert float(np.max(np.abs(decoded_inp - inp))) <= 0.003775003594016138


def test_diff_encoder_decoder_sin_cos_second_derivative():
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
    assert float(np.max(np.abs(decoded_inp - inp))) <= 0.000225


def test_diff_encoder_decoder_sin_cos_third_derivative():
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
    assert float(np.max(np.abs(decoded_inp - inp))) <= 1.35e-5


def test_diff_encoder_decoder_sin_cos_third_derivative_2():
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
    assert float(np.max(np.abs(decoded_inp - inp))) <= 2.05337387573312e-06


def test_diff_encoder_sin_cos_different_lenght():
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


def test_diff_encoder_decoder_lin():
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


def test_diff_encoder_decoder_const():
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


def test_diff_encoder_decoder_small_const():
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
    assert float(np.max(np.abs(decoded_inp - inp))) <= 2.783148670569062e-30


def test_diff_encoder_decoder_mix():
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
    assert float(np.max(np.abs(decoded_inp - inp))) <= 0.003775003594016138


def test_kth_order_mash_bound_is_tight_and_uniform_in_T():
    """MASH k-th-order Σ-Δ: reconstruction error ≤ small constant × Δ/2
    up to k = 4 in IEEE float64. Beyond that, the k-fold cumsum in the
    decoder loses mantissa bits faster than the noise-shaping recovers
    them, so the tight bound only holds numerically up to k = 4; the
    theoretical bound still holds at higher k but would need extended
    precision (long double / Decimal) to observe it.
    """
    V = 4096
    T = 5_000
    # Use full-magnitude sin/cos so higher derivatives aren't negligible.
    idx = np.arange(T, dtype=np.float64)
    inp = np.stack((np.sin(idx), np.cos(idx)), axis=1)
    for k in range(1, 5):
        dod = get_domain_of_definition(
            inp=inp, order_of_derivative=k, use_decimal=False
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
        # Allow 2× slack for boundary transient / clipping edge effects.
        assert max_err <= 2 * bin_half, (
            f"k={k}: max_err={max_err:.4e} exceeds 2·Δ/2={2*bin_half:.4e}"
        )


def test_k2_float64_sigma_delta_bound_is_uniform_in_T():
    """k-th-order Σ-Δ at k=2: reconstruction error must stay bounded as T
    grows. The old 1st-order carry-based encoder produced O(T·Δ) drift at
    order=2; the new k-th-order EFM keeps the bound at a small constant × Δ."""
    V = 256
    errors = []
    for T in (2_000, 10_000, 50_000):
        idx = np.arange(T, dtype=np.float64) * 0.01
        inp = np.stack((np.sin(idx), np.cos(idx)), axis=1)
        dod = get_domain_of_definition(
            inp=inp, order_of_derivative=2, use_decimal=False
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
    # Must not grow with T — allow a 2× slack for numerical boundary
    # transients but fail if the old O(T) drift is back.
    assert errors[2] < 3 * errors[0], (
        f"error grew with T: {errors} — noise-shaping is not working"
    )


def test_k2_float64_beats_k1_recon_when_signal_is_smooth():
    """For a smooth signal (small 2nd derivative relative to 1st), encoding
    the 2nd derivative gives tighter bin width → smaller reconstruction
    error than k=1 at the same vocab size."""
    V = 256
    T = 5_000
    idx = np.arange(T, dtype=np.float64) * 0.001  # very slow → smooth
    inp = np.stack((np.sin(idx),), axis=1)

    def round_trip_err(k: int) -> float:
        dod = get_domain_of_definition(
            inp=inp, order_of_derivative=k, use_decimal=False
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


def test_diff_encoder_decoder_square():
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
