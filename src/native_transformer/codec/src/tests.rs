use super::*;

fn dyadic(x: f64) -> (BigInt, i32) {
    assert!(x.is_finite());
    let bits = x.to_bits();
    let e = ((bits >> 52) & 2047) as i32;
    let m = (bits & ((1_u64 << 52) - 1)) | if e == 0 { 0 } else { 1_u64 << 52 };
    let value = BigInt::from(m);
    (
        if bits >> 63 == 0 { value } else { -value },
        if e == 0 { -1074 } else { e - 1075 },
    )
}

// Independent integer bit decomposition; no Rational arithmetic or codec code.
fn exact_level(x: f64, scale: f64, fraction: f64) -> i64 {
    let (x, ex) = dyadic(x);
    let (s, es) = dyadic(scale);
    let (f, ef) = dyadic(fraction);
    let e = (ex + es).min(ef).min(-1);
    let n = ((x * s) << (ex + es - e) as usize) - (f << (ef - e) as usize)
        + (BigInt::from(1) << (-1 - e) as usize);
    (n >> (-e) as usize).to_i64().unwrap()
}

#[test]
fn level_ties_and_tiny_products_match_independent_integer_bits() {
    let mut checked = 0;
    let mut rounded_first_misses = 0;
    for scale in [0.1, 1.1, std::f64::consts::PI, 1e-100, 1e100] {
        for i in -50..50 {
            let f = f64::from((i + 50) % 8) / 8.;
            let tie = (f64::from(i) + f - 0.5) / scale;
            for x in [
                tie.next_down(),
                tie,
                tie.next_up(),
                f64::from(i) * 1e-70 / scale,
            ] {
                let expected = exact_level(x, scale, f);
                assert_eq!(level(x, scale, f, 0).unwrap(), expected);
                rounded_first_misses +=
                    usize::from((scale * x - f + 0.5).floor() as i64 != expected);
                checked += 1;
            }
        }
    }
    assert_eq!(checked, 2000);
    assert!(
        rounded_first_misses > 20,
        "exercise the reference's rounded-product bug"
    );
    let smallest = f64::from_bits(1);
    assert_eq!(level(-smallest, 0.1, 0.5, 0).unwrap(), -1);
    assert_eq!(level(smallest, 0.1, 0.5, 0).unwrap(), 0);
    assert_eq!(level(0., 0.1, 0.5, 0).unwrap(), 0);
    assert!(level(2f64.powi(53), 1., 0., 0).is_err());
    assert!(level(2f64.powi(52), 1., 0., 11).is_err());
    // Reject out-of-range scaled values before converting levels to int64.
    assert!(level(50e-70, 1e100, 0., 0).is_err());
}

#[test]
fn selected_binary64_scale_is_exactly_below_the_endpoint() {
    for q in [
        Rational::new(1.into(), 10.into()),
        Rational::new(31.into(), 3.into()),
        rational(f64::MAX).unwrap() * integer(2),
        Rational::new(1.into(), BigInt::from(1) << 1100_usize),
    ] {
        let f = below(&q).unwrap();
        assert!(rational(f).unwrap() <= q);
        if f < f64::MAX {
            assert!(rational(f.next_up()).unwrap() > q);
        }
    }
}

#[test]
fn modular_anchor_matches_unbounded_coefficient_recurrence() {
    for lags in [
        vec![1],
        vec![1, 1],
        vec![1, 1, 1, 1, 1],
        vec![1, 4],
        vec![1, 2, 3],
    ] {
        let op = Difference::new(lags).unwrap();
        let p = op.degree();
        let bits = 53.min(64 - op.lags.len());
        let unit = BigInt::from(1) << bits;
        let unit_float = (1_u64 << bits) as f64;
        let prefix: Vec<f64> = (0..p).map(|i| [0.2, -2.75, 1024.5][i % 3]).collect();
        for tokenizer in [Tokenizer::Mash, Tokenizer::MashHalf] {
            let (fraction, steps) = codec::anchor(&prefix, 256, &op, tokenizer).unwrap();
            let mut anchor: Vec<BigInt> = prefix
                .iter()
                .map(|x| BigInt::from(((x - x.floor()) * unit_float).round_ties_even() as u64))
                .collect();
            let step = if tokenizer == Tokenizer::MashHalf {
                &unit / 2
            } else {
                BigInt::zero()
            };
            for t in p..p + 256 {
                let value = &step
                    - op.coefficients
                        .iter()
                        .enumerate()
                        .skip(1)
                        .map(|(j, c)| c * &anchor[t - j])
                        .sum::<BigInt>();
                anchor.push(value);
            }
            let numerators: Vec<BigInt> = anchor
                .iter()
                .map(|a| ((a % &unit) + &unit) % &unit)
                .collect();
            for (i, n) in numerators.iter().enumerate() {
                assert_eq!(fraction[i], n.to_f64().unwrap() / unit_float);
            }
            for t in p..p + 256 {
                let difference = op
                    .coefficients
                    .iter()
                    .enumerate()
                    .map(|(j, c)| c * &numerators[t - j])
                    .sum::<BigInt>();
                assert_eq!(steps[t - p], (difference >> bits).to_i64().unwrap());
            }
        }
    }
}

fn roundtrip(
    input: &[f64],
    vocab: i64,
    domain: Domain,
    op: Difference,
    tokenizer: Tokenizer,
) -> Encoded {
    let matrix = Matrix::new(1, input.to_vec()).unwrap();
    let operators = [op];
    let encoded = encode(&matrix, vocab, &[domain], &operators, tokenizer).unwrap();
    assert_eq!(encoded.clipped_tokens, 0);
    let decoded = decode(
        &encoded.prefix,
        &encoded.grid,
        &encoded.tokens,
        vocab,
        &operators,
        tokenizer,
    )
    .unwrap();
    for (&x, &y) in input.iter().zip(decoded.values()) {
        let bound = 0.5 / encoded.grid.scale[0] + 1e-12 * (1. + x.abs());
        assert!(
            (x - y).abs() <= bound,
            "{tokenizer:?}: {x} vs {y}, bound {bound}"
        );
    }
    encoded
}

#[test]
fn reference_rounding_tie_regressions_remain_in_range() {
    for tokenizer in [Tokenizer::Lgt, Tokenizer::Mash] {
        let encoded = roundtrip(
            &[0.030147312353493265, 1.8992806782700757],
            64,
            Domain::symmetric(1.8691333659165825),
            Difference::order(1).unwrap(),
            tokenizer,
        );
        assert!(encoded.tokens.values().iter().all(|q| *q > 0 && *q < 64));
        let values = [0.06800385238021311, 1.4734168015712845];
        let exact = rational(values[1]).unwrap() - rational(values[0]).unwrap();
        let lower = below(&exact).unwrap();
        let bound = if rational(lower).unwrap() < exact {
            lower.next_up()
        } else {
            lower
        };
        let encoded = roundtrip(
            &values,
            64,
            Domain::symmetric(bound),
            Difference::order(1).unwrap(),
            tokenizer,
        );
        assert!(rational(encoded.grid.scale[0]).unwrap() * rational(bound).unwrap() <= integer(31));
    }
}

#[test]
fn constant_mash_is_exact_and_half_step_has_the_reference_pattern() {
    let constant = vec![17.125; 32];
    for order in 1..4 {
        let op = Difference::order(order).unwrap();
        let encoded = roundtrip(
            &constant,
            64,
            Domain::symmetric(1.),
            op.clone(),
            Tokenizer::Mash,
        );
        assert!(encoded.tokens.values().iter().all(|q| *q == 32));
        let decoded = decode(
            &encoded.prefix,
            &encoded.grid,
            &encoded.tokens,
            64,
            &[op],
            Tokenizer::Mash,
        )
        .unwrap();
        assert_eq!(decoded.values(), constant);
    }
    let encoded = roundtrip(
        &constant,
        64,
        Domain::symmetric(1.),
        Difference::order(1).unwrap(),
        Tokenizer::MashHalf,
    );
    assert_eq!(&encoded.tokens.values()[..6], &[32, 31, 32, 31, 32, 31]);
}

#[test]
fn grid_uses_coefficient_headroom_and_handles_asymmetric_ranges() {
    let op = Difference::new(vec![1, 2, 3]).unwrap();
    assert_eq!(op.headroom(), BigInt::from(6));
    let grid = level_grid(64, &[Domain::symmetric(1.)], &[op], Tokenizer::Mash).unwrap();
    assert_eq!(grid.scale[0], 29.);
    for domain in [
        Domain { low: -1., high: 2. },
        Domain { low: 1., high: 5. },
        Domain {
            low: -5.,
            high: -1.,
        },
        Domain { low: 0., high: 1. },
    ] {
        for tokenizer in [Tokenizer::Lgt, Tokenizer::Mash, Tokenizer::MashHalf] {
            let values: Vec<_> = (0..81)
                .map(|i| domain.low + (domain.high - domain.low) * f64::from(i) / 80.)
                .collect();
            roundtrip(
                &values,
                64,
                domain,
                Difference::order(0).unwrap(),
                tokenizer,
            );
        }
    }
    let grid = level_grid(
        64,
        &[Domain::symmetric(0.)],
        &[Difference::order(1).unwrap()],
        Tokenizer::Mash,
    )
    .unwrap();
    assert_eq!(grid.scale[0], 2f64.powi(20));
    assert_eq!(grid.zero[0], 32);
}

#[test]
fn mixed_seasonal_columns_match_individual_encoders_and_decoders() {
    let ops = [
        Difference::order(1).unwrap(),
        Difference::order(2).unwrap(),
        Difference::new(vec![1, 4]).unwrap(),
    ];
    let domains = [
        Domain::symmetric(10.),
        Domain::symmetric(10.),
        Domain::symmetric(20.),
    ];
    let mut values = Vec::new();
    for i in 0..128 {
        let t = f64::from(i);
        values.extend([
            0.031 * t + (t * 0.3).sin(),
            -0.011 * t + (t * 0.7).cos(),
            0.007 * t + (t * 0.5).sin(),
        ]);
    }
    let input = Matrix::new(3, values).unwrap();
    for tokenizer in [Tokenizer::Lgt, Tokenizer::Mash, Tokenizer::MashHalf] {
        let encoded = encode(&input, 128, &domains, &ops, tokenizer).unwrap();
        assert_eq!(encoded.clipped_tokens, 0);
        let decoded = decode(
            &encoded.prefix,
            &encoded.grid,
            &encoded.tokens,
            128,
            &ops,
            tokenizer,
        )
        .unwrap();
        for c in 0..3 {
            let start = 5 - ops[c].degree();
            let column: Vec<_> = (start..128).map(|r| input.values[r * 3 + c]).collect();
            let separate = roundtrip(&column, 128, domains[c], ops[c].clone(), tokenizer);
            assert_eq!(
                separate.grid.scale[0].to_bits(),
                encoded.grid.scale[c].to_bits()
            );
            assert_eq!(
                separate.tokens.values(),
                (0..123)
                    .map(|r| encoded.tokens.values[r * 3 + c])
                    .collect::<Vec<_>>()
            );
            let own = decode(
                &separate.prefix,
                &separate.grid,
                &separate.tokens,
                128,
                &[ops[c].clone()],
                tokenizer,
            )
            .unwrap();
            for r in 5..128 {
                assert_eq!(
                    decoded.values[r * 3 + c].to_bits(),
                    own.values[r - start].to_bits()
                );
            }
        }
    }
}

#[test]
fn appending_future_rows_cannot_change_existing_tokens() {
    let prefix: Vec<_> = (0..32).map(|i| (f64::from(i) * 0.4).sin()).collect();
    let mut extended = prefix.clone();
    extended.extend([100., -300., 40.]);
    let op = [Difference::order(2).unwrap()];
    let domain = [Domain::symmetric(2.)];
    let prefix = Matrix::new(1, prefix).unwrap();
    let extended = Matrix::new(1, extended).unwrap();
    for tokenizer in [Tokenizer::Mash, Tokenizer::Lgt, Tokenizer::MashHalf] {
        let before = encode(&prefix, 64, &domain, &op, tokenizer).unwrap();
        let after = encode(&extended, 64, &domain, &op, tokenizer).unwrap();
        assert_eq!(
            before.tokens.values(),
            &after.tokens.values()[..before.tokens.rows()]
        );
        assert_eq!(before.clipped_tokens, 0);
        assert!(after.clipped_tokens > 0);
    }
}

#[test]
fn forecast_boundary_uses_each_encoders_grid_from_the_raw_tail() {
    let op = [Difference::order(2).unwrap()];
    let domains = [Domain::symmetric(4.)];
    let whole = [0.11, 0.31, 0.74, 1.49, 1.12, 1.98, 3.17];
    for tokenizer in [Tokenizer::Mash, Tokenizer::Lgt, Tokenizer::MashHalf] {
        let full = encode(
            &Matrix::new(1, whole.to_vec()).unwrap(),
            64,
            &domains,
            &op,
            tokenizer,
        )
        .unwrap();
        let future = encode(
            &Matrix::new(1, whole[2..].to_vec()).unwrap(),
            64,
            &domains,
            &op,
            tokenizer,
        )
        .unwrap();
        let decoded = decode(
            &future.prefix,
            &future.grid,
            &future.tokens,
            64,
            &op,
            tokenizer,
        )
        .unwrap();
        assert_eq!(future.clipped_tokens, 0);
        for (x, y) in whole[2..].iter().zip(decoded.values()) {
            assert!((x - y).abs() <= 0.5 / future.grid.scale[0] + 1e-12);
        }
        if tokenizer == Tokenizer::Lgt {
            assert_eq!(future.tokens.values(), &full.tokens.values()[2..]);
        } else {
            assert_ne!(
                future.tokens.values(),
                &full.tokens.values()[2..],
                "prefix-dependent {tokenizer:?}"
            );
        }
    }
}

#[test]
fn invalid_inputs_and_inverse_overflow_are_errors() {
    assert!(codec::check_inverse(&[i64::MAX], &[1], &[1]).is_err());
    assert!(codec::check_inverse(&[i64::MIN], &[-1], &[1]).is_err());
    assert_eq!(
        codec::check_inverse(&[3, 5], &[2, 2, 2], &[1, 1]).unwrap(),
        vec![3, 5, 9, 15, 23]
    );
    let ops = [Difference::order(1).unwrap()];
    for vocab in [0, 3, 63, 1_i64 << 62] {
        assert!(level_grid(vocab, &[Domain::symmetric(1.)], &ops, Tokenizer::Mash).is_err());
    }
    for values in [
        vec![0., f64::NAN],
        vec![0., f64::INFINITY],
        vec![2f64.powi(54); 2],
    ] {
        assert!(
            encode(
                &Matrix::new(1, values).unwrap(),
                64,
                &[Domain::symmetric(1.)],
                &ops,
                Tokenizer::Mash
            )
            .is_err()
        );
    }
    assert!(level_grid(64, &[Domain { low: 2., high: 1. }], &ops, Tokenizer::Mash).is_err());
    assert!(Difference::new(vec![0]).is_err());
    assert!(Difference::order(62).is_err());
}

#[test]
fn every_fibonacci97_window_matches_all_three_encoder_formulas() {
    // The actual DiffGPT test's codec geometry: four raw values, three step
    // tokens, V512, D96, order1. This is codec evidence, not training.
    let states: Vec<(i64, i64)> = (0..97)
        .flat_map(|a| (0..97).map(move |b| (a, b)))
        .filter(|&(a, b)| (a, b) != (0, 0))
        .collect();
    assert_eq!(states.len(), 9408);
    let windows: Vec<[i64; 4]> = states
        .iter()
        .map(|&(a, b)| [a, b, (a + b) % 97, (a + 2 * b) % 97])
        .collect();
    let values: Vec<f64> = (0..4)
        .flat_map(|r| windows.iter().map(move |w| w[r] as f64))
        .collect();
    let ops = vec![Difference::order(1).unwrap(); states.len()];
    let matrix = Matrix::new(states.len(), values).unwrap();
    for tokenizer in [Tokenizer::Mash, Tokenizer::Lgt, Tokenizer::MashHalf] {
        let encoded = encode(
            &matrix,
            512,
            &vec![Domain::symmetric(96.); states.len()],
            &ops,
            tokenizer,
        )
        .unwrap();
        assert_eq!(encoded.clipped_tokens, 0);
        let mut scale: f64 = if tokenizer == Tokenizer::MashHalf {
            511. / 192.
        } else {
            85. / 32.
        };
        if tokenizer == Tokenizer::MashHalf
            && rational(scale).unwrap() * integer(192) > integer(511)
        {
            scale = scale.next_down();
        }
        assert!(
            encoded
                .grid
                .scale
                .iter()
                .all(|s| s.to_bits() == scale.to_bits())
        );
        assert!(encoded.grid.zero.iter().all(|z| *z == 256));
        let decoded = decode(
            &encoded.prefix,
            &encoded.grid,
            &encoded.tokens,
            512,
            &ops,
            tokenizer,
        )
        .unwrap();
        for (c, w) in windows.iter().enumerate() {
            let levels: Vec<i64> = w
                .iter()
                .enumerate()
                .map(|(i, x)| match tokenizer {
                    Tokenizer::Mash => (85 * (x - w[0]) + 16).div_euclid(32),
                    Tokenizer::Lgt => (85 * x + 16).div_euclid(32),
                    Tokenizer::MashHalf => {
                        // Full, unbounded anchor A_i=fl(s*a)+i/2, independently of
                        // the native modular-anchor and inverse-difference code.
                        let a = rational(scale * w[0] as f64).unwrap() + integer(i as i64) * half();
                        let whole = floor(&a);
                        let fraction = (a - Rational::from_integer(whole.clone()))
                            .to_f64()
                            .unwrap();
                        exact_level(*x as f64, scale, fraction) - whole.to_i64().unwrap()
                    }
                })
                .collect();
            for r in 0..3 {
                assert_eq!(
                    encoded.tokens.values[r * states.len() + c],
                    levels[r + 1] - levels[r] + 256,
                    "{tokenizer:?}, {w:?}, row{r}"
                );
            }
            for (r, &raw) in w.iter().enumerate() {
                let prediction = decoded.values[r * states.len() + c];
                assert_eq!(prediction.round() as i64, raw);
                assert!((prediction - raw as f64).abs() <= 0.5 / scale + 1e-12);
            }
        }
    }
}

#[test]
fn asymmetric_positive_negative_steps_reconstruct_across_operators() {
    for lags in [vec![], vec![1], vec![1, 1], vec![1, 4]] {
        let op = Difference::new(lags).unwrap();
        for (low, high) in [(-1, 2), (1, 5), (-5, -1), (0, 4)] {
            let mut raw: Vec<i64> = (0..op.degree()).map(|i| (i as i64) % 3 - 1).collect();
            for t in op.degree()..op.degree() + 48 {
                let step = low + (t as i64) % (high - low + 1);
                let prior: BigInt = op
                    .coefficients
                    .iter()
                    .enumerate()
                    .skip(1)
                    .map(|(j, c)| c * raw[t - j])
                    .sum();
                raw.push((BigInt::from(step) - prior).to_i64().unwrap());
            }
            for tokenizer in [Tokenizer::Lgt, Tokenizer::Mash, Tokenizer::MashHalf] {
                roundtrip(
                    &raw.iter().map(|v| *v as f64).collect::<Vec<_>>(),
                    128,
                    Domain {
                        low: low as f64,
                        high: high as f64,
                    },
                    op.clone(),
                    tokenizer,
                );
            }
        }
    }
}
