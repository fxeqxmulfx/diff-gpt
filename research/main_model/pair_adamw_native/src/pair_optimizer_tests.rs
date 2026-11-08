//! CPU geometry/update oracle written independently of the optimizer owner map.
use crate::{Config, Engine, Result, layout, optimizer::Optimizer, schedule::Schedule};
use serde_json::{Value, json};

type Pair = (usize, usize, Option<(usize, usize)>);
fn config(w: usize, h: usize, l: usize, phase: bool) -> Config {
    Config {
        width: w,
        heads: h,
        layers: l,
        vocab: 11,
        max_batch: 2,
        max_length: 4,
        normalized: true,
        ssmax: true,
        gain: h as f32 / w as f32,
        lr: 0.001,
        decay: 0.1,
        warmup: 2000,
        phase_optimizer: phase,
        architecture: "complex-mhc-lerope".into(),
    }
}
fn geometry(c: &Config) -> (Vec<Pair>, usize, usize, usize) {
    let (w, h, l) = (c.width, c.heads, c.layers);
    let d = w / h;
    let hidden = ((8 * w + 1) / 3).div_ceil(2) * 2;
    let base = c.vocab * w + l * (4 * w * w + 3 * w * hidden);
    let frequencies = base + 2 * l * (96 * w + 27);
    let n = frequencies + d / 2;
    let mut pairs = Vec::new();
    for token in 0..c.vocab {
        for head in 0..h {
            for i in 0..d / 2 {
                let re = token * w + head * d + i;
                pairs.push((re, re + d / 2, None));
            }
        }
    }
    for layer in 0..l {
        let offset = c.vocab * w + layer * (4 * w * w + 3 * w * hidden);
        for (at, rows, cols, rgroups, cgroups) in [
            (offset, 3 * w, w, 3 * h, h),
            (offset + 3 * w * w, w, w, h, h),
            (offset + 4 * w * w, 2 * hidden, w, 2, h),
            (offset + 4 * w * w + 2 * hidden * w, w, hidden, h, 1),
        ] {
            let rh = rows / rgroups / 2;
            let ch = cols / cgroups / 2;
            for rg in 0..rgroups {
                for r in 0..rh {
                    for cg in 0..cgroups {
                        for col in 0..ch {
                            let re = at + (rg * 2 * rh + r) * cols + cg * 2 * ch + col;
                            let im = re + rh * cols;
                            pairs.push((re, im, Some((re + ch, im + ch))));
                        }
                    }
                }
            }
        }
    }
    (pairs, base, frequencies, n)
}
fn set_pair(x: &mut [f32], pair: Pair, re: f32, im: f32, variance: bool) {
    let (a, b, duplicate) = pair;
    x[a] = re;
    x[b] = im;
    if let Some((negative, real)) = duplicate {
        x[negative] = if variance { im } else { -im };
        x[real] = re;
    }
}
fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}
fn clipped(c: &Config, gradient: &[f32]) -> Vec<f64> {
    let (pairs, base, frequencies, n) = geometry(c);
    let mut g: Vec<_> = gradient.iter().map(|&x| f64::from(x)).collect();
    let mut norm = 0.;
    for &(a, b, duplicate) in &pairs {
        if let Some((negative, real)) = duplicate {
            g[a] += g[real];
            g[b] -= g[negative];
        }
        norm += g[a] * g[a] + g[b] * g[b];
    }
    norm += g[base..frequencies].iter().map(|x| x * x).sum::<f64>();
    let norm = norm.sqrt().max(1.);
    for pair in pairs {
        let (a, b, duplicate) = pair;
        g[a] /= norm;
        g[b] /= norm;
        if let Some((negative, real)) = duplicate {
            g[negative] = -g[b];
            g[real] = g[a];
        }
    }
    for x in &mut g[base..frequencies] {
        *x /= norm;
    }
    let frequency_norm = g[frequencies..n]
        .iter()
        .map(|x| x * x)
        .sum::<f64>()
        .sqrt()
        .max(1.);
    for x in &mut g[frequencies..n] {
        *x /= frequency_norm;
    }
    g
}
fn oracle(c: &Config, old: &[f32], gradient: &[f32], step: usize) -> (Vec<f64>, Vec<f64>) {
    let (pairs, base, frequencies, n) = geometry(c);
    let g = clipped(c, gradient);
    let mut next: Vec<_> = old.iter().map(|&x| f64::from(x)).collect();
    let lr = f64::from(Schedule::new(100000).unwrap().rate(step));
    let bc1 = 1. - 0.9f64.powf((step + 1) as f64);
    let bc2 = 1. - 0.95f64.powf((step + 1) as f64);
    let mut write = |index: usize, moment: f64, variance: f64, decay: f64| {
        next[index] = (1. - lr * decay) * f64::from(old[index])
            - lr * (moment / bc1) / ((variance / bc2).sqrt() + 1e-8);
        next[n + index] = moment;
        next[2 * n + index] = variance;
    };
    for (a, b, duplicate) in pairs {
        let ma = 0.9 * f64::from(old[n + a]) + 0.1 * g[a];
        let mb = 0.9 * f64::from(old[n + b]) + 0.1 * g[b];
        let v = 0.95 * f64::from(old[2 * n + a]) + 0.05 * (g[a] * g[a] + g[b] * g[b]) / 2.;
        write(a, ma, v, 0.1);
        write(b, mb, v, 0.1);
        if let Some((negative, real)) = duplicate {
            write(negative, -mb, v, 0.1);
            write(real, ma, v, 0.1);
        }
    }
    for i in base..n {
        let m = 0.9 * f64::from(old[n + i]) + 0.1 * g[i];
        let v = 0.95 * f64::from(old[2 * n + i]) + 0.05 * g[i] * g[i];
        write(i, m, v, if i < frequencies { 0.1 } else { 0. });
    }
    (next, g)
}
fn record(name: &str, value: &Value) -> Result<()> {
    if let Some(root) = std::env::var_os("PAIR_VALIDATION_OUTPUT") {
        let path = std::path::Path::new(&root).join(format!("{name}.json"));
        std::fs::create_dir_all(&root)?;
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?;
        file.write_all(&serde_json::to_vec_pretty(value)?)?;
    }
    eprintln!("{}: {}", name, value["checks"]);
    Ok(())
}

#[test]
fn independent_geometry_covers_exact_complex_and_real_owners() {
    for (w, h, l) in [(64, 4, 2), (128, 4, 1), (128, 1, 6)] {
        let c = config(w, h, l, true);
        let (pairs, base, frequencies, n) = geometry(&c);
        let owners = Optimizer::owners(&c).unwrap();
        assert_eq!(owners.len(), pairs.len() + (frequencies - base) / 2);
        let declared: std::collections::BTreeSet<_> = pairs.iter().copied().collect();
        let actual: std::collections::BTreeSet<_> = owners[..pairs.len()]
            .iter()
            .map(|&[a, b, neg, re, k]| {
                (
                    a as usize,
                    b as usize,
                    if k == 0 {
                        Some((neg as usize, re as usize))
                    } else {
                        None
                    },
                )
            })
            .collect();
        assert_eq!(declared, actual);
        let mut visits = vec![0u32; n];
        for (a, b, d) in pairs {
            visits[a] += 1;
            visits[b] += 1;
            if let Some((negative, real)) = d {
                visits[negative] += 1;
                visits[real] += 1;
            }
        }
        for &[a, b, _, _, kind] in &owners[actual.len()..] {
            assert_eq!(kind, 1);
            visits[a as usize] += 1;
            visits[b as usize] += 1;
        }
        for [a, b, _, _, kind] in Optimizer::frequency_owners(&c) {
            assert_eq!(kind, 1);
            assert!(a as usize >= frequencies);
            visits[a as usize] += 1;
            visits[b as usize] += 1;
        }
        assert!(visits.iter().all(|&x| x == 1));
        assert_eq!(layout::initialize_model(&c, 1).len(), n);
    }
}

#[test]
#[ignore = "requires CUDA"]
fn pair_updates_match_independent_fp64_and_preserve_real_groups() -> Result<()> {
    let mut checks = 0;
    let mut misses = Vec::new();
    let mut worst_ratio = 0f64;
    for (w, h, l) in [(64, 4, 2), (128, 4, 1), (128, 1, 1)] {
        let c = config(w, h, l, true);
        let (pairs, base, frequencies, n) = geometry(&c);
        let initial = layout::initialize_model(&c, 19);
        let mut e = Engine::new(c.clone(), &initial)?;
        let mut opt = Optimizer::new(&e, 100000)?;
        let mut control = Engine::new(config(w, h, l, false), &initial)?;
        let mut control_opt = Optimizer::new(&control, 100000)?;
        for scale in [0., 1e-8, 0.003, 1., 1e8] {
            for state in 0..3 {
                let gradient: Vec<f32> = (0..n)
                    .map(|i| (((i * 7) % 31) as f32 - 15.) * scale / 17.)
                    .collect();
                let g = clipped(&c, &gradient);
                let mut m = vec![0f32; n];
                let mut v = vec![0f32; n];
                if state != 0 {
                    for (j, &pair) in pairs.iter().enumerate() {
                        let (a, b, _) = pair;
                        let ma = if state == 2 {
                            (-g[a] / 9.) as f32
                        } else {
                            ((j % 5) as f32 - 2.) * 0.0003
                        };
                        let mb = if state == 2 {
                            (-g[b] / 9.) as f32
                        } else {
                            ((j % 7) as f32 - 3.) * 0.0002
                        };
                        set_pair(&mut m, pair, ma, mb, false);
                        let vv = 0.00002 + (j % 3) as f32 * 0.00001;
                        set_pair(&mut v, pair, vv, vv, true);
                    }
                    for i in base..n {
                        m[i] = ((i % 5) as f32 - 2.) * 0.0002;
                        v[i] = 0.00002 + (i % 3) as f32 * 0.00001;
                    }
                }
                let mut old = initial.clone();
                old.extend(&m);
                old.extend(&v);
                for step in [0, 1, 1999, 2000, 79999, 80000, 90000, 99999] {
                    opt.restore(&mut e, &old)?;
                    control_opt.restore(&mut control, &old)?;
                    e.gradients.write(&gradient)?;
                    control.gradients.write(&gradient)?;
                    opt.step(&e, step)?;
                    control_opt.step(&control, step)?;
                    let actual = opt.save(&e)?;
                    let baseline = control_opt.save(&control)?;
                    for part in 0..3 {
                        assert!(
                            same_bits(
                                &actual[part * n + base..(part + 1) * n],
                                &baseline[part * n + base..(part + 1) * n]
                            ),
                            "real/frequency update changed"
                        );
                    }
                    let (expected, clipped_g) = oracle(&c, &old, &gradient, step);
                    for index in 0..3 * n {
                        let coordinate = index % n;
                        let floor = if index < n {
                            2e-8
                        } else if index < 2 * n {
                            1e-12
                                + 4. * f64::from(f32::EPSILON)
                                    * (f64::from(old[n + coordinate]).abs()
                                        + 0.1 * clipped_g[coordinate].abs())
                        } else {
                            1e-18
                        };
                        let bound = floor + 4e-6 * expected[index].abs();
                        let error = (f64::from(actual[index]) - expected[index]).abs();
                        worst_ratio = worst_ratio.max(error / bound);
                        if !actual[index].is_finite() || error > bound {
                            misses.push(json!({"width":w,"heads":h,"layers":l,"scale":scale,"state":state,"step":step,"coordinate":index,"expected":expected[index],"actual":actual[index],"bound":bound,"error":error}));
                        }
                    }
                    if scale == 0. && state == 0 {
                        assert!(same_bits(&actual[frequencies..n], &initial[frequencies..n]));
                    }
                    checks += 1;
                }
            }
        }
        let mut invalid = opt.save(&e)?;
        invalid[2 * n + pairs[0].1] += 0.01;
        assert!(opt.restore(&mut e, &invalid).is_err());
    }
    record(
        "fp64-oracle",
        &json!({"checks":checks,"passed":misses.is_empty(),"misses":misses,"worst_error_over_bound":worst_ratio,"real_and_frequency_groups_bit_exact":true}),
    )?;
    assert_eq!(checks, 360);
    assert!(
        misses.is_empty(),
        "{} strict coordinate misses retained",
        misses.len()
    );
    Ok(())
}

fn rotate_state(c: &Config, state: &[f32], angle: f64, independent: bool) -> Vec<f32> {
    let (pairs, _, _, n) = geometry(c);
    let mut out = state.to_vec();
    for (j, pair) in pairs.into_iter().enumerate() {
        let (a, b, _) = pair;
        let (s, cos) = (angle
            + if independent {
                (j % 17) as f64 * 0.13
            } else {
                0.
            })
        .sin_cos();
        for offset in [0, n] {
            let re = (cos * f64::from(state[offset + a]) - s * f64::from(state[offset + b])) as f32;
            let im = (s * f64::from(state[offset + a]) + cos * f64::from(state[offset + b])) as f32;
            set_pair(&mut out[offset..offset + n], pair, re, im, false);
        }
    }
    out
}
#[test]
#[ignore = "requires CUDA"]
fn pair_updates_commute_with_rotations_over_repeated_steps() -> Result<()> {
    let c = config(64, 4, 2, true);
    let (pairs, base, _, n) = geometry(&c);
    let initial = layout::initialize_model(&c, 23);
    let mut e = Engine::new(c.clone(), &initial)?;
    let mut rotated = Engine::new(c.clone(), &initial)?;
    let mut opt = Optimizer::new(&e, 100000)?;
    let mut other = Optimizer::new(&rotated, 100000)?;
    let mut checks = 0;
    let mut misses = Vec::new();
    let mut worst = 0f64;
    for scale in [1e-8, 0.003, 1e8] {
        for (angle, independent) in [
            (0., false),
            (std::f64::consts::FRAC_PI_2, false),
            (-std::f64::consts::FRAC_PI_2, false),
            (std::f64::consts::PI, false),
            (0.71, false),
            (0.23, true),
        ] {
            let mut state = initial.clone();
            state.resize(3 * n, 0.);
            for (j, &pair) in pairs.iter().enumerate() {
                set_pair(
                    &mut state[n..2 * n],
                    pair,
                    ((j % 3) as f32 - 1.) * 0.0002,
                    ((j % 5) as f32 - 2.) * 0.0003,
                    false,
                );
                set_pair(&mut state[2 * n..], pair, 0.00002, 0.00002, true);
            }
            opt.restore(&mut e, &state)?;
            other.restore(&mut rotated, &rotate_state(&c, &state, angle, independent))?;
            for k in 0..19 {
                let mut gradient = vec![0f32; n];
                let mut rg = gradient.clone();
                for (j, &(a, b, _)) in pairs.iter().enumerate() {
                    let ga = ((j * 7 + k) % 23) as f32 * scale / 17. - scale * 0.7;
                    let gb = ((j * 11 + k) % 29) as f32 * scale / 19. - scale * 0.6;
                    let (s, cos) = (angle
                        + if independent {
                            (j % 17) as f64 * 0.13
                        } else {
                            0.
                        })
                    .sin_cos();
                    gradient[a] = ga;
                    gradient[b] = gb;
                    rg[a] = (cos * f64::from(ga) - s * f64::from(gb)) as f32;
                    rg[b] = (s * f64::from(ga) + cos * f64::from(gb)) as f32;
                }
                for i in base..n {
                    gradient[i] = ((i % 13) as f32 - 6.) * scale / 11.;
                    rg[i] = gradient[i];
                }
                e.gradients.write(&gradient)?;
                rotated.gradients.write(&rg)?;
                opt.step(&e, 2000 + k)?;
                other.step(&rotated, 2000 + k)?;
                let original = opt.save(&e)?;
                let actual = other.save(&rotated)?;
                let expected = rotate_state(&c, &original, angle, independent);
                for index in 0..3 * n {
                    let a = f64::from(actual[index]);
                    let x = f64::from(expected[index]);
                    let bound =
                        if index < 2 * n { 8e-8 } else { 1e-18 } + 8e-6 * a.abs().max(x.abs());
                    let error = (a - x).abs();
                    worst = worst.max(error / bound);
                    if !a.is_finite() || error > bound {
                        misses.push(json!({"scale":scale,"angle":angle,"independent":independent,"step":k,"coordinate":index,"expected":x,"actual":a,"bound":bound,"error":error}));
                    }
                }
                checks += 1;
            }
        }
    }
    record(
        "phase-rotations",
        &json!({"checks":checks,"passed":misses.is_empty(),"misses":misses,"worst_error_over_bound":worst}),
    )?;
    assert_eq!(checks, 342);
    assert!(misses.is_empty());
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn optimizer_choice_preserves_initial_forward_and_backward() -> Result<()> {
    let mut checks = 0;
    for (w, h, l) in [(64, 4, 2), (128, 4, 1), (128, 1, 1)] {
        let c = config(w, h, l, true);
        let p = layout::initialize_model(&c, 31);
        let mut candidate = Engine::new(c, &p)?;
        let mut control = Engine::new(config(w, h, l, false), &p)?;
        for (tokens, labels, batch, length) in [
            (
                vec![1, 2, 3, 4, 5, 6],
                vec![u32::MAX, 2, 3, u32::MAX, 5, 6],
                2,
                3,
            ),
            (
                vec![1, 2, 3, 4],
                vec![u32::MAX, u32::MAX, u32::MAX, 7],
                1,
                4,
            ),
        ] {
            candidate.forward(&tokens, &labels, batch, length)?;
            control.forward(&tokens, &labels, batch, length)?;
            candidate.backward(batch, length)?;
            control.backward(batch, length)?;
            assert!(same_bits(
                &candidate.logits.read()?,
                &control.logits.read()?
            ));
            assert!(same_bits(
                &candidate.gradients.read()?,
                &control.gradients.read()?
            ));
            checks += 1;
        }
    }
    record(
        "initial-forward-backward",
        &json!({"checks":checks,"passed":true,"logits_and_gradients_bit_exact":true}),
    )?;
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn pair_kernel_tails_canaries_and_duplicate_writes() -> Result<()> {
    use kernite_cuda::{Cuda, Grid};
    let cuda = Cuda::new()?;
    let kernels = crate::optimizer_kernels::kernels::Kernels::load(&cuda)?;
    let mut checks = 0;
    for count in [1usize, 7, 255, 256, 257, 513] {
        let complex_count = count - 1;
        let n = 4 * count + 13;
        let map: Vec<[u32; 5]> = (0..count)
            .map(|i| {
                let a = (4 * i) as u32;
                [
                    a,
                    a + 1,
                    a + 2,
                    a + 3,
                    u32::from(i + 1 == count || i % 2 == 1),
                ]
            })
            .collect();
        let mut written = vec![false; n];
        for &[a, b, neg, re, kind] in &map {
            written[a as usize] = true;
            written[b as usize] = true;
            if kind == 0 {
                written[neg as usize] = true;
                written[re as usize] = true;
            }
        }
        let old_p = vec![0.023f32; n];
        let old_m = vec![0.0003f32; n];
        let mut old_v = vec![0.0002f32; n];
        old_v[4 * (count - 1) + 1] = 0.0004;
        let p = cuda.upload(&old_p)?;
        let m = cuda.upload(&old_m)?;
        let v = cuda.upload(&old_v)?;
        let gradient: Vec<f32> = (0..n)
            .map(|i| {
                if i < 4 * count {
                    ((i * 7) % 19) as f32 / 11. - 0.4
                } else {
                    1e20
                }
            })
            .collect();
        let expanded = cuda.upload(&gradient)?;
        let owners = cuda.upload(map.as_flattened())?;
        let contracted = cuda.upload(&vec![12345f32; 2 * count + 19])?;
        let parts_count = (2 * count).div_ceil(256);
        let parts = cuda.upload(&vec![23456f32; parts_count + 7])?;
        let norm = cuda.upload(&[34567f32; 5])?;
        let grid = Grid::new(count.div_ceil(256) as u32, 256);
        // SAFETY: all buffers include canaries; each launch receives only its
        // logical length. The ownership map assigns disjoint output slots.
        unsafe {
            kernels.contract(
                grid,
                contracted.ptr(),
                expanded.ptr(),
                owners.ptr(),
                count as u32,
            )?;
            kernels.norm_partial(
                Grid::new(parts_count as u32, 32),
                parts.ptr(),
                contracted.ptr(),
                (2 * count) as u32,
            )?;
            kernels.norm_finish(
                Grid::new(1, 32),
                norm.ptr(),
                parts.ptr(),
                parts_count as u32,
            )?;
            kernels.adam_pair_coordinates(
                grid,
                p.ptr(),
                m.ptr(),
                v.ptr(),
                contracted.ptr(),
                owners.ptr(),
                norm.ptr(),
                count as u32,
                complex_count as u32,
                0.001,
                0.1,
                0.1,
                0.05f32.sqrt(),
            )?;
        }
        let g = contracted.read()?;
        assert!(g[2 * count..].iter().all(|&x| x == 12345.));
        assert!(parts.read()?[parts_count..].iter().all(|&x| x == 23456.));
        assert!(norm.read()?[1..].iter().all(|&x| x == 34567.));
        let actual_p = p.read()?;
        let actual_m = m.read()?;
        let actual_v = v.read()?;
        for i in 0..n {
            if !written[i] {
                assert_eq!(actual_p[i].to_bits(), old_p[i].to_bits());
                assert_eq!(actual_m[i].to_bits(), old_m[i].to_bits());
                assert_eq!(actual_v[i].to_bits(), old_v[i].to_bits());
            }
        }
        for (i, &[a, b, negative, real, kind]) in map.iter().enumerate() {
            let (a, b, negative, real) = (a as usize, b as usize, negative as usize, real as usize);
            assert_eq!(
                g[2 * i],
                gradient[a] + if kind == 0 { gradient[real] } else { 0. }
            );
            assert_eq!(
                g[2 * i + 1],
                gradient[b] - if kind == 0 { gradient[negative] } else { 0. }
            );
            if i < complex_count {
                assert_eq!(actual_v[a].to_bits(), actual_v[b].to_bits());
            }
            if kind == 0 {
                assert_eq!(actual_p[real], actual_p[a]);
                assert_eq!(actual_p[negative], -actual_p[b]);
                assert_eq!(actual_m[real], actual_m[a]);
                assert_eq!(actual_m[negative], -actual_m[b]);
                assert_eq!(actual_v[real], actual_v[a]);
                assert_eq!(actual_v[negative], actual_v[b]);
            }
        }
        assert_ne!(actual_v[4 * (count - 1)], actual_v[4 * (count - 1) + 1]);
        checks += 1;
    }
    record(
        "tails-canaries",
        &json!({"checks":checks,"passed":true,"logical_pair_counts":[1,7,255,256,257,513]}),
    )?;
    Ok(())
}
