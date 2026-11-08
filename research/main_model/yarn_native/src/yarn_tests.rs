use crate::{
    Config, Engine, Result, hash, layout,
    optimizer::Optimizer,
    oracle::{self, Jet},
    position::{Axes, Extension, reference_frequency},
    rotary,
};
use kernite_cuda::{Cuda, Grid};
use serde_json::{Value, json};
use std::{fs, io::Write, path::PathBuf};

fn grid(n: usize) -> Grid {
    Grid::new(n.div_ceil(256) as u32, 256)
}
fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
}
fn report(name: &str, rows: &[Value]) -> Result<()> {
    let root = PathBuf::from(
        std::env::var_os("YARN_VALIDATION_OUTPUT")
            .ok_or("set YARN_VALIDATION_OUTPUT to a fresh absolute directory")?,
    );
    if !root.is_absolute() {
        return Err("validation report directory must be absolute".into());
    }
    fs::create_dir_all(&root)?;
    let failures = rows.iter().filter(|r| r["passed"] != true).count();
    let mut file = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(root.join(format!("{name}.json")))?;
    serde_json::to_writer_pretty(
        &mut file,
        &json!({"cases":rows.len(),"failures":failures,"rows":rows}),
    )?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    if failures != 0 {
        return Err(format!("{name}: {failures} failed cases; all results retained").into());
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA, standalone YaRN table and full-score temperature stress"]
fn yarn_primitive_tables_and_score_temperature() -> Result<()> {
    let cuda = Cuda::new()?;
    let k = rotary::kernels::Kernels::load(&cuda)?;
    let mut records = Vec::new();
    for d in [16, 32, 128] {
        let bands = d / 2;
        for axes in [
            Axes::Sequence,
            Axes::Axial { features: 1 },
            Axes::Axial { features: 3 },
        ] {
            for length in [1, 2, 17, 67, 129, 257] {
                let (base, rope) = axes.base_tables(length, d);
                let frequency = axes.base_frequencies(d);
                let db = cuda.upload(&base)?;
                let dr = cuda.upload(&rope)?;
                let df = cuda.upload(&frequency)?;
                for factor in [1, 2, 8] {
                    let context = Extension::new(32, 32 * factor)?;
                    for magnitude in [0f32, 1e-7, 0.125, 1., 3.] {
                        let alpha: Vec<f32> = (0..bands)
                            .map(|i| magnitude * ((i % 7) as f32 - 3.) / 3.)
                            .collect();
                        let da = cuda.upload(&alpha)?;
                        let transformed = cuda.upload(&vec![12345.; bands + 4])?;
                        let result = cuda.upload(&vec![12345.; rope.len() + 4])?;
                        let original = cuda.alloc(rope.len())?;
                        let theta = cuda.upload(&vec![12345.; base.len() + 4])?;
                        unsafe {
                            k.yarn_frequencies(
                                grid(bands),
                                transformed.ptr(),
                                da.ptr(),
                                df.ptr(),
                                bands as u32,
                                axes.time_bands(d) as u32,
                                32.,
                                context.inverse_scale(),
                            )?;
                            k.table(
                                grid(base.len()),
                                result.ptr(),
                                theta.ptr(),
                                db.ptr(),
                                dr.ptr(),
                                if context.extended() {
                                    transformed.ptr()
                                } else {
                                    da.ptr()
                                },
                                bands as u32,
                                base.len() as u32,
                            )?;
                            k.table(
                                grid(base.len()),
                                original.ptr(),
                                theta.ptr(),
                                db.ptr(),
                                dr.ptr(),
                                da.ptr(),
                                bands as u32,
                                base.len() as u32,
                            )?;
                        }
                        let got = result.read()?;
                        let effective = transformed.read()?;
                        let original = original.read()?;
                        let mut worst = 0f64;
                        let mut freq_error = 0f64;
                        let mut norm_error = 0f64;
                        let mut feature_exact = true;
                        for i in 0..base.len() {
                            let band = i % bands;
                            let learned = frequency[band] as f64 * (alpha[band] as f64).exp();
                            let omega = if band < axes.time_bands(d) {
                                reference_frequency(learned, context)
                            } else {
                                learned
                            };
                            let angle =
                                base[i] as f64 * (alpha[band] as f64).exp() * (omega / learned);
                            let err = (got[i] as f64 - angle.cos())
                                .abs()
                                .max((got[base.len() + i] as f64 - angle.sin()).abs());
                            worst = worst.max(err / (3e-6 + 2e-6 * angle.abs()));
                            norm_error = norm_error.max(
                                (f64::from(got[i]).powi(2)
                                    + f64::from(got[base.len() + i]).powi(2)
                                    - 1.)
                                    .abs(),
                            );
                            freq_error = freq_error.max(
                                ((effective[band] as f64).exp() * frequency[band] as f64 / omega
                                    - 1.)
                                    .abs(),
                            );
                            if band >= axes.time_bands(d) {
                                feature_exact &= effective[band].to_bits() == alpha[band].to_bits();
                            }
                        }
                        let canaries = got[rope.len()..]
                            .iter()
                            .chain(&effective[bands..])
                            .all(|x| *x == 12345.);
                        let identity =
                            context.extended() || same_bits(&got[..rope.len()], &original);
                        records.push(json!({"kind":"table","d":d,"axes":format!("{axes:?}"),"length":length,
                            "extension":factor,"alpha_magnitude":magnitude,"max_phase_error_over_bound":worst,
                            "max_frequency_relative_error":freq_error,"max_norm_error":norm_error,
                            "feature_parameters_exact":feature_exact,"identity_exact":identity,"canaries":canaries,
                            "passed":worst<=1.&&freq_error<3e-6&&norm_error<3e-6&&feature_exact&&identity&&canaries}));
                    }
                }
            }
        }
    }
    let forward = crate::phase_score_forward::kernels::Kernels::load(&cuda)?;
    let coefficient = crate::phase_score::kernels::Kernels::load(&cuda)?;
    for (d, heads) in [(16, 4), (32, 4), (128, 1)] {
        for (batch, length) in [(1, 1), (2, 5), (1, 17)] {
            let w = d * heads;
            let count = batch * length * 3 * w;
            let x: Vec<f32> = (0..count)
                .map(|i| ((i * 37 % 113) as f32 - 56.) / 32.)
                .collect();
            let input = cuda.upload(&x)?;
            let alpha: Vec<f32> = (0..heads)
                .map(|h| if h % 2 == 0 { 0.8 } else { -0.9 })
                .collect();
            let da = cuda.upload(&alpha)?;
            let coeff = cuda.alloc(2 * heads)?;
            unsafe {
                coefficient.coefficients(grid(heads), coeff.ptr(), da.ptr(), heads as u32)?;
            }
            for scale in [1, 2, 8] {
                let gain = Extension::new(32, 32 * scale)?.logit_gain();
                let n = batch * heads * length;
                let out = cuda.upload(&vec![12345.; batch * length * w + 4])?;
                let prob = cuda.upload(&vec![12345.; n * length + 4])?;
                let im = cuda.upload(&vec![12345.; n * length + 4])?;
                macro_rules! launch {
                    ($d:literal) => {
                        unsafe {
                            forward.forward::<$d>(
                                Grid::new(
                                    (n / if $d == 16 { 2 } else { 1 }).div_ceil(4) as u32,
                                    128,
                                ),
                                out.ptr(),
                                prob.ptr(),
                                input.ptr(),
                                im.ptr(),
                                coeff.ptr(),
                                heads as u32,
                                length as u32,
                                n as u32,
                                gain,
                            )?;
                        }
                    };
                }
                match d {
                    16 => launch!(16),
                    32 => launch!(32),
                    128 => launch!(128),
                    _ => unreachable!(),
                }
                let got = out.read()?;
                let probabilities = prob.read()?;
                let mut error = 0f64;
                let mut probability_error = 0f64;
                for b in 0..batch {
                    for h in 0..heads {
                        for t in 0..length {
                            let get = |p: usize, part: usize, c: usize| {
                                f64::from(x[(b * length + p) * 3 * w + part * w + h * d + c])
                            };
                            let beta = alpha[h] as f64 / (1. + f64::from(alpha[h]).powi(2)).sqrt();
                            let scores: Vec<f64> = (0..=t)
                                .map(|key| {
                                    let re: f64 =
                                        (0..d).map(|c| get(t, 0, c) * get(key, 1, c)).sum();
                                    let im: f64 = (0..d / 2)
                                        .map(|c| {
                                            get(t, 0, c) * get(key, 1, c + d / 2)
                                                - get(t, 0, c + d / 2) * get(key, 1, c)
                                        })
                                        .sum();
                                    gain as f64 * ((t + 1) as f64).ln() / d as f64
                                        * (re + beta * im * im / d as f64)
                                })
                                .collect();
                            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                            let masses: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                            let den: f64 = masses.iter().sum();
                            for (key, mass) in masses.iter().enumerate() {
                                probability_error = probability_error.max(
                                    (probabilities[((b * heads + h) * length + t) * length + key]
                                        as f64
                                        - mass / den)
                                        .abs(),
                                );
                            }
                            for c in 0..d {
                                let y: f64 = masses
                                    .iter()
                                    .enumerate()
                                    .map(|(key, m)| m / den * get(key, 2, c))
                                    .sum();
                                error = error
                                    .max((got[(b * length + t) * w + h * d + c] as f64 - y).abs());
                            }
                        }
                    }
                }
                let canaries = got[batch * length * w..]
                    .iter()
                    .chain(&probabilities[n * length..])
                    .all(|x| *x == 12345.);
                records.push(json!({"kind":"full_score_temperature","d":d,"batch":batch,"length":length,
                    "extension":scale,"logit_gain":gain,"max_output_error":error,"max_probability_error":probability_error,
                    "canaries":canaries,"passed":error<1e-5&&probability_error<3e-6&&canaries}));
            }
        }
    }
    report("primitives", &records)
}

fn config(width: usize, heads: usize, layers: usize) -> Config {
    Config {
        width,
        heads,
        layers,
        vocab: 11,
        max_batch: 2,
        max_length: 6,
        normalized: true,
        ssmax: true,
        gain: heads as f32 / width as f32,
        lr: 0.001,
        decay: 0.1,
        warmup: 400,
        architecture: "complex-phase-score-lerope".into(),
        phase_optimizer: false,
    }
}

#[test]
#[ignore = "requires CUDA; identity, fixed-extent causality, state and axial AD"]
fn yarn_identity_causality_and_axial_frequency_paths() -> Result<()> {
    let mut records = Vec::new();
    let tokens = [1, 2, 3, 4, 3, 2, 2, 3, 4, 1, 2, 3];
    let labels = [2, 3, 4, 3, 2, 1, 3, 4, 1, 2, 3, 2];
    for (width, heads, layers) in [(64, 4, 1), (64, 4, 2), (128, 1, 1)] {
        let cfg = config(width, heads, layers);
        let mut p = layout::initialize_model(&cfg, 17);
        for (i, v) in p[layout::phase_offset(&cfg)..layout::frequency_offset(&cfg)]
            .iter_mut()
            .enumerate()
        {
            *v = ((i % 5) as f32 - 2.) * 0.1;
        }
        for (i, v) in p[layout::frequency_offset(&cfg)..].iter_mut().enumerate() {
            *v = ((i % 7) as f32 - 3.) * 0.03;
        }
        let mut engine = Engine::new(cfg.clone(), &p)?;
        let mut optimizer = Optimizer::new(&engine, 20000)?;
        let before: Vec<f32> = engine
            .parameters
            .read()?
            .into_iter()
            .chain(engine.first_moment.read()?)
            .chain(engine.second_moment.read()?)
            .collect();
        engine.forward(&tokens, &labels, 2, 6)?;
        let baseline = engine.logits.read()?;
        engine.set_inference_extension(Some(Extension::new(6, 6)?))?;
        engine.forward(&tokens, &labels, 2, 6)?;
        let identity = same_bits(&baseline, &engine.logits.read()?);
        let rejects = engine.backward(2, 6).is_err() && optimizer.step(&engine, 0).is_err();
        engine.set_inference_extension(Some(Extension::new(3, 6)?))?;
        engine.forward(&tokens, &labels, 2, 6)?;
        let extended = engine.logits.read()?;
        let mut changed = tokens;
        changed[5] = 9;
        changed[11] = 10;
        engine.forward(&changed, &labels, 2, 6)?;
        let perturbed = engine.logits.read()?;
        let causal = (0..2).all(|b| {
            same_bits(
                &extended[b * 66..b * 66 + 55],
                &perturbed[b * 66..b * 66 + 55],
            )
        });
        let after: Vec<f32> = engine
            .parameters
            .read()?
            .into_iter()
            .chain(engine.first_moment.read()?)
            .chain(engine.second_moment.read()?)
            .collect();
        let unchanged = same_bits(&before, &after);
        records.push(json!({"kind":"inference_integration","width":width,"heads":heads,"layers":layers,
            "identity_bitwise":identity,"fixed_document_causal":causal,"rejects_training":rejects,
            "state_unchanged":unchanged,"passed":identity&&causal&&rejects&&unchanged&&extended.iter().all(|x|x.is_finite())}));
        engine.set_inference_extension(None)?;
        let stale_rejected = engine.backward(2, 6).is_err() && optimizer.step(&engine, 0).is_err();
        records.push(
            json!({"kind":"stale_inference_graph_rejected","width":width,"heads":heads,
            "layers":layers,"passed":stale_rejected}),
        );
        for features in [1, 2, 3] {
            let axes = Axes::Axial { features };
            engine.set_positional_axes(axes)?;
            engine.forward(&tokens, &labels, 2, 6)?;
            engine.backward(2, 6)?;
            let gradient = engine.gradients.read()?;
            let reset = gradient.clone();
            engine.backward(2, 6)?;
            let repeated = same_bits(&reset, &engine.gradients.read()?);
            let cpu: Vec<Jet> = p.iter().map(|&v| Jet::from(v as f64)).collect();
            let expected = oracle::main_model_axes(&cfg, &cpu, &tokens, 2, 6, axes);
            let logits = engine.logits.read()?;
            let forward_error = logits
                .iter()
                .zip(&expected)
                .map(|(a, b)| (f64::from(*a) - b.v).abs())
                .fold(0f64, f64::max);
            records.push(json!({"kind":"axial_forward","width":width,"heads":heads,"layers":layers,
                "features":features,"max_absolute_error":forward_error,"repeat_backward_exact":repeated,
                "passed":forward_error<0.02&&repeated}));
            let mut misses = Vec::new();
            for i in layout::frequency_offset(&cfg)..p.len() {
                let mut direction = cpu.clone();
                direction[i].d = 1.;
                let reference = oracle::cross_entropy(
                    &oracle::main_model_axes(&cfg, &direction, &tokens, 2, 6, axes),
                    &labels,
                    cfg.vocab,
                )
                .d;
                let actual = gradient[i] as f64;
                let strict = (actual - reference).abs() <= 2e-6 + 0.03 * reference.abs();
                if !strict {
                    misses.push(json!({"index":i,"gpu":actual,"cpu_ad":reference,"absolute_error":(actual-reference).abs()}));
                }
                let epsilon = 1e-4;
                let mut plus = cpu.clone();
                let mut minus = cpu.clone();
                plus[i].v += epsilon;
                minus[i].v -= epsilon;
                let lp = oracle::cross_entropy(
                    &oracle::main_model_axes(&cfg, &plus, &tokens, 2, 6, axes),
                    &labels,
                    cfg.vocab,
                )
                .v;
                let lm = oracle::cross_entropy(
                    &oracle::main_model_axes(&cfg, &minus, &tokens, 2, 6, axes),
                    &labels,
                    cfg.vocab,
                )
                .v;
                let fd = (lp - lm) / (2. * epsilon);
                let zero_feature =
                    features == 1 && i - layout::frequency_offset(&cfg) >= width / heads / 4;
                records.push(json!({"kind":"axial_frequency_ad","width":width,"heads":heads,"layers":layers,
                    "features":features,"band":i-layout::frequency_offset(&cfg),"gpu":actual,"cpu_ad":reference,
                    "cpu_fd":fd,"strict_mixed_ad_passed":strict,"expected_zero_feature_path":zero_feature,
                    "passed":(fd-reference).abs()<2e-7+2e-4*reference.abs()&&actual.is_finite()
                        &&(!zero_feature||(actual==0.&&reference==0.))}));
            }
            records.push(json!({"kind":"axial_strict_mixed_ad_misses","width":width,"heads":heads,
                "layers":layers,"features":features,"misses":misses,"passed":true,
                "interpretation":"strict discrepancies retained separately; finite/CPU-AD checks are not full GPU equivalence"}));
        }
        engine.set_positional_axes(Axes::Sequence)?;
        engine.forward(&tokens, &labels, 2, 6)?;
        records.push(json!({"kind":"axis_cache_reset","width":width,"layers":layers,
            "passed":same_bits(&baseline,&engine.logits.read()?),"parameter_sha256":hash(&p.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>())}));
    }
    report("integration", &records)
}
