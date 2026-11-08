//! Independent complex arithmetic oracle, launch guards and model integration.
use crate::{
    Config, Engine, Result, layout,
    optimizer::Optimizer,
    oracle::{self, Jet},
    runner,
};
use kernite_cuda::{Cuda, Grid};
use serde_json::{Value, json};
use std::{fs, io::Write, path::Path};

const GUARD: f32 = 12345.;
fn record(name: &str, value: Value) -> Result<()> {
    if let Some(root) = std::env::var_os("PHASE_VALIDATION_OUTPUT") {
        let path = Path::new(&root).join(format!("{name}.json"));
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?;
        file.write_all(&serde_json::to_vec_pretty(&value)?)?;
        file.sync_all()?;
    }
    Ok(())
}
fn grid(n: usize) -> Grid {
    Grid::new(n.div_ceil(256) as u32, 256)
}
fn config(w: usize, heads: usize, layers: usize, phase: bool) -> Config {
    Config {
        width: w,
        heads,
        layers,
        vocab: 11,
        max_batch: 3,
        max_length: 4,
        normalized: true,
        ssmax: true,
        gain: heads as f32 / w as f32,
        lr: 0.001,
        decay: 0.1,
        warmup: 400,
        architecture: if phase {
            "complex-phase-mhc-lerope"
        } else {
            "complex-mhc-lerope"
        }
        .into(),
        phase_optimizer: false,
    }
}
fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}
#[derive(Clone, Copy, Default)]
struct C(f64, f64);
impl C {
    fn add(self, z: Self) -> Self {
        Self(self.0 + z.0, self.1 + z.1)
    }
    fn mul(self, z: Self) -> Self {
        Self(self.0 * z.0 - self.1 * z.1, self.0 * z.1 + self.1 * z.0)
    }
    fn conj(self) -> Self {
        Self(self.0, -self.1)
    }
    fn scale(self, s: f64) -> Self {
        Self(self.0 * s, self.1 * s)
    }
    fn dot(self, z: Self) -> f64 {
        self.0 * z.0 + self.1 * z.1
    }
    fn j(self) -> Self {
        Self(-self.1, self.0)
    }
}
fn phases(a: [f32; 3]) -> [C; 4] {
    std::array::from_fn(|i| {
        if i == 0 {
            C(1., 0.)
        } else {
            let value = f64::from(a[i - 1]);
            let den = value.hypot(1.);
            C(1. / den, value / den)
        }
    })
}
struct Oracle {
    y: Vec<f64>,
    dx: Vec<f64>,
    db: Vec<f64>,
    dg: Vec<f64>,
    da: [f64; 3],
    abs_da: [f64; 3],
}
// Complex-number scalar oracle; indexing is independent of the PTX partner helper.
fn reference(
    (w, heads, rows): (usize, usize, usize),
    a: [f32; 3],
    x: &[f32],
    b: &[f32],
    dy: &[f32],
    g: &[f32],
    scale: f32,
) -> Oracle {
    let mut o = Oracle {
        y: vec![0.; rows * 4 * w],
        dx: vec![0.; rows * 4 * w],
        db: vec![0.; rows * w],
        dg: vec![0.; rows * 24],
        da: [0.; 3],
        abs_da: [0.; 3],
    };
    let u = phases(a);
    let d = w / heads;
    for row in 0..rows {
        for head in 0..heads {
            for band in 0..d / 2 {
                let re = head * d + band;
                let im = re + d / 2;
                let pair = |v: &[f32], stream: usize| {
                    C(
                        f64::from(v[(row * 4 + stream) * w + re]),
                        f64::from(v[(row * 4 + stream) * w + im]),
                    )
                };
                let branch = C(f64::from(b[row * w + re]), f64::from(b[row * w + im]));
                let rotated: Vec<_> = (0..4).map(|j| u[j].conj().mul(pair(x, j))).collect();
                for i in 0..4 {
                    let upstream = pair(dy, i);
                    let mut sum = C::default();
                    let post = f64::from(g[row * 24 + 4 + i]) * f64::from(scale);
                    let br = upstream.scale(post);
                    o.db[row * w + re] += br.0;
                    o.db[row * w + im] += br.1;
                    o.dg[row * 24 + 4 + i] += upstream.dot(branch) * f64::from(scale);
                    for j in 0..4 {
                        let p = f64::from(g[row * 24 + 8 + 4 * i + j]);
                        sum = sum.add(rotated[j].scale(p));
                        let z = u[i].mul(rotated[j]);
                        o.dg[row * 24 + 8 + 4 * i + j] += upstream.dot(z);
                        let back = u[j].mul(u[i].conj().mul(upstream)).scale(p);
                        o.dx[(row * 4 + j) * w + re] += back.0;
                        o.dx[(row * 4 + j) * w + im] += back.1;
                        let term = p * upstream.dot(z.j());
                        if i != j {
                            if i > 0 {
                                let v = term / (1. + f64::from(a[i - 1]).powi(2));
                                o.da[i - 1] += v;
                                o.abs_da[i - 1] += v.abs();
                            }
                            if j > 0 {
                                let v = -term / (1. + f64::from(a[j - 1]).powi(2));
                                o.da[j - 1] += v;
                                o.abs_da[j - 1] += v.abs();
                            }
                        }
                    }
                    let y = u[i].mul(sum).add(branch.scale(post));
                    o.y[(row * 4 + i) * w + re] = y.0;
                    o.y[(row * 4 + i) * w + im] = y.1;
                }
            }
        }
    }
    o
}
fn comparison(actual: &[f32], expected: &[f64]) -> Value {
    let mut ratio = 0f64;
    let mut misses = 0;
    let mut worst = Value::Null;
    for (i, (&got, &want)) in actual.iter().zip(expected).enumerate() {
        let bound = 2e-6 + 2e-5 * want.abs().max(f64::from(got).abs());
        let error = (f64::from(got) - want).abs();
        if !error.is_finite() || error > bound {
            misses += 1;
        }
        if error / bound > ratio || !error.is_finite() {
            ratio = error / bound;
            worst = json!({"index":i,"native":got,"reference":want,"bound":bound,"error":error});
        }
    }
    json!({"passed":misses==0,"misses":misses,"max_error_over_bound":ratio,"worst":worst})
}

#[test]
fn phase_coordinates_preserve_common_initialization_and_have_unique_owners() -> Result<()> {
    for (w, h, l) in [(64, 4, 1), (64, 4, 2), (128, 1, 1), (128, 4, 6)] {
        let base = config(w, h, l, false);
        let candidate = config(w, h, l, true);
        let old = layout::initialize_model(&base, 17);
        let new = layout::initialize_model(&candidate, 17);
        let offset = layout::base_count(&base) + 2 * l * (96 * w + 27);
        assert_eq!(layout::phase_offset(&candidate), offset);
        assert_eq!(layout::frequency_offset(&candidate), offset + 6 * l);
        assert_eq!(new.len(), old.len() + 6 * l);
        assert_eq!(
            layout::independent_count(&candidate),
            layout::independent_count(&base) + 6 * l
        );
        assert!(same_bits(&old[..offset], &new[..offset]));
        assert!(same_bits(&old[offset..], &new[offset + 6 * l..]));
        assert!(new[offset..offset + 6 * l].iter().all(|x| x.to_bits() == 0));
        let mut seen = vec![0u8; new.len()];
        let base_owners = Optimizer::owners(&candidate)?;
        for [a, b, nb, aa, kind] in base_owners
            .iter()
            .copied()
            .chain(Optimizer::frequency_owners(&candidate))
        {
            seen[a as usize] += 1;
            seen[b as usize] += 1;
            if kind == 0 {
                seen[nb as usize] += 1;
                seen[aa as usize] += 1;
            }
        }
        assert!(seen.iter().all(|x| *x == 1));
        for pair in base_owners {
            if pair[0] as usize >= offset {
                assert_eq!(pair[4], 1);
            }
        }
        layout::validate_state(&candidate, &new, &vec![0.; new.len()], &vec![0.; new.len()])?;
    }
    Ok(())
}

#[test]
fn scalar_ad_checks_the_independent_complex_adjoint_formulas() {
    let (w, heads) = (8, 2);
    let cfg = config(w, heads, 1, true);
    let x: Vec<f32> = (0..4 * w).map(|i| (i as f32 - 13.) / 16.).collect();
    let b: Vec<f32> = (0..w).map(|i| (i as f32 - 3.) / 16.).collect();
    let dy: Vec<f32> = (0..4 * w)
        .map(|i| ((i * 7 % 19) as f32 - 9.) / 16.)
        .collect();
    let mut g = vec![0.25f32; 24];
    for i in 0..4 {
        g[4 + i] = 0.25 * (i + 1) as f32;
    }
    let a = [0.37, -0.81, 1.7];
    let expected = reference((w, heads, 1), a, &x, &b, &dy, &g, 0.5);
    for group in 0..4 {
        let reference: &[f64] = match group {
            0 => &expected.dx,
            1 => &expected.db,
            2 => &expected.dg,
            _ => &expected.da[..],
        };
        for (index, want) in reference.iter().enumerate() {
            let mut xj: Vec<_> = x.iter().map(|&v| Jet::from(f64::from(v))).collect();
            let mut bj: Vec<_> = b.iter().map(|&v| Jet::from(f64::from(v))).collect();
            let mut gj: Vec<_> = g.iter().map(|&v| Jet::from(f64::from(v))).collect();
            let mut aj: Vec<_> = a.iter().map(|&v| Jet::from(f64::from(v))).collect();
            match group {
                0 => xj[index].d = 1.,
                1 => bj[index].d = 1.,
                2 => gj[index].d = 1.,
                _ => aj[index].d = 1.,
            }
            let got: f64 = oracle::write_phase_mhc(&cfg, &xj, &bj, &gj, &aj, 0.5)
                .iter()
                .zip(&dy)
                .map(|(v, &d)| v.d * f64::from(d))
                .sum();
            assert!(
                (got - want).abs() <= 1e-12 * (1. + want.abs()),
                "group{group} index{index}: {got} versus {want}"
            );
        }
    }
}

#[test]
#[ignore = "requires CUDA"]
fn phase_skip_forward_and_all_adjoints_match_independent_complex_oracle() -> Result<()> {
    skip_oracle(false)
}

#[test]
#[ignore = "requires CUDA"]
fn normalized_state_skip_and_finite_matrix_amplification() -> Result<()> {
    skip_oracle(true)
}

// The supplementary normalized-state screen does not replace or relax the
// original extreme-amplitude screen. Its separate report retains every miss.
fn skip_oracle(normalized_state: bool) -> Result<()> {
    let cuda = Cuda::new()?;
    let k = crate::phase_mhc::kernels::Kernels::load(&cuda)?;
    let real = crate::mhc::kernels::Kernels::load(&cuda)?;
    let phase_sets = [
        [0., 0., 0.],
        [1e-8, -1e-8, 0.5e-8],
        [0.001, -0.002, 0.003],
        [1., -1., 0.5],
        [16., -16., 8.],
        [1e10, -1e10, 3e9],
        [1e30, -1e30, 3e29],
        [0.37, -0.81, 1.7],
    ];
    let mut records = Vec::new();
    for (w, heads) in [(64, 4), (64, 1), (128, 4), (128, 1), (8, 2)] {
        for rows in [1, 3, 5] {
            for (phase_id, &a) in phase_sets.iter().enumerate() {
                let scales: &[f32] = if normalized_state {
                    &[1., 2., 6.] // depths; exact-real radius, not an FP32 proof
                } else {
                    &[0., 1e-8, 1., 1e8]
                };
                for &amplitude in scales {
                    for map in 0..4 {
                        let n = rows * 4 * w;
                        let bn = rows * w;
                        let mut x: Vec<_> = (0..n)
                            .map(|i| ((i * 17 % 31) as f32 - 15.) * amplitude / 16.)
                            .collect();
                        let mut b: Vec<_> = (0..bn)
                            .map(|i| ((i * 7 % 19) as f32 - 9.) * amplitude / 16.)
                            .collect();
                        if normalized_state {
                            let radius =
                                (w as f64).sqrt() * (1. + 2. * (2. * f64::from(amplitude)).sqrt());
                            for stream in x.chunks_exact_mut(w) {
                                let norm = stream
                                    .iter()
                                    .map(|&v| f64::from(v).powi(2))
                                    .sum::<f64>()
                                    .sqrt();
                                for v in stream {
                                    *v = (f64::from(*v) * radius / norm) as f32;
                                }
                            }
                            // Zero branch isolates the finite skip-map amplification.
                            // Branch adjoints still use nonzero upstream gradients.
                            b.fill(0.);
                        }
                        let dy: Vec<_> =
                            (0..n).map(|i| ((i * 13 % 23) as f32 - 11.) / 16.).collect();
                        let mut gates = vec![0.5; rows * 24];
                        for row in 0..rows {
                            for i in 0..4 {
                                gates[row * 24 + 4 + i] = 0.25 * (i + 1) as f32;
                                for j in 0..4 {
                                    gates[row * 24 + 8 + 4 * i + j] = match map {
                                        0 => {
                                            if i == j {
                                                1.
                                            } else {
                                                0.
                                            }
                                        }
                                        1 => 0.25,
                                        2 => {
                                            if j == (i + 1) % 4 {
                                                1.
                                            } else {
                                                0.
                                            }
                                        }
                                        _ => {
                                            if j == 0 {
                                                0.9997
                                            } else {
                                                0.0001
                                            }
                                        }
                                    };
                                }
                            }
                        }
                        let scale = 0.5;
                        let expected = reference((w, heads, rows), a, &x, &b, &dy, &gates, scale);
                        let pa = cuda.upload(&a)?;
                        let table = cuda.upload(&[GUARD; 49])?;
                        let xx = cuda.upload(&x)?;
                        let bb = cuda.upload(&b)?;
                        let yy = cuda.upload(&dy)?;
                        let gg = cuda.upload(&gates)?;
                        let out = cuda.upload(&vec![GUARD; n + 5])?;
                        let dx = cuda.upload(&vec![GUARD; n + 5])?;
                        let db = cuda.upload(&vec![GUARD; bn + 5])?;
                        let dg = cuda.upload(&vec![GUARD; rows * 24 + 5])?;
                        let da = cuda.upload(&[GUARD; 8])?;
                        let dread = cuda.upload(&vec![0.; bn])?;
                        unsafe {
                            k.table(Grid::new(1, 32), table.ptr(), pa.ptr())?;
                            k.write(
                                grid(n),
                                out.ptr(),
                                xx.ptr(),
                                bb.ptr(),
                                gg.ptr(),
                                table.ptr(),
                                w as u32,
                                heads as u32,
                                n as u32,
                                scale,
                            )?;
                            k.write_backward(
                                grid(bn),
                                dx.ptr(),
                                db.ptr(),
                                yy.ptr(),
                                gg.ptr(),
                                table.ptr(),
                                w as u32,
                                heads as u32,
                                bn as u32,
                                scale,
                            )?;
                            real.gate_grad(
                                Grid::new((rows * 6) as u32, 128),
                                dg.ptr(),
                                xx.ptr(),
                                bb.ptr(),
                                yy.ptr(),
                                dread.ptr(),
                                w as u32,
                                rows as u32,
                                scale,
                            )?;
                            k.skip_gate_grad(
                                Grid::new((rows * 4) as u32, 128),
                                dg.ptr(),
                                xx.ptr(),
                                yy.ptr(),
                                table.ptr(),
                                w as u32,
                                heads as u32,
                                rows as u32,
                            )?;
                            k.phase_grad(
                                Grid::new(3, 32),
                                da.ptr(),
                                xx.ptr(),
                                yy.ptr(),
                                gg.ptr(),
                                table.ptr(),
                                w as u32,
                                heads as u32,
                                rows as u32,
                            )?;
                        }
                        let tv = table.read()?;
                        let ov = out.read()?;
                        let xv = dx.read()?;
                        let bv = db.read()?;
                        let gv = dg.read()?;
                        let av = da.read()?;
                        let checks = [
                            comparison(&ov, &expected.y),
                            comparison(&xv, &expected.dx),
                            comparison(&bv, &expected.db),
                            comparison(&gv, &expected.dg),
                        ];
                        let phase_checks:Vec<_>=(0..3).map(|i|{
                let bound=3e-6+2e-5*expected.abs_da[i];let error=(f64::from(av[i])-expected.da[i]).abs();
                json!({"parameter":i,"native":av[i],"reference":expected.da[i],"error":error,"bound":bound,"passed":error.is_finite()&&error<=bound})
            }).collect();
                        let unit_error = (0..4)
                            .map(|i| {
                                (f64::from(tv[2 * i]).powi(2) + f64::from(tv[2 * i + 1]).powi(2)
                                    - 1.)
                                    .abs()
                            })
                            .chain((0..16).map(|i| {
                                (f64::from(tv[12 + 2 * i]).powi(2)
                                    + f64::from(tv[13 + 2 * i]).powi(2)
                                    - 1.)
                                    .abs()
                            }))
                            .fold(0f64, f64::max);
                        let canaries = tv[44..]
                            .iter()
                            .chain(&ov[n..])
                            .chain(&xv[n..])
                            .chain(&bv[bn..])
                            .chain(&gv[rows * 24..])
                            .chain(&av[3..])
                            .all(|&v| v == GUARD);
                        let amplification = if normalized_state {
                            let input_norm =
                                x.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>().sqrt();
                            let native = ov[..n]
                                .iter()
                                .map(|&v| f64::from(v).powi(2))
                                .sum::<f64>()
                                .sqrt()
                                / input_norm;
                            let reference =
                                expected.y.iter().map(|v| v * v).sum::<f64>().sqrt() / input_norm;
                            let bound = 2e-6 + 2e-5 * reference.abs();
                            json!({"native":native,"reference":reference,"bound":bound,
                                "passed":(native-reference).abs()<=bound,
                                "scope":"observed finite-map energy ratio, not an operator-norm proof"})
                        } else {
                            json!({"passed":true,"scope":"not measured"})
                        };
                        let passed = checks.iter().all(|v| v["passed"] == true)
                            && phase_checks.iter().all(|v| v["passed"] == true)
                            && unit_error <= 2e-6
                            && canaries
                            && amplification["passed"] == true;
                        records.push(json!({"width":w,"heads":heads,"rows":rows,"phase_id":phase_id,"amplitude":amplitude,"map":map,"passed":passed,
                "forward":checks[0],"input_adjoint":checks[1],"branch_adjoint":checks[2],"gate_adjoint":checks[3],"phase_adjoint":phase_checks,
                "unit_energy_error":unit_error,"canaries":canaries,"finite_map_amplification":amplification}));
                    }
                }
            }
        }
    }
    let misses = records.iter().filter(|r| r["passed"] != true).count();
    record(
        if normalized_state {
            "normalized-state-oracle"
        } else {
            "kernel-oracle"
        },
        json!({"cases":records,"missed_cases":misses,"normalized_state":normalized_state,
            "radius_scope":"exact-real row-normalized skip bound; finite inputs tested, no FP32 proof",
            "source_sha256":runner::source_hash(),"ptx_sha256":runner::ptx_hash()}),
    )?;
    assert_eq!(
        misses, 0,
        "new phase-routing oracle misses retained in kernel-oracle.json"
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn phase_model_integration_and_complete_new_gradient_paths() -> Result<()> {
    let tokens = [1, 2, 3, 4, 3, 2];
    let labels = [2, 3, 4, 3, 2, 1];
    let mut records = Vec::new();
    for (w, heads, layers) in [(64, 4, 1), (64, 4, 2), (128, 1, 1)] {
        let c0 = config(w, heads, layers, false);
        let p0 = layout::initialize_model(&c0, 17);
        let mut control = Engine::new(c0.clone(), &p0)?;
        control.forward(&tokens, &labels, 2, 3)?;
        control.backward(2, 3)?;
        let baseline_logits = control.logits.read()?;
        let baseline_gradient = control.gradients.read()?;
        let baseline_input = control.embedding_gradient_probe(6)?;
        drop(control);
        let cfg = config(w, heads, layers, true);
        let offset = layout::phase_offset(&cfg);
        let frequencies = layout::frequency_offset(&cfg);
        let phases = 6 * layers;
        let mut parameters = layout::initialize_model(&cfg, 17);
        let mut engine = Engine::new(cfg.clone(), &parameters)?;
        engine.forward(&tokens, &labels, 2, 3)?;
        engine.backward(2, 3)?;
        let initial_logits = engine.logits.read()?;
        let gradient = engine.gradients.read()?;
        let old = layout::frequency_offset(&c0);
        let logits_equal = same_bits(
            &baseline_logits[..6 * cfg.vocab],
            &initial_logits[..6 * cfg.vocab],
        );
        let gradients_equal = same_bits(&baseline_gradient[..old], &gradient[..offset])
            && same_bits(&baseline_gradient[old..], &gradient[frequencies..]);
        let inputs_equal = same_bits(&baseline_input, &engine.embedding_gradient_probe(6)?);
        let initial_parameters_equal = same_bits(&p0[..old], &parameters[..offset])
            && same_bits(&p0[old..], &parameters[frequencies..]);
        engine.backward(2, 3)?;
        let repeat_equal = same_bits(&gradient, &engine.gradients.read()?);
        records.push(json!({"kind":"initial_integration","width":w,"heads":heads,"layers":layers,
            "passed":logits_equal&&gradients_equal&&inputs_equal&&initial_parameters_equal&&repeat_equal,
            "logits_equal":logits_equal,"original_gradients_equal":gradients_equal,"input_gradients_equal":inputs_equal,
            "common_initial_parameters_equal":initial_parameters_equal,"backward_reset_equal":repeat_equal,
            "phase_gradient_nonzero":gradient[offset..frequencies].iter().any(|v|*v!=0.)}));

        for phase_state in 0..2 {
            for (i, value) in parameters[offset..frequencies].iter_mut().enumerate() {
                *value = if phase_state == 0 {
                    0.
                } else {
                    [0.37, -0.81, 1.7][i % 3] * (1. + (i / 3) as f32 * 0.1)
                };
            }
            engine.parameters.write(&parameters)?;
            engine.forward(&tokens, &labels, 2, 3)?;
            engine.backward(2, 3)?;
            let native = engine.gradients.read()?;
            let logits = engine.logits.read()?;
            let mut directions: Vec<(String, usize)> =
                (offset..frequencies).map(|i| ("phase".into(), i)).collect();
            for layer in 0..2 * layers {
                let router = layout::base_count(&cfg) + layer * (96 * w + 27);
                for gate in 0..24 {
                    let group = if gate < 4 {
                        "read"
                    } else if gate < 8 {
                        "write"
                    } else {
                        "skip"
                    };
                    directions.push((format!("router{layer}-{group}"), router + 96 * w + gate));
                }
                for group in 0..3 {
                    directions.push((format!("router{layer}-scale"), router + 96 * w + 24 + group));
                    directions.push((
                        format!("router{layer}-projection"),
                        router + (4 * group) * (4 * w) + group,
                    ));
                }
            }
            directions.push(("frequency".into(), frequencies));
            directions.push(("frequency".into(), parameters.len() - 1));
            let mut expected = Vec::new();
            let mut finite_differences = Vec::new();
            for (group, at) in &directions {
                let mut p: Vec<Jet> = parameters
                    .iter()
                    .map(|&v| Jet::from(f64::from(v)))
                    .collect();
                p[*at].d = 1.;
                let value = oracle::main_model(&cfg, &p, &tokens, 2, 3);
                let derivative = oracle::cross_entropy(&value, &labels, cfg.vocab).d;
                expected.push(derivative);
                {
                    p[*at].d = 0.;
                    let center = p[*at].v;
                    let h = 1e-5 * (1. + center.abs());
                    p[*at].v = center + h;
                    let plus = oracle::cross_entropy(
                        &oracle::main_model(&cfg, &p, &tokens, 2, 3),
                        &labels,
                        cfg.vocab,
                    )
                    .v;
                    p[*at].v = center - h;
                    let minus = oracle::cross_entropy(
                        &oracle::main_model(&cfg, &p, &tokens, 2, 3),
                        &labels,
                        cfg.vocab,
                    )
                    .v;
                    let difference = (plus - minus) / (2. * h);
                    let bound = 2e-8 + 2e-5 * derivative.abs();
                    finite_differences.push(
                        json!({"path":group,"parameter":at,"AD":derivative,"finite_difference":difference,
                        "bound":bound,"passed":(difference-derivative).abs()<=bound}),
                    );
                }
            }
            let mut norm = std::collections::BTreeMap::<String, f64>::new();
            for ((group, _), &value) in directions.iter().zip(&expected) {
                *norm.entry(group.clone()).or_default() += value * value;
            }
            let comparisons: Vec<_> = directions
                .iter()
                .zip(&expected)
                .map(|((group, at), &want)| {
                    let bound = 5e-4 * norm[group].sqrt() + 1e-12;
                    let got = f64::from(native[*at]);
                    let error = (got - want).abs();
                    json!({"path":group,"parameter":at,"native":got,"reference":want,"bound":bound,
                    "error":error,"passed":error.is_finite()&&error<=bound})
                })
                .collect();
            let mut changed = tokens;
            changed[2] = 9;
            changed[5] = 10;
            engine.forward(&changed, &labels, 2, 3)?;
            let future = engine.logits.read()?;
            let causal = [0, 1, 3, 4].iter().all(|&row| {
                same_bits(
                    &logits[row * cfg.vocab..(row + 1) * cfg.vocab],
                    &future[row * cfg.vocab..(row + 1) * cfg.vocab],
                )
            });
            let effect = phase_state == 0
                || !same_bits(&logits[..6 * cfg.vocab], &initial_logits[..6 * cfg.vocab]);
            let fd_passed = finite_differences.iter().all(|r| r["passed"] == true);
            records.push(json!({"kind":"full_model_AD","width":w,"heads":heads,"layers":layers,"phase_state":phase_state,
                "passed":causal&&effect&&fd_passed,"strict_AD_passed":comparisons.iter().all(|r|r["passed"]==true),
                "directions":comparisons,"finite_differences":finite_differences,"causal":causal,"nonzero_phase_effect":effect,
                "independent_phase_parameters":phases}));
        }
    }
    let implementation_passed = records.iter().all(|r| r["passed"] == true);
    let strict_misses: usize = records
        .iter()
        .filter_map(|r| r["directions"].as_array())
        .map(|ds| ds.iter().filter(|d| d["passed"] != true).count())
        .sum();
    record(
        "full-model",
        json!({"cases":records,"implementation_passed":implementation_passed,"strict_AD_misses":strict_misses,
        "source_sha256":runner::source_hash(),"ptx_sha256":runner::ptx_hash(),"dtype":crate::precision::TAG,
        "all_affine_FP32_is_diagnostic_only":true}),
    )?;
    eprintln!(
        "Full-model phase integration: implementation_passed={implementation_passed}, strict_AD_misses={strict_misses}"
    );
    assert!(
        implementation_passed,
        "full-model phase routing implementation checks failed; strict AD misses are separately retained"
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn real_phase_optimizer_and_shifted_frequencies_match_independent_fp64() -> Result<()> {
    use crate::schedule::Schedule;
    let mut records = Vec::new();
    for (width, heads, layers, budget) in [(64, 4, 2, 20000), (128, 1, 1, 100000)] {
        let mut cfg = config(width, heads, layers, true);
        let schedule = Schedule::new(budget)?;
        cfg.warmup = schedule.warmup;
        let base = layout::base_count(&cfg);
        let phase = layout::phase_offset(&cfg);
        let frequency = layout::frequency_offset(&cfg);
        let mut initial = layout::initialize_model(&cfg, 19);
        for (i, p) in initial[phase..].iter_mut().enumerate() {
            *p = 0.04 * (i as f32 - 2.);
        }
        let n = initial.len();
        let mut engine = Engine::new(cfg.clone(), &initial)?;
        let mut optimizer = Optimizer::new(&engine, budget)?;
        let before_guard = optimizer.save(&engine)?;
        assert!(engine.update(0).is_err());
        assert!(same_bits(&before_guard, &optimizer.save(&engine)?));
        for (base_scale, phase_scale, frequency_scale) in [
            (0., 0., 0.),
            (1e-8, 1e-8, 1e-8),
            (1e8, 1e-8, 1e-8),
            (1e-8, 1e8, 1e-8),
            (1e-8, 1e-8, 1e8),
            (1e8, 1e8, 1e8),
        ] {
            for step in [
                0,
                1,
                schedule.warmup - 1,
                schedule.warmup,
                schedule.cooldown_start - 1,
                schedule.cooldown_start,
                (schedule.cooldown_start + budget) / 2,
                budget - 1,
            ] {
                let gradient: Vec<f32> = (0..n)
                    .map(|i| {
                        let scale = if i < phase {
                            base_scale
                        } else if i < frequency {
                            phase_scale
                        } else {
                            frequency_scale
                        };
                        ((i * 7 % 29) as f32 - 14.) * scale / 17.
                    })
                    .collect();
                let mut m = vec![0.; n];
                let mut v = vec![0.; n];
                if step > 1 {
                    for i in base..n {
                        m[i] = ((i - base) % 3) as f32 * 0.0003 - 0.0003;
                        v[i] = 0.00002 + ((i - base) % 2) as f32 * 0.00001;
                    }
                }
                // Independent-coordinate contraction from matrix geometry,
                // without the optimizer's packed ownership map or GPU reductions.
                let mut coordinates = Vec::new();
                for [a, b, nb, aa, kind] in layout::pairs(&cfg) {
                    let mut ga = f64::from(gradient[a as usize]);
                    let mut gb = f64::from(gradient[b as usize]);
                    if kind == 0 {
                        ga += f64::from(gradient[aa as usize]);
                        gb -= f64::from(gradient[nb as usize]);
                    }
                    coordinates.push((a as usize, ga));
                    coordinates.push((b as usize, gb));
                }
                coordinates.extend((base..frequency).map(|i| (i, f64::from(gradient[i]))));
                let base_norm = coordinates
                    .iter()
                    .map(|(_, g)| g * g)
                    .sum::<f64>()
                    .sqrt()
                    .max(1.);
                let frequency_norm = gradient[frequency..]
                    .iter()
                    .map(|&g| f64::from(g).powi(2))
                    .sum::<f64>()
                    .sqrt()
                    .max(1.);
                coordinates.extend((frequency..n).map(|i| (i, f64::from(gradient[i]))));
                engine.parameters.write(&initial)?;
                engine.first_moment.write(&m)?;
                engine.second_moment.write(&v)?;
                engine.gradients.write(&gradient)?;
                optimizer.step(&engine, step)?;
                let actual = optimizer.save(&engine)?; // verifies all expanded ties/state
                let rate = f64::from(schedule.rate(step));
                let t = (step + 1) as f64;
                let bc1 = 1. - 0.9f64.powf(t);
                let bc2 = 1. - 0.95f64.powf(t);
                let mut misses = 0;
                let mut max_ratio = 0f64;
                let mut worst = Value::Null;
                for &(i, unscaled) in &coordinates {
                    let is_frequency = i >= frequency;
                    let g = unscaled
                        / if is_frequency {
                            frequency_norm
                        } else {
                            base_norm
                        };
                    let first = 0.9 * f64::from(m[i]) + 0.1 * g;
                    let second = 0.95 * f64::from(v[i]) + 0.05 * g * g;
                    let decay = if is_frequency { 0. } else { 0.1 };
                    let parameter = f64::from(initial[i]) * (1. - rate * decay)
                        - rate * (first / bc1) / ((second / bc2).sqrt() + 1e-8);
                    let cancellation =
                        4. * f64::from(f32::EPSILON) * (f64::from(m[i]).abs() + 0.1 * g.abs());
                    for (part, expected, floor) in [
                        (0, parameter, 2e-8),
                        (1, first, 1e-12 + cancellation),
                        (2, second, 1e-18),
                    ] {
                        let have = f64::from(actual[part * n + i]);
                        let error = (have - expected).abs();
                        let bound = floor + 4e-6 * expected.abs();
                        if !error.is_finite() || error > bound {
                            misses += 1;
                        }
                        if error / bound > max_ratio || !error.is_finite() {
                            max_ratio = error / bound;
                            worst = json!({"parameter":i,"state_part":part,"native":have,
                                "reference":expected,"error":error,"bound":bound});
                        }
                    }
                    if is_frequency && frequency_scale == 0. && step <= 1 {
                        assert_eq!(actual[i].to_bits(), initial[i].to_bits(), "frequency decay");
                    }
                }
                records.push(
                    json!({"width":width,"heads":heads,"layers":layers,"budget":budget,
                    "step":step,"base_scale":base_scale,"phase_scale":phase_scale,
                    "frequency_scale":frequency_scale,"base_clip_denominator":base_norm,
                    "frequency_clip_denominator":frequency_norm,"comparisons":3*coordinates.len(),
                    "phase_parameters":frequency-phase,"passed":misses==0,"misses":misses,
                    "max_error_over_bound":max_ratio,"worst":worst}),
                );
            }
        }
    }
    let misses: usize = records
        .iter()
        .map(|r| r["misses"].as_u64().unwrap() as usize)
        .sum();
    assert_eq!(records.len(), 96);
    record(
        "phase-optimizer",
        json!({"cases":records,"misses":misses,
        "source_sha256":runner::source_hash(),"ptx_sha256":runner::ptx_hash()}),
    )?;
    assert_eq!(
        misses, 0,
        "phase/group optimizer misses retained in phase-optimizer.json"
    );
    eprintln!("96 independent phase/base/frequency optimizer cases passed");
    Ok(())
}
