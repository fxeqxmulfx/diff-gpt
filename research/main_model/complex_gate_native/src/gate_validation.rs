//! Independent sigmoid/adjoint references and complete-model gate integration.
use crate::{
    Config, Engine, Result, gate, hash, layout,
    oracle::{self, Jet},
    random::Random,
    runner,
};
use kernite_cuda::{Cuda, Grid};
use serde_json::{Value, json};
use std::{fs, path::Path};

fn grid(n: usize) -> Grid {
    Grid::new(n.div_ceil(256) as u32, 256)
}
fn equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}
fn reference(z: f64) -> (f64, f64) {
    let e = (-z.abs()).exp();
    (
        if z >= 0. { 1. / (1. + e) } else { e / (1. + e) },
        e / (1. + e).powi(2),
    )
}
fn cfg(w: usize, h: usize, l: usize, gated: bool) -> Config {
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
        warmup: 400,
        phase_optimizer: false,
        architecture: if gated {
            "complex-mhc-complex-predictor-gated"
        } else {
            "complex-mhc"
        }
        .into(),
    }
}

fn primitives(records: &mut Vec<Value>) -> Result<()> {
    let cuda = Cuda::new()?;
    let k = gate::kernels::Kernels::load(&cuda)?;
    for n in [1, 7, 31, 32, 33, 255, 256, 257, 1025] {
        for pattern in ["zero", "random", "saturated", "extreme"] {
            let mut rng = Random::new(91);
            let logits: Vec<f32> = (0..n)
                .map(|i| match pattern {
                    "zero" => 0.,
                    "random" => rng.gauss(0., 3.) as f32,
                    "saturated" => {
                        [-100., -80., -40., -20., -1., 0., 1., 20., 40., 80., 100.][i % 11]
                    }
                    _ => [-1e4, -100., -1e-20, 0., 1e-20, 100., 1e4][i % 7],
                })
                .collect();
            for scale in [1e-20f32, 1., 1e20] {
                let a: Vec<f32> = (0..n)
                    .map(|i| ((i % 19) as f32 - 9.) / 7. * scale)
                    .collect();
                let dy: Vec<f32> = (0..n)
                    .map(|i| ((i % 13) as f32 - 6.) / 5. * if scale > 1e10 { 1e10 } else { 1. })
                    .collect();
                let da = cuda.upload(&a)?;
                let dz = cuda.upload(&logits)?;
                let out = cuda.upload(&vec![12345.; n + 4])?;
                let s = cuda.upload(&vec![12345.; n + 4])?;
                let ds = cuda.upload(&vec![12345.; n + 4])?;
                let tails = cuda.upload(&vec![12345.; n + 4])?;
                let mut padded_dy = dy.clone();
                padded_dy.extend([12345.; 4]);
                let incoming = cuda.upload(&padded_dy)?;
                let dlogit = cuda.upload(&vec![12345.; n + 4])?;
                // SAFETY: all valid arrays cover n; the extra four elements are canaries.
                unsafe {
                    k.forward(
                        grid(n),
                        out.ptr(),
                        s.ptr(),
                        ds.ptr(),
                        tails.ptr(),
                        da.ptr(),
                        dz.ptr(),
                        n as u32,
                    )?;
                    k.backward(
                        grid(n),
                        incoming.ptr(),
                        dlogit.ptr(),
                        incoming.ptr(),
                        da.ptr(),
                        tails.ptr(),
                        dz.ptr(),
                        n as u32,
                    )?;
                }
                let arrays = [
                    out.read()?,
                    s.read()?,
                    ds.read()?,
                    incoming.read()?,
                    dlogit.read()?,
                ];
                let mut max_ratio = 0f64;
                let mut failures = Vec::new();
                for i in 0..n {
                    let (sigmoid, slope) = reference(logits[i] as f64);
                    let expected = [
                        a[i] as f64 * sigmoid,
                        sigmoid,
                        slope,
                        dy[i] as f64 * sigmoid,
                        dy[i] as f64 * a[i] as f64 * slope,
                    ];
                    for j in 0..5 {
                        let error = (arrays[j][i] as f64 - expected[j]).abs();
                        let floor = 8.
                            * f32::from_bits(1) as f64
                            * if j == 4 {
                                (a[i] as f64 * dy[i] as f64).abs().max(1.)
                            } else {
                                1.
                            };
                        let tolerance = 3e-6 * expected[j].abs() + floor;
                        max_ratio = max_ratio.max(error / tolerance.max(1e-300));
                        if !arrays[j][i].is_finite() || error > tolerance {
                            failures.push(json!({"index":i,"path":j,"native":arrays[j][i],"reference":expected[j],"error":error,"tolerance":tolerance}));
                        }
                    }
                }
                let canaries = arrays.iter().all(|a| a[n..].iter().all(|x| *x == 12345.))
                    && tails.read()?[n..].iter().all(|x| *x == 12345.);
                let positive_tail = (0..n)
                    .filter(|&i| logits[i] == 20. || logits[i] == 40. || logits[i] == 80.)
                    .all(|i| arrays[2][i] > 0.);
                records.push(json!({"kind":"primitive","n":n,"pattern":pattern,"scale":scale,"passed":failures.is_empty()&&canaries&&positive_tail,"max_error_over_bound":max_ratio,"canaries":canaries,"positive_saturation_slope_retained":positive_tail,"failures":failures}));
            }
        }
    }
    Ok(())
}

fn model_checks(records: &mut Vec<Value>) -> Result<()> {
    let tokens = [1, 2, 3, 4, 3, 2];
    let labels = [2, 3, 4, 3, 2, 1];
    for (w, h, l) in [(64, 4, 1), (64, 4, 2), (128, 4, 1), (128, 1, 1)] {
        let c = cfg(w, h, l, true);
        let original = layout::initialize_model(&cfg(w, h, l, false), 13);
        let initial = layout::initialize_model(&c, 13);
        let common = layout::common_count(&c);
        records.push(json!({"kind":"initialization","width":w,"heads":h,"layers":l,
            "passed":equal(&initial[..common],&original)&&initial[common..].iter().all(|x|*x==0.),
            "extra_stored_real_entries":initial.len()-original.len(),"extra_independent_real_parameters":layout::gate_independent_count(&c),
            "common_initial_sha256":hash(&original.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>())}));
        for magnitude in [0., 0.02f32, 2.] {
            let mut rng = Random::new(812);
            let mut p = initial.clone();
            for [a, b, nb, aa, _] in layout::pairs(&c)
                .into_iter()
                .filter(|owner| owner[0] as usize >= common)
            {
                p[a as usize] = rng.gauss(0., magnitude as f64) as f32;
                p[b as usize] = rng.gauss(0., magnitude as f64) as f32;
                p[nb as usize] = -p[b as usize];
                p[aa as usize] = p[a as usize];
            }
            let mut e = Engine::new(c.clone(), &p)?;
            e.forward(&tokens, &labels, 2, 3)?;
            let logits = e.logits.read()?;
            let before = e.diagnostic_snapshot(6)?;
            e.backward(2, 3)?;
            let g = e.gradients.read()?;
            let after = e.diagnostic_snapshot(6)?;
            e.backward(2, 3)?;
            let overwrite = equal(&g, &e.gradients.read()?);
            let mut all_gemm_checks = true;
            let mut gemm_error = 0f64;
            for layer in 0..l {
                let input = &before[&format!("layer{layer}.attention_input_normalized")];
                let dz = &after[&format!("layer{layer}.grad_gate_logits")];
                let din = &after[&format!("layer{layer}.grad_gate_input")];
                let offset = layout::gate_offset(&c, layer);
                let logit = &before[&format!("layer{layer}.gate_logits")];
                let half = w / h / 2;
                // Direct complex arithmetic, independent of layout::pairs and
                // the expanded GEMM: every predictor output and every A/B adjoint.
                for oh in 0..h {
                    for oc in 0..half {
                        let ore = oh * 2 * half + oc;
                        let oim = ore + half;
                        for row in 0..6 {
                            let mut expected = [0f64; 2];
                            let mut sums = [0f64; 2];
                            for ih in 0..h {
                                for ic in 0..half {
                                    let ire = ih * 2 * half + ic;
                                    let iim = ire + half;
                                    let a = p[offset + ore * w + ire] as f64;
                                    let b = p[offset + oim * w + ire] as f64;
                                    let xr = input[row * w + ire] as f64;
                                    let xi = input[row * w + iim] as f64;
                                    expected[0] += a * xr - b * xi;
                                    expected[1] += b * xr + a * xi;
                                    sums[0] += (a * xr).abs() + (b * xi).abs();
                                    sums[1] += (b * xr).abs() + (a * xi).abs();
                                }
                            }
                            for (component, out) in [ore, oim].into_iter().enumerate() {
                                let err = (logit[row * w + out] as f64 - expected[component]).abs();
                                let tol = 2e-5 * sums[component] + 1e-12;
                                all_gemm_checks &= err.is_finite() && err <= tol;
                                gemm_error = gemm_error.max(err / tol);
                            }
                        }
                        for ih in 0..h {
                            for ic in 0..half {
                                let ire = ih * 2 * half + ic;
                                let iim = ire + half;
                                let mut expected = [0f64; 2];
                                let mut sums = [0f64; 2];
                                for row in 0..6 {
                                    let xr = input[row * w + ire] as f64;
                                    let xi = input[row * w + iim] as f64;
                                    let dr = dz[row * w + ore] as f64;
                                    let di = dz[row * w + oim] as f64;
                                    expected[0] += dr * xr + di * xi;
                                    expected[1] += di * xr - dr * xi;
                                    sums[0] += (dr * xr).abs() + (di * xi).abs();
                                    sums[1] += (di * xr).abs() + (dr * xi).abs();
                                }
                                let actual = [
                                    g[offset + ore * w + ire] + g[offset + oim * w + iim],
                                    g[offset + oim * w + ire] - g[offset + ore * w + iim],
                                ];
                                for component in 0..2 {
                                    let err = (actual[component] as f64 - expected[component]).abs();
                                    let tol = 2e-5 * sums[component] + 1e-12;
                                    all_gemm_checks &= err.is_finite() && err <= tol;
                                    gemm_error = gemm_error.max(err / tol);
                                }
                            }
                        }
                    }
                }
                // CPU sums over actual input/adjoint tensors test every gate coordinate.
                for out in 0..w {
                    for col in 0..w {
                        let mut expected = 0f64;
                        let mut sum_abs = 0f64;
                        for row in 0..6 {
                            let term = dz[row * w + out] as f64 * input[row * w + col] as f64;
                            expected += term;
                            sum_abs += term.abs();
                        }
                        let err = (g[offset + out * w + col] as f64 - expected).abs();
                        let tol = 0.0012 * sum_abs + 1e-12;
                        all_gemm_checks &= err.is_finite() && err <= tol;
                        gemm_error = gemm_error.max(err / tol);
                    }
                }
                for row in 0..6 {
                    for col in 0..w {
                        let mut expected = 0f64;
                        let mut sum_abs = 0f64;
                        for out in 0..w {
                            let term = dz[row * w + out] as f64 * p[offset + out * w + col] as f64;
                            expected += term;
                            sum_abs += term.abs();
                        }
                        let err = (din[row * w + col] as f64 - expected).abs();
                        let tol = 0.0012 * sum_abs + 1e-12;
                        all_gemm_checks &= err.is_finite() && err <= tol;
                        gemm_error = gemm_error.max(err / tol);
                    }
                }
            }
            let mut directions = Vec::new();
            let mut rng = Random::new(31);
            // Gate directions per head/layer plus the upstream embedding direction
            // check the extra input-adjoint path through the shared normalization.
            let mut blocks = vec![("embedding".to_owned(), 0, c.vocab * w)];
            for layer in 0..l {
                for head in 0..h {
                    blocks.push((
                        format!("gate-{layer}-head-{head}"),
                        layout::gate_offset(&c, layer) + head * (w / h) * w,
                        (w / h) * w,
                    ));
                }
            }
            for (name, start, count) in blocks {
                let mut coordinates = Vec::new();
                for [a, b, nb, aa, kind] in layout::pairs(&c) {
                    if a as usize >= start && (a as usize) < start + count {
                        coordinates.push((a as usize, (kind == 0).then_some((aa as usize, 1.))));
                        coordinates.push((b as usize, (kind == 0).then_some((nb as usize, -1.))));
                    }
                }
                let contracted: Vec<f64> = coordinates.iter().map(|&(at, duplicate)| {
                    g[at] as f64 + duplicate.map_or(0., |(other, sign)| sign * g[other] as f64)
                }).collect();
                let norm = contracted.iter().map(|v| v*v).sum::<f64>().sqrt();
                for mode in 0..3 {
                    let mut direction = vec![0f64; p.len()];
                    let mut independent_direction = vec![0f64; coordinates.len()];
                    if mode == 1 {
                        let idx = (0..contracted.len())
                            .max_by(|&a, &b| contracted[a].abs().total_cmp(&contracted[b].abs()))
                            .unwrap();
                        independent_direction[idx] = 1.;
                    } else {
                        for (i, value) in independent_direction.iter_mut().enumerate() {
                            *value = if mode == 2 {
                                contracted[i]
                            } else {
                                rng.gauss(0., 1.)
                            };
                        }
                        let length = independent_direction.iter().map(|v| v * v).sum::<f64>().sqrt();
                        if length > 0. {
                            for x in &mut independent_direction {
                                *x /= length;
                            }
                        }
                    }
                    for (&(at, duplicate), &value) in coordinates.iter().zip(&independent_direction) {
                        direction[at] = value;
                        if let Some((other, sign)) = duplicate {
                            direction[other] = sign * value;
                        }
                    }
                    let actual = g
                        .iter()
                        .zip(&direction)
                        .map(|(&v, &d)| v as f64 * d)
                        .sum::<f64>();
                    let dual: Vec<Jet> = p
                        .iter()
                        .zip(&direction)
                        .map(|(&v, &d)| Jet { v: v as f64, d })
                        .collect();
                    let expected = oracle::cross_entropy(
                        &oracle::main_model(&c, &dual, &tokens, 2, 3),
                        &labels,
                        c.vocab,
                    )
                    .d;
                    let error = (actual - expected).abs();
                    let tolerance = 5e-4 * norm + 1e-12;
                    directions.push(json!({"block":name,"direction":mode,"native":actual,"reference":expected,"error":error,"tolerance":tolerance,"passed":error.is_finite()&&error<=tolerance}));
                }
            }
            let changed = [1, 2, 9, 4, 3, 8];
            e.forward(&changed, &labels, 2, 3)?;
            let changed_logits = e.logits.read()?;
            let causal = [0, 1, 3, 4].iter().all(|&r| {
                equal(
                    &logits[r * 11..(r + 1) * 11],
                    &changed_logits[r * 11..(r + 1) * 11],
                )
            });
            let prefix_tokens = [1, 2, 4, 3];
            let prefix_labels = [2, 3, 3, 2];
            e.forward(&prefix_tokens, &prefix_labels, 2, 2)?;
            let prefix = e.logits.read()?;
            let prefix_equal = [(0, 0), (1, 1), (2, 3), (3, 4)]
                .iter()
                .all(|&(a, b)| equal(&prefix[a * 11..(a + 1) * 11], &logits[b * 11..(b + 1) * 11]));
            let finite = g.iter().chain(&logits).all(|x| x.is_finite());
            let gate_half = magnitude != 0.
                || before
                    .iter()
                    .filter(|(name, _)| name.ends_with(".gate_values"))
                    .all(|(_, v)| v.iter().all(|x| *x == 0.5));
            records.push(json!({"kind":"model_integration","width":w,"heads":h,"layers":l,"gate_init_magnitude":magnitude,
                "passed":overwrite&&all_gemm_checks&&causal&&prefix_equal&&finite&&gate_half,
                "gradient_overwrite_exact":overwrite,"gate_matrix_and_input_adjoints_passed":all_gemm_checks,"gemm_error_over_bound":gemm_error,
                "causal_future_invariance_exact":causal,"prefix_invariance_exact":prefix_equal,"finite":finite,"zero_weights_gate_half":gate_half,
                "strict_scalar_AD_all_passed":directions.iter().all(|v|v["passed"]==true),"strict_scalar_AD_directions":directions}));
        }
    }
    Ok(())
}

pub fn run(root: &Path) -> Result<()> {
    if root.exists() {
        return Err("gate validation output exists".into());
    }
    fs::create_dir_all(root)?;
    for (name, body) in crate::sources::FILES {
        let path = root.join("source").join(name);
        fs::create_dir_all(path.parent().ok_or("missing parent")?)?;
        fs::write(path, body)?;
    }
    let mut records = Vec::new();
    let result = (|| -> Result<()> {
        primitives(&mut records)?;
        model_checks(&mut records)
    })();
    let strict_misses = records
        .iter()
        .filter_map(|v| v["strict_scalar_AD_directions"].as_array())
        .flatten()
        .filter(|v| v["passed"] != true)
        .count();
    let passed = result.is_ok() && records.iter().all(|v| v["passed"] == true);
    fs::write(
        root.join("report.json"),
        serde_json::to_vec_pretty(
            &json!({"implementation_checks_passed":passed,"strict_scalar_AD_misses":strict_misses,"all_strict_checks_passed":passed&&strict_misses==0,
        "source_sha256":runner::source_hash(),"ptx_sha256":runner::ptx_hash(),"cases":records,"training_started":false,
        "execution_error":result.as_ref().err().map(|e|e.to_string()),"remaining_validation":["optimizer_and_checkpoint_restore","frozen_control_replay","maximum_shapes","sanitizers","classify_scalar_AD_misses_before_training"]}),
        )?,
    )?;
    result?;
    if passed {
        Ok(())
    } else {
        Err("gate implementation checks failed; inspect report.json".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::Optimizer;
    #[test]
    fn complex_predictor_commutes_with_quarter_turn_and_rejects_broken_ties() -> Result<()> {
        for (w,h,l) in [(64,4,2),(128,1,1),(128,4,1)] {
            let c=cfg(w,h,l,true);
            let mut p=layout::initialize_model(&c,19);
            let common=layout::common_count(&c);
            for [a,b,nb,aa,_] in layout::pairs(&c).into_iter().filter(|o|o[0] as usize>=common) {
                assert_eq!(p[a as usize].to_bits(),0);
                assert_eq!(p[b as usize].to_bits(),0);
                assert_eq!(p[nb as usize].to_bits(),(-0f32).to_bits());
                assert_eq!(p[aa as usize].to_bits(),0);
                p[a as usize]=((a%17) as f32-8.)/32.;
                p[b as usize]=((b%13) as f32-6.)/32.;
                p[nb as usize]=-p[b as usize];
                p[aa as usize]=p[a as usize];
            }
            layout::validate(&c,&p)?;
            let half=w/h/2;
            let rotate=|x:&[f64]| {
                let mut out=vec![0.;w];
                for head in 0..h { for k in 0..half {
                    let re=head*2*half+k;
                    out[re]=-x[re+half]; out[re+half]=x[re];
                }}
                out
            };
            let input:Vec<f64>=(0..w).map(|i|((i%13) as f64-6.)/8.).collect();
            for layer in 0..l {
                let offset=layout::gate_offset(&c,layer);
                let apply=|x:&[f64]| (0..w).map(|r|(0..w).map(|col|p[offset+r*w+col] as f64*x[col]).sum::<f64>()).collect::<Vec<_>>();
                // Dyadic inputs make this exact; no learned sigmoid is included
                // in this property of the complex-linear predictor alone.
                assert_eq!(apply(&rotate(&input)),rotate(&apply(&input)));
            }
            let owner=layout::pairs(&c).into_iter().find(|o|o[0] as usize>=common).unwrap();
            let mut broken=p.clone(); broken[owner[2] as usize]+=0.125;
            assert!(layout::validate(&c,&broken).is_err());
            let m=p.clone(); let mut v=vec![0.25;p.len()];
            layout::validate_state(&c,&p,&m,&v)?;
            v[owner[3] as usize]=0.5;
            assert!(layout::validate_state(&c,&p,&m,&v).is_err());
        }
        Ok(())
    }
    #[test]
    fn gate_initialization_and_optimizer_cover_each_real_coordinate_once() -> Result<()> {
        for (w, h, l) in [
            (64, 4, 1),
            (128, 4, 2),
            (128, 4, 6),
            (128, 1, 1),
            (256, 2, 2),
        ] {
            let c = cfg(w, h, l, true);
            let p = layout::initialize_model(&c, 19);
            let old = layout::initialize_model(&cfg(w, h, l, false), 19);
            assert!(equal(&p[..old.len()], &old));
            assert_eq!(p.len() - old.len(), l * w * w);
            assert!(p[old.len()..].iter().all(|x| *x == 0.));
            assert_eq!(layout::gate_independent_count(&c), l*w*w/2);
            assert_eq!(layout::independent_count(&c)-layout::independent_count(&cfg(w,h,l,false)),l*w*w/2);
            let mut seen = vec![0; p.len()];
            let owners = Optimizer::owners(&c)?;
            let old_owners = Optimizer::owners(&cfg(w,h,l,false))?;
            assert_eq!(&owners[..old_owners.len()], old_owners.as_slice());
            assert_eq!(owners.len() * 2, layout::independent_count(&c));
            for [a, b, nb, aa, kind] in owners {
                seen[a as usize] += 1;
                seen[b as usize] += 1;
                if kind == 0 {
                    seen[nb as usize] += 1;
                    seen[aa as usize] += 1;
                }
            }
            assert!(seen.iter().all(|x| *x == 1));
            layout::validate(&c, &p)?;
        }
        Ok(())
    }
}
