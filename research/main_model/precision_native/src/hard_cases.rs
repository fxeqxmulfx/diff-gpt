//! Hard cancellation and a raw-adjoint range case with a finite RMS control.
use super::{Pair, push};
use crate::{Result, attention, random::Random};
use kernite_cuda::{Cuda, Grid};
use serde_json::{Value, json};

pub(super) fn append(pairs: &mut Vec<Pair>) {
    let mut rng = Random::new(7318301);
    for d in [16, 32, 128] {
        for case in 0..1024 {
            let mut a: Vec<f32> = (0..d / 2).map(|_| rng.gauss(0., 1.) as f32).collect();
            let mut b: Vec<f32> = (0..d / 2).map(|_| rng.gauss(0., 1.) as f32).collect();
            if case % 2 == 1 {
                for x in a.iter_mut().chain(&mut b) {
                    *x *= 2f32.powi(-(rng.below(80) as i32));
                }
            }
            for v in [&mut a, &mut b] {
                let rms =
                    (v.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / (d / 2) as f64).sqrt();
                for x in v {
                    *x = (*x as f64 / rms * 0.999) as f32;
                }
            }
            let permutation = rng.sample(d, d);
            let mut x = vec![0.; d];
            let mut y = vec![0.; d];
            for i in 0..d / 2 {
                x[permutation[2 * i]] = a[i];
                x[permutation[2 * i + 1]] = a[i];
                y[permutation[2 * i]] = b[i];
                y[permutation[2 * i + 1]] = -b[i];
            }
            if case % 4 == 0 {
                let i = permutation[0];
                y[i] = f32::from_bits(y[i].to_bits() + 1);
            }
            assert!(x.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() < d as f64);
            assert!(y.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() < d as f64);
            push(
                pairs,
                format!("adversarial_cancellation/d{d}/{case}"),
                "bounded_adversarial",
                &x,
                &y,
            );
        }
        let x = vec![2f32.powi(60); d];
        let g: Vec<_> = (0..d)
            .map(|i| {
                if i < d / 2 {
                    2f32.powi(80)
                } else {
                    -2f32.powi(80)
                }
            })
            .collect();
        push(
            pairs,
            format!("finite_norm_high_gain/d{d}"),
            "finite_norm_raw_backward_dot",
            &x,
            &g,
        );
    }
}

pub(super) fn norm_controls(cuda: &Cuda) -> Result<Vec<Value>> {
    let k = attention::kernels::Kernels::load(cuda)?;
    let mut reports = vec![];
    for d in [16, 32, 128] {
        let heads = 4;
        let width = heads * d;
        let mut x = vec![0f32; 3 * width];
        let mut g = x.clone();
        for i in 0..2 * width {
            x[i] = 2f32.powi(60);
            g[i] = if i % d < d / 2 {
                2f32.powi(80)
            } else {
                -2f32.powi(80)
            };
        }
        let a = cuda.upload(&x)?;
        let b = cuda.upload(&g)?;
        let y = cuda.alloc::<f32>(x.len())?;
        let dx = cuda.alloc::<f32>(x.len())?;
        macro_rules! run {
            ($d:literal) => {
                unsafe {
                    k.norm::<$d>(
                        Grid::new((3 * heads) as u32, 32),
                        y.ptr(),
                        a.ptr(),
                        heads as u32,
                    )?;
                    k.norm_backward::<$d>(
                        Grid::new(
                            (if $d == 16 { 3 * heads / 2 } else { 3 * heads }) as u32,
                            32,
                        ),
                        dx.ptr(),
                        a.ptr(),
                        b.ptr(),
                        heads as u32,
                    )?;
                }
            };
        }
        match d {
            16 => run!(16),
            32 => run!(32),
            128 => run!(128),
            _ => unreachable!(),
        }
        let values = y.read()?;
        let gradients = dx.read()?;
        let inv = 1. / (2f64.powi(120) + 2f64.powi(-23)).sqrt();
        let mut forward_error = 0f64;
        let mut backward_relative = 0f64;
        for i in 0..3 * width {
            let want = if i < 2 * width {
                2f64.powi(60) * inv
            } else {
                0.
            };
            let want_grad = g[i] as f64 * inv;
            forward_error = forward_error.max((values[i] as f64 - want).abs());
            backward_relative = backward_relative
                .max((gradients[i] as f64 - want_grad).abs() / want_grad.abs().max(1.));
        }
        let finite = values.iter().chain(&gradients).all(|x| x.is_finite());
        let passed = finite && forward_error <= 2e-5 && backward_relative <= 5e-4;
        let report = json!({"head_width":d,"raw_input":2f64.powi(60),"upstream_magnitude":2f64.powi(80),"raw_product_magnitude":2f64.powi(140),"exact_dot":0,"forward_energy_sum":d as f64*2f64.powi(120),"forward_max_error":forward_error,"backward_relative_error":backward_relative,"finite":finite,"passed":passed,"oracle":"constant_x_and_zero_mean_tangent_g_implies_dx=g/sqrt(x_squared+epsilon)"});
        if !passed {
            return Err(format!("RMS high-gain baseline control failed: {report}").into());
        }
        reports.push(report);
    }
    Ok(reports)
}
