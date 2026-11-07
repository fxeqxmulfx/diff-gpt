use serde_json::{Value, json};

pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
pub fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}
pub fn cosine(a: &[f64], b: &[f64]) -> Option<f64> {
    let den = norm(a) * norm(b);
    (den > 0.).then(|| dot(a, b) / den)
}
pub fn summary(x: &[f64]) -> Value {
    let mut finite: Vec<_> = x.iter().copied().filter(|v| v.is_finite()).collect();
    finite.sort_by(f64::total_cmp);
    if finite.is_empty() {
        return json!({"count":x.len(),"finite":0});
    }
    let at = |q: f64| finite[((finite.len() - 1) as f64 * q).round() as usize];
    json!({"count":x.len(),"finite":finite.len(),"mean":finite.iter().sum::<f64>()/finite.len() as f64,
        "rms":(dot(&finite,&finite)/finite.len() as f64).sqrt(),"l2":norm(&finite),
        "min":at(0.),"p01":at(0.01),"p05":at(0.05),"median":at(0.5),"p95":at(0.95),"p99":at(0.99),"max":at(1.)})
}

// Singular-value energy participation rank = (sum s^2)^2/sum s^4.
// This is a continuous concentration statistic, not a numerical rank cutoff.
pub fn matrix(x: &[f64], rows: usize, cols: usize) -> Value {
    assert_eq!(x.len(), rows * cols);
    let n = rows.min(cols);
    let mut gram = vec![0.; n * n];
    for i in 0..n {
        for j in 0..=i {
            let v = if rows <= cols {
                dot(&x[i * cols..(i + 1) * cols], &x[j * cols..(j + 1) * cols])
            } else {
                (0..rows).map(|r| x[r * cols + i] * x[r * cols + j]).sum()
            };
            gram[i * n + j] = v;
            gram[j * n + i] = v;
        }
    }
    let energy = dot(x, x);
    let mut max_eigen: f64 = 0.;
    for seed in 1..=3 {
        let mut v: Vec<_> = (0..n).map(|i| ((i * seed + 1) as f64).sin()).collect();
        let length = norm(&v);
        v.iter_mut().for_each(|x| *x /= length);
        for _ in 0..64 {
            let mut next: Vec<_> = gram.chunks_exact(n).map(|row| dot(row, &v)).collect();
            let length = norm(&next);
            if length == 0. {
                break;
            }
            next.iter_mut().for_each(|x| *x /= length);
            v = next;
        }
        let av: Vec<_> = gram.chunks_exact(n).map(|row| dot(row, &v)).collect();
        max_eigen = max_eigen.max(dot(&v, &av));
    }
    let row_norms: Vec<_> = x.chunks_exact(cols).map(norm).collect();
    json!({"entries":summary(x),"row_norms":summary(&row_norms),"frobenius":energy.sqrt(),
        "spectral_norm_estimate":max_eigen.sqrt(),"stable_rank_estimate":if max_eigen>0.{Some(energy/max_eigen)}else{None},
        "singular_energy_participation":if energy>0.{Some(energy*energy/dot(&gram,&gram))}else{None}})
}

#[test]
fn matrix_spectrum_distinguishes_rank_one_and_full_rank() {
    let identity = matrix(&[1., 0., 0., 1.], 2, 2);
    assert!((identity["stable_rank_estimate"].as_f64().unwrap() - 2.).abs() < 1e-12);
    assert_eq!(identity["singular_energy_participation"], 2.);
    let rank_one = matrix(&[1., 2., 2., 4.], 2, 2);
    assert!((rank_one["spectral_norm_estimate"].as_f64().unwrap() - 5.).abs() < 1e-12);
    assert!((rank_one["stable_rank_estimate"].as_f64().unwrap() - 1.).abs() < 1e-12);
}
