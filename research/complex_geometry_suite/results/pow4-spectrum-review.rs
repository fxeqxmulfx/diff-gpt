// Standalone FP64 arithmetic check; no training or supplied model features.
// For K(a,b)=g(a+b mod p), singular values are |DFT(g)|: reversing one input
// converts K into a circulant matrix. The four-character construction below
// is constant on fourth-power fibers, hence defines a readout of the label.
use std::f64::consts::TAU;
const P: usize = 97;
fn fourth(x: usize) -> usize {
    let s = x * x % P;
    s * s % P
}
fn character(k: usize, x: usize) -> (f64, f64) {
    let angle = TAU * ((k * x) % P) as f64 / P as f64;
    (angle.cos(), angle.sin())
}
fn spectrum(g: &[(f64, f64)]) -> Vec<f64> {
    (0..P)
        .map(|k| {
            let (mut re, mut im) = (0., 0.);
            for (s, &(a, b)) in g.iter().enumerate() {
                let (c, d) = character(k, s);
                re += a * c + b * d;
                im += b * c - a * d;
            }
            re.hypot(im)
        })
        .collect()
}
fn effective_rank(s: &[f64]) -> f64 {
    s.iter().sum::<f64>().powi(2) / s.iter().map(|x| x * x).sum::<f64>()
}
fn main() {
    let roots: Vec<_> = (0..P).filter(|&r| fourth(r) == 1).collect();
    assert_eq!(roots, vec![1, 22, 75, 96]);
    let g: Vec<_> = (0..P)
        .map(|s| {
            roots.iter().fold((0., 0.), |(a, b), &r| {
                let (c, d) = character(r, s);
                (a + c, b + d)
            })
        })
        .collect();
    let mut labels: [Option<(f64, f64)>; P] = [None; P];
    let mut counts = [0usize; P];
    let mut fiber_error = 0f64;
    for (s, &(re, im)) in g.iter().enumerate() {
        let t = fourth(s);
        counts[t] += 1;
        if let Some((a, b)) = labels[t] {
            fiber_error = fiber_error.max((re - a).hypot(im - b));
        } else {
            labels[t] = Some((re, im));
        }
    }
    assert!(fiber_error < 1e-11);
    assert_eq!(counts.iter().filter(|&&x| x > 0).count(), 25);
    assert_eq!(counts[0], 1);
    assert!(counts[1..].iter().all(|&x| x == 0 || x == 4));
    let special = spectrum(&g);
    let mut dft_error = 0f64;
    for (k, &v) in special.iter().enumerate() {
        dft_error = dft_error.max((v - if roots.contains(&k) { P as f64 } else { 0. }).abs());
    }
    assert!(dft_error < 1e-10);
    let ordinary: Vec<_> = (0..P).map(|s| character(1, fourth(s))).collect();
    let mut sv = spectrum(&ordinary);
    sv.sort_by(f64::total_cmp);
    let rank = effective_rank(&special);
    assert!((rank - 4.).abs() < 1e-10);
    println!(
        "{{\"prime\":{P},\"fourth_roots_of_one\":{roots:?},\"reachable_labels\":25,\"four_character_readout\":{{\"effective_rank\":{rank},\"max_fiber_error\":{fiber_error},\"max_dft_error\":{dft_error},\"singular_values\":{special:?}}},\"fixed_character_of_fourth_power\":{{\"effective_rank\":{},\"minimum_singular_value\":{},\"maximum_singular_value\":{},\"singular_values_sorted\":{sv:?}}},\"passed\":true,\"training_performed\":false}}",
        effective_rank(&sv),
        sv[0],
        sv[P - 1]
    );
}
