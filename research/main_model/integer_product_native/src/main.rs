//! Exact-product representation validation; no model updates.
#[path = "kernels.rs"]
mod product;
use kernite_cuda::{Cuda, Grid};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    error::Error,
    fs::{self, File},
    io::{BufWriter, Write},
    path::Path,
    time::Instant,
};
type Result<T> = std::result::Result<T, Box<dyn Error>>;
fn hash(v: &[u8]) -> String {
    Sha256::digest(v)
        .iter()
        .map(|x| format!("{x:02x}"))
        .collect()
}
fn pairs() -> (Vec<f32>, Vec<f32>) {
    let mut a = vec![];
    let mut b = vec![];
    let mut push = |x: u32, y: u32| {
        a.push(f32::from_bits(x));
        b.push(f32::from_bits(y));
    };
    for ea in 0..255u32 {
        for eb in [0, 1, 126, 127, 128, 253, 254] {
            for ma in [0, 1, 0x400000, 0x7ffffe, 0x7fffff] {
                for mb in [0, 1, 0x400000, 0x7ffffe, 0x7fffff] {
                    for sa in [0, 0x80000000] {
                        for sb in [0, 0x80000000] {
                            push(sa | (ea << 23) | ma, sb | (eb << 23) | mb);
                        }
                    }
                }
            }
        }
    }
    let exceptional = [
        0, 0x80000000, 1, 0x80000001, 0x3f800000, 0xbf800000, 0x7f800000, 0xff800000, 0x7fc00001,
        0x7f800001,
    ];
    for x in exceptional {
        for y in exceptional {
            push(x, y);
        }
    }
    let mut r = 0x1234567887654321u64;
    for _ in 0..1_000_003 {
        r ^= r << 13;
        r ^= r >> 7;
        r ^= r << 17;
        let x = r as u32;
        r ^= r << 13;
        r ^= r >> 7;
        r ^= r << 17;
        push(x, r as u32);
    }
    (a, b)
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 2 {
        return Err("usage: exact-wide-product FRESH_ROOT".into());
    }
    let root = Path::new(&args[1]);
    if root.exists() {
        return Err("output exists".into());
    }
    fs::create_dir_all(root)?;
    for (name, body) in [
        ("main.rs", include_str!("main.rs")),
        ("kernels.rs", include_str!("kernels.rs")),
        ("Cargo.toml", include_str!("../Cargo.toml")),
        ("Cargo.lock", include_str!("../Cargo.lock")),
        ("PLAN.txt", include_str!("../../INTEGER_PRODUCT_PLAN.txt")),
    ] {
        fs::write(root.join(name), body)?;
    }
    let (a, b) = pairs();
    let cuda = Cuda::new()?;
    let k = product::kernels::Kernels::load(&cuda)?;
    let aa = cuda.upload(&a)?;
    let bb = cuda.upload(&b)?;
    let old = cuda.alloc::<f64>(a.len())?;
    let new = cuda.alloc::<f64>(a.len())?;
    let grid = Grid::new(a.len().div_ceil(256) as u32, 256);
    unsafe {
        k.products::<false>(grid, old.ptr(), aa.ptr(), bb.ptr(), a.len() as u32)?;
        k.products::<true>(grid, new.ptr(), aa.ptr(), bb.ptr(), a.len() as u32)?;
    }
    let reference = old.read()?;
    let actual = new.read()?;
    let mut failures = vec![];
    let mut finite_count = 0;
    let mut nonfinite_count = 0;
    for i in 0..a.len() {
        let cpu = a[i] as f64 * b[i] as f64;
        let matches = |x: f64, y: f64| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan());
        if a[i].is_finite() && b[i].is_finite() {
            finite_count += 1;
        } else {
            nonfinite_count += 1;
        }
        if !matches(actual[i], reference[i]) || !matches(actual[i], cpu) {
            failures.push(json!({"index":i,"a_bits":a[i].to_bits(),"b_bits":b[i].to_bits(),"actual_bits":actual[i].to_bits(),"gpu_reference_bits":reference[i].to_bits(),"cpu_bits":cpu.to_bits()}));
        }
    }
    let mut timed = vec![];
    for repetition in 0..3 {
        for integer in if repetition % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        } {
            for _ in 0..3 {
                unsafe {
                    if integer {
                        k.products::<true>(grid, new.ptr(), aa.ptr(), bb.ptr(), a.len() as u32)?;
                    } else {
                        k.products::<false>(grid, old.ptr(), aa.ptr(), bb.ptr(), a.len() as u32)?;
                    }
                }
            }
            cuda.sync()?;
            let started = Instant::now();
            for _ in 0..100 {
                unsafe {
                    if integer {
                        k.products::<true>(grid, new.ptr(), aa.ptr(), bb.ptr(), a.len() as u32)?;
                    } else {
                        k.products::<false>(grid, old.ptr(), aa.ptr(), bb.ptr(), a.len() as u32)?;
                    }
                }
            }
            cuda.sync()?;
            timed.push(json!({"integer":integer,"repetition":repetition,"milliseconds_per_launch":started.elapsed().as_secs_f64()*10.}));
        }
    }
    let mut input = BufWriter::new(File::create_new(root.join("inputs.f32"))?);
    for v in a.iter().chain(&b) {
        input.write_all(&v.to_le_bytes())?;
    }
    input.flush()?;
    let mut output = BufWriter::new(File::create_new(root.join("outputs.f64"))?);
    for v in &actual {
        output.write_all(&v.to_le_bytes())?;
    }
    output.flush()?;
    let report = json!({"pairs":a.len(),"finite_pairs_bitwise_compared":finite_count,"nonfinite_class_and_infinity_sign_compared":nonfinite_count,"mismatches":failures.len(),"first_mismatches":failures.iter().take(16).collect::<Vec<_>>(),"gpu":cuda.name(),"sm":cuda.sm(),"ptx_sha256":hash(product::kernels::PTX.as_bytes()),"executable_sha256":hash(&fs::read(std::env::current_exe()?)?),"timings":timed,"timing_scope":"elementwise_product_only_including_input_output_device_memory_not_model_training","no_runtime_precision_selection":true,"passed":failures.is_empty()});
    fs::write(
        root.join("report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    fs::write(root.join("kernels.ptx"), product::kernels::PTX)?;
    println!("{report}");
    if !failures.is_empty() {
        return Err("exact product mismatch".into());
    }
    Ok(())
}
