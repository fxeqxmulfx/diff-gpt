//! Standalone audit of raw retained inputs/outputs, independent of the runner.
use std::{collections::BTreeMap, error::Error, fs, path::Path};
type Result<T> = std::result::Result<T, Box<dyn Error>>;
fn labels(root: &Path) -> Result<BTreeMap<(usize, String), usize>> {
    let mut out = BTreeMap::new();
    for line in fs::read_to_string(root.join("labels.tsv"))?.lines() {
        let mut parts = line.splitn(3, '\t');
        let width: usize = parts.next().ok_or("missing width")?.parse()?;
        let index: usize = parts.next().ok_or("missing index")?.parse()?;
        let label = parts.next().ok_or("missing label")?.to_owned();
        if out.insert((width, label), index).is_some() {
            return Err("duplicate pair label".into());
        }
    }
    Ok(out)
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 { return Err("usage: check-retained OLD_SCREEN NEW_SCREEN".into()); }
    let old = Path::new(&args[1]);
    let new = Path::new(&args[2]);
    let old_labels = labels(old)?;
    let new_labels = labels(new)?;
    if new_labels.len() != old_labels.len() + 3 { return Err("unexpected new case count".into()); }
    for ((width,label),_) in new_labels.iter().filter(|(key,_)| !old_labels.contains_key(*key)) {
        if label != &format!("product_cancellation/d{width}") { return Err("unexpected extra case".into()); }
    }
    let mut inputs = 0;
    let mut outputs = 0;
    for width in [16,32,128] {
        let a = fs::read(old.join(format!("inputs-d{width}.f32")))?;
        let b = fs::read(new.join(format!("inputs-d{width}.f32")))?;
        let old_n = old_labels.keys().filter(|(d,_)| *d == width).count();
        let new_n = new_labels.keys().filter(|(d,_)| *d == width).count();
        if a.len() != old_n * width * 8 || b.len() != new_n * width * 8 { return Err("input length mismatch".into()); }
        for ((_,label),&i) in old_labels.iter().filter(|((d,_),_)| *d == width) {
            let j = *new_labels.get(&(width,label.clone())).ok_or("retained pair missing")?;
            for part in 0..2 {
                let ai = (part*old_n+i)*width*4;
                let bi = (part*new_n+j)*width*4;
                if a[ai..ai+width*4] != b[bi..bi+width*4] { return Err(format!("changed input: {label}").into()); }
                inputs += width;
            }
        }
        for mode in 0..8 {
            let a = fs::read(old.join(format!("outputs-d{width}-mode{mode}.f64")))?;
            let b = fs::read(new.join(format!("outputs-d{width}-mode{mode}.f64")))?;
            if a.len() != old_n*8 || b.len() != new_n*8 { return Err("output length mismatch".into()); }
            for ((_,label),&i) in old_labels.iter().filter(|((d,_),_)| *d == width) {
                let j = new_labels[&(width,label.clone())];
                if a[i*8..(i+1)*8] != b[j*8..(j+1)*8] { return Err(format!("changed output: {label} mode{mode}").into()); }
                outputs += 1;
            }
        }
    }
    println!("{{\"retained_pairs\":{},\"input_floats_bit_exact\":{inputs},\"old_variant_outputs_bit_exact\":{outputs},\"passed\":true}}", old_labels.len());
    Ok(())
}
