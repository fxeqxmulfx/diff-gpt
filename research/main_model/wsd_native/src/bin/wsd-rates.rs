//! Read-only diagnosis of JSON float round trips in frozen WSD histories.
use serde_json::{Value, json};
use std::{env, fs};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
#[path = "../schedule.rs"]
mod schedule;
fn main() -> Result<()> {
    let path = env::args().nth(1).ok_or("history path required")?;
    let schedule = schedule::Schedule::parse(schedule::ARM)?;
    let mut observations = 0usize;
    let mut mismatches = Vec::new();
    let mut wrong_f32 = 0usize;
    let mut wrong_std_parse = 0usize;
    let mut max_f64_ulps = 0u64;
    for line in fs::read_to_string(path)?.lines() {
        let v: Value = serde_json::from_str(line)?;
        let step = v["step"].as_u64().ok_or("step")? as usize;
        let expected = schedule.rate(0.002, step);
        for field in ["next_embedding_lr", "next_matrix_lr"] {
            let parsed = v[field].as_f64().ok_or("rate")?;
            let prefix = format!("\"{field}\":");
            let raw = line.split_once(&prefix).ok_or("raw rate")?.1
                .split([',', '}']).next().ok_or("raw rate")?.trim();
            let std_parsed: f64 = raw.parse()?;
            let ulps = parsed.to_bits().abs_diff((expected as f64).to_bits());
            max_f64_ulps = max_f64_ulps.max(ulps);
            wrong_f32 += usize::from((parsed as f32).to_bits() != expected.to_bits());
            wrong_std_parse += usize::from(std_parsed != expected as f64);
            if parsed != expected as f64 && mismatches.len() < 5 {
                mismatches.push(json!({"step":step,"field":field,"raw":raw,"parsed":parsed,
                    "expected_f32_promoted_to_f64":expected as f64,"f64_ulps":ulps,
                    "std_parse_exact":std_parsed==expected as f64,
                    "f32_bits_exact":(parsed as f32).to_bits()==expected.to_bits()}));
            }
        }
        observations += 1;
    }
    println!("{}",json!({"schedule":schedule.name(),"observations":observations,
        "wrong_f32_values":wrong_f32,"wrong_standard_parser_values":wrong_std_parse,
        "max_serde_f64_ulps":max_f64_ulps,"first_mismatches":mismatches}));
    if wrong_f32 != 0 || wrong_std_parse != 0 { return Err("actual schedule mismatch".into()); }
    Ok(())
}
