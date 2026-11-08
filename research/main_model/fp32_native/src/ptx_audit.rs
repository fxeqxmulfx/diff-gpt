//! Assert the exact generated model/optimizer modules contain no FP64 arithmetic.
use crate::{Result, hash, runner};
use serde_json::json;
use std::{fs, path::Path};
pub fn run(root: &Path) -> Result<()> {
    if root.exists() {
        return Err("PTX audit output exists".into());
    }
    fs::create_dir_all(root)?;
    let mut rows = Vec::new();
    for (name, ptx) in [
        ("attention", crate::attention::kernels::PTX),
        ("base_ops", crate::base_ops::kernels::PTX),
        ("sequence_ops", crate::sequence_ops::kernels::PTX),
        ("complex", crate::complex::kernels::PTX),
        ("gemm", crate::gemm::kernels::PTX),
        ("optimizer", crate::optimizer_kernels::kernels::PTX),
        ("mhc", crate::mhc::kernels::PTX),
        (
            "ordinary_attention",
            crate::ordinary_attention::kernels::PTX,
        ),
        ("plain_product", crate::plain_product::kernels::PTX),
        ("stable_loss", crate::stable_loss::kernels::PTX),
        ("stable_attention", crate::stable_attention::kernels::PTX),
        ("stable_chain", crate::stable_chain::kernels::PTX),
        (
            "centered_attention",
            crate::centered_attention::kernels::PTX,
        ),
    ] {
        let fp64_lines: Vec<_> = ptx.lines().filter(|line| line.contains(".f64")).collect();
        fs::write(root.join(format!("{name}.ptx")), ptx)?;
        rows.push(json!({"module":name,"sha256":hash(ptx.as_bytes()),"fp64_lines":fp64_lines}));
    }
    let passed = rows
        .iter()
        .all(|r| r["fp64_lines"].as_array().unwrap().is_empty());
    fs::write(
        root.join("report.json"),
        serde_json::to_vec_pretty(&json!({
            "passed":passed,"criterion":"no .f64 registers/instructions in every generated model/optimizer module",
            "modules":rows,"ptx_sha256":runner::ptx_hash(),"source_sha256":runner::source_hash(),
            "host_schedule_and_independent_cpu_references_may_use_fp64":true
        }))?,
    )?;
    if passed {
        Ok(())
    } else {
        Err("FP64 remains in generated model PTX".into())
    }
}
