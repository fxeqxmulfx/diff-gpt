//! Reproducible source port for the isolated FP32 engineering candidate.
//! Generates a source fork; never edits the selected or frozen implementations.
use std::{fs, path::Path};
fn function(text: &mut String, name: &str, replacement: &str) {
    let start = text.find(&format!("    fn {name}")).unwrap();
    let brace = start + text[start..].find('{').unwrap();
    let mut depth = 0;
    let mut end = brace;
    for (offset, byte) in text.as_bytes()[brace..].iter().enumerate() {
        if *byte == b'{' { depth += 1; }
        if *byte == b'}' { depth -= 1; }
        if depth == 0 { end = brace + offset + 1; break; }
    }
    text.replace_range(start..end, replacement);
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = Path::new("research/main_model/fp32_native/src");
    fs::create_dir_all(root)?;
    let modules = [
        ("attention", "research/main_model/optimized_native/src/attention.rs"),
        ("ordinary_attention", "research/main_model/optimized_native/src/ordinary_attention.rs"),
        ("mhc", "research/main_model/optimized_native/src/mhc.rs"),
        ("base_ops", "research/complex_geometry_suite/native/src/base_ops.rs"),
        ("stable_loss", "research/complex_geometry_suite/diagnostic/src/stable_loss.rs"),
        ("stable_attention", "research/complex_geometry_suite/diagnostic/src/stable_attention.rs"),
        ("stable_chain", "research/complex_geometry_suite/diagnostic/src/stable_chain.rs"),
        ("centered_attention", "research/complex_geometry_suite/diagnostic/src/centered_attention.rs"),
        ("optimizer_kernels", "research/main_model/wsd_native/src/optimizer_kernels.rs"),
    ];
    for (name, path) in modules {
        let mut text = fs::read_to_string(path)?;
        // All floating-point calculations in this source become binary32.
        text = text.replace("f64", "f32").replace("FP64", "FP32");
        text = text.replace("cvt.rn.f32.f32(", "mov.f32(")
            .replace("cvt.f32.f32(", "mov.f32(");
        // Every old binary64 global buffer uses 4-byte elements now.
        if name == "mhc" || name == "centered_attention" {
            text = text.replace("mad.wide.u32(a, i, 8, p)", "mad.wide.u32(a, i, 4, p)");
        }
        if name == "optimizer_kernels" {
            text = text.replace("mad.wide.u32(p, b, 8, out)", "mad.wide.u32(p, b, 4, out)")
                .replace("mad.wide.u32(p, i, 8, parts)", "mad.wide.u32(p, i, 4, parts)");
        }
        if name == "stable_chain" {
            text = text.replace("mul.lo.u32(sat, at, 8)", "mul.lo.u32(sat, at, 4)")
                .replace("mad.lo.u32(sat, col, 8, smem)", "mad.lo.u32(sat, col, 4, smem)")
                .replace("mad.lo.u32(sat, k, 8, smem)", "mad.lo.u32(sat, k, 4, smem)");
        }
        let sums: &[(&str, bool)] = match name {
            "attention" => &[("sum64", false), ("norm_sum64", true)],
            "ordinary_attention" => &[("sum", true)],
            "base_ops" | "stable_loss" | "optimizer_kernels" | "centered_attention" => &[("sum64", false)],
            "stable_attention" | "stable_chain" => &[("sum", false)],
            _ => &[],
        };
        for (helper, generic) in sums {
            let suffix = if *generic { "<const STEPS: usize>" } else { "" };
            let steps = if *generic { "STEPS" } else { "5" };
            function(&mut text, helper, &format!(
                "    fn {helper}{suffix}(v: f32) {{\n        let x: f32;\n        for j in 0..{steps} {{\n            shfl.sync.bfly.b32(x, v, 1 << j, 31, 0xffffffff);\n            add.rn.f32(v, v, x);\n        }}\n    }}"));
        }
        if name == "mhc" {
            text = text.replace("// Range reduction and twelve Taylor terms. Arguments here are <=0.\n    // Below -700 the discarded mass is <1e-304, far below FP32 storage.",
                "// Native fixed FP32 exponential for nonpositive arguments.");
            function(&mut text, "shuffle", "    fn shuffle(v: f32, shift: u32) {\n        let x: f32;\n        shfl.sync.bfly.b32(x, v, shift, 31, 0xffffffff);\n        mov.f32(v, x);\n    }");
            function(&mut text, "exp64", "    fn exp64(y: f32, x: f32) {\n        let z: f32;\n        mul.rn.f32(z, x, 1.4426950408889634f32);\n        ex2.approx.f32(y, z);\n    }");
            function(&mut text, "log64", "    fn log64(out: f32, x: f32) {\n        lg2.approx.f32(out, x);\n        mul.rn.f32(out, out, 0.6931471805599453f32);\n    }");
        }
        if name == "stable_loss" {
            function(&mut text, "log_positive", "    fn log_positive(out: f32, x: f32) {\n        lg2.approx.f32(out, x);\n        mul.rn.f32(out, out, 0.6931471805599453f32);\n    }");
        }
        assert!(!text.contains(".f64"));
        fs::write(root.join(format!("{name}.rs")), text)?;
    }
    for (name, path) in [
        ("engine", "research/main_model/optimized_native/src/engine.rs"),
        ("engine_mhc", "research/main_model/optimized_native/src/engine_mhc.rs"),
        ("optimizer", "research/main_model/wsd_suite_native/src/optimizer.rs"),
    ] {
        let text = fs::read_to_string(path)?.replace("Buffer<f64>", "Buffer<f32>");
        fs::write(root.join(format!("{name}.rs")), text)?;
    }
    Ok(())
}
