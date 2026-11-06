"""Audit the complete diagnostic grid and export every component measurement."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path

from research.early99_reporting.round2_audit import require, verify_hash

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).parent
RESULTS = STUDY / "results"
PATH_TOLERANCE = 1e-5
STATES = [
    ("initial", "original"),
    ("initial", "normalized"),
    ("final", "original_lr1"),
    ("final", "original_lr2"),
    ("final", "normalized_lr2"),
]


def digest(path: Path) -> str:
    """Hash an immutable input or generated evidence file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(values: list) -> dict:
    """Retain the complete count, median and range without dropping undefined values."""
    require(
        values and all(value is not None for value in values),
        "Undefined diagnostic statistic",
    )
    return {
        "count": len(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def write_csv(name: str, rows: list) -> Path:
    """Export a rectangular table without dropping any declared observation."""
    path = RESULTS / name
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def audit(data: dict, plan: dict) -> dict:
    """Verify paired inputs, observer coverage and each recorded path reconstruction."""
    rows = data["rows"]
    expected = {
        (stage, arm, seed, batch)
        for stage, arm in STATES
        for seed in plan["seeds"]
        for batch in range(plan["training_batches"]["count"])
    }
    require(
        len(rows) == len(expected)
        and {(r["stage"], r["arm"], r["seed"], r["batch"]) for r in rows} == expected,
        "Incomplete or duplicate diagnostic grid",
    )
    require(
        data["optimizer_updates"] == 0
        and data["new_heldout_scores"] is False
        and data["selection_uses_this_diagnostic"] is False,
        "Diagnostic changed experimental scope",
    )
    batch_pairs = {}
    errors = []
    common = [
        "token_embedding_table.lookup.output",
        "token_embedding_table.output",
        "ln_f.input",
        "ln_f.output",
        "lm_head.input",
        "lm_head.output",
    ]
    common += [
        "blocks.0." + suffix
        for suffix in [
            "input",
            "output",
            "ln_1.input",
            "ln_1.output",
            "ln_2.input",
            "ln_2.output",
            "sa.c_attn.output",
            "sa.q_norm.input",
            "sa.q_norm.output",
            "sa.k_norm.input",
            "sa.k_norm.output",
            "sa.rope.q.output",
            "sa.rope.k.output",
            "sa.attention.input0",
            "sa.attention.input1",
            "sa.attention.input2",
            "sa.attention.input3",
            "sa.attention.output",
            "sa.softmax.input0",
            "sa.softmax.output",
            "sa.xsa.input0",
            "sa.xsa.input1",
            "sa.xsa.input2",
            "sa.xsa.output",
            "sa.c_proj.linear.output",
            "sa.c_proj.output",
            "ffwd.c_fc.output",
            "ffwd.silu.input0",
            "ffwd.silu.output",
            "ffwd.c_proj.input",
            "ffwd.c_proj.linear.output",
            "ffwd.c_proj.output",
        ]
    ]
    for row in rows:
        require(
            row["exact_forward_and_all_parameter_gradient_parity"]
            and row["unchanged_model_and_rng"],
            "Observer neutrality not checked",
        )
        paired = [row["batch_sha256"], row["targets_sha256"], row["probe_sha256"]]
        require(
            batch_pairs.setdefault(row["batch"], paired) == paired,
            "Unpaired data, labels or probe",
        )
        require(
            set(common) <= set(row["executed_sites"]), "Missing transformer component"
        )
        if row["arm"].startswith("normalized"):
            for prefix in [
                "token_embedding_table.norm",
                "blocks.0.sa.c_proj.norm",
                "blocks.0.ffwd.c_proj.norm",
            ]:
                require(
                    prefix + ".input" in row["executed_sites"]
                    and prefix + ".output" in row["executed_sites"],
                    "Missing added normalization",
                )
        for objective in row["objectives"].values():
            for site, values in objective["sites"].items():
                activation = values["activation"]
                if site.endswith(".softmax.input0"):
                    batch, heads, queries, keys = activation["shape"]
                    require(
                        queries == keys
                        and activation["nonfinite"]
                        == batch * heads * queries * (queries - 1) // 2,
                        "Unexpected attention mask/nonfinite entries",
                    )
                else:
                    require(
                        activation["nonfinite"] == 0, f"Nonfinite activation: {site}"
                    )
                if values["gradient"] is None:
                    require(
                        site.endswith(".attention.input3"),
                        f"Missing active gradient: {site}",
                    )
                else:
                    require(
                        values["gradient"]["nonfinite"] == 0,
                        f"Nonfinite gradient: {site}",
                    )
            for parameter in objective["parameters"].values():
                require(
                    parameter["weight"]["nonfinite"]
                    == parameter["gradient"]["nonfinite"]
                    == 0,
                    "Nonfinite parameter/gradient",
                )
            errors.extend(
                path["gradient_paths"]["reconstruction_relative_l2"]
                for path in objective["residual_paths"]
            )
            errors.append(objective["tied_weight_paths"]["reconstruction_relative_l2"])
    require(max(errors) <= PATH_TOLERANCE, "A gradient path did not reconstruct")
    return {
        "state_batch_measurements": len(rows),
        "objectives": sum(len(r["objectives"]) for r in rows),
        "gradient_decompositions": len(errors),
        "max_path_reconstruction_relative_l2": max(errors),
        "all_required_components_present": True,
        "all_gradients_and_unmasked_activations_finite": True,
        "paired_batches_targets_and_probes": True,
    }


def main() -> None:
    """Write complete CSVs, per-seed summaries and paired comparisons."""
    data_path = RESULTS / "measurements.json"
    data = json.loads(data_path.read_text())
    plan = json.loads((STUDY / "plan.json").read_text())
    validation = json.loads((RESULTS / "validation.json").read_text())
    verify_hash(STUDY / "plan.json", data["plan_sha256"])
    verify_hash(RESULTS / "validation.json", data["validation_sha256"])
    for collection in [
        plan["existing_artifact_sha256"],
        plan["existing_source_sha256"],
        validation["files_sha256"],
    ]:
        for path, expected in collection.items():
            verify_hash(ROOT / path, expected)
    checked = audit(data, plan)
    sites, parameters, residuals, ties, attention = [], [], [], [], []
    for row in data["rows"]:
        identity = {key: row[key] for key in ["stage", "arm", "seed", "batch"]}
        attention.extend({**identity, **metrics} for metrics in row["attention"])
        for objective, values in row["objectives"].items():
            key = {**identity, "objective": objective}
            for site, metrics in values["sites"].items():
                sites.append(
                    {
                        **key,
                        "site": site,
                        **{
                            f"activation_{k}": v
                            for k, v in metrics["activation"].items()
                            if k != "shape"
                        },
                        **{
                            f"gradient_{k}": metrics["gradient"][k]
                            if metrics["gradient"]
                            else None
                            for k in [
                                "rms",
                                "mean",
                                "abs_max",
                                "zero_fraction",
                                "nonfinite",
                            ]
                        },
                    }
                )
            for name, metrics in values["parameters"].items():
                parameters.append(
                    {
                        **key,
                        "parameter": name,
                        "weight_rms": metrics["weight"]["rms"],
                        "gradient_rms": metrics["gradient"]["rms"],
                        "gradient_to_weight_l2": metrics["gradient_to_weight_l2"],
                    }
                )
            residuals.extend(
                {
                    **key,
                    **{k: v for k, v in path.items() if k != "gradient_paths"},
                    **path["gradient_paths"],
                }
                for path in values["residual_paths"]
            )
            ties.append({**key, **values["tied_weight_paths"]})
    files = [
        write_csv(name, rows)
        for name, rows in [
            ("components.csv", sites),
            ("parameters.csv", parameters),
            ("residual-paths.csv", residuals),
            ("tied-weight-paths.csv", ties),
            ("attention.csv", attention),
        ]
    ]
    summaries = []
    for stage, arm in STATES:
        for seed in [None, *plan["seeds"]]:
            rows = [
                r
                for r in data["rows"]
                if r["stage"] == stage
                and r["arm"] == arm
                and (seed is None or r["seed"] == seed)
            ]
            obj = "unit_logit_probe"
            summaries.append(
                {
                    "stage": stage,
                    "arm": arm,
                    "seed": seed,
                    "batches": len(rows),
                    "task_loss": distribution(
                        [r["objectives"]["task_loss"]["value"] for r in rows]
                    ),
                    "unit_probe_embedding_gradient_rms": distribution(
                        [
                            r["objectives"][obj]["sites"][
                                "token_embedding_table.lookup.output"
                            ]["gradient"]["rms"]
                            for r in rows
                        ]
                    ),
                    "attention_normalized_entropy": distribution(
                        [
                            a["normalized_entropy_mean_nonsingleton"]
                            for r in rows
                            for a in r["attention"]
                        ]
                    ),
                    "unit_probe_q_gradient_rms": distribution(
                        [
                            r["objectives"][obj]["parameters"][
                                "blocks.0.sa.c_attn.weight.q"
                            ]["gradient"]["rms"]
                            for r in rows
                        ]
                    ),
                    "unit_probe_k_gradient_rms": distribution(
                        [
                            r["objectives"][obj]["parameters"][
                                "blocks.0.sa.c_attn.weight.k"
                            ]["gradient"]["rms"]
                            for r in rows
                        ]
                    ),
                    "unit_probe_v_gradient_rms": distribution(
                        [
                            r["objectives"][obj]["parameters"][
                                "blocks.0.sa.c_attn.weight.v"
                            ]["gradient"]["rms"]
                            for r in rows
                        ]
                    ),
                    "unit_probe_attention_branch_identity_ratio": distribution(
                        [
                            r["objectives"][obj]["residual_paths"][0]["gradient_paths"][
                                "second_to_first_l2"
                            ]
                            for r in rows
                        ]
                    ),
                    "unit_probe_feedforward_branch_identity_ratio": distribution(
                        [
                            r["objectives"][obj]["residual_paths"][1]["gradient_paths"][
                                "second_to_first_l2"
                            ]
                            for r in rows
                        ]
                    ),
                    "unit_probe_tied_path_cosine": distribution(
                        [
                            r["objectives"][obj]["tied_weight_paths"]["cosine"]
                            for r in rows
                        ]
                    ),
                }
            )
    paired = []
    for stage, control, candidate in [
        ("initial", "original", "normalized"),
        ("final", "original_lr1", "normalized_lr2"),
        ("final", "original_lr2", "normalized_lr2"),
    ]:
        old = {
            (r["seed"], r["batch"]): r
            for r in data["rows"]
            if r["stage"] == stage and r["arm"] == control
        }
        new = {
            (r["seed"], r["batch"]): r
            for r in data["rows"]
            if r["stage"] == stage and r["arm"] == candidate
        }
        for key in old:
            for parameter in old[key]["objectives"]["unit_logit_probe"]["parameters"]:
                base = old[key]["objectives"]["unit_logit_probe"]["parameters"][
                    parameter
                ]["gradient"]["rms"]
                value = new[key]["objectives"]["unit_logit_probe"]["parameters"][
                    parameter
                ]["gradient"]["rms"]
                paired.append(
                    {
                        "stage": stage,
                        "control": control,
                        "candidate": candidate,
                        "seed": key[0],
                        "batch": key[1],
                        "parameter": parameter,
                        "control_gradient_rms": base,
                        "candidate_gradient_rms": value,
                        "ratio": value / base if base else None,
                    }
                )
    files.append(write_csv("paired-unit-probe-gradients.csv", paired))
    summary = {
        "audit": checked,
        "summaries": summaries,
        "paired_parameter_unit_probe_gradients": paired,
        "limitations": data["limitations"],
        "source_sha256": digest(Path(__file__)),
        "measurements_sha256": digest(data_path),
        "export_sha256": {str(p.relative_to(ROOT)): digest(p) for p in files},
    }
    summary_path = RESULTS / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(checked, indent=2))


if __name__ == "__main__":
    main()
