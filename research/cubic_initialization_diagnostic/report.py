"""Audit all initialized states and summarize scales without choosing an initializer."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from research.cubic_initialization_diagnostic.run import RESULTS, STUDY, verify_inputs
from research.cubic_usual_gain.readiness import sha256
from research.early99_reporting.round2_audit import (
    read,
    require,
    verify_csv,
    verify_hash,
)
from research.transformer_path_diagnostic.report import distribution

TOLERANCE = 1e-5
INITIAL_STD = 0.02


def audit(data: dict, declared: dict) -> dict:
    """Require every state, all finite active paths and every local reconstruction."""
    expected = {
        (shape["name"], arm, seed)
        for shape in declared["shapes"]
        for arm in declared["arms"]
        for seed in declared["seeds"]
    }
    rows = data["rows"]
    require(
        len(rows) == len(expected)
        and {(r["size"], r["arm"], r["seed"]) for r in rows} == expected,
        "Incomplete or duplicate initialized grid",
    )
    require(
        data["complete"]
        and data["scientific_optimizer_updates"] == 0
        and data["new_heldout_scores"] is False,
        "Changed diagnostic scope",
    )
    errors = []
    for row in rows:
        require(
            row["exact_forward_and_all_parameter_gradient_parity"]
            and row["unchanged_model_and_rng"],
            "Observer changed execution",
        )
        require(
            set(row["objectives"]) == set(declared["objectives"]),
            "Incomplete objectives",
        )
        for layer in range(row["layers"]):
            prefix = f"blocks.{layer}"
            for suffix in [
                "input",
                "output",
                "ln_1.input",
                "ln_1.output",
                "ln_2.output",
                "sa.q_norm.input",
                "sa.q_norm.output",
                "sa.k_norm.input",
                "sa.k_norm.output",
                "sa.rope.q.output",
                "sa.rope.k.output",
                "sa.softmax.output",
                "sa.xsa.output",
                "sa.c_proj.linear.output",
                "sa.c_proj.output",
                "ffwd.c_fc.output",
                "ffwd.silu.output",
                "ffwd.c_proj.input",
                "ffwd.c_proj.linear.output",
                "ffwd.c_proj.output",
            ]:
                require(
                    prefix + "." + suffix in row["executed_sites"],
                    "Unobserved transformer component",
                )
        for objective in row["objectives"].values():
            require(
                len(objective["residual_paths"]) == 2 * row["layers"],
                "Missing residual path",
            )
            errors.extend(
                path["gradient_paths"]["reconstruction_relative_l2"]
                for path in objective["residual_paths"]
            )
            errors.append(objective["tied_weight_paths"]["reconstruction_relative_l2"])
            for site, values in objective["sites"].items():
                activation, gradient = values["activation"], values["gradient"]
                if site.endswith(".softmax.input0"):
                    batch, heads, queries, keys = activation["shape"]
                    require(
                        queries == keys
                        and activation["nonfinite"]
                        == batch * heads * queries * (queries - 1) // 2,
                        "Incorrect causal mask",
                    )
                else:
                    require(
                        activation["nonfinite"] == 0, f"Nonfinite activation: {site}"
                    )
                require(
                    gradient is not None or site.endswith(".attention.input3"),
                    "Missing active gradient",
                )
                if gradient is not None:
                    require(gradient["nonfinite"] == 0, "Nonfinite gradient")
            for value in objective["parameters"].values():
                require(
                    value["weight"]["nonfinite"] == value["gradient"]["nonfinite"] == 0,
                    "Nonfinite parameter or gradient",
                )
    require(max(errors) <= TOLERANCE, "Invalid path reconstruction")
    return {
        "states": len(rows),
        "objective_parity_checks": 2 * len(rows),
        "path_reconstructions": len(errors),
        "maximum_reconstruction_relative_l2": max(errors),
        "finite_active_values_and_gradients": True,
    }


def tables(rows: list[dict]) -> tuple[list, list]:
    """Retain every component and parameter piece, including Q/K/V and SwiGLU splits."""
    components, parameters = [], []
    for row in rows:
        identity = {
            key: row[key] for key in ["size", "arm", "seed", "layers", "width", "heads"]
        }
        for objective, values in row["objectives"].items():
            for site, metrics in values["sites"].items():
                record = {**identity, "objective": objective, "site": site}
                for kind in ["activation", "gradient"]:
                    summary = metrics[kind]
                    for field in [
                        "elements",
                        "nonfinite",
                        "rms",
                        "mean",
                        "abs_max",
                        "zero_fraction",
                    ]:
                        record[kind + "_" + field] = (
                            summary[field] if summary is not None else None
                        )
                components.append(record)
            for name, metrics in values["parameters"].items():
                weight, gradient = metrics["weight"], metrics["gradient"]
                shared = name == "token_embedding_table.weight"
                fan_out, fan_in = weight["shape"]
                parameters.append(
                    {
                        **identity,
                        "objective": objective,
                        "parameter": name,
                        "role": "shared_embedding_readout" if shared else "projection",
                        "fan_in": fan_in,
                        "fan_out": fan_out,
                        "weight_rms": weight["rms"],
                        "weight_mean": weight["mean"],
                        "weight_population_std": math.sqrt(
                            max(0, weight["rms"] ** 2 - weight["mean"] ** 2)
                        ),
                        "independent_unit_input_variance_reference": None
                        if shared
                        else fan_in * INITIAL_STD**2,
                        "gradient_rms": gradient["rms"],
                        "gradient_to_weight_l2": metrics["gradient_to_weight_l2"],
                    }
                )
    return components, parameters


def summarize(rows: list[dict], declared: dict) -> list:
    """Report all-seed distributions at every layer without selecting a best arm."""
    groups = []
    for shape in declared["shapes"]:
        for arm in declared["arms"]:
            selected = [
                r for r in rows if r["size"] == shape["name"] and r["arm"] == arm
            ]
            require(
                len(selected) == len(declared["seeds"])
                and {r["seed"] for r in selected} == set(declared["seeds"]),
                "Missing summary seed",
            )
            for layer in range(shape["layers"]):
                prefix = f"blocks.{layer}"
                group = {
                    "size": shape["name"],
                    "arm": arm,
                    "layer": layer,
                    "seeds": declared["seeds"],
                }
                sites = {
                    "q_before_norm": prefix + ".sa.q_norm.input",
                    "k_before_norm": prefix + ".sa.k_norm.input",
                    "attention_before_output_norm": prefix + ".sa.c_proj.linear.output",
                    "attention_branch": prefix + ".sa.c_proj.output",
                    "swiglu_gate_value": prefix + ".ffwd.c_fc.output",
                    "swiglu_product": prefix + ".ffwd.c_proj.input",
                    "ffn_before_output_norm": prefix + ".ffwd.c_proj.linear.output",
                    "ffn_branch": prefix + ".ffwd.c_proj.output",
                    "raw_embedding": "token_embedding_table.lookup.output",
                    "logits": "lm_head.output",
                }
                for label, site in sites.items():
                    group[label + "_activation_rms"] = distribution(
                        [
                            r["objectives"]["unit_logit_probe"]["sites"][site][
                                "activation"
                            ]["rms"]
                            for r in selected
                        ]
                    )
                    group[label + "_probe_gradient_rms"] = distribution(
                        [
                            r["objectives"]["unit_logit_probe"]["sites"][site][
                                "gradient"
                            ]["rms"]
                            for r in selected
                        ]
                    )
                for label in ["q", "k", "v"]:
                    group[label + "_parameter_probe_gradient_rms"] = distribution(
                        [
                            r["objectives"]["unit_logit_probe"]["parameters"][
                                prefix + ".sa.c_attn.weight." + label
                            ]["gradient"]["rms"]
                            for r in selected
                        ]
                    )
                group["attention_normalized_entropy"] = distribution(
                    [
                        h["normalized_entropy_mean_nonsingleton"]
                        for r in selected
                        for h in r["attention"]
                        if h["layer"] == layer
                    ]
                )
                if arm.startswith("normalized"):
                    for label, site in [
                        ("embedding", "token_embedding_table.norm"),
                        ("attention_output", prefix + ".sa.c_proj.norm"),
                        ("ffn_output", prefix + ".ffwd.c_proj.norm"),
                    ]:
                        group[label + "_local_rms_probe_vjp_gain"] = distribution(
                            [
                                r["objectives"]["unit_logit_probe"]["rms_local_paths"][
                                    site
                                ]["local_vjp_gain"]
                                for r in selected
                            ]
                        )
                groups.append(group)
    return groups


def main() -> None:
    """Write complete CSVs and tie the descriptive summaries to immutable raw data."""
    declared = verify_inputs()
    data_path = RESULTS / "measurements.json"
    data = read(data_path)
    verify_hash(STUDY / "plan.json", data["plan_sha256"])
    verify_hash(STUDY / "run.py", data["measurement_source_sha256"])
    checks = audit(data, declared)
    components, parameters = tables(data["rows"])
    lookup = {(r["size"], r["arm"], r["seed"]): r for r in data["rows"]}
    pairs = []
    for shape in declared["shapes"]:
        for seed in declared["seeds"]:
            inverse = lookup[shape["name"], "normalized_inverse", seed]
            usual = lookup[shape["name"], "normalized_usual", seed]
            require(
                inverse["initial_parameters_sha256"]
                == usual["initial_parameters_sha256"],
                "Unpaired initialized weights",
            )
            for layer in range(shape["layers"]):
                pair = {"size": shape["name"], "seed": seed, "layer": layer}
                for piece in ["q", "k", "v"]:
                    key = f"blocks.{layer}.sa.c_attn.weight.{piece}"
                    field = "unit_logit_probe"
                    old = inverse["objectives"][field]["parameters"][key]["gradient"][
                        "rms"
                    ]
                    new = usual["objectives"][field]["parameters"][key]["gradient"][
                        "rms"
                    ]
                    require(old > 0, "Undefined paired gradient ratio")
                    pair[piece + "_usual_over_inverse_probe_gradient_rms"] = new / old
                pairs.append(pair)
    outputs = []
    for name, rows in [
        ("components", components),
        ("parameters", parameters),
        ("paired_gain_gradients", pairs),
    ]:
        path = RESULTS / f"{name}.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(rows[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
        verify_csv(path, rows)
        outputs.append(path)
    summary = {
        "complete": True,
        "checks": checks,
        "groups": summarize(data["rows"], declared),
        "component_rows": len(components),
        "parameter_rows": len(parameters),
        "paired_gain_gradients": pairs,
        "measurements_sha256": sha256(data_path),
        "reporter_sha256": sha256(Path(__file__)),
        "csv_sha256": {path.name: sha256(path) for path in outputs},
        "limits": declared["analysis_limits"],
        "new_initializer_selected": False,
    }
    (RESULTS / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(checks))


if __name__ == "__main__":
    main()
