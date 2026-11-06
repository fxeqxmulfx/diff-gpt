"""Apply the real score envelopes to the four declared cubic-sum model shapes."""

from __future__ import annotations

import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path

import torch

from research.cubic_capacity.factories import ROOT, factories, plan
from research.cubic_capacity.readiness import verify_training_ready
from research.current_grokking.data import build
from research.current_grokking.protocol import Case
from research.early99.experiments.factories import baseline_model
from research.early99_reporting.round2_audit import read, require, verify_hash
from research.stored_rope_guarantees.check import digest as sha256

STUDY = Path(__file__).parent
RESULTS = STUDY / "results"
HEAD_WIDTH = 32
CONTEXT = 2
STATEMENTS = 4


def main() -> None:
    """Read fixed buffers and evaluate formulas, without forwards or training."""
    verify_training_ready()
    declared = read(STUDY / "plan.json")
    proof = read(RESULTS / "guarantees.json")
    require(
        proof["statements"] == STATEMENTS
        and proof["warnings"] == 0
        and not proof["extra_axioms"],
        "Incomplete proof",
    )
    for group in [declared["sources_sha256"], proof["source_sha256"]]:
        for name, digest in group.items():
            verify_hash(ROOT / name, digest)
    verify_hash(STUDY / "plan.json", proof["plan_sha256"])
    for name, digest in proof["external_source_sha256"].items():
        verify_hash(Path(proof["external_project"]) / name, digest)
    certificate = read(
        ROOT / "research/stored_rope_guarantees/results/application.json"
    )
    cube = next(
        row for row in certificate["rows"] if row["case"] == "arithmetic-sum-cube"
    )
    require(cube["all_pairs_within_declared_bound"], "Missing cubic table certificate")
    bound = Fraction(
        declared["stored_pair_bound"]["numerator"],
        declared["stored_pair_bound"]["denominator"],
    )
    search = plan()
    rows, inventories = [], []
    for shape in search["shapes"]:
        spec = next(
            s
            for s in search["configurations"]
            if s["size"] == ("depth" if shape["reuse_existing"] else shape["name"])
            and s["arm"] == "normalized_lr2"
        )
        case = Case(**shape["case"])
        task = build(case, "cpu")
        torch.manual_seed(0)
        model = factories(spec).model(task, case, "cpu").model
        require(
            task.context == CONTEXT and case.width // case.heads == HEAD_WIDTH,
            "Changed context or head width",
        )
        require(
            model.token_embedding_table.weight is model.lm_head.weight, "Untied weights"
        )
        for label, table in [("cos", model.rope_cos), ("sin", model.rope_sin)]:
            raw = table.contiguous().numpy().tobytes()
            key = f"runs/stored-rope-guarantees/tables/{case.name}/{label}.bin"
            require(
                hashlib.sha256(raw).hexdigest() == cube["table_sha256"][key],
                "Stored table differs from exact certificate",
            )
        usual = baseline_model(task, case, "cpu").model
        inverse_gain = model.blocks[0].sa.attention_gain().flatten()[0].item()
        usual_gain = usual.blocks[0].sa.attention_gain().flatten()[0].item()
        inventories.append(
            {
                "size": shape["name"],
                "context": task.context,
                "head_width": HEAD_WIDTH,
                "inverse_gain_stored_float32": inverse_gain,
                "usual_gain_stored_float32": usual_gain,
                "usual_gain_ideal_real_approx": 1 / math.sqrt(HEAD_WIDTH),
                "tied_weights": True,
                "tables_match_prior_exact_certificate": True,
            }
        )
        for label, gain, factor in [
            ("inverse_width_ssmax", inverse_gain, math.log(CONTEXT)),
            ("usual_gain_ssmax", usual_gain, math.log(CONTEXT)),
            ("original_softmax", usual_gain, 1.0),
        ]:
            for constant in [Fraction(1), bound]:
                score_bound = gain * HEAD_WIDTH * factor * float(constant)
                top = 1 / (1 + math.exp(-2 * score_bound))
                rows.append(
                    {
                        "size": shape["name"],
                        "head_width": HEAD_WIDTH,
                        "visible_keys": CONTEXT,
                        "rule": label,
                        "gain_stored_float32_as_real": gain,
                        "table_energy_bound_numerator": constant.numerator,
                        "table_energy_bound_denominator": constant.denominator,
                        "real_score_bound_approx": score_bound,
                        "probability_upper_approx": top,
                        "probability_lower_approx": 1 - top,
                    }
                )
    record = {
        "complete": True,
        "scientific_optimizer_updates": 0,
        "heldout_scores_computed": 0,
        "scope": (
            "Real-arithmetic envelopes using the actual stored gain as a real "
            "constant. Decimal endpoints are floating approximations, not a "
            "certificate for rounded forward execution or convergence."
        ),
        "capacity_shape_contexts_checked": len(inventories),
        "inventories": inventories,
        "rows": rows,
        "source_plan_sha256": sha256(STUDY / "plan.json"),
        "proof_record_sha256": sha256(RESULTS / "guarantees.json"),
        "application_source_sha256": sha256(Path(__file__)),
        "frozen_cube_table_certificate": cube,
    }
    (RESULTS / "application.json").write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "shapes": len(inventories),
                "probability_envelopes": len(rows),
                "optimizer_updates": 0,
                "current_size": rows[:6],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
