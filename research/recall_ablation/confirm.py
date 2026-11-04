"""Confirm a passed final architecture on fresh seeds using the original loop."""

import argparse
import ast
import hashlib
import json
from pathlib import Path

import torch

from research.recall_ablation.features import ARCHITECTURE_STAGES
from research.recall_ablation.run import ROOT, TRAIN_ROWS, run, split, write_json


def training_signature(source: str) -> str:
    """Fingerprint training math while excluding provenance/eligibility branches."""
    module = ast.parse(source)
    constants = {"TASK", "RATE", "WARMUP", "BATCH", "OBSERVE", "TARGET", "TRAIN_ROWS"}

    def assigns(node: ast.stmt, name: str) -> bool:
        return isinstance(node, ast.Assign) and any(
            isinstance(item, ast.Name) and item.id == name
            for target in node.targets
            for item in ast.walk(target)
        )

    body = [
        node
        for node in module.body
        if (
            isinstance(node, (ast.Import, ast.ImportFrom))
            or any(assigns(node, name) for name in constants)
            or (isinstance(node, ast.FunctionDef) and node.name == "split")
        )
    ]
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    model_start = next(
        i for i, node in enumerate(function.body) if assigns(node, "model")
    )
    metadata_start = next(
        i for i, node in enumerate(function.body) if assigns(node, "metadata")
    )
    history_start = next(
        i for i, node in enumerate(function.body) if assigns(node, "history")
    )
    body.extend(function.body[model_start:metadata_start])
    body.extend(function.body[history_start:])
    canonical = ast.dump(
        ast.Module(body=body, type_ignores=[]), include_attributes=False
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def main() -> None:
    """Reuse the reference recipe without requiring unrun per-seed ablations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        default=ROOT / "runs/recall-ablation/prefix-stable/seed1/result.json",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "runs/recall-ablation")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 2])
    args = parser.parse_args()
    raw_reference = args.reference.read_bytes()
    reference = json.loads(raw_reference)
    if (
        reference.get("status") != "passed"
        or reference.get("stage") not in ARCHITECTURE_STAGES
    ):
        msg = "confirmation requires a passed architectural screen"
        raise ValueError(msg)
    if len(set(args.seeds)) != len(args.seeds) or reference["seed"] in args.seeds:
        msg = "confirmation seeds must be distinct fresh seeds, excluding the reference"
        raise ValueError(msg)
    driver_name = "research/recall_ablation/run.py"
    frozen_driver = args.reference.parent / "source" / driver_name
    frozen_bytes = frozen_driver.read_bytes()
    if hashlib.sha256(frozen_bytes).hexdigest() != reference["sources"][driver_name]:
        msg = "the reference training-driver snapshot does not match its recorded hash"
        raise ValueError(msg)
    signature = training_signature(frozen_bytes.decode())
    if signature != training_signature((ROOT / driver_name).read_text()):
        msg = "confirmation changed the accepted training algorithm"
        raise ValueError(msg)
    reference["_training_signature_sha256"] = signature
    args.stage = reference["stage"]
    args.threads = reference["threads"]
    args.max_updates = reference["max_updates"]
    args.compile = reference["compiled"]
    for seed in args.seeds:
        directory = args.output / args.stage / f"seed{seed}"
        if directory.exists():
            msg = f"refusing to overwrite an existing confirmation: {directory}"
            raise FileExistsError(msg)
    reference["_result_path"] = str(args.reference.resolve())
    reference["_result_sha256"] = hashlib.sha256(raw_reference).hexdigest()
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("highest")
    train, train_hash = split("train", TRAIN_ROWS)
    validation, validation_hash = split("validation", 512)
    data = {
        "train": train,
        "validation": validation,
        "fingerprints": {"train": train_hash, "validation": validation_hash},
    }
    results = []
    for seed in args.seeds:
        result = run(args, seed, data, confirmation_reference=reference)
        results.append(
            {
                "seed": seed,
                "status": result["status"],
                "first99": result["first99"],
                "endpoint": result["endpoint"],
            }
        )
        print(json.dumps({"confirmation": results[-1]}), flush=True)
    write_json(
        args.output / args.stage / "confirmation_summary.json",
        {
            "reference_path": reference["_result_path"],
            "reference_sha256": reference["_result_sha256"],
            "reference_seed": reference["seed"],
            "reference_first99": reference["first99"],
            "results": results,
            "all_requested_seeds_passed": all(
                row["status"] == "passed" for row in results
            ),
            "test_loaded": False,
        },
    )


if __name__ == "__main__":
    main()
