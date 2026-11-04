"""Independent endpoint, source, sampler and optimizer-state replay."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from lab.infrastructure.benchmarks.samplers import EpochSampler

from diff_gpt.domain.model.gpt import GPT
from research.current_grokking import data
from research.current_grokking.protocol import ROOT, Case
from research.current_grokking.run import write
from research.current_grokking.train import Logits, Stepper


def audit(directory: Path) -> dict:
    """Rebuild a fresh public GPT and replay the actual retained endpoint on CPU."""
    torch.set_num_threads(2)
    result = json.loads((directory / "result.json").read_text())
    checkpoint = torch.load(
        directory / "checkpoint.pt", weights_only=True, map_location="cpu"
    )
    metadata = checkpoint["metadata"]
    case = Case(**metadata["case"])
    task = data.build(case, "cpu")
    if task.fingerprints != metadata["data_fingerprints"]:
        message = "Rebuilt data differ from the trained split fingerprints"
        raise ValueError(message)
    model = Logits(
        GPT(
            task.vocab,
            case.width,
            block_size=task.context,
            n_head=case.heads,
            n_layer=case.layers,
            use_checkpoint=False,
            pad_vocab_size_to=1,
        )
    )
    model.model.load_state_dict(checkpoint["model"])
    stepper = Stepper(model, task, case, "cpu", compiled=False)
    measured = {
        "train": stepper.evaluate("train"),
        "validation": stepper.evaluate(task.selection),
    }
    exact_counts = {}
    numeric_checks = {}
    for name, metrics in measured.items():
        original = result["endpoint"][name]
        for key in (
            "correct",
            "targets",
            "exact",
            "examples",
            "example_count",
            "target_count",
            "field_correct",
            "native_correct",
        ):
            if key in metrics and key in original:
                exact_counts[f"{name}.{key}"] = metrics[key] == original[key]
        for key in (
            "loss",
            "sequence_accuracy",
            "token_accuracy",
            "decoded_accuracy",
            "native_token_accuracy",
            "accuracy",
            "skill",
            "relative_ce_gain",
            "mse",
            "mae",
            "bayes_loss",
        ):
            if key in metrics and key in original:
                first, second = metrics[key], original[key]
                numeric_checks[f"{name}.{key}"] = (
                    first is None and second is None
                    if first is None or second is None
                    else math.isclose(first, second, rel_tol=1e-5, abs_tol=1e-7)
                )
    sampler = EpochSampler(
        len(task.splits["train"]), case.batch, wrap=False, seed=metadata["batch_seed"]
    )
    for _ in range(checkpoint["completed"]):
        sampler.next()
    state = sampler.state()
    sampling_matches = all(
        torch.equal(value, checkpoint["sampler"][key].cpu())
        if isinstance(value, torch.Tensor)
        else value == checkpoint["sampler"][key]
        for key, value in state.items()
    )
    optimizer_matches = all(
        int(state["step"]) == checkpoint["completed"]
        for state in checkpoint["optimizer"]["state"].values()
    )
    sources_match = all(
        hashlib.sha256((directory / "source" / name).read_bytes()).hexdigest()
        == expected
        for name, expected in metadata["sources"].items()
    )
    for resumption in metadata.get("resumptions", []):
        sources_match &= all(
            hashlib.sha256(
                (directory / resumption["directory"] / "source" / name).read_bytes()
            ).hexdigest()
            == expected
            for name, expected in resumption["sources"].items()
        )
    parameters_finite = all(
        bool(torch.isfinite(parameter).all()) for parameter in model.parameters()
    )
    report = {
        "checkpoint_sha256": hashlib.sha256(
            (directory / "checkpoint.pt").read_bytes()
        ).hexdigest(),
        "case": case.name,
        "seed": metadata["seed"],
        "device": "cpu",
        "replayed": measured,
        "exact_count_checks": exact_counts,
        "exact_count_checks_available": bool(exact_counts),
        "all_exact_counts_match": all(exact_counts.values()) if exact_counts else None,
        "numeric_metric_checks": numeric_checks,
        "all_numeric_metrics_close": all(numeric_checks.values()),
        "numeric_metric_tolerance": {"relative": 1e-5, "absolute": 1e-7},
        "sampler_replay_matches": sampling_matches,
        "optimizer_steps_match": optimizer_matches,
        "source_snapshot_hashes_match": sources_match,
        "parameters_finite": parameters_finite,
        "completed_matches": checkpoint["completed"] == result["completed"],
    }
    write(directory / "audit.json", report)
    if (
        not sampling_matches
        or not optimizer_matches
        or not sources_match
        or not parameters_finite
        or not report["completed_matches"]
    ):
        message = (
            "Endpoint provenance/state audit failed; details retained in audit.json"
        )
        raise AssertionError(message)
    return report


def main() -> None:
    """Audit explicit cases or every finished case with no retained audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "runs/current-grokking")
    parser.add_argument("--case")
    args = parser.parse_args()
    for path in sorted(args.root.glob("*/seed*/result.json")):
        if args.case is not None and path.parent.parent.name != args.case:
            continue
        if (path.parent / "audit.json").exists():
            continue
        result = audit(path.parent)
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in (
                        "case",
                        "seed",
                        "all_exact_counts_match",
                        "sampler_replay_matches",
                        "optimizer_steps_match",
                    )
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
