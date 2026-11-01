"""
Replay every frozen query-routing checkpoint on CPU without training.

This is an independent execution-path check of the compiled GPU experiment,
not a second learning cohort. Retain all curves, failures and rounding errors.
"""

import hashlib
import json
import math
from pathlib import Path

import torch

from diff_gpt.application.grokking import residual_attention_probe as probe
from diff_gpt.domain.optimizer.schedule_free import ScheduleFreeAdamW


def require(condition: object, detail: str) -> None:
    """Keep replay validation active even when Python optimization is enabled."""
    if not condition:
        raise ValueError(detail)


def replay(directory: Path) -> dict:
    """Verify source, data, optimizer clock, sampler and all final outcomes."""
    torch.set_num_threads(1)
    report = json.loads((directory / "results.json").read_text())
    protocol = json.loads((directory / "protocol.json").read_text())
    require(report["protocol"] == protocol, 'report["protocol"] == protocol')
    require(
        hashlib.sha256(Path(probe.__file__).read_bytes()).hexdigest()
        == protocol["source_sha256"],
        "Probe source hash mismatch",
    )
    require(
        len(report["results"]) == len(probe.CASES),
        'len(report["results"]) == len(probe.CASES)',
    )
    for case, record in zip(probe.CASES, report["results"], strict=True):
        require(case.label == record["label"], 'case.label == record["label"]')
        checkpoint_path = directory / f"{case.label}.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        require(
            checkpoint["step"] == probe.UPDATES, 'checkpoint["step"] == probe.UPDATES'
        )
        require(
            checkpoint["optimizer"]["param_groups"][0]["step"] == probe.UPDATES,
            'checkpoint["optimizer"]["param_groups"][0]["step"] == probe.UPDATES',
        )
        model = probe.Probe(case)
        model.load_state_dict(checkpoint["model"])
        optimizer = ScheduleFreeAdamW(model.parameters())
        optimizer.load_state_dict(checkpoint["optimizer"])
        splits = {
            name: probe.dataset(case.seed + offset)
            for name, offset in (("train", 1000), ("heldout", 2000))
        }
        for name, (values, _, _) in splits.items():
            require(
                hashlib.sha256(values.numpy().tobytes()).hexdigest()
                == record["input_sha256"][name],
                "Split input hash mismatch",
            )
        rng = torch.Generator().manual_seed(case.seed + 3000)
        for _ in range(probe.UPDATES):
            torch.randint(512, (probe.BATCH,), generator=rng)
        require(
            torch.equal(rng.get_state(), checkpoint["sampler"]),
            'torch.equal(rng.get_state(), checkpoint["sampler"])',
        )
        actual = probe.observe(model, optimizer, splits, probe.UPDATES)
        recorded = record["curves"][-1]
        errors = {}
        for split in ("train", "heldout"):
            for metric, expected in recorded["splits"][split].items():
                value = actual["splits"][split][metric]
                errors[f"{split}.{metric}"] = abs(value - expected)
                if metric == "routing_accuracy":
                    require(value == expected, "value == expected")
                else:
                    require(
                        math.isclose(value, expected, abs_tol=5e-06, rel_tol=3e-05),
                        str((case.label, split, metric, value, expected)),
                    )
        require(
            actual["finite_gradient"] and recorded["finite_gradient"],
            'actual["finite_gradient"] and recorded["finite_gradient"]',
        )
        final = actual["splits"]["heldout"]
        success = (
            final["mse"] <= probe.MSE_TARGET
            and final["routing_accuracy"] >= probe.ACCURACY_TARGET
        )
        require(success == record["success"], 'success == record["success"]')
        record["checkpoint_sha256"] = hashlib.sha256(
            checkpoint_path.read_bytes()
        ).hexdigest()
        record["cpu_replay"] = {
            "final": actual,
            "absolute_metric_errors": errors,
            "same_success_classification": True,
            "sampler_and_optimizer_clock_verified": True,
        }
        print(json.dumps({"label": case.label, "success": success}), flush=True)
    report["replay_scope"] = (
        "All21 final checkpoints on CPU; source and input SHA256 verified; "
        "optimizer step and exact sampler state verified; original outcomes retained. "
        "Same query direction at each nonzero scale; seeds vary data and minibatches. "
        "Mixture metrics use averaged weights; gradients use training queries. "
        "An extra FP32 evaluation roundtrip can introduce rounding differences."
    )
    report["training_source_commit"] = "386ae01"
    report["lean_bound_commit"] = "1a9b522e"
    report["raw_results_sha256"] = hashlib.sha256(
        (directory / "results.json").read_bytes()
    ).hexdigest()
    return report
