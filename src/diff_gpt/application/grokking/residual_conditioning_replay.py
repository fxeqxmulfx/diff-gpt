"""Audit the complete coordinate-repair cohort with independent FP64 equations."""

import copy
import hashlib
import json
import math
from typing import TYPE_CHECKING

import torch

from diff_gpt.application.grokking import residual_attention_probe as original
from diff_gpt.application.grokking.residual_attention_replay import require
from diff_gpt.application.grokking.residual_conditioning import (
    ROOT,
    make_model,
    read_protocol,
    sha256,
)
from diff_gpt.domain.optimizer.schedule_free import ScheduleFreeAdamW

if TYPE_CHECKING:
    from pathlib import Path


def independent_metrics(state: dict, splits: dict, eps: float) -> dict:
    """Evaluate physical-query equations without calling the model or optimizer."""
    query = state["model"]["query"].double()
    scale = state["model"].get("query_scale", torch.tensor(1.0)).double()
    group = state["optimizer"]["param_groups"][0]
    base = state["optimizer"]["state"][group["params"][0]]["base"].double()
    average = query + (1 - 1 / group["betas"][0]) * (base - query)

    def scores(coordinate: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        physical = coordinate * scale
        normalized = physical / (physical.square().mean() + eps).sqrt()
        keys = values / (values.square().mean(-1, keepdim=True) + eps).sqrt()
        return (keys * normalized).sum(-1) / query.numel() ** 0.5

    result = {"splits": {}}
    for name, (inputs, targets, chosen) in splits.items():
        values = inputs.double()
        logits = scores(average, values)
        weights = logits.softmax(-1)
        output = (weights[..., None] * values).sum(-2)
        result["splits"][name] = {
            "mse": float((output - targets.double()).square().mean()),
            "routing_accuracy": float((logits.argmax(-1) == chosen).double().mean()),
            "useful_mass": float(weights.gather(1, chosen[:, None]).mean()),
            "max_abs_score": float(logits.abs().max()),
            "entropy": float(
                -(weights * weights.clamp_min(1e-30).log()).sum(-1).mean()
            ),
        }
    coordinate = query.clone().requires_grad_()
    inputs, targets, _ = splits["train"]
    values = inputs.double()
    weights = scores(coordinate, values).softmax(-1)
    loss = ((weights[..., None] * values).sum(-2) - targets.double()).square().mean()
    gradient = torch.autograd.grad(loss, coordinate)[0]
    require(bool(torch.isfinite(gradient).all()), "Nonfinite independent gradient")
    result["query_gradient_l2"] = float(gradient.norm())
    result["physical_query_gradient_l2"] = float(gradient.norm() / scale)
    return result


def validate_initial(
    case: dict, initial: dict, conditioned: dict, protocol: dict
) -> dict:
    """Check unreset histories and the exact intended change, with FP32 tolerance."""
    group = initial["optimizer"]["param_groups"][0]
    if case["resume_source"]:
        source = torch.load(
            ROOT / case["resume_source"], map_location="cpu", weights_only=True
        )
        require(
            sha256(ROOT / case["resume_source"]) == case["resume_sha256"], "Source hash"
        )
        torch.testing.assert_close(
            initial["model"]["query"], source["model"]["query"], rtol=0, atol=0
        )
        torch.testing.assert_close(
            initial["optimizer"], source["optimizer"], rtol=0, atol=0
        )
        require(initial["step"] == source["step"], "Source step")
        require(torch.equal(initial["sampler"], source["sampler"]), "Source sampler")
    else:
        torch.manual_seed(case["seed"])
        model = make_model(case, "cpu")
        optimizer = ScheduleFreeAdamW(model.parameters(), **protocol["optimizer"])
        torch.testing.assert_close(initial["model"], model.state_dict(), rtol=0, atol=0)
        torch.testing.assert_close(
            initial["optimizer"], optimizer.state_dict(), rtol=0, atol=0
        )
        require(initial["step"] == 0, "Fresh step")
        rng = torch.Generator().manual_seed(case["seed"] + 3000)
        require(torch.equal(initial["sampler"], rng.get_state()), "Fresh sampler")
    require(
        group == conditioned["optimizer"]["param_groups"][0],
        "Conditioning changed clocks",
    )
    require(initial["step"] == conditioned["step"], "Conditioning changed step")
    require(
        torch.equal(initial["sampler"], conditioned["sampler"]),
        "Conditioning changed sampler",
    )
    query = initial["model"]["query"].double()
    old = initial["optimizer"]["state"][group["params"][0]]
    new = conditioned["optimizer"]["state"][group["params"][0]]
    base = old["base"].double()
    average = query + (1 - 1 / group["betas"][0]) * (base - query)
    maximum = max(float(v.square().mean().sqrt()) for v in (query, base, average))
    factor = (
        maximum
        if case["mode"] == "conditioned"
        and maximum > protocol["maximum_coordinate_rms"]
        else 1.0
    )
    actual_scale = float(conditioned["model"].get("query_scale", torch.tensor(1.0)))
    require(math.isclose(actual_scale, factor, rel_tol=3e-6), "Wrong scale factor")
    # The saved initial observation includes one FP32 eval/train roundtrip.
    torch.testing.assert_close(
        conditioned["model"]["query"].double(), query / factor, rtol=3e-6, atol=2e-7
    )
    torch.testing.assert_close(
        new["base"].double(), base / factor, rtol=3e-6, atol=2e-7
    )
    torch.testing.assert_close(
        new["second"].double(),
        old["second"].double() * factor**2,
        rtol=5e-6,
        atol=1e-25,
    )
    return {
        "source_state_exact": True,
        "clocks_and_sampler_unchanged": True,
        "factor": actual_scale,
    }


def replay(directory: Path) -> dict:
    """Retain outcomes while checking sources, checkpoints and metric equations."""
    torch.set_num_threads(1)
    protocol = read_protocol()
    report = json.loads((directory / "results.json").read_text())
    require(report["protocol"] == protocol, "Protocol/report mismatch")
    require(
        json.loads((directory / "protocol.json").read_text()) == protocol,
        "Copied protocol mismatch",
    )
    require(
        report["protocol_sha256"] == sha256(directory / "protocol.json"),
        "Protocol hash",
    )
    require(len(report["results"]) == len(protocol["cases"]), "Incomplete cohort")
    require(
        not report["not_started"] and "failure" not in report, "Failed cohort retained"
    )
    for case, record in zip(protocol["cases"], report["results"], strict=True):
        require(case == record["case"], "Case order mismatch")
        path = directory / case["label"]
        require(
            json.loads((path / "result.json").read_text()) == record,
            "Per-case report mismatch",
        )
        require(
            sha256(path / "checkpoint.pt") == record["checkpoint_sha256"],
            "Checkpoint hash",
        )
        initial, conditioned, final = [
            torch.load(path / name, map_location="cpu", weights_only=True)
            for name in ("initial.pt", "conditioned_initial.pt", "checkpoint.pt")
        ]
        state_check = validate_initial(case, initial, conditioned, protocol)
        source_step = initial["step"]
        end = source_step + case["updates"]
        group = final["optimizer"]["param_groups"][0]
        require(
            group["training"]
            and group["step"] == end == final["step"] == record["checkpoint_step"],
            "Final clock",
        )
        require(
            record["completed"] and record["source_step"] == source_step,
            "Incomplete case",
        )
        require(
            [r["step"] for r in record["curves"]]
            == list(range(source_step, end + 1, protocol["observe_every"])),
            "Curve intervals",
        )
        require(
            all(r["finite_gradient"] for r in record["curves"]),
            "Nonfinite curve gradient",
        )
        for tensor in (
            *final["model"].values(),
            *final["optimizer"]["state"][group["params"][0]].values(),
        ):
            require(bool(torch.isfinite(tensor).all()), "Nonfinite final state")
        sampler = torch.Generator().set_state(initial["sampler"])
        for _ in range(case["updates"]):
            torch.randint(
                protocol["train_rows"], (protocol["batch"],), generator=sampler
            )
        require(
            torch.equal(sampler.get_state(), final["sampler"]), "Final sampler mismatch"
        )
        splits = {
            name: original.dataset(case["seed"] + offset, rows=protocol[f"{name}_rows"])
            for name, offset in (("train", 1000), ("heldout", 2000))
        }
        for name, (values, _, _) in splits.items():
            require(
                hashlib.sha256(values.numpy().tobytes()).hexdigest()
                == record["input_sha256"][name],
                "Input hash",
            )
        errors = {}
        checks = (
            ("before", initial, record["before_conditioning"]),
            ("after", conditioned, record["curves"][0]),
            ("final", final, record["curves"][-1]),
        )
        independent = {}
        for label, state, observed in checks:
            actual = independent_metrics(state, splits, protocol["eps"])
            independent[label] = actual
            for split, metrics in actual["splits"].items():
                for metric, value in metrics.items():
                    expected = observed["splits"][split][metric]
                    errors[f"{label}.{split}.{metric}"] = abs(value - expected)
                    require(
                        value == expected
                        if metric == "routing_accuracy"
                        else math.isclose(value, expected, abs_tol=5e-6, rel_tol=3e-5),
                        str((case["label"], label, split, metric, value, expected)),
                    )
            errors[f"{label}.query_gradient_l2"] = abs(
                actual["query_gradient_l2"] - observed["query_gradient_l2"]
            )
        metrics = independent["final"]["splits"]["heldout"]
        success = (
            metrics["mse"] <= protocol["success"]["heldout_mse_at_most"]
            and metrics["routing_accuracy"]
            >= protocol["success"]["heldout_routing_at_least"]
        )
        require(success == record["success"], "Outcome mismatch")
        record["replay"] = {
            "state": state_check,
            "sampler_verified": True,
            "independent_fp64": independent,
            "absolute_errors": errors,
            "initial_sha256": sha256(path / "initial.pt"),
            "conditioned_initial_sha256": sha256(path / "conditioned_initial.pt"),
        }
        print(json.dumps({"label": case["label"], "success": success}), flush=True)
    report["replay_scope"] = (
        "CPU FP64 physical-query equations for all30 initial/conditioned/final states; "
        "no model or optimizer forward in metric reference; exact saved source state "
        "and sampler checks. Metrics at abs5e-6/rel3e-5, routing counts exact; "
        "gradient differences reported, not thresholded near zero."
    )
    report["training_source_commit"] = "ec39481"
    report["raw_results_sha256"] = sha256(directory / "results.json")
    report["runtime_note"] = (
        "Wall times include compilation and evaluation. The mutated state kernel "
        "compiled but skipped CUDA graph capture; warnings retained."
    )
    return copy.deepcopy(report)
