"""Small paired retrieval training under predeclared Q/K magnitude stress."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from diff_gpt.domain.model.rms_norm import rms_norm

ROOT = Path(__file__).resolve().parents[2]
PLAN = Path(__file__).with_name("plan.json")


def require(message: str, *, condition: bool) -> None:
    """Keep scientific checks enabled even if Python optimizations are requested."""
    if not condition:
        raise ValueError(message)


def digest(tensors: dict[str, torch.Tensor]) -> str:
    """Hash named, ordered tensor values without depending on archive metadata."""
    result = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        result.update(name.encode())
        result.update(value.contiguous().numpy().tobytes())
    return result.hexdigest()


def data(count: int, settings: dict, rng: torch.Generator) -> dict[str, torch.Tensor]:
    """Generate independent continuous keys, permuted values and a repeated query."""
    keys = torch.randn(count, settings["keys"], settings["dimension"], generator=rng)
    selected = torch.randint(settings["keys"], (count,), generator=rng)
    values = torch.rand(count, settings["keys"], generator=rng).argsort(-1)
    rows = torch.arange(count)
    return {
        "keys": keys,
        "query": keys[rows, selected].clone(),
        "values": F.one_hot(values, settings["keys"]).float(),
        "target": values[rows, selected],
    }


class Retrieval(nn.Module):
    """Learn only two projections; normalization and stress add no parameters."""

    def __init__(self, settings: dict, scale: float, *, normalized: bool) -> None:
        """Keep parameter draws identical in every paired scale/norm condition."""
        super().__init__()
        dimension = settings["dimension"]
        self.query = nn.Linear(dimension, dimension, bias=False)
        self.key = nn.Linear(dimension, dimension, bias=False)
        self.scale, self.normalized = scale, normalized
        self.gain = math.log(settings["keys"]) / math.sqrt(dimension)
        self.readout_scale = settings["readout_scale"]

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply scale before optional RMS, then SSMax and a fixed value readout."""
        query = self.query(batch["query"]) * self.scale
        keys = self.key(batch["keys"]) * self.scale
        if self.normalized:
            query, keys = rms_norm(query), rms_norm(keys)
        scores = (query[:, None] * keys).sum(-1) * self.gain
        weights = scores.softmax(-1)
        logits = (weights[:, :, None] * batch["values"]).sum(1) * self.readout_scale
        return logits, weights, scores


@torch.inference_mode()
def evaluate(model: Retrieval, split: dict) -> dict:
    """Evaluate the entire fixed split, including saturation diagnostics."""
    logits, weights, scores = model(split)
    return {
        "accuracy": (logits.argmax(-1) == split["target"]).float().mean().item(),
        "loss": F.cross_entropy(logits, split["target"]).item(),
        "maximum_absolute_score": scores.abs().max().item(),
        "mean_attention_entropy": -(weights * weights.clamp_min(1e-38).log())
        .sum(-1)
        .mean()
        .item(),
        "nearly_one_hot_fraction": (weights.max(-1).values > 1 - 1e-6)
        .float()
        .mean()
        .item(),
    }


def checks(settings: dict, splits: dict) -> dict:
    """Check an independent distance oracle and the finite-epsilon scale equation."""
    for split in splits.values():
        nearest = (split["keys"] - split["query"][:, None]).square().sum(-1).argmin(-1)
        actual = split["values"][torch.arange(len(nearest)), nearest].argmax(-1)
        require(
            "Distance oracle disagrees", condition=torch.equal(actual, split["target"])
        )
        require(
            "Values must be a permutation of the one-hot classes",
            condition=torch.equal(
                split["values"].sum(1), torch.ones_like(split["values"][:, 0])
            ),
        )
    torch.manual_seed(123)
    value = torch.randn(
        7, settings["dimension"], dtype=torch.float64, requires_grad=True
    )
    eps = torch.finfo(torch.float64).eps
    for scale in settings["projection_scales"]:
        actual = rms_norm(value * scale)
        expected = (
            value / (value.square().mean(-1, keepdim=True) + eps / scale**2).sqrt()
        )
        torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)
        upstream = torch.arange(value.numel(), dtype=value.dtype).reshape_as(value)
        ga = torch.autograd.grad((actual * upstream).sum(), value, retain_graph=True)[0]
        ge = torch.autograd.grad((expected * upstream).sum(), value, retain_graph=True)[
            0
        ]
        torch.testing.assert_close(ga, ge, rtol=1e-12, atol=1e-12)
    sample = {name: tensor[:7] for name, tensor in splits["train"].items()}
    for normalized in [True, False]:
        for scale in [1.0, 64.0]:
            model = Retrieval(settings, scale, normalized=normalized)
            query = F.linear(sample["query"], model.query.weight) * scale
            keys = F.linear(sample["keys"], model.key.weight) * scale
            if normalized:
                query = F.rms_norm(query, (settings["dimension"],))
                keys = F.rms_norm(keys, (settings["dimension"],))
            expected_scores = torch.bmm(query[:, None], keys.transpose(1, 2))[:, 0]
            expected_scores = (
                expected_scores
                * math.log(settings["keys"])
                / math.sqrt(settings["dimension"])
            )
            logits, weights, scores = model(sample)
            torch.testing.assert_close(scores, expected_scores, rtol=1e-5, atol=2e-3)
            torch.testing.assert_close(
                weights, expected_scores.softmax(-1), rtol=1e-5, atol=1e-5
            )
            expected_logits = (
                torch.bmm(weights[:, None], sample["values"])[:, 0]
                * settings["readout_scale"]
            )
            torch.testing.assert_close(logits, expected_logits)
    return {
        "distance_oracle": "passed",
        "value_permutations": "passed",
        "independent_rms_scale_forward_and_backward": "passed",
        "independent_attention_and_fixed_readout": "passed",
    }


def run_cell(
    settings: dict,
    splits: dict,
    output: Path,
    seed: int,
    scale: float,
    *,
    normalized: bool,
) -> dict:
    """Run one immutable cell and preserve the complete observation history."""
    name = f"{'qk' if normalized else 'noqk'}-scale{scale:g}-seed{seed}"
    cell = output / name
    if (cell / "result.json").exists():
        return json.loads((cell / "result.json").read_text())
    if cell.exists():
        message = f"Incomplete preserved cell requires explicit recovery: {cell}"
        raise RuntimeError(message)
    cell.mkdir()
    torch.manual_seed(seed)
    model = Retrieval(settings, scale, normalized=normalized)
    initial = digest(model.state_dict())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings["learning_rate"],
        betas=tuple(settings["betas"]),
        eps=settings["optimizer_epsilon"],
        weight_decay=settings["weight_decay"],
        foreach=False,
    )
    rng = torch.Generator().manual_seed(10000 + seed)
    train = splits["train"]
    order = torch.empty(0, dtype=torch.int64)
    history = []
    zero_gradients = 0
    gradient_norm = None
    start = time.perf_counter()
    status = "budget_exhausted"
    for step in range(settings["updates"] + 1):
        if step % settings["observe_updates"] == 0:
            record = {
                "step": step,
                "epochs": step * settings["batch"] / settings["train_examples"],
                "train": evaluate(model, train),
                "heldout": evaluate(model, splits["heldout"]),
                "last_gradient_l2": gradient_norm,
                "seconds": time.perf_counter() - start,
            }
            history.append(record)
            with (cell / "history.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            if not all(
                math.isfinite(v)
                for part in ["train", "heldout"]
                for v in record[part].values()
            ):
                status = "numerical_failure"
                break
            if record["heldout"]["accuracy"] >= settings["target"]:
                status = "passed"
                break
        if step == settings["updates"]:
            break
        offset = step * settings["batch"] % settings["train_examples"]
        if offset == 0:
            order = torch.randperm(settings["train_examples"], generator=rng)
        indices = order[offset : offset + settings["batch"]]
        batch = {k: v[indices] for k, v in train.items()}
        optimizer.param_groups[0]["lr"] = settings["learning_rate"] * min(
            1.0, step / settings["warmup_updates"]
        )
        optimizer.zero_grad(set_to_none=True)
        logits, _, _ = model(batch)
        F.cross_entropy(logits, batch["target"]).backward()
        gradient_norm = math.sqrt(
            sum(p.grad.double().square().sum().item() for p in model.parameters())
        )
        zero_gradients += gradient_norm == 0
        if not math.isfinite(gradient_norm):
            status = "numerical_gradient_failure"
            break
        optimizer.step()
    result = {
        "name": name,
        "seed": seed,
        "scale": scale,
        "qk_rms": normalized,
        "status": status,
        "completed_updates": step,
        "first99_epochs": history[-1]["epochs"] if status == "passed" else None,
        "initial_parameters_sha256": initial,
        "parameters": sum(p.numel() for p in model.parameters()),
        "initial": history[0],
        "final": history[-1],
        "exact_zero_gradient_updates": zero_gradients,
        "seconds": time.perf_counter() - start,
    }
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "result": result,
            "history": history,
            "sampler_rng": rng.get_state(),
        },
        cell / "checkpoint.pt",
    )
    (cell / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return result


def main() -> None:
    """Freeze all settings and source hashes before the paired experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/qk-scale-stress/v1")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("highest")
    settings = json.loads(PLAN.read_text())
    rng = torch.Generator().manual_seed(settings["data_seed"])
    splits = {
        label: data(settings[key], settings, rng)
        for label, key in [("train", "train_examples"), ("heldout", "heldout_examples")]
    }
    validation = checks(settings, splits)
    sources = [Path(__file__), ROOT / "src/diff_gpt/domain/model/rms_norm.py"]
    manifest = {
        "settings": settings,
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
        "data_sha256": {label: digest(split) for label, split in splits.items()},
        "validation": validation,
        "torch": torch.__version__,
        "device": "cpu",
        "dtype": "float32",
    }
    if not args.run:
        print(json.dumps(manifest, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    plan = args.output / "plan.json"
    if plan.exists():
        require(
            "Frozen stress source/settings changed",
            condition=json.loads(plan.read_text()) == manifest,
        )
    else:
        plan.write_text(json.dumps(manifest, indent=2) + "\n")
        torch.save(splits, args.output / "data.pt")
        for source in sources:
            destination = args.output / "source" / source.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())
    rows = []
    for scale in settings["projection_scales"]:
        for seed in settings["model_seeds"]:
            rows.extend(
                run_cell(
                    settings,
                    splits,
                    args.output,
                    seed,
                    scale,
                    normalized=normalized,
                )
                for normalized in settings["qk_rms"]
            )
    for seed in settings["model_seeds"]:
        require(
            "Paired cells changed initial parameters",
            condition=len(
                {r["initial_parameters_sha256"] for r in rows if r["seed"] == seed}
            )
            == 1,
        )
    summary = {
        "manifest": manifest,
        "runs": rows,
        "expected_cells": 24,
        "all_initialization_pairs_match": True,
        "interpretation": settings["interpretation"],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
