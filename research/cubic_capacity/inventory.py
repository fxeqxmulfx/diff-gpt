"""Verify all declared initialized models, task data and paired sample orders."""

from __future__ import annotations

import json

import torch

# isort: off
from research.current_grokking import run as _bootstrap  # noqa: F401
from lab.infrastructure.benchmarks.samplers import EpochSampler, gather
# isort: on

from research.cubic_capacity.factories import (
    STUDY,
    case_for,
    configuration,
    factories,
    plan,
)
from research.current_grokking.data import build, tensor_digest
from research.early99.experiments.initialization import parameter_metadata
from research.early99_reporting.round2_audit import require


def inventory() -> list:
    """Check every size/arm/seed with no optimizer update or held-out measurement."""
    search = plan()
    reference, order_reference, records = {}, {}, []
    for specification in search["configurations"]:
        case = case_for(specification)
        task = build(case, "cpu")
        require(
            task.fingerprints == search["task"]["data_fingerprints"],
            "Changed data fingerprints",
        )
        shape = next(
            row for row in search["shapes"] if row["name"] == specification["size"]
        )
        cfg = configuration(specification)
        require(
            cfg.learning_rate_at(case, 0, len(task.splits["train"])) == 0,
            "Warmup must begin at zero",
        )
        require(
            cfg.learning_rate_at(case, case.warmup, len(task.splits["train"]))
            == case.learning_rate * specification["learning_rate_multiplier"],
            "Incorrect peak rate",
        )
        require(
            cfg.learning_rate_at(case, case.max_updates, len(task.splits["train"]))
            == case.learning_rate * specification["learning_rate_multiplier"],
            "Rate must remain constant",
        )
        for seed in search["seeds"]:
            torch.manual_seed(seed)
            model = factories(specification).model(task, case, "cpu")
            gpt = model.model
            parameters = [
                (name, tensor_digest(value)) for name, value in gpt.named_parameters()
            ]
            buffers = [
                (name, tensor_digest(value))
                for name, value in gpt.named_buffers()
                if not name.endswith("inverse_width_gain")
            ]
            key = (specification["size"], seed)
            state = (parameters, buffers, torch.get_rng_state().clone())
            if key not in reference:
                reference[key] = state
            expected = reference[key]
            require(
                state[:2] == expected[:2] and torch.equal(state[2], expected[2]),
                "Architecture/rate changed initialization, buffers, ordering or RNG",
            )
            require(
                gpt.token_embedding_table.weight is gpt.lm_head.weight,
                "Tied weights changed",
            )
            metadata = parameter_metadata(gpt)
            require(
                metadata["parameters"] == shape["unique_parameters"],
                "Unexpected parameter count",
            )
            for block in gpt.blocks:
                require(block.sa.head_dim == shape["head_width"], "Head width changed")
                require(
                    block.sa.q_norm.eps is None and block.sa.k_norm.eps is None,
                    "Q/K RMS epsilon changed",
                )
                gain = block.sa.attention_gain()
                if specification["architecture"] == "inverse_width":
                    require(
                        torch.equal(
                            gain, torch.full_like(gain, 1 / shape["head_width"])
                        ),
                        "Inverse gain changed",
                    )
                    require(
                        block.sa.c_proj.residual_scale
                        == block.ffwd.c_proj.residual_scale
                        == (2 * case.layers) ** -0.5,
                        "Depth coefficient changed",
                    )
                else:
                    require(
                        torch.equal(
                            gain.flatten(),
                            block.sa.log_alpha.exp() / shape["head_width"],
                        ),
                        "Original gain changed",
                    )
                    require(
                        not hasattr(block.sa.c_proj, "residual_scale"),
                        "Original branch was scaled",
                    )
            sampler = EpochSampler(
                len(task.splits["train"]), case.batch, wrap=False, seed=seed + 10000
            )
            batches = [
                gather(sampler.next()[0])
                for _ in range(
                    2 * ((len(task.splits["train"]) + case.batch - 1) // case.batch)
                )
            ]
            indices = torch.cat(batches)
            for epoch in indices.chunk(2):
                require(
                    torch.equal(
                        epoch.sort().values, torch.arange(len(task.splits["train"]))
                    ),
                    "Sampler duplicated or omitted an example",
                )
            order = tensor_digest(indices)
            require(
                order_reference.setdefault(seed, order) == order,
                "Capacity/architecture changed the paired sample order",
            )
            records.append(
                {
                    "name": specification["name"],
                    "size": specification["size"],
                    "arm": specification["arm"],
                    "seed": seed,
                    "case_width": case.width,
                    "case_layers": case.layers,
                    "head_width": shape["head_width"],
                    **metadata,
                    "data_fingerprints": task.fingerprints,
                    "first_two_epochs_indices_sha256": order,
                    "smallest_batch": min(len(batch) for batch in batches),
                    "initial_weights_buffers_rng_and_tying_verified": True,
                }
            )
    require(
        len(records) == search["new_training_cells"],
        "Incomplete initialization inventory",
    )
    return records


if __name__ == "__main__":
    rows = inventory()
    output = STUDY / "results/inventory.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(
        json.dumps({"complete": True, "optimizer_updates": 0, "rows": rows}, indent=2)
        + "\n"
    )
    print(json.dumps({"verified_initializations": len(rows)}))
