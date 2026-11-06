"""Verify the gain-only intervention against every retained inverse-width seed."""

from __future__ import annotations

import json

import torch
from torch.testing import assert_close

from research.cubic_usual_gain.factories import ROOT, STUDY, case_for, factories, plan
from research.current_grokking.data import build, tensor_digest
from research.early99.experiments.initialization import parameter_metadata
from research.early99.experiments.scalable_softmax import ScalableSoftmaxAttention
from research.early99_dimension_gain.factories import dimension_model
from research.early99_reporting.round2_audit import read, require


def inventory(seed: int) -> dict:
    """Check tensors, RNG, tying and forward/backward parity after restoring gain."""
    declared = plan()
    require(seed in declared["seeds"], "Undeclared seed")
    case = case_for()
    task = build(case, "cpu")
    torch.manual_seed(seed)
    model = factories().model(task, case, "cpu")
    rng = torch.get_rng_state().clone()
    metadata = parameter_metadata(model.model)
    control = read(
        ROOT
        / declared["controls"]["normalized_lr2"]
        / case.name
        / f"seed{seed}"
        / "protocol.json"
    )
    require(
        all(control[key] == value for key, value in metadata.items()),
        "Changed paired initialization",
    )
    require(task.fingerprints == declared["task"]["data_fingerprints"], "Changed data")
    torch.manual_seed(seed)
    inverse = dimension_model(task, case, "cpu")
    require(torch.equal(rng, torch.get_rng_state()), "Changed RNG consumption")
    actual_params = [
        (name, tensor_digest(p)) for name, p in model.model.named_parameters()
    ]
    control_params = [
        (name, tensor_digest(p)) for name, p in inverse.model.named_parameters()
    ]
    require(actual_params == control_params, "Changed parameter names/order/values")
    buffers = dict(inverse.model.named_buffers())
    require(
        all(
            torch.equal(value, buffers[name])
            for name, value in model.model.named_buffers()
        ),
        "Changed shared buffers",
    )
    require(
        model.model.token_embedding_table.weight is model.model.lm_head.weight,
        "Untied weights",
    )
    actual_gains = []
    for block, parent in zip(model.model.blocks, inverse.model.blocks, strict=True):
        require(
            block.sa.use_qk_norm
            and block.sa.q_norm.eps is None
            and block.sa.k_norm.eps is None,
            "Changed Q/K normalization",
        )
        require(
            block.sa.c_proj.residual_scale
            == block.ffwd.c_proj.residual_scale
            == (2 * case.layers) ** -0.5,
            "Changed residual scale",
        )
        gain = block.sa.attention_gain()
        require(
            torch.equal(gain.flatten(), block.sa.log_alpha.exp() / block.sa.head_dim),
            "Changed usual gain formula",
        )
        actual_gains.append(gain.flatten().tolist())
        # Remove precisely the inverse-gain override from the paired control.
        parent.sa = ScalableSoftmaxAttention(parent.sa, normalize_qk=True)
    indices = torch.arange(64)
    batch = task.inputs(indices, static=True)
    expected, _ = task.forward(model, batch)
    restored, _ = task.forward(inverse, batch)
    assert_close(expected, restored, rtol=0, atol=0)
    torch.manual_seed(11000 + seed)
    cotangent = torch.randn_like(expected)
    left = torch.autograd.grad(expected, list(model.model.parameters()), cotangent)
    right = torch.autograd.grad(restored, list(inverse.model.parameters()), cotangent)
    for value, reference in zip(left, right, strict=True):
        assert_close(value, reference, rtol=0, atol=0)
    return {
        "seed": seed,
        **metadata,
        "data_fingerprints": task.fingerprints,
        "actual_gains": actual_gains,
        "inverse_gain": 1 / (case.width // case.heads),
        "matched_weights_rng_buffers_and_tying": True,
        "restored_inverse_forward_all_parameter_gradients_exact": True,
        "optimizer_updates": 0,
    }


if __name__ == "__main__":
    torch.set_num_threads(1)
    rows = [inventory(seed) for seed in plan()["seeds"]]
    destination = STUDY / "results/inventory.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(
        json.dumps({"complete": True, "rows": rows}, indent=2) + "\n"
    )
    print(json.dumps({"verified_initializations": len(rows), "optimizer_updates": 0}))
