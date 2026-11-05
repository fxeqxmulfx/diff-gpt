"""Independent LeRoPE optimizer, capture and added-capacity verification."""

from __future__ import annotations

import copy

import pytest
import torch
from torch.testing import assert_close

from research.current_grokking.data import arithmetic
from research.current_grokking.protocol import Case
from research.current_grokking.train import Stepper
from research.early99.experiments.factories import (
    Configuration,
    FactoryStepper,
    configured_factories,
)
from research.early99.experiments.initialization import (
    parameter_metadata,
    verify_paired_initialization,
)


@pytest.mark.parametrize(
    "model_variant",
    [
        "peri-rms-embedding-scaled-lerope",
        "peri-rms-embedding-scaled-ssmax-lerope",
        "peri-rms-embedding-scaled-ssmax-lerope-log-angle",
    ],
)
def test_pairing_allows_only_declared_extra_frequency_vector(
    model_variant: str,
) -> None:
    """New capacity is explicit; extra or changed parent tensors still fail pairing."""
    case = Case("pairing", "arithmetic", {"operation": "div", "prime": 7}, width=16)
    task = arithmetic(case, "cpu")
    configuration = Configuration("lerope-check", 2, model_variant=model_variant)
    torch.manual_seed(4)
    base = configured_factories(Configuration()).model(task, case, "cpu")
    torch.manual_seed(4)
    new = configured_factories(configuration).model(task, case, "cpu")
    metadata = {"case": {"width": case.width, "heads": case.heads}}
    base_metadata = metadata | parameter_metadata(base.model)
    new_metadata = (
        metadata
        | parameter_metadata(new.model)
        | {"experiment": configuration.metadata(case)}
    )
    verify_paired_initialization(base_metadata, new_metadata)
    with pytest.raises(ValueError, match="capacity"):
        verify_paired_initialization(
            base_metadata, new_metadata | {"parameters": new_metadata["parameters"] + 1}
        )
    with pytest.raises(ValueError, match="initialization"):
        verify_paired_initialization(
            base_metadata, new_metadata | {"shared_initial_parameters_sha256": "bad"}
        )
    altered = copy.deepcopy(new_metadata)
    altered["added_parameters"]["lerope.log_scales"]["initial_values"][0] = 1.0
    with pytest.raises(ValueError, match="zero frequency vector"):
        verify_paired_initialization(base_metadata, altered)
    with pytest.raises(ValueError, match="Undeclared"):
        verify_paired_initialization(
            base_metadata, new_metadata | {"experiment": Configuration().metadata(case)}
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("loss_gain", [1, 10_000])
@pytest.mark.parametrize(
    "model_variant",
    [
        "peri-rms-embedding-scaled-lerope",
        "peri-rms-embedding-scaled-ssmax-lerope",
        "peri-rms-embedding-scaled-ssmax-lerope-log-angle",
    ],
)
def test_lerope_captured_updates_clip_only_frequencies_and_restore_both_groups(
    loss_gain: int,
    model_variant: str,
) -> None:
    """Match independently clipped native updates before and after graph preparation."""
    torch.compiler.reset()
    case = Case(
        "lerope-capture",
        "arithmetic",
        {"operation": "div", "prime": 7},
        width=32,
        batch=8,
        warmup=3,
    )
    task = arithmetic(case, "cuda")
    configuration = Configuration("lerope-capture", 2, model_variant=model_variant)
    factories = configured_factories(configuration)
    torch.manual_seed(1)
    model = factories.model(task, case, "cuda")
    initial = copy.deepcopy(model.state_dict())
    stepper = FactoryStepper(
        model,
        task,
        case,
        "cuda",
        configuration=configuration,
        factories=factories,
        compiled=True,
    )
    objective = stepper.objective
    # A test-only multiplier makes the frequency clipping active. Scientific
    # losses and original parameter gradients are never scaled this way.
    stepper.objective = lambda batch: objective(batch) * loss_gain
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    stepper.prepare([8, 5])
    assert_close(model.state_dict(), initial, rtol=0, atol=0)
    assert_close(torch.get_rng_state(), cpu_rng, rtol=0, atol=0)
    assert_close(torch.cuda.get_rng_state(), cuda_rng, rtol=0, atol=0)
    assert all(int(state["step"]) == 0 for state in stepper.optimizer.state.values())
    torch.manual_seed(1)
    reference = factories.model(task, case, "cuda")
    eager = Stepper(reference, task, case, "cuda", compiled=False)
    frequency = reference.model.lerope.log_scales
    eager.optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for p in reference.parameters() if p is not frequency],
                "weight_decay": case.weight_decay,
            },
            {"params": [frequency], "weight_decay": 0.0},
        ],
        lr=eager.rate,
        betas=(0.9, 0.98),
        eps=1e-8,
        capturable=True,
        fused=True,
    )
    clipping_seen = False
    sizes = [8, 8, 5, 8, 8, 5, 8]
    for completed, size in enumerate(sizes):
        if completed == len(sizes) - 1:
            saved = copy.deepcopy(stepper.optimizer.state_dict())
            saved_model = copy.deepcopy(model.state_dict())
            saved_rng = torch.cuda.get_rng_state()
            stepper.prepare([8, 5])
            assert_close(stepper.optimizer.state_dict(), saved, rtol=0, atol=0)
            assert_close(model.state_dict(), saved_model, rtol=0, atol=0)
            assert_close(torch.cuda.get_rng_state(), saved_rng, rtol=0, atol=0)
        before_eval = copy.deepcopy(stepper.optimizer.state_dict())
        stepper.evaluate("validation")
        assert_close(stepper.optimizer.state_dict(), before_eval, rtol=0, atol=0)
        stepper.step(torch.arange(size), completed)
        eager.rate.fill_(case.learning_rate * 2 * min(1.0, completed / case.warmup))
        reference.train()
        eager.optimizer.zero_grad(set_to_none=True)
        loss = (
            eager.objective(task.inputs(torch.arange(size, device="cuda"), static=True))
            * loss_gain
        )
        loss.backward()
        raw_norm = frequency.grad.norm()
        clipping_seen |= bool(raw_norm > 1)
        with torch.no_grad():
            frequency.grad.mul_((1 / (raw_norm + 1e-6)).clamp(max=1))
        eager.optimizer.step()
        assert all(
            group["lr"] is stepper.rate for group in stepper.optimizer.param_groups
        )
    torch.cuda.synchronize()
    if loss_gain > 1:
        assert clipping_seen
    assert torch.count_nonzero(model.model.lerope.log_scales) > 0
    assert all(
        int(state["step"]) == len(sizes) for state in stepper.optimizer.state.values()
    )
    assert_close(model.state_dict(), reference.state_dict(), rtol=3e-5, atol=3e-6)
    # Compare moments in original loss units. The stress-only loss multiplier
    # scales unclipped first/second moments by gain/gain^2; frequency moments
    # stay in their clipped units. The scientific gain=1 check is unscaled.
    moment_states = [
        copy.deepcopy(opt.state_dict()) for opt in (stepper.optimizer, eager.optimizer)
    ]
    for state in moment_states:
        for key in state["param_groups"][0]["params"]:
            state["state"][key]["exp_avg"].div_(loss_gain)
            state["state"][key]["exp_avg_sq"].div_(loss_gain**2)
    assert_close(moment_states[0], moment_states[1], rtol=3e-5, atol=3e-6)
