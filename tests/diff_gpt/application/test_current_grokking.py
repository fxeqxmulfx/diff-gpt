"""Data/oracle, outcome and state-integrity checks for the current-GPT sweep."""

from typing import TYPE_CHECKING

import pytest
import torch
from torch.testing import assert_close

from diff_gpt.domain.model.gpt import GPT
from research.current_grokking.data import arithmetic
from research.current_grokking.protocol import Case, cases
from research.current_grokking.run import outcome, run
from research.current_grokking.train import Logits, Stepper

if TYPE_CHECKING:
    from pathlib import Path

IGNORE = -100
ARITHMETIC_LABELS = 40
SUITE_VARIANTS = 37
NEW_PROBES = 6
CHECKPOINT_UPDATES = 2
GRAPH_UPDATES = 3


@pytest.mark.parametrize("operation", ["add", "sub", "mul", "div", "half-sum"])
def test_arithmetic_labels_follow_independent_integer_oracle(operation: str) -> None:
    """Every retained row obeys its mathematical rule; split inputs never overlap."""
    task = arithmetic(
        Case("oracle", "arithmetic", {"operation": operation, "prime": 7}), "cpu"
    )
    seen = set()
    for split in ("train", "validation"):
        rows = task.splits[split].rows
        keys = {tuple(map(int, row)) for row in rows[:, 0]}
        assert not seen & keys
        seen |= keys
        for tokens, targets, _ in rows:
            a, b = map(int, tokens)
            expected = {
                "add": (a + b) % 7,
                "sub": (a - b) % 7,
                "mul": a * b % 7,
                "half-sum": (a + b) * 4 % 7,
            }.get(operation)
            if operation == "div":
                expected = next(q for q in range(7) if b * q % 7 == a)
            assert int(targets[0]) == IGNORE
            assert int(targets[1]) == expected
    assert len(seen) == 7 * (6 if operation == "div" else 7)


def test_every_catalog_arithmetic_label_and_suite_variant_is_planned() -> None:
    """The large requested matrix cannot silently become just the easy cases."""
    rows = cases()
    assert (
        len([row for row in rows if row.name.startswith("arithmetic-")])
        == ARITHMETIC_LABELS
    )
    assert len([row for row in rows if row.name.startswith("suite-")]) == SUITE_VARIANTS
    assert len([row for row in rows if row.kind == "probe"]) == NEW_PROBES
    assert {"fashion", "text", "series", "associative", "recurrence"} <= {
        row.kind for row in rows
    }


def test_outcome_distinguishes_fit_delay_fast_learning_and_bounded_miss() -> None:
    """Late fitting alone, early heldout success and a fitted plateau differ."""
    case = Case("outcome", "arithmetic")

    def row(step: int, train: float, heldout: float) -> dict:
        return {
            "step": step,
            "train": {"sequence_accuracy": train},
            "validation": {"sequence_accuracy": heldout},
        }

    plateau = [row(0, 0.01, 0.01), row(250, 1.0, 0.01), row(500, 1.0, 0.02)]
    assert (
        outcome(plateau, case)["classification"] == "target_not_reached_within_budget"
    )
    assert (
        outcome([*plateau, row(750, 1.0, 1.0)], case)["classification"]
        == "delayed_generalization"
    )
    fast = [row(0, 0.01, 0.01), row(250, 1.0, 1.0)]
    assert outcome(fast, case)["classification"] == "generalized_without_low_plateau"
    late_fit = [
        row(0, 0.01, 0.01),
        row(250, 0.4, 0.01),
        row(500, 0.8, 0.01),
        row(750, 1.0, 1.0),
    ]
    assert outcome(late_fit, case)["classification"] != "delayed_generalization"


def test_cpu_checkpoint_preserves_sampler_moments_and_completed_updates(
    tmp_path: Path,
) -> None:
    """No best restoration or hidden warmup step may enter a retained endpoint."""
    case = Case(
        "checkpoint-check",
        "arithmetic",
        {"operation": "div", "prime": 7},
        width=16,
        heads=2,
        batch=8,
        max_updates=CHECKPOINT_UPDATES,
        observe=1,
        after_target=0,
    )
    result = run(case, 1, tmp_path, "cpu", compiled=False)
    saved = torch.load(tmp_path / case.name / "seed1/checkpoint.pt", weights_only=True)
    assert result["completed"] == CHECKPOINT_UPDATES
    assert all(
        int(state["step"]) == CHECKPOINT_UPDATES
        for state in saved["optimizer"]["state"].values()
    )
    assert saved["sampler"]["cursor"] == CHECKPOINT_UPDATES * case.batch
    replay = run(case, 1, tmp_path, "cpu", resume=True, compiled=False)
    assert replay["endpoint"] == result["endpoint"]
    assert replay["first_target"] == result["first_target"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_graph_prepare_is_neutral_and_updates_match_issued_operations() -> None:
    """Verify the actual public-GPT/native-AdamW graph rather than a toy model."""
    case = Case(
        "graph-check",
        "arithmetic",
        {"operation": "div", "prime": 7},
        width=32,
        batch=8,
        warmup=0,
    )
    task = arithmetic(case, "cuda")
    torch.manual_seed(1)
    model = Logits(
        GPT(
            7, 32, block_size=2, n_layer=1, use_checkpoint=False, pad_vocab_size_to=1
        ).cuda()
    )
    initial = {name: value.clone() for name, value in model.state_dict().items()}
    stepper = Stepper(model, task, case, "cuda", compiled=True)
    stepper.prepare([8])
    assert_close(model.state_dict(), initial, rtol=0, atol=0)
    assert all(int(state["step"]) == 0 for state in stepper.optimizer.state.values())
    torch.manual_seed(1)
    reference = Logits(
        GPT(
            7, 32, block_size=2, n_layer=1, use_checkpoint=False, pad_vocab_size_to=1
        ).cuda()
    )
    eager = Stepper(reference, task, case, "cuda", compiled=False)
    eager.rate.fill_(case.learning_rate)
    indices = torch.arange(8)
    for step in range(GRAPH_UPDATES):
        stepper.step(indices, step)
        eager.update(indices.cuda())
    torch.cuda.synchronize()
    assert all(
        int(state["step"]) == GRAPH_UPDATES
        for state in stepper.optimizer.state.values()
    )
    assert_close(model.state_dict(), reference.state_dict(), rtol=2e-5, atol=2e-6)
