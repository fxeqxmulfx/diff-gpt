"""Keep fresh-seed confirmation from silently changing the accepted experiment."""

from copy import deepcopy
from pathlib import Path

import pytest

import research.recall_ablation.run as recall_run
from research.recall_ablation.confirm import training_signature
from research.recall_ablation.run import validate_confirmation


def _protocols() -> tuple[dict, dict]:
    candidate = {
        "stage": "prefix-stable",
        "seed": 0,
        "batch_seed": 10000,
        "initial_parameters_sha256": "new-initialization",
        "git_commit": "new-driver",
        "features": ["passing-architecture"],
        "optimizer": "AdamW",
        "learning_rate": 3e-4,
        "max_updates": 3000,
        "threads": 4,
        "compiled": True,
        "split_fingerprints": {"train": "train-hash", "validation": "validation-hash"},
        "embedding_normalization": None,
        "sources": {
            "src/model.py": "model-hash",
            "research/recall_ablation/run.py": "new-driver",
        },
    }
    reference = deepcopy(candidate)
    reference.update(
        seed=1,
        batch_seed=10001,
        initial_parameters_sha256="reference-initialization",
        git_commit="old-driver",
        status="passed",
        first99=2600,
    )
    reference["sources"]["research/recall_ablation/run.py"] = "old-driver"
    return candidate, reference


def test_confirmation_allows_new_seed_initialization_and_its_sampler() -> None:
    """Fresh weights and seed-derived shuffling are the intended changes."""
    candidate, reference = _protocols()
    validate_confirmation(candidate, reference)


@pytest.mark.parametrize(
    "field",
    [
        "features",
        "optimizer",
        "learning_rate",
        "max_updates",
        "threads",
        "compiled",
        "split_fingerprints",
    ],
)
def test_confirmation_rejects_recipe_or_architecture_changes(field: str) -> None:
    """A positive reference must not authorize another model or training budget."""
    candidate, reference = _protocols()
    candidate[field] = "changed"
    with pytest.raises(ValueError, match=field):
        validate_confirmation(candidate, reference)


def test_confirmation_requires_a_passed_reference() -> None:
    """A censored feature cannot advance to final confirmation."""
    candidate, reference = _protocols()
    reference.update(status="censored", first99=None)
    with pytest.raises(ValueError, match="passed reference"):
        validate_confirmation(candidate, reference)


def test_confirmation_requires_the_same_stage() -> None:
    """A passing architecture does not authorize a different stage label."""
    candidate, reference = _protocols()
    candidate["stage"] = "different-stage"
    with pytest.raises(ValueError, match="same stage"):
        validate_confirmation(candidate, reference)


def test_confirmation_rejects_changed_model_source() -> None:
    """Keep the executable model/data sources identical to the passed screen."""
    candidate, reference = _protocols()
    candidate["sources"]["src/model.py"] = "changed"
    with pytest.raises(ValueError, match="model/data source"):
        validate_confirmation(candidate, reference)


def test_confirmation_rejects_missing_even_optional_protocol_fields() -> None:
    """A missing field is not equivalent to an explicitly recorded None."""
    candidate, reference = _protocols()
    del reference["embedding_normalization"]
    with pytest.raises(ValueError, match="embedding_normalization"):
        validate_confirmation(candidate, reference)


def test_confirmation_requires_the_original_sampler_seed_rule() -> None:
    """Do not accidentally reuse the reference seed's shuffle for a new model."""
    candidate, reference = _protocols()
    candidate["batch_seed"] = reference["batch_seed"]
    with pytest.raises(ValueError, match="batch_seed"):
        validate_confirmation(candidate, reference)


def test_confirmation_signature_allows_provenance_changes_only() -> None:
    """Eligibility checks can change without changing the training calculation."""
    source = Path(recall_run.__file__).read_text()
    changed = source.replace(
        "confirmation requires a passed reference", "a different eligibility message"
    )
    assert source != changed
    assert training_signature(source) == training_signature(changed)


def test_confirmation_signature_rejects_training_changes() -> None:
    """A different learning-rate constant cannot masquerade as the same loop."""
    source = Path(recall_run.__file__).read_text()
    changed = source.replace("RATE = 3e-4", "RATE = 1e-4")
    assert source != changed
    assert training_signature(source) != training_signature(changed)
