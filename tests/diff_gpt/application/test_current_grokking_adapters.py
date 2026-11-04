"""Independent codec, numeric-generation and series-scoring regressions."""

from dataclasses import fields

import pytest
import torch

from diff_gpt.application.grokking import series as original_series
from diff_gpt.domain.model.gpt import GPT
from research.current_grokking.adapters import RecurrenceTask, SeriesTask
from research.current_grokking.data import build
from research.current_grokking.protocol import Case, cases
from research.current_grokking.train import Logits, Stepper


@pytest.mark.parametrize("tokenizer", ["lgt", "mash", "mash-half"])
def test_temporal_adapter_uses_the_public_codec_without_future_grid_fit(
    tokenizer: str,
) -> None:
    """MASH-half must not reuse the old integer-difference shortcut."""
    case = Case(
        "temporal-codec-check",
        "recurrence",
        {"task": "fibonacci", "order": 1, "context": 2, "tokenizer": tokenizer},
    )
    task = RecurrenceTask(case, "cpu")
    assert not set(task.metadata["train_orbits"]) & set(
        task.metadata["validation_orbits"]
    )
    assert "actual public encode" in task.metadata["encoder"]
    rows = task.splits["validation"].rows
    assert rows.shape[-1] == case.data["context"]
    assert torch.all((rows[:, 1, -1] >= 0) & (rows[:, 1, -1] < task.vocab))
    assert torch.isfinite(task.decoder).all()
    indices = rows[:, 2, 0]
    decoded = torch.round(
        task.decoder[indices, 0]
        + (rows[:, 1, -1] - task.vocab // 2) * task.decoder[indices, 1]
    )
    torch.testing.assert_close(
        decoded, task.raw[indices, -1], rtol=0, atol=0, check_dtype=False
    )


def test_series_scores_match_the_existing_independent_evaluator() -> None:
    """Proper CE, decoded risk and known-floor skill retain their definitions."""
    case = Case(
        "series-score-check",
        "series",
        {
            "dataset": "ar",
            "series_length": 512,
            "max_train": 32,
            "max_eval": 16,
            "context": 1,
        },
        metric="skill",
    )
    task = SeriesTask(case, "cpu")
    assert task.baseline == 0.0
    model = Logits(
        GPT(
            task.vocab,
            16,
            block_size=task.context,
            n_head=2,
            n_layer=1,
            use_checkpoint=False,
            pad_vocab_size_to=1,
        )
    )
    stepper = Stepper(model, task, case, "cpu", compiled=False)
    actual = stepper.evaluate("validation")
    config = original_series.Config(data=task.config, width=16, layers=1, heads=2)
    expected = original_series.evaluate(
        model.model,
        {
            field.name: torch.as_tensor(value.copy())
            for field in fields(task.dataset.splits["validation"])
            if (value := getattr(task.dataset.splits["validation"], field.name))
            is not None
        },
        task.dataset,
        "validation",
        config,
    )
    for key in ("loss", "accuracy", "mse", "mae", "skill", "bayes_loss"):
        if expected[key] is None:
            assert actual[key] is None
        else:
            assert actual[key] == pytest.approx(expected[key], rel=2e-6, abs=1e-7)


class AnswerOracle:
    """A scorer fixture only; its outputs are never reported as trained GPT results."""

    def __init__(self, task: object, *, early_eos: bool) -> None:
        """Look up held-out answers solely to verify the evaluation implementation."""
        self.vocab, self.early_eos = task.vocab, early_eos
        examples = task.splits["validation"].examples
        self.width = len(examples[0].prompt)
        self.eos = examples[0].answer[-1]
        self.answers = {example.prompt: example.answer for example in examples}

    def __call__(self, tokens: torch.Tensor) -> torch.Tensor:
        """Emit the oracle sequence, or intentionally stop before any numeric token."""
        logits = torch.full((*tokens.shape, self.vocab), -1000.0)
        for row in range(len(tokens)):
            key = tuple(map(int, tokens[row, : self.width]))
            answer = self.answers[key]
            for position in range(self.width - 1, tokens.shape[1]):
                index = min(position - self.width + 1, len(answer) - 1)
                token = self.eos if self.early_eos else answer[index]
                logits[row, position, token] = 1000
        return logits


def test_numeric_proxy_scores_free_generation_and_penalizes_early_eos() -> None:
    """A missing vector must fail MSE rather than receive an artificial zero."""
    case = next(case for case in cases() if case.name == "gpt-probe-linear-transport")
    task = build(case, "cpu")
    rows = task.splits["validation"]
    for early, expected_accuracy in ((False, 1.0), (True, 0.0)):
        oracle = AnswerOracle(task, early_eos=early)
        sums = task.accumulator()
        task.accumulate(oracle, rows[: len(rows)], sums, static=True)
        result = task.metrics(sums.tolist(), rows)
        assert result["sequence_accuracy"] == expected_accuracy
        if early:
            assert result["numeric_output_coverage"] == 0
            assert result["mse"] > 1
        else:
            assert result["numeric_output_coverage"] == 1
            assert result["mse"] == pytest.approx(0.0)


def test_synthetic_selection_excludes_repeated_training_prompts() -> None:
    """A high score on repeated small-domain inputs cannot count as novel success."""
    case = next(case for case in cases() if case.name == "suite-parity-none")
    task = build(case, "cpu")
    train = {
        example.prompt or example.tokens for example in task.splits["train"].examples
    }
    heldout = {
        example.prompt or example.tokens
        for example in task.splits[task.selection].examples
    }
    assert task.selection.endswith("/novel")
    assert not train & heldout
