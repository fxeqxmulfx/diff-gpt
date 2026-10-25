"""Validate probe semantics and prove that logging does not change training."""

import json
from typing import TYPE_CHECKING

import pytest
import torch

from benchmarks.grokking.mqar_div97 import (
    RunConfig,
    Task,
    attention,
    build,
    run_experiment,
    score,
    verify_checkpoint,
)
from benchmarks.grokking.mqar_div97_diagnostics import DiagnosticConfig, Diagnostics

if TYPE_CHECKING:
    from pathlib import Path

ATTENTION_TOLERANCE = 1e-6


@pytest.mark.parametrize(
    "settings",
    [{}, {"rope": "10"}, {"labels": "dividend"}],
)
def test_probe_reconstructs_real_attention_without_side_effects(
    tmp_path: Path, settings: dict[str, object]
) -> None:
    """Match SDPA without changing hooks, RNG, weights or gradients."""
    config = RunConfig(width=16, layers=2, heads=2, loads=2, keys=4, **settings)
    task = Task(config)
    model = build(config, "cpu").eval()
    with torch.no_grad():
        for p in model.parameters():
            p.grad = torch.ones_like(p)
    expected_weights = {name: p.clone() for name, p in model.named_parameters()}
    inputs, targets = task.tensors(task.evaluation["test"], "cpu")
    rng = torch.get_rng_state().clone()
    logger = Diagnostics(
        tmp_path, task, inputs, targets, DiagnosticConfig(frames=16, snapshot_every=0)
    )
    measured = logger.measure(model)
    probabilities = attention(model, inputs[:16])
    assert torch.equal(rng, torch.get_rng_state())
    assert not model.training
    for name, p in model.named_parameters():
        torch.testing.assert_close(p, expected_weights[name], atol=0, rtol=0)
        torch.testing.assert_close(p.grad, torch.ones_like(p), atol=0, rtol=0)
    assert not any(module._forward_hooks for module in model.modules())  # noqa: SLF001
    for layer, expected in zip(measured["layers"], probabilities, strict=True):
        assert layer["attention_reconstruction_max_error"] < ATTENTION_TOLERANCE
        torch.testing.assert_close(
            torch.tensor(layer["mean_attention"]), expected.mean(0)
        )
        positions = logger.sources * logger.columns + logger.columns - 1
        for h, head in enumerate(layer["heads"]):
            rows = torch.arange(len(logger.inputs))
            a = expected[rows, h, -1, positions[:, 0]]
            b = expected[rows, h, -1, positions[:, 1]]
            assert head["attention_a_mean"] == pytest.approx(a.mean().item())
            assert head["attention_b_mean"] == pytest.approx(b.mean().item())
            assert head["a_over_b_fraction_mean"] == pytest.approx(
                (a > b).float().mean().item()
            )
            same = positions[:, 0] == positions[:, 1]
            for name, selected in (("same", same), ("distinct", ~same)):
                assert head[f"{name}_frames"] == selected.sum().item()
                if selected.any():
                    assert head[f"{name}_attention_a_minus_b_mean"] == pytest.approx(
                        (a - b)[selected].mean().item(), abs=ATTENTION_TOLERANCE
                    )
                    assert head[f"{name}_a_over_b_fraction_mean"] == pytest.approx(
                        (a > b)[selected].float().mean().item()
                    )
            assert 0 <= head["xsa_removed_energy_fraction_mean"] <= 1 + 1e-6
    json.dumps(measured, allow_nan=False)


def test_peri_stream_diagnostics_measure_the_added_branch(tmp_path: Path) -> None:
    """The logged branch is the one actually added, after output normalization."""
    task = Task(RunConfig(width=16, layers=2, heads=2, loads=2))
    model = build(task.config, "cpu").eval()
    logger = Diagnostics(
        tmp_path,
        task,
        *task.tensors(task.evaluation["test"], "cpu"),
        DiagnosticConfig(frames=16, snapshot_every=0),
    )
    captured = {}

    def before(_module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        captured["before"] = args[0]

    def after(_module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        captured["after"] = args[0]

    hooks = [
        model.blocks[0].register_forward_pre_hook(before),
        model.blocks[0].ln_2.register_forward_pre_hook(after),
    ]
    measured = logger.measure(model)
    for hook in hooks:
        hook.remove()
    streams = measured["layers"][0]["answer_streams"]
    added = captured["after"] - captured["before"]
    rms = added[:, -1].square().mean(-1).sqrt().mean().item()
    assert streams["attention_output_rms"]["mean"] == pytest.approx(rms)
    assert streams["residual_rms"]["mean"] == pytest.approx(1.0, abs=1e-3)
    assert torch.isfinite(added).all()


def test_qk_norm_separates_projection_scale_from_logits(tmp_path: Path) -> None:
    """Scaling raw Q/K changes their logged norms but preserves normalized logits."""
    task = Task(RunConfig(width=16, layers=1, heads=2, loads=2, keys=4))
    model = build(task.config, "cpu").double().eval()
    logger = Diagnostics(
        tmp_path,
        task,
        *task.tensors(task.evaluation["test"], "cpu"),
        DiagnosticConfig(frames=16, snapshot_every=0),
    )
    first = logger.measure(model)["layers"][0]["heads"]
    width = task.config.width
    with torch.no_grad():
        model.blocks[0].sa.c_attn.weight[:width].mul_(2)
        model.blocks[0].sa.c_attn.weight[width : 2 * width].mul_(3)
    second = logger.measure(model)["layers"][0]["heads"]
    for original, scaled in zip(first, second, strict=True):
        assert scaled["q_l2_mean"] == pytest.approx(2 * original["q_l2_mean"])
        assert scaled["k_l2_mean"] == pytest.approx(3 * original["k_l2_mean"])
        assert scaled["a_minus_b_logit_mean"] == pytest.approx(
            original["a_minus_b_logit_mean"], abs=1e-7
        )
        assert scaled["a_minus_b_cosine_mean"] == pytest.approx(
            original["a_minus_b_cosine_mean"], abs=1e-7
        )


@pytest.mark.parametrize("every", [1, 2])
def test_logged_training_matches_unlogged_weights_and_metrics(
    tmp_path: Path, every: int
) -> None:
    """Extra forwards and snapshots preserve the entire training trajectory exactly."""
    config = RunConfig(
        width=8, heads=1, loads=2, keys=4, layers=2, epochs=2, labels="dividend"
    )
    baseline, logged = tmp_path / "baseline", tmp_path / "logged"
    ordinary = run_experiment(config, baseline)
    result = run_experiment(
        config,
        logged,
        logging=DiagnosticConfig(every=every, frames=16, snapshot_every=1),
    )
    expected = torch.load(baseline / "checkpoint.pt", weights_only=True)
    actual = torch.load(logged / "checkpoint.pt", weights_only=True)
    torch.testing.assert_close(actual["model"], expected["model"], atol=0, rtol=0)
    torch.testing.assert_close(
        actual["optimizer"], expected["optimizer"], atol=0, rtol=0
    )
    assert result["metrics"] == ordinary["metrics"]
    assert result["query_metrics"] == ordinary["query_metrics"]
    assert verify_checkpoint(logged)["metrics"] == result["metrics"]
    records = [
        json.loads(line)
        for line in (logged / "diagnostics.jsonl").read_text().splitlines()
    ]
    assert [(r["epoch"], r["state"]) for r in records] == [
        (epoch, state)
        for epoch in range(config.epochs + 1)
        if epoch % every == 0
        for state in ("query", "evaluation")
    ]
    assert all(
        r["optimizer"]["state_to_average_l2"] == 0
        for r in records
        if r["state"] == "evaluation"
    )
    assert any(
        r["optimizer"]["state_to_average_l2"] > 0
        for r in records
        if r["state"] == "query"
    )
    steps = [
        json.loads(line) for line in (logged / "steps.jsonl").read_text().splitlines()
    ]
    assert len(steps) == config.epochs // every * Task(config).updates
    assert "blocks.0.sa.q.weight" in steps[-1]["parameters"]
    assert "blocks.0.sa.k.weight" in steps[-1]["parameters"]
    assert any(p["update_l2"] > 0 for p in steps[-1]["parameters"].values())
    task = Task(config)
    inputs, targets = task.tensors(task.evaluation["test"], "cpu")
    for epoch in range(config.epochs + 1):
        saved = torch.load(
            logged / "snapshots" / f"epoch-{epoch:05d}.pt", weights_only=True
        )
        assert saved["data_protocol"] == config.data_protocol
        model = build(config, "cpu").eval()
        for state, key in (("query", "query_metrics"), ("evaluation", "metrics")):
            model.load_state_dict(saved[f"{state}_model"])
            measured, _ = score(model, inputs, targets)
            assert measured == saved[key]["test"]
    assert (logged / "source/benchmarks/grokking/mqar_div97_diagnostics.py").exists()


@pytest.mark.parametrize(
    "settings", [{"every": 0}, {"frames": 0}, {"snapshot_every": -1}]
)
def test_invalid_diagnostic_settings_fail(settings: dict[str, int]) -> None:
    """Invalid logging intervals cannot start a run."""
    with pytest.raises(ValueError, match="require"):
        DiagnosticConfig(**settings)


def test_unsupported_layout_fails_before_creating_run(tmp_path: Path) -> None:
    """The two-recalled-operand probes reject a directly supplied divisor."""
    directory = tmp_path / "invalid"
    with pytest.raises(ValueError, match="two operands"):
        run_experiment(RunConfig(operands=1), directory, logging=DiagnosticConfig())
    assert not directory.exists()
