"""Check isolated architecture equations, initialization and actual training paths."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F
from torch.testing import assert_close

from diff_gpt.domain.model.gpt import GPT
from diff_gpt.domain.model.kv_cache import KVCache
from diff_gpt.domain.model.rope import apply_rope

# isort: off
# Import the frozen external-lab bootstrap before importing its samplers.
from research.current_grokking import run as _bootstrap  # noqa: F401
from lab.infrastructure.benchmarks.samplers import EpochSampler, gather

# isort: on
from research.current_grokking.data import arithmetic, build
from research.current_grokking.protocol import Case
from research.current_grokking.train import Stepper
from research.early99.experiments.factories import baseline_model
from research.early99.protocol import selected, specification
from research.early99_core_ablations import trainer
from research.early99_core_ablations.factories import (
    MECHANISMS,
    Configuration,
    FactoryStepper,
    configured_factories,
    install_mechanism,
)
from research.early99_core_ablations.run import prepare_output
from research.early99_cosine import trainer as native_trainer
from research.early99_execution.journal import verify_saved_journal

INTERRUPT_AT = 4
COMPILER_RELATIVE_L2_LIMIT = 1e-3


def small_case() -> Case:
    """Use a real exact arithmetic evaluator with a short final batch."""
    return Case(
        "core-ablation-validation",
        "arithmetic",
        {"operation": "div", "prime": 7},
        width=16,
        heads=2,
        layers=1,
        batch=8,
        learning_rate=0.003,
        weight_decay=0.1,
        warmup=2,
        max_updates=12,
        observe=2,
        after_target=0,
        target=1.0,
    )


def prepared(mechanism: str, head_width: int, dtype: torch.dtype) -> GPT:
    """Install exactly one addition on a deterministic original model."""
    torch.manual_seed(23)
    model = GPT(
        13,
        2 * head_width,
        block_size=16,
        n_head=2,
        n_layer=2,
        use_checkpoint=False,
        pad_vocab_size_to=1,
    ).to(dtype=dtype)
    install_mechanism(model, mechanism)
    return model


def test_configuration_keeps_native_lr2_and_describes_only_actual_mechanism() -> None:
    """Reject combined recipes and verify warmup, exact gain and normalization sites."""
    for changes in [
        {"mechanism": "combined"},
        {"model_variant": "peri-rms"},
        {"learning_rate_multiplier": 4},
        {"beta2": 0.95},
        {"schedule": "cosine"},
    ]:
        with pytest.raises(ValueError, match="one declared original-model"):
            Configuration(**changes)
    for mechanism in MECHANISMS:
        config = Configuration(mechanism=mechanism)
        case = replace(small_case(), warmup=50)
        metadata = config.metadata(case)
        assert metadata["optimizer"] == "torch.optim.AdamW"
        assert metadata["qk_rms"]
        assert metadata["embedding_rms"] == (mechanism == "embedding_rms")
        assert metadata["branch_output_rms"] == (mechanism == "scaled_peri_rms")
        assert ("scalable_softmax" in metadata) == (mechanism == "ssmax")
        for completed, multiplier in [(0, 0), (25, 1), (50, 2), (1000, 2)]:
            assert config.learning_rate_at(case, completed, 21) == pytest.approx(
                multiplier * case.learning_rate
            )
        model = prepared(mechanism, 8, torch.float32)
        assert model.lm_head.weight is model.token_embedding_table.weight
        for block in model.blocks:
            assert_close(
                block.sa.attention_gain(),
                block.sa.log_alpha.exp().view(1, 2, 1, 1) / 8,
                rtol=0,
                atol=0,
            )
            for branch in (block.sa, block.ffwd):
                assert hasattr(branch.c_proj, "norm") == (
                    mechanism == "scaled_peri_rms"
                )
                if mechanism == "scaled_peri_rms":
                    assert branch.c_proj.residual_scale == (2 * model.n_layer) ** -0.5


def test_all_35_specs_preserve_initial_tensors_order_capacity_and_rng() -> None:
    """Compare each complete factory with the original on every frozen task shape."""
    inventory = json.loads(
        Path(
            "research/early99_optimizer_followups/parameter-inventory.json"
        ).read_text()
    )
    capacities = {r["case"]: r["parameters"] for r in inventory["rows"]}
    for row in selected("all"):
        case = specification(row)
        task = build(case, "cpu")
        torch.manual_seed(7)
        original = baseline_model(task, case, "cpu")
        rng = torch.get_rng_state().clone()
        for mechanism in MECHANISMS:
            config = Configuration(mechanism=mechanism)
            torch.manual_seed(7)
            actual = configured_factories(config).model(task, case, "cpu")
            assert torch.equal(torch.get_rng_state(), rng)
            assert_close(actual.state_dict(), original.state_dict(), rtol=0, atol=0)
            assert list(dict(actual.named_parameters())) == list(
                dict(original.named_parameters())
            )
            assert_close(
                dict(actual.named_buffers()),
                dict(original.named_buffers()),
                rtol=0,
                atol=0,
            )
            assert sum(p.numel() for p in actual.parameters()) == capacities[case.name]
            assert (
                actual.model.token_embedding_table.weight is actual.model.lm_head.weight
            )


def independent_logits(
    model: GPT, tokens: torch.Tensor, mechanism: str, eps: float
) -> torch.Tensor:
    """Compute full equations with functional ops; never call a candidate module."""
    batch, length = tokens.shape
    width, heads = model.n_embd, model.n_head
    head_width = width // heads

    def rms(value: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(value, (value.shape[-1],), eps=eps)

    value = F.embedding(tokens, model.lm_head.weight)
    if mechanism == "embedding_rms":
        value = rms(value)
    allowed = torch.ones(length, length, dtype=torch.bool).tril()
    for block in model.blocks:
        att = block.sa
        q, k, v = [
            part.reshape(batch, length, heads, head_width).transpose(1, 2)
            for part in F.linear(rms(value), att.c_attn.weight).chunk(3, -1)
        ]
        q, k = [
            apply_rope(rms(part), model.rope_cos[:length], model.rope_sin[:length])
            for part in (q, k)
        ]
        scores = (
            (q @ k.transpose(-2, -1))
            * att.log_alpha.exp().view(1, heads, 1, 1)
            / head_width
        )
        if mechanism == "ssmax":
            scores = scores * torch.arange(1, length + 1, dtype=value.dtype).log().view(
                1, 1, length, 1
            )
        weights = scores.masked_fill(~allowed, -torch.inf).softmax(-1)
        attended = weights @ v
        unit = v / v.norm(dim=-1, keepdim=True).clamp_min(att.xsa_epsilon)
        exclusive = attended - (attended * unit).sum(-1, keepdim=True) * unit
        branch = F.linear(
            exclusive.transpose(1, 2).reshape(batch, length, width), att.c_proj.weight
        )
        if mechanism == "scaled_peri_rms":
            branch = rms(branch) * (2 * model.n_layer) ** -0.5
        branch = torch.cat([torch.zeros_like(branch[:, :1]), branch[:, 1:]], dim=1)
        value = value + branch
        gate, content = F.linear(rms(value), block.ffwd.c_fc.weight).chunk(2, -1)
        branch = F.linear(F.silu(gate) * content, block.ffwd.c_proj.weight)
        if mechanism == "scaled_peri_rms":
            branch = rms(branch) * (2 * model.n_layer) ** -0.5
        value = value + branch
    return F.linear(rms(value), model.lm_head.weight)


@pytest.mark.parametrize("mechanism", MECHANISMS)
@pytest.mark.parametrize("head_width", [8, 32])
def test_full_forward_and_all_parameter_gradients_against_independent_equations(
    mechanism: str, head_width: int
) -> None:
    """Keep actual FP32 epsilon in FP64 references and exercise clipped XSA too."""
    model = prepared(mechanism, head_width, torch.float64)
    eps = torch.finfo(torch.float32).eps
    for module in model.modules():
        if isinstance(module, torch.nn.RMSNorm):
            module.eps = eps
    tokens = torch.tensor([[2, 4, 8, 1, 3, 5, 6], [3, 7, 2, 6, 0, 4, 1]])
    parameters = list(model.parameters())
    for clipping_epsilon in (1e-12, 0.25):
        for block in model.blocks:
            block.sa.xsa_epsilon = clipping_epsilon
        actual = model(tokens)[0]
        expected = independent_logits(model, tokens, mechanism, eps)
        assert_close(actual, expected, rtol=2e-8, atol=2e-9)
        probe = torch.randn_like(actual)
        first = torch.autograd.grad((actual * probe).sum(), parameters)
        second = torch.autograd.grad((expected * probe).sum(), parameters)
        for a, b in zip(first, second, strict=True):
            assert torch.isfinite(a).all()
            assert_close(a, b, rtol=3e-7, atol=3e-8)


def cache(batch: int, head_width: int) -> KVCache:
    """Use a cache matching the small two-layer model."""
    return KVCache(
        batch_size=batch, num_heads=2, seq_len=16, head_dim=head_width, num_layers=2
    )


@pytest.mark.parametrize("mechanism", MECHANISMS)
def test_causal_chunked_shared_cache_and_singleton(mechanism: str) -> None:
    """Check global SSMax counts, guarded repeated prefixes and selected readout."""
    head_width = 8
    model = prepared(mechanism, head_width, torch.float32).eval()
    tokens = torch.tensor([[2, 2, 2, 4, 8, 1, 3], [3, 3, 7, 2, 6, 0, 4]])
    with torch.no_grad():
        dense = model(tokens)[0]
        for chunks in [[(i, i + 1) for i in range(7)], [(0, 2), (2, 5), (5, 7)]]:
            state = cache(2, head_width)
            actual = torch.cat(
                [model(tokens[:, a:b], kv_cache=state)[0] for a, b in chunks], dim=1
            )
            assert_close(actual, dense, rtol=3e-5, atol=4e-6)
        changed = tokens.clone()
        changed[:, 4:] = (changed[:, 4:] + 1) % 13
        assert_close(model(changed)[0][:, :4], dense[:, :4], rtol=0, atol=0)
        prompts = tokens[:, :3]
        suffix = torch.tensor([[5, 6], [8, 9], [2, 4], [6, 8]])
        prefix = cache(2, head_width)
        model(prompts, kv_cache=prefix)
        shared = cache(4, head_width)
        shared.share_prefix(prefix)
        actual = torch.cat(
            [model(suffix[:, i : i + 1], kv_cache=shared)[0] for i in range(2)], dim=1
        )
        full = torch.cat([prompts.repeat_interleave(2, dim=0), suffix], dim=1)
        assert_close(actual, model(full)[0][:, 3:], rtol=3e-5, atol=4e-6)
        positions = torch.tensor([[1, 5], [2, 4]])
        assert_close(
            model(tokens, positions=positions)[0],
            dense.gather(1, positions[..., None].expand(-1, -1, 13)),
            rtol=0,
            atol=0,
        )
        value = model.blocks[0].ln_1(model.token_embedding_table(tokens[:, :1]))
        assert (
            torch.count_nonzero(
                model.blocks[0].sa(value, model.rope_cos[:1], model.rope_sin[:1])
            )
            == 0
        )


def test_source_archive_rejects_changed_mechanism_or_source(tmp_path: Path) -> None:
    """Preserve every reused implementation, fixed proof dependency and config."""
    config = Configuration()
    plan = prepare_output(tmp_path, config, "cpu", compiled=False)
    for name in [
        "research/early99_core_ablations/factories.py",
        "research/early99_cosine/trainer.py",
        "research/early99/guarantees/PeriBounds.lean",
    ]:
        assert name in plan["source_sha256"]
    prepare_output(tmp_path, config, "cpu", compiled=False)
    with pytest.raises(ValueError, match="settings/sources changed"):
        prepare_output(
            tmp_path, Configuration(mechanism="ssmax"), "cpu", compiled=False
        )
    path = tmp_path / "experiment_source/research/early99_core_ablations/factories.py"
    path.write_text(path.read_text() + "\n# changed\n")
    with pytest.raises(ValueError, match="Archived experiment source changed"):
        prepare_output(tmp_path, config, "cpu", compiled=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mechanism", MECHANISMS)
def test_cuda_graphs_match_eager_native_adamw_and_restore_nonempty_state(
    tmp_path: Path,
    mechanism: str,
) -> None:
    """Separate capture equivalence from compiler rounding through real updates."""
    torch.compiler.reset()
    torch.set_float32_matmul_precision("highest")
    head_width = 32 if mechanism == "ssmax" else 8
    case = replace(small_case(), width=2 * head_width, layers=2)
    config = Configuration(mechanism=mechanism)
    task = arithmetic(case, "cuda")
    torch.manual_seed(11)
    model = configured_factories(config).model(task, case, "cuda")
    reference = copy.deepcopy(model)
    actual = FactoryStepper(
        model,
        task,
        case,
        "cuda",
        configuration=config,
        factories=configured_factories(config),
        journal_path=tmp_path / "execution-journal.json",
        compiled=True,
    )
    parent = Stepper(reference, task, config.training_case(case), "cuda", compiled=True)
    uncompiled = Stepper(
        copy.deepcopy(model), task, config.training_case(case), "cuda", compiled=False
    )

    def eager_step(indices: torch.Tensor, completed: int) -> None:
        # Both native optimizers run without a CUDA graph. One reuses the
        # compiled loss; the other measures the separate effect of compilation.
        for other in (parent, uncompiled):
            other.rate.fill_(
                config.learning_rate_at(case, completed, len(task.splits["train"]))
            )
            other.update(indices.to("cuda"))

    def check_compiler_rounding() -> None:
        # Use the existing numerical-stress normwise tolerance (1e-3) for
        # different floating-point evaluation orders, while retaining the
        # original stricter elementwise tolerances for the capture comparison.
        tensors = list(
            zip(model.parameters(), uncompiled.model.parameters(), strict=True)
        )
        for first, second in zip(
            actual.optimizer.state.values(),
            uncompiled.optimizer.state.values(),
            strict=True,
        ):
            assert first["step"].item() == second["step"].item()
            tensors.extend(
                (first[field], second[field]) for field in ("exp_avg", "exp_avg_sq")
            )
        for first, second in tensors:
            assert torch.isfinite(first).all()
            assert torch.isfinite(second).all()
            relative = (
                first.double() - second.double()
            ).norm() / second.double().norm().clamp_min(1e-30)
            assert relative.item() <= COMPILER_RELATIVE_L2_LIMIT

    sampler = EpochSampler(
        len(task.splits["train"]), case.batch, wrap=False, seed=10011
    )
    initial, rng, cuda_rng = (
        copy.deepcopy(model.state_dict()),
        torch.get_rng_state(),
        torch.cuda.get_rng_state(),
    )
    actual.prepare(sampler.sizes())
    assert_close(model.state_dict(), initial, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), rng)
    assert torch.equal(torch.cuda.get_rng_state(), cuda_rng)
    assert not actual.execution_journal.path.exists()
    assert all(float(state["step"]) == 0 for state in actual.optimizer.state.values())
    sizes = []
    for completed in range(case.max_updates):
        parts, _ = sampler.next()
        indices = gather(parts)
        sizes.append(len(indices))
        actual.step(indices, completed)
        if completed == 0:
            assert_close(model.state_dict(), initial, rtol=0, atol=0)
        eager_step(indices, completed)
        assert_close(model.state_dict(), reference.state_dict(), rtol=5e-4, atol=4e-6)
        assert_close(
            actual.optimizer.state_dict(),
            parent.optimizer.state_dict(),
            rtol=5e-4,
            atol=4e-6,
        )
        check_compiler_rounding()
        summary = actual.execution_journal.summary(completed + 1)
        assert summary["actual_examples_lower"] == sum(sizes)
        assert summary["actual_examples_upper"] == sum(sizes)
    assert min(sizes) < case.batch
    measured = actual.evaluate(task.selection)
    expected = parent.evaluate(task.selection)
    assert measured["sequence_accuracy"] == expected["sequence_accuracy"]
    assert measured["loss"] == pytest.approx(expected["loss"], rel=1e-4, abs=3e-6)

    # Exercise the actual nonempty-moment CUDA restore path, preserving the
    # original journal until the first genuine continued update.
    continued_model = copy.deepcopy(model)
    continued = FactoryStepper(
        continued_model,
        task,
        case,
        "cuda",
        configuration=config,
        factories=configured_factories(config),
        journal_path=actual.execution_journal.path,
        compiled=True,
    )
    continued.optimizer.load_state_dict(actual.optimizer.state_dict())
    for group in continued.optimizer.param_groups:
        group["lr"] = continued.rate
    values, moments = (
        copy.deepcopy(continued_model.state_dict()),
        copy.deepcopy(continued.optimizer.state_dict()),
    )
    raw_journal = continued.execution_journal.path.read_bytes()
    rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    continued.prepare(sampler.sizes())
    assert_close(continued_model.state_dict(), values, rtol=0, atol=0)
    assert_close(continued.optimizer.state_dict(), moments, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), rng)
    assert torch.equal(torch.cuda.get_rng_state(), cuda_rng)
    assert continued.execution_journal.path.read_bytes() == raw_journal
    parts, _ = sampler.next()
    indices = gather(parts)
    continued.step(indices, case.max_updates)
    eager_step(indices, case.max_updates)
    assert_close(
        continued_model.state_dict(), reference.state_dict(), rtol=5e-4, atol=4e-6
    )
    assert_close(
        continued.optimizer.state_dict(),
        parent.optimizer.state_dict(),
        rtol=5e-4,
        atol=4e-6,
    )
    summary = continued.execution_journal.summary(case.max_updates + 1)
    assert (
        summary["actual_examples_lower"]
        == summary["actual_examples_upper"]
        == sum(sizes) + len(indices)
    )
    verify_saved_journal(
        tmp_path,
        case.max_updates + 1,
        len(task.splits["train"]),
        case.batch,
        [{"completed": case.max_updates}],
        summary,
    )


@pytest.mark.parametrize("mechanism", MECHANISMS)
def test_trainer_resume_restores_native_state_and_journal_exposure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mechanism: str,
) -> None:
    """Compare a real uninterrupted trajectory with checkpointed continuation."""
    case, config = small_case(), Configuration(mechanism=mechanism)
    monkeypatch.setattr(native_trainer, "CHECKPOINT_INTERVAL", 2)
    full = trainer.run(
        case, 4, tmp_path / "full", "cpu", configuration=config, compiled=False
    )
    save = native_trainer.save_checkpoint

    def interrupt(
        path: Path,
        model: object,
        stepper: object,
        sampler: object,
        completed: int,
        history: list,
        metadata: dict,
    ) -> None:
        save(path, model, stepper, sampler, completed, history, metadata)
        if completed == INTERRUPT_AT:
            message = "Interrupted after checkpoint"
            raise RuntimeError(message)

    monkeypatch.setattr(native_trainer, "save_checkpoint", interrupt)
    with pytest.raises(RuntimeError, match="Interrupted after checkpoint"):
        trainer.run(
            case, 4, tmp_path / "resume", "cpu", configuration=config, compiled=False
        )
    monkeypatch.setattr(native_trainer, "save_checkpoint", save)
    resumed = trainer.run(
        case,
        4,
        tmp_path / "resume",
        "cpu",
        configuration=config,
        compiled=False,
        resume=True,
    )
    first = torch.load(
        tmp_path / "full" / case.name / "seed4/checkpoint.pt", weights_only=True
    )
    second = torch.load(
        tmp_path / "resume" / case.name / "seed4/checkpoint.pt", weights_only=True
    )
    for key in ["model", "optimizer", "sampler", "cpu_rng"]:
        assert_close(first[key], second[key], rtol=0, atol=0)
    assert resumed["endpoint"]["validation"] == full["endpoint"]["validation"]
    assert (
        resumed["execution_journal"]["actual_examples_lower"]
        == full["execution_journal"]["actual_examples_lower"]
    )
    assert (
        resumed["execution_journal"]["actual_examples_upper"]
        == full["execution_journal"]["actual_examples_upper"]
    )
    assert resumed["execution_journal"]["resumed_segments"] == 1
