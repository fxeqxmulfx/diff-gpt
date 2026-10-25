"""Check associative division frames against plain interpreters and the codec."""

from dataclasses import asdict, replace
from itertools import product
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from diff_gpt.application.grokking.div97 import (
    BLOCK,
    TRAIN,
    VOCAB,
    EpochSampler,
    P,
    token_table,
)
from diff_gpt.application.grokking.mqar_div97 import (
    COLUMNS,
    EMPTY,
    IGNORE,
    INVERSE,
    LEVELS,
    PROBES,
    TRAIN_STREAM,
    VALUE,
    Frames,
    RunConfig,
    Task,
    attention,
    build,
    codec,
    config_from_record,
    copy_metrics,
    copy_queries,
    error_classes,
    memory_partition,
    queries,
    run_experiment,
    stream,
    verify_checkpoint,
)
from diff_gpt.application.grokking.mqar_div97_copy_audit import VARIANTS, variant
from diff_gpt.domain.encoder_decoder import encode
from diff_gpt.domain.model.gpt import GPT
from diff_gpt.infrastructure import provenance

if TYPE_CHECKING:
    from pathlib import Path

QUERY_LAYOUTS = [
    RunConfig(fixed=True, loads=2),
    RunConfig(keys=8, loads=8),
    RunConfig(keys=32, loads=8, overwrites=8),
    RunConfig(keys=16, loads=16, gap=8),
    RunConfig(keys=8, loads=4, operands=1),
]
TINY = {"width": 8, "heads": 1, "epochs": 1}


def replay_queries(frames: Frames, rows: np.ndarray, config: RunConfig) -> None:
    """Run every frame through a dictionary of registers and compare."""
    for frame, row, sources, previous in zip(
        frames.raw, rows, frames.sources[:, 0], frames.previous[:, 0], strict=True
    ):
        registers: dict[int, list[tuple[int, int]]] = {}
        for position, (key, empty, value) in enumerate(frame[: config.writes]):
            assert -config.keys <= key < 0
            assert empty == EMPTY
            assert 0 <= value < P
            history = registers.setdefault(int(key), [])
            # A rewrite always changes the value of its key.
            assert not history or history[-1][1] != value
            history.append((position, int(value)))
        assert len(registers) == config.loads
        dividend, divisor, answer = frame[-1]
        assert answer == row[2]
        latest = registers[int(dividend)]
        assert latest[-1] == (sources[0], row[0])
        assert previous[0] == (latest[-2][1] if len(latest) > 1 else -1)
        if config.operands == 1:
            assert divisor == row[1]
            assert (sources[1], previous[1]) == (-1, -1)
            continue
        assert divisor != dividend
        latest = registers[int(divisor)]
        assert latest[-1] == (sources[1], row[1])
        assert previous[1] == (latest[-2][1] if len(latest) > 1 else -1)
        assert max(sources) < config.writes - config.gap


def test_every_level_has_its_own_token_and_the_actual_codec() -> None:
    """Keys, the empty level and residues are disjoint; residues are div97's."""
    table = codec()
    assert np.unique(table.tokens).size == LEVELS.size
    np.testing.assert_array_equal(table.levels[table.tokens], LEVELS)
    np.testing.assert_array_equal(table.decoded[table.tokens], LEVELS)
    np.testing.assert_array_equal(table.tokens[-EMPTY:], token_table()[0])
    frames = Task(RunConfig(keys=32, loads=8, overwrites=8)).evaluation["test"]
    for frame in frames.raw[:16]:
        _, _, tokens = encode(
            frame.astype(np.float64),
            VOCAB,
            np.full(len(COLUMNS), 96.0),
            0,
            tokenizer="lgt",
        )
        np.testing.assert_array_equal(tokens, table.encode(frame))


@pytest.mark.parametrize("config", QUERY_LAYOUTS, ids=RunConfig.name)
def test_query_frames_follow_a_reference_interpreter(config: RunConfig) -> None:
    """The query reads the latest values of its keys: the equation's operands."""
    task = Task(config)
    for split, frames in task.evaluation.items():
        rows = task.rows[:TRAIN] if split == "train" else task.rows[TRAIN:]
        replay_queries(frames, rows, config)
    batch = next(iter(task.epoch(1)))
    order = EpochSampler(config.seed)
    replay_queries(batch, task.rows[next(iter(order.batches()))], config)


@pytest.mark.parametrize("labels", ["division", "dividend"])
def test_program_frames_follow_a_reference_interpreter(labels: str) -> None:
    """Each compute row divides current registers in place, within its split."""
    config = RunConfig(keys=16, loads=4, computed=6, labels=labels)
    task = Task(config)
    for split, frames in task.evaluation.items():
        for index, frame in enumerate(frames.raw):
            # Per key: its value, the row that wrote it, the value before, depth.
            registers: dict[int, tuple[int, int, int, int]] = {}
            for position, (key, empty, value) in enumerate(frame[: config.writes]):
                assert empty == EMPTY
                assert 0 < value < P
                held = registers.get(int(key))
                assert held is None or held[0] != value
                before = -1 if held is None else held[0]
                registers[int(key)] = (int(value), position, before, 0)
            assert len(registers) == config.loads
            seen = set()
            for step, (i, j, answer) in enumerate(frame[config.writes :]):
                assert i != j
                u, source_u, before_u, depth_u = registers[int(i)]
                w, source_w, before_w, depth_w = registers[int(j)]
                assert task.allowed[split][u, w]
                assert (u, w) not in seen
                seen.add((u, w))
                assert answer == (u * INVERSE[w] % P if labels == "division" else u)
                assert frames.operands[index, step].tolist() == [u, w]
                assert frames.sources[index, step].tolist() == [source_u, source_w]
                assert frames.previous[index, step].tolist() == [before_u, before_w]
                depth = 1 + max(depth_u, depth_w)
                assert frames.depth[index, step] == depth
                registers[int(i)] = (int(answer), config.writes + step, u, depth)


def test_fixed_frames_are_the_anchor_equations_in_its_batch_order() -> None:
    """Level 0 reads the div97 equations and batches, and starts from its weights."""
    config = RunConfig(fixed=True, loads=2, width=8, heads=1, layers=1)
    task = Task(config)
    sampler = EpochSampler(config.seed)
    for epoch in (1, 2):
        for frames, batch in zip(task.epoch(epoch), sampler.batches(), strict=True):
            expected = [
                [(-1, EMPTY, a), (-2, EMPTY, b), (-1, -2, y)]
                for a, b, y in task.rows[batch]
            ]
            np.testing.assert_array_equal(frames.raw, expected)
    torch.manual_seed(config.seed)
    model = build(config, "cpu")
    torch.manual_seed(config.seed)
    anchor = GPT(
        vocab_size=VOCAB,
        n_embd=config.width,
        block_size=BLOCK,
        n_head=config.heads,
        n_layer=config.layers,
        use_checkpoint=False,
    )
    weights, anchor_weights = model.state_dict(), anchor.state_dict()
    assert list(weights) == list(anchor_weights)
    for name, weight in weights.items():
        assert torch.equal(weight, anchor_weights[name])


@pytest.mark.parametrize(
    "config",
    [
        RunConfig(keys=8, loads=4),
        RunConfig(keys=8, loads=4, computed=3),
        RunConfig(loads=2, labels="dividend"),
    ],
    ids=RunConfig.name,
)
def test_only_compute_answers_are_supervised(config: RunConfig) -> None:
    """Targets are the y of compute rows; the loss of GPT scores those alone."""
    task = Task(config)
    frames = task.evaluation["test"].select(slice(8))
    inputs, targets = task.tensors(frames, "cpu")
    supervised = (targets[0] != IGNORE).nonzero().flatten().tolist()
    rows = range(config.writes, config.rows)
    assert supervised == [len(COLUMNS) * row + VALUE - 1 for row in rows]
    assert torch.equal(targets != IGNORE, (targets[0] != IGNORE).expand_as(targets))
    np.testing.assert_array_equal(
        task.codec.levels[targets[:, supervised].numpy()],
        frames.raw[:, config.writes :, VALUE],
    )
    if config.computed == 1:
        changed = frames.raw.copy()
        changed[:, -1, VALUE] = (changed[:, -1, VALUE] + 1) % P
        changed_inputs, _ = task.tensors(replace(frames, raw=changed), "cpu")
        assert torch.equal(inputs, changed_inputs)
    model = build(replace(config, **TINY), "cpu")
    logits, loss = model(inputs, targets)
    expected = F.cross_entropy(
        logits[:, supervised].flatten(0, 1), targets[:, supervised].flatten()
    )
    torch.testing.assert_close(loss, expected)


def test_contexts_are_reproducible_and_fresh_unless_fixed() -> None:
    """Each epoch draws new contexts from its own stream unless they are fixed."""
    config = RunConfig(keys=8, loads=4)
    fresh, fixed = Task(config), Task(replace(config, contexts="fixed"))
    sampler = EpochSampler(config.seed)
    for epoch in (1, 2):
        batch = next(iter(sampler.batches()))
        drawn = queries(
            fresh.rows[batch], config, stream(config, TRAIN_STREAM, epoch, 0)
        )
        np.testing.assert_array_equal(next(iter(fresh.epoch(epoch))).raw, drawn.raw)
        # The train split is scored in contexts that training never shows.
        assert not np.array_equal(drawn.raw, fresh.evaluation["train"].raw[batch])
        np.testing.assert_array_equal(
            next(iter(fixed.epoch(epoch))).raw, fixed.evaluation["train"].raw[batch]
        )
    for split, frames in Task(config).evaluation.items():
        np.testing.assert_array_equal(frames.raw, fresh.evaluation[split].raw)


@pytest.mark.parametrize("labels", ["dividend", "divisor"])
@pytest.mark.parametrize(
    "settings",
    [
        {"loads": 2},
        {"loads": 2, "overwrites": 4},
        {"loads": 2, "fixed": True},
        {"loads": 4, "overwrites": 3, "gap": 2},
    ],
)
def test_independent_copy_reads_memory(
    labels: str, settings: dict[str, object]
) -> None:
    """Identical queries read one register twice, including after overwrites."""
    config = RunConfig(labels=labels, **settings)
    for split in ("train", "test"):
        frames = copy_queries(128, config, split, stream(config, 42))
        for n, frame in enumerate(frames.raw):
            registers = {}
            for position, (key, empty, value) in enumerate(frame[: config.writes]):
                assert empty == EMPTY
                held = registers.get(key, (-1, -1, -1))
                assert held[0] != value
                registers[key] = (value, position, held[0])
            assert len(registers) == config.loads
            keys = np.array(list(registers))
            values = np.array([registers[key][0] for key in keys])
            assert memory_partition(-keys, values) == split
            for operand, key in enumerate(frame[-1, :2]):
                value, source, previous = registers[key]
                assert frames.operands[n, 0, operand] == value
                assert frames.sources[n, 0, operand] == source
                assert frames.previous[n, 0, operand] == previous
                assert source < config.writes - config.gap
            assert frame[-1, VALUE] == registers[frame[-1, int(labels == "divisor")]][0]


@pytest.mark.parametrize("labels", ["dividend", "divisor"])
def test_copy_quartets_close_the_other_query_shortcut(labels: str) -> None:
    """Every memory has all four counterfactuals; the complement rule gets half."""
    config = RunConfig(loads=2, labels=labels)
    frames = copy_queries(512, config, "test", stream(config, 44))
    relevant = int(labels == "divisor")
    other_query = 1 - relevant
    shortcut, exact_answers = [], []
    for quartet in frames.raw.reshape(-1, 4, config.rows, len(COLUMNS)):
        assert (quartet[:, : config.writes] == quartet[0, : config.writes]).all()
        memory = {key: value for key, _, value in quartet[0, : config.writes]}
        assert {tuple(row[-1, :2]) for row in quartet} == set(product(memory, repeat=2))
        for frame in quartet:
            other = next(key for key in memory if key != frame[-1, other_query])
            shortcut.append(memory[other])
            exact_answers.append(memory[frame[-1, relevant]])
    np.testing.assert_array_equal(exact_answers, frames.raw[:, -1, VALUE])
    good = copy_metrics(frames, np.array(exact_answers), config)
    assert all(value == 1.0 for value in good.values())
    bad = copy_metrics(frames, np.array(shortcut), config)
    assert bad["distinct_accuracy"] == 1.0
    collisions = np.mean(frames.raw[:, 0, VALUE] == frames.raw[:, 1, VALUE])
    assert collisions < 1
    for name in ("same_accuracy", "context_exact", "irrelevant_query_agreement"):
        assert bad[name] == collisions
    assert np.mean(np.array(shortcut) == exact_answers) == (1 + collisions) / 2


def test_copy_split_is_by_memory_and_is_reproducible() -> None:
    """Query order, write order, and seed cannot move a memory between splits."""
    config = RunConfig(loads=2, keys=2, labels="dividend")
    memories = {}
    for seed, split in product((11, 12), ("train", "test")):
        frames = copy_queries(
            512, config, split, stream(replace(config, seed=seed), 45)
        )
        for frame in frames.raw:
            keys, values = -frame[:2, 0], frame[:2, VALUE]
            assert memory_partition(keys, values) == split
            assert memory_partition(keys[::-1], values[::-1]) == split
            canonical = tuple(sorted(zip(keys.tolist(), values.tolist(), strict=True)))
            assert memories.setdefault(canonical, split) == split
    fresh, repeated = Task(config), Task(config)
    for split in ("train", "test"):
        np.testing.assert_array_equal(
            fresh.evaluation[split].raw, repeated.evaluation[split].raw
        )
    first = next(iter(fresh.epoch(1))).raw
    np.testing.assert_array_equal(first, next(iter(repeated.epoch(1))).raw)
    assert not np.array_equal(first, next(iter(fresh.epoch(2))).raw)
    fixed = Task(replace(config, contexts="fixed"))
    batch = next(iter(EpochSampler(config.seed).batches()))
    np.testing.assert_array_equal(
        next(iter(fixed.epoch(1))).raw, fixed.evaluation["train"].raw[batch]
    )


@pytest.mark.parametrize("labels", ["dividend", "divisor"])
def test_copy_audit_relabels_changed_queries(labels: str) -> None:
    """An input intervention carries the new correct answer, never the old label."""
    config = RunConfig(loads=2, labels=labels)
    frames = copy_queries(32, config, "test", stream(config, 46))
    for name in VARIANTS:
        changed = variant(frames, config, name)
        for n, frame in enumerate(changed.raw):
            memory = {
                key: (value, row) for row, (key, _, value) in enumerate(frame[:2])
            }
            for operand, key in enumerate(frame[-1, :2]):
                value, source = memory[key]
                assert changed.operands[n, 0, operand] == value
                assert changed.sources[n, 0, operand] == source
            relevant = frame[-1, int(labels == "divisor")]
            assert frame[-1, VALUE] == memory[relevant][0]
        if name in {
            "original",
            "swap_loads",
            "flip_second_query" if labels == "dividend" else "flip_first_query",
        }:
            np.testing.assert_array_equal(
                changed.raw[:, -1, VALUE], frames.raw[:, -1, VALUE]
            )


def test_error_classes_name_the_recognizable_mistakes() -> None:
    """Swapped operands, copies, stale values and non-values are told apart."""
    size = 5
    raw = np.zeros((size, 1, len(COLUMNS)), dtype=np.int64)
    raw[:, -1, VALUE] = 2  # 6 / 3
    frames = Frames(
        raw=raw,
        operands=np.tile([[[6, 3]]], (size, 1, 1)),
        sources=np.zeros((size, 1, 2), dtype=np.int64),
        previous=np.tile([[[12, 4]]], (size, 1, 1)),
        depth=np.ones((size, 1), dtype=np.int64),
    )
    levels = np.array([2, 3 * INVERSE[6] % P, 6, 12 * INVERSE[3] % P, -5])
    assert error_classes(levels, frames) == {
        "test_swap": 1 / size,
        "test_copy": 1 / size,
        "test_stale": 1 / size,
        "test_nonvalue": 1 / size,
    }


@pytest.mark.parametrize(
    "settings",
    [
        {},
        {"rope_layers": (True, False)},
        {"rope_layers": (False, False)},
        {"rope_base": 500.0},
    ],
)
def test_attention_probabilities_rebuild_each_attention_output(
    settings: dict[str, object],
) -> None:
    """The replicated softmax, values, XSA and projection equal the module."""
    model = GPT(
        vocab_size=VOCAB, n_embd=16, block_size=12, n_head=2, n_layer=2, **settings
    )
    model.eval()
    generator = torch.Generator().manual_seed(0)
    inputs = torch.randint(0, VOCAB, (3, 11), generator=generator)
    outputs: list[tuple[torch.Tensor, torch.Tensor]] = []
    hooks = [
        block.sa.register_forward_hook(
            lambda _, args, out: outputs.append((args[0], out))
        )
        for block in model.blocks
    ]
    with torch.no_grad():
        model(inputs)
    for hook in hooks:
        hook.remove()
    probabilities = attention(model, inputs)
    with torch.no_grad():
        for block, (x, out), p in zip(
            model.blocks, outputs, probabilities, strict=True
        ):
            sa = block.sa
            torch.testing.assert_close(p.sum(-1), torch.ones(p.shape[:-1]))
            assert torch.all(p.triu(1) == 0)
            _, _, v = sa.c_attn(x).split(sa.n_embd, dim=2)
            v = v.view(*x.shape[:2], sa.n_head, sa.head_dim).transpose(1, 2)
            y = p @ v
            unit = F.normalize(v, dim=-1)
            y -= (y * unit).sum(-1, keepdim=True) * unit
            rebuilt = sa.c_proj(y.transpose(1, 2).reshape(x.shape))
            torch.testing.assert_close(rebuilt, out, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "config",
    [
        RunConfig(fixed=True, loads=2, layers=1, **TINY),
        RunConfig(keys=8, loads=4, overwrites=2, **TINY),
        RunConfig(keys=8, loads=4, operands=1, labels="random", **TINY),
        RunConfig(keys=8, loads=4, computed=3, **TINY),
        RunConfig(loads=2, layers=2, labels="dividend", rope="10", **TINY),
        RunConfig(loads=2, layers=2, labels="dividend", rope_base=500.0, **TINY),
        RunConfig(loads=2, layers=2, labels="dividend", **TINY),
    ],
    ids=RunConfig.name,
)
def test_saved_weights_replay_metrics_and_public_predict(
    tmp_path: Path, config: RunConfig
) -> None:
    """The checkpoint replays both splits, query weights and DiffGPT.predict."""
    directory = tmp_path / "run"
    result = run_experiment(config, directory, git_head=provenance.git_head)
    verified = verify_checkpoint(directory)
    assert verified["epoch"] == result["epoch"] == 1
    assert verified["metrics"] == result["metrics"]
    public = verified["diffgpt_predict"]
    assert public["frames"] == len(PROBES)
    assert public["rows"] == len(PROBES) * config.computed
    assert public["disagreements_with_benchmark"] == 0
    assert set(verified["query_metrics"]) == {"train", "test"}
    breakdown = "by_depth" if config.computed > 1 else "by_distance"
    assert breakdown in result["breakdown"]


@pytest.mark.slow
def test_full_public_predict_scores_every_test_frame(tmp_path: Path) -> None:
    """Every test frame goes through DiffGPT.predict and agrees with the logits."""
    directory = tmp_path / "run"
    config = RunConfig(keys=8, loads=4, computed=2, **TINY)
    run_experiment(config, directory, git_head=provenance.git_head)
    public = verify_checkpoint(directory, full_predict=True)["diffgpt_predict"]
    assert public["complete_test_split"]
    assert public["disagreements_with_benchmark"] == 0
    assert public["frames"] == len(Task(config).evaluation["test"].raw)


@pytest.mark.parametrize(
    "settings",
    [
        {"loads": 40, "keys": 32},
        {"loads": 4, "gap": 3},
        {"computed": 2, "operands": 1},
        {"fixed": True, "loads": 3},
        {"labels": "sum"},
        {"rope": ""},
        {"rope": "1"},
        {"rope": "12"},
        {"rope_base": 0.0},
    ],
)
def test_impossible_frames_are_rejected(settings: dict[str, object]) -> None:
    """The generators never get a layout they cannot draw."""
    with pytest.raises(ValueError, match=r"require|must|fixed|register"):
        RunConfig(**settings)


def test_current_records_replay_the_same_config() -> None:
    """The saved architecture marker identifies the fixed normalization formula."""
    config = RunConfig(loads=2, labels="dividend", **TINY)
    current = {
        "config": asdict(config),
        "normalization": "peri_rms_qk_rms",
        "data_protocol": config.data_protocol,
    }
    assert config_from_record(current) == config


def test_old_copy_data_cannot_silently_replay_as_independent_queries() -> None:
    """Identical model shapes do not imply identical data or evaluation."""
    config = RunConfig(loads=2, labels="dividend", **TINY)
    with pytest.raises(ValueError, match="frozen source"):
        config_from_record(
            {"config": asdict(config), "normalization": "peri_rms_qk_rms"}
        )


@pytest.mark.parametrize(
    "normalization",
    [None, "pre", "pre_embed", "peri_rms"],
)
def test_historical_architectures_cannot_silently_load_as_peri(
    normalization: str | None,
) -> None:
    """Parameter-free norms can change the function without changing weight shapes."""
    with pytest.raises(ValueError, match="Peri/QK-RMSNorm"):
        config_from_record(
            {"config": asdict(RunConfig()), "normalization": normalization}
        )
