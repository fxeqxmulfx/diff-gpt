"""
Grokking a / b mod 97 through associative recall down the rows of a frame.

Run with ``uv run --locked -m benchmarks.grokking.mqar_div97 --help``. A frame
has the columns (a, b, y). A load row (k, empty, v) writes the value v to the
key k. A compute row (i, j, y) is supervised with y = R[i] / R[j] mod 97, R[k]
being the latest value written to k, and writes R[i] := y. One compute row
after the loads is associative recall that feeds division; several make a
register machine whose rows read what earlier rows computed. Every compute
row divides an equation of the browser split of ``benchmarks.grokking.div97``:
train and test frames retrieve alike and differ in their equations alone, so
the test accuracy measures division. The single-step, two-query copying
controls instead draw memory first, query its keys independently (including
i == j), and split by the canonical key/value memory, never by the query.

For the two-layer N=2 copying control with RoPE only in the first layer,
use ``--loads 2 --labels dividend --layers 2 --rope 10``.
Add ``--diagnostics-every 1 --snapshot-every 10`` to record both optimizer
weight states, per-head routing, norms, gradients and paired weight snapshots.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from benchmarks.grokking.div97 import (
    BATCH,
    DOMAIN,
    LIBRARY_SOURCES,
    NORMALIZATION,
    REPLAY_LOSS_TOLERANCE,
    ROOT,
    TOTAL,
    TRAIN,
    VOCAB,
    Criterion,
    EpochSampler,
    P,
    digest,
    division_rows,
    require_current_model,
    write_json,
)
from benchmarks.grokking.mqar_div97_diagnostics import DiagnosticConfig, Diagnostics
from diff_gpt.application.diff_gpt import DiffGPT
from diff_gpt.domain.encoder_decoder import ENCODER_VERSION, decode, encode
from diff_gpt.domain.model.gpt import GPT
from diff_gpt.domain.model.rope import apply_rope
from diff_gpt.domain.optimizer.schedule_free import ScheduleFreeAdamW
from diff_gpt.domain.sampler.temperature import TemperatureSampler

if TYPE_CHECKING:
    from collections.abc import Iterator

COLUMNS = ["a", "b", "y"]
VALUE = COLUMNS.index("y")
EMPTY = -96  # Column b of a load row: the lowest level, below every key.
MAX_KEYS = 95  # Keys are the levels -1..-95, values the levels 0..96.
LEVELS = np.arange(EMPTY, P, dtype=np.int64)
UNUSED = EMPTY - 1  # The level of a token that no level of a frame reaches.
INVERSE = np.array([0, *(pow(value, -1, P) for value in range(1, P))])
IGNORE = -100
EVALUATION_BATCH = 512
ATTENTION_FRAMES = 512
RECALL_OPERANDS = 2
PROBES = (0, 19, -1)
LABELS = ("division", "dividend", "divisor", "random")
CONTEXTS = ("fresh", "fixed")
RANDOM_LABELS = 0x72616E646F6D  # The random-label stream of div97.
TRAIN_STREAM, PROBE_STREAM, TEST_STREAM, ORDER_STREAM = range(4)


@dataclass(frozen=True)
class RunConfig:
    """
    One run: the frame layout, the model and ScheduleFreeAdamW.

    `loads` distinct keys out of `keys` are written once each, and
    `overwrites` more writes rebind some of them to new values, all in a
    random order. `gap` keeps the operands out of the last load rows.
    `operands` 1 gives the divisor in the compute row itself. `computed` > 1
    turns the compute rows into a register machine. `fixed` is the layout of
    div97: keys 1 and 2, written in that order and always queried.
    `contexts` "fresh" draws new loads around the equations every epoch.
    `rope` is "all" or one 0/1 per layer: "10" applies RoPE only in layer 0.
    `rope_base` is the base of the RoPE frequencies, GPT's `rope_base`.
    The model always uses parameter-free Peri-RMSNorm and per-head QK-RMSNorm
    with standard 1/sqrt(head_dim) attention scaling.
    """

    width: int = 128
    layers: int = 2
    heads: int = 4
    rope: str = "all"
    rope_base: float = 10_000.0
    seed: int = 11
    learning_rate: float = 0.01
    weight_decay: float = 1.0
    beta2: float = 0.98
    warmup: int = 10
    epochs: int = 2000
    after99: int = 25
    labels: str = "division"
    keys: int = 32
    loads: int = 8
    overwrites: int = 0
    gap: int = 0
    operands: int = 2
    computed: int = 1
    fixed: bool = False
    contexts: str = "fresh"

    def __post_init__(self) -> None:
        """Reject frames that the generators cannot draw."""
        problems = []
        if self.labels not in LABELS:
            problems.append(f"labels must be one of {LABELS}")
        if self.contexts not in CONTEXTS:
            problems.append(f"contexts must be one of {CONTEXTS}")
        if self.rope != "all" and (
            not self.rope
            or len(self.rope) != self.layers
            or set(self.rope) - {"0", "1"}
        ):
            problems.append('rope must be "all" or one 0/1 per layer')
        if not 0 < self.rope_base < math.inf:
            problems.append("rope_base must be positive and finite")
        if self.operands not in {1, 2}:
            problems.append("operands must be 1 or 2")
        if not self.operands <= self.loads <= self.keys <= MAX_KEYS:
            problems.append(f"require operands <= loads <= keys <= {MAX_KEYS}")
        if self.overwrites < 0 or not 0 <= self.gap <= self.loads - self.operands:
            problems.append("require overwrites >= 0 and 0 <= gap <= loads - operands")
        if self.computed < 1:
            problems.append("computed must be positive")
        if self.computed > 1 and (self.operands == 1 or self.gap or self.fixed):
            problems.append("the register machine has two operands, no gap, free keys")
        if self.fixed and (self.loads != self.operands or self.overwrites or self.gap):
            problems.append("a fixed frame loads its operands once, without a gap")
        if problems:
            msg = "; ".join(problems)
            raise ValueError(msg)

    @property
    def writes(self) -> int:
        """Return the load rows in front of the first compute row."""
        return self.loads + self.overwrites

    @property
    def rows(self) -> int:
        """Return the rows of one frame."""
        return self.writes + self.computed

    @property
    def block(self) -> int:
        """Return the tokens of one frame: the block of GPT and DiffGPT.predict."""
        return len(COLUMNS) * self.rows

    @property
    def independent_copy(self) -> bool:
        """Identify recall controls whose two queries must be independent."""
        return (
            self.computed == 1
            and self.operands == RECALL_OPERANDS
            and self.labels in {"dividend", "divisor"}
        )

    @property
    def data_protocol(self) -> str:
        """Version the sampling and split separately from the model."""
        return "independent_copy_v1" if self.independent_copy else "equations_v1"

    def name(self) -> str:
        """Return a unique readable directory component for these settings."""
        parts = [self.labels, "fixed" if self.fixed else f"n{self.loads}-k{self.keys}"]
        parts.extend(("normperi", "qkrms"))
        if self.independent_copy:
            parts.append("independent")
        if self.overwrites:
            parts.append(f"r{self.overwrites}")
        if self.gap:
            parts.append(f"g{self.gap}")
        if self.operands == 1:
            parts.append("one")
        if self.computed > 1:
            parts.append(f"m{self.computed}")
        if self.contexts != "fresh":
            parts.append(f"{self.contexts}-contexts")
        rope = f"-rope{self.rope}" if self.rope != "all" else ""
        if self.rope_base != RunConfig.rope_base:  # The default keeps old names.
            rope += f"-base{self.rope_base:g}"
        parts.append(
            f"w{self.width}-l{self.layers}-h{self.heads}{rope}"
            f"-lr{self.learning_rate:g}-wd{self.weight_decay:g}"
            f"-b2{self.beta2:g}-s{self.seed}"
        )
        return "-".join(parts)


def config_from_record(record: dict[str, Any]) -> RunConfig:
    """Return a saved run's config; reject historical models and data protocols."""
    require_current_model(record)
    config = RunConfig(**record["config"])
    if config.independent_copy and record.get("data_protocol") != config.data_protocol:
        msg = (
            "checkpoint copying data used a different query/split protocol; "
            "use the run's frozen source/ for historical datasets"
        )
        raise ValueError(msg)
    return config


@dataclass(frozen=True)
class Codec:
    """
    The k=0 codec of div97 on every level of a frame.

    `tokens` holds the token of each level -96..96 at index level + 96,
    `levels` the level of each token (UNUSED for one no level reaches) and
    `decoded` the integer DiffGPT.predict decodes from each token.
    """

    tokens: np.ndarray
    levels: np.ndarray
    decoded: np.ndarray
    metadata: dict[str, Any]

    def encode(self, raw: np.ndarray) -> np.ndarray:
        """Return the tokens of raw levels, elementwise."""
        return self.tokens[raw - EMPTY]


def codec() -> Codec:
    """
    Encode every level of a frame with the actual codec of div97.

    Vocabulary 256 and domain 96 give the residues the tokens of that
    benchmark, so a fixed frame starts from its initialization. The negative
    half of the grid, which residues leave unused, holds the keys -1..-95
    and the empty level -96: GPT has no column embedding, so every role
    needs its own tokens.
    """
    raw = LEVELS.astype(np.float64).reshape(-1, 1)
    start, grid, tokens = encode(raw, VOCAB, DOMAIN, 0, tokenizer="lgt")
    decoded = decode(start, grid, tokens, VOCAB, 0, tokenizer="lgt")
    table = tokens.reshape(-1).astype(np.int64)
    if np.unique(table).size != LEVELS.size or not np.array_equal(
        np.rint(decoded).reshape(-1), LEVELS
    ):
        msg = "every key, value and the empty level need their own token"
        raise ValueError(msg)
    levels = np.full(VOCAB, UNUSED, dtype=np.int64)
    levels[table] = LEVELS
    every = np.arange(VOCAB).reshape(-1, 1)
    public = decode(np.empty((0, 1)), grid, every, VOCAB, 0, tokenizer="lgt")
    return Codec(
        tokens=table,
        levels=levels,
        decoded=np.rint(public).astype(np.int64).reshape(-1),
        metadata={
            "tokenizer": "lgt",
            "encoder_version": ENCODER_VERSION,
            "order_of_derivative": 0,
            "vocab_size": VOCAB,
            "domain": DOMAIN.tolist(),
            "scale": grid.scale.tolist(),
            "value_tokens": table[-EMPTY:].tolist(),
            "key_tokens": table[1:-EMPTY][::-1].tolist(),
            "empty_token": int(table[0]),
        },
    )


@dataclass(frozen=True)
class Frames:
    """
    Raw frames and, for each compute row, where its operands come from.

    `raw` (B, T, 3) holds keys as levels -1..-K, the empty level and values
    0..96. Per compute row, (B, M, 2): `operands` the values R[i], R[j] (or
    the given divisor), `sources` the rows of their latest writes (-1 for a
    given divisor) and `previous` the values those keys held before them
    (-1 for none). `depth` (B, M) is one more than the deepest operand, a
    load being depth 0.
    """

    raw: np.ndarray
    operands: np.ndarray
    sources: np.ndarray
    previous: np.ndarray
    depth: np.ndarray

    def select(self, index: np.ndarray | slice) -> Frames:
        """Return the frames at `index`."""
        return Frames(*(getattr(self, field.name)[index] for field in fields(self)))

    @staticmethod
    def concatenate(parts: list[Frames]) -> Frames:
        """Return the frames of all `parts`, in order."""
        return Frames(
            *(
                np.concatenate([getattr(part, field.name) for part in parts])
                for field in fields(Frames)
            )
        )


def stream(config: RunConfig, *purpose: int) -> np.random.Generator:
    """Return the generator of one purpose of this run, independent of the others."""
    return np.random.default_rng([config.seed, *purpose])


def _schedule(
    rng: np.random.Generator, size: int, config: RunConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw each frame's keys, the order of its writes and each key's last write."""
    if config.fixed:
        keys = np.tile(np.arange(1, config.loads + 1), (size, 1))
        order = np.tile(np.arange(config.loads), (size, 1))
    else:
        alphabet = np.tile(np.arange(1, config.keys + 1), (size, 1))
        keys = rng.permuted(alphabet, axis=1)[:, : config.loads]
        once = np.tile(np.arange(config.loads), (size, 1))
        again = rng.integers(0, config.loads, size=(size, config.overwrites))
        order = rng.permuted(np.concatenate((once, again), axis=1), axis=1)
    written = order[:, :, None] == np.arange(config.loads)
    last = np.where(written, np.arange(config.writes)[:, None], -1).max(axis=1)
    return keys, order, last


def _operand_layout(
    rng: np.random.Generator, size: int, config: RunConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Draw write schedules and the slots of the operands, outside the last gap rows."""
    keys, order, last = _schedule(rng, size, config)
    if config.fixed:
        return keys, order, last, np.tile(np.arange(config.operands), (size, 1))
    slots = np.empty((size, config.operands), dtype=np.int64)
    pending = np.arange(size)
    while pending.size:
        eligible = last[pending] < config.writes - config.gap
        scores = np.where(eligible, rng.random(eligible.shape), -1.0)
        chosen = np.argsort(-scores, axis=1, kind="stable")[:, : config.operands]
        valid = np.take_along_axis(eligible, chosen, axis=1).all(axis=1)
        slots[pending[valid]] = chosen[valid]
        pending = pending[~valid]
        if pending.size:
            keys[pending], order[pending], last[pending] = _schedule(
                rng, pending.size, config
            )
    return keys, order, last, slots


def _values_ending_at(
    rng: np.random.Generator,
    order: np.ndarray,
    last: np.ndarray,
    finals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Assign the writes backward, so that each key ends at its final value.

    Every rewrite changes the value of its key. Returns the value of every
    write and, per key, the value its latest write replaced (-1 for none).
    """
    size, writes = order.shape
    index = np.arange(size)
    following = finals.copy()
    previous = np.full(finals.shape, -1, dtype=np.int64)
    values = np.empty((size, writes), dtype=np.int64)
    for position in range(writes - 1, -1, -1):
        slot = order[:, position]
        latest = last[index, slot] == position
        after = following[index, slot]
        draw = rng.integers(0, P - 1, size=size)
        draw += draw >= after
        value = np.where(latest, after, draw)
        # Going backward, the first earlier write of a key precedes its latest.
        first = ~latest & (previous[index, slot] < 0)
        previous[index[first], slot[first]] = value[first]
        following[index, slot] = value
        values[:, position] = value
    return values, previous


def queries(rows: np.ndarray, config: RunConfig, rng: np.random.Generator) -> Frames:
    """
    Load the operands of each equation under random keys, then query them.

    `rows` holds (a, b, label) per frame; a and b become the latest values
    of the queried keys, every other write draws a residue 0..96.
    """
    size = len(rows)
    index = np.arange(size)
    keys, order, last, slots = _operand_layout(rng, size, config)
    finals = rng.integers(0, P, size=(size, config.loads))
    for operand in range(config.operands):
        finals[index, slots[:, operand]] = rows[:, operand]
    values, previous = _values_ending_at(rng, order, last, finals)
    raw = np.empty((size, config.rows, len(COLUMNS)), dtype=np.int64)
    raw[:, : config.writes, 0] = -np.take_along_axis(keys, order, axis=1)
    raw[:, : config.writes, 1] = EMPTY
    raw[:, : config.writes, VALUE] = values
    raw[:, -1, 0] = -keys[index, slots[:, 0]]
    raw[:, -1, 1] = -keys[index, slots[:, 1]] if config.operands > 1 else rows[:, 1]
    raw[:, -1, VALUE] = rows[:, 2]
    sources = np.full((size, 1, 2), -1, dtype=np.int64)
    replaced = np.full((size, 1, 2), -1, dtype=np.int64)
    for operand in range(config.operands):
        sources[:, 0, operand] = last[index, slots[:, operand]]
        replaced[:, 0, operand] = previous[index, slots[:, operand]]
    return Frames(
        raw=raw,
        operands=rows[:, None, :2].copy(),
        sources=sources,
        previous=replaced,
        depth=np.ones((size, 1), dtype=np.int64),
    )


def memory_partition(keys: np.ndarray, values: np.ndarray) -> str:
    """Assign a memory map to one split, independently of queries and write order."""
    canonical = np.stack((keys, values), axis=1)[np.argsort(keys)]
    hashed = hashlib.blake2b(
        canonical.astype("<i8", copy=False).tobytes(),
        digest_size=8,
        person=b"mqar-copy-v1",
    ).digest()
    return "train" if hashed[0] % 2 == 0 else "test"


def copy_queries(
    count: int, config: RunConfig, split: str, rng: np.random.Generator
) -> Frames:
    """
    Draw memory, then all ordered query pairs with replacement for that memory.

    Every N=2 context contributes a complete quartet when count is divisible
    by four (all benchmark batches are). Values are uniform on 0..96, including
    equal values. Labels and source metadata are read from memory after writing.
    A larger context's final query block may be truncated in random order.
    """
    if split not in {"train", "test"} or count < 1:
        msg = "require a train/test split and a positive frame count"
        raise ValueError(msg)
    parts: list[Frames] = []
    pending = count
    while pending:
        size = max(16, math.ceil(pending / (config.loads - config.gap) ** 2) * 3)
        keys, order, last = _schedule(rng, size, config)
        finals = rng.integers(0, P, size=(size, config.loads))
        values, previous = _values_ending_at(rng, order, last, finals)
        for row in range(size):
            if memory_partition(keys[row], finals[row]) != split:
                continue
            eligible = np.flatnonzero(last[row] < config.writes - config.gap)
            if len(eligible) < config.operands:
                continue
            slots = np.stack(np.meshgrid(eligible, eligible, indexing="ij"), axis=-1)
            slots = rng.permutation(slots.reshape(-1, 2))[:pending]
            length = len(slots)
            raw = np.empty((length, config.rows, len(COLUMNS)), dtype=np.int64)
            raw[:, : config.writes, 0] = -keys[row, order[row]]
            raw[:, : config.writes, 1] = EMPTY
            raw[:, : config.writes, VALUE] = values[row]
            raw[:, -1, :2] = -keys[row, slots]
            operands = finals[row, slots]
            raw[:, -1, VALUE] = operands[:, int(config.labels == "divisor")]
            parts.append(
                Frames(
                    raw=raw,
                    operands=operands[:, None],
                    sources=last[row, slots][:, None],
                    previous=previous[row, slots][:, None],
                    depth=np.ones((length, 1), dtype=np.int64),
                )
            )
            pending -= length
            if not pending:
                break
    return Frames.concatenate(parts)


def _program_attempt(
    size: int,
    config: RunConfig,
    allowed: np.ndarray,
    operation: np.ndarray,
    rng: np.random.Generator,
) -> tuple[Frames, np.ndarray]:
    """Draw register-machine frames; flag those that ran out of allowed pairs."""
    index = np.arange(size)
    keys, order, _ = _schedule(rng, size, config)
    register = np.zeros((size, config.loads), dtype=np.int64)
    source = np.full((size, config.loads), -1, dtype=np.int64)
    before = np.full((size, config.loads), -1, dtype=np.int64)
    level = np.zeros((size, config.loads), dtype=np.int64)
    raw = np.empty((size, config.rows, len(COLUMNS)), dtype=np.int64)
    for position in range(config.writes):
        slot = order[:, position]
        held = register[index, slot]
        fresh = rng.integers(1, P, size=size)
        changed = rng.integers(1, P - 1, size=size)
        changed += changed >= held
        value = np.where(held == 0, fresh, changed)
        before[index, slot] = np.where(held == 0, -1, held)
        register[index, slot] = value
        source[index, slot] = position
        raw[:, position, 0] = -keys[index, slot]
        raw[:, position, 1] = EMPTY
        raw[:, position, VALUE] = value
    steps = (size, config.computed, 2)
    operands = np.empty(steps, dtype=np.int64)
    sources = np.empty(steps, dtype=np.int64)
    replaced = np.empty(steps, dtype=np.int64)
    depth = np.empty(steps[:2], dtype=np.int64)
    distinct = ~np.eye(config.loads, dtype=bool)
    used = np.zeros((size, P, P), dtype=bool)
    valid = np.ones(size, dtype=bool)
    for step in range(config.computed):
        u, w = register[:, :, None], register[:, None, :]
        open_pairs = allowed[u, w] & distinct & ~used[index[:, None, None], u, w]
        scores = np.where(open_pairs, rng.random(open_pairs.shape), -1.0)
        choice = scores.reshape(size, -1).argmax(axis=1)
        i, j = np.divmod(choice, config.loads)
        valid &= open_pairs[index, i, j]
        dividend, divisor = register[index, i], register[index, j]
        answer = operation[dividend, divisor]
        row = config.writes + step
        operands[:, step] = np.stack((dividend, divisor), axis=1)
        sources[:, step] = np.stack((source[index, i], source[index, j]), axis=1)
        replaced[:, step] = np.stack((before[index, i], before[index, j]), axis=1)
        depth[:, step] = 1 + np.maximum(level[index, i], level[index, j])
        raw[:, row] = np.stack((-keys[index, i], -keys[index, j], answer), axis=1)
        used[index, dividend, divisor] = True
        before[index, i] = dividend
        register[index, i] = answer
        source[index, i] = row
        level[index, i] = depth[:, step]
    return Frames(raw, operands, sources, replaced, depth), valid


def programs(
    size: int,
    config: RunConfig,
    allowed: np.ndarray,
    operation: np.ndarray,
    rng: np.random.Generator,
) -> Frames:
    """
    Draw register-machine frames: nonzero loads, then in-place compute rows.

    Every compute row picks distinct keys i, j whose current values form an
    `allowed` equation not used before in its frame, and writes
    R[i] := operation[R[i], R[j]]. A frame that runs out of such pairs is
    drawn again.
    """
    done: list[Frames] = []
    pending = size
    while pending:
        frames, valid = _program_attempt(pending, config, allowed, operation, rng)
        done.append(frames.select(valid))
        pending -= int(np.count_nonzero(valid))
    return Frames.concatenate(done)


def labeled_rows(config: RunConfig) -> np.ndarray:
    """Return the browser split of div97 with this run's label in column 2."""
    rows = division_rows(config.seed)
    if config.labels == "dividend":
        rows[:, 2] = rows[:, 0]
    elif config.labels == "divisor":
        rows[:, 2] = rows[:, 1]
    elif config.labels == "random":
        rng = np.random.default_rng(config.seed ^ RANDOM_LABELS)
        rows[:, 2] = rng.integers(0, P, size=TOTAL)
    return rows


def allowed_pairs(rows: np.ndarray) -> dict[str, np.ndarray]:
    """Mark each half's equations with a nonzero dividend: register-machine pairs."""
    allowed = {}
    for split, part in (("train", rows[:TRAIN]), ("test", rows[TRAIN:])):
        mask = np.zeros((P, P), dtype=bool)
        nonzero = part[part[:, 0] > 0]
        mask[nonzero[:, 0], nonzero[:, 1]] = True
        allowed[split] = mask
    return allowed


def operation_table(config: RunConfig) -> np.ndarray:
    """Return what a register-machine row writes for each pair; values stay 1..96."""
    dividend, divisor = np.meshgrid(np.arange(P), np.arange(P), indexing="ij")
    if config.labels == "division":
        return dividend * INVERSE[divisor] % P
    if config.labels == "dividend":
        return dividend
    if config.labels == "divisor":
        return divisor
    return np.random.default_rng(config.seed ^ RANDOM_LABELS).integers(
        1, P, size=(P, P)
    )


class Task:
    """Every deterministic part of a run: the codec, equations and frames."""

    def __init__(self, config: RunConfig) -> None:
        """Build the evaluation frames; training frames follow epoch by epoch."""
        self.config = config
        self.codec = codec()
        self.rows = labeled_rows(config)
        self.allowed = allowed_pairs(self.rows)
        self.operation = operation_table(config)
        self.sampler = EpochSampler(config.seed)
        counts = {"train": TRAIN, "test": TOTAL - TRAIN}
        if config.computed == 1:
            self.sequences, self.batch = TRAIN, BATCH
        else:
            counts = {
                split: math.ceil(np.count_nonzero(mask) / config.computed)
                for split, mask in self.allowed.items()
            }
            self.sequences = counts["train"]
            self.batch = math.ceil(BATCH / config.computed)
        rows = np.arange(config.writes, config.rows)
        self.positions = len(COLUMNS) * rows + VALUE - 1
        self.fixed_contexts = (
            self.frames("train", counts["train"], stream(config, TRAIN_STREAM))
            if config.contexts == "fixed"
            else None
        )
        self.evaluation = {
            "train": self.fixed_contexts
            if self.fixed_contexts is not None
            else self.frames("train", counts["train"], stream(config, PROBE_STREAM)),
            "test": self.frames("test", counts["test"], stream(config, TEST_STREAM)),
        }

    @property
    def updates(self) -> int:
        """Return the optimizer steps of one epoch."""
        return math.ceil(self.sequences / self.batch)

    def frames(
        self,
        split: str,
        count: int,
        rng: np.random.Generator,
        index: np.ndarray | None = None,
    ) -> Frames:
        """Draw `count` frames of `split`; queries take the equations at `index`."""
        if self.config.independent_copy:
            return copy_queries(count, self.config, split, rng)
        if self.config.computed > 1:
            return programs(
                count, self.config, self.allowed[split], self.operation, rng
            )
        equations = self.rows[:TRAIN] if split == "train" else self.rows[TRAIN:]
        return queries(
            equations[np.arange(count) if index is None else index], self.config, rng
        )

    def epoch(self, epoch: int) -> Iterator[Frames]:
        """Yield the batches of one complete pass; epochs must come in order."""
        if self.config.computed == 1:
            batches = (np.asarray(batch) for batch in self.sampler.batches())
        else:
            order = stream(self.config, ORDER_STREAM, epoch).permutation(self.sequences)
            batches = (
                order[start : start + self.batch]
                for start in range(0, self.sequences, self.batch)
            )
        for number, batch in enumerate(batches):
            if self.fixed_contexts is not None:
                yield self.fixed_contexts.select(batch)
            else:
                rng = stream(self.config, TRAIN_STREAM, epoch, number)
                yield self.frames("train", len(batch), rng, batch)

    def tensors(self, frames: Frames, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Return inputs, and targets that supervise the y of each compute row."""
        tokens = self.codec.encode(frames.raw).reshape(len(frames.raw), -1)
        inputs = torch.as_tensor(np.ascontiguousarray(tokens[:, :-1]), device=device)
        targets = torch.full_like(inputs, IGNORE)
        answers = torch.as_tensor(tokens[:, self.positions + 1], device=device)
        targets[:, torch.as_tensor(self.positions, device=device)] = answers
        return inputs, targets


def build(config: RunConfig, device: str) -> GPT:
    """Return the repository GPT at this run's size, initialized from the torch RNG."""
    return GPT(
        vocab_size=VOCAB,
        n_embd=config.width,
        block_size=config.block,
        n_head=config.heads,
        n_layer=config.layers,
        use_checkpoint=False,
        rope_base=config.rope_base,
        rope_layers=None
        if config.rope == "all"
        else tuple(bit == "1" for bit in config.rope),
    ).to(device)


@torch.no_grad()
def score(
    model: GPT, inputs: torch.Tensor, targets: torch.Tensor
) -> tuple[dict[str, float], torch.Tensor]:
    """Score every supervised token by argmax over the complete vocabulary."""
    supervised = targets[0] != IGNORE
    total = torch.zeros((), dtype=torch.float64, device=inputs.device)
    predictions = []
    for x, y in zip(
        inputs.split(EVALUATION_BATCH), targets.split(EVALUATION_BATCH), strict=True
    ):
        logits, _ = model(x)
        logits = logits[:, supervised]
        total += F.cross_entropy(
            logits.flatten(0, 1), y[:, supervised].flatten(), reduction="sum"
        )
        predictions.append(logits.argmax(-1))
    predicted = torch.cat(predictions)
    truth = targets[:, supervised]
    return {
        "loss": total.item() / truth.numel(),
        "accuracy": (predicted == truth).double().mean().item(),
    }, predicted


def evaluate_splits(
    model: GPT, data: dict[str, tuple[torch.Tensor, torch.Tensor]]
) -> tuple[dict[str, dict[str, float]], dict[str, torch.Tensor]]:
    """Score both complete splits; return their metrics and argmax tokens."""
    metrics, predicted = {}, {}
    for split, (inputs, targets) in data.items():
        metrics[split], predicted[split] = score(model, inputs, targets)
    return metrics, predicted


@torch.no_grad()
def rollout(model: GPT, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Answer each compute row after the model's own earlier answers, by argmax."""
    positions = (targets[0] != IGNORE).nonzero().flatten().tolist()
    outputs = []
    for batch in inputs.split(EVALUATION_BATCH):
        x = batch.clone()
        answers = []
        for position in positions:
            logits, _ = model(x[:, : position + 1])
            token = logits[:, -1].argmax(-1)
            answers.append(token)
            if position + 1 < x.shape[1]:
                x[:, position + 1] = token
        outputs.append(torch.stack(answers, dim=1))
    return torch.cat(outputs)


@torch.no_grad()
def attention(model: GPT, inputs: torch.Tensor) -> list[torch.Tensor]:
    """Return the causal attention probabilities of every layer, (B, H, T, T)."""
    captured: list[tuple[torch.Tensor, ...]] = []
    hooks = [
        block.sa.register_forward_pre_hook(lambda _, args: captured.append(args))
        for block in model.blocks
    ]
    try:
        model(inputs)
    finally:
        for hook in hooks:
            hook.remove()
    length = inputs.shape[1]
    causal = torch.ones(length, length, dtype=torch.bool, device=inputs.device).tril()
    probabilities = []
    for block, (x, cos, sin) in zip(model.blocks, captured, strict=True):
        sa = block.sa
        q, k, _ = sa.c_attn(x).split(sa.n_embd, dim=2)
        q, k = (
            t.view(len(x), length, sa.n_head, sa.head_dim).transpose(1, 2)
            for t in (q, k)
        )
        q, k = F.rms_norm(q, (sa.head_dim,)), F.rms_norm(k, (sa.head_dim,))
        if sa.use_rope:
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        scale = 1 / math.sqrt(sa.head_dim)
        scores = q @ k.transpose(-1, -2) * scale
        probabilities.append(scores.masked_fill(~causal, -math.inf).softmax(-1))
    return probabilities


def operand_attention(
    model: GPT, inputs: torch.Tensor, frames: Frames, operands: int
) -> dict[str, float]:
    """Return the answer's attention to each operand value, the top head per layer."""
    rows = torch.arange(len(inputs), device=inputs.device)
    result = {}
    for layer, probabilities in enumerate(attention(model, inputs)):
        answer = probabilities[:, :, -1]
        for operand, name in enumerate(("dividend", "divisor")[:operands]):
            sources = frames.sources[:, 0, operand]
            positions = torch.as_tensor(
                len(COLUMNS) * sources + VALUE, device=inputs.device
            )
            mass = answer[rows, :, positions]
            result[f"attention_{name}_l{layer}"] = mass.mean(dim=0).max().item()
    return result


def error_classes(levels: np.ndarray, frames: Frames) -> dict[str, float]:
    """Return the share of frames whose wrong answer is a recognizable mistake."""
    dividend, divisor = frames.operands[:, 0].T
    stale_dividend, stale_divisor = frames.previous[:, 0].T
    wrong = levels != frames.raw[:, -1, VALUE]
    kinds = {
        "test_swap": (dividend > 0) & (levels == divisor * INVERSE[dividend] % P),
        "test_copy": (levels == dividend) | (levels == divisor),
        "test_stale": (
            (stale_dividend >= 0) & (levels == stale_dividend * INVERSE[divisor] % P)
        )
        | ((stale_divisor > 0) & (levels == dividend * INVERSE[stale_divisor] % P)),
        "test_nonvalue": (levels < 0) | (levels >= P),
    }
    return {name: float(np.mean(wrong & kind)) for name, kind in kinds.items()}


COPY_METRICS = (
    "same_accuracy",
    "distinct_accuracy",
    "context_exact",
    "irrelevant_query_agreement",
)


def copy_metrics(
    frames: Frames, levels: np.ndarray, config: RunConfig
) -> dict[str, float]:
    """Score both query types and counterfactual consistency on shared memories."""
    correct = levels == frames.raw[:, -1, VALUE]
    same = frames.raw[:, -1, 0] == frames.raw[:, -1, 1]
    memory = frames.raw[:, : config.writes].reshape(len(levels), -1)
    _, groups = np.unique(memory, axis=0, return_inverse=True)
    exact = np.ones(groups.max() + 1, dtype=bool)
    np.logical_and.at(exact, groups, correct)
    relevant = frames.raw[:, -1, int(config.labels == "divisor")]
    _, queries = np.unique(
        np.column_stack((memory, relevant)), axis=0, return_inverse=True
    )
    minimum = np.full(queries.max() + 1, np.iinfo(np.int64).max, dtype=np.int64)
    maximum = np.full_like(minimum, np.iinfo(np.int64).min)
    np.minimum.at(minimum, queries, levels)
    np.maximum.at(maximum, queries, levels)
    return {
        "same_accuracy": float(correct[same].mean()) if same.any() else 0.0,
        "distinct_accuracy": float(correct[~same].mean()) if (~same).any() else 0.0,
        "context_exact": float(exact.mean()),
        "irrelevant_query_agreement": float((minimum == maximum).mean()),
    }


def diagnostic_fields(config: RunConfig) -> list[str]:
    """Name the diagnostics recorded on the test frames every epoch."""
    if config.computed > 1:
        return ["test_rollout_accuracy", "test_rollout_exact"]
    names = [
        f"attention_{name}_l{layer}"
        for layer in range(config.layers)
        for name in ("dividend", "divisor")[: config.operands]
    ]
    if config.labels == "division":
        return ["test_swap", "test_copy", "test_stale", "test_nonvalue", *names]
    if config.independent_copy:
        return [
            *names,
            *(
                f"{state}{split}_{metric}"
                for state in ("query_", "")
                for split in ("train", "test")
                for metric in COPY_METRICS
            ),
        ]
    return names


def diagnose(
    model: GPT,
    task: Task,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    predicted: torch.Tensor,
) -> dict[str, float]:
    """Measure what the test answers read and how they fail."""
    config, frames = task.config, task.evaluation["test"]
    if config.computed > 1:
        correct = rollout(model, inputs, targets) == targets[:, targets[0] != IGNORE]
        return {
            "test_rollout_accuracy": correct.double().mean().item(),
            "test_rollout_exact": correct.all(dim=1).double().mean().item(),
        }
    result = {}
    if config.labels == "division":
        levels = task.codec.levels[predicted[:, 0].cpu().numpy()]
        result.update(error_classes(levels, frames))
    result.update(
        operand_attention(
            model,
            inputs[:ATTENTION_FRAMES],
            frames.select(slice(ATTENTION_FRAMES)),
            config.operands,
        )
    )
    return result


def breakdown(
    model: GPT,
    task: Task,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    predicted: torch.Tensor,
) -> dict[str, Any]:
    """Split the final test accuracy by operand distance, rewrites or depth."""
    config, frames = task.config, task.evaluation["test"]
    truth = targets[:, targets[0] != IGNORE]
    correct = (predicted == truth).cpu().numpy()

    def groups(by: np.ndarray, **scores: np.ndarray) -> dict[str, dict[str, Any]]:
        return {
            str(value): {
                "rows": int(count),
                **{
                    name: float(hits[by == value].mean())
                    for name, hits in scores.items()
                },
            }
            for value, count in zip(*np.unique(by, return_counts=True), strict=True)
        }

    if config.computed > 1:
        rolled = (rollout(model, inputs, targets) == truth).cpu().numpy()
        return {
            "by_depth": groups(frames.depth, teacher_forced=correct, rollout=rolled)
        }
    sources = frames.sources[:, 0, : config.operands]
    distance = config.writes - sources.min(axis=1)
    rewritten = (frames.previous[:, 0, : config.operands] >= 0).any(axis=1)
    result = {
        "by_distance": groups(distance, accuracy=correct[:, 0]),
        "by_rewrite": groups(rewritten, accuracy=correct[:, 0]),
    }
    if config.independent_copy:
        same = frames.raw[:, -1, 0] == frames.raw[:, -1, 1]
        result["by_same_query"] = groups(same, accuracy=correct[:, 0])
    return result


def _metadata(task: Task, model: GPT, device: str) -> dict[str, Any]:
    config = task.config
    sources = [
        Path(__file__),
        ROOT / "benchmarks/grokking/mqar_div97_diagnostics.py",
        ROOT / "benchmarks/grokking/div97.py",
        *LIBRARY_SOURCES,
    ]
    tokens_per_epoch = task.sequences * (config.block - 1)
    return {
        "config": asdict(config),
        "normalization": NORMALIZATION,
        "data_protocol": config.data_protocol,
        "git_head": subprocess.check_output(  # noqa: S603 - fixed git argv, no shell.
            [shutil.which("git") or "/usr/bin/git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip(),
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": device,
        "device_name": torch.cuda.get_device_name()
        if device == "cuda"
        else platform.processor(),
        "tf32": False,
        "precision": "float32",
        "codec": task.codec.metadata,
        "columns": COLUMNS,
        "frame_rows": config.rows,
        "load_rows": config.writes,
        "compute_rows": config.computed,
        "block_size": config.block,
        "supervised_positions": task.positions.tolist(),
        "sequences_per_epoch": task.sequences,
        "batch_size": task.batch,
        "updates_per_epoch": task.updates,
        "parameters": sum(p.numel() for p in model.parameters()),
        "flops_per_token": model.estimate_flops(),
        "train_flops_per_epoch": model.estimate_flops() * tokens_per_epoch,
        "data": {
            **(
                {
                    "split": (
                        "BLAKE2b of sorted latest key/value map; query independent"
                    ),
                    "queries": "Cartesian product of eligible keys, including i == j",
                    "values": "independent uniform residues 0..96",
                    "contexts_per_epoch": task.sequences // RECALL_OPERANDS**2
                    if config.loads == RECALL_OPERANDS
                    else None,
                }
                if config.independent_copy
                else {
                    "equations": {
                        split: int(np.count_nonzero(mask))
                        if config.computed > 1
                        else TRAIN
                        for split, mask in task.allowed.items()
                    }
                }
            ),
            **{
                f"{split}_frames": len(frames.raw)
                for split, frames in task.evaluation.items()
            },
            **{
                f"{split}_sha256": digest(frames.raw)
                for split, frames in task.evaluation.items()
            },
        },
        "evaluation": "complete fixed train/test probes every epoch; after99 stopping",
    }


def run_experiment(
    config: RunConfig,
    directory: Path,
    *,
    device: str = "cpu",
    log_every: int = 25,
    logging: DiagnosticConfig | None = None,
) -> dict[str, Any]:
    """Train the repository GPT and ScheduleFreeAdamW while retaining every epoch."""
    task = Task(config)
    if logging is not None:
        logging.validate_task(task)
    directory.mkdir(parents=True, exist_ok=False)
    data = {
        split: task.tensors(frames, device) for split, frames in task.evaluation.items()
    }
    torch.manual_seed(config.seed)
    model = build(config, device)
    metadata = _metadata(task, model, device)
    write_json(directory / "metadata.json", metadata)
    if logging is not None:
        for relative in metadata["source_sha256"]:
            target = directory / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)
    optimizer = ScheduleFreeAdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, config.beta2),
        weight_decay=config.weight_decay,
        warmup=config.warmup,
    )
    logger = (
        Diagnostics(directory, task, *data["test"], logging)
        if logging is not None
        else None
    )
    criterion = Criterion()
    start_time = time.monotonic()
    header = [
        "epoch",
        "step",
        "seconds",
        "flops",
        "train_loss",
        "train_accuracy",
        "test_loss",
        "test_accuracy",
        "query_train_accuracy",
        "query_test_accuracy",
        *diagnostic_fields(config),
    ]
    with (directory / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for epoch in range(config.epochs + 1):
            if epoch:
                model.train()
                optimizer.train()
                for frames in task.epoch(epoch):
                    inputs, targets = task.tensors(frames, device)
                    optimizer.zero_grad(set_to_none=True)
                    _, loss = model(inputs, targets)
                    loss.backward()
                    before = logger.before_step(epoch, model) if logger else None
                    optimizer.step()
                    if logger:
                        logger.after_step(epoch, model, optimizer, before, loss)
            model.eval()
            # Query and averaged evaluation weights are distinct model states.
            query_metrics, query_predicted = evaluate_splits(model, data)
            if logger:
                logger.capture(epoch, "query", model, optimizer, query_metrics)
            optimizer.eval()
            metrics, predicted = evaluate_splits(model, data)
            if logger:
                logger.capture(epoch, "evaluation", model, optimizer, metrics)
            if not all(
                math.isfinite(value)
                for split in metrics.values()
                for value in split.values()
            ):
                msg = f"nonfinite evaluation at epoch {epoch}"
                raise ValueError(msg)
            diagnostics = diagnose(model, task, *data["test"], predicted["test"])
            if config.independent_copy:
                for state, predictions in (
                    ("query_", query_predicted),
                    ("", predicted),
                ):
                    for split, tokens in predictions.items():
                        levels = task.codec.levels[tokens[:, 0].cpu().numpy()]
                        diagnostics.update(
                            {
                                f"{state}{split}_{name}": value
                                for name, value in copy_metrics(
                                    task.evaluation[split], levels, config
                                ).items()
                            }
                        )
            criterion.observe(
                epoch, metrics["train"]["accuracy"], metrics["test"]["accuracy"]
            )
            elapsed = time.monotonic() - start_time
            row = {
                "epoch": epoch,
                "step": epoch * task.updates,
                "seconds": elapsed,
                "flops": epoch * metadata["train_flops_per_epoch"],
                **{
                    f"{split}_{key}": value
                    for split, values in metrics.items()
                    for key, value in values.items()
                },
                **{
                    f"query_{split}_accuracy": values["accuracy"]
                    for split, values in query_metrics.items()
                },
                **diagnostics,
            }
            writer.writerow(row)
            handle.flush()
            status = {
                "config": asdict(config),
                "normalization": NORMALIZATION,
                "data_protocol": config.data_protocol,
                "epoch": epoch,
                "seconds": elapsed,
                "metrics": metrics,
                "query_metrics": query_metrics,
                "diagnostics": diagnostics,
                "criterion": asdict(criterion),
            }
            write_json(directory / "status.json", status)
            if epoch % log_every == 0 or epoch in (config.epochs, criterion.first99):
                print(
                    json.dumps(
                        {"run": config.name(), **row, "criterion": asdict(criterion)}
                    ),
                    flush=True,
                )
            if (
                criterion.first99 is not None
                and epoch >= criterion.first99 + config.after99
            ):
                break
    status["breakdown"] = breakdown(model, task, *data["test"], predicted["test"])
    # The saved model is exactly the scored evaluation state. Retain optimizer
    # iterates too, so averaged and query predictions can both be audited.
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "config": asdict(config),
            "normalization": NORMALIZATION,
            "data_protocol": config.data_protocol,
            "sampler_order": task.sampler.order,
            "sampler_rng": task.sampler.random.state,
        },
        directory / "checkpoint.pt",
    )
    write_json(directory / "result.json", status)
    return status


def public_forecasts(wrapper: DiffGPT, frames: Frames, writes: int) -> np.ndarray:
    """Forecast every compute row by DiffGPT.predict: a and b given, y sampled."""
    forecasts = []
    for frame in frames.raw.astype(np.float64):
        context = pd.DataFrame(frame[:writes], columns=COLUMNS)
        future = pd.DataFrame(frame[writes:], columns=COLUMNS).assign(y=0.0)
        forecast = wrapper.predict(
            context,
            future_covariates_df=future,
            target_columns=["y"],
            sampler=TemperatureSampler(temperature=0.0),
        )
        forecasts.append(np.rint(forecast["y"].to_numpy()[writes:]).astype(np.int64))
    return np.stack(forecasts)


@torch.no_grad()
def verify_checkpoint(
    directory: Path,
    device: str = "cpu",
    *,
    full_predict: bool = False,
) -> dict[str, Any]:
    """Reload the saved averaged weights; replay both splits and DiffGPT.predict."""
    metadata = json.loads((directory / "metadata.json").read_text())
    config = config_from_record(metadata)
    if metadata["codec"]["encoder_version"] != ENCODER_VERSION:
        msg = "checkpoint was trained with a different encoder version"
        raise ValueError(msg)
    checkpoint = torch.load(
        directory / "checkpoint.pt", map_location=device, weights_only=True
    )
    if config_from_record(checkpoint) != config:
        msg = "checkpoint configuration disagrees with its metadata"
        raise ValueError(msg)
    task = Task(config)
    for split, frames in task.evaluation.items():
        if digest(frames.raw) != metadata["data"][f"{split}_sha256"]:
            msg = f"regenerated {split} frames differ from the measured ones"
            raise ValueError(msg)
    data = {
        split: task.tensors(frames, device) for split, frames in task.evaluation.items()
    }
    model = build(config, device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    metrics, predicted = evaluate_splits(model, data)
    for split, values in metrics.items():
        expected = checkpoint["metrics"][split]
        if (
            abs(values["loss"] - expected["loss"]) > REPLAY_LOSS_TOLERANCE
            or values["accuracy"] != expected["accuracy"]
        ):
            msg = f"checkpoint {split} metrics disagree: {values} != {expected}"
            raise ValueError(msg)
    # DiffGPT.predict answers each compute row after its own earlier answers.
    frames, (inputs, targets) = task.evaluation["test"], data["test"]
    tokens = (
        predicted["test"] if config.computed == 1 else rollout(model, inputs, targets)
    )
    expected_levels = task.codec.decoded[tokens.cpu().numpy()]
    count = len(frames.raw)
    indices = np.arange(count) if full_predict else np.unique(np.array(PROBES) % count)
    wrapper = DiffGPT(
        model,
        order_of_derivative=0,
        domain_of_definition=np.repeat(DOMAIN, len(COLUMNS)),
        tokenizer="lgt",
    )
    public = public_forecasts(wrapper, frames.select(indices), config.writes)
    disagreements = int(
        np.count_nonzero((public != expected_levels[indices]).any(axis=1))
    )
    if disagreements and not full_predict:
        msg = "DiffGPT.predict disagrees with the benchmark logits and codec"
        raise ValueError(msg)
    truth = frames.raw[indices, config.writes :, VALUE]
    correct = int(np.count_nonzero(public == truth))
    result: dict[str, Any] = {
        "epoch": checkpoint["epoch"],
        "metrics": metrics,
        "diffgpt_predict": {
            "frames": len(indices),
            "rows": int(truth.size),
            "correct": correct,
            "accuracy": correct / truth.size,
            "disagreements_with_benchmark": disagreements,
            "complete_test_split": full_predict,
        },
    }
    optimizer = ScheduleFreeAdamW(model.parameters())
    optimizer.load_state_dict(checkpoint["optimizer"])
    optimizer.train()
    query_metrics, _ = evaluate_splits(model, data)
    with (directory / "history.csv").open(newline="") as handle:
        last = list(csv.DictReader(handle))[-1]
    for split, values in query_metrics.items():
        if values["accuracy"] != float(last[f"query_{split}_accuracy"]):
            msg = f"restored query weights disagree with the measured {split} accuracy"
            raise ValueError(msg)
    result["query_metrics"] = query_metrics
    write_json(directory / "verification.json", result)
    return result


def _arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "runs/grokking-div97-mqar"
    )
    parser.add_argument("--plan", type=Path)
    parser.add_argument(
        "--verify", type=Path, help="Directory containing a saved checkpoint"
    )
    parser.add_argument(
        "--full-predict",
        action="store_true",
        help="Verify every test frame through DiffGPT.predict",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--diagnostics-every",
        type=int,
        default=0,
        help="Extended probes for both optimizer states every N epochs; 0 disables",
    )
    parser.add_argument("--diagnostics-frames", type=int, default=512)
    parser.add_argument(
        "--snapshot-every",
        type=int,
        default=10,
        help="With diagnostics, save both weight states every N epochs; 0 disables",
    )
    types = {"int": int, "float": float, "str": str}
    choices = {"labels": LABELS, "contexts": CONTEXTS}
    for field in fields(RunConfig):
        flag = "--" + field.name.replace("_", "-")
        if field.type == "bool":
            parser.add_argument(
                flag, action=argparse.BooleanOptionalAction, default=field.default
            )
        else:
            parser.add_argument(
                flag,
                type=types[field.type],
                default=field.default,
                choices=choices.get(field.name),
            )
    return parser


def main() -> None:
    """Run one experiment, a resumable JSON plan, or checkpoint verification."""
    args = _arguments().parse_args()
    logging = (
        DiagnosticConfig(
            every=args.diagnostics_every,
            frames=args.diagnostics_frames,
            snapshot_every=args.snapshot_every,
        )
        if args.diagnostics_every
        else None
    )
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("highest")
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if args.verify:
        print(
            json.dumps(
                verify_checkpoint(
                    args.verify, args.device, full_predict=args.full_predict
                ),
                indent=2,
            )
        )
        return
    if args.plan:
        configs = [RunConfig(**config) for config in json.loads(args.plan.read_text())]
    else:
        configs = [
            RunConfig(
                **{field.name: getattr(args, field.name) for field in fields(RunConfig)}
            )
        ]
    args.output.mkdir(parents=True, exist_ok=True)
    for config in configs:
        directory = args.output / config.name()
        if (directory / "result.json").exists():
            recorded = json.loads((directory / "metadata.json").read_text())
            if config_from_record(recorded) != config:
                msg = f"different configuration already exists at {directory}"
                raise ValueError(msg)
            if not (directory / "verification.json").exists():
                verify_checkpoint(directory, args.device)
            print(f"Completed: {directory}", flush=True)
            continue
        if directory.exists():
            msg = (
                f"incomplete run exists at {directory}; "
                "preserve it and use another --output"
            )
            raise FileExistsError(msg)
        run_experiment(
            config,
            directory,
            device=args.device,
            log_every=args.log_every,
            logging=logging,
        )
        verify_checkpoint(directory, args.device)


if __name__ == "__main__":
    main()
