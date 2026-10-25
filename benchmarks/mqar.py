"""
Multi-query associative recall (MQAR): train the GPT at one length, test longer.

Run with ``uv run --locked -m benchmarks.mqar --help``. The task is the lab's
MQAR (``/home/misha/lean_projects/transformer/python``: ``lab.domain.tasks.MQAR``
and ``lab.infrastructure.benchmarks.synthetic``): Procedure 1 of Zoology
(arXiv:2312.04927v1, Appendix E.1) with a BOS token. After BOS, `pairs` rows
(key, value) bind keys to values; the rest of the input is random values,
among which `queries` of the written keys recur, and each recurring key is
supervised with its value. Every random draw is the lab's, in the lab's
order, so each split is the lab's split and has its fingerprint.

The GPT trains at `length` tokens and is tested at that length and at each
longer `ood` length: a longer test loads the same weights into a GPT whose
block_size is that length, so RoPE turns the queries and keys by angles never
trained on. `rope` and `rope_base` set the GPT's RoPE layers and base.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch.nn import functional as F

from diff_gpt.domain.model.gpt import GPT
from diff_gpt.domain.optimizer.schedule_free import ScheduleFreeAdamW

if TYPE_CHECKING:
    from collections.abc import Iterator

BOS = 1
IDENTITY_BASE = 36  # The lab's structural tokens and digits come first.
IGNORE = -100
SPLIT_VERSION = 2  # The lab's GENERATOR_VERSION: hashed into split seeds.
PAD_VOCAB_TO = 64
SCORE_BUDGET = 1 << 25  # Attention scores per evaluation batch: B·heads·T².
EVALUATION_FIELDS = ("test", "ood")  # Settings that leave training unchanged.
ROOT = Path(__file__).resolve().parents[1]

Example = tuple[list[int], list[int]]  # Tokens and their targets.


@dataclass(frozen=True)
class MQAR:
    """
    MQAR task: `pairs` writes bind keys to values, `queries` written keys recur.

    Keys are the `symbols` tokens from IDENTITY_BASE, values the next
    `symbols`. Queries sit at least `query_gap` tokens past the writes, at
    positions drawn without replacement with weight position^-alpha.
    `overwrites` of the writes rebind a key already written, each to a value
    other than its current one; a query is supervised with the latest value
    of its key. Queries ask distinct keys.
    """

    symbols: int = 32
    pairs: int = 8
    queries: int = 4
    query_gap: int = 0
    alpha: float = 0.1
    overwrites: int = 0

    def __post_init__(self) -> None:
        """Reject controls the lab's MQAR rejects; ValueError."""
        checks = {
            "require 0 <= overwrites < pairs": 0 <= self.overwrites < self.pairs,
            "require 1 <= queries <= pairs - overwrites <= symbols": (
                1 <= self.queries <= self.pairs - self.overwrites <= self.symbols
            ),
            "a rewrite needs a second value": not self.overwrites or self.symbols > 1,
            "query_gap must be nonnegative": self.query_gap >= 0,
            "alpha must be finite and nonnegative": (
                math.isfinite(self.alpha) and self.alpha >= 0
            ),
        }
        for msg, ok in checks.items():
            if not ok:
                raise ValueError(msg)

    @property
    def vocab(self) -> int:
        """Return the number of token ids the task uses, from 0."""
        return IDENTITY_BASE + 2 * self.symbols

    @property
    def end(self) -> int:
        """Return the position after the writes: BOS, then (key, value) rows."""
        return 1 + 2 * self.pairs

    def identity(self, length: int, min_length: int | None) -> dict[str, Any]:
        """
        Return what seeds and fingerprints a split of problems of these lengths.

        A split without rewrites keeps the lab's historical seed: the
        `overwrites` control appears only when nonzero.
        """
        overwrites = {"overwrites": self.overwrites} if self.overwrites else {}
        return {
            "task": "mqar",
            "length": length,
            "min_length": min_length,
            "symbols": self.symbols,
            "pairs": self.pairs,
            "queries": self.queries,
            "query_gap": self.query_gap,
            "alpha": float(self.alpha),
            **overwrites,
        }


def uniform_draws(rng: random.Random, start: int, stop: int, count: int) -> list[int]:
    """
    Return `count` draws of `rng.randrange(start, stop)`, from the same stream.

    CPython's `randrange` draws `getrandbits(n.bit_length())` until it falls
    below n = stop - start; so does this, without the calls around each draw.
    """
    width = stop - start
    if width < 1:
        msg = "need a nonempty range"
        raise ValueError(msg)
    bits = width.bit_length()
    draws = []
    for _ in range(count):
        draw = rng.getrandbits(bits)
        while draw >= width:
            draw = rng.getrandbits(bits)
        draws.append(start + draw)
    return draws


def weighted_positions(
    available: range, count: int, alpha: float, rng: random.Random
) -> list[int]:
    """
    Return `count` of the positions, without replacement, weighted position^-alpha.

    Exponential races in log space: the smallest log(E) + alpha·log(position),
    E ~ Exp(1), win; ties go to the earlier position.
    """
    tiny = sys.float_info.min
    scores = sorted(
        (
            math.log(-math.log(max(rng.random(), tiny))) + alpha * math.log(position),
            position,
        )
        for position in available
    )
    return [position for _, position in scores[:count]]


def sample(task: MQAR, rng: random.Random, length: int) -> list[int]:
    """Return the tokens of one problem of `length`, drawn as the lab draws them."""
    first_value = IDENTITY_BASE + task.symbols
    distinct = rng.sample(
        range(IDENTITY_BASE, first_value), task.pairs - task.overwrites
    )
    keys = distinct + [rng.choice(distinct) for _ in range(task.overwrites)]
    if task.overwrites:
        rng.shuffle(keys)
    values: list[int] = []
    current: dict[int, int] = {}
    for key in keys:
        # A rewrite draws uniformly among the values other than the current one.
        value = rng.randrange(first_value, task.vocab - (key in current))
        if key in current and value >= current[key]:
            value += 1
        current[key] = value
        values.append(value)
    tokens = [BOS, *uniform_draws(rng, first_value, task.vocab, length - 1)]
    for row, (key, value) in enumerate(zip(keys, values, strict=True)):
        tokens[1 + 2 * row : 3 + 2 * row] = [key, value]
    available = range(task.end + task.query_gap, length)
    positions = weighted_positions(available, task.queries, task.alpha, rng)
    for position, key in zip(
        positions, rng.sample(distinct, task.queries), strict=True
    ):
        tokens[position] = key
    return tokens


def labels(task: MQAR, tokens: list[int]) -> list[int]:
    """
    Return the target of every position: a query's latest value, else IGNORE.

    The oracle reads the raw input alone and checks the task's contract on
    it: valid writes, queries of distinct written keys past `query_gap`,
    value tokens everywhere else; ValueError otherwise.
    """
    first_value = IDENTITY_BASE + task.symbols
    end = task.end
    if not tokens or tokens[0] != BOS or len(tokens) < end:
        msg = "incomplete MQAR prefix"
        raise ValueError(msg)
    table: dict[int, int] = {}
    for position in range(1, end, 2):
        key, value = tokens[position], tokens[position + 1]
        if (
            not IDENTITY_BASE <= key < first_value
            or not first_value <= value < task.vocab
        ):
            msg = "invalid MQAR write"
            raise ValueError(msg)
        if table.get(key) == value or (key in table and not task.overwrites):
            msg = "a rewrite must change the value of an earlier key"
            raise ValueError(msg)
        table[key] = value
    if len(table) != task.pairs - task.overwrites:
        msg = "the writes must bind pairs - overwrites distinct keys"
        raise ValueError(msg)
    targets = [IGNORE] * len(tokens)
    seen: set[int] = set()
    for position in range(end, len(tokens)):
        token = tokens[position]
        if IDENTITY_BASE <= token < first_value:
            if position < end + task.query_gap or token not in table or token in seen:
                msg = "an MQAR query must repeat one distinct prior key"
                raise ValueError(msg)
            targets[position] = table[token]
            seen.add(token)
        elif not first_value <= token < task.vocab:
            msg = "MQAR fillers must be value tokens"
            raise ValueError(msg)
    if len(seen) != task.queries:
        msg = "wrong number of supervised queries"
        raise ValueError(msg)
    return targets


def stream(
    task: MQAR, length: int, min_length: int | None, split: str, seed: int
) -> Iterator[Example]:
    """
    Yield every example a split can hold, in order: the lab's endless stream.

    Its generator is seeded by the problem, the data seed and the split name.
    Each draw is two problems of one length in random order, so row parity
    hides nothing.
    """
    identity = json.dumps(
        {
            "version": SPLIT_VERSION,
            "spec": task.identity(length, min_length),
            "seed": seed,
            "split": split,
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(identity.encode()).digest()
    # The lab's splits are Mersenne Twister streams; nothing here is a secret.
    rng = random.Random(int.from_bytes(digest, "big"))  # noqa: S311
    minimum = length if min_length is None else min_length
    while True:
        size = rng.randint(minimum, length)
        pair = (sample(task, rng, size), sample(task, rng, size))
        for tokens in pair[::-1] if rng.random() < 0.5 else pair:  # noqa: PLR2004
            yield tokens, labels(task, tokens)


def build_split(
    task: MQAR,
    length: int,
    split: str,
    seed: int,
    count: int,
    min_length: int | None = None,
) -> list[Example]:
    """Return the first `count` examples of the split's stream."""
    examples = stream(task, length, min_length, split, seed)
    return [next(examples) for _ in range(count)]


def fingerprint(
    task: MQAR,
    length: int,
    split: str,
    seed: int,
    examples: list[Example],
    min_length: int | None = None,
) -> str:
    """
    Return the lab's fingerprint of a split: SHA-256 of its examples.

    The lab hashes each example's fields, the positions where the
    supervised target changes included, with its name, problem and seed.
    """
    records = []
    for tokens, targets in examples:
        changes = []
        previous = None
        for target in targets:
            changes.append(target not in (IGNORE, previous))
            if target != IGNORE:
                previous = target
        records.append(
            {
                "task": "mqar",
                "tokens": tokens,
                "targets": targets,
                "changes": changes,
                "prompt": [],
                "answer": [],
                "generation_limit": 0,
            }
        )
    payload = {
        "version": SPLIT_VERSION,
        "name": split,
        "spec": task.identity(length, min_length),
        "seed": seed,
        "examples": records,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def tensors(examples: list[Example], device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the tokens and targets of equally long examples as (N, T) tensors."""
    tokens = torch.tensor([tokens for tokens, _ in examples], dtype=torch.int64)
    targets = torch.tensor([targets for _, targets in examples], dtype=torch.int64)
    return tokens.to(device), targets.to(device)


@dataclass(frozen=True)
class RunConfig:
    """
    One run: the task, its splits, the GPT and ScheduleFreeAdamW.

    Every split is drawn with `data_seed`; `seed` sets the initialization
    and the order of the training examples. Training takes `steps` batches
    of `batch` examples of `length` tokens, in shuffled epochs. `test`
    examples are scored at `length` and at every `ood` length. `rope` is
    "all" or one 0/1 per layer: "10" applies RoPE only in layer 0.
    `rope_base` is the base of the RoPE frequencies, GPT's `rope_base`.
    """

    symbols: int = 64
    pairs: int = 8
    queries: int = 8
    query_gap: int = 0
    alpha: float = 0.1
    overwrites: int = 0
    length: int = 64
    ood: tuple[int, ...] = (128, 256, 512)
    train: int = 100_000
    test: int = 2_000
    data_seed: int = 0
    width: int = 128
    heads: int = 4
    layers: int = 2
    rope: str = "all"
    rope_base: float = 10_000.0
    learning_rate: float = 3e-3
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.03
    warmup: int = 1_000
    batch: int = 64
    steps: int = 20_000
    seed: int = 0
    evaluate_every: int = 1_000

    def __post_init__(self) -> None:
        """Check the controls of the task and of the run; ValueError."""
        required = self.task.end + self.task.query_gap + self.task.queries
        if self.length < required:
            msg = f"MQAR needs at least {required} tokens"
            raise ValueError(msg)
        longer = all(length > self.length for length in self.ood)
        if not longer or len(set(self.ood)) != len(self.ood):
            msg = "ood lengths must be distinct and longer than length"
            raise ValueError(msg)
        if min(self.train, self.test, self.steps, self.batch, self.evaluate_every) < 1:
            msg = "train, test, steps, batch and evaluate_every must be positive"
            raise ValueError(msg)
        if self.rope != "all" and (
            not self.rope
            or len(self.rope) != self.layers
            or set(self.rope) - {"0", "1"}
        ):
            msg = 'rope must be "all" or one 0/1 per layer'
            raise ValueError(msg)
        if not 0 < self.rope_base < math.inf:
            msg = "rope_base must be positive and finite"
            raise ValueError(msg)

    @property
    def task(self) -> MQAR:
        """Return the task the run trains on."""
        return MQAR(
            symbols=self.symbols,
            pairs=self.pairs,
            queries=self.queries,
            query_gap=self.query_gap,
            alpha=self.alpha,
            overwrites=self.overwrites,
        )

    @property
    def vocab_size(self) -> int:
        """Return the GPT vocabulary: the task's, padded to a multiple of 64."""
        return -(-self.task.vocab // PAD_VOCAB_TO) * PAD_VOCAB_TO

    @property
    def name(self) -> str:
        """Return the run directory name: the settings that shape training."""
        overwrites = f"-o{self.overwrites}" if self.overwrites else ""
        gap = f"-gap{self.query_gap}" if self.query_gap else ""
        rope = f"-rope{self.rope}" if self.rope != "all" else ""
        if self.rope_base != RunConfig.rope_base:  # The default keeps old names.
            rope += f"-base{self.rope_base:g}"
        return (
            f"mqar-s{self.symbols}-p{self.pairs}-q{self.queries}{overwrites}{gap}"
            f"-a{self.alpha:g}-L{self.length}-n{self.train}-d{self.data_seed}"
            f"-w{self.width}-h{self.heads}-l{self.layers}{rope}"
            f"-lr{self.learning_rate:g}-wd{self.weight_decay:g}"
            f"-b{self.beta1:g}-{self.beta2:g}-wu{self.warmup}"
            f"-bs{self.batch}-steps{self.steps}-s{self.seed}"
        )

    def training(self) -> dict[str, Any]:
        """Return the settings that shape the trained weights and their curve."""
        return {
            key: value
            for key, value in asdict(self).items()
            if key not in EVALUATION_FIELDS
        }

    def model(self, block_size: int) -> GPT:
        """Return the run's GPT over `block_size` positions, on the CPU."""
        return GPT(
            vocab_size=self.vocab_size,
            n_embd=self.width,
            block_size=block_size,
            n_head=self.heads,
            n_layer=self.layers,
            use_checkpoint=False,
            rope_base=self.rope_base,
            rope_layers=None
            if self.rope == "all"
            else tuple(bit == "1" for bit in self.rope),
        )


@dataclass
class Splits:
    """The training split and a test split at each tested length, on one device."""

    train: tuple[torch.Tensor, torch.Tensor]
    test: dict[int, tuple[torch.Tensor, torch.Tensor]]
    fingerprints: dict[str, str]


def build_splits(config: RunConfig, device: str) -> Splits:
    """
    Draw the run's splits with the lab's seeds and record their fingerprints.

    The test split at `length` shares the training problem; each longer
    length L is the lab's held-out problem `length-L`, without min_length.
    """
    task, seed = config.task, config.data_seed
    train = build_split(task, config.length, "train", seed, config.train)
    fingerprints = {"train": fingerprint(task, config.length, "train", seed, train)}
    test = {}
    for length in (config.length, *config.ood):
        examples = build_split(task, length, "test", seed, config.test)
        name = "test" if length == config.length else f"test/length-{length}"
        fingerprints[name] = fingerprint(task, length, "test", seed, examples)
        test[length] = tensors(examples, device)
    return Splits(tensors(train, device), test, fingerprints)


@torch.no_grad()
def evaluate(
    model: GPT, tokens: torch.Tensor, targets: torch.Tensor
) -> dict[str, float]:
    """
    Score the model on a split: loss and accuracies at the supervised positions.

    `token_accuracy` counts correct queries, `sequence_accuracy` examples
    whose every query is correct; predictions are the argmax of the logits.
    """
    batch = max(1, SCORE_BUDGET // (model.n_head * tokens.size(1) ** 2))
    loss = torch.zeros((), dtype=torch.float64, device=tokens.device)
    correct = torch.zeros((), dtype=torch.int64, device=tokens.device)
    exact = torch.zeros((), dtype=torch.int64, device=tokens.device)
    for start in range(0, tokens.size(0), batch):
        x, y = tokens[start : start + batch], targets[start : start + batch]
        logits, _ = model(x)
        supervised = y != IGNORE
        losses = F.cross_entropy(logits.transpose(1, 2), y, reduction="none")
        hits = (logits.argmax(-1) == y) & supervised
        loss += losses[supervised].double().sum()
        correct += hits.sum()
        exact += (hits | ~supervised).all(dim=1).sum()
    queries = int((targets != IGNORE).sum())
    return {
        "loss": float(loss) / queries,
        "token_accuracy": int(correct) / queries,
        "sequence_accuracy": int(exact) / tokens.size(0),
    }


def evaluate_lengths(
    config: RunConfig, state: dict[str, torch.Tensor], splits: Splits, device: str
) -> dict[str, dict[str, float]]:
    """
    Score the weights `state` on the test split of every length, keyed by it.

    A longer length loads them into a GPT whose block_size is that length.
    """
    results = {}
    for length, (tokens, targets) in splits.test.items():
        model = config.model(length).to(device)
        model.load_state_dict(state)
        model.eval()
        results[str(length)] = evaluate(model, tokens, targets)
    return results


def train(
    config: RunConfig, splits: Splits, device: str, *, log: bool
) -> tuple[GPT, list[dict[str, Any]]]:
    """
    Train the GPT on the training split; return it with evaluation weights.

    Each epoch visits the training examples in a fresh order; a batch never
    spans two epochs. Every `evaluate_every` steps the averaged weights are
    scored on the test split at the training length.
    """
    torch.manual_seed(config.seed)
    model = config.model(config.length).to(device)
    optimizer = ScheduleFreeAdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
        warmup=config.warmup,
    )
    tokens, targets = splits.train
    test_tokens, test_targets = splits.test[config.length]
    order = torch.Generator().manual_seed(config.seed)
    epoch = torch.empty(0, dtype=torch.int64)
    history = []
    running = torch.zeros((), device=device)
    model.train()
    for step in range(1, config.steps + 1):
        if epoch.numel() < config.batch:
            epoch = torch.randperm(tokens.size(0), generator=order)
        indices, epoch = epoch[: config.batch].to(device), epoch[config.batch :]
        _, loss = model(tokens[indices], targets[indices])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        running += loss.detach()
        if step % config.evaluate_every == 0 or step == config.steps:
            optimizer.eval()
            model.eval()
            seen = step % config.evaluate_every or config.evaluate_every
            row = {
                "step": step,
                "train_loss": float(running) / seen,
                **evaluate(model, test_tokens, test_targets),
            }
            history.append(row)
            if log:
                print(json.dumps(row), flush=True)
            running.zero_()
            optimizer.train()
            model.train()
    optimizer.eval()
    model.eval()
    return model, history


def run(
    config: RunConfig, output: Path, device: str, *, log: bool = True
) -> dict[str, Any]:
    """
    Train one run, or read it back, and score its weights at every length.

    `output/<name>/model.pt` holds the evaluation weights and
    `output/<name>/result.json` the settings, fingerprints, FLOPs, learning
    curve and the scores of the latest evaluation. Saved weights are scored
    again, not retrained, so `test` and `ood` may change between calls;
    ValueError if the saved run was trained with other settings.
    """
    directory = output / config.name
    model_path, result_path = directory / "model.pt", directory / "result.json"
    splits = build_splits(config, device)
    if result_path.exists():
        result = json.loads(result_path.read_text())
        # A setting added after the run was trained had its default.
        trained = {**RunConfig().training(), **result["training"]}
        if trained != config.training():
            msg = f"{directory} holds a run trained with other settings"
            raise ValueError(msg)
        state = torch.load(model_path, map_location="cpu", weights_only=True)
    else:
        started = time.perf_counter()
        model, history = train(config, splits, device, log=log)
        seconds = time.perf_counter() - started
        state = {
            name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
        }
        trained_tokens = config.steps * config.batch * config.length
        result = {
            "name": config.name,
            "training": config.training(),
            "parameters": sum(p.numel() for p in model.parameters()),
            "train_flops": model.estimate_flops() * trained_tokens,
            "train_seconds": seconds,
            "device": torch.cuda.get_device_name(device)
            if device.startswith("cuda")
            else device,
            "torch": torch.__version__,
            "history": history,
        }
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(state, model_path)
        result_path.write_text(json.dumps(result, indent=2) + "\n")
    result["evaluation"] = {key: getattr(config, key) for key in EVALUATION_FIELDS}
    result["fingerprints"] = splits.fingerprints
    result["test"] = evaluate_lengths(config, state, splits, device)
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    """Train the configured runs, one per seed, and print their scores."""
    defaults = RunConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/mqar")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[defaults.seed])
    parser.add_argument("--ood", type=int, nargs="*", default=list(defaults.ood))
    for item in RunConfig.__dataclass_fields__.values():
        if item.name in {"seed", "ood"}:
            continue
        parser.add_argument(
            "--" + item.name.replace("_", "-"),
            type=type(getattr(defaults, item.name)),
            default=getattr(defaults, item.name),
        )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    # Full FP32 matmuls: TF32 rounds their inputs to 10 bits of mantissa.
    torch.set_float32_matmul_precision("highest")
    settings = {
        name: getattr(args, name)
        for name in RunConfig.__dataclass_fields__
        if name not in {"seed", "ood"}
    }
    for seed in args.seeds:
        config = RunConfig(**settings, ood=tuple(args.ood), seed=seed)
        result = run(config, args.output, args.device, log=not args.quiet)
        scores = {
            key: round(value["token_accuracy"], 4)
            for key, value in result["test"].items()
        }
        print(
            json.dumps({"name": result["name"], "token_accuracy": scores}), flush=True
        )


if __name__ == "__main__":
    main()
