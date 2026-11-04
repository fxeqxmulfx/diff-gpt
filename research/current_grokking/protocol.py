"""Predeclared task matrix; every trained network is the public GPT."""

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEAN = Path("/home/misha/lean_projects/transformer")
ARITHMETIC = Path("/home/misha/rust_projects/grokking/experiments/training")


@dataclass(frozen=True)
class Case:
    """A task, serialization, model size and bounded training contract."""

    name: str
    kind: str
    data: dict = field(default_factory=dict)
    width: int = 128
    layers: int = 1
    heads: int = 4
    batch: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 0.1
    warmup: int = 50
    max_updates: int = 100_000
    observe: int = 250
    after_target: int = 500
    metric: str = "sequence_accuracy"
    target: float = 0.99
    data_seed: int = 1


def cases() -> list[Case]:
    """Cover each catalog label, then the distinct memory/series/probe tasks."""
    catalog = json.loads((ROOT / "research/grokking_task_inventory.json").read_text())
    rows = []
    names = [row["name"] for row in catalog["arithmetic_function_catalog"]["tasks"]]
    names.sort(key=lambda name: (name != "div", name))
    rows.extend(
        Case(
            "arithmetic-" + name.replace("/", "-"),
            "arithmetic",
            {"operation": name, "prime": 97},
            max_updates=10000 if name == "random" else 100000,
        )
        for name in names
    )
    rows.extend(
        Case(f"division-p{prime}", "arithmetic", {"operation": "div", "prime": prime})
        for prime in (17, 193)
    )
    for prime in (17, 97, 193):
        rows.extend(
            Case(
                f"power-division-p{prime}-k{power}",
                "arithmetic",
                {"operation": "power-division", "prime": prime, "power": power},
            )
            for power in (0, 2, 3, 5)
        )
        rows.extend(
            Case(
                f"modular-chain-p{prime}-l{steps}",
                "chain",
                {"prime": prime, "steps": steps},
                layers=4,
            )
            for steps in (1, 2, 4, 8)
        )
    rows.extend(
        Case(
            variant["label"],
            "synthetic",
            {"label": variant["label"]},
            width=64,
            layers=2,
            batch=32,
            target=0.95,
        )
        for variant in catalog["synthetic_algorithmic_suite"]["variants"]
    )
    rows.append(
        Case(
            "random-lm-control",
            "synthetic",
            {"label": "control-seed0"},
            width=64,
            layers=2,
            batch=32,
            max_updates=10_000,
            target=0.95,
        )
    )
    for name, kind, data, budget in [
        ("basis-depth-easy", "basis", {"task": "depth", "mode": "easy"}, 20_000),
        ("basis-depth-hard", "basis", {"task": "depth", "mode": "hard"}, 100_000),
        ("basis-recall-easy", "basis", {"task": "recall", "mode": "easy"}, 20_000),
        ("basis-recall-hard", "basis", {"task": "recall", "mode": "hard"}, 3000),
        ("basis-parity16", "basis", {"task": "parity", "bits": 16}, 100_000),
        ("basis-parity20", "basis", {"task": "parity", "bits": 20}, 100_000),
        ("basis-parity24", "basis", {"task": "parity", "bits": 24}, 100_000),
    ]:
        hard = data.get("mode") == "hard"
        task = data["task"]
        rows.append(
            Case(
                name,
                kind,
                data,
                width=128 if hard else 64,
                layers=6 if hard else 2,
                batch=64 if task == "recall" else 16,
                learning_rate=0.0003 if task in ("recall", "parity") else 0.001,
                max_updates=budget,
                observe=50 if task == "recall" else 250,
            )
        )
    rows.extend(
        Case(f"differenced-division-k{order}", "differenced_division", {"order": order})
        for order in (1, 2)
    )
    rows.extend(
        Case(
            f"recurrence-{task}-k{order}-c{context}-{tokenizer}",
            "recurrence",
            {"task": task, "order": order, "context": context, "tokenizer": tokenizer},
            heads=1 if task == "fibonacci" else 4,
            batch=512,
        )
        for (task, order, context, tokenizer) in [
            ("division", 1, 4, "lgt"),
            ("division", 2, 4, "lgt"),
            ("division", 1, 48, "lgt"),
            ("fibonacci", 1, 2, "lgt"),
            ("fibonacci", 1, 2, "mash"),
            ("fibonacci", 1, 2, "mash-half"),
            ("fibonacci", 2, 2, "lgt"),
        ]
    )
    rows.extend(
        Case(
            f"associative-{labels}-n{loads}-l{compute}-w{overwrites}",
            "associative",
            {
                "loads": loads,
                "labels": labels,
                "compute": compute,
                "overwrites": overwrites,
            },
            layers=2,
            batch=256,
        )
        for (loads, labels, compute, overwrites) in [
            (2, "dividend", 1, 0),
            (2, "divisor", 1, 0),
            (2, "division", 1, 0),
            (8, "division", 1, 0),
            (8, "division", 2, 0),
            (8, "division", 4, 0),
            (8, "division", 1, 4),
            (2, "random", 1, 0),
        ]
    )
    rows.extend(
        Case(
            "weather-formula-" + formula,
            "series",
            {"dataset": "weather_formula", "formula": formula, "max_train": 0},
            width=64,
            batch=512,
            max_updates=30000,
            metric="skill",
            target=0.9,
        )
        for formula in ("VPact", "VPdef", "H2OC", "sh")
    )
    for dataset, order, context in [
        ("weather", 0, 4),
        ("weather", 1, 4),
        ("ar", 0, 1),
        ("seasonal_ar", 0, 24),
        ("gaussian_ar", 0, 1),
        ("gaussian_seasonal_ar", 0, 24),
        ("weather", 0, 8),
        ("etth1", 0, 24),
        ("m4_hourly", 0, 24),
    ]:
        exact_floor = dataset in (
            "ar",
            "seasonal_ar",
            "gaussian_ar",
            "gaussian_seasonal_ar",
        )
        rows.append(
            Case(
                f"series-{dataset}-k{order}-c{context}",
                "series",
                {"dataset": dataset, "order": order, "context": context},
                width=64,
                batch=512,
                max_updates=30_000,
                observe=0,
                metric="skill" if exact_floor else "relative_ce_gain",
                target=0.9 if exact_floor else 0.1,
            )
        )
    for task in ("div97", "fibonacci"):
        rows.extend(
            Case(
                f"noise-{task}-{noise:g}",
                "series",
                {"dataset": task, "noise": noise, "order": int(task == "fibonacci")},
                heads=1 if task == "fibonacci" else 4,
                batch=512,
                metric="accuracy",
                observe=0,
            )
            for noise in (0.01, 0.05, 0.2)
        )
    rows.extend(
        Case(
            "gpt-probe-" + name,
            "probe",
            {"task": name},
            width=64,
            layers=2,
            batch=32,
            max_updates=30000,
            metric="sequence_accuracy"
            if name in ("order-discrimination", "key-value-binding")
            else "negative_mse",
            target=0.99
            if name in ("order-discrimination", "key-value-binding")
            else -0.001,
        )
        for name in (
            "routing-mixture",
            "branch-amplitudes",
            "conditional-amplitudes",
            "linear-transport",
            "order-discrimination",
            "key-value-binding",
        )
    )
    rows.append(
        Case(
            "fashion-mnist-gpt",
            "fashion",
            width=64,
            layers=2,
            batch=64,
            max_updates=30_000,
            target=0.9,
        )
    )
    rows.append(
        Case(
            "tiny-shakespeare-gpt",
            "text",
            width=64,
            layers=4,
            batch=32,
            max_updates=10_000,
            metric="negative_loss",
            target=-1.63,
        )
    )
    rows.append(
        Case(
            "basis-parity-noise20",
            "basis",
            {"task": "parity", "bits": 16, "noise": 0.2},
            width=64,
            layers=2,
            batch=64,
            learning_rate=0.0003,
        )
    )
    rows.extend(
        Case(
            f"language-l{depth}",
            "basis",
            {"task": "depth", "mode": "hard", "blocks": depth, "neutral_fraction": 0.0},
            width=128,
            layers=6,
            batch=16,
        )
        for depth in range(1, 7)
    )
    rows.extend(
        Case(
            f"series-{name}-shuffled-time",
            "series",
            {"dataset": name, "context": context, "shuffled_time": True},
            width=64,
            batch=512,
            max_updates=30000,
            metric="relative_ce_gain",
            target=0.1,
        )
        for (name, context) in (("weather", 8), ("etth1", 24), ("m4_hourly", 24))
    )
    if len({row.name for row in rows}) != len(rows):
        message = "Task matrix contains duplicate case names"
        raise ValueError(message)
    # These protocols historically observe once per complete shuffled epoch.
    return [replace(row, observe=0) if row.kind == "series" else row for row in rows]


def records() -> list[dict]:
    """Serialize all frozen case settings without runtime objects."""
    return [asdict(row) for row in cases()]
