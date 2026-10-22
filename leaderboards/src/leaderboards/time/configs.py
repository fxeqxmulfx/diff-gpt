"""
The 98 TIME tasks, named as the leaderboard names them.

A task is a TIME dataset at one term, "<dataset>/<freq>/<term>" (e.g.
"SG_Weather/D/short"). The datasets, their terms and their prediction, test
and validation lengths are those of timebench's config/datasets.yaml at the
commit pinned in pyproject.toml, and the terms of a dataset are the ones the
file defines, as TIME's experiments take them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from timebench.evaluation.data import load_dataset_config
from timebench.evaluation.utils import get_available_terms

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "datasets" / "time"  # huggingface.co/datasets/Real-TSF/TIME
HOSTED = ROOT / "datasets" / "time-output" / "results"  # Real-TSF/TIME-Output
RUNS = ROOT / "runs" / "leaderboards" / "time"
SPLITS = ("test", "val")


@dataclass(frozen=True)
class Task:
    """One task: a TIME dataset at one term."""

    name: str  # TIME's dataset name, "<dataset>/<freq>", e.g. "SG_Weather/D".
    term: str  # "short", "medium" or "long".
    prediction_length: int
    test_length: int  # The steps of the test windows, at the end of each series.
    val_length: int  # The steps of the validation windows, right before them.

    @property
    def key(self) -> str:
        """The leaderboard's name, e.g. "SG_Weather/D/short"."""
        return f"{self.name}/{self.term}"

    @property
    def file_key(self) -> str:
        """`key` as a file name: "/" becomes "__"."""
        return self.key.replace("/", "__")


def tasks(config: dict | None = None) -> list[Task]:
    """Return the tasks of `config` (default: timebench's), in its order."""
    if config is None:
        config = load_dataset_config()
    return [
        Task(
            name,
            term,
            settings[term]["prediction_length"],
            settings["test_length"],
            settings["val_length"],
        )
        for name, settings in config["datasets"].items()
        for term in get_available_terms(name, config)
    ]


def select(tasks: list[Task], only: list[str] | None) -> list[Task]:
    """Return the tasks named in `only` (keys or file keys), or all."""
    if not only:
        return tasks
    wanted = {name.replace("__", "/") for name in only}
    unknown = wanted - {task.key for task in tasks}
    if unknown:
        msg = f"unknown tasks: {sorted(unknown)}"
        raise ValueError(msg)
    return [task for task in tasks if task.key in wanted]
