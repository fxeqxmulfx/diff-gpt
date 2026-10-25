"""
The 97 GIFT-Eval configurations, named as the leaderboard names them.

The dataset lists, the terms and the names follow gift-eval's
notebooks/naive.ipynb at commit 3c7be6d, the code leaderboards.vendored
copies. A configuration is a gift-eval dataset at one term, and the
leaderboard calls it "<dataset>/<frequency>/<term>": the dataset part is
lowercased and renamed by PRETTY_NAMES, and a dataset name without a
frequency takes it from notebooks/dataset_properties.json, which also gives
the domain and the number of variates of each row.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
GIFT_REPO = ROOT.parent / "gift-eval"  # The clone at commit 3c7be6d.
DATA = ROOT / "datasets" / "gift-eval"  # huggingface.co/datasets/Salesforce/GiftEval
RUNS = ROOT / "runs" / "leaderboards" / "gift-eval"
SPLITS = ("test", "val")

SHORT_DATASETS = [
    "m4_yearly",
    "m4_quarterly",
    "m4_monthly",
    "m4_weekly",
    "m4_daily",
    "m4_hourly",
    "electricity/15T",
    "electricity/H",
    "electricity/D",
    "electricity/W",
    "solar/10T",
    "solar/H",
    "solar/D",
    "solar/W",
    "hospital",
    "covid_deaths",
    "us_births/D",
    "us_births/M",
    "us_births/W",
    "saugeenday/D",
    "saugeenday/M",
    "saugeenday/W",
    "temperature_rain_with_missing",
    "kdd_cup_2018_with_missing/H",
    "kdd_cup_2018_with_missing/D",
    "car_parts_with_missing",
    "restaurant",
    "hierarchical_sales/D",
    "hierarchical_sales/W",
    "LOOP_SEATTLE/5T",
    "LOOP_SEATTLE/H",
    "LOOP_SEATTLE/D",
    "SZ_TAXI/15T",
    "SZ_TAXI/H",
    "M_DENSE/H",
    "M_DENSE/D",
    "ett1/15T",
    "ett1/H",
    "ett1/D",
    "ett1/W",
    "ett2/15T",
    "ett2/H",
    "ett2/D",
    "ett2/W",
    "jena_weather/10T",
    "jena_weather/H",
    "jena_weather/D",
    "bitbrains_fast_storage/5T",
    "bitbrains_fast_storage/H",
    "bitbrains_rnd/5T",
    "bitbrains_rnd/H",
    "bizitobs_application",
    "bizitobs_service",
    "bizitobs_l2c/5T",
    "bizitobs_l2c/H",
]
MED_LONG_DATASETS = [
    "electricity/15T",
    "electricity/H",
    "solar/10T",
    "solar/H",
    "kdd_cup_2018_with_missing/H",
    "LOOP_SEATTLE/5T",
    "LOOP_SEATTLE/H",
    "SZ_TAXI/15T",
    "M_DENSE/H",
    "ett1/15T",
    "ett1/H",
    "ett2/15T",
    "ett2/H",
    "jena_weather/10T",
    "jena_weather/H",
    "bitbrains_fast_storage/5T",
    "bitbrains_rnd/5T",
    "bizitobs_application",
    "bizitobs_service",
    "bizitobs_l2c/5T",
    "bizitobs_l2c/H",
]
PRETTY_NAMES = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}


@dataclass(frozen=True)
class Config:
    """One configuration: a gift-eval dataset at one term."""

    name: str  # gift-eval's dataset name, e.g. "electricity/15T".
    term: str  # "short", "medium" or "long".
    key: str  # The leaderboard's name, e.g. "electricity/15T/short".
    domain: str
    num_variates: int

    @property
    def file_key(self) -> str:
        """`key` as a file name: "/" becomes "__"."""
        return self.key.replace("/", "__")


def leaderboard_key(name: str, term: str, properties: dict[str, dict]) -> str:
    """Return the leaderboard's name of dataset `name` at `term`."""
    dataset, _, freq = name.partition("/")
    dataset = PRETTY_NAMES.get(dataset.lower(), dataset.lower())
    return f"{dataset}/{freq or properties[dataset]['frequency']}/{term}"


def configurations(gift_repo: Path = GIFT_REPO) -> list[Config]:
    """
    Return the 97 configurations: the short terms, then the medium, the long.

    Their keys, domains and numbers of variates are checked against the
    leaderboard's scripts/dataset_configurations.csv.
    """
    properties = json.loads(
        (gift_repo / "notebooks" / "dataset_properties.json").read_text()
    )
    with (gift_repo / "scripts" / "dataset_configurations.csv").open() as file:
        board = {row["dataset"]: row for row in csv.DictReader(file)}
    pairs = [(name, "short") for name in SHORT_DATASETS] + [
        (name, term) for term in ("medium", "long") for name in MED_LONG_DATASETS
    ]
    configs = []
    for name, term in pairs:
        key = leaderboard_key(name, term, properties)
        row = properties[key.split("/")[0]]
        config = Config(name, term, key, row["domain"], int(row["num_variates"]))
        listed = board.get(key, {})
        if (listed.get("domain"), listed.get("num_variates")) != (
            config.domain,
            str(config.num_variates),
        ):
            msg = f"{key}: {config} disagrees with dataset_configurations.csv"
            raise ValueError(msg)
        configs.append(config)
    if sorted(config.key for config in configs) != sorted(board):
        msg = "the configurations differ from dataset_configurations.csv"
        raise ValueError(msg)
    return configs


def select(configs: list[Config], only: list[str] | None) -> list[Config]:
    """Return the configurations named in `only` (keys or file keys), or all."""
    if not only:
        return configs
    wanted = {name.replace("__", "/") for name in only}
    unknown = wanted - {config.key for config in configs}
    if unknown:
        msg = f"unknown configurations: {sorted(unknown)}"
        raise ValueError(msg)
    return [config for config in configs if config.key in wanted]
