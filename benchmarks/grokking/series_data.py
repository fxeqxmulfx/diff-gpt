"""
Frozen, train-fitted datasets for the real-series grokking experiments.

Tabulation uses labels only to report an empirical oracle, never to train the
model or fit the codec. Weather equations hold out whole quantized input pairs.
Temporal datasets keep chronological gaps and explicitly report vocabulary
coverage exclusions. All stochastic choices use independent, local RNG streams.
"""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from benchmarks.grokking import div97, recurrence
from benchmarks.grokking import series_gaussian as gaussian
from benchmarks.grokking.series_oracle import conditional_reference
from diff_gpt.encoder_decoder import (
    ENCODER_VERSION,
    TokenClippingWarning,
    decode,
    encode,
    get_domain_of_definition,
)

if TYPE_CHECKING:
    from collections.abc import Callable

ROOT = Path(__file__).resolve().parents[2]
WEATHER = ROOT / "datasets/long_term_forecast/weather/weather.csv"
ETTH1 = ROOT / "datasets/long_term_forecast/ETT-small/ETTh1.csv"
M4 = ROOT / "datasets/short_term_forecast/m4"
FORMULAS = {
    "VPact": ("rh (%)", "VPmax (mbar)", "VPact (mbar)"),
    "VPdef": ("VPmax (mbar)", "VPact (mbar)", "VPdef (mbar)"),
    "H2OC": ("p (mbar)", "VPact (mbar)", "H2OC (mmol/mol)"),
    "sh": ("p (mbar)", "VPact (mbar)", "sh (g/kg)"),
}
MIN_STD = 1e-12
TRAIN_PAIR_FRACTION = 0.5
VOCAB_MULTIPLE = 64
MIN_CALIBRATION = 2
GAUSSIAN_DATASETS = {"gaussian_ar", "gaussian_seasonal_ar"}


def array_hash(array: np.ndarray) -> str:
    """Hash shape, dtype and canonical bytes without converting floats to ints."""
    value = np.ascontiguousarray(array.astype(array.dtype.newbyteorder("<")))
    header = f"{value.dtype.str}:{value.shape}:".encode()
    return hashlib.sha256(header + value.tobytes()).hexdigest()


@dataclass(frozen=True)
class DataConfig:
    """Dataset and codec choices, separate from model/optimizer parameters."""

    dataset: str = "weather_formula"
    seed: int = 11
    order: int = 0
    vocab_size: int = 64
    tokenizer: str = "lgt"
    context: int = 2
    formula: str = "VPact"
    noise: float = 0.0
    random_labels: bool = False
    shuffled_time: bool = False
    train_fraction: float = 0.6
    max_train: int = 2048
    max_eval: int = 2048
    series_length: int = 8192
    period: int = 24
    m4_series: int = 8
    calibration_rows: int = 512
    domain_multiplier: float = 1.0
    source: str = ""


@dataclass
class Split:
    """One complete frozen split; only positions with column >= 0 have a loss."""

    x: np.ndarray
    y: np.ndarray
    clean_y: np.ndarray
    column: np.ndarray
    ids: np.ndarray
    truth: np.ndarray
    base: np.ndarray
    step: np.ndarray
    past: np.ndarray | None = None
    bayes: np.ndarray | None = None
    seasonal: np.ndarray | None = None

    def hashes(self) -> dict[str, str]:
        """Include clean labels, decoder state and reference predictions."""
        return {
            key: array_hash(value)
            for key in self.__dataclass_fields__
            if (value := getattr(self, key)) is not None
        }


@dataclass
class Dataset:
    """Model-ready splits with independent baseline and oracle diagnostics."""

    splits: dict[str, Split]
    columns: list[str]
    metadata: dict[str, Any]
    references: dict[str, dict[str, dict[str, float | None]]]

    def fingerprint(self) -> dict[str, dict[str, str]]:
        """Make changes to any evaluated or trained array detectable on resume."""
        return {name: split.hashes() for name, split in self.splits.items()}


def token_coverage(splits: dict[str, Split]) -> dict[str, dict[str, Any]]:
    """Audit actual prompt/answer token IDs against the optimized training set."""
    train = splits["train"]
    seen = set(train.x.ravel()) | set(train.y[train.y >= 0])
    return {
        name: {
            "covered": not bool(
                missing := (set(split.x.ravel()) | set(split.y[split.y >= 0])) - seen
            ),
            "missing_tokens": sorted(map(int, missing)),
        }
        for name, split in splits.items()
        if name != "train"
    }


def tabulation(contexts: np.ndarray, targets: np.ndarray) -> dict[str, float | int]:
    """Return the best lookup accuracy and conditional empirical CE in nats."""
    if not len(targets):
        msg = "cannot tabulate an empty split"
        raise ValueError(msg)
    _, group = np.unique(contexts, axis=0, return_inverse=True)
    pairs, counts = np.unique(
        np.column_stack((group, targets)), axis=0, return_counts=True
    )
    totals = np.bincount(group)
    best = np.zeros(len(totals), dtype=np.int64)
    np.maximum.at(best, pairs[:, 0], counts)
    entropy = -np.sum(counts * np.log(counts / totals[pairs[:, 0]])) / len(targets)
    return {
        "accuracy": float(best.sum() / len(targets)),
        "loss": float(entropy),
        "contexts": len(totals),
        "observations": len(targets),
    }


def _references(
    splits: dict[str, Split], columns: list[str], vocab: int, *, oracle: bool
) -> dict[str, dict[str, dict[str, float | None]]]:
    result = {name: {} for name in splits}
    train = splits["train"]
    for index, column in enumerate(columns):
        labels = train.y[train.column == index]
        counts = np.bincount(labels, minlength=vocab).astype(np.float64)
        # Jeffreys smoothing is fit on training labels alone and gives finite CE
        # when an evaluated output value was absent from supervised training.
        probability = (counts + 0.5) / (counts.sum() + 0.5 * vocab)
        for name, split in splits.items():
            mask = split.column == index
            target = split.y[mask]
            selected = np.nonzero(mask)[0]
            # Each series window supervises one value of each column. Formula
            # equations and anchors supervise one value total.
            ceiling = tabulation(split.x[selected], target) if oracle else None
            floor = float(ceiling["loss"]) if ceiling is not None else None
            accuracy = float(ceiling["accuracy"]) if ceiling is not None else None
            if split.bayes is not None:
                p = split.bayes[:, index]
                floor = float(-np.sum(p * np.log(np.maximum(p, 1e-300))) / len(p))
                accuracy = float(p.max(axis=-1).mean())
            result[name][column] = {
                "baseline_loss": float(-np.log(probability[target]).mean()),
                "baseline_accuracy": float((target == counts.argmax()).mean()),
                "floor_loss": floor,
                "ceiling_accuracy": accuracy,
            }
            if split.bayes is not None:
                result[name][column]["baseline_loss"] = float(
                    -np.sum(p * np.log(probability)) / len(p)
                )
                result[name][column]["baseline_accuracy"] = float(
                    p[:, counts.argmax()].mean()
                )
    return result


def _finish(
    splits: dict[str, Split],
    columns: list[str],
    metadata: dict[str, Any],
    config: DataConfig,
    *,
    oracle: bool = False,
) -> Dataset:
    metadata.update(
        {
            "config": asdict(config),
            "encoder_version": ENCODER_VERSION,
            "split_rows": {name: len(split.x) for name, split in splits.items()},
            "token_coverage": token_coverage(splits),
        }
    )
    data = Dataset(
        splits,
        columns,
        metadata,
        _references(splits, columns, config.vocab_size, oracle=oracle),
    )
    metadata["hashes"] = data.fingerprint()
    return data


def _read_weather(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = pd.read_csv(path)
    numeric = frame.drop(columns="date").astype(np.float64)
    missing = numeric.eq(-9999) | ~np.isfinite(numeric)
    # Fill forward causally. No interpolation across chronological splits and
    # no backwards fill from future observations. Initial missing rows drop.
    cleaned = numeric.mask(missing).ffill()
    valid = cleaned.notna().all(axis=1)
    return cleaned.loc[valid], {
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "missing_by_column": missing.sum().loc[lambda s: s > 0].to_dict(),
        "missing_policy": "causal forward fill; drop leading incomplete rows",
        "dropped_leading_rows": int((~valid).sum()),
    }


def _fit(
    values: np.ndarray, config: DataConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std < MIN_STD] = 1.0
    domain = get_domain_of_definition((values - mean) / std, config.order)
    domain *= config.domain_multiplier
    return mean, std, domain


def _encode(
    values: np.ndarray, config: DataConfig, domain: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", TokenClippingWarning)
        _, scale, tokens = encode(
            values, config.vocab_size, domain, config.order, tokenizer=config.tokenizer
        )
    clipped = sum(
        int(str(w.message).split()[0])
        for w in caught
        if isinstance(w.message, TokenClippingWarning)
    )
    return tokens, scale, clipped


def _cap(indices: np.ndarray, maximum: int, rng: np.random.Generator) -> np.ndarray:
    if maximum and len(indices) > maximum:
        return np.sort(rng.choice(indices, maximum, replace=False))
    return np.sort(indices)


def weather_formula(config: DataConfig) -> Dataset:
    """Hold out quantized input combinations with train-only codec calibration."""
    if config.order != 0 or config.formula not in FORMULAS:
        msg = "independent weather equations require k=0 and a registered formula"
        raise ValueError(msg)
    frame, provenance = _read_weather(Path(config.source) if config.source else WEATHER)
    names = FORMULAS[config.formula]
    values = frame.loc[:, list(names)].to_numpy()
    rng = np.random.default_rng(config.seed ^ 0xCA11B)
    calibration_size = max(
        MIN_CALIBRATION, min(len(values) // 20, config.calibration_rows)
    )
    calibration = np.sort(rng.choice(len(values), calibration_size, replace=False))
    mean, std, domain = _fit(values[calibration], config)
    tokens, scale, clipped = _encode((values - mean) / std, config, domain)
    pairs, groups = np.unique(tokens[:, :2], axis=0, return_inverse=True)
    train_group = rng.random(len(pairs)) < TRAIN_PAIR_FRACTION
    train_group[np.unique(groups[calibration])] = True
    # Repair coverage by assigning entire pairs to training, never individual
    # rows. This is independent of output labels. Each input position is checked.
    for position in range(2):
        for value in np.unique(tokens[:, position]):
            matching = np.flatnonzero(pairs[:, position] == value)
            if not train_group[matching].any():
                train_group[matching[0]] = True
    train_ids = np.flatnonzero(train_group[groups])
    test_ids = np.flatnonzero(~train_group[groups])
    # The pair assignment is already frozen. Optimization may sample rows from
    # its training side; calibration rows remain in that optimized sample.
    train_ids = _cap(
        train_ids, config.max_train, np.random.default_rng(config.seed ^ 0x7AA1)
    )
    # Include input token representatives in the actual optimized sample.
    full_train = np.flatnonzero(train_group[groups])
    representatives = [*train_ids, *calibration]
    for position in range(2):
        present = set(tokens[train_ids, position])
        for value in np.unique(tokens[test_ids, position]):
            if value not in present:
                representatives.append(
                    int(
                        full_train[
                            np.flatnonzero(tokens[full_train, position] == value)[0]
                        ]
                    )
                )
                present.add(value)
    train_ids = np.unique(representatives)
    test_ids = _cap(
        test_ids, config.max_eval, np.random.default_rng(config.seed ^ 0x7E57)
    )
    if not len(test_ids):
        msg = "pair split left no held-out equations"
        raise ValueError(msg)
    split_ids = {"train": train_ids, "test": test_ids}
    target_tokens = tokens[:, 2].copy()
    if config.random_labels:
        target_tokens = np.random.default_rng(config.seed ^ 0x1ABE1).permutation(
            target_tokens
        )
    splits = {}
    for name, ids in split_ids.items():
        x = tokens[ids, :2].copy()
        y = np.full_like(x, -100)
        y[:, -1] = target_tokens[ids]
        column = np.full_like(x, -1)
        column[:, -1] = 0
        base = np.full((len(ids), 1), mean[2] + 0.5 / scale[2] * std[2])
        step = np.full_like(base, std[2] / scale[2])
        splits[name] = Split(x, y, y.copy(), column, ids, values[ids, 2:3], base, step)
    training_pairs = set(map(tuple, splits["train"].x))
    if training_pairs & set(map(tuple, splits["test"].x)):
        msg = "quantized equation pairs overlap"
        raise RuntimeError(msg)
    coverage = {
        names[position]: bool(
            set(splits["test"].x[:, position]) <= set(splits["train"].x[:, position])
        )
        for position in range(2)
    }
    metadata = {
        **provenance,
        "split": "whole quantized input pairs; calibration pairs train-only",
        "codec_fit": "frozen training calibration subset only",
        "calibration_ids": calibration.tolist(),
        "calibration_pairs_in_test": 0,
        "fit_population_is_training_only": not bool(
            set(groups[calibration]) & set(groups[test_ids])
        ),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "domain": domain.tolist(),
        "scale": scale.tolist(),
        "clipped_tokens": clipped,
        "input_columns": list(names[:2]),
        "input_token_coverage": coverage,
        "unique_pairs": len(pairs),
        "full_table_oracle": tabulation(tokens[:, :2], target_tokens),
        "target_label_sha256": array_hash(target_tokens),
        "oracle_usage": (
            "evaluation diagnostic only; empirical, not a population Bayes floor"
        ),
        "raw_sha256": array_hash(values),
    }
    result = _finish(splits, [config.formula], metadata, config, oracle=True)
    for name, split in splits.items():
        reference = result.references[name][config.formula]
        reference["empirical_split_floor_loss"] = reference["floor_loss"]
        reference.update(
            conditional_reference(
                tokens[:, :2], target_tokens, split.x, split.y[:, -1], config.vocab_size
            )
        )
        if config.random_labels:
            prior = np.bincount(target_tokens, minlength=config.vocab_size) / len(
                target_tokens
            )
            reference.update(
                {
                    "input_independent_control": True,
                    "floor_loss": float(
                        -np.sum(prior * np.log(np.maximum(prior, 1e-300)))
                    ),
                    "ceiling_accuracy": float(prior.max()),
                    "oracle_realized_loss": float(
                        -np.log(prior[split.y[:, -1]]).mean()
                    ),
                    "oracle_realized_accuracy": float(
                        (split.y[:, -1] == prior.argmax()).mean()
                    ),
                }
            )
    metadata["oracle_population"] = (
        "entire equation table; conditional entropy weighted by "
        "each frozen split's contexts"
    )
    if config.random_labels:
        metadata["oracle_population"] = (
            "known marginal of the full-table random label permutation"
        )
    return result


def corrupt_answers(
    raw: np.ndarray, train: int, fraction: float, seed: int
) -> tuple[np.ndarray, dict[str, Any]]:
    """Replace a fixed fraction of training answers; leave all prompts/test clean."""
    if not 0 <= fraction <= 1:
        msg = "noise must lie in [0, 1]"
        raise ValueError(msg)
    result = raw.copy()
    rng = np.random.default_rng(seed ^ 0xC077)
    selected = np.sort(rng.choice(train, round(train * fraction), replace=False))
    result[selected, -1] = rng.integers(0, div97.P, len(selected))
    return result, {
        "selected_training_answers": len(selected),
        "changed_training_answers": int(np.count_nonzero(result[:, -1] != raw[:, -1])),
        "selected_ids_sha256": array_hash(selected),
        "heldout_answers_changed": bool(np.any(result[train:] != raw[train:])),
    }


def anchor(config: DataConfig) -> Dataset:
    """Expose the exact existing anchor splits/codecs to the shared noise loop."""
    if config.vocab_size != div97.VOCAB:
        msg = "anchor comparisons require the original vocabulary 256"
        raise ValueError(msg)
    if config.dataset == "div97":
        if config.order != 0:
            msg = "the division anchor has k=0"
            raise ValueError(msg)
        raw = div97.division_rows(config.seed)
        train = div97.TRAIN
        lookup, codec = div97.token_table(0)
        tokens = lookup[raw]
        noisy_raw, corruption = corrupt_answers(raw, train, config.noise, config.seed)
        noisy_tokens = lookup[noisy_raw]
        info = {
            "codec": codec,
            "raw_sha256": array_hash(raw),
            "split": "original SplitMix64 equation split",
        }
    else:
        temporal = recurrence.Config(
            task="fibonacci",
            context=config.context,
            order=config.order,
            seed=config.seed,
            readout="field",
        )
        raw, train, info = recurrence.dataset(temporal)
        tokens, _ = recurrence.encode_windows(raw, temporal)
        noisy_raw, corruption = corrupt_answers(raw, train, config.noise, config.seed)
        noisy_tokens, _ = recurrence.encode_windows(noisy_raw, temporal)
    splits = {}
    for name, ids in (
        ("train", np.arange(train)),
        ("test", np.arange(train, len(raw))),
    ):
        x = tokens[ids, :-1].copy()
        clean_y = np.full_like(x, -100)
        clean_y[:, -1] = tokens[ids, -1]
        y = clean_y.copy()
        y[:, -1] = noisy_tokens[ids, -1]
        column = np.full_like(x, -1)
        column[:, -1] = 0
        if config.dataset == "div97":
            scale = np.asarray(info["codec"]["scale"])[0]
            base = np.full((len(ids), 1), 0.5 / scale)
            step = np.full_like(base, 1 / scale)
        else:
            base = raw[ids, -2:-1].astype(np.float64)
            step = np.ones_like(base)
        splits[name] = Split(
            x,
            y,
            clean_y,
            column,
            ids,
            raw[ids, -1:].astype(np.float64),
            base,
            step,
            past=raw[ids, :-1],
        )
    # Fix the evaluation vocabulary from clean training, independent of epsilon.
    # Seed 23's exact orbit split has four windows containing +/-96 differences
    # that occur only in its test orbits. Preserve that split for re-anchoring;
    # the series/noise adapter reports and removes those unsupported windows.
    supported = set(splits["train"].x.ravel()) | set(
        splits["train"].clean_y[splits["train"].clean_y >= 0]
    )
    test = splits["test"]
    keep = np.isin(test.x, list(supported)).all(axis=1) & np.isin(
        test.clean_y[:, -1], list(supported)
    )
    splits["test"] = _subset(test, keep)
    info["full_split_test_rows"] = len(test.x)
    info["test_rows"] = int(keep.sum())
    info["coverage_excluded_windows"] = {"test": int((~keep).sum())}
    info["coverage_population"] = (
        "fixed from clean training token support at every noise level"
    )
    info.update(
        {
            "corruption": corruption,
            "accuracy_criterion": True,
            "readout": "field" if config.dataset == "fibonacci" else "argmax",
        }
    )
    return _finish(splits, ["x"], info, config, oracle=True)


def generate_ar(length: int, seed: int, period: int = 1) -> np.ndarray:
    """
    Lattice AR: x[t] = floor(0.75*x[t-period]) + U{-1,0,1}.

    The reachable state space from zero is {-4,...,1}. Unlike Gaussian AR through
    a lossy quantizer, this process has an exactly enumerable token Bayes floor.
    A burn-in removes the initial deterministic zero state.
    """
    if length < 1 or period < 1:
        msg = "AR length and period must be positive"
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    burn = 256 * period
    values = np.zeros(length + burn + period, dtype=np.int64)
    innovations = rng.integers(-1, 2, len(values))
    for t in range(period, len(values)):
        values[t] = (3 * values[t - period]) // 4 + innovations[t]
    return values[burn + period :, None].astype(np.float64)


def _raw_series(
    config: DataConfig,
) -> tuple[list[np.ndarray], list[str], dict[str, Any], int]:
    if config.dataset == "weather":
        frame, metadata = _read_weather(
            Path(config.source) if config.source else WEATHER
        )
        return [frame.to_numpy()], frame.columns.tolist(), metadata, 144
    if config.dataset == "etth1":
        path = Path(config.source) if config.source else ETTH1
        frame = pd.read_csv(path).drop(columns="date").astype(np.float64)
        return (
            [frame.to_numpy()],
            frame.columns.tolist(),
            {"source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
            24,
        )
    if config.dataset == "m4_hourly":
        directory = Path(config.source) if config.source else M4
        info = pd.read_csv(directory / "M4-info.csv")
        # These local archives are NPY object arrays despite their .npz suffix.
        archive = np.load(directory / "training.npz", allow_pickle=True)
        ids = info.index[info["SP"] == "Hourly"].to_numpy()
        rng = np.random.default_rng(config.seed ^ 0x44)
        chosen = np.sort(
            rng.choice(ids, min(config.m4_series, len(ids)), replace=False)
        )
        series = [
            np.asarray(archive[index], dtype=np.float64)[:, None] for index in chosen
        ]
        return (
            series,
            ["x"],
            {
                "series_ids": chosen.tolist(),
                "source_sha256": hashlib.sha256(
                    (directory / "training.npz").read_bytes()
                ).hexdigest(),
            },
            24,
        )
    if config.dataset in GAUSSIAN_DATASETS:
        period = config.period if config.dataset == "gaussian_seasonal_ar" else 1
        if config.order != 0 or config.context != period:
            msg = (
                "the Gaussian token oracle requires k=0 and one observation "
                "of the relevant phase (context = lag)"
            )
            raise ValueError(msg)
        end = int(config.series_length * config.train_fraction)
        val_end = int(config.series_length * (config.train_fraction + 1) / 2)
        lengths = (end, val_end - end, config.series_length - val_end)
        # Independent stationary realizations make the heldout law independent
        # of all training observations, including the fitted codec statistics.
        values = np.concatenate(
            [
                gaussian.generate(size, config.seed ^ nonce, period)
                for size, nonce in zip(
                    lengths, (0x47547261696E, 0x4756616C, 0x4754657374), strict=True
                )
            ]
        )
        return (
            [values],
            ["x"],
            {
                "generator": "stationary Gaussian linear AR",
                "coefficient": gaussian.COEFFICIENT,
                "innovation_std": gaussian.INNOVATION_STD,
                "lag": period,
                "independent_split_realizations": True,
                "raw_sha256": array_hash(values),
            },
            period,
        )
    period = config.period if config.dataset == "seasonal_ar" else 1
    if config.order != 0 or config.context < period:
        msg = "the exact lattice-AR floor requires k=0 and context >= AR lag"
        raise ValueError(msg)
    values = generate_ar(config.series_length, config.seed, period)
    return (
        [values],
        ["x"],
        {
            "generator": "floor(3*x[t-lag]/4) + uniform(-1,0,1)",
            "lag": period,
            "raw_sha256": array_hash(values),
        },
        period,
    )


def _windows(values: np.ndarray, starts: np.ndarray, length: int) -> np.ndarray:
    return values[starts[:, None] + np.arange(length)[None, :]]


def _make_series_split(
    raw: np.ndarray,
    starts: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    domain: np.ndarray,
    config: DataConfig,
    ids: np.ndarray,
    period: int,
) -> tuple[Split, int]:
    features = raw.shape[1]
    length = config.order + config.context + 1
    windows = _windows(raw, starts, length)
    normalized = (windows - mean) / std
    # One encode call per column, with each window treated as its own column;
    # this preserves MASH's prefix anchoring without differencing across windows.
    transposed = normalized.transpose(1, 0, 2).reshape(length, -1)
    tokens, scales, clipped = _encode(transposed, config, np.tile(domain, len(starts)))
    encoded = tokens.reshape(config.context + 1, len(starts), features).transpose(
        1, 0, 2
    )
    stream = encoded.reshape(len(starts), -1)
    x = stream[:, :-1].copy()
    y = np.full_like(x, -100)
    y[:, -features:] = stream[:, -features:]
    column = np.full_like(x, -1)
    column[:, -features:] = np.arange(features)
    # Integrate on this window's original grid. Reanchoring MASH at the final
    # raw observation would change its grid and give another point prediction.
    prefix = (
        normalized[:, : config.order]
        .transpose(1, 0, 2)
        .reshape(config.order, len(starts) * features)
    )
    if config.order:
        context_tokens = encoded[:, :-1].transpose(1, 0, 2).reshape(config.context, -1)
        prefix = decode(
            prefix,
            scales,
            context_tokens,
            config.vocab_size,
            config.order,
            tokenizer=config.tokenizer,
        )[-config.order :]
    center = np.full(
        (1, len(starts) * features), config.vocab_size // 2, dtype=np.int64
    )
    base = (
        decode(
            prefix,
            scales,
            center,
            config.vocab_size,
            config.order,
            tokenizer=config.tokenizer,
        )[-1].reshape(len(starts), features)
        * std
        + mean
    )
    shifted = (
        decode(
            prefix,
            scales,
            center + 1,
            config.vocab_size,
            config.order,
            tokenizer=config.tokenizer,
        )[-1].reshape(len(starts), features)
        * std
        + mean
    )
    seasonal = (
        raw[starts + length - 1 - period]
        if np.all(starts + length - 1 >= period)
        else None
    )
    split = Split(
        x,
        y,
        y.copy(),
        column,
        ids,
        windows[:, -1],
        base,
        shifted - base,
        seasonal=seasonal,
    )
    if config.dataset in {"ar", "seasonal_ar"}:
        state = raw[starts + length - 1 - period, 0].astype(np.int64)
        candidates = ((3 * state) // 4)[:, None] + np.arange(-1, 2)[None, :]
        candidate_tokens, _, extra_clipped = _encode(
            (candidates.reshape(-1, 1) - mean) / std, config, domain
        )
        if clipped or extra_clipped:
            msg = "AR training domain does not cover the transition support"
            raise ValueError(msg)
        # Check lossless encoding of ALL reachable states, not only samples.
        states, _, _ = _encode((np.arange(-4, 2)[:, None] - mean) / std, config, domain)
        if len(np.unique(states)) != 6:  # noqa: PLR2004
            msg = "AR state quantization aliases states; exact floor invalid"
            raise ValueError(msg)
        bayes = np.zeros((len(starts), 1, config.vocab_size), dtype=np.float64)
        for row, encoded_candidates in enumerate(candidate_tokens.reshape(-1, 3)):
            np.add.at(bayes[row, 0], encoded_candidates, 1 / 3)
        split.bayes = bayes
    if config.dataset in GAUSSIAN_DATASETS:
        edges = gaussian.bin_edges(
            float(mean[0]), float(std[0]), float(scales[0]), config.vocab_size
        )
        probabilities, _ = gaussian.transition(tuple(edges))
        split.bayes = probabilities[encoded[:, 0, 0], None, :].copy()
    return split, clipped


def _subset(split: Split, keep: np.ndarray) -> Split:
    return Split(
        **{
            key: value[keep] if (value := getattr(split, key)) is not None else None
            for key in split.__dataclass_fields__
        }
    )


def series(config: DataConfig) -> Dataset:
    """Use separate chronological segments, with full raw-window gaps."""
    sequences, columns, provenance, period = _raw_series(config)
    rng = np.random.default_rng(config.seed ^ 0x5E71E5)
    pieces: dict[str, list[Split]] = {
        name: [] for name in ("train", "validation", "test")
    }
    fit_info = []
    boundaries = []
    clipped = dict.fromkeys(pieces, 0)
    length = config.context + config.order + 1
    for series_id, original in enumerate(sequences):
        values = original
        if not np.isfinite(values).all():
            msg = "real series contains nonfinite observations"
            raise ValueError(msg)
        end = int(len(values) * config.train_fraction)
        validation_end = int(len(values) * (config.train_fraction + 1) / 2)
        if end <= length + period or len(values) - validation_end <= 2 * length:
            msg = "series too short for train/validation/test and context gaps"
            raise ValueError(msg)
        fit_values = (
            gaussian.generate(config.calibration_rows, config.seed ^ 0xCA11B, period)
            if config.dataset in GAUSSIAN_DATASETS
            else values[:end]
        )
        mean, std, domain = _fit(fit_values, config)
        if config.shuffled_time:
            values = values.copy()
            # Shuffle independently within segments so time controls retain the
            # same chronological populations and never exchange heldout rows.
            for first, stop in (
                (0, end),
                (end, validation_end),
                (validation_end, len(values)),
            ):
                values[first:stop] = values[first:stop][rng.permutation(stop - first)]
        fit_record = {
            "series_id": series_id,
            "mean": mean.tolist(),
            "std": std.tolist(),
            "domain": domain.tolist(),
            "fit_end": end,
        }
        if config.dataset in GAUSSIAN_DATASETS:
            _, scale, _ = encode(
                np.zeros((1, 1)),
                config.vocab_size,
                domain,
                0,
                tokenizer=config.tokenizer,
            )
            edges = gaussian.bin_edges(
                float(mean[0]), float(std[0]), float(scale[0]), config.vocab_size
            )
            _, diagnostics = gaussian.transition(tuple(edges))
            fit_record.update(
                {
                    "fit_population": (
                        "independent training-only calibration realization"
                    ),
                    "calibration_rows": config.calibration_rows,
                    "calibration_sha256": array_hash(fit_values),
                    "bayes_numerics": diagnostics,
                }
            )
            fit_record.pop("fit_end")
        fit_info.append(fit_record)
        for name, first, stop in (
            ("train", 0, end),
            ("validation", end + length, validation_end),
            ("test", validation_end + length, len(values)),
        ):
            starts = np.arange(max(first, period), stop - length + 1)
            maximum = config.max_train if name == "train" else config.max_eval
            maximum = max(1, maximum // len(sequences)) if maximum else 0
            split_seed = {"train": 0x7AA1, "validation": 0xA11D, "test": 0x7E57}[name]
            starts = _cap(
                starts,
                maximum,
                np.random.default_rng(config.seed ^ split_seed ^ (series_id * 65537)),
            )
            ids = np.column_stack(
                (np.full(len(starts), series_id), starts, starts + length - 1)
            )
            split, count = _make_series_split(
                values, starts, mean, std, domain, config, ids, period
            )
            clipped[name] += count
            pieces[name].append(split)
            boundaries.append(
                {
                    "series": series_id,
                    "split": name,
                    "first_raw_row": int(starts.min()),
                    "last_raw_row": int(starts.max() + length - 1),
                }
            )
    splits = {}
    for name, parts in pieces.items():
        splits[name] = Split(
            **{
                key: np.concatenate([getattr(part, key) for part in parts], axis=0)
                if getattr(parts[0], key) is not None
                else None
                for key in Split.__dataclass_fields__
            }
        )
    # Full contexts are checked column-wise, not only the supervised targets.
    train = splits["train"]
    features = len(columns)
    seen = [
        set(train.x[:, index::features].ravel()) | set(train.y[train.column == index])
        for index in range(features)
    ]
    excluded = {}
    for name in ("validation", "test"):
        split = splits[name]
        keep = np.ones(len(split.x), dtype=bool)
        for index in range(features):
            keep &= np.isin(split.x[:, index::features], list(seen[index])).all(axis=1)
            keep &= np.isin(split.y[:, -features + index], list(seen[index]))
        excluded[name] = int((~keep).sum())
        if not keep.any():
            msg = (
                f"{name} has no windows with training token coverage; "
                "use a coarser vocabulary or more training windows"
            )
            raise ValueError(msg)
        splits[name] = _subset(split, keep)
        if config.dataset in GAUSSIAN_DATASETS:
            # Coverage selection also conditions on the realized target token.
            # Use that conditional law rather than the unrestricted Gaussian
            # law, or expected CE would describe a different test population.
            probabilities = splits[name].bayes
            allowed = np.isin(np.arange(config.vocab_size), list(seen[0]))
            probabilities *= allowed[None, None, :]
            probabilities /= probabilities.sum(axis=-1, keepdims=True)
    provenance.update(
        {
            "split": "chronological train/validation/test with raw-window gaps",
            "gap_rows": length,
            "boundaries": boundaries,
            "codec_fit": "independent training-only calibration"
            if config.dataset in GAUSSIAN_DATASETS
            else "each series training segment only",
            "fit": fit_info,
            "clipped_tokens_before_coverage_filter": clipped,
            "coverage_excluded_windows": excluded,
            "evaluation_population": (
                "frozen windows whose per-column tokens occurred in "
                "optimized training windows"
            ),
            "bayes_conditioned_on_target_coverage": config.dataset in GAUSSIAN_DATASETS,
            "seasonal_period": period,
            "floor_kind": "exact token transition entropy"
            if config.dataset in {"ar", "seasonal_ar"}
            else (
                "known Gaussian token conditional entropy; "
                "convergence-checked quadrature"
            )
            if config.dataset in GAUSSIAN_DATASETS
            else "unknown",
        }
    )
    return _finish(splits, columns, provenance, config)


REGISTRY: dict[str, Callable[[DataConfig], Dataset]] = {
    "weather_formula": weather_formula,
    "div97": anchor,
    "fibonacci": anchor,
    "weather": series,
    "etth1": series,
    "m4_hourly": series,
    "ar": series,
    "seasonal_ar": series,
    "gaussian_ar": series,
    "gaussian_seasonal_ar": series,
}


def dataset(config: DataConfig) -> Dataset:
    """Build and validate a named, deterministic dataset."""
    if config.dataset not in REGISTRY:
        msg = f"unknown dataset {config.dataset!r}; choose from {sorted(REGISTRY)}"
        raise ValueError(msg)
    if (
        config.vocab_size < VOCAB_MULTIPLE
        or config.vocab_size % VOCAB_MULTIPLE
        or config.context < 1
    ):
        msg = "vocabulary must be a positive multiple of 64 and context positive"
        raise ValueError(msg)
    if config.order not in {0, 1} or not 0 < config.train_fraction < 1:
        msg = "series experiments support k=0/1 and 0 < train_fraction < 1"
        raise ValueError(msg)
    if config.noise and config.dataset not in {"div97", "fibonacci"}:
        msg = "answer corruption is defined only for the two anchors"
        raise ValueError(msg)
    if config.shuffled_time and config.dataset in {
        "ar",
        "seasonal_ar",
        *GAUSSIAN_DATASETS,
    }:
        msg = "shuffled time invalidates the known autoregressive Bayes floor"
        raise ValueError(msg)
    if (
        not 0 < config.domain_multiplier < np.inf
        or config.calibration_rows < MIN_CALIBRATION
    ):
        msg = "domain multiplier must be finite/positive and calibration_rows >= 2"
        raise ValueError(msg)
    return REGISTRY[config.dataset](config)
