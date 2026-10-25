"""Guard scientific splits, quantization ceilings, noise and exact resume."""

import csv
import json
import warnings
from dataclasses import asdict, replace
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
import torch

from diff_gpt.application.grokking.div97 import Criterion, division_rows
from diff_gpt.application.grokking.series import Config, run, skill, verify
from diff_gpt.application.grokking.series_data import (
    DataConfig,
    array_hash,
    corrupt_answers,
    dataset,
    generate_ar,
    tabulation,
)
from diff_gpt.application.grokking.series_numeric_codec import (
    binary_prompts,
    bits_dataset,
)
from diff_gpt.application.grokking.series_oracle import conditional_reference
from diff_gpt.application.grokking.series_weather_audit import formula_comparators
from diff_gpt.domain.encoder_decoder import TokenClippingWarning, encode

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def weather_path(tmp_path: Path) -> Path:
    """Create a periodic table with an exact same-row vapor-pressure formula."""
    rng = np.random.default_rng(7)
    row = np.arange(1024)
    rh = 40 + 5 * (row % 9)
    vapor = 5 + 0.5 * ((row // 9) % 11)
    frame = pd.DataFrame(
        {
            "date": row,
            "rh (%)": rh,
            "VPmax (mbar)": vapor,
            "VPact (mbar)": rh * vapor / 100,
            "wv (m/s)": rng.integers(0, 3, len(row)).astype(np.float64),
        }
    )
    frame.loc[20, "wv (m/s)"] = -9999
    path = tmp_path / "weather.csv"
    frame.to_csv(path, index=False)
    return path


def test_tabulation_counts_ambiguity_and_entropy() -> None:
    """Conflicting targets contribute majority accuracy and conditional entropy."""
    contexts = np.array([[1, 2], [1, 2], [1, 2], [2, 1]])
    result = tabulation(contexts, np.array([0, 0, 1, 3]))
    assert result["accuracy"] == pytest.approx(3 / 4)
    assert result["contexts"] == len(np.unique(contexts, axis=0))
    assert result["loss"] == pytest.approx(-(2 * np.log(2 / 3) + np.log(1 / 3)) / 4)


def test_full_table_oracle_keeps_ambiguity_in_a_one_row_test() -> None:
    """Capping the test does not turn an ambiguous input into a deterministic one."""
    contexts = np.array([[1, 2]] * 4)
    targets = np.array([0, 0, 1, 1])
    reference = conditional_reference(contexts, targets, contexts[:1], targets[:1], 64)
    assert reference["floor_loss"] == pytest.approx(np.log(2))
    assert reference["ceiling_accuracy"] == pytest.approx(1 / 2)
    assert tabulation(contexts[:1], targets[:1])["loss"] == 0
    with pytest.raises(ValueError, match="context"):
        conditional_reference(contexts, targets, np.array([[2, 1]]), targets[:1], 64)


def test_formula_ceiling_uses_the_full_population(weather_path: Path) -> None:
    """Full-table ambiguity survives sparse evaluation of heldout token pairs."""
    rng = np.random.default_rng(9)
    rh, vapor = rng.uniform(40, 95, 4096), rng.uniform(5, 40, 4096)
    pd.DataFrame(
        {
            "date": np.arange(len(rh)),
            "rh (%)": rh,
            "VPmax (mbar)": vapor,
            "VPact (mbar)": rh * vapor / 100,
        }
    ).to_csv(weather_path, index=False)
    data = dataset(DataConfig(source=str(weather_path), max_train=128, max_eval=64))
    reference = data.references["test"]["VPact"]
    assert reference["floor_loss"] > reference["empirical_split_floor_loss"]
    assert reference["ceiling_accuracy"] < 1


def test_random_formula_control_has_no_input_rule_skill(
    weather_path: Path, tmp_path: Path
) -> None:
    """Marginal calibration of independent labels cannot count as rule learning."""
    config = Config(
        data=DataConfig(
            source=str(weather_path), random_labels=True, max_train=0, max_eval=64
        ),
        width=8,
        heads=1,
        epochs=1,
    )
    data = dataset(config.data)
    assert data.references["test"]["VPact"]["input_independent_control"]
    assert data.references["test"]["VPact"]["floor_loss"] > 0
    result = run(config, tmp_path / "control", "cpu")
    assert result["metrics"]["test"]["skill"] is None
    assert result["query_metrics"]["test"]["skill"] is None
    assert result["criteria"]["averaged"]["VPact"]["first99"] is None


def test_formula_pair_split_calibration_and_coverage(weather_path: Path) -> None:
    """No pair or calibration row leaks, including capped training samples."""
    config = DataConfig(source=str(weather_path), max_train=128, max_eval=128)
    data = dataset(config)
    train, test = data.splits["train"], data.splits["test"]
    assert not set(map(tuple, train.x)) & set(map(tuple, test.x))
    assert not set(train.ids) & set(test.ids)
    assert set(data.metadata["calibration_ids"]) <= set(train.ids)
    assert data.metadata["fit_population_is_training_only"]
    assert all(data.metadata["input_token_coverage"].values())
    assert data.metadata["missing_by_column"] == {"wv (m/s)": 1}
    assert not np.isin(-9999, train.truth)
    for position in range(2):
        assert set(test.x[:, position]) <= set(train.x[:, position])
    assert data.fingerprint() == dataset(config).fingerprint()


def test_formula_fit_excludes_all_heldout_values(weather_path: Path) -> None:
    """Calibration statistics can be independently recomputed from train rows."""
    config = DataConfig(source=str(weather_path), max_train=128, max_eval=128)
    data = dataset(config)
    values = pd.read_csv(weather_path)[
        ["rh (%)", "VPmax (mbar)", "VPact (mbar)"]
    ].to_numpy()
    calibration = values[data.metadata["calibration_ids"]]
    np.testing.assert_allclose(data.metadata["mean"], calibration.mean(0))
    np.testing.assert_allclose(data.metadata["std"], calibration.std(0))


@pytest.mark.parametrize("vocab", [64, 128, 256])
def test_binary_prompt_round_trip_preserves_quantized_values(vocab: int) -> None:
    """Changing the input alphabet loses neither quantization levels nor order."""
    values = np.arange(vocab).reshape(-1, 2)
    encoded = binary_prompts(values, vocab)
    bits = (vocab - 1).bit_length()
    recovered = encoded.reshape(len(values), 2, bits) @ (
        2 ** np.arange(bits - 1, -1, -1)
    )
    np.testing.assert_array_equal(values, recovered)


def test_binary_codec_keeps_splits_labels_resolution_and_ceiling(
    weather_path: Path,
) -> None:
    """Change the input representation alone in the codec comparison."""
    config = DataConfig(source=str(weather_path), max_train=128, max_eval=64)
    plain = dataset(config)
    binary = bits_dataset(replace(config, dataset="weather_formula_bits"))
    assert plain.references == binary.references
    for key in ("mean", "std", "domain", "scale", "full_table_oracle"):
        assert plain.metadata[key] == binary.metadata[key]
    for name, split in plain.splits.items():
        encoded = binary.splits[name]
        np.testing.assert_array_equal(encoded.ids, split.ids)
        np.testing.assert_array_equal(encoded.y[:, -1], split.y[:, -1])
        np.testing.assert_array_equal(encoded.truth, split.truth)
        np.testing.assert_array_equal(
            encoded.x, binary_prompts(split.x, config.vocab_size)
        )


@pytest.mark.parametrize("fraction", [0.0, 0.01, 0.05, 0.2])
def test_noise_changes_only_fixed_training_answers(fraction: float) -> None:
    """Test windows and every training prompt remain byte-identical."""
    raw = division_rows(11)
    corrupted, info = corrupt_answers(raw, len(raw) // 2, fraction, 11)
    np.testing.assert_array_equal(corrupted[:, :-1], raw[:, :-1])
    np.testing.assert_array_equal(corrupted[len(raw) // 2 :], raw[len(raw) // 2 :])
    assert info["selected_training_answers"] == round(len(raw) / 2 * fraction)
    assert not info["heldout_answers_changed"]
    np.testing.assert_array_equal(
        corrupted, corrupt_answers(raw, len(raw) // 2, fraction, 11)[0]
    )


@pytest.mark.parametrize(("name", "order"), [("div97", 0), ("fibonacci", 1)])
def test_anchor_noise_does_not_enter_model_inputs(name: str, order: int) -> None:
    """Replacing the final raw value changes its supervision alone."""
    clean = dataset(DataConfig(dataset=name, order=order, vocab_size=256))
    noisy = dataset(DataConfig(dataset=name, order=order, vocab_size=256, noise=0.2))
    for split in clean.splits:
        np.testing.assert_array_equal(clean.splits[split].x, noisy.splits[split].x)
        np.testing.assert_array_equal(
            clean.splits[split].clean_y, noisy.splits[split].clean_y
        )
    np.testing.assert_array_equal(clean.splits["test"].y, noisy.splits["test"].y)
    assert np.any(clean.splits["train"].y != noisy.splits["train"].y)


def test_fibonacci_coverage_population_stays_fixed_across_noise() -> None:
    """Remove the same four unsupported windows at every corruption level."""
    config = DataConfig(dataset="fibonacci", order=1, vocab_size=256, seed=23)
    clean = dataset(config)
    noisy = dataset(replace(config, noise=0.2))
    assert clean.metadata["coverage_excluded_windows"] == {"test": 4}
    assert all(value["covered"] for value in clean.metadata["token_coverage"].values())
    assert all(value["covered"] for value in noisy.metadata["token_coverage"].values())
    np.testing.assert_array_equal(clean.splits["test"].ids, noisy.splits["test"].ids)


@pytest.mark.parametrize("order", [0, 1])
@pytest.mark.parametrize("tokenizer", ["lgt", "mash"])
def test_temporal_gaps_train_only_fit_and_vocabulary(
    weather_path: Path, order: int, tokenizer: str
) -> None:
    """Raw windows are disjoint across populations and codec inversion agrees."""
    config = DataConfig(
        dataset="weather",
        source=str(weather_path),
        order=order,
        tokenizer=tokenizer,
        context=3,
        max_train=0,
        max_eval=64,
    )
    data = dataset(config)
    train = data.splits["train"]
    features = len(data.columns)
    bounds = {value["split"]: value for value in data.metadata["boundaries"]}
    for first, second in (("train", "validation"), ("validation", "test")):
        assert (
            bounds[second]["first_raw_row"] - bounds[first]["last_raw_row"]
            >= config.context + order
        )
    for name, split in data.splits.items():
        predicted = split.base + split.step * (
            split.y[:, -features:] - config.vocab_size // 2
        )
        np.testing.assert_array_less(
            np.abs(predicted - split.truth), split.step / 2 + 1e-10
        )
        if name == "train":
            continue
        for index in range(features):
            seen = set(train.x[:, index::features].ravel()) | set(
                train.y[train.column == index]
            )
            assert set(split.x[:, index::features].ravel()) <= seen
            assert set(split.y[split.column == index]) <= seen
    original = pd.read_csv(weather_path)
    original.loc[900:, "rh (%)"] += 10000
    original.to_csv(weather_path, index=False)
    altered = dataset(config)
    assert data.metadata["fit"] == altered.metadata["fit"]
    np.testing.assert_array_equal(train.x, altered.splits["train"].x)


@pytest.mark.parametrize("shuffled", [False, True])
def test_multivariate_formula_audit_uses_the_actual_future_row(
    weather_path: Path, tmp_path: Path, *, shuffled: bool
) -> None:
    """Formula references stay aligned with teacher-forced operands after shuffling."""
    frame = pd.read_csv(weather_path)
    rng = np.random.default_rng(15)
    frame["rh (%)"] = rng.uniform(30, 95, len(frame))
    frame["VPmax (mbar)"] = rng.uniform(5, 40, len(frame))
    frame["VPact (mbar)"] = frame["rh (%)"] * frame["VPmax (mbar)"] / 100
    frame.to_csv(weather_path, index=False)
    config = DataConfig(
        dataset="weather",
        source=str(weather_path),
        context=2,
        max_train=256,
        max_eval=64,
        shuffled_time=shuffled,
    )
    data = dataset(config)
    np.savez(
        tmp_path / "frozen_splits.npz",
        **{f"{name}__ids": split.ids for name, split in data.splits.items()},
    )
    audit = formula_comparators(
        tmp_path, {"config": {"data": asdict(config)}, "data": data.metadata}
    )
    assert audit["native_full_prompt_bayes_floor"] is None
    ceiling = audit["columns"]["VPact (mbar)"]["splits"]["test"]
    assert ceiling["oracle_realized_loss"] >= 0
    assert 0 < ceiling["ceiling_accuracy"] <= 1
    # Independently recover the same-row operands and target from the frozen
    # prompt. Their alignment must agree with raw row IDs used by the audit.
    features = len(data.columns)
    split = data.splits["test"]
    operands = split.x[:, config.context * features : config.context * features + 2]
    targets = split.y[:, -features + 2]
    fit = data.metadata["fit"][0]
    values = frame[data.columns[:3]].to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", TokenClippingWarning)
        _, _, population = encode(
            (values - np.asarray(fit["mean"])[:3]) / np.asarray(fit["std"])[:3],
            config.vocab_size,
            np.asarray(fit["domain"])[:3],
            0,
            tokenizer="lgt",
        )
    expected = conditional_reference(
        population[:, :2], population[:, 2], operands, targets, config.vocab_size
    )
    assert ceiling == expected


def test_generator_hash_fixture_and_exact_bayes_floor() -> None:
    """The lattice generator and quantized transition entropy are fixed."""
    assert (
        array_hash(generate_ar(32, 11, 4))
        == "84e83df89368a1babf460bcab793fb8043cebfcbad0d549549edebd04ba6f6ef"
    )
    config = DataConfig(
        dataset="ar", context=4, series_length=2048, max_train=128, max_eval=64
    )
    data = dataset(config)
    for name, split in data.splits.items():
        assert split.bayes is not None
        np.testing.assert_allclose(split.bayes.sum(-1), 1.0)
        assert data.references[name]["x"]["floor_loss"] == pytest.approx(np.log(3))
        assert data.references[name]["x"]["ceiling_accuracy"] == pytest.approx(1 / 3)
        labels = split.y[:, -1]
        assert np.all(split.bayes[np.arange(len(labels)), 0, labels] > 0)


def test_skill_criterion_requires_a_plateau_and_records_overshoot() -> None:
    """Normalized loss satisfies the anchor rule without hiding below-floor CE."""
    criterion = Criterion()
    for epoch in range(30):
        criterion.observe(epoch, skill(0.01, 2, 0), skill(1.9, 2, 0))
    criterion.observe(30, 1.0, 0.91)
    criterion.observe(31, 1.0, 0.995)
    assert criterion.delayed
    assert criterion.plateau_epochs == epoch
    assert skill(0.5, 2, 1) == pytest.approx(3 / 2)
    assert skill(2.1, 2, 1) < 0
    assert skill(1, 2, None) is None
    assert skill(1, 1, 1) is None
    immediate = Criterion()
    immediate.observe(1, 1.0, 1.0)
    assert not immediate.delayed


def _small_config() -> Config:
    return Config(
        data=DataConfig(
            dataset="ar", context=4, series_length=2048, max_train=64, max_eval=32
        ),
        width=8,
        heads=1,
        epochs=1,
        checkpoint_every=1,
        batch_size=32,
    )


def test_resume_is_identical_and_replays_both_weight_states(tmp_path: Path) -> None:
    """Durable checkpoint resume reproduces every parameter and optimizer tensor."""
    torch.set_num_threads(2)
    config = _small_config()
    interrupted, whole = tmp_path / "interrupted", tmp_path / "whole"
    run(config, interrupted, "cpu")
    extended = replace(config, epochs=3)
    resumed = run(extended, interrupted, "cpu", resume=True)
    uninterrupted = run(extended, whole, "cpu")
    assert resumed["metrics"] == uninterrupted["metrics"]
    assert resumed["query_metrics"] == uninterrupted["query_metrics"]
    assert resumed["criteria"] == uninterrupted["criteria"]
    assert resumed["training_flops"] == uninterrupted["training_flops"]
    first = torch.load(interrupted / "checkpoint.pt", weights_only=True)
    second = torch.load(whole / "checkpoint.pt", weights_only=True)
    for name, value in first["model"].items():
        torch.testing.assert_close(value, second["model"][name], atol=0, rtol=0)
    for name, value in first["query_model"].items():
        torch.testing.assert_close(value, second["query_model"][name], atol=0, rtol=0)
    for index, state in first["optimizer"]["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(
                value, second["optimizer"]["state"][index][name], atol=0, rtol=0
            )
    assert first["sampler_state"] == second["sampler_state"]
    assert first["sampler_order"] == second["sampler_order"]
    verification = verify(interrupted, "cpu")
    assert verification["both_weight_states"]
    assert verification["explicit_scored_query_weights"]


def test_resume_rejects_changed_experiment_before_mutating_run(tmp_path: Path) -> None:
    """Different splits, codec or source cannot silently extend an old curve."""
    config = _small_config()
    directory = tmp_path / "run"
    run(config, directory, "cpu")
    before = {
        path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()
    }
    changed = replace(config, epochs=2, data=replace(config.data, seed=17))
    with pytest.raises(ValueError, match="identical"):
        run(changed, directory, "cpu", resume=True)
    after = {
        path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()
    }
    assert before == after


def test_resume_discards_history_after_durable_checkpoint(tmp_path: Path) -> None:
    """An interruption after CSV flush but before checkpoint leaves no duplicates."""
    config = _small_config()
    directory = tmp_path / "run"
    run(config, directory, "cpu")
    path = directory / "history.csv"
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows, fields = list(reader), reader.fieldnames
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writerow({**rows[-1], "epoch": 2})
    run(replace(config, epochs=2), directory, "cpu", resume=True)
    with path.open(newline="") as handle:
        assert [int(row["epoch"]) for row in csv.DictReader(handle)] == [0, 1, 2]
    assert (
        json.loads((directory / "result.json").read_text())["epoch"]
        == config.epochs + 1
    )
