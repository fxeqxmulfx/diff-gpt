"""Check the rescoring of synthetic_pretrain checkpoints on fresh holdout windows."""

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from benchmarks import synthetic_pretrain, synthetic_rescore
from benchmarks.synthetic_pretrain import forecast_scores, holdout_windows, load_split

if TYPE_CHECKING:
    from pathlib import Path

# The tiny run: a block of 32 tokens, order 1 (one prefix row), horizon 8, so
# its holdout windows have 32 + 1 - 8 = 25 context rows.
BLOCK = 32
HORIZON = 8
SCORE_WINDOWS = 2
NUM_SAMPLES = 4


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Train a run of four updates on a corpus of two sources, on the CPU."""
    root = tmp_path_factory.mktemp("rescore")
    rng = np.random.default_rng(0)
    corpus = root / "corpus"
    for split, n in (("train", 6), ("holdout", 3)):
        (corpus / split).mkdir(parents=True)
        for name in ("sinewave", "ou_process"):
            walks = np.cumsum(rng.normal(size=(n, 80)), axis=1)
            np.save(corpus / split / f"{name}.npy", walks.astype(np.float32))
    (root / "manifest.json").write_text('{"splits": {}}\n')
    output = root / "run"
    synthetic_pretrain.main(
        [
            *("--corpus", str(corpus), "--output", str(output)),
            *("--block-size", str(BLOCK), "--width", "16", "--layers", "1"),
            *("--heads", "2", "--vocab", "64", "--domain-windows", "64"),
            *("--batch-size", "4", "--max-iters", "4", "--eval-interval", "2"),
            *("--eval-iters", "1", "--warmup", "1"),
            *("--score-windows", str(SCORE_WINDOWS), "--horizon", str(HORIZON)),
            *("--num-samples", str(NUM_SAMPLES), "--device", "cpu"),
        ],
    )
    return output


def _rescore(run: Path, output: Path, *extra: str) -> dict:
    synthetic_rescore.main(
        [
            *("--run", str(run), "--output", str(output)),
            *("--num-samples", str(NUM_SAMPLES), "--device", "cpu"),
            *extra,
        ],
    )
    return json.loads(output.read_text())


def test_the_run_s_own_windows_and_seed_give_its_last_scores(
    tiny_run: Path,
    tmp_path: Path,
) -> None:
    """
    Rescoring last.bin as the run scored it reproduces the run's last scores.

    The run's windows are --score-windows per source drawn with --score-seed
    0, at the run's context and horizon, and its samples are seeded with
    --score-seed too: the same draw gives the same scores of every source.
    """
    record = _rescore(
        tiny_run,
        tmp_path / "own.json",
        *("--per-source", str(SCORE_WINDOWS), "--window-seed", "0"),
        *("--sample-seed", "0", "--score-samples", "2"),
    )
    lines = (tiny_run / "scores.jsonl").read_text().splitlines()
    last = json.loads(lines[-1])
    assert (record["context"], record["horizon"]) == (BLOCK + 1 - HORIZON, HORIZON)
    assert record["windows"] == 2 * SCORE_WINDOWS
    assert record["scores"]["mase"] == last["mase"]
    assert record["scores"]["crps"] == last["crps"]
    assert record["scores"]["sources"] == last["sources"]
    fewer = record["fewer_samples"]["2"]
    assert (fewer["samples"], fewer["groups"]) == (2, 2)
    assert record["temperature"] == 1.0
    assert record["tokens_per_second"] > 0


def test_another_context_and_horizon_score_their_own_windows(
    tiny_run: Path,
    tmp_path: Path,
) -> None:
    """
    A shorter context and a longer horizon draw those windows of the holdout.

    10 context rows and 20 points take 10 - 1 + 20 = 29 of the 32 tokens. The
    naive scores of every source are those of the windows holdout_windows
    draws with the same seed, so the record scored them.
    """
    record = _rescore(
        tiny_run,
        tmp_path / "long.json",
        *("--context", "10", "--horizon", "20", "--per-source", "3"),
        *("--window-seed", "5"),
    )
    assert (record["context"], record["horizon"], record["windows"]) == (10, 20, 6)
    config = json.loads((tiny_run / "config.json").read_text())
    holdout = load_split(tiny_run.parent / "corpus", "holdout")
    windows = holdout_windows(holdout, 3, 10, 20, seed=5)
    naive = np.repeat(windows.contexts[:, -1:, None], 20, axis=2)  # (N, 1, H)
    expected = forecast_scores(naive, windows, list(holdout))
    for name, scores in expected["sources"].items():
        assert record["scores"]["sources"][name]["naive_mase"] == pytest.approx(
            scores["naive_mase"],
        )
        assert record["scores"]["sources"][name]["naive_crps"] == pytest.approx(
            scores["naive_crps"],
        )
    assert np.isfinite([record["scores"]["mase"], record["scores"]["crps"]]).all()
    assert record["git_head"]
    assert config["args"]["block_size"] == BLOCK


@pytest.mark.parametrize(
    "extra",
    [
        ["--context", "20", "--horizon", "20"],  # 20 - 1 + 20 = 39 > 32 tokens
        ["--context", "1"],  # no row past the prefix
        ["--horizon", "0"],
        ["--score-samples", str(NUM_SAMPLES + 1)],
    ],
)
def test_draws_that_do_not_fit_are_refused(
    tiny_run: Path,
    tmp_path: Path,
    extra: list[str],
) -> None:
    """A context and horizon past the block, or too many samples, write nothing."""
    output = tmp_path / "refused.json"
    with pytest.raises(SystemExit):
        _rescore(tiny_run, output, *extra)
    assert not output.exists()


def test_an_existing_output_is_kept(tiny_run: Path, tmp_path: Path) -> None:
    """--output must not exist: an earlier record is never overwritten."""
    output = tmp_path / "kept.json"
    output.write_text("{}\n")
    with pytest.raises(SystemExit):
        _rescore(tiny_run, output)
    assert output.read_text() == "{}\n"


def test_other_encoder_settings_or_corpus_are_refused(
    tiny_run: Path,
    tmp_path: Path,
) -> None:
    """
    Weights trained with other encoder settings, or a changed corpus, are refused.

    A copy of the run whose config.json names another domain no longer
    matches the checkpoint's encoder settings; a copy of the corpus with
    another manifest is not the corpus the run trained on.
    """
    config = json.loads((tiny_run / "config.json").read_text())
    other = tmp_path / "other"
    other.mkdir()
    (other / "last.bin").write_bytes((tiny_run / "last.bin").read_bytes())
    changed = {
        **config,
        "domain_of_definition": [2 * d for d in config["domain_of_definition"]],
    }
    (other / "config.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="other encoder settings"):
        _rescore(other, tmp_path / "domain.json")
    corpus = tmp_path / "copy" / "corpus"
    (corpus / "holdout").mkdir(parents=True)
    for path in (tiny_run.parent / "corpus" / "holdout").iterdir():
        (corpus / "holdout" / path.name).write_bytes(path.read_bytes())
    (corpus.parent / "manifest.json").write_text('{"splits": {"other": 1}}\n')
    moved = tmp_path / "moved"
    moved.mkdir()
    (moved / "last.bin").write_bytes((tiny_run / "last.bin").read_bytes())
    elsewhere = {**config, "args": {**config["args"], "corpus": str(corpus)}}
    (moved / "config.json").write_text(json.dumps(elsewhere))
    with pytest.raises(SystemExit):
        _rescore(moved, tmp_path / "manifest.json")
    assert not (tmp_path / "manifest.json").exists()
