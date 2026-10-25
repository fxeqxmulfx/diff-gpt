"""Check the synthetic pretraining script: its holdout scores and a tiny run."""

import csv
import hashlib
import io
import json
import random
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from diff_gpt.application.diff_gpt import DiffGPT
from diff_gpt.application.synthetic_pretrain import (
    RESUME_STATE,
    TEMPOPFN_WEIGHTS,
    _rng_states,
    _set_rng_states,
    forecast_scores,
    gradient_checkpointing,
    holdout_windows,
)
from diff_gpt.domain.model.gpt import GPT
from diff_gpt.interfaces.synthetic_pretrain import main

if TYPE_CHECKING:
    from pathlib import Path

    from diff_gpt.application.synthetic_pretrain import Windows

CONTEXT = 24
HORIZON = 8
PER_SOURCE = 4
# A forecast halfway from the naive one to the truth halves both errors.
HALFWAY = 0.5
# The tiny runs of the resume tests, but for --warmup and --max-iters.
TINY = (
    *("--block-size", "32", "--width", "16", "--layers", "1"),
    *("--heads", "2", "--vocab", "64", "--domain-windows", "64"),
    *("--batch-size", "4", "--eval-interval", "2", "--eval-iters", "1"),
    *("--score-windows", "2", "--horizon", "8", "--num-samples", "4"),
    *("--score-samples", "2", "--device", "cpu"),
)
# 6 warmup steps span a resume after 4 updates. An update is 3.9e6 FLOPs,
# so a copy every 1e7 FLOPs keeps the weights after 4, 6 and 8 updates.
RESUMED = (*TINY, "--warmup", "6", "--keep-every", "1e7")
TIMINGS = frozenset({"seconds", "score_seconds"})


def _series() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        "a": np.cumsum(rng.normal(size=(3, 40)), axis=1),
        "b": 5.0 + np.cumsum(rng.normal(size=(2, 40)), axis=1),
    }


def _windows() -> Windows:
    return holdout_windows(_series(), PER_SOURCE, CONTEXT, HORIZON, seed=0)


def _samples(point: np.ndarray) -> np.ndarray:
    """Return five samples of the (N, H) `point`: each quantile is the point."""
    return np.repeat(point[:, None, :], 5, axis=1)


def test_holdout_windows_are_slices_of_the_series() -> None:
    """Check that each window is a slice of a series of its source, in order."""
    series = _series()
    windows = _windows()
    assert windows.sources.tolist() == [0] * PER_SOURCE + [1] * PER_SOURCE
    assert windows.contexts.shape == (2 * PER_SOURCE, CONTEXT)
    assert windows.futures.shape == (2 * PER_SOURCE, HORIZON)
    names = list(series)
    for k, context, future in zip(*windows, strict=True):
        window = np.concatenate([context, future])
        data = series[names[k]]
        assert any(
            np.array_equal(row[o : o + len(window)], window)
            for row in data
            for o in range(data.shape[1] - len(window) + 1)
        )


def test_scores_are_ratios_to_the_naive_forecast() -> None:
    """
    Check the scores of the naive forecast and of one halfway to the truth.

    Samples that all equal a point forecast make every quantile that point,
    so the naive forecast scores 1 and the halfway one 0.5, on both scores.
    """
    windows = _windows()
    naive = np.repeat(windows.contexts[:, -1:], HORIZON, axis=1)
    halfway = naive + HALFWAY * (windows.futures - naive)
    for point, expected in ((naive, 1.0), (halfway, HALFWAY)):
        scores = forecast_scores(_samples(point), windows, ["a", "b"])
        assert scores["mase"] == pytest.approx(expected)
        assert scores["crps"] == pytest.approx(expected)
        assert set(scores["sources"]) == {"a", "b"}


def test_fewer_samples_average_the_scores_of_their_groups() -> None:
    """
    Check scores of groups of samples: their mean, of the groups that fit.

    Of six samples, four are the naive forecast (scores 1) and the last two
    are halfway to the truth (scores 0.5). Groups of two are two naive ones
    and a halfway one, which score (1 + 1 + 0.5) / 3; groups of four leave
    the last two samples out, so their one group is naive. Groups of all
    six score as the samples do without groups.
    """
    windows = _windows()
    naive = np.repeat(windows.contexts[:, -1:], HORIZON, axis=1)
    halfway = naive + HALFWAY * (windows.futures - naive)
    samples = np.concatenate([_samples(naive)[:, :4], _samples(halfway)[:, :2]], axis=1)
    names = ["a", "b"]
    mean = (2.0 + HALFWAY) / 3
    for group, expected, groups in ((2, mean, 3), (4, 1.0, 1)):
        scores = forecast_scores(samples, windows, names, group=group)
        assert scores["mase"] == pytest.approx(expected)
        assert scores["crps"] == pytest.approx(expected)
        assert (scores["samples"], scores["groups"]) == (group, groups)
    assert forecast_scores(samples, windows, names, group=6) == forecast_scores(
        samples,
        windows,
        names,
    )
    for group in (0, 7):
        with pytest.raises(ValueError, match="group must lie in"):
            forecast_scores(samples, windows, names, group=group)


def test_scores_skip_missing_values() -> None:
    """A missing future value counts for nothing; naive is the last value seen."""
    windows = _windows()
    windows.futures[0, 3] = np.nan
    windows.contexts[1, -1] = np.nan
    last = windows.contexts[:, -1].copy()
    last[1] = windows.contexts[1, -2]
    naive = np.repeat(last[:, None], HORIZON, axis=1)
    scores = forecast_scores(_samples(naive), windows, ["a", "b"])
    assert scores["mase"] == pytest.approx(1.0)
    assert scores["crps"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "precision",
    [[], ["--train-dtype", "bfloat16", "--score-dtype", "float16"]],
)
def test_a_tiny_run_writes_its_curve_and_checkpoints(
    tmp_path: Path,
    precision: list[str],
) -> None:
    """
    Check a CPU run of four updates on a corpus of two sources.

    The history holds the evaluations after 0, 2 and 4 updates with their
    FLOPs, each one scored, with all four samples, with two and with three,
    and best.bin loads into the model config.json describes, with the
    digest of the manifest one level above the corpus, where synthetic/'s
    export writes it. Trained in mixed precision and scored in float16 too,
    the checkpoints hold the float32 master weights. Scores of more samples
    than are drawn are refused.
    """
    rng = np.random.default_rng(0)
    corpus = tmp_path / "corpus"
    for split, n in (("train", 6), ("holdout", 3)):
        (corpus / split).mkdir(parents=True)
        for name in ("sinewave", "ou_process"):
            walks = np.cumsum(rng.normal(size=(n, 80)), axis=1)
            np.save(corpus / split / f"{name}.npy", walks.astype(np.float32))
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"splits": {}}\n')
    output = tmp_path / "run"
    main(
        [
            *("--corpus", str(corpus), "--output", str(output)),
            *("--block-size", "32", "--width", "16", "--layers", "1"),
            *("--heads", "2", "--vocab", "64", "--domain-windows", "64"),
            *("--batch-size", "4", "--max-iters", "4", "--eval-interval", "2"),
            *("--eval-iters", "1", "--warmup", "1", "--score-windows", "2"),
            *("--horizon", "8", "--num-samples", "4", "--device", "cpu"),
            *("--score-samples", "2", "3"),
            *precision,
        ],
    )
    config = json.loads((output / "config.json").read_text())
    with (output / "history.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    steps = [0, 2, 4]
    assert [int(row["step"]) for row in rows] == steps
    flops = config["flops_per_token"] * config["tokens_per_update"]
    assert [int(row["train_flops"]) for row in rows] == [flops * s for s in steps]
    scored = ("crps", "mase_s2", "crps_s2", "mase_s3", "crps_s3", "score_seconds")
    assert all(row[field] for row in rows for field in scored)
    records = [
        json.loads(line) for line in (output / "scores.jsonl").read_text().splitlines()
    ]
    assert [record["step"] for record in records] == steps
    for record in records:
        assert (record["samples"], record["groups"]) == (4, 1)
        fewer = [record["fewer_samples"][size] for size in ("2", "3")]
        assert [(f["samples"], f["groups"]) for f in fewer] == [(2, 2), (3, 1)]
    with pytest.raises(SystemExit):
        main(
            [
                *("--corpus", str(corpus), "--output", str(tmp_path / "more")),
                *("--num-samples", "4", "--score-samples", "5"),
            ],
        )
    assert not (tmp_path / "more").exists()
    assert config["sources"]["ou_process"]["weight"] == TEMPOPFN_WEIGHTS["ou_process"]
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert config["manifest_sha256"] == digest
    model = DiffGPT(
        model=GPT(vocab_size=64, n_embd=16, block_size=32, n_head=2, n_layer=1),
        order_of_derivative=1,
        domain_of_definition=np.array(config["domain_of_definition"]),
        tokenizer=config["tokenizer"],
    )
    model.load(str(output / "best.bin"))
    checkpoint = torch.load(output / "best.bin", weights_only=True)
    assert {v.dtype for v in checkpoint["model"].values()} == {torch.float32}
    with pytest.raises(FileExistsError):
        main(["--corpus", str(corpus), "--output", str(output)])


def _corpus(root: Path) -> Path:
    """Write a corpus of two sources under `root`, with its manifest beside it."""
    rng = np.random.default_rng(0)
    corpus = root / "corpus"
    for split, n in (("train", 6), ("holdout", 3)):
        (corpus / split).mkdir(parents=True)
        for name in ("sinewave", "ou_process"):
            walks = np.cumsum(rng.normal(size=(n, 80)), axis=1)
            np.save(corpus / split / f"{name}.npy", walks.astype(np.float32))
    (root / "manifest.json").write_text('{"splits": {}}\n')
    return corpus


def _history(run: Path) -> list[dict[str, str]]:
    """Return the rows of the run's history.csv without the timings."""
    with (run / "history.csv").open(newline="") as handle:
        return [
            {key: value for key, value in row.items() if key not in TIMINGS}
            for row in csv.DictReader(handle)
        ]


def _same(a: object, b: object) -> bool:
    """Return whether `a` and `b` hold the same values and tensors, bit for bit."""
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.equal(b)
    if isinstance(a, dict):
        return (
            isinstance(b, dict)
            and a.keys() == b.keys()
            and all(_same(a[key], b[key]) for key in a)
        )
    if isinstance(a, list | tuple):
        return (
            isinstance(b, list | tuple)
            and type(a) is type(b)
            and len(a) == len(b)
            and all(_same(x, y) for x, y in zip(a, b, strict=True))
        )
    return a == b


def test_a_resumed_run_goes_on_as_an_unbroken_one(tmp_path: Path) -> None:
    """
    Check that 4 updates, a crash and a resume to 8 give the run of 8 updates.

    The crash comes after the evaluation of 6 updates wrote its rows, its
    scores and its checkpoints, but before its state.pt: the files of a run
    of 6 updates over the state.pt of a run of 4, and a history row cut
    short. The resume drops what came after 4 updates and goes on through
    the rest of the warmup: the same history but for the timings, the same
    scores, the same weights in every checkpoint and kept copy, and the
    same state to go on from, bit for bit.
    """
    data = ("--corpus", str(_corpus(tmp_path)))
    straight, crashed, later = (tmp_path / name for name in ("8", "4", "6"))
    main([*data, *RESUMED, "--output", str(straight), "--max-iters", "8"])
    main([*data, *RESUMED, "--output", str(crashed), "--max-iters", "4"])
    main([*data, *RESUMED, "--output", str(later), "--max-iters", "6"])
    for path in later.iterdir():
        if path.name not in {"config.json", RESUME_STATE} and path.is_file():
            (crashed / path.name).write_bytes(path.read_bytes())
    with (crashed / "history.csv").open("ab") as handle:
        handle.write(b"8,315")
    main(["--resume", "--output", str(crashed), "--max-iters", "8"])
    assert _history(crashed) == _history(straight)
    assert [int(row["step"]) for row in _history(straight)] == [0, 2, 4, 6, 8]
    scores = (straight / "scores.jsonl").read_text()
    assert (crashed / "scores.jsonl").read_text() == scores
    config = json.loads((straight / "config.json").read_text())
    flops = config["flops_per_token"] * config["tokens_per_update"]
    kept = [f"step_{step}_flops_{flops * step:.4e}.bin" for step in (4, 6, 8)]
    names = sorted(path.name for path in straight.glob("*.bin"))
    assert names == sorted(["best.bin", "last.bin", *kept])
    assert sorted(path.name for path in crashed.glob("*.bin")) == names
    for name in names:
        assert _same(
            torch.load(straight / name, weights_only=True),
            torch.load(crashed / name, weights_only=True),
        ), name
    ends = [
        torch.load(run / RESUME_STATE, weights_only=True) for run in (straight, crashed)
    ]
    for end in ends:
        assert end.pop("seconds") > 0
        assert (end["step"], end["evaluations"], end["kept"]) == (8, 5, 3)
        assert end["optimizer"]["param_groups"][0]["step"] == 8  # noqa: PLR2004
    assert _same(*ends)
    resumed = json.loads((crashed / "config.json").read_text())
    assert resumed["args"]["max_iters"] == 8  # noqa: PLR2004
    assert [(r["step"], r["max_iters"]) for r in resumed["resumes"]] == [(4, 8)]
    model = DiffGPT(
        model=GPT(vocab_size=64, n_embd=16, block_size=32, n_head=2, n_layer=1),
        order_of_derivative=1,
        domain_of_definition=np.array(config["domain_of_definition"]),
        tokenizer=config["tokenizer"],
    )
    model.load(str(crashed / kept[0]))


def test_resume_takes_the_settings_of_the_run(tmp_path: Path) -> None:
    """
    Check what --resume refuses: each exits and leaves the run as it was.

    It takes no settings but --max-iters, which must lie above the updates
    made, needs a state.pt, and the corpus must be the run's: here its
    manifest changed. A --keep-every of 0 is refused before a run starts.
    """
    data = ("--corpus", str(_corpus(tmp_path)))
    run = tmp_path / "run"
    main([*data, *TINY, "--warmup", "1", "--output", str(run), "--max-iters", "2"])

    def files() -> dict[str, bytes]:
        return {
            path.name: path.read_bytes() for path in run.iterdir() if path.is_file()
        }

    before = files()
    resume = ["--resume", "--output", str(run)]
    for extra in (["--vocab", "32"], ["--max-iters", "4", "--seed", "1"], []):
        with pytest.raises(SystemExit):
            main([*resume, *extra])
    for extra in (["--max-iters", "2"], ["--max-iters", "1"]):
        with pytest.raises(SystemExit):
            main([*resume, *extra])
    (tmp_path / "manifest.json").write_text('{"splits": {"other": 1}}\n')
    with pytest.raises(SystemExit):
        main([*resume, "--max-iters", "4"])
    assert files() == before
    old = tmp_path / "old"
    old.mkdir()
    (old / "config.json").write_bytes(before["config.json"])
    with pytest.raises(SystemExit):
        main(["--resume", "--output", str(old), "--max-iters", "4"])
    with pytest.raises(SystemExit):
        main([*data, "--output", str(tmp_path / "zero"), "--keep-every", "0"])
    assert not (tmp_path / "zero").exists()


def test_gradient_checkpointing_is_a_gpus_by_default() -> None:
    """Auto recomputes the activations on a GPU, whose memory needs it, alone."""
    assert gradient_checkpointing("auto", "cuda")
    assert gradient_checkpointing("auto", "cuda:0")
    assert not gradient_checkpointing("auto", "cpu")
    assert gradient_checkpointing("on", "cpu")
    assert not gradient_checkpointing("off", "cuda")


def test_kept_activations_train_the_run_of_recomputed_ones(tmp_path: Path) -> None:
    """
    A CPU run keeps its activations and trains the run that recomputes them.

    The same history but for the timings, the same scores, and the same
    weights in every checkpoint and in state.pt, bit for bit.
    """
    data = ("--corpus", str(_corpus(tmp_path)), *TINY, "--warmup", "1")
    kept, recomputed = tmp_path / "kept", tmp_path / "recomputed"
    main([*data, "--max-iters", "4", "--output", str(kept)])
    main(
        [
            *(*data, "--max-iters", "4", "--output", str(recomputed)),
            *("--gradient-checkpointing", "on"),
        ],
    )
    runs = (kept, recomputed)
    configs = [json.loads((run / "config.json").read_text()) for run in runs]
    assert [config["gradient_checkpointing"] for config in configs] == [False, True]
    assert _history(kept) == _history(recomputed)
    scores = [(run / "scores.jsonl").read_text() for run in runs]
    assert scores[0] == scores[1]
    for name in ("best.bin", "last.bin"):
        saved = [torch.load(run / name, weights_only=True) for run in runs]
        assert _same(*saved), name
    ends = [torch.load(run / RESUME_STATE, weights_only=True) for run in runs]
    for end in ends:
        assert end.pop("seconds") > 0
    assert _same(*ends)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU"),
        ),
    ],
)
def test_saved_rng_states_draw_again_what_they_drew(device: str) -> None:
    """
    The global RNG states, through a file, make every generator repeat its draws.

    Nothing in a run draws from them, but code that did would go on the same.
    """
    buffer = io.BytesIO()
    torch.save(_rng_states(device), buffer)
    buffer.seek(0)
    saved = torch.load(buffer, weights_only=True)

    def draw() -> tuple[torch.Tensor, torch.Tensor, np.ndarray, float]:
        return (
            torch.rand(3),
            torch.rand(3, device=device).cpu(),
            np.random.random(3),  # noqa: NPY002 - the global numpy state
            random.random(),  # noqa: S311 - not for cryptography
        )

    first = draw()
    _set_rng_states(saved)
    again = draw()
    assert torch.equal(first[0], again[0])
    assert torch.equal(first[1], again[1])
    assert np.array_equal(first[2], again[2])
    assert first[3] == again[3]
