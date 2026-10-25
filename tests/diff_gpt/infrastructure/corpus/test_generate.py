"""The determinism of `generate`: the series of a seed, whatever process makes it."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

from diff_gpt.infrastructure.corpus import export, sources

if TYPE_CHECKING:
    from pathlib import Path

FILE_SIZE = sources.CHUNK
THREADS = (
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
SETTINGS = (*THREADS, "PYO_GUI_WX")


def generate(
    root: Path,
    source: str,
    series: range,
    hash_seed: str = "0",
) -> subprocess.CompletedProcess[str]:
    """Run `generate` on the `series` of `source` in train, in a new interpreter."""
    return subprocess.run(  # noqa: S603 - our own argv, no shell.
        [
            *(
                sys.executable,
                "-m",
                "diff_gpt.interfaces.corpus.generate",
                "train",
                source,
            ),
            *("--start", str(series.start), "--stop", str(series.stop)),
            *("--file-size", str(FILE_SIZE), "--root", str(root)),
        ],
        env={**os.environ, "PYTHONHASHSEED": hash_seed},
        capture_output=True,
        text=True,
        check=False,
    )


def test_generate_refuses_a_random_hash_seed(tmp_path: Path) -> None:
    """TempoPFN salts its parameter seeds with hash(class name)."""
    run = generate(tmp_path, "sinewave", range(FILE_SIZE), hash_seed="random")
    assert run.returncode != 0
    assert "PYTHONHASHSEED='random'" in run.stderr
    assert not (tmp_path / "arrow").exists()


@pytest.mark.parametrize("source", ["sinewave", "audio_network_topology"])
def test_files_made_apart_equal_files_made_together(
    tmp_path: Path, source: str
) -> None:
    """Two files of one process are byte for byte two files made one per process."""
    together, apart = tmp_path / "together", tmp_path / "apart"
    runs = [
        generate(together, source, range(2 * FILE_SIZE)),
        generate(apart, source, range(FILE_SIZE, 2 * FILE_SIZE)),
        generate(apart, source, range(FILE_SIZE)),
    ]
    assert [run.returncode for run in runs] == [0, 0, 0], runs[-1].stderr[-4000:]

    export.export(together)
    export.export(apart)

    npy = f"npy/train/{source}.npy"
    assert (together / npy).read_bytes() == (apart / npy).read_bytes()


def settings_after(imports: str) -> dict[str, object]:
    """Return what `imports` loads and sets in a new interpreter without SETTINGS."""
    report = (
        "import json, os, sys; print(json.dumps({"
        "'loaded': [name for name in ('numpy', 'torch') if name in sys.modules], "
        f"'settings': {{name: os.environ.get(name) for name in {SETTINGS!r}}}}}))"
    )
    run = subprocess.run(  # noqa: S603 - our own argv, no shell.
        [sys.executable, "-c", f"{imports}\n{report}"],
        env={name: value for name, value in os.environ.items() if name not in SETTINGS},
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr[-4000:]
    return json.loads(run.stdout.splitlines()[-1])


def test_thread_settings_belong_to_the_clis() -> None:
    """
    The corpus code sets no thread counts; the package of its CLIs does.

    Importing diff_gpt.infrastructure.corpus and its modules, which load numpy,
    torch and TempoPFN's modules, leaves the five thread variables and
    PYO_GUI_WX unset; importing diff_gpt.interfaces.corpus, which
    `python -m` imports before a CLI's module, sets them before numpy loads.
    """
    imports = (
        "from diff_gpt.infrastructure.corpus import "
        "export, generate, schedule, sources, stats"
    )
    assert settings_after(imports) == {
        "loaded": ["numpy", "torch"],
        "settings": dict.fromkeys(SETTINGS),
    }
    assert settings_after("import diff_gpt.interfaces.corpus") == {
        "loaded": [],
        "settings": {**dict.fromkeys(THREADS, "1"), "PYO_GUI_WX": "0"},
    }
