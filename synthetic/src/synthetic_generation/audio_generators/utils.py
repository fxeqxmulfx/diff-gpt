"""
Stand-in for TempoPFN's `src.synthetic_generation.audio_generators.utils`.

TempoPFN renders each audio series on a new offline pyo server and never sets
the server's global seed, so pyo seeds its noise objects (Noise, BrownNoise)
from the clock: audio_financial_volatility, audio_multi_scale_fractal and
audio_network_topology came out different on every run. These are TempoPFN's
`run_offline_pyo` and `normalize_waveform` (Apache-2.0), changed in one
place: `run_offline_pyo` sets the global seed of the server from numpy's
global generator. The audio wrappers reseed that generator at the start of
each batch with the seed of its first series and draw nothing else from it,
so each series gets its noise from its batch's seed and its index in it.
"""

from __future__ import annotations

import os
import tempfile
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from pyo import NewTable, Server, TableRec

if TYPE_CHECKING:
    from collections.abc import Callable

# pyo seeds each random object with the global seed plus a multiple of the
# count of objects of its kind on the server, as a C int; 0 means the clock.
_SEED_BOUND = 2**30


def run_offline_pyo(
    synth_builder: Callable[[], object],
    server_duration: float,
    sample_rate: int,
    length: int,
) -> np.ndarray:
    """Render the graph of `synth_builder` offline, seeded, into `length` samples."""
    seed = int(np.random.randint(1, _SEED_BOUND))  # noqa: NPY002 - seeded per batch by TempoPFN.
    with (
        Path(os.devnull).open("w") as devnull,
        redirect_stdout(devnull),
        redirect_stderr(devnull),
    ):
        server = Server(sr=sample_rate, nchnls=1, duplex=0, audio="offline")
        # The offline server needs a file name even when it records to a table.
        wav = (
            Path(tempfile.gettempdir())
            / f"pyo_offline_{os.getpid()}_{time.time_ns()}.wav"
        )
        server.recordOptions(dur=server_duration, filename=str(wav), fileformat=0)
        server.boot()
        server.setGlobalSeed(seed)
        table = NewTable(length=server_duration, chnls=1)
        synth = synth_builder()
        _recorder = TableRec(synth, table, fadetime=0.01).play()
        server.start()
        server.stop()
        server.shutdown()
        wav.unlink(missing_ok=True)

    waveform = np.array(table.getTable())
    if waveform.size > length:
        return waveform[:length]
    return np.concatenate([waveform, np.zeros(length - waveform.size, waveform.dtype)])


def normalize_waveform(values: np.ndarray) -> np.ndarray:
    """Scale `values` to a maximum absolute value of 1, unless they are all 0."""
    max_abs = np.max(np.abs(values)) if values.size > 0 else 0.0
    if max_abs > 0:
        return values / max_abs
    return values
