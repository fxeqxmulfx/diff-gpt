"""
The synthetic pretraining corpus of diff-gpt, made with TempoPFN's generators.

`generate` runs a range of series of one source of one split in a process,
deterministically, and writes TempoPFN's Arrow batches; `schedule` runs the
`generate` processes of whole splits; `export` turns the batches into float32
.npy arrays for the diff-gpt loader and writes the manifest; `stats` sums up
their values. Each process runs single threaded, one per CPU that `schedule`
may use: importing the package sets the thread counts below, which numpy,
BLAS and torch read when they load. It also keeps pyo from printing that it
found no WxPython.
"""

import os

for _variable in (
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_variable] = "1"
os.environ.setdefault("PYO_GUI_WX", "0")
