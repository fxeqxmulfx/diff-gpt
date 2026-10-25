"""
The command lines of the synthetic corpus: export, generate, schedule, stats.

`python -m diff_gpt.interfaces.corpus.<command>` imports this package before
the command's module, and so before anything that module imports: diff_gpt
and diff_gpt.interfaces hold docstrings only. The code below then runs before
numpy loads. It sets the thread counts that numpy, BLAS and torch read when
they load to 1, so that each process runs single threaded, one per CPU that
`schedule` may use, and schedule's `generate` processes inherit them with its
environment; it also keeps pyo from printing that it found no WxPython. They
are settings of the process, so they are a command line's: importing
diff_gpt.infrastructure.corpus sets none of them.
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
