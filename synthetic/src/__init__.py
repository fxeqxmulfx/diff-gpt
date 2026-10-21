"""
TempoPFN's package `src`, loaded from its clone, with four modules replaced.

TempoPFN imports its own code as `src.*` from the root of its repository.
This package appends the `src` folder of the clone to its own path, so that
`src.synthetic_generation`, `src.data` and the rest load from there, unmodified.
The clone is `TEMPOPFN_ROOT`, by default the folder TempoPFN next to this
repository. Python writes no bytecode caches into it.

Four modules resolve here first, to stand-ins:
- `src.gift_eval`. TempoPFN's package imports its forecasting model
  (flash-linear-attention, triton) and gluonts, and `src.data.augmentations`,
  which both offline augmentation scripts import, needs it only for the
  GIFT-Eval `Dataset` of an analysis the corpus never runs
  (`analyze_datasets_for_augmentation`, which builds nan_stats.json from the
  GIFT-Eval train splits);
- `src.synthetic_generation.audio_generators.utils`, whose pyo rendering is
  seeded here: see that module;
- `src.synthetic_generation.kernel_synth.kernel_synth`, whose GP prior is
  sampled here by a pivoted Cholesky factorization in place of an SVD, 57
  times as fast, from a kernel matrix built from the lags of the grid: see
  that module;
- `src.synthetic_generation.cauker.cauker_generator`, whose roots draw
  independent noises here, on the CPU, as in the original CauKer, where
  TempoPFN draws one noise for all roots of a series on the GPU: see that
  module.

`sources.environment` records a digest of each module here, so that the runs
of a source cannot mix the output of two versions of them.
"""

import os
import sys
from pathlib import Path

TEMPOPFN_ROOT = Path(
    os.environ.get(
        "TEMPOPFN_ROOT",
        Path(__file__).resolve().parents[3] / "TempoPFN",
    ),
)

sys.dont_write_bytecode = True
__path__.append(str(TEMPOPFN_ROOT / "src"))
