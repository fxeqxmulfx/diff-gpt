"""
TempoPFN's package `src`, loaded from its clone, with four modules replaced.

TempoPFN imports its own code as `src.*` from the root of its repository.
This package appends the `src` folder of the clone to its own path, so that
`src.synthetic_generation`, `src.data` and the rest load from there, unmodified.
A finder first on sys.meta_path serves the imports of `src` and `src.<rest>`
with this package and its modules, so that each file loads once, under its
name here: the stand-ins below still replace the clone's files, and pickles
and isinstance checks see one class.
The clone is `TEMPOPFN_ROOT`, by default the folder TempoPFN next to this
repository. Importing this package keeps the whole process from writing
bytecode caches, so none is written into the clone; Python still reads the
caches there are.

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

import importlib
import importlib.abc
import importlib.util
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, override

from diff_gpt.application import paths

if TYPE_CHECKING:
    from collections.abc import Sequence
    from importlib.machinery import ModuleSpec
    from types import ModuleType

TEMPOPFN_ROOT = Path(
    os.environ.get(
        "TEMPOPFN_ROOT",
        paths.ROOT.parent / "TempoPFN",
    ),
)

sys.dont_write_bytecode = True
__path__.append(str(TEMPOPFN_ROOT / "src"))


class _Alias(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Serve the imports of `src` and `src.<rest>` with this package's modules."""

    def __init__(self) -> None:
        """Hold the specs of the modules imported under their `src` names."""
        self._specs: dict[str, ModuleSpec] = {}

    @override
    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        """Return this loader's spec for `src` and `src.<rest>`, None for the rest."""
        if fullname == "src" or fullname.startswith("src."):
            return importlib.util.spec_from_loader(fullname, self)
        return None

    @override
    def create_module(self, spec: ModuleSpec) -> ModuleType:
        """Import the module of this package that `spec` names under `src`."""
        module = importlib.import_module(__name__ + spec.name.removeprefix("src"))
        self._specs[spec.name] = module.__spec__
        return module

    @override
    def exec_module(self, module: ModuleType) -> None:
        """Give the module back its own spec: it ran when it was imported."""
        module.__spec__ = self._specs.pop(module.__spec__.name)


sys.meta_path.insert(0, _Alias())
