"""TempoPFN's KernelSynth, from its clone, with `kernel_synth` replaced: see `src`."""

from src import TEMPOPFN_ROOT

__path__.append(str(TEMPOPFN_ROOT / "src" / "synthetic_generation" / "kernel_synth"))
