"""TempoPFN's CauKer, from its clone, with `cauker_generator` replaced: see `src`."""

from diff_gpt.infrastructure.corpus.tempopfn import TEMPOPFN_ROOT

__path__.append(str(TEMPOPFN_ROOT / "src" / "synthetic_generation" / "cauker"))
