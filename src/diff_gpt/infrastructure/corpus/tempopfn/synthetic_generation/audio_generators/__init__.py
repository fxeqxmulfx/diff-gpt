"""TempoPFN's audio generators, from its clone, with `utils` replaced: see `src`."""

from diff_gpt.infrastructure.corpus.tempopfn import TEMPOPFN_ROOT

__path__.append(
    str(TEMPOPFN_ROOT / "src" / "synthetic_generation" / "audio_generators")
)
