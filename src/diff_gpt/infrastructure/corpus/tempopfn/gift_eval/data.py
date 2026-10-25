"""
Stand-in for TempoPFN's `src.gift_eval.data`.

`src.data.augmentations` imports `Dataset` from it at module level, but only
`analyze_datasets_for_augmentation` uses it, to read the GIFT-Eval train
splits. The corpus is synthetic only, for a zero-shot entry, so this
`Dataset` refuses to load anything.
"""


class Dataset:
    """A GIFT-Eval dataset that cannot be loaded."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Refuse to load the dataset named by `args` and `kwargs`."""
        msg = (
            f"refusing to load the GIFT-Eval dataset {args} {kwargs}: the "
            "synthetic corpus must not read GIFT-Eval data"
        )
        raise RuntimeError(msg)
