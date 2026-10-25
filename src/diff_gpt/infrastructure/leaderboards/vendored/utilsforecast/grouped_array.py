"""
Series stored end to end in one array, read one series at a time.

This is utilsforecast 0.2.17's `utilsforecast/grouped_array.py` (Apache-2.0, a
copy of the license in
src/diff_gpt/infrastructure/leaderboards/LICENSE-utilsforecast), cut to
GroupedArray's `[]`, the one method of it that statsforecast's GroupedArray
uses: it iterates over its series with it. Unchanged but for the style of
diff_gpt.infrastructure.leaderboards.vendored.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class GroupedArray:
    """Series i is data[indptr[i]:indptr[i + 1]]; n_groups series in all."""

    data: np.ndarray
    indptr: np.ndarray
    n_groups: int

    def __getitem__(self, idx: int) -> np.ndarray:
        """Return series `idx`."""
        if idx < 0:
            idx = self.n_groups + idx
        return self.data[self.indptr[idx] : self.indptr[idx + 1]]
