# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
The iteration helpers that gluonts' datasets and evaluation use.

This is gluonts 0.15.1's `gluonts/itertools.py` (Apache-2.0, a copy of the
license in src/diff_gpt/infrastructure/leaderboards/LICENSE-gluonts), cut to
SizedIterable, `prod`, `batcher` and Map. Unchanged but for the style of
diff_gpt.infrastructure.leaderboards.vendored.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator


@runtime_checkable
class SizedIterable(Protocol):
    """An iterable with a length."""

    def __len__(self) -> int:
        """Return the number of elements."""
        ...

    def __iter__(self) -> Iterator:
        """Iterate over the elements."""
        ...


def prod(xs: Iterable) -> object:
    """Compute the product of the elements of an iterable object."""
    p = 1
    for x in xs:
        p *= x
    return p


def batcher[T](iterable: Iterable[T], batch_size: int) -> Iterator[list[T]]:
    """
    Group elements from `iterable` into batches of size `batch_size`.

    >>> list(batcher("ABCDEFG", 3))
    [['A', 'B', 'C'], ['D', 'E', 'F'], ['G']]

    Unlike the grouper proposed in the documentation of itertools, `batcher`
    doesn't fill up missing values.
    """
    it: Iterator[T] = iter(iterable)

    def get_batch() -> list[T]:
        return list(itertools.islice(it, batch_size))

    # has an empty list so that we have a 2D array for sure
    return iter(get_batch, [])


@dataclass
class Map:
    """`fn` mapped over `iterable`, lazily, with the length of `iterable`."""

    fn: Callable
    iterable: SizedIterable

    def __iter__(self) -> Iterator:
        """Iterate over `fn` of each element."""
        return map(self.fn, self.iterable)

    def __len__(self) -> int:
        """Return the length of `iterable`."""
        return len(self.iterable)
