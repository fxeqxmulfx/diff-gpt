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
The types of gluonts' datasets: an entry is a dict, a dataset iterates entries.

This is gluonts 0.15.1's `gluonts/dataset/__init__.py` (Apache-2.0, a copy of
the license in leaderboards/LICENSE-gluonts), cut to DataEntry and Dataset.
Unchanged but for the style of leaderboards.vendored.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterator

# Dictionary used for data flowing through the transformations.
DataEntry = dict[str, Any]


@runtime_checkable
class Dataset(Protocol):
    """An iterable of entries with a length."""

    def __iter__(self) -> Iterator[DataEntry]:
        """Iterate over the entries."""
        raise NotImplementedError

    def __len__(self) -> int:
        """Return the number of entries."""
        raise NotImplementedError
