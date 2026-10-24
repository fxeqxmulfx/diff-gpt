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
A transformation of a dataset, entry by entry, applied lazily.

This is gluonts 0.15.1's `gluonts/transform/_base.py` (Apache-2.0, a copy of
the license in leaderboards/LICENSE-gluonts), cut to Transformation, without
its `chain` and `+`, and TransformedDataset. Unchanged but for the style of
leaderboards.vendored.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

from leaderboards.vendored.gluonts.dataset import Dataset

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from leaderboards.vendored.gluonts.dataset import DataEntry


class Transformation(metaclass=abc.ABCMeta):
    """
    Base class for all Transformations.

    A Transformation processes works on a stream (iterator) of dictionaries.
    """

    @abc.abstractmethod
    def __call__(
        self, data_it: Iterable[DataEntry], *, is_train: bool
    ) -> Iterable[DataEntry]:
        """Transform the entries of `data_it`."""

    def apply(self, dataset: Dataset, *, is_train: bool = True) -> TransformedDataset:
        """Return `dataset`, transformed lazily."""
        return TransformedDataset(dataset, self, is_train=is_train)


class TransformedDataset(Dataset):
    """
    A dataset of the elements of base_dataset, each transformed.

    This only supports SimpleTransformations, which do the same thing at
    prediction and training time.

    Parameters
    ----------
    base_dataset
        Dataset to transform
    transformations
        List of transformations to apply

    """

    def __init__(
        self,
        base_dataset: Dataset,
        transformation: Transformation,
        *,
        is_train: bool = True,
    ) -> None:
        """Transform `base_dataset` with `transformation`."""
        self.base_dataset = base_dataset
        self.transformation = transformation
        self.is_train = is_train

    def __len__(self) -> int:
        """Return the number of transformed entries."""
        # NOTE this is unsafe when transformations are run with is_train = True
        # since some transformations may not be deterministic
        # (instance splitter)
        return sum(1 for _ in self)

    def __iter__(self) -> Iterator[DataEntry]:
        """Iterate over the transformed entries."""
        yield from self.transformation(self.base_dataset, is_train=self.is_train)
