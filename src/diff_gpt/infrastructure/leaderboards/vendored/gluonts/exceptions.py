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
The error that gluonts' processing of a dataset entry raises on bad data.

This is gluonts 0.15.1's `gluonts/exceptions.py` (Apache-2.0, a copy of the
license in leaderboards/LICENSE-gluonts), cut to GluonTSException, without
its `guard`, and GluonTSDataError, the error that the copy of
`gluonts/dataset/common.py` raises. Unchanged but for the style of
leaderboards.vendored.
"""


class GluonTSException(Exception):
    """Base class for all GluonTS exceptions."""


class GluonTSDataError(GluonTSException):
    """An exception indicating an error with the input data."""
