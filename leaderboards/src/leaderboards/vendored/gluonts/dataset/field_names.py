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
The names of the fields of a dataset entry.

This is gluonts 0.15.1's `gluonts/dataset/field_names.py` (Apache-2.0, a copy
of the license in leaderboards/LICENSE-gluonts), cut to the names that the
copies use. Unchanged but for the style of leaderboards.vendored.
"""


class FieldName:
    """A bundle of default field names, for clients instantiating transformers."""

    START = "start"
    TARGET = "target"

    FEAT_STATIC_CAT = "feat_static_cat"
    FEAT_STATIC_REAL = "feat_static_real"
    FEAT_DYNAMIC_CAT = "feat_dynamic_cat"
    FEAT_DYNAMIC_REAL = "feat_dynamic_real"
    PAST_FEAT_DYNAMIC_REAL = "past_feat_dynamic_real"
    FEAT_DYNAMIC_REAL_LEGACY = "dynamic_feat"
