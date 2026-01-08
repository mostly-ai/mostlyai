# Copyright 2025 MOSTLY AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""fixed combinations constraint handler."""

from __future__ import annotations

import hashlib
import json
import logging

import pandas as pd

from mostlyai.sdk._data.constraints.types.base import ConstraintHandler
from mostlyai.sdk.client._constraint_types import FixedCombinations

_LOG = logging.getLogger(__name__)


def _generate_internal_column_name(prefix: str, columns: list[str]) -> str:
    """generate a deterministic internal column name."""
    key = "|".join(columns)
    hash_suffix = hashlib.md5(key.encode()).hexdigest()[:8]
    columns_str = "_".join(col.upper() for col in columns)
    return f"__CONSTRAINT_{prefix}_{columns_str}_{hash_suffix}__"


class FixedCombinationsHandler(ConstraintHandler):
    """handler for FixedCombinations constraints."""

    def __init__(self, constraint: FixedCombinations):
        self.constraint = constraint
        self.table_name = constraint.table_name
        self.columns = constraint.columns
        self.merged_name = _generate_internal_column_name("FC", self.columns)

    def get_internal_column_names(self) -> list[str]:
        return [self.merged_name]

    def to_internal(self, df: pd.DataFrame) -> pd.DataFrame:
        self._validate_columns(df, self.columns)

        def merge_row(row):
            values = [row[col] if pd.notna(row[col]) else None for col in self.columns]
            # JSON serialization handles all escaping automatically
            return json.dumps(values, ensure_ascii=False)

        df[self.merged_name] = df.apply(merge_row, axis=1)
        return df

    def to_original(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.merged_name in df.columns:

            def split_row(merged_value: str) -> list[str]:
                if pd.isna(merged_value):
                    return [""] * len(self.columns)
                elif merged_value == "_RARE_":
                    return ["_RARE_"] * len(self.columns)
                try:
                    values = json.loads(merged_value)
                    return [str(v) if v is not None else "" for v in values]
                except json.JSONDecodeError:
                    _LOG.error(f"failed to decode JSON for {merged_value}; using empty values")
                    return [""] * len(self.columns)

            split_values = df[self.merged_name].apply(split_row)
            split_df = pd.DataFrame(split_values.tolist(), index=df.index)

            # preserve original index
            original_index = df.index
            split_df.index = original_index

            # assign to original columns
            for i, col in enumerate(self.columns):
                df[col] = split_df[i].values

            # drop the merged column
            df = df.drop(columns=[self.merged_name])
        return df

    def get_encoding_types(self) -> dict[str, str]:
        # always use TABULAR encoding for constraints, regardless of model_type
        # constraints merge columns which requires categorical encoding
        return {self.merged_name: "TABULAR_CATEGORICAL"}
