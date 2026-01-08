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

"""inequality constraint handler."""

from __future__ import annotations

import hashlib
import logging

import pandas as pd

from mostlyai.sdk._data.constraints.types.base import ConstraintHandler
from mostlyai.sdk.client._constraint_types import Inequality
from mostlyai.sdk.domain import ModelEncodingType

_LOG = logging.getLogger(__name__)


def _generate_internal_column_name(prefix: str, columns: list[str]) -> str:
    """generate a deterministic internal column name."""
    key = "|".join(columns)
    hash_suffix = hashlib.md5(key.encode()).hexdigest()[:8]
    columns_str = "_".join(col.upper() for col in columns)
    return f"__CONSTRAINT_{prefix}_{columns_str}_{hash_suffix}__"


class InequalityHandler(ConstraintHandler):
    """handler for Inequality constraints (low <= high)."""

    _DATETIME_EPOCH = pd.Timestamp("1970-01-01")  # reference epoch for delta representation

    def __init__(self, constraint: Inequality, table=None):
        self.constraint = constraint
        self.table_name = constraint.table_name
        self.low_column = constraint.low_column
        self.high_column = constraint.high_column
        self._delta_column = _generate_internal_column_name("INEQ_DELTA", [self.low_column, self.high_column])

        # determine if this is a datetime constraint based on table encoding types
        self._is_datetime = False
        if table and table.columns:
            datetime_encodings = {
                ModelEncodingType.tabular_datetime,
                # ModelEncodingType.tabular_datetime_relative,  # not supported yet
            }
            # check if either column is datetime-encoded
            self._is_datetime = all(
                col.model_encoding_type in datetime_encodings
                for col in table.columns
                if col.name in {self.low_column, self.high_column}
            )

    def __repr__(self) -> str:
        return f"{self.low_column} <= {self.high_column}"

    def get_internal_column_names(self) -> list[str]:
        return [self._delta_column]

    def to_internal(self, df: pd.DataFrame) -> pd.DataFrame:
        self._validate_columns(df, [self.low_column, self.high_column])
        low = df[self.low_column]
        high = df[self.high_column]

        # handle NAs: if either low or high is NA, set delta to NA
        na_mask = low.isna() | high.isna()

        zero = pd.Timedelta(0) if self._is_datetime else 0
        delta = high - low

        # set delta to NA where either low or high is NA
        delta[na_mask] = pd.NA

        # only check violations where we have valid values
        violations = (delta < zero) & ~na_mask
        if violations.any():
            _LOG.warning(f"detected {violations.sum() / len(delta) * 100:.2f}% inequality violations for {self}")

        # convert timedelta to datetime using epoch (for datetime constraints)
        if self._is_datetime:
            # represent delta as datetime: epoch + timedelta
            # preserve NAs during conversion
            delta = delta.where(na_mask, self._DATETIME_EPOCH + delta)
        # for numeric, keep as-is (numeric delta)

        df[self._delta_column] = delta
        return df

    def to_original(self, df: pd.DataFrame) -> pd.DataFrame:
        """transform from internal schema back to original schema."""
        if self._delta_column not in df.columns:
            return df

        # prepare data and types
        low = df[self.low_column]
        delta = df[self._delta_column]
        high_dtype = df[self.high_column].dtype

        # convert datetime back to timedelta if needed
        if self._is_datetime:
            # ensure delta is datetime dtype before subtraction
            delta = pd.to_datetime(delta)
            # preserve NAs during conversion
            delta = delta - self._DATETIME_EPOCH
            delta = delta.astype("timedelta64[ns]")  # explicit dtype conversion

        # RECONSTRUCTION
        low_na = low.isna()
        delta_na = delta.isna()

        # if low != NA and delta != NA, then high = low + delta
        both_valid = ~low_na & ~delta_na
        df.loc[both_valid, self.high_column] = low[both_valid] + delta[both_valid]

        # if low != NA and delta == NA, then high = NA
        low_valid_delta_na = ~low_na & delta_na
        df.loc[low_valid_delta_na, self.high_column] = pd.NA
        # else, if low == NA, then high is kept as is

        # if any violations made it through, apply a primitive correction
        high = df[self.high_column]
        both_valid_for_check = ~low_na & ~high.isna()
        violations = both_valid_for_check & (high < low)
        df.loc[violations, self.high_column] = low[violations]

        # convert back to original dtype
        if pd.api.types.is_integer_dtype(high_dtype):
            df[self.high_column] = df[self.high_column].astype(high_dtype)

        return df.drop(columns=[self._delta_column])

    def get_encoding_types(self) -> dict[str, str]:
        # use TABULAR_DATETIME for datetime constraints to preserve precision
        # use TABULAR_NUMERIC_AUTO for numeric constraints
        if self._is_datetime:
            return {self._delta_column: "TABULAR_DATETIME"}
        return {self._delta_column: "TABULAR_NUMERIC_AUTO"}
