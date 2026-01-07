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

"""base constraint handler class."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class ConstraintHandler(ABC):
    """abstract base class for constraint handlers."""

    @abstractmethod
    def get_internal_column_names(self) -> list[str]:
        """return list of internal column names created by this handler."""
        pass

    @abstractmethod
    def to_internal(self, df: pd.DataFrame) -> pd.DataFrame:
        """transform dataframe (in-place) from user schema to internal schema."""
        pass

    @abstractmethod
    def to_original(self, df: pd.DataFrame) -> pd.DataFrame:
        """transform dataframe (in-place) from internal schema back to user schema."""
        pass

    @abstractmethod
    def get_encoding_types(self) -> dict[str, str]:
        """return encoding types for internal columns."""
        pass

    def _validate_columns(self, df: pd.DataFrame, columns: list[str]) -> None:
        """validate that all required columns exist in the dataframe."""
        missing_cols = set(columns) - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"Columns {sorted(missing_cols)} required by {self.__class__.__name__} "
                f"not found in dataframe. Available columns: {sorted(df.columns)}"
            )
