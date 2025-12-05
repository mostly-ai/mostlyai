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

"""constraint transformation utilities."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from mostlyai.sdk.domain import FixedCombination, Inequality, OneHotEncoding, Range

if TYPE_CHECKING:
    from mostlyai.sdk.domain import Generator, ModelType

_LOG = logging.getLogger(__name__)


def _generate_internal_column_name(prefix: str, columns: list[str]) -> str:
    """generate a deterministic internal column name."""
    key = "|".join(columns)
    hash_suffix = hashlib.md5(key.encode()).hexdigest()[:8]
    return f"__constraint_{prefix}_{hash_suffix}"


def get_tgt_meta_path(workspace_dir: Path) -> Path:
    """get tgt-meta directory path in OriginalData."""
    path = workspace_dir / "OriginalData" / "tgt-meta"
    path.mkdir(parents=True, exist_ok=True)
    return path


class ConstraintHandler(ABC):
    """abstract base class for constraint handlers."""

    @abstractmethod
    def get_internal_column_names(self) -> list[str]:
        """return list of internal column names created by this handler."""
        pass

    @abstractmethod
    def get_original_columns(self) -> list[str]:
        """return original column names involved in this constraint."""
        pass

    @abstractmethod
    def to_internal(self, df: pd.DataFrame) -> pd.DataFrame:
        """transform dataframe from user schema to internal schema."""
        pass

    @abstractmethod
    def to_original(self, df: pd.DataFrame, seed_data: pd.DataFrame | None = None) -> pd.DataFrame:
        """transform dataframe from internal schema back to user schema."""
        pass

    @abstractmethod
    def get_encoding_types(self) -> dict[str, str]:
        """return encoding types for internal columns."""
        pass

    def _validate_columns(self, df: pd.DataFrame, columns: list[str]) -> None:
        """validate that all required columns exist in the dataframe."""
        missing_cols = set(columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"columns {sorted(missing_cols)} not found in dataframe")


class FixedCombinationHandler(ConstraintHandler):
    """handler for FixedCombination constraints."""

    # use record separator (ASCII 30) as delimiter - very unlikely to appear in text data
    # csv module handles escaping automatically using quotes
    _SEPARATOR = "\x1e"

    def __init__(self, constraint: FixedCombination):
        self.constraint = constraint
        self.columns = constraint.columns
        # use separator for column name (for display/debugging, actual data uses _SEPARATOR)
        self.merged_name = "|".join(self.columns)

    def get_internal_column_names(self) -> list[str]:
        return [self.merged_name]

    def get_original_columns(self) -> list[str]:
        return list(self.columns)

    def to_internal(self, df: pd.DataFrame) -> pd.DataFrame:
        self._validate_columns(df, self.columns)
        df = df.copy()

        # use csv module to handle escaping automatically
        def merge_row(row):
            values = ["" if pd.isna(row[col]) else str(row[col]) for col in self.columns]
            s = io.StringIO()
            writer = csv.writer(s, delimiter=self._SEPARATOR, lineterminator="", quoting=csv.QUOTE_MINIMAL)
            writer.writerow(values)
            return s.getvalue()

        df[self.merged_name] = df.apply(merge_row, axis=1)
        return df

    def to_original(self, df: pd.DataFrame, seed_data: pd.DataFrame | None = None) -> pd.DataFrame:
        df = df.copy()
        if self.merged_name in df.columns:
            # check if all columns in this combination were seeded
            if seed_data is not None:
                seed_cols = set(seed_data.columns)
                if all(col in seed_cols for col in self.columns):
                    # all columns were seeded - preserve them from seed_data
                    if len(seed_data) == len(df):
                        for col in self.columns:
                            df[col] = seed_data[col].values
                        df = df.drop(columns=[self.merged_name])
                        return df

            # split the merged column using csv module (handles escaping automatically)
            def split_row(merged_value: str) -> list[str]:
                if pd.isna(merged_value):
                    return [""] * len(self.columns)
                reader = csv.reader(io.StringIO(merged_value), delimiter=self._SEPARATOR)
                try:
                    parts = next(reader)
                except StopIteration:
                    parts = []
                # pad or truncate to expected number of columns
                parts = (parts + [""] * len(self.columns))[: len(self.columns)]
                return parts

            split_values = df[self.merged_name].astype(str).apply(split_row)
            split_df = pd.DataFrame(split_values.tolist(), index=df.index)

            # assign to original columns
            for i, col in enumerate(self.columns):
                # if this column was seeded, preserve seed value
                if seed_data is not None and col in seed_data.columns and len(seed_data) == len(df):
                    df[col] = seed_data[col].values
                else:
                    df[col] = split_df[i]

            # drop the merged column
            df = df.drop(columns=[self.merged_name])
        return df

    def get_encoding_types(self) -> dict[str, str]:
        return {self.merged_name: "TABULAR_CATEGORICAL"}


class InequalityHandler(ConstraintHandler):
    """handler for Inequality constraints (low <= high or low < high if strict_boundaries=True)."""

    # fixed epsilon values for strict boundaries
    _NUMERIC_EPSILON = 1e-10
    _DATETIME_EPSILON = pd.Timedelta(microseconds=1)

    def __init__(self, constraint: Inequality):
        self.constraint = constraint
        self.low_column = constraint.low_column
        self.high_column = constraint.high_column
        self.strict_boundaries = constraint.strict_boundaries
        self._delta_column = _generate_internal_column_name("ineq_delta", [self.low_column, self.high_column])

    def get_internal_column_names(self) -> list[str]:
        return [self._delta_column]

    def get_original_columns(self) -> list[str]:
        return [self.low_column, self.high_column]

    def to_internal(self, df: pd.DataFrame) -> pd.DataFrame:
        self._validate_columns(df, [self.low_column, self.high_column])
        df = df.copy()
        low = df[self.low_column]
        high = df[self.high_column]

        is_datetime = pd.api.types.is_datetime64_any_dtype(low) or pd.api.types.is_datetime64_any_dtype(high)

        if is_datetime:
            low = pd.to_datetime(low)
            high = pd.to_datetime(high)
            delta = high - low
            violations = delta < pd.Timedelta(0)
        else:
            low = pd.to_numeric(low, errors="coerce")
            high = pd.to_numeric(high, errors="coerce")
            delta = high - low
            violations = delta < 0

        if violations.any():
            _LOG.warning(
                f"correcting {violations.sum()} inequality violations for {self.low_column} <= {self.high_column}"
            )
            delta = delta.abs()

        # enforce strict boundaries if needed
        if self.strict_boundaries:
            if is_datetime:
                zero_mask = delta <= pd.Timedelta(0)
                epsilon = self._DATETIME_EPSILON
            else:
                zero_mask = delta <= 0
                # use 1 for integer types, epsilon for float types
                is_integer = pd.api.types.is_integer_dtype(low) or pd.api.types.is_integer_dtype(high)
                epsilon = 1 if is_integer else self._NUMERIC_EPSILON
                # only convert to float if using epsilon (not for integer types)
                if zero_mask.any() and not is_integer and not pd.api.types.is_float_dtype(delta):
                    delta = delta.astype(float)

            if zero_mask.any():
                _LOG.warning(
                    f"correcting {zero_mask.sum()} equality violations for strict inequality {self.low_column} < {self.high_column}"
                )
                delta = delta.where(delta > (pd.Timedelta(0) if is_datetime else 0), epsilon)

        df[self._delta_column] = delta
        return df

    def to_original(self, df: pd.DataFrame, seed_data: pd.DataFrame | None = None) -> pd.DataFrame:
        df = df.copy()
        if self._delta_column in df.columns:
            # prepare delta and determine data types
            low = df[self.low_column]
            delta = df[self._delta_column]

            is_datetime = pd.api.types.is_datetime64_any_dtype(low)

            if is_datetime:
                low = pd.to_datetime(low)
                if not pd.api.types.is_timedelta64_dtype(delta):
                    delta = pd.to_timedelta(delta)
            else:
                low = pd.to_numeric(low, errors="coerce")
                delta = pd.to_numeric(delta, errors="coerce")
                # preserve original dtype for reconstruction
                original_dtype = low.dtype

            # enforce strict boundaries if needed
            if self.strict_boundaries:
                zero_mask = delta <= (pd.Timedelta(0) if is_datetime else 0)
                if zero_mask.any():
                    if is_datetime:
                        epsilon = self._DATETIME_EPSILON
                    else:
                        # use 1 for integer types, epsilon for float types
                        is_integer = pd.api.types.is_integer_dtype(low)
                        epsilon = 1 if is_integer else self._NUMERIC_EPSILON
                        # only convert to float if using epsilon (not for integer types)
                        if not is_integer and not pd.api.types.is_float_dtype(delta):
                            delta = delta.astype(float)
                    delta = delta.where(delta > (pd.Timedelta(0) if is_datetime else 0), epsilon)

            # handle seed data with row-by-row imputation support
            if seed_data is not None and len(seed_data) > 0:
                seed_cols = set(seed_data.columns)
                seed_len = len(seed_data)
                df_len = len(df)

                # align seed data length with df if needed
                if seed_len < df_len:
                    # pad seed_data with NaN rows to match df length
                    padding = pd.DataFrame(index=range(df_len - seed_len), columns=seed_data.columns)
                    seed_data = pd.concat([seed_data, padding], ignore_index=True)
                elif seed_len > df_len:
                    # truncate seed_data to match df length
                    seed_data = seed_data.iloc[:df_len].copy()

                # determine which rows have seeded values (non-null) for each column
                low_seeded_mask = None
                high_seeded_mask = None

                if self.low_column in seed_cols:
                    low_seeded_mask = seed_data[self.low_column].notna()
                    # preserve seeded low values
                    df.loc[low_seeded_mask, self.low_column] = seed_data.loc[low_seeded_mask, self.low_column].values
                    # update low variable to include seeded values
                    if is_datetime:
                        low = pd.to_datetime(df[self.low_column])
                    else:
                        low = pd.to_numeric(df[self.low_column], errors="coerce")

                if self.high_column in seed_cols:
                    high_seeded_mask = seed_data[self.high_column].notna()
                    # preserve seeded high values
                    df.loc[high_seeded_mask, self.high_column] = seed_data.loc[
                        high_seeded_mask, self.high_column
                    ].values

                # handle reconstruction based on which columns are seeded per row
                if low_seeded_mask is not None and high_seeded_mask is not None:
                    # both columns may be seeded: check row-by-row
                    # rows where both are seeded are already preserved above, no action needed
                    only_low_seeded = low_seeded_mask & ~high_seeded_mask
                    only_high_seeded = ~low_seeded_mask & high_seeded_mask
                    neither_seeded = ~low_seeded_mask & ~high_seeded_mask

                    # rows where only low is seeded: reconstruct high = low + delta
                    if only_low_seeded.any():
                        high_vals = (low + delta).loc[only_low_seeded]
                        df.loc[only_low_seeded, self.high_column] = high_vals.values

                    # rows where only high is seeded: reconstruct low = high - delta
                    if only_high_seeded.any():
                        high = df[self.high_column]
                        if is_datetime:
                            high = pd.to_datetime(high)
                        else:
                            high = pd.to_numeric(high, errors="coerce")
                        low_vals = (high - delta).loc[only_high_seeded]
                        df.loc[only_high_seeded, self.low_column] = low_vals.values

                    # rows where neither is seeded: reconstruct high = low + delta (normal case)
                    if neither_seeded.any():
                        high_vals = (low + delta).loc[neither_seeded]
                        df.loc[neither_seeded, self.high_column] = high_vals.values

                elif low_seeded_mask is not None:
                    # only low_column may be seeded: reconstruct high for all rows using low + delta
                    # (low may be seeded or generated depending on the row)
                    df[self.high_column] = (low + delta).values

                elif high_seeded_mask is not None:
                    # only high_column may be seeded: reconstruct low for rows where high is seeded
                    high = df[self.high_column]
                    if is_datetime:
                        high = pd.to_datetime(high)
                    else:
                        high = pd.to_numeric(high, errors="coerce")

                    # reconstruct low for rows where high is seeded: low = high - delta
                    if high_seeded_mask.any():
                        low_vals = (high - delta).loc[high_seeded_mask]
                        df.loc[high_seeded_mask, self.low_column] = low_vals.values

                    # reconstruct high for rows where high is not seeded: high = low + delta
                    if (~high_seeded_mask).any():
                        high_vals = (low + delta).loc[~high_seeded_mask]
                        df.loc[~high_seeded_mask, self.high_column] = high_vals.values

                # preserve original dtype for reconstructed columns
                if not is_datetime and pd.api.types.is_integer_dtype(original_dtype):
                    df[self.low_column] = df[self.low_column].astype(original_dtype)
                    df[self.high_column] = df[self.high_column].astype(original_dtype)
            else:
                # no seed data: normal reconstruction high = low + delta
                df[self.high_column] = low + delta
                # preserve original dtype if it was integer
                if not is_datetime and pd.api.types.is_integer_dtype(original_dtype):
                    df[self.high_column] = df[self.high_column].astype(original_dtype)

            df = df.drop(columns=[self._delta_column])
        return df

    def get_encoding_types(self) -> dict[str, str]:
        return {self._delta_column: "TABULAR_NUMERIC_AUTO"}


class RangeHandler(ConstraintHandler):
    """handler for Range constraints (low <= middle <= high)."""

    def __init__(self, constraint: Range):
        self.constraint = constraint
        self.low_column = constraint.low_column
        self.middle_column = constraint.middle_column
        self.high_column = constraint.high_column
        cols = [self.low_column, self.middle_column, self.high_column]
        self._delta1_column = _generate_internal_column_name("range_d1", cols)
        self._delta2_column = _generate_internal_column_name("range_d2", cols)

    def get_internal_column_names(self) -> list[str]:
        return [self._delta1_column, self._delta2_column]

    def get_original_columns(self) -> list[str]:
        return [self.low_column, self.middle_column, self.high_column]

    def to_internal(self, df: pd.DataFrame) -> pd.DataFrame:
        self._validate_columns(df, [self.low_column, self.middle_column, self.high_column])
        df = df.copy()
        low = df[self.low_column]
        middle = df[self.middle_column]
        high = df[self.high_column]

        if pd.api.types.is_datetime64_any_dtype(low):
            low = pd.to_datetime(low)
            middle = pd.to_datetime(middle)
            high = pd.to_datetime(high)
            delta1 = middle - low
            delta2 = high - middle
            zero = pd.Timedelta(0)
        else:
            low = pd.to_numeric(low, errors="coerce")
            middle = pd.to_numeric(middle, errors="coerce")
            high = pd.to_numeric(high, errors="coerce")
            delta1 = middle - low
            delta2 = high - middle
            zero = 0

        violations1 = delta1 < zero
        violations2 = delta2 < zero
        if violations1.any() or violations2.any():
            total = (violations1 | violations2).sum()
            _LOG.warning(
                f"correcting {total} range violations for {self.low_column} <= {self.middle_column} <= {self.high_column}"
            )
            delta1 = delta1.abs()
            delta2 = delta2.abs()

        df[self._delta1_column] = delta1
        df[self._delta2_column] = delta2
        return df

    def to_original(self, df: pd.DataFrame, seed_data: pd.DataFrame | None = None) -> pd.DataFrame:
        df = df.copy()
        if self._delta1_column in df.columns and self._delta2_column in df.columns:
            # check if constraint columns were seeded
            if seed_data is not None and len(seed_data) == len(df):
                seed_cols = set(seed_data.columns)
                # if all three columns were seeded, preserve them and skip reconstruction
                if all(col in seed_cols for col in [self.low_column, self.middle_column, self.high_column]):
                    df[self.low_column] = seed_data[self.low_column].values
                    df[self.middle_column] = seed_data[self.middle_column].values
                    df[self.high_column] = seed_data[self.high_column].values
                    df = df.drop(columns=[self._delta1_column, self._delta2_column])
                    return df
                # if individual columns were seeded, preserve them
                for col in [self.low_column, self.middle_column, self.high_column]:
                    if col in seed_cols:
                        df[col] = seed_data[col].values

            low = df[self.low_column]
            delta1 = df[self._delta1_column]
            delta2 = df[self._delta2_column]

            if pd.api.types.is_datetime64_any_dtype(low):
                low = pd.to_datetime(low)
                if not pd.api.types.is_timedelta64_dtype(delta1):
                    delta1 = pd.to_timedelta(delta1)
                if not pd.api.types.is_timedelta64_dtype(delta2):
                    delta2 = pd.to_timedelta(delta2)
            else:
                low = pd.to_numeric(low, errors="coerce")
                delta1 = pd.to_numeric(delta1, errors="coerce")
                delta2 = pd.to_numeric(delta2, errors="coerce")

            # only reconstruct columns that weren't seeded
            if seed_data is None or self.middle_column not in seed_data.columns:
                df[self.middle_column] = low + delta1
            if seed_data is None or self.high_column not in seed_data.columns:
                df[self.high_column] = low + delta1 + delta2
            df = df.drop(columns=[self._delta1_column, self._delta2_column])
        return df

    def get_encoding_types(self) -> dict[str, str]:
        return {
            self._delta1_column: "TABULAR_NUMERIC_AUTO",
            self._delta2_column: "TABULAR_NUMERIC_AUTO",
        }


class OneHotEncodingHandler(ConstraintHandler):
    """handler for OneHotEncoding constraints (exactly one column has value 1)."""

    def __init__(self, constraint: OneHotEncoding):
        self.constraint = constraint
        self.columns = constraint.columns
        self._internal_column = _generate_internal_column_name("onehot", self.columns)

    def get_internal_column_names(self) -> list[str]:
        return [self._internal_column]

    def get_original_columns(self) -> list[str]:
        return list(self.columns)

    def to_internal(self, df: pd.DataFrame) -> pd.DataFrame:
        self._validate_columns(df, self.columns)
        df = df.copy()

        def find_active_column(row):
            for col in self.columns:
                val = row[col]
                if pd.notna(val) and val == 1:
                    return col
            return None

        df[self._internal_column] = df.apply(find_active_column, axis=1)
        return df

    def to_original(self, df: pd.DataFrame, seed_data: pd.DataFrame | None = None) -> pd.DataFrame:
        df = df.copy()
        if self._internal_column in df.columns:
            # check if all columns in this one-hot were seeded
            if seed_data is not None and len(seed_data) == len(df):
                seed_cols = set(seed_data.columns)
                if all(col in seed_cols for col in self.columns):
                    # all columns were seeded - preserve them from seed_data
                    for col in self.columns:
                        df[col] = seed_data[col].values
                    df = df.drop(columns=[self._internal_column])
                    return df

            for col in self.columns:
                # if this column was seeded, preserve seed value
                if seed_data is not None and col in seed_data.columns and len(seed_data) == len(df):
                    df[col] = seed_data[col].values
                else:
                    df[col] = (df[self._internal_column] == col).astype(int)
            # handle null/unknown values by setting all to 0
            null_mask = df[self._internal_column].isna()
            for col in self.columns:
                df.loc[null_mask, col] = 0
            df = df.drop(columns=[self._internal_column])
        return df

    def get_encoding_types(self) -> dict[str, str]:
        return {self._internal_column: "TABULAR_CATEGORICAL"}


def _create_constraint_handler(constraint: FixedCombination | Inequality | Range | OneHotEncoding) -> ConstraintHandler:
    """factory function to create appropriate handler for a constraint."""
    if isinstance(constraint, FixedCombination):
        return FixedCombinationHandler(constraint)
    elif isinstance(constraint, Inequality):
        return InequalityHandler(constraint)
    elif isinstance(constraint, Range):
        return RangeHandler(constraint)
    elif isinstance(constraint, OneHotEncoding):
        return OneHotEncodingHandler(constraint)
    else:
        raise ValueError(f"unknown constraint type: {type(constraint)}")


class ConstraintTranslator:
    """translates data between user schema and internal schema for constraints."""

    def __init__(self, constraints: list[FixedCombination | Inequality | Range | OneHotEncoding]):
        self.constraints = constraints
        self.handlers = [_create_constraint_handler(c) for c in constraints]

    def to_internal(self, df: pd.DataFrame) -> pd.DataFrame:
        """transform dataframe from user schema to internal schema."""
        for handler in self.handlers:
            df = handler.to_internal(df)
        return df

    def to_original(self, df: pd.DataFrame, seed_data: pd.DataFrame | None = None) -> pd.DataFrame:
        """transform dataframe from internal schema back to user schema."""
        for handler in self.handlers:
            df = handler.to_original(df, seed_data=seed_data)
        return df

    def get_internal_columns(self, original_columns: list[str]) -> list[str]:
        """get list of column names in internal schema."""
        columns_to_remove = set()
        columns_to_add = []

        for handler in self.handlers:
            # compute columns to remove based on handler type
            if isinstance(handler, InequalityHandler):
                columns_to_remove.add(handler.high_column)
            elif isinstance(handler, RangeHandler):
                columns_to_remove.add(handler.middle_column)
                columns_to_remove.add(handler.high_column)
            elif isinstance(handler, OneHotEncodingHandler):
                columns_to_remove.update(handler.columns)
            # FixedCombinationHandler keeps all original columns, so nothing to remove
            columns_to_add.extend(handler.get_internal_column_names())

        internal_columns = [c for c in original_columns if c not in columns_to_remove]
        internal_columns.extend(columns_to_add)
        return internal_columns

    def get_original_columns(self, internal_columns: list[str]) -> list[str]:
        """get list of column names in original schema from internal schema."""
        internal_to_original = {}
        for handler in self.handlers:
            for internal_col in handler.get_internal_column_names():
                internal_to_original[internal_col] = handler.get_original_columns()

        original_columns = []
        seen = set()
        for col in internal_columns:
            if col in internal_to_original:
                for orig_col in internal_to_original[col]:
                    if orig_col not in seen:
                        original_columns.append(orig_col)
                        seen.add(orig_col)
            else:
                if col not in seen:
                    original_columns.append(col)
                    seen.add(col)
        return original_columns

    def get_encoding_types(self) -> dict[str, str]:
        """get combined encoding types for all internal columns."""
        encoding_types = {}
        for handler in self.handlers:
            encoding_types.update(handler.get_encoding_types())
        return encoding_types

    def get_columns_to_remove(self) -> set[str]:
        """get all columns that should be removed from encoding types."""
        columns = set()
        for handler in self.handlers:
            # compute columns to remove based on handler type
            if isinstance(handler, InequalityHandler):
                columns.add(handler.high_column)
            elif isinstance(handler, RangeHandler):
                columns.add(handler.middle_column)
                columns.add(handler.high_column)
            elif isinstance(handler, OneHotEncodingHandler):
                columns.update(handler.columns)
            # FixedCombinationHandler keeps all original columns, so nothing to remove
        return columns

    @property
    def merged_columns(self) -> list[tuple[list[str], str]]:
        """backward compatibility: get list of (original_columns, merged_name) for FixedCombination handlers."""
        result = []
        for handler in self.handlers:
            if isinstance(handler, FixedCombinationHandler):
                result.append((list(handler.columns), handler.merged_name))
        return result

    def get_all_internal_column_names(self) -> list[str]:
        """get all internal column names from all handlers."""
        result = []
        for handler in self.handlers:
            result.extend(handler.get_internal_column_names())
        return result

    @staticmethod
    def from_generator_config(
        generator: Generator,
        table_name: str,
    ) -> tuple[ConstraintTranslator | None, list[str] | None]:
        """create constraint translator from generator configuration."""
        table = next((t for t in generator.tables if t.name == table_name), None)
        if not table:
            return None, None

        constraints = None
        if table.tabular_model_configuration and table.tabular_model_configuration.constraints:
            constraints = table.tabular_model_configuration.constraints
        if not constraints and table.language_model_configuration:
            if table.language_model_configuration.constraints:
                constraints = table.language_model_configuration.constraints

        if not constraints:
            return None, None

        translator = ConstraintTranslator(constraints)
        current_columns = [c.name for c in table.columns] if table.columns else None

        if not current_columns:
            return translator, None

        original_columns = translator.get_original_columns(current_columns)
        return translator, original_columns


def _align_seed_data(
    df: pd.DataFrame, seed_data: pd.DataFrame, columns: list[str], context_key_col: str | None = None
) -> pd.DataFrame | None:
    """align seed data with dataframe for constraint column preservation.

    handles both subject tables (1:1 row alignment) and sequential tables
    (alignment by context key + row index).

    Args:
        df: generated dataframe to align with
        seed_data: seed dataframe (may contain __row_idx__ and context key)
        columns: list of columns to extract from seed_data
        context_key_col: context key column name for sequential tables

    Returns:
        aligned dataframe with requested columns, or None if alignment fails
    """
    if seed_data is None:
        return None

    # check if this is a sequential table (has __row_idx__)
    if "__row_idx__" in seed_data.columns and context_key_col is not None:
        # sequential table: align by context key + row index
        if context_key_col not in df.columns:
            _LOG.warning(f"context key column {context_key_col} not found in dataframe, skipping seed alignment")
            return None

        # add row index to df for alignment
        df_with_idx = df.copy()
        df_with_idx["__row_idx__"] = df_with_idx.groupby(context_key_col).cumcount()

        # merge seed data
        aligned = pd.merge(
            df_with_idx[[context_key_col, "__row_idx__"]],
            seed_data,
            on=[context_key_col, "__row_idx__"],
            how="left",
        )

        # extract requested columns if they exist
        available_cols = [col for col in columns if col in aligned.columns]
        if not available_cols:
            return None

        return aligned[available_cols].reset_index(drop=True)

    # subject table: 1:1 row alignment (allow partial seed - fewer rows than df)
    if len(seed_data) > len(df):
        _LOG.warning(
            f"seed data length ({len(seed_data)}) is greater than dataframe length ({len(df)}), skipping seed alignment"
        )
        return None

    # extract requested columns if they exist in seed_data
    available_cols = [col for col in columns if col in seed_data.columns]
    if not available_cols:
        return None

    # for subject tables, align first N rows where N is seed size
    # if seed has fewer rows, we'll align the first N rows
    aligned = seed_data[available_cols].copy()
    if len(aligned) < len(df):
        # pad with NaN rows to match df length (handlers will only use first N rows)
        padding = pd.DataFrame(index=range(len(df) - len(aligned)), columns=available_cols)
        aligned = pd.concat([aligned, padding], ignore_index=True)

    return aligned.reset_index(drop=True)


def preprocess_constraints_for_training(
    *,
    generator: Generator,
    workspace_dir: Path,
    model_type: ModelType,
    target_table_name: str,
) -> list[str] | None:
    """preprocess constraint transformations for training data."""
    target_table = next((t for t in generator.tables if t.name == target_table_name), None)
    if not target_table:
        _LOG.info(f"table {target_table_name} not found in generator")
        return None

    if model_type.value == "language":
        model_config = target_table.language_model_configuration
    else:
        model_config = target_table.tabular_model_configuration

    if not model_config or not model_config.constraints:
        return None

    _LOG.info(f"preprocessing constraints for table {target_table_name} in {model_type} model")
    translator = ConstraintTranslator(model_config.constraints)

    tgt_data_dir = workspace_dir / "OriginalData" / "tgt-data"
    if not tgt_data_dir.exists():
        _LOG.warning(f"data directory not found: {tgt_data_dir}")
        return None

    parquet_files = sorted(list(tgt_data_dir.glob("part.*.parquet")))
    for parquet_file in parquet_files:
        df = pd.read_parquet(parquet_file)
        df_transformed = translator.to_internal(df)
        df_transformed.to_parquet(parquet_file, index=True)

    original_columns = [c.name for c in target_table.columns] if target_table.columns else []
    _update_meta_with_internal_columns(workspace_dir, target_table_name, translator, parquet_files)
    internal_columns = translator.get_internal_columns(original_columns)
    return internal_columns


def _update_meta_with_internal_columns(
    workspace_dir: Path,
    table_name: str,
    translator: ConstraintTranslator,
    parquet_files: list[Path],
) -> None:
    """update tgt-meta to reflect internal column structure after transformation."""
    if not parquet_files:
        return

    meta_dir = get_tgt_meta_path(workspace_dir)
    encoding_types_file = meta_dir / "encoding-types.json"

    if encoding_types_file.exists():
        with open(encoding_types_file) as f:
            encoding_types = json.load(f)

        encoding_types.update(translator.get_encoding_types())
        for col in translator.get_columns_to_remove():
            encoding_types.pop(col, None)

        with open(encoding_types_file, "w") as f:
            json.dump(encoding_types, f, indent=2)

        _LOG.info(f"updated encoding-types.json with internal columns for {table_name}")
