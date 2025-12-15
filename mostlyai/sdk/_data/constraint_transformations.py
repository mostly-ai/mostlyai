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

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from mostlyai.sdk.domain import FixedCombination, Generator, Inequality, ModelType, OneHotEncoding, Range

_LOG = logging.getLogger(__name__)

# type alias for constraint types
ConstraintType = FixedCombination | Inequality | Range | OneHotEncoding


def _generate_internal_column_name(prefix: str, columns: list[str], model_type: ModelType = ModelType.tabular) -> str:
    """generate a deterministic internal column name.

    use model type prefix to avoid pydantic validation errors with language models.
    """
    key = "|".join(columns)
    hash_suffix = hashlib.md5(key.encode()).hexdigest()[:8]
    # use model type prefix instead of underscores (pydantic doesn't allow leading underscores)
    model_prefix = "language" if model_type == ModelType.language else "tabular"
    return f"{model_prefix}_constraint_{prefix}_{hash_suffix}"


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

    @abstractmethod
    def get_table_column_tuples(self) -> list[tuple[str, str]]:
        """return list of (table_name, column_name) tuples involved in this constraint."""
        pass

    @abstractmethod
    def get_columns_to_remove(self) -> set[str]:
        """columns excluded from training (reconstructed from internal representation)."""
        pass

    def _validate_columns(self, df: pd.DataFrame, columns: list[str]) -> None:
        """validate that all required columns exist in the dataframe."""
        missing_cols = set(columns) - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"Columns {sorted(missing_cols)} required by {self.__class__.__name__} "
                f"not found in dataframe. Available columns: {sorted(df.columns)}"
            )

    def _align_seed_data_simple(self, df: pd.DataFrame, seed_data: pd.DataFrame | None) -> pd.DataFrame | None:
        """align seed data to match df length (1:1 row alignment).

        handles length mismatches by padding with NaN or truncating.

        Args:
            df: target dataframe to align with
            seed_data: seed dataframe (may have different length)

        Returns:
            aligned seed dataframe, or None if seed_data is empty/None
        """
        if seed_data is None or len(seed_data) == 0:
            return None
        seed_len, df_len = len(seed_data), len(df)
        if seed_len < df_len:
            # pad with NaN rows to match df length
            padding = pd.DataFrame(index=range(df_len - seed_len), columns=seed_data.columns)
            return pd.concat([seed_data, padding], ignore_index=True)
        elif seed_len > df_len:
            # truncate to match df length
            return seed_data.iloc[:df_len].reset_index(drop=True)
        return seed_data.reset_index(drop=True)

    def _is_datetime_column(self, series: pd.Series) -> bool:
        """check if series is datetime type."""
        return pd.api.types.is_datetime64_any_dtype(series)

    def _normalize_numeric(self, series: pd.Series) -> pd.Series:
        """convert series to numeric, coercing errors."""
        return pd.to_numeric(series, errors="coerce")

    def _normalize_datetime(self, series: pd.Series) -> pd.Series:
        """convert series to datetime."""
        return pd.to_datetime(series)

    def _ensure_delta_type(self, delta: pd.Series, is_datetime: bool) -> pd.Series:
        """ensure delta has correct type (timedelta or numeric)."""
        if is_datetime:
            if not pd.api.types.is_timedelta64_dtype(delta):
                return pd.to_timedelta(delta)
            return delta
        return pd.to_numeric(delta, errors="coerce")

    def validate_against_generator(self, generator: Generator) -> None:
        """validate that tables and columns referenced by this constraint exist in the generator.

        Args:
            generator: Generator to validate against.

        Raises:
            ValueError: If table or columns don't exist.
        """
        table_column_tuples = self.get_table_column_tuples()
        if not table_column_tuples:
            return

        # build table map
        table_map = {table.name: table for table in (generator.tables or [])}

        # validate each (table, column) tuple
        for table_name, column_name in table_column_tuples:
            if table_name not in table_map:
                raise ValueError(f"table '{table_name}' referenced by constraint not found in generator")

            table = table_map[table_name]
            table_columns = {col.name for col in (table.columns or []) if col.included}

            if column_name not in table_columns:
                raise ValueError(
                    f"column '{column_name}' in table '{table_name}' referenced by constraint not found or not included"
                )


class FixedCombinationHandler(ConstraintHandler):
    """handler for FixedCombination constraints."""

    def __init__(self, constraint: FixedCombination, model_type: ModelType = ModelType.tabular):
        self.constraint = constraint
        self.table_name = constraint.table_name
        self.columns = constraint.columns
        self.model_type = model_type
        self.merged_name = _generate_internal_column_name("fixedcomb", self.columns, model_type)

    def get_internal_column_names(self) -> list[str]:
        return [self.merged_name]

    def get_original_columns(self) -> list[str]:
        return list(self.columns)

    def to_internal(self, df: pd.DataFrame) -> pd.DataFrame:
        self._validate_columns(df, self.columns)
        df = df.copy()

        def merge_row(row):
            values = [row[col] if pd.notna(row[col]) else None for col in self.columns]
            # JSON serialization handles all escaping automatically
            return json.dumps(values, ensure_ascii=False)

        df[self.merged_name] = df.apply(merge_row, axis=1)
        return df

    def to_original(self, df: pd.DataFrame, seed_data: pd.DataFrame | None = None) -> pd.DataFrame:
        df = df.copy()
        if self.merged_name in df.columns:
            aligned_seed = self._align_seed_data_simple(df, seed_data)

            # check if all columns in this combination were seeded
            if aligned_seed is not None:
                seed_cols = set(aligned_seed.columns)
                if all(col in seed_cols for col in self.columns):
                    # all columns were seeded - preserve them from seed_data
                    for col in self.columns:
                        # use mask to only assign non-null seed values
                        seed_mask = aligned_seed[col].notna()
                        df.loc[seed_mask, col] = aligned_seed.loc[seed_mask, col].values
                    df = df.drop(columns=[self.merged_name])
                    return df

            def split_row(merged_value: str) -> list[str]:
                if pd.isna(merged_value):
                    return [""] * len(self.columns)
                try:
                    values = json.loads(merged_value)
                    return [str(v) if v is not None else "" for v in values]
                except json.JSONDecodeError:
                    return [""] * len(self.columns)

            # handle empty dataframe case
            if len(df) == 0:
                for col in self.columns:
                    df[col] = pd.Series([], dtype=str)
                df = df.drop(columns=[self.merged_name])
                return df

            split_values = df[self.merged_name].astype(str).apply(split_row)
            split_df = pd.DataFrame(split_values.tolist(), index=df.index)

            # reset df index to align with split_df and seed data
            df = df.reset_index(drop=True)
            split_df = split_df.reset_index(drop=True)

            # assign to original columns
            for i, col in enumerate(self.columns):
                # first assign split values
                df[col] = split_df[i].values
                # then override with seed values where available
                if aligned_seed is not None and col in aligned_seed.columns:
                    seed_mask = aligned_seed[col].notna()
                    if seed_mask.any():
                        df.loc[seed_mask, col] = aligned_seed.loc[seed_mask, col].values

            # drop the merged column
            df = df.drop(columns=[self.merged_name])
        return df

    def get_encoding_types(self) -> dict[str, str]:
        # always use TABULAR encoding for constraints, regardless of model_type
        # constraints merge columns which requires categorical encoding
        return {self.merged_name: "TABULAR_CATEGORICAL"}

    def get_table_column_tuples(self) -> list[tuple[str, str]]:
        """return list of (table_name, column_name) tuples involved in this constraint."""
        return [(self.table_name, col) for col in self.columns]

    def get_columns_to_remove(self) -> set[str]:
        return set()  # keeps original columns alongside merged column


class InequalityHandler(ConstraintHandler):
    """handler for Inequality constraints (low <= high or low < high if strict_boundaries=True)."""

    _NUMERIC_EPSILON = 1e-8  # conservative practical float64 difference (1 for integers)
    _INTEGER_EPSILON = 1  # Epsilon for integer types
    _DATETIME_EPSILON = pd.Timedelta(microseconds=1)  # smallest reliable timedelta

    def __init__(self, constraint: Inequality, model_type: ModelType = ModelType.tabular):
        self.constraint = constraint
        self.table_name = constraint.table_name
        self.low_column = constraint.low_column
        self.high_column = constraint.high_column
        self.strict_boundaries = constraint.strict_boundaries
        self.model_type = model_type
        self._delta_column = _generate_internal_column_name(
            "ineq_delta", [self.low_column, self.high_column], model_type
        )

    def _enforce_strict_delta(
        self, delta: pd.Series, is_datetime: bool, is_integer: bool, log_warning: bool = True
    ) -> pd.Series:
        """enforce strict boundaries by ensuring delta > 0.

        Args:
            delta: delta series to enforce
            is_datetime: whether the data is datetime type
            is_integer: whether the data is integer type (for numeric)
            log_warning: whether to log a warning when correcting violations

        Returns:
            corrected delta series with all values > 0
        """
        zero = pd.Timedelta(0) if is_datetime else 0
        zero_mask = delta <= zero
        if not zero_mask.any():
            return delta

        if is_datetime:
            epsilon = self._DATETIME_EPSILON
        elif is_integer:
            epsilon = self._INTEGER_EPSILON
        else:
            epsilon = self._NUMERIC_EPSILON
            if not pd.api.types.is_float_dtype(delta):
                delta = delta.astype(float)

        if log_warning:
            _LOG.warning(
                f"correcting {zero_mask.sum()} equality violations for strict inequality "
                f"{self.low_column} < {self.high_column}"
            )
        return delta.where(delta > zero, epsilon)

    def get_internal_column_names(self) -> list[str]:
        return [self._delta_column]

    def get_original_columns(self) -> list[str]:
        return [self.low_column, self.high_column]

    def to_internal(self, df: pd.DataFrame) -> pd.DataFrame:
        self._validate_columns(df, [self.low_column, self.high_column])
        df = df.copy()
        low = df[self.low_column]
        high = df[self.high_column]

        is_datetime = self._is_datetime_column(low) or self._is_datetime_column(high)

        if is_datetime:
            low = self._normalize_datetime(low)
            high = self._normalize_datetime(high)
            delta = high - low
            zero = pd.Timedelta(0)
        else:
            low = self._normalize_numeric(low)
            high = self._normalize_numeric(high)
            delta = high - low
            zero = 0

        violations = delta < zero
        if violations.any():
            _LOG.warning(
                f"correcting {violations.sum()} inequality violations for {self.low_column} <= {self.high_column}"
            )
            delta = delta.abs()

        # enforce strict boundaries if needed
        if self.strict_boundaries:
            is_integer = pd.api.types.is_integer_dtype(low) or pd.api.types.is_integer_dtype(high)
            delta = self._enforce_strict_delta(delta, is_datetime, is_integer)

        df[self._delta_column] = delta
        return df

    def to_original(self, df: pd.DataFrame, seed_data: pd.DataFrame | None = None) -> pd.DataFrame:
        """transform from internal schema back to original schema."""
        df = df.copy()
        if self._delta_column not in df.columns:
            return df

        # check if low_column exists (should always exist, but be defensive)
        if self.low_column not in df.columns:
            _LOG.warning(f"low_column '{self.low_column}' not found in dataframe, skipping constraint transformation")
            return df

        # prepare data and types
        is_datetime = self._is_datetime_column(df[self.low_column])
        if is_datetime:
            low = self._normalize_datetime(df[self.low_column])
            delta = self._ensure_delta_type(df[self._delta_column], is_datetime=True)
            original_dtype = None
        else:
            low = self._normalize_numeric(df[self.low_column])
            delta = self._ensure_delta_type(df[self._delta_column], is_datetime=False)
            original_dtype = low.dtype

        # enforce strict boundaries if needed
        if self.strict_boundaries:
            delta = self._enforce_strict_delta(
                delta, is_datetime, pd.api.types.is_integer_dtype(low), log_warning=False
            )

        # align seed data
        aligned = self._align_seed_data_simple(df, seed_data)
        low_mask = (
            aligned[self.low_column].notna() if aligned is not None and self.low_column in aligned.columns else None
        )
        high_mask = (
            aligned[self.high_column].notna() if aligned is not None and self.high_column in aligned.columns else None
        )

        # default reconstruction: high = low + delta
        df[self.high_column] = low + delta

        # apply seed overrides
        if aligned is not None:
            if low_mask is not None and low_mask.any():
                target_dtype = df[self.low_column].dtype
                values = aligned.loc[low_mask, self.low_column].values
                df.loc[low_mask, self.low_column] = pd.Series(values, dtype=target_dtype).values
                low = (
                    self._normalize_datetime(df[self.low_column])
                    if is_datetime
                    else self._normalize_numeric(df[self.low_column])
                )
                target_dtype = df[self.high_column].dtype
                values = (low + delta).loc[low_mask].values
                df.loc[low_mask, self.high_column] = pd.Series(values, dtype=target_dtype).values
            if high_mask is not None and high_mask.any():
                target_dtype = df[self.high_column].dtype
                values = aligned.loc[high_mask, self.high_column].values
                df.loc[high_mask, self.high_column] = pd.Series(values, dtype=target_dtype).values
                # reconstruct low_column from high_column when only high is seeded
                only_high = high_mask & ~(low_mask if low_mask is not None else pd.Series([False] * len(df)))
                if only_high.any():
                    high = (
                        self._normalize_datetime(df[self.high_column])
                        if is_datetime
                        else self._normalize_numeric(df[self.high_column])
                    )
                    target_dtype = df[self.low_column].dtype
                    values = (high - delta).loc[only_high].values
                    df.loc[only_high, self.low_column] = pd.Series(values, dtype=target_dtype).values

        # preserve dtype
        if not is_datetime and original_dtype and pd.api.types.is_integer_dtype(original_dtype):
            df[self.low_column] = df[self.low_column].astype(original_dtype)
            df[self.high_column] = df[self.high_column].astype(original_dtype)

        return df.drop(columns=[self._delta_column])

    def get_encoding_types(self) -> dict[str, str]:
        # always use TABULAR encoding for constraints, regardless of model_type
        return {self._delta_column: "TABULAR_NUMERIC_AUTO"}

    def get_table_column_tuples(self) -> list[tuple[str, str]]:
        """return list of (table_name, column_name) tuples involved in this constraint."""
        return [(self.table_name, self.low_column), (self.table_name, self.high_column)]

    def get_columns_to_remove(self) -> set[str]:
        return {self.high_column}


class RangeHandler(ConstraintHandler):
    """handler for Range constraints (low <= middle <= high)."""

    def __init__(self, constraint: Range, model_type: ModelType = ModelType.tabular):
        self.constraint = constraint
        self.table_name = constraint.table_name
        self.low_column = constraint.low_column
        self.middle_column = constraint.middle_column
        self.high_column = constraint.high_column
        self.model_type = model_type
        cols = [self.low_column, self.middle_column, self.high_column]
        self._delta1_column = _generate_internal_column_name("range_d1", cols, model_type)
        self._delta2_column = _generate_internal_column_name("range_d2", cols, model_type)

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

        is_datetime = self._is_datetime_column(low)
        if is_datetime:
            low = self._normalize_datetime(low)
            middle = self._normalize_datetime(middle)
            high = self._normalize_datetime(high)
            zero = pd.Timedelta(0)
        else:
            low = self._normalize_numeric(low)
            middle = self._normalize_numeric(middle)
            high = self._normalize_numeric(high)
            zero = 0

        delta1 = middle - low
        delta2 = high - middle

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
        if self._delta1_column not in df.columns or self._delta2_column not in df.columns:
            return df

        # prepare data and types
        is_datetime = self._is_datetime_column(df[self.low_column])
        low = (
            self._normalize_datetime(df[self.low_column])
            if is_datetime
            else self._normalize_numeric(df[self.low_column])
        )
        delta1 = self._ensure_delta_type(df[self._delta1_column], is_datetime)
        delta2 = self._ensure_delta_type(df[self._delta2_column], is_datetime)

        # align seed data and get masks
        aligned = self._align_seed_data_simple(df, seed_data)
        low_mask = (
            aligned[self.low_column].notna() if aligned is not None and self.low_column in aligned.columns else None
        )
        mid_mask = (
            aligned[self.middle_column].notna()
            if aligned is not None and self.middle_column in aligned.columns
            else None
        )
        high_mask = (
            aligned[self.high_column].notna() if aligned is not None and self.high_column in aligned.columns else None
        )

        # default reconstruction
        df[self.middle_column] = low + delta1
        df[self.high_column] = low + delta1 + delta2

        # apply seed overrides
        if aligned is not None:
            if low_mask is not None and low_mask.any():
                target_dtype = df[self.low_column].dtype
                values = aligned.loc[low_mask, self.low_column].values
                df.loc[low_mask, self.low_column] = pd.Series(values, dtype=target_dtype).values
                low = (
                    self._normalize_datetime(df[self.low_column])
                    if is_datetime
                    else self._normalize_numeric(df[self.low_column])
                )
                # recompute for seeded low rows
                target_dtype = df[self.middle_column].dtype
                values = (low + delta1).loc[low_mask].values
                df.loc[low_mask, self.middle_column] = pd.Series(values, dtype=target_dtype).values
                target_dtype = df[self.high_column].dtype
                values = (low + delta1 + delta2).loc[low_mask].values
                df.loc[low_mask, self.high_column] = pd.Series(values, dtype=target_dtype).values
            if mid_mask is not None and mid_mask.any():
                target_dtype = df[self.middle_column].dtype
                values = aligned.loc[mid_mask, self.middle_column].values
                df.loc[mid_mask, self.middle_column] = pd.Series(values, dtype=target_dtype).values
            if high_mask is not None and high_mask.any():
                target_dtype = df[self.high_column].dtype
                values = aligned.loc[high_mask, self.high_column].values
                df.loc[high_mask, self.high_column] = pd.Series(values, dtype=target_dtype).values

        return df.drop(columns=[self._delta1_column, self._delta2_column])

    def get_encoding_types(self) -> dict[str, str]:
        # always use TABULAR encoding for constraints, regardless of model_type
        return {
            self._delta1_column: "TABULAR_NUMERIC_AUTO",
            self._delta2_column: "TABULAR_NUMERIC_AUTO",
        }

    def get_table_column_tuples(self) -> list[tuple[str, str]]:
        """return list of (table_name, column_name) tuples involved in this constraint."""
        return [
            (self.table_name, self.low_column),
            (self.table_name, self.middle_column),
            (self.table_name, self.high_column),
        ]

    def get_columns_to_remove(self) -> set[str]:
        return {self.middle_column, self.high_column}


class OneHotEncodingHandler(ConstraintHandler):
    """handler for OneHotEncoding constraints (exactly one column has value 1)."""

    def __init__(self, constraint: OneHotEncoding, model_type: ModelType = ModelType.tabular):
        self.constraint = constraint
        self.table_name = constraint.table_name
        self.columns = constraint.columns
        self.model_type = model_type
        self._internal_column = _generate_internal_column_name("onehot", self.columns, model_type)

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
        # always use TABULAR encoding for constraints, regardless of model_type
        return {self._internal_column: "TABULAR_CATEGORICAL"}

    def get_table_column_tuples(self) -> list[tuple[str, str]]:
        """return list of (table_name, column_name) tuples involved in this constraint."""
        return [(self.table_name, col) for col in self.columns]

    def get_columns_to_remove(self) -> set[str]:
        return set(self.columns)


def _create_constraint_handler(
    constraint: ConstraintType, model_type: ModelType = ModelType.tabular
) -> ConstraintHandler:
    """factory function to create appropriate handler for a constraint."""
    if isinstance(constraint, FixedCombination):
        return FixedCombinationHandler(constraint, model_type)
    elif isinstance(constraint, Inequality):
        return InequalityHandler(constraint, model_type)
    elif isinstance(constraint, Range):
        return RangeHandler(constraint, model_type)
    elif isinstance(constraint, OneHotEncoding):
        return OneHotEncodingHandler(constraint, model_type)
    else:
        raise ValueError(f"unknown constraint type: {type(constraint)}")


class ConstraintTranslator:
    """translates data between user schema and internal schema for constraints."""

    def __init__(self, constraints: list[ConstraintType], model_type: ModelType = ModelType.tabular):
        self.constraints = constraints
        self.model_type = model_type
        self.handlers = [_create_constraint_handler(c, model_type) for c in constraints]

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

    def _compute_columns_to_remove(self) -> set[str]:
        """compute columns that should be removed when transforming to internal schema."""
        return set().union(*(h.get_columns_to_remove() for h in self.handlers))

    def get_internal_columns(self, original_columns: list[str]) -> list[str]:
        """get list of column names in internal schema."""
        columns_to_remove = self._compute_columns_to_remove()
        internal_columns = [c for c in original_columns if c not in columns_to_remove]
        for handler in self.handlers:
            internal_columns.extend(handler.get_internal_column_names())
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
        return self._compute_columns_to_remove()

    def get_all_internal_column_names(self) -> list[str]:
        """get all internal column names from all handlers."""
        result = []
        for handler in self.handlers:
            result.extend(handler.get_internal_column_names())
        return result

    def align_seed_data(
        self, df: pd.DataFrame, seed_data: pd.DataFrame | None, context_key_col: str | None = None
    ) -> pd.DataFrame | None:
        """align seed data for all constraint columns.

        Args:
            df: generated dataframe to align with
            seed_data: seed dataframe (may contain __row_idx__ and context key)
            context_key_col: context key column name for sequential tables

        Returns:
            aligned dataframe with constraint columns, or None if alignment fails
        """
        if seed_data is None or len(seed_data) == 0:
            return None

        columns = list(set().union(*(set(h.get_original_columns()) for h in self.handlers)))
        if "__row_idx__" in seed_data.columns and context_key_col:
            return self._align_sequential(df, seed_data, columns, context_key_col)
        return self._align_subject(df, seed_data, columns)

    def _align_sequential(
        self, df: pd.DataFrame, seed_data: pd.DataFrame, columns: list[str], context_key_col: str
    ) -> pd.DataFrame | None:
        """align seed data for sequential tables using context key + row index."""
        if context_key_col not in df.columns:
            _LOG.warning(f"context key column {context_key_col} not found, skipping seed alignment")
            return None

        df_with_idx = df[[context_key_col]].copy()
        df_with_idx["__row_idx__"] = df.groupby(context_key_col).cumcount()

        aligned = pd.merge(df_with_idx, seed_data, on=[context_key_col, "__row_idx__"], how="left")
        available_cols = [c for c in columns if c in aligned.columns]
        return aligned[available_cols].reset_index(drop=True) if available_cols else None

    def _align_subject(self, df: pd.DataFrame, seed_data: pd.DataFrame, columns: list[str]) -> pd.DataFrame | None:
        """align seed data for subject tables (1:1 row alignment with padding/truncation)."""
        available_cols = [c for c in columns if c in seed_data.columns]
        if not available_cols:
            return None

        seed_len, df_len = len(seed_data), len(df)
        if seed_len > df_len:
            return seed_data[available_cols].iloc[:df_len].reset_index(drop=True)
        elif seed_len < df_len:
            padding = pd.DataFrame(index=range(df_len - seed_len), columns=available_cols)
            return pd.concat([seed_data[available_cols], padding], ignore_index=True)
        return seed_data[available_cols].reset_index(drop=True)

    @staticmethod
    def from_generator_config(
        generator: Generator,
        table_name: str,
    ) -> tuple[ConstraintTranslator | None, list[str] | None]:
        """create constraint translator from generator configuration.

        reads constraints from generator root level and filters by table_name.
        """
        if not generator.constraints:
            return None, None

        table_constraints = [c for c in generator.constraints if c.table_name == table_name]

        if not table_constraints:
            return None, None

        table = next((t for t in generator.tables if t.name == table_name), None)
        if not table:
            return None, None

        model_type = ModelType.language if table.language_model_configuration else ModelType.tabular

        translator = ConstraintTranslator(table_constraints, model_type)
        current_columns = [c.name for c in table.columns] if table.columns else None

        if not current_columns:
            return translator, None

        original_columns = translator.get_original_columns(current_columns)
        return translator, original_columns


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
        _LOG.debug(f"table {target_table_name} not found in generator")
        return None

    if not generator.constraints:
        return None

    table_constraints = [c for c in generator.constraints if c.table_name == target_table_name]
    if not table_constraints:
        return None

    _LOG.info(f"preprocessing constraints for table {target_table_name} in {model_type} model")
    translator = ConstraintTranslator(table_constraints, model_type)

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

    meta_dir = workspace_dir / "OriginalData" / "tgt-meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    encoding_types_file = meta_dir / "encoding-types.json"

    if encoding_types_file.exists():
        with open(encoding_types_file) as f:
            encoding_types = json.load(f)

        encoding_types.update(translator.get_encoding_types())
        for col in translator.get_columns_to_remove():
            encoding_types.pop(col, None)

        with open(encoding_types_file, "w") as f:
            json.dump(encoding_types, f, indent=2)

        _LOG.debug(f"updated encoding-types.json with internal columns for {table_name}")
