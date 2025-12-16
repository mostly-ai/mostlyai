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

from mostlyai.sdk.domain import (
    FixedCombination,
    Generator,
    Inequality,
    ModelType,
)

_LOG = logging.getLogger(__name__)

# type alias for constraint types
ConstraintType = FixedCombination | Inequality


def _generate_internal_column_name(prefix: str, columns: list[str]) -> str:
    """generate a deterministic internal column name."""
    key = "|".join(columns)
    hash_suffix = hashlib.md5(key.encode()).hexdigest()[:8]
    columns_str = "_".join(col.upper() for col in columns)
    return f"__TABULAR_CONSTRAINT_{prefix.upper()}_{columns_str}_{hash_suffix}__"


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
    def to_original(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def _validate_columns(self, df: pd.DataFrame, columns: list[str]) -> None:
        """validate that all required columns exist in the dataframe."""
        missing_cols = set(columns) - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"Columns {sorted(missing_cols)} required by {self.__class__.__name__} "
                f"not found in dataframe. Available columns: {sorted(df.columns)}"
            )

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

    def __init__(self, constraint: FixedCombination):
        self.constraint = constraint
        self.table_name = constraint.table_name
        self.columns = constraint.columns
        self.merged_name = _generate_internal_column_name("fixedcomb", self.columns)

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

    def to_original(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if self.merged_name in df.columns:

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

            # reset df index to align with split_df
            df = df.reset_index(drop=True)
            split_df = split_df.reset_index(drop=True)

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

    def get_table_column_tuples(self) -> list[tuple[str, str]]:
        """return list of (table_name, column_name) tuples involved in this constraint."""
        return [(self.table_name, col) for col in self.columns]


class InequalityHandler(ConstraintHandler):
    """handler for Inequality constraints (low <= high or low < high if strict_boundaries=True)."""

    _NUMERIC_EPSILON = 1e-8  # conservative practical float64 difference (1 for integers)
    _INTEGER_EPSILON = 1  # Epsilon for integer types
    _DATETIME_EPSILON = pd.Timedelta(microseconds=1)  # smallest reliable timedelta

    def __init__(self, constraint: Inequality):
        self.constraint = constraint
        self.table_name = constraint.table_name
        self.low_column = constraint.low_column
        self.high_column = constraint.high_column
        self.strict_boundaries = constraint.strict_boundaries
        self._delta_column = _generate_internal_column_name("ineq_delta", [self.low_column, self.high_column])

    def _repr_boundaries(self) -> str:
        """return string representation of inequality boundaries."""
        return (
            f"{self.low_column} <= {self.high_column}"
            if not self.strict_boundaries
            else f"{self.low_column} < {self.high_column}"
        )

    def _enforce_strict_delta(self, delta: pd.Series, is_datetime: bool, is_integer: bool) -> pd.Series:
        """enforce strict boundaries by ensuring delta > 0.

        Args:
            delta: delta series to enforce
            is_datetime: whether the data is datetime type
            is_integer: whether the data is integer type (for numeric)

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

        _LOG.warning(
            f"correcting {zero_mask.sum()} equality violations for strict inequality {self._repr_boundaries()}"
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

        is_datetime = pd.api.types.is_datetime64_any_dtype(low)
        zero = pd.Timedelta(0) if is_datetime else 0
        delta = high - low
        violations = delta < zero
        if violations.any():
            _LOG.warning(f"correcting {violations.sum()} inequality violations for {self._repr_boundaries()}")
            delta[violations] = zero

        # enforce strict boundaries if needed
        if self.strict_boundaries:
            is_integer = pd.api.types.is_integer_dtype(low) or pd.api.types.is_integer_dtype(high)
            delta = self._enforce_strict_delta(delta, is_datetime, is_integer)

        df[self._delta_column] = delta
        return df

    def to_original(self, df: pd.DataFrame) -> pd.DataFrame:
        """transform from internal schema back to original schema."""
        df = df.copy()
        if self._delta_column not in df.columns:
            return df

        # prepare data and types
        is_datetime = pd.api.types.is_datetime64_any_dtype(df[self.low_column])
        low = df[self.low_column]
        delta = df[self._delta_column]
        original_dtype = None if is_datetime else low.dtype

        # ensure delta has correct type (numeric or timedelta)
        delta = self._ensure_delta_type(delta, is_datetime)

        # enforce strict boundaries if needed
        if self.strict_boundaries:
            delta = self._enforce_strict_delta(delta, is_datetime, pd.api.types.is_integer_dtype(low))

        # default reconstruction: high = low + delta
        df[self.high_column] = low + delta

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


def _create_constraint_handler(constraint: ConstraintType) -> ConstraintHandler:
    """factory function to create appropriate handler for a constraint."""
    if isinstance(constraint, FixedCombination):
        return FixedCombinationHandler(constraint)
    elif isinstance(constraint, Inequality):
        return InequalityHandler(constraint)
    else:
        raise ValueError(f"unknown constraint type: {type(constraint)}")


class ConstraintTranslator:
    """translates data between user schema and internal schema for constraints."""

    def __init__(self, constraints: list[ConstraintType]):
        self.constraints = constraints
        self.handlers = [_create_constraint_handler(c) for c in constraints]

    def to_internal(self, df: pd.DataFrame) -> pd.DataFrame:
        """transform dataframe from user schema to internal schema."""
        for handler in self.handlers:
            df = handler.to_internal(df)
        return df

    def to_original(self, df: pd.DataFrame) -> pd.DataFrame:
        """transform dataframe from internal schema back to user schema."""
        for handler in self.handlers:
            df = handler.to_original(df)
        return df

    def get_internal_columns(self, original_columns: list[str]) -> list[str]:
        """get list of column names in internal schema."""
        # keep all original columns and add internal constraint columns
        internal_columns = list(original_columns)
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
        """create constraint translator from generator configuration.

        reads constraints from generator root level and filters by table_name.
        """
        if not generator.constraints:
            return None, None

        table = next((t for t in generator.tables if t.name == table_name), None)
        if not table:
            return None, None

        # filter by table_name
        table_constraints = [c for c in generator.constraints if c.table_name == table_name]
        if not table_constraints:
            return None, None

        translator = ConstraintTranslator(table_constraints)
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

    # filter by table_name
    table_constraints = [c for c in generator.constraints if c.table_name == target_table_name]
    if not table_constraints:
        return None

    _LOG.info(f"preprocessing constraints for table {target_table_name}")
    translator = ConstraintTranslator(table_constraints)

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

        with open(encoding_types_file, "w") as f:
            json.dump(encoding_types, f, indent=2)

        _LOG.debug(f"updated encoding-types.json with internal columns for {table_name}")
