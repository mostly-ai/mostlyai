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
    ModelEncodingType,
)

_LOG = logging.getLogger(__name__)

# type alias for constraint types
ConstraintType = FixedCombination | Inequality


def _generate_internal_column_name(prefix: str, columns: list[str]) -> str:
    """generate a deterministic internal column name."""
    key = "|".join(columns)
    hash_suffix = hashlib.md5(key.encode()).hexdigest()[:8]
    columns_str = "_".join(col.upper() for col in columns)
    return f"__TABULAR_CONSTRAINT_{prefix}_{columns_str}_{hash_suffix}__"


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


class FixedCombinationHandler(ConstraintHandler):
    """handler for FixedCombination constraints."""

    def __init__(self, constraint: FixedCombination):
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

            split_values = df[self.merged_name].astype(str).apply(split_row)
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


class InequalityHandler(ConstraintHandler):
    """handler for Inequality constraints (low <= high or low < high if strict_boundaries=True)."""

    _NUMERIC_EPSILON = 1e-8  # conservative practical float64 difference (1 for integers)
    _INTEGER_EPSILON = 1  # Epsilon for integer types
    _DATETIME_EPSILON = pd.Timedelta(microseconds=1)  # smallest reliable timedelta
    _DATETIME_EPOCH = pd.Timestamp("1970-01-01")  # reference epoch for delta representation

    def __init__(self, constraint: Inequality, table=None):
        self.constraint = constraint
        self.table_name = constraint.table_name
        self.low_column = constraint.low_column
        self.high_column = constraint.high_column
        self.strict_boundaries = constraint.strict_boundaries
        self._delta_column = _generate_internal_column_name("INEQ_DELTA", [self.low_column, self.high_column])

        # determine if this is a datetime constraint based on table encoding types
        self._is_datetime = False
        if table and table.columns:
            datetime_encodings = {
                ModelEncodingType.tabular_datetime,
                ModelEncodingType.tabular_datetime_relative,
            }
            # check if either column is datetime-encoded
            self._is_datetime = any(
                col.model_encoding_type in datetime_encodings
                for col in table.columns
                if col.name in {self.low_column, self.high_column}
            )

    def _repr_boundaries(self) -> str:
        """return string representation of inequality boundaries."""
        return (
            f"{self.low_column} <= {self.high_column}"
            if not self.strict_boundaries
            else f"{self.low_column} < {self.high_column}"
        )

    def _enforce_strict_delta(self, delta: pd.Series) -> pd.Series:
        """enforce strict boundaries by ensuring delta > 0."""
        zero = pd.Timedelta(0) if self._is_datetime else 0
        zero_mask = delta <= zero
        if not zero_mask.any():
            return delta

        is_integer = pd.api.types.is_integer_dtype(delta)
        epsilon = (
            self._DATETIME_EPSILON
            if self._is_datetime
            else self._INTEGER_EPSILON
            if is_integer
            else self._NUMERIC_EPSILON
        )

        _LOG.warning(
            f"correcting {zero_mask.sum()} equality violations for strict inequality {self._repr_boundaries()}"
        )
        return delta.where(delta > zero, epsilon)

    def get_internal_column_names(self) -> list[str]:
        return [self._delta_column]

    def to_internal(self, df: pd.DataFrame) -> pd.DataFrame:
        self._validate_columns(df, [self.low_column, self.high_column])
        low = df[self.low_column]
        high = df[self.high_column]

        zero = pd.Timedelta(0) if self._is_datetime else 0
        delta = high - low
        violations = delta < zero
        if violations.any():
            _LOG.warning(f"correcting {violations.sum()} inequality violations for {self._repr_boundaries()}")
            delta[violations] = zero

        # enforce strict boundaries if needed
        if self.strict_boundaries:
            delta = self._enforce_strict_delta(delta)

        # convert timedelta to datetime using epoch (for datetime constraints)
        if self._is_datetime:
            # represent delta as datetime: epoch + timedelta
            delta = self._DATETIME_EPOCH + delta
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

        # convert datetime back to timedelta if needed
        if self._is_datetime:
            delta = delta - self._DATETIME_EPOCH

        # enforce strict boundaries if needed
        if self.strict_boundaries:
            delta = self._enforce_strict_delta(delta)

        # default reconstruction: high = low + delta
        df[self.high_column] = low + delta

        return df.drop(columns=[self._delta_column])

    def get_encoding_types(self) -> dict[str, str]:
        # use TABULAR_DATETIME for datetime constraints to preserve precision
        # use TABULAR_NUMERIC_AUTO for numeric constraints
        if self._is_datetime:
            return {self._delta_column: "TABULAR_DATETIME"}
        return {self._delta_column: "TABULAR_NUMERIC_AUTO"}


def _create_constraint_handler(constraint: ConstraintType, table=None) -> ConstraintHandler:
    """factory function to create appropriate handler for a constraint."""
    if isinstance(constraint, FixedCombination):
        return FixedCombinationHandler(constraint)
    elif isinstance(constraint, Inequality):
        return InequalityHandler(constraint, table=table)
    else:
        raise ValueError(f"unknown constraint type: {type(constraint)}")


class ConstraintTranslator:
    """translates data between user schema and internal schema for constraints."""

    def __init__(self, constraints: list[ConstraintType], table=None):
        self.constraints = constraints
        self.table = table
        self.handlers = [_create_constraint_handler(c, table=table) for c in constraints]

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

    def get_all_column_names(self, original_column_names: list[str]) -> list[str]:
        """get list of all column names (original and internal constraint columns)."""
        all_column_names = list(original_column_names)
        for handler in self.handlers:
            all_column_names.extend(handler.get_internal_column_names())
        return all_column_names

    def get_encoding_types(self) -> dict[str, str]:
        """get combined encoding types for all internal columns."""
        encoding_types = {}
        for handler in self.handlers:
            encoding_types.update(handler.get_encoding_types())
        return encoding_types

    @staticmethod
    def from_generator_config(
        generator: Generator,
        table_name: str,
    ) -> ConstraintTranslator | None:
        """create constraint translator from generator configuration for a specific table."""
        if not generator.constraints:
            return None

        table = next((t for t in generator.tables if t.name == table_name), None)
        if not table:
            return None

        # filter by table_name
        table_constraints = [c for c in generator.constraints if c.table_name == table_name]
        if not table_constraints:
            return None

        # pass table to translator so handlers can check column types
        constraint_translator = ConstraintTranslator(table_constraints, table=table)
        return constraint_translator


def preprocess_constraints_for_training(
    *,
    generator: Generator,
    workspace_dir: Path,
    target_table_name: str,
) -> list[str] | None:
    """preprocess constraint transformations for training data:
    - transform constraints from user schema to internal schema (if any)
    - update tgt-meta (encoding-types) and tgt-data with internal columns (if any)
    - return list of all column names (original and internal constraint columns) for use in training
    """
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
    # pass table to translator so handlers can check column types
    constraint_translator = ConstraintTranslator(table_constraints, table=target_table)

    tgt_data_dir = workspace_dir / "OriginalData" / "tgt-data"
    if not tgt_data_dir.exists():
        _LOG.warning(f"data directory not found: {tgt_data_dir}")
        return None

    parquet_files = sorted(list(tgt_data_dir.glob("part.*.parquet")))
    for parquet_file in parquet_files:
        df = pd.read_parquet(parquet_file)
        df_transformed = constraint_translator.to_internal(df)
        df_transformed.to_parquet(parquet_file, index=True)

    original_columns = [c.name for c in target_table.columns] if target_table.columns else []
    _update_meta_with_internal_columns(workspace_dir, target_table_name, constraint_translator, parquet_files)
    all_column_names = constraint_translator.get_all_column_names(original_columns)
    return all_column_names


def _update_meta_with_internal_columns(
    workspace_dir: Path,
    table_name: str,
    constraint_translator: ConstraintTranslator,
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

        encoding_types.update(constraint_translator.get_encoding_types())

        with open(encoding_types_file, "w") as f:
            json.dump(encoding_types, f, indent=2)

        _LOG.debug(f"updated encoding-types.json with internal columns for {table_name}")
