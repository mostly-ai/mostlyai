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

"""internal typed constraint classes for validation and transformation.

These classes are internal-only and not part of the public API.
The public API uses ConstraintConfig with a dict config.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic import Field, field_validator, model_validator

from mostlyai.sdk.client.base import CustomBaseModel

if TYPE_CHECKING:
    from mostlyai.sdk.domain import Constraint, ConstraintConfig, SourceColumn


class BaseConstraint(CustomBaseModel, ABC):
    """base class for constraint types with validation interface."""

    table_name: str = Field(alias="tableName")

    @abstractmethod
    def validate(
        self,
        table_columns: dict[str, SourceColumn],
        column_usage: dict[str, dict[str, int]],
        constraint_idx: int,
    ) -> None:
        """validate constraint against table columns and track usage."""
        pass


class FixedCombinations(BaseConstraint):
    """internal typed representation of FixedCombinations constraint."""

    columns: list[str]

    @field_validator("columns")
    @classmethod
    def validate_columns(cls, columns):
        if len(columns) < 2:
            raise ValueError(f"FixedCombinations requires at least 2 columns, got {len(columns)}.")
        return columns

    def validate(
        self,
        table_columns: dict[str, SourceColumn],
        column_usage: dict[str, dict[str, int]],
        constraint_idx: int,
    ) -> None:
        """validate FixedCombinations constraint."""
        # import here to avoid circular imports
        from mostlyai.sdk.domain import ModelEncodingType

        missing_cols = set(self.columns) - set(table_columns.keys())
        if missing_cols:
            raise ValueError(
                f"columns {sorted(missing_cols)} in table '{self.table_name}' referenced by constraint not found"
            )

        for col_name in self.columns:
            encoding = table_columns[col_name].model_encoding_type or ModelEncodingType.auto

            _validate_tabular_encoding(col_name, self.table_name, encoding)

            if encoding not in (
                ModelEncodingType.auto,
                ModelEncodingType.tabular_categorical,
            ):
                raise ValueError(
                    f"Column '{col_name}' in table '{self.table_name}' must have TABULAR_CATEGORICAL encoding type"
                )

            _track_column_usage(self.table_name, col_name, constraint_idx, column_usage)


class Inequality(BaseConstraint):
    """internal typed representation of Inequality constraint."""

    low_column: str = Field(alias="lowColumn")
    high_column: str = Field(alias="highColumn")

    @model_validator(mode="after")
    def validate_columns(self):
        if self.low_column == self.high_column:
            raise ValueError(f"low_column and high_column must be different, both are '{self.low_column}'.")
        return self

    def validate(
        self,
        table_columns: dict[str, SourceColumn],
        column_usage: dict[str, dict[str, int]],
        constraint_idx: int,
    ) -> None:
        """validate Inequality constraint."""
        # import here to avoid circular imports
        from mostlyai.sdk.domain import ModelEncodingType

        missing_cols = []
        for col_name in [self.low_column, self.high_column]:
            if col_name not in table_columns:
                missing_cols.append(col_name)

        if missing_cols:
            raise ValueError(f"columns {missing_cols} in table '{self.table_name}' referenced by constraint not found")

        low_encoding = table_columns[self.low_column].model_encoding_type or ModelEncodingType.auto
        high_encoding = table_columns[self.high_column].model_encoding_type or ModelEncodingType.auto

        _validate_tabular_encoding(self.low_column, self.table_name, low_encoding)
        _validate_tabular_encoding(self.high_column, self.table_name, high_encoding)

        if low_encoding != ModelEncodingType.auto and high_encoding != ModelEncodingType.auto:
            _validate_compatible_encodings(self, low_encoding, high_encoding)

        for col_name in [self.low_column, self.high_column]:
            _track_column_usage(self.table_name, col_name, constraint_idx, column_usage)


def _validate_tabular_encoding(col_name: str, table_name: str, encoding: Any) -> None:
    """validate that column uses TABULAR model encoding."""
    from mostlyai.sdk.domain import ModelEncodingType, ModelType

    if encoding != ModelEncodingType.auto and not encoding.value.startswith(ModelType.tabular.value):
        raise ValueError(f"Column '{col_name}' in table '{table_name}' is not part of TABULAR model")


def _validate_compatible_encodings(
    constraint: Inequality,
    low_encoding: Any,
    high_encoding: Any,
) -> None:
    """validate that both columns have compatible encoding types."""
    from mostlyai.sdk.domain import ModelEncodingType

    numeric_types = {
        ModelEncodingType.tabular_numeric_auto,
        ModelEncodingType.tabular_numeric_discrete,
        ModelEncodingType.tabular_numeric_binned,
        ModelEncodingType.tabular_numeric_digit,
    }
    datetime_types = {
        ModelEncodingType.tabular_datetime,
        # ModelEncodingType.tabular_datetime_relative,  # not supported yet
    }

    both_numeric = low_encoding in numeric_types and high_encoding in numeric_types
    both_datetime = low_encoding in datetime_types and high_encoding in datetime_types

    if not (both_numeric or both_datetime):
        raise ValueError(
            f"Columns '{constraint.low_column}' and '{constraint.high_column}' in table "
            f"'{constraint.table_name}' must both be either numeric or datetime encoding types"
        )


def _track_column_usage(
    table_name: str,
    col_name: str,
    constraint_idx: int,
    column_usage: dict[str, dict[str, int]],
) -> None:
    """track column usage and detect overlaps."""
    if table_name not in column_usage:
        column_usage[table_name] = {}

    if col_name in column_usage[table_name]:
        raise ValueError(f"column '{col_name}' in table '{table_name}' is referenced by multiple constraints")

    column_usage[table_name][col_name] = constraint_idx


def convert_constraint_config_to_typed(
    constraint_config: ConstraintConfig | Constraint,
) -> BaseConstraint:
    """convert ConstraintConfig or Constraint to typed constraint object."""
    # import here to avoid circular imports
    from mostlyai.sdk.domain import ConstraintType

    # config is now a plain dict[str, Any]
    config_dict = constraint_config.config

    if not config_dict:
        raise ValueError(f"constraint config is missing required fields. Got config: {config_dict}")

    # Pydantic with populate_by_name=True will accept both snake_case and camelCase
    if constraint_config.type == ConstraintType.fixed_combinations:
        return FixedCombinations(**config_dict)
    elif constraint_config.type == ConstraintType.inequality:
        return Inequality(**config_dict)
    raise ValueError(f"unknown constraint type: {constraint_config.type}")
