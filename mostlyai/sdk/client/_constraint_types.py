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

from typing import TYPE_CHECKING

from pydantic import field_validator, model_validator

from mostlyai.sdk.client.base import CustomBaseModel

if TYPE_CHECKING:
    from mostlyai.sdk.domain import Constraint, ConstraintConfig


class FixedCombination(CustomBaseModel):
    """internal typed representation of FixedCombination constraint."""

    table_name: str
    columns: list[str]

    @field_validator("columns")
    @classmethod
    def validate_columns(cls, columns):
        if len(columns) < 2:
            raise ValueError(f"FixedCombination requires at least 2 columns, got {len(columns)}.")
        return columns


class Inequality(CustomBaseModel):
    """internal typed representation of Inequality constraint."""

    table_name: str
    low_column: str
    high_column: str

    @model_validator(mode="after")
    def validate_columns(self):
        if self.low_column == self.high_column:
            raise ValueError(f"low_column and high_column must be different, both are '{self.low_column}'.")
        return self


def convert_constraint_config_to_typed(
    constraint_config: ConstraintConfig | Constraint,
) -> FixedCombination | Inequality:
    """convert ConstraintConfig or Constraint to typed constraint object."""
    # import here to avoid circular imports
    from mostlyai.sdk.domain import ConstraintType

    # config is now a plain dict[str, Any]
    config_dict = constraint_config.config

    # validate that we have the required fields
    if not config_dict or ("table_name" not in config_dict and "low_column" not in config_dict):
        raise ValueError(
            f"constraint config is missing required fields. "
            f"Expected 'table_name' and either 'columns' (for FixedCombination) or 'low_column'/'high_column' (for Inequality). "
            f"Got config: {config_dict}"
        )

    if constraint_config.type == ConstraintType.fixed_combination:
        return FixedCombination(**config_dict)
    elif constraint_config.type == ConstraintType.inequality:
        return Inequality(**config_dict)
    raise ValueError(f"unknown constraint type: {constraint_config.type}")
