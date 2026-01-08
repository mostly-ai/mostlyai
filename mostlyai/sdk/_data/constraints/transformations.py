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

import json
import logging
from pathlib import Path

import pandas as pd

from mostlyai.sdk._data.constraints.types import (
    ConstraintHandler,
    FixedCombinationsHandler,
    InequalityHandler,
)
from mostlyai.sdk.client._constraint_types import (
    FixedCombinations,
    Inequality,
    convert_constraint_config_to_typed,
)
from mostlyai.sdk.domain import Generator

_LOG = logging.getLogger(__name__)

# type alias for constraint types
ConstraintType = FixedCombinations | Inequality


def _create_constraint_handler(constraint: ConstraintType, table=None) -> ConstraintHandler:
    """factory function to create appropriate handler for a constraint."""
    if isinstance(constraint, FixedCombinations):
        return FixedCombinationsHandler(constraint)
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

        # convert constraints to typed objects and filter by table_name
        typed_constraints = []
        for constraint in generator.constraints:
            typed_constraint = convert_constraint_config_to_typed(constraint)
            if typed_constraint.table_name == table_name:
                typed_constraints.append(typed_constraint)

        if not typed_constraints:
            return None

        # pass table to translator so handlers can check column types
        constraint_translator = ConstraintTranslator(typed_constraints, table=table)
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

    # convert constraints to typed objects and filter by table_name
    typed_constraints = []
    for constraint in generator.constraints:
        typed_constraint = convert_constraint_config_to_typed(constraint)
        if typed_constraint.table_name == target_table_name:
            typed_constraints.append(typed_constraint)

    if not typed_constraints:
        return None

    _LOG.info(f"preprocessing constraints for table {target_table_name}")
    # pass table to translator so handlers can check column types
    constraint_translator = ConstraintTranslator(typed_constraints, table=target_table)

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
    else:
        _LOG.error(f"encoding-types.json not found to update internal columns for {table_name}")
