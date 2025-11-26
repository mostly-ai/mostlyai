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

"""constraint transformation utilities for fixed combinations."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from mostlyai.sdk.domain import FixedCombination

if TYPE_CHECKING:
    from mostlyai.sdk.domain import Generator


def get_tgt_meta_path(workspace_dir: Path) -> Path:
    """get tgt-meta directory path in OriginalData."""
    path = workspace_dir / "OriginalData" / "tgt-meta"
    path.mkdir(parents=True, exist_ok=True)
    return path


class ConstraintTranslator:
    """translates data between user schema and internal schema for constraints."""

    def __init__(self, constraints: list[FixedCombination]):
        """initialize translator with constraint definitions.

        Args:
            constraints: List of FixedCombination constraint objects.
        """
        self.constraints = constraints
        self.merged_columns: list[tuple[list[str], str]] = []

        # precompute merged column names
        for constraint in constraints:
            columns = constraint.columns
            merged_name = "|".join(columns)
            self.merged_columns.append((columns, merged_name))

    def to_internal(self, df: pd.DataFrame) -> pd.DataFrame:
        """transform dataframe from user schema to internal schema.

        Args:
            df: DataFrame with original column structure.

        Returns:
            DataFrame with constrained columns merged.
        """
        df = df.copy()

        for columns, merged_name in self.merged_columns:
            # merge columns into one with | separator
            df[merged_name] = df[columns].astype(str).agg("|".join, axis=1)
            # drop original columns
            df = df.drop(columns=columns)

        return df

    def to_original(self, df: pd.DataFrame) -> pd.DataFrame:
        """transform dataframe from internal schema back to user schema.

        Args:
            df: DataFrame with merged constraint columns.

        Returns:
            DataFrame with original column structure.
        """
        df = df.copy()

        for columns, merged_name in self.merged_columns:
            if merged_name in df.columns:
                # split merged column back into original columns
                split_values = df[merged_name].str.split("|", n=len(columns) - 1, expand=True)
                for i, col in enumerate(columns):
                    df[col] = split_values[i]
                # drop merged column
                df = df.drop(columns=[merged_name])

        return df

    def get_internal_columns(self, original_columns: list[str]) -> list[str]:
        """get list of column names in internal schema.

        Args:
            original_columns: List of original column names.

        Returns:
            List of column names with constraint columns merged.
        """
        columns_to_remove = set()
        columns_to_add = []

        for columns, merged_name in self.merged_columns:
            columns_to_remove.update(columns)
            columns_to_add.append(merged_name)

        internal_columns = [c for c in original_columns if c not in columns_to_remove]
        internal_columns.extend(columns_to_add)

        return internal_columns

    @staticmethod
    def from_generator_config(
        generator: Generator,
        table_name: str,
    ) -> tuple[ConstraintTranslator | None, list[str] | None]:
        """create constraint translator from generator configuration.

        This loads constraints directly from the generator config without needing
        any external files. The generator object contains all necessary information.

        Args:
            generator: Generator object with table configurations.
            table_name: Name of the table to get constraints for.

        Returns:
            Tuple of (translator, original_columns) if constraints exist,
            or (None, None) if no constraints are defined.

        Example:
            >>> translator, columns = ConstraintTranslator.from_generator_config(
            ...     generator=g,
            ...     table_name="customers"
            ... )
            >>> if translator:
            ...     df_transformed = translator.to_original(df_internal)
        """
        # find table in generator
        table = next((t for t in generator.tables if t.name == table_name), None)
        if not table:
            return None, None

        # check for constraints in model configurations
        constraints = None

        # try tabular model first
        if table.tabular_model_configuration and table.tabular_model_configuration.constraints:
            constraints = table.tabular_model_configuration.constraints

        # try language model if tabular doesn't have constraints
        if not constraints and table.language_model_configuration:
            if table.language_model_configuration.constraints:
                constraints = table.language_model_configuration.constraints

        if not constraints:
            return None, None

        # create translator from constraints
        translator = ConstraintTranslator(constraints)

        # get original columns from generator
        original_columns = [c.name for c in table.columns] if table.columns else None

        return translator, original_columns
