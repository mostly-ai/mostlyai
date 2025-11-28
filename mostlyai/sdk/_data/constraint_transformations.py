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

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from mostlyai.sdk.domain import FixedCombination

if TYPE_CHECKING:
    from mostlyai.sdk.domain import Generator, ModelType

_LOG = logging.getLogger(__name__)


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

        Original columns are kept in the DataFrame but excluded from training
        via encoding-types.json. This avoids generator config mutation.

        Args:
            df: DataFrame with original column structure.

        Returns:
            DataFrame with merged column added (original columns kept).
        """
        df = df.copy()

        for columns, merged_name in self.merged_columns:
            df[merged_name] = df[columns].astype(str).agg("|".join, axis=1)

        return df

    def to_original(self, df: pd.DataFrame) -> pd.DataFrame:
        """transform dataframe from internal schema back to user schema.

        Overrides existing original columns with de-transformed values from merged column.

        Args:
            df: DataFrame with merged constraint columns.

        Returns:
            DataFrame with original columns overridden and merged column dropped.
        """
        df = df.copy()

        for columns, merged_name in self.merged_columns:
            if merged_name in df.columns:
                split_values = df[merged_name].str.split("|", n=len(columns) - 1, expand=True)
                for i, col in enumerate(columns):
                    df[col] = split_values[i]
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

    def get_original_columns(self, internal_columns: list[str]) -> list[str]:
        """get list of column names in original schema from internal schema.

        This reverses the transformation: replaces merged columns with original columns.

        Args:
            internal_columns: List of internal column names (with merged columns).

        Returns:
            List of original column names (with columns split).
        """
        original_columns = []

        for col in internal_columns:
            # check if this is a merged column
            is_merged = False
            for columns, merged_name in self.merged_columns:
                if col == merged_name:
                    # replace merged column with original columns
                    original_columns.extend(columns)
                    is_merged = True
                    break

            if not is_merged:
                # keep non-merged column as is
                original_columns.append(col)

        return original_columns

    @staticmethod
    def from_generator_config(
        generator: Generator,
        table_name: str,
    ) -> tuple[ConstraintTranslator | None, list[str] | None]:
        """create constraint translator from generator configuration.

        This loads constraints directly from the generator config without needing
        any external files. The generator object contains all necessary information.

        Importantly, this method works even if generator.columns has been modified
        to internal schema - it will reverse-engineer the original columns from
        the constraints + internal columns.

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

        # get current columns from generator (might be internal or original schema)
        current_columns = [c.name for c in table.columns] if table.columns else None

        if not current_columns:
            return translator, None

        # reverse-engineer original columns from current columns
        # if current columns are in internal schema (have merged columns),
        # this will convert them back to original schema
        original_columns = translator.get_original_columns(current_columns)

        return translator, original_columns


def preprocess_constraints_for_training(
    *,
    generator: Generator,
    workspace_dir: Path,
    model_type: ModelType,
    target_table_name: str,
) -> list[str] | None:
    """preprocess constraint transformations for training data.

    applies constraint transformations to parquet files and updates metadata.
    returns the internal column list if constraints were applied, None otherwise.

    Args:
        generator: Generator configuration.
        workspace_dir: Workspace directory path.
        model_type: Model type being trained.
        target_table_name: Name of the table to process.

    Returns:
        List of internal column names if constraints were applied, None otherwise.
    """
    # get target table configuration
    target_table = next((t for t in generator.tables if t.name == target_table_name), None)
    if not target_table:
        _LOG.info(f"table {target_table_name} not found in generator")
        return None

    # get model configuration based on model type
    if model_type.value == "language":
        model_config = target_table.language_model_configuration
    else:
        model_config = target_table.tabular_model_configuration

    # early return if no model config or no constraints
    if not model_config or not model_config.constraints:
        return None

    _LOG.info(f"preprocessing constraints for table {target_table_name} in {model_type} model")

    # create constraint translator
    translator = ConstraintTranslator(model_config.constraints)

    # get data directory
    tgt_data_dir = workspace_dir / "OriginalData" / "tgt-data"

    if not tgt_data_dir.exists():
        _LOG.warning(f"data directory not found: {tgt_data_dir}")
        return None

    # process all parquet files
    parquet_files = sorted(list(tgt_data_dir.glob("part.*.parquet")))

    for parquet_file in parquet_files:
        # read data
        df = pd.read_parquet(parquet_file)

        # apply transformation
        df_transformed = translator.to_internal(df)

        # write back to same file
        df_transformed.to_parquet(parquet_file, index=True)

    # save metadata with original columns
    original_columns = [c.name for c in target_table.columns] if target_table.columns else []

    # update tgt-meta with internal column structure
    _update_meta_with_internal_columns(workspace_dir, target_table_name, translator, parquet_files)

    # get internal columns to return (for training purposes)
    internal_columns = translator.get_internal_columns(original_columns)

    # NOTE: we do NOT modify target_table.columns here!
    # generator config must remain immutable after validation/auto-completion.
    # the internal columns are returned and used only for training,
    # without mutating the generator object.

    return internal_columns


def _update_meta_with_internal_columns(
    workspace_dir: Path,
    table_name: str,
    translator: ConstraintTranslator,
    parquet_files: list[Path],
) -> None:
    """update tgt-meta to reflect internal column structure after transformation.

    Args:
        workspace_dir: Workspace directory path.
        table_name: Name of the table.
        translator: Constraint translator with transformation info.
        parquet_files: List of parquet files that were transformed.
    """
    # read a sample file to get transformed columns
    if not parquet_files:
        return

    # update encoding-types.json to include merged columns
    meta_dir = get_tgt_meta_path(workspace_dir)
    encoding_types_file = meta_dir / "encoding-types.json"

    if encoding_types_file.exists():
        with open(encoding_types_file) as f:
            encoding_types = json.load(f)

        # get the columns that were merged
        for constraint in translator.constraints:
            merged_columns = constraint.columns
            merged_name = "|".join(merged_columns)

            # force merged column to be TABULAR_CATEGORICAL to preserve valid combinations
            # this is critical for the constraint to work properly
            encoding_types[merged_name] = "TABULAR_CATEGORICAL"

            # remove original columns from encoding types
            for col in merged_columns:
                encoding_types.pop(col, None)

        # write back
        with open(encoding_types_file, "w") as f:
            json.dump(encoding_types, f, indent=2)

        _LOG.info(f"updated encoding-types.json with internal columns for {table_name}")
