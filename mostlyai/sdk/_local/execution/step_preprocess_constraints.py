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

"""preprocessing step for constraint transformations."""

import logging
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from mostlyai.sdk._data.constraint_transformations import ConstraintTranslator, get_tgt_meta_path
from mostlyai.sdk.domain import Generator, ModelType

_LOG = logging.getLogger(__name__)


def execute_step_preprocess_constraints(
    *,
    generator: Generator,
    workspace_dir: Path,
    model_type: ModelType,
    target_table_name: str,
    update_progress: Callable,
) -> None:
    """preprocess constraint transformations for training data.

    Args:
        generator: Generator configuration.
        workspace_dir: Workspace directory path.
        model_type: Model type being trained.
        target_table_name: Name of the table to process.
        update_progress: Progress update callback.
    """
    # get target table configuration
    target_table = next((t for t in generator.tables if t.name == target_table_name), None)
    if not target_table:
        _LOG.info(f"table {target_table_name} not found in generator")
        return

    # get model configuration based on model type
    if model_type == ModelType.language:
        model_config = target_table.language_model_configuration
    else:
        model_config = target_table.tabular_model_configuration

    # early return if no model config or no constraints
    if not model_config or not model_config.constraints:
        return

    _LOG.info(f"preprocessing constraints for table {target_table_name} in {model_type} model")

    # create constraint translator
    translator = ConstraintTranslator(model_config.constraints)

    # get data directory - note: workspace_dir is already at the model level (e.g. .../test:tabular/)
    # and contains OriginalData/tgt-data/ directly (not OriginalData/TABULAR/tgt-data/test/)
    tgt_data_dir = workspace_dir / "OriginalData" / "tgt-data"

    if not tgt_data_dir.exists():
        _LOG.warning(f"data directory not found: {tgt_data_dir}")
        return

    # process all parquet files
    parquet_files = sorted(list(tgt_data_dir.glob("part.*.parquet")))

    for i, parquet_file in enumerate(parquet_files, start=1):
        # read data
        df = pd.read_parquet(parquet_file)

        # apply transformation
        df_transformed = translator.to_internal(df)

        # write back to same file
        df_transformed.to_parquet(parquet_file, index=True)

        # update progress
        progress = i / len(parquet_files)
        update_progress(progress)

    # save metadata with original columns
    original_columns = [c.name for c in target_table.columns]

    # note: we no longer save constraints.json - constraints are read directly from generator config

    # update tgt-meta with internal column structure
    _update_meta_with_internal_columns(workspace_dir, target_table_name, translator, parquet_files)

    # update generator columns to reflect internal schema for training
    internal_columns = translator.get_internal_columns(original_columns)

    # update the column list
    from mostlyai.sdk.domain import SourceColumn

    target_table.columns = [SourceColumn(name=col) for col in internal_columns]

    # update progress to 100%
    update_progress(1.0)


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
    import json

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
