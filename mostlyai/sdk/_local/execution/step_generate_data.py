# Copyright 2024-2025 MOSTLY AI
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
import uuid
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from mostlyai.sdk import _data as data
from mostlyai.sdk._data.base import Schema
from mostlyai.sdk._data.util.common import TABLE_COLUMN_INFIX, TEMPORARY_PRIMARY_KEY
from mostlyai.sdk._local.execution.migration import migrate_workspace
from mostlyai.sdk.domain import Generator, ModelType, SyntheticDataset


def execute_step_generate_data(
    *,
    generator: Generator,
    synthetic_dataset: SyntheticDataset,
    target_table_name: str,
    model_type: ModelType,
    sample_seed: pd.DataFrame | None = None,
    schema: Schema,
    workspace_dir: Path,
    update_progress: Callable,
):
    # import ENGINE here to avoid pre-mature loading of large ENGINE dependencies
    import mostlyai.engine as engine

    # ensure backward compatibility
    migrate_workspace(workspace_dir)

    tgt_g_table = next(t for t in generator.tables if t.name == target_table_name)
    tgt_sd_table = next(t for t in synthetic_dataset.tables if t.name == target_table_name)
    config = tgt_sd_table.configuration

    ctx_data = None  # context is taken implicitly from workspace dir, if needed
    if any(fk.is_context for fk in (tgt_g_table.foreign_keys or [])) or model_type == ModelType.language:
        data.pull_context(
            tgt=tgt_g_table.name,
            schema=schema,
            max_sample_size=None,
            model_type=model_type,
            workspace_dir=workspace_dir,
        )
        ### TODO: FIX
        # Hack that enables single table, single text column generation
        ctx_data_dir = workspace_dir / "OriginalData" / "ctx-data"
        if not ctx_data_dir.exists():
            ctx_data_dir.mkdir(parents=True)
            ctx_primary_key = f"{tgt_g_table.name}{TABLE_COLUMN_INFIX}{TEMPORARY_PRIMARY_KEY}"
            dummy_ctx_length = config.sample_size if sample_seed is None else sample_seed.shape[0]
            dummy_ctx = pd.DataFrame({ctx_primary_key: [str(uuid.uuid4()) for _ in range(dummy_ctx_length)]})
            dummy_ctx.to_parquet(ctx_data_dir / "part.00000-ctx.parquet")
        ### TODO: FIX

    if model_type == ModelType.language:
        model_config = tgt_g_table.language_model_configuration
    else:
        model_config = tgt_g_table.tabular_model_configuration

    # handle sample size / seed
    is_subject = not (any(fk.is_context for fk in (tgt_g_table.foreign_keys or [])))
    if is_subject:
        if sample_seed is None and config.sample_size is not None:
            sample_size = config.sample_size
        else:
            sample_size = None
    else:
        sample_size = None

    # ensure disallowed arguments are set to None
    if model_type == ModelType.language:
        rare_category_replacement_method = None
        rebalancing = None
        imputation = None
        fairness = None
    else:  # model_type == ModelType.tabular
        rare_category_replacement_method = model_config.rare_category_replacement_method
        rebalancing = config.rebalancing
        imputation = config.imputation
        fairness = config.fairness

    # call GENERATE
    engine.generate(
        ctx_data=ctx_data,
        seed_data=sample_seed,
        sample_size=sample_size,
        batch_size=None,
        sampling_temperature=config.sampling_temperature,
        sampling_top_p=config.sampling_top_p,
        device=None,
        rare_category_replacement_method=rare_category_replacement_method,
        rebalancing=rebalancing,
        imputation=imputation,
        fairness=fairness,
        workspace_dir=workspace_dir,
        update_progress=update_progress,
    )
