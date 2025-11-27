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


import logging
import shutil
import time
import traceback
from pathlib import Path

import pandas as pd

import mostlyai.engine as engine
from mostlyai.sdk._data.base import NonContextRelation, Schema
from mostlyai.sdk._data.non_context import (
    CHILDREN_COUNT_COLUMN_NAME,
    ParentChildMatcher,
    analyze_df,
    encode_df,
    get_cardinalities,
    prepare_training_pairs_for_fk_model,
    pull_fk_model_training_data,
    safe_name,
    store_fk_model,
    train_fk_model,
)
from mostlyai.sdk._data.progress_callback import ProgressCallback, ProgressCallbackWrapper
from mostlyai.sdk._local.execution.step_pull_training_data import create_training_schema
from mostlyai.sdk.domain import Connector, Generator

_LOG = logging.getLogger(__name__)


def execute_train_non_context_models_for_single_table(
    *,
    tgt_table_name: str,
    schema: Schema,
    fk_models_workspace_dir: Path,
    update_progress: ProgressCallback,
):
    non_ctx_relations = [rel for rel in schema.non_context_relations if rel.child.table == tgt_table_name]
    if not non_ctx_relations:
        # no non-context relations, so no parent-child matchers to train
        return

    fk_models_workspace_dir.mkdir(parents=True, exist_ok=True)

    for non_ctx_relation in non_ctx_relations:
        tgt_parent_key = non_ctx_relation.child.column
        fk_model_workspace_dir = fk_models_workspace_dir / safe_name(tgt_parent_key)

        train_non_context_models_for_single_relation(
            tgt_table_name=tgt_table_name,
            non_ctx_relation=non_ctx_relation,
            schema=schema,
            fk_model_workspace_dir=fk_model_workspace_dir,
        )

        # report progress after each FK model training
        update_progress(advance=1)


def train_fk_matching_model(
    *,
    parent_data: pd.DataFrame,
    tgt_data: pd.DataFrame,
    parent_primary_key: str,
    tgt_parent_key: str,
    parent_encoding_types: dict[str, any] | None,
    tgt_encoding_types: dict[str, any] | None,
    fk_model_workspace_dir: Path,
):
    """
    Train and save neural network model for parent-child matching.

    This trains a ParentChildMatcher that learns to match children to parents
    based on feature similarity, then saves it to disk.

    Args:
        parent_data: Parent table data
        tgt_data: Target/child table data
        parent_primary_key: Primary key column in parent data
        tgt_parent_key: Foreign key column in target data
        parent_encoding_types: User-defined encoding types for parent columns
        tgt_encoding_types: User-defined encoding types for target columns
        fk_model_workspace_dir: Directory to save model artifacts
    """
    tgt_data_columns = [c for c in tgt_data.columns if c != tgt_parent_key]
    parent_data_columns = [c for c in parent_data.columns if c != parent_primary_key]

    matching_model_dir = fk_model_workspace_dir / "fk_matching_model"
    tgt_stats_dir = matching_model_dir / "tgt-stats"
    analyze_df(
        df=tgt_data,
        parent_key=tgt_parent_key,
        data_columns=tgt_data_columns,
        encoding_types=tgt_encoding_types,
        stats_dir=tgt_stats_dir,
    )

    parent_stats_dir = matching_model_dir / "parent-stats"
    analyze_df(
        df=parent_data,
        primary_key=parent_primary_key,
        data_columns=parent_data_columns,
        encoding_types=parent_encoding_types,
        stats_dir=parent_stats_dir,
    )

    tgt_encoded_data = encode_df(
        df=tgt_data,
        stats_dir=tgt_stats_dir,
        include_primary_key=False,
    )

    parent_encoded_data = encode_df(
        df=parent_data,
        stats_dir=parent_stats_dir,
    )

    parent_cardinalities = get_cardinalities(stats_dir=parent_stats_dir)
    tgt_cardinalities = get_cardinalities(stats_dir=tgt_stats_dir)
    model = ParentChildMatcher(
        parent_cardinalities=parent_cardinalities,
        child_cardinalities=tgt_cardinalities,
    )

    parent_pd, tgt_pd, labels_pd = prepare_training_pairs_for_fk_model(
        parent_encoded_data=parent_encoded_data,
        tgt_encoded_data=tgt_encoded_data,
        parent_primary_key=parent_primary_key,
        tgt_parent_key=tgt_parent_key,
    )

    train_fk_model(
        model=model,
        parent_pd=parent_pd,
        tgt_pd=tgt_pd,
        labels=labels_pd,
    )

    store_fk_model(
        model=model,
        fk_model_workspace_dir=fk_model_workspace_dir,
    )


def train_cardinality_model(
    *,
    parent_data,
    parent_primary_key: str,
    fk_model_workspace_dir: Path,
):
    """
    Train engine-based model to predict number of children per parent.

    This trains a TabularARGN model using mostlyai-engine to predict the
    CHILDREN_COUNT_COLUMN_NAME column, which represents how many children
    each parent should have.

    Args:
        parent_data: Parent table data with CHILDREN_COUNT_COLUMN_NAME already added
        parent_primary_key: Primary key column name in parent data
        fk_model_workspace_dir: Directory to save model artifacts
    """
    cardinality_workspace_dir = fk_model_workspace_dir / "cardinality_model"
    cardinality_workspace_dir.mkdir(parents=True, exist_ok=True)

    _LOG.info(f"[NonContext Cardinality] Split data | n_rows={len(parent_data)}")
    engine.split(
        tgt_data=parent_data,
        tgt_primary_key=parent_primary_key,
        workspace_dir=cardinality_workspace_dir,
        update_progress=lambda **kwargs: None,
    )

    _LOG.info("[NonContext Cardinality] Analyze data")
    engine.analyze(
        workspace_dir=cardinality_workspace_dir,
        update_progress=lambda **kwargs: None,
    )

    _LOG.info("[NonContext Cardinality] Encode data")
    engine.encode(
        workspace_dir=cardinality_workspace_dir,
        update_progress=lambda **kwargs: None,
    )

    _LOG.info("[NonContext Cardinality] Train model | model=MOSTLY_AI/Medium")
    engine.train(
        model="MOSTLY_AI/Medium",
        workspace_dir=cardinality_workspace_dir,
        enable_flexible_generation=False,
        update_progress=lambda **kwargs: None,
    )


def train_non_context_models_for_single_relation(
    *,
    tgt_table_name: str,
    non_ctx_relation: NonContextRelation,
    schema: Schema,
    fk_model_workspace_dir: Path,
):
    """
    Train both FK matching and cardinality models for a non-context relation.

    This orchestrates training of:
    1. FK matching model (ParentChildMatcher) - matches children to parents
    2. Cardinality model (engine-based) - predicts children count per parent

    Args:
        tgt_table_name: Name of target/child table
        non_ctx_relation: Non-context relation to train models for
        schema: Schema containing table definitions
        fk_model_workspace_dir: Directory to save model artifacts
    """
    t0 = time.time()

    tgt_table = schema.tables[tgt_table_name]
    tgt_parent_key = non_ctx_relation.child.column

    parent_table = schema.tables[non_ctx_relation.parent.table]
    parent_primary_key = non_ctx_relation.parent.column
    parent_table_name = non_ctx_relation.parent.table

    parent_data, tgt_data = pull_fk_model_training_data(
        tgt_table=tgt_table,
        parent_table=parent_table,
        tgt_parent_key=tgt_parent_key,
        schema=schema,
    )

    parent_empty_or_key_only = parent_data.empty or len(parent_data.columns) <= 1
    tgt_empty_or_key_only = tgt_data.empty or len(tgt_data.columns) <= 1
    if parent_empty_or_key_only or tgt_empty_or_key_only:
        _LOG.info(
            f"[NonContext] Skip FK model training | relation={tgt_table_name}.{tgt_parent_key}->{parent_table_name}.{parent_primary_key} | reason=empty_or_key_only"
        )
        return

    fk_model_workspace_dir.mkdir(parents=True, exist_ok=True)

    # Add children count column to parent data
    children_counts = tgt_data[tgt_parent_key].value_counts()
    children_counts_mapped = parent_data[parent_primary_key].map(children_counts).fillna(0).astype(int)
    parent_data_with_counts = parent_data.assign(**{CHILDREN_COUNT_COLUMN_NAME: children_counts_mapped})
    max_count = children_counts_mapped.max() if len(children_counts_mapped) > 0 else 0
    avg_count = children_counts_mapped.mean() if len(children_counts_mapped) > 0 else 0
    _LOG.info(
        f"[NonContext] Prepare parent data with children counts | max_children={max_count} | avg_children={avg_count:.2f}"
    )

    _LOG.info(
        f"[NonContext Matching] Train FK matching model | relation={tgt_table_name}.{tgt_parent_key}->{parent_table_name}.{parent_primary_key}"
    )
    train_fk_matching_model(
        parent_data=parent_data_with_counts,
        tgt_data=tgt_data,
        parent_primary_key=parent_primary_key,
        tgt_parent_key=tgt_parent_key,
        parent_encoding_types=parent_table.encoding_types,
        tgt_encoding_types=tgt_table.encoding_types,
        fk_model_workspace_dir=fk_model_workspace_dir,
    )

    _LOG.info(
        f"[NonContext Cardinality] Train cardinality model | relation={tgt_table_name}.{tgt_parent_key}->{parent_table_name}.{parent_primary_key}"
    )
    train_cardinality_model(
        parent_data=parent_data_with_counts,
        parent_primary_key=parent_primary_key,
        fk_model_workspace_dir=fk_model_workspace_dir,
    )

    _LOG.info(
        f"[NonContext] All models trained | relation={tgt_table_name}.{tgt_parent_key}->{parent_table_name}.{parent_primary_key} | "
        f"time={time.time() - t0:.2f}s | workspace={fk_model_workspace_dir}"
    )


def clean_up_non_context_models_dirs(fk_models_workspace_dir: Path):
    # ensure OriginalData is not persisted
    for path in fk_models_workspace_dir.absolute().rglob("*"):
        if path.name == "OriginalData":
            shutil.rmtree(path)


def execute_step_finalize_training(
    *,
    generator: Generator,
    connectors: list[Connector],
    job_workspace_dir: Path,
    update_progress: ProgressCallback | None = None,
):
    schema = create_training_schema(generator=generator, connectors=connectors)

    # calculate total number of non-context relations to train
    total_non_ctx_relations = sum(
        len([rel for rel in schema.non_context_relations if rel.child.table == tgt_table_name])
        for tgt_table_name in schema.tables
    )

    with ProgressCallbackWrapper(update_progress, description="Finalize training") as progress:
        # initialize progress with total count
        progress.update(completed=0, total=max(1, total_non_ctx_relations))

        for tgt_table_name in schema.tables:
            fk_models_workspace_dir = job_workspace_dir / "FKModelsStore" / tgt_table_name
            try:
                execute_train_non_context_models_for_single_table(
                    tgt_table_name=tgt_table_name,
                    schema=schema,
                    fk_models_workspace_dir=fk_models_workspace_dir,
                    update_progress=progress.update,
                )
            except Exception as e:
                _LOG.error(
                    f"[NonContext] FK model training failed | table={tgt_table_name} | error={e}\n{traceback.format_exc()}"
                )
                continue
            finally:
                clean_up_non_context_models_dirs(fk_models_workspace_dir=fk_models_workspace_dir)
