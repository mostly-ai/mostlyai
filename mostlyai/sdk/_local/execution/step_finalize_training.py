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
import time
import traceback
from pathlib import Path

from mostlyai.sdk._data.base import NonContextRelation, Schema
from mostlyai.sdk._data.non_context import (
    ParentChildMatcher,
    analyze_df,
    encode_df,
    get_cardinalities,
    prepare_training_data_for_cardinality_model,
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


def execute_train_fk_models_for_single_table(
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

        execute_train_fk_model_for_single_non_context_relation(
            tgt_table_name=tgt_table_name,
            non_ctx_relation=non_ctx_relation,
            schema=schema,
            fk_model_workspace_dir=fk_model_workspace_dir,
        )

        # report progress after each FK model training
        update_progress(advance=1)


def execute_train_fk_model_for_single_non_context_relation(
    *,
    tgt_table_name: str,
    non_ctx_relation: NonContextRelation,
    schema: Schema,
    fk_model_workspace_dir: Path,
):
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

    if parent_data.empty or tgt_data.empty:
        return

    tgt_data_columns = [c for c in tgt_data.columns if c != tgt_parent_key]
    parent_data_columns = [c for c in parent_data.columns if c != parent_primary_key]

    fk_model_workspace_dir.mkdir(parents=True, exist_ok=True)

    tgt_stats_dir = fk_model_workspace_dir / "tgt-stats"
    analyze_df(
        df=tgt_data,
        parent_key=tgt_parent_key,
        data_columns=tgt_data_columns,
        stats_dir=tgt_stats_dir,
    )

    parent_stats_dir = fk_model_workspace_dir / "parent-stats"
    analyze_df(
        df=parent_data,
        primary_key=parent_primary_key,
        data_columns=parent_data_columns,
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

    # Train Cardinality Model using engine BEFORE saving FK model
    # This ensures we only save FK models if cardinality training also succeeds
    _LOG.info(f"Training Cardinality Model with engine for {tgt_table_name}.{tgt_parent_key}")

    # Prepare parent data with children count column
    parent_data_with_counts = prepare_training_data_for_cardinality_model(
        parent_data=parent_data,
        tgt_data=tgt_data,
        parent_primary_key=parent_primary_key,
        tgt_parent_key=tgt_parent_key,
    )

    # Define cardinality workspace
    cardinality_workspace_dir = fk_model_workspace_dir / "cardinality_engine"
    cardinality_workspace_dir.mkdir(parents=True, exist_ok=True)

    # Import engine here to avoid premature loading
    import mostlyai.engine as engine

    # Run engine.split - Split data into train/validation sets
    _LOG.info("Splitting cardinality training data")
    engine.split(
        tgt_data=parent_data_with_counts,
        workspace_dir=cardinality_workspace_dir,
        update_progress=lambda **kwargs: None,
    )

    # Run engine.analyze - Analyze training data statistics
    _LOG.info("Analyzing cardinality training data")
    engine.analyze(
        workspace_dir=cardinality_workspace_dir,
        update_progress=lambda **kwargs: None,
    )

    # Run engine.encode - Encode training data
    _LOG.info("Encoding cardinality training data")
    engine.encode(
        workspace_dir=cardinality_workspace_dir,
        update_progress=lambda **kwargs: None,
    )

    # Run engine.train - Train model on __children_count column
    _LOG.info("Training cardinality model with engine")
    engine.train(
        model="MOSTLY_AI/Small",
        workspace_dir=cardinality_workspace_dir,
        update_progress=lambda **kwargs: None,
    )

    _LOG.info(f"Successfully trained engine-based cardinality model at {cardinality_workspace_dir}")

    # Save FK model only after cardinality model training succeeds
    store_fk_model(
        model=model,
        fk_model_workspace_dir=fk_model_workspace_dir,
    )

    _LOG.info(
        f"Trained FK model and Cardinality model for relation: {tgt_table_name}.{tgt_parent_key} -> {parent_table_name}.{parent_primary_key} | "
        f"time: {time.time() - t0:.2f}s | models saved: {fk_model_workspace_dir}"
    )


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
                execute_train_fk_models_for_single_table(
                    tgt_table_name=tgt_table_name,
                    schema=schema,
                    fk_models_workspace_dir=fk_models_workspace_dir,
                    update_progress=progress.update,
                )
            except Exception as e:
                _LOG.error(f"FK model training failed for table {tgt_table_name}: {e}\n{traceback.format_exc()}")
                continue
