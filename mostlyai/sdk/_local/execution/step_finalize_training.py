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
from pathlib import Path

from mostlyai.sdk._data.base import NonContextRelation, Schema
from mostlyai.sdk._data.non_context import (
    ParentChildMatcher,
    analyze_df,
    encode_df,
    get_cardinalities,
    prepare_training_data,
    pull_fk_model_training_data,
    safe_name,
    store_fk_model,
    train,
)
from mostlyai.sdk._local.execution.step_pull_training_data import create_training_schema
from mostlyai.sdk.domain import Connector, Generator

_LOG = logging.getLogger(__name__)


def execute_train_fk_models_for_single_table(
    *,
    tgt_table_name: str,
    schema: Schema,
    fk_models_workspace_dir: Path,
):
    non_ctx_relations = [rel for rel in schema.non_context_relations if rel.child.table == tgt_table_name]
    if not non_ctx_relations:
        # no non-context relations, so no parent-child matchers to train
        return

    fk_models_workspace_dir.mkdir(parents=True, exist_ok=True)

    for non_ctx_relation in non_ctx_relations:
        tgt_parent_key = non_ctx_relation.child.column
        fk_model_workspace_dir = fk_models_workspace_dir / safe_name(tgt_parent_key)

        execute_train_fk_model_for_single_relation(
            tgt_table_name=tgt_table_name,
            non_ctx_relation=non_ctx_relation,
            schema=schema,
            fk_model_workspace_dir=fk_model_workspace_dir,
        )


def execute_train_fk_model_for_single_relation(
    *,
    tgt_table_name: str,
    non_ctx_relation: NonContextRelation,
    schema: Schema,
    fk_model_workspace_dir: Path,
):
    tgt_table = schema.tables[tgt_table_name]
    tgt_primary_key = tgt_table.primary_key
    tgt_parent_key = non_ctx_relation.child.column
    tgt_foreign_keys = [fk.column for fk in tgt_table.foreign_keys]
    tgt_data_columns = [c for c in tgt_table.columns if c != tgt_table.primary_key and c not in tgt_foreign_keys]

    parent_table = schema.tables[non_ctx_relation.parent.table]
    parent_primary_key = non_ctx_relation.parent.column
    parent_foreign_keys = [fk.column for fk in parent_table.foreign_keys]
    parent_data_columns = [
        c for c in parent_table.columns if c != parent_table.primary_key and c not in parent_foreign_keys
    ]
    parent_table_name = non_ctx_relation.parent.table

    parent_data, tgt_data = pull_fk_model_training_data(
        tgt_table=tgt_table,
        tgt_parent_key=tgt_parent_key,
        parent_table=parent_table,
        parent_primary_key=parent_primary_key,
    )

    if parent_data.empty or tgt_data.empty:
        # no data to train matcher model, so skip
        return

    fk_model_workspace_dir.mkdir(parents=True, exist_ok=True)

    tgt_stats_dir = fk_model_workspace_dir / "tgt-stats"
    analyze_df(
        df=tgt_data,
        primary_key=tgt_primary_key,
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

    parent_pd, child_pd, labels_pd = prepare_training_data(
        parent_encoded_data=parent_encoded_data,
        tgt_encoded_data=tgt_encoded_data,
        parent_primary_key=parent_primary_key,
        tgt_parent_key=tgt_parent_key,
        sample_size=None,  # no additional sampling - already done in data pull phase
    )

    train(
        model=model,
        parent_pd=parent_pd,
        child_pd=child_pd,
        labels=labels_pd,
    )

    store_fk_model(model=model, fk_model_workspace_dir=fk_model_workspace_dir)

    _LOG.info(f"Child-parent matcher model trained and stored for parent table: {parent_table_name}")


def execute_step_finalize_training(
    *,
    generator: Generator,
    connectors: list[Connector],
    job_workspace_dir: Path,
):
    schema = create_training_schema(generator=generator, connectors=connectors)
    for tgt_table_name in schema.tables:
        fk_models_workspace_dir = job_workspace_dir / "FKModelsStore" / tgt_table_name
        try:
            execute_train_fk_models_for_single_table(
                tgt_table_name=tgt_table_name,
                schema=schema,
                fk_models_workspace_dir=fk_models_workspace_dir,
            )
        except Exception as e:
            _LOG.error(f"FK model training failed for table {tgt_table_name}: {e}")
            continue
