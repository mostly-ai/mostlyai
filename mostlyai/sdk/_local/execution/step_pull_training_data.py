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

from collections.abc import Callable
from pathlib import Path

import pandas as pd

from mostlyai.sdk import _data as data
from mostlyai.sdk._data.base import ForeignKey, Schema
from mostlyai.sdk._data.conversions import create_container_from_connector
from mostlyai.sdk._data.file.utils import make_data_table_from_container
from mostlyai.sdk._data.smart_select import (
    ParentChildMatcher,
    encode_df,
    pre_training,
    prepare_training_data,
    store_model,
    train,
)
from mostlyai.sdk.domain import Connector, Generator, ModelType


def execute_train_child_parent_matchers(
    *,
    tgt_table_name: str,
    schema: Schema,
    workspace_dir: Path,
):
    tgt_data = schema.tables[tgt_table_name].read_data()

    non_ctx_relations = [rel for rel in schema.non_context_relations if rel.child.table == tgt_table_name]
    if not non_ctx_relations:
        # no non-context relations, so no parent-child matchers to train
        return

    tgt_table = schema.tables[tgt_table_name]
    tgt_foreign_keys = [fk.column for fk in tgt_table.foreign_keys]
    tgt_data_columns = [c for c in tgt_table.columns if c != tgt_table.primary_key and c not in tgt_foreign_keys]

    for non_ctx_relation in non_ctx_relations:
        parent_table = schema.tables[non_ctx_relation.parent.table]
        parent_foreign_keys = [fk.column for fk in parent_table.foreign_keys]
        parent_data_columns = [
            c for c in parent_table.columns if c != parent_table.primary_key and c not in parent_foreign_keys
        ]
        parent_table_name = non_ctx_relation.parent.table
        parent_data = schema.tables[parent_table_name].read_data()

        tgt_primary_key = schema.tables[tgt_table_name].primary_key
        tgt_parent_key = non_ctx_relation.child.column
        parent_primary_key = non_ctx_relation.parent.column

        smart_select_workspace_dir = workspace_dir / "SmartSelectModelStore"
        execute_train_child_parent_matcher(
            tgt_data=tgt_data,
            parent_data=parent_data,
            tgt_primary_key=tgt_primary_key,
            tgt_parent_key=tgt_parent_key,
            tgt_data_columns=tgt_data_columns,
            parent_primary_key=parent_primary_key,
            parent_data_columns=parent_data_columns,
            parent_table_name=parent_table_name,
            smart_select_workspace_dir=smart_select_workspace_dir,
        )


def execute_train_child_parent_matcher(
    *,
    tgt_data: pd.DataFrame,
    parent_data: pd.DataFrame,
    tgt_primary_key: str,
    tgt_parent_key: str,
    tgt_data_columns: list[str],
    parent_primary_key: str,
    parent_table_name: str,
    parent_data_columns: list[str],
    smart_select_workspace_dir: Path,
):
    smart_select_workspace_dir.mkdir(parents=True, exist_ok=True)

    tgt_data_columns = [
        c for c in tgt_data_columns if c != tgt_primary_key and c != tgt_parent_key and c in tgt_data.columns
    ]
    parent_data_columns = [c for c in parent_data_columns if c != parent_primary_key and c in parent_data.columns]

    # fit tgt encoders
    tgt_pre_training_dir = smart_select_workspace_dir / f"pre_training[{tgt_parent_key}]"
    pre_training(
        df=tgt_data,
        primary_key=tgt_primary_key,
        parent_key=tgt_parent_key,
        data_columns=tgt_data_columns,
        pre_training_dir=tgt_pre_training_dir,
    )

    # fit parent encoders
    parent_pre_training_dir = smart_select_workspace_dir / f"pre_training[{parent_table_name}]"
    if not parent_pre_training_dir.exists():
        pre_training(
            df=parent_data,
            primary_key=parent_primary_key,
            data_columns=parent_data_columns,
            pre_training_dir=parent_pre_training_dir,
        )
    else:
        print(f"Parent table `{parent_table_name}` pre-training already done, skipping")

    # encode tgt data
    tgt_encoded_data = encode_df(
        df=tgt_data,
        pre_training_dir=tgt_pre_training_dir,
        include_primary_key=False,
    )

    # encode parent data
    parent_encoded_data = encode_df(
        df=parent_data,
        pre_training_dir=parent_pre_training_dir,
    )

    # initialize child-parent matcher model
    parent_dim = parent_encoded_data.shape[1] - 1
    child_dim = tgt_encoded_data.shape[1] - 1
    hidden_dim = 32
    emb_dim = 8
    model = ParentChildMatcher(
        parent_dim=parent_dim,
        child_dim=child_dim,
        hidden_dim=hidden_dim,
        emb_dim=emb_dim,
    )

    # create positive and negative pairs for training
    parent_vecs, child_vecs, labels = prepare_training_data(
        df_parents_encoded=parent_encoded_data,
        df_children_encoded=tgt_encoded_data,
        parent_primary_key=parent_primary_key,
        children_foreign_key=tgt_parent_key,
        sample_size=1_000,
    )

    # train model
    train(
        model=model,
        parent_vecs=parent_vecs,
        child_vecs=child_vecs,
        labels=labels,
        do_plot_losses=False,
    )

    # store model
    store_model(model=model, smart_select_workspace_dir=smart_select_workspace_dir)

    print(f"Child-parent matcher model trained and stored for parent table: {parent_table_name}")


def execute_step_pull_training_data(
    *,
    generator: Generator,
    connectors: list[Connector],
    model_type: ModelType,
    target_table_name: str,
    workspace_dir: Path,
    update_progress: Callable,
) -> tuple[list[str], int]:
    schema = _create_training_schema(generator=generator, connectors=connectors)

    # fetch total rows
    tgt_table_total_rows = schema.tables[target_table_name].row_count
    # fetch columns
    tgt_table_columns = schema.tables[target_table_name].columns

    # fetch model_config
    tgt_table = next(t for t in generator.tables if t.name == target_table_name)
    if model_type == ModelType.language:
        model_config = tgt_table.language_model_configuration
    else:
        model_config = tgt_table.tabular_model_configuration

    # call PULL
    data.pull(
        tgt=target_table_name,
        schema=schema,
        model_type=model_type,
        max_sample_size=model_config.max_sample_size,
        workspace_dir=workspace_dir,
        update_progress=update_progress,
    )

    # handle parent-child matcher models training (for non-context relations)
    execute_train_child_parent_matchers(
        tgt_table_name=target_table_name,
        schema=schema,
        workspace_dir=workspace_dir,
    )
    return tgt_table_columns, tgt_table_total_rows


def _create_training_schema(generator: Generator, connectors: list[Connector]) -> Schema:
    tables = {}
    for table in generator.tables:
        # create DataContainer
        connector_id = table.source_connector_id
        connector = next(c for c in connectors if c.id == connector_id)
        container = create_container_from_connector(connector)
        container.set_location(table.location)
        # create DataTable
        data_table = make_data_table_from_container(container, lazy_fetch_primary_key=False)
        data_table.name = table.name
        data_table.primary_key = table.primary_key
        if table.columns:
            data_table.columns = [c.name for c in table.columns if c.included]
            data_table.encoding_types = {c.name: c.model_encoding_type for c in table.columns if c.included}
        data_table.is_output = False
        data_table.foreign_keys = [
            ForeignKey(column=fk.column, referenced_table=fk.referenced_table, is_context=fk.is_context)
            for fk in table.foreign_keys or []
        ]
        tables[table.name] = data_table
    return Schema(tables=tables)
