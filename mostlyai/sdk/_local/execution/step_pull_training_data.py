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

from _AI_SMART_SELECT.smart_select import (
    ParentChildMatcher,
    encode_df,
    load_model,
    pre_training,
    prepare_training_data,
    store_model,
    train,
)
from mostlyai.sdk import _data as data
from mostlyai.sdk._data.base import ForeignKey, Schema
from mostlyai.sdk._data.conversions import create_container_from_connector
from mostlyai.sdk._data.file.utils import make_data_table_from_container
from mostlyai.sdk.domain import Connector, Generator, ModelType


class SmartSelect:
    def __init__(self):
        pass

    def execute_train_smart_select_model(self, *, tgt_table_name: str, schema: Schema, workspace_dir: Path):
        tgt_data = schema.tables[tgt_table_name].read_data()

        non_ctx_relations = [rel for rel in schema.non_context_relations if rel.child.table == tgt_table_name]
        if not non_ctx_relations:
            return
        assert len(non_ctx_relations) == 1
        non_ctx_relation = non_ctx_relations[0]

        non_ctx_table_name = non_ctx_relation.parent.table
        non_ctx_data = schema.tables[non_ctx_table_name].read_data()

        tgt_primary_key = schema.tables[tgt_table_name].primary_key
        tgt_non_context_key = non_ctx_relation.child.column
        non_ctx_primary_key = non_ctx_relation.parent.column

        smart_select_workspace_dir = workspace_dir / "smart_select"
        self.execute_train_smart_select_model_2(
            tgt_data=tgt_data,
            non_ctx_data=non_ctx_data,
            tgt_primary_key=tgt_primary_key,
            tgt_non_context_key=tgt_non_context_key,
            non_ctx_primary_key=non_ctx_primary_key,
            smart_select_workspace_dir=smart_select_workspace_dir,
        )

    def execute_train_smart_select_model_2(
        self,
        *,
        tgt_data: pd.DataFrame,
        non_ctx_data: pd.DataFrame,
        tgt_primary_key: str,
        tgt_non_context_key: str,
        non_ctx_primary_key: str,
        smart_select_workspace_dir: Path,
    ):
        smart_select_workspace_dir.mkdir(parents=True, exist_ok=True)

        # fit tgt encoders
        tgt_pre_training_dir = smart_select_workspace_dir / "pre_training[tgt]"
        pre_training(
            df=tgt_data,
            primary_key=tgt_primary_key,
            foreign_key=tgt_non_context_key,
            pre_training_dir=tgt_pre_training_dir,
        )

        # fit non_ctx encoders
        non_ctx_pre_training_dir = smart_select_workspace_dir / "pre_training[non_ctx]"
        pre_training(
            df=non_ctx_data,
            primary_key=non_ctx_primary_key,
            pre_training_dir=non_ctx_pre_training_dir,
        )

        # encode tgt data
        tgt_encoded_data = encode_df(
            df=tgt_data,
            pre_training_dir=tgt_pre_training_dir,
            drop_primary_key=True,
        )

        # encode non_ctx data
        non_ctx_encoded_data = encode_df(
            df=non_ctx_data,
            pre_training_dir=non_ctx_pre_training_dir,
        )

        # make discriminator
        parent_dim = non_ctx_encoded_data.shape[1] - 1
        child_dim = tgt_encoded_data.shape[1] - 1
        hidden_dim = 32
        emb_dim = 8
        model = ParentChildMatcher(
            parent_dim=parent_dim,
            child_dim=child_dim,
            hidden_dim=hidden_dim,
            emb_dim=emb_dim,
        )

        # 1st training: random negative sampling
        parent_vecs, child_vecs, labels = prepare_training_data(
            df_parent_encoded=non_ctx_encoded_data,
            df_child_encoded=tgt_encoded_data,
            parent_primary_key=non_ctx_primary_key,
            child_foreign_key=tgt_non_context_key,
            n_children=1_000,
            n_false_parents=1,
            negative_sampling_strategy="random",
        )
        train(model, parent_vecs, child_vecs, labels, do_plot_losses=False)

        # store / load model
        store_model(model, smart_select_workspace_dir)
        model = load_model(smart_select_workspace_dir)

        # 2nd training: hard negative sampling
        parent_vecs, child_vecs, labels = prepare_training_data(
            df_parent_encoded=non_ctx_encoded_data,
            df_child_encoded=tgt_encoded_data,
            parent_primary_key=non_ctx_primary_key,
            child_foreign_key=tgt_non_context_key,
            n_children=1_000,
            n_false_parents=1,
            negative_sampling_strategy="hard",
            model=model,
        )
        train(model, parent_vecs, child_vecs, labels, do_plot_losses=False)

        # store model
        store_model(model, smart_select_workspace_dir)

        print("SmartSelect model trained and stored")


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

    # handle SmartSelect model training
    smart_select = SmartSelect()
    smart_select.execute_train_smart_select_model(
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
