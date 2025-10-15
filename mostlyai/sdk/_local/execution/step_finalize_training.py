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


from pathlib import Path

import pandas as pd
from torch import Generator

from mostlyai.sdk._data.base import Schema
from mostlyai.sdk._data.fk_models import (
    ParentChildMatcher,
    analyze_df,
    encode_df,
    get_cardinalities,
    prepare_training_data,
    pull_fk_training_data,
    set_seeds,
    store_fk_model,
    train,
)
from mostlyai.sdk._local.execution.step_pull_training_data import create_training_schema
from mostlyai.sdk.domain import Connector


def execute_train_fk_models_for_single_table(
    *,
    tgt_table_name: str,
    schema: Schema,
    fk_models_workspace_dir: Path,
    max_parent_sample_size: int = 10000,
    max_children_per_parent: int = 1,
):
    non_ctx_relations = [rel for rel in schema.non_context_relations if rel.child.table == tgt_table_name]
    if not non_ctx_relations:
        # no non-context relations, so no parent-child matchers to train
        return

    fk_models_workspace_dir.mkdir(parents=True, exist_ok=True)

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

        tgt_primary_key = schema.tables[tgt_table_name].primary_key
        tgt_parent_key = non_ctx_relation.child.column
        parent_primary_key = non_ctx_relation.parent.column

        parent_data, tgt_data = pull_fk_training_data(
            schema=schema,
            non_ctx_relation=non_ctx_relation,
            max_parent_sample_size=max_parent_sample_size,
            max_children_per_parent=max_children_per_parent,
        )

        # Skip if no data was pulled
        if parent_data is None or tgt_data is None:
            continue

        execute_train_fk_models_for_single_relation(
            tgt_data=tgt_data,
            parent_data=parent_data,
            tgt_primary_key=tgt_primary_key,
            tgt_parent_key=tgt_parent_key,
            tgt_data_columns=tgt_data_columns,
            parent_primary_key=parent_primary_key,
            parent_data_columns=parent_data_columns,
            parent_table_name=parent_table_name,
            fk_models_workspace_dir=fk_models_workspace_dir,
        )


def execute_train_fk_models_for_single_relation(
    *,
    tgt_data: pd.DataFrame,
    parent_data: pd.DataFrame,
    tgt_primary_key: str,
    tgt_parent_key: str,
    tgt_data_columns: list[str],
    parent_primary_key: str,
    parent_table_name: str,
    parent_data_columns: list[str],
    fk_models_workspace_dir: Path,
):
    fk_models_workspace_dir.mkdir(parents=True, exist_ok=True)

    tgt_data_columns = [
        c for c in tgt_data_columns if c != tgt_primary_key and c != tgt_parent_key and c in tgt_data.columns
    ]
    parent_data_columns = [c for c in parent_data_columns if c != parent_primary_key and c in parent_data.columns]

    # fit tgt encoders
    tgt_pre_training_dir = fk_models_workspace_dir / f"pre_training[{tgt_parent_key}]"
    analyze_df(
        df=tgt_data,
        primary_key=tgt_primary_key,
        parent_key=tgt_parent_key,
        data_columns=tgt_data_columns,
        pre_training_dir=tgt_pre_training_dir,
    )

    # fit parent encoders
    parent_pre_training_dir = fk_models_workspace_dir / f"pre_training[{parent_table_name}]"
    if not parent_pre_training_dir.exists():
        analyze_df(
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
    parent_cardinalities = get_cardinalities(pre_training_dir=parent_pre_training_dir)
    tgt_cardinalities = get_cardinalities(pre_training_dir=tgt_pre_training_dir)
    model = ParentChildMatcher(
        parent_cardinalities=parent_cardinalities,
        child_cardinalities=tgt_cardinalities,
    )

    # create positive and negative pairs for training
    parent_pd, child_pd, labels_pd = prepare_training_data(
        parent_encoded_data=parent_encoded_data,
        tgt_encoded_data=tgt_encoded_data,
        parent_primary_key=parent_primary_key,
        tgt_parent_key=tgt_parent_key,
        sample_size=None,  # No additional sampling - already done in data pull phase
    )

    # train model
    train(
        model=model,
        parent_pd=parent_pd,
        child_pd=child_pd,
        labels=labels_pd,
        do_plot_losses=False,
    )

    # store model
    store_fk_model(model=model, tgt_parent_key=tgt_parent_key, fk_models_workspace_dir=fk_models_workspace_dir)

    print(f"Child-parent matcher model trained and stored for parent table: {parent_table_name}")


def execute_step_finalize_training(
    *,
    generator: Generator,
    connectors: list[Connector],
    job_workspace_dir: Path,
    max_parent_sample_size: int = 10000,
    max_children_per_parent: int = 1,
):
    set_seeds(42)

    schema = create_training_schema(generator=generator, connectors=connectors)
    for tgt_table_name in schema.tables:
        fk_models_workspace_dir = job_workspace_dir / "FKModelsStore" / tgt_table_name
        execute_train_fk_models_for_single_table(
            tgt_table_name=tgt_table_name,
            schema=schema,
            fk_models_workspace_dir=fk_models_workspace_dir,
            max_parent_sample_size=max_parent_sample_size,
            max_children_per_parent=max_children_per_parent,
        )
