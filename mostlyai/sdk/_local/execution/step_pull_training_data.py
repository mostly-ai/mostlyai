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

from mostlyai.sdk import _data as data
from mostlyai.sdk._data.base import ForeignKey, Schema
from mostlyai.sdk._data.conversions import create_container_from_connector
from mostlyai.sdk._data.db.base import SqlAlchemyTable
from mostlyai.sdk._data.file.utils import make_data_table_from_container
from mostlyai.sdk.domain import Connector, Generator, ModelType


def execute_step_pull_training_data(
    *,
    generator: Generator,
    connectors: list[Connector],
    model_type: ModelType,
    target_table_name: str,
    workspace_dir: Path,
    update_progress: Callable,
) -> tuple[list[str], int]:
    schema = create_training_schema(generator=generator, connectors=connectors)

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
    return tgt_table_columns, tgt_table_total_rows


def create_training_schema(generator: Generator, connectors: list[Connector]) -> Schema:
    tables = {}
    for table in generator.tables:
        # create DataContainer
        connector_id = table.source_connector_id
        connector = next(c for c in connectors if c.id == connector_id)
        container = create_container_from_connector(connector)
        meta = container.set_location(table.location)
        # create DataTable
        data_table = make_data_table_from_container(container, lazy_fetch_primary_key=False)
        # preserve actual database table name for database containers
        if isinstance(data_table, SqlAlchemyTable) and meta and "table_name" in meta:
            data_table.name_in_db = meta["table_name"]
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
