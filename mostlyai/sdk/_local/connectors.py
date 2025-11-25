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

import logging
from io import BytesIO
from pathlib import Path

import pandas as pd
from fastapi import HTTPException

from mostlyai.sdk._data.auto_detect import auto_detect_encoding_types_and_pk
from mostlyai.sdk._data.base import Schema
from mostlyai.sdk._data.conversions import create_container_from_connector
from mostlyai.sdk._data.db.base import SqlAlchemyContainer
from mostlyai.sdk._data.file.utils import make_data_table_from_container, read_data_table_from_path
from mostlyai.sdk._data.metadata_objects import ColumnSchema, ConstraintSchema, TableSchema
from mostlyai.sdk._data.util.common import encrypt, get_passphrase, run_with_timeout_unsafe
from mostlyai.sdk._local.storage import write_connector_to_json
from mostlyai.sdk.domain import (
    Connector,
    ConnectorAccessType,
    ConnectorConfig,
    ConnectorDeleteDataConfig,
    ConnectorPatchConfig,
    ConnectorReadDataConfig,
    ConnectorType,
    ConnectorWriteDataConfig,
)

_LOG = logging.getLogger(__name__)


def create_connector(home_dir: Path, config: ConnectorConfig, test_connection: bool = True) -> Connector:
    config = encrypt_connector_config(config)
    connector = Connector(**config.model_dump())
    if test_connection:
        do_test_connection(connector)
    connector_dir = home_dir / "connectors" / connector.id
    write_connector_to_json(connector_dir, connector)
    return connector


def encrypt_connector_config(config: ConnectorConfig | ConnectorPatchConfig) -> ConnectorConfig | ConnectorPatchConfig:
    # mimic the encryption of secrets and ssl parameters in local mode
    attrs = (attr for attr in (config.secrets, config.ssl) if attr is not None)
    for attr in attrs:
        for k, v in attr.items():
            attr[k] = encrypt(v, get_passphrase())
    return config


def do_test_connection(connector: Connector) -> bool:
    # mimic the test connection service in local mode
    try:
        _ = create_container_from_connector(connector)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return True


def _data_table_from_connector_and_location(connector: Connector, location: str, is_output: bool):
    if connector.type == ConnectorType.file_upload:
        raise HTTPException(status_code=400, detail="Connector type FILE_UPLOAD is disallowed for this operation")
    container = create_container_from_connector(connector)
    meta = container.set_location(location)
    data_table = make_data_table_from_container(container, is_output=is_output)
    data_table.name = meta["table_name"] if hasattr(container, "dbname") else "data"
    return data_table


def read_data_from_connector(connector: Connector, config: ConnectorReadDataConfig) -> pd.DataFrame:
    if connector.access_type not in {ConnectorAccessType.read_data, ConnectorAccessType.write_data}:
        raise HTTPException(status_code=400, detail="Connector does not have read access")

    try:
        data_table = _data_table_from_connector_and_location(
            connector=connector, location=config.location, is_output=False
        )
        return data_table.read_data(limit=config.limit, shuffle=config.shuffle)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def write_data_to_connector(connector: Connector, config: ConnectorWriteDataConfig) -> None:
    if connector.access_type != ConnectorAccessType.write_data:
        raise HTTPException(status_code=400, detail="Connector does not have write access")

    try:
        data_table = _data_table_from_connector_and_location(
            connector=connector, location=config.location, is_output=True
        )
        df = pd.read_parquet(BytesIO(config.file))
        data_table.write_data(df, if_exists=config.if_exists.value.lower())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def delete_data_from_connector(connector: Connector, config: ConnectorDeleteDataConfig) -> None:
    if connector.access_type != ConnectorAccessType.write_data:
        raise HTTPException(status_code=400, detail="Connector does not have write access")

    try:
        data_table = _data_table_from_connector_and_location(
            connector=connector, location=config.location, is_output=True
        )
        data_table.drop()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def query_data_from_connector(connector: Connector, sql: str) -> pd.DataFrame:
    if connector.access_type not in {ConnectorAccessType.read_data, ConnectorAccessType.write_data}:
        raise HTTPException(status_code=400, detail="Connector does not have query access")

    try:
        data_container = create_container_from_connector(connector)
        return data_container.query(sql)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def _data_tables_to_table_schemas(
    schema: Schema,
    ordered_table_names: list[str],
    ordered_locations: list[str],
) -> list[TableSchema]:
    table_schemas = []
    for table_name, location in zip(ordered_table_names, ordered_locations):
        table = schema.tables[table_name]
        # auto-detect encoding types
        encoding_types, primary_key = auto_detect_encoding_types_and_pk(table)
        _LOG.info(f"auto-detected {encoding_types=} and {primary_key=} for table {table_name}")
        table.encoding_types |= encoding_types
        # fetch table schema
        columns = [
            ColumnSchema(
                name=col,
                original_data_type=str(table.dtypes[col].wrapped),
                default_model_encoding_type=table.encoding_types[col].value,
            )
            for col in table.columns
        ]
        relations = schema.subset(relations_to=[table_name]).relations
        constraints = [
            ConstraintSchema(foreign_key=rel.child.column, referenced_table=rel.parent.table) for rel in relations
        ]
        # fetch table row count - but time-box to 10secs
        total_rows = run_with_timeout_unsafe(lambda: table.row_count, timeout=10)
        # instantiate TableSchema
        table_schema = TableSchema(
            name=table_name,
            total_rows=total_rows,
            primary_key=table.primary_key or primary_key,
            columns=columns,
            constraints=constraints,
            location=location,
        )
        table_schemas.append(table_schema)

    return table_schemas


def _collapse_table_schemas(table_schemas: list[TableSchema]) -> TableSchema:
    n = len(table_schemas)
    assert n > 0
    if n == 1:
        return table_schemas[0]
    else:
        table_schema = table_schemas[0]
        table_schema.children = table_schemas[1:]
        return table_schema


def fetch_location_schema(connector: Connector, location: str, include_children: bool = False) -> TableSchema:
    container = create_container_from_connector(connector)
    meta = container.set_location(location)
    if isinstance(container, SqlAlchemyContainer):
        db_schema, table_name = meta["db_schema"], meta["table_name"]
        if include_children:
            container.filtered_tables = container.get_children(table_name)
        else:
            container.filtered_tables = [table_name]
        container.fetch_schema()
        schema = container.schema
        ordered_table_names = container.filtered_tables
        ordered_locations = [".".join([db_schema, table_name]) for table_name in ordered_table_names]
    else:
        location = meta["location"]
        table = read_data_table_from_path(container)
        schema = Schema(tables={table.name: table})
        ordered_table_names = [table.name]
        ordered_locations = [location]

    table_schemas = _data_tables_to_table_schemas(
        schema=schema,
        ordered_table_names=ordered_table_names,
        ordered_locations=ordered_locations,
    )
    table_schema = _collapse_table_schemas(table_schemas)
    return table_schema
