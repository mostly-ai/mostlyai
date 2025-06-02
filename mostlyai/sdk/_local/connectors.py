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

from io import BytesIO
from pathlib import Path

import pandas as pd
from fastapi import HTTPException

from mostlyai.sdk._data.conversions import create_container_from_connector
from mostlyai.sdk._data.file.utils import make_data_table_from_container
from mostlyai.sdk._data.util.common import encrypt, get_passphrase
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
