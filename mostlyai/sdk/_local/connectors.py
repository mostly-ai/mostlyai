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

from pathlib import Path

from fastapi import HTTPException

from mostlyai.sdk._data.conversions import create_container_from_connector
from mostlyai.sdk._data.util.common import encrypt, get_passphrase
from mostlyai.sdk._local.storage import write_connector_to_json
from mostlyai.sdk.domain import ConnectorConfig, Connector, ConnectorPatchConfig


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
