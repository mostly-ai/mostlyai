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
from pydoc import locate
from typing import Any

from mostlyai.sdk._data.db.base import SqlAlchemyContainer
from mostlyai.sdk._data.file.base import FileContainer
from mostlyai.sdk._data.file.container.bucket_based import BucketBasedContainer
from mostlyai.sdk.domain import Connector, ConnectorType

_LOG = logging.getLogger(__name__)

CONNECTOR_TYPE_CONTAINER_CLASS_MAP = {
    ConnectorType.mysql: "mostlyai.sdk._data.db.mysql.MysqlContainer",
    ConnectorType.postgres: "mostlyai.sdk._data.db.postgresql.PostgresqlContainer",
    ConnectorType.redshift: "mostlyai.sdk._data.db.redshift.RedshiftContainer",
    ConnectorType.mssql: "mostlyai.sdk._data.db.mssql.MssqlContainer",
    ConnectorType.oracle: "mostlyai.sdk._data.db.oracle.OracleContainer",
    ConnectorType.mariadb: "mostlyai.sdk._data.db.mysql.MariadbContainer",
    ConnectorType.snowflake: "mostlyai.sdk._data.db.snowflake.SnowflakeContainer",
    ConnectorType.bigquery: "mostlyai.sdk._data.db.bigquery.BigQueryContainer",
    ConnectorType.databricks: "mostlyai.sdk._data.db.databricks.DatabricksContainer",
    ConnectorType.hive: "mostlyai.sdk._data.db.hive.HiveContainer",
    ConnectorType.sqlite: "mostlyai.sdk._data.db.sqlite.SqliteContainer",
    ConnectorType.azure_storage: "mostlyai.sdk._data.file.container.azure.AzureBlobFileContainer",
    ConnectorType.google_cloud_storage: "mostlyai.sdk._data.file.container.gcs.GcsContainer",
    ConnectorType.s3_storage: "mostlyai.sdk._data.file.container.aws.AwsS3FileContainer",
    ConnectorType.file_upload: "mostlyai.sdk._data.file.base.LocalFileContainer",
}

CONNECTOR_TYPE_CONTAINER_PARAMS_CLASS_MAP = {
    ConnectorType.mysql: "mostlyai.sdk._data.metadata_objects.SqlAlchemyContainerParameters",
    ConnectorType.postgres: "mostlyai.sdk._data.metadata_objects.SqlAlchemyContainerParameters",
    ConnectorType.redshift: "mostlyai.sdk._data.metadata_objects.SqlAlchemyContainerParameters",
    ConnectorType.mssql: "mostlyai.sdk._data.metadata_objects.SqlAlchemyContainerParameters",
    ConnectorType.oracle: "mostlyai.sdk._data.metadata_objects.OracleContainerParameters",
    ConnectorType.mariadb: "mostlyai.sdk._data.metadata_objects.SqlAlchemyContainerParameters",
    ConnectorType.snowflake: "mostlyai.sdk._data.metadata_objects.SnowflakeContainerParameters",
    ConnectorType.bigquery: "mostlyai.sdk._data.metadata_objects.BigQueryContainerParameters",
    ConnectorType.databricks: "mostlyai.sdk._data.metadata_objects.DatabricksContainerParameters",
    ConnectorType.hive: "mostlyai.sdk._data.metadata_objects.SqlAlchemyContainerParameters",
    ConnectorType.sqlite: "mostlyai.sdk._data.metadata_objects.SqlAlchemyContainerParameters",
    ConnectorType.azure_storage: "mostlyai.sdk._data.metadata_objects.AzureBlobFileContainerParameters",
    ConnectorType.google_cloud_storage: "mostlyai.sdk._data.metadata_objects.GcsContainerParameters",
    ConnectorType.s3_storage: "mostlyai.sdk._data.metadata_objects.AwsS3FileContainerParameters",
    ConnectorType.file_upload: "mostlyai.sdk._data.metadata_objects.LocalFileContainerParameters",
}


def convert_connector_params_to_container_params(connector: Connector) -> dict[str, Any]:
    """
    Merge `config`, `secrets` and `ssl` of the Connector into one dictionary
    and then validate it individually based on the connector type.
    """
    connector.config = connector.config or {}
    connector.secrets = connector.secrets or {}
    connector.ssl = connector.ssl or {}

    container_params_cls_path = CONNECTOR_TYPE_CONTAINER_PARAMS_CLASS_MAP.get(connector.type)
    container_params_pydantic_cls = locate(container_params_cls_path)

    container_params = connector.config | connector.secrets | connector.ssl
    container_params = container_params_pydantic_cls.model_validate(container_params).model_dump()
    return container_params


def create_container_from_connector(
    connector: Connector,
) -> SqlAlchemyContainer | BucketBasedContainer | FileContainer:
    container_cls_path = CONNECTOR_TYPE_CONTAINER_CLASS_MAP.get(connector.type)
    if not container_cls_path:
        raise ValueError("Unsupported connector type!")
    container_cls = locate(container_cls_path)
    container_params = convert_connector_params_to_container_params(connector)
    container = container_cls(**container_params)
    # Check if the container is accessible before __repr__ (workaround broken logic of sa and __repr__)
    is_accessible = container.is_accessible()
    _LOG.info(f"Container accessible: {is_accessible}")
    _LOG.info(f"Container created: {container}")
    return container
