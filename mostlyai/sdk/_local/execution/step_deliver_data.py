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

from mostlyai.sdk._data.base import DataContainer, Schema
from mostlyai.sdk._data.conversions import create_container_from_connector
from mostlyai.sdk._data.db.base import SqlAlchemyContainer
from mostlyai.sdk._data.file.container.bucket_based import BucketBasedContainer
from mostlyai.sdk._data.file.table.parquet import ParquetDataTable
from mostlyai.sdk._data.file.utils import make_data_table_from_container
from mostlyai.sdk._data.push import push_data, push_data_by_copying
from mostlyai.sdk.domain import Connector, Generator, SyntheticDatasetDelivery


def execute_step_deliver_data(
    *,
    generator: Generator,
    delivery: SyntheticDatasetDelivery,
    connector: Connector | None,
    schema: Schema,
    job_workspace_dir: Path,
):
    # skip DELIVER_DATA step if no destination connector is provided
    if connector is None:
        return

    # create destination container
    container = create_container_from_connector(connector)
    container.set_location(delivery.location)

    overwrite_tables = delivery.overwrite_tables
    for table_name in schema.tables:
        local_path = job_workspace_dir / "FinalizedSyntheticData" / table_name / "parquet"
        if isinstance(container, BucketBasedContainer):
            bucket_path = container.path / table_name
            push_data_by_copying(
                source=local_path,
                destination=bucket_path,
                overwrite_tables=overwrite_tables,
            )
        elif isinstance(container, SqlAlchemyContainer):
            src_table = ParquetDataTable(path=local_path)
            table = _create_destination_table(table_name, generator, container)
            push_data(
                source=src_table,
                destination=table,
                schema=schema,
                overwrite_tables=overwrite_tables,
            )
        else:
            raise ValueError(f"Unsupported destination container type: {container}")


def _create_destination_table(
    table_name: str,
    generator: Generator,
    data_container: DataContainer,
):
    # create destination table
    source_table = next(t for t in generator.tables if t.name == table_name)
    data_table = make_data_table_from_container(data_container)
    data_table.name = source_table.name
    data_table.primary_key = source_table.primary_key
    data_table.columns = [c.name for c in source_table.columns if c.included]
    data_table.encoding_types = {c.name: c.model_encoding_type for c in source_table.columns if c.included}
    data_table.is_output = True
    return data_table
