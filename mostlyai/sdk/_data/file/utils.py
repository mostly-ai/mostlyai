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
import re

from mostlyai.sdk._data.exceptions import MostlyDataException
from mostlyai.sdk._data.base import DataContainer, DataTable
from mostlyai.sdk._data.db.base import SqlAlchemyContainer
from mostlyai.sdk._data.file.base import (
    FileContainer,
    FileDataTable,
    get_file_name_and_type,
    FILE_TYPE_COMPRESSED,
    FileType,
)
from mostlyai.sdk._data.file.table.csv import CsvDataTable
from mostlyai.sdk._data.file.table.feather import FeatherDataTable
from mostlyai.sdk._data.file.table.json import JsonDataTable
from mostlyai.sdk._data.file.table.parquet import ParquetDataTable

_LOG = logging.getLogger(__name__)


FILE_EXT_DATA_TABLE_CLASS_MAP = {
    FileType.csv: CsvDataTable,
    FileType.tsv: CsvDataTable,
    FileType.parquet: ParquetDataTable,
    FileType.feather: FeatherDataTable,
    FileType.json: JsonDataTable,
}


def read_data_table_from_path(
    container_object: FileContainer, return_class: bool = False
) -> FileDataTable | type[FileDataTable]:
    # determine table_name
    table_name = container_object.path.absolute().name
    for ext in FILE_TYPE_COMPRESSED:
        table_name = re.sub(f"\\.{ext}$", "", table_name, flags=re.IGNORECASE)
    if "." in table_name:
        table_name = ".".join(table_name.split(".")[:-1])
    # determine file extension
    _LOG.info(f"detect data files for `{container_object.path}`")
    file_list = container_object.list_valid_files()
    _LOG.info(container_object.path)
    _LOG.info(file_list)
    _LOG.info(f"detected {len(file_list)} files: {file_list}")
    if len(file_list) == 0:
        raise MostlyDataException("No data files found.")
    _, file_type = get_file_name_and_type(file_list[0])
    data_table_cls = FILE_EXT_DATA_TABLE_CLASS_MAP.get(file_type)
    if return_class:
        return data_table_cls
    _LOG.info("create FileDataTable")
    data_table = data_table_cls(
        container=container_object,
        path=file_list,
        name=table_name,
    )
    return data_table


def make_data_table_from_container(container: DataContainer) -> DataTable:
    if isinstance(container, SqlAlchemyContainer):
        # handle DB containers
        data_table_class = container.table_class()
    elif isinstance(container, FileContainer):
        # handle local fs and bucket containers
        data_table_class = read_data_table_from_path(container, return_class=True)
    else:
        raise RuntimeError(f"Unknown container type: {type(container)}")
    return data_table_class(container=container)
