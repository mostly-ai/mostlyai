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

import csv
import logging

import pandas as pd
import pyarrow.csv as pa_csv
import pyarrow.dataset as ds
import smart_open

from mostlyai.sdk._data.file.base import (
    FILE_DATA_TABLE_LAZY_INIT_FIELDS,
    FileContainer,
    FileDataTable,
    LocalFileContainer,
)

CSV_DATA_TABLE_LAZY_INIT_FIELDS = FILE_DATA_TABLE_LAZY_INIT_FIELDS + [
    "delimiter",
]

_LOG = logging.getLogger(__name__)


class CsvDataTable(FileDataTable):
    DATA_TABLE_TYPE = "csv"
    LAZY_INIT_FIELDS = frozenset(CSV_DATA_TABLE_LAZY_INIT_FIELDS)
    IS_WRITE_APPEND_ALLOWED = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delimiter: str | None = None

    @classmethod
    def container_class(cls) -> type["FileContainer"]:
        return LocalFileContainer

    def _get_delimiter(self) -> str:
        try:
            # only use first file to determine CSV delimiter for all files
            file = self.container.list_valid_files()[0]  # AnyPath
            # substitute scheme prefix to work with smart_open (e.g. Azure in particular)
            file = f"{self.container.delimiter_prefix}{str(file).split('//')[-1]}"
            header = smart_open.open(
                file,
                "r",
                errors="backslashreplace",
                transport_params=self.container.transport_params,
            ).readline()
            sniffer = csv.Sniffer()
            try:
                delimiter = sniffer.sniff(header, ",;|\t' '").delimiter
            except csv.Error:
                # happens for example for single column CSV files
                delimiter = ","
            return delimiter
        except Exception as err:
            _LOG.warning(f"{err=} of {type(err)=}, defaulting to ',' delimiter")
            return ","

    def _get_dataset_format(self, **kwargs):
        if "delimiter" in kwargs:
            delimiter = kwargs.get("delimiter")
        else:
            delimiter = self.delimiter
        fmt = ds.CsvFileFormat(
            parse_options=pa_csv.ParseOptions(
                delimiter=delimiter,
                # silently drop any invalid rows
                invalid_row_handler=lambda x: "skip",
            ),
            # use 100MB to increase reliability of dtype detection;
            # if e.g. dtype is detected as int64 and then later on in the file a float occurs,
            # an error is raised; the same issue also occurs if multiple CSV files are provided
            # with the first file consisting of integers and the others of floats; so, there still
            # might be scenarios where errors occur due to dtype mismatch in CSV chunks; in these
            # cases we shall advise to convert the source data to Parquet
            read_options=pa_csv.ReadOptions(block_size=100 * 1024 * 1024),
            # add additional formats for datetime conversion
            convert_options=pa_csv.ConvertOptions(
                timestamp_parsers=[
                    pa_csv.ISO8601,
                    "%m/%d/%Y %H:%M:%S",
                    "%m/%d/%Y %H:%M",
                    "%m/%d/%Y",
                ]
            ),
        )
        return fmt

    def _lazy_fetch(self, item: str) -> None:
        if item == "delimiter":
            self.delimiter = self._get_delimiter()
        else:
            super()._lazy_fetch(item)
            return

    def write_data(self, df: pd.DataFrame, if_exists: str = "append", **kwargs):
        mode = self.handle_if_exists(if_exists)
        df.to_csv(
            self.container.path_str,
            mode=mode,
            # write the header only during an initial "write", not during "append"
            header=(mode == "w"),
            storage_options=self.container.storage_options,
            index=False,
        )
