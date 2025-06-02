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

import functools
import logging
import time
from collections.abc import Iterable
from typing import Any

import pandas as pd
import pyarrow.dataset as ds
import smart_open
from pyarrow import json as pa_json

from mostlyai.sdk._data.base import order_df_by
from mostlyai.sdk._data.dtype import (
    coerce_dtypes_by_encoding,
    is_date_dtype,
    is_timestamp_dtype,
    pyarrow_to_pandas_map,
)
from mostlyai.sdk._data.file.base import FileContainer, FileDataTable, LocalFileContainer
from mostlyai.sdk._data.util.common import OrderBy

_LOG = logging.getLogger(__name__)


class JsonDataTable(FileDataTable):
    DATA_TABLE_TYPE = "json"
    # append is only supported when to_json(..., orient="records", lines=True)
    IS_WRITE_APPEND_ALLOWED = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def container_class(cls) -> type["FileContainer"]:
        return LocalFileContainer

    def read_data(
        self,
        where: dict[str, Any] | None = None,
        limit: int | None = None,
        columns: list[str] | None = None,
        shuffle: bool | None = False,
        order_by: OrderBy | None = None,
        do_coerce_dtypes: bool | None = False,
    ) -> pd.DataFrame:
        t0 = time.time()
        if where:
            filters = []
            for c, v in where.items():
                # make sure values is a list of unique values
                values = list(set(v)) if (isinstance(v, Iterable) and not isinstance(v, str)) else [v]
                filters.append(ds.field(c).isin(values))
            filter = functools.reduce(lambda x, y: x & y, filters)
        else:
            filter = ds.scalar(True)
        files = [f"{self.container.path_prefix}{file}" for file in self.container.valid_files_without_scheme]
        df = pd.concat(
            [
                pa_json.read_json(
                    smart_open.open(
                        file,
                        "rb",
                        transport_params=self.container.transport_params,
                    )
                )
                .filter(filter)
                .to_pandas(
                    # convert to pyarrow DTypes
                    types_mapper=pyarrow_to_pandas_map.get,
                    # reduce memory of conversion
                    # see https://arrow.apache.org/docs/python/pandas.html#reducing-memory-use-in-table-to-pandas
                    split_blocks=True,
                )
                for file in files
            ],
            axis=0,
        )
        if columns:
            df = df[columns]
        if shuffle:
            df = df.sample(frac=1)
        if limit is not None:
            df = df.head(limit)
        if order_by:
            df = order_df_by(df, order_by)
        if do_coerce_dtypes:
            df = coerce_dtypes_by_encoding(df, self.encoding_types)
        df = df.reset_index(drop=True)
        _LOG.info(f"read {self.DATA_TABLE_TYPE} data `{self.name}` {df.shape} in {time.time() - t0:.2f}s")
        return df

    @functools.cached_property
    def row_count(self) -> int:
        # Note: this currently reads all data; optimize later
        return len(self.read_data())

    def _get_columns(self):
        # Note: this currently reads all data; optimize later
        return list(self.read_data().columns)

    def _get_dataset_format(self) -> ds.FileFormat:
        return ds.JsonFileFormat()

    def fetch_dtypes(self) -> dict[str, Any]:
        # Note: this currently reads all data; optimize later
        return self.read_data().dtypes.to_dict()

    def write_data(self, df: pd.DataFrame, if_exists: str = "append", **kwargs):
        # Convert to ISO format so that pyarrow.json.read_json can auto-detect these
        for c in df:
            if is_date_dtype(df[c]):
                df[c] = df[c].dt.strftime("%Y-%m-%d")
            elif is_timestamp_dtype(df[c]):
                # we need to strip off any milliseconds
                df[c] = (
                    df[c]
                    .dt.tz_localize(None)
                    .astype("timestamp[us][pyarrow]")
                    .dt.strftime("%Y-%m-%d %H:%M:%S")
                    .str[:-7]
                )
        mode = self.handle_if_exists(if_exists)
        df.to_json(self.container.path_str, orient="records", lines=True, mode=mode)
        # raise MostlyException("write to cloud buckets not yet supported")
