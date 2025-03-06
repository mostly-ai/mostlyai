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


import pandas as pd
import pyarrow as pa
import pytest

from mostlyai.sdk._data.dtype import (
    is_boolean_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_string_dtype,
    is_timestamp_dtype,
)
from mostlyai.sdk._data.file.table.json import JsonDataTable


@pytest.mark.parametrize("file_name", ["sample.json", "sample.json.gz"])
@pytest.mark.parametrize("sample_parquet_file", ["pyarrow"], indirect=True)
def test_read_write_data(tmp_path, sample_parquet_file, file_name):
    sample_data = pd.read_parquet(sample_parquet_file, dtype_backend="pyarrow")
    sample_data["date"] = sample_data["date"].astype(pd.ArrowDtype(pa.date32()))
    # write
    table1 = JsonDataTable(path=tmp_path / file_name, is_output=True)
    table1.write_data(sample_data)
    # read
    table2 = JsonDataTable(path=tmp_path / file_name, is_output=False)
    data = table2.read_data()
    chunk_df = next(table2.read_chunks())  # assuming a single chunk, due to a small size
    # compare data
    assert data.shape == sample_data.shape
    assert chunk_df.shape == sample_data.shape
    # check dtypes from data
    assert is_integer_dtype(data["id"])
    assert is_boolean_dtype(data["bool"])
    assert is_integer_dtype(data["int"])
    assert is_float_dtype(data["float"])
    assert is_timestamp_dtype(data["date"])
    assert is_timestamp_dtype(data["ts_s"])
    assert is_timestamp_dtype(data["ts_ns"])
    assert is_timestamp_dtype(data["ts_tz"])
    assert is_string_dtype(data["text"])
    # check dtypes from meta-data
    dtypes = table2.dtypes
    assert pd.api.types.is_bool_dtype(dtypes["bool"].wrapped)
    assert pd.api.types.is_integer_dtype(dtypes["int"].wrapped)
    assert pd.api.types.is_float_dtype(dtypes["float"].wrapped)
    assert pd.api.types.is_datetime64_any_dtype(dtypes["date"].wrapped)
    assert pd.api.types.is_datetime64_any_dtype(dtypes["ts_s"].wrapped)
    assert pd.api.types.is_datetime64_any_dtype(dtypes["ts_ns"].wrapped)
    assert pd.api.types.is_datetime64_any_dtype(dtypes["ts_tz"].wrapped)
    assert pd.api.types.is_string_dtype(dtypes["text"].wrapped)
