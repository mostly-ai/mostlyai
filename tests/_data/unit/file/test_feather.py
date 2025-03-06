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
import pytest

from mostlyai.sdk._data.dtype import (
    is_boolean_dtype,
    is_date_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_string_dtype,
    is_timestamp_dtype,
)
from mostlyai.sdk._data.file.table.feather import FeatherDataTable


@pytest.mark.parametrize("sample_parquet_file", ["pyarrow"], indirect=True)
def test_read_write_data(tmp_path, sample_parquet_file):
    # write data
    sample_data = pd.read_parquet(sample_parquet_file, dtype_backend="pyarrow")
    table1 = FeatherDataTable(path=tmp_path / "sample.feather", is_output=True)
    table1.write_data(sample_data)
    # read data
    table2 = FeatherDataTable(path=tmp_path / "sample.feather")
    data = table2.read_data()
    # compare data
    assert data.shape == sample_data.shape
    assert is_integer_dtype(data["id"])
    assert is_boolean_dtype(data["bool"])
    assert is_integer_dtype(data["int"])
    assert is_float_dtype(data["float"])
    assert is_date_dtype(data["date"])
    assert is_timestamp_dtype(data["ts_s"])
    assert is_timestamp_dtype(data["ts_ns"])
    assert is_timestamp_dtype(data["ts_tz"])
    assert is_string_dtype(data["text"])
