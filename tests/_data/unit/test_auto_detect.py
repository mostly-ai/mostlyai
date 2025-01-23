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

import uuid
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from mostlyai.sdk._data.auto_detect import (
    auto_detect_encoding_type,
    auto_detect_encoding_types_and_pk,
)
from mostlyai.sdk._data.file.table.parquet import ParquetDataTable
from mostlyai.sdk.domain import ModelEncodingType


class TestAutoDetectEncodingType:
    @pytest.mark.parametrize(
        "values",
        [
            [str(uuid.uuid4()) for _ in range(20000)],
            [str(uuid.uuid4()) for _ in range(20)] + ["cat1", "cat2"] * 90,  # 20 out of 200 are unique > 0.05
        ],
    )
    def test_language_text(self, values):
        series = pd.Series(values)
        assert auto_detect_encoding_type(series) == ModelEncodingType.language_text

    @pytest.mark.parametrize(
        "values",
        [
            pd.date_range("20230101", periods=100),  # regular dates
            ["2023-01-01", "2023-02-01", "", None],  # dates with missing values
        ],
    )
    def test_tabular_datetime(self, values):
        series = pd.Series(values)
        assert auto_detect_encoding_type(series) == ModelEncodingType.tabular_datetime

    @pytest.mark.parametrize(
        "values",
        [
            ["21.314, -23.22315", "-1.2321, 22.33", "0, 0", None],
            [
                "34.052235, -118.243683",
                "40.712776, -74.005974",
                "37.774929, -122.419418",
            ]
            * 5_000,
            ["   ", "-90.0, 180.0", "0.0, 0.0", "45.0, 90.0", "", None],
        ],
    )
    def test_tabular_latlong(self, values):
        series = pd.Series(values)
        assert auto_detect_encoding_type(series) == ModelEncodingType.tabular_lat_long

    @pytest.mark.parametrize(
        "values",
        [
            ["21.314, -23.22315", "0", None] * 2,
            [
                "34.052235, -118.243683",
                "40.712776. -74.005974",
                "37.774929, -122.419418",
            ]
            * 5000,
            ["2023-01-01", "2023-02-01", "abc", None] * 2,
            ["id1", "id2"] + ["cat1", "cat2"] * 49,  # 2 out of 100 are unique < 0.05
        ],
    )
    def test_tabular_categorical(self, values):
        series = pd.Series(values)
        assert auto_detect_encoding_type(series) == ModelEncodingType.tabular_categorical


@pytest.fixture
def data_table(tmp_path):
    pqt_path = Path(tmp_path) / "table.parquet"
    num_rows = 20_000

    base_date = datetime(2020, 1, 1)
    dt_column = [(base_date + timedelta(days=i)).isoformat() for i in range(num_rows)]
    int_column = list(range(num_rows))
    float_column = [i * 0.1 for i in range(num_rows)]
    lat_long_column = ["48.20849, 16.37208"] * num_rows
    str_column = [f"string_{i}" for i in range(num_rows)]

    pd.DataFrame(
        {
            "dt": dt_column,
            "int": int_column,
            "float": float_column,
            "lat_long": lat_long_column,
            "str": str_column,
            "id": str_column,
        }
    ).to_parquet(pqt_path)

    return ParquetDataTable(path=pqt_path, name="table")


@pytest.mark.parametrize(
    "sample,expected_pk",
    [
        (pd.DataFrame(), None),
        (pd.DataFrame({"pk": [1, 2, 3]}), None),
        (pd.DataFrame({"some_id": [1, 2, 3]}), "some_id"),
        (pd.DataFrame({"some_id": [1, 2, None]}), None),
        (pd.DataFrame({"pk": [1, 2, 3], "id": ["1", "2", "3"]}), "id"),
        (pd.DataFrame({"id": [1.0, 2.0, 3.0]}), None),
        (pd.DataFrame({"id": ["1" * 36, "2", "3"]}), "id"),
        (pd.DataFrame({"id": ["1" * 37, "2", "3"]}), None),
    ],
)
def test_auto_detect_primary_key(sample, expected_pk, tmp_path):
    pqt_path = Path(tmp_path) / "sample.parquet"
    sample.to_parquet(pqt_path)
    table = ParquetDataTable(path=pqt_path, name="sample")
    _, pk = auto_detect_encoding_types_and_pk(table)
    assert pk == expected_pk


def test_auto_detect_encoding_types_and_pk(data_table):
    assert data_table.encoding_types == {
        "dt": ModelEncodingType.tabular_categorical,
        "float": ModelEncodingType.tabular_numeric_auto,
        "id": ModelEncodingType.tabular_categorical,
        "int": ModelEncodingType.tabular_numeric_auto,
        "lat_long": ModelEncodingType.tabular_categorical,
        "str": ModelEncodingType.tabular_categorical,
    }  # ensure that only tabular_categorical columns are being auto-detected
    encoding_types, primary_key = auto_detect_encoding_types_and_pk(data_table)
    assert encoding_types == {
        "dt": ModelEncodingType.tabular_datetime,
        "lat_long": ModelEncodingType.tabular_lat_long,
        "str": ModelEncodingType.language_text,
    }
    assert primary_key == "id"
