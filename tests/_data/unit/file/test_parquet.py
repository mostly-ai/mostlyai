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

import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest
from mostlyai.sdk._data.dtype import VirtualVarchar
from mostlyai.sdk._data.file.table.parquet import ParquetDataTable

SCRIPT_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
FIXTURES_DIR = SCRIPT_DIR / "fixtures"
PQT_FIXTURES_DIR = FIXTURES_DIR / "parquet"


def test_row_count(tmp_path):
    # test existing file
    n = 100
    fn = tmp_path / "hello.parquet"
    pd.DataFrame({"a": range(n)}).to_parquet(fn)
    table = ParquetDataTable(path=fn)
    assert table.row_count == n
    # test existing directory
    n = 100
    dir = tmp_path / "hello"
    dir.mkdir()
    pd.DataFrame({"a": range(n)}).to_parquet(dir / "a.parquet")
    pd.DataFrame({"a": range(n)}).to_parquet(dir / "b.parquet")
    table = ParquetDataTable(path=dir)
    assert table.row_count == 2 * n
    # test missing file
    table = ParquetDataTable(path=tmp_path / "missing.parquet")
    assert table.row_count == 0
    # test missing directory
    table = ParquetDataTable(path=tmp_path / "missing")
    assert table.row_count == 0


@pytest.mark.parametrize(
    "dtype_backend",
    [
        "numpy_nullable",
        "pyarrow",
    ],
)
def test_read_data(sample_csv_file, dtype_backend):
    # create parquet file on the fly
    fn = tempfile.NamedTemporaryFile(suffix=".parquet").name
    df = pd.read_csv(sample_csv_file, engine="pyarrow", dtype_backend=dtype_backend)
    df.to_parquet(fn, engine="pyarrow")
    orig_df = pd.read_parquet(fn, engine="pyarrow", dtype_backend="pyarrow")
    table = ParquetDataTable(path=fn, name="sample")
    # test metadata
    assert table.columns == list(orig_df.columns)
    dtypes = table.dtypes
    assert pd.api.types.is_integer_dtype(dtypes["id"].wrapped)
    assert pd.api.types.is_bool_dtype(dtypes["bool"].wrapped)
    assert pd.api.types.is_integer_dtype(dtypes["int"].wrapped)
    assert pd.api.types.is_float_dtype(dtypes["float"].wrapped)
    assert pd.api.types.is_datetime64_any_dtype(dtypes["date"].wrapped)
    assert pd.api.types.is_datetime64_any_dtype(dtypes["ts_s"].wrapped)
    assert pd.api.types.is_datetime64_any_dtype(dtypes["ts_ns"].wrapped)
    assert pd.api.types.is_datetime64_any_dtype(dtypes["ts_tz"].wrapped)
    assert pd.api.types.is_string_dtype(dtypes["text"].wrapped)
    # test content
    df = table.read_data()
    assert df.shape == orig_df.shape
    for col in df:
        assert df[col].isna().sum() == orig_df[col].isna().sum()
    df = table.read_data(where={"id": 1})
    assert len(df) == 1
    df = table.read_data(where={"id": [2, 3]}, columns=["float", "id"])
    assert len(df) == 2
    assert set(df.columns) == {"id", "float"}
    df = table.read_data(limit_n_rows=3, is_shuffle=True)
    assert len(df) == 3


def test_filter_data(tmp_path):
    # create test data
    fn = tmp_path / "data.parquet"
    df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            "str": ["a", "b"] * 6,
            "int": [1, 2] * 6,
        }
    )
    df.to_parquet(fn)
    tbl = ParquetDataTable(path=fn, primary_key="id")
    # check data
    assert tbl.read_data().shape == df.shape
    # check where
    assert len(tbl.read_data(where={"id": -1})) == 0
    assert len(tbl.read_data(where={"id": 1})) == 1
    assert len(tbl.read_data(where={"id": [3, 4]})) == 2
    assert len(tbl.read_data(where={"id": [pd.NA, pd.NaT, np.nan, None]})) == 0
    assert len(tbl.read_data(where={"str": "a", "int": 2})) == 0
    assert len(tbl.read_data(where={"str": "a", "int": 1})) == 6
    # check cols
    s = tbl.read_data(columns=["str", "int"])
    assert all(s.columns == ["str", "int"])
    # check shuffle
    s = tbl.read_data(is_shuffle=True)
    assert any(s["id"] != df["id"])
    # check shuffle + limit
    s = tbl.read_data(is_shuffle=True, limit_n_rows=6)
    assert sorted(s["id"]) != sorted(df["id"].head(6))
    # check shuffle + limit + where
    s = tbl.read_data(where={"str": "a"}, is_shuffle=True, limit_n_rows=3)
    assert all(s["id"].isin(df.loc[df["str"] == "a", "id"]))
    assert all(s["int"] == 1)
    # test empty data
    df = pd.DataFrame(columns=["id", "str", "int"])
    df.to_parquet(fn, index=False)
    tbl = ParquetDataTable(path=fn, primary_key="id")
    assert len(tbl.read_data(where={"id": [None, "uuid"]})) == 0
    # test null-only pk
    df = pd.DataFrame({"id": [None, None, None]})
    df.to_parquet(fn, index=False)
    tbl = ParquetDataTable(path=fn, primary_key="id")
    assert len(tbl.read_data(where={"id": [None, "uuid"]})) == 3


@pytest.fixture()
def simple_partitions_generator():
    df_1 = pd.DataFrame({"id": [1, 2, 3], "str": ["a", "b", "c"]})
    df_2 = pd.DataFrame({"id": [4, 5, 6], "str": ["d", "e", "f"]})
    base = "some_base_directory/"
    part_names = [base + p for p in ["part.000000.parquet", "part.000001.parquet"]]

    def partitions_generator():
        yield from zip(part_names, [df_1, df_2])

    return partitions_generator


def test_write_data(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}, dtype="int64[pyarrow]")
    file_path = f"file://{tmp_path}/output.parquet"
    data_table = ParquetDataTable(path=file_path, is_output=True)
    data_table.write_data(df)
    input_data_table = ParquetDataTable(path=file_path)
    df_read = input_data_table.read_data()
    assert df_read.equals(df)


def test_parquet_ignore_index_column():
    table = ParquetDataTable(path=PQT_FIXTURES_DIR / "file_zoo" / "extra_index_column.parquet")
    assert "__index_level_0__" not in table.columns


def test_parquet_ignore_complex_columns(tmp_path):
    pd.DataFrame(
        {
            "complex1": [["a"], ["b"]],
            "complex2": [{"a": 1}, {"b": 2}],
            "complex3": [("a",), ("b",)],
            "text": ["a", "b"],
            "number": [30, 40],
        }
    ).to_parquet(tmp_path / "test.parquet")

    table = ParquetDataTable(path=tmp_path)
    assert table.columns == ["text", "number"]


def test_mismatch_columns():
    table = ParquetDataTable(path=PQT_FIXTURES_DIR / "file_zoo" / "mismatch")
    assert table.columns == ["col1"]


def test_no_columns():
    table = ParquetDataTable(path=PQT_FIXTURES_DIR / "file_zoo" / "no_columns")
    assert table.columns == []
    assert table.read_data().empty


def test_no_records():
    table = ParquetDataTable(path=PQT_FIXTURES_DIR / "file_zoo" / "no_records")
    assert table.columns == ["col_a", "col_b"]
    assert table.read_data().empty


def test_categorical(tmp_path):
    df = pd.DataFrame({"cat": pd.Categorical(["a", "b", None])})
    table1 = ParquetDataTable(path=tmp_path / "test.parquet", is_output=True)
    table1.write_data(df)
    table2 = ParquetDataTable(path=tmp_path / "test.parquet")
    assert table2.read_data().equals(df)
    assert isinstance(table2.dtypes["cat"].wrapped, pd.CategoricalDtype)
    assert table2.dtypes["cat"].to_virtual() == VirtualVarchar()
