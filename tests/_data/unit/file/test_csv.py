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
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pytest
from mostlyai.sdk._data.file.table.csv import CsvDataTable

SCRIPT_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
FIXTURES_DIR = SCRIPT_DIR / "fixtures"
CSV_FIXTURES_DIR = FIXTURES_DIR / "csv"


def test_read_data():
    fn = CSV_FIXTURES_DIR / "sample.csv"
    orig_df = ds.dataset(source=fn, format=ds.CsvFileFormat()).scanner().to_table().to_pandas()
    table = CsvDataTable(path=fn, name="sample")
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
    df = table.read_data(do_coerce_dtypes=True)
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
    fn = tmp_path / "data.csv"
    df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            "str": ["a", "b"] * 6,
            "int": [1, 2] * 6,
        }
    )
    df.to_csv(fn, index=False)
    tbl = CsvDataTable(path=fn, primary_key="id")
    # check data
    assert tbl.read_data().shape == df.shape
    # check where
    assert len(tbl.read_data(where={"id": -1})) == 0
    assert len(tbl.read_data(where={"id": 1})) == 1
    assert len(tbl.read_data(where={"id": [3, 4]})) == 2
    assert len(tbl.read_data(where={"str": "a", "int": 2})) == 0
    assert len(tbl.read_data(where={"str": "a", "int": 1})) == 6
    assert len(tbl.read_data(where={"id": [pd.NA, pd.NaT, np.nan, None]})) == 0
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
    df.to_csv(fn, index=False)
    tbl = CsvDataTable(path=fn, primary_key="id")
    assert len(tbl.read_data(where={"id": [None, "uuid"]})) == 0
    # test null-only pk
    df = pd.DataFrame({"id": [None, None, None]})
    df.to_csv(fn, index=False)
    tbl = CsvDataTable(path=fn, primary_key="id")
    assert len(tbl.read_data(where={"id": [None, "uuid"]})) == 3


def test_write_data(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    csv_data_table = CsvDataTable(path=f"file://{tmp_path}/test.csv")
    csv_data_table.write_data(df)
    df_read = csv_data_table.read_data()
    assert df_read.shape == df.shape


def test_csv_format(csv_file):
    file_path, supported, delimiter = csv_file
    csv_data_table = CsvDataTable(path=file_path)
    sample = csv_data_table.read_data()
    assert sample is not None
    if supported:
        assert len(sample.columns.tolist()) == 3
        assert sample.columns.tolist() == ["Name", "Age", "Gender"]
    else:
        assert len(sample.columns.tolist()) == 1
        assert sample.columns.tolist() == [f"Name{delimiter}Age{delimiter}Gender"]


def _create_temp_csv_file(data: list[list[str]], delimiter: str, quote_char: str) -> str:
    with tempfile.NamedTemporaryFile(mode="w+", newline="", delete=False, encoding="utf-8", suffix=".csv") as temp_file:
        writer = csv.writer(
            temp_file,
            delimiter=delimiter,
            quotechar=quote_char,
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writerows(data)
        temp_file.flush()
        return temp_file.name


def _create_data_list(header: list[str], num_rows: int) -> list[list[str]]:
    names = [
        "John",
        "Jane",
        "David",
        "Alice",
        "Bob",
        "Cathy",
        "Daniel",
        "Eva",
        "Fred",
        "Grace",
    ]
    ages = list(range(18, 61))
    genders = ["Male", "Female"]

    data_list = [header]

    for _ in range(num_rows):
        name = random.choice(names)
        age = str(random.choice(ages))
        gender = random.choice(genders)
        data_list.append([name, age, gender])

    return data_list


def test_double_quote_handling(tmp_path):
    df = pd.DataFrame({"a": ['"apple"', '"banana"', '"cherry"']})
    df.to_csv(tmp_path / "data.csv", index=False)
    t = CsvDataTable(path=tmp_path / "data.csv")
    assert sorted(t.read_data()["a"].tolist()) == ['"apple"', '"banana"', '"cherry"']


def test_single_quote_handling(tmp_path):
    df = pd.DataFrame({"b": ["'apple'", "'banana'", "'cherry'"]})
    df.to_csv(tmp_path / "data.csv", index=False)
    t = CsvDataTable(path=tmp_path / "data.csv")
    assert sorted(t.read_data()["b"].tolist()) == ["'apple'", "'banana'", "'cherry'"]


def test_escape_character_handling(tmp_path):
    df = pd.DataFrame({"c": ["apple,red", "banana,yellow", "cherry,red"]})
    df.to_csv(tmp_path / "data.csv", index=False, escapechar="\\")
    t = CsvDataTable(path=tmp_path / "data.csv")
    assert sorted(t.read_data()["c"].tolist()) == [
        "apple,red",
        "banana,yellow",
        "cherry,red",
    ]


@pytest.fixture(
    params=[
        # supported delimiters
        (",", '"', True),
        (";", "'", True),
        ("\t", '"', True),
        ("|", '"', True),
        ("'", "'", True),
        (" ", '"', True),
        # unsupported delimiters
        ("/", "'", False),
        ("$", '"', False),
    ]
)
def csv_file(request):
    delimiter, quotechar, supported = request.param
    header = ["Name", "Age", "Gender"]
    data = _create_data_list(header, 5)
    file_path = _create_temp_csv_file(data, delimiter, quotechar)
    yield file_path, supported, delimiter
    os.remove(file_path)


def test_read_file_zoo():
    files = Path(f"{CSV_FIXTURES_DIR}/file_zoo").glob("*")
    for file in files:
        table = CsvDataTable(path=f"file://{file}")
        assert table.delimiter is not None
        assert table.columns is not None
        assert table.dtypes is not None
        assert table.read_data() is not None


def test_non_utf8_encoding():
    fn = CSV_FIXTURES_DIR / "file_zoo" / "enc_iso8859_1.csv"
    table = CsvDataTable(path=fn)
    df = table.read_data(do_coerce_dtypes=True)
    # check that we get a column with string values, rather than
    # binary values; otherwise ENGINE would crash for those
    assert isinstance(df["category"][0], str)


def test_read_tsv_from_folder(tmp_path):
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    df.to_csv(tmp_path / "data.tsv", index=False, sep="\t")
    t = CsvDataTable(path=tmp_path)
    assert len(t.read_data().columns) == 2


def test_read_csv_dtype_mismatch(tmp_path):
    # write CSV file larger than 1MB
    fn = tmp_path / "data.csv"
    with open(fn, "a") as f:
        f.write("x,y\n")
        for i in range(2_000_000):
            f.write("1,\n")
        f.write(",.3\n")
    # read CSV and ensure that no error is thrown due to dtype mismatch
    t = CsvDataTable(path=tmp_path / "data.csv").read_data()
    assert t.shape[1] == 2