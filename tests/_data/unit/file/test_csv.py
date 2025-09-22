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
import datetime
import os
import random
import tempfile

import duckdb
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pytest

from mostlyai.sdk._data.file.table.csv import CsvDataTable


def test_read_data(sample_csv_file):
    sample_df = ds.dataset(source=sample_csv_file, format=ds.CsvFileFormat()).scanner().to_table().to_pandas()
    tbl = CsvDataTable(path=sample_csv_file, name="sample")
    # test metadata
    assert tbl.columns == list(sample_df.columns)
    dtypes = tbl.dtypes
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
    df = tbl.read_data(do_coerce_dtypes=True)
    assert df.shape == sample_df.shape
    for col in df:
        assert df[col].isna().sum() == sample_df[col].isna().sum()
    df = tbl.read_data(where={"id": 1})
    assert len(df) == 1
    df = tbl.read_data(where={"id": [2, 3]}, columns=["float", "id"])
    assert len(df) == 2
    assert set(df.columns) == {"id", "float"}
    df = tbl.read_data(limit=3, shuffle=True)
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
    s = tbl.read_data(shuffle=True)
    assert any(s["id"] != df["id"])
    # check shuffle + limit
    s = tbl.read_data(shuffle=True, limit=6)
    assert sorted(s["id"]) != sorted(df["id"].head(6))
    # check shuffle + limit + where
    s = tbl.read_data(where={"str": "a"}, shuffle=True, limit=3)
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


@pytest.mark.parametrize(
    "scenario, file_extension, sep",
    [
        ("csv_file", "csv", ","),  # Test with a single CSV file
        ("tsv_file", "tsv", "\t"),  # Test with a single TSV file
        ("directory", "csv", ","),  # Test with a directory containing multiple CSV files
    ],
)
def test_query(tmp_path, scenario, file_extension, sep):
    rows = 99
    data = {
        "num": list(range(rows)),  # Numbers 0-98
        "str": [chr(97 + i % 3) for i in range(rows)],  # Rotating 'a', 'b', 'c'
        "dt": [datetime.datetime(2023, 1, 1) + datetime.timedelta(days=i) for i in range(rows)],  # Sequential dates
        "cat": [f"c{i % 3 + 1}" for i in range(rows)],  # Categories c1, c2, c3
    }
    df = pd.DataFrame(data)
    category_count = rows // 3

    if scenario == "directory":
        dir_path = tmp_path / "data_dir"
        dir_path.mkdir()
        chunk_size = 25
        for i in range(0, rows, chunk_size):
            chunk = df.iloc[i : i + chunk_size]
            chunk.to_csv(dir_path / f"part{i // chunk_size}.{file_extension}", sep=sep, index=False)
        table_name = f"read_csv_auto('{dir_path}/*.csv')"
    else:
        file_path = tmp_path / f"data.{file_extension}"
        df.to_csv(file_path, sep=sep, index=False)
        table_name = f"'{file_path}'"

    # basic select
    query_base = f"SELECT * FROM {table_name}"
    result = duckdb.query(query_base).to_df()
    assert len(result) == rows
    assert set(result.columns) == {"num", "str", "dt", "cat"}

    # where
    filter_query = f"{query_base} WHERE cat = 'c1'"
    result = duckdb.query(filter_query).to_df()
    assert len(result) == category_count
    assert all(row == "c1" for row in result["cat"])

    # aggregation
    agg_query = f"""
        SELECT cat, COUNT(*) as count, AVG(num) as avg_num
        FROM {table_name}
        GROUP BY cat
        ORDER BY cat
    """
    result = duckdb.query(agg_query).to_df()
    assert len(result) == 3  # 3 categories
    assert list(result["cat"]) == ["c1", "c2", "c3"]
    assert all(result["count"] == category_count)

    # order by
    order_query = f"""
        SELECT * FROM {table_name}
        ORDER BY num DESC
        LIMIT 5
    """
    result = duckdb.query(order_query).to_df()
    assert len(result) == 5
    assert list(result["num"]) == [98, 97, 96, 95, 94]

    # limit
    limit_query = f"SELECT * FROM {table_name} LIMIT 10"
    result = duckdb.query(limit_query).to_df()
    assert len(result) == 10


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
        ("'", "'", False),
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


@pytest.mark.parametrize(
    "csv_string",
    [
        "a,b\n",  # empty csv
        '"a"\n1\n',  # single column
        "a,b\n1,🌈\n",  # non-ascii character
        '"a", "  b  "\n1,2\n',  # quoted fields with spaces
        "a|b\n1|2\n",  # pipe-separated
        "a\tb\n1\t2\n",  # tab-separated
        "a;b\n1;2\n",  # semicolon-separated
        'a,b\n1,"Hello, World"\n',  # unquoted field with comma
    ],
)
def test_edge_cases(csv_string, tmp_path):
    with tempfile.NamedTemporaryFile(dir=tmp_path, suffix=".csv", mode="w") as f:
        f.write(csv_string)
        f.flush()
        table = CsvDataTable(path=f.name)
        assert table.delimiter is not None
        assert table.columns is not None
        assert table.dtypes is not None
        assert table.read_data() is not None


def test_compressed_csv(tmp_path):
    data = b"a,b\n1,2\n"

    import bz2

    fn = tmp_path / "data.csv.bz2"
    with bz2.open(fn, "wb") as f:
        f.write(data)
    table_bz2 = CsvDataTable(path=fn)
    assert table_bz2.delimiter is not None
    assert table_bz2.columns is not None

    import gzip

    fn = tmp_path / "data.csv.gz"
    with gzip.open(fn, "wb") as f:
        f.write(data)
    table_gz = CsvDataTable(path=fn)
    assert table_gz.delimiter is not None
    assert table_gz.columns is not None


def test_non_utf8_encoding(tmp_path):
    fn = tmp_path / "data.csv"
    with open(fn, "w", encoding="iso-8859-1") as f:
        f.write("category\none é two\n")
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
