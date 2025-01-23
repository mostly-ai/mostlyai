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

import pandas as pd

from mostlyai.sdk._data.file.container.bucket_based import BucketBasedContainer
from mostlyai.sdk._data.file.base import LocalFileContainer
from mostlyai.sdk._data.file.table.csv import CsvDataTable
from mostlyai.sdk._data.file.table.parquet import ParquetDataTable
from mostlyai.sdk._data.file.utils import read_data_table_from_path

SCRIPT_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
FIXTURES_DIR = SCRIPT_DIR / "fixtures"
CSV_FIXTURES_DIR = FIXTURES_DIR / "csv"


def test_read_data_table_from_local_path(tmp_path):
    def _create_csv(file):
        file.parent.mkdir(exist_ok=True)
        pd.DataFrame({"x": [1]}).to_csv(file)

    def _create_pqt(file):
        file.parent.mkdir(exist_ok=True)
        pd.DataFrame({"x": [1]}).to_parquet(file)

    # test single CSV file
    file = tmp_path / "folder1" / "my_data.2023.csv.gz"
    _create_csv(file)
    container = LocalFileContainer(file_path=file)
    table = read_data_table_from_path(container)
    assert table.name == "my_data.2023"
    assert isinstance(table, CsvDataTable)

    # test multiple CSV files, mixed with some parquet
    file1 = tmp_path / "folder2" / "my_data.2023.csv.gz"
    file2 = tmp_path / "folder2" / "my_data.2024.csv.gz"
    file3 = tmp_path / "folder2" / "my_data.2025.parquet"
    _create_csv(file1)
    _create_csv(file2)
    _create_pqt(file3)
    container = LocalFileContainer(file_path=tmp_path / "folder2")
    table = read_data_table_from_path(container)
    assert table.name == "folder2"
    assert isinstance(table, ParquetDataTable)

    # test multiple Parquet files
    file1 = tmp_path / "folder3" / "my_data.2023.parquet"
    file2 = tmp_path / "folder3" / "my_data.2024.parquet"
    _create_pqt(file1)
    _create_pqt(file2)
    container = LocalFileContainer(file_path=tmp_path / "folder3")
    table = read_data_table_from_path(container)
    assert table.name == "folder3"
    assert isinstance(table, ParquetDataTable)


def test_normalize_bucket_location():
    uris = [
        "s3:///bucketname/bucketpath/filename.xlsx/",
        "s3:///bucketname/bucketpath/filename.xlsx",
        "s3:///bucketname/bucketpath/",
        "s3:///bucketname/bucketpath",
        "s3:///bucketname/",
        "s3:///bucketname",
        "s3://bucketname/bucketpath/filename.xlsx/",
        "s3://bucketname/bucketpath/filename.xlsx",
        "s3://bucketname/bucketpath/",
        "s3://bucketname/bucketpath",
        "s3://bucketname/",
        "s3://bucketname",
        "/bucketname/bucketpath/filename.xlsx/",
        "/bucketname/bucketpath/filename.xlsx",
        "/bucketname/bucketpath/",
        "/bucketname/bucketpath",
        "bucketname/bucketpath/filename.xlsx/",
        "bucketname/bucketpath/filename.xlsx",
        "bucketname/bucketpath/",
        "bucketname/bucketpath",
        "bucketname/",
        "bucketname",
    ]
    results = [BucketBasedContainer.normalize_bucket_location(uri) for uri in uris]
    assert results == [
        "bucketname/bucketpath/filename.xlsx",
        "bucketname/bucketpath/filename.xlsx",
        "bucketname/bucketpath",
        "bucketname/bucketpath",
        "bucketname/",
        "bucketname/",
        "bucketname/bucketpath/filename.xlsx",
        "bucketname/bucketpath/filename.xlsx",
        "bucketname/bucketpath",
        "bucketname/bucketpath",
        "bucketname/",
        "bucketname/",
        "bucketname/bucketpath/filename.xlsx",
        "bucketname/bucketpath/filename.xlsx",
        "bucketname/bucketpath",
        "bucketname/bucketpath",
        "bucketname/bucketpath/filename.xlsx",
        "bucketname/bucketpath/filename.xlsx",
        "bucketname/bucketpath",
        "bucketname/bucketpath",
        "bucketname/",
        "bucketname/",
    ]
