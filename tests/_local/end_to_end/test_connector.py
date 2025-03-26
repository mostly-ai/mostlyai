# Copyright 2024-2025 MOSTLY AI
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

from mostlyai.sdk import MostlyAI
import pandas as pd
from sqlalchemy import create_engine
import pytest
from pathlib import Path


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({"num": [1, 2, 3, 4, 5, 6, 7, 8], "let": ["a", "b", "c", "d", "e", "f", "g", "h"]})


def prepare_data_for_read(data_format, df, tmp_path):
    if data_format == "csv":
        file_path = tmp_path / "test_data.csv"
        df.to_csv(file_path, index=False)
        return str(file_path)
    elif data_format == "sqlite":
        db_path = tmp_path / "test_data.sqlite"
        engine = create_engine(f"sqlite:///{db_path}")
        df.to_sql("data", engine, index=False, if_exists="replace")
        return "main.data"


def test_connector(tmp_path, sample_dataframe):
    mostly = MostlyAI(local=True, local_dir=tmp_path, quiet=True)

    c = mostly.connect(
        config={
            "name": "Test 1",
            "type": "S3_STORAGE",
            "access_type": "SOURCE",
            "config": {
                "access_key": "XXX",
            },
            "secrets": {
                "secret_key": "YYY",
            },
        },
        test_connection=False,
    )
    assert c.name == "Test 1"
    c.update(name="Test 2", test_connection=False)
    assert c.name == "Test 2"

    c.delete()


@pytest.mark.parametrize(
    "data_format, connector_type",
    [("csv", "FILE_UPLOAD"), ("sqlite", "SQLITE")],
)
def test_read_data(tmp_path, sample_dataframe, data_format, connector_type):
    mostly = MostlyAI(local=True, local_dir=tmp_path, quiet=True)

    location = prepare_data_for_read(data_format, sample_dataframe, tmp_path)

    connector_config = {
        "name": f"Test {connector_type} Connector",
        "type": connector_type,
        "access_type": "READ_DATA",
        "config": {},
    }

    if connector_type == "SQLITE":
        connector_config["config"]["database"] = str(tmp_path / "test_data.sqlite")

    c = mostly.connect(
        config=connector_config,
        test_connection=False,
    )

    read_df = c.read_data(location=location)
    pd.testing.assert_frame_equal(read_df, sample_dataframe, check_dtype=False)

    limited_df = c.read_data(location=location, limit=3)
    pd.testing.assert_frame_equal(limited_df, sample_dataframe.head(3), check_dtype=False)

    shuffled_df = c.read_data(location=location, shuffle=True)
    assert not shuffled_df.equals(sample_dataframe)
    for col in sample_dataframe.columns:
        assert set(shuffled_df[col]) == set(sample_dataframe[col])

    limited_shuffled_df = c.read_data(location=location, limit=3, shuffle=True)
    assert len(limited_shuffled_df) == 3
    for col in sample_dataframe.columns:
        assert set(limited_shuffled_df[col]).issubset(set(sample_dataframe[col]))

    c.delete()


@pytest.mark.parametrize(
    "connector_type, location_format",
    [
        ("SQLITE", "main.data"),
        ("FILE_UPLOAD", "{tmp_path}/test_write.csv"),
        ("FILE_UPLOAD", "{tmp_path}/test_write.parquet"),
    ],
)
def test_write_data(tmp_path, sample_dataframe, connector_type, location_format):
    mostly = MostlyAI(local=True, local_dir=tmp_path, quiet=True)

    connector_config = {
        "name": f"Test {connector_type} Connector",
        "type": connector_type,
        "access_type": "WRITE_DATA",
        "config": {},
    }

    if connector_type == "SQLITE":
        connector_config["config"]["database"] = str(tmp_path / "test_write.sqlite")

    location = location_format.format(tmp_path=tmp_path) if "{tmp_path}" in location_format else location_format

    midpoint = len(sample_dataframe) // 2
    first_half = sample_dataframe.iloc[:midpoint].copy()
    second_half = sample_dataframe.iloc[midpoint:].copy()

    c = mostly.connect(config=connector_config)

    # parquet is not "appendable", unlike the other formats being tested
    if "parquet" in location:
        c.write_data(data=sample_dataframe, location=location)
    else:
        c.write_data(data=first_half, location=location, if_exists="fail")
        c.write_data(data=first_half, location=location, if_exists="replace")
        c.write_data(data=second_half, location=location, if_exists="append")

    if connector_type == "SQLITE":
        engine = create_engine(f"sqlite:///{tmp_path}/test_write.sqlite")
        read_df = pd.read_sql_table("data", con=engine)
    else:
        read_df = pd.read_csv(location) if "csv" in location else pd.read_parquet(location)

    assert len(read_df) == len(first_half) + len(second_half)

    for col in sample_dataframe.columns:
        assert set(first_half[col]).union(set(second_half[col])) == set(read_df[col])

    if connector_type == "SQLITE":
        with pytest.raises(Exception):
            c.write_data(data=sample_dataframe, location=location, if_exists="fail")

    # test delete functionality
    c.write_data(data=None, location=location)

    if connector_type == "SQLITE":
        engine = create_engine(f"sqlite:///{tmp_path}/test_write.sqlite")
        with pytest.raises(Exception):
            pd.read_sql_table("data", con=engine)
    else:
        assert not Path(location).exists()

    c.delete()
