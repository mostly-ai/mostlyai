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
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({"pk": [1, 2, 3, 4, 5, 6, 7, 8], "let": ["a", "b", "c", "d", "e", "f", "g", "h"]})


@pytest.fixture
def linked_dataframe(sample_dataframe):
    total_repeats = sum(sample_dataframe["pk"])
    linked_df = pd.DataFrame(
        {
            "fk": np.repeat(sample_dataframe["pk"], sample_dataframe["pk"]),
            "letter": np.tile(["A", "B", "C", "D"], total_repeats)[:total_repeats],
        }
    )
    linked_df.reset_index(drop=True, inplace=True)
    return linked_df


@pytest.fixture
def dataset(sample_dataframe, linked_dataframe):
    yield {"subject": sample_dataframe, "linked": linked_dataframe}


def prepare_data_for_read(data_format, dataset, tmp_path):
    file_paths = {}
    for table_name, df in dataset.items():
        file_path = tmp_path / f"test_data_{table_name}.csv"
        df.to_csv(file_path, index=False)
        file_paths[table_name] = str(file_path)

    if data_format == "csv":
        return file_paths["subject"], file_paths["linked"]
    elif data_format == "sqlite":
        db_path = tmp_path / "test_data.sqlite"
        engine = create_engine(f"sqlite:///{db_path}")
        for table_name, df in dataset.items():
            df.to_sql(table_name, engine, index=False, if_exists="replace")
        return "main.subject", "main.linked"


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
def test_read_data(tmp_path, dataset, data_format, connector_type):
    mostly = MostlyAI(local=True, local_dir=tmp_path, quiet=True)

    subject_location, linked_location = prepare_data_for_read(data_format, dataset, tmp_path)

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

    read_df_subject = c.read_data(location=subject_location)
    pd.testing.assert_frame_equal(read_df_subject, dataset["subject"], check_dtype=False)

    read_df_linked = c.read_data(location=linked_location)
    pd.testing.assert_frame_equal(read_df_linked, dataset["linked"], check_dtype=False)

    c.delete()


@pytest.mark.parametrize(
    "connector_type, location_format",
    [
        ("SQLITE", "main.data"),
        ("FILE_UPLOAD", "{tmp_path}/test_write.csv"),
        ("FILE_UPLOAD", "{tmp_path}/test_write.parquet"),
    ],
)
def test_write_and_delete_data(tmp_path, sample_dataframe, connector_type, location_format):
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
    c.delete_data(location=location)

    if connector_type == "SQLITE":
        engine = create_engine(f"sqlite:///{tmp_path}/test_write.sqlite")
        with pytest.raises(Exception):
            pd.read_sql_table("data", con=engine)
    else:
        assert not Path(location).exists()

    c.delete()


@pytest.mark.parametrize(
    "data_format, connector_type",
    [
        # ("csv", "FILE_UPLOAD"),  # local file-system access is disabled
        ("sqlite", "SQLITE")
    ],
)
def test_query(tmp_path, dataset, data_format, connector_type):
    mostly = MostlyAI(local=True, local_dir=tmp_path, quiet=True)

    subject_location, linked_location = prepare_data_for_read(data_format, dataset, tmp_path)

    connector_config = {
        "name": f"Test {connector_type} Connector",
        "type": connector_type,
        "access_type": "READ_DATA",
        "config": {},
    }

    if connector_type == "SQLITE":
        connector_config["config"]["database"] = str(tmp_path / "test_data.sqlite")

    subject_table_name = subject_location if connector_type == "FILE_UPLOAD" else "subject"
    linked_table_name = linked_location if connector_type == "FILE_UPLOAD" else "linked"

    c = mostly.connect(
        config=connector_config,
        test_connection=False,
    )

    query_result = c.query(sql=f"SELECT * FROM '{subject_table_name}'")
    pd.testing.assert_frame_equal(query_result, dataset["subject"], check_dtype=False)

    # perform a join query:
    join_query = f"""
        SELECT s.pk, s.let, l.fk
        FROM '{subject_table_name}' s
        JOIN '{linked_table_name}' l ON s.pk = l.fk
        WHERE l.fk IN (2, 3)
        ORDER BY l.fk, s.pk
    """
    join_result = c.query(sql=join_query)

    expected_join_df = dataset["linked"][dataset["linked"]["fk"].isin([2, 3])].copy()
    expected_join_df = expected_join_df.merge(
        dataset["subject"], left_on="fk", right_on="pk", suffixes=("_linked", "_subject")
    )
    expected_join_df = expected_join_df[["pk", "let", "fk"]]
    expected_join_df.sort_values(by=["fk", "pk"], inplace=True)
    expected_join_df.reset_index(drop=True, inplace=True)

    pd.testing.assert_frame_equal(join_result, expected_join_df, check_dtype=False)

    # perform an aggregation query:
    aggregation_query = f"""
        SELECT l.fk, COUNT(*) as count
        FROM '{linked_table_name}' l
        GROUP BY l.fk
        ORDER BY l.fk
    """
    aggregation_result = c.query(sql=aggregation_query)

    expected_aggregation_df = pd.DataFrame({"fk": dataset["linked"]["fk"].unique()})
    expected_aggregation_df["count"] = expected_aggregation_df["fk"]

    pd.testing.assert_frame_equal(aggregation_result, expected_aggregation_df, check_dtype=False)

    c.delete()
