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
from datetime import date, datetime, time
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pytest as pytest
import sqlalchemy as sa
from mostlyai.sdk._data.base import ForeignKey, Schema
from mostlyai.sdk._data.db.sqlite import SqliteDType, SqliteTable, SqliteContainer
from mostlyai.sdk._data.dtype import PandasDType
from mostlyai.sdk._data.file.table.csv import CsvDataTable
from mostlyai.sdk._data.file.table.parquet import ParquetDataTable
from mostlyai.sdk._data.push import adapt_dtypes_to_destination, push_data


@pytest.fixture
def user_df_with_self_ref():
    # populate reports_to_user_id and follow referential integrity
    user_df = pd.DataFrame({"id": np.arange(1, 101)})
    n = len(user_df)
    n_admins = n // 10
    user_df["reports_to_user_id"] = np.random.randint(n_admins + 1, n + 1, size=n)
    user_df.loc[user_df["id"] <= n_admins, "reports_to_user_id"] = np.nan
    user_df["id"] = user_df["id"].astype("Int64")
    user_df["reports_to_user_id"] = user_df["reports_to_user_id"].astype("Int64")
    return user_df


@pytest.fixture
def source_table_partitioned(tmp_path):
    source_data = pd.DataFrame(
        {
            "id": np.arange(1_000),
            "cat": ["blue", "red"] * 500,
            "str": list("abcdefghij") * 100,
        }
    )
    source_data_path = tmp_path / "source"
    source_data_path.mkdir(exist_ok=True, parents=True)
    n_partitions = 10
    get_data_partitioned = np.array_split(source_data, n_partitions)
    # save each part into a separate parquet file
    for i, df_part in enumerate(get_data_partitioned):
        part_fn = source_data_path / f"part.{i:06}.parquet"
        df_part.to_parquet(part_fn, index=False)
    return ParquetDataTable(path=source_data_path)


@pytest.fixture
def user_table_with_self_ref(tmp_path, user_df_with_self_ref):
    path = Path(tmp_path) / "user.csv"
    CsvDataTable(path=path, is_output=True).write_data(user_df_with_self_ref)
    return CsvDataTable(path=path)


@pytest.mark.parametrize("destination_type", [ParquetDataTable, CsvDataTable])
def test_push_data_to_file_based_destination(source_table_partitioned, destination_type, tmp_path):
    is_dir_output = not destination_type.IS_WRITE_APPEND_ALLOWED
    ext = destination_type.DATA_TABLE_TYPE
    destination_path = Path(tmp_path) / (ext if is_dir_output else f"destination.{ext}")
    destination = destination_type(path=destination_path, is_output=True)

    push_data(source=source_table_partitioned, destination=destination)

    # ensure data is equal
    source_data = source_table_partitioned.read_data()
    dest_data = destination_type(path=destination_path).read_data()
    pd.testing.assert_frame_equal(dest_data, source_data, check_dtype=False)
    # ensure partitioned file stems are equal (when applicable)
    if is_dir_output:
        source_file_stems = [Path(f).stem for f in source_table_partitioned.dataset.files]
        destination_file_stems = [Path(f).stem for f in destination_path.iterdir()]
        assert sorted(source_file_stems) == sorted(destination_file_stems)


def create_sqlite_db(path):
    uri = f"sqlite+pysqlite:///{path}"
    df = pd.DataFrame({"id": [1, 2, 3]})
    df.to_sql(index=False, name="temp", con=uri)
    return SqliteContainer(dbname=path)


@pytest.fixture
def two_table_schema(tmp_path):
    container = create_sqlite_db(f"{tmp_path}/temp_db.db")
    common_kwargs = dict(primary_key="id", is_output=True)
    tables = {
        "ctx": SqliteTable(name="ctx", container=container, foreign_keys=[], **common_kwargs),
        "tgt": SqliteTable(
            name="tgt",
            container=container,
            foreign_keys=[ForeignKey(column="ctx_id", referenced_table="ctx", is_context=True)],
            **common_kwargs,
        ),
    }
    schema = Schema(tables=tables)
    return schema


def create_two_table_data(
    key_gen_type: Literal["seq", "uuid"] = "seq", ctx_size=10, seq_length=2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gen_function_map = dict(
        seq=lambda length: list(range(length)),
        uuid=lambda length: list(str(uuid.uuid4()) for _ in range(length)),
    )
    gen_func = gen_function_map[key_gen_type]

    ctx_df = pd.DataFrame({"id": gen_func(ctx_size)})
    tgt_size = ctx_size * seq_length
    tgt_df = pd.DataFrame(
        {
            "id": gen_func(ctx_size * seq_length),
            "ctx_id": gen_func(ctx_size) * seq_length,
            "str3": ["abc"] * tgt_size,
            "date": [date(2020, 3, 13)] * tgt_size,
            "time": [time(13, 45, 30)] * tgt_size,
            "datetime": [datetime(2001, 4, 30, 13, 45, 15)] * tgt_size,
            "float": [3.141592] * tgt_size,
            "decimal": [12.34] * tgt_size,
            "int": [99] * tgt_size,
        }
    )
    return ctx_df, tgt_df


@pytest.mark.parametrize("key_gen_type", ["seq", "uuid"])
@pytest.mark.parametrize(
    "dtypes",
    [
        None,
        {
            "id": SqliteDType(sa.VARCHAR(2)),
            "ctx_id": SqliteDType(sa.INTEGER()),
            "datetime": SqliteDType(sa.DATE()),  # TODO this coercion is not supported yet
            "str3": SqliteDType(sa.VARCHAR(2)),
            "decimal": SqliteDType(sa.INTEGER()),
        },
    ],
)
def test_adapt_dtypes_to_destination(tmp_path, key_gen_type, two_table_schema, dtypes):
    _, tgt_df = create_two_table_data(key_gen_type=key_gen_type)
    tgt_table = two_table_schema.tables["tgt"]
    tgt_table.dtypes = dtypes
    adapted_tgt_df = adapt_dtypes_to_destination(tgt_data=tgt_df, schema=two_table_schema, tgt="tgt")
    for column in tgt_df.columns:
        if column in ("id", "ctx_id"):
            # ensure referential integrity
            assert PandasDType(adapted_tgt_df[column]).encompasses(tgt_df[column])
        elif dtypes and column in dtypes:
            # ensure casting would have likely happened
            assert dtypes[column].encompasses(adapted_tgt_df[column])
        else:
            # ensure no changes to such columns
            pd.testing.assert_series_equal(adapted_tgt_df[column], tgt_df[column])
