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

from mostlyai.sdk._data.db.sqlite import SqliteContainer, SqliteTable


@pytest.fixture()
def temp_table(tmp_path):
    container = SqliteContainer(dbname=str(tmp_path / "database.db"))
    return SqliteTable(name="data", container=container, is_output=True)


def test_count_rows(temp_table):
    df = pd.DataFrame({"id": [1, 2, 3]})
    temp_table.write_data(df, if_exists="replace")
    assert temp_table.row_count == df.shape[0]


@pytest.mark.parametrize(
    "write_chunk_size",
    [0, 100],
)
def test_write_data_empty(temp_table, write_chunk_size):
    df = pd.DataFrame({"id": [], "col": []})
    temp_table.WRITE_CHUNK_SIZE = write_chunk_size
    temp_table.write_data(df, if_exists="replace")
    df_read = temp_table.read_data()
    assert df_read.empty


def test_name_in_db_differs_from_logical_name(temp_table):
    # create a table with actual name "users_v2" but logical name "users"
    temp_table.name = "users_v2"
    df = pd.DataFrame({"name": ["alice", "bob", "charlie"]})
    temp_table.write_data(df, if_exists="replace")

    # now access with different logical name but same name_in_db
    table_with_alias = SqliteTable(name="users", container=temp_table.container, name_in_db="users_v2", is_output=False)
    df_read = table_with_alias.read_data()

    assert df_read.shape == df.shape
    pd.testing.assert_frame_equal(
        df_read.sort_values("name").reset_index(drop=True),
        df.sort_values("name").reset_index(drop=True),
        check_dtype=False,
    )
