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

import datetime

import pandas as pd
from sqlalchemy.dialects import postgresql

from mostlyai.sdk._data.db.types_coercion import coerce_to_sql_dtype
from mostlyai.sdk._data.dtype import BOOL, FLOAT64, INT64, STRING


def test_coerce_to_sql_dtype():
    # this test is primarily to check if appropriate function was selected for coercion

    def _assert_coerced(in_data, in_sql_dtype, out_data, out_pd_dtype):
        s = coerce_to_sql_dtype(pd.Series(in_data), in_sql_dtype)
        pd.testing.assert_series_equal(s, pd.Series(out_data, dtype=out_pd_dtype))

    _assert_coerced(
        in_data=["a", True],
        in_sql_dtype=postgresql.BOOLEAN(),
        out_data=[pd.NA, True],
        out_pd_dtype=BOOL,
    )
    _assert_coerced(
        in_data=["a", "2020-01-01"],
        in_sql_dtype=postgresql.DATE(),
        out_data=[pd.NA, "2020-01-01"],
        out_pd_dtype="datetime64[ns]",
    )
    _assert_coerced(
        in_data=["a", 1.0],
        in_sql_dtype=postgresql.FLOAT(),
        out_data=[pd.NA, 1.0],
        out_pd_dtype=FLOAT64,
    )
    _assert_coerced(
        in_data=["a", 1],
        in_sql_dtype=postgresql.INTEGER(),
        out_data=[pd.NA, 1],
        out_pd_dtype=INT64,
    )
    _assert_coerced(
        in_data=["abcde", "abc"],
        in_sql_dtype=postgresql.VARCHAR(4),
        out_data=["abcd", "abc"],
        out_pd_dtype=STRING,
    )
    _assert_coerced(
        in_data=["abcde", "10:20"],
        in_sql_dtype=postgresql.TIME(),
        out_data=[pd.NA, datetime.time(10, 20)],
        out_pd_dtype="object",
    )
