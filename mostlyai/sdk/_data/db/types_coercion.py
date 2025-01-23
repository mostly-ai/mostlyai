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

"""
Provides function for data coercion based on SQL dtype.
"""

import datetime
import decimal
import logging
from collections import defaultdict
from typing import Any

import pandas as pd
from sqlalchemy.dialects import postgresql

from mostlyai.sdk._data.dtype import (
    bool_coerce,
    datetime_coerce,
    float_coerce,
    int_coerce,
    str_coerce,
    time_coerce,
)

_LOG = logging.getLogger(__name__)

# primary mapping from SQL dtype class to coercion function
SQL_DTYPE__COERCE = {
    # postgresql
    postgresql.VARCHAR: str_coerce
    # NOTE: we may add more of specific type coercions
}


# fallback mapping from python type to coercion function
PYTHON_DTYPE__COERCE = {
    bool: bool_coerce,
    datetime.time: time_coerce,
    datetime.date: datetime_coerce,
    datetime.datetime: datetime_coerce,
    decimal.Decimal: float_coerce,
    float: float_coerce,
    int: int_coerce,
    str: str_coerce,
}

COERCE__SQL_DTYPE__KWARGS = defaultdict(
    lambda: lambda _: {},
    {str_coerce: lambda sql_dtype: {"max_length": getattr(sql_dtype, "length", None)}},
)


def coerce_to_sql_dtype(s: pd.Series, sql_dtype: Any) -> pd.Series | None:
    """
    Coerces data to given SQL dtype.

    :param s: data
    :param sql_dtype: target SQL dtype for data to fit
    :return: coerced data if coercion function found for SQL dtype, otherwise None
    """

    # check if coercion function is available for SQL dtype class
    sql_dtype_class = type(sql_dtype)
    if sql_dtype_class in SQL_DTYPE__COERCE:
        coerce = SQL_DTYPE__COERCE[sql_dtype_class]
        coerce_kwargs = COERCE__SQL_DTYPE__KWARGS[coerce](sql_dtype)
        return coerce(s, **coerce_kwargs)

    # check if coercion function is available for SQL dtype's python_type
    # note that, not every SQL dtype has python_type implemented!
    try:
        python_dtype = sql_dtype.python_type
    except NotImplementedError:
        _LOG.warning(f"python_type not implemented for SQL dtype {sql_dtype}")
    else:
        if python_dtype in PYTHON_DTYPE__COERCE:
            coerce = PYTHON_DTYPE__COERCE[python_dtype]
            coerce_kwargs = COERCE__SQL_DTYPE__KWARGS[coerce](sql_dtype)
            return coerce(s, **coerce_kwargs)

    _LOG.warning(f"coercion function not found for column {s.name} and SQL dtype {sql_dtype}")
    # return None if coercion function not found
    return None
