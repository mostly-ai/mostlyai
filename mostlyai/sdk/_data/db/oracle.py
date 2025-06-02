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

import logging
import os
import sys
from urllib.parse import quote

import numpy as np
import oracledb
import pandas as pd
import sqlalchemy as sa
import sqlalchemy.dialects.oracle as oracle
import sqlalchemy.dialects.oracle as sa_oracle
from pandas.io.sql import SQLDatabase, SQLTable
from sqlalchemy import text
from sqlalchemy.dialects.oracle.base import OracleDialect

from mostlyai.sdk._data.db.base import DBDType, SqlAlchemyContainer, SqlAlchemyTable
from mostlyai.sdk._data.dtype import STRING
from mostlyai.sdk._data.exceptions import MostlyDataException
from mostlyai.sdk._data.util.common import run_with_timeout_unsafe

_LOG = logging.getLogger(__name__)


def configure_oracle_client():
    # this enables using Oracle without Oracle Instant Client
    # see more: https://levelup.gitconnected.com/using-python-oracledb-1-0-with-sqlalchemy-pandas-django-and-flask-5d84e910cb19
    oracledb.version = "8.3.0"
    oracledb.defaults.prefetchrows = 10_000
    sys.modules["cx_Oracle"] = oracledb
    oracle_home = os.getenv("ORACLE_HOME")
    if oracle_home:
        oracledb.init_oracle_client(lib_dir=oracle_home)


class OracleDType(DBDType):
    UNBOUNDED_VARCHAR_ALLOWED = False
    FROM_VIRTUAL_INTEGER = sa_oracle.NUMBER
    FROM_VIRTUAL_DATETIME = sa.TIMESTAMP

    @classmethod
    def sa_dialect_class(cls):
        return OracleDialect


class OracleContainer(SqlAlchemyContainer):
    SCHEMES = ["oracle"]
    SA_CONNECT_ARGS_ACCESS_ENGINE = {"tcp_connect_timeout": 3}
    IS_ACCESSIBLE_TIMEOUT = 15  # seconds
    SQL_FETCH_FOREIGN_KEYS = """
    SELECT
        A.TABLE_NAME,
        A.COLUMN_NAME,
        B.TABLE_NAME AS REFERENCED_TABLE_NAME,
        B.COLUMN_NAME AS REFERENCED_COLUMN_NAME
    FROM
        ALL_CONS_COLUMNS A
        JOIN ALL_CONSTRAINTS C ON A.CONSTRAINT_NAME = C.CONSTRAINT_NAME
        JOIN ALL_CONS_COLUMNS B ON B.CONSTRAINT_NAME = C.R_CONSTRAINT_NAME
    WHERE
        C.CONSTRAINT_TYPE = 'R'
        AND C.OWNER = :schema_name
        AND A.OWNER = :schema_name
        AND B.OWNER = :schema_name
    """
    INIT_DEFAULT_VALUES = {"dbname": "ORCL", "port": "1521"}

    def post_init_hook(self):
        super().post_init_hook()
        configure_oracle_client()

    @property
    def sa_uri(self):
        # User and password are needed to avoid double-encoding of @ character
        username = quote(self.username)
        password = quote(self.password)
        connection_type = self.props.get("connection_type", "SID")
        suffix = f"?service_name={self.dbname}" if connection_type == "SERVICE_NAME" else self.dbname
        return f"oracle+cx_oracle://{username}:{password}@{self.host}:{self.port}/{suffix}"

    @classmethod
    def table_class(cls):
        return OracleTable

    def is_accessible(self) -> bool:
        try:
            return run_with_timeout_unsafe(super().is_accessible, timeout=self.IS_ACCESSIBLE_TIMEOUT, do_raise=True)
        except TimeoutError as e:
            _LOG.error(f"Database connection failed with error: {e}")
            raise MostlyDataException("Cannot establish connection. Host, port, or credentials are incorrect.")
        except Exception:
            raise

    def _is_schema_exist(self) -> bool:
        with self.init_sa_connection() as connection:
            if self.dbschema is None:
                return True

            result = connection.execute(
                text("SELECT COUNT(*) FROM all_users WHERE upper(username) = upper(:schema_name)"),
                {"schema_name": self.dbschema},
            ).scalar()
            return result > 0

    def does_database_exist(self) -> bool:
        try:
            with self.init_sa_connection("db_exist_check") as connection:
                connection.close()
                return True
        except sa.exc.SQLAlchemyError:
            return False

    def all_schemas(self) -> list:
        # by the time this code was written, sql alchemy did not proper read
        # schemas from oracle using inspect, the workaround is to use raw sql
        # and dividing into three queries (inherited and not/null inherited)
        with self.init_sa_connection() as connection:
            inherited = connection.execute(sa.text("SELECT username FROM all_users WHERE INHERITED = 'YES'")).fetchall()
            not_inherited = connection.execute(
                sa.text("SELECT username FROM all_users WHERE INHERITED = 'NO'")
            ).fetchall()
            none_inherited = connection.execute(
                sa.text("SELECT username FROM all_users WHERE INHERITED is NULL")
            ).fetchall()

            total = inherited + not_inherited + none_inherited
            return [x[0] for x in total]


class OracleTable(SqlAlchemyTable):
    DATA_TABLE_TYPE = "oracle"
    SA_RANDOM = sa.text("dbms_random.value")
    # Oracle has 1000 upper bound on number of items that can be within IN clause
    # read more: https://stackoverflow.com/a/17844383
    SA_MAX_VALS_PER_IN_CLAUSE = 1_000
    SA_CONN_DIALECT_PROPS = {"arraysize": 100_000}

    @classmethod
    def dtype_class(cls):
        return OracleDType

    @classmethod
    def container_class(cls):
        return OracleContainer

    def create_table(self, df: pd.DataFrame | None = None, **kwargs) -> None:
        # HACK alert: build dummy SQLTable only to get access to the way pd.to_sql infers dtypes
        dummy_sql_db = SQLDatabase(None)
        dummy_sql_table = SQLTable(name="DUMMY", pandas_sql_engine=dummy_sql_db, frame=df)
        to_sql_dtypes = {
            column: dtype
            for column, dtype, _ in dummy_sql_table._get_column_names_and_types(dummy_sql_table._sqlalchemy_type)
        }

        dtypes = kwargs.pop("dtype", {})
        for column in df.columns:
            if any((column in dtypes, column not in to_sql_dtypes)):
                # skip if dtype is already set
                # skip if pandas hasn't inferred dtype
                continue
            if to_sql_dtypes[column] == sa.Text:
                # TEXT would be later mapped to very inefficient CLOB
                # we want to avoid that, so we map to VARCHAR2 (2*max_length)
                max_length = df[column].astype(STRING).str.len().max()
                if pd.isna(max_length):
                    # handle empty or all-null columns
                    max_length = 256 / 1.5
                # add 200% to max_length and ensure at least VARCHAR2(100) to be on the safe side
                # truncate at 4000 (max VARCHAR2 length in STANDARD mode)
                dtypes[column] = oracle.VARCHAR2(int(np.clip(a=3.0 * max_length, a_min=100, a_max=4_000)))
            elif isinstance(to_sql_dtypes[column], sa.Float):
                # SQLAlchemy resolves floats to FLOAT with precision, which is not translatable to Oracle,
                # we use FLOAT with default binary_precision from Oracle instead
                dtypes[column] = oracle.FLOAT()
        kwargs = kwargs | {"dtype": dtypes}
        super().create_table(df, **kwargs)

    def write_data(self, df: pd.DataFrame, **kwargs):
        self.INIT_WRITE_CHUNK = configure_oracle_client
        super().write_data(df, **kwargs)
        self.container.get_sa_engine().dispose()  # clean pool after writing data to Oracle db
