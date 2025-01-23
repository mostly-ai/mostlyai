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
from urllib.parse import quote, urlencode

import pyodbc
import sqlalchemy as sa
from sqlalchemy.dialects.mssql.base import MSDialect

from mostlyai.sdk._data.db.base import DBDType, SqlAlchemyContainer, SqlAlchemyTable

# disable pyodbc pooling, SQLAlchemy utilises its own mechanism
# read more: https://docs.sqlalchemy.org/en/14/dialects/mssql.html#pyodbc-pooling-connection-close-behavior
pyodbc.pooling = False

_LOG = logging.getLogger(__name__)


class MssqlDType(DBDType):
    FROM_VIRTUAL_TIMESTAMP = sa.DATETIME

    @classmethod
    def sa_dialect_class(cls):
        return MSDialect


class MssqlContainer(SqlAlchemyContainer):
    SCHEMES = ["mssql"]
    SA_CONNECT_ARGS_ACCESS_ENGINE = {"timeout": 3}
    SA_CONNECTION_KWARGS = {
        "ssl": "True",
    }
    SA_SSL_ATTR_KEY_MAP = {
        "root_certificate_path": "Certificate",
    }
    SQL_FETCH_FOREIGN_KEYS = """
    SELECT
        FK.TABLE_NAME as TABLE_NAME,
        CU.COLUMN_NAME as COLUMN_NAME,
        PK.TABLE_NAME as REFERENCED_TABLE_NAME,
        PT.COLUMN_NAME as REFERENCED_COLUMN_NAME
    FROM
        INFORMATION_SCHEMA.REFERENTIAL_CONSTRAINTS C
        INNER JOIN INFORMATION_SCHEMA.TABLE_CONSTRAINTS FK ON C.CONSTRAINT_NAME = FK.CONSTRAINT_NAME
        INNER JOIN INFORMATION_SCHEMA.TABLE_CONSTRAINTS PK ON C.UNIQUE_CONSTRAINT_NAME = PK.CONSTRAINT_NAME
        INNER JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE CU ON C.CONSTRAINT_NAME = CU.CONSTRAINT_NAME
        INNER JOIN (
            SELECT
                i1.TABLE_NAME,
                i2.COLUMN_NAME
            FROM
                INFORMATION_SCHEMA.TABLE_CONSTRAINTS i1
                INNER JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE i2 ON i1.CONSTRAINT_NAME = i2.CONSTRAINT_NAME
            WHERE
                i1.CONSTRAINT_TYPE = 'PRIMARY KEY'
                AND i1.TABLE_SCHEMA = :schema_name
        ) PT ON PT.TABLE_NAME = PK.TABLE_NAME
    WHERE
        FK.TABLE_SCHEMA = :schema_name
        AND PK.TABLE_SCHEMA = :schema_name
        AND CU.TABLE_SCHEMA = :schema_name;
    """
    _MSSQL_PYODBC_DRIVER = "ODBC Driver 18 for SQL Server"
    INIT_DEFAULT_VALUES = {"dbname": "master", "port": "1433"}

    @property
    def sa_uri(self):
        # User and password are needed to avoid double-encoding of @ character
        username = quote(self.username)
        password = quote(self.password)
        props = urlencode(
            {
                "driver": self._MSSQL_PYODBC_DRIVER,
                "TrustServerCertificate": "yes",
                **self.sa_engine_connection_kwargs,
            }
        )
        uri = f"mssql+pyodbc://{username}:{password}@{self.host}:{self.port}/{self.dbname}?{props}"
        return uri

    @property
    def sa_create_engine_kwargs(self) -> dict:
        return {
            # improves speed of write dramatically
            # read more: https://docs.sqlalchemy.org/en/14/dialects/mssql.html#fast-executemany-mode
            "fast_executemany": True
        }

    @classmethod
    def table_class(cls):
        return MssqlTable

    def does_database_exist(self) -> bool:
        try:
            with self.init_sa_connection("db_exist_check") as connection:
                result = connection.execute(sa.text(f"SELECT name FROM sys.databases WHERE name='{self.dbname}'"))
                if result.fetchone():
                    db_exist = True
                else:
                    db_exist = False
                return db_exist
        except Exception:
            return False


class MssqlTable(SqlAlchemyTable):
    DATA_TABLE_TYPE = "mssql"
    SA_RANDOM = sa.func.newid()
    # MSSQL has upper bound of 2100 on number of bound parameters in a query,
    # each batch contributes len(batch) bound parameters to the counter,
    # so max batch size must be significantly smaller than 2100 to
    # leave some space for other bound parameters in a query, e.g. column names
    SA_MAX_VALS_PER_BATCH = 1_900

    @classmethod
    def dtype_class(cls):
        return MssqlDType

    @classmethod
    def container_class(cls):
        return MssqlContainer
