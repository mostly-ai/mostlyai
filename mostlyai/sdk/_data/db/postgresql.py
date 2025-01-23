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
from urllib.parse import quote

import numpy as np
import sqlalchemy as sa
from psycopg2.extensions import AsIs, register_adapter
from sqlalchemy.dialects.postgresql.base import PGDialect

from mostlyai.sdk._data.db.base import DBDType, SqlAlchemyContainer, SqlAlchemyTable

_LOG = logging.getLogger(__name__)

register_adapter(np.int64, AsIs)
register_adapter(np.float64, AsIs)


class PostgresqlDType(DBDType):
    FROM_VIRTUAL_DATETIME = sa.TIMESTAMP

    @classmethod
    def sa_dialect_class(cls):
        return PGDialect


class PostgresqlContainer(SqlAlchemyContainer):
    SCHEMES = ["postgresql"]
    SA_CONNECTION_KWARGS = {"sslmode": "require"}
    SA_SSL_ATTR_KEY_MAP = {
        "root_certificate_path": "sslrootcert",
        "ssl_certificate_path": "sslcert",
        "ssl_certificate_key_path": "sslkey",
    }
    SQL_FETCH_FOREIGN_KEYS = """
    SELECT
        kcu.table_name AS TABLE_NAME,
        kcu.column_name AS COLUMN_NAME,
        ccu.table_name AS REFERENCED_TABLE_NAME,
        ccu.column_name AS REFERENCED_COLUMN_NAME
    FROM
        information_schema.table_constraints AS tc
        JOIN information_schema.key_column_usage AS kcu
            ON tc.constraint_name = kcu.constraint_name
            AND tc.table_schema = kcu.table_schema
        JOIN information_schema.constraint_column_usage AS ccu
            ON ccu.constraint_name = tc.constraint_name
            AND ccu.table_schema = tc.table_schema
    WHERE
        tc.constraint_type = 'FOREIGN KEY'
        AND tc.table_schema = :schema_name;
    """
    INIT_DEFAULT_VALUES = {"dbname": "", "port": "5432"}

    @property
    def sa_uri(self):
        # User and password are needed to avoid double-encoding of @ character
        username = quote(self.username)
        password = quote(self.password)
        return f"postgresql+psycopg2://{username}:{password}@{self.host}:{self.port}/{self.dbname}"

    @property
    def sa_create_engine_kwargs(self) -> dict:
        return {
            # read more: https://docs.sqlalchemy.org/en/14/dialects/postgresql.html#psycopg2-fast-execution-helpers
            "executemany_mode": "values_only",
            # the following knobs can be used in attempts to improve speed of writing
            # however, no better setting was found than the default
            # "executemany_values_page_size": 1_000,  # default
            # "executemany_batch_page_size": 100      # default
        }

    @classmethod
    def table_class(cls):
        return PostgresqlTable

    def does_database_exist(self) -> bool:
        try:
            with self.init_sa_connection() as connection:
                result = connection.execute(sa.text(f"SELECT 1 FROM pg_database WHERE datname='{self.dbname}'"))
            return bool(result.rowcount)
        except Exception as e:
            _LOG.error(f"Error when checking if database exists: {e}")
            return False


class PostgresqlTable(SqlAlchemyTable):
    DATA_TABLE_TYPE = "postgresql"
    SA_RANDOM = sa.func.random()

    @classmethod
    def dtype_class(cls):
        return PostgresqlDType

    @classmethod
    def container_class(cls):
        return PostgresqlContainer
