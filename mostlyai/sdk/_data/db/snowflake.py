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

import sqlalchemy as sa
from snowflake.sqlalchemy import URL
from snowflake.sqlalchemy.snowdialect import SnowflakeDialect

from mostlyai.sdk._data.db.base import DBDType, SqlAlchemyContainer, SqlAlchemyTable
from mostlyai.sdk._data.exceptions import MostlyDataException

_LOG = logging.getLogger(__name__)

ACCOUNT_SUFFIX_TO_REMOVE = ".snowflakecomputing.com"
ACCOUNT_PREFIX_TO_REMOVE = "https://"
DEFAULT_WAREHOUSE = "COMPUTE_WH"


class SnowflakeDType(DBDType):
    @classmethod
    def sa_dialect_class(cls):
        return SnowflakeDialect


class SnowflakeContainer(SqlAlchemyContainer):
    SCHEMES = ["snowflake"]
    INIT_DEFAULT_VALUES = {"dbname": ""}

    def __init__(self, *args, account, **kwargs):
        super().__init__(*args, **kwargs)
        self.account = account

    @property
    def sa_uri(self):
        # User and password are needed to avoid double-encoding of @ character
        username = quote(self.username)
        password = quote(self.password)
        self.host = self.host or DEFAULT_WAREHOUSE
        return URL(
            user=username,
            password=password,
            account=self.account.replace(ACCOUNT_SUFFIX_TO_REMOVE, "").replace(ACCOUNT_PREFIX_TO_REMOVE, ""),
            warehouse=self.host,
            database=self.dbname,
            cache_column_metadata=True,
        )

    @classmethod
    def table_class(cls):
        return SnowflakeTable

    def does_database_exist(self) -> bool:
        with self.init_sa_connection("db_exist_check") as connection:
            result = connection.execute(sa.text(f"SHOW DATABASES LIKE '{self.dbname}'"))
        if result.rowcount:
            db_exist = True
        else:
            db_exist = False
        return db_exist

    def _is_schema_exist(self) -> bool:
        if self.dbschema is None:
            return True
        with self.use_sa_engine() as sa_engine:
            schema_names = sa.inspect(sa_engine).get_schema_names()
        return self.dbschema.lower() in schema_names

    def is_accessible(self) -> bool:
        try:
            with self.init_sa_connection("access_check"):
                if self.dbname and not self.does_database_exist():
                    raise MostlyDataException(f"Database `{self.dbname}` does not exist.")
                elif self.schema and not self._is_schema_exist():
                    raise MostlyDataException(f"Schema `{self.dbschema}` does not exist.")
                else:
                    return True
        except sa.exc.SQLAlchemyError as e:
            error_message = str(e).lower()
            _LOG.error(f"Database connection failed with error: {e}")
            if "password" in error_message or "user" or ("account name" and "snowflake") in error_message:
                raise MostlyDataException("Credentials are incorrect.")
            else:
                raise


class SnowflakeTable(SqlAlchemyTable):
    DATA_TABLE_TYPE = "snowflake"
    SA_RANDOM = sa.func.random()

    @classmethod
    def dtype_class(cls):
        return SnowflakeDType

    @classmethod
    def container_class(cls):
        return SnowflakeContainer
