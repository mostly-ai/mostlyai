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
from sqlalchemy import text
from sqlalchemy.dialects.mysql.base import MySQLDialect

from mostlyai.sdk._data.db.base import DBDType, SqlAlchemyContainer, SqlAlchemyTable

_LOG = logging.getLogger(__name__)

_MY_SQL_PRIVILEGE_DB_NAME = "mysql"
_MY_SQL_PRIVILEGE_TABLE_NAME = "user"


class MysqlDType(DBDType):
    UNBOUNDED_VARCHAR_ALLOWED = False

    @classmethod
    def sa_dialect_class(cls):
        return MySQLDialect


class BaseMySqlContainer(SqlAlchemyContainer):
    SCHEMES = ["mysql", "mariadb"]
    SA_CONNECTION_KWARGS = {}
    SA_SSL_ATTR_KEY_MAP = {
        "root_certificate_path": "ssl_ca",
        "ssl_certificate_path": "ssl_cert",
        "ssl_certificate_key_path": "ssl_key",
    }
    DIALECT = ""
    SQL_FETCH_FOREIGN_KEYS = """
    SELECT
        TABLE_NAME as TABLE_NAME,
        COLUMN_NAME as COLUMN_NAME,
        REFERENCED_TABLE_NAME as REFERENCED_TABLE_NAME,
        REFERENCED_COLUMN_NAME as REFERENCED_COLUMN_NAME
    FROM
        information_schema.KEY_COLUMN_USAGE
    WHERE
        CONSTRAINT_SCHEMA = :schema_name AND
        REFERENCED_TABLE_NAME IS NOT NULL;
    """
    INIT_DEFAULT_VALUES = {"dbname": "", "port": "3306"}

    def _get_uri_without_dbname(self):
        # User and password are needed to avoid double-encoding of @ character
        username = quote(self.username)
        password = quote(self.password)
        return f"{self.DIALECT}+mysqlconnector://{username}:{password}@{self.host}:{self.port}"

    @property
    def sa_uri(self):
        sa_uri = f"{self._get_uri_without_dbname()}/{self.dbname}"
        return sa_uri

    @property
    def sa_uri_for_does_database_exist(self):
        return self._get_uri_without_dbname()

    @classmethod
    def table_class(cls):
        return MysqlLikeTable

    def does_database_exist(self) -> bool:
        with self.init_sa_connection("db_exist_check") as connection:
            result = connection.execute(
                text(f"SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA WHERE SCHEMA_NAME='{self.dbname}'")
            )
        return bool(result.rowcount)

    def update_dbschema(self, dbschema: str | None) -> None:
        # schema (as a prefix) is equivalent to db name in mysql
        self.dbname = dbschema or self.INIT_DEFAULT_VALUES.get("dbname")
        # reset engine
        self._sa_engine_for_read = None
        self._sa_engine_for_write = None


class MysqlContainer(BaseMySqlContainer):
    DIALECT = "mysql"


class MariadbContainer(BaseMySqlContainer):
    DIALECT = "mariadb"


class MysqlLikeTable(SqlAlchemyTable):
    DATA_TABLE_TYPE = "mysql"
    SA_RANDOM = sa.func.rand()

    @classmethod
    def dtype_class(cls):
        return MysqlDType

    @classmethod
    def container_class(cls):
        return MysqlContainer
