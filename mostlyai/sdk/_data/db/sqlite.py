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

import sqlalchemy as sa
from sqlalchemy.dialects.sqlite.base import SQLiteDialect

from mostlyai.sdk._data.db.base import DBDType, SqlAlchemyContainer, SqlAlchemyTable

_LOG = logging.getLogger(__name__)


class SqliteDType(DBDType):
    @classmethod
    def sa_dialect_class(cls):
        return SQLiteDialect


class SqliteContainer(SqlAlchemyContainer):
    SCHEMES = ["sqlite"]
    SA_CONNECT_ARGS_ACCESS_ENGINE = {"timeout": 3}

    @property
    def sa_uri(self):
        return f"sqlite+pysqlite:///{self.dbname}"

    @classmethod
    def table_class(cls):
        return SqliteTable

    def _is_schema_exist(self):
        return True

    def does_database_exist(self) -> bool:
        return True


class SqliteTable(SqlAlchemyTable):
    DATA_TABLE_TYPE = "sqlite"
    SA_RANDOM = sa.func.random()

    @classmethod
    def dtype_class(cls):
        return SqliteDType

    @classmethod
    def container_class(cls):
        return SqliteContainer
