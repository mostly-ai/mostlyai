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
import math
import re
from typing import Literal
from urllib.parse import quote

import numpy as np
import pandas as pd
import sqlalchemy as sa
from impala.dbapi import connect as impala_connect
from impala.sqlalchemy import ImpalaDialect, _impala_type_to_sqlalchemy_type
from pyhive.sqlalchemy_hive import HiveDialect
from sqlalchemy.dialects import registry

from mostlyai.sdk._data.db.base import DBDType, SqlAlchemyContainer, SqlAlchemyTable
from mostlyai.sdk._data.exceptions import MostlyDataException
from mostlyai.sdk._data.util.common import run_with_timeout_unsafe

_LOG = logging.getLogger(__name__)


registry.register("impala_mostly", "mostlyai.sdk._data.db.hive", "MostlyImpalaDialect")


class MostlyImpalaDialect(ImpalaDialect):
    def has_table(self, connection, table_name, schema=None, **kwargs):
        # adds **kwargs to ImpalaDialect.has_table to work with SA2.0
        return super().has_table(connection, table_name, schema)

    def get_table_names(self, connection, schema=None, **kw):
        # copied from ImpalaDialect.get_table_names and adjusted to work with SA2.0
        query = "SHOW TABLES"
        if schema is not None:
            escaped_schema = re.sub(r"^[\`]{1,}|[\`]{1,}$", "", schema)
            escaped_schema = re.sub(r"^|$", "`", escaped_schema)
            query += " IN %s" % escaped_schema
        query = sa.text(query)  # MOSTLY AI CHANGE
        tables = [tup[1] if len(tup) > 1 else tup[0] for tup in connection.execute(query).fetchall()]
        return tables

    def get_columns(self, connection, table_name, schema=None, **kwargs):
        # copied from ImpalaDialect.get_columns and adjusted to work with SA2.0
        # pylint: disable=unused-argument
        name = table_name
        if schema is not None:
            name = f"{schema}.{name}"
        query = "SELECT * FROM %s LIMIT 0" % name
        query = sa.text(query)  # MOSTLY AI CHANGE
        cursor = connection.execute(query)
        schema = cursor.cursor.description
        # We need to fetch the empty results otherwise these queries remain in
        # flight
        cursor.fetchall()
        column_info = []
        for col in schema:
            column_info.append(
                {
                    "name": col[0],
                    "type": _impala_type_to_sqlalchemy_type[col[1]],
                    "nullable": True,
                    "autoincrement": False,
                }
            )
        return column_info


class HiveDType(DBDType):
    @classmethod
    def sa_dialect_class(cls):
        return HiveDialect


class HiveContainer(SqlAlchemyContainer):
    SCHEMES = ["hive"]
    # it's not possible to set timeout for hive engine
    # see: https://github.com/dropbox/PyHive/issues/162
    # instead, we use Timeout context manager to handle timeouts
    SA_CONNECT_ARGS_ACCESS_ENGINE = {}
    IS_ACCESSIBLE_TIMEOUT = 60  # seconds
    INIT_DEFAULT_VALUES = {"dbname": "", "port": "10000"}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.impala_conn = None

    @property
    def sa_create_engine_kwargs(self) -> dict:
        if not self.impala_conn:
            if self.kerberos_enabled and self.ssl_enabled:
                with self.use_ssl_connection():
                    self.impala_conn = impala_connect(
                        host=self.host,
                        port=int(self.port),
                        use_ssl=True,
                        auth_mechanism="GSSAPI",  # Kerberos
                        kerberos_service_name=self.kerberos_service_name,
                        ca_cert=getattr(self, "ca_certificate_path", None),
                    )
            elif self.kerberos_enabled and not self.ssl_enabled:
                self.impala_conn = impala_connect(
                    host=self.host,
                    port=int(self.port),
                    use_ssl=False,
                    auth_mechanism="GSSAPI",  # Kerberos
                    kerberos_service_name=self.kerberos_service_name,
                )
            else:  # not kerberos_enabled and not ssl_enabled
                self.impala_conn = impala_connect(
                    host=self.host,
                    port=int(self.port),
                    use_ssl=False,
                    auth_mechanism="LDAP",  # User/Password
                    user=self.username,
                    password=self.password,
                )
        return {"creator": lambda: self.impala_conn}

    def _get_uri_without_dbname(self, method: Literal["read", "write"] = "read"):
        username_password = ""
        if not self.kerberos_enabled and self.username is not None and self.password is not None:
            username = quote(self.username)
            password = quote(self.password)
            username_password = f"{username}:{password}"
        # for both kerberos and user/password we use the same protocol
        # due to impyla bug we need a separate protocol for read / write
        protocol = "impala_mostly" if method == "write" else "hive"
        return f"{protocol}://{username_password}@{self.host}:{self.port}"

    def _get_uri_query_params(self):
        if self.kerberos_enabled:
            parsed = self._parse_kerberos_service_principal()
            return f"auth=KERBEROS&kerberos_service_name={parsed['primary']}"
        return "auth=CUSTOM"

    @property
    def sa_uri(self):
        sa_uri = f"{self._get_uri_without_dbname('read')}/{self.dbname}?{self._get_uri_query_params()}"
        return sa_uri

    @property
    def sa_uri_for_does_database_exist(self):
        return f"{self._get_uri_without_dbname('read')}/?{self._get_uri_query_params()}"

    @property
    def sa_uri_for_write(self):
        return f"{self._get_uri_without_dbname('write')}/?{self._get_uri_query_params()}"

    def is_accessible(self) -> bool:
        try:
            return run_with_timeout_unsafe(super().is_accessible, timeout=self.IS_ACCESSIBLE_TIMEOUT, do_raise=True)
        except TimeoutError as e:
            _LOG.error(f"Database connection failed with error: {e}")
            raise MostlyDataException("Cannot establish connection. Host, port, or credentials are incorrect.")
        except Exception:
            raise

    @classmethod
    def table_class(cls):
        return HiveTable

    def use_sa_engine(self, mode="read_data", dispose=False):
        # always keep the engine alive to reuse the cached connection
        return super().use_sa_engine(mode=mode, dispose=dispose)

    def does_database_exist(self) -> bool:
        with self.init_sa_connection("db_exist_check") as connection:
            result = connection.execute(sa.text("SHOW DATABASES"))
            for (db_name,) in result:
                if db_name == self.dbname:
                    return True
        return False


class HiveTable(SqlAlchemyTable):
    DATA_TABLE_TYPE = "hive"
    SA_RANDOM = sa.func.rand()
    SA_MULTIPLE_INSERTS = True
    MIN_WRITE_CHUNK_SIZE = 100
    MAX_WRITE_CHUNK_SIZE = 10_000
    MAX_DATA_POINTS_PER_WRITE_CHUNK = 1_000_000
    WRITE_CHUNKS_N_JOBS = 1

    def calculate_write_chunk_size(self, df: pd.DataFrame) -> int:
        n_cols = max(len(df.columns), 1)
        chunk_size = math.ceil(self.MAX_DATA_POINTS_PER_WRITE_CHUNK / n_cols)
        return int(np.clip(chunk_size, self.MIN_WRITE_CHUNK_SIZE, self.MAX_WRITE_CHUNK_SIZE))

    @classmethod
    def dtype_class(cls):
        return HiveDType

    @classmethod
    def container_class(cls):
        return HiveContainer
