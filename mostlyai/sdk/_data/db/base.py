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

import abc
import base64
import decimal
import enum
import functools
import hashlib
import logging
import os
import re
import shutil
import socket
import subprocess
import tempfile
import time
import uuid
from collections import defaultdict
from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
import sqlalchemy as sa
import sqlalchemy.sql.sqltypes as sa_types
import sshtunnel
from joblib import Parallel, delayed
from sqlalchemy import Table
from sqlalchemy.orm import sessionmaker

from mostlyai.sdk._data.base import (
    DataContainer,
    DataTable,
    ForeignKey,
    order_df_by,
)
from mostlyai.sdk._data.db.types_coercion import coerce_to_sql_dtype
from mostlyai.sdk._data.dtype import (
    FLOAT64,
    STRING,
    VirtualBoolean,
    VirtualDate,
    VirtualDatetime,
    VirtualDType,
    VirtualFloat,
    VirtualInteger,
    VirtualTimestamp,
    VirtualVarchar,
    WrappedDType,
    coerce_dtypes_by_encoding,
)
from mostlyai.sdk._data.exceptions import MostlyDataException
from mostlyai.sdk._data.util.common import ColumnSort, OrderBy, assert_read_only_sql, prepare_ssl_path
from mostlyai.sdk._data.util.kerberos import is_kerberos_ticket_alive

_LOG = logging.getLogger(__name__)

ConnectionMode = Literal["read_data", "write_data", "access_check", "db_exist_check"]

KRB5_CONF_TEMPLATE = """
[libdefaults]
    default_realm = {kerberos_realm}
    dns_lookup_realm = false
    dns_lookup_kdc = false
    forwardable = true
    rdns = false
[realms]
    {kerberos_realm} = {{
        kdc = {kerberos_kdc_host}
    }}
"""
SSL_ATTRIBUTES = [
    "root_certificate",
    "ssl_certificate",
    "ssl_certificate_key",
    "ca_certificate",
]


class DBContainer(DataContainer, abc.ABC):
    pass


class DBDType(WrappedDType, abc.ABC):
    # from_virtual() / to_virtual() hooks
    UNBOUNDED_VARCHAR_ALLOWED = True
    DEFAULT_VARCHAR_LENGTH = 255
    FROM_VIRTUAL_BOOLEAN = sa.BOOLEAN
    FROM_VIRTUAL_INTEGER = sa.INTEGER
    FROM_VIRTUAL_FLOAT = sa.FLOAT
    FROM_VIRTUAL_DATETIME = sa.DATETIME
    FROM_VIRTUAL_TIMESTAMP = sa.TIMESTAMP

    DEFAULT_NAME_SA_TYPE_MAP = defaultdict(
        lambda: sa_types.String,
        {
            "VARCHAR": sa_types.String,
            "INTEGER": sa_types.Integer,
            "DATETIME": sa_types.DateTime,
            "TIMESTAMP": sa_types.DateTime,
            "DATE": sa_types.Date,
            "TIME": sa_types.Time,
            "BOOLEAN": sa_types.Boolean,
            "NULL": sa_types.NullType,
            "NullType": sa_types.NullType,
            "FLOAT": sa_types.Float,
            "DOUBLE": sa_types.Float,
            "DECIMAL": sa_types.Numeric,
        },
    )

    @classmethod
    def from_virtual(cls, dtype: VirtualDType) -> "WrappedDType":
        if isinstance(dtype, VirtualBoolean):
            return cls(cls.FROM_VIRTUAL_BOOLEAN())
        elif isinstance(dtype, VirtualInteger):
            return cls(cls.FROM_VIRTUAL_INTEGER())
        elif isinstance(dtype, VirtualFloat):
            return cls(cls.FROM_VIRTUAL_FLOAT())
        elif isinstance(dtype, VirtualVarchar):
            length = dtype.length if cls.UNBOUNDED_VARCHAR_ALLOWED else dtype.length or cls.DEFAULT_VARCHAR_LENGTH
            return cls(sa.VARCHAR(length))
        elif isinstance(dtype, VirtualDate):
            return cls(sa.DATE())
        elif isinstance(dtype, VirtualDatetime):
            return cls(cls.FROM_VIRTUAL_DATETIME())
        elif isinstance(dtype, VirtualTimestamp):
            return cls(cls.FROM_VIRTUAL_TIMESTAMP())
        else:
            raise NotImplementedError(f"handler for virtual dtype [{dtype}] is not specified")

    def to_virtual(self) -> VirtualDType:
        wrapped_type = type(self.wrapped).__name__.upper()

        if "BOOLEAN" in wrapped_type:
            return VirtualBoolean()
        elif any(t in wrapped_type for t in ["INTEGER", "BIGINT", "NUMBER", "SMALLINT", "TINYINT", "MEDIUMINT"]):
            return VirtualInteger()
        elif any(t in wrapped_type for t in ["FLOAT", "NUMERIC", "MONEY", "DOUBLE", "DECIMAL", "REAL"]):
            return VirtualFloat()
        elif any(t in wrapped_type for t in ["VARCHAR", "STRING"]):
            return VirtualVarchar(self.wrapped.length)
        elif "TEXT" in wrapped_type:
            return VirtualVarchar(4096)
        elif "DATETIME" in wrapped_type:
            return VirtualDatetime()
        elif "DATE" in wrapped_type:
            return VirtualDate()
        elif "TIMESTAMP" in wrapped_type:
            return VirtualTimestamp()
        else:
            _LOG.warning(
                f"no virtual type for [{self.wrapped}] found; returning VirtualVarchar({self.DEFAULT_VARCHAR_LENGTH})"
            )
            return VirtualVarchar(self.DEFAULT_VARCHAR_LENGTH)

    def coerce(self, data: pd.Series) -> pd.Series:
        return coerce_to_sql_dtype(data, self.wrapped)

    @staticmethod
    def _parse_name_and_args(s: str) -> tuple[str | None, tuple]:
        match = re.match(r"(\w+)(?:\(([^)]*)\))?", s)
        name = None
        args = ()
        if match:
            name = match.group(1)
            args_str = match.group(2)
            if args_str is not None:
                try:
                    args = tuple(map(int, args_str.split(",")))
                except ValueError:
                    _LOG.warning(f"Could not parse {args=} as integers. Falling back to no args")

        return name, args

    @classmethod
    def sa_dialect_class(cls):
        return sa.engine.DefaultDialect

    @classmethod
    def sa_dialect_types(cls) -> list[type]:
        dialect_class = cls.sa_dialect_class()
        if not hasattr(dialect_class, "ischema_names") or not isinstance(dialect_class.ischema_names, dict):
            return []

        types = dialect_class.ischema_names.values()
        return types

    @classmethod
    def _name_to_sa_class(cls, name: str | None) -> type | None:
        types = cls.sa_dialect_types()
        name_type_map = {t.__name__: t for t in types}
        sa_class = name_type_map.get(name) or cls.DEFAULT_NAME_SA_TYPE_MAP.get(name)
        return sa_class

    @classmethod
    def parse(cls, dtype: str) -> Optional["WrappedDType"]:
        name, args = cls._parse_name_and_args(dtype)
        sa_class = cls._name_to_sa_class(name)
        if not sa_class:
            return None
        try:
            return cls(sa_class(*args))
        except TypeError:
            _LOG.warning(f"Could not instantiate {sa_class} with {args}. Falling back to no args")
            return cls(sa_class())


class DBTable(DataTable, abc.ABC):
    pass


class SqlAlchemyContainer(DBContainer, abc.ABC):
    SA_CONNECT_ARGS_ACCESS_ENGINE: dict[str, Any] = {"connect_timeout": 3}
    SA_CONNECTION_KWARGS: dict[str, str] = {}
    SA_SSL_ATTR_KEY_MAP: dict[str, str] = {}
    SQL_FETCH_FOREIGN_KEYS = None
    INIT_DEFAULT_VALUES: dict[str, Any] = {}
    SECRET_ATTR_NAME: str = "password"

    def __init__(
        self,
        *args,
        username: str | None = None,
        password: str | None = None,
        host: str | None = None,
        port: str | None = None,
        dbname: str | None = None,
        dbschema: str | None = None,
        do_decrypt_secret: bool = True,
        ssl_enabled: bool = False,
        root_certificate: str | None = None,
        ssl_certificate: str | None = None,
        ssl_certificate_key: str | None = None,
        ca_certificate: str | None = None,
        enable_ssh: bool = False,
        ssh_host: str | None = None,
        ssh_port: str | None = None,
        ssh_username: str | None = None,
        ssh_password: str | None = None,
        ssh_private_key_path: str | None = None,
        kerberos_enabled: bool = False,
        kerberos_kdc_host: str | None = None,
        kerberos_krb5_conf: str | None = None,
        kerberos_service_principal: str | None = None,
        kerberos_client_principal: str | None = None,
        kerberos_keytab: str | None = None,
        **kwargs,
    ):
        self.username = username
        self.password = password
        self.host = host
        self.port = port or self.INIT_DEFAULT_VALUES.get("port")
        self.dbname = dbname or self.INIT_DEFAULT_VALUES.get("dbname")
        self.dbschema = dbschema
        self.do_decrypt_secret = do_decrypt_secret

        # SSL
        self.ssl_enabled = ssl_enabled
        self.root_certificate = root_certificate
        self.ssl_certificate = ssl_certificate
        self.ssl_certificate_key = ssl_certificate_key
        self.ca_certificate = ca_certificate

        # SSH
        self.enable_ssh = enable_ssh
        self.ssh_host = ssh_host
        self.ssh_port = ssh_port
        self.ssh_username = ssh_username
        self.ssh_password = ssh_password
        self.ssh_private_key_path = ssh_private_key_path

        # Kerberos
        self.kerberos_enabled = kerberos_enabled
        self.kerberos_kdc_host = kerberos_kdc_host
        self.kerberos_krb5_conf = kerberos_krb5_conf
        self.kerberos_service_principal = kerberos_service_principal
        self.kerberos_client_principal = kerberos_client_principal
        self.kerberos_keytab = kerberos_keytab

        self.props = kwargs
        self.sa_metadata = sa.MetaData(schema=self.dbschema)
        self.filtered_tables = kwargs.get("filtered_tables")
        self._sa_engine_for_read = None
        self._sa_engine_for_write = None
        self._foreign_keys = None
        super().__init__(*args, **kwargs)
        self.post_init_hook()

        # secure connection flags
        # - SSH tunnel should be reused if already exist
        # - SSL should reuse keys if they are present
        # - Kerberos should reuse tickets if they are present
        self._is_ssl_active = False
        self._is_ssh_active = False
        self._is_kerberos_active = False

    def __repr__(self):
        # password field will be masked for security reasons
        return f"{self.__class__.__name__}(sa_uri={self.get_sa_engine().url.__repr__()}, dbschema={self.dbschema})"

    def post_init_hook(self):
        if self.do_decrypt_secret:
            self.decrypt_secret()
        if self.ssl_enabled:
            self.load_ssl_files_and_prepare_ssl_paths()

    def get_sa_engine(self, mode: ConnectionMode = "read_data") -> sa.engine.Engine:
        default_kwargs = {
            "url": self.sa_uri,
            "connect_args": self.sa_engine_connection_kwargs,
            **self.sa_create_engine_kwargs,
        }

        def get_sa_engine_for_read_data() -> sa.engine.Engine:
            # default read engine is cached
            if self._sa_engine_for_read is None:
                self._sa_engine_for_read = sa.create_engine(**default_kwargs)
            return self._sa_engine_for_read

        def get_sa_engine_for_write_data() -> sa.engine.Engine:
            # default write engine is cached
            if self._sa_engine_for_write is None:
                default_kwargs["url"] = self.sa_uri_for_write
                self._sa_engine_for_write = sa.create_engine(**default_kwargs)
            return self._sa_engine_for_write

        def get_sa_engine_for_access_check() -> sa.engine.Engine:
            # adhoc engine is always created anew
            connect_args = self.SA_CONNECT_ARGS_ACCESS_ENGINE | default_kwargs["connect_args"]
            kwargs = default_kwargs | {"connect_args": connect_args}
            return sa.create_engine(**kwargs)

        def get_sa_engine_for_db_exist_check() -> sa.engine.Engine:
            # adhoc engine is always created anew
            connect_args = self.SA_CONNECT_ARGS_ACCESS_ENGINE | default_kwargs["connect_args"]
            url = self.sa_uri_for_does_database_exist
            kwargs = default_kwargs | {"url": url, "connect_args": connect_args}
            return sa.create_engine(**kwargs)

        if mode == "read_data":
            return get_sa_engine_for_read_data()
        elif mode == "write_data":
            return get_sa_engine_for_write_data()
        elif mode == "access_check":
            return get_sa_engine_for_access_check()
        elif mode == "db_exist_check":
            return get_sa_engine_for_db_exist_check()

    @contextmanager
    def use_sa_engine(
        self,
        mode: ConnectionMode = "read_data",
        dispose: bool = True,
    ) -> Generator[sa.engine.Engine, None, None]:
        with self.kerberized(), self.use_ssh_tunnel(), self.use_ssl_connection():
            sa_engine = self.get_sa_engine(mode=mode)
            yield sa_engine
            if dispose:
                sa_engine.dispose()

    @contextmanager
    def kerberized(self):
        if not self.kerberos_enabled or self._is_kerberos_active:
            yield
            return

        self._is_kerberos_active = True

        def is_ticket_alive(service_principal: str) -> bool:
            try:
                p_result = subprocess.run(
                    ["klist", "-c", krb5_cache_path],
                    capture_output=True,
                    check=True,
                )
                result = p_result.stdout.decode("utf-8")
                is_alive = is_kerberos_ticket_alive(result, service_principal)
                _LOG.info(f"Kerberos ticket for `{service_principal}` is_alive={is_alive}")
                return is_alive
            except subprocess.CalledProcessError:
                return False
            except Exception as e:
                _LOG.info(f"Error occurred while checking a Kerberos ticket state {e}; Assuming is_alive=False")
                return False

        def kinit():
            # create temporary keytab file for kinit
            with tempfile.NamedTemporaryFile("wb") as keytab_file:
                # decrypt keytab and store it to the temporary file
                kerberos_keytab = base64.b64decode(self.decrypt(self.kerberos_keytab))
                keytab_file.write(kerberos_keytab)
                keytab_file.flush()
                keytab_path = keytab_file.name
                try:
                    # if client principal exists, use it to request the ticket
                    # otherwise, use service principal instead
                    principal = self.kerberos_client_principal or self.kerberos_service_principal
                    subprocess.run(
                        [
                            "kinit",
                            "-c",
                            krb5_cache_path,
                            "-kt",
                            keytab_path,
                            principal,
                        ],
                        capture_output=True,
                        check=True,
                    )
                    _LOG.info("Kerberos `kinit` succeeded")
                except subprocess.CalledProcessError as e:
                    _LOG.error("Kerberos `kinit` failed")
                    _LOG.error(f"stdout: {e.stdout.decode()}")
                    _LOG.error(f"stderr: {e.stderr.decode()}")
                    raise MostlyDataException("Kerberos initialization failed.")

        # resolve kerberos ticket cache for provided principal
        hashing = hashlib.sha256()
        hash_input = f"{self.kerberos_service_principal}{self.kerberos_keytab}"
        hashing.update(hash_input.encode())
        cache_name = hashing.hexdigest()
        krb5_cache_path = f"{tempfile.gettempdir()}/{cache_name}"

        # create temporary krb5.conf file
        parsed = self._parse_kerberos_service_principal()
        if self.kerberos_krb5_conf is not None:
            krb5_conf = self.kerberos_krb5_conf
        else:
            krb5_conf = KRB5_CONF_TEMPLATE.format(
                kerberos_kdc_host=self.kerberos_kdc_host, kerberos_realm=parsed["realm"]
            )
        with tempfile.NamedTemporaryFile("w") as krb5_conf_file:
            krb5_conf_file.write(krb5_conf)
            krb5_conf_file.flush()
            # set KRB5_CONFIG for kerberos configuration file
            # set KRB5CCNAME for isolated kerberos ticket cache
            os.environ["KRB5_CONFIG"] = krb5_conf_file.name
            os.environ["KRB5CCNAME"] = krb5_cache_path
            if not is_ticket_alive(service_principal=self.kerberos_service_principal):
                # if cache does not present valid ticket, kinit a new one
                kinit()
            try:
                yield
            finally:
                # clean up environment
                del os.environ["KRB5_CONFIG"]
                del os.environ["KRB5CCNAME"]
        self._is_kerberos_active = False

    def _parse_kerberos_service_principal(self):
        # should match 'primary@REALM' or 'primary/instance@REALM'
        pattern = r"^(?P<primary>[^/@]+)(/(?P<instance>[^/@]+))?@(?P<realm>[^/@]+)$"
        match = re.match(pattern, self.kerberos_service_principal)
        if not match:
            raise MostlyDataException("Invalid Kerberos principal format. Expected format 'primary[/instance]@REALM'")
        components = match.groupdict()
        components["instance"] = components.get("instance")
        return components

    @property
    def kerberos_service_name(self) -> str:
        if self.kerberos_service_principal and isinstance(self.kerberos_service_principal, str):
            return self.kerberos_service_principal.split("/")[0]  # e.g. hive/... -> hive
        return ""

    @contextmanager
    def use_ssh_tunnel(self):
        if not self.enable_ssh or self._is_ssh_active:
            yield
            return

        self._is_ssh_active = True
        db_host = self.host
        db_port = self.port
        with use_ssh_tunnel(
            ssh_host=self.ssh_host,
            ssh_port=self.ssh_port,
            ssh_username=self.ssh_username,
            db_host=db_host,
            db_port=db_port,
            ssh_password=self.ssh_password,
            ssh_private_key_path=self.ssh_private_key_path,
        ) as (ssh_tunnel_host, ssh_tunnel_port):
            # temporarily update the host and port to the tunnel host and port
            self.update_host_and_port(ssh_tunnel_host, ssh_tunnel_port)
            yield
            self.update_host_and_port(db_host, db_port)
        self._is_ssh_active = False

    @contextmanager
    def use_ssl_connection(self):
        if not self.ssl_enabled or self._is_ssl_active:
            yield
            return

        self._is_ssl_active = True
        self.store_ssl_files()
        yield
        self.clear_ssl_files()
        self._is_ssl_active = False

    @contextmanager
    def init_sa_connection(
        self,
        mode: ConnectionMode = "read_data",
        dispose: bool = True,
    ):
        with (
            self.use_sa_engine(mode, dispose) as engine,
            engine.connect() as connection,
        ):
            yield connection

    @property
    @abc.abstractmethod
    def sa_uri(self):
        pass

    @property
    def sa_uri_for_does_database_exist(self):
        return self.sa_uri

    @property
    def sa_uri_for_write(self):
        return self.sa_uri

    @property
    def sa_engine_connection_kwargs(self):
        if self.ssl_enabled:
            for attr, key in self.SA_SSL_ATTR_KEY_MAP.items():
                if hasattr(self, attr):
                    self.SA_CONNECTION_KWARGS[key] = getattr(self, attr)
            return self.SA_CONNECTION_KWARGS
        return {}

    @property
    def sa_create_engine_kwargs(self) -> dict:
        return {}

    @classmethod
    @abc.abstractmethod
    def table_class(cls):
        pass

    def is_accessible(self) -> bool:
        try:
            with self.init_sa_connection("access_check"):
                if not self._is_schema_exist():
                    raise MostlyDataException(f"Schema `{self.dbschema}` does not exist.")
                else:
                    return True
        except Exception as e:
            error_message = str(e).lower()
            _LOG.error(f"Database connection failed with error: {e}")
            if self.host is not None and not self.is_host_reachable(self.host, int(self.port)):
                raise MostlyDataException("Cannot resolve host.")
            elif self.port is not None and not self.is_port_accessible(self.host, int(self.port)):
                raise MostlyDataException(f"Cannot reach port `{self.port}`.")
            elif any(keyword in error_message for keyword in ["ssl", "SSL", "certificate"]):
                raise MostlyDataException("SSL certificates are missing or incorrect.")
            elif "password" in error_message or "user" in error_message:
                raise MostlyDataException("Credentials are incorrect.")
            elif not self.does_database_exist():
                raise MostlyDataException("Database does not exist.")
            else:
                raise

    def _is_schema_exist(self) -> bool:
        if self.dbschema is None:
            return True
        with self.use_sa_engine() as sa_engine:
            schema_names = sa.inspect(sa_engine).get_schema_names()
        return self.dbschema in schema_names

    @abc.abstractmethod
    def does_database_exist(self) -> bool:
        pass

    def query(self, sql: str) -> pd.DataFrame:
        assert_read_only_sql(sql)
        with self.init_sa_connection() as engine:
            try:
                return pd.read_sql(sql, engine, dtype_backend="pyarrow")
            except sa.exc.SQLAlchemyError as e:
                _LOG.error(f"Error executing query: {str(e)}")
                raise

    def drop_all(self):
        if self.sa_metadata:
            with self.use_sa_engine() as sa_engine:
                self.sa_metadata.drop_all(bind=sa_engine)

    def fetch_table(self, table_name: str):
        if table_name and self.table_class():
            return self.table_class()(name=table_name, container=self)

    def get_view_list(self) -> list[str]:
        with self.use_sa_engine() as sa_engine:
            return list(sa.inspect(sa_engine).get_view_names(schema=self.dbschema))

    def get_table_list(self) -> list[str]:
        with self.use_sa_engine() as sa_engine:
            return list(sa.inspect(sa_engine).get_table_names(schema=self.dbschema))

    def get_sa_table(self, table_name: str) -> sa.Table | None:
        if table_name not in self.get_object_list():
            return None
        try:
            with self.use_sa_engine() as sa_engine:
                sa_table = Table(
                    table_name,
                    self.sa_metadata,
                    autoload_with=sa_engine,
                    schema=self.dbschema,
                )
        except sa.exc.NoSuchTableError:
            return None
        return sa_table

    def fetch_schema(self):
        object_names = self.get_object_list()
        if not object_names:
            return
        for object_name in object_names:
            if self.filtered_tables and object_name not in self.filtered_tables:
                continue
            sa_table = self.get_sa_table(table_name=object_name)
            if sa_table is None:
                continue

            table = self.table_class()(
                name=sa_table.name,
                container=self,
                is_view=object_name in self.get_view_list(),
            )

            if not table.primary_key:
                table.primary_key = self.get_primary_key(object_name)

            self.schema.tables[sa_table.name] = table

            foreign_keys = self.get_foreign_keys(sa_table.name)
            table.foreign_keys = [
                ForeignKey(
                    column=fk["COLUMN_NAME"],
                    referenced_table=fk["REFERENCED_TABLE_NAME"],
                    is_context=False,  # to be on the safe side
                )
                for fk in foreign_keys
            ]

    def get_object_list(self):
        return self.get_table_list() + self.get_view_list()

    def get_primary_key(self, table_name: str):
        pass

    @property
    def foreign_keys(self) -> list[dict]:
        if self.SQL_FETCH_FOREIGN_KEYS is not None and self._foreign_keys is None:
            query = sa.text(self.SQL_FETCH_FOREIGN_KEYS)
            params = {"schema_name": self.dbschema}
            with self.init_sa_connection() as conn:
                result = conn.execute(query, params)
                # some SQL dialects may force "keys" to be non-case sensitive, thus ensure they are uppercase
                self._foreign_keys = [{k.upper(): v for k, v in dict(row).items()} for row in result.mappings()]
        elif self._foreign_keys is None:
            self._foreign_keys = []
        return self._foreign_keys

    def _get_foreign_keys_naive(self, table_name: str):
        foreign_keys = []
        sa_table = self.get_sa_table(table_name)
        if sa_table is None:
            return foreign_keys
        for sa_column in sa_table.columns:
            for fk in sa_column.foreign_keys:
                if fk.column.table.primary_key:
                    foreign_keys.append(
                        {
                            "REFERENCED_TABLE_NAME": fk.column.table.name,
                            "REFERENCED_COLUMN_NAME": fk.column.name,
                            "TABLE_NAME": sa_table.name,
                            "COLUMN_NAME": sa_column.name,
                        }
                    )
        return foreign_keys

    @lru_cache
    def get_foreign_keys(self, table_name: str):
        if self.foreign_keys:
            foreign_keys = [fk for fk in self.foreign_keys if fk.get("TABLE_NAME") == table_name]
        else:
            foreign_keys = self._get_foreign_keys_naive(table_name)
        return foreign_keys

    def get_children(self, table_name: str) -> list[str]:
        visited = []
        queue = [table_name]

        while queue:
            current_table = queue.pop(0)
            if current_table not in visited:
                visited.append(current_table)

                # Find all tables that have a foreign key to the current table
                for table_name_in_db in self.get_table_list():
                    for fk in self.get_foreign_keys(table_name_in_db):
                        if fk["REFERENCED_TABLE_NAME"] == current_table:
                            queue.append(table_name_in_db)

        return visited

    def list_locations(self, prefix: str | None) -> list[str]:
        """
        If the prefix is None or "", return the list of locations above the table-level.
          - MYSQL/MARIADB: the locations are the databases
          - The rest: the locations are the schemas
        Otherwise, return the list of tables of the database/schema named f"{prefix}".
        """
        if not prefix:
            # List the schemas
            self.update_dbschema(None)
            return sorted(list(set(self.all_schemas())))
        else:
            prefix = prefix.rstrip(".")  # remove trailing dot if present
            # List the tables of the provided schema named {prefix}
            self.update_dbschema(prefix)
            locations = [".".join([prefix, loc]) for loc in self.get_object_list()]
            return sorted(list(set(locations)))

    def update_dbschema(self, dbschema: str | None) -> None:
        """
        Resetting schema in order to list tables belonging to the new schema
        """
        self.dbschema = dbschema
        self._sa_engine_for_read = None  # reset engine
        self.sa_metadata = sa.MetaData(schema=self.dbschema)

    def update_host_and_port(self, host: str, port: str) -> None:
        self.host = host
        self.port = port
        self._sa_engine_for_read = None  # reset engine

    @functools.cached_property
    def decryption_path(self) -> str:
        return tempfile.mkdtemp()

    def load_ssl_files_and_prepare_ssl_paths(self):
        # loads ssl files, decrypts them and keeps them in memory
        # prepares ssl paths where ssl files will be stored and clear on demand
        for attr in SSL_ATTRIBUTES:
            encrypted_content = getattr(self, attr)  # e.g. value of root_certificate
            if encrypted_content:
                content_attr = f"_{attr}"  # e.g. _root_certificate
                decrypted_content = self.decrypt(encrypted_content)
                setattr(self, content_attr, decrypted_content)  # e.g. sets _root_certificate
                path_attr = f"{attr}_path"  # e.g. root_certificate_path
                decrypted_path = prepare_ssl_path(path_attr, self.decryption_path)
                setattr(self, path_attr, decrypted_path)  # e.g. sets root_certificate_path

    def store_ssl_files(self):
        for attr in SSL_ATTRIBUTES:
            path_attr = f"{attr}_path"  # e.g. root_certificate_path
            if hasattr(self, path_attr):
                path = Path(getattr(self, path_attr))  # e.g. value of root_certificate_path
                content_attr = f"_{attr}"  # e.g. _root_certificate
                content = getattr(self, content_attr)  # e.g. value of _root_certificate
                # store ssl file
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content.encode())
                path.chmod(0o600)

    def clear_ssl_files(self):
        # remove the decryption directory
        shutil.rmtree(self.decryption_path, ignore_errors=True)

    def all_schemas(self) -> list:
        with self.use_sa_engine() as sa_engine:
            schemas = sa.inspect(sa_engine).get_schema_names()
            return schemas

    @staticmethod
    def is_host_reachable(host: str, port: int) -> bool:
        """
        Check if host name is resolvable
        :param host: Hostname or IP address
        :param port: Port number
        :return: True if host is resolvable, False otherwise
        """
        try:
            socket.getaddrinfo(host, port)
            return True
        except socket.gaierror:
            return False

    @staticmethod
    def is_port_accessible(host: str, port: int) -> bool:
        """
        Check if host is reachable on given port number with a timeout
        :param host: Hostname or IP address
        :param port: Port number
        :return: True if host is reachable, False otherwise
        """
        try:
            sock = socket.create_connection((host, port), timeout=3)
            sock.close()
            return True
        except (OSError, TimeoutError):
            return False

    def set_location(self, location: str) -> dict:
        separator = "."
        location = location.rstrip(separator)
        parts = location.split(separator)
        db_schema = parts[0]
        table_name = parts[-1] if len(parts) > 1 else None
        self.update_dbschema(db_schema)
        return {"location": location, "db_schema": db_schema, "table_name": table_name}


@contextmanager
def use_ssh_tunnel(
    ssh_host: str,
    ssh_port: str,
    ssh_username: str,
    db_host: str,
    db_port: str,
    ssh_password: str | None = None,
    ssh_private_key_path: str | None = None,
) -> Generator[tuple[str, str], None, None]:
    auth_kwargs = {"ssh_password": ssh_password} if ssh_password else {"ssh_pkey": ssh_private_key_path}
    _LOG.info("Opening SSH tunnel")
    with sshtunnel.open_tunnel(
        (ssh_host, int(ssh_port)),
        ssh_username=ssh_username,
        remote_bind_address=(db_host, int(db_port)),
        **auth_kwargs,
    ) as ssh_tunnel:
        ssh_tunnel_host = ssh_tunnel.local_bind_host
        ssh_tunnel_port = str(ssh_tunnel.local_bind_port)
        yield ssh_tunnel_host, ssh_tunnel_port
    _LOG.info(f"SSH tunnel [{ssh_tunnel_host=} {ssh_tunnel_port=}] closed")


class SqlAlchemyTable(DBTable, abc.ABC):
    ENABLE_ORDER_AND_LIMIT_ON_SQL: bool = True
    IS_SERVER_SIDE_CURSOR_AVAILABLE: bool = True
    SA_RANDOM: sa.sql.Executable | None = None  # must be overridden
    SA_MAX_VALS_PER_BATCH: int = 10_000
    SA_MAX_VALS_PER_IN_CLAUSE: int | None = None
    SA_CONN_DIALECT_PROPS: dict[str, Any] | None = None
    SA_MULTIPLE_INSERTS = False
    WRITE_CHUNK_SIZE: int = 1_000
    INIT_WRITE_CHUNK: Callable | None = None
    WRITE_CHUNKS_N_JOBS: int = 4
    SA_DIALECT_PARAMSTYLE: str | None = None

    #################### CONSTRUCTORS & MAGIC METHODS ####################

    def __init__(self, *args, **kwargs):
        self.is_view = kwargs.get("is_view", False)
        self.lazy_fetch_primary_key = kwargs.get("lazy_fetch_primary_key", True)
        self.name_in_db = kwargs.get("name_in_db")  # actual db table name, may differ from self.name
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, is_output={self.is_output})"

    #################### ABSTRACT METHODS ####################

    @classmethod
    @abc.abstractmethod
    def container_class(cls) -> type["SqlAlchemyContainer"]:
        pass

    @classmethod
    @abc.abstractmethod
    def dtype_class(cls) -> type["WrappedDType"]:
        pass

    #################### READ QUERY BUILDING & UTILS ####################

    @property
    def db_table_name(self) -> str:
        # use name_in_db (actual db table name) if set, otherwise fall back to self.name
        return self.name_in_db or self.name

    @property
    def _sa_table(self):
        with self.container.use_sa_engine() as sa_engine:
            return sa.Table(
                self.db_table_name,
                self.container.sa_metadata,
                autoload_with=sa_engine,
            )

    def _sa_set_conn_dialect_props(self, conn: sa.engine.Connection):
        skip_msg = "skipping setting connection dialect properties"
        if not self.SA_CONN_DIALECT_PROPS:
            _LOG.debug(f"SA_CONN_DIALECT_PROPS is not set, {skip_msg}")
            return

        if not hasattr(conn, "dialect"):
            _LOG.warning(f"dialect attribute doesn't exist in {conn}, {skip_msg}")
            return

        _LOG.info(f"Setting dialect properties to {self.SA_CONN_DIALECT_PROPS}")
        for k, v in self.SA_CONN_DIALECT_PROPS.items():
            if not hasattr(conn.dialect, k):
                _LOG.warning(f"Attribute {k} doesn't exist in {conn.dialect}")
                continue

            setattr(conn.dialect, k, v)
            _LOG.info(f"Attribute {k} was set to {v}")

    def _sa_select(self, columns: list[str] | None = None) -> sa.sql.Selectable:
        sa_columns = [c for c in self._sa_table.columns if columns is None or c.name in columns]
        return sa.select(*sa_columns).select_from(self._sa_table)

    def _sa_where(self, stmt: sa.sql.Selectable, where: dict[str, Any] | None = None) -> list[sa.sql.Selectable]:
        # we only support where clause on a single column for DBs
        assert not where or len(where) == 1
        if where is None:
            return [stmt]

        where_column, where_values = next(iter(where.items()))
        where_values = list(set(where_values))  # make sure values are unique
        if self.dtypes[where_column].to_virtual() == VirtualInteger():
            where_values_series = pd.Series(where_values)
            special_where_values = [None] if None in where_values else []
            where_values = (
                pd.to_numeric(where_values_series, errors="coerce", dtype_backend="pyarrow").dropna().tolist()
                + special_where_values
            )

        stmts = []
        for batch_values in self._chunkify(where_values, self.SA_MAX_VALS_PER_BATCH):
            # split into multiple queries
            column = self._sa_table.columns[where_column]
            if not self.SA_MAX_VALS_PER_IN_CLAUSE:
                # IN clause will have unlimited elements
                batch_stmt = stmt.where(column.in_(batch_values))
            else:
                # split into multiple IN clauses connected by OR
                # e.g. SELECT c FROM t WHERE c IN (1, 2) OR c IN (3, 4)
                ins = [
                    column.in_(chunk_ids) for chunk_ids in self._chunkify(batch_values, self.SA_MAX_VALS_PER_IN_CLAUSE)
                ]
                batch_stmt = stmt.where(sa.or_(*ins))
            stmts.append(batch_stmt)
        return stmts if stmts else [stmt]

    def _sa_order(
        self,
        stmt: sa.sql.Selectable,
        shuffle: bool | None = False,
        orderby: OrderBy | None = None,
    ) -> sa.sql.Selectable:
        assert not (shuffle and orderby is not None)  # providing both shuffle and orderby is forbidden
        if shuffle:
            # only shuffle is provided
            stmt = stmt.order_by(self.SA_RANDOM)
        elif orderby:
            # only orderby is provided
            if isinstance(orderby, (tuple, str)):
                orderby = [orderby]
            orderby = [(e, "asc") if isinstance(e, str) else e for e in orderby]
            for column, direction in orderby:
                _column = self._sa_table.columns[column]
                _orderby = _column.asc() if direction == "asc" else _column.desc()
                stmt = stmt.order_by(_orderby)
        return stmt

    @staticmethod
    def _sa_limit(stmt: sa.sql.Selectable, n: int | None = None) -> sa.sql.Selectable:
        if n is None:
            return stmt
        return stmt.limit(n)

    def _sa_execute(self, stmts: list[sa.sql.Selectable]) -> pd.DataFrame:
        n_queries = len(stmts)
        i_query = 1

        def sa_execute_one(stmt: sa.sql.Selectable) -> pd.DataFrame:
            nonlocal i_query
            with self.container.init_sa_connection(dispose=False) as conn:
                try:
                    if self.SA_DIALECT_PARAMSTYLE is not None:
                        conn.dialect.paramstyle = self.SA_DIALECT_PARAMSTYLE
                    self._sa_set_conn_dialect_props(conn)
                    df = pd.read_sql_query(stmt, conn, dtype_backend="pyarrow")
                except sa.exc.SQLAlchemyError:
                    _LOG.exception(f"[{i_query}/{n_queries}] query failed:\n{stmt}")
                    raise
                _LOG.info(f"[{i_query}/{n_queries}] batch shape: {df.shape}")
                i_query += 1
                return df

        def safe_concat(dfs: list[pd.DataFrame]) -> pd.DataFrame:
            return pd.concat(dfs, axis=0) if dfs else pd.DataFrame()

        try:
            df = safe_concat([sa_execute_one(stmt) for stmt in stmts])
        finally:
            # dispose connection pool only once
            self.container.get_sa_engine().dispose()
        return df

    @staticmethod
    def _df_order(
        df: pd.DataFrame,
        shuffle: bool | None = False,
        orderby: list[ColumnSort] | ColumnSort | None = None,
    ) -> pd.DataFrame:
        assert not (shuffle and orderby is not None)  # providing both shuffle and orderby is forbidden
        if shuffle:
            # only shuffle is provided
            _LOG.info("in memory shuffle")
            df = df.sample(frac=1)
        elif orderby:
            # only orderby is provided
            _LOG.info("in memory orderby")
            df = order_df_by(df, orderby)
        return df.reset_index(drop=True)

    @staticmethod
    def _df_limit(df: pd.DataFrame, n: int) -> pd.DataFrame:
        _LOG.info("in memory limit")
        return df.head(n)

    def _get_primary_key(self) -> str | None:
        assert not self.is_output
        primary_key_constraint = self._sa_table.primary_key
        primary_key = next(iter(primary_key_constraint.columns), None)
        return primary_key.name if primary_key is not None else None

    def _get_dtypes(self) -> dict[str, WrappedDType]:
        assert not self.is_output
        return {c.name: self.dtype_class()(wrapped=c.type) for c in self._sa_table.columns}

    @staticmethod
    def _chunkify(lst: Any, n_per_chunk: int):
        lst = list(lst)
        for i in range(0, len(lst), n_per_chunk):
            yield lst[i : i + n_per_chunk]

    #################### PUBLIC METHODS ####################

    def _lazy_fetch(self, item: str) -> None:
        if self.is_output:
            # output tables are assumed to be initially missing,
            # thus don't read any metadata for them
            return None

        if item == "columns":
            self.columns = [c.name for c in self._sa_table.columns]
        elif item == "primary_key":
            self.primary_key = self._get_primary_key() if self.lazy_fetch_primary_key else None
        elif item == "dtypes":
            self.dtypes = self._get_dtypes()
        else:
            super()._lazy_fetch(item)
            return

    def is_accessible(self) -> bool:
        return self.container.is_accessible()

    def read_chunks_by_query(
        self,
        where: dict[str, Any],
        columns: list[str] | None = None,
        do_coerce_dtypes: bool = True,
        fetch_chunk_size: int | None = None,
        yield_chunk_size: int | None = None,
    ) -> Iterable[pd.DataFrame]:
        t00 = time.time()
        assert where is not None and len(where) == 1
        if fetch_chunk_size is None:
            fetch_chunk_size = 1_000
        yield_chunk_size = yield_chunk_size if yield_chunk_size is not None else fetch_chunk_size

        total_time = 0
        chunk_idx = 0
        chunks_df = pd.DataFrame()
        where_key, where_values = next(iter(where.items()))
        for chunk_values in self._chunkify(where_values, fetch_chunk_size):
            t0 = time.time()
            chunk_where = {where_key: chunk_values}
            chunk_df = self.read_data(where=chunk_where, columns=columns, do_coerce_dtypes=do_coerce_dtypes)
            # accumulate chunks
            chunks_df = pd.concat([chunks_df, chunk_df], ignore_index=True)
            if len(chunks_df) >= yield_chunk_size:
                # yield data once it reaches the yield_chunk_size
                yield chunks_df
                chunks_df = pd.DataFrame()
            chunk_idx += 1
            total_time += time.time() - t0
            if chunk_idx % 10 == 0:
                _LOG.info(f"processed {chunk_idx} chunks in {total_time:.2f}s")
        if len(chunks_df) > 0 or len(chunks_df.columns) > 0:
            # yield the remaining data
            yield chunks_df
        _LOG.info(f"finished reading {chunk_idx} chunks in {time.time() - t00:.2f}s (strategy=query)")

    def read_chunks_by_scan(
        self,
        where: dict[str, Any] | None = None,
        columns: list[str] | None = None,
        do_coerce_dtypes: bool = True,
        fetch_chunk_size: int | None = None,
        yield_chunk_size: int | None = None,
    ) -> Iterable[pd.DataFrame]:
        t00 = time.time()
        fetch_chunk_size = fetch_chunk_size if fetch_chunk_size is not None else 100_000
        yield_chunk_size = yield_chunk_size if yield_chunk_size is not None else fetch_chunk_size
        stmt = self._sa_select(columns)
        chunk_idx = 0
        total_time = 0
        # use context managers to ensure proper cleanup of engine and session
        with (
            self.container.use_sa_engine() as sa_engine,
            sessionmaker(bind=sa_engine)() as session,
        ):
            results_generator = None
            try:
                conn = session.connection()
                results_generator = pd.read_sql_query(stmt, conn, dtype_backend="pyarrow", chunksize=fetch_chunk_size)
                chunks_df = pd.DataFrame()
                for chunk_df in results_generator:
                    t0 = time.time()
                    if where is not None:
                        where_column, where_values = next(iter(where.items()))
                        chunk_df = chunk_df[chunk_df[where_column].isin(where_values)]
                    if do_coerce_dtypes:
                        chunk_df = coerce_dtypes_by_encoding(chunk_df, self.encoding_types)
                    # accumulate chunks
                    chunks_df = pd.concat([chunks_df, chunk_df], ignore_index=True)
                    if len(chunks_df) >= yield_chunk_size:
                        # yield data once it reaches the yield_chunk_size
                        yield chunks_df
                        chunks_df = pd.DataFrame()
                    chunk_idx += 1
                    total_time += time.time() - t0
                    if chunk_idx % 10 == 0:
                        _LOG.info(f"processed {chunk_idx} chunks in {total_time:.2f}s")
                if len(chunks_df) > 0 or len(chunks_df.columns) > 0:
                    # yield the remaining data
                    yield chunks_df
                _LOG.info(f"finished reading {chunk_idx} chunks in {time.time() - t00:.2f}s (strategy=scan)")
            finally:
                # explicitly close the results generator to prevent connection leaks
                if results_generator is not None:
                    results_generator.close()

    def is_column_indexed(self, column: str) -> bool:
        with self.container.use_sa_engine() as sa_engine:
            indexes = sa.inspect(sa_engine).get_indexes(table_name=self.db_table_name, schema=self.container.dbschema)
        for index in indexes:
            if column in index.get("column_names", []):
                return True
        return False

    def read_chunks(
        self,
        where: dict[str, Any] | None = None,
        columns: list[str] | None = None,
        do_coerce_dtypes: bool = True,
        fetch_chunk_size: int | None = None,
        yield_chunk_size: int | None = None,
    ) -> Iterable[pd.DataFrame]:
        # determine read strategy based on whether the where column is indexed
        has_index = False
        if where is not None:
            # we only support where clause on a single column for DBs
            assert len(where) == 1
            key_column, _ = next(iter(where.items()))
            has_index = self.is_column_indexed(key_column)
        strategy = "query" if has_index else "scan"

        # read chunks based on strategy
        if strategy == "query":
            yield from self.read_chunks_by_query(
                where=where,
                columns=columns,
                do_coerce_dtypes=do_coerce_dtypes,
                fetch_chunk_size=fetch_chunk_size,
                yield_chunk_size=yield_chunk_size,
            )
        elif strategy == "scan" and self.IS_SERVER_SIDE_CURSOR_AVAILABLE:
            yield from self.read_chunks_by_scan(
                where=where,
                columns=columns,
                do_coerce_dtypes=do_coerce_dtypes,
                fetch_chunk_size=fetch_chunk_size,
                yield_chunk_size=yield_chunk_size,
            )
        else:
            # reads everything at once; potentially memory unsafe
            yield self.read_data(
                where=where,
                columns=columns,
                do_coerce_dtypes=do_coerce_dtypes,
            )

    def read_data(
        self,
        where: dict[str, Any] | None = None,
        limit: int | None = None,
        columns: list[str] | None = None,
        shuffle: bool | None = None,
        order_by: OrderBy | None = None,
        do_coerce_dtypes: bool | None = False,
    ) -> pd.DataFrame:
        t0 = time.time()
        _LOG.info(f"read from table `{self.db_table_name}` started")
        columns = columns if columns is not None else self.columns
        stmt = self._sa_select(columns)
        stmts = self._sa_where(stmt, where)
        if len(stmts) == 1 and self.ENABLE_ORDER_AND_LIMIT_ON_SQL:
            # single statement; include ordering and limiting in the SQL statement
            # and execute it
            stmt = self._sa_order(stmts[0], shuffle, order_by)
            stmt = self._sa_limit(stmt, limit)
            df = self._sa_execute([stmt])
        else:  # len(stmts) > 1 or self.ENABLE_ORDER_AND_LIMIT_ON_SQL is False
            # multiple statements; execute them, concatenate results and apply
            # ordering and limiting on the final result (in memory)
            df = self._sa_execute(stmts)
            df = self._df_order(df, shuffle, order_by)
            df = self._df_limit(df, limit)
        if do_coerce_dtypes:
            df = coerce_dtypes_by_encoding(df, self.encoding_types)
        _LOG.info(f"read DB data `{self.db_table_name}` {df.shape} in {time.time() - t0:.2f}s")
        return df

    def create_table(self, df: pd.DataFrame | None = None, **kwargs) -> None:
        _LOG.info("create DB table schema")

        if df is None:
            # in case not provided, create an empty pd.DataFrame based on self.columns
            df = pd.DataFrame(columns=self.columns)

        with self.container.use_sa_engine(mode="write_data") as sa_engine:
            try:
                df.head(0).to_sql(
                    name=self.db_table_name,
                    con=sa_engine,
                    schema=self.container.dbschema,
                    index=False,
                    **kwargs,
                )
            except ValueError:
                if kwargs.get("if_exists") == "fail":
                    raise MostlyDataException("Destination location already exists.")
                raise

        dtypes_msg = f"with dtypes=`{kwargs['dtype']}`" if "dtype" in kwargs else ""
        _LOG.info(f"Successfully created table `{self.db_table_name}` schema {dtypes_msg}")

    def write_chunks(self, chunks: Iterable[pd.DataFrame], dtypes: dict[str, Any], **kwargs) -> None:
        _LOG.info(f"write data in {len(chunks)} chunks (n_jobs={self.WRITE_CHUNKS_N_JOBS})")
        with self.container.use_sa_engine(mode="write_data") as sa_engine:
            Parallel(n_jobs=self.WRITE_CHUNKS_N_JOBS)(
                delayed(_write_chunk)(
                    chunk=chunk,
                    sa_engine_uri=sa_engine.url,
                    sa_create_engine_kwargs=self.container.sa_create_engine_kwargs,
                    sa_create_engine_connect_kwargs=self.container.sa_engine_connection_kwargs,
                    sa_multiple_inserts=self.SA_MULTIPLE_INSERTS,
                    table_name=self.db_table_name,
                    table_schema=self.container.dbschema,
                    table_dtypes=dtypes,
                    chunk_init=self.INIT_WRITE_CHUNK,
                    **kwargs,
                )
                for chunk in chunks
            )

    def calculate_write_chunk_size(self, df: pd.DataFrame) -> int:
        return self.WRITE_CHUNK_SIZE

    def write_data(
        self,
        df: pd.DataFrame,
        if_exists: Literal["append", "replace", "fail"] = "append",
        **kwargs,
    ) -> None:
        t0 = time.time()
        assert self.is_output
        _LOG.info(f"write data {df.shape} to table `{self.db_table_name}` started")

        df = _normalize_for_sql(df)

        # map mostly-data dtypes to SQLAlchemy dtypes
        dtypes = self.dtype_class().from_dtypes_unwrap(self.dtypes)

        # create table before parallel writes
        self.create_table(df, dtype=dtypes, if_exists=if_exists)

        # calculate write chunk size and ensure it is > 0
        write_chunk_size = max(self.calculate_write_chunk_size(df), 1)

        # chunkify and write data in chunks
        chunks = [df[i : i + write_chunk_size] for i in range(0, len(df), write_chunk_size)]
        self.write_chunks(chunks, dtypes)

        _LOG.info(f"write to table `{self.db_table_name}` finished in {time.time() - t0:.2f}s")

    def has_child(self) -> bool:
        return len(self.container.schema.get_relations_from_table(str(self.name))) > 0

    def drop(self, drop_all: bool = False):
        with self.container.use_sa_engine() as sa_engine:
            self._sa_table.drop(sa_engine)

    @functools.cached_property
    def row_count(self) -> int:
        stmt = sa.select(sa.func.count()).select_from(self._sa_table)
        return self._sa_execute([stmt]).loc[0, "count_1"]


def _normalize_for_sql(df: pd.DataFrame) -> pd.DataFrame:
    # ensure pd.DataFrame.to_sql(...) works alongside pyarrow backend
    apply_map = {
        np.ndarray: lambda x: x.tolist() if isinstance(x, np.ndarray) else x,
        enum.Enum: lambda x: x.value if isinstance(x, enum.Enum) else x,
    }
    astype_map = {
        decimal.Decimal: FLOAT64,
        uuid.UUID: STRING,
    }

    for col in df.columns:
        series = df[col]
        if series.empty or series.dropna().empty:
            continue

        sample = series.dropna().iloc[0]

        for typ, astype_target in astype_map.items():
            if isinstance(sample, typ):
                df[col] = series.astype(astype_target)
                break
        else:
            for typ, apply_func in apply_map.items():
                if isinstance(sample, typ):
                    df[col] = series.apply(apply_func)
                    break

    return df


def _write_chunk(
    chunk: pd.DataFrame,
    sa_engine_uri: str,
    sa_create_engine_kwargs: dict[str, Any],
    sa_create_engine_connect_kwargs: dict[str, Any],
    sa_multiple_inserts: bool,
    table_name: str,
    table_schema: str,
    table_dtypes: dict[str, Any],
    chunk_init: Callable | None,
) -> None:
    # this function is parallelized and thus must be self-contained
    # sqlalchemy engine cannot be pickled, so we recreate it here
    if chunk_init:
        chunk_init()
    sa_engine = sa.create_engine(
        url=sa_engine_uri,
        connect_args=sa_create_engine_connect_kwargs,
        **sa_create_engine_kwargs,
    )
    method = "multi" if sa_multiple_inserts else None

    try:
        chunk.to_sql(
            table_name,
            sa_engine,
            schema=table_schema,
            dtype=table_dtypes,
            # we assume that the table has already been created before
            # parallelized writes kicked in
            method=method,
            if_exists="append",
            index=False,
        )
    finally:
        # ensure connections get closed
        sa_engine.dispose()
