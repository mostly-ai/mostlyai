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
import warnings
from typing import TYPE_CHECKING
from urllib.parse import quote

import redshift_connector
import sqlalchemy as sa
from sqlalchemy.dialects import registry
from sqlalchemy.dialects.postgresql.base import PGDialect

from mostlyai.sdk._data.db.base import DBDType
from mostlyai.sdk._data.db.postgresql import PostgresqlContainer, PostgresqlTable

# suppress ssl deprecation warnings from redshift_connector
warnings.filterwarnings(
    "ignore", message="ssl.SSLContext.*without protocol argument is deprecated", category=DeprecationWarning
)
warnings.filterwarnings("ignore", message="ssl.PROTOCOL_TLS is deprecated", category=DeprecationWarning)

if TYPE_CHECKING:
    import pandas as pd

_LOG = logging.getLogger(__name__)


# minimal DBAPI wrapper for redshift_connector
class RedshiftDBAPI:
    paramstyle = "format"  # redshift_connector uses format style (%s)
    apilevel = "2.0"
    threadsafety = 1

    @staticmethod
    def connect(*args, **kwargs):
        # map sqlalchemy parameter names to redshift-connector parameter names
        param_mapping = {"username": "user", "dbname": "database"}
        redshift_kwargs = {param_mapping.get(key, key): value for key, value in kwargs.items()}
        return redshift_connector.connect(**redshift_kwargs)

    Error = redshift_connector.Error
    InterfaceError = redshift_connector.InterfaceError
    DatabaseError = redshift_connector.DatabaseError
    DataError = redshift_connector.DataError
    OperationalError = redshift_connector.OperationalError
    IntegrityError = redshift_connector.IntegrityError
    InternalError = redshift_connector.InternalError
    ProgrammingError = redshift_connector.ProgrammingError
    NotSupportedError = redshift_connector.NotSupportedError


# minimal redshift dialect that avoids postgresql-specific introspection
class RedshiftDialect(PGDialect):
    name = "redshift"
    driver = "redshift-connector"
    supports_statement_cache = False
    default_paramstyle = "format"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.on_connect = lambda: None

    @classmethod
    def import_dbapi(cls):
        return RedshiftDBAPI

    def initialize(self, connection):
        # skip postgresql-specific initialization that doesn't work with redshift
        pass

    @property
    def server_version_info(self):
        # return a stable version tuple to avoid version comparison issues
        return (8, 4, 0)

    def _get_relnames_for_relkinds(self, connection, schema, relkinds, scope):
        # use information_schema for redshift compatibility
        if not relkinds:
            return []

        from sqlalchemy import text

        # map relkinds to table types: "r"=tables, "v"=views, "m"=materialized views
        type_map = {"r": "BASE TABLE", "v": "VIEW", "m": "MATERIALIZED VIEW"}
        conditions = [f"table_type = '{type_map[k]}'" for k in relkinds if k in type_map]

        if not conditions:
            return []

        result = connection.execute(
            text(f"""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = :schema AND ({" OR ".join(conditions)})
            ORDER BY table_name
        """),
            {"schema": schema or "public"},
        )
        return [row[0] for row in result.fetchall()]

    def get_domains(self, connection, schema=None, **kw):
        # redshift doesn't support domains, return empty list to avoid pg_collation queries
        return []

    def _load_domains(self, connection, schema=None, **kw):
        # override to prevent domain loading that causes pg_collation errors
        return []

    def _get_domain_query(self, schema):
        # override to prevent domain queries that cause pg_collation errors
        from sqlalchemy import text

        return text("SELECT 1 WHERE 1=0")  # return empty result

    def _load_enums(self, connection, schema=None, **kw):
        # override to prevent enum loading that causes syntax errors in redshift
        return []

    def _enum_query(self, schema):
        # override to prevent enum queries that cause syntax errors in redshift
        from sqlalchemy import text

        return text("SELECT 1 WHERE 1=0")  # return empty result

    def get_multi_pk_constraint(self, connection, schema=None, **kw):
        # override to prevent primary key constraint queries that cause syntax errors in redshift
        return {}

    def _reflect_constraint(self, connection, constraint_name, table_name, schema, **kw):
        # override to prevent constraint reflection that causes syntax errors in redshift
        return []

    def get_multi_unique_constraints(self, connection, schema=None, **kw):
        # override to prevent unique constraint queries that cause syntax errors in redshift
        return {}

    def get_multi_indexes(self, connection, schema=None, **kw):
        # override to prevent index queries that cause syntax errors in redshift
        return {}


def _ensure_dialect_registered():
    """ensure the redshift dialect is registered with sqlalchemy"""
    try:
        # try to load the dialect to see if it's already registered
        registry.load("redshift.redshift_connector")
    except Exception:
        # dialect not registered, register it now
        registry.register("redshift.redshift_connector", __name__, "RedshiftDialect")


# register the dialect immediately when module is imported
_ensure_dialect_registered()


def configure_redshift_dialect():
    """configure redshift dialect for parallel write processes"""
    # ensure the dialect is registered in the parallel process
    try:
        # try to load the dialect to see if it's already registered
        registry.load("redshift.redshift_connector")
    except Exception:
        # dialect not registered, register it now
        registry.register("redshift.redshift_connector", "mostlyai.sdk._data.db.redshift", "RedshiftDialect")


class RedshiftDType(DBDType):
    FROM_VIRTUAL_DATETIME = sa.TIMESTAMP

    @classmethod
    def sa_dialect_class(cls):
        return RedshiftDialect


class RedshiftContainer(PostgresqlContainer):
    SCHEMES = ["redshift"]
    # redshift-connector enables ssl by default; don't pass postgres-specific ssl args
    SA_CONNECTION_KWARGS = {}
    SA_SSL_ATTR_KEY_MAP = {}
    # redshift-connector doesn't support connect_timeout parameter
    SA_CONNECT_ARGS_ACCESS_ENGINE = {}
    INIT_DEFAULT_VALUES = {"dbname": "", "port": "5439"}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def sa_uri(self):
        # user and password are needed to avoid double-encoding of @ character
        username = quote(self.username)
        password = quote(self.password)
        # use redshift+redshift_connector dialect
        return f"redshift+redshift_connector://{username}:{password}@{self.host}:{self.port}/{self.dbname}"

    @property
    def sa_create_engine_kwargs(self) -> dict:
        # redshift-connector doesn't support executemany_* parameters
        # use only standard sqlalchemy engine parameters
        return {
            # optimize connection pooling for redshift
            "pool_size": 5,
            "max_overflow": 10,
            "pool_timeout": 30,
            "pool_recycle": 3600,
        }

    @classmethod
    def table_class(cls):
        return RedshiftTable


class RedshiftTable(PostgresqlTable):
    DATA_TABLE_TYPE = "redshift"
    SA_RANDOM = sa.func.rand()

    # performance optimizations for redshift
    # larger chunks for better redshift performance (redshift prefers fewer, larger operations)
    WRITE_CHUNK_SIZE = 10_000
    # enable multiple inserts for better performance
    SA_MULTIPLE_INSERTS = True
    # reduce parallel jobs to avoid overwhelming redshift (redshift has limited concurrent connections)
    WRITE_CHUNKS_N_JOBS = 2

    def write_data(self, df: "pd.DataFrame", **kwargs):
        # set the chunk initialization function for parallel writes
        self.INIT_WRITE_CHUNK = configure_redshift_dialect
        super().write_data(df, **kwargs)

    def calculate_write_chunk_size(self, df: "pd.DataFrame") -> int:
        # optimize chunk size based on dataframe size for redshift
        if len(df) < 1000:
            return len(df)  # write all at once for small datasets
        elif len(df) < 100_000:
            return 5_000  # medium chunks for medium datasets
        else:
            return self.WRITE_CHUNK_SIZE  # use default for large datasets

    @property
    def _sa_table(self):
        # quote table name to handle special characters like dots
        with self.container.use_sa_engine() as sa_engine:
            quoted_name = sa.quoted_name(self.name, quote=True)
            return sa.Table(
                quoted_name, self.container.sa_metadata, schema=self.container.dbschema, autoload_with=sa_engine
            )
