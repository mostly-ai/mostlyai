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
from sqlalchemy import text, types
from sqlalchemy.dialects import registry
from sqlalchemy.engine.default import DefaultDialect

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


# minimal redshift dialect that inherits from DefaultDialect instead of PGDialect
class RedshiftDialect(DefaultDialect):
    name = "redshift"
    driver = "redshift_connector"
    supports_statement_cache = False
    default_paramstyle = "format"

    # redshift-specific capabilities
    supports_alter = True
    supports_unicode_statements = True
    supports_unicode_binds = True
    supports_native_boolean = True
    supports_native_decimal = True
    supports_schemas = True
    supports_sequences = False  # redshift doesn't support sequences
    supports_identity_columns = False  # redshift doesn't support identity columns
    supports_comments = True
    supports_default_values = True
    supports_empty_inserts = False
    supports_multivalues_insert = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.on_connect = lambda: None
        # set a stable version tuple to avoid version comparison issues
        self.server_version_info = (8, 4, 0)

    @classmethod
    def import_dbapi(cls):
        return RedshiftDBAPI

    def _execute_information_schema_query(self, connection, query, params):
        """helper method to execute information_schema queries with common error handling"""
        return connection.execute(text(query), params)

    def _get_schema_or_default(self, schema):
        """helper method to get schema name with default fallback"""
        return schema or "public"

    def _query_table_names_by_type(self, connection, schema, table_type):
        """helper method to query table names by type (BASE TABLE or VIEW)"""
        schema = self._get_schema_or_default(schema)

        query = """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = :schema AND table_type = :table_type
            ORDER BY table_name
        """

        result = self._execute_information_schema_query(connection, query, {"schema": schema, "table_type": table_type})
        return [row[0] for row in result.fetchall()]

    def get_table_names(self, connection, schema=None, **kw):
        """get table names using information_schema for redshift compatibility"""
        return self._query_table_names_by_type(connection, schema, "BASE TABLE")

    def get_view_names(self, connection, schema=None, **kw):
        """get view names using information_schema for redshift compatibility"""
        return self._query_table_names_by_type(connection, schema, "VIEW")

    def get_columns(self, connection, table_name, schema=None, **kw):
        """get column information using information_schema for redshift compatibility"""
        schema = self._get_schema_or_default(schema)

        query = """
            SELECT
                column_name,
                data_type,
                is_nullable,
                column_default,
                character_maximum_length,
                numeric_precision,
                numeric_scale
            FROM information_schema.columns
            WHERE table_schema = :schema AND table_name = :table_name
            ORDER BY ordinal_position
        """

        result = self._execute_information_schema_query(connection, query, {"schema": schema, "table_name": table_name})

        columns = []
        for row in result.fetchall():
            columns.append(
                {
                    "name": row[0],
                    "type": self._get_column_type(row[1], row[4], row[5], row[6]),
                    "nullable": row[2] == "YES",
                    "default": row[3],
                }
            )
        return columns

    def _get_column_type(self, data_type, max_length, precision, scale):
        """map redshift data types to sqlalchemy types"""
        type_map = {
            "character varying": lambda: types.VARCHAR(max_length) if max_length else types.VARCHAR(),
            "character": lambda: types.CHAR(max_length) if max_length else types.CHAR(),
            "varchar": lambda: types.VARCHAR(max_length) if max_length else types.VARCHAR(),
            "char": lambda: types.CHAR(max_length) if max_length else types.CHAR(),
            "text": lambda: types.TEXT(),
            "integer": lambda: types.INTEGER(),
            "bigint": lambda: types.BIGINT(),
            "smallint": lambda: types.SMALLINT(),
            "decimal": lambda: types.DECIMAL(precision, scale) if precision else types.DECIMAL(),
            "numeric": lambda: types.NUMERIC(precision, scale) if precision else types.NUMERIC(),
            "real": lambda: types.REAL(),
            "double precision": lambda: types.FLOAT(),
            "boolean": lambda: types.BOOLEAN(),
            "date": lambda: types.DATE(),
            "timestamp": lambda: types.TIMESTAMP(),
            "timestamp without time zone": lambda: types.TIMESTAMP(),
            "timestamp with time zone": lambda: types.TIMESTAMP(timezone=True),
            "time": lambda: types.TIME(),
            "time without time zone": lambda: types.TIME(),
            "time with time zone": lambda: types.TIME(timezone=True),
        }

        return type_map.get(data_type.lower(), lambda: types.NULLTYPE)()

    def get_pk_constraint(self, connection, table_name, schema=None, **kw):
        """get primary key constraint information (non-enforced in redshift)"""
        schema = self._get_schema_or_default(schema)

        query = """
            SELECT
                t.constraint_name,
                c.column_name
            FROM
                information_schema.table_constraints t
            JOIN
                information_schema.key_column_usage c
                  ON t.constraint_name = c.constraint_name
                 AND t.table_schema   = c.table_schema
                 AND t.table_name     = c.table_name
            WHERE
                t.table_schema = :schema
                AND t.table_name = :table_name
                AND t.constraint_type = 'PRIMARY KEY'
            ORDER BY
                c.ordinal_position
        """

        result = self._execute_information_schema_query(connection, query, {"schema": schema, "table_name": table_name})

        columns = []
        constraint_name = None
        for row in result.fetchall():
            if constraint_name is None:
                constraint_name = row[0]
            columns.append(row[1])

        return {"constrained_columns": columns, "name": constraint_name}

    def get_foreign_keys(self, connection, table_name, schema=None, **kw):
        """get foreign key constraint information (non-enforced in redshift)"""
        schema = self._get_schema_or_default(schema)

        query = """
            SELECT
                tc.constraint_name,
                kcu.column_name,
                ccu.table_schema AS foreign_table_schema,
                ccu.table_name AS foreign_table_name,
                ccu.column_name AS foreign_column_name
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
                AND tc.table_schema = :schema
                AND tc.table_name = :table_name
            ORDER BY
                tc.constraint_name,
                kcu.ordinal_position
        """

        result = self._execute_information_schema_query(connection, query, {"schema": schema, "table_name": table_name})

        # group by constraint name
        fkeys = {}
        for row in result:
            const_name, col_name, ref_schema, ref_table, ref_col = row
            if const_name not in fkeys:
                fkeys[const_name] = {
                    "name": const_name,
                    "constrained_columns": [],
                    "referred_schema": ref_schema,
                    "referred_table": ref_table,
                    "referred_columns": [],
                }
            fkeys[const_name]["constrained_columns"].append(col_name)
            fkeys[const_name]["referred_columns"].append(ref_col)

        return list(fkeys.values())

    def get_indexes(self, connection, table_name, schema=None, **kw):
        """get index information"""
        # redshift doesn't support traditional indexes, return empty
        return []

    def get_unique_constraints(self, connection, table_name, schema=None, **kw):
        """get unique constraint information"""
        # redshift doesn't enforce unique constraints, return empty
        return []

    def has_table(self, connection, table_name, schema=None, **kw):
        """check if table exists"""
        schema = self._get_schema_or_default(schema)

        query = """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema = :schema AND table_name = :table_name
        """

        result = self._execute_information_schema_query(connection, query, {"schema": schema, "table_name": table_name})
        return result.scalar() > 0


def _ensure_dialect_registered():
    """ensure the redshift dialect is registered with sqlalchemy"""
    registry.register("redshift.redshift_connector", __name__, "RedshiftDialect")


# register the dialect immediately when module is imported
_ensure_dialect_registered()


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
        self.INIT_WRITE_CHUNK = _ensure_dialect_registered
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
