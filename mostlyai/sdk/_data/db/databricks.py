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
import re
import tempfile
import time
from pathlib import Path
from typing import Literal

import pandas as pd
import sqlalchemy as sa
from azure.identity import ClientSecretCredential
from databricks.sqlalchemy import DatabricksDialect
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from mostlyai.sdk._data.db.base import DBDType, SqlAlchemyContainer, SqlAlchemyTable
from mostlyai.sdk._data.exceptions import MostlyDataException
from mostlyai.sdk._data.util.common import calculate_rows_per_chunk_for_df

AZURE_DATABRICKS_SERVICE = "2ff814a6-3304-4ab8-85cb-cd0e6f879c1d"

_LOG = logging.getLogger(__name__)

HOSTNAME_PREFIX_TO_REMOVE = "https://"
HTTP_PATH_PREFIX_TO_REMOVE = "https://"
USER_AGENT_ENTRY = "mostlyai_SD_platform"  # This tag is required for Databricks to identify the connections from us


class DatabricksDType(DBDType):
    @classmethod
    def sa_dialect_class(cls):
        return DatabricksDialect


class DatabricksContainer(SqlAlchemyContainer):
    SCHEMES = ["databricks"]
    INIT_DEFAULT_VALUES = {"dbname": ""}

    def __init__(
        self,
        *args,
        http_path,
        client_id=None,
        client_secret=None,
        tenant_id=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.http_path = http_path
        self.client_id = client_id
        self.client_secret = client_secret
        self.tenant_id = tenant_id
        if not client_secret:
            # Normal auth
            self.decrypt_secret()
        else:
            # Service Principal auth
            self.decrypt_secret("client_secret")

    def post_init_hook(self):
        # skip secret decryption here
        if self.ssl_enabled:
            self.load_ssl_files_and_prepare_ssl_paths()

    @property
    def sa_uri(self):
        self.dbschema = self.dbschema.lower() if isinstance(self.dbschema, str) else "default"
        if re.search("[^A-Za-z0-9_]", self.dbschema):
            raise MostlyDataException(
                f"Schema `{self.dbschema}` contains invalid characters. "
                f"Only letters, digits, and underscores are allowed.",
            )
        self.dbname = self.dbname.lower()
        if re.search("[^A-Za-z0-9_]", self.dbname):
            raise MostlyDataException(
                f"Catalog `{self.dbname}` contains invalid characters. "
                f"Only letters, digits, and underscores are allowed.",
            )
        self.host = self.host.replace(HOSTNAME_PREFIX_TO_REMOVE, "")
        self.http_path = self.http_path.replace(HTTP_PATH_PREFIX_TO_REMOVE, "").replace(self.host, "")
        if self.client_secret:  # using service principal
            if not all([self.client_id, self.client_secret, self.tenant_id]):
                raise MostlyDataException(
                    "Provide a token or service principal credentials (client_id, client_secret, tenant_id).",
                )
            credential = ClientSecretCredential(self.tenant_id, self.client_id, self.client_secret)
            access_token = credential.get_token(f"{AZURE_DATABRICKS_SERVICE}/.default").token
        else:
            access_token = f"{self.password}"
        return (
            f"databricks://token:{access_token}@{self.host}?"
            f"http_path={self.http_path}"
            f"&catalog={self.dbname}"
            f"&schema={self.dbschema}"
        )

    @property
    def sa_create_engine_kwargs(self) -> dict:
        return {"future": True, "echo": False}

    @property
    def sa_engine_connection_kwargs(self):
        return {
            "_user_agent_entry": USER_AGENT_ENTRY,
            # only retry the connection three times with an interval of 10 seconds
            # so that we don't wait too long when the cluster is still starting and temporarily unavailable
            "_retry_delay_max": 10.0,
            "_retry_stop_after_attempts_count": 3,
            "use_inline_params": "silent",
        }

    @classmethod
    def table_class(cls):
        return DatabricksTable

    def _check_schema_name(self, result: sa.engine.cursor.CursorResult) -> bool:
        return any(str(row[0]) == self.dbschema for row in result)

    def does_database_exist(self) -> bool:
        pass

    def is_accessible(self) -> bool:
        try:
            with self.init_sa_connection("access_check") as connection:
                result = connection.execute(sa.text("SHOW DATABASES"))
                if not self._check_schema_name(result):
                    raise MostlyDataException(f"Schema `{self.dbschema}` does not exist.")
            return True
        except Exception as e:
            error_message = str(e).lower()
            _LOG.error(f"Database connection failed with error: {e}")
            if any(
                keyword in error_message
                for keyword in [
                    "invalid access token",
                    "credential",
                    "request to server",
                    "401 error",
                ]
            ):
                raise MostlyDataException("Access token is incorrect.")
            elif f"catalog '{self.dbname}' not found" in error_message:
                raise MostlyDataException(f"Catalog `{self.dbname}` does not exist.")
            elif "temporarily unavailable" in error_message:
                raise MostlyDataException(
                    "Server is temporarily unavailable. Check if the cluster or SQL warehouse is running.",
                )
            elif any(
                keyword in error_message
                for keyword in [
                    "nodename",
                    "servname",
                    "max retries exceeded",
                ]
            ):
                raise MostlyDataException("Cannot resolve host or HTTP path.")
            elif "catalog" in error_message and "not found" in error_message:
                raise MostlyDataException("Catalog is not accessible.")
            elif "check your tenant" in error_message:
                raise MostlyDataException("Tenant ID is incorrect.")
            elif "application with identifier" in error_message:
                raise MostlyDataException("Client ID is incorrect.")
            elif "invalid client secret" in error_message:
                raise MostlyDataException("Client secret is incorrect.")
            else:
                raise MostlyDataException(f"Connectivity check failed: {str(e)}")

    def _valid_table_name(self, table_name: str) -> bool:
        """Validate table name."""
        return table_name in self.get_object_list()

    def _fetch_foreign_keys(self, table_name: str) -> list[dict]:
        """
        Fetch foreign keys for a table
        """
        if not self._valid_table_name(table_name):
            raise MostlyDataException(f"Table `{table_name}` does not exist.")
        query = text(
            """
        SELECT
            kcu.table_name as TABLE_NAME,
            kcu.column_name as COLUMN_NAME,
            rcu.table_name as REFERENCED_TABLE_NAME,
            rcu.column_name as REFERENCED_COLUMN_NAME
        FROM information_schema.table_constraints as tc
        JOIN information_schema.key_column_usage as kcu
            ON tc.constraint_name = kcu.constraint_name
        JOIN information_schema.referential_constraints as rc
            ON rc.constraint_name = tc.constraint_name
        JOIN information_schema.constraint_column_usage as rcu
            ON rcu.constraint_name = rc.constraint_name
        WHERE tc.table_name = :table_name
            AND tc.constraint_type = 'FOREIGN KEY'
        """
        )
        try:
            with self.init_sa_connection() as connection:
                result = connection.execute(query, {"table_name": table_name})
            return [dict(row) for row in result]
        except Exception:
            _LOG.debug(f"Error while executing {query=}")
            return []

    def _fetch_primary_key(self, table_name: str) -> list[str] | None:
        """
        Fetch primary key of a table from Databricks via SQLAlchemy
        """
        if not self._valid_table_name(table_name):
            raise MostlyDataException(f"Table `{table_name}` does not exist.")
        query = text(
            """
            SELECT ccu.CONSTRAINT_NAME, ccu.TABLE_NAME, ccu.COLUMN_NAME
            FROM information_schema.constraint_column_usage AS ccu
            INNER JOIN information_schema.table_constraints AS tc
            ON ccu.CONSTRAINT_CATALOG = tc.CONSTRAINT_CATALOG
            AND ccu.CONSTRAINT_SCHEMA = tc.CONSTRAINT_SCHEMA
            AND ccu.CONSTRAINT_NAME = tc.CONSTRAINT_NAME
            WHERE ccu.TABLE_NAME = :table_name
            AND tc.CONSTRAINT_TYPE = 'PRIMARY KEY'
        """
        )
        try:
            with self.init_sa_connection() as connection:
                result = connection.execute(query, {"table_name": table_name})
            return next((row["COLUMN_NAME"] for row in result), None)
        except SQLAlchemyError:
            _LOG.debug(f"Error while executing {query=}")
            return None

    def get_primary_key(self, table_name: str):
        return self._fetch_primary_key(table_name)

    def get_foreign_keys(self, table_name: str):
        return self._fetch_foreign_keys(table_name)

    def all_schemas(self) -> list:
        # by the time this code was written, sql alchemy did not proper read
        # schemas from databricks using inspect, the workaround is to use raw sql
        with self.init_sa_connection() as connection:
            total = connection.execute(sa.text("SHOW SCHEMAS")).fetchall()
            return [x[0] for x in total]


class DatabricksTable(SqlAlchemyTable):
    DATA_TABLE_TYPE: str = "databricks"
    SA_RANDOM = sa.func.random()
    SA_MULTIPLE_INSERTS = True
    SA_DIALECT_PARAMSTYLE = "pyformat"

    @classmethod
    def dtype_class(cls) -> DatabricksDType:
        return DatabricksDType

    @classmethod
    def container_class(cls) -> DatabricksContainer:
        # The payload size limit when sending data to the Databricks REST API is 16 MB per request
        # Because of this, we chunk the data into smaller pieces and send them in parallel
        return DatabricksContainer

    def calculate_write_chunk_size(self, df: pd.DataFrame) -> int:
        return calculate_rows_per_chunk_for_df(df)

    def _execute(self, query: str) -> None:
        with self.container.init_sa_connection() as connection:
            cursor = connection.connection.cursor()
            cursor.execute(query)

    def create_table(self, df: pd.DataFrame | None = None, **kwargs) -> None:
        if_exists = kwargs.get("if_exists", "append")
        with self.container.init_sa_connection() as connection:
            cursor = connection.connection.cursor()
            cursor.columns(schema_name=self.container.dbschema, table_name=self.db_table_name)
            result = cursor.fetchall()
            if result:  # the table already exists
                if if_exists == "replace":
                    self.drop_table_if_exists()
                    kwargs["if_exists"] = "append"
                elif if_exists == "fail":
                    raise MostlyDataException("Destination location already exists.")
            # create an empty table without schema
            query = f"CREATE TABLE IF NOT EXISTS {self.container.dbname}.{self.container.dbschema}.{self.db_table_name}"
            cursor.execute(query)
            _LOG.info(f"created table `{self.db_table_name}` under schema `{self.container.dbschema}`")

    def create_volume(self, volume_name) -> None:
        self.drop_volume_if_exists(volume_name)
        query = f"CREATE VOLUME IF NOT EXISTS {self.container.dbname}.{self.container.dbschema}.{volume_name};"
        self._execute(query)
        _LOG.info(f"created temporary volume `{volume_name}` under schema `{self.container.dbschema}`")

    def drop_volume_if_exists(self, volume_name: str) -> None:
        query = f"DROP VOLUME IF EXISTS {self.container.dbname}.{self.container.dbschema}.{volume_name};"
        self._execute(query)
        _LOG.info(f"dropped temporary volume `{volume_name}` if existed")

    def drop_table_if_exists(self) -> None:
        query = f"DROP TABLE IF EXISTS {self.container.dbname}.{self.container.dbschema}.{self.db_table_name};"
        self._execute(query)
        _LOG.info(f"dropped table `{self.db_table_name}` if existed")

    def copy_data_from_volume_to_table(self, volume_name: str) -> None:
        query = (
            f"COPY INTO {self.container.dbname}.{self.container.dbschema}.{self.db_table_name} "
            f"FROM '/Volumes/{self.container.dbname}/{self.container.dbschema}/{volume_name}/' "
            "FILEFORMAT = PARQUET "
            "FORMAT_OPTIONS ('inferSchema' = 'true') "
            "COPY_OPTIONS ('mergeSchema' = 'true');"
        )
        self._execute(query)
        _LOG.info(f"copied data from volume `{volume_name}` to table `{self.db_table_name}`")

    def write_data(
        self,
        df: pd.DataFrame,
        if_exists: Literal["append", "replace", "fail"] = "append",
        **kwargs,
    ) -> None:
        t0 = time.time()
        assert self.is_output
        _LOG.info(f"write data {df.shape} to table `{self.db_table_name}` started")
        # include a unique timestamp in the volume name to avoid conflicts
        volume_name = f"{self.name}_temp_{time.time_ns()}"
        try:
            self.create_table(df, if_exists=if_exists)
            self.create_volume(volume_name)
            # write the whole partition in one go as it's faster than writing small chunks in parallel
            self.write_data_to_volume(df=df, volume_name=volume_name)
            self.copy_data_from_volume_to_table(volume_name)

            _LOG.info(f"write to table `{self.db_table_name}` finished in {time.time() - t0:.2f}s")
        except Exception as e:
            if if_exists != "append":
                self.drop_table_if_exists()
            raise e
        finally:
            self.drop_volume_if_exists(volume_name)

    def write_data_to_volume(
        self,
        df: pd.DataFrame,
        volume_name: str,
    ) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            filename = f"{self.name}.parquet"
            df.to_parquet(
                Path(tempdir) / filename,
                index=False,
                engine="pyarrow",
                # this will coerce Pandas' dtype `datetime[ns]` to a dtype that can be read by Spark
                use_deprecated_int96_timestamps=True,
            )
            # initialize a new engine with the extra `staging_allowed_local_path` argument
            # this is required for uploading local files to Databricks volumes
            sa_engine = sa.create_engine(
                url=self.container.sa_uri,
                connect_args=self.container.sa_engine_connection_kwargs | {"staging_allowed_local_path": tempdir},
                **self.container.sa_create_engine_kwargs,
            )
            try:
                with sa_engine.connect() as connection:
                    cursor = connection.connection.cursor()
                    query = (
                        f"PUT '{tempdir}/{filename}' INTO "
                        f"'/Volumes/{self.container.dbname}/{self.container.dbschema}/{volume_name}/{filename}' OVERWRITE"
                    )
                    cursor.execute(query)
                    _LOG.info(f"uploaded a partition to volume `{volume_name}`")
            finally:
                # ensure connections get closed
                sa_engine.dispose()
