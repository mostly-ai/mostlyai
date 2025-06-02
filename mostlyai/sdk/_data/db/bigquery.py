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
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd
from google.api_core.exceptions import NotFound
from google.cloud import bigquery
from google.oauth2.service_account import Credentials
from sqlalchemy import exc
from sqlalchemy_bigquery import BigQueryDialect

from mostlyai.sdk._data.db.base import DBDType, SqlAlchemyContainer, SqlAlchemyTable
from mostlyai.sdk._data.exceptions import MostlyDataException
from mostlyai.sdk._data.util.common import validate_gcs_key_file

_LOG = logging.getLogger(__name__)


class BigQueryDType(DBDType):
    @classmethod
    def sa_dialect_class(cls):
        return BigQueryDialect


class BigQueryContainer(SqlAlchemyContainer):
    SCHEMES = ["bigquery"]
    project_id: str
    client: bigquery.Client

    def post_init_hook(self):
        self.decrypt_secret()
        self.project_id = self.password["project_id"]
        self.dbschema = f"{self.project_id}.{self.dbname}"
        try:
            credentials = Credentials.from_service_account_info(self.password)
        except ValueError:
            raise MostlyDataException("Key file is incorrect.")
        self.client = bigquery.Client(credentials=credentials)

    @property
    def sa_create_engine_kwargs(self) -> dict:
        return {
            "credentials_info": self.password,
        }

    def decrypt_secret(self, secret_attr_name: str | None = None) -> None:
        super().decrypt_secret()
        self.password = validate_gcs_key_file(self.password)
        if not self.password:
            raise MostlyDataException("Key file is incorrect.")

    @property
    def sa_uri(self):
        return f"bigquery://{self.project_id}/{self.dbname}"

    @classmethod
    def table_class(cls):
        return BigQueryTable

    def get_view_list(self) -> list[str]:
        if not self.does_database_exist():
            return []
        return [table.table_id for table in self.client.list_tables(self.dbname) if table.table_type == "VIEW"]

    def get_table_list(self) -> list[str]:
        if not self.does_database_exist():
            return []
        return [table.table_id for table in self.client.list_tables(self.dbname) if table.table_type == "TABLE"]

    def does_database_exist(self) -> bool:
        datasets = list(self.client.list_datasets())
        return self.dbname in [dataset.dataset_id for dataset in datasets]

    def _is_schema_exist(self) -> bool:
        pass

    def is_accessible(self) -> bool:
        # TODO check connection without relying on database existence, if not provided (if possible)
        try:
            if self.dbname and not self.does_database_exist():
                raise MostlyDataException(f"Database `{self.dbname}` does not exist.")
            return True
        except (exc.SQLAlchemyError, NotFound, Exception) as e:
            error_message = str(e).lower()
            _LOG.error(f"Database connection failed with error: {e}")
            if "not found" and "project" in error_message:
                raise MostlyDataException(f"Project `{self.project_id}` does not exist.")
            else:
                raise

    def _valid_table_name(self, table_name: str) -> bool:
        """Validate table name."""
        return table_name in self.get_object_list()

    def _fetch_foreign_keys(self, table_name):
        """
        Fetch foreign keys for a table
        Upon scrutinizing the schema reflection for Big Query, it is able to discern
        the dataset lists and glean information about the columns to a certain extent.
        However, when it is attempted to retrieve the foreign keys using an inspector,
        an empty list is returned. This may be the cause of Big Query does not support
        foreign keys in a standard manner.
        This is the ticket for the issue:
        https://github.com/googleapis/python-bigquery-sqlalchemy/issues/866
        """
        if not self._valid_table_name(table_name):
            raise MostlyDataException(f"Table `{table_name}` does not exist.")
        table_constraints_sql = f"{self.dbschema}.INFORMATION_SCHEMA.TABLE_CONSTRAINTS"
        key_column_usage_sql = f"{self.dbschema}.INFORMATION_SCHEMA.KEY_COLUMN_USAGE"
        constraint_column_usage_sql = f"{self.dbschema}.INFORMATION_SCHEMA.CONSTRAINT_COLUMN_USAGE"
        query = f"""
        SELECT
            tc.CONSTRAINT_NAME,
            kcu.TABLE_NAME,
            kcu.COLUMN_NAME,
            ccu.TABLE_NAME as REFERENCED_TABLE_NAME,
            ccu.COLUMN_NAME as REFERENCED_COLUMN_NAME
        FROM `{table_constraints_sql}` as tc
        JOIN `{key_column_usage_sql}` as kcu
            ON tc.CONSTRAINT_NAME = kcu.CONSTRAINT_NAME
        JOIN `{constraint_column_usage_sql}` as ccu
            ON ccu.CONSTRAINT_NAME = tc.CONSTRAINT_NAME
        JOIN `{key_column_usage_sql}` as kcu_pk
            ON ccu.TABLE_NAME = kcu_pk.TABLE_NAME
                AND ccu.COLUMN_NAME = kcu_pk.COLUMN_NAME
        JOIN `{table_constraints_sql}` as tc_pk
            ON kcu_pk.CONSTRAINT_NAME = tc_pk.CONSTRAINT_NAME
        WHERE tc.TABLE_NAME = '{table_name}'
            AND tc.CONSTRAINT_TYPE = 'FOREIGN KEY'
            AND tc_pk.CONSTRAINT_TYPE = 'PRIMARY KEY'
        """
        try:
            rows = self.client.query(query).result()  # Waits for query to finish
            return [dict(row) for row in rows]
        except Exception:
            _LOG.debug(f"Error while executing {query=}")
            return []

    def _fetch_primary_key(self, table_name: str) -> list[str] | None:
        """
        Fetch primary key of a table
        The same comment as above applies here.
        """
        if not self._valid_table_name(table_name):
            raise MostlyDataException(f"Table `{table_name}` does not exist.")
        constraints_table = f"{self.dbschema}.INFORMATION_SCHEMA.TABLE_CONSTRAINTS"
        columns_table = f"{self.dbschema}.INFORMATION_SCHEMA.KEY_COLUMN_USAGE"
        query = f"""
            SELECT C.CONSTRAINT_NAME, C.TABLE_NAME, K.COLUMN_NAME
            FROM `{constraints_table}` AS C
            INNER JOIN `{columns_table}` AS K
            ON C.CONSTRAINT_NAME = K.CONSTRAINT_NAME
            WHERE C.TABLE_NAME = '{table_name}' AND C.CONSTRAINT_TYPE = 'PRIMARY KEY'
        """
        try:
            rows = self.client.query(query).result()  # Waits for query to finish
            return next((row["COLUMN_NAME"] for row in rows), None)
        except Exception:
            _LOG.debug(f"Error while executing {query=}")
            return None

    def get_primary_key(self, table_name: str):
        return self._fetch_primary_key(table_name)

    def get_foreign_keys(self, table_name: str):
        return self._fetch_foreign_keys(table_name)

    def update_dbschema(self, dbschema: str | None) -> None:
        self.dbname = dbschema
        self.dbschema = f"{self.project_id}.{self.dbname}"
        # reset engine
        self._sa_engine_for_read = None
        self._sa_engine_for_write = None


class BigQueryTable(SqlAlchemyTable):
    DATA_TABLE_TYPE = "bigquery"
    SA_RANDOM = None  # Replace with your equivalent function in BigQuery

    @classmethod
    def dtype_class(cls):
        return BigQueryDType

    @classmethod
    def container_class(cls):
        return BigQueryContainer

    def write_chunks(self, chunks: Iterable[pd.DataFrame], dtypes: dict[str, Any], **kwargs) -> None:
        _LOG.info(f"write data in parallel in {len(chunks)} chunks")

        # Here joblib Parallel, delayed also works but it's slower than ThreadPoolExecutor
        # For 100 records, Parallel, delayed takes 7s, ThreadPoolExecutor takes 4s
        with ThreadPoolExecutor() as executor:
            for chunk in chunks:
                executor.submit(
                    _write_chunk,
                    credentials_info=self.container.sa_create_engine_kwargs["credentials_info"],
                    dataframe=chunk,
                    table_id=f"{self.container.dbschema}.{self.name}",
                    if_exists=True,
                )


def _write_chunk(
    credentials_info: dict[str, str | list | dict],
    dataframe: pd.DataFrame,
    table_id: str,
    if_exists: bool,
) -> None:
    # Create a BigQuery client in the worker process
    credentials = Credentials.from_service_account_info(credentials_info)
    client = bigquery.Client(credentials=credentials)
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_APPEND" if if_exists else "WRITE_EMPTY",
    )
    job = client.load_table_from_dataframe(dataframe, table_id, job_config=job_config)
    # Wait for the job to complete
    job.result()
