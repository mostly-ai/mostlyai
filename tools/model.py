# Copyright 2024-2025 MOSTLY AI
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
import uuid
import zipfile
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal

import pandas as pd
import rich
from pydantic import Field, field_validator, model_validator

from mostlyai.sdk.client._base_utils import convert_to_base64, read_table_from_path
from mostlyai.sdk.client.base import CustomBaseModel
from mostlyai.sdk.domain import (
    ArtifactPatchConfig,
    ConnectorAccessType,
    ConnectorPatchConfig,
    DatasetConnector,
    DatasetPatchConfig,
    Generator,
    GeneratorConfig,
    GeneratorPatchConfig,
    JobProgress,
    ModelEncodingType,
    ModelType,
    ProgressStatus,
    SourceColumnConfig,
    SyntheticDatasetDelivery,
    SyntheticDatasetPatchConfig,
    SyntheticDatasetReportType,
)


class Connector:
    OPEN_URL_PARTS: ClassVar[list] = ["d", "connectors"]

    @model_validator(mode="before")
    @classmethod
    def add_required_fields(cls, values):
        if isinstance(values, dict):
            if "id" not in values:
                values["id"] = str(uuid.uuid4())
            if values.get("name") is None:
                values["name"] = "New connector"
        return values

    def update(
        self,
        name: str | None = None,
        description: str | None = None,
        access_type: ConnectorAccessType | None = None,
        config: dict[str, Any] | None = None,
        secrets: dict[str, str] | None = None,
        ssl: dict[str, str] | None = None,
        test_connection: bool | None = True,
    ) -> None:
        """
        Update a connector with specific parameters.

        Args:
            name (str | None): The name of the connector.
            description (str | None): The description of the connector.
            access_type (ConnectorAccessType | None): The access type of the connector.
            config (dict[str, Any] | None): Connector configuration.
            secrets (dict[str, str] | None): Secret values for the connector.
            ssl (dict[str, str] | None): SSL configuration for the connector.
            test_connection (bool | None): If true, validates the connection before saving.
        """
        patch_config = ConnectorPatchConfig(
            name=name,
            description=description,
            access_type=access_type,
            config=config,
            secrets=secrets,
            ssl=ssl,
        )
        self.client._update(
            connector_id=self.id,
            config=patch_config,
            test_connection=test_connection,
        )
        self.reload()

    def delete(self) -> None:
        """
        Delete the connector.
        """
        return self.client._delete(connector_id=self.id)

    def locations(self, prefix: str = "") -> list[str]:
        """
        List connector locations.

        List the available databases, schemas, tables, or folders for a connector.
        For storage connectors, this returns list of buckets for empty prefix and otherwise the folders and files then within a path, specified via prefix.
        For DB connectors, this returns list of schemas (or databases for DBs without schema), respectively list of tables if `prefix` is provided.

        The formats of the locations are:

        - Cloud storage:
            - `AZURE_STORAGE`: `container/path`
            - `GOOGLE_CLOUD_STORAGE`: `bucket/path`
            - `S3_STORAGE`: `bucket/path`
        - Database:
            - `BIGQUERY`: `dataset.table`
            - `DATABRICKS`: `schema.table`
            - `HIVE`: `database.table`
            - `MARIADB`: `database.table`
            - `MSSQL`: `schema.table`
            - `MYSQL`: `database.table`
            - `ORACLE`: `schema.table`
            - `POSTGRES`: `schema.table`
            - `SNOWFLAKE`: `schema.table`

        Args:
            prefix (str): The prefix to filter the results by. Defaults to an empty string.

        Returns:
            list[str]: A list of locations (schemas, databases, directories, etc.).

        Example:
            ```python
            c.locations()  # list all schemas / databases for a DB connector; list all buckets for a storage connector
            c.locations('db_name')  # list all tables in 'db_name' for a DB connector
            c.locations('s3://my_bucket')  # list all objects in 'my_bucket' for a S3 storage connector
            c.locations('gs://my_bucket/path/to/folder')  # list all objects in 'my_bucket/path/to/folder' for a GCP storage connector
            c.locations('az://my_container/path/to/folder')  # list all objects in 'my_container/path/to/folder' for a AZURE storage connector
            ```
        """
        return self.client._locations(connector_id=self.id, prefix=prefix)

    def schema(self, location: str) -> list[dict[str, Any]]:
        """
        Retrieve the schema of the table at a connector location.
        This method is available for all connectors.

        Args:
            location (str): The location of the table.

        Returns:
            list[dict[str, Any]]: The retrieved schema.

        Example:
            ```python
            c.schema('db_name.table_name')  # get the schema of 'table_name' in 'db_name' for a DB connector
            c.schema('s3://my_bucket/path/to/file.csv')  # get the schema of 'file.csv' in 'my_bucket' for a S3 storage connector
            ```
        """
        return self.client._schema(connector_id=self.id, location=location)

    def read_data(self, location: str, limit: int | None = None, shuffle: bool = False) -> pd.DataFrame:
        """
        Retrieve data from the specified location within the connector.
        This method is only available for connectors of access_type READ_DATA or WRITE_DATA.

        Args:
            location (str): The target location within the connector to read data from.
            limit (int | None, optional): The maximum number of rows to return. Returns all if not specified.
            shuffle (bool | None, optional): Whether to shuffle the results.

        Returns:
            pd.DataFrame: A DataFrame containing the retrieved data.

        Example:
            ```python
            df = c.read_data('db_name.table_name', limit=100)  # fetch first 100 rows from 'table_name' in 'db_name' for a DB connector
            df = c.read_data('s3://my_bucket/path/to/file.csv')  # read all data from 'file.csv' in 'my_bucket' for a S3 storage connector
            ```
        """
        return self.client._read_data(connector_id=self.id, location=location, limit=limit, shuffle=shuffle)

    def write_data(
        self, data: pd.DataFrame | None, location: str, if_exists: Literal["append", "replace", "fail"] = "fail"
    ) -> None:
        """
        Write data to the specified location within the connector.
        This method is only available for connectors of access_type WRITE_DATA.

        Args:
            data (pd.DataFrame | None): The DataFrame to write, or None to delete the location.
            location (str): The target location within the connector to write data to.
            if_exists (Literal["append", "replace", "fail"]): The behavior if the target location already exists (append, replace, fail). Default is "fail".

        Example:
            ```python
            c.write_data(df, 'db_name.table_name', if_exists='fail')  # write data to 'table_name' in 'db_name' for a DB connector (if it doesn't exist)
            c.write_data(df, 's3://my_bucket/path/to/file.csv')  # write data to 'file.csv' in 'my_bucket' for a S3 storage connector
            ```
        """
        self.client._write_data(connector_id=self.id, data=data, location=location, if_exists=if_exists.upper())

    def delete_data(self, location: str) -> None:
        """
        Delete data from the specified location within the connector.
        This method is only available for connectors of access_type WRITE_DATA.

        Args:
            location (str): The target location within the connector to delete data from.

        Example:
            ```python
            c.delete_data('db_name.table_name')  # drop table data from 'table_name' in 'db_name' for a DB connector
            c.delete_data('s3://my_bucket/path/to/file.csv')  # delete data from 'file.csv' in 'my_bucket' for a S3 storage connector
            ```
        """
        self.client._delete_data(connector_id=self.id, location=location)

    def query(self, sql: str) -> pd.DataFrame:
        """
        Execute a read-only SQL query against the connector's data source.

        Queries can include statements like SELECT, SHOW, or DESCRIBE, but must not modify data or state.
        For file-based connectors (S3_STORAGE, GOOGLE_CLOUD_STORAGE, AZURE_STORAGE) queries are executed using DuckDB. Use connector-type-specific prefixes. See examples.

        Args:
            sql (str): The SQL query to execute.

        Returns:
            pd.DataFrame: The result of the query as a Pandas DataFrame.

        Example:
            ```python
            df = c.query("SELECT count(*) FROM schema.table")  # for DB connectors
            df = c.query("SELECT count(*) FROM read_csv_auto('s3://bucket/path/to/file.csv')")  # query a single CSV file from S3 storage
            df = c.query("SELECT count(*) FROM read_parquet('gs://bucket/path/to/folder/*.parquet')")  # query a folder with PQT files from GCP storage
            df = c.query("SELECT count(*) FROM read_json_auto('az://bucket/path/to/file.json')")  # query a single JSON file from AZURE storage
            ```
        """
        return self.client._query(connector_id=self.id, sql=sql)


class Generator:
    OPEN_URL_PARTS: ClassVar[list] = ["d", "generators"]
    training: Annotated[Any | None, Field(exclude=True)] = None

    @model_validator(mode="before")
    @classmethod
    def add_required_fields(cls, values):
        if isinstance(values, dict):
            if "id" not in values:
                values["id"] = str(uuid.uuid4())
            if values.get("name") is None:
                values["name"] = "New generator"
            if "training_status" not in values:
                values["training_status"] = ProgressStatus.new
        return values

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training = self.Training(self)

    def update(
        self,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        """
        Update a generator with specific parameters.

        Args:
            name (str | None): The name of the generator.
            description (str | None): The description of the generator.
        """
        patch_config = GeneratorPatchConfig(
            name=name,
            description=description,
        )
        self.client._update(generator_id=self.id, config=patch_config)
        self.reload()

    def delete(self) -> None:
        """
        Delete the generator.
        """
        return self.client._delete(generator_id=self.id)

    def config(self) -> GeneratorConfig:
        """
        Retrieve writable generator properties.

        Returns:
            GeneratorConfig: The generator properties as a configuration object.
        """
        return self.client._config(generator_id=self.id)

    def export_to_file(
        self,
        file_path: str | Path | None = None,
    ) -> Path:
        """
        Export generator and save to file.

        Args:
            file_path (str | Path | None): The file path to save the generator.

        Returns:
            Path: The path to the saved file.
        """
        bytes, filename = self.client._export_to_file(generator_id=self.id)
        file_path = Path(file_path or ".")
        if file_path.is_dir():
            file_path = file_path / filename
        file_path.write_bytes(bytes)
        return file_path

    def clone(self, training_status: Literal["new", "continue"] = "new") -> Generator:
        """
        Clone the generator.

        Args:
            training_status (Literal["new", "continue"]): The training status of the cloned generator. Default is "new".

        Returns:
            Generator: The cloned generator object.
        """
        generator = self.client._clone(generator_id=self.id, training_status=training_status)
        gid = generator.id
        if self.client.local:
            rich.print(f"Created generator [dodger_blue2]{gid}[/]")
        else:
            rich.print(
                f"Created generator [link={self.client.base_url}/d/generators/{gid} dodger_blue2 underline]{gid}[/]"
            )
        return generator

    def reports(self, file_path: str | Path | None = None, display: bool = False) -> Path | None:
        """
        Download or display the quality assurance reports.

        If display is True, the report is rendered inline via IPython display and no file is downloaded.
        Otherwise, the report is downloaded and saved to file_path (or a default location if None).

        Note that reports are not available for generators that were trained with less than 100 samples or had `enable_model_report` set to `False`.

        Args:
            file_path (str | Path | None): The file path to save the zipped reports (ignored if display=True).
            display (bool): If True, render the report inline instead of downloading it.

        Returns:
            Path | None: The path to the saved file if downloading, or None if display=True.
        """
        reports = {}
        for table in self.tables:
            if table.tabular_model_metrics:
                reports[f"{table.name}-tabular.html"] = self.client._report(
                    generator_id=self.id,
                    source_table_id=table.id,
                    model_type="TABULAR",
                    short_lived_file_token=(self.metadata.short_lived_file_token if self.metadata else None),
                )
            if table.language_model_metrics:
                reports[f"{table.name}-language.html"] = self.client._report(
                    generator_id=self.id,
                    source_table_id=table.id,
                    model_type="LANGUAGE",
                    short_lived_file_token=(self.metadata.short_lived_file_token if self.metadata else None),
                )

        if display and rich.console._is_jupyter():
            from IPython.display import HTML, display  # noqa
            import html  # noqa

            iframes = ""
            for content in reports.values():
                content = html.escape(content, quote=True)
                iframes += f'<p><iframe srcdoc="{content}" width="100%" height="600"></iframe></p> '

            display(HTML(iframes))
            return None
        else:
            file_path = Path(file_path or ".")
            if file_path.is_dir():
                file_path = file_path / f"generator-{self.id[:8]}-reports.zip"
            if file_path.exists():
                file_path.unlink()
            with zipfile.ZipFile(file_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
                for filename, content in reports.items():
                    zip_file.writestr(filename, content)
            if display:
                rich.print(f"Reports saved to {file_path}")
            return file_path

    class Training:
        def __init__(self, _generator: "Generator"):
            self.generator = _generator

        def start(self) -> None:
            """
            Start training.
            """
            rich.print("Started generator training")
            self.generator.client._training_start(self.generator.id)

        def cancel(self) -> None:
            """
            Cancel training.
            """
            self.generator.client._training_cancel(self.generator.id)
            self.generator.reload()

        def progress(self) -> JobProgress:
            """
            Retrieve job progress of training.

            Returns:
                JobProgress: The job progress of the training process.
            """
            return self.generator.client._training_progress(self.generator.id)

        def wait(self, progress_bar: bool = True, interval: float = 2) -> None:
            """
            Poll training progress and loop until training has completed.

            Args:
                progress_bar (bool): If true, displays the progress bar. Default is True.
                interval (float): The interval in seconds to poll the job progress. Default is 2 seconds.
            """
            self.generator.client._training_wait(self.generator.id, progress_bar=progress_bar, interval=interval)
            self.generator.reload()
            if self.generator.training_status == ProgressStatus.done:
                rich.print(
                    ":tada: [bold green]Your generator is ready![/] "
                    "Use it to create synthetic data. "
                    "Publish it so others can do the same."
                )

        def logs(self, file_path: str | Path | None = None) -> Path:
            """
            Download the training logs and save to file.

            Args:
                file_path (str | Path | None): The file path to save the logs. Default is the current working directory.

            Returns:
                Path: The path to the saved file.
            """
            bytes, filename = self.generator.client._training_logs(
                generator_id=self.generator.id,
                short_lived_file_token=self.generator.metadata.short_lived_file_token
                if self.generator.metadata
                else None,
            )
            file_path = Path(file_path or ".")
            if file_path.is_dir():
                file_path = file_path / filename
            file_path.write_bytes(bytes)
            return file_path


class GeneratorConfig:
    @field_validator("tables", mode="after")
    @classmethod
    def validate_unique_table_names(cls, tables):
        defined_tables = [t.name for t in tables or []]
        if len(defined_tables) != len(set(defined_tables)):
            raise ValueError("Table names must be unique.")
        return tables

    @field_validator("tables", mode="after")
    @classmethod
    def validate_each_referenced_table_exist_and_has_primary_key(cls, tables):
        table_map = {table.name: table for table in tables or []}
        for table in tables or []:
            for fk in table.foreign_keys or []:
                ref_table = table_map.get(fk.referenced_table)
                if not ref_table:
                    raise ValueError(
                        f"Foreign key in table '{table.name}' references a non-existent table: '{fk.referenced_table}'."
                    )
                if not ref_table.primary_key:
                    raise ValueError(f"Referenced table '{fk.referenced_table}' does not have a primary key.")
        return tables

    @field_validator("tables", mode="after")
    @classmethod
    def validate_no_circular_context_references(cls, tables):
        if not tables:
            return tables
        table_map = {table.name: table for table in tables}
        visited = set()
        for table in tables:
            if table.name in visited:
                continue
            current_table = table
            seen_tables = set()
            while current_table:
                if current_table.name in seen_tables:
                    raise ValueError(f"Circular reference detected in tables: {', '.join(seen_tables)}")
                seen_tables.add(current_table.name)
                context_fk = next((fk for fk in (current_table.foreign_keys or []) if fk.is_context), None)
                if not context_fk or not context_fk.referenced_table:
                    break
                current_table = table_map.get(context_fk.referenced_table)
            visited.update(seen_tables)
        return tables


class SourceTableConfig:
    @field_validator("data", mode="before")
    @classmethod
    def convert_data_before(cls, value):
        # an empty (pd.DataFrame()) parquet in base64 is 800 chars. Assuming a shorter str is a URI
        if isinstance(value, Path) or (isinstance(value, str) and len(value) > 0 and len(value) < 512):
            _, value = read_table_from_path(value)
        return (
            convert_to_base64(value)
            if isinstance(value, pd.DataFrame)
            or (value.__class__.__name__ == "DataFrame" and value.__class__.__module__.startswith("pyspark.sql"))
            else value
        )

    @field_validator("columns", mode="before")
    @classmethod
    def filter_excluded_columns(cls, columns):
        if columns is None or not isinstance(columns, list):
            return columns
        included_columns = []
        for column in columns:
            is_included = any(
                (
                    isinstance(column, dict) and bool(column.get("included", True)),
                    isinstance(column, SourceColumn) and column.included,
                    isinstance(column, SourceColumnConfig),
                )
            )
            if is_included:
                included_columns.append(column)
        return included_columns

    @model_validator(mode="after")
    def add_model_configuration(self):
        # Check if the table has a tabular and/or a language model
        keys = [fk.column for fk in self.foreign_keys or []]
        if self.primary_key:
            keys.append(self.primary_key)
        model_columns = [c for c in self.columns if c.name not in keys] if self.columns is not None else None
        if model_columns is None:
            # auto detection haven't been run yet, so we assume both models are present to retain model configurations given by the user
            has_tabular_model = True
            has_language_model = True
        elif len(model_columns) == 0:
            # this table doesn't have any columns other than PK/FKs
            has_tabular_model = True
            has_language_model = False
        else:
            enc_types = [c.model_encoding_type or ModelEncodingType.auto for c in model_columns]
            has_tabular_model = any(
                enc_type.startswith(ModelType.tabular) or enc_type == ModelEncodingType.auto for enc_type in enc_types
            )
            has_language_model = any(
                enc_type.startswith(ModelType.language) or enc_type == ModelEncodingType.auto for enc_type in enc_types
            )
        # Always train tabular model for tables with a primary key or linked tables to model sequences
        if self.primary_key or (self.foreign_keys and any(fk.is_context for fk in self.foreign_keys)):
            has_tabular_model = True
        # Remove model configurations that are not applicable for the model type
        if self.tabular_model_configuration and not has_tabular_model:
            self.tabular_model_configuration = None
        if self.language_model_configuration and not has_language_model:
            self.language_model_configuration = None
        # Add default model configurations if none were provided
        if has_tabular_model:
            default_model = "MOSTLY_AI/Medium"
            if not self.tabular_model_configuration:
                self.tabular_model_configuration = ModelConfiguration(model=default_model)
            elif not self.tabular_model_configuration.model:
                self.tabular_model_configuration.model = default_model
        if has_language_model:
            default_model = "MOSTLY_AI/LSTMFromScratch-3m"
            if not self.language_model_configuration:
                self.language_model_configuration = ModelConfiguration(model=default_model, max_sequence_window=None)
            elif not self.language_model_configuration.model:
                self.language_model_configuration.model = default_model
            # language models atm do not support max_sequence_window; thus set configuration to None
            self.language_model_configuration.max_sequence_window = None
        return self

    @field_validator("columns", mode="after")
    @classmethod
    def validate_unique_columns(cls, columns):
        if columns:
            defined_columns = [c.name for c in columns]
            if len(defined_columns) != len(set(defined_columns)):
                raise ValueError("Column names must be unique.")
        return columns

    @model_validator(mode="after")
    def validate_unique_keys(self):
        pk = self.primary_key or []
        fks = [fk.column for fk in self.foreign_keys or []]
        if len(fks) != len(set(fks)):
            raise ValueError("Foreign key column names must be unique.")
        if pk in fks:
            raise ValueError("Primary key column name must not be defined as foreign key.")
        return self

    @field_validator("foreign_keys", mode="after")
    @classmethod
    def validate_at_most_one_context_fk(cls, foreign_keys):
        if foreign_keys:
            context_fks = [fk for fk in foreign_keys if fk.is_context]
            if len(context_fks) > 1:
                raise ValueError("At most one context foreign key is allowed")
        return foreign_keys

    @model_validator(mode="after")
    def validate_keys_exists_in_columns(self):
        if self.columns is not None:
            column_names = {col.name for col in self.columns}
            pk = self.primary_key
            if pk and pk not in column_names:
                raise ValueError(f"Primary key column '{pk}' does not exist in the table's columns.")
            for fk in self.foreign_keys or []:
                if fk.column not in column_names:
                    raise ValueError(f"Foreign key column '{fk.column}' does not exist in the table's columns.")
        return self

    @model_validator(mode="after")
    def validate_pk_and_fks_are_not_overlapping(self):
        primary_key = self.primary_key
        foreign_keys = [fk.column for fk in self.foreign_keys or []]
        if primary_key and primary_key in foreign_keys:
            raise ValueError(f"Column '{primary_key}' is both a primary key and a foreign key.")
        return self

    @model_validator(mode="after")
    def validate_data_or_connector_is_provided(self):
        if self.data is None and (self.source_connector_id is None or self.location is None):
            raise ValueError(
                "At least one input source must be provided: either `data`, or both `source_connector_id` and `location`."
            )
        elif self.data is not None and (self.source_connector_id is not None or self.location is not None):
            raise ValueError(
                "Only one input source is allowed: either `data`, or both `source_connector_id` and `location`."
            )
        return self


class SourceColumn:
    @model_validator(mode="before")
    @classmethod
    def add_required_fields(cls, values):
        if isinstance(values, dict):
            if "id" not in values:
                values["id"] = str(uuid.uuid4())
            if "model_encoding_type" not in values:
                values["model_encoding_type"] = ModelEncodingType.auto
            if "included" not in values:
                values["included"] = True
        return values


class ModelConfiguration:
    @model_validator(mode="after")
    def validate_differential_privacy_config(self):
        if self.differential_privacy:
            if not self.value_protection:
                self.differential_privacy.value_protection_epsilon = None
            else:
                if self.differential_privacy.value_protection_epsilon is None:
                    self.differential_privacy.value_protection_epsilon = 1.0
        return self


class SyntheticTableConfiguration:
    @field_validator("sample_seed_dict", mode="before")
    @classmethod
    def convert_dict_before(cls, value):
        return convert_to_base64(value, format="jsonl") if isinstance(value, (dict, pd.DataFrame)) else value

    @field_validator("sample_seed_data", mode="before")
    @classmethod
    def convert_data_before(cls, value):
        return (
            convert_to_base64(value)
            if isinstance(value, pd.DataFrame)
            or (value.__class__.__name__ == "DataFrame" and value.__class__.__module__.startswith("pyspark.sql"))
            else value
        )

    @model_validator(mode="after")
    def add_required_fields(self):
        if self.sampling_temperature is None:
            self.sampling_temperature = 1.0
        if self.sampling_top_p is None:
            self.sampling_top_p = 1.0
        return self


class SyntheticDataset:
    OPEN_URL_PARTS: ClassVar[list] = ["d", "synthetic-datasets"]
    generation: Annotated[Any | None, Field(exclude=True)] = None

    @model_validator(mode="before")
    @classmethod
    def add_required_fields(cls, values):
        if isinstance(values, dict):
            if "id" not in values:
                values["id"] = str(uuid.uuid4())
            if values.get("name") is None:
                values["name"] = "New synthetic dataset"
            if "generation_status" not in values:
                values["generation_status"] = ProgressStatus.new
        return values

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generation = self.Generation(self)

    def update(
        self,
        name: str | None = None,
        description: str | None = None,
        delivery: SyntheticDatasetDelivery | None = None,
    ) -> None:
        """
        Update a synthetic dataset with specific parameters.

        Args:
            name (str | None): The name of the synthetic dataset.
            description (str | None): The description of the synthetic dataset.
            delivery (SyntheticDatasetDelivery | None): The delivery configuration for the synthetic dataset.
        """
        patch_config = SyntheticDatasetPatchConfig(
            name=name,
            description=description,
            delivery=delivery,
        )
        self.client._update(
            synthetic_dataset_id=self.id,
            config=patch_config,
        )
        self.reload()

    def delete(self) -> None:
        """
        Delete the synthetic dataset.
        """
        return self.client._delete(synthetic_dataset_id=self.id)

    def config(self) -> "SyntheticDatasetConfig":
        """
        Retrieve writable synthetic dataset properties.

        Returns:
            SyntheticDatasetConfig: The synthetic dataset properties as a configuration object.
        """
        return self.client._config(synthetic_dataset_id=self.id)

    def download(
        self,
        file_path: str | Path | None = None,
        format: Literal["parquet", "csv", "json"] = "parquet",
    ) -> Path:
        """
        Download synthetic dataset and save to file.

        Args:
            file_path (str | Path | None): The file path to save the synthetic dataset.
            format (Literal["parquet", "csv", "json"]): The format of the synthetic dataset. Default is "parquet".

        Returns:
            Path: The path to the saved file.
        """
        bytes, filename = self.client._download(
            synthetic_dataset_id=self.id,
            ds_format=format.upper(),
            short_lived_file_token=self.metadata.short_lived_file_token if self.metadata else None,
        )
        file_path = Path(file_path or ".")
        if file_path.is_dir():
            file_path = file_path / filename
        file_path.write_bytes(bytes)
        return file_path

    def data(self, return_type: Literal["auto", "dict"] = "auto") -> pd.DataFrame | dict[str, pd.DataFrame]:
        """
        Download synthetic dataset and return as dictionary of pandas DataFrames.

        Args:
            return_type (Literal["auto", "dict"]): The type of the return value. "dict" will always provide a dictionary of DataFrames. "auto" will return a single DataFrame for a single-table generator, and a dictionary of DataFrames for a multi-table generator. Default is "auto".

        Returns:
            Union[pd.DataFrame, dict[str, pd.DataFrame]]: The synthetic dataset. See return_type for the format of the return value.
        """
        dfs = self.client._data(
            synthetic_dataset_id=self.id,
            short_lived_file_token=self.metadata.short_lived_file_token if self.metadata else None,
        )
        if return_type == "auto" and len(dfs) == 1:
            return list(dfs.values())[0]
        else:
            return dfs

    def reports(self, file_path: str | Path | None = None, display: bool = False) -> Path | None:
        """
        Download or display the quality assurance reports.

        If display is True, the report is rendered inline via IPython display and no file is downloaded.
        Otherwise, the report is downloaded and saved to file_path (or a default location if None).

        Note that reports are not available for synthetic datasets that generated less than 100 samples or had `enable_data_report` set to `False`.

        Args:
            file_path (str | Path | None): The file path to save the zipped reports (ignored if display=True).
            display (bool): If True, render the report inline instead of downloading it.

        Returns:
            Path | None: The path to the saved file if downloading, or None if display=True.
        """
        reports = {}
        for report_type in [SyntheticDatasetReportType.model, SyntheticDatasetReportType.data]:
            report_infix = "" if report_type == SyntheticDatasetReportType.model else "-data"
            for table in self.tables:
                if table.tabular_model_metrics:
                    reports[f"{table.name}-tabular{report_infix}.html"] = self.client._report(
                        synthetic_dataset_id=self.id,
                        synthetic_table_id=table.id,
                        model_type="TABULAR",
                        report_type=report_type,
                        short_lived_file_token=(self.metadata.short_lived_file_token if self.metadata else None),
                    )
                if table.language_model_metrics:
                    reports[f"{table.name}-language{report_infix}.html"] = self.client._report(
                        synthetic_dataset_id=self.id,
                        synthetic_table_id=table.id,
                        model_type="LANGUAGE",
                        report_type=report_type,
                        short_lived_file_token=(self.metadata.short_lived_file_token if self.metadata else None),
                    )

        if display and rich.console._is_jupyter():
            from IPython.display import HTML, display  # noqa
            import html  # noqa

            iframes = ""
            for content in reports.values():
                content = html.escape(content, quote=True)
                iframes += f'<p><iframe srcdoc="{content}" width="100%" height="600"></iframe></p> '

            display(HTML(iframes))
            return None
        else:
            file_path = Path(file_path or ".")
            if file_path.is_dir():
                file_path = file_path / f"synthetic-dataset-{self.id[:8]}-reports.zip"
            if file_path.exists():
                file_path.unlink()
            with zipfile.ZipFile(file_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
                for filename, content in reports.items():
                    zip_file.writestr(filename, content)
            if display:
                rich.print(f"Reports saved to {file_path}")
            return file_path

    class Generation:
        def __init__(self, _synthetic_dataset: "SyntheticDataset"):
            self.synthetic_dataset = _synthetic_dataset

        def start(self) -> None:
            """
            Start the generation process.
            """
            self.synthetic_dataset.client._generation_start(self.synthetic_dataset.id)
            rich.print("Started synthetic dataset generation")

        def cancel(self) -> None:
            """
            Cancel the generation process.
            """
            self.synthetic_dataset.client._generation_cancel(self.synthetic_dataset.id)
            self.synthetic_dataset.reload()

        def progress(self) -> JobProgress:
            """
            Retrieve the progress of the generation process.

            Returns:
                JobProgress: The progress of the generation process.
            """
            return self.synthetic_dataset.client._generation_progress(self.synthetic_dataset.id)

        def wait(self, progress_bar: bool = True, interval: float = 2) -> None:
            """
            Poll the generation progress and wait until the process is complete.

            Args:
                progress_bar (bool): If true, displays a progress bar. Default is True.
                interval (float): Interval in seconds to poll the job progress. Default is 2 seconds.
            """
            self.synthetic_dataset.client._generation_wait(
                self.synthetic_dataset.id, progress_bar=progress_bar, interval=interval
            )
            self.synthetic_dataset.reload()
            if self.synthetic_dataset.generation_status == ProgressStatus.done:
                rich.print(
                    ":tada: [bold green]Your synthetic dataset is ready![/] "
                    "Use it to consume the generated data. "
                    "Publish it so others can do the same."
                )

        def logs(self, file_path: str | Path | None = None) -> Path:
            """
            Download the generation logs and save to file.

            Args:
                file_path (str | Path | None): The file path to save the logs. Default is the current working directory.

            Returns:
                Path: The path to the saved file.
            """
            bytes, filename = self.synthetic_dataset.client._generation_logs(
                synthetic_dataset_id=self.synthetic_dataset.id,
                short_lived_file_token=self.synthetic_dataset.metadata.short_lived_file_token
                if self.synthetic_dataset.metadata
                else None,
            )
            file_path = Path(file_path or ".")
            if file_path.is_dir():
                file_path = file_path / filename
            file_path.write_bytes(bytes)
            return file_path


class SyntheticDatasetConfig:
    @field_validator("tables", mode="after")
    @classmethod
    def validate_unique_table_names(cls, tables):
        if not tables:
            return tables
        defined_tables = [t.name for t in tables]
        if len(defined_tables) != len(set(defined_tables)):
            raise ValueError("Table names must be unique.")
        return tables

    def validate_against_generator(self, generator: Generator) -> None:
        _SyntheticDataConfigValidation(synthetic_config=self, generator=generator)


class SyntheticProbeConfig:
    @field_validator("tables", mode="after")
    @classmethod
    def validate_unique_table_names(cls, tables):
        if not tables:
            return tables
        defined_tables = [t.name for t in tables]
        if len(defined_tables) != len(set(defined_tables)):
            raise ValueError("Table names must be unique.")
        return tables

    def validate_against_generator(self, generator: Generator) -> None:
        _SyntheticDataConfigValidation(synthetic_config=self, generator=generator)


class SourceTable:
    @model_validator(mode="before")
    @classmethod
    def add_required_fields(cls, values):
        if isinstance(values, dict):
            if "id" not in values:
                values["id"] = str(uuid.uuid4())
        return values


class SyntheticTable:
    @model_validator(mode="before")
    @classmethod
    def add_required_fields(cls, values):
        if isinstance(values, dict):
            if "id" not in values:
                values["id"] = str(uuid.uuid4())
        return values


class SyntheticTableConfig:
    @model_validator(mode="after")
    def add_configuration(self):
        if self.configuration is None:
            self.configuration = SyntheticTableConfiguration()
        return self

    def validate_against_source_table(self, source_table: SourceTable, is_probe: bool) -> None:
        self._maybe_set_sample_size(source_table, is_probe)
        _SyntheticTableConfigValidation(synthetic_table=self, source_table=source_table)

    def _maybe_set_sample_size(self, source_table: SourceTable, is_probe: bool) -> None:
        config = self.configuration
        is_subject = not any(fk.is_context for fk in source_table.foreign_keys or [])
        if (
            not config.sample_size
            and is_subject
            and not (config.sample_seed_connector_id or config.sample_seed_dict or config.sample_seed_data)
        ):
            config.sample_size = 1 if is_probe else source_table.total_rows
        elif not is_subject:
            config.sample_size = None


class SourceForeignKey:
    @model_validator(mode="before")
    @classmethod
    def add_required_fields(cls, values):
        if isinstance(values, dict):
            if "id" not in values:
                values["id"] = str(uuid.uuid4())
        return values


class _SyntheticTableConfigValidation(CustomBaseModel):
    """
    Validation logic for SyntheticTableConfig against SourceTable
    """

    synthetic_table: SyntheticTableConfig
    source_table: SourceTable

    @model_validator(mode="after")
    def validate_rebalancing_config(self):
        config = self.synthetic_table.configuration
        if config and config.rebalancing:
            rebalancing_column = config.rebalancing.column
            rebalancing_col = next(
                (col for col in self.source_table.columns or [] if col.name == rebalancing_column),
                None,
            )
            if not rebalancing_col:
                raise ValueError(
                    f"Rebalancing column '{rebalancing_column}' not found in table '{self.source_table.name}'"
                )
            if not rebalancing_col.model_encoding_type == ModelEncodingType.tabular_categorical:
                raise ValueError(
                    f"Rebalancing column '{rebalancing_column}' in table '{self.source_table.name}' must be categorical"
                )
            if not rebalancing_col.included:
                raise ValueError(
                    f"Rebalancing column '{rebalancing_column}' in table '{self.source_table.name}' must have `included=True`"
                )
            for value in config.rebalancing.probabilities.keys():
                if value not in rebalancing_col.value_range.values:
                    raise ValueError(
                        f"Rebalancing value '{value}' not found in table '{self.source_table.name}' column '{rebalancing_column}'"
                    )
        return self

    @model_validator(mode="after")
    def validate_imputation_config(self):
        config = self.synthetic_table.configuration
        if config and config.imputation:
            has_tabular_model = self.source_table.tabular_model_configuration is not None
            if not has_tabular_model:
                raise ValueError(f"Table '{self.source_table.name}' specifies imputation but has no tabular model")

            for col in config.imputation.columns:
                if not any(gcol.name == col for gcol in self.source_table.columns or []):
                    raise ValueError(f"Imputation column '{col}' not found in table '{self.source_table.name}'")
        return self

    @model_validator(mode="after")
    def validate_fairness_config(self):
        config = self.synthetic_table.configuration
        if config and config.fairness:
            has_tabular_model = self.source_table.tabular_model_configuration is not None
            if not has_tabular_model:
                raise ValueError(f"Table '{self.source_table.name}' specifies fairness but has no tabular model")

            target_col = config.fairness.target_column
            if not any(col.name == target_col for col in self.source_table.columns or []):
                raise ValueError(f"Fairness target column '{target_col}' not found in table '{self.source_table.name}'")

            for col in config.fairness.sensitive_columns:
                if not any(gcol.name == col for gcol in self.source_table.columns or []):
                    raise ValueError(f"Fairness sensitive column '{col}' not found in table '{self.source_table.name}'")

            if target_col in config.fairness.sensitive_columns:
                raise ValueError(f"Target column '{target_col}' cannot be a sensitive column")
        return self

    @model_validator(mode="after")
    def validate_data_report_disabled_if_both_model_reports_disabled(self):
        configs = [
            cfg
            for cfg in [
                self.source_table.tabular_model_configuration,
                self.source_table.language_model_configuration,
            ]
            if cfg
        ]

        if all(cfg.enable_model_report is False for cfg in configs):
            if self.synthetic_table.configuration is not None:
                self.synthetic_table.configuration.enable_data_report = False
        return self


class _SyntheticDataConfigValidation(CustomBaseModel):
    """
    Validation logic for SyntheticDatasetConfig and SyntheticProbeConfig against Generator
    """

    synthetic_config: SyntheticDatasetConfig | SyntheticProbeConfig
    generator: Generator

    @model_validator(mode="after")
    def add_missing_tables(self):
        generator_table_map = {t.name: t for t in self.generator.tables}
        if self.synthetic_config.tables is None:
            self.synthetic_config.tables = []
        synthetic_table_map = {t.name: t for t in self.synthetic_config.tables}

        missing_tables = set(generator_table_map.keys()) - set(synthetic_table_map.keys())
        for t in missing_tables:
            self.synthetic_config.tables.append(SyntheticTableConfig(name=t))
        return self

    @model_validator(mode="after")
    def validate_no_extra_tables(self):
        generator_table_map = {t.name: t for t in self.generator.tables}
        synthetic_table_map = {t.name: t for t in self.synthetic_config.tables or []}

        generator_tables = set(generator_table_map.keys())
        extra_tables = set(synthetic_table_map.keys()) - generator_tables
        if extra_tables:
            raise ValueError(
                f"Tables {extra_tables} are not present in the generator. Only {generator_tables} are available."
            )
        return self

    @model_validator(mode="after")
    def validate_tables(self):
        generator_table_map = {t.name: t for t in self.generator.tables}
        synthetic_table_map = {t.name: t for t in self.synthetic_config.tables or []}

        for table_name, synthetic_table in synthetic_table_map.items():
            generator_table = generator_table_map[table_name]
            synthetic_table.validate_against_source_table(
                generator_table, is_probe=isinstance(self.synthetic_config, SyntheticProbeConfig)
            )
        return self


class ProgressStep:
    @model_validator(mode="after")
    def add_required_fields(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
        if self.status is None:
            self.status = ProgressStatus.new
        if self.compute_name is None:
            self.compute_name = "SDK"
        return self


class RebalancingConfig:
    @field_validator("probabilities", mode="after")
    def validate_probabilities(cls, v):
        if not all(0 <= v <= 1 for v in v.values()):
            raise ValueError("the probabilities must be between 0 and 1")
        if not sum(v.values()) <= 1:
            raise ValueError("the sum of probabilities must be less than or equal to 1")
        return v


class ConnectorListItem:
    OPEN_URL_PARTS: ClassVar[list] = ["d", "connectors"]

    def __getattr__(self, item):
        if item in {"update", "delete", "locations", "schema"}:

            def delegated_method(*args, **kwargs):
                connector = self.client.get(self.id)
                result = getattr(connector, item)(*args, **kwargs)
                if item == "update":
                    self.reload()
                return result

            return delegated_method
        return object.__getattribute__(self, item)


class GeneratorListItem:
    OPEN_URL_PARTS: ClassVar[list] = ["d", "generators"]

    def __getattr__(self, item):
        if item in {"update", "delete", "clone", "config", "export_to_file"}:

            def delegated_method(*args, **kwargs):
                generator = Generator(id=self.id, client=self.client)
                result = getattr(generator, item)(*args, **kwargs)
                if item == "update":
                    self.reload()
                return result

            return delegated_method
        return object.__getattribute__(self, item)


class SyntheticDatasetListItem:
    OPEN_URL_PARTS: ClassVar[list] = ["d", "synthetic-datasets"]

    def __getattr__(self, item):
        if item in {"update", "delete", "config", "download", "data"}:

            def delegated_method(*args, **kwargs):
                sd = self.client.get(self.id)
                result = getattr(sd, item)(*args, **kwargs)
                if item == "update":
                    self.reload()
                return result

            return delegated_method
        return object.__getattribute__(self, item)


class Dataset:
    OPEN_URL_PARTS: ClassVar[list] = ["d", "datasets"]

    @model_validator(mode="before")
    @classmethod
    def add_required_fields(cls, values):
        if isinstance(values, dict):
            if "id" not in values:
                values["id"] = str(uuid.uuid4())
            if values.get("name") is None:
                values["name"] = "New dataset"
        return values

    @field_validator("files", mode="after")
    @classmethod
    def initialize_file_list(cls, values):
        if values is None:
            values = []
        return values

    def update(
        self,
        name: str | None = None,
        description: str | None = None,
        connectors: list[DatasetConnector] | None = None,
    ) -> None:
        """
        Update a dataset with specific parameters.

        Args:
            name (str | None): The name of the connector.
            description (str | None): The description of the connector.
            connectors (list[DatasetConnector] | None): The connectors of the dataset.
        """
        patch_config = DatasetPatchConfig(
            name=name,
            description=description,
            connectors=connectors,
        )
        self.client._update(
            dataset_id=self.id,
            config=patch_config,
        )
        self.reload()

    def delete(self) -> None:
        """
        Delete the dataset.
        """
        return self.client._delete(dataset_id=self.id)

    def download_file(
        self,
        dataset_file_path: str | Path,
        output_file_path: str | Path | None = None,
    ) -> Path:
        """
        Download the dataset file.

        Args:
            file_path (str | Path | None): The file path to save the dataset file.

        Returns:
            Path: The path to the saved file.
        """
        bytes, filename = self.client._download_file(dataset_id=self.id, file_path=str(dataset_file_path))
        output_file_path = Path(output_file_path or ".")
        if output_file_path.is_dir():
            output_file_path = output_file_path / filename
        output_file_path.write_bytes(bytes)
        return output_file_path

    def upload_file(
        self,
        file_path: str | Path,
    ) -> None:
        """
        Upload the dataset file.
        """
        self.client._upload_file(dataset_id=self.id, file_path=str(file_path))
        self.reload()

    def delete_file(
        self,
        file_path: str | Path,
    ) -> None:
        """
        Delete the dataset file.
        """
        self.client._delete_file(dataset_id=self.id, file_path=str(file_path))
        self.reload()


class Artifact:
    OPEN_URL_PARTS: ClassVar[list] = ["d", "artifacts"]

    @model_validator(mode="before")
    @classmethod
    def add_required_fields(cls, values):
        if isinstance(values, dict):
            if "id" not in values:
                values["id"] = str(uuid.uuid4())
        return values

    def update(
        self,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        """
        Update the artifact.

        Args:
            name (str | None): The name of the artifact.
            description (str | None): The description of the artifact.
        """
        patch_config = ArtifactPatchConfig(
            name=name,
            description=description,
        )
        self.client._update(artifact_id=self.id, config=patch_config)
        self.reload()
