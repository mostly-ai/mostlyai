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

from mostlyai.sdk.client._base_utils import convert_to_base64
from mostlyai.sdk.client.base import CustomBaseModel
from mostlyai.sdk.domain import (
    JobProgress,
    SyntheticDatasetFormat,
    ConnectorPatchConfig,
    GeneratorPatchConfig,
    SyntheticDatasetDelivery,
    SyntheticDatasetPatchConfig,
    GeneratorConfig,
    ModelEncodingType,
    ProgressStatus,
    ModelConfiguration,
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
        return values

    def update(
        self,
        name: str | None = None,
        config: dict[str, Any] | None = None,
        secrets: dict[str, str] | None = None,
        ssl: dict[str, str] | None = None,
        test_connection: bool | None = True,
    ) -> None:
        """
        Update a connector with specific parameters.

        Args:
            name: The name of the connector.
            config (dict[str, Any], optional): Connector configuration.
            secrets (dict[str, str], optional): Secret values for the connector.
            ssl (dict[str, str], optional): SSL configuration for the connector.
            test_connection: If true, validates the connection before saving.
        """
        patch_config = ConnectorPatchConfig(
            name=name,
            config=config,
            secrets=secrets,
            ssl=ssl,
        )
        self.client._update(
            connector_id=self.id,
            config=patch_config.model_dump(exclude_none=True),
            test_connection=test_connection,
        )
        self.reload()

    def delete(self) -> None:
        """
        Delete the connector.

        Returns:
            None
        """
        return self.client._delete(connector_id=self.id)

    def locations(self, prefix: str = "") -> list:
        """
        List connector locations.

        List the available databases, schemas, tables, or folders for a connector.
        For storage connectors, this returns list of folders and files at root, respectively at `prefix` level.
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
            prefix: The prefix to filter the results by.

        Returns:
            list: A list of locations (schemas, databases, directories, etc.)."""
        return self.client._locations(connector_id=self.id, prefix=prefix)

    def schema(self, location: str) -> list[dict[str, Any]]:
        """
        Retrieve the schema of the table at a connector location.
        Please refer to `locations()` for the format of the location.

        Args:
            location: The location of the table.

        Returns:
            list[dict[str, Any]]: The retrieved schema.
        """
        return self.client._schema(connector_id=self.id, location=location)


class Generator:
    OPEN_URL_PARTS: ClassVar[list] = ["d", "generators"]
    training: Annotated[Any | None, Field(exclude=True)] = None

    @model_validator(mode="before")
    @classmethod
    def add_required_fields(cls, values):
        if isinstance(values, dict):
            if "id" not in values:
                values["id"] = str(uuid.uuid4())
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
            name: The name of the generator.
            description: The description of the generator.
        """
        patch_config = GeneratorPatchConfig(
            name=name,
            description=description,
        )
        self.client._update(generator_id=self.id, config=patch_config.model_dump(exclude_none=True))
        self.reload()

    def delete(self) -> None:
        """
        Delete the generator.

        Returns:
            None
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
            file_path: The file path to save the generator.

        Returns:
            The path to the saved file.
        """
        bytes, filename = self.client._export_to_file(generator_id=self.id)
        file_path = Path(file_path or ".")
        if file_path.is_dir():
            file_path = file_path / filename
        file_path.write_bytes(bytes)
        return file_path

    def clone(self, training_status: Literal["NEW", "CONTINUE"] = "NEW") -> "Generator":
        """
        Clone the generator.

        Args:
            training_status (Literal["NEW", "CONTINUE"]): The training status of the cloned generator.

        Returns:
            Generator: The cloned generator object.
        """
        return self.client._clone(generator_id=self.id, training_status=training_status)

    def reports(self, file_path: str | Path | None = None) -> Path:
        """
        Download the quality assurance reports and save to file.

        Args:
            file_path: The file path to save the zipped reports.

        Returns:
            The path to the saved file.
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

        file_path = Path(file_path or ".")
        if file_path.is_dir():
            file_path = file_path / f"generator-{self.id[:8]}-reports.zip"
        if file_path.exists():
            file_path.unlink()
        with zipfile.ZipFile(file_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for filename, content in reports.items():
                zip_file.writestr(filename, content)
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
                progress_bar: If true, displays the progress bar.
                interval: The interval in seconds to poll the job progress.
            """
            self.generator.client._training_wait(self.generator.id, progress_bar=progress_bar, interval=interval)
            self.generator.reload()
            if self.generator.training_status == ProgressStatus.done:
                rich.print(
                    ":tada: [bold green]Your generator is ready![/] "
                    "Use it to create synthetic data. "
                    "Publish it so others can do the same."
                )


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
        return convert_to_base64(value) if isinstance(value, pd.DataFrame) else value

    @model_validator(mode="after")
    @classmethod
    def add_model_configuration(cls, values):
        # Check if the table has a tabular and/or a language model
        if values.columns:
            keys = [fk.column for fk in values.foreign_keys or []]
            if values.primary_key:
                keys.append(values.primary_key)
            model_columns = [c for c in values.columns if c.name not in keys]
            enc_types = [c.model_encoding_type or ModelEncodingType.auto for c in model_columns]
            has_tabular_model = any(not enc_type.startswith("LANGUAGE_") for enc_type in enc_types)
            has_language_model = any(enc_type.startswith("LANGUAGE_") for enc_type in enc_types)
        else:
            has_tabular_model = True
            has_language_model = False
        if values.foreign_keys:
            # Always train tabular model for linked tables to model sequences
            for fk in values.foreign_keys:
                if fk.is_context:
                    has_tabular_model = True
        # Raise error if model configurations were incorrectly provided
        if values.tabular_model_configuration and not has_tabular_model:
            raise ValueError("Tabular model configuration is not applicable for language models.")
        if values.language_model_configuration and not has_language_model:
            raise ValueError("Language model configuration is not applicable for tabular models.")
        # Add default model configurations if none were provided
        if has_tabular_model and not values.tabular_model_configuration:
            default_model = "MOSTLY_AI/Medium"
            values.tabular_model_configuration = ModelConfiguration(model=default_model)
        if has_language_model and not values.language_model_configuration:
            default_model = "MOSTLY_AI/LSTMFromScratch-3m"
            values.language_model_configuration = ModelConfiguration(model=default_model, max_sequence_window=None)
        if has_language_model and values.language_model_configuration:
            # language models atm do not support max_sequence_window; thus set configuration to None
            values.language_model_configuration.max_sequence_window = None
        return values

    @field_validator("columns", mode="after")
    @classmethod
    def validate_unique_columns(cls, columns):
        if columns:
            defined_columns = [c.name for c in columns]
            if len(defined_columns) != len(set(defined_columns)):
                raise ValueError("Column names must be unique.")
        return columns

    @model_validator(mode="after")
    @classmethod
    def validate_unique_keys(cls, values):
        pk = values.primary_key or []
        fks = [fk.column for fk in values.foreign_keys or []]
        if len(fks) != len(set(fks)):
            raise ValueError("Foreign key column names must be unique.")
        if pk in fks:
            raise ValueError("Primary key column name must not be defined as foreign key.")
        return values

    @field_validator("foreign_keys", mode="after")
    @classmethod
    def validate_at_most_one_context_fk(cls, foreign_keys):
        if foreign_keys:
            context_fks = [fk for fk in foreign_keys if fk.is_context]
            if len(context_fks) > 1:
                raise ValueError("At most one context foreign key is allowed")
        return foreign_keys

    @model_validator(mode="after")
    @classmethod
    def validate_keys_exists_in_columns(cls, values):
        if values.columns:
            column_names = {col.name for col in values.columns}
            pk = values.primary_key
            if pk is not None and pk not in column_names:
                raise ValueError(f"Primary key column '{pk}' does not exist in the table's columns.")
            for fk in values.foreign_keys or []:
                if fk.column not in column_names:
                    raise ValueError(f"Foreign key column '{fk.column}' does not exist in the table's columns.")
        return values


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


class SyntheticTableConfiguration:
    @field_validator("sample_seed_dict", mode="before")
    @classmethod
    def convert_dict_before(cls, value):
        return convert_to_base64(value, format="jsonl") if isinstance(value, dict) else value

    @field_validator("sample_seed_data", mode="before")
    @classmethod
    def convert_data_before(cls, value):
        return convert_to_base64(value) if isinstance(value, pd.DataFrame) else value

    @model_validator(mode="after")
    @classmethod
    def add_required_fields(cls, values):
        if values.sampling_temperature is None:
            values.sampling_temperature = 1.0
        if values.sampling_top_p is None:
            values.sampling_top_p = 1.0
        return values

    # @model_validator(mode="after")
    # @classmethod
    # def mutually_exclusive_fields(cls, values):
    #     seed_fields = [
    #         field
    #         for field in [values.sample_seed_connector_id, values.sample_seed_dict, values.sample_seed_data]
    #         if field is not None
    #     ]
    #     if len(seed_fields) > 1:
    #         raise ValueError(
    #             "Only one of sample_seed_connector_id, sample_seed_dict and sample_seed_data can be provided"
    #         )

    #     if seed_fields and values.sample_size is not None:
    #         raise ValueError(
    #             "sample_seed_connector_id, sample_seed_dict and sample_seed_data are mutually exclusive with sample_size"
    #         )
    #     return values


class SyntheticDataset:
    OPEN_URL_PARTS: ClassVar[list] = ["d", "synthetic-datasets"]
    generation: Annotated[Any | None, Field(exclude=True)] = None

    @model_validator(mode="before")
    @classmethod
    def add_required_fields(cls, values):
        if isinstance(values, dict):
            if "id" not in values:
                values["id"] = str(uuid.uuid4())
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
            name: The name of the synthetic dataset.
            description: The description of the synthetic dataset.
            delivery: The delivery configuration for the synthetic dataset.
        """
        patch_config = SyntheticDatasetPatchConfig(
            name=name,
            description=description,
            delivery=delivery,
        )
        self.client._update(
            synthetic_dataset_id=self.id,
            config=patch_config.model_dump(exclude_none=True),
        )
        self.reload()

    def delete(self) -> None:
        """
        Delete the synthetic dataset.

        Returns:
            None
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
        format: SyntheticDatasetFormat = "PARQUET",
    ) -> Path:
        """
        Download synthetic dataset and save to file.

        Args:
            file_path: The file path to save the synthetic dataset.
            format: The format of the synthetic dataset.

        Returns:
            The path to the saved file.
        """
        bytes, filename = self.client._download(
            synthetic_dataset_id=self.id,
            ds_format=format,
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
            return_type (Literal["auto", "dict"]): The format of the returned data.

        Returns:
            Union[pd.DataFrame, dict[str, pd.DataFrame]]: The synthetic dataset as a dictionary of pandas DataFrames.
        """
        dfs = self.client._data(
            synthetic_dataset_id=self.id,
            short_lived_file_token=self.metadata.short_lived_file_token if self.metadata else None,
        )
        if return_type == "auto" and len(dfs) == 1:
            return list(dfs.values())[0]
        else:
            return dfs

    def reports(self, file_path: str | Path | None = None) -> Path:
        """
        Download the quality assurance reports and save to file.

        Args:
            file_path: The file path to save the zipped reports.

        Returns:
            The path to the saved file.
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

        file_path = Path(file_path or ".")
        if file_path.is_dir():
            file_path = file_path / f"synthetic-dataset-{self.id[:8]}-reports.zip"
        if file_path.exists():
            file_path.unlink()
        with zipfile.ZipFile(file_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for filename, content in reports.items():
                zip_file.writestr(filename, content)
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
                progress_bar: If true, displays a progress bar.
                interval: Interval in seconds to poll the job progress.
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
    @classmethod
    def add_configuration(cls, values):
        if values.configuration is None:
            values.configuration = SyntheticTableConfiguration()
        return values

    def validate_against_source_table(self, source_table: SourceTable) -> None:
        _SyntheticTableConfigValidation(synthetic_table=self, source_table=source_table)


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
    def validate_rebalancing_config(cls, validation):
        config = validation.synthetic_table.configuration
        if config and config.rebalancing:
            has_tabular_model = validation.source_table.tabular_model_configuration is not None
            if not has_tabular_model:
                raise ValueError(
                    f"Table '{validation.source_table.name}' specifies rebalancing but has no tabular model"
                )

            rebalancing_column = config.rebalancing.column
            rebalancing_col = next(
                (col for col in validation.source_table.columns or [] if col.name == rebalancing_column),
                None,
            )
            if not rebalancing_col:
                raise ValueError(
                    f"Rebalancing column '{rebalancing_column}' not found in table '{validation.source_table.name}'"
                )
            if not rebalancing_col.model_encoding_type == ModelEncodingType.tabular_categorical:
                raise ValueError(
                    f"Rebalancing column '{rebalancing_column}' in table '{validation.source_table.name}' must be categorical"
                )
            for value in config.rebalancing.probabilities.keys():
                if value not in rebalancing_col.value_range.values:
                    raise ValueError(
                        f"Rebalancing value '{value}' not found in table '{validation.source_table.name}' column '{rebalancing_column}'"
                    )
        return validation

    @model_validator(mode="after")
    def validate_imputation_config(cls, validation):
        config = validation.synthetic_table.configuration
        if config and config.imputation:
            has_tabular_model = validation.source_table.tabular_model_configuration is not None
            if not has_tabular_model:
                raise ValueError(
                    f"Table '{validation.source_table.name}' specifies imputation but has no tabular model"
                )

            for col in config.imputation.columns:
                if not any(gcol.name == col for gcol in validation.source_table.columns or []):
                    raise ValueError(f"Imputation column '{col}' not found in table '{validation.source_table.name}'")
        return validation

    @model_validator(mode="after")
    def validate_fairness_config(cls, validation):
        config = validation.synthetic_table.configuration
        if config and config.fairness:
            has_tabular_model = validation.source_table.tabular_model_configuration is not None
            if not has_tabular_model:
                raise ValueError(f"Table '{validation.source_table.name}' specifies fairness but has no tabular model")

            target_col = config.fairness.target_column
            if not any(col.name == target_col for col in validation.source_table.columns or []):
                raise ValueError(
                    f"Fairness target column '{target_col}' not found in table '{validation.source_table.name}'"
                )

            for col in config.fairness.sensitive_columns:
                if not any(gcol.name == col for gcol in validation.source_table.columns or []):
                    raise ValueError(
                        f"Fairness sensitive column '{col}' not found in table '{validation.source_table.name}'"
                    )

            if target_col in config.fairness.sensitive_columns:
                raise ValueError(f"Target column '{target_col}' cannot be a sensitive column")
        return validation


class _SyntheticDataConfigValidation(CustomBaseModel):
    """
    Validation logic for SyntheticDatasetConfig and SyntheticProbeConfig against Generator
    """

    synthetic_config: SyntheticDatasetConfig | SyntheticProbeConfig
    generator: Generator

    @model_validator(mode="after")
    def add_missing_tables(cls, validation):
        generator_table_map = {t.name: t for t in validation.generator.tables}
        if validation.synthetic_config.tables is None:
            validation.synthetic_config.tables = []
        synthetic_table_map = {t.name: t for t in validation.synthetic_config.tables}

        missing_tables = set(generator_table_map.keys()) - set(synthetic_table_map.keys())
        for t in missing_tables:
            validation.synthetic_config.tables.append(SyntheticTableConfig(name=t))
        return validation

    @model_validator(mode="after")
    def validate_no_extra_tables(cls, validation):
        generator_table_map = {t.name: t for t in validation.generator.tables}
        synthetic_table_map = {t.name: t for t in validation.synthetic_config.tables or []}

        extra_tables = set(synthetic_table_map.keys()) - set(generator_table_map.keys())
        if extra_tables:
            raise ValueError(f"Extra tables in synthetic config not present in generator: {extra_tables}")
        return validation

    @model_validator(mode="after")
    def validate_tables(cls, validation):
        generator_table_map = {t.name: t for t in validation.generator.tables}
        synthetic_table_map = {t.name: t for t in validation.synthetic_config.tables or []}

        for table_name, synthetic_table in synthetic_table_map.items():
            generator_table = generator_table_map[table_name]
            synthetic_table.validate_against_source_table(generator_table)
        return validation

    @model_validator(mode="after")
    def set_default_sample_sizes(cls, validation):
        generator_table_map = {t.name: t for t in validation.generator.tables}
        synthetic_table_map = {t.name: t for t in validation.synthetic_config.tables or []}

        for table_name, synthetic_table in synthetic_table_map.items():
            generator_table = generator_table_map[table_name]
            config = synthetic_table.configuration

            if config:
                is_subject = not any(fk.is_context for fk in generator_table.foreign_keys or [])
                if (
                    not config.sample_size
                    and is_subject
                    and not (config.sample_seed_connector_id or config.sample_seed_dict or config.sample_seed_data)
                ):
                    config.sample_size = (
                        1
                        if isinstance(validation.synthetic_config, SyntheticProbeConfig)
                        else generator_table.total_rows
                    )
                elif not is_subject:
                    config.sample_size = None
        return validation


class ProgressStep:
    @model_validator(mode="after")
    @classmethod
    def add_required_fields(cls, values):
        if values.id is None:
            values.id = str(uuid.uuid4())
        if values.status is None:
            values.status = ProgressStatus.new
        if values.compute_name is None:
            values.compute_name = "SDK"
        return values


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
