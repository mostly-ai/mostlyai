# Copyright 2024 MOSTLY AI
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

from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal

import pandas as pd
from pydantic import Field, field_validator

from mostlyai.client._base_utils import convert_to_base64
from mostlyai.domain import (
    JobProgress,
    SyntheticDatasetFormat,
    ConnectorPatchConfig,
    GeneratorPatchConfig,
    SyntheticDatasetDelivery,
    SyntheticDatasetPatchConfig,
    SyntheticDatasetConfig,
    GeneratorConfig,
)


class Connector:
    OPEN_URL_PARTS: ClassVar[list] = ["d", "connectors"]

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
        self.client._update(
            generator_id=self.id, config=patch_config.model_dump(exclude_none=True)
        )
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

    class Training:
        def __init__(self, _generator: "Generator"):
            self.generator = _generator

        def start(self) -> None:
            """
            Start training.
            """
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
            self.generator.client._training_wait(
                self.generator.id, progress_bar=progress_bar, interval=interval
            )
            self.generator.reload()


class SourceTableConfig:
    @field_validator("data", mode="before")
    @classmethod
    def validate_data_before(cls, value):
        return convert_to_base64(value) if isinstance(value, pd.DataFrame) else value


class SyntheticTableConfiguration:
    @field_validator("sample_seed_dict", mode="before")
    @classmethod
    def validate_dict_before(cls, value):
        return (
            convert_to_base64(value, format="jsonl")
            if isinstance(value, dict)
            else value
        )

    @field_validator("sample_seed_data", mode="before")
    @classmethod
    def validate_data_before(cls, value):
        return convert_to_base64(value) if isinstance(value, pd.DataFrame) else value


class SyntheticDataset:
    OPEN_URL_PARTS: ClassVar[list] = ["d", "synthetic-datasets"]
    generation: Annotated[Any | None, Field(exclude=True)] = None

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

    def config(self) -> SyntheticDatasetConfig:
        """
        Retrieve writable synthetic dataset properties.

        Returns:
            SyntheticDatasetConfig: The synthetic dataset properties as a configuration object.
        """
        return self.client._config(synthetic_dataset_id=self.id)

    def download(
        self,
        format: SyntheticDatasetFormat = "PARQUET",
        file_path: str | Path | None = None,
    ) -> Path:
        """
        Download synthetic dataset and save to file.

        Args:
            format: The format of the synthetic dataset.
            file_path: The file path to save the synthetic dataset.

        Returns:
            The path to the saved file.
        """
        bytes, filename = self.client._download(
            synthetic_dataset_id=self.id,
            ds_format=format,
            short_lived_file_token=self.metadata.short_lived_file_token,
        )
        file_path = Path(file_path or ".")
        if file_path.is_dir():
            file_path = file_path / filename
        file_path.write_bytes(bytes)
        return file_path

    def data(
        self, return_type: Literal["auto", "dict"] = "auto"
    ) -> pd.DataFrame | dict[str, pd.DataFrame]:
        """
        Download synthetic dataset and return as dictionary of pandas DataFrames.

        Args:
            return_type (Literal["auto", "dict"]): The format of the returned data.

        Returns:
            Union[pd.DataFrame, dict[str, pd.DataFrame]]: The synthetic dataset as a dictionary of pandas DataFrames.
        """
        dfs = self.client._data(
            synthetic_dataset_id=self.id,
            short_lived_file_token=self.metadata.short_lived_file_token,
        )
        if return_type == "auto" and len(dfs) == 1:
            return list(dfs.values())[0]
        else:
            return dfs

    class Generation:
        def __init__(self, _synthetic_dataset: "SyntheticDataset"):
            self.synthetic_dataset = _synthetic_dataset

        def start(self) -> None:
            """
            Start the generation process.
            """
            self.synthetic_dataset.client._generation_start(self.synthetic_dataset.id)

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
            return self.synthetic_dataset.client._generation_progress(
                self.synthetic_dataset.id
            )

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
