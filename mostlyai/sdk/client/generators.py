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

import os
import re
import tempfile
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
import rich

from mostlyai.sdk.client._base_utils import convert_to_base64, read_table_from_path
from mostlyai.sdk.client._utils import job_wait
from mostlyai.sdk.client.base import (
    DELETE,
    GET,
    PATCH,
    POST,
    Paginator,
    _MostlyBaseClient,
)
from mostlyai.sdk.domain import (
    Generator,
    GeneratorCloneConfig,
    GeneratorConfig,
    GeneratorListItem,
    GeneratorPatchConfig,
    JobProgress,
    ModelType,
)


class _MostlyGeneratorsClient(_MostlyBaseClient):
    SECTION = ["generators"]

    # PUBLIC METHODS #

    def list(
        self,
        offset: int = 0,
        limit: int | None = None,
        status: str | list[str] | None = None,
        search_term: str | None = None,
        owner_id: str | list[str] | None = None,
        visibility: str | list[str] | None = None,
        created_from: str | None = None,
        created_to: str | None = None,
        sort_by: str | list[str] | None = None,
    ) -> Iterator[GeneratorListItem]:
        """
        List generators.

        Paginate through all generators accessible by the user.

        Args:
            offset: Offset for the entities in the response.
            limit: Limit for the number of entities in the response.
            status: Filter by training status.
            search_term: Filter by name or description.
            owner_id: Filter by owner ID.
            visibility: Filter by visibility (e.g., PUBLIC, PRIVATE or UNLISTED).
            created_from: Filter by creation date, not older than this date. Format: YYYY-MM-DD.
            created_to: Filter by creation date, not younger than this date. Format: YYYY-MM-DD.
            sort_by: Sort by field. Either NO_OF_THREADS, NO_OF_LIKES, RECENCY, or NO_OF_SYNTHETIC_DATASETS.

        Returns:
            Iterator[GeneratorListItem]: An iterator over generator list items.

        Example for listing all generators:
            ```python
            from mostlyai.sdk import MostlyAI
            mostly = MostlyAI()
            for g in mostly.generators.list():
                print(f"Generator `{g.name}` ({g.training_status}, {g.id})")
            ```

        Example for searching trained generators via key word:
            ```python
            from mostlyai.sdk import MostlyAI
            mostly = MostlyAI()
            generators = list(mostly.generators.list(search_term="census", status="DONE"))
            print(f"Found {len(generators)} generators")
            ```
        """
        status = ",".join(status) if isinstance(status, list) else status
        with Paginator(
            self,
            GeneratorListItem,
            offset=offset,
            limit=limit,
            status=status,
            search_term=search_term,
            owner_id=owner_id,
            visibility=visibility,
            created_from=created_from,
            created_to=created_to,
            sort_by=sort_by,
        ) as paginator:
            yield from paginator

    def get(self, generator_id: str) -> Generator:
        """
        Retrieve a generator by its ID.

        Args:
            generator_id: The unique identifier of the generator.

        Returns:
            Generator: The retrieved generator object.

        Example for retrieving a generator:
            ```python
            from mostlyai.sdk import MostlyAI
            mostly = MostlyAI()
            g = mostly.generators.get('INSERT_YOUR_GENERATOR_ID')
            g
            ```
        """
        if not isinstance(generator_id, str) or len(generator_id) != 36:
            raise ValueError("The provided generator_id must be a UUID string")
        response = self.request(verb=GET, path=[generator_id], response_type=Generator)
        return response

    def create(self, config: GeneratorConfig | dict) -> Generator:
        """
        Create a generator. The generator will be in the NEW state and will need to be trained before it can be used.

        See [`mostly.train`](api_client.md#mostlyai.sdk.client.api.MostlyAI.train) for more details.

        Args:
            config: Configuration for the generator.

        Returns:
            The created generator object.

        Example for creating a generator:
            ```python
            from mostlyai.sdk import MostlyAI
            mostly = MostlyAI()
            g = mostly.generators.create(
                config={
                    "name": "US Census",
                    "tables": [{
                        "name": "census",
                        "data": trn_df,
                    }]
                )
            )
            print("status:", g.training_status)
            # status: NEW
            g.training.start()  # start training
            print("status:", g.training_status)
            # status: QUEUED
            g.training.wait()   # wait for training to complete
            print("status:", g.training_status)
            # status: DONE
            ```
        """
        if isinstance(config, dict):
            for table in config.get("tables", []):
                # convert `data` to base64-encoded Parquet files
                if table.get("data") is not None:
                    if isinstance(table["data"], (str, Path)):
                        name, df = read_table_from_path(table["data"])
                        table["data"] = convert_to_base64(df)
                        if "name" not in table:
                            table["name"] = name
                        del df
                    elif isinstance(table["data"], pd.DataFrame) or (
                        table["data"].__class__.__name__ == "DataFrame"
                        and table["data"].__class__.__module__.startswith("pyspark.sql")
                    ):
                        table["data"] = convert_to_base64(table["data"])
                    else:
                        raise ValueError("data must be a DataFrame or a file path")
                if table.get("columns"):
                    # convert `columns` to list[dict], if provided as list[str]
                    table["columns"] = [{"name": col} if isinstance(col, str) else col for col in table["columns"]]
            config = GeneratorConfig.model_validate(config)
        generator = self.request(verb=POST, path=[], json=config, response_type=Generator)
        gid = generator.id
        if self.local:
            rich.print(f"Created generator [dodger_blue2]{gid}[/]")
        else:
            rich.print(f"Created generator [link={self.base_url}/d/generators/{gid} dodger_blue2 underline]{gid}[/]")
        return generator

    def import_from_file(
        self,
        file_path: str | Path,
    ) -> Generator:
        """
        Import a generator from a file.

        Args:
            file_path: Local file path or URL of the generator to import.

        Returns:
            The imported generator object.

        Example:
            ```python
            from mostlyai.sdk import MostlyAI
            mostly = MostlyAI()

            # Import from local file
            g = mostly.generators.import_from_file('path/to/generator')

            # Or import from URL
            g = mostly.generators.import_from_file('https://example.com/path/to/generator.zip')
            ```
        """
        # check if file_path is a URL
        parsed_url = urlparse(str(file_path))
        is_url = parsed_url.scheme in ["http", "https"]

        temp_file = None
        try:
            if is_url:
                # download the file from URL to a temporary file
                response = requests.get(str(file_path), stream=True)
                response.raise_for_status()  # raise an exception for HTTP errors
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
                temp_file.write(response.content)
                temp_file.close()
                file_path = temp_file.name

            with open(file_path, "rb") as f:
                generator = self.request(
                    verb=POST,
                    path=["import-from-file"],
                    headers={
                        "Accept": "application/json, text/plain, */*",
                    },
                    files={"file": f},
                    response_type=Generator,
                )

            gid = generator.id
            if self.local:
                rich.print(f"Imported generator [dodger_blue2]{gid}[/]")
            else:
                rich.print(
                    f"Imported generator [link={self.base_url}/d/generators/{gid} dodger_blue2 underline]{gid}[/]"
                )
            return generator

        finally:
            # clean up the temporary file if it exists
            if temp_file and os.path.exists(temp_file.name):
                os.unlink(temp_file.name)

    # PRIVATE METHODS #
    def _export_to_file(
        self,
        generator_id: str,
    ) -> tuple[bytes, str | None]:
        response = self.request(
            verb=GET,
            path=[generator_id, "export-to-file"],
            headers={
                "Content-Type": "application/octet-stream",
                "Accept": "application/json, text/plain, */*",
            },
            raw_response=True,
        )
        content_bytes = response.content
        # Check if 'Content-Disposition' header is present
        if "Content-Disposition" in response.headers:
            content_disposition = response.headers["Content-Disposition"]
            filename = re.findall(r"filename(?:=|\*=UTF-8'')(.+)", content_disposition)[0]
        else:
            filename = f"generator-{generator_id[:8]}.zip"
        return content_bytes, filename

    def _update(self, generator_id: str, config: GeneratorPatchConfig) -> Generator:
        response = self.request(
            verb=PATCH,
            path=[generator_id],
            json=config,
            exclude_none_in_json=True,
            response_type=Generator,
        )
        return response

    def _delete(self, generator_id: str) -> None:
        response = self.request(verb=DELETE, path=[generator_id])
        return response

    def _clone(self, generator_id: str, training_status: str) -> Generator:
        response = self.request(
            verb=POST,
            path=[generator_id, "clone"],
            json=GeneratorCloneConfig(training_status=training_status.upper()),
            response_type=Generator,
        )
        return response

    def _config(self, generator_id: str) -> GeneratorConfig:
        response = self.request(verb=GET, path=[generator_id, "config"], response_type=GeneratorConfig)
        return response

    def _report(
        self,
        generator_id: str,
        source_table_id: str,
        model_type: ModelType = ModelType.tabular,
        short_lived_file_token: str | None = None,
    ) -> tuple[str, str | None]:
        response = self.request(
            verb=GET,
            path=[generator_id, "tables", source_table_id, "report"],
            params={
                "modelType": model_type.upper() if isinstance(model_type, str) else model_type.value,
                "slft": short_lived_file_token,
            },
            headers={
                "Accept": "text/html, text/plain, */*",
            },
            raw_response=True,
        )
        return response.text

    def _training_start(self, generator_id: str) -> None:
        self.request(verb=POST, path=[generator_id, "training", "start"])

    def _training_cancel(self, generator_id: str) -> None:
        self.request(verb=POST, path=[generator_id, "training", "cancel"])

    def _training_progress(self, generator_id: str) -> JobProgress:
        response = self.request(verb=GET, path=[generator_id, "training"], response_type=JobProgress)
        return response

    def _training_wait(self, generator_id: str, progress_bar: bool, interval: float) -> Generator:
        job_wait(lambda: self._training_progress(generator_id), interval, progress_bar)
        generator = self.get(generator_id)
        return generator

    def _training_logs(self, generator_id: str, short_lived_file_token: str | None = None) -> tuple[bytes, str]:
        response = self.request(
            verb=GET,
            path=[generator_id, "training", "logs"],
            params={
                "slft": short_lived_file_token,
            },
            headers={
                "Content-Type": "application/zip",
                "Accept": "application/json, text/plain, */*",
            },
            raw_response=True,
        )
        content_bytes = response.content
        filename = f"generator-{generator_id[:8]}-logs.zip"
        return content_bytes, filename
