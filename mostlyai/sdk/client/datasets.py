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

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import rich

from mostlyai.sdk.client.base import (
    DELETE,
    GET,
    PATCH,
    POST,
    Paginator,
    _MostlyBaseClient,
)
from mostlyai.sdk.domain import (
    Dataset,
    DatasetConfig,
    DatasetListItem,
    DatasetPatchConfig,
)


class _MostlyDatasetsClient(_MostlyBaseClient):
    SECTION = ["datasets"]

    # PUBLIC METHODS #

    def list(
        self,
        offset: int = 0,
        limit: int | None = None,
        search_term: str | None = None,
        owner_id: str | list[str] | None = None,
        visibility: str | list[str] | None = None,
        created_from: str | None = None,
        created_to: str | None = None,
        sort_by: str | list[str] | None = None,
    ) -> Iterator[DatasetListItem]:
        """
        List datasets.

        Args:
            offset: Offset for the entities in the response.
            limit: Limit for the number of entities in the response.
            status: Filter by generation status.
            search_term: Filter by name or description.
            owner_id: Filter by owner ID.
            visibility: Filter by visibility (e.g., PUBLIC, PRIVATE or UNLISTED).
            created_from: Filter by creation date, not older than this date. Format: YYYY-MM-DD.
            created_to: Filter by creation date, not younger than this date. Format: YYYY-MM-DD.
            sort_by: Sort by field. Either NO_OF_THREADS, NO_OF_LIKES, or RECENCY.

        Returns:
            An iterator over datasets.

        Example for listing all datasets:
            ```python
            from mostlyai.sdk import MostlyAI
            mostly = MostlyAI()
            for ds in mostly.datasets.list():
                print(f"Dataset `{ds.name}` ({ds.id})")
            ```

        Example for searching datasets via key word:
            ```python
            from mostlyai.sdk import MostlyAI
            mostly = MostlyAI()
            datasets = list(mostly.datasets.list(search_term="census"))
            print(f"Found {len(datasets)} datasets")
        """
        with Paginator(
            self,
            DatasetListItem,
            offset=offset,
            limit=limit,
            search_term=search_term,
            owner_id=owner_id,
            visibility=visibility,
            created_from=created_from,
            created_to=created_to,
            sort_by=sort_by,
        ) as paginator:
            yield from paginator

    def get(self, dataset_id: str) -> Dataset:
        """
        Retrieve a dataset by its ID.

        Args:
            dataset_id: The unique identifier of the dataset.

        Returns:
            Dataset: The retrieved dataset object.

        Example for retrieving a dataset:
            ```python
            from mostlyai.sdk import MostlyAI
            mostly = MostlyAI()
            ds = mostly.datasets.get('INSERT_YOUR_DATASET_ID')
            ds
            ```
        """
        if not isinstance(dataset_id, str) or len(dataset_id) != 36:
            raise ValueError("The provided dataset_id must be a UUID string")
        response = self.request(verb=GET, path=[dataset_id], response_type=Dataset)
        return response

    def create(self, config: DatasetConfig | dict[str, Any]) -> Dataset:
        """
        Create a dataset with optional connectors and files.

        Args:
            config: Configuration for the dataset.

        Returns:
            The created dataset object.

        Example for creating a dataset with a connector:
            ```python
            from mostlyai.sdk import MostlyAI
            mostly = MostlyAI()
            ds = mostly.datasets.create(
                config=DatasetConfig(
                    name="INSERT_YOUR_DATASET_NAME",
                    description="INSERT_YOUR_DATASET_INSTRUCTIONS",
                    connectors=[
                        DatasetConnector(
                            connector_id="INSERT_YOUR_CONNECTOR_ID",
                            locations=["LOCATION_1", "LOCATION_2"],
                        )
                    ],
                )
            )
            ```

        Example for creating a dataset with files:
            ```python
            from mostlyai.sdk import MostlyAI
            mostly = MostlyAI()
            ds = mostly.datasets.create(
                config=DatasetConfig(
                    name="INSERT_YOUR_DATASET_NAME",
                    description="INSERT_YOUR_DATASET_INSTRUCTIONS",
                )
            )
            ds.upload_file("path/to/file_1.csv.gz")
            ds.upload_file("path/to/file_2.txt")
            ```
        """
        config = DatasetConfig.model_validate(config)
        dataset = self.request(
            verb=POST,
            path=[],
            json=config,
            response_type=Dataset,
        )
        dsid = dataset.id
        if self.local:
            rich.print(f"Created dataset [dodger_blue2]{dsid}[/]")
        else:
            rich.print(f"Created dataset [link={self.base_url}/d/datasets/{dsid} dodger_blue2 underline]{dsid}[/]")
        return dataset

    def _update(
        self,
        dataset_id: str,
        config: DatasetPatchConfig,
    ) -> Dataset:
        response = self.request(
            verb=PATCH,
            path=[dataset_id],
            json=config,
            exclude_none_in_json=True,
            response_type=Dataset,
        )
        return response

    def _delete(self, dataset_id: str) -> None:
        response = self.request(verb=DELETE, path=[dataset_id])
        return response

    def _config(self, dataset_id: str) -> DatasetConfig:
        response = self.request(verb=GET, path=[dataset_id, "config"], response_type=DatasetConfig)
        return response

    def _download_file(
        self,
        dataset_id: str,
        file_path: str,
    ) -> tuple[bytes, str | None]:
        response = self.request(
            verb=GET,
            path=[dataset_id, "file"],
            params={"filepath": file_path},
            headers={
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
            filename = Path(file_path).name
        return content_bytes, filename

    def _upload_file(
        self,
        dataset_id: str,
        file_path: str | Path,
    ) -> None:
        with open(file_path, "rb") as f:
            _ = self.request(
                verb=POST,
                path=[dataset_id, "file"],
                headers={
                    "Accept": "application/json, text/plain, */*",
                },
                files={"file": f},
            )
            rich.print(f"Uploaded file `{file_path}` to Dataset [dodger_blue2]{dataset_id}[/]")

    def _delete_file(
        self,
        dataset_id: str,
        file_path: str | Path,
    ) -> None:
        _ = self.request(
            verb=DELETE,
            path=[dataset_id, "file"],
            params={
                "filepath": file_path,
            },
        )
        rich.print(f"Deleted file `{file_path}` from Dataset [dodger_blue2]{dataset_id}[/]")
