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

import io
import re
import zipfile
from typing import Any
from collections.abc import Iterator

import pandas as pd
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
    JobProgress,
    SyntheticDataset,
    SyntheticDatasetFormat,
    SyntheticDatasetListItem,
    SyntheticDatasetConfig,
    SyntheticProbeConfig,
    SyntheticDatasetPatchConfig,
    SyntheticDatasetReportType,
    ModelType,
)
from mostlyai.sdk.client._utils import job_wait


class _MostlySyntheticDatasetsClient(_MostlyBaseClient):
    SECTION = ["synthetic-datasets"]

    # PUBLIC METHODS #

    def list(
        self,
        offset: int = 0,
        limit: int | None = None,
        status: str | list[str] | None = None,
        search_term: str | None = None,
    ) -> Iterator[SyntheticDatasetListItem]:
        """
        List synthetic datasets.

        Paginate through all synthetic datasets accessible by the user.

        Example for listing all synthetic datasets:
            ```python
            from mostlyai.sdk import MostlyAI
            mostly = MostlyAI()
            for sd in mostly.synthetic_datasets.list():
                print(f"Synthetic Dataset `{sd.name}` ({sd.generation_status}, {sd.id})")
            ```

        Example for searching generated synthetic datasets via key word:
            ```python
            from mostlyai.sdk import MostlyAI
            mostly = MostlyAI()
            datasets = list(mostly.synthetic_datasets.list(search_term="census", status="DONE"))
            print(f"Found {len(datasets)} synthetic datasets")
            ```

        Args:
            offset: Offset for the entities in the response.
            limit: Limit for the number of entities in the response.
            status: Filter by generation status.
            search_term: Filter by name or description.

        Returns:
            An iterator over synthetic datasets.
        """
        status = ",".join(status) if isinstance(status, list) else status
        with Paginator(
            self,
            SyntheticDatasetListItem,
            offset=offset,
            limit=limit,
            status=status,
            search_term=search_term,
        ) as paginator:
            yield from paginator

    def get(self, synthetic_dataset_id: str) -> SyntheticDataset:
        """
        Retrieve a synthetic dataset by its ID.

        Example for retrieving a synthetic dataset:
            ```python
            from mostlyai.sdk import MostlyAI
            mostly = MostlyAI()
            sd = mostly.synthetic_datasets.get('INSERT_YOUR_SYNTHETIC_DATASET_ID')
            sd
            ```

        Args:
            synthetic_dataset_id: The unique identifier of the synthetic dataset.

        Returns:
            SyntheticDataset: The retrieved synthetic dataset object.
        """
        if not isinstance(synthetic_dataset_id, str) or len(synthetic_dataset_id) != 36:
            raise ValueError("The provided synthetic_dataset_id must be a UUID string")
        response = self.request(verb=GET, path=[synthetic_dataset_id], response_type=SyntheticDataset)
        return response

    def create(self, config: SyntheticDatasetConfig | dict[str, Any]) -> SyntheticDataset:
        """
        Create a synthetic dataset. The synthetic dataset will be in the NEW state and will need to be generated before it can be used.

        See [`mostly.generate`](api_client.md#mostlyai.sdk.client.api.MostlyAI.generate) for more details.

        Example for creating a synthetic dataset:
            ```python
            from mostlyai.sdk import MostlyAI
            mostly = MostlyAI()
            sd = mostly.synthetic_datasets.create(
                config=SyntheticDatasetConfig(
                    generator_id="INSERT_YOUR_GENERATOR_ID",
                )
            )
            print("status:", sd.generation_status)
            # status: NEW
            sd.generation.start()  # start generation
            print("status:", sd.generation_status)
            # status: QUEUED
            sd.generation.wait()   # wait for generation to complete
            print("status:", sd.generation_status)
            # status: DONE
            ```

        Args:
            config: Configuration for the synthetic dataset.

        Returns:
            The created synthetic dataset object.
        """
        synthetic_dataset = self.request(
            verb=POST,
            path=[],
            json=config,
            response_type=SyntheticDataset,
        )
        sid = synthetic_dataset.id
        gid = synthetic_dataset.generator_id
        rich.print(
            f"Created synthetic dataset [link={self.base_url}/d/synthetic-datasets/{sid} blue underline]{sid}[/] with generator [link={self.base_url}/d/generators/{gid} blue underline]{gid}[/]"
        )
        return synthetic_dataset

    # PRIVATE METHODS #

    def _update(
        self,
        synthetic_dataset_id: str,
        config: SyntheticDatasetPatchConfig | dict[str, Any],
    ) -> SyntheticDataset:
        response = self.request(
            verb=PATCH,
            path=[synthetic_dataset_id],
            json=config,
            response_type=SyntheticDataset,
        )
        return response

    def _delete(self, synthetic_dataset_id: str) -> None:
        response = self.request(verb=DELETE, path=[synthetic_dataset_id])
        return response

    def _config(self, synthetic_dataset_id: str) -> SyntheticDatasetConfig:
        response = self.request(
            verb=GET,
            path=[synthetic_dataset_id, "config"],
            response_type=SyntheticDatasetConfig,
        )
        return response

    def _download(
        self,
        synthetic_dataset_id: str,
        ds_format: SyntheticDatasetFormat = SyntheticDatasetFormat.parquet,
        short_lived_file_token: str | None = None,
    ) -> (bytes, str | None):
        response = self.request(
            verb=GET,
            path=[synthetic_dataset_id, "download"],
            params={
                "format": ds_format.upper() if isinstance(ds_format, str) else ds_format.value,
                "slft": short_lived_file_token,
            },
            headers={
                "Content-Type": "application/zip",
                "Accept": "application/json, text/plain, */*",
            },
            raw_response=True,
        )
        content_bytes = response.content
        # Check if 'Content-Disposition' header is present
        if "Content-Disposition" in response.headers:
            content_disposition = response.headers["Content-Disposition"]
            filename = re.findall("filename=(.+)", content_disposition)[0]
        else:
            filename = f"synthetic-dataset-{synthetic_dataset_id[:8]}.zip"
        return content_bytes, filename

    def _data(self, synthetic_dataset_id: str, short_lived_file_token: str | None) -> dict[str, pd.DataFrame]:
        # download pqt
        pqt_zip_bytes, filename = self._download(
            synthetic_dataset_id=synthetic_dataset_id,
            ds_format=SyntheticDatasetFormat.parquet,
            short_lived_file_token=short_lived_file_token,
        )
        # read each parquet file into a pandas dataframe
        with zipfile.ZipFile(io.BytesIO(pqt_zip_bytes), "r") as z:
            dir_list = {name.split("/")[0] for name in z.namelist()}
            dfs = {}
            for table in dir_list:
                pqt_files = [
                    name for name in z.namelist() if name.startswith(f"{table}/") and name.endswith(".parquet")
                ]
                dfs[table] = pd.concat([pd.read_parquet(z.open(name)) for name in pqt_files], axis=0)
                dfs[table].name = table
        return dfs

    def _report(
        self,
        synthetic_dataset_id: str,
        synthetic_table_id: str,
        model_type: ModelType = ModelType.tabular,
        report_type: SyntheticDatasetReportType = SyntheticDatasetReportType.model,
        short_lived_file_token: str | None = None,
    ) -> (str, str | None):
        response = self.request(
            verb=GET,
            path=[synthetic_dataset_id, "tables", synthetic_table_id, "report"],
            params={
                "modelType": model_type.upper() if isinstance(model_type, str) else model_type.value,
                "reportType": report_type.upper() if isinstance(report_type, str) else report_type.value,
                "slft": short_lived_file_token,
            },
            headers={
                "Accept": "text/html, text/plain, */*",
            },
            raw_response=True,
        )
        return response.text

    def _generation_start(self, synthetic_dataset_id: str) -> None:
        self.request(verb=POST, path=[synthetic_dataset_id, "generation", "start"])

    def _generation_cancel(self, synthetic_dataset_id: str) -> None:
        self.request(verb=POST, path=[synthetic_dataset_id, "generation", "cancel"])

    def _generation_progress(self, synthetic_dataset_id: str) -> JobProgress:
        response = self.request(
            verb=GET,
            path=[synthetic_dataset_id, "generation"],
            response_type=JobProgress,
        )
        return response

    def _generation_wait(self, synthetic_dataset_id: str, progress_bar: bool, interval: float) -> SyntheticDataset:
        job_wait(
            lambda: self._generation_progress(synthetic_dataset_id),
            interval,
            progress_bar,
        )
        synthetic_dataset = self.get(synthetic_dataset_id)
        return synthetic_dataset


class _MostlySyntheticProbesClient(_MostlyBaseClient):
    SECTION = ["synthetic-probes"]

    def create(self, config: SyntheticProbeConfig | dict[str, Any]) -> pd.DataFrame | dict[str, pd.DataFrame]:
        """
        Create a synthetic probe.

        See [`mostly.probe`](api_client.md#mostlyai.sdk.client.api.MostlyAI.probe) for more details.

        Args:
            config: Configuration for the synthetic probe.

        Returns:
            A dictionary mapping probe names to pandas DataFrames.
        """
        dicts = self.request(
            verb=POST,
            path=[],
            json=config,
        )
        return {dct["name"]: pd.DataFrame(dct["rows"]) for dct in dicts}