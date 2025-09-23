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

from __future__ import annotations

import io
from collections.abc import Iterator
from typing import Any

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
    Connector,
    ConnectorConfig,
    ConnectorDeleteDataConfig,
    ConnectorListItem,
    ConnectorPatchConfig,
    ConnectorQueryConfig,
    ConnectorReadDataConfig,
    ConnectorWriteDataConfig,
    IfExists,
)


class _MostlyConnectorsClient(_MostlyBaseClient):
    SECTION = ["connectors"]

    # PUBLIC METHODS #

    def list(
        self,
        offset: int = 0,
        limit: int | None = None,
        search_term: str | None = None,
        access_type: str | None = None,
        owner_id: str | list[str] | None = None,
        visibility: str | list[str] | None = None,
        created_from: str | None = None,
        created_to: str | None = None,
        sort_by: str | list[str] | None = None,
    ) -> Iterator[ConnectorListItem]:
        """
        List connectors.

        Paginate through all connectors accessible by the user. Only connectors that are independent of a table will be returned.

        Example for listing all connectors:
            ```python
            from mostlyai.sdk import MostlyAI
            mostly = MostlyAI()
            for c in mostly.connectors.list():
                print(f"Connector `{c.name}` ({c.access_type}, {c.type}, {c.id})")
            ```

        Args:
            offset: Offset for entities in the response.
            limit: Limit for the number of entities in the response.
            search_term: Filter by search term in the name and description.
            access_type: Filter by access type (e.g., READ_PROTECTED, READ_DATA or WRITE_DATA).
            owner_id: Filter by owner ID.
            visibility: Filter by visibility (e.g., PUBLIC, PRIVATE or UNLISTED).
            created_from: Filter by creation date, not older than this date. Format: YYYY-MM-DD.
            created_to: Filter by creation date, not younger than this date. Format: YYYY-MM-DD.
            sort_by: Sort by field. Either NO_OF_THREADS, NO_OF_LIKES, RECENCY or NO_OF_GENERATORS.

        Returns:
            Iterator[ConnectorListItem]: An iterator over connector list items.
        """
        with Paginator(
            self,
            ConnectorListItem,
            offset=offset,
            limit=limit,
            search_term=search_term,
            access_type=access_type,
            owner_id=owner_id,
            visibility=visibility,
            created_from=created_from,
            created_to=created_to,
            sort_by=sort_by,
        ) as paginator:
            yield from paginator

    def get(self, connector_id: str) -> Connector:
        """
        Retrieve a connector by its ID.

        Args:
            connector_id: The unique identifier of the connector.

        Returns:
            Connector: The retrieved connector object.

        Example for retrieving a connector:
            ```python
            from mostlyai.sdk import MostlyAI
            mostly = MostlyAI()
            c = mostly.connectors.get('INSERT_YOUR_CONNECTOR_ID')
            c
            ```
        """
        if not isinstance(connector_id, str) or len(connector_id) != 36:
            raise ValueError("The provided connector_id must be a UUID string")
        response = self.request(verb=GET, path=[connector_id], response_type=Connector)
        return response

    def create(
        self,
        config: ConnectorConfig | dict[str, Any],
        test_connection: bool | None = True,
    ) -> Connector:
        """
        Create a connector and optionally validate the connection before saving.

        See [`mostly.connect`](api_client.md#mostlyai.sdk.client.api.MostlyAI.connect) for more details.

        Args:
            config: Configuration for the connector.
            test_connection: Whether to test the connection before saving the connector

        Returns:
            The created connector object.
        """
        config = ConnectorConfig.model_validate(config)
        connector = self.request(
            verb=POST,
            path=[],
            json=config,
            params={"testConnection": test_connection},
            response_type=Connector,
        )
        cid = connector.id
        if self.local:
            rich.print(f"Created connector [dodger_blue2]{cid}[/]")
        else:
            rich.print(f"Created connector [link={self.base_url}/d/connectors/{cid} dodger_blue2 underline]{cid}[/]")
        return connector

    # PRIVATE METHODS #

    def _update(
        self,
        connector_id: str,
        config: ConnectorPatchConfig,
        test_connection: bool | None = True,
    ) -> Connector:
        response = self.request(
            verb=PATCH,
            path=[connector_id],
            json=config,
            params={"testConnection": test_connection},
            exclude_none_in_json=True,
            response_type=Connector,
        )
        return response

    def _delete(self, connector_id: str) -> None:
        self.request(verb=DELETE, path=[connector_id])

    def _config(self, connector_id: str) -> ConnectorConfig:
        response = self.request(verb=GET, path=[connector_id, "config"], response_type=ConnectorConfig)
        return response

    def _locations(self, connector_id: str, prefix: str = "") -> list:
        response = self.request(verb=GET, path=[connector_id, "locations"], params={"prefix": prefix})
        return response

    def _schema(self, connector_id: str, location: str) -> list[dict[str, Any]]:
        response = self.request(verb=GET, path=[connector_id, "schema"], params={"location": location})
        return response

    def _read_data(
        self, connector_id: str, location: str, limit: int | None = None, shuffle: bool = False
    ) -> pd.DataFrame:
        response = self.request(
            verb=POST,
            path=[connector_id, "read-data"],
            json=ConnectorReadDataConfig(location=location, limit=limit, shuffle=shuffle),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/octet-stream, application/json",
            },
            raw_response=True,
        )
        content_bytes = response.content
        df = pd.read_parquet(io.BytesIO(content_bytes))
        return df

    def _write_data(
        self, connector_id: str, data: pd.DataFrame | None, location: str, if_exists: IfExists = IfExists.fail
    ) -> None:
        files = {}
        if data is not None:
            buffer = io.BytesIO()
            data.to_parquet(buffer, index=False)
            buffer.seek(0)
            files = {
                "file": ("data.parquet", buffer, "application/octet-stream"),
            }
        non_file_config = ConnectorWriteDataConfig(file=b"", location=location, if_exists=if_exists).model_dump(
            mode="json", by_alias=True
        )
        non_file_config.pop("file")
        self.request(
            verb="POST",
            path=[connector_id, "write-data"],
            files=files,
            data=non_file_config,
        )

    def _delete_data(self, connector_id: str, location: str) -> None:
        self.request(
            verb="POST",
            path=[connector_id, "delete-data"],
            json=ConnectorDeleteDataConfig(location=location),
        )

    def _query(self, connector_id: str, sql: str) -> pd.DataFrame:
        response = self.request(
            verb=POST,
            path=[connector_id, "query"],
            json=ConnectorQueryConfig(sql=sql),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/octet-stream, application/json",
            },
            raw_response=True,
        )
        content_bytes = response.content
        df = pd.read_parquet(io.BytesIO(content_bytes))
        return df
