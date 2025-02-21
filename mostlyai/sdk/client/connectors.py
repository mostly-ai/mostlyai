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
from typing import Any
from collections.abc import Iterator

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
    ConnectorListItem,
    ConnectorPatchConfig,
    ConnectorConfig,
)


class _MostlyConnectorsClient(_MostlyBaseClient):
    SECTION = ["connectors"]

    # PUBLIC METHODS #

    def list(
        self,
        offset: int = 0,
        limit: int | None = None,
        access_type: str | None = None,
        search_term: str | None = None,
        owner_id: str | list[str] | None = None,
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
            access_type: Filter by access type (e.g., "SOURCE" or "DESTINATION").
            search_term: Filter by string in the connector name.
            owner_id: Filter by owner ID.

        Returns:
            Iterator[ConnectorListItem]: An iterator over connector list items.
        """
        with Paginator(
            self,
            ConnectorListItem,
            offset=offset,
            limit=limit,
            access_type=access_type,
            search_term=search_term,
            owner_id=owner_id,
        ) as paginator:
            yield from paginator

    def get(self, connector_id: str) -> Connector:
        """
        Retrieve a connector by its ID.

        Example for retrieving a connector:
            ```python
            from mostlyai.sdk import MostlyAI
            mostly = MostlyAI()
            c = mostly.connectors.get('INSERT_YOUR_CONNECTOR_ID')
            c
            ```

        Args:
            connector_id: The unique identifier of the connector.

        Returns:
            Connector: The retrieved connector object.
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
        connector = self.request(
            verb=POST,
            path=[],
            json=config,
            params={"test_connection": test_connection},
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
        config: ConnectorPatchConfig | dict[str, Any],
        test_connection: bool | None = True,
    ) -> Connector:
        response = self.request(
            verb=PATCH,
            path=[connector_id],
            json=config,
            params={"test_connection": test_connection},
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
