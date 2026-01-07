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

from __future__ import annotations

import rich

from mostlyai.sdk.client.base import (
    DELETE,
    GET,
    POST,
    _MostlyBaseClient,
)
from mostlyai.sdk.domain import (
    Integration,
    IntegrationAuthorizationRequest,
)


class _MostlyIntegrationsClient(_MostlyBaseClient):
    SECTION = ["integrations"]

    # PUBLIC METHODS #

    def list(self) -> list[Integration]:
        """
        List integrations.

        Returns all integrations accessible by the user.

        Example for listing all integrations:
            ```python
            from mostlyai.sdk import MostlyAI
            mostly = MostlyAI()
            integrations = mostly.integrations.list()
            for i in integrations:
                print(f"Integration `{i.provider_name}` ({i.status}, {i.provider_id})")
            ```

        Returns:
            list[Integration]: A list of integration objects.
        """
        response = self.request(verb=GET, path=[])
        return [Integration(**item) for item in response]

    def get(self, provider_id: str) -> Integration:
        """
        Retrieve an integration by its provider ID.

        Args:
            provider_id: The provider identifier (e.g., "google", "slack", "github").

        Returns:
            Integration: The retrieved integration object.

        Example for retrieving an integration:
            ```python
            from mostlyai.sdk import MostlyAI
            mostly = MostlyAI()
            i = mostly.integrations.get('google')
            i
            ```
        """
        response = self.request(verb=GET, path=[provider_id], response_type=Integration)
        return response

    def authorize(
        self,
        provider: str,
        scope_ids: list[str],
    ) -> str:
        """
        Generate an OAuth authorization URL for connecting an integration.

        Args:
            provider: The OAuth provider identifier (e.g., "google", "slack", "github").
            scope_ids: List of scope identifiers for this integration.

        Returns:
            str: The OAuth authorization URL.

        Example for generating an authorization URL:
            ```python
            from mostlyai.sdk import MostlyAI
            mostly = MostlyAI()
            url = mostly.integrations.authorize(
                provider='google',
                scope_ids=['550e8400-e29b-41d4-a716-446655440000']
            )
            print(f"Visit this URL to authorize: {url}")
            ```
        """
        config = IntegrationAuthorizationRequest(scope_ids=scope_ids)
        response = self.request(
            verb=POST,
            path=[provider, "authorize"],
            json=config,
            response_type=dict,
            do_response_dict_snake_case=True,
        )
        return response.get("authorization_url", "")

    def refresh(self, provider_id: str) -> None:
        """
        Refresh integration OAuth token.

        Refresh the OAuth access token for an integration.
        If the integration has a refresh token, it will be used to obtain new access tokens.
        If no refresh token exists, a new authorization flow must be initiated.

        Args:
            provider_id: The provider identifier (e.g., "google", "slack", "github").

        Example for refreshing an integration token:
            ```python
            from mostlyai.sdk import MostlyAI
            mostly = MostlyAI()
            mostly.integrations.refresh('google')
            ```

        Raises:
            APIError: If the integration is not found (404) or no refresh token is available (400).
        """
        self.request(verb=POST, path=[provider_id, "refresh"])

    def disconnect(self, provider_id: str) -> None:
        """
        Disconnect an integration.

        Args:
            provider_id: The provider identifier (e.g., "google", "slack", "github").

        Example for disconnecting an integration:
            ```python
            from mostlyai.sdk import MostlyAI
            mostly = MostlyAI()
            mostly.integrations.disconnect('google')
            ```
        """
        self.request(verb=DELETE, path=[provider_id])
        rich.print(
            f"Disconnected integration [link={self.base_url}/d/integrations/{provider_id} dodger_blue2 underline]{provider_id}[/]"
        )
