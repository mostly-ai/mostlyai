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

import rich

from mostlyai.sdk.client.base import (
    DELETE,
    GET,
    POST,
    _MostlyBaseClient,
)
from mostlyai.sdk.domain import (
    Integration,
    IntegrationAuthorizationConfig,
    IntegrationProvidersConfig,
    Provider,
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

    def get(self, provider_id: str | Provider) -> Integration:
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
        provider_id_str = provider_id.value if isinstance(provider_id, Provider) else provider_id
        response = self.request(verb=GET, path=[provider_id_str], response_type=Integration)
        return response

    def providers(self) -> list[IntegrationProvidersConfig]:
        """
        List all available integration providers.

        Returns:
            list[IntegrationProvidersConfig]: A list of available integration providers configuration objects.

        Example for listing providers:
            ```python
            from mostlyai.sdk import MostlyAI
            mostly = MostlyAI()
            providers = mostly.integrations.providers()
            for p in providers:
                print(f"Provider: {p.name} ({p.id})")
            ```
        """
        response = self.request(verb=GET, path=["providers"], do_include_client=False)
        return [IntegrationProvidersConfig(**item) for item in response["providers"]]

    def authorize(
        self,
        provider: Provider | str,
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
        provider_enum = Provider(provider) if isinstance(provider, str) else provider
        config = IntegrationAuthorizationConfig(provider=provider_enum, scope_ids=scope_ids)
        response = self.request(
            verb=POST,
            path=["authorize"],
            json=config,
            response_type=dict,
            do_response_dict_snake_case=True,
        )
        return response.get("authorization_url", "")

    def disconnect(self, provider_id: str | Provider) -> None:
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
        provider_id_str = provider_id.value if isinstance(provider_id, Provider) else provider_id
        self.request(verb=DELETE, path=[provider_id_str])
        if self.local:
            rich.print(f"Disconnected integration [dodger_blue2]{provider_id_str}[/]")
        else:
            rich.print(
                f"Disconnected integration [link={self.base_url}/d/integrations/{provider_id_str} dodger_blue2 underline]{provider_id_str}[/]"
            )
