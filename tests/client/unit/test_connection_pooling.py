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

import respx
from httpx import Response

from mostlyai.sdk.client.base import _MostlyBaseClient


class TestConnectionPooling:
    """Tests for httpx connection pooling"""

    def test_shared_client_instance(self):
        """Verify that the same httpx.Client instance is reused across requests"""
        client = _MostlyBaseClient(api_key="test_key")

        # The client should have a _client attribute
        assert hasattr(client, "_client")
        assert client._client is not None

        # Store reference to the httpx client
        httpx_client_id = id(client._client)

        # Make multiple requests and verify the same client is used
        with respx.mock:
            route = respx.get("https://app.mostly.ai/api/v2/test").mock(
                return_value=Response(200, json={"status": "ok"})
            )

            # Make first request
            client.request(path="test", verb="GET")
            assert route.called

            # Verify same client is still used
            assert id(client._client) == httpx_client_id

            # Make second request
            client.request(path="test", verb="GET")

            # Verify same client is still used
            assert id(client._client) == httpx_client_id

        # Clean up
        client.close()

    def test_close_method(self):
        """Verify that close method properly closes the httpx client"""
        client = _MostlyBaseClient(api_key="test_key")

        # Client should be open initially
        assert not client._client.is_closed

        # Close the client
        client.close()

        # Client should now be closed
        assert client._client.is_closed

    def test_context_manager(self):
        """Verify that the client works as a context manager"""
        with _MostlyBaseClient(api_key="test_key") as client:
            # Client should be open inside context
            assert not client._client.is_closed

            # Should be able to make requests
            with respx.mock:
                respx.get("https://app.mostly.ai/api/v2/test").mock(return_value=Response(200, json={"status": "ok"}))
                result = client.request(path="test", verb="GET")
                assert result == {"status": "ok"}

        # Client should be closed after exiting context
        assert client._client.is_closed

    def test_multiple_clients_have_separate_pools(self):
        """Verify that different client instances have separate connection pools"""
        client1 = _MostlyBaseClient(api_key="test_key_1")
        client2 = _MostlyBaseClient(api_key="test_key_2")

        # Each client should have its own httpx.Client instance
        assert id(client1._client) != id(client2._client)

        # Clean up
        client1.close()
        client2.close()
