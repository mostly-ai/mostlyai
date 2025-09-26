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

import re
from unittest import mock

import pytest
import respx
from httpx import NetworkError, Response

from mostlyai.sdk.client.base import DEFAULT_BASE_URL, CustomBaseModel, Paginator, _MostlyBaseClient
from mostlyai.sdk.client.exceptions import APIError, APIStatusError


@pytest.fixture
def mostly_base_client():
    """Fixture to provide a _MostlyBaseClient instance."""
    return _MostlyBaseClient(api_key="test_api_key")


@pytest.fixture
def more_specific_client():
    class MoreSpecificClient(_MostlyBaseClient):
        SECTION = ["more", "specific"]

    return MoreSpecificClient(api_key="test_api_key")


class TestMostlyBaseClient:
    def test_initialization(self):
        # Test with all parameters provided
        client = _MostlyBaseClient(base_url="https://custom.url", api_key="12345")
        assert client.base_url == "https://custom.url"
        assert client.api_key == "12345"

        # Test with all required parameters provided
        client = _MostlyBaseClient(api_key="12345")
        assert client.base_url == DEFAULT_BASE_URL
        assert client.api_key == "12345"

    @respx.mock
    def test_request_success(self, mostly_base_client):
        mock_url = respx.get("https://app.mostly.ai/api/v2/test").mock(
            return_value=Response(200, json={"success": True})
        )
        response = mostly_base_client.request(path="test", verb="GET")

        assert mock_url.called
        assert response == {"success": True}

    @respx.mock
    def test_request_http_error(self, mostly_base_client):
        respx.get("https://app.mostly.ai/api/v2/test").mock(return_value=Response(404, json={"message": "Not found"}))

        with pytest.raises(APIStatusError) as excinfo:
            mostly_base_client.request(path="test", verb="GET")

        assert "HTTP 404: Not found" in str(excinfo.value)

    @respx.mock
    def test_client_request_network_error(self, mostly_base_client):
        respx.get("https://app.mostly.ai/api/v2/test").mock(side_effect=NetworkError("Network error"))

        with pytest.raises(APIError) as excinfo:
            mostly_base_client.request("test", "GET")

        assert "An error occurred while requesting" in str(excinfo.value)

    @respx.mock
    def test_client_post_request(self, mostly_base_client):
        test_data = CustomBaseModel(name="Test")
        respx.post("https://app.mostly.ai/api/v2/create").mock(return_value=Response(201, json={"success": True}))

        response = mostly_base_client.request("create", "POST", json=test_data)

        assert response == {"success": True}

    @respx.mock
    def test_more_specific_client_request(self, more_specific_client):
        mock_url = respx.get("https://app.mostly.ai/api/v2/more/specific/test").mock(
            return_value=Response(200, json={"success": True})
        )
        response = more_specific_client.request(path="test", verb="GET")

        assert mock_url.called
        assert response == {"success": True}


class TestPaginator:
    @respx.mock
    def test_iteration(self, mostly_base_client):
        # Define mock responses
        offset_page_map = {
            0: {"results": [{"id": 1}, {"id": 2}], "totalCount": 4},
            2: {"results": [{"id": 3}, {"id": 4}], "totalCount": 4},
            4: {"results": [], "totalCount": 4},
        }

        # Using a callback to differentiate the response based on offset and page
        def request_callback(request):
            url_pattern = re.compile(r"offset=(\d+)")
            match = re.search(url_pattern, str(request.url.query))
            offset = int(match.group(1))

            return Response(200, json=offset_page_map.get(offset))

        # Mock any GET request and use the callback to handle it
        respx.get(url=mock.ANY).mock(side_effect=request_callback)

        paginator = Paginator(mostly_base_client, dict, limit=2)

        items = list(paginator)
        assert [item["id"] for item in items] == [1, 2]

    @respx.mock
    def test_paginator_no_results(self, mostly_base_client):
        respx.get("https://app.mostly.ai/api/v2?offset=0&limit=50").mock(
            return_value=Response(200, json={"results": [], "totalCount": 0})
        )

        paginator = Paginator(mostly_base_client, dict)

        items = list(paginator)
        assert len(items) == 0
