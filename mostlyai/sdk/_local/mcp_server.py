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

import atexit
import re
from threading import Thread
import time
import httpx
import yaml
from fastmcp import FastMCP
from fastmcp.server.openapi import RouteMap, MCPType
from fastmcp.utilities.logging import configure_logging as fastmcp_configure_logging


class MCPServer:
    def __init__(
        self,
        api_client_kwargs: dict,
        port: int = 8081,
    ):
        self.port = port
        openapi_yaml = httpx.get(
            "https://raw.githubusercontent.com/mostly-ai/mostly-openapi/refs/heads/main/public-api.yaml"
        ).text
        # temp: fix syntax of public API
        openapi_yaml = re.sub(r"(\d\d\d):", r'"\1":', openapi_yaml)
        self._openapi_spec = yaml.safe_load(openapi_yaml)
        api_client_kwargs["base_url"] = api_client_kwargs["base_url"].rstrip("/") + "/api/v2"
        # translate uds to async transport
        if api_client_kwargs["uds"] is not None:
            api_client_kwargs["transport"] = httpx.AsyncHTTPTransport(uds=api_client_kwargs["uds"])
        api_client_kwargs.pop("uds")
        self._api_client = httpx.AsyncClient(**api_client_kwargs)
        self._server = None
        self._thread = None
        self._started = False
        self.start()

    def _create_server(self):
        fastmcp_configure_logging(level="ERROR")
        self._server = FastMCP.from_openapi(
            openapi_spec=self._openapi_spec,
            client=self._api_client,
            name="MOSTLY AI MCP Server",
            # temp: only include endpoints related to connectors
            route_maps=[
                RouteMap(methods="*", pattern=r".*\/connectors.*", mcp_type=MCPType.TOOL),
                RouteMap(methods="*", pattern=r"^(?!.*\/connectors).*", mcp_type=MCPType.EXCLUDE),
            ],
            log_level="ERROR",
        )

    def _run_server(self):
        # TODO: switch to streamable-http once it's supported by Cursor
        self._server.run(transport="sse", host="127.0.0.1", port=self.port, log_level="error")

    def start(self):
        if not self._server:
            self._create_server()
            self._thread = Thread(target=self._run_server, daemon=True)
            self._thread.start()
            self._started = True
            atexit.register(self.stop)
            # give the server a moment to start
            time.sleep(1)

    def stop(self):
        if self._server and self._started:
            self._thread.join()
            self._started = False

    def __del__(self):
        self.stop()
