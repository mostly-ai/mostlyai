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
import tempfile
import time
from pathlib import Path

# import random
from threading import Thread

import rich
from fastapi import FastAPI
import uvicorn
from mostlyai.sdk._local.routes import Routes

# import socket
import os


class LocalServer:
    def __init__(
        self,
        home_dir: str | Path | None = None,
        # host: str | None = None,
        # port: int | None = None,
        uds: str | None = None,
    ):
        self.home_dir = Path(home_dir or "~/mostlyai").expanduser()
        # self.host = host or "127.0.0.1"
        # self.port = port or self._find_available_port()
        self.uds = uds or tempfile.NamedTemporaryFile(prefix="mostlyai-", suffix=".sock", delete=False).name
        self.base_url = "http://127.0.0.1"
        self._app = FastAPI(
            root_path="/api/v2",
            title="Synthetic Data SDK ✨",
            description="Welcome! This is your Local Server instance of the Synthetic Data SDK. "
            "Connect via the MOSTLY AI client to train models and generate synthetic data locally. "
            "Share the knowledge of your synthetic data generators with your team or the world by "
            "deploying these then to a MOSTLY AI platform. Enjoy!",
            version="1.0.0",
        )
        routes = Routes(self.home_dir)
        self._app.include_router(routes.router)
        self._server = None
        self._thread = None
        self.start()  # Automatically start the server during initialization

    # def _find_available_port(self) -> int:
    #     def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    #         with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    #             return s.connect_ex((host, port)) == 0
    #
    #     failed_ports = []
    #     min_port, max_port = 49152, 65535
    #     for _ in range(10):
    #         port = random.randint(min_port, max_port)
    #         if not is_port_in_use(port, self.host):
    #             return port
    #         failed_ports.append(port)
    #     raise ValueError(
    #         f"Could not find an available port in range {min_port}-{max_port} after 10 attempts. Tried ports: {failed_ports}"
    #     )

    def _clear_socket_file(self):
        if os.path.exists(self.uds):
            os.remove(self.uds)

    def _create_server(self):
        self._clear_socket_file()
        config = uvicorn.Config(self._app, uds=self.uds, log_level="error", reload=False)
        self._server = uvicorn.Server(config)

    def _run_server(self):
        if self._server:
            self._server.run()

    def start(self):
        if not self._server:
            # rich.print(
            #     f"Starting server on [link={self.base_url}]{self.base_url}[/] using [link=file://{self.home_dir}]file://{self.home_dir}[/]"
            # )
            rich.print(f"Starting server using [link=file://{self.home_dir}]file://{self.home_dir}[/]")
            self._create_server()
            self._thread = Thread(target=self._run_server, daemon=True)
            self._thread.start()
            # make sure the socket file is cleaned up on exit
            atexit.register(self._clear_socket_file)
            time.sleep(0.5)  # give the server a moment to start

    def stop(self):
        if self._server and self._server.started:
            rich.print(f"Stopping server on {self.base_url} for {self.home_dir.absolute()}.")
            self._server.should_exit = True  # Signal the server to shut down
            self._thread.join()  # Wait for the server thread to finish
            self._clear_socket_file()

    def __enter__(self):
        # Ensure the server is running
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Stop the server when exiting the context
        self.stop()

    def __del__(self):
        # Backup cleanup in case `stop` was not called explicitly or via context
        if self._server and self._server.started:
            print("Automatically shutting down server")
            self.stop()
