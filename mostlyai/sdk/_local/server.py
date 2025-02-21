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

from threading import Thread

import psutil
import rich
import torch
from fastapi import FastAPI
import uvicorn
from mostlyai.sdk._local.routes import Routes

import os


class LocalServer:
    def __init__(
        self,
        home_dir: str | Path | None = None,
    ):
        self.home_dir = Path(home_dir or "~/mostlyai").expanduser()
        self.home_dir.mkdir(parents=True, exist_ok=True)
        self.uds = tempfile.NamedTemporaryFile(prefix=".mostlyai-", suffix=".sock", delete=False).name
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
            msg = f"Starting Synthetic Data SDK in local mode using [link=file://{self.home_dir} dodger_blue2 underline]{self.home_dir}[/]"
            # report local compute resources
            compute = f"{psutil.virtual_memory().total / (1024**3):.0f} GB RAM, {psutil.cpu_count(logical=True)} CPUs"
            if torch.cuda.is_available():
                compute += f", {torch.cuda.device_count()}x {torch.cuda.get_device_name()}"
            else:
                compute += ", 0 GPUs"
            msg += f" ({compute})"
            rich.print(msg)
            self._create_server()
            self._thread = Thread(target=self._run_server, daemon=True)
            self._thread.start()
            # make sure the socket file is cleaned up on exit
            atexit.register(self._clear_socket_file)
            # give the server a moment to start
            time.sleep(0.5)

    def stop(self):
        if self._server and self._server.started:
            rich.print("Stopping Synthetic Data SDK in local mode")
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
