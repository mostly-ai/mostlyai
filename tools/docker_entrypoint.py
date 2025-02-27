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


def main() -> None:
    """
    Entrypoint for the MostlyAI SDK Docker image.
    Can be called without any arguments which would start in a Local mode, running on port 8080.
    Alternatively, any arguments can be passed as key-value pairs and they will be used when initiating the MostlyAI class.

    Example:
    ```bash
    docker run mostlyai/mostlyai
    # Connected to http://127.0.0.1:8080 with 16 GB RAM, 11 CPUs, 0 GPUs available

    docker run mostlyai/mostlyai --local=True --local_port=8082 --ssl_verify=False
    # Connected to http://127.0.0.1:8082 with 16 GB RAM, 11 CPUs, 0 GPUs available
    ```
    """
    from argparse import ArgumentParser
    from time import sleep

    parser = ArgumentParser(description="MostlyAI SDK Docker Entrypoint.")
    _, args = parser.parse_known_args()
    kwargs = {}
    for arg in args:
        if arg.startswith("--"):
            key, value = arg.lstrip("--").split("=", 1)
            kwargs[key] = value
    if len(kwargs) == 0:
        kwargs = {"local": True, "local_port": 8080}

    print("Startup may take a few seconds while libraries are being loaded...")

    from mostlyai.sdk import MostlyAI

    MostlyAI(**kwargs)

    try:
        while True:
            sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")


if __name__ == "__main__":
    main()
