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
import os
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import rich
from rich.prompt import Prompt

from mostlyai import sdk
from mostlyai.sdk.client.base import GET, _MostlyBaseClient, DEFAULT_BASE_URL
from mostlyai.sdk.client.connectors import _MostlyConnectorsClient
from mostlyai.sdk.client.generators import _MostlyGeneratorsClient
from mostlyai.sdk.domain import (
    Connector,
    CurrentUser,
    Generator,
    SyntheticDataset,
    ModelType,
    ConnectorConfig,
    GeneratorConfig,
    SourceTableConfig,
    SyntheticDatasetConfig,
    SyntheticProbeConfig,
    AboutService,
)
from mostlyai.sdk.client.synthetic_datasets import (
    _MostlySyntheticDatasetsClient,
    _MostlySyntheticProbesClient,
)
from mostlyai.sdk.client._base_utils import convert_to_base64, read_table_from_path
from mostlyai.sdk.client._utils import (
    harmonize_sd_config,
    Seed,
    check_local_mode_available,
    validate_api_key,
    validate_base_url,
)


class MostlyAI(_MostlyBaseClient):
    """
    Instantiate an SDK instance, either in CLIENT or in LOCAL mode.

    Example for instantiating the SDK in CLIENT mode with explicit arguments:
        ```python
        from mostlyai.sdk import MostlyAI
        mostly = MostlyAI(
            api_key='INSERT_YOUR_API_KEY',
            base_url='https://app.mostly.ai',
        )
        mostly
        # MostlyAI(base_url='https://app.mostly.ai', api_key='***')
        ```

    Example for instantiating the SDK in CLIENT mode with environment variables:
        ```python
        import os
        from mostlyai.sdk import MostlyAI
        os.environ["MOSTLY_API_KEY"] = "INSERT_YOUR_API_KEY"
        os.environ["MOSTLY_BASE_URL"] = "https://app.mostly.ai"
        mostly = MostlyAI()
        mostly
        # MostlyAI(base_url='https://app.mostly.ai', api_key='***')
        ```

    Example for instantiating the SDK in LOCAL mode, and connect via Unix Domain Socket (UDS):
        ```python
        from mostlyai.sdk import MostlyAI
        mostly = MostlyAI(local=True)
        mostly
        # MostlyAI(local=True)
        ```

    Example for instantiating the SDK in LOCAL mode, and connect via Transmission Control Protocol (TCP):
        ```python
        from mostlyai.sdk import MostlyAI
        mostly = MostlyAI(local=True, local_port=8080)
        mostly
        # MostlyAI(local=True, local_port=8080)
        ```

    Args:
        base_url: The base URL. If not provided, a default value is used.
        api_key: The API key for authenticating. If not provided, it would rely on environment variables.
        local: Whether to run in local mode or not.
        local_dir: The directory to use for local mode. If not provided, `~/mostlyai` will be used.
        local_port: The port to use for local mode with TCP transport. If not provided, UDS transport is used by default.
        timeout: Timeout for HTTPS requests in seconds.
        ssl_verify: Whether to verify SSL certificates.
        quiet: Whether to suppress rich output.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        local: bool | None = None,
        local_dir: str | Path | None = None,
        local_port: int | None = None,
        timeout: float = 60.0,
        ssl_verify: bool = True,
        quiet: bool = False,
    ):
        import warnings

        # suppress deprecation warnings, also those stemming from external libs
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # determine SDK mode: either CLIENT or LOCAL mode
        mode: Literal["CLIENT", "LOCAL", None] = None
        if base_url is not None or api_key is not None:
            mode = "CLIENT"
        elif local is not None:
            mode = "LOCAL" if bool(local) else "CLIENT"
        elif os.getenv("MOSTLY_LOCAL"):
            mode = "LOCAL" if os.getenv("MOSTLY_LOCAL").lower()[:1] in ["1", "t", "y"] else "CLIENT"
        elif os.getenv("MOSTLY_API_KEY"):
            mode = "CLIENT"
        else:
            # prompt for CLIENT or LOCAL setup, if not yet determined
            choice = Prompt.ask(
                "Select your desired SDK mode:\n\n"
                "1) Run in [bold]CLIENT mode[/bold] 📡, connecting to a remote MOSTLY AI platform\n\n"
                "2) Run in [bold]LOCAL mode[/bold] 🏠, operating offline using your own compute\n\n"
                "Enter your choice",
                choices=["1", "2"],
                default="1",
            )
            if choice == "1":
                mode = "CLIENT"
                base_url = os.getenv("MOSTLY_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
                base_url = Prompt.ask("Enter the [bold]Base URL[/bold] 🌐 of the MOSTLY AI platform", default=base_url)
                api_key_url = f"{base_url}/settings/api-keys"
                api_key = Prompt.ask(
                    f"Enter your [bold]API key[/bold] 🔑 for {base_url} (obtain [link={api_key_url} dodger_blue2 underline]here[/link])",
                    default="mostly-xxx",
                    password=True,
                )
                rich.print(
                    "[dim][bold]Note[/bold]: To skip this prompt in the future, instantiate via [bold]MostlyAI(base_url=..., api_key=...)[/bold].\n\n"
                    "Alternatively set [bold]MOSTLY_BASE_URL[/bold] and [bold]MOSTLY_API_KEY[/bold] as environment variables.[/dim]"
                )
            else:
                mode = "LOCAL"
                rich.print(
                    "[dim][bold]Note[/bold]: To skip this prompt in the future, instantiate via [bold]MostlyAI(local=True)[/bold].\n\n"
                    "Alternatively set [bold]MOSTLY_LOCAL=1[/bold] as an environment variable.[/dim]"
                )

        if mode == "LOCAL":
            check_local_mode_available()
            from mostlyai.sdk._local.server import LocalServer  # noqa

            self.local_server = LocalServer(home_dir=local_dir, port=local_port)
            home_dir = self.local_server.home_dir
            base_url = self.local_server.base_url
            api_key = "local"
            uds = self.local_server.uds
        elif mode == "CLIENT":
            if base_url is None:
                base_url = os.getenv("MOSTLY_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
            validate_base_url(base_url)
            if api_key is None:
                api_key = os.getenv("MOSTLY_API_KEY")
            validate_api_key(api_key)
            home_dir = None
            uds = None
        else:
            raise ValueError("Invalid SDK mode")

        client_kwargs = {
            "base_url": base_url,
            "api_key": api_key,
            "uds": uds,
            "timeout": timeout,
            "ssl_verify": ssl_verify,
        }
        super().__init__(**client_kwargs)
        self.connectors = _MostlyConnectorsClient(**client_kwargs)
        self.generators = _MostlyGeneratorsClient(**client_kwargs)
        self.synthetic_datasets = _MostlySyntheticDatasetsClient(**client_kwargs)
        self.synthetic_probes = _MostlySyntheticProbesClient(**client_kwargs)
        if mode == "LOCAL":
            rich.print(f"Initializing [bold]Synthetic Data SDK[/bold] {sdk.__version__} in [bold]LOCAL mode[/bold] 🏠")
            if self.local_server.uds:
                msg = f"Connected to [link=file://{home_dir} dodger_blue2 underline]{home_dir}[/]"
            else:
                msg = f"Connected to [link={self.base_url} dodger_blue2 underline]{self.base_url}[/]"
            import torch  # noqa
            import psutil  # noqa

            msg += f" with {psutil.virtual_memory().total / (1024**3):.0f} GB RAM"
            msg += f", {psutil.cpu_count(logical=True)} CPUs"
            if torch.cuda.is_available():
                msg += f", {torch.cuda.device_count()}x {torch.cuda.get_device_name()}"
            else:
                msg += ", 0 GPUs"
            msg += " available"
            rich.print(msg)
        elif mode == "CLIENT":
            rich.print(f"Initializing [bold]Synthetic Data SDK[/bold] {sdk.__version__} in [bold]CLIENT mode[/bold] 📡")
            try:
                server_version = self.about().version
                email = self.me().email
                msg = f"Connected to [link={self.base_url} dodger_blue2 underline]{self.base_url}[/] {server_version}"
                msg += f" as [bold]{email}[/bold]" if email else ""
                rich.print(msg)
            except Exception as e:
                rich.print(f"Failed to connect to {self.base_url}: {e}.")
        else:
            raise ValueError("Invalid SDK mode")

        if quiet:
            rich.get_console().quiet = True

    def __repr__(self) -> str:
        if self.local:
            if self.local_server.uds:
                return "MostlyAI(local=True)"
            return f"MostlyAI(local=True, local_port={self.local_server.port})"
        return f"MostlyAI(base_url='{self.base_url}', api_key=***)"

    def connect(
        self,
        config: ConnectorConfig | dict[str, Any],
        test_connection: bool | None = True,
    ) -> Connector:
        """
        Create a connector and optionally validate the connection before saving.

        See [ConnectorConfig](api_domain.md#mostlyai.sdk.domain.ConnectorConfig) for more information on the available configuration parameters.

        Example for creating a connector to a AWS S3 storage:
            ```python
            from mostlyai.sdk import MostlyAI
            mostly = MostlyAI()
            c = mostly.connect(
                config={
                    'type': 'S3_STORAGE',
                    'config': {
                        'accessKey': '...',
                    },
                    'secrets': {
                        'secretKey': '...'
                    }
                }
            )
            ```

        Args:
            config: Configuration for the connector. Can be either a ConnectorConfig object or an equivalent dictionary.
            test_connection: Whether to validate the connection before saving.

        The structures of the `config`, `secrets` and `ssl` parameters depend on the connector `type`:

        - Cloud storage:
          ```yaml
          - type: AZURE_STORAGE
            config:
              accountName: string
              clientId: string (required for auth via service principal)
              tenantId: string (required for auth via service principal)
            secrets:
              accountKey: string (required for regular auth)
              clientSecret: string (required for auth via service principal)

          - type: GOOGLE_CLOUD_STORAGE
            config:
            secrets:
              keyFile: string

          - type: S3_STORAGE
            config:
              accessKey: string
              endpointUrl: string (only needed for S3-compatible storage services other than AWS)
            secrets:
              secretKey: string
          ```
        - Database:
          ```yaml
          - type: BIGQUERY
            config:
            secrets:
              keyFile: string

          - type: DATABRICKS
            config:
              host: string
              httpPath: string
              catalog: string
              clientId: string (required for auth via service principal)
              tenantId: string (required for auth via service principal)
            secrets:
              accessToken: string (required for regular auth)
              clientSecret: string (required for auth via service principal)

          - type: HIVE
            config:
              host: string
              port: integer, default: 10000
              username: string (required for regular auth)
              kerberosEnabled: boolean, default: false
              kerberosPrincipal: string (required if kerberosEnabled)
              kerberosKrb5Conf: string (required if kerberosEnabled)
              sslEnabled: boolean, default: false
            secrets:
              password: string (required for regular auth)
              kerberosKeytab: base64-encoded string (required if kerberosEnabled)
            ssl:
              caCertificate: base64-encoded string

          - type: MARIADB
            config:
              host: string
              port: integer, default: 3306
              username: string
            secrets:
              password: string

          - type: MSSQL
            config:
              host: string
              port: integer, default: 1433
              username: string
              database: string
            secrets:
             password: string

          - type: MYSQL
            config:
              host: string
              port: integer, default: 3306
              username: string
            secrets:
              password: string

          - type: ORACLE
            config:
              host: string
              port: integer, default: 1521
              username: string
              connectionType: enum {SID, SERVICE_NAME}, default: SID
              database: string, default: ORCL
            secrets:
              password: string

          - type: POSTGRES
            config:
              host: string
              port: integer, default: 5432
              username: string
              database: string
              sslEnabled: boolean, default: false
            secrets:
              password: string
            ssl:
              rootCertificate: base64-encoded string
              sslCertificate: base64-encoded string
              sslCertificateKey: base64-encoded string

          - type: SNOWFLAKE
            config:
              account: string
              username: string
              warehouse: string, default: COMPUTE_WH
              database: string
            secrets:
              password: string
          ```

        Returns:
            Connector: The created connector.
        """
        c = self.connectors.create(config=config, test_connection=test_connection)
        return c

    def train(
        self,
        config: GeneratorConfig | dict | None = None,
        data: pd.DataFrame | str | Path | None = None,
        name: str | None = None,
        start: bool = True,
        wait: bool = True,
        progress_bar: bool = True,
    ) -> Generator:
        """
        Train a generator.

        See [GeneratorConfig](api_domain.md#mostlyai.sdk.domain.GeneratorConfig) for more information on the available configuration parameters.

        Example of short-hand notation, reading data from path:
            ```python
            from mostlyai.sdk import MostlyAI
            mostly = MostlyAI()
            g = mostly.train(
                data='https://github.com/mostly-ai/public-demo-data/raw/dev/census/census.csv.gz',
            )
            ```

        Example of short-hand notation, passing data as pandas DataFrame:
            ```python
            # read original data
            import pandas as pd
            df_original = pd.read_csv('https://github.com/mostly-ai/public-demo-data/raw/dev/titanic/titanic.csv')
            # instantiate client
            from mostlyai.sdk import MostlyAI
            mostly = MostlyAI()
            # train generator
            g = mostly.train(
                name='census',
                data=df_original,
            )
            ```

        Example configuration using GeneratorConfig:
            ```python
            # read original data
            import pandas as pd
            df_original = pd.read_csv('https://github.com/mostly-ai/public-demo-data/raw/dev/titanic/titanic.csv')
            # instantiate client
            from mostlyai.sdk import MostlyAI
            mostly = MostlyAI()
            # configure generator via GeneratorConfig
            from mostlyai.sdk.domain import GeneratorConfig, SourceTableConfig
            g = mostly.train(
                config=GeneratorConfig(
                    name='census',
                    tables=[
                        SourceTableConfig(
                            name='data',
                            data=df_original
                        )
                    ]
                )
            )
            ```

        Example configuration using a dictionary:
            ```python
            # read original data
            import pandas as pd
            df_original = pd.read_csv('https://github.com/mostly-ai/public-demo-data/raw/dev/titanic/titanic.csv')
            # instantiate client
            from mostlyai.sdk import MostlyAI
            mostly = MostlyAI()
            # configure generator via dictionary
            g = mostly.train(
                config={
                    'name': 'census',
                    'tables': [
                        {
                            'name': 'data',
                            'data': df_original
                        }
                    ]
                }
            )
            ```

        Args:
            config: The configuration parameters of the generator to be created. Either `config` or `data` must be provided.
            data: A single pandas DataFrame, or a path to a CSV or PARQUET file. Either `config` or `data` must be provided.
            name: Name of the generator.
            start: Whether to start training immediately.
            wait: Whether to wait for training to finish.
            progress_bar: Whether to display a progress bar during training.

        Returns:
            Generator: The created generator.
        """
        if data is None and config is None:
            raise ValueError("Either config or data must be provided")
        if data is not None and config is not None:
            raise ValueError("Either config or data must be provided, but not both")
        if config is not None and isinstance(config, (pd.DataFrame, str, Path)) is None:
            # map config to data, in case user incorrectly provided data as first argument
            data = config
        if isinstance(data, (str, Path)):
            name, df = read_table_from_path(data)
            config = GeneratorConfig(
                name=name,
                tables=[SourceTableConfig(data=convert_to_base64(df), name=name)],
            )
        elif isinstance(data, pd.DataFrame) or (
            data.__class__.__name__ == "DataFrame" and data.__class__.__module__.startswith("pyspark.sql")
        ):
            df = data
            config = GeneratorConfig(
                tables=[SourceTableConfig(data=convert_to_base64(df), name="data")],
            )
        if isinstance(config, dict):
            config = GeneratorConfig(**config)
        if name is not None:
            config.name = name
        g = self.generators.create(config)
        if start:
            g.training.start()
        if start and wait:
            g.training.wait(progress_bar=progress_bar)
        return g

    def generate(
        self,
        generator: Generator | str | None = None,
        config: SyntheticDatasetConfig | dict | None = None,
        size: int | dict[str, int] | None = None,
        seed: Seed | dict[str, Seed] | None = None,
        name: str | None = None,
        start: bool = True,
        wait: bool = True,
        progress_bar: bool = True,
    ) -> SyntheticDataset:
        """
        Generate synthetic data.

        See [SyntheticDatasetConfig](api_domain.md#mostlyai.sdk.domain.SyntheticDatasetConfig) for more information on the available configuration parameters.

        Example configuration using short-hand notation:
            ```python
            from mostlyai.sdk import MostlyAI
            mostly = MostlyAI()
            sd = mostly.generate(generator=g, size=1000)
            ```

        Example configuration using SyntheticDatasetConfig:
            ```python
            from mostlyai.sdk import MostlyAI
            mostly = MostlyAI()
            sd = mostly.generate(
                config=SyntheticDatasetConfig(
                    generator=g,
                    tables=[
                        SyntheticTableConfig(
                            name="data",
                            configuration=SyntheticTableConfiguration(
                                sample_size=1000,
                                sampling_temperature=0.9,
                            )
                        )
                    ]
                )
            )
            ```

        Example configuration using a dictionary:
            ```python
            from mostlyai.sdk import MostlyAI
            mostly = MostlyAI()
            sd = mostly.generate(
                config={
                    'generator': g,
                    'tables': [
                        {
                            'name': 'data',
                            'configuration': {
                                'sample_size': 1000,
                                'sampling_temperature': 0.9,
                            }
                        }
                    ]
                }
            )
            ```

        Args:
            generator: The generator instance or its UUID.
            config: Configuration for the synthetic dataset.
            size : Sample size(s) for the subject table(s).
            seed: Seed data for the subject table(s).
            name: Name of the synthetic dataset.
            start: Whether to start generation immediately.
            wait: Whether to wait for generation to finish.
            progress_bar: Whether to display a progress bar during generation.

        Returns:
            SyntheticDataset: The created synthetic dataset.
        """
        config = harmonize_sd_config(
            generator,
            get_generator=self.generators.get,
            size=size,
            seed=seed,
            config=config,
            config_type=SyntheticDatasetConfig,
            name=name,
        )
        sd = self.synthetic_datasets.create(config)
        if start:
            sd.generation.start()
        if start and wait:
            sd.generation.wait(progress_bar=progress_bar)
        return sd

    def probe(
        self,
        generator: Generator | str | None = None,
        size: int | dict[str, int] | None = None,
        seed: Seed | dict[str, Seed] | None = None,
        config: SyntheticProbeConfig | dict | None = None,
        return_type: Literal["auto", "dict"] = "auto",
    ) -> pd.DataFrame | dict[str, pd.DataFrame]:
        """
        Probe a generator.

        See [SyntheticProbeConfig](api_domain.md#mostlyai.sdk.domain.SyntheticProbeConfig) for more information on the available configuration parameters.

        Example for probing a generator for 10 synthetic samples:
            ```python
            from mostlyai.sdk import MostlyAI
            mostly = MostlyAI()
            probe = mostly.probe(
                generator='INSERT_YOUR_GENERATOR_ID',
                size=10
            )
            ```

        Example for conditional probing a generator for 10 synthetic samples:
            ```python
            import pandas as pd
            from mostlyai.sdk import MostlyAI
            mostly = MostlyAI()
            g = mostly.generators.get('INSERT_YOUR_GENERATOR_ID')
            print('columns:', [c.name for c in g.tables[0].columns])
            # columns: ['age', 'workclass', 'fnlwgt', ...]
            col = g.tables[0].columns[1]
            print(col.name, col.value_range.values)
            # workclass: ['Federal-gov', 'Local-gov', 'Never-worked', ...]
            mostly.probe(
                generator=g,
                seed=pd.DataFrame({
                    'age': [63, 45],
                    'sex': ['Female', 'Male'],
                    'workclass': ['Sales', 'Tech-support'],
                }),
            )
            ```

        Args:
            generator: The generator instance or its UUID.
            size: Sample size(s) for the subject table(s).
            seed: Seed data for the subject table(s).
            config: Configuration for the probe.
            return_type: Format of the return value. "auto" for pandas DataFrame if a single table, otherwise a dictionary.

        Returns:
            The created synthetic probe.
        """
        config = harmonize_sd_config(
            generator,
            get_generator=self.generators.get,
            size=size,
            seed=seed,
            config=config,
            config_type=SyntheticProbeConfig,
        )
        dfs = self.synthetic_probes.create(config)
        if return_type == "auto" and len(dfs) == 1:
            return list(dfs.values())[0]
        else:
            return dfs

    def me(self) -> CurrentUser:
        """
        Retrieve information about the current user.

        Example for retrieving information about the current user:
            ```python
            from mostlyai.sdk import MostlyAI
            mostly = MostlyAI()
            mostly.me()
            # {'id': '488f2f26-...', 'first_name': 'Tom', ...}
            ```

        Returns:
            Information about the current user.
        """
        return self.request(verb=GET, path=["users", "me"], response_type=CurrentUser)

    def about(self) -> AboutService:
        """
        Retrieve information about the platform.

        Example for retrieving information about the platform:
            ```python
            from mostlyai.sdk import MostlyAI
            mostly = MostlyAI()
            mostly.about()
            # {'version': 'v316', 'assistant': True}
            ```

        Returns:
            Information about the platform.
        """
        return self.request(verb=GET, path=["about"], response_type=AboutService)

    def models(self) -> dict[str : list[str]]:
        """
        Retrieve a list of available models of a specific type.

        Example for retrieving available models:
            ```python
            from mostlyai.sdk import MostlyAI
            mostly = MostlyAI()
            mostly.models()
            # {
            #    'TABULAR": ['MOSTLY_AI/Small', 'MOSTLY_AI/Medium', 'MOSTLY_AI/Large'],
            #    'LANGUAGE": ['MOSTLY_AI/LSTMFromScratch-3m', 'microsoft/phi-1_5', ..],
            # }
            ```

        Returns:
            A dictionary with list of available models for each ModelType.
        """
        return {model_type.value: self.request(verb=GET, path=["models", model_type.value]) for model_type in ModelType}

    def computes(self) -> list[dict[str, Any]]:
        """
        Retrieve a list of available compute resources, that can be used for executing tasks.

        Example for retrieving available compute resources:
            ```python
            from mostlyai.sdk import MostlyAI
            mostly = MostlyAI()
            mostly.computes()
            # [{'id': '...', 'name': 'CPU Large',...]
            ```

        Returns:
            A list of available compute resources.
        """
        return self.request(verb=GET, path=["computes"])
