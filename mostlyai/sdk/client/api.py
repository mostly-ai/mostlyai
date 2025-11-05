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
from mostlyai.sdk.client._base_utils import convert_to_base64, read_table_from_path
from mostlyai.sdk.client._utils import (
    Seed,
    check_local_mode_available,
    harmonize_sd_config,
    validate_base_url,
)
from mostlyai.sdk.client.artifacts import _MostlyArtifactsClient
from mostlyai.sdk.client.base import DEFAULT_BASE_URL, GET, _MostlyBaseClient
from mostlyai.sdk.client.connectors import _MostlyConnectorsClient
from mostlyai.sdk.client.datasets import _MostlyDatasetsClient
from mostlyai.sdk.client.exceptions import APIError
from mostlyai.sdk.client.generators import _MostlyGeneratorsClient
from mostlyai.sdk.client.synthetic_datasets import (
    _MostlySyntheticDatasetsClient,
    _MostlySyntheticProbesClient,
)
from mostlyai.sdk.domain import (
    AboutService,
    Connector,
    ConnectorConfig,
    CurrentUser,
    Generator,
    GeneratorConfig,
    ModelType,
    SourceTableConfig,
    SyntheticDataset,
    SyntheticDatasetConfig,
    SyntheticProbeConfig,
)


class MostlyAI(_MostlyBaseClient):
    """
    Instantiate an SDK instance, either in CLIENT or in LOCAL mode.

    Args:
        base_url (str | None): The base URL. If not provided, env var `MOSTLY_BASE_URL` is used if available, otherwise `https://app.mostly.ai`.
        api_key (str | None): The API key for authenticating. If not provided, env var `MOSTLY_API_KEY` is used if available.
        bearer_token (str | None): The bearer token for authenticating. If not provided, env var `MOSTLY_BEARER_TOKEN` is used if available. Takes precedence over api_key.
        local (bool | None): Whether to run in local mode or not. If not provided, user is prompted to choose between CLIENT and LOCAL mode.
        local_dir (str | Path | None): The directory to use for local mode. If not provided, `~/mostlyai` is used.
        local_port (int | None): The port to use for local mode with TCP transport. If not provided, UDS transport is used.
        timeout (float): Timeout for HTTPS requests in seconds. Default is 60 seconds.
        ssl_verify (bool): Whether to verify SSL certificates. Default is True.
        test_connection (bool): Whether to test the connection during initialization. Default is True.
        quiet (bool): Whether to suppress rich output. Default is False.

    Example for SDK in CLIENT mode with explicit arguments:
        ```python
        from mostlyai.sdk import MostlyAI
        mostly = MostlyAI(
            api_key='INSERT_YOUR_API_KEY',
            base_url='https://app.mostly.ai',
        )
        mostly
        # MostlyAI(base_url='https://app.mostly.ai', api_key='***')
        ```

    Example for SDK in CLIENT mode with bearer token:
        ```python
        from mostlyai.sdk import MostlyAI
        mostly = MostlyAI(
            bearer_token='INSERT_YOUR_BEARER_TOKEN',
            base_url='https://app.mostly.ai',
        )
        mostly
        # MostlyAI(base_url='https://app.mostly.ai', bearer_token='***')
        ```

    Example for SDK in CLIENT mode with environment variables:
        ```python
        import os
        from mostlyai.sdk import MostlyAI
        os.environ["MOSTLY_API_KEY"] = "INSERT_YOUR_API_KEY"
        os.environ["MOSTLY_BASE_URL"] = "https://app.mostly.ai"
        mostly = MostlyAI()
        mostly
        # MostlyAI(base_url='https://app.mostly.ai', api_key='***')
        ```

    Example for SDK in CLIENT mode with bearer token environment variable:
        ```python
        import os
        from mostlyai.sdk import MostlyAI
        os.environ["MOSTLY_BEARER_TOKEN"] = "INSERT_YOUR_BEARER_TOKEN"
        os.environ["MOSTLY_BASE_URL"] = "https://app.mostly.ai"
        mostly = MostlyAI()
        mostly
        # MostlyAI(base_url='https://app.mostly.ai', bearer_token='***')
        ```

    Example for SDK in LOCAL mode connecting via UDS:
        ```python
        from mostlyai.sdk import MostlyAI
        mostly = MostlyAI(local=True)
        mostly
        # MostlyAI(local=True)
        ```

    Example for SDK in LOCAL mode connecting via TCP:
        ```python
        from mostlyai.sdk import MostlyAI
        mostly = MostlyAI(local=True, local_port=8080)
        mostly
        # MostlyAI(local=True, local_port=8080)
        ```
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        bearer_token: str | None = None,
        local: bool | None = None,
        local_dir: str | Path | None = None,
        local_port: int | None = None,
        timeout: float = 60.0,
        ssl_verify: bool = True,
        test_connection: bool = True,
        quiet: bool = False,
    ):
        import warnings

        # suppress deprecation warnings, also those stemming from external libs
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # determine SDK mode: either CLIENT or LOCAL mode
        mode: Literal["CLIENT", "LOCAL", None] = None
        if base_url is not None or api_key is not None or bearer_token is not None:
            mode = "CLIENT"
        elif local is not None:
            mode = "LOCAL" if bool(local) else "CLIENT"
        elif os.getenv("MOSTLY_LOCAL"):
            mode = "LOCAL" if os.getenv("MOSTLY_LOCAL").lower()[:1] in ["1", "t", "y"] else "CLIENT"
        elif os.getenv("MOSTLY_API_KEY") or os.getenv("MOSTLY_BEARER_TOKEN"):
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
                api_key = os.getenv("MOSTLY_API_KEY", "")
            if bearer_token is None:
                bearer_token = os.getenv("MOSTLY_BEARER_TOKEN", "")
            home_dir = None
            uds = None
        else:
            raise ValueError("Invalid SDK mode")

        # Set quiet mode BEFORE any rich output
        if quiet:
            rich.get_console().quiet = True

        client_kwargs = {
            "base_url": base_url,
            "api_key": api_key,
            "bearer_token": bearer_token,
            "uds": uds,
            "timeout": timeout,
            "ssl_verify": ssl_verify,
        }
        super().__init__(**client_kwargs)
        self.connectors = _MostlyConnectorsClient(**client_kwargs)
        self.generators = _MostlyGeneratorsClient(**client_kwargs)
        self.datasets = _MostlyDatasetsClient(**client_kwargs)
        self.artifacts = _MostlyArtifactsClient(**client_kwargs)
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
            if test_connection:
                try:
                    server_version = self.about().version
                    email = self.me().email
                    msg = (
                        f"Connected to [link={self.base_url} dodger_blue2 underline]{self.base_url}[/] {server_version}"
                    )
                    msg += f" as [bold]{email}[/bold]" if email else ""
                    rich.print(msg)
                except Exception as e:
                    rich.print(f"Failed to connect to {self.base_url} : {e}.")
        else:
            raise ValueError("Invalid SDK mode")

    def __repr__(self) -> str:
        if self.local:
            if self.local_server.uds:
                return "MostlyAI(local=True)"
            return f"MostlyAI(local=True, local_port={self.local_server.port})"
        if self.bearer_token:
            return f"MostlyAI(base_url='{self.base_url}', bearer_token=***)"
        return f"MostlyAI(base_url='{self.base_url}', api_key=***)"

    def connect(
        self,
        config: ConnectorConfig | dict[str, Any],
        test_connection: bool | None = True,
    ) -> Connector:
        """
        Create a connector and optionally validate the connection before saving.

        There are 3 access types for a connector (which are independent of the connector type):

        - `READ_PROTECTED`:  The connector is restricted to being used solely as a source for training a generator. Direct data access is not permitted, only schema access via `c.locations(prefix)` and `c.schema(location)` is available.
        - `READ_DATA`: This connector allows full read access. It can also be used as a source for training a generator.
        - `WRITE_DATA`: This connector allows full read and write access. It can be also used as a source for training a generator, as well as a destination for delivering a synthetic dataset.

        Args:
            config (ConnectorConfig | dict[str, Any]): Configuration for the connector. Can be either a ConnectorConfig object or an equivalent dictionary.
            test_connection (bool | None): Whether to validate the connection before saving. Default is True.

        Returns:
            Connector: The created connector.

        Example for creating a connector to a AWS S3 storage:
            ```python
            from mostlyai.sdk import MostlyAI
            mostly = MostlyAI()
            c = mostly.connect(
                config={
                    'access_type': 'READ_PROTECTED',  # or 'READ_DATA' or 'WRITE_DATA'
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

          - type: REDSHIFT
            config:
              host: string
              port: integer, default: 5439
              username: string
              database: string
            secrets:
              password: string
          ```
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
        Create a generator resource. Once trained, it will include the model as well as optionally a model report.

        Note: A generator is initially being configured. That training job can be either launched immediately or later. One can check progress via `g.training.progress()`. Once the job has finished, the generator is available for use.

        Args:
            config (GeneratorConfig | dict | None): The configuration parameters of the generator to be created. Either `config` or `data` must be provided.
            data (pd.DataFrame | str | Path | None): A single pandas DataFrame, or a path to a CSV or PARQUET file. Either `config` or `data` must be provided.
            name (str | None): Name of the generator.
            start (bool): Whether to start training immediately. Default is True.
            wait (bool): Whether to wait for training to finish. Default is True.
            progress_bar (bool): Whether to display a progress bar during training. Default is True.

        Returns:
            Generator: The created generator.

        Example of a single flat table with default configurations:
            ```python
            # read original data
            import pandas as pd
            df = pd.read_csv('https://github.com/mostly-ai/public-demo-data/raw/dev/census/census10k.parquet')
            # instantiate client
            from mostlyai.sdk import MostlyAI
            mostly = MostlyAI()
            # train generator
            g = mostly.train(
                name='census',
                data=df,     # alternatively, pass a path to a CSV or PARQUET file
                start=True,  # start training immediately
                wait=True,   # wait for training to finish
            )
            ```

        Example of a single flat table with custom configurations:
            ```python
            # read original data
            import pandas as pd
            df = pd.read_csv('https://github.com/mostly-ai/public-demo-data/raw/dev/baseball/players.csv.gz')
            # instantiate client
            from mostlyai.sdk import MostlyAI
            mostly = MostlyAI()
            # configure generator via dictionary
            g = mostly.train(
                config={                                             # see `mostlyai.sdk.domain.GeneratorConfig`
                    'name': 'Baseball Players',
                    'tables': [
                        {                                            # see `mostlyai.sdk.domain.SourceTableConfig`
                            'name': 'players',                       # name of the table (required)
                            'data': df,                              # either provide data as a pandas DataFrame
                            'source_connector_id': None,             # - or pass a source_connector_id
                            'location': None,                        # - together with a table location
                            'primary_key': 'id',                     # specify the primary key column, if one is present
                            'tabular_model_configuration': {         # see `mostlyai.sdk.domain.ModelConfiguration`; all settings are optional!
                                'model': 'MOSTLY_AI/Medium',         # check `mostly.models()` for available models
                                'batch_size': None,                  # set a custom physical training batch size
                                'max_sample_size': 100_000,          # cap sample size to 100k; set to None for max accuracy
                                'max_epochs': 50,                    # cap training to 50 epochs; set to None for max accuracy
                                'max_training_time': 60,             # cap runtime to 60min; set to None for max accuracy
                                'enable_flexible_generation': True,  # allow seed, imputation, rebalancing and fairness; set to False for max accuracy
                                'value_protection': True,            # privacy protect value ranges; set to False for allowing all seen values
                                'differential_privacy': {            # set DP configs if explicitly requested
                                    'max_epsilon': 5.0,                # - max DP epsilon value, used as stopping criterion
                                    'noise_multiplier': 1.5,           # - noise multiplier for DP-SGD training
                                    'max_grad_norm': 1.0,              # - max grad norm for DP-SGD training
                                    'delta': 1e-5,                     # - delta value for DP-SGD training
                                    'value_protection_epsilon': 2.0,   # - DP epsilon for determining value ranges / data domains
                                },
                                'enable_model_report': True,         # generate a model report, including quality metrics
                            },
                            'columns': [                             # list columns (optional); see `mostlyai.sdk.domain.ModelEncodingType`
                                {'name': 'id', 'model_encoding_type': 'TABULAR_CATEGORICAL'},
                                {'name': 'bats', 'model_encoding_type': 'TABULAR_CATEGORICAL'},
                                {'name': 'throws', 'model_encoding_type': 'TABULAR_CATEGORICAL'},
                                {'name': 'birthDate', 'model_encoding_type': 'TABULAR_DATETIME'},
                                {'name': 'weight', 'model_encoding_type': 'TABULAR_NUMERIC_AUTO'},
                                {'name': 'height', 'model_encoding_type': 'TABULAR_NUMERIC_AUTO'},
                            ],
                        }
                    ]
                },
                start=True,  # start training immediately
                wait=True,   # wait for training to finish
            )
            ```

        Example of a multi-table sequential dataset (time series):
            ```python
            # read original data
            import pandas as pd
            df_purchases = pd.read_csv('https://github.com/mostly-ai/public-demo-data/raw/dev/cdnow/purchases.csv.gz')
            df_users = df_purchases[['users_id']].drop_duplicates()  # create a table representing subjects / groups, if not already present
            # instantiate client
            from mostlyai.sdk import MostlyAI
            mostly = MostlyAI()
            # train generator
            g = mostly.train(config={
                'name': 'CDNOW',                      # name of the generator
                'tables': [{                          # provide list of all tables
                    'name': 'users',
                    'data': df_users,
                    'primary_key': 'users_id',        # define PK column
                }, {
                    'name': 'purchases',
                    'data': df_purchases,
                    'foreign_keys': [{                 # define FK columns, with one providing the context
                        'column': 'users_id',
                        'referenced_table': 'users',
                        'is_context': True
                    }],
                    'tabular_model_configuration': {
                        'max_sample_size': 10_000,     # cap sample size to 10k users; set to None for max accuracy
                        'max_training_time': 60,       # cap runtime to 60min; set to None for max accuracy
                        'max_sequence_window': 10,     # optionally limit the sequence window
                    },
                }],
            }, start=True, wait=True)
            ```

        Example of a multi-table relational dataset with non-context foreign key:
            ```python
            # read original data
            import pandas as pd
            repo_url = 'https://github.com/mostly-ai/public-demo-data/raw/refs/heads/dev/berka/data'
            accounts_df = pd.read_csv(f'{repo_url}/account.csv.gz')
            disp_df = pd.read_csv(f'{repo_url}/disp.csv.gz')
            clients_df = pd.read_csv(f'{repo_url}/client.csv.gz')
            # instantiate client
            from mostlyai.sdk import MostlyAI
            mostly = MostlyAI()
            # train generator
            g = mostly.train(config={
                'name': 'BERKA',
                'tables': [{
                    'name': 'clients',
                    'data': clients_df,
                    'primary_key': 'client_id',       # define PK column
                }, {
                    'name': 'accounts',
                    'data': accounts_df,
                    'primary_key': 'account_id',      # define PK column
                }, {
                    'name': 'disp',
                    'data': disp_df,
                    'primary_key': 'disp_id',         # define PK column
                    'foreign_keys': [{                # define FK columns: max 1 Context FK allowed; referenced context tables must NOT result in circular references;
                        'column': 'client_id',
                        'referenced_table': 'clients',
                        'is_context': True            # Context FK: the `disp` records that belong to the same `client` will be learned and generated together - with the context of the parent;
                                                      # -> patterns between child and parent (and grand-parent) and between siblings belonging to the same parent will all be retained;
                    }, {
                        'column': 'account_id',
                        'referenced_table': 'accounts',
                        'is_context': False           # Non-Context FK: a dedicated model will be trained to learn matching a `disp` record with a suitable `account` record;
                                                      # -> patterns between child and parent will be retained, but not between siblings belonging to the same parent;
                    }],
                }],
            }, start=True, wait=True)
            ```

        Example of a single flat table with TABULAR and LANGUAGE models:
            ```python
            # read original data
            import pandas as pd
            df = pd.read_parquet('https://github.com/mostly-ai/public-demo-data/raw/dev/headlines/headlines.parquet')

            # instantiate SDK
            from mostlyai.sdk import MostlyAI
            mostly = MostlyAI()

            # print out available LANGUAGE models
            print(mostly.models()["LANGUAGE"])

            # train a generator
            g = mostly.train(config={
                'name': 'Headlines',
                'tables': [{
                    'name': 'headlines',
                    'data': df,
                    'columns': [                                 # configure TABULAR + LANGUAGE cols
                        {'name': 'category', 'model_encoding_type': 'TABULAR_CATEGORICAL'},
                        {'name': 'date', 'model_encoding_type': 'TABULAR_DATETIME'},
                        {'name': 'headline', 'model_encoding_type': 'LANGUAGE_TEXT'},
                    ],
                    'tabular_model_configuration': {              # tabular model configuration (optional)
                        'max_sample_size': None,                  # eg. use all availabel training samples for max accuracy
                        'max_training_time': None,                # eg. set no upper time limit for max accuracy
                    },
                    'language_model_configuration': {             # language model configuration (optional)
                        'max_sample_size': 50_000,                # eg. cap sample size to 50k; set None for max accuracy
                        'max_training_time': 60,                  # eg. cap runtime to 60min; set None for max accuracy
                        'model': 'MOSTLY_AI/LSTMFromScratch-3m',  # use a light-weight LSTM model, trained from scratch (GPU recommended)
                        #'model': 'microsoft/phi-1.5',            # alternatively use a pre-trained HF-hosted LLM model (GPU required)
                    }
                }],
            }, start=True, wait=True)
            ```
        """
        if data is None and config is None:
            raise ValueError("Either config or data must be provided")
        if data is not None and config is not None:
            raise ValueError("Either config or data must be provided, but not both")
        if config is not None and isinstance(config, (pd.DataFrame, str, Path)) is None:
            # map config to data, in case user incorrectly provided data as first argument
            data = config
        if isinstance(data, (str, Path)):
            table_name, df = read_table_from_path(data)
            if name is None:
                name = table_name
            config = GeneratorConfig(
                name=name,
                tables=[SourceTableConfig(data=convert_to_base64(df), name=table_name)],
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
        generator: Generator | str,
        config: SyntheticDatasetConfig | dict | None = None,
        size: int | dict[str, int] | None = None,
        seed: Seed | dict[str, Seed] | None = None,
        name: str | None = None,
        start: bool = True,
        wait: bool = True,
        progress_bar: bool = True,
    ) -> SyntheticDataset:
        """
        Create a synthetic dataset resource. Once generated, it will include the data as well as optionally a data report.

        Note: A synthetic dataset is initially being configured. That generation job can be either launched immediately or later. One can check progress via `sd.generation.progress()`. Once the job has finished, the synthetic dataset is available for download via `sd.data()`, and the reports are available via `sd.reports()`.

        Args:
            generator (Generator | str): The generator instance or its UUID.
            config (SyntheticDatasetConfig | dict | None): Configuration for the synthetic dataset.
            size (int | dict[str, int] | None): Sample size(s) for the subject table(s).
            seed (Seed | dict[str, Seed] | None): Either a single Seed for the subject table, or a dictionary with table names as keys and Seeds as values. Seed can either be a DataFrame or a path to a CSV or PARQUET file. Check generator details (`generator.tables[i].columns[j].value_range`) for possible value ranges.
            name (str | None): Name of the synthetic dataset.
            start (bool): Whether to start generation immediately. Default is True.
            wait (bool): Whether to wait for generation to finish. Default is True.
            progress_bar (bool): Whether to display a progress bar during generation. Default is True.

        Returns:
            SyntheticDataset: The created synthetic dataset.

        Example configuration using short-hand notation:
            ```python
            from mostlyai.sdk import MostlyAI
            mostly = MostlyAI()
            sd = mostly.generate(generator=g, size=1000)
            ```

        Example configuration using a dictionary:
            ```python
            from mostlyai.sdk import MostlyAI
            mostly = MostlyAI()
            sd = mostly.generate(
                generator=g,
                config={
                    'tables': [
                        {
                            'name': 'data',
                            'configuration': {  # all parameters are optional!
                                'sample_size': None,  # set to None to generate as many samples as original; otherwise, set to an integer; only applicable for subject tables
                                # 'sample_seed_data': seed_df,  # provide a DataFrame to conditionally generate samples; only applicable for subject tables
                                'sampling_temperature': 1.0,
                                'sampling_top_p': 1.0,
                                'rebalancing': {
                                    'column': 'age',
                                    'probabilities': {'male': 0.5, 'female': 0.5},
                                },
                                'imputation': {
                                    'columns': ['age'],
                                },
                                'fairness': {
                                    'target_column': 'income',
                                    'sensitive_columns': ['gender'],
                                },
                                'enable_data_report': True,  # disable for faster generation
                            }
                        }
                    ]
                }
            )
            ```
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
        generator: Generator | str,
        size: int | dict[str, int] | None = None,
        seed: Seed | dict[str, Seed] | None = None,
        config: SyntheticProbeConfig | dict | None = None,
        return_type: Literal["auto", "dict"] = "auto",
    ) -> pd.DataFrame | dict[str, pd.DataFrame]:
        """
        Probe a generator for a new synthetic dataset (synchronously).

        Args:
            generator (Generator | str): The generator instance or its UUID.
            size (int | dict[str, int] | None): Sample size(s) for the subject table(s). Default is 1, if no seed is provided.
            seed (Seed | dict[str, Seed] | None): Either a single Seed for the subject table, or a dictionary with table names as keys and Seeds as values. Seed can either be a DataFrame or a path to a CSV or PARQUET file. Check generator details (`generator.tables[i].columns[j].value_range`) for possible value ranges. Seed data may contain null values; columns listed in `imputation.columns` will have nulls generated, while unlisted columns keep nulls as-is.
            config (SyntheticProbeConfig | dict | None): Configuration for the probe.
            return_type (Literal["auto", "dict"]): The type of the return value. "dict" will always provide a dictionary of DataFrames. "auto" will return a single DataFrame for a single-table generator, and a dictionary of DataFrames for a multi-table generator. Default is "auto".

        Returns:
            pd.DataFrame | dict[str, pd.DataFrame]: The created synthetic probe. See return_type for the format of the return value.

        Example for probing a generator for 10 synthetic samples:
            ```python
            from mostlyai.sdk import MostlyAI
            mostly = MostlyAI()
            data = mostly.probe('INSERT_YOUR_GENERATOR_ID', size=10, return_type="dict")
            ```

        Example for conditional probing based on a seed DataFrame:
            ```python
            import pandas as pd
            from mostlyai.sdk import MostlyAI
            mostly = MostlyAI()
            seed = pd.DataFrame({'col1': ['x', 'y'], 'col2': [13, 74]})
            data = mostly.probe('INSERT_YOUR_GENERATOR_ID', seed=seed, return_type="dict")
            ```

        Example for advanced probing configuration:
            ```python
            import pandas as pd
            from mostlyai.sdk import MostlyAI
            mostly = MostlyAI()
            # seed with nulls for columns to be imputed
            seed = pd.DataFrame({'country': ['US', 'CA'], 'age': [None, None], 'income': [50000, None]})
            data = mostly.probe(
                'INSERT_YOUR_GENERATOR_ID',
                seed=seed,
                config={
                    'tables': [{
                        'name': 'tbl1',
                        'configuration': {
                            'sample_size': 100,
                            'sampling_temperature': 1.0,
                            'sampling_top_p': 1.0,
                            'rebalancing': {'column': 'country', 'probabilities': {'US': 0.5, 'CA': 0.3}},
                            'imputation': {'columns': ['age']},  # impute age nulls; income null stays as-is
                            'fairness': {'target_column': 'income', 'sensitive_columns': ['gender']},
                        }
                    }]
                },
                return_type="dict"
            )
            ```

        Example for multi-table conditional probing (e.g., time-series with 100 simulations):
            ```python
            # create 100 simulations for a specific user profile and first 2 purchases
            user_ids = [f"sim-{i:03d}" for i in range(100)]
            seed_users = pd.DataFrame({'users_id': user_ids})
            seed_purchases = pd.DataFrame({
                'users_id': [uid for uid in user_ids for _ in range(2)],
                'date': pd.to_datetime(['1997-01-12', '1997-01-12'] * 100),
                'cds': [1, 5] * 100,
                'amt': [12.00, 77.00] * 100,
            })
            data = mostly.probe(
                'INSERT_YOUR_GENERATOR_ID',
                seed={'users': seed_users, 'purchases': seed_purchases}
            )
            # Note: For multi-table seeds, provide unique PK/FK values to match records between tables
            ```
        """
        config = harmonize_sd_config(
            generator,
            get_generator=self.generators.get,
            size=size,
            seed=seed,
            config=config,
            config_type=SyntheticProbeConfig,
        )

        try:
            dfs = self.synthetic_probes.create(config)
        except APIError as e:
            # translate timeout error into a more informative message for probe requests
            if "timed out" in str(e).lower():
                raise APIError("Probing timed out. Please try `generate()` instead.")
            else:
                raise e

        if return_type == "auto" and len(dfs) == 1:
            return list(dfs.values())[0]
        else:
            return dfs

    def me(self) -> CurrentUser:
        """
        Retrieve information about the current user.

        Returns:
            CurrentUser: Information about the current user.

        Example for retrieving information about the current user:
            ```python
            from mostlyai.sdk import MostlyAI
            mostly = MostlyAI()
            mostly.me()
            # {'id': '488f2f26-...', 'first_name': 'Tom', ...}
            ```
        """
        return self.request(verb=GET, path=["users", "me"], response_type=CurrentUser)

    def about(self) -> AboutService:
        """
        Retrieve information about the platform.

        Returns:
            AboutService: Information about the platform.

        Example for retrieving information about the platform:
            ```python
            from mostlyai.sdk import MostlyAI
            mostly = MostlyAI()
            mostly.about()
            # {'version': 'v316', 'assistant': True}
            ```
        """
        return self.request(verb=GET, path=["about"], response_type=AboutService)

    def models(self) -> dict[str : list[str]]:
        """
        Retrieve a list of available models of a specific type.

        Returns:
            dict[str, list[str]]: A dictionary with list of available models for each ModelType.

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
        """
        return {model_type.value: self.request(verb=GET, path=["models", model_type.value]) for model_type in ModelType}

    def computes(self) -> list[dict[str, Any]]:
        """
        Retrieve a list of available compute resources, that can be used for executing tasks.
        Returns:
            list[dict[str, Any]]: A list of available compute resources.

        Example for retrieving available compute resources:
            ```python
            from mostlyai.sdk import MostlyAI
            mostly = MostlyAI()
            mostly.computes()
            # [{'id': '...', 'name': 'CPU Large',...]
            ```
        """
        return self.request(verb=GET, path=["computes"])
