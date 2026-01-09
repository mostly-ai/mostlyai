# API Reference

## MOSTLY AI Client

Instantiate an SDK instance, either in CLIENT or in LOCAL mode.

Parameters:

| Name              | Type    | Description                                                            | Default                                                                                                                                |
| ----------------- | ------- | ---------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `base_url`        | \`str   | None\`                                                                 | The base URL. If not provided, env var MOSTLY_BASE_URL is used if available, otherwise https://app.mostly.ai.                          |
| `api_key`         | \`str   | None\`                                                                 | The API key for authenticating. If not provided, env var MOSTLY_API_KEY is used if available.                                          |
| `bearer_token`    | \`str   | None\`                                                                 | The bearer token for authenticating. If not provided, env var MOSTLY_BEARER_TOKEN is used if available. Takes precedence over api_key. |
| `local`           | \`bool  | None\`                                                                 | Whether to run in local mode or not. If not provided, user is prompted to choose between CLIENT and LOCAL mode.                        |
| `local_dir`       | \`str   | Path                                                                   | None\`                                                                                                                                 |
| `local_port`      | \`int   | None\`                                                                 | The port to use for local mode with TCP transport. If not provided, UDS transport is used.                                             |
| `timeout`         | `float` | Timeout for HTTPS requests in seconds. Default is 60 seconds.          | `60.0`                                                                                                                                 |
| `ssl_verify`      | `bool`  | Whether to verify SSL certificates. Default is True.                   | `True`                                                                                                                                 |
| `test_connection` | `bool`  | Whether to test the connection during initialization. Default is True. | `True`                                                                                                                                 |
| `quiet`           | `bool`  | Whether to suppress rich output. Default is False.                     | `False`                                                                                                                                |

Example for SDK in CLIENT mode with explicit arguments

```python
from mostlyai.sdk import MostlyAI
mostly = MostlyAI(
    api_key='INSERT_YOUR_API_KEY',
    base_url='https://app.mostly.ai',
)
mostly
# MostlyAI(base_url='https://app.mostly.ai', api_key='***')
```

Example for SDK in CLIENT mode with bearer token

```python
from mostlyai.sdk import MostlyAI
mostly = MostlyAI(
    bearer_token='INSERT_YOUR_BEARER_TOKEN',
    base_url='https://app.mostly.ai',
)
mostly
# MostlyAI(base_url='https://app.mostly.ai', bearer_token='***')
```

Example for SDK in CLIENT mode with environment variables

```python
import os
from mostlyai.sdk import MostlyAI
os.environ["MOSTLY_API_KEY"] = "INSERT_YOUR_API_KEY"
os.environ["MOSTLY_BASE_URL"] = "https://app.mostly.ai"
mostly = MostlyAI()
mostly
# MostlyAI(base_url='https://app.mostly.ai', api_key='***')
```

Example for SDK in CLIENT mode with bearer token environment variable

```python
import os
from mostlyai.sdk import MostlyAI
os.environ["MOSTLY_BEARER_TOKEN"] = "INSERT_YOUR_BEARER_TOKEN"
os.environ["MOSTLY_BASE_URL"] = "https://app.mostly.ai"
mostly = MostlyAI()
mostly
# MostlyAI(base_url='https://app.mostly.ai', bearer_token='***')
```

Example for SDK in LOCAL mode connecting via UDS

```python
from mostlyai.sdk import MostlyAI
mostly = MostlyAI(local=True)
mostly
# MostlyAI(local=True)
```

Example for SDK in LOCAL mode connecting via TCP

```python
from mostlyai.sdk import MostlyAI
mostly = MostlyAI(local=True, local_port=8080)
mostly
# MostlyAI(local=True, local_port=8080)
```

### mostlyai.sdk.client.api.MostlyAI.about

```python
about()
```

Retrieve information about the platform.

Returns:

| Name           | Type           | Description                     |
| -------------- | -------------- | ------------------------------- |
| `AboutService` | `AboutService` | Information about the platform. |

Example for retrieving information about the platform

```python
from mostlyai.sdk import MostlyAI
mostly = MostlyAI()
mostly.about()
# {'version': 'v316', 'assistant': True}
```

### mostlyai.sdk.client.api.MostlyAI.computes

```python
computes()
```

Retrieve a list of available compute resources, that can be used for executing tasks. Returns: list\[dict[str, Any]\]: A list of available compute resources.

Example for retrieving available compute resources

```python
from mostlyai.sdk import MostlyAI
mostly = MostlyAI()
mostly.computes()
# [{'id': '...', 'name': 'CPU Large',...]
```

### mostlyai.sdk.client.api.MostlyAI.connect

```python
connect(config, test_connection=True)
```

Create a connector and optionally validate the connection before saving.

There are 3 access types for a connector (which are independent of the connector type):

- `READ_PROTECTED`: The connector is restricted to being used solely as a source for training a generator. Direct data access is not permitted, only schema access via `c.locations(prefix)` and `c.schema(location)` is available.
- `READ_DATA`: This connector allows full read access. It can also be used as a source for training a generator.
- `WRITE_DATA`: This connector allows full read and write access. It can be also used as a source for training a generator, as well as a destination for delivering a synthetic dataset.

Parameters:

| Name              | Type              | Description      | Default                                                                                              |
| ----------------- | ----------------- | ---------------- | ---------------------------------------------------------------------------------------------------- |
| `config`          | \`ConnectorConfig | dict[str, Any]\` | Configuration for the connector. Can be either a ConnectorConfig object or an equivalent dictionary. |
| `test_connection` | \`bool            | None\`           | Whether to validate the connection before saving. Default is True.                                   |

Returns:

| Name        | Type        | Description            |
| ----------- | ----------- | ---------------------- |
| `Connector` | `Connector` | The created connector. |

Example for creating a connector to a AWS S3 storage

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

### mostlyai.sdk.client.api.MostlyAI.generate

```python
generate(
    generator,
    config=None,
    size=None,
    seed=None,
    name=None,
    start=True,
    wait=True,
    progress_bar=True,
)
```

Create a synthetic dataset resource. Once generated, it will include the data as well as optionally a data report.

Note: A synthetic dataset is initially being configured. That generation job can be either launched immediately or later. One can check progress via `sd.generation.progress()`. Once the job has finished, the synthetic dataset is available for download via `sd.data()`, and the reports are available via `sd.reports()`.

Parameters:

| Name           | Type                     | Description                                                           | Default                             |
| -------------- | ------------------------ | --------------------------------------------------------------------- | ----------------------------------- |
| `generator`    | \`Generator              | str\`                                                                 | The generator instance or its UUID. |
| `config`       | \`SyntheticDatasetConfig | dict                                                                  | None\`                              |
| `size`         | \`int                    | dict[str, int]                                                        | None\`                              |
| `seed`         | \`Seed                   | dict[str, Seed]                                                       | None\`                              |
| `name`         | \`str                    | None\`                                                                | Name of the synthetic dataset.      |
| `start`        | `bool`                   | Whether to start generation immediately. Default is True.             | `True`                              |
| `wait`         | `bool`                   | Whether to wait for generation to finish. Default is True.            | `True`                              |
| `progress_bar` | `bool`                   | Whether to display a progress bar during generation. Default is True. | `True`                              |

Returns:

| Name               | Type               | Description                    |
| ------------------ | ------------------ | ------------------------------ |
| `SyntheticDataset` | `SyntheticDataset` | The created synthetic dataset. |

Example configuration using short-hand notation

```python
from mostlyai.sdk import MostlyAI
mostly = MostlyAI()
sd = mostly.generate(generator=g, size=1000)
```

Example configuration using a dictionary

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

### mostlyai.sdk.client.api.MostlyAI.me

```python
me()
```

Retrieve information about the current user.

Returns:

| Name          | Type          | Description                         |
| ------------- | ------------- | ----------------------------------- |
| `CurrentUser` | `CurrentUser` | Information about the current user. |

Example for retrieving information about the current user

```python
from mostlyai.sdk import MostlyAI
mostly = MostlyAI()
mostly.me()
# {'id': '488f2f26-...', 'first_name': 'Tom', ...}
```

### mostlyai.sdk.client.api.MostlyAI.models

```python
models()
```

Retrieve a list of available models of a specific type.

Returns:

| Type                    | Description                                                                            |
| ----------------------- | -------------------------------------------------------------------------------------- |
| `dict[str:(list[str])]` | dict\[str, list[str]\]: A dictionary with list of available models for each ModelType. |

Example for retrieving available models

```python
from mostlyai.sdk import MostlyAI
mostly = MostlyAI()
mostly.models()
# {
#    'TABULAR": ['MOSTLY_AI/Small', 'MOSTLY_AI/Medium', 'MOSTLY_AI/Large'],
#    'LANGUAGE": ['MOSTLY_AI/LSTMFromScratch-3m', 'microsoft/phi-1_5', ..],
# }
```

### mostlyai.sdk.client.api.MostlyAI.probe

```python
probe(
    generator,
    size=None,
    seed=None,
    config=None,
    return_type="auto",
)
```

Probe a generator for a new synthetic dataset (synchronously).

Parameters:

| Name          | Type                      | Description                                                                                                                                                                                                                             | Default                             |
| ------------- | ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------- |
| `generator`   | \`Generator               | str\`                                                                                                                                                                                                                                   | The generator instance or its UUID. |
| `size`        | \`int                     | dict[str, int]                                                                                                                                                                                                                          | None\`                              |
| `seed`        | \`Seed                    | dict[str, Seed]                                                                                                                                                                                                                         | None\`                              |
| `return_type` | `Literal['auto', 'dict']` | The type of the return value. "dict" will always provide a dictionary of DataFrames. "auto" will return a single DataFrame for a single-table generator, and a dictionary of DataFrames for a multi-table generator. Default is "auto". | `'auto'`                            |

Returns:

| Type        | Description            |
| ----------- | ---------------------- |
| \`DataFrame | dict[str, DataFrame]\` |

Example for probing a generator for 10 synthetic samples

```python
from mostlyai.sdk import MostlyAI
mostly = MostlyAI()
data = mostly.probe('INSERT_YOUR_GENERATOR_ID', size=10, return_type="dict")
```

Example for conditional probing based on a seed DataFrame

```python
import pandas as pd
from mostlyai.sdk import MostlyAI
mostly = MostlyAI()
seed = pd.DataFrame({'col1': ['x', 'y'], 'col2': [13, 74]})
data = mostly.probe('INSERT_YOUR_GENERATOR_ID', seed=seed, return_type="dict")
```

Example for advanced probing configuration

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

### mostlyai.sdk.client.api.MostlyAI.train

```python
train(
    config=None,
    data=None,
    name=None,
    start=True,
    wait=True,
    progress_bar=True,
)
```

Create a generator resource. Once trained, it will include the model as well as optionally a model report.

Note: A generator is initially being configured. That training job can be either launched immediately or later. One can check progress via `g.training.progress()`. Once the job has finished, the generator is available for use.

Parameters:

| Name           | Type              | Description                                                         | Default                |
| -------------- | ----------------- | ------------------------------------------------------------------- | ---------------------- |
| `config`       | \`GeneratorConfig | dict                                                                | None\`                 |
| `data`         | \`DataFrame       | str                                                                 | Path                   |
| `name`         | \`str             | None\`                                                              | Name of the generator. |
| `start`        | `bool`            | Whether to start training immediately. Default is True.             | `True`                 |
| `wait`         | `bool`            | Whether to wait for training to finish. Default is True.            | `True`                 |
| `progress_bar` | `bool`            | Whether to display a progress bar during training. Default is True. | `True`                 |

Returns:

| Name        | Type        | Description            |
| ----------- | ----------- | ---------------------- |
| `Generator` | `Generator` | The created generator. |

Example of a single flat table with default configurations

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

Example of a single flat table with custom configurations

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

Example of a multi-table relational dataset with non-context foreign key

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

Example of a single flat table with TABULAR and LANGUAGE models

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

Example with constraints to preserve valid combinations and enforce inequalities

```python
# constraints ensure that synthetic data only contains combinations of values that existed in training data
# and enforce logical relationships like departure_time < arrival_time
import numpy as np
import pandas as pd
departure_times = pd.date_range(start='2024-01-01 08:00', periods=40, freq='2h')
flight_durations = np.clip(np.random.normal(2.5, 0.2, 40), 2, 3)
df = pd.DataFrame({
    'origin_airport': ['JFK', 'JFK', 'LAX', 'LAX'] * 10,
    'destination_airport': ['LAX', 'ORD', 'ORD', 'JFK'] * 10,
    'departure_time': departure_times,
    'arrival_time': departure_times + pd.to_timedelta(flight_durations, unit='h'),
})
from mostlyai.sdk import MostlyAI
mostly = MostlyAI()
g = mostly.train(
    config={
        'name': 'flights',
        'tables': [{
            'name': 'flights',
            'data': df,
        }],
        'constraints': [
            {'type': 'FixedCombinations', 'config': {'table_name': 'flights', 'columns': ['origin_airport', 'destination_airport']}},  # ensures valid route combinations
            {'type': 'Inequality', 'config': {'table_name': 'flights', 'low_column': 'departure_time', 'high_column': 'arrival_time'}},  # ensures departure <= arrival
        ]
    },
    start=True,
    wait=True
)
# synthetic data will never generate impossible combinations like: origin='JFK', destination='JFK'
# and will always satisfy departure_time < arrival_time
```

## Generators

### mostlyai.sdk.client.generators.\_MostlyGeneratorsClient.create

```python
create(config)
```

Create a generator. The generator will be in the NEW state and will need to be trained before it can be used.

See [`mostly.train`](https://mostly-ai.github.io/mostlyai/api_client/#mostlyai.sdk.client.api.MostlyAI.train) for more details.

Parameters:

| Name     | Type              | Description | Default                          |
| -------- | ----------------- | ----------- | -------------------------------- |
| `config` | \`GeneratorConfig | dict\`      | Configuration for the generator. |

Returns:

| Type        | Description                   |
| ----------- | ----------------------------- |
| `Generator` | The created generator object. |

Example for creating a generator

```python
from mostlyai.sdk import MostlyAI
mostly = MostlyAI()
g = mostly.generators.create(
    config={
        "name": "US Census",
        "tables": [{
            "name": "census",
            "data": trn_df,
        }]
    )
)
print("status:", g.training_status)
# status: NEW
g.training.start()  # start training
print("status:", g.training_status)
# status: QUEUED
g.training.wait()   # wait for training to complete
print("status:", g.training_status)
# status: DONE
```

### mostlyai.sdk.client.generators.\_MostlyGeneratorsClient.get

```python
get(generator_id)
```

Retrieve a generator by its ID.

Parameters:

| Name           | Type  | Description                             | Default    |
| -------------- | ----- | --------------------------------------- | ---------- |
| `generator_id` | `str` | The unique identifier of the generator. | *required* |

Returns:

| Name        | Type        | Description                     |
| ----------- | ----------- | ------------------------------- |
| `Generator` | `Generator` | The retrieved generator object. |

Example for retrieving a generator

```python
from mostlyai.sdk import MostlyAI
mostly = MostlyAI()
g = mostly.generators.get('INSERT_YOUR_GENERATOR_ID')
g
```

### mostlyai.sdk.client.generators.\_MostlyGeneratorsClient.import_from_file

```python
import_from_file(file_path)
```

Import a generator from a file.

Parameters:

| Name        | Type  | Description | Default                                            |
| ----------- | ----- | ----------- | -------------------------------------------------- |
| `file_path` | \`str | Path\`      | Local file path or URL of the generator to import. |

Returns:

| Type        | Description                    |
| ----------- | ------------------------------ |
| `Generator` | The imported generator object. |

Example

```python
from mostlyai.sdk import MostlyAI
mostly = MostlyAI()

# Import from local file
g = mostly.generators.import_from_file('path/to/generator')

# Or import from URL
g = mostly.generators.import_from_file('https://example.com/path/to/generator.zip')
```

### mostlyai.sdk.client.generators.\_MostlyGeneratorsClient.list

```python
list(
    offset=0,
    limit=None,
    status=None,
    search_term=None,
    owner_id=None,
    visibility=None,
    created_from=None,
    created_to=None,
    sort_by=None,
)
```

List generators.

Paginate through all generators accessible by the user.

Parameters:

| Name           | Type  | Description                              | Default                                                                  |
| -------------- | ----- | ---------------------------------------- | ------------------------------------------------------------------------ |
| `offset`       | `int` | Offset for the entities in the response. | `0`                                                                      |
| `limit`        | \`int | None\`                                   | Limit for the number of entities in the response.                        |
| `status`       | \`str | list[str]                                | None\`                                                                   |
| `search_term`  | \`str | None\`                                   | Filter by name or description.                                           |
| `owner_id`     | \`str | list[str]                                | None\`                                                                   |
| `visibility`   | \`str | list[str]                                | None\`                                                                   |
| `created_from` | \`str | None\`                                   | Filter by creation date, not older than this date. Format: YYYY-MM-DD.   |
| `created_to`   | \`str | None\`                                   | Filter by creation date, not younger than this date. Format: YYYY-MM-DD. |
| `sort_by`      | \`str | list[str]                                | None\`                                                                   |

Returns:

| Type                          | Description                                                           |
| ----------------------------- | --------------------------------------------------------------------- |
| `Iterator[GeneratorListItem]` | Iterator\[GeneratorListItem\]: An iterator over generator list items. |

Example for listing all generators

```python
from mostlyai.sdk import MostlyAI
mostly = MostlyAI()
for g in mostly.generators.list():
    print(f"Generator `{g.name}` ({g.training_status}, {g.id})")
```

Example for searching trained generators via key word

```python
from mostlyai.sdk import MostlyAI
mostly = MostlyAI()
generators = list(mostly.generators.list(search_term="census", status="DONE"))
print(f"Found {len(generators)} generators")
```

## Generator

A generator is a set models that can generate synthetic data.

The generator can be trained on one or more source tables. A quality assurance report is generated for each model.

Parameters:

| Name              | Type                | Description                           | Default                                                                                                                                                                                                                                                                                              |
| ----------------- | ------------------- | ------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `id`              | `str`               | The unique identifier of a generator. | *required*                                                                                                                                                                                                                                                                                           |
| `name`            | `str`               | The name of a generator.              | *required*                                                                                                                                                                                                                                                                                           |
| `description`     | \`str               | None\`                                | The description of a generator.                                                                                                                                                                                                                                                                      |
| `training_status` | `ProgressStatus`    |                                       | *required*                                                                                                                                                                                                                                                                                           |
| `training_time`   | \`AwareDatetime     | None\`                                | The UTC date and time when the training has finished.                                                                                                                                                                                                                                                |
| `usage`           | \`GeneratorUsage    | None\`                                |                                                                                                                                                                                                                                                                                                      |
| `metadata`        | \`Metadata          | None\`                                |                                                                                                                                                                                                                                                                                                      |
| `accuracy`        | \`float             | None\`                                | The overall accuracy of the trained generator. This is the average of the overall accuracy scores of all trained models.                                                                                                                                                                             |
| `tables`          | \`list[SourceTable] | None\`                                | The tables of this generator                                                                                                                                                                                                                                                                         |
| `constraints`     | \`list[Constraint]  | None\`                                | The data constraints to apply.                                                                                                                                                                                                                                                                       |
| `random_state`    | \`int               | None\`                                | Seed for the random number generators. If None, the random number generator is initialized randomly, yielding different results for every run. Setting it to a specific integer ensures reproducible results across runs. Useful when consistent results are desired, e.g. for testing or debugging. |
| `training`        | \`Any               | None\`                                |                                                                                                                                                                                                                                                                                                      |

### mostlyai.sdk.domain.Generator.Training

#### mostlyai.sdk.domain.Generator.Training.cancel

```python
cancel()
```

Cancel training.

#### mostlyai.sdk.domain.Generator.Training.logs

```python
logs(file_path=None)
```

Download the training logs and save to file.

Parameters:

| Name        | Type  | Description | Default |
| ----------- | ----- | ----------- | ------- |
| `file_path` | \`str | Path        | None\`  |

Returns:

| Name   | Type   | Description                 |
| ------ | ------ | --------------------------- |
| `Path` | `Path` | The path to the saved file. |

#### mostlyai.sdk.domain.Generator.Training.progress

```python
progress()
```

Retrieve job progress of training.

Returns:

| Name          | Type          | Description                               |
| ------------- | ------------- | ----------------------------------------- |
| `JobProgress` | `JobProgress` | The job progress of the training process. |

#### mostlyai.sdk.domain.Generator.Training.start

```python
start()
```

Start training.

#### mostlyai.sdk.domain.Generator.Training.wait

```python
wait(progress_bar=True, interval=2)
```

Poll training progress and loop until training has completed.

Parameters:

| Name           | Type    | Description                                                             | Default |
| -------------- | ------- | ----------------------------------------------------------------------- | ------- |
| `progress_bar` | `bool`  | If true, displays the progress bar. Default is True.                    | `True`  |
| `interval`     | `float` | The interval in seconds to poll the job progress. Default is 2 seconds. | `2`     |

### mostlyai.sdk.domain.Generator.clone

```python
clone(training_status='new')
```

Clone the generator.

Parameters:

| Name              | Type                         | Description                                                    | Default |
| ----------------- | ---------------------------- | -------------------------------------------------------------- | ------- |
| `training_status` | `Literal['new', 'continue']` | The training status of the cloned generator. Default is "new". | `'new'` |

Returns:

| Name        | Type        | Description                  |
| ----------- | ----------- | ---------------------------- |
| `Generator` | `Generator` | The cloned generator object. |

### mostlyai.sdk.domain.Generator.config

```python
config()
```

Retrieve writable generator properties.

Returns:

| Name              | Type              | Description                                         |
| ----------------- | ----------------- | --------------------------------------------------- |
| `GeneratorConfig` | `GeneratorConfig` | The generator properties as a configuration object. |

### mostlyai.sdk.domain.Generator.delete

```python
delete()
```

Delete the generator.

### mostlyai.sdk.domain.Generator.export_to_file

```python
export_to_file(file_path=None)
```

Export generator and save to file.

Parameters:

| Name        | Type  | Description | Default |
| ----------- | ----- | ----------- | ------- |
| `file_path` | \`str | Path        | None\`  |

Returns:

| Name   | Type   | Description                 |
| ------ | ------ | --------------------------- |
| `Path` | `Path` | The path to the saved file. |

### mostlyai.sdk.domain.Generator.reports

```python
reports(file_path=None, display=False)
```

Download or display the quality assurance reports.

If display is True, the report is rendered inline via IPython display and no file is downloaded. Otherwise, the report is downloaded and saved to file_path (or a default location if None).

Note that reports are not available for generators that were trained with less than 100 samples or had `enable_model_report` set to `False`.

Parameters:

| Name        | Type   | Description                                                  | Default |
| ----------- | ------ | ------------------------------------------------------------ | ------- |
| `file_path` | \`str  | Path                                                         | None\`  |
| `display`   | `bool` | If True, render the report inline instead of downloading it. | `False` |

Returns:

| Type   | Description |
| ------ | ----------- |
| \`Path | None\`      |

### mostlyai.sdk.domain.Generator.update

```python
update(name=None, description=None)
```

Update a generator with specific parameters.

Parameters:

| Name          | Type  | Description | Default                           |
| ------------- | ----- | ----------- | --------------------------------- |
| `name`        | \`str | None\`      | The name of the generator.        |
| `description` | \`str | None\`      | The description of the generator. |

## Synthetic Datasets

### mostlyai.sdk.client.synthetic_datasets.\_MostlySyntheticDatasetsClient.create

```python
create(config)
```

Create a synthetic dataset. The synthetic dataset will be in the NEW state and will need to be generated before it can be used.

See [`mostly.generate`](https://mostly-ai.github.io/mostlyai/api_client/#mostlyai.sdk.client.api.MostlyAI.generate) for more details.

Parameters:

| Name     | Type                     | Description      | Default                                  |
| -------- | ------------------------ | ---------------- | ---------------------------------------- |
| `config` | \`SyntheticDatasetConfig | dict[str, Any]\` | Configuration for the synthetic dataset. |

Returns:

| Type               | Description                           |
| ------------------ | ------------------------------------- |
| `SyntheticDataset` | The created synthetic dataset object. |

Example for creating a synthetic dataset

```python
from mostlyai.sdk import MostlyAI
mostly = MostlyAI()
sd = mostly.synthetic_datasets.create(
    config=SyntheticDatasetConfig(
        generator_id="INSERT_YOUR_GENERATOR_ID",
    )
)
print("status:", sd.generation_status)
# status: NEW
sd.generation.start()  # start generation
print("status:", sd.generation_status)
# status: QUEUED
sd.generation.wait()   # wait for generation to complete
print("status:", sd.generation_status)
# status: DONE
```

### mostlyai.sdk.client.synthetic_datasets.\_MostlySyntheticDatasetsClient.get

```python
get(synthetic_dataset_id)
```

Retrieve a synthetic dataset by its ID.

Parameters:

| Name                   | Type  | Description                                     | Default    |
| ---------------------- | ----- | ----------------------------------------------- | ---------- |
| `synthetic_dataset_id` | `str` | The unique identifier of the synthetic dataset. | *required* |

Returns:

| Name               | Type               | Description                             |
| ------------------ | ------------------ | --------------------------------------- |
| `SyntheticDataset` | `SyntheticDataset` | The retrieved synthetic dataset object. |

Example for retrieving a synthetic dataset

```python
from mostlyai.sdk import MostlyAI
mostly = MostlyAI()
sd = mostly.synthetic_datasets.get('INSERT_YOUR_SYNTHETIC_DATASET_ID')
sd
```

### mostlyai.sdk.client.synthetic_datasets.\_MostlySyntheticDatasetsClient.list

```python
list(
    offset=0,
    limit=None,
    status=None,
    search_term=None,
    owner_id=None,
    visibility=None,
    created_from=None,
    created_to=None,
    sort_by=None,
)
```

List synthetic datasets.

Paginate through all synthetic datasets accessible by the user.

Parameters:

| Name           | Type  | Description                              | Default                                                                  |
| -------------- | ----- | ---------------------------------------- | ------------------------------------------------------------------------ |
| `offset`       | `int` | Offset for the entities in the response. | `0`                                                                      |
| `limit`        | \`int | None\`                                   | Limit for the number of entities in the response.                        |
| `status`       | \`str | list[str]                                | None\`                                                                   |
| `search_term`  | \`str | None\`                                   | Filter by name or description.                                           |
| `owner_id`     | \`str | list[str]                                | None\`                                                                   |
| `visibility`   | \`str | list[str]                                | None\`                                                                   |
| `created_from` | \`str | None\`                                   | Filter by creation date, not older than this date. Format: YYYY-MM-DD.   |
| `created_to`   | \`str | None\`                                   | Filter by creation date, not younger than this date. Format: YYYY-MM-DD. |
| `sort_by`      | \`str | list[str]                                | None\`                                                                   |

Returns:

| Type                                 | Description                          |
| ------------------------------------ | ------------------------------------ |
| `Iterator[SyntheticDatasetListItem]` | An iterator over synthetic datasets. |

Example for listing all synthetic datasets

```python
from mostlyai.sdk import MostlyAI
mostly = MostlyAI()
for sd in mostly.synthetic_datasets.list():
    print(f"Synthetic Dataset `{sd.name}` ({sd.generation_status}, {sd.id})")
```

Example for searching generated synthetic datasets via key word

```python
from mostlyai.sdk import MostlyAI
mostly = MostlyAI()
datasets = list(mostly.synthetic_datasets.list(search_term="census", status="DONE"))
print(f"Found {len(datasets)} synthetic datasets")
```

## Synthetic Dataset

A synthetic dataset is created based on a trained generator.

It consists of synthetic samples, as well as a quality assurance report.

Parameters:

| Name                | Type                       | Description                                   | Default                                                                                                                                                                                                                                                                                              |
| ------------------- | -------------------------- | --------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `id`                | `str`                      | The unique identifier of a synthetic dataset. | *required*                                                                                                                                                                                                                                                                                           |
| `generator_id`      | \`str                      | None\`                                        | The unique identifier of a generator.                                                                                                                                                                                                                                                                |
| `metadata`          | \`Metadata                 | None\`                                        |                                                                                                                                                                                                                                                                                                      |
| `name`              | `str`                      | The name of a synthetic dataset.              | *required*                                                                                                                                                                                                                                                                                           |
| `description`       | \`str                      | None\`                                        | The description of a synthetic dataset.                                                                                                                                                                                                                                                              |
| `generation_status` | `ProgressStatus`           |                                               | *required*                                                                                                                                                                                                                                                                                           |
| `generation_time`   | \`AwareDatetime            | None\`                                        | The UTC date and time when the generation has finished.                                                                                                                                                                                                                                              |
| `tables`            | \`list[SyntheticTable]     | None\`                                        | The tables of this synthetic dataset.                                                                                                                                                                                                                                                                |
| `delivery`          | \`SyntheticDatasetDelivery | None\`                                        |                                                                                                                                                                                                                                                                                                      |
| `accuracy`          | \`float                    | None\`                                        | The overall accuracy of the trained generator. This is the average of the overall accuracy scores of all trained models.                                                                                                                                                                             |
| `usage`             | \`SyntheticDatasetUsage    | None\`                                        |                                                                                                                                                                                                                                                                                                      |
| `compute`           | \`str                      | None\`                                        | The unique identifier of a compute resource. Not applicable for SDK.                                                                                                                                                                                                                                 |
| `random_state`      | \`int                      | None\`                                        | Seed for the random number generators. If None, the random number generator is initialized randomly, yielding different results for every run. Setting it to a specific integer ensures reproducible results across runs. Useful when consistent results are desired, e.g. for testing or debugging. |
| `generation`        | \`Any                      | None\`                                        |                                                                                                                                                                                                                                                                                                      |

### mostlyai.sdk.domain.SyntheticDataset.Generation

#### mostlyai.sdk.domain.SyntheticDataset.Generation.cancel

```python
cancel()
```

Cancel the generation process.

#### mostlyai.sdk.domain.SyntheticDataset.Generation.logs

```python
logs(file_path=None)
```

Download the generation logs and save to file.

Parameters:

| Name        | Type  | Description | Default |
| ----------- | ----- | ----------- | ------- |
| `file_path` | \`str | Path        | None\`  |

Returns:

| Name   | Type   | Description                 |
| ------ | ------ | --------------------------- |
| `Path` | `Path` | The path to the saved file. |

#### mostlyai.sdk.domain.SyntheticDataset.Generation.progress

```python
progress()
```

Retrieve the progress of the generation process.

Returns:

| Name          | Type          | Description                             |
| ------------- | ------------- | --------------------------------------- |
| `JobProgress` | `JobProgress` | The progress of the generation process. |

#### mostlyai.sdk.domain.SyntheticDataset.Generation.start

```python
start()
```

Start the generation process.

#### mostlyai.sdk.domain.SyntheticDataset.Generation.wait

```python
wait(progress_bar=True, interval=2)
```

Poll the generation progress and wait until the process is complete.

Parameters:

| Name           | Type    | Description                                                         | Default |
| -------------- | ------- | ------------------------------------------------------------------- | ------- |
| `progress_bar` | `bool`  | If true, displays a progress bar. Default is True.                  | `True`  |
| `interval`     | `float` | Interval in seconds to poll the job progress. Default is 2 seconds. | `2`     |

### mostlyai.sdk.domain.SyntheticDataset.config

```python
config()
```

Retrieve writable synthetic dataset properties.

Returns:

| Name                     | Type                     | Description                                                 |
| ------------------------ | ------------------------ | ----------------------------------------------------------- |
| `SyntheticDatasetConfig` | `SyntheticDatasetConfig` | The synthetic dataset properties as a configuration object. |

### mostlyai.sdk.domain.SyntheticDataset.data

```python
data(return_type='auto')
```

Download synthetic dataset and return as dictionary of pandas DataFrames.

Parameters:

| Name          | Type                      | Description                                                                                                                                                                                                                             | Default  |
| ------------- | ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| `return_type` | `Literal['auto', 'dict']` | The type of the return value. "dict" will always provide a dictionary of DataFrames. "auto" will return a single DataFrame for a single-table generator, and a dictionary of DataFrames for a multi-table generator. Default is "auto". | `'auto'` |

Returns:

| Type        | Description            |
| ----------- | ---------------------- |
| \`DataFrame | dict[str, DataFrame]\` |

### mostlyai.sdk.domain.SyntheticDataset.delete

```python
delete()
```

Delete the synthetic dataset.

### mostlyai.sdk.domain.SyntheticDataset.download

```python
download(file_path=None, format='parquet')
```

Download synthetic dataset and save to file.

Parameters:

| Name        | Type                                | Description                                                | Default     |
| ----------- | ----------------------------------- | ---------------------------------------------------------- | ----------- |
| `file_path` | \`str                               | Path                                                       | None\`      |
| `format`    | `Literal['parquet', 'csv', 'json']` | The format of the synthetic dataset. Default is "parquet". | `'parquet'` |

Returns:

| Name   | Type   | Description                 |
| ------ | ------ | --------------------------- |
| `Path` | `Path` | The path to the saved file. |

### mostlyai.sdk.domain.SyntheticDataset.reports

```python
reports(file_path=None, display=False)
```

Download or display the quality assurance reports.

If display is True, the report is rendered inline via IPython display and no file is downloaded. Otherwise, the report is downloaded and saved to file_path (or a default location if None).

Note that reports are not available for synthetic datasets that generated less than 100 samples or had `enable_data_report` set to `False`.

Parameters:

| Name        | Type   | Description                                                  | Default |
| ----------- | ------ | ------------------------------------------------------------ | ------- |
| `file_path` | \`str  | Path                                                         | None\`  |
| `display`   | `bool` | If True, render the report inline instead of downloading it. | `False` |

Returns:

| Type   | Description |
| ------ | ----------- |
| \`Path | None\`      |

### mostlyai.sdk.domain.SyntheticDataset.update

```python
update(name=None, description=None, delivery=None)
```

Update a synthetic dataset with specific parameters.

Parameters:

| Name          | Type                       | Description | Default                                               |
| ------------- | -------------------------- | ----------- | ----------------------------------------------------- |
| `name`        | \`str                      | None\`      | The name of the synthetic dataset.                    |
| `description` | \`str                      | None\`      | The description of the synthetic dataset.             |
| `delivery`    | \`SyntheticDatasetDelivery | None\`      | The delivery configuration for the synthetic dataset. |

## Connectors

### mostlyai.sdk.client.connectors.\_MostlyConnectorsClient.create

```python
create(config, test_connection=True)
```

Create a connector and optionally validate the connection before saving.

See [`mostly.connect`](https://mostly-ai.github.io/mostlyai/api_client/#mostlyai.sdk.client.api.MostlyAI.connect) for more details.

Parameters:

| Name              | Type              | Description      | Default                                                    |
| ----------------- | ----------------- | ---------------- | ---------------------------------------------------------- |
| `config`          | \`ConnectorConfig | dict[str, Any]\` | Configuration for the connector.                           |
| `test_connection` | \`bool            | None\`           | Whether to test the connection before saving the connector |

Returns:

| Type        | Description                   |
| ----------- | ----------------------------- |
| `Connector` | The created connector object. |

### mostlyai.sdk.client.connectors.\_MostlyConnectorsClient.get

```python
get(connector_id)
```

Retrieve a connector by its ID.

Parameters:

| Name           | Type  | Description                             | Default    |
| -------------- | ----- | --------------------------------------- | ---------- |
| `connector_id` | `str` | The unique identifier of the connector. | *required* |

Returns:

| Name        | Type        | Description                     |
| ----------- | ----------- | ------------------------------- |
| `Connector` | `Connector` | The retrieved connector object. |

Example for retrieving a connector

```python
from mostlyai.sdk import MostlyAI
mostly = MostlyAI()
c = mostly.connectors.get('INSERT_YOUR_CONNECTOR_ID')
c
```

### mostlyai.sdk.client.connectors.\_MostlyConnectorsClient.list

```python
list(
    offset=0,
    limit=None,
    search_term=None,
    access_type=None,
    owner_id=None,
    visibility=None,
    created_from=None,
    created_to=None,
    sort_by=None,
)
```

List connectors.

Paginate through all connectors accessible by the user. Only connectors that are independent of a table will be returned.

Example for listing all connectors

```python
from mostlyai.sdk import MostlyAI
mostly = MostlyAI()
for c in mostly.connectors.list():
    print(f"Connector `{c.name}` ({c.access_type}, {c.type}, {c.id})")
```

Parameters:

| Name           | Type  | Description                          | Default                                                                  |
| -------------- | ----- | ------------------------------------ | ------------------------------------------------------------------------ |
| `offset`       | `int` | Offset for entities in the response. | `0`                                                                      |
| `limit`        | \`int | None\`                               | Limit for the number of entities in the response.                        |
| `search_term`  | \`str | None\`                               | Filter by search term in the name and description.                       |
| `access_type`  | \`str | None\`                               | Filter by access type (e.g., READ_PROTECTED, READ_DATA or WRITE_DATA).   |
| `owner_id`     | \`str | list[str]                            | None\`                                                                   |
| `visibility`   | \`str | list[str]                            | None\`                                                                   |
| `created_from` | \`str | None\`                               | Filter by creation date, not older than this date. Format: YYYY-MM-DD.   |
| `created_to`   | \`str | None\`                               | Filter by creation date, not younger than this date. Format: YYYY-MM-DD. |
| `sort_by`      | \`str | list[str]                            | None\`                                                                   |

Returns:

| Type                          | Description                                                           |
| ----------------------------- | --------------------------------------------------------------------- |
| `Iterator[ConnectorListItem]` | Iterator\[ConnectorListItem\]: An iterator over connector list items. |

## Connector

A connector is a connection to a data source or a data destination.

Parameters:

| Name          | Type                  | Description                           | Default                                                                                                                                                                              |
| ------------- | --------------------- | ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `id`          | `str`                 | The unique identifier of a connector. | *required*                                                                                                                                                                           |
| `name`        | \`str                 | None\`                                | The name of a connector.                                                                                                                                                             |
| `description` | \`str                 | None\`                                | The description of a connector.                                                                                                                                                      |
| `type`        | `ConnectorType`       |                                       | *required*                                                                                                                                                                           |
| `access_type` | \`ConnectorAccessType | None\`                                |                                                                                                                                                                                      |
| `config`      | \`dict[str, Any]      | None\`                                |                                                                                                                                                                                      |
| `secrets`     | \`dict[str, str]      | None\`                                |                                                                                                                                                                                      |
| `ssl`         | \`dict[str, str]      | None\`                                |                                                                                                                                                                                      |
| `metadata`    | \`Metadata            | None\`                                |                                                                                                                                                                                      |
| `usage`       | \`ConnectorUsage      | None\`                                |                                                                                                                                                                                      |
| `table_id`    | \`str                 | None\`                                | Optional. ID of a source table or a synthetic table, that this connector belongs to. If not set, then this connector is managed independently of any generator or synthetic dataset. |

### mostlyai.sdk.domain.Connector.delete

```python
delete()
```

Delete the connector.

### mostlyai.sdk.domain.Connector.delete_data

```python
delete_data(location)
```

Delete data from the specified location within the connector. This method is only available for connectors of access_type WRITE_DATA.

Parameters:

| Name       | Type  | Description                                                   | Default    |
| ---------- | ----- | ------------------------------------------------------------- | ---------- |
| `location` | `str` | The target location within the connector to delete data from. | *required* |

Example

```python
c.delete_data('db_name.table_name')  # drop table data from 'table_name' in 'db_name' for a DB connector
c.delete_data('s3://my_bucket/path/to/file.csv')  # delete data from 'file.csv' in 'my_bucket' for a S3 storage connector
```

### mostlyai.sdk.domain.Connector.locations

```python
locations(prefix='')
```

List connector locations.

List the available databases, schemas, tables, or folders for a connector. For storage connectors, this returns list of buckets for empty prefix and otherwise the folders and files then within a path, specified via prefix. For DB connectors, this returns list of schemas (or databases for DBs without schema), respectively list of tables if `prefix` is provided.

The formats of the locations are:

- Cloud storage:
  - `AZURE_STORAGE`: `container/path`
  - `GOOGLE_CLOUD_STORAGE`: `bucket/path`
  - `S3_STORAGE`: `bucket/path`
- Database:
  - `BIGQUERY`: `dataset.table`
  - `DATABRICKS`: `schema.table`
  - `HIVE`: `database.table`
  - `MARIADB`: `database.table`
  - `MSSQL`: `schema.table`
  - `MYSQL`: `database.table`
  - `ORACLE`: `schema.table`
  - `POSTGRES`: `schema.table`
  - `SNOWFLAKE`: `schema.table`

Parameters:

| Name     | Type  | Description                                                       | Default |
| -------- | ----- | ----------------------------------------------------------------- | ------- |
| `prefix` | `str` | The prefix to filter the results by. Defaults to an empty string. | `''`    |

Returns:

| Type        | Description                                                               |
| ----------- | ------------------------------------------------------------------------- |
| `list[str]` | list\[str\]: A list of locations (schemas, databases, directories, etc.). |

Example

```python
c.locations()  # list all schemas / databases for a DB connector; list all buckets for a storage connector
c.locations('db_name')  # list all tables in 'db_name' for a DB connector
c.locations('s3://my_bucket')  # list all objects in 'my_bucket' for a S3 storage connector
c.locations('gs://my_bucket/path/to/folder')  # list all objects in 'my_bucket/path/to/folder' for a GCP storage connector
c.locations('az://my_container/path/to/folder')  # list all objects in 'my_container/path/to/folder' for a AZURE storage connector
```

### mostlyai.sdk.domain.Connector.query

```python
query(sql)
```

Execute a read-only SQL query against the connector's data source.

Queries can include statements like SELECT, SHOW, or DESCRIBE, but must not modify data or state. For file-based connectors (S3_STORAGE, GOOGLE_CLOUD_STORAGE, AZURE_STORAGE) queries are executed using DuckDB. Use connector-type-specific prefixes. See examples.

Parameters:

| Name  | Type  | Description               | Default    |
| ----- | ----- | ------------------------- | ---------- |
| `sql` | `str` | The SQL query to execute. | *required* |

Returns:

| Type        | Description                                                  |
| ----------- | ------------------------------------------------------------ |
| `DataFrame` | pd.DataFrame: The result of the query as a Pandas DataFrame. |

Example

```python
df = c.query("SELECT count(*) FROM schema.table")  # for DB connectors
df = c.query("SELECT count(*) FROM read_csv_auto('s3://bucket/path/to/file.csv')")  # query a single CSV file from S3 storage
df = c.query("SELECT count(*) FROM read_parquet('gs://bucket/path/to/folder/*.parquet')")  # query a folder with PQT files from GCP storage
df = c.query("SELECT count(*) FROM read_json_auto('az://bucket/path/to/file.json')")  # query a single JSON file from AZURE storage
```

### mostlyai.sdk.domain.Connector.read_data

```python
read_data(location, limit=None, shuffle=False)
```

Retrieve data from the specified location within the connector. This method is only available for connectors of access_type READ_DATA or WRITE_DATA.

Parameters:

| Name       | Type   | Description                                                 | Default                                                             |
| ---------- | ------ | ----------------------------------------------------------- | ------------------------------------------------------------------- |
| `location` | `str`  | The target location within the connector to read data from. | *required*                                                          |
| `limit`    | \`int  | None\`                                                      | The maximum number of rows to return. Returns all if not specified. |
| `shuffle`  | \`bool | None\`                                                      | Whether to shuffle the results.                                     |

Returns:

| Type        | Description                                              |
| ----------- | -------------------------------------------------------- |
| `DataFrame` | pd.DataFrame: A DataFrame containing the retrieved data. |

Example

```python
df = c.read_data('db_name.table_name', limit=100)  # fetch first 100 rows from 'table_name' in 'db_name' for a DB connector
df = c.read_data('s3://my_bucket/path/to/file.csv')  # read all data from 'file.csv' in 'my_bucket' for a S3 storage connector
```

### mostlyai.sdk.domain.Connector.schema

```python
schema(location)
```

Retrieve the schema of the table at a connector location. This method is available for all connectors.

Parameters:

| Name       | Type  | Description                | Default    |
| ---------- | ----- | -------------------------- | ---------- |
| `location` | `str` | The location of the table. | *required* |

Returns:

| Type                   | Description                                   |
| ---------------------- | --------------------------------------------- |
| `list[dict[str, Any]]` | list\[dict[str, Any]\]: The retrieved schema. |

Example

```python
c.schema('db_name.table_name')  # get the schema of 'table_name' in 'db_name' for a DB connector
c.schema('s3://my_bucket/path/to/file.csv')  # get the schema of 'file.csv' in 'my_bucket' for a S3 storage connector
```

### mostlyai.sdk.domain.Connector.update

```python
update(
    name=None,
    description=None,
    access_type=None,
    config=None,
    secrets=None,
    ssl=None,
    test_connection=True,
)
```

Update a connector with specific parameters.

Parameters:

| Name              | Type                  | Description | Default                                          |
| ----------------- | --------------------- | ----------- | ------------------------------------------------ |
| `name`            | \`str                 | None\`      | The name of the connector.                       |
| `description`     | \`str                 | None\`      | The description of the connector.                |
| `access_type`     | \`ConnectorAccessType | None\`      | The access type of the connector.                |
| `config`          | \`dict[str, Any]      | None\`      | Connector configuration.                         |
| `secrets`         | \`dict[str, str]      | None\`      | Secret values for the connector.                 |
| `ssl`             | \`dict[str, str]      | None\`      | SSL configuration for the connector.             |
| `test_connection` | \`bool                | None\`      | If true, validates the connection before saving. |

### mostlyai.sdk.domain.Connector.write_data

```python
write_data(data, location, if_exists='fail')
```

Write data to the specified location within the connector. This method is only available for connectors of access_type WRITE_DATA.

Parameters:

| Name        | Type                                   | Description                                                                                    | Default                                                 |
| ----------- | -------------------------------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| `data`      | \`DataFrame                            | None\`                                                                                         | The DataFrame to write, or None to delete the location. |
| `location`  | `str`                                  | The target location within the connector to write data to.                                     | *required*                                              |
| `if_exists` | `Literal['append', 'replace', 'fail']` | The behavior if the target location already exists (append, replace, fail). Default is "fail". | `'fail'`                                                |

Example

```python
c.write_data(df, 'db_name.table_name', if_exists='fail')  # write data to 'table_name' in 'db_name' for a DB connector (if it doesn't exist)
c.write_data(df, 's3://my_bucket/path/to/file.csv')  # write data to 'file.csv' in 'my_bucket' for a S3 storage connector
```
