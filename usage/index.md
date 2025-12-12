# Usage Examples

## Single-table tabular data

Train a tabular model on the US Census Income dataset, with differential privacy guarantees.

```python
import pandas as pd
from mostlyai.sdk import MostlyAI

# load original data
repo_url = 'https://github.com/mostly-ai/public-demo-data'
df_original = pd.read_csv(f'{repo_url}/raw/dev/census/census.csv.gz')

# instantiate SDK
mostly = MostlyAI()

# train a generator
g = mostly.train(config={
        'name': 'US Census Income',          # name of the generator
        'tables': [{                         # provide list of table(s)
            'name': 'census',                # name of the table
            'data': df_original,             # the original data as pd.DataFrame
            'tabular_model_configuration': { # tabular model configuration (optional)
                'max_training_time': 2,      # cap runtime for demo; set None for max accuracy
                # model, max_epochs,,..      # further model configurations (optional)
                'differential_privacy': {    # differential privacy configuration (optional)
                    'max_epsilon': 5.0,      # - max epsilon value, used as stopping criterion
                    'delta': 1e-5,           # - delta value
                }
            },
            # columns, keys, compute,..      # further table configurations (optional)
        }]
    },
    start=True,                              # start training immediately (default: True)
    wait=True,                               # wait for completion (default: True)
)
g
```

Probe the generator for 100 new synthetic samples.

```python
df_samples = mostly.probe(g, size=100)
df_samples
```

Probe the generator for a 28-year old male Cuban and a 44-year old female Mexican.

```python
df_samples = mostly.probe(g, seed=pd.DataFrame({
    'age': [28, 44],
    'sex': ['Male', 'Female'],
    'native_country': ['Cuba', 'Mexico'],
}))
df_samples
```

Create a new Synthetic Dataset via a batch job to conditionally generate 1'000'000 statistically representative synthetic samples.

```python
sd = mostly.generate(g, size=1_000_000)
df_synthetic = sd.data()
df_synthetic
```

## Time-series data

Train a two-table generator on a time-series dataset with a parent-child relationship.

```python
import pandas as pd
from mostlyai.sdk import MostlyAI

# load original time series data
repo_url = 'https://github.com/mostly-ai/public-demo-data/raw/refs/heads/dev/cdnow/'
df_original_purchases = pd.read_csv(f'{repo_url}/purchases.csv.gz')[['users_id', 'date', 'cds', 'amt']]

# ensure correct data type for DATE column
df_original_purchases['date'] = pd.to_datetime(df_original_purchases['date'])

# extract subject table from time-series data
df_original_users = df_original_purchases[['users_id']].drop_duplicates()

# instantiate SDK
mostly = MostlyAI()

# train a generator
g = mostly.train(config={
    'name': 'CDNOW',                      # name of the generator
    'tables': [{                          # provide list of table(s)
        'name': 'users',                  # name of the table
        'data': df_original_users,        # the original data as pd.DataFrame
        'primary_key': 'users_id',
    }, {
        'name': 'purchases',              # name of the table
        'data': df_original_purchases,    # the original data as pd.DataFrame
        'foreign_keys': [                 # foreign key configurations
            {'column': 'users_id', 'referenced_table': 'users', 'is_context': True},
        ],
        'tabular_model_configuration': {
            'max_training_time': 2,       # cap runtime for demo; set None for max accuracy
        },
    }],
})

# show Model QA reports
g.reports(display=True)
```

Generate a new dataset of 1k synthetic users and their synthetic purchases.

```python
syn = mostly.probe(g, size=1000)
syn['purchases'].sort_values(['users_id', 'date'])
```

Conditionally generate 100 synthetic simulations for a specific customer profile with their first 2 purchases.

> **Note:** For multi-table seeds, you must provide unique PK/FK values to match records between the seed tables.

```python
# define seed for 100 simulations for a specific user and her first 2 purchases
user_ids = [f"sim-{i:03d}" for i in range(100)]
seed_users = pd.DataFrame({
    'users_id': user_ids,
})
seed_purchases = pd.DataFrame({
    'users_id': [uid for uid in user_ids for _ in range(2)],
    'date': pd.to_datetime(['1997-01-12', '1997-01-12'] * 100),
    'cds': [1, 5] * 100,
    'amt': [12.00, 77.00] * 100,
})
syn = mostly.probe(g, seed={
    'users': seed_users,
    'purchases': seed_purchases,
})
syn['purchases'] = syn['purchases'].sort_values(['users_id', 'date'])
syn
```

## Multi-table tabular data

Train a 3-table tabular generator on baseball players and their seasonal statistics.

```python
import pandas as pd
from mostlyai.sdk import MostlyAI

# load original data
repo_url = 'https://github.com/mostly-ai/public-demo-data/raw/dev/baseball'
df_original_players = pd.read_csv(f'{repo_url}/players.csv.gz')[['id', 'country', 'weight', 'height']]
df_original_batting = pd.read_csv(f'{repo_url}/batting.csv.gz')[['players_id', 'G', 'AB', 'R', 'H', 'HR']]
df_original_fielding = pd.read_csv(f'{repo_url}/fielding.csv.gz')[['players_id', 'POS', 'G', 'PO', 'A', 'E', 'DP']]

# instantiate SDK
mostly = MostlyAI()

# train a generator
g = mostly.train(config={
    'name': 'Baseball',                   # name of the generator
    'tables': [{                          # provide list of table(s)
        'name': 'players',                # name of the table
        'data': df_original_players,      # the original data as pd.DataFrame
        'primary_key': 'id',
        'tabular_model_configuration': {
            'max_training_time': 2,       # cap runtime for demo; set None for max accuracy
        },
    }, {
        'name': 'batting',                # name of the table
        'data': df_original_batting,      # the original data as pd.DataFrame
        'foreign_keys': [                 # foreign key configurations
            {'column': 'players_id', 'referenced_table': 'players', 'is_context': True},
        ],
        'tabular_model_configuration': {
            'max_training_time': 2,       # cap runtime for demo; set None for max accuracy
        },
    }, {
        'name': 'fielding',               # name of the table
        'data': df_original_fielding,     # the original data as pd.DataFrame
        'foreign_keys': [                 # foreign key configurations
            {'column': 'players_id', 'referenced_table': 'players', 'is_context': True},
        ],
        'tabular_model_configuration': {
            'max_training_time': 2,       # cap runtime for demo; set None for max accuracy
        },
    }],
})

# show Model QA reports
g.reports(display=True)
```

Generate a new dataset of 10k synthetic players and their synthetic season stats.

```python
sd = mostly.generate(g, size=10_000)
syn = sd.data()
display(syn['players'].sort_values('id'))
display(syn['batting'].sort_values('players_id'))
display(syn['fielding'].sort_values('players_id'))
```

## Multi-table relational data with non-context foreign keys

Train a 3-table relational generator on the BERKA dataset with both context and non-context foreign keys.

**Understanding Foreign Key Types:**

- **Context FK** (`is_context: True`): Child records that belong to the same parent are learned and generated together with the context of the parent. This preserves patterns between child and parent (and grand-parent) as well as between siblings belonging to the same parent.
- **Non-Context FK** (`is_context: False`): A dedicated model is trained to learn matching a child record with a suitable parent record. This preserves patterns between child and parent, but not between siblings belonging to the same parent.

**Subject Tables:** Tables that do not have a context foreign key are considered subject tables. In the BERKA example below, `clients` and `accounts` are subject tables, while `disp` is a child table with a context FK to `clients`.

```python
import pandas as pd
from mostlyai.sdk import MostlyAI

# load original data from BERKA dataset
repo_url = 'https://github.com/mostly-ai/public-demo-data/raw/refs/heads/dev/berka/data'
accounts_df = pd.read_csv(f'{repo_url}/account.csv.gz')
disp_df = pd.read_csv(f'{repo_url}/disp.csv.gz')
clients_df = pd.read_csv(f'{repo_url}/client.csv.gz')

# instantiate SDK
mostly = MostlyAI()

# train a generator
g = mostly.train(config={
    'name': 'BERKA',                      # name of the generator
    'tables': [{                          # provide list of table(s)
        'name': 'clients',                # name of the table
        'data': clients_df,               # the original data as pd.DataFrame
        'primary_key': 'client_id',       # define PK column
        'tabular_model_configuration': {
            'max_training_time': 2,       # cap runtime for demo; set None for max accuracy
        },
    }, {
        'name': 'accounts',               # name of the table
        'data': accounts_df,              # the original data as pd.DataFrame
        'primary_key': 'account_id',      # define PK column
        'tabular_model_configuration': {
            'max_training_time': 2,       # cap runtime for demo; set None for max accuracy
        },
    }, {
        'name': 'disp',                   # name of the table
        'data': disp_df,                  # the original data as pd.DataFrame
        'primary_key': 'disp_id',         # define PK column
        'foreign_keys': [{                # define FK columns: max 1 Context FK allowed
            'column': 'client_id',
            'referenced_table': 'clients',
            'is_context': True            # Context FK
        }, {
            'column': 'account_id',
            'referenced_table': 'accounts',
            'is_context': False           # Non-Context FK
        }],
        'tabular_model_configuration': {
            'max_training_time': 2,       # cap runtime for demo; set None for max accuracy
        },
    }],
})

# show Model QA reports
g.reports(display=True)
```

Generate a new dataset with 1000 synthetic clients and 800 synthetic accounts, along with their related dispositions.

```python
sd = mostly.generate(g, size={'clients': 1_000, 'accounts': 800})
syn = sd.data()
display(syn['clients'].sort_values('client_id'))
display(syn['accounts'].sort_values('account_id'))
display(syn['disp'].sort_values('disp_id'))
```

## Tabular and textual data

Train a multi-model generator on a single flat table, that consists both of tabular and of textual columns.

Note, that the usage of a GPU, with 24GB or more, is strongly recommended for training language models.

```python
import pandas as pd
from mostlyai.sdk import MostlyAI

# load original data with news headlines
repo_url = 'https://github.com/mostly-ai/public-demo-data'
trn_df = pd.read_parquet(f'{repo_url}/raw/refs/heads/dev/headlines/headlines.parquet')

# instantiate SDK
mostly = MostlyAI()

# print out available LANGUAGE models
print(mostly.models()["LANGUAGE"])

# train a generator; increase max_training_time to improve quality
g = mostly.train(config={
    'name': 'Headlines',                   # name of the generator
    'tables': [{                           # provide list of table(s)
        'name': 'headlines',               # name of the table
        'data': trn_df,                    # the original data as pd.DataFrame
        'columns': [                       # configure TABULAR + LANGUAGE cols
            {'name': 'category', 'model_encoding_type': 'TABULAR_CATEGORICAL'},
            {'name': 'date', 'model_encoding_type': 'TABULAR_DATETIME'},
            {'name': 'headline', 'model_encoding_type': 'LANGUAGE_TEXT'},
        ],
        'tabular_model_configuration': {             # tabular model configuration (optional)
            'max_training_time': 5,                  # cap runtime for demo; set None for max accuracy
        },
        'language_model_configuration': {             # language model configuration (optional)
            'max_training_time': 5,                   # cap runtime for demo; set None for max accuracy
            'model': 'MOSTLY_AI/LSTMFromScratch-3m',  # use a light-weight LSTM model, trained from scratch (GPU recommended)
            #'model': 'microsoft/phi-1.5',            # alternatively use a pre-trained HF-hosted LLM model (GPU required)
        }
    }],
})
```

Conditionally generate 100 new headlines for the WELLNESS category.

```python
df_seed = pd.DataFrame({'category': ['WELLNESS'] * 100})
sd = mostly.generate(g, seed=df_seed)
df_synthetic = sd.data()
```

## Usage of connectors

Leverage connectors for fetching original data as well as for delivering synthetic datasets.

See [ConnectorConfig](https://mostly-ai.github.io/mostlyai/api_domain/#mostlyai.sdk.domain.ConnectorConfig) for the full list of available connectors, and their corresponding configuration parameters.

```python
import pandas as pd
from mostlyai.sdk import MostlyAI

# instantiate SDK
mostly = MostlyAI()

# define a source connector for reading
src_c = mostly.connect(config={
    "name": "My S3 Source Storage",
    "type": "POSTGRES",
    "access_type": "SOURCE",
    "config": {
        "host": "INSERT_YOUR_DB_HOST",
        "username": "INSERT_YOUR_DB_USER",
        "database": "INSERT_YOUR_DB_NAME",
    },
    "secrets": {
        "password": "INSERT_YOUR_DB_PWD",
    }
})

# define a destination connector for writing
dest_c = mostly.connect(config={
    "name": "My S3 Destination Storage",
    "type": "S3_STORAGE",
    "access_type": "DESTINATION",
    "config": {
        "access_key": "INSERT_YOUR_ACCESS_KEY",
    },
    "secrets": {
        "secret_key": "INSERT_YOUR_SECRET_KEY",
    },
})

# list available source locations
src_c.locations()
```

Train a generator on a dataset fetched from the source connector.

```python
# train a generator; increase max_training_time to improve quality
g = mostly.train(config={
    'name': 'Housing',                      # name of the generator
    'tables': [{                            # provide list of table(s)
        'name': 'housing',                  # name of the table
        'source_connector_id': src_c.id,    # the ID of the source connector
        'location': 'bucket/path_to_data',  # the location of the source data
        'tabular_model_configuration': {    # tabular model configuration (optional)
            'max_epochs': 20,               # cap runtime for demo; set None for max accuracy
        },
    }],
}, start=True, wait=True)
```

Generate a synthetic dataset, and deliver it to a destination connector.

```python
sd = mostly.generate(g, config={
    "name": "Housing",                             # name of the synthetic dataset
    "delivery": {                                  # delivery configuration (optional)
        "destination_connector_id": dest_c.id,     # the ID of the destination connector
        "location": "bucket/path_to_destination",  # the location of the destination data
        "overwrite_tables": True,                   # overwrite existing tables (default: False)
    }
})
```

## Usage of datasets (CLIENT mode only)

Create datasets to train [generators](#tabular-and-textual-data) or generate [artifacts](https://docs.mostly.ai/assistant/artifacts) via the MOSTLY AI Assistant.

Example 1: Create a dataset with a connector

```python
from mostlyai.sdk import MostlyAI
mostly = MostlyAI()
ds = mostly.datasets.create(
    config={
        "name": "My Database",
        "description": "Some instructions...",
        "connectors": [
            {
                "id": "e43aa845-8d77-4cda-bc9e-10da9a4196a9"  # the UUID of the source connector
            }
        ]
    }
)
```

Example 2: Create a dataset with files

```python
from mostlyai.sdk import MostlyAI
mostly = MostlyAI()
ds = mostly.datasets.create(
    config={
        "name": "My Dataset",
        "description": "Some instructions...",
    }
)
ds.upload_file("path/to/file_1.csv.gz")
ds.upload_file("path/to/file_2.txt")
```
