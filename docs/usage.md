---
hide:
  - navigation
---

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
mostly = MostlyAI(local=True)  # or: MostlyAI(base_url='xxx', api_key='xxx')

# train a generator
g = mostly.train(config={
        'name': 'US Census Income',          # name of the generator
        'tables': [{                         # provide list of table(s)
            'name': 'census',                # name of the table
            'data': df_original,             # the original data as pd.DataFrame
            'tabular_model_configuration': { # tabular model configuration (optional)
                'max_training_time': 2,      # - limit training time (in minutes)
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

## Multi-table tabular data

Train a multi-table tabular generator on baseball players and their seasonal statistics.

```python
import pandas as pd
from mostlyai.sdk import MostlyAI

# load original data
repo_url = 'https://github.com/mostly-ai/public-demo-data'
df_original_players = pd.read_csv(f'{repo_url}/raw/dev/baseball/players.csv.gz')
df_original_players = df_original_players[['id', 'country', 'weight', 'height']]
df_original_seasons = pd.read_csv(f'{repo_url}/raw/dev/baseball/batting.csv.gz')
df_original_seasons = df_original_seasons[['players_id', 'year', 'team', 'G', 'AB', 'HR']]

# instantiate SDK
mostly = MostlyAI(local=True)  # or: MostlyAI(base_url='xxx', api_key='xxx')

# train a generator
g = mostly.train(config={
    'name': 'Baseball',                   # name of the generator
    'tables': [{                          # provide list of table(s)
        'name': 'players',                # name of the table
        'data': df_original_players,      # the original data as pd.DataFrame
        'primary_key': 'id',
    }, {
        'name': 'seasons',                # name of the table
        'data': df_original_seasons,      # the original data as pd.DataFrame
        'foreign_keys': [                 # foreign key configurations
            {'column': 'players_id', 'referenced_table': 'players', 'is_context': True},
        ],
    }],
}, start=True, wait=True)
```

Generate a new dataset of 10k synthetic players and their synthetic season stats.
```python
sd = mostly.generate(g, size=10_000)
df_synthetic_players = sd.data()['players']
display(df_synthetic_players)
df_synthetic_seasons = sd.data()['seasons']
display(df_synthetic_seasons)
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
mostly = MostlyAI(local=True)  # or: MostlyAI(base_url='xxx', api_key='xxx')

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
            'max_training_time': 5,                  # - limit training time (in minutes)
        },
        'language_model_configuration': {             # language model configuration (optional)
            'max_training_time': 5,                   # - limit training time (in minutes)
            'model': 'MOSTLY_AI/LSTMFromScratch-3m',  # use a light-weight LSTM model, trained from scratch (GPU recommended)
            #'model': 'microsoft/phi-1.5',            # alternatively use a pre-trained HF-hosted LLM model (GPU required)
        }
    }],
}, start=True, wait=True)
```

Conditionally generate 100 new headlines for the WELLNESS category.

```python
df_seed = pd.DataFrame({'category': ['WELLNESS'] * 100})
sd = mostly.generate(g, seed=df_seed)
df_synthetic = sd.data()
```

## Usage of connectors

Leverage connectors for fetching original data as well as for delivering synthetic datasets.

See [ConnectorConfig](../api_domain/#mostlyai.sdk.domain.ConnectorConfig) for the full list of available connectors, and their corresponding configuration parameters.

```python
import pandas as pd
from mostlyai.sdk import MostlyAI

# instantiate SDK
mostly = MostlyAI(local=True)  # or: MostlyAI(base_url='xxx', api_key='xxx')

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
            'max_epochs': 20,               # - limit training epochs
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
