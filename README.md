
# Synthetic Data SDK ✨

[![Documentation](https://img.shields.io/badge/docs-latest-green)](https://mostly-ai.github.io/mostlyai/) [![PyPI Downloads](https://static.pepy.tech/badge/mostlyai)](https://pepy.tech/projects/mostlyai) ![license](https://img.shields.io/github/license/mostly-ai/mostlyai) ![GitHub Release](https://img.shields.io/github/v/release/mostly-ai/mostlyai) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mostlyai) [![GitHub stars](https://img.shields.io/github/stars/mostly-ai/mostlyai?style=social)](https://github.com/mostly-ai/mostlyai)

[SDK Documentation](https://mostly-ai.github.io/mostlyai/) | [Platform Documentation](https://mostly.ai/docs) | [Usage Examples](https://mostly-ai.github.io/mostlyai/usage/)

The Synthetic Data SDK is a Python toolkit for high-fidelity, privacy-safe **Synthetic Data**.

- **Client mode** connects to a remote [MOSTLY AI platform](https://app.mostly.ai/) for training & generating synthetic data there.
- **Local mode** trains and generates synthetic data locally on your own compute resources.
- Generators, that were trained locally, can be easily imported to a platform for further sharing.

## Overview

The SDK allows you to programmatically create, browse and manage 3 key resources:

1. **Generators** - Train a synthetic data generator on your existing tabular or language data assets
2. **Synthetic Datasets** - Use a generator to create any number of synthetic samples to your needs
3. **Connectors** - Connect to any data source within your organization, for reading and writing data

| Intent                                        | Primitive                         | Documentation                                                                                                     |
|-----------------------------------------------|-----------------------------------|-------------------------------------------------------------------------------------------------------------------|
| Train a Generator on tabular or language data | `g = mostly.train(config)`        | see [mostly.train](https://mostly-ai.github.io/mostlyai/api_client/#mostlyai.sdk.client.api.MostlyAI.train)       |
| Generate any number of synthetic data records | `sd = mostly.generate(g, config)` | see [mostly.generate](https://mostly-ai.github.io/mostlyai/api_client/#mostlyai.sdk.client.api.MostlyAI.generate) |
| Live probe the generator on demand            | `df = mostly.probe(g, config)`    | see [mostly.probe](https://mostly-ai.github.io/mostlyai/api_client/#mostlyai.sdk.client.api.MostlyAI.probe)       |
| Connect to any data source within your org    | `c = mostly.connect(config)`      | see [mostly.connect](https://mostly-ai.github.io/mostlyai/api_client/#mostlyai.sdk.client.api.MostlyAI.connect)   |

## Installation

**Client mode only**

```shell
pip install -U mostlyai
```

**Client + Local mode**

```shell
pip install -U 'mostlyai[local]'       # for CPU
#pip install -U 'mostlyai[local-gpu]'  # for GPU
```

NOTE: installing `mostlyai[local]` on Linux requires `--extra-index-url https://download.pytorch.org/whl/cpu` to be specified.

**Optional Connectors**

Add any of the following extras for further data connectors support: `databricks`, `googlebigquery`, `hive`, `mssql`, `mysql`, `oracle`, `postgres`, `snowflake`.

E.g.
```shell
pip install -U 'mostlyai[local, databricks, snowflake]'
```

## Quick Start  [![Run on Colab](https://img.shields.io/badge/Open%20in-Colab-blue?logo=google-colab)](https://colab.research.google.com/github/mostly-ai/mostlyai/blob/main/docs/tutorials/getting-started/getting-started.ipynb)

Generate your first samples based on your own trained generator with a few lines of code. For local mode, initialize the SDK with `local=True`. For client mode, initialize the SDK with `base_url` and `api_key` obtained from your [account settings page](https://app.mostly.ai/settings/api-keys).

```python
import pandas as pd
from mostlyai.sdk import MostlyAI

# load original data
repo_url = "https://github.com/mostly-ai/public-demo-data/raw/refs/heads/dev"
df_original = pd.read_csv(f"{repo_url}/census/census.csv.gz").sample(n=5_000)

# initialize the SDK in local or client mode
mostly = MostlyAI(local=True)                       # local mode
# mostly = MostlyAI(base_url='xxx', api_key='xxx')  # client mode

# train a synthetic data generator
g = mostly.train(
    config={
        "name": "US Census Income",
        "tables": [
            {
                "name": "census",
                "data": df_original,
                "tabular_model_configuration": {  # tabular model configuration (optional)
                    "max_training_time": 1,  # - limit training time (in minutes)
                    # model, max_epochs,,..        # further model configurations (optional)
                    # 'differential_privacy': {    # differential privacy configuration (optional)
                    #     'max_epsilon': 5.0,      # - max epsilon value, used as stopping criterion
                    #     'delta': 1e-5,           # - delta value
                    # }
                },
                # columns, keys, compute,..      # further table configurations (optional)
            }
        ],
    },
    start=True,  # start training immediately (default: True)
    wait=True,  # wait for completion (default: True)
)
```

Once the generator has been trained, you can use it to generate synthetic data samples. Either via probing:

```python
# probe for some representative synthetic samples
df_samples = mostly.probe(g, size=100)
df_samples
```

or by creating a synthetic dataset entity for larger data volumes:

```python
# generate a large representative synthetic dataset
sd = mostly.generate(g, size=100_000)
df_synthetic = sd.data()
df_synthetic
```

or by conditionally probing / generating synthetic data:

```python
# create 100 seed records of 24y old Mexicans
df_seed = pd.DataFrame({
    'age': [24] * 100,
    'native_country': ['Mexico'] * 100,
})
# conditionally probe, based on provided seed
df_samples = mostly.probe(g, seed=df_seed)
df_samples
```

## Key Features

- **Broad Data Support**
    - Mixed-type data (categorical, numerical, geospatial, text, etc.)
    - Single-table, multi-table, and time-series
- **Multiple Model Types**
    - TabularARGN for SOTA tabular performance
    - Fine-tune HuggingFace-based language models
    - Efficient LSTM for text synthesis from scratch
- **Advanced Training Options**
    - GPU/CPU support
    - Differential Privacy
    - Progress Monitoring
- **Automated Quality Assurance**
    - Quality metrics for fidelity and privacy
    - In-depth HTML reports for visual analysis
- **Flexible Sampling**
    - Up-sample to any data volumes
    - Conditional generation by any columns
    - Re-balance underrepresented segments
    - Context-aware data imputation
    - Statistical fairness controls
    - Rule-adherence via temperature
- **Seamless Integration**
    - Connect to external data sources (DBs, cloud storages)
    - Fully permissive open-source license

## Citation

Please consider citing our project if you find it useful:

```bibtex
@software{mostlyai,
    author = {{MOSTLY AI}},
    title = {{MOSTLY AI SDK}},
    url = {https://github.com/mostly-ai/mostlyai},
    year = {2025}
}
```
