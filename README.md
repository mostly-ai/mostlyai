
# Synthetic Data SDK ✨

[![Documentation](https://img.shields.io/badge/docs-latest-green)](https://mostly-ai.github.io/mostlyai/) [![stats](https://pepy.tech/badge/mostlyai)](https://pypi.org/project/mostlyai/) ![license](https://img.shields.io/github/license/mostly-ai/mostlyai) ![GitHub Release](https://img.shields.io/github/v/release/mostly-ai/mostlyai) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mostlyai)

[SDK Documentation](https://mostly-ai.github.io/mostlyai/) | [Platform Documentation](https://mostly.ai/docs) | [Usage Examples](https://mostly-ai.github.io/mostlyai/usage/)

The official SDK of [MOSTLY AI](https://app.mostly.ai/), a Python toolkit for high-fidelity, privacy-safe **Synthetic Data**.

- **Client mode** connects to a remote MOSTLY AI platform for training & generating synthetic data there.
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
pip install -U `mostlyai[local, databricks, snowflake]`
```

## Quick Start

For client mode, initialize with `base_url` and `api_key` obtained from your [account settings page](https://app.mostly.ai/settings/api-keys). For local mode, initialize the client simply with `local=True`.

```python
import pandas as pd
from mostlyai.sdk import MostlyAI

# 1) Initialize the SDK in local or client mode
mostly = MostlyAI(local=True)
# mostly = MostlyAI(base_url='https://app.mostly.ai', api_key='YOUR_API_KEY')

# 2) Load your original data
trn_df = pd.read_csv('https://github.com/mostly-ai/public-demo-data/raw/dev/census/census.csv.gz')

# 3) Train a synthetic data generator
g = mostly.train(name='census', data=trn_df)  # shorthand syntax for 1-table config

# 4) Live probe small synthetic samples
df_samples = mostly.probe(g, size=10)

# 5) Generate a full synthetic dataset
sd = mostly.generate(g, size=100_000)
syn_df = sd.data()
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
