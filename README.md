# Synthetic Data SDK ✨

[![GitHub Release](https://img.shields.io/github/v/release/mostly-ai/mostlyai)](https://github.com/mostly-ai/mostlyai/releases)
[![Documentation](https://img.shields.io/badge/docs-latest-green)](https://mostly-ai.github.io/mostlyai/)
[![PyPI Downloads](https://static.pepy.tech/badge/mostlyai)](https://pepy.tech/projects/mostlyai)
[![License](https://img.shields.io/github/license/mostly-ai/mostlyai)](https://github.com/mostly-ai/mostlyai/blob/main/LICENSE)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mostlyai)](https://pypi.org/project/mostlyai/)
[![GitHub stars](https://img.shields.io/github/stars/mostly-ai/mostlyai?style=social)](https://github.com/mostly-ai/mostlyai/stargazers)

[Documentation](https://mostly-ai.github.io/mostlyai/) | [Technical White Paper](https://arxiv.org/abs/2508.00718) | [Usage Examples](https://mostly-ai.github.io/mostlyai/usage/) | [Free Cloud Service](https://app.mostly.ai/)

The **Synthetic Data SDK** is a Python toolkit for high-fidelity, privacy-safe **Synthetic Data**.

- **LOCAL** mode trains and generates synthetic data locally on your own compute resources.
- **CLIENT** mode connects to a remote MOSTLY AI platform for training & generating synthetic data there.
- Generators, that were trained locally, can be easily imported to a platform for further sharing.

## Overview

The SDK allows you to programmatically create, browse and manage 3 key resources:

1. **Generators** - Train a synthetic data generator on your existing tabular or language data assets
2. **Synthetic Datasets** - Use a generator to create any number of synthetic samples to your needs
3. **Connectors** - Connect to any data source within your organization, for reading and writing data

| Intent                                        | Primitive                         | API Reference                                                                                                 |
|-----------------------------------------------|-----------------------------------|---------------------------------------------------------------------------------------------------------------|
| Train a Generator on tabular or language data | `g = mostly.train(config)`        | [mostly.train](https://mostly-ai.github.io/mostlyai/api_client/#mostlyai.sdk.client.api.MostlyAI.train)       |
| Generate any number of synthetic data records | `sd = mostly.generate(g, config)` | [mostly.generate](https://mostly-ai.github.io/mostlyai/api_client/#mostlyai.sdk.client.api.MostlyAI.generate) |
| Live probe the generator on demand            | `df = mostly.probe(g, config)`    | [mostly.probe](https://mostly-ai.github.io/mostlyai/api_client/#mostlyai.sdk.client.api.MostlyAI.probe)       |
| Connect to any data source within your org    | `c = mostly.connect(config)`      | [mostly.connect](https://mostly-ai.github.io/mostlyai/api_client/#mostlyai.sdk.client.api.MostlyAI.connect)   |

https://github.com/user-attachments/assets/9e233213-a259-455c-b8ed-d1f1548b492f

## Key Features

- **Broad Data Support**
  - Mixed-type data (categorical, numerical, geospatial, text, etc.)
  - Single-table, multi-table, and time-series
- **Multiple Model Types**
  - State-of-the-art performance via TabularARGN
  - DNN-based match making for graph relations
  - Fine-tune Hugging Face hosted language models
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
  - Conditional simulations based on any columns
  - Re-balance underrepresented segments
  - Context-aware data imputation
  - Statistical fairness controls
  - Rule-adherence via temperature
- **Seamless Integration**
  - Connect to external data sources (DBs, cloud storages)
  - Fully permissive open-source license

## Quick Start <a href="https://colab.research.google.com/github/mostly-ai/mostlyai/blob/main/docs/tutorials/getting-started/getting-started.ipynb" target="_blank"><img src="https://img.shields.io/badge/Open%20in-Colab-blue?logo=google-colab" alt="Run on Colab"></a>

Install the SDK via `pip` (see [Installation](#installation) for further details):

```shell
pip install -U mostlyai  # or 'mostlyai[local]' for LOCAL mode
```

Generate synthetic samples using a pre-trained generator:

```python
# initialize the SDK
from mostlyai.sdk import MostlyAI
mostly = MostlyAI()

# import a trained generator
g = mostly.generators.import_from_file(
  "https://github.com/mostly-ai/public-demo-data/raw/dev/census/census-generator.zip"
)

# probe for 1000 representative synthetic samples
df = mostly.probe(g, size=1000)
df
```

Generate synthetic samples based on fixed column values:

```python
# create 10k records of 24y male respondents
df = mostly.probe(g, seed=[{"age": 24, "sex": "Male"}] * 10_000)
df
```

And now train your very own synthetic data generator:

```python
# load original data
import pandas as pd
original_df = pd.read_csv(
  "https://github.com/mostly-ai/public-demo-data/raw/dev/titanic/titanic.csv"
)

# train a single-table generator, with default configs
g = mostly.train(
  name="Quick Start Demo - Titanic",
  data=original_df,
)

# display the quality assurance report
g.reports(display=True)

# generate a representative synthetic dataset, with default configs
sd = mostly.generate(g)
df = sd.data()

# or simply probe for some samples
df = mostly.probe(g, size=100)
df
```

## Performance

The SDK is being developed with a focus on efficiency, accuracy, and flexibility, with best-in-class performance across all three. Results will ultimately depend on the training data itself (size, structure, and content), on the available compute (CPU vs GPU), as well as on the chosen training configurations (model, epochs, samples, etc.). Thus, a crawl / walk / run approach is recommended — starting with a subset of samples training for a limited amount of time, to then gradually scale up, to yield optimal results for use case at hand.

### Tabular Models

Tabular models within the SDK are built on TabularARGN ([arXiv:2501.12012](https://arxiv.org/abs/2501.12012)), which achieves best-in-class synthetic data quality while being 1–2 orders of magnitude more efficient than comparable models. This efficiency enables the training and generation of millions of synthetic records within minutes, even on CPU environments.

![TabularARGN Benchmark](https://raw.githubusercontent.com/mostly-ai/mostlyai/refs/heads/main/docs/TabularARGN-benchmark.png)

### Language Models

The default language model is a basic, non-pre-trained LSTM (`LSTMFromScratch-3m`), particularly effective for textual data with limited scope (short lengths, narrow variety) and sufficient training samples.

Alternatively, any pre-trained language model, that is available via the [Hugging Face Hub](https://huggingface.co/) and that supports the [AutoModelForCausalLM](https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoModelForCausalLM) class, can be selected to be then fine-tuned on the provided training data. These models start out already with a general world knowledge, and then adapt to the training data for generating high-fidelity synthetic samples even in sparse data domains. The final performance will once again largely depend on the chosen model configurations.

In either case, a modern GPU is highly recommended when working with language models.

## Installation

Use `pip` (or better `uv pip`) to install the official `mostlyai` package via PyPI. Python 3.10 or higher is required.

It is highly recommended to install the package within a dedicated virtual environment using `uv` (see [here](https://docs.astral.sh/uv/)):

<details>

  <summary>Setup of <code>uv</code> on Unix / macOS</summary>

```shell
# Install uv if you don't have it yet
curl -Ls https://astral.sh/uv/install.sh | bash

# Create and activate a Python 3.12 environment with uv
mkdir ~/synthetic-data-sdk; cd ~/synthetic-data-sdk
uv venv -p 3.12

# Activate virtual environment
source .venv/bin/activate
```

</details>

<details>

  <summary>Setup of <code>uv</code> on Windows</summary>

```shell
# Install uv if you don't have it yet
irm https://astral.sh/uv/install.ps1 | iex

# Create and activate a Python 3.12 environment with uv
mkdir ~/synthetic-data-sdk; cd ~/synthetic-data-sdk
uv venv -p 3.12

# Activate virtual environment
.venv\Scripts\activate
```

</details>

<details>

  <summary>Run Jupyter Lab session via <code>uv</code></summary>

```shell
# Optionally launch jupyter session after SDK installation
uv run --with jupyter jupyter lab
```

</details>

### CLIENT mode

This is a light-weight installation for using the SDK in CLIENT mode only. It communicates to a MOSTLY AI platform to perform requested tasks. See e.g. [app.mostly.ai](https://app.mostly.ai/) for a free-to-use hosted version.

```shell
uv pip install -U mostlyai
```

### CLIENT + LOCAL mode

This is a full installation for using the SDK in both CLIENT and LOCAL mode. It includes all dependencies, incl. PyTorch, for training and generating synthetic data locally.

```shell
uv pip install -U 'mostlyai[local]'
```

or alternatively for a GPU setup on Linux (needed for LLM finetuning and inference):

```shell
uv pip install -U 'mostlyai[local-gpu]'
```

On Linux, one can explicitly install the CPU-only variant of torch together with `mostlyai[local]`:

```shell
# uv pip install
uv pip install --index-strategy unsafe-first-match -U torch==2.8.0+cpu torchvision==0.23.0+cpu 'mostlyai[local]' --extra-index-url https://download.pytorch.org/whl/cpu
```

```shell
# standard pip install
pip install -U torch==2.8.0+cpu torchvision==0.23.0+cpu 'mostlyai[local]' --extra-index-url https://download.pytorch.org/whl/cpu
```


> **Note for Google Colab users**: Installing any of the local extras (`mostlyai[local]`, or `mostlyai[local-gpu]`) might need restarting the runtime after installation for the changes to take effect.

### Data Connectors

Add any of the following extras for further data connectors support in LOCAL mode: `databricks`, `googlebigquery`, `hive`, `mssql`, `mysql`, `oracle`, `postgres`, `redshift`, `snowflake`. E.g.

```shell
uv pip install -U 'mostlyai[local, databricks, snowflake]'
```

### Using Docker

As an alternative, you can also build a Docker image, which provides you with an isolated environment for running the SDK in LOCAL mode, with all connector dependencies pre-installed. This approach ensures a consistent runtime environment across all systems. Before proceeding, make sure [Docker](https://docs.docker.com/get-started/get-docker/) is installed on your system.

<details>

  <summary>Get the image</summary>

  <ul>
  <li><strong>Pull from official repository</strong></li>
  </ul>

  <code>docker pull --platform=linux/amd64 ghcr.io/mostly-ai/sdk</code>

  <ul>
  <li><strong>Pull from official repository</strong></li>
  </ul>

  If your environment is capable of executing Makefile (see <a href="https://github.com/mostly-ai/mostlyai/blob/main/Makefile#L47-L73">here</a>), then execute <code>make docker-build</code>.

  Otherwise, use <code>docker buildx build . --platform=linux/amd64 -t ghcr.io/mostly-ai/sdk</code> instead.

</details>

<details>

  <summary>Start the container</summary>

  <p>This will launch the SDK in LOCAL mode on port 8080 inside the container.</p>

  <p>If your environment is capable of executing Makefile, then execute <code>make docker-run</code>. Or <code>make docker-run HOST_PORT=8080</code> to forward to a host port of your choice. One could also mount the <code>local_dir</code> via <code>make docker-run HOST_LOCAL_DIR=/path/to/host/folder</code> to make the generators and synthetic datasets directly accessible from the host.</p>

  <p>Otherwise, use <code>docker run --platform=linux/amd64 -p 8080:8080 ghcr.io/mostly-ai/sdk</code> instead. Optionally, you can use the <code>-v</code> flag to mount a <a href="https://docs.docker.com/engine/storage/volumes/#syntax">volume</a> for passing files between the host and the container.</p>

</details>

<details>

  <summary>Connect to the container</summary>

  <p>You can now connect to the SDK running within the container by initializing the SDK in <code>CLIENT</code>> mode on the host machine.</p>

  ```python
  from mostlyai.sdk import MostlyAI

  mostly = MostlyAI(base_url="http://localhost:8080")
  ```

</details>

### Air-gapped Environments

For air-gapped environments (without internet access), you must install the package using the provided wheel files, including any optional dependencies you require.

If your application depends on a Hugging Face language model, you’ll also need to manually download and transfer the model files.

<details>

  <summary>Download models from Hugging Face Hub</summary>

<p>On a machine with internet access, run the following Python script, to download the Hugging Face model to your local Hugging Face cache.</p>

```python
#! uv pip install huggingface-hub
from pathlib import Path
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id="Qwen/Qwen2.5-Coder-0.5B",  # change accordingly
    token=None,  # insert your HF TOKEN for gated models
)
print(f"COPY `{Path(path).parent.parent}`")
```

Next, transfer the printed directory to the air-gapped environment's cache directory located at `~/.cache/huggingface/hub/` (respectively to `HF_HOME`, if that environment variable has been set).

</details>


## Citation

Please consider citing our project if you find it useful:

```bibtex
@misc{mostlyai,
      title={Democratizing Tabular Data Access with an Open-Source Synthetic-Data SDK},
      author={Ivona Krchova and Mariana Vargas Vieyra and Mario Scriminaci and Andrey Sidorenko},
      year={2025},
      eprint={2508.00718},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2508.00718},
}
```
