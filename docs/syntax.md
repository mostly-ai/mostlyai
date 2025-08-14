---
hide:
  - navigation
---

# MOSTLY AI Cheat Sheet

## Initialization

```python
from mostlyai.sdk import MostlyAI

# local mode (with TCP port)
mostly = MostlyAI(
    local=True,
    local_dir='~/mostlyai',
    local_port=8080,
)

# client mode
mostly = MostlyAI(
    base_url='https://app.mostly.ai', # or set env var `MOSTLY_BASE_URL`
    api_key='INSERT_YOUR_API_KEY',    # or set env var `MOSTLY_API_KEY`
)
```

## Generators

```python
# shorthand syntax to train a new single-table generator
g = mostly.train(name: str, data: pd.DataFrame)

# train a new generator
g = mostly.train(config: dict | GeneratorConfig, start: bool, wait: bool)

# configure a new generator and launch separately
g = mostly.generators.create(config: dict | GeneratorConfig)
g.training.start()
g.training.wait()
g.training.progress()  # includes loss messages
g.training.logs()

# download a ZIP of quality assurance reports
g.reports()

# iterate over all your available generators
for g in mostly.generators.list():
    print(g.id, g.name)

# fetch a generator by id
g = mostly.generators.get(id: str)

# fetch a generator's configuration
config = g.config()

# open a generator in a new browser tab
g.open()

# update a generator
g.update(name: str, ...)

# delete a generator
g.delete()

# export a generator as a ZIP file
fn = g.export_to_file()
# import a generator from a ZIP file
g = mostly.generators.import_from_file(fn)

# continue training of an existing generator (with connectors)
g2 = g.clone("continue")
g2.training.start()
g2.training.wait()
```

## Synthetic Datasets

```python
# shorthand syntax for generating a new synthetic dataset
sd = mostly.generate(g, size: int)

# shorthand syntax for conditionally generating a new synthetic dataset
sd = mostly.generate(g, seed: pd.DataFrame)

# generate a new synthetic dataset
sd = mostly.generate(g, config: dict | SyntheticDatasetConfig, start: bool, wait: bool)

# configure a new synthetic dataset and launch separately
sd = mostly.synthetic_datasets.create(config: dict | SyntheticDatasetConfig)
sd.generation.start()
sd.generation.wait()
sd.generation.progress()
sd.generation.logs()

# download a ZIP of quality assurance reports
sd.reports()

# iterate over all your available synthetic datasets
for sd in mostly.synthetic_datasets.list():
    print(sd.id, sd.name)

# fetch a synthetic dataset by id
sd = mostly.synthetic_datasets.get(id: str)

# fetch a synthetic dataset's configuration
config = sd.config()

# open a synthetic dataset in a new browser tab
sd.open()

# download a synthetic dataset
sd.download(file: str, format: str)

# fetch the synthetic dataset's data
syn_df = sd.data()

# update a synthetic dataset
sd.update(name: str, ...)

# delete a synthetic dataset
sd.delete()
```

Synthetic probes allow to instantly generate synthetic samples on demand, without storing these on the platform. This feature depends on the availability of **Live Probing** on the platform. The syntax is similar to generating a synthetic dataset, with the notable difference that its return value is already the synthetic data as pandas DataFrame(s).
```python
# shorthand syntax for probing for synthetic samples
syn_df = mostly.probe(g, size: int)

# shorthand syntax for conditionally probing for synthetic samples
syn_df = mostly.probe(g, seed: pd.DataFrame)

# probe for synthetic samples
syn_df = mostly.probe(g, config: dict | SyntheticDatasetConfig)
```

## Connectors

Connectors can be used both as a source of original data for training a generator, as well as a destination for delivering the generated synthetic data samples to. See [ConnectorConfig](api_domain.md#mostlyai.sdk.domain.ConnectorConfig) for the full list of available connectors, and their corresponding configuration parameters.

```python
# create a new connector
c = mostly.connect(config: dict | ConnectorConfig)

# fetch a connector by id
c = mostly.connectors.get(id: str)

# list all locations of a connector
c.locations(prefix: str)

# fetch schema for a specific location
c.schema(location: str)

# iterate over all your available connectors
for c in mostly.connectors.list():
    print(c.id, c.name)

# update a connector
c.update(name: str, ...)

# open a connector in a new browser tab
c.open()

# delete a connector
c.delete()
```

## Datasets

Datasets can be used to train generators or create artifacts and may be created with or without a corresponding connector. Datasets are only available in `client` mode.

```python
# create a new dataset
ds = mostly.datasets.create(config: dict | DatasetConfig)

# retrieve a dataset by id
ds = mostly.datasets.get(dataset_id: str)

# list datasets (iterator with optional filters)
for ds in mostly.datasets.list(
    offset: int = 0,
    limit: int | None = None,
    search_term: str | None = None,
    owner_id: str | list[str] | None = None,
    visibility: str | list[str] | None = None,     # e.g., PUBLIC, PRIVATE, UNLISTED
    created_from: str | None = None,               # YYYY-MM-DD
    created_to: str | None = None,                 # YYYY-MM-DD
    sort_by: str | list[str] | None = None,        # NO_OF_THREADS | NO_OF_LIKES | RECENCY
):
    print(ds.id, ds.name)

# fetch the dataset's config (read-only view of config)
cfg = mostly.datasets._config(dataset_id: str)

# update an existing dataset (partial patch)
ds = mostly.datasets._update(dataset_id: str, config: dict | DatasetPatchConfig)

# delete a dataset
mostly.datasets._delete(dataset_id: str)

# download a file from a dataset
content_bytes, filename = mostly.datasets._download_file(
    dataset_id: str,
    file_path: str,                                 # path inside the dataset
)

# upload a file to a dataset
mostly.datasets._upload_file(
    dataset_id: str,
    file_path: str | Path,                          # local path to file
)

# delete a file from a dataset
mostly.datasets._delete_file(
    dataset_id: str,
    file_path: str | Path,                          # path inside the dataset
)
```

## Miscellaneous

```python
# fetch info on your user account
mostly.me()

# fetch info about the platform
mostly.about()

# list all available models
mostly.models()

# list all available computes
mostly.computes()
```
