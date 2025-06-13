# Schema References for `mostlyai.sdk.domain`

This module is auto-generated to represent `pydantic`-based classes of the defined schema in the [Public API](https://github.com/mostly-ai/mostly-openapi/blob/main/public-api.yaml).

### mostlyai.sdk.domain

#### AboutService

General information about the service.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `version` | `str | None` | The version number of the service. | `None` | | `assistant` | `bool | None` | A flag indicating if the assistant is enabled. | `None` |

#### AccountType

The type of account, either a user or an organization.

#### Accuracy

Metrics regarding the accuracy of synthetic data, measured as the closeness of discretized lower dimensional marginal distributions.

1. **Univariate Accuracy**: The accuracy of the univariate distributions for all target columns.
1. **Bivariate Accuracy**: The accuracy of all pair-wise distributions for target columns, as well as for target columns with respect to the context columns.
1. **Trivariate Accuracy**: The accuracy of all three-way distributions for target columns.
1. **Coherence Accuracy**: The accuracy of the auto-correlation for all target columns.

Accuracy is defined as 100% - [Total Variation Distance](https://en.wikipedia.org/wiki/Total_variation_distance_of_probability_measures) (TVD), whereas TVD is half the sum of the absolute differences of the relative frequencies of the corresponding distributions.

These accuracies are calculated for all discretized univariate, and bivariate distributions. In case of sequential data, also for all coherence distributions. Overall metrics are then calculated as the average across these accuracies.

All metrics can be compared against a theoretical maximum accuracy, which is calculated for a same-sized holdout. The accuracy metrics shall be as close as possible to the theoretical maximum, but not significantly higher, as this would indicate overfitting.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `overall` | `float | None` | Overall accuracy of synthetic data, averaged across univariate, bivariate, and coherence. | `None` | | `univariate` | `float | None` | Average accuracy of discretized univariate distributions. | `None` | | `bivariate` | `float | None` | Average accuracy of discretized bivariate distributions. | `None` | | `trivariate` | `float | None` | Average accuracy of discretized trivariate distributions. | `None` | | `coherence` | `float | None` | Average accuracy of discretized coherence distributions. Only applicable for sequential data. | `None` | | `overall_max` | `float | None` | Expected overall accuracy of a same-sized holdout. Serves as a reference for overall. | `None` | | `univariate_max` | `float | None` | Expected univariate accuracy of a same-sized holdout. Serves as a reference for univariate. | `None` | | `bivariate_max` | `float | None` | Expected bivariate accuracy of a same-sized holdout. Serves as a reference for bivariate. | `None` | | `trivariate_max` | `float | None` | Expected trivariate accuracy of a same-sized holdout. Serves as a reference for trivariate. | `None` | | `coherence_max` | `float | None` | Expected coherence accuracy of a same-sized holdout. Serves as a reference for coherence. | `None` |

#### Compute

A compute resource for executing tasks.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `id` | `str | None` | The unique identifier of a compute resource. Not applicable for SDK. | `None` | | `name` | `str | None` | The name of a compute resource. | `None` | | `type` | `ComputeType | None` | An enumeration. | `None` | | `config` | `dict[str, Any] | None` | An enumeration. | `None` | | `secrets` | `dict[str, Any] | None` | An enumeration. | `None` | | `resources` | `ComputeResources | None` | An enumeration. | `None` | | `order_index` | `int | None` | The index for determining the sort order when listing computes | `None` |

#### ComputeConfig

The configuration for creating a new compute resource.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `name` | `str | None` | The name of a compute resource. | `None` | | `type` | `ComputeType | None` | An enumeration. | `None` | | `resources` | `ComputeResources | None` | An enumeration. | `None` | | `config` | `dict[str, Any] | None` | An enumeration. | `None` | | `secrets` | `dict[str, Any] | None` | An enumeration. | `None` | | `order_index` | `int | None` | The index for determining the sort order when listing computes | `None` |

#### ComputeListItem

Essential compute details for listings.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `id` | `str | None` | The unique identifier of a compute resource. Not applicable for SDK. | `None` | | `type` | `ComputeType | None` | An enumeration. | `None` | | `name` | `str | None` | The name of a compute resource. | `None` | | `resources` | `ComputeResources | None` | An enumeration. | `None` |

#### ComputeResources

A set of available hardware resources for a compute resource.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `cpus` | `int | None` | The number of CPU cores | `None` | | `memory` | `float | None` | The amount of memory in GB | `None` | | `gpus` | `int | None` | The number of GPUs | `0` | | `gpu_memory` | `float | None` | The amount of GPU memory in GB | `0` |

#### ComputeType

The type of compute.

#### Connector

A connector is a connection to a data source or a data destination.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `id` | `str` | The unique identifier of a connector. | *required* | | `name` | `str | None` | The name of a connector. | `None` | | `description` | `str | None` | The description of a connector. | `None` | | `type` | `ConnectorType` | An enumeration. | *required* | | `access_type` | `ConnectorAccessType | None` | An enumeration. | `<ConnectorAccessType.read_protected: 'READ_PROTECTED'>` | | `config` | `dict[str, Any] | None` | An enumeration. | `None` | | `secrets` | `dict[str, str] | None` | An enumeration. | `None` | | `ssl` | `dict[str, str] | None` | An enumeration. | `None` | | `metadata` | `Metadata | None` | An enumeration. | `None` | | `usage` | `ConnectorUsage | None` | An enumeration. | `None` | | `table_id` | `str | None` | Optional. ID of a source table or a synthetic table, that this connector belongs to. If not set, then this connector is managed independently of any generator or synthetic dataset. | `None` |

##### delete

```python
delete()

```

Delete the connector.

##### delete_data

```python
delete_data(location)

```

Delete data from the specified location within the connector. This method is only available for connectors of access_type WRITE_DATA.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `location` | `str` | The target location within the connector to delete data from. | *required* |

Example

```python
c.delete_data('db_name.table_name')  # drop table data from 'table_name' in 'db_name' for a DB connector
c.delete_data('s3://my_bucket/path/to/file.csv')  # delete data from 'file.csv' in 'my_bucket' for a S3 storage connector

```

##### locations

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

| Name | Type | Description | Default | | --- | --- | --- | --- | | `prefix` | `str` | The prefix to filter the results by. Defaults to an empty string. | `''` |

Returns:

| Type | Description | | --- | --- | | `list[str]` | list\[str\]: A list of locations (schemas, databases, directories, etc.). |

Example

```python
c.locations()  # list all schemas / databases for a DB connector; list all buckets for a storage connector
c.locations('db_name')  # list all tables in 'db_name' for a DB connector
c.locations('s3://my_bucket')  # list all objects in 'my_bucket' for a S3 storage connector
c.locations('gs://my_bucket/path/to/folder')  # list all objects in 'my_bucket/path/to/folder' for a GCP storage connector
c.locations('az://my_container/path/to/folder')  # list all objects in 'my_container/path/to/folder' for a AZURE storage connector

```

##### query

```python
query(sql)

```

Execute a read-only SQL query against the connector's data source.

Queries can include statements like SELECT, SHOW, or DESCRIBE, but must not modify data or state. For file-based connectors (S3_STORAGE, GOOGLE_CLOUD_STORAGE, AZURE_STORAGE) queries are executed using DuckDB. Use connector-type-specific prefixes. See examples.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `sql` | `str` | The SQL query to execute. | *required* |

Returns:

| Type | Description | | --- | --- | | `DataFrame` | pd.DataFrame: The result of the query as a Pandas DataFrame. |

Example

```python
df = c.query("SELECT count(*) FROM schema.table")  # for DB connectors
df = c.query("SELECT count(*) FROM read_csv_auto('s3://bucket/path/to/file.csv')")  # query a single CSV file from S3 storage
df = c.query("SELECT count(*) FROM read_parquet('gs://bucket/path/to/folder/*.parquet')")  # query a folder with PQT files from GCP storage
df = c.query("SELECT count(*) FROM read_json('az://bucket/path/to/file.json')")  # query a single JSON file from AZURE storage

```

##### read_data

```python
read_data(location, limit=None, shuffle=False)

```

Retrieve data from the specified location within the connector. This method is only available for connectors of access_type READ_DATA or WRITE_DATA.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `location` | `str` | The target location within the connector to read data from. | *required* | | `limit` | `int | None` | The maximum number of rows to return. Returns all if not specified. | `None` | | `shuffle` | `bool | None` | Whether to shuffle the results. | `False` |

Returns:

| Type | Description | | --- | --- | | `DataFrame` | pd.DataFrame: A DataFrame containing the retrieved data. |

Example

```python
df = c.read_data('db_name.table_name', limit=100)  # fetch first 100 rows from 'table_name' in 'db_name' for a DB connector
df = c.read_data('s3://my_bucket/path/to/file.csv')  # read all data from 'file.csv' in 'my_bucket' for a S3 storage connector

```

##### schema

```python
schema(location)

```

Retrieve the schema of the table at a connector location. This method is available for all connectors.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `location` | `str` | The location of the table. | *required* |

Returns:

| Type | Description | | --- | --- | | `list[dict[str, Any]]` | list\[dict[str, Any]\]: The retrieved schema. |

Example

```python
c.schema('db_name.table_name')  # get the schema of 'table_name' in 'db_name' for a DB connector
c.schema('s3://my_bucket/path/to/file.csv')  # get the schema of 'file.csv' in 'my_bucket' for a S3 storage connector

```

##### update

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

| Name | Type | Description | Default | | --- | --- | --- | --- | | `name` | `str | None` | The name of the connector. | `None` | | `description` | `str | None` | The description of the connector. | `None` | | `access_type` | `ConnectorAccessType | None` | The access type of the connector. | `None` | | `config` | `dict[str, Any] | None` | Connector configuration. | `None` | | `secrets` | `dict[str, str] | None` | Secret values for the connector. | `None` | | `ssl` | `dict[str, str] | None` | SSL configuration for the connector. | `None` | | `test_connection` | `bool | None` | If true, validates the connection before saving. | `True` |

##### write_data

```python
write_data(data, location, if_exists='fail')

```

Write data to the specified location within the connector. This method is only available for connectors of access_type WRITE_DATA.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `data` | `DataFrame | None` | The DataFrame to write, or None to delete the location. | *required* | | `location` | `str` | The target location within the connector to write data to. | *required* | | `if_exists` | `Literal['append', 'replace', 'fail']` | The behavior if the target location already exists (append, replace, fail). Default is "fail". | `'fail'` |

Example

```python
c.write_data(df, 'db_name.table_name', if_exists='fail')  # write data to 'table_name' in 'db_name' for a DB connector (if it doesn't exist)
c.write_data(df, 's3://my_bucket/path/to/file.csv')  # write data to 'file.csv' in 'my_bucket' for a S3 storage connector

```

#### ConnectorAccessType

The access permissions of a connector.

- `READ_PROTECTED`: The connector is restricted to being used solely as a source for training a generator. Direct data access is not permitted, only schema access is available.
- `READ_DATA`: This connector allows full read access. It can also be used as a source for training a generator.
- `WRITE_DATA`: This connector allows full read and write access. It can be also used as a source for training a generator, as well as a destination for delivering a synthetic dataset.
- `SOURCE`: DEPRECATED - equivalent to READ_PROTECTED
- `DESTINATION`: DEPRECATED - equivalent to WRITE_DATA

#### ConnectorConfig

The structures of the config, secrets and ssl parameters depend on the connector type.

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
      sslEnabled: boolean, default: false
    secrets:
      secretKey: string
    ssl:
      caCertificate: base64-encoded string

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
      kerberosServicePrincipal: string (required if kerberosEnabled)
      kerberosClientPrincipal: string (optional if kerberosEnabled)
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

  - type: SQLITE
    config:
      database: string

  ```

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `name` | `str | None` | The name of a connector. | `None` | | `description` | `str | None` | The description of a connector. | `None` | | `type` | `ConnectorType` | An enumeration. | *required* | | `access_type` | `ConnectorAccessType | None` | An enumeration. | `<ConnectorAccessType.read_protected: 'READ_PROTECTED'>` | | `config` | `dict[str, Any] | None` | An enumeration. | `None` | | `secrets` | `dict[str, str] | None` | An enumeration. | `None` | | `ssl` | `dict[str, str] | None` | An enumeration. | `None` |

#### ConnectorDeleteDataConfig

Configuration for deleting data from a connector.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `location` | `str` | Specifies the target within the connector to delete. The format of this parameter varies by connector type. | *required* |

#### ConnectorListItem

Essential connector details for listings.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `id` | `str` | The unique identifier of a connector. | *required* | | `name` | `str | None` | The name of a connector. | `None` | | `description` | `str | None` | The description of a connector. | `None` | | `type` | `ConnectorType` | An enumeration. | *required* | | `access_type` | `ConnectorAccessType | None` | An enumeration. | `<ConnectorAccessType.read_protected: 'READ_PROTECTED'>` | | `metadata` | `Metadata | None` | An enumeration. | `None` | | `usage` | `ConnectorUsage | None` | An enumeration. | `None` |

#### ConnectorQueryConfig

Attributes: sql (str): SQL read-only (e.g. SELECT) statement to execute.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `sql` | `str` | SQL read-only (e.g. SELECT) statement to execute. | *required* |

#### ConnectorReadDataConfig

Configuration for reading data from a connector.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `location` | `str | None` | Specifies the target within the connector from which to retrieve the data. The format of this parameter varies by connector type. | `None` | | `limit` | `int | None` | The maximum number of rows to return. Return all if not specified. | `None` | | `shuffle` | `bool | None` | Whether to shuffle the results. | `False` |

#### ConnectorType

The type of a connector.

The type determines the structure of the config, secrets and ssl parameters.

- `MYSQL`: MySQL database
- `POSTGRES`: PostgreSQL database
- `MSSQL`: Microsoft SQL Server database
- `ORACLE`: Oracle database
- `MARIADB`: MariaDB database
- `SNOWFLAKE`: Snowflake cloud data platform
- `BIGQUERY`: Google BigQuery cloud data warehouse
- `HIVE`: Apache Hive database
- `DATABRICKS`: Databricks cloud data platform
- `SQLITE`: SQLite database
- `AZURE_STORAGE`: Azure Blob Storage
- `GOOGLE_CLOUD_STORAGE`: Google Cloud Storage
- `S3_STORAGE`: Amazon S3 Storage
- `FILE_UPLOAD`: File upload

#### ConnectorUsage

Usage statistics of a connector.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `no_of_generators` | `int | None` | Number of generators using this connector. | `None` | | `no_of_likes` | `int | None` | Number of likes of this connector. | `None` |

#### ConnectorWriteDataConfig

Configuration for writing data to a connector.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `file` | `bytes` | Binary Parquet file containing the data to write. | *required* | | `location` | `str` | Specifies the target within the connector to which to write the data. The format of this parameter varies by connector type. | *required* | | `if_exists` | `IfExists | None` | The behavior if the target location already exists. APPEND: Append the data to the existing target. REPLACE: Replace the existing target with the new data. FAIL: Fail if the target already exists. | `None` |

#### Credits

The credit balance and limit for the current time period

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `current` | `float | None` | The credit balance for the current time period | `None` | | `limit` | `float | None` | The credit limit for the current time period. If empty, then there is no limit. | `None` | | `period_start` | `datetime | None` | The UTC date and time when the current time period started | `None` | | `period_end` | `datetime | None` | The UTC date and time when the current time period ends | `None` |

#### CurrentUser

Information on the current user.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `id` | `str | None` | The unique identifier of a user. | `None` | | `name` | `str | None` | The name of a user. Contains only alphanumeric characters, hyphens, and underscores. Must start or end with alphanumeric. It must be globally case-insensitive unique considering organizations and users. | `None` | | `first_name` | `str | None` | First name of a user | `None` | | `last_name` | `str | None` | Last name of a user | `None` | | `email` | `str | None` | The email of a user | `None` | | `avatar` | `str | None` | The URL of the user's avatar | `None` | | `settings` | `dict[str, Any] | None` | An enumeration. | `None` | | `usage` | `UserUsage | None` | An enumeration. | `None` | | `unread_notifications` | `int | None` | Number of unread notifications for the user | `None` | | `organizations` | `list[OrganizationListItem] | None` | The organizations the user belongs to | `None` |

#### DifferentialPrivacyConfig

The optional differential privacy configuration for training the model. If not provided, then no differential privacy will be applied.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `max_epsilon` | `float | None` | Specifies the maximum allowable epsilon value. If the training process exceeds this threshold, it will be terminated early. Only model checkpoints with epsilon values below this limit will be retained. If not provided, the training will proceed without early termination based on epsilon constraints. | `10.0` | | `delta` | `float | None` | The delta value for differential privacy. It is the probability of the privacy guarantee not holding. The smaller the delta, the more confident you can be that the privacy guarantee holds. This delta will be equally distributed between the analysis and the training phase. | `'1e-5'` | | `noise_multiplier` | `float | None` | Determines how much noise while training the model with differential privacy. This is the ratio of the standard deviation of the Gaussian noise to the L2-sensitivity of the function to which the noise is added. | `1.5` | | `max_grad_norm` | `float | None` | Determines the maximum impact of a single sample on updating the model weights during training with differential privacy. This is the maximum norm of the per-sample gradients. | `1.0` | | `value_protection_epsilon` | `float | None` | The DP epsilon of the privacy budget for determining the value ranges, which are gathered prior to the model training during the analysis step. Only applicable if value protection is True. Privacy budget will be equally distributed between the columns. For categorical we calculate noisy histograms and use a noisy threshold. For numeric and datetime we calculate bounds based on noisy histograms. | `1.0` |

#### Distances

Metrics regarding the nearest neighbor distances between training, holdout, and synthetic samples in an numerically encoded space. Useful for assessing the novelty / privacy of synthetic data.

The provided data is first down-sampled, so that the number of samples match across all datasets. Note, that for an optimal sensitivity of this privacy assessment it is recommended to use a 50/50 split between training and holdout data, and then generate synthetic data of the same size.

The numerical encodings of these samples are then computed, and the nearest neighbor distances are calculated for each synthetic sample to the training and holdout samples. Based on these nearest neighbor distances the following metrics are calculated:

- Identical Match Share (IMS): The share of synthetic samples that are identical to a training or holdout sample.
- Distance to Closest Record (DCR): The average distance of synthetic to training or holdout samples.
- Nearest Neighbor Distance Ratio (NNDR): The 10-th smallest ratio of the distance to nearest and second nearest neighbor.

For privacy-safe synthetic data we expect to see about as many identical matches, and about the same distances for synthetic samples to training, as we see for synthetic samples to holdout.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `ims_training` | `float | None` | Share of synthetic samples that are identical to a training sample. | `None` | | `ims_holdout` | `float | None` | Share of synthetic samples that are identical to a holdout sample. Serves as a reference for ims_training. | `None` | | `ims_trn_hol` | `float | None` | Share of training samples that are identical to a holdout sample. Serves as a reference for ims_training. | `None` | | `dcr_training` | `float | None` | Average nearest-neighbor distance between synthetic and training samples. | `None` | | `dcr_holdout` | `float | None` | Average nearest-neighbor distance between synthetic and holdout samples. Serves as a reference for dcr_training. | `None` | | `dcr_trn_hol` | `float | None` | Average nearest-neighbor distance between training and holdout samples. Serves as a reference for dcr_training. | `None` | | `dcr_share` | `float | None` | Share of synthetic samples that are closer to a training sample than to a holdout sample. This should not be significantly larger than 50%. | `None` | | `nndr_training` | `float | None` | 10th smallest nearest-neighbor distance ratio between synthetic and training samples. | `None` | | `nndr_holdout` | `float | None` | 10th smallest nearest-neighbor distance ratio between synthetic and holdout samples. | `None` | | `nndr_trn_hol` | `float | None` | 10th smallest nearest-neighbor distance ratio between training and holdout samples. | `None` |

#### ErrorEvent

An error event containing an error message

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `event` | `Literal[str] | None` | An enumeration. | `None` | | `data` | `ErrorMessage | None` | An enumeration. | `None` |

#### ErrorMessage

An error message

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `message` | `str | None` | The error message | `None` |

#### FairnessConfig

Configure a fairness objective for the table. Only applicable for a subject table. The generated synthetic data will maintain robust statistical parity between the target column and the specified sensitive columns. All these columns must be categorical.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `target_column` | `str` | The name of the target column. | *required* | | `sensitive_columns` | `list[str]` | The names of the sensitive columns. | *required* |

#### FilterByUser

Determines whether to filter usage reports for all users or only the current user.

- `ALL`: Filter usage reports for all users. Only accessible for SuperAdmins.
- `ME`: Filter usage reports for the current user.

#### Generator

A generator is a set models that can generate synthetic data.

The generator can be trained on one or more source tables. A quality assurance report is generated for each model.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `id` | `str` | The unique identifier of a generator. | *required* | | `name` | `str | None` | The name of a generator. | `None` | | `description` | `str | None` | The description of a generator. | `None` | | `training_status` | `ProgressStatus` | An enumeration. | *required* | | `training_time` | `datetime | None` | The UTC date and time when the training has finished. | `None` | | `usage` | `GeneratorUsage | None` | An enumeration. | `None` | | `metadata` | `Metadata | None` | An enumeration. | `None` | | `accuracy` | `float | None` | The overall accuracy of the trained generator. This is the average of the overall accuracy scores of all trained models. | `None` | | `tables` | `list[SourceTable] | None` | The tables of this generator | `None` | | `random_state` | `int | None` | Seed for the random number generators. If None, the random number generator is initialized randomly, yielding different results for every run. Setting it to a specific integer ensures reproducible results across runs. Useful when consistent results are desired, e.g. for testing or debugging. | `None` | | `training` | `Any | None` | An enumeration. | `None` |

##### Training

###### cancel

```python
cancel()

```

Cancel training.

###### logs

```python
logs(file_path=None)

```

Download the training logs and save to file.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `file_path` | `str | Path | None` | The file path to save the logs. Default is the current working directory. | `None` |

Returns:

| Name | Type | Description | | --- | --- | --- | | `Path` | `Path` | The path to the saved file. |

###### progress

```python
progress()

```

Retrieve job progress of training.

Returns:

| Name | Type | Description | | --- | --- | --- | | `JobProgress` | `JobProgress` | The job progress of the training process. |

###### start

```python
start()

```

Start training.

###### wait

```python
wait(progress_bar=True, interval=2)

```

Poll training progress and loop until training has completed.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `progress_bar` | `bool` | If true, displays the progress bar. Default is True. | `True` | | `interval` | `float` | The interval in seconds to poll the job progress. Default is 2 seconds. | `2` |

##### clone

```python
clone(training_status='new')

```

Clone the generator.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `training_status` | `Literal['new', 'continue']` | The training status of the cloned generator. Default is "new". | `'new'` |

Returns:

| Name | Type | Description | | --- | --- | --- | | `Generator` | `Generator` | The cloned generator object. |

##### config

```python
config()

```

Retrieve writable generator properties.

Returns:

| Name | Type | Description | | --- | --- | --- | | `GeneratorConfig` | `GeneratorConfig` | The generator properties as a configuration object. |

##### delete

```python
delete()

```

Delete the generator.

##### export_to_file

```python
export_to_file(file_path=None)

```

Export generator and save to file.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `file_path` | `str | Path | None` | The file path to save the generator. | `None` |

Returns:

| Name | Type | Description | | --- | --- | --- | | `Path` | `Path` | The path to the saved file. |

##### reports

```python
reports(file_path=None, display=False)

```

Download or display the quality assurance reports.

If display is True, the report is rendered inline via IPython display and no file is downloaded. Otherwise, the report is downloaded and saved to file_path (or a default location if None).

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `file_path` | `str | Path | None` | The file path to save the zipped reports (ignored if display=True). | `None` | | `display` | `bool` | If True, render the report inline instead of downloading it. | `False` |

Returns:

| Type | Description | | --- | --- | | `Path | None` | Path | None: The path to the saved file if downloading, or None if display=True. |

##### update

```python
update(name=None, description=None)

```

Update a generator with specific parameters.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `name` | `str | None` | The name of the generator. | `None` | | `description` | `str | None` | The description of the generator. | `None` |

#### GeneratorCloneTrainingStatus

The training status of the new generator. The available options are:

- `NEW`: The new generator will re-use existing data and model configurations.
- `CONTINUE`: The new generator will re-use existing data and model configurations, as well as model weights.

#### GeneratorConfig

The configuration for creating a new generator.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `name` | `str | None` | The name of a generator. | `None` | | `description` | `str | None` | The description of a generator. | `None` | | `random_state` | `int | None` | Seed for the random number generators. If None, the random number generator is initialized randomly, yielding different results for every run. Setting it to a specific integer ensures reproducible results across runs. Useful when consistent results are desired, e.g. for testing or debugging. | `None` | | `tables` | `list[SourceTableConfig] | None` | The tables of a generator | `None` |

#### GeneratorImportFromFileConfig

Configuration for importing a generator from a file.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `file` | `bytes` | An enumeration. | *required* |

#### GeneratorListItem

Essential generator details for listings.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `id` | `str` | The unique identifier of a generator. | *required* | | `name` | `str | None` | The name of a generator. | `None` | | `description` | `str | None` | The description of a generator. | `None` | | `training_status` | `ProgressStatus` | An enumeration. | *required* | | `training_time` | `datetime | None` | The UTC date and time when the training has finished. | `None` | | `usage` | `GeneratorUsage | None` | An enumeration. | `None` | | `metadata` | `Metadata | None` | An enumeration. | `None` | | `accuracy` | `float | None` | The overall accuracy of the trained generator. This is the average of the overall accuracy scores of all trained models. | `None` |

#### GeneratorUsage

Usage statistics of a generator.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `total_datapoints` | `int | None` | The total number of datapoints generated by this generator. Deprecated: This field is no longer valid and will always return -1. It will be removed in a future version. | `None` | | `total_compute_time` | `int | None` | The total compute time in seconds used for training this generator. This is the sum of the elapsed compute time of all training tasks. | `None` | | `total_credits` | `float | None` | The amount of credits consumed for training the generator. | `None` | | `total_virtual_cpu_time` | `float | None` | The total virtual CPU time in seconds used for training this generator. This is the sum of the elapsed time multiplied by number of allocated virtual CPUs across all training tasks. | `None` | | `total_virtual_gpu_time` | `float | None` | The total virtual GPU time in seconds used for training this generator. This is the sum of the elapsed time multiplied by number of allocated virtual GPUs across all training tasks. | `None` | | `no_of_synthetic_datasets` | `int | None` | Number of synthetic datasets generated by this generator. | `None` | | `no_of_likes` | `int | None` | Number of likes of this generator. | `None` |

#### HeartbeatEvent

A heartbeat event to keep the connection alive

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `event` | `Literal[str] | None` | An enumeration. | `None` |

#### IfExists

The behavior if the target location already exists.

- `APPEND`: Append the data to the existing target.
- `REPLACE`: Replace the existing target with the new data.
- `FAIL`: Fail if the target already exists.

#### ImputationConfig

Configure imputation.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `columns` | `list[str]` | The names of the columns to be imputed. Imputed columns will suppress the sampling of NULL values. | *required* |

#### JobProgress

The progress of a job.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `id` | `str | None` | An enumeration. | `None` | | `start_date` | `datetime | None` | The UTC date and time when the job has started. If the job has not started yet, then this is None. | `None` | | `end_date` | `datetime | None` | The UTC date and time when the job has ended. If the job is still, then this is None. | `None` | | `progress` | `ProgressValue | None` | An enumeration. | `None` | | `status` | `ProgressStatus | None` | An enumeration. | `None` | | `steps` | `list[ProgressStep] | None` | An enumeration. | `None` |

#### MemberRole

The role of the user in the organization

- `VIEWER`: The user can view and use all resources of the organization
- `CONTRIBUTOR`: The user can create new resources for an organization, and becomes resource ADMIN
- `ADMIN`: The user can manage members and all resources of an organization

#### MessageEvent

A message event containing an assistant message delta

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `event` | `Literal[str] | None` | An enumeration. | `None` | | `data` | `AssistantMessageDelta | None` | An enumeration. | `None` |

#### MessageStreamEvent

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `root` | `MessageEvent | HeartbeatEvent | ErrorEvent` | An event in the server-sent event stream | *required* |

#### Metadata

The metadata of a resource.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `creator_id` | `str | None` | The unique identifier of a user. | `None` | | `creator_name` | `str | None` | The name of a user. Contains only alphanumeric characters, hyphens, and underscores. Must start or end with alphanumeric. It must be globally case-insensitive unique considering organizations and users. | `None` | | `created_at` | `datetime | None` | The UTC date and time when the resource has been created. | `None` | | `owner_id` | `str | None` | The unique identifier of an account (either a user or an organization). | `None` | | `owner_name` | `str | None` | The name of an account (either a user or an organization). | `None` | | `owner_type` | `AccountType | None` | An enumeration. | `None` | | `owner_image` | `str | None` | The URL of the account's image. | `None` | | `visibility` | `Visibility | None` | An enumeration. | `None` | | `current_user_permission_level` | `PermissionLevel | None` | An enumeration. | `None` | | `current_user_like_status` | `bool | None` | A boolean indicating whether the user has liked the entity or not | `None` | | `short_lived_file_token` | `str | None` | An auto-generated short-lived file token (slft) for accessing resource artifacts. The token is always restricted to a single resource, only valid for 60 minutes, and only accepted by API endpoints that allow to download single files. | `None` |

#### ModelConfiguration

The training configuration for the model

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `model` | `str | None` | The model to be used for training. | `None` | | `max_sample_size` | `int | None` | The maximum number of samples to consider for training. If not provided, then all available samples will be taken. | `None` | | `batch_size` | `int | None` | The physical batch size used for training the model. If not provided, batchSize will be chosen automatically. | `None` | | `gradient_accumulation_steps` | `int | None` | Steps to accumulate gradients before optimizer update. If not provided, gradientAccumulationSteps will be chosen automatically. | `None` | | `max_training_time` | `float | None` | The maximum number of minutes to train the model. | `14400` | | `max_epochs` | `float | None` | The maximum number of epochs to train the model. | `100` | | `max_sequence_window` | `int | None` | The maximum sequence window to consider for training. Only applicable for TABULAR models. | `100` | | `enable_flexible_generation` | `bool | None` | If true, then the trained generator can be used for conditional generation, rebalancing, imputation and fairness. If none of these will be needed, then one can gain extra accuracy by disabling this feature. This will then result in a fixed column order being fed into the training process, rather than a column order, that is randomly permuted for every batch. | `True` | | `value_protection` | `bool | None` | Defines if Rare Category, Extreme value, or Sequence length protection will be applied. | `True` | | `rare_category_replacement_method` | `RareCategoryReplacementMethod | None` | Specifies how rare categories will be sampled. Only applicable if value protection has been enabled. CONSTANT: Replace rare categories by a constant _RARE_ token. SAMPLE: Replace rare categories by a sample from non-rare categories. | `<RareCategoryReplacementMethod.constant: 'CONSTANT'>` | | `differential_privacy` | `DifferentialPrivacyConfig | None` | An enumeration. | `None` | | `compute` | `str | None` | The unique identifier of a compute resource. Not applicable for SDK. | `None` | | `enable_model_report` | `bool | None` | If false, then the Model report is not generated. | `True` |

#### ModelEncodingType

The encoding type used for model training and data generation.

- `AUTO`: Model chooses among available encoding types based on the column's data type.
- `TABULAR_CATEGORICAL`: Model samples from existing (non-rare) categories.
- `TABULAR_NUMERIC_AUTO`: Model chooses among 3 numeric encoding types based on the values.
- `TABULAR_NUMERIC_DISCRETE`: Model samples from existing discrete numerical values.
- `TABULAR_NUMERIC_BINNED`: Model samples from binned buckets, to then sample randomly within a bucket.
- `TABULAR_NUMERIC_DIGIT`: Model samples each digit of a numerical value.
- `TABULAR_CHARACTER`: Model samples each character of a string value.
- `TABULAR_DATETIME`: Model samples each part of a datetime value.
- `TABULAR_DATETIME_RELATIVE`: Model samples the relative difference between datetimes within a sequence.
- `TABULAR_LAT_LONG`: Model samples a latitude-longitude column. The format is "latitude,longitude".
- `LANGUAGE_TEXT`: Model will sample free text, using a LANGUAGE model.
- `LANGUAGE_CATEGORICAL`: Model samples from existing (non-rare) categories, using a LANGUAGE model.
- `LANGUAGE_NUMERIC`: Model samples from the valid numeric value range, using a LANGUAGE model.
- `LANGUAGE_DATETIME`: Model samples from the valid datetime value range, using a LANGUAGE model.

#### ModelMetrics

Metrics regarding the quality of synthetic data, measured in terms of accuracy, similarity, and distances.

1. **Accuracy**: Metrics regarding the accuracy of synthetic data, measured as the closeness of discretized lower dimensional marginal distributions.
1. **Similarity**: Metrics regarding the similarity of the full joint distributions of samples within an embedding space.
1. **Distances**: Metrics regarding the nearest neighbor distances between training, holdout, and synthetic samples in an numeric encoding space. Useful for assessing the novelty / privacy of synthetic data.

The quality of synthetic data is assessed by comparing these metrics to the same metrics of a holdout dataset. The holdout dataset is a subset of the original training data, that was not used for training the synthetic data generator. The metrics of the synthetic data should be as close as possible to the metrics of the holdout data.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `accuracy` | `Accuracy | None` | An enumeration. | `None` | | `distances` | `Distances | None` | An enumeration. | `None` | | `similarity` | `Similarity | None` | An enumeration. | `None` |

#### ModelType

The type of model.

- `TABULAR`: A generative AI model tailored towards tabular data, trained from scratch.
- `LANGUAGE`: A generative AI model build upon a (pre-trained) language model.

#### Notification

A notification for a user.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `id` | `str` | The unique identifier of the notification. | *required* | | `type` | `NotificationType` | An enumeration. | *required* | | `message` | `str` | The message of the notification. | *required* | | `status` | `NotificationStatus` | An enumeration. | *required* | | `created_at` | `datetime` | The UTC date and time when the notification has been created. | *required* | | `resource_uri` | `str | None` | The service URI of the entity | `None` |

#### NotificationStatus

The status of the notification.

#### NotificationType

The type of the notification

#### Organization

An organization that owns resources.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `id` | `str` | The unique identifier of an organization. | *required* | | `name` | `str` | The name of an organization. Contains only alphanumeric characters, hyphens, and underscores. Must start or end with alphanumeric. It must be globally case-insensitive unique. | *required* | | `display_name` | `str` | The display name of an organization. | *required* | | `description` | `str | None` | The description of an organization. Supports markdown. | `None` | | `logo` | `str | None` | The URL of the organization's logo. | `None` | | `email` | `str | None` | The email address of the organization. | `None` | | `website` | `str | None` | The URL of the organization's website. | `None` | | `members` | `list[UserListItem] | None` | An enumeration. | `None` | | `metadata` | `OrganizationMetadata | None` | An enumeration. | `None` |

#### OrganizationConfig

The configuration for creating a new organization.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `name` | `str` | The name of an organization. Contains only alphanumeric characters, hyphens, and underscores. Must start or end with alphanumeric. It must be globally case-insensitive unique. | *required* | | `display_name` | `str` | The display name of an organization. | *required* | | `description` | `str | None` | The description of an organization. Supports markdown. | `None` | | `logo_base64` | `str | None` | The base64-encoded image of the organization's logo. | `None` | | `email` | `str | None` | The email address of the organization. | `None` | | `website` | `str | None` | The URL of the organization's website. | `None` |

#### OrganizationInvite

A non-personalized time-boxed invite to join an organization.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `token` | `str | None` | The generated token, encrypting organization, expiration timestamp, and role (VIEW). | `None` | | `link` | `str | None` | The generated invite link. | `None` | | `expiration_date` | `datetime | None` | The expiration date of the invite link. 72 hours after creation. | `None` | | `organization_id` | `str | None` | The unique identifier of an organization. | `None` |

#### OrganizationListItem

Essential organization details for listings.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `id` | `str` | The unique identifier of an organization. | *required* | | `name` | `str | None` | The name of an organization. Contains only alphanumeric characters, hyphens, and underscores. Must start or end with alphanumeric. It must be globally case-insensitive unique. | `None` | | `display_name` | `str` | The display name of an organization. | *required* | | `description` | `str | None` | The description of an organization. Supports markdown. | `None` | | `logo` | `str | None` | The URL of the organization's logo. | `None` | | `metadata` | `OrganizationMetadata | None` | An enumeration. | `None` |

#### OrganizationMember

A member of an organization with their associated role.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `user` | `UserListItem | None` | An enumeration. | `None` | | `role` | `MemberRole | None` | An enumeration. | `None` |

#### OrganizationMetadata

The metadata of an organization.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `current_user_member_role` | `MemberRole | None` | An enumeration. | `None` |

#### PaginatedTotalCount

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `root` | `int` | The total number of entities within the list | *required* |

#### ParallelGenerationJobs

The number of currently running generation jobs and the limit

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `current` | `int | None` | The number of currently running generation jobs. | `None` | | `limit` | `int | None` | The maximum number of running generation jobs at any time. If empty, then there is no limit. | `None` |

#### ParallelTrainingJobs

The number of currently running training jobs and the limit

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `current` | `int | None` | The number of currently running training jobs | `None` | | `limit` | `int | None` | The maximum number of running training jobs at any time. If empty, then there is no limit. | `None` |

#### PermissionLevel

The permission level of the user with respect to this resource

- `VIEW`: The user can view and use the resource
- `ADMIN`: The user can edit, delete and transfer ownership of the resource

#### Probe

The generated synthetic samples returned as a result of the probe.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `name` | `str | None` | The name of the table. | `None` | | `rows` | `list[dict[str, Any]] | None` | An array of sample data objects. | `None` |

#### ProgressStatus

The status of a job or a step.

- `NEW`: The job/step is being configured, and has not started yet
- `CONTINUE`: The job/step is being configured, but has existing artifacts
- `ON_HOLD`: The job/step has been started, but is kept on hold
- `QUEUED`: The job/step has been started, and is awaiting for resources to execute
- `IN_PROGRESS`: The job/step is currently running
- `DONE`: The job/step has finished successfully
- `FAILED`: The job/step has failed
- `CANCELED`: The job/step has been canceled

#### ProgressStep

The progress of a step.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `id` | `str | None` | The unique identifier of the step. | `None` | | `model_label` | `str | None` | The unique label for the model, consisting of table name and a suffix for the model type. This will be empty for steps that are not related to a model. | `None` | | `compute_name` | `str | None` | The name of a compute resource. | `None` | | `restarts` | `int | None` | The number of previous restarts for the corresponding task. | `0` | | `task_type` | `TaskType | None` | An enumeration. | `None` | | `step_code` | `StepCode | None` | An enumeration. | `None` | | `start_date` | `datetime | None` | The UTC date and time when the job has started. If the job has not started yet, then this is None. | `None` | | `end_date` | `datetime | None` | The UTC date and time when the job has ended. If the job is still, then this is None. | `None` | | `compute_resources` | `ComputeResources | None` | An enumeration. | `None` | | `messages` | `list[dict[str, Any]] | None` | An enumeration. | `None` | | `error_message` | `str | None` | An enumeration. | `None` | | `progress` | `ProgressValue | None` | An enumeration. | `None` | | `status` | `ProgressStatus | None` | An enumeration. | `None` |

#### ProgressValue

The progress of a job or a step.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `value` | `int | None` | An enumeration. | `0` | | `max` | `int | None` | An enumeration. | `1` |

#### RareCategoryReplacementMethod

Specifies how rare categories will be sampled. Only applicable if value protection has been enabled.

- `CONSTANT`: Replace rare categories by a constant `_RARE_` token.
- `SAMPLE`: Replace rare categories by a sample from non-rare categories.

#### RebalancingConfig

Configure rebalancing.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `column` | `str` | The name of the column to be rebalanced. Only applicable for a subject table. Only applicable for categorical columns. | *required* | | `probabilities` | `dict[str, float]` | The target distribution of samples values. The keys are the categorical values, and the values are the probabilities. | *required* |

#### SetVisibilityConfig

Configuration for setting the visibility of a resource.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `visibility` | `Visibility` | An enumeration. | *required* |

#### Similarity

Metrics regarding the similarity of the full joint distributions of samples within an embedding space.

1. **Cosine Similarity**: The cosine similarity between the centroids of synthetic and training samples.
1. **Discriminator AUC**: The AUC of a discriminative model to distinguish between synthetic and training samples.

The Model2Vec model [potion-base-8M](https://huggingface.co/minishlab/potion-base-8M) is used to compute the embeddings of a string-ified representation of individual records. In case of sequential data the records, that belong to the same group, are being concatenated. We then calculate the cosine similarity between the centroids of the provided datasets within the embedding space.

Again, we expect the similarity metrics to be as close as possible to 1, but not significantly higher than what is measured for the holdout data, as this would again indicate overfitting.

In addition, a discriminative ML model is trained to distinguish between training and synthetic samples. The ability of this model to distinguish between training and synthetic samples is measured by the AUC score. For synthetic data to be considered realistic, the AUC score should be close to 0.5, which indicates that the synthetic data is indistinguishable from the training data.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `cosine_similarity_training_synthetic` | `float | None` | Cosine similarity between training and synthetic centroids. | `None` | | `cosine_similarity_training_holdout` | `float | None` | Cosine similarity between training and holdout centroids. Serves as a reference for cosine_similarity_training_synthetic. | `None` | | `discriminator_auc_training_synthetic` | `float | None` | Cross-validated AUC of a discriminative model to distinguish between training and synthetic samples. | `None` | | `discriminator_auc_training_holdout` | `float | None` | Cross-validated AUC of a discriminative model to distinguish between training and holdout samples. Serves as a reference for discriminator_auc_training_synthetic. | `None` |

#### SourceColumn

A column as part of a source table.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `id` | `str` | The unique identifier of a source column. | *required* | | `name` | `str` | The name of a source column. It must be unique within a source table. | *required* | | `included` | `bool | None` | If true, the column will be included in the training. If false, the column will be excluded from the training. | `True` | | `model_encoding_type` | `ModelEncodingType` | An enumeration. | *required* | | `value_range` | `SourceColumnValueRange | None` | An enumeration. | `None` |

#### SourceColumnConfig

The configuration for a source column when creating a new generator.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `name` | `str` | The name of a source column. It must be unique within a source table. | *required* | | `model_encoding_type` | `ModelEncodingType | None` | An enumeration. | `<ModelEncodingType.auto: 'AUTO'>` |

#### SourceColumnValueRange

The (privacy-safe) range of values detected within a source column. These values can then be used as seed values for conditional generation. For CATEGORICAL and NUMERIC_DISCRETE encoding types, this will be given as a list of unique values, sorted by popularity. For other NUMERIC and for DATETIME encoding types, this will be given as a min and max value. Note, that this property will only be populated, once the analysis step for the training of the generator has been completed.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `min` | `str | None` | The minimum value of the column. For dates, this is represented in ISO format. | `None` | | `max` | `str | None` | The maximum value of the column. For dates, this is represented in ISO format. | `None` | | `values` | `list[str] | None` | The list of distinct values of the column. Limited to a maximum of 1000 values. | `None` | | `has_null` | `bool | None` | If true, null value was detected within the column. | `None` |

#### SourceForeignKey

A foreign key relationship in a source table.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `id` | `str` | The unique identifier of a foreign key. | *required* | | `column` | `str` | The column name of a foreign key. | *required* | | `referenced_table` | `str` | The table name of the referenced table. That table must have a primary key already defined. | *required* | | `is_context` | `bool` | If true, then the foreign key will be considered as a context relation. Note, that only one foreign key relation per table can be a context relation. | *required* |

#### SourceForeignKeyConfig

Configuration for defining a foreign key relationship in a source table.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `column` | `str` | The column name of a foreign key. | *required* | | `referenced_table` | `str` | The table name of the referenced table. That table must have a primary key already defined. | *required* | | `is_context` | `bool` | If true, then the foreign key will be considered as a context relation. Note, that only one foreign key relation per table can be a context relation. | *required* |

#### SourceTable

A table as part of a generator.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `id` | `str` | The unique identifier of a source table. | *required* | | `source_connector_id` | `str | None` | The unique identifier of a connector. | `None` | | `location` | `str | None` | The location of a source table. Together with the source connector it uniquely identifies a source, and samples data from there. | `None` | | `name` | `str` | The name of a source table. It must be unique within a generator. | *required* | | `primary_key` | `str | None` | The column name of the primary key. | `None` | | `columns` | `list[SourceColumn] | None` | The columns of this generator table. | `None` | | `foreign_keys` | `list[SourceForeignKey] | None` | The foreign keys of a table. | `None` | | `tabular_model_metrics` | `ModelMetrics | None` | An enumeration. | `None` | | `language_model_metrics` | `ModelMetrics | None` | An enumeration. | `None` | | `tabular_model_configuration` | `ModelConfiguration | None` | An enumeration. | `None` | | `language_model_configuration` | `ModelConfiguration | None` | An enumeration. | `None` | | `total_rows` | `int | None` | The total number of rows in the source table while fetching data for training. | `None` |

#### SourceTableAddConfig

Configuration for adding a new source table to a generator.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `source_connector_id` | `str` | The unique identifier of a connector. | *required* | | `location` | `str` | The location of a source table. Together with the source connector it uniquely identifies a source, and samples data from there. | *required* | | `name` | `str | None` | The name of a source table. It must be unique within a generator. | `None` | | `include_children` | `bool | None` | If true, all tables that are referenced by foreign keys will be included. If false, only the selected table will be included. | `False` | | `tabular_model_configuration` | `ModelConfiguration | None` | An enumeration. | `None` | | `language_model_configuration` | `ModelConfiguration | None` | An enumeration. | `None` |

#### SourceTableConfig

The configuration for a source table when creating a new generator.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `name` | `str` | The name of a source table. It must be unique within a generator. | *required* | | `source_connector_id` | `str | None` | The unique identifier of a connector. | `None` | | `location` | `str | None` | The location of a source table. Together with the source connector it uniquely identifies a source, and samples data from there. | `None` | | `data` | `str | None` | The base64-encoded string derived from a Parquet file containing the specified source table. | `None` | | `tabular_model_configuration` | `ModelConfiguration | None` | An enumeration. | `None` | | `language_model_configuration` | `ModelConfiguration | None` | An enumeration. | `None` | | `primary_key` | `str | None` | The column name of the primary key. | `None` | | `foreign_keys` | `list[SourceForeignKeyConfig] | None` | The foreign key configurations of this table. | `None` | | `columns` | `list[SourceColumnConfig] | None` | The column configurations of this table. | `None` |

#### StepCode

The unique code for the step.

#### SyntheticDataset

A synthetic dataset is created based on a trained generator.

It consists of synthetic samples, as well as a quality assurance report.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `id` | `str` | The unique identifier of a synthetic dataset. | *required* | | `generator_id` | `str | None` | The unique identifier of a generator. | `None` | | `metadata` | `Metadata | None` | An enumeration. | `None` | | `name` | `str | None` | The name of a synthetic dataset. | `None` | | `description` | `str | None` | The description of a synthetic dataset. | `None` | | `generation_status` | `ProgressStatus` | An enumeration. | *required* | | `generation_time` | `datetime | None` | The UTC date and time when the generation has finished. | `None` | | `tables` | `list[SyntheticTable] | None` | The tables of this synthetic dataset. | `None` | | `delivery` | `SyntheticDatasetDelivery | None` | An enumeration. | `None` | | `accuracy` | `float | None` | The overall accuracy of the trained generator. This is the average of the overall accuracy scores of all trained models. | `None` | | `usage` | `SyntheticDatasetUsage | None` | An enumeration. | `None` | | `compute` | `str | None` | The unique identifier of a compute resource. Not applicable for SDK. | `None` | | `random_state` | `int | None` | Seed for the random number generators. If None, the random number generator is initialized randomly, yielding different results for every run. Setting it to a specific integer ensures reproducible results across runs. Useful when consistent results are desired, e.g. for testing or debugging. | `None` | | `generation` | `Any | None` | An enumeration. | `None` |

##### Generation

###### cancel

```python
cancel()

```

Cancel the generation process.

###### logs

```python
logs(file_path=None)

```

Download the generation logs and save to file.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `file_path` | `str | Path | None` | The file path to save the logs. Default is the current working directory. | `None` |

Returns:

| Name | Type | Description | | --- | --- | --- | | `Path` | `Path` | The path to the saved file. |

###### progress

```python
progress()

```

Retrieve the progress of the generation process.

Returns:

| Name | Type | Description | | --- | --- | --- | | `JobProgress` | `JobProgress` | The progress of the generation process. |

###### start

```python
start()

```

Start the generation process.

###### wait

```python
wait(progress_bar=True, interval=2)

```

Poll the generation progress and wait until the process is complete.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `progress_bar` | `bool` | If true, displays a progress bar. Default is True. | `True` | | `interval` | `float` | Interval in seconds to poll the job progress. Default is 2 seconds. | `2` |

##### config

```python
config()

```

Retrieve writable synthetic dataset properties.

Returns:

| Name | Type | Description | | --- | --- | --- | | `SyntheticDatasetConfig` | `SyntheticDatasetConfig` | The synthetic dataset properties as a configuration object. |

##### data

```python
data(return_type='auto')

```

Download synthetic dataset and return as dictionary of pandas DataFrames.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `return_type` | `Literal['auto', 'dict']` | The type of the return value. "dict" will always provide a dictionary of DataFrames. "auto" will return a single DataFrame for a single-table generator, and a dictionary of DataFrames for a multi-table generator. Default is "auto". | `'auto'` |

Returns:

| Type | Description | | --- | --- | | `DataFrame | dict[str, DataFrame]` | Union\[pd.DataFrame, dict[str, pd.DataFrame]\]: The synthetic dataset. See return_type for the format of the return value. |

##### delete

```python
delete()

```

Delete the synthetic dataset.

##### download

```python
download(file_path=None, format='parquet')

```

Download synthetic dataset and save to file.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `file_path` | `str | Path | None` | The file path to save the synthetic dataset. | `None` | | `format` | `Literal['parquet', 'csv', 'json']` | The format of the synthetic dataset. Default is "parquet". | `'parquet'` |

Returns:

| Name | Type | Description | | --- | --- | --- | | `Path` | `Path` | The path to the saved file. |

##### reports

```python
reports(file_path=None, display=False)

```

Download or display the quality assurance reports.

If display is True, the report is rendered inline via IPython display and no file is downloaded. Otherwise, the report is downloaded and saved to file_path (or a default location if None).

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `file_path` | `str | Path | None` | The file path to save the zipped reports (ignored if display=True). | `None` | | `display` | `bool` | If True, render the report inline instead of downloading it. | `False` |

Returns:

| Type | Description | | --- | --- | | `Path | None` | Path | None: The path to the saved file if downloading, or None if display=True. |

##### update

```python
update(name=None, description=None, delivery=None)

```

Update a synthetic dataset with specific parameters.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `name` | `str | None` | The name of the synthetic dataset. | `None` | | `description` | `str | None` | The description of the synthetic dataset. | `None` | | `delivery` | `SyntheticDatasetDelivery | None` | The delivery configuration for the synthetic dataset. | `None` |

#### SyntheticDatasetConfig

The configuration for creating a new synthetic dataset.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `generator_id` | `str | None` | The unique identifier of a generator. | `None` | | `name` | `str | None` | The name of a synthetic dataset. | `None` | | `description` | `str | None` | The description of a synthetic dataset. | `None` | | `random_state` | `int | None` | Seed for the random number generators. If None, the random number generator is initialized randomly, yielding different results for every run. Setting it to a specific integer ensures reproducible results across runs. Useful when consistent results are desired, e.g. for testing or debugging. | `None` | | `tables` | `list[SyntheticTableConfig] | None` | An enumeration. | `None` | | `delivery` | `SyntheticDatasetDelivery | None` | An enumeration. | `None` | | `compute` | `str | None` | The unique identifier of a compute resource. Not applicable for SDK. | `None` |

#### SyntheticDatasetDelivery

Configuration for delivering a synthetic dataset to a destination.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `overwrite_tables` | `bool` | If true, tables in the destination will be overwritten. If false, any tables exist, the delivery will fail. | *required* | | `destination_connector_id` | `str` | The unique identifier of a connector. | *required* | | `location` | `str` | The location for the destination connector. | *required* |

#### SyntheticDatasetListItem

Essential synthetic dataset details for listings.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `id` | `str` | The unique identifier of a synthetic dataset. | *required* | | `metadata` | `Metadata | None` | An enumeration. | `None` | | `name` | `str | None` | The name of a synthetic dataset. | `None` | | `description` | `str | None` | The description of a synthetic dataset. | `None` | | `generation_status` | `ProgressStatus` | An enumeration. | *required* | | `generation_time` | `datetime | None` | The UTC date and time when the generation has finished. | `None` | | `usage` | `SyntheticDatasetUsage | None` | An enumeration. | `None` |

#### SyntheticDatasetUsage

Usage statistics of a synthetic dataset.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `total_datapoints` | `int | None` | The number of datapoints in the synthetic dataset. Deprecated: This field is no longer valid and will always return -1. It will be removed in a future version. | `None` | | `total_compute_time` | `int | None` | The total compute time in seconds used for generating this synthetic dataset. This is the sum of the compute time of all trained tasks. | `None` | | `total_credits` | `float | None` | The amount of credits consumed for generating the synthetic dataset. | `None` | | `total_virtual_cpu_time` | `float | None` | The total virtual CPU time in seconds used for training this generator. This is the sum of the elapsed time multiplied by number of allocated virtual CPUs across all training tasks. | `None` | | `total_virtual_gpu_time` | `float | None` | The total virtual GPU time in seconds used for training this generator. This is the sum of the elapsed time multiplied by number of allocated virtual GPUs across all training tasks. | `None` | | `no_of_likes` | `int | None` | Number of likes of this synthetic dataset. | `None` | | `no_of_downloads` | `int | None` | Number of downloads of this synthetic dataset. | `None` |

#### SyntheticProbeConfig

The configuration for probing for new synthetic samples.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `generator_id` | `str | None` | The unique identifier of a generator. | `None` | | `random_state` | `int | None` | Seed for the random number generators. If None, the random number generator is initialized randomly, yielding different results for every run. Setting it to a specific integer ensures reproducible results across runs. Useful when consistent results are desired, e.g. for testing or debugging. | `None` | | `tables` | `list[SyntheticTableConfig] | None` | An enumeration. | `None` |

#### SyntheticTable

A synthetic table that will be generated.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `id` | `str | None` | The unique identifier of a synthetic table. | `None` | | `name` | `str` | The name of a source table. It must be unique within a generator. | *required* | | `configuration` | `SyntheticTableConfiguration | None` | An enumeration. | `None` | | `tabular_model_metrics` | `ModelMetrics | None` | An enumeration. | `None` | | `language_model_metrics` | `ModelMetrics | None` | An enumeration. | `None` | | `foreign_keys` | `list[SourceForeignKey] | None` | The foreign keys of a table. | `None` | | `total_rows` | `int | None` | The total number of rows for that table in the generated synthetic dataset. | `None` | | `total_datapoints` | `int | None` | Deprecated: This field is no longer valid and will always return -1. It will be removed in a future version. | `None` | | `source_table_total_rows` | `int | None` | The total number of rows in the source table while fetching data for training. | `None` |

#### SyntheticTableConfig

The configuration for a synthetic table when creating a new synthetic dataset.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `name` | `str` | The name of a synthetic table. This matches the name of a corresponding SourceTable. | *required* | | `configuration` | `SyntheticTableConfiguration | None` | An enumeration. | `None` |

#### SyntheticTableConfiguration

The sample configuration for a synthetic table

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `sample_size` | `int | None` | Number of generated samples. Only applicable for subject tables. If neither size nor seed is provided, then the default behavior for Synthetic Datasets is to generate the same size of samples as the original, and the default behavior for Synthetic Probes is to generate one subject only. | `None` | | `sample_seed_connector_id` | `str | None` | The connector id of the seed data for conditional generation. Only applicable for subject tables. | `None` | | `sample_seed_dict` | `str | None` | The base64-encoded string derived from a json line file containing the specified sample seed data. This allows conditional live probing via non-python clients. Only applicable for subject tables. | `None` | | `sample_seed_data` | `str | None` | The base64-encoded string derived from a Parquet file containing the specified sample seed data. This allows conditional generation as well as live probing via python clients. Only applicable for subject tables. | `None` | | `sampling_temperature` | `float | None` | temperature for sampling | `1.0` | | `sampling_top_p` | `float | None` | topP for sampling | `1.0` | | `rebalancing` | `RebalancingConfig | None` | An enumeration. | `None` | | `imputation` | `ImputationConfig | None` | An enumeration. | `None` | | `fairness` | `FairnessConfig | None` | An enumeration. | `None` | | `enable_data_report` | `bool | None` | If false, then the Data report is not generated. If enableDataReport is set to false on generator, then enableDataReport is automatically set to false. | `True` |

#### TaskType

The type of the task.

#### TransferOwnershipConfig

The configuration for transferring ownership of a resource to an account.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `account_id` | `str | None` | The unique identifier of an account (either a user or an organization). | `None` |

#### User

The public attributes of a user of the service.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `id` | `str | None` | The unique identifier of a user. | `None` | | `name` | `str | None` | The name of a user. Contains only alphanumeric characters, hyphens, and underscores. Must start or end with alphanumeric. It must be globally case-insensitive unique considering organizations and users. | `None` | | `first_name` | `str | None` | First name of a user | `None` | | `last_name` | `str | None` | Last name of a user | `None` | | `avatar` | `str | None` | The URL of the user's avatar | `None` | | `organizations` | `list[OrganizationListItem] | None` | The organizations the user belongs to | `None` |

#### UserListItem

Essential information about a user for public listings.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `id` | `str | None` | The unique identifier of a user. | `None` | | `name` | `str | None` | The name of a user. Contains only alphanumeric characters, hyphens, and underscores. Must start or end with alphanumeric. It must be globally case-insensitive unique considering organizations and users. | `None` | | `first_name` | `str | None` | First name of a user | `None` | | `last_name` | `str | None` | Last name of a user | `None` | | `avatar` | `str | None` | The URL of the user's avatar | `None` |

#### UserSettingsAssistantUpdateConfig

Configuration for updating a user's assistant-related settings.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `about_user_message` | `str | None` | The instruction what the Assistant should know about the user to provide better response | `None` | | `about_model_message` | `str | None` | The instruction how the Assistant should respond | `None` |

#### UserSettingsProfileUpdateConfig

Configuration for updating a user's profile settings.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `name` | `str | None` | The name of a user. Contains only alphanumeric characters, hyphens, and underscores. Must start or end with alphanumeric. It must be globally case-insensitive unique considering organizations and users. | `None` | | `first_name` | `str | None` | First name of a user | `None` | | `last_name` | `str | None` | Last name of a user | `None` | | `avatar` | `str | None` | The base64-encoded image of the user's avatar | `None` |

#### UserSettingsUpdateConfig

The configuration for updating user settings.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `profile` | `UserSettingsProfileUpdateConfig | None` | An enumeration. | `None` | | `assistant` | `UserSettingsAssistantUpdateConfig | None` | An enumeration. | `None` |

#### UserUsage

Usage statistics and limits for the current user.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `credits` | `Credits | None` | The credit balance and limit for the current time period | `None` | | `parallel_training_jobs` | `ParallelTrainingJobs | None` | The number of currently running training jobs and the limit | `None` | | `parallel_generation_jobs` | `ParallelGenerationJobs | None` | The number of currently running generation jobs and the limit | `None` |

#### Visibility

Indicates the visibility of the resource.

- `PUBLIC` - Everyone can access the resource.
- `UNLISTED`- Anyone with the direct link can access the resource. No public listings.
- `PRIVATE` - Accessible only by the owner. For organizations, all members can access.

#### \_SyntheticDataConfigValidation

Validation logic for SyntheticDatasetConfig and SyntheticProbeConfig against Generator

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `synthetic_config` | `SyntheticDatasetConfig | SyntheticProbeConfig` | An enumeration. | *required* | | `generator` | `Generator` | An enumeration. | *required* |

#### \_SyntheticTableConfigValidation

Validation logic for SyntheticTableConfig against SourceTable

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `synthetic_table` | `SyntheticTableConfig` | An enumeration. | *required* | | `source_table` | `SourceTable` | An enumeration. | *required* |
