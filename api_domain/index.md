# Schema References for `mostlyai.sdk.domain`

This module is auto-generated to represent `pydantic`-based classes of the defined schema in the [Public API](https://github.com/mostly-ai/mostly-openapi/blob/main/public-api.yaml).

### mostlyai.sdk.domain

#### AboutService

General information about the service.

Parameters:

| Name        | Type   | Description | Default                                        |
| ----------- | ------ | ----------- | ---------------------------------------------- |
| `version`   | \`str  | None\`      | The version number of the service.             |
| `assistant` | \`bool | None\`      | A flag indicating if the assistant is enabled. |

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

| Name             | Type    | Description | Default                                                                                       |
| ---------------- | ------- | ----------- | --------------------------------------------------------------------------------------------- |
| `overall`        | \`float | None\`      | Overall accuracy of synthetic data, averaged across univariate, bivariate, and coherence.     |
| `univariate`     | \`float | None\`      | Average accuracy of discretized univariate distributions.                                     |
| `bivariate`      | \`float | None\`      | Average accuracy of discretized bivariate distributions.                                      |
| `trivariate`     | \`float | None\`      | Average accuracy of discretized trivariate distributions.                                     |
| `coherence`      | \`float | None\`      | Average accuracy of discretized coherence distributions. Only applicable for sequential data. |
| `overall_max`    | \`float | None\`      | Expected overall accuracy of a same-sized holdout. Serves as a reference for overall.         |
| `univariate_max` | \`float | None\`      | Expected univariate accuracy of a same-sized holdout. Serves as a reference for univariate.   |
| `bivariate_max`  | \`float | None\`      | Expected bivariate accuracy of a same-sized holdout. Serves as a reference for bivariate.     |
| `trivariate_max` | \`float | None\`      | Expected trivariate accuracy of a same-sized holdout. Serves as a reference for trivariate.   |
| `coherence_max`  | \`float | None\`      | Expected coherence accuracy of a same-sized holdout. Serves as a reference for coherence.     |

#### Artifact

A shareable artifact generated from an assistant conversation.

Parameters:

| Name            | Type              | Description                                                             | Default    |
| --------------- | ----------------- | ----------------------------------------------------------------------- | ---------- |
| `id`            | `str`             | The unique identifier of an artifact.                                   | *required* |
| `name`          | `str`             | The name/title of an artifact.                                          | *required* |
| `file_name`     | `str`             | The filename of an artifact.                                            | *required* |
| `description`   | `str`             | The description/prompt of an artifact that explains how to recreate it. | *required* |
| `type`          | \`ArtifactType    | None\`                                                                  |            |
| `preview`       | \`ArtifactPreview | None\`                                                                  |            |
| `download_url`  | `str`             | URL to download the artifact.                                           | *required* |
| `shareable_url` | `str`             | Public URL where the artifact can be viewed and shared.                 | *required* |
| `usage`         | \`ArtifactUsage   | None\`                                                                  |            |
| `metadata`      | \`Metadata        | None\`                                                                  |            |

##### update

```python
update(name=None, description=None)
```

Update the artifact.

Parameters:

| Name          | Type  | Description | Default                          |
| ------------- | ----- | ----------- | -------------------------------- |
| `name`        | \`str | None\`      | The name of the artifact.        |
| `description` | \`str | None\`      | The description of the artifact. |

#### ArtifactConfig

Configuration for creating an artifact from an assistant message.

Parameters:

| Name                   | Type  | Description                                   | Default    |
| ---------------------- | ----- | --------------------------------------------- | ---------- |
| `assistant_message_id` | `str` | The unique identifier of a assistant message. | *required* |

#### ArtifactListItem

Essential artifact details for listings.

Parameters:

| Name            | Type              | Description                                                             | Default    |
| --------------- | ----------------- | ----------------------------------------------------------------------- | ---------- |
| `id`            | `str`             | The unique identifier of an artifact.                                   | *required* |
| `name`          | `str`             | The name/title of an artifact.                                          | *required* |
| `file_name`     | `str`             | The filename of an artifact.                                            | *required* |
| `description`   | `str`             | The description/prompt of an artifact that explains how to recreate it. | *required* |
| `type`          | \`ArtifactType    | None\`                                                                  |            |
| `preview`       | \`ArtifactPreview | None\`                                                                  |            |
| `download_url`  | `str`             | URL to download the artifact.                                           | *required* |
| `shareable_url` | `str`             | Public URL where the artifact can be viewed and shared.                 | *required* |
| `usage`         | \`ArtifactUsage   | None\`                                                                  |            |
| `metadata`      | \`Metadata        | None\`                                                                  |            |

#### ArtifactPreview

Preview information for an artifact.

Parameters:

| Name    | Type  | Description | Default                                  |
| ------- | ----- | ----------- | ---------------------------------------- |
| `image` | \`str | None\`      | URL to a preview image for the artifact. |

#### ArtifactType

The type of artifact content.

#### ArtifactUsage

Usage statistics of an artifact.

Parameters:

| Name            | Type  | Description | Default                                                 |
| --------------- | ----- | ----------- | ------------------------------------------------------- |
| `no_of_threads` | \`int | None\`      | Number of assistant threads started from this artifact. |

#### BillingCycle

The billing cycle of the subscription.

Parameters:

| Name         | Type            | Description | Default                                               |
| ------------ | --------------- | ----------- | ----------------------------------------------------- |
| `start_date` | \`AwareDatetime | None\`      | The UTC date and time when the billing cycle started. |
| `end_date`   | \`AwareDatetime | None\`      | The UTC date and time when the billing cycle ended.   |

#### BillingInfo

Billing information for an account.

Parameters:

| Name                  | Type            | Description                                                            | Default                                                       |
| --------------------- | --------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------- |
| `customer_portal_uri` | `AnyUrl`        | The URL of the customer portal for managing billing and subscriptions. | *required*                                                    |
| `billing_cycle`       | `BillingCycle`  |                                                                        | *required*                                                    |
| `cancels_at`          | \`AwareDatetime | None\`                                                                 | The UTC date and time when the subscription will be canceled. |
| `current_plan`        | `Plan`          |                                                                        | *required*                                                    |

#### BillingInterval

The billing cycle for a plan.

#### Compute

A compute resource for executing tasks.

Parameters:

| Name          | Type                    | Description | Default                                                              |
| ------------- | ----------------------- | ----------- | -------------------------------------------------------------------- |
| `id`          | \`str                   | None\`      | The unique identifier of a compute resource. Not applicable for SDK. |
| `name`        | \`str                   | None\`      | The name of a compute resource.                                      |
| `type`        | \`Literal['KUBERNETES'] | None\`      | The type of compute.                                                 |
| `config`      | \`dict[str, Any]        | None\`      |                                                                      |
| `secrets`     | \`dict[str, Any]        | None\`      |                                                                      |
| `resources`   | \`ComputeResources      | None\`      |                                                                      |
| `order_index` | \`int                   | None\`      | The index for determining the sort order when listing computes       |

#### ComputeConfig

The configuration for creating a new compute resource.

Parameters:

| Name          | Type                    | Description | Default                                                        |
| ------------- | ----------------------- | ----------- | -------------------------------------------------------------- |
| `name`        | \`str                   | None\`      | The name of a compute resource.                                |
| `type`        | \`Literal['KUBERNETES'] | None\`      | The type of compute.                                           |
| `resources`   | \`ComputeResources      | None\`      |                                                                |
| `config`      | \`dict[str, Any]        | None\`      |                                                                |
| `secrets`     | \`dict[str, Any]        | None\`      |                                                                |
| `order_index` | \`int                   | None\`      | The index for determining the sort order when listing computes |

#### ComputeListItem

Essential compute details for listings.

Parameters:

| Name        | Type                    | Description | Default                                                              |
| ----------- | ----------------------- | ----------- | -------------------------------------------------------------------- |
| `id`        | \`str                   | None\`      | The unique identifier of a compute resource. Not applicable for SDK. |
| `type`      | \`Literal['KUBERNETES'] | None\`      | The type of compute.                                                 |
| `name`      | \`str                   | None\`      | The name of a compute resource.                                      |
| `resources` | \`ComputeResources      | None\`      |                                                                      |

#### ComputeResources

A set of available hardware resources for a compute resource.

Parameters:

| Name         | Type    | Description | Default                                    |
| ------------ | ------- | ----------- | ------------------------------------------ |
| `cpus`       | \`int   | None\`      | The number of CPU cores                    |
| `memory`     | \`float | None\`      | The amount of memory in GB                 |
| `gpus`       | \`int   | None\`      | The number of GPUs                         |
| `gpu_memory` | \`float | None\`      | The amount of GPU memory in GB             |
| `storage`    | \`float | None\`      | Ephemeral storage in GiB (Kubernetes only) |

#### Connector

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

| Name       | Type  | Description                                                   | Default    |
| ---------- | ----- | ------------------------------------------------------------- | ---------- |
| `location` | `str` | The target location within the connector to delete data from. | *required* |

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

##### query

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

##### read_data

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

##### schema

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

| Name              | Type                  | Description | Default                                          |
| ----------------- | --------------------- | ----------- | ------------------------------------------------ |
| `name`            | \`str                 | None\`      | The name of the connector.                       |
| `description`     | \`str                 | None\`      | The description of the connector.                |
| `access_type`     | \`ConnectorAccessType | None\`      | The access type of the connector.                |
| `config`          | \`dict[str, Any]      | None\`      | Connector configuration.                         |
| `secrets`         | \`dict[str, str]      | None\`      | Secret values for the connector.                 |
| `ssl`             | \`dict[str, str]      | None\`      | SSL configuration for the connector.             |
| `test_connection` | \`bool                | None\`      | If true, validates the connection before saving. |

##### write_data

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

  - type: REDSHIFT
    config:
      host: string
      port: integer, default: 5439
      username: string
      database: string
    secrets:
      password: string

  - type: SQLITE
    config:
      database: string
  ```

Parameters:

| Name          | Type                  | Description | Default                         |
| ------------- | --------------------- | ----------- | ------------------------------- |
| `name`        | \`str                 | None\`      | The name of a connector.        |
| `description` | \`str                 | None\`      | The description of a connector. |
| `type`        | `ConnectorType`       |             | *required*                      |
| `access_type` | \`ConnectorAccessType | None\`      |                                 |
| `config`      | \`dict[str, Any]      | None\`      |                                 |
| `secrets`     | \`dict[str, str]      | None\`      |                                 |
| `ssl`         | \`dict[str, str]      | None\`      |                                 |

#### ConnectorDeleteDataConfig

Configuration for deleting data from a connector.

Parameters:

| Name       | Type  | Description                                                                                                 | Default    |
| ---------- | ----- | ----------------------------------------------------------------------------------------------------------- | ---------- |
| `location` | `str` | Specifies the target within the connector to delete. The format of this parameter varies by connector type. | *required* |

#### ConnectorListItem

Essential connector details for listings.

Parameters:

| Name          | Type                  | Description                           | Default                         |
| ------------- | --------------------- | ------------------------------------- | ------------------------------- |
| `id`          | `str`                 | The unique identifier of a connector. | *required*                      |
| `name`        | \`str                 | None\`                                | The name of a connector.        |
| `description` | \`str                 | None\`                                | The description of a connector. |
| `type`        | `ConnectorType`       |                                       | *required*                      |
| `access_type` | \`ConnectorAccessType | None\`                                |                                 |
| `metadata`    | \`Metadata            | None\`                                |                                 |
| `usage`       | \`ConnectorUsage      | None\`                                |                                 |

#### ConnectorQueryConfig

Attributes: sql (str): SQL read-only (e.g. SELECT) statement to execute.

Parameters:

| Name  | Type  | Description                                       | Default    |
| ----- | ----- | ------------------------------------------------- | ---------- |
| `sql` | `str` | SQL read-only (e.g. SELECT) statement to execute. | *required* |

#### ConnectorReadDataConfig

Configuration for reading data from a connector.

Parameters:

| Name       | Type   | Description | Default                                                                                                                           |
| ---------- | ------ | ----------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `location` | \`str  | None\`      | Specifies the target within the connector from which to retrieve the data. The format of this parameter varies by connector type. |
| `limit`    | \`int  | None\`      | The maximum number of rows to return. Return all if not specified.                                                                |
| `shuffle`  | \`bool | None\`      | Whether to shuffle the results.                                                                                                   |

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
- `REDSHIFT`: Amazon Redshift cloud data warehouse
- `SQLITE`: SQLite database
- `AZURE_STORAGE`: Azure Blob Storage
- `GOOGLE_CLOUD_STORAGE`: Google Cloud Storage
- `S3_STORAGE`: Amazon S3 Storage
- `FILE_UPLOAD`: File upload

#### ConnectorUsage

Usage statistics of a connector.

Parameters:

| Name               | Type  | Description | Default                                           |
| ------------------ | ----- | ----------- | ------------------------------------------------- |
| `no_of_generators` | \`int | None\`      | Number of generators using this connector.        |
| `no_of_likes`      | \`int | None\`      | Number of likes of this connector.                |
| `no_of_threads`    | \`int | None\`      | Number of assistant threads using this connector. |

#### ConnectorWriteDataConfig

Configuration for writing data to a connector.

Parameters:

| Name        | Type       | Description                                                                                                                  | Default                                                                                                                                                                                              |
| ----------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `file`      | `bytes`    | Binary Parquet file containing the data to write.                                                                            | *required*                                                                                                                                                                                           |
| `location`  | `str`      | Specifies the target within the connector to which to write the data. The format of this parameter varies by connector type. | *required*                                                                                                                                                                                           |
| `if_exists` | \`IfExists | None\`                                                                                                                       | The behavior if the target location already exists. APPEND: Append the data to the existing target. REPLACE: Replace the existing target with the new data. FAIL: Fail if the target already exists. |

#### Constraint

A data constraint to apply.

Parameters:

| Name     | Type             | Description                            | Default    |
| -------- | ---------------- | -------------------------------------- | ---------- |
| `id`     | `str`            | The unique identifier of a constraint. | *required* |
| `type`   | `ConstraintType` |                                        | *required* |
| `config` | `dict[str, Any]` |                                        | *required* |

#### ConstraintConfig

A constraint to apply during data generation.

Parameters:

| Name     | Type             | Description | Default    |
| -------- | ---------------- | ----------- | ---------- |
| `type`   | `ConstraintType` |             | *required* |
| `config` | `dict[str, Any]` |             | *required* |

#### ConstraintType

The type of constraint. If type is 'Inequality', then 'table_name', 'lowColumn' and 'highColumn' are required. If type is 'FixedCombinations', then 'table_name' and 'columns' are required.

#### CreditType

Determines which type of credits to include in the usage report.

- `FREE`: Filter usage reports for free credits (natural month cycle).
- `PAID`: Filter usage reports for paid credits (billing cycle).

#### CurrentUser

Information on the current user.

Parameters:

| Name                   | Type                         | Description | Default                                                                                                                                                                                                    |
| ---------------------- | ---------------------------- | ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `id`                   | \`str                        | None\`      | The unique identifier of a user.                                                                                                                                                                           |
| `name`                 | \`str                        | None\`      | The name of a user. Contains only alphanumeric characters, hyphens, and underscores. Must start or end with alphanumeric. It must be globally case-insensitive unique considering organizations and users. |
| `first_name`           | \`str                        | None\`      | First name of a user                                                                                                                                                                                       |
| `last_name`            | \`str                        | None\`      | Last name of a user                                                                                                                                                                                        |
| `email`                | \`str                        | None\`      | The email of a user                                                                                                                                                                                        |
| `avatar`               | \`str                        | None\`      | The URL of the user's avatar                                                                                                                                                                               |
| `settings`             | \`dict[str, Any]             | None\`      |                                                                                                                                                                                                            |
| `usage`                | \`UserUsage                  | None\`      |                                                                                                                                                                                                            |
| `unread_notifications` | \`int                        | None\`      | Number of unread notifications for the user                                                                                                                                                                |
| `plan`                 | \`UserPlan                   | None\`      |                                                                                                                                                                                                            |
| `organizations`        | \`list[OrganizationListItem] | None\`      | The organizations the user belongs to                                                                                                                                                                      |
| `secrets`              | \`list[UserSecretKey]        | None\`      | The list of secret key names for this user (secret values are never exposed via API)                                                                                                                       |
| `created_at`           | \`AwareDatetime              | None\`      | The UTC date and time when the user has been created.                                                                                                                                                      |
| `last_activity_at`     | \`AwareDatetime              | None\`      | The UTC date and time of users last activity                                                                                                                                                               |

#### Dataset

A dataset to be consumed via the assistant.

Parameters:

| Name          | Type                     | Description                         | Default                                          |
| ------------- | ------------------------ | ----------------------------------- | ------------------------------------------------ |
| `id`          | `str`                    | The unique identifier of a dataset. | *required*                                       |
| `name`        | `str`                    | The name of a dataset.              | *required*                                       |
| `description` | \`str                    | None\`                              | The description of / instructions for a dataset. |
| `connectors`  | \`list[DatasetConnector] | None\`                              |                                                  |
| `files`       | \`list[str]              | None\`                              |                                                  |
| `usage`       | \`DatasetUsage           | None\`                              |                                                  |
| `metadata`    | \`Metadata               | None\`                              |                                                  |

##### delete

```python
delete()
```

Delete the dataset.

##### delete_file

```python
delete_file(file_path)
```

Delete the dataset file.

##### download_file

```python
download_file(dataset_file_path, output_file_path=None)
```

Download the dataset file.

Parameters:

| Name        | Type  | Description | Default |
| ----------- | ----- | ----------- | ------- |
| `file_path` | \`str | Path        | None\`  |

Returns:

| Name   | Type   | Description                 |
| ------ | ------ | --------------------------- |
| `Path` | `Path` | The path to the saved file. |

##### update

```python
update(name=None, description=None, connectors=None)
```

Update a dataset with specific parameters.

Parameters:

| Name          | Type                     | Description | Default                           |
| ------------- | ------------------------ | ----------- | --------------------------------- |
| `name`        | \`str                    | None\`      | The name of the connector.        |
| `description` | \`str                    | None\`      | The description of the connector. |
| `connectors`  | \`list[DatasetConnector] | None\`      | The connectors of the dataset.    |

##### upload_file

```python
upload_file(file_path)
```

Upload the dataset file.

#### DatasetConfig

The configuration for creating a dataset.

Parameters:

| Name          | Type                           | Description | Default                                          |
| ------------- | ------------------------------ | ----------- | ------------------------------------------------ |
| `name`        | \`str                          | None\`      | The name of a dataset.                           |
| `description` | \`str                          | None\`      | The description of / instructions for a dataset. |
| `connectors`  | \`list[DatasetConnectorConfig] | None\`      |                                                  |

#### DatasetConnector

Configuration for a dataset connector.

Parameters:

| Name          | Type                  | Description | Default                               |
| ------------- | --------------------- | ----------- | ------------------------------------- |
| `id`          | \`str                 | None\`      | The unique identifier of a connector. |
| `name`        | \`str                 | None\`      | The name of a connector.              |
| `description` | \`str                 | None\`      | The description of a connector.       |
| `type`        | \`ConnectorType       | None\`      |                                       |
| `access_type` | \`ConnectorAccessType | None\`      |                                       |
| `metadata`    | \`Metadata            | None\`      |                                       |
| `usage`       | \`ConnectorUsage      | None\`      |                                       |
| `locations`   | \`list[str]           | None\`      |                                       |

#### DatasetConnectorConfig

Configuration for a dataset connector.

Parameters:

| Name        | Type        | Description                           | Default    |
| ----------- | ----------- | ------------------------------------- | ---------- |
| `id`        | `str`       | The unique identifier of a connector. | *required* |
| `locations` | \`list[str] | None\`                                |            |

#### DatasetConnectorLocationsConfig

Configuration for adding connector to dataset.

Parameters:

| Name        | Type        | Description | Default |
| ----------- | ----------- | ----------- | ------- |
| `locations` | \`list[str] | None\`      |         |

#### DatasetListItem

Essential dataset details for listings.

Parameters:

| Name          | Type           | Description                         | Default                                          |
| ------------- | -------------- | ----------------------------------- | ------------------------------------------------ |
| `id`          | `str`          | The unique identifier of a dataset. | *required*                                       |
| `name`        | `str`          | The name of a dataset.              | *required*                                       |
| `description` | \`str          | None\`                              | The description of / instructions for a dataset. |
| `usage`       | \`DatasetUsage | None\`                              |                                                  |
| `metadata`    | \`Metadata     | None\`                              |                                                  |

#### DatasetUsage

Usage statistics of a dataset.

Parameters:

| Name            | Type  | Description | Default                                         |
| --------------- | ----- | ----------- | ----------------------------------------------- |
| `no_of_likes`   | \`int | None\`      | Number of likes of this dataset.                |
| `no_of_threads` | \`int | None\`      | Number of assistant threads using this dataset. |

#### DifferentialPrivacyConfig

The optional differential privacy configuration for training the model. If not provided, then no differential privacy will be applied.

Parameters:

| Name                       | Type    | Description | Default                                                                                                                                                                                                                                                                                                                                                                                                       |
| -------------------------- | ------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `max_epsilon`              | \`float | None\`      | Specifies the maximum allowable epsilon value. If the training process exceeds this threshold, it will be terminated early. Only model checkpoints with epsilon values below this limit will be retained. If not provided, the training will proceed without early termination based on epsilon constraints.                                                                                                  |
| `delta`                    | \`float | None\`      | The delta value for differential privacy. It is the probability of the privacy guarantee not holding. The smaller the delta, the more confident you can be that the privacy guarantee holds. This delta will be equally distributed between the analysis and the training phase.                                                                                                                              |
| `noise_multiplier`         | \`float | None\`      | Determines how much noise while training the model with differential privacy. This is the ratio of the standard deviation of the Gaussian noise to the L2-sensitivity of the function to which the noise is added.                                                                                                                                                                                            |
| `max_grad_norm`            | \`float | None\`      | Determines the maximum impact of a single sample on updating the model weights during training with differential privacy. This is the maximum norm of the per-sample gradients.                                                                                                                                                                                                                               |
| `value_protection_epsilon` | \`float | None\`      | The DP epsilon of the privacy budget for determining the value ranges, which are gathered prior to the model training during the analysis step. Only applicable if value protection is True. Privacy budget will be equally distributed between the columns. For categorical we calculate noisy histograms and use a noisy threshold. For numeric and datetime we calculate bounds based on noisy histograms. |

#### Distances

Metrics regarding the nearest neighbor distances between training, holdout, and synthetic samples in an numerically encoded space. Useful for assessing the novelty / privacy of synthetic data.

The provided data is first down-sampled, so that the number of samples match across all datasets. Note, that for an optimal sensitivity of this privacy assessment it is recommended to use a 50/50 split between training and holdout data, and then generate synthetic data of the same size.

The numerical encodings of these samples are then computed, and the nearest neighbor distances are calculated for each synthetic sample to the training and holdout samples. Based on these nearest neighbor distances the following metrics are calculated:

- Identical Match Share (IMS): The share of synthetic samples that are identical to a training or holdout sample.
- Distance to Closest Record (DCR): The average distance of synthetic to training or holdout samples.
- Nearest Neighbor Distance Ratio (NNDR): The 10-th smallest ratio of the distance to nearest and second nearest neighbor.

For privacy-safe synthetic data we expect to see about as many identical matches, and about the same distances for synthetic samples to training, as we see for synthetic samples to holdout.

Parameters:

| Name            | Type    | Description | Default                                                                                                                                     |
| --------------- | ------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `ims_training`  | \`float | None\`      | Share of synthetic samples that are identical to a training sample.                                                                         |
| `ims_holdout`   | \`float | None\`      | Share of synthetic samples that are identical to a holdout sample. Serves as a reference for ims_training.                                  |
| `ims_trn_hol`   | \`float | None\`      | Share of training samples that are identical to a holdout sample. Serves as a reference for ims_training.                                   |
| `dcr_training`  | \`float | None\`      | Average nearest-neighbor distance between synthetic and training samples.                                                                   |
| `dcr_holdout`   | \`float | None\`      | Average nearest-neighbor distance between synthetic and holdout samples. Serves as a reference for dcr_training.                            |
| `dcr_trn_hol`   | \`float | None\`      | Average nearest-neighbor distance between training and holdout samples. Serves as a reference for dcr_training.                             |
| `dcr_share`     | \`float | None\`      | Share of synthetic samples that are closer to a training sample than to a holdout sample. This should not be significantly larger than 50%. |
| `nndr_training` | \`float | None\`      | 10th smallest nearest-neighbor distance ratio between synthetic and training samples.                                                       |
| `nndr_holdout`  | \`float | None\`      | 10th smallest nearest-neighbor distance ratio between synthetic and holdout samples.                                                        |
| `nndr_trn_hol`  | \`float | None\`      | 10th smallest nearest-neighbor distance ratio between training and holdout samples.                                                         |

#### ErrorEvent

An error event containing an error message

Parameters:

| Name    | Type               | Description | Default |
| ------- | ------------------ | ----------- | ------- |
| `event` | \`Literal['error'] | None\`      |         |
| `data`  | \`ErrorMessage     | None\`      |         |

#### ErrorMessage

An error message

Parameters:

| Name      | Type  | Description | Default           |
| --------- | ----- | ----------- | ----------------- |
| `message` | \`str | None\`      | The error message |

#### FairnessConfig

Configure a fairness objective for the table. Only applicable for a subject table. The generated synthetic data will maintain robust statistical parity between the target column and the specified sensitive columns. All these columns must be categorical.

Parameters:

| Name                | Type        | Description                         | Default    |
| ------------------- | ----------- | ----------------------------------- | ---------- |
| `target_column`     | `str`       | The name of the target column.      | *required* |
| `sensitive_columns` | `list[str]` | The names of the sensitive columns. | *required* |

#### FilterByUser

Determines whether to filter usage reports for all users or only the current user.

- `ALL`: Filter usage reports for all users. Only accessible for SuperAdmins.
- `ME`: Filter usage reports for the current user.

#### Free

Usage statistics of free credits.

Parameters:

| Name      | Type               | Description | Default |
| --------- | ------------------ | ----------- | ------- |
| `daily`   | \`UsageCreditStats | None\`      |         |
| `monthly` | \`UsageCreditStats | None\`      |         |

#### Generator

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

| Name        | Type  | Description | Default |
| ----------- | ----- | ----------- | ------- |
| `file_path` | \`str | Path        | None\`  |

Returns:

| Name   | Type   | Description                 |
| ------ | ------ | --------------------------- |
| `Path` | `Path` | The path to the saved file. |

###### progress

```python
progress()
```

Retrieve job progress of training.

Returns:

| Name          | Type          | Description                               |
| ------------- | ------------- | ----------------------------------------- |
| `JobProgress` | `JobProgress` | The job progress of the training process. |

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

| Name           | Type    | Description                                                             | Default |
| -------------- | ------- | ----------------------------------------------------------------------- | ------- |
| `progress_bar` | `bool`  | If true, displays the progress bar. Default is True.                    | `True`  |
| `interval`     | `float` | The interval in seconds to poll the job progress. Default is 2 seconds. | `2`     |

##### clone

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

##### config

```python
config()
```

Retrieve writable generator properties.

Returns:

| Name              | Type              | Description                                         |
| ----------------- | ----------------- | --------------------------------------------------- |
| `GeneratorConfig` | `GeneratorConfig` | The generator properties as a configuration object. |

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

| Name        | Type  | Description | Default |
| ----------- | ----- | ----------- | ------- |
| `file_path` | \`str | Path        | None\`  |

Returns:

| Name   | Type   | Description                 |
| ------ | ------ | --------------------------- |
| `Path` | `Path` | The path to the saved file. |

##### reports

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

##### update

```python
update(name=None, description=None)
```

Update a generator with specific parameters.

Parameters:

| Name          | Type  | Description | Default                           |
| ------------- | ----- | ----------- | --------------------------------- |
| `name`        | \`str | None\`      | The name of the generator.        |
| `description` | \`str | None\`      | The description of the generator. |

#### GeneratorCloneTrainingStatus

The training status of the new generator. The available options are:

- `NEW`: The new generator will re-use existing data and model configurations.
- `CONTINUE`: The new generator will re-use existing data and model configurations, as well as model weights.

#### GeneratorConfig

The configuration for creating a new generator.

Parameters:

| Name           | Type                      | Description | Default                                                                                                                                                                                                                                                                                              |
| -------------- | ------------------------- | ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `name`         | \`str                     | None\`      | The name of a generator.                                                                                                                                                                                                                                                                             |
| `description`  | \`str                     | None\`      | The description of a generator.                                                                                                                                                                                                                                                                      |
| `random_state` | \`int                     | None\`      | Seed for the random number generators. If None, the random number generator is initialized randomly, yielding different results for every run. Setting it to a specific integer ensures reproducible results across runs. Useful when consistent results are desired, e.g. for testing or debugging. |
| `tables`       | \`list[SourceTableConfig] | None\`      | The tables of a generator                                                                                                                                                                                                                                                                            |
| `constraints`  | \`list[ConstraintConfig]  | None\`      | The data constraints to apply.                                                                                                                                                                                                                                                                       |

##### \_track_column_usage

```python
_track_column_usage(
    table_name, col_name, constraint_idx, column_usage
)
```

track column usage and detect overlaps.

##### validate_constraints

```python
validate_constraints()
```

validate that constraints reference existing tables and columns.

#### GeneratorImportFromFileConfig

Configuration for importing a generator from a file.

Parameters:

| Name   | Type    | Description | Default    |
| ------ | ------- | ----------- | ---------- |
| `file` | `bytes` |             | *required* |

#### GeneratorListItem

Essential generator details for listings.

Parameters:

| Name              | Type             | Description                           | Default                                                                                                                  |
| ----------------- | ---------------- | ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `id`              | `str`            | The unique identifier of a generator. | *required*                                                                                                               |
| `name`            | `str`            | The name of a generator.              | *required*                                                                                                               |
| `description`     | \`str            | None\`                                | The description of a generator.                                                                                          |
| `training_status` | `ProgressStatus` |                                       | *required*                                                                                                               |
| `training_time`   | \`AwareDatetime  | None\`                                | The UTC date and time when the training has finished.                                                                    |
| `usage`           | \`GeneratorUsage | None\`                                |                                                                                                                          |
| `metadata`        | \`Metadata       | None\`                                |                                                                                                                          |
| `accuracy`        | \`float          | None\`                                | The overall accuracy of the trained generator. This is the average of the overall accuracy scores of all trained models. |

#### GeneratorUsage

Usage statistics of a generator.

Parameters:

| Name                       | Type    | Description | Default                                                                                                                                                                               |
| -------------------------- | ------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `total_datapoints`         | \`int   | None\`      | The total number of datapoints generated by this generator. Deprecated: This field is no longer valid and will always return -1. It will be removed in a future version.              |
| `total_compute_time`       | \`int   | None\`      | The total compute time in seconds used for training this generator. This is the sum of the elapsed compute time of all training tasks.                                                |
| `total_credits`            | \`float | None\`      | The amount of credits consumed for training the generator.                                                                                                                            |
| `total_virtual_cpu_time`   | \`float | None\`      | The total virtual CPU time in seconds used for training this generator. This is the sum of the elapsed time multiplied by number of allocated virtual CPUs across all training tasks. |
| `total_virtual_gpu_time`   | \`float | None\`      | The total virtual GPU time in seconds used for training this generator. This is the sum of the elapsed time multiplied by number of allocated virtual GPUs across all training tasks. |
| `no_of_synthetic_datasets` | \`int   | None\`      | Number of synthetic datasets generated by this generator.                                                                                                                             |
| `no_of_likes`              | \`int   | None\`      | Number of likes of this generator.                                                                                                                                                    |
| `no_of_threads`            | \`int   | None\`      | Number of assistant threads using this generator.                                                                                                                                     |

#### HeartbeatEvent

A heartbeat event to keep the connection alive

Parameters:

| Name    | Type                   | Description | Default |
| ------- | ---------------------- | ----------- | ------- |
| `event` | \`Literal['heartbeat'] | None\`      |         |

#### IfExists

The behavior if the target location already exists.

- `APPEND`: Append the data to the existing target.
- `REPLACE`: Replace the existing target with the new data.
- `FAIL`: Fail if the target already exists.

#### ImputationConfig

Configure imputation.

Parameters:

| Name      | Type        | Description                                                                                        | Default    |
| --------- | ----------- | -------------------------------------------------------------------------------------------------- | ---------- |
| `columns` | `list[str]` | The names of the columns to be imputed. Imputed columns will suppress the sampling of NULL values. | *required* |

#### Integration

An OAuth2 integration provider with connection status. If connected, includes integration details. If not connected, shows NOT_CONNECTED status.

Parameters:

| Name               | Type                     | Description                                    | Default                                                                |
| ------------------ | ------------------------ | ---------------------------------------------- | ---------------------------------------------------------------------- |
| `provider_id`      | `str`                    | The provider identifier                        | *required*                                                             |
| `provider_name`    | `str`                    | Display name of the provider                   | *required*                                                             |
| `icon`             | \`str                    | None\`                                         | svg image                                                              |
| `is_promoted`      | \`bool                   | None\`                                         | Whether this provider is promoted in the UI                            |
| `scope_ids`        | \`list[str]              | None\`                                         | List of IDs of the currently connected scopes (empty if not connected) |
| `available_scopes` | `list[IntegrationScope]` | List of all available scopes for this provider | *required*                                                             |
| `status`           | `IntegrationStatus`      |                                                | *required*                                                             |

#### IntegrationAuthorizationRequest

Request to generate an OAuth authorization URL

Parameters:

| Name        | Type        | Description                                    | Default    |
| ----------- | ----------- | ---------------------------------------------- | ---------- |
| `scope_ids` | `list[str]` | List of scope identifiers for this integration | *required* |

#### IntegrationProvidersConfig

Configuration for a single integration provider

Parameters:

| Name                | Type                        | Description                        | Default                                                                                 |
| ------------------- | --------------------------- | ---------------------------------- | --------------------------------------------------------------------------------------- |
| `id`                | `str`                       | Unique identifier for the provider | *required*                                                                              |
| `name`              | `str`                       | Display name of the provider       | *required*                                                                              |
| `description`       | \`str                       | None\`                             | Description of the provider                                                             |
| `icon`              | \`str                       | None\`                             | svg image                                                                               |
| `is_promoted`       | \`bool                      | None\`                             | Whether this provider is promoted in the UI                                             |
| `authorization_url` | `str`                       | OAuth authorization endpoint URL   | *required*                                                                              |
| `access_token_url`  | `str`                       | OAuth token endpoint URL           | *required*                                                                              |
| `scopes`            | \`list[ProviderScopeConfig] | None\`                             | Available OAuth scopes for this provider                                                |
| `additional_params` | \`dict[str, str]            | None\`                             | Additional parameters to include in OAuth requests                                      |
| `client_id`         | `str`                       | OAuth client ID                    | *required*                                                                              |
| `client_secret`     | `str`                       | OAuth client secret                | *required*                                                                              |
| `scope_delimiter`   | \`str                       | None\`                             | Delimiter used to join multiple scope values (e.g., space for GitHub, comma for others) |

#### IntegrationProvidersConfigList

Configuration to create or update multiple integration providers

Parameters:

| Name        | Type                               | Description | Default    |
| ----------- | ---------------------------------- | ----------- | ---------- |
| `providers` | `list[IntegrationProvidersConfig]` |             | *required* |

#### IntegrationScope

OAuth scope information for an integration

Parameters:

| Name          | Type  | Description                     | Default                               |
| ------------- | ----- | ------------------------------- | ------------------------------------- |
| `id`          | `str` | Unique identifier for the scope | *required*                            |
| `name`        | `str` | Display name of the scope       | *required*                            |
| `description` | \`str | None\`                          | Description of what this scope allows |
| `value`       | `str` | The actual OAuth scope value    | *required*                            |

#### IntegrationStatus

Status of an integration connection

#### JobProgress

The progress of a job.

Parameters:

| Name         | Type                 | Description | Default                                                                                            |
| ------------ | -------------------- | ----------- | -------------------------------------------------------------------------------------------------- |
| `id`         | \`str                | None\`      |                                                                                                    |
| `start_date` | \`AwareDatetime      | None\`      | The UTC date and time when the job has started. If the job has not started yet, then this is None. |
| `end_date`   | \`AwareDatetime      | None\`      | The UTC date and time when the job has ended. If the job is still, then this is None.              |
| `progress`   | \`ProgressValue      | None\`      |                                                                                                    |
| `status`     | \`ProgressStatus     | None\`      |                                                                                                    |
| `steps`      | \`list[ProgressStep] | None\`      |                                                                                                    |

#### MemberRole

The role of the user in the organization

- `VIEWER`: The user can view and use all resources of the organization
- `CONTRIBUTOR`: The user can create new resources for an organization, and becomes resource ADMIN
- `ADMIN`: The user can manage members and all resources of an organization

#### MessageEvent

A message event containing an assistant message delta

Parameters:

| Name    | Type                    | Description | Default |
| ------- | ----------------------- | ----------- | ------- |
| `event` | \`Literal['message']    | None\`      |         |
| `data`  | \`AssistantMessageDelta | None\`      |         |

#### MessageStreamEvent

Parameters:

| Name   | Type           | Description    | Default      |
| ------ | -------------- | -------------- | ------------ |
| `root` | \`MessageEvent | HeartbeatEvent | ErrorEvent\` |

#### Metadata

The metadata of a resource.

Parameters:

| Name                            | Type              | Description | Default                                                                                                                                                                                                                                   |
| ------------------------------- | ----------------- | ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `creator_id`                    | \`str             | None\`      | The unique identifier of a user.                                                                                                                                                                                                          |
| `creator_name`                  | \`str             | None\`      | The name of a user. Contains only alphanumeric characters, hyphens, and underscores. Must start or end with alphanumeric. It must be globally case-insensitive unique considering organizations and users.                                |
| `created_at`                    | \`AwareDatetime   | None\`      | The UTC date and time when the resource has been created.                                                                                                                                                                                 |
| `owner_id`                      | \`str             | None\`      | The unique identifier of an account (either a user or an organization).                                                                                                                                                                   |
| `owner_name`                    | \`str             | None\`      | The name of an account (either a user or an organization).                                                                                                                                                                                |
| `owner_type`                    | \`AccountType     | None\`      |                                                                                                                                                                                                                                           |
| `owner_image`                   | \`str             | None\`      | The URL of the account's image.                                                                                                                                                                                                           |
| `visibility`                    | \`Visibility      | None\`      |                                                                                                                                                                                                                                           |
| `current_user_permission_level` | \`PermissionLevel | None\`      |                                                                                                                                                                                                                                           |
| `current_user_like_status`      | \`bool            | None\`      | A boolean indicating whether the user has liked the entity or not                                                                                                                                                                         |
| `short_lived_file_token`        | \`str             | None\`      | An auto-generated short-lived file token (slft) for accessing resource artifacts. The token is always restricted to a single resource, only valid for 60 minutes, and only accepted by API endpoints that allow to download single files. |

#### ModelConfiguration

The training configuration for the model

Parameters:

| Name                               | Type                            | Description | Default                                                                                                                                                                                                                                                                                                                                                                  |
| ---------------------------------- | ------------------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `model`                            | \`str                           | None\`      | The model to be used for training.                                                                                                                                                                                                                                                                                                                                       |
| `max_sample_size`                  | \`int                           | None\`      | The maximum number of samples to consider for training. If not provided, then all available samples will be taken.                                                                                                                                                                                                                                                       |
| `batch_size`                       | \`int                           | None\`      | The physical batch size used for training the model. If not provided, batchSize will be chosen automatically.                                                                                                                                                                                                                                                            |
| `gradient_accumulation_steps`      | \`int                           | None\`      | Steps to accumulate gradients before optimizer update. If not provided, gradientAccumulationSteps will be chosen automatically.                                                                                                                                                                                                                                          |
| `max_training_time`                | \`float                         | None\`      | The maximum number of minutes to train the model.                                                                                                                                                                                                                                                                                                                        |
| `max_epochs`                       | \`float                         | None\`      | The maximum number of epochs to train the model.                                                                                                                                                                                                                                                                                                                         |
| `max_sequence_window`              | \`int                           | None\`      | The maximum sequence window to consider for training. Only applicable for TABULAR models.                                                                                                                                                                                                                                                                                |
| `enable_flexible_generation`       | \`bool                          | None\`      | If true, then the trained generator can be used for conditional simulation, rebalancing, imputation and fairness. If none of these will be needed, then one can gain extra accuracy by disabling this feature. This will then result in a fixed column order being fed into the training process, rather than a column order, that is randomly permuted for every batch. |
| `value_protection`                 | \`bool                          | None\`      | Defines if Rare Category, Extreme value, or Sequence length protection will be applied.                                                                                                                                                                                                                                                                                  |
| `rare_category_replacement_method` | \`RareCategoryReplacementMethod | None\`      | Specifies how rare categories will be sampled. Only applicable if value protection has been enabled. CONSTANT: Replace rare categories by a constant _RARE_ token. SAMPLE: Replace rare categories by a sample from non-rare categories.                                                                                                                                 |
| `differential_privacy`             | \`DifferentialPrivacyConfig     | None\`      |                                                                                                                                                                                                                                                                                                                                                                          |
| `compute`                          | \`str                           | None\`      | The unique identifier of a compute resource. Not applicable for SDK.                                                                                                                                                                                                                                                                                                     |
| `enable_model_report`              | \`bool                          | None\`      | If false, then the Model report is not generated.                                                                                                                                                                                                                                                                                                                        |

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

| Name         | Type         | Description | Default |
| ------------ | ------------ | ----------- | ------- |
| `accuracy`   | \`Accuracy   | None\`      |         |
| `distances`  | \`Distances  | None\`      |         |
| `similarity` | \`Similarity | None\`      |         |

#### ModelType

The type of model.

- `TABULAR`: A generative AI model tailored towards tabular data, trained from scratch.
- `LANGUAGE`: A generative AI model build upon a (pre-trained) language model.

#### Notification

A notification for a user.

Parameters:

| Name           | Type                 | Description                                                   | Default                       |
| -------------- | -------------------- | ------------------------------------------------------------- | ----------------------------- |
| `id`           | `str`                | The unique identifier of the notification.                    | *required*                    |
| `type`         | `NotificationType`   |                                                               | *required*                    |
| `message`      | `str`                | The message of the notification.                              | *required*                    |
| `status`       | `NotificationStatus` |                                                               | *required*                    |
| `created_at`   | `AwareDatetime`      | The UTC date and time when the notification has been created. | *required*                    |
| `resource_uri` | \`str                | None\`                                                        | The service URI of the entity |

#### NotificationStatus

The status of the notification.

#### NotificationType

The type of the notification

#### Organization

An organization that owns resources.

Parameters:

| Name           | Type                   | Description                                                                                                                                                                     | Default                                                |
| -------------- | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| `id`           | `str`                  | The unique identifier of an organization.                                                                                                                                       | *required*                                             |
| `name`         | `str`                  | The name of an organization. Contains only alphanumeric characters, hyphens, and underscores. Must start or end with alphanumeric. It must be globally case-insensitive unique. | *required*                                             |
| `display_name` | `str`                  | The display name of an organization.                                                                                                                                            | *required*                                             |
| `description`  | \`str                  | None\`                                                                                                                                                                          | The description of an organization. Supports markdown. |
| `logo`         | \`str                  | None\`                                                                                                                                                                          | The URL of the organization's logo.                    |
| `email`        | \`str                  | None\`                                                                                                                                                                          | The email address of the organization.                 |
| `website`      | \`str                  | None\`                                                                                                                                                                          | The URL of the organization's website.                 |
| `members`      | \`list[UserListItem]   | None\`                                                                                                                                                                          |                                                        |
| `metadata`     | \`OrganizationMetadata | None\`                                                                                                                                                                          |                                                        |

#### OrganizationConfig

The configuration for creating a new organization.

Parameters:

| Name           | Type  | Description                                                                                                                                                                     | Default                                                |
| -------------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| `name`         | `str` | The name of an organization. Contains only alphanumeric characters, hyphens, and underscores. Must start or end with alphanumeric. It must be globally case-insensitive unique. | *required*                                             |
| `display_name` | `str` | The display name of an organization.                                                                                                                                            | *required*                                             |
| `description`  | \`str | None\`                                                                                                                                                                          | The description of an organization. Supports markdown. |
| `logo_base64`  | \`str | None\`                                                                                                                                                                          | The base64-encoded image of the organization's logo.   |
| `email`        | \`str | None\`                                                                                                                                                                          | The email address of the organization.                 |
| `website`      | \`str | None\`                                                                                                                                                                          | The URL of the organization's website.                 |

#### OrganizationInvite

A non-personalized time-boxed invite to join an organization.

Parameters:

| Name              | Type            | Description | Default                                                                              |
| ----------------- | --------------- | ----------- | ------------------------------------------------------------------------------------ |
| `token`           | \`str           | None\`      | The generated token, encrypting organization, expiration timestamp, and role (VIEW). |
| `link`            | \`str           | None\`      | The generated invite link.                                                           |
| `expiration_date` | \`AwareDatetime | None\`      | The expiration date of the invite link. 72 hours after creation.                     |
| `organization_id` | \`str           | None\`      | The unique identifier of an organization.                                            |

#### OrganizationListItem

Essential organization details for listings.

Parameters:

| Name           | Type                   | Description                               | Default                                                                                                                                                                         |
| -------------- | ---------------------- | ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `id`           | `str`                  | The unique identifier of an organization. | *required*                                                                                                                                                                      |
| `name`         | \`str                  | None\`                                    | The name of an organization. Contains only alphanumeric characters, hyphens, and underscores. Must start or end with alphanumeric. It must be globally case-insensitive unique. |
| `display_name` | `str`                  | The display name of an organization.      | *required*                                                                                                                                                                      |
| `description`  | \`str                  | None\`                                    | The description of an organization. Supports markdown.                                                                                                                          |
| `logo`         | \`str                  | None\`                                    | The URL of the organization's logo.                                                                                                                                             |
| `metadata`     | \`OrganizationMetadata | None\`                                    |                                                                                                                                                                                 |

#### OrganizationMember

A member of an organization with their associated role.

Parameters:

| Name   | Type           | Description | Default |
| ------ | -------------- | ----------- | ------- |
| `user` | \`UserListItem | None\`      |         |
| `role` | \`MemberRole   | None\`      |         |

#### OrganizationMetadata

The metadata of an organization.

Parameters:

| Name                       | Type         | Description | Default |
| -------------------------- | ------------ | ----------- | ------- |
| `current_user_member_role` | \`MemberRole | None\`      |         |

#### PaginatedTotalCount

Parameters:

| Name   | Type  | Description                                  | Default    |
| ------ | ----- | -------------------------------------------- | ---------- |
| `root` | `int` | The total number of entities within the list | *required* |

#### Paid

Usage statistics of paid credits.

Parameters:

| Name      | Type               | Description | Default |
| --------- | ------------------ | ----------- | ------- |
| `monthly` | \`UsageCreditStats | None\`      |         |

#### ParallelGenerationJobs

The number of currently running generation jobs and the limit

Parameters:

| Name      | Type  | Description | Default                                                                                      |
| --------- | ----- | ----------- | -------------------------------------------------------------------------------------------- |
| `current` | \`int | None\`      | The number of currently running generation jobs.                                             |
| `limit`   | \`int | None\`      | The maximum number of running generation jobs at any time. If empty, then there is no limit. |

#### ParallelTrainingJobs

The number of currently running training jobs and the limit

Parameters:

| Name      | Type  | Description | Default                                                                                    |
| --------- | ----- | ----------- | ------------------------------------------------------------------------------------------ |
| `current` | \`int | None\`      | The number of currently running training jobs                                              |
| `limit`   | \`int | None\`      | The maximum number of running training jobs at any time. If empty, then there is no limit. |

#### PaymentUrl

The URL for purchasing a plan.

Parameters:

| Name    | Type     | Description | Default |
| ------- | -------- | ----------- | ------- |
| `value` | \`AnyUrl | None\`      |         |

#### PermissionLevel

The permission level of the user with respect to this resource

- `VIEW`: The user can view and use the resource
- `ADMIN`: The user can edit, delete and transfer ownership of the resource

#### Plan

A billing plan available for purchase.

Parameters:

| Name               | Type              | Description               | Default                      |
| ------------------ | ----------------- | ------------------------- | ---------------------------- |
| `id`               | `str`             | The identifier of a plan. | *required*                   |
| `name`             | `str`             | The name of the plan.     | *required*                   |
| `billing_interval` | \`BillingInterval | None\`                    |                              |
| `price`            | `Price`           |                           | *required*                   |
| `description`      | \`str             | None\`                    | The description of the plan. |
| `features`         | \`list[str]       | None\`                    |                              |
| `metadata`         | \`dict[str, str]  | None\`                    | The metadata of the plan.    |

#### PlanUpdateConfig

Request to upgrade or downgrade the user's plan.

Parameters:

| Name               | Type              | Description | Default                   |
| ------------------ | ----------------- | ----------- | ------------------------- |
| `id`               | \`str             | None\`      | The identifier of a plan. |
| `billing_interval` | \`BillingInterval | None\`      |                           |

#### Price

The price information for a plan.

Parameters:

| Name       | Type             | Description                      | Default                             |
| ---------- | ---------------- | -------------------------------- | ----------------------------------- |
| `id`       | \`str            | None\`                           | The unique identifier of the price. |
| `value`    | `float`          | The price value.                 | *required*                          |
| `currency` | `str`            | The currency code for the price. | *required*                          |
| `metadata` | `dict[str, str]` | The metadata of the price.       | *required*                          |

#### Probe

The generated synthetic samples returned as a result of the probe.

Parameters:

| Name   | Type                     | Description | Default                          |
| ------ | ------------------------ | ----------- | -------------------------------- |
| `name` | \`str                    | None\`      | The name of the table.           |
| `rows` | \`list\[dict[str, Any]\] | None\`      | An array of sample data objects. |

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

| Name                | Type                     | Description | Default                                                                                                                                                 |
| ------------------- | ------------------------ | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `id`                | \`str                    | None\`      | The unique identifier of the step.                                                                                                                      |
| `model_label`       | \`str                    | None\`      | The unique label for the model, consisting of table name and a suffix for the model type. This will be empty for steps that are not related to a model. |
| `compute_name`      | \`str                    | None\`      | The name of a compute resource.                                                                                                                         |
| `restarts`          | \`int                    | None\`      | The number of previous restarts for the corresponding task.                                                                                             |
| `task_type`         | \`TaskType               | None\`      |                                                                                                                                                         |
| `step_code`         | \`StepCode               | None\`      |                                                                                                                                                         |
| `start_date`        | \`AwareDatetime          | None\`      | The UTC date and time when the job has started. If the job has not started yet, then this is None.                                                      |
| `end_date`          | \`AwareDatetime          | None\`      | The UTC date and time when the job has ended. If the job is still, then this is None.                                                                   |
| `compute_resources` | \`ComputeResources       | None\`      |                                                                                                                                                         |
| `messages`          | \`list\[dict[str, Any]\] | None\`      |                                                                                                                                                         |
| `error_message`     | \`str                    | None\`      |                                                                                                                                                         |
| `progress`          | \`ProgressValue          | None\`      |                                                                                                                                                         |
| `status`            | \`ProgressStatus         | None\`      |                                                                                                                                                         |

#### ProgressValue

The progress of a job or a step.

Parameters:

| Name    | Type  | Description | Default |
| ------- | ----- | ----------- | ------- |
| `value` | \`int | None\`      |         |
| `max`   | \`int | None\`      |         |

#### ProviderScopeConfig

OAuth scope definition

Parameters:

| Name          | Type   | Description                     | Default                                   |
| ------------- | ------ | ------------------------------- | ----------------------------------------- |
| `id`          | `str`  | Unique identifier for the scope | *required*                                |
| `name`        | `str`  | Display name of the scope       | *required*                                |
| `description` | \`str  | None\`                          | Description of what this scope allows     |
| `value`       | `str`  | The actual OAuth scope value    | *required*                                |
| `is_default`  | \`bool | None\`                          | Whether this scope is selected by default |

#### RareCategoryReplacementMethod

Specifies how rare categories will be sampled. Only applicable if value protection has been enabled.

- `CONSTANT`: Replace rare categories by a constant `_RARE_` token.
- `SAMPLE`: Replace rare categories by a sample from non-rare categories.

#### RebalancingConfig

Configure rebalancing.

Parameters:

| Name            | Type               | Description                                                                                                            | Default    |
| --------------- | ------------------ | ---------------------------------------------------------------------------------------------------------------------- | ---------- |
| `column`        | `str`              | The name of the column to be rebalanced. Only applicable for a subject table. Only applicable for categorical columns. | *required* |
| `probabilities` | `dict[str, float]` | The target distribution of samples values. The keys are the categorical values, and the values are the probabilities.  | *required* |

#### SetVisibilityConfig

Configuration for setting the visibility of a resource.

Parameters:

| Name         | Type         | Description | Default    |
| ------------ | ------------ | ----------- | ---------- |
| `visibility` | `Visibility` |             | *required* |

#### Similarity

Metrics regarding the similarity of the full joint distributions of samples within an embedding space.

1. **Cosine Similarity**: The cosine similarity between the centroids of synthetic and training samples.
1. **Discriminator AUC**: The AUC of a discriminative model to distinguish between synthetic and training samples.

The Model2Vec model [potion-base-8M](https://huggingface.co/minishlab/potion-base-8M) is used to compute the embeddings of a string-ified representation of individual records. In case of sequential data the records, that belong to the same group, are being concatenated. We then calculate the cosine similarity between the centroids of the provided datasets within the embedding space.

Again, we expect the similarity metrics to be as close as possible to 1, but not significantly higher than what is measured for the holdout data, as this would again indicate overfitting.

In addition, a discriminative ML model is trained to distinguish between training and synthetic samples. The ability of this model to distinguish between training and synthetic samples is measured by the AUC score. For synthetic data to be considered realistic, the AUC score should be close to 0.5, which indicates that the synthetic data is indistinguishable from the training data.

Parameters:

| Name                                   | Type    | Description | Default                                                                                                                                                            |
| -------------------------------------- | ------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `cosine_similarity_training_synthetic` | \`float | None\`      | Cosine similarity between training and synthetic centroids.                                                                                                        |
| `cosine_similarity_training_holdout`   | \`float | None\`      | Cosine similarity between training and holdout centroids. Serves as a reference for cosine_similarity_training_synthetic.                                          |
| `discriminator_auc_training_synthetic` | \`float | None\`      | Cross-validated AUC of a discriminative model to distinguish between training and synthetic samples.                                                               |
| `discriminator_auc_training_holdout`   | \`float | None\`      | Cross-validated AUC of a discriminative model to distinguish between training and holdout samples. Serves as a reference for discriminator_auc_training_synthetic. |

#### SourceColumn

A column as part of a source table.

Parameters:

| Name                  | Type                     | Description                                                           | Default                                                                                                        |
| --------------------- | ------------------------ | --------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `id`                  | `str`                    | The unique identifier of a source column.                             | *required*                                                                                                     |
| `name`                | `str`                    | The name of a source column. It must be unique within a source table. | *required*                                                                                                     |
| `included`            | \`bool                   | None\`                                                                | If true, the column will be included in the training. If false, the column will be excluded from the training. |
| `model_encoding_type` | `ModelEncodingType`      |                                                                       | *required*                                                                                                     |
| `value_range`         | \`SourceColumnValueRange | None\`                                                                |                                                                                                                |

#### SourceColumnConfig

The configuration for a source column when creating a new generator.

Parameters:

| Name                  | Type                | Description                                                           | Default    |
| --------------------- | ------------------- | --------------------------------------------------------------------- | ---------- |
| `name`                | `str`               | The name of a source column. It must be unique within a source table. | *required* |
| `model_encoding_type` | \`ModelEncodingType | None\`                                                                |            |

#### SourceColumnValueRange

The (privacy-safe) range of values detected within a source column. These values can then be used as seed values for conditional simulation. For CATEGORICAL and NUMERIC_DISCRETE encoding types, this will be given as a list of unique values, sorted by popularity. For other NUMERIC and for DATETIME encoding types, this will be given as a min and max value. Note, that this property will only be populated, once the analysis step for the training of the generator has been completed.

Parameters:

| Name       | Type        | Description | Default                                                                         |
| ---------- | ----------- | ----------- | ------------------------------------------------------------------------------- |
| `min`      | \`str       | None\`      | The minimum value of the column. For dates, this is represented in ISO format.  |
| `max`      | \`str       | None\`      | The maximum value of the column. For dates, this is represented in ISO format.  |
| `values`   | \`list[str] | None\`      | The list of distinct values of the column. Limited to a maximum of 1000 values. |
| `has_null` | \`bool      | None\`      | If true, null value was detected within the column.                             |

#### SourceForeignKey

A foreign key relationship in a source table.

Parameters:

| Name               | Type   | Description                                                                                                                                           | Default    |
| ------------------ | ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `id`               | `str`  | The unique identifier of a foreign key.                                                                                                               | *required* |
| `column`           | `str`  | The column name of a foreign key.                                                                                                                     | *required* |
| `referenced_table` | `str`  | The table name of the referenced table. That table must have a primary key already defined.                                                           | *required* |
| `is_context`       | `bool` | If true, then the foreign key will be considered as a context relation. Note, that only one foreign key relation per table can be a context relation. | *required* |

#### SourceForeignKeyConfig

Configuration for defining a foreign key relationship in a source table.

Parameters:

| Name               | Type   | Description                                                                                                                                           | Default    |
| ------------------ | ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `column`           | `str`  | The column name of a foreign key.                                                                                                                     | *required* |
| `referenced_table` | `str`  | The table name of the referenced table. That table must have a primary key already defined.                                                           | *required* |
| `is_context`       | `bool` | If true, then the foreign key will be considered as a context relation. Note, that only one foreign key relation per table can be a context relation. | *required* |

#### SourceTable

A table as part of a generator.

Parameters:

| Name                           | Type                     | Description                                                       | Default                                                                                                                          |
| ------------------------------ | ------------------------ | ----------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `id`                           | `str`                    | The unique identifier of a source table.                          | *required*                                                                                                                       |
| `source_connector_id`          | \`str                    | None\`                                                            | The unique identifier of a connector.                                                                                            |
| `location`                     | \`str                    | None\`                                                            | The location of a source table. Together with the source connector it uniquely identifies a source, and samples data from there. |
| `name`                         | `str`                    | The name of a source table. It must be unique within a generator. | *required*                                                                                                                       |
| `primary_key`                  | \`str                    | None\`                                                            | The column name of the primary key.                                                                                              |
| `columns`                      | \`list[SourceColumn]     | None\`                                                            | The columns of this generator table.                                                                                             |
| `foreign_keys`                 | \`list[SourceForeignKey] | None\`                                                            | The foreign keys of a table.                                                                                                     |
| `tabular_model_metrics`        | \`ModelMetrics           | None\`                                                            |                                                                                                                                  |
| `language_model_metrics`       | \`ModelMetrics           | None\`                                                            |                                                                                                                                  |
| `tabular_model_configuration`  | \`ModelConfiguration     | None\`                                                            |                                                                                                                                  |
| `language_model_configuration` | \`ModelConfiguration     | None\`                                                            |                                                                                                                                  |
| `total_rows`                   | \`int                    | None\`                                                            | The total number of rows in the source table while fetching data for training.                                                   |

#### SourceTableAddConfig

Configuration for adding a new source table to a generator.

Parameters:

| Name                           | Type                 | Description                                                                                                                      | Default                                                                                                                       |
| ------------------------------ | -------------------- | -------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `source_connector_id`          | `str`                | The unique identifier of a connector.                                                                                            | *required*                                                                                                                    |
| `location`                     | `str`                | The location of a source table. Together with the source connector it uniquely identifies a source, and samples data from there. | *required*                                                                                                                    |
| `name`                         | \`str                | None\`                                                                                                                           | The name of a source table. It must be unique within a generator.                                                             |
| `include_children`             | \`bool               | None\`                                                                                                                           | If true, all tables that are referenced by foreign keys will be included. If false, only the selected table will be included. |
| `tabular_model_configuration`  | \`ModelConfiguration | None\`                                                                                                                           |                                                                                                                               |
| `language_model_configuration` | \`ModelConfiguration | None\`                                                                                                                           |                                                                                                                               |

#### SourceTableConfig

The configuration for a source table when creating a new generator.

Parameters:

| Name                           | Type                           | Description                                                       | Default                                                                                                                          |
| ------------------------------ | ------------------------------ | ----------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `name`                         | `str`                          | The name of a source table. It must be unique within a generator. | *required*                                                                                                                       |
| `source_connector_id`          | \`str                          | None\`                                                            | The unique identifier of a connector.                                                                                            |
| `location`                     | \`str                          | None\`                                                            | The location of a source table. Together with the source connector it uniquely identifies a source, and samples data from there. |
| `data`                         | \`str                          | None\`                                                            | The base64-encoded string derived from a Parquet file containing the specified source table.                                     |
| `tabular_model_configuration`  | \`ModelConfiguration           | None\`                                                            |                                                                                                                                  |
| `language_model_configuration` | \`ModelConfiguration           | None\`                                                            |                                                                                                                                  |
| `primary_key`                  | \`str                          | None\`                                                            | The column name of the primary key.                                                                                              |
| `foreign_keys`                 | \`list[SourceForeignKeyConfig] | None\`                                                            | The foreign key configurations of this table.                                                                                    |
| `columns`                      | \`list[SourceColumnConfig]     | None\`                                                            | The column configurations of this table.                                                                                         |

##### extract_columns_from_data_if_missing

```python
extract_columns_from_data_if_missing()
```

extract column names from base64 data if columns are not provided.

#### StepCode

The unique code for the step.

#### SyntheticDataset

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

| Name        | Type  | Description | Default |
| ----------- | ----- | ----------- | ------- |
| `file_path` | \`str | Path        | None\`  |

Returns:

| Name   | Type   | Description                 |
| ------ | ------ | --------------------------- |
| `Path` | `Path` | The path to the saved file. |

###### progress

```python
progress()
```

Retrieve the progress of the generation process.

Returns:

| Name          | Type          | Description                             |
| ------------- | ------------- | --------------------------------------- |
| `JobProgress` | `JobProgress` | The progress of the generation process. |

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

| Name           | Type    | Description                                                         | Default |
| -------------- | ------- | ------------------------------------------------------------------- | ------- |
| `progress_bar` | `bool`  | If true, displays a progress bar. Default is True.                  | `True`  |
| `interval`     | `float` | Interval in seconds to poll the job progress. Default is 2 seconds. | `2`     |

##### config

```python
config()
```

Retrieve writable synthetic dataset properties.

Returns:

| Name                     | Type                     | Description                                                 |
| ------------------------ | ------------------------ | ----------------------------------------------------------- |
| `SyntheticDatasetConfig` | `SyntheticDatasetConfig` | The synthetic dataset properties as a configuration object. |

##### data

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

| Name        | Type                                | Description                                                | Default     |
| ----------- | ----------------------------------- | ---------------------------------------------------------- | ----------- |
| `file_path` | \`str                               | Path                                                       | None\`      |
| `format`    | `Literal['parquet', 'csv', 'json']` | The format of the synthetic dataset. Default is "parquet". | `'parquet'` |

Returns:

| Name   | Type   | Description                 |
| ------ | ------ | --------------------------- |
| `Path` | `Path` | The path to the saved file. |

##### reports

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

##### update

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

#### SyntheticDatasetConfig

The configuration for creating a new synthetic dataset.

Parameters:

| Name           | Type                         | Description | Default                                                                                                                                                                                                                                                                                              |
| -------------- | ---------------------------- | ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `generator_id` | \`str                        | None\`      | The unique identifier of a generator.                                                                                                                                                                                                                                                                |
| `name`         | \`str                        | None\`      | The name of a synthetic dataset.                                                                                                                                                                                                                                                                     |
| `description`  | \`str                        | None\`      | The description of a synthetic dataset.                                                                                                                                                                                                                                                              |
| `random_state` | \`int                        | None\`      | Seed for the random number generators. If None, the random number generator is initialized randomly, yielding different results for every run. Setting it to a specific integer ensures reproducible results across runs. Useful when consistent results are desired, e.g. for testing or debugging. |
| `tables`       | \`list[SyntheticTableConfig] | None\`      |                                                                                                                                                                                                                                                                                                      |
| `delivery`     | \`SyntheticDatasetDelivery   | None\`      |                                                                                                                                                                                                                                                                                                      |
| `compute`      | \`str                        | None\`      | The unique identifier of a compute resource. Not applicable for SDK.                                                                                                                                                                                                                                 |

#### SyntheticDatasetDelivery

Configuration for delivering a synthetic dataset to a destination.

Parameters:

| Name                       | Type   | Description                                                                                                 | Default    |
| -------------------------- | ------ | ----------------------------------------------------------------------------------------------------------- | ---------- |
| `overwrite_tables`         | `bool` | If true, tables in the destination will be overwritten. If false, any tables exist, the delivery will fail. | *required* |
| `destination_connector_id` | `str`  | The unique identifier of a connector.                                                                       | *required* |
| `location`                 | `str`  | The location for the destination connector.                                                                 | *required* |

#### SyntheticDatasetListItem

Essential synthetic dataset details for listings.

Parameters:

| Name                | Type                    | Description                                   | Default                                                 |
| ------------------- | ----------------------- | --------------------------------------------- | ------------------------------------------------------- |
| `id`                | `str`                   | The unique identifier of a synthetic dataset. | *required*                                              |
| `metadata`          | \`Metadata              | None\`                                        |                                                         |
| `name`              | `str`                   | The name of a synthetic dataset.              | *required*                                              |
| `description`       | \`str                   | None\`                                        | The description of a synthetic dataset.                 |
| `generation_status` | `ProgressStatus`        |                                               | *required*                                              |
| `generation_time`   | \`AwareDatetime         | None\`                                        | The UTC date and time when the generation has finished. |
| `usage`             | \`SyntheticDatasetUsage | None\`                                        |                                                         |

#### SyntheticDatasetUsage

Usage statistics of a synthetic dataset.

Parameters:

| Name                     | Type    | Description | Default                                                                                                                                                                               |
| ------------------------ | ------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `total_datapoints`       | \`int   | None\`      | The number of datapoints in the synthetic dataset. Deprecated: This field is no longer valid and will always return -1. It will be removed in a future version.                       |
| `total_compute_time`     | \`int   | None\`      | The total compute time in seconds used for generating this synthetic dataset. This is the sum of the compute time of all trained tasks.                                               |
| `total_credits`          | \`float | None\`      | The amount of credits consumed for generating the synthetic dataset.                                                                                                                  |
| `total_virtual_cpu_time` | \`float | None\`      | The total virtual CPU time in seconds used for training this generator. This is the sum of the elapsed time multiplied by number of allocated virtual CPUs across all training tasks. |
| `total_virtual_gpu_time` | \`float | None\`      | The total virtual GPU time in seconds used for training this generator. This is the sum of the elapsed time multiplied by number of allocated virtual GPUs across all training tasks. |
| `no_of_likes`            | \`int   | None\`      | Number of likes of this synthetic dataset.                                                                                                                                            |
| `no_of_downloads`        | \`int   | None\`      | Number of downloads of this synthetic dataset.                                                                                                                                        |
| `no_of_threads`          | \`int   | None\`      | Number of assistant threads using this synthetic dataset.                                                                                                                             |

#### SyntheticProbeConfig

The configuration for probing for new synthetic samples.

Parameters:

| Name           | Type                         | Description | Default                                                                                                                                                                                                                                                                                              |
| -------------- | ---------------------------- | ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `generator_id` | \`str                        | None\`      | The unique identifier of a generator.                                                                                                                                                                                                                                                                |
| `random_state` | \`int                        | None\`      | Seed for the random number generators. If None, the random number generator is initialized randomly, yielding different results for every run. Setting it to a specific integer ensures reproducible results across runs. Useful when consistent results are desired, e.g. for testing or debugging. |
| `tables`       | \`list[SyntheticTableConfig] | None\`      |                                                                                                                                                                                                                                                                                                      |

#### SyntheticTable

A synthetic table that will be generated.

Parameters:

| Name                      | Type                          | Description                                                       | Default                                                                                                      |
| ------------------------- | ----------------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `id`                      | \`str                         | None\`                                                            | The unique identifier of a synthetic table.                                                                  |
| `name`                    | `str`                         | The name of a source table. It must be unique within a generator. | *required*                                                                                                   |
| `configuration`           | \`SyntheticTableConfiguration | None\`                                                            |                                                                                                              |
| `tabular_model_metrics`   | \`ModelMetrics                | None\`                                                            |                                                                                                              |
| `language_model_metrics`  | \`ModelMetrics                | None\`                                                            |                                                                                                              |
| `foreign_keys`            | \`list[SourceForeignKey]      | None\`                                                            | The foreign keys of a table.                                                                                 |
| `total_rows`              | \`int                         | None\`                                                            | The total number of rows for that table in the generated synthetic dataset.                                  |
| `total_datapoints`        | \`int                         | None\`                                                            | Deprecated: This field is no longer valid and will always return -1. It will be removed in a future version. |
| `source_table_total_rows` | \`int                         | None\`                                                            | The total number of rows in the source table while fetching data for training.                               |

#### SyntheticTableConfig

The configuration for a synthetic table when creating a new synthetic dataset.

Parameters:

| Name            | Type                          | Description                                                                          | Default    |
| --------------- | ----------------------------- | ------------------------------------------------------------------------------------ | ---------- |
| `name`          | `str`                         | The name of a synthetic table. This matches the name of a corresponding SourceTable. | *required* |
| `configuration` | \`SyntheticTableConfiguration | None\`                                                                               |            |

#### SyntheticTableConfiguration

The sample configuration for a synthetic table

Parameters:

| Name                       | Type                | Description | Default                                                                                                                                                                                                                                                                                         |
| -------------------------- | ------------------- | ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `sample_size`              | \`int               | None\`      | Number of generated samples. Only applicable for subject tables. If neither size nor seed is provided, then the default behavior for Synthetic Datasets is to generate the same size of samples as the original, and the default behavior for Synthetic Probes is to generate one subject only. |
| `sample_seed_connector_id` | \`str               | None\`      | The connector id of the seed data for conditional simulation                                                                                                                                                                                                                                    |
| `sample_seed_dict`         | \`str               | None\`      | The base64-encoded string derived from a json line file containing the specified sample seed data. This allows conditional live probing via non-python clients.                                                                                                                                 |
| `sample_seed_data`         | \`str               | None\`      | The base64-encoded string derived from a Parquet file containing the specified sample seed data. This allows conditional simulation as well as live probing via python clients.                                                                                                                 |
| `sampling_temperature`     | \`float             | None\`      | temperature for sampling                                                                                                                                                                                                                                                                        |
| `sampling_top_p`           | \`float             | None\`      | topP for sampling                                                                                                                                                                                                                                                                               |
| `rebalancing`              | \`RebalancingConfig | None\`      |                                                                                                                                                                                                                                                                                                 |
| `imputation`               | \`ImputationConfig  | None\`      |                                                                                                                                                                                                                                                                                                 |
| `fairness`                 | \`FairnessConfig    | None\`      |                                                                                                                                                                                                                                                                                                 |
| `enable_data_report`       | \`bool              | None\`      | If false, then the Data report is not generated. If enableDataReport is set to false on generator, then enableDataReport is automatically set to false.                                                                                                                                         |

#### TaskType

The type of the task.

#### TransferOwnershipConfig

The configuration for transferring ownership of a resource to an account.

Parameters:

| Name         | Type  | Description | Default                                                                 |
| ------------ | ----- | ----------- | ----------------------------------------------------------------------- |
| `account_id` | \`str | None\`      | The unique identifier of an account (either a user or an organization). |

#### UsageCreditStats

User credits statistics and limits for the current user.

Parameters:

| Name           | Type            | Description | Default                                                          |
| -------------- | --------------- | ----------- | ---------------------------------------------------------------- |
| `current`      | \`float         | None\`      | The current credit balance for the user.                         |
| `limit`        | \`float         | None\`      | The credit limit for the user. If empty, then there is no limit. |
| `period_start` | \`AwareDatetime | None\`      | The UTC date and time when the current time period started.      |
| `period_end`   | \`AwareDatetime | None\`      | The UTC date and time when the current time period ends.         |

#### User

The public attributes of a user of the service.

Parameters:

| Name            | Type                         | Description | Default                                                                                                                                                                                                    |
| --------------- | ---------------------------- | ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `id`            | \`str                        | None\`      | The unique identifier of a user.                                                                                                                                                                           |
| `name`          | \`str                        | None\`      | The name of a user. Contains only alphanumeric characters, hyphens, and underscores. Must start or end with alphanumeric. It must be globally case-insensitive unique considering organizations and users. |
| `first_name`    | \`str                        | None\`      | First name of a user                                                                                                                                                                                       |
| `last_name`     | \`str                        | None\`      | Last name of a user                                                                                                                                                                                        |
| `avatar`        | \`str                        | None\`      | The URL of the user's avatar                                                                                                                                                                               |
| `organizations` | \`list[OrganizationListItem] | None\`      | The organizations the user belongs to                                                                                                                                                                      |

#### UserCredits

Usage statistics and limits for the current user.

Parameters:

| Name   | Type   | Description                       | Default                           |
| ------ | ------ | --------------------------------- | --------------------------------- |
| `free` | `Free` | Usage statistics of free credits. | *required*                        |
| `paid` | \`Paid | None\`                            | Usage statistics of paid credits. |

#### UserListItem

Essential information about a user for public listings.

Parameters:

| Name         | Type  | Description | Default                                                                                                                                                                                                    |
| ------------ | ----- | ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `id`         | \`str | None\`      | The unique identifier of a user.                                                                                                                                                                           |
| `name`       | \`str | None\`      | The name of a user. Contains only alphanumeric characters, hyphens, and underscores. Must start or end with alphanumeric. It must be globally case-insensitive unique considering organizations and users. |
| `first_name` | \`str | None\`      | First name of a user                                                                                                                                                                                       |
| `last_name`  | \`str | None\`      | Last name of a user                                                                                                                                                                                        |
| `avatar`     | \`str | None\`      | The URL of the user's avatar                                                                                                                                                                               |

#### UserPlan

The type of the user subscription plan.

#### UserSecretConfig

Request body for creating a new user secret

Parameters:

| Name    | Type  | Description                                                                                                                         | Default    |
| ------- | ----- | ----------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `key`   | `str` | The name of a user secret environment variable. Must match pattern: A-Z\_\* Examples: MY_API_KEY, DATABASE_PASSWORD, AWS_SECRET_KEY | *required* |
| `value` | `str` | The secret value (will be encrypted and stored securely)                                                                            | *required* |

#### UserSecretKey

Parameters:

| Name   | Type  | Description                                                                                                                         | Default    |
| ------ | ----- | ----------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `root` | `str` | The name of a user secret environment variable. Must match pattern: A-Z\_\* Examples: MY_API_KEY, DATABASE_PASSWORD, AWS_SECRET_KEY | *required* |

#### UserSettingsAssistantUpdateConfig

Configuration for updating a user's assistant-related settings.

Parameters:

| Name                  | Type  | Description | Default                                                                                  |
| --------------------- | ----- | ----------- | ---------------------------------------------------------------------------------------- |
| `about_user_message`  | \`str | None\`      | The instruction what the Assistant should know about the user to provide better response |
| `about_model_message` | \`str | None\`      | The instruction how the Assistant should respond                                         |

#### UserSettingsProfileUpdateConfig

Configuration for updating a user's profile settings.

Parameters:

| Name         | Type  | Description | Default                                                                                                                                                                                                    |
| ------------ | ----- | ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `name`       | \`str | None\`      | The name of a user. Contains only alphanumeric characters, hyphens, and underscores. Must start or end with alphanumeric. It must be globally case-insensitive unique considering organizations and users. |
| `first_name` | \`str | None\`      | First name of a user                                                                                                                                                                                       |
| `last_name`  | \`str | None\`      | Last name of a user                                                                                                                                                                                        |
| `avatar`     | \`str | None\`      | The base64-encoded image of the user's avatar                                                                                                                                                              |

#### UserSettingsUpdateConfig

The configuration for updating user settings.

Parameters:

| Name        | Type                                | Description | Default |
| ----------- | ----------------------------------- | ----------- | ------- |
| `profile`   | \`UserSettingsProfileUpdateConfig   | None\`      |         |
| `assistant` | \`UserSettingsAssistantUpdateConfig | None\`      |         |

#### UserUsage

Usage statistics and limits for the current user.

Parameters:

| Name                       | Type                     | Description | Default                                                       |
| -------------------------- | ------------------------ | ----------- | ------------------------------------------------------------- |
| `parallel_training_jobs`   | \`ParallelTrainingJobs   | None\`      | The number of currently running training jobs and the limit   |
| `parallel_generation_jobs` | \`ParallelGenerationJobs | None\`      | The number of currently running generation jobs and the limit |

#### Visibility

Indicates the visibility of the resource.

- `PUBLIC` - Everyone can access the resource.
- `UNLISTED`- Anyone with the direct link can access the resource. No public listings.
- `PRIVATE` - Accessible only by the owner. For organizations, all members can access.

#### \_SyntheticDataConfigValidation

Validation logic for SyntheticDatasetConfig and SyntheticProbeConfig against Generator

Parameters:

| Name               | Type                     | Description            | Default    |
| ------------------ | ------------------------ | ---------------------- | ---------- |
| `synthetic_config` | \`SyntheticDatasetConfig | SyntheticProbeConfig\` |            |
| `generator`        | `Generator`              |                        | *required* |

#### \_SyntheticTableConfigValidation

Validation logic for SyntheticTableConfig against SourceTable

Parameters:

| Name              | Type                   | Description | Default    |
| ----------------- | ---------------------- | ----------- | ---------- |
| `synthetic_table` | `SyntheticTableConfig` |             | *required* |
| `source_table`    | `SourceTable`          |             | *required* |
