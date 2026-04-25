# Copyright 2025 MOSTLY AI
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

from __future__ import annotations

import uuid
import zipfile
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal

import pandas as pd
import rich
from pydantic import AwareDatetime, Field, field_validator, model_validator

from mostlyai.sdk.client._base_utils import convert_to_base64, convert_to_df, read_table_from_path
from mostlyai.sdk.client._constraint_types import convert_constraint_config_to_typed
from mostlyai.sdk.client.base import CustomBaseModel


class ConnectorAccessType(str, Enum):
    """
    The access permissions of a connector.

    - `READ_PROTECTED`:  The connector is restricted to being used solely as a source for training a generator. Direct data access is not permitted, only schema access is available.
    - `READ_DATA`: This connector allows full read access. It can also be used as a source for training a generator.
    - `WRITE_DATA`: This connector allows full read and write access. It can be also used as a source for training a generator, as well as a destination for delivering a synthetic dataset.
    - `SOURCE`: DEPRECATED - equivalent to READ_PROTECTED
    - `DESTINATION`: DEPRECATED - equivalent to WRITE_DATA

    """

    read_protected = "READ_PROTECTED"
    read_data = "READ_DATA"
    write_data = "WRITE_DATA"
    source = "SOURCE"
    destination = "DESTINATION"


class ConnectorType(str, Enum):
    """
    The type of a connector.

    The type determines the structure of the config, secrets and ssl parameters.

    - `MYSQL`: MySQL database
    - `POSTGRES`: PostgreSQL database
    - `MSSQL`: Microsoft SQL Server database
    - `ORACLE`: Oracle database
    - `MARIADB`: MariaDB database
    - `SNOWFLAKE`: Snowflake cloud data platform
    - `BIGQUERY`: Google BigQuery cloud data warehouse
    - `SQLITE`: SQLite database
    - `AZURE_STORAGE`: Azure Blob Storage
    - `GOOGLE_CLOUD_STORAGE`: Google Cloud Storage
    - `S3_STORAGE`: Amazon S3 Storage
    - `FILE_UPLOAD`: File upload

    """

    mysql = "MYSQL"
    postgres = "POSTGRES"
    mssql = "MSSQL"
    oracle = "ORACLE"
    mariadb = "MARIADB"
    snowflake = "SNOWFLAKE"
    bigquery = "BIGQUERY"
    sqlite = "SQLITE"
    azure_storage = "AZURE_STORAGE"
    google_cloud_storage = "GOOGLE_CLOUD_STORAGE"
    s3_storage = "S3_STORAGE"
    file_upload = "FILE_UPLOAD"


class ConnectorConfig(CustomBaseModel):
    """
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

    """

    name: str | None = Field(None, description="The name of a connector.")
    description: str | None = Field(None, description="The description of a connector.")
    type: ConnectorType
    access_type: ConnectorAccessType | None = Field(ConnectorAccessType.read_protected, alias="accessType")
    config: dict[str, Any] | None = None
    secrets: Annotated[dict[str, str] | None, Field(repr=False)] = None
    ssl: Annotated[dict[str, str] | None, Field(repr=False)] = None


class ConnectorPatchConfig(CustomBaseModel):
    """
    See ConnectorConfig for details on the structure of the connection parameters.

    """

    name: str | None = Field(None, description="The name of a connector.")
    description: str | None = Field(None, description="The description of a connector.")
    access_type: ConnectorAccessType | None = Field(ConnectorAccessType.read_protected, alias="accessType")
    config: dict[str, Any] | None = None
    secrets: Annotated[dict[str, str] | None, Field(repr=False)] = None
    ssl: Annotated[dict[str, str] | None, Field(repr=False)] = None


class ConnectorDeleteDataConfig(CustomBaseModel):
    """
    Configuration for deleting data from a connector.
    """

    location: str = Field(
        ...,
        description="Specifies the target within the connector to delete. The format of this parameter varies by connector type.",
    )


class ConnectorReadDataConfig(CustomBaseModel):
    """
    Configuration for reading data from a connector.
    """

    location: str | None = Field(
        None,
        description="Specifies the target within the connector from which to retrieve the data. The format of this parameter varies by connector type.",
    )
    limit: int | None = Field(
        None, description="The maximum number of rows to return. Return all if not specified.", ge=1
    )
    shuffle: bool | None = Field(False, description="Whether to shuffle the results.")


class IfExists(str, Enum):
    """
    The behavior if the target location already exists.

    - `APPEND`: Append the data to the existing target.
    - `REPLACE`: Replace the existing target with the new data.
    - `FAIL`: Fail if the target already exists.

    """

    append = "APPEND"
    replace = "REPLACE"
    fail = "FAIL"


class ConnectorWriteDataConfig(CustomBaseModel):
    """
    Configuration for writing data to a connector.
    """

    file: bytes = Field(..., description="Binary Parquet file containing the data to write.")
    location: str = Field(
        ...,
        description="Specifies the target within the connector to which to write the data. The format of this parameter varies by connector type.",
    )
    if_exists: IfExists | None = Field(
        None,
        alias="ifExists",
        description="The behavior if the target location already exists.\n\n- `APPEND`: Append the data to the existing target.\n- `REPLACE`: Replace the existing target with the new data.\n- `FAIL`: Fail if the target already exists.\n",
    )


class ConnectorQueryConfig(CustomBaseModel):
    sql: str = Field(..., description="SQL read-only (e.g. SELECT) statement to execute.")


class SourceColumnValueRange(CustomBaseModel):
    """
    The (privacy-safe) range of values detected within a source column. These values can then be used as seed values
    for conditional simulation. For CATEGORICAL and NUMERIC_DISCRETE encoding types, this will be given as a list
    of unique values, sorted by popularity. For other NUMERIC and for DATETIME encoding types, this will be given
    as a min and max value. Note, that this property will only be populated, once the analysis step for the training
    of the generator has been completed.

    """

    min: str | None = Field(
        None, description="The minimum value of the column. For dates, this is represented in ISO format."
    )
    max: str | None = Field(
        None, description="The maximum value of the column. For dates, this is represented in ISO format."
    )
    values: list[str] | None = Field(
        None, description="The list of distinct values of the column. Limited to a maximum of 1000 values."
    )
    has_null: bool | None = Field(None, description="If true, null value was detected within the column.")


class SourceForeignKey(CustomBaseModel):
    """
    A foreign key relationship in a source table.
    """

    id: str = Field(..., description="The unique identifier of a foreign key.")
    column: str = Field(..., description="The column name of a foreign key.", min_length=1)
    referenced_table: str = Field(
        ...,
        alias="referencedTable",
        description="The table name of the referenced table. That table must have a primary key already defined.",
        min_length=1,
    )
    is_context: bool = Field(
        ...,
        alias="isContext",
        description="If true, then the foreign key will be considered as a context relation.\nNote, that only one foreign key relation per table can be a context relation.\n",
    )

    @model_validator(mode="before")
    @classmethod
    def add_required_fields(cls, values):
        if isinstance(values, dict):
            if "id" not in values:
                values["id"] = str(uuid.uuid4())
        return values


class GeneratorCloneTrainingStatus(str, Enum):
    """
    The training status of the new generator. The available options are:

    - `NEW`: The new generator will re-use existing data and model configurations.
    - `CONTINUE`: The new generator will re-use existing data and model configurations, as well as model weights.

    """

    new = "NEW"
    continue_ = "CONTINUE"


class ConstraintType(str, Enum):
    """
    The type of constraint. If type is 'Inequality', then 'table_name', 'lowColumn' and 'highColumn' are required. If type is 'FixedCombinations', then 'table_name' and 'columns' are required.

    """

    inequality = "Inequality"
    fixed_combinations = "FixedCombinations"


class GeneratorPatchConfig(CustomBaseModel):
    """
    The configuration for updating a generator.
    """

    name: str | None = Field(None, description="The name of a generator.")
    description: str | None = Field(None, description="The description of a generator.")
    random_state: int | None = Field(
        None,
        alias="randomState",
        description="Seed for the random number generators. If None, the random number generator is initialized randomly, yielding different results for every run.\nSetting it to a specific integer ensures reproducible results across runs.\nUseful when consistent results are desired, e.g. for testing or debugging.\n",
    )


class SourceForeignKeyConfig(CustomBaseModel):
    """
    Configuration for defining a foreign key relationship in a source table.
    """

    column: str = Field(..., description="The column name of a foreign key.", min_length=1)
    referenced_table: str = Field(
        ...,
        alias="referencedTable",
        description="The table name of the referenced table. That table must have a primary key already defined.",
        min_length=1,
    )
    is_context: bool = Field(
        ...,
        alias="isContext",
        description="If true, then the foreign key will be considered as a context relation.\nNote, that only one foreign key relation per table can be a context relation.\n",
    )


class RareCategoryReplacementMethod(str, Enum):
    """
    Specifies how rare categories will be sampled.
    Only applicable if value protection has been enabled.

    - `CONSTANT`: Replace rare categories by a constant `_RARE_` token.
    - `SAMPLE`: Replace rare categories by a sample from non-rare categories.

    """

    constant = "CONSTANT"
    sample = "SAMPLE"


class TaskType(str, Enum):
    """
    The type of the task.
    """

    sync = "SYNC"
    train_tabular = "TRAIN_TABULAR"
    train_language = "TRAIN_LANGUAGE"
    finalize_training = "FINALIZE_TRAINING"
    generate_tabular = "GENERATE_TABULAR"
    generate_language = "GENERATE_LANGUAGE"
    finalize_generation = "FINALIZE_GENERATION"
    probe_tabular = "PROBE_TABULAR"
    probe_language = "PROBE_LANGUAGE"
    finalize_probing = "FINALIZE_PROBING"
    generate = "GENERATE"
    probe = "PROBE"


class StepCode(str, Enum):
    """
    The unique code for the step.
    """

    pull_training_data = "PULL_TRAINING_DATA"
    analyze_training_data = "ANALYZE_TRAINING_DATA"
    encode_training_data = "ENCODE_TRAINING_DATA"
    train_model = "TRAIN_MODEL"
    generate_data = "GENERATE_DATA"
    generate_model_report_data = "GENERATE_MODEL_REPORT_DATA"
    create_model_report = "CREATE_MODEL_REPORT"
    finalize_training = "FINALIZE_TRAINING"
    generate_data_tabular = "GENERATE_DATA_TABULAR"
    generate_data_language = "GENERATE_DATA_LANGUAGE"
    create_data_report = "CREATE_DATA_REPORT"
    create_data_report_tabular = "CREATE_DATA_REPORT_TABULAR"
    create_data_report_language = "CREATE_DATA_REPORT_LANGUAGE"
    finalize_generation = "FINALIZE_GENERATION"
    deliver_data = "DELIVER_DATA"
    finalize_probing = "FINALIZE_PROBING"


class ProgressValue(CustomBaseModel):
    """
    The progress of a job or a step.
    """

    value: int | None = 0
    max: int | None = 1


class ProgressStatus(str, Enum):
    """
    The status of a job or a step.

    - `NEW`: The job/step is being configured, and has not started yet
    - `CONTINUE`: The job/step is being configured, but has existing artifacts
    - `ON_HOLD`: The job/step has been started, but is kept on hold
    - `QUEUED`: The job/step has been started, and is awaiting for resources to execute
    - `IN_PROGRESS`: The job/step is currently running
    - `DONE`: The job/step has finished successfully
    - `FAILED`: The job/step has failed
    - `CANCELED`: The job/step has been canceled

    """

    new = "NEW"
    continue_ = "CONTINUE"
    on_hold = "ON_HOLD"
    queued = "QUEUED"
    in_progress = "IN_PROGRESS"
    done = "DONE"
    failed = "FAILED"
    canceled = "CANCELED"


class SyntheticDatasetFormat(str, Enum):
    csv = "CSV"
    parquet = "PARQUET"
    xlsx = "XLSX"


class SyntheticDatasetReportType(str, Enum):
    model = "MODEL"
    data = "DATA"


class SyntheticDatasetDelivery(CustomBaseModel):
    """
    Configuration for delivering a synthetic dataset to a destination.
    """

    overwrite_tables: bool = Field(
        ...,
        alias="overwriteTables",
        description="If true, tables in the destination will be overwritten.\nIf false, any tables exist, the delivery will fail.\n",
    )
    destination_connector_id: str = Field(
        ..., alias="destinationConnectorId", description="The unique identifier of a connector."
    )
    location: str = Field(..., description="The location for the destination connector.")


class ModelType(str, Enum):
    """
    The type of model.

    - `TABULAR`: A generative AI model tailored towards tabular data, trained from scratch.
    - `LANGUAGE`: A generative AI model build upon a (pre-trained) language model.

    """

    tabular = "TABULAR"
    language = "LANGUAGE"


class ModelEncodingType(str, Enum):
    """
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

    """

    auto = "AUTO"
    tabular_categorical = "TABULAR_CATEGORICAL"
    tabular_numeric_auto = "TABULAR_NUMERIC_AUTO"
    tabular_numeric_discrete = "TABULAR_NUMERIC_DISCRETE"
    tabular_numeric_binned = "TABULAR_NUMERIC_BINNED"
    tabular_numeric_digit = "TABULAR_NUMERIC_DIGIT"
    tabular_character = "TABULAR_CHARACTER"
    tabular_datetime = "TABULAR_DATETIME"
    tabular_datetime_relative = "TABULAR_DATETIME_RELATIVE"
    tabular_lat_long = "TABULAR_LAT_LONG"
    language_text = "LANGUAGE_TEXT"
    language_categorical = "LANGUAGE_CATEGORICAL"
    language_numeric = "LANGUAGE_NUMERIC"
    language_datetime = "LANGUAGE_DATETIME"


class RebalancingConfig(CustomBaseModel):
    """
    Configure rebalancing.
    """

    column: str = Field(
        ...,
        description="The name of the column to be rebalanced.  Only applicable for a subject table.\nOnly applicable for categorical columns.\n",
    )
    probabilities: dict[str, float] = Field(
        ...,
        description="The target distribution of samples values.\nThe keys are the categorical values, and the values are the probabilities.\n",
        examples=[[{"US": 0.8}, {"male": 0.5, "female": 0.5}]],
    )

    @field_validator("probabilities", mode="after")
    def validate_probabilities(cls, v):
        if not all(0 <= v <= 1 for v in v.values()):
            raise ValueError("the probabilities must be between 0 and 1")
        if not sum(v.values()) <= 1:
            raise ValueError("the sum of probabilities must be less than or equal to 1")
        return v


class ImputationConfig(CustomBaseModel):
    """
    Configure imputation.
    """

    columns: list[str] = Field(
        ...,
        description="The names of the columns to be imputed.\nImputed columns will suppress the sampling of NULL values.\n",
    )


class FairnessConfig(CustomBaseModel):
    """
    Configure a fairness objective for the table.
    Only applicable for a subject table.
    The generated synthetic data will maintain robust statistical parity between the target column and
    the specified sensitive columns. All these columns must be categorical.

    """

    target_column: str = Field(..., alias="targetColumn", description="The name of the target column.", min_length=1)
    sensitive_columns: list[str] = Field(
        ..., alias="sensitiveColumns", description="The names of the sensitive columns."
    )


class DifferentialPrivacyConfig(CustomBaseModel):
    """
    The optional differential privacy configuration for training the model.
    If not provided, then no differential privacy will be applied.

    """

    max_epsilon: float | None = Field(
        10.0,
        alias="maxEpsilon",
        description="Specifies the maximum allowable epsilon value. If the training process exceeds this threshold, it will be terminated early. Only model checkpoints with epsilon values below this limit will be retained.\nIf not provided, the training will proceed without early termination based on epsilon constraints.\n",
        gt=0.0,
        le=10000.0,
    )
    delta: float | None = Field(
        1e-05,
        description="The delta value for differential privacy. It is the probability of the privacy guarantee not holding.\nThe smaller the delta, the more confident you can be that the privacy guarantee holds.\nThis delta will be equally distributed between the analysis and the training phase.\n",
        gt=0.0,
        le=1.0,
    )
    noise_multiplier: float | None = Field(
        1.5,
        alias="noiseMultiplier",
        description="Determines how much noise while training the model with differential privacy. This is the ratio of the standard deviation of the Gaussian noise to the L2-sensitivity of the function to which the noise is added.\n",
        ge=0.0,
        le=10000.0,
    )
    max_grad_norm: float | None = Field(
        1.0,
        alias="maxGradNorm",
        description="Determines the maximum impact of a single sample on updating the model weights during training with differential privacy. This is the maximum norm of the per-sample gradients.\n",
        ge=0.0,
        le=10000.0,
    )
    value_protection_epsilon: float | None = Field(
        1.0,
        alias="valueProtectionEpsilon",
        description="The DP epsilon of the privacy budget for determining the value ranges, which are gathered prior to the model training during the analysis step. Only applicable if value protection is True.\nPrivacy budget will be equally distributed between the columns. For categorical we calculate noisy histograms and use a noisy threshold. For numeric and datetime we calculate bounds based on noisy histograms.\n",
        gt=0.0,
        le=10000.0,
    )


class Accuracy(CustomBaseModel):
    """
    Metrics regarding the accuracy of synthetic data, measured as the closeness of discretized lower dimensional
    marginal distributions.

    1. **Univariate Accuracy**: The accuracy of the univariate distributions for all target columns.
    2. **Bivariate Accuracy**: The accuracy of all pair-wise distributions for target columns, as well as for target
    columns with respect to the context columns.
    3. **Trivariate Accuracy**: The accuracy of all three-way distributions for target columns.
    4. **Coherence Accuracy**: The accuracy of the auto-correlation for all target columns.

    Accuracy is defined as 100% - [Total Variation Distance](https://en.wikipedia.org/wiki/Total_variation_distance_of_probability_measures) (TVD),
    whereas TVD is half the sum of the absolute differences of the relative frequencies of the corresponding
    distributions.

    These accuracies are calculated for all discretized univariate, and bivariate distributions. In case of sequential
    data, also for all coherence distributions. Overall metrics are then calculated as the average across these
    accuracies.

    All metrics can be compared against a theoretical maximum accuracy, which is calculated for a same-sized holdout.
    The accuracy metrics shall be as close as possible to the theoretical maximum, but not significantly higher, as
    this would indicate overfitting.

    """

    overall: float | None = Field(
        None,
        description="Overall accuracy of synthetic data, averaged across univariate, bivariate, and coherence.\n",
        ge=0.0,
        le=1.0,
    )
    univariate: float | None = Field(
        None, description="Average accuracy of discretized univariate distributions.\n", ge=0.0, le=1.0
    )
    bivariate: float | None = Field(
        None, description="Average accuracy of discretized bivariate distributions.\n", ge=0.0, le=1.0
    )
    trivariate: float | None = Field(
        None, description="Average accuracy of discretized trivariate distributions.\n", ge=0.0, le=1.0
    )
    coherence: float | None = Field(
        None,
        description="Average accuracy of discretized coherence distributions. Only applicable for sequential data.\n",
        ge=0.0,
        le=1.0,
    )
    overall_max: float | None = Field(
        None,
        alias="overallMax",
        description="Expected overall accuracy of a same-sized holdout. Serves as a reference for `overall`.\n",
        ge=0.0,
        le=1.0,
    )
    univariate_max: float | None = Field(
        None,
        alias="univariateMax",
        description="Expected univariate accuracy of a same-sized holdout. Serves as a reference for `univariate`.\n",
        ge=0.0,
        le=1.0,
    )
    bivariate_max: float | None = Field(
        None,
        alias="bivariateMax",
        description="Expected bivariate accuracy of a same-sized holdout. Serves as a reference for `bivariate`.\n",
        ge=0.0,
        le=1.0,
    )
    trivariate_max: float | None = Field(
        None,
        alias="trivariateMax",
        description="Expected trivariate accuracy of a same-sized holdout. Serves as a reference for `trivariate`.\n",
        ge=0.0,
        le=1.0,
    )
    coherence_max: float | None = Field(
        None,
        alias="coherenceMax",
        description="Expected coherence accuracy of a same-sized holdout. Serves as a reference for `coherence`.\n",
        ge=0.0,
        le=1.0,
    )


class Similarity(CustomBaseModel):
    """
    Metrics regarding the similarity of the full joint distributions of samples within an embedding space.

    1. **Cosine Similarity**: The cosine similarity between the centroids of synthetic and training samples.
    2. **Discriminator AUC**: The AUC of a discriminative model to distinguish between synthetic and training samples.

    The Model2Vec model [potion-base-8M](https://huggingface.co/minishlab/potion-base-8M) is
    used to compute the embeddings of a string-ified representation of individual records. In case of sequential data
    the records, that belong to the same group, are being concatenated. We then calculate the cosine similarity
    between the centroids of the provided datasets within the embedding space.

    Again, we expect the similarity metrics to be as close as possible to 1, but not significantly higher than what is
    measured for the holdout data, as this would again indicate overfitting.

    In addition, a discriminative ML model is trained to distinguish between training and synthetic samples. The
    ability of this model to distinguish between training and synthetic samples is measured by the AUC score. For
    synthetic data to be considered realistic, the AUC score should be close to 0.5, which indicates that the synthetic
    data is indistinguishable from the training data.

    """

    cosine_similarity_training_synthetic: float | None = Field(
        None,
        alias="cosineSimilarityTrainingSynthetic",
        description="Cosine similarity between training and synthetic centroids.",
        ge=-1.0,
        le=1.0,
    )
    cosine_similarity_training_holdout: float | None = Field(
        None,
        alias="cosineSimilarityTrainingHoldout",
        description="Cosine similarity between training and holdout centroids.\nServes as a reference for `cosine_similarity_training_synthetic`.\n",
        ge=-1.0,
        le=1.0,
    )
    discriminator_auc_training_synthetic: float | None = Field(
        None,
        alias="discriminatorAUCTrainingSynthetic",
        description="Cross-validated AUC of a discriminative model to distinguish between training and synthetic samples.",
        ge=0.0,
        le=1.0,
    )
    discriminator_auc_training_holdout: float | None = Field(
        None,
        alias="discriminatorAUCTrainingHoldout",
        description="Cross-validated AUC of a discriminative model to distinguish between training and holdout samples.\nServes as a reference for `discriminator_auc_training_synthetic`.\n",
        ge=0.0,
        le=1.0,
    )


class Distances(CustomBaseModel):
    """
    Metrics regarding the nearest neighbor distances between training, holdout, and synthetic samples in an numerically
    encoded space. Useful for assessing the novelty / privacy of synthetic data.

    The provided data is first down-sampled, so that the number of samples match across all datasets. Note, that for
    an optimal sensitivity of this privacy assessment it is recommended to use a 50/50 split between training and
    holdout data, and then generate synthetic data of the same size.

    The numerical encodings of these samples are then computed, and the nearest neighbor distances are calculated for each
    synthetic sample to the training and holdout samples. Based on these nearest neighbor distances the following
    metrics are calculated:
    - Identical Match Share (IMS): The share of synthetic samples that are identical to a training or holdout sample.
    - Distance to Closest Record (DCR): The average distance of synthetic to training or holdout samples.
    - Nearest Neighbor Distance Ratio (NNDR): The 10-th smallest ratio of the distance to nearest and second nearest neighbor.

    For privacy-safe synthetic data we expect to see about as many identical matches, and about the same distances
    for synthetic samples to training, as we see for synthetic samples to holdout.

    """

    ims_training: float | None = Field(
        None,
        alias="imsTraining",
        description="Share of synthetic samples that are identical to a training sample.",
        ge=0.0,
        le=1.0,
    )
    ims_holdout: float | None = Field(
        None,
        alias="imsHoldout",
        description="Share of synthetic samples that are identical to a holdout sample. Serves as a reference for `ims_training`.",
        ge=0.0,
        le=1.0,
    )
    ims_trn_hol: float | None = Field(
        None,
        alias="imsTrnHol",
        description="Share of training samples that are identical to a holdout sample. Serves as a reference for `ims_training`.",
        ge=0.0,
        le=1.0,
    )
    dcr_training: float | None = Field(
        None,
        alias="dcrTraining",
        description="Average nearest-neighbor distance between synthetic and training samples.",
        ge=0.0,
    )
    dcr_holdout: float | None = Field(
        None,
        alias="dcrHoldout",
        description="Average nearest-neighbor distance between synthetic and holdout samples. Serves as a reference for `dcr_training`.",
        ge=0.0,
    )
    dcr_trn_hol: float | None = Field(
        None,
        alias="dcrTrnHol",
        description="Average nearest-neighbor distance between training and holdout samples. Serves as a reference for `dcr_training`.",
        ge=0.0,
    )
    dcr_share: float | None = Field(
        None,
        alias="dcrShare",
        description="Share of synthetic samples that are closer to a training sample than to a holdout sample. This should not be significantly larger than 50%.",
        ge=0.0,
        le=1.0,
    )
    nndr_training: float | None = Field(
        None,
        alias="nndrTraining",
        description="10th smallest nearest-neighbor distance ratio between synthetic and training samples.",
        ge=0.0,
    )
    nndr_holdout: float | None = Field(
        None,
        alias="nndrHoldout",
        description="10th smallest nearest-neighbor distance ratio between synthetic and holdout samples.",
        ge=0.0,
    )
    nndr_trn_hol: float | None = Field(
        None,
        alias="nndrTrnHol",
        description="10th smallest nearest-neighbor distance ratio between training and holdout samples.",
        ge=0.0,
    )


class Metadata(CustomBaseModel):
    """
    The metadata of a resource.
    """

    creator_id: str | None = Field(None, alias="creatorId", description="The unique identifier of a user.")
    creator_name: str | None = Field(
        None,
        alias="creatorName",
        description="The name of a user.\nContains only alphanumeric characters, hyphens, and underscores. Must start or end with alphanumeric.\nIt must be globally case-insensitive unique considering organizations and users.\n",
    )
    created_at: AwareDatetime | None = Field(
        None,
        alias="createdAt",
        description="The UTC date and time when the resource has been created.",
        examples=["2023‐09‐07T18:40:39Z"],
    )


class ConnectorListItem(CustomBaseModel):
    """
    Essential connector details for listings.
    """

    id: str = Field(..., description="The unique identifier of a connector.")
    name: str | None = Field(None, description="The name of a connector.")
    description: str | None = Field(None, description="The description of a connector.")
    type: ConnectorType
    access_type: ConnectorAccessType | None = Field(ConnectorAccessType.read_protected, alias="accessType")
    metadata: Metadata | None = None
    OPEN_URL_PARTS: ClassVar[list] = ["d", "connectors"]

    def __getattr__(self, item):
        if item in {"update", "delete", "locations", "schema"}:

            def delegated_method(*args, **kwargs):
                connector = self.client.get(self.id)
                result = getattr(connector, item)(*args, **kwargs)
                if item == "update":
                    self.reload()
                return result

            return delegated_method
        return object.__getattribute__(self, item)


class Connector(CustomBaseModel):
    """
    A connector is a connection to a data source or a data destination.

    """

    id: str = Field(..., description="The unique identifier of a connector.")
    name: str | None = Field(None, description="The name of a connector.")
    description: str | None = Field(None, description="The description of a connector.")
    type: ConnectorType
    access_type: ConnectorAccessType | None = Field(ConnectorAccessType.read_protected, alias="accessType")
    config: dict[str, Any] | None = None
    secrets: Annotated[dict[str, str] | None, Field(repr=False)] = None
    ssl: Annotated[dict[str, str] | None, Field(repr=False)] = None
    metadata: Metadata | None = None
    table_id: str | None = Field(
        None,
        alias="tableId",
        description="Optional. ID of a source table or a synthetic table, that this connector belongs to.\nIf not set, then this connector is managed independently of any generator or synthetic dataset.\n",
    )
    OPEN_URL_PARTS: ClassVar[list] = ["d", "connectors"]

    @model_validator(mode="before")
    @classmethod
    def add_required_fields(cls, values):
        if isinstance(values, dict):
            if "id" not in values:
                values["id"] = str(uuid.uuid4())
            if values.get("name") is None:
                values["name"] = "New connector"
        return values

    def update(
        self,
        name: str | None = None,
        description: str | None = None,
        access_type: ConnectorAccessType | None = None,
        config: dict[str, Any] | None = None,
        secrets: Annotated[dict[str, str] | None, Field(repr=False)] = None,
        ssl: Annotated[dict[str, str] | None, Field(repr=False)] = None,
        test_connection: bool | None = True,
    ) -> None:
        """
        Update a connector with specific parameters.

        Args:
            name (str | None): The name of the connector.
            description (str | None): The description of the connector.
            access_type (ConnectorAccessType | None): The access type of the connector.
            config (dict[str, Any] | None): Connector configuration.
            secrets (dict[str, str] | None): Secret values for the connector.
            ssl (dict[str, str] | None): SSL configuration for the connector.
            test_connection (bool | None): If true, validates the connection before saving.
        """
        patch_config = ConnectorPatchConfig(
            name=name, description=description, access_type=access_type, config=config, secrets=secrets, ssl=ssl
        )
        self.client._update(connector_id=self.id, config=patch_config, test_connection=test_connection)
        self.reload()

    def delete(self) -> None:
        """
        Delete the connector.
        """
        return self.client._delete(connector_id=self.id)

    def locations(self, prefix: str = "") -> list[str]:
        """
        List connector locations.

        List the available databases, schemas, tables, or folders for a connector.
        For storage connectors, this returns list of buckets for empty prefix and otherwise the folders and files then within a path, specified via prefix.
        For DB connectors, this returns list of schemas (or databases for DBs without schema), respectively list of tables if `prefix` is provided.

        The formats of the locations are:

        - Cloud storage:
            - `AZURE_STORAGE`: `container/path`
            - `GOOGLE_CLOUD_STORAGE`: `bucket/path`
            - `S3_STORAGE`: `bucket/path`
        - Database:
            - `BIGQUERY`: `dataset.table`
            - `MARIADB`: `database.table`
            - `MSSQL`: `schema.table`
            - `MYSQL`: `database.table`
            - `ORACLE`: `schema.table`
            - `POSTGRES`: `schema.table`
            - `SNOWFLAKE`: `schema.table`

        Args:
            prefix (str): The prefix to filter the results by. Defaults to an empty string.

        Returns:
            list[str]: A list of locations (schemas, databases, directories, etc.).

        Example:
            ```python
            c.locations()  # list all schemas / databases for a DB connector; list all buckets for a storage connector
            c.locations('db_name')  # list all tables in 'db_name' for a DB connector
            c.locations('s3://my_bucket')  # list all objects in 'my_bucket' for a S3 storage connector
            c.locations('gs://my_bucket/path/to/folder')  # list all objects in 'my_bucket/path/to/folder' for a GCP storage connector
            c.locations('az://my_container/path/to/folder')  # list all objects in 'my_container/path/to/folder' for a AZURE storage connector
            ```
        """
        return self.client._locations(connector_id=self.id, prefix=prefix)

    def schema(self, location: str) -> list[dict[str, Any]]:
        """
        Retrieve the schema of the table at a connector location.
        This method is available for all connectors.

        Args:
            location (str): The location of the table.

        Returns:
            list[dict[str, Any]]: The retrieved schema.

        Example:
            ```python
            c.schema('db_name.table_name')  # get the schema of 'table_name' in 'db_name' for a DB connector
            c.schema('s3://my_bucket/path/to/file.csv')  # get the schema of 'file.csv' in 'my_bucket' for a S3 storage connector
            ```
        """
        return self.client._schema(connector_id=self.id, location=location)

    def read_data(self, location: str, limit: int | None = None, shuffle: bool = False) -> pd.DataFrame:
        """
        Retrieve data from the specified location within the connector.
        This method is only available for connectors of access_type READ_DATA or WRITE_DATA.

        Args:
            location (str): The target location within the connector to read data from.
            limit (int | None, optional): The maximum number of rows to return. Returns all if not specified.
            shuffle (bool | None, optional): Whether to shuffle the results.

        Returns:
            pd.DataFrame: A DataFrame containing the retrieved data.

        Example:
            ```python
            df = c.read_data('db_name.table_name', limit=100)  # fetch first 100 rows from 'table_name' in 'db_name' for a DB connector
            df = c.read_data('s3://my_bucket/path/to/file.csv')  # read all data from 'file.csv' in 'my_bucket' for a S3 storage connector
            ```
        """
        return self.client._read_data(connector_id=self.id, location=location, limit=limit, shuffle=shuffle)

    def write_data(
        self, data: pd.DataFrame | None, location: str, if_exists: Literal["append", "replace", "fail"] = "fail"
    ) -> None:
        """
        Write data to the specified location within the connector.
        This method is only available for connectors of access_type WRITE_DATA.

        Args:
            data (pd.DataFrame | None): The DataFrame to write, or None to delete the location.
            location (str): The target location within the connector to write data to.
            if_exists (Literal["append", "replace", "fail"]): The behavior if the target location already exists (append, replace, fail). Default is "fail".

        Example:
            ```python
            c.write_data(df, 'db_name.table_name', if_exists='fail')  # write data to 'table_name' in 'db_name' for a DB connector (if it doesn't exist)
            c.write_data(df, 's3://my_bucket/path/to/file.csv')  # write data to 'file.csv' in 'my_bucket' for a S3 storage connector
            ```
        """
        self.client._write_data(connector_id=self.id, data=data, location=location, if_exists=if_exists.upper())

    def delete_data(self, location: str) -> None:
        """
        Delete data from the specified location within the connector.
        This method is only available for connectors of access_type WRITE_DATA.

        Args:
            location (str): The target location within the connector to delete data from.

        Example:
            ```python
            c.delete_data('db_name.table_name')  # drop table data from 'table_name' in 'db_name' for a DB connector
            c.delete_data('s3://my_bucket/path/to/file.csv')  # delete data from 'file.csv' in 'my_bucket' for a S3 storage connector
            ```
        """
        self.client._delete_data(connector_id=self.id, location=location)

    def query(self, sql: str) -> pd.DataFrame:
        """
        Execute a read-only SQL query against the connector's data source.

        Queries can include statements like SELECT, SHOW, or DESCRIBE, but must not modify data or state.
        For file-based connectors (S3_STORAGE, GOOGLE_CLOUD_STORAGE, AZURE_STORAGE) queries are executed using DuckDB. Use connector-type-specific prefixes. See examples.

        Args:
            sql (str): The SQL query to execute.

        Returns:
            pd.DataFrame: The result of the query as a Pandas DataFrame.

        Example:
            ```python
            df = c.query("SELECT count(*) FROM schema.table")  # for DB connectors
            df = c.query("SELECT count(*) FROM read_csv_auto('s3://bucket/path/to/file.csv')")  # query a single CSV file from S3 storage
            df = c.query("SELECT count(*) FROM read_parquet('gs://bucket/path/to/folder/*.parquet')")  # query a folder with PQT files from GCP storage
            df = c.query("SELECT count(*) FROM read_json_auto('az://bucket/path/to/file.json')")  # query a single JSON file from AZURE storage
            ```
        """
        return self.client._query(connector_id=self.id, sql=sql)


class GeneratorListItem(CustomBaseModel):
    """
    Essential generator details for listings.
    """

    id: str = Field(..., description="The unique identifier of a generator.")
    name: str = Field(..., description="The name of a generator.")
    description: str | None = Field(None, description="The description of a generator.")
    training_status: ProgressStatus = Field(..., alias="trainingStatus")
    training_time: AwareDatetime | None = Field(
        None, alias="trainingTime", description="The UTC date and time when the training has finished."
    )
    metadata: Metadata | None = None
    accuracy: float | None = Field(
        None,
        description="The overall accuracy of the trained generator.\nThis is the average of the overall accuracy scores of all trained models.\n",
    )
    OPEN_URL_PARTS: ClassVar[list] = ["d", "generators"]

    def __getattr__(self, item):
        if item in {"update", "delete", "clone", "config", "export_to_file"}:

            def delegated_method(*args, **kwargs):
                generator = Generator(id=self.id, client=self.client)
                result = getattr(generator, item)(*args, **kwargs)
                if item == "update":
                    self.reload()
                return result

            return delegated_method
        return object.__getattribute__(self, item)


class Probe(CustomBaseModel):
    """
    The generated synthetic samples returned as a result of the probe.
    """

    name: str | None = Field(None, description="The name of the table.")
    rows: list[dict[str, Any]] | None = Field(None, description="An array of sample data objects.")


class SourceColumn(CustomBaseModel):
    """
    A column as part of a source table.
    """

    id: str = Field(..., description="The unique identifier of a source column.")
    name: str = Field(
        ...,
        description="The name of a source column. It must be unique within a source table.",
        max_length=256,
        min_length=1,
    )
    included: bool | None = Field(
        True,
        description="If true, the column will be included in the training.\nIf false, the column will be excluded from the training.\n",
    )
    model_encoding_type: ModelEncodingType = Field(..., alias="modelEncodingType")
    value_range: SourceColumnValueRange | None = Field(None, alias="valueRange")

    @model_validator(mode="before")
    @classmethod
    def add_required_fields(cls, values):
        if isinstance(values, dict):
            if "id" not in values:
                values["id"] = str(uuid.uuid4())
            if "model_encoding_type" not in values:
                values["model_encoding_type"] = ModelEncodingType.auto
            if "included" not in values:
                values["included"] = True
        return values


class GeneratorCloneConfig(CustomBaseModel):
    """
    The configuration for cloning a generator.
    """

    training_status: GeneratorCloneTrainingStatus | None = Field(
        GeneratorCloneTrainingStatus.new, alias="trainingStatus"
    )


class Constraint(CustomBaseModel):
    """
    A data constraint to apply.
    """

    id: str = Field(..., description="The unique identifier of a constraint.")
    type: ConstraintType
    config: dict[str, Any]

    @model_validator(mode="before")
    @classmethod
    def add_required_fields(cls, values):
        if isinstance(values, dict):
            if "id" not in values:
                values["id"] = str(uuid.uuid4())
        return values


class ConstraintConfig(CustomBaseModel):
    """
    A constraint to apply during data generation.
    """

    type: ConstraintType
    config: dict[str, Any]


class SourceColumnConfig(CustomBaseModel):
    """
    The configuration for a source column when creating a new generator.
    """

    name: str = Field(
        ...,
        description="The name of a source column. It must be unique within a source table.",
        max_length=256,
        min_length=1,
    )
    model_encoding_type: ModelEncodingType | None = Field(ModelEncodingType.auto, alias="modelEncodingType")


class ModelConfiguration(CustomBaseModel):
    """
    The training configuration for the model
    """

    model: str | None = Field(
        None,
        description="The model to be used for training.",
        examples=[
            [
                "MOSTLY_AI/Small",
                "MOSTLY_AI/Medium",
                "MOSTLY_AI/Large",
                "MOSTLY_AI/LSTMFromScratch-3m",
                "microsoft/phi-1_5",
            ]
        ],
    )
    max_sample_size: int | None = Field(
        None,
        alias="maxSampleSize",
        description="The maximum number of samples to consider for training.\nIf not provided, then all available samples will be taken.\n",
        ge=1,
    )
    batch_size: int | None = Field(
        None,
        alias="batchSize",
        description="The physical batch size used for training the model.\nIf not provided, batchSize will be chosen automatically.\n",
        ge=1,
    )
    gradient_accumulation_steps: int | None = Field(
        None,
        alias="gradientAccumulationSteps",
        description="Steps to accumulate gradients before optimizer update.\nIf not provided, gradientAccumulationSteps will be chosen automatically.\n",
        ge=1,
    )
    max_training_time: float | None = Field(
        14400, alias="maxTrainingTime", description="The maximum number of minutes to train the model.", ge=0.0
    )
    max_epochs: float | None = Field(
        100, alias="maxEpochs", description="The maximum number of epochs to train the model.", ge=0.0
    )
    max_sequence_window: int | None = Field(
        100,
        alias="maxSequenceWindow",
        description="The maximum sequence window to consider for training.\nOnly applicable for TABULAR models.\n",
        ge=1,
    )
    enable_flexible_generation: bool | None = Field(
        True,
        alias="enableFlexibleGeneration",
        description="If true, then the trained generator can be used for conditional simulation, rebalancing, imputation and fairness.\nIf none of these will be needed, then one can gain extra accuracy by disabling this feature. This will then result in a fixed\ncolumn order being fed into the training process, rather than a column order, that is randomly permuted for every batch.\n",
    )
    value_protection: bool | None = Field(
        True,
        alias="valueProtection",
        description="Defines if Rare Category, Extreme value, or Sequence length protection will be applied.\n",
    )
    rare_category_replacement_method: RareCategoryReplacementMethod | None = Field(
        RareCategoryReplacementMethod.constant,
        alias="rareCategoryReplacementMethod",
        description="Specifies how rare categories will be sampled.\nOnly applicable if value protection has been enabled.\n\n- `CONSTANT`: Replace rare categories by a constant `_RARE_` token.\n- `SAMPLE`: Replace rare categories by a sample from non-rare categories.\n",
    )
    differential_privacy: DifferentialPrivacyConfig | None = Field(None, alias="differentialPrivacy")
    compute: str | None = Field(
        None, description="The unique identifier of a compute resource. Not applicable for SDK."
    )
    enable_model_report: bool | None = Field(
        True, alias="enableModelReport", description="If false, then the Model report is not generated.\n"
    )

    @model_validator(mode="after")
    def validate_differential_privacy_config(self):
        if self.differential_privacy:
            if not self.value_protection:
                self.differential_privacy.value_protection_epsilon = None
            elif self.differential_privacy.value_protection_epsilon is None:
                self.differential_privacy.value_protection_epsilon = 1.0
        return self


class ProgressStep(CustomBaseModel):
    """
    The progress of a step.
    """

    id: str | None = Field(None, description="The unique identifier of the step.")
    model_label: str | None = Field(
        None,
        alias="modelLabel",
        description="The unique label for the model, consisting of table name and a suffix for the model type.\nThis will be empty for steps that are not related to a model.\n",
        examples=[["census:tabular", "census:language"]],
    )
    compute_name: str | None = Field(
        None, alias="computeName", description="The name of a compute resource.", min_length=1
    )
    restarts: int | None = Field(0, description="The number of previous restarts for the corresponding task.")
    task_type: TaskType | None = Field(None, alias="taskType")
    step_code: StepCode | None = Field(None, alias="stepCode")
    start_date: AwareDatetime | None = Field(
        None,
        alias="startDate",
        description="The UTC date and time when the job has started.\nIf the job has not started yet, then this is None.\n",
        examples=["2024-01-25T12:34:56Z"],
    )
    end_date: AwareDatetime | None = Field(
        None,
        alias="endDate",
        description="The UTC date and time when the job has ended.\nIf the job is still, then this is None.\n",
        examples=["2024-01-25T12:34:56Z"],
    )
    messages: list[dict[str, Any]] | None = None
    error_message: str | None = Field(None, alias="errorMessage")
    progress: ProgressValue | None = None
    status: ProgressStatus | None = None

    @model_validator(mode="after")
    def add_required_fields(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
        if self.status is None:
            self.status = ProgressStatus.new
        if self.compute_name is None:
            self.compute_name = "SDK"
        return self


class SyntheticDatasetListItem(CustomBaseModel):
    """
    Essential synthetic dataset details for listings.
    """

    id: str = Field(..., description="The unique identifier of a synthetic dataset.")
    metadata: Metadata | None = None
    name: str = Field(..., description="The name of a synthetic dataset.")
    description: str | None = Field(None, description="The description of a synthetic dataset.")
    generation_status: ProgressStatus = Field(..., alias="generationStatus")
    generation_time: AwareDatetime | None = Field(
        None, alias="generationTime", description="The UTC date and time when the generation has finished."
    )
    OPEN_URL_PARTS: ClassVar[list] = ["d", "synthetic-datasets"]

    def __getattr__(self, item):
        if item in {"update", "delete", "config", "download", "data"}:

            def delegated_method(*args, **kwargs):
                sd = self.client.get(self.id)
                result = getattr(sd, item)(*args, **kwargs)
                if item == "update":
                    self.reload()
                return result

            return delegated_method
        return object.__getattribute__(self, item)


class SyntheticTableConfiguration(CustomBaseModel):
    """
    The sample configuration for a synthetic table
    """

    sample_size: int | None = Field(
        None,
        alias="sampleSize",
        description="Number of generated samples. Only applicable for subject tables.\nIf neither size nor seed is provided, then the default behavior for Synthetic Datasets is to generate the\nsame size of samples as the original, and the default behavior for Synthetic Probes is to generate one\nsubject only.\n",
        ge=1,
    )
    sample_seed_connector_id: str | None = Field(
        None, alias="sampleSeedConnectorId", description="The connector id of the seed data for conditional simulation"
    )
    sample_seed_dict: str | None = Field(
        None,
        alias="sampleSeedDict",
        description="The base64-encoded string derived from a json line file containing the specified sample seed data.\nThis allows conditional live probing via non-python clients.\n",
    )
    sample_seed_data: str | None = Field(
        None,
        alias="sampleSeedData",
        description="The base64-encoded string derived from a Parquet file containing the specified sample seed data.\nThis allows conditional simulation as well as live probing via python clients.\n",
    )
    sampling_temperature: float | None = Field(
        1.0, alias="samplingTemperature", description="temperature for sampling", ge=0.0, le=2.0
    )
    sampling_top_p: float | None = Field(1.0, alias="samplingTopP", description="topP for sampling", ge=0.9, le=1.0)
    rebalancing: RebalancingConfig | None = None
    imputation: ImputationConfig | None = None
    fairness: FairnessConfig | None = None
    enable_data_report: bool | None = Field(
        True,
        alias="enableDataReport",
        description="If false, then the Data report is not generated.\nIf enableDataReport is set to false on generator, then enableDataReport is automatically set to false.\n",
    )

    @field_validator("sample_seed_dict", mode="before")
    @classmethod
    def convert_dict_before(cls, value):
        return convert_to_base64(value, format="jsonl") if isinstance(value, (dict, pd.DataFrame)) else value

    @field_validator("sample_seed_data", mode="before")
    @classmethod
    def convert_data_before(cls, value):
        return (
            convert_to_base64(value)
            if isinstance(value, pd.DataFrame)
            or (value.__class__.__name__ == "DataFrame" and value.__class__.__module__.startswith("pyspark.sql"))
            else value
        )

    @model_validator(mode="after")
    def add_required_fields(self):
        if self.sampling_temperature is None:
            self.sampling_temperature = 1.0
        if self.sampling_top_p is None:
            self.sampling_top_p = 1.0
        return self


class SyntheticDatasetPatchConfig(CustomBaseModel):
    """
    The configuration for updating a synthetic dataset.
    """

    name: str | None = Field(None, description="The name of a synthetic dataset.")
    description: str | None = Field(None, description="The description of a synthetic dataset.")
    delivery: SyntheticDatasetDelivery | None = None
    compute: str | None = Field(
        None, description="The unique identifier of a compute resource. Not applicable for SDK."
    )
    random_state: int | None = Field(
        None,
        alias="randomState",
        description="Seed for the random number generators. If None, the random number generator is initialized randomly, yielding different results for every run.\nSetting it to a specific integer ensures reproducible results across runs.\nUseful when consistent results are desired, e.g. for testing or debugging.\n",
    )


class SyntheticTableConfig(CustomBaseModel):
    """
    The configuration for a synthetic table when creating a new synthetic dataset.
    """

    name: str = Field(
        ..., description="The name of a synthetic table. This matches the name of a corresponding SourceTable."
    )
    configuration: SyntheticTableConfiguration | None = None

    @model_validator(mode="after")
    def add_configuration(self):
        if self.configuration is None:
            self.configuration = SyntheticTableConfiguration()
        return self

    def validate_against_source_table(self, source_table: SourceTable, is_probe: bool) -> None:
        self._maybe_set_sample_size(source_table, is_probe)
        _SyntheticTableConfigValidation(synthetic_table=self, source_table=source_table)

    def _maybe_set_sample_size(self, source_table: SourceTable, is_probe: bool) -> None:
        config = self.configuration
        is_subject = not any(fk.is_context for fk in source_table.foreign_keys or [])
        if (
            not config.sample_size
            and is_subject
            and (not (config.sample_seed_connector_id or config.sample_seed_dict or config.sample_seed_data))
        ):
            config.sample_size = 1 if is_probe else source_table.total_rows
        elif not is_subject:
            config.sample_size = None


class ModelMetrics(CustomBaseModel):
    """
    Metrics regarding the quality of synthetic data, measured in terms of accuracy, similarity, and distances.

    1. **Accuracy**: Metrics regarding the accuracy of synthetic data, measured as the closeness of discretized lower
    dimensional marginal distributions.
    2. **Similarity**: Metrics regarding the similarity of the full joint distributions of samples within an embedding
    space.
    3. **Distances**: Metrics regarding the nearest neighbor distances between training, holdout, and synthetic samples
    in an numeric encoding space. Useful for assessing the novelty / privacy of synthetic data.

    The quality of synthetic data is assessed by comparing these metrics to the same metrics of a holdout dataset.
    The holdout dataset is a subset of the original training data, that was not used for training the synthetic data
    generator. The metrics of the synthetic data should be as close as possible to the metrics of the holdout data.

    """

    accuracy: Accuracy | None = None
    distances: Distances | None = None
    similarity: Similarity | None = None


class SourceTable(CustomBaseModel):
    """
    A table as part of a generator.
    """

    id: str = Field(..., description="The unique identifier of a source table.")
    source_connector_id: str | None = Field(
        None, alias="sourceConnectorId", description="The unique identifier of a connector."
    )
    location: str | None = Field(
        None,
        description="The location of a source table. Together with the source connector it uniquely\nidentifies a source, and samples data from there.\n",
    )
    name: str = Field(
        ...,
        description="The name of a source table. It must be unique within a generator.",
        max_length=256,
        min_length=1,
    )
    primary_key: str | None = Field(None, alias="primaryKey", description="The column name of the primary key.")
    columns: list[SourceColumn] | None = Field(None, description="The columns of this generator table.")
    foreign_keys: list[SourceForeignKey] | None = Field(
        None, alias="foreignKeys", description="The foreign keys of a table."
    )
    tabular_model_metrics: ModelMetrics | None = Field(None, alias="tabularModelMetrics")
    language_model_metrics: ModelMetrics | None = Field(None, alias="languageModelMetrics")
    tabular_model_configuration: ModelConfiguration | None = Field(None, alias="tabularModelConfiguration")
    language_model_configuration: ModelConfiguration | None = Field(None, alias="languageModelConfiguration")
    total_rows: int | None = Field(
        None,
        alias="totalRows",
        description="The total number of rows in the source table while fetching data for training.\n",
    )

    @model_validator(mode="before")
    @classmethod
    def add_required_fields(cls, values):
        if isinstance(values, dict):
            if "id" not in values:
                values["id"] = str(uuid.uuid4())
        return values


class SourceTableConfig(CustomBaseModel):
    """
    The configuration for a source table when creating a new generator.
    """

    name: str = Field(
        ...,
        description="The name of a source table. It must be unique within a generator.",
        max_length=256,
        min_length=1,
    )
    source_connector_id: str | None = Field(
        None, alias="sourceConnectorId", description="The unique identifier of a connector."
    )
    location: str | None = Field(
        None,
        description="The location of a source table. Together with the source connector it uniquely\nidentifies a source, and samples data from there.\n",
    )
    data: str | None = Field(
        None,
        description="The base64-encoded string derived from a Parquet file containing the specified source table.\n",
    )
    tabular_model_configuration: ModelConfiguration | None = Field(None, alias="tabularModelConfiguration")
    language_model_configuration: ModelConfiguration | None = Field(None, alias="languageModelConfiguration")
    primary_key: str | None = Field(None, alias="primaryKey", description="The column name of the primary key.")
    foreign_keys: list[SourceForeignKeyConfig] | None = Field(
        None, alias="foreignKeys", description="The foreign key configurations of this table."
    )
    columns: list[SourceColumnConfig] | None = Field(None, description="The column configurations of this table.")

    @field_validator("data", mode="before")
    @classmethod
    def convert_data_before(cls, value):
        if isinstance(value, Path) or (isinstance(value, str) and len(value) > 0 and (len(value) < 512)):
            _, value = read_table_from_path(value)
        return (
            convert_to_base64(value)
            if isinstance(value, pd.DataFrame)
            or (value.__class__.__name__ == "DataFrame" and value.__class__.__module__.startswith("pyspark.sql"))
            else value
        )

    @field_validator("columns", mode="before")
    @classmethod
    def filter_excluded_columns(cls, columns):
        if columns is None or not isinstance(columns, list):
            return columns
        included_columns = []
        for column in columns:
            is_included = any(
                (
                    isinstance(column, dict) and bool(column.get("included", True)),
                    isinstance(column, SourceColumn) and column.included,
                    isinstance(column, SourceColumnConfig),
                )
            )
            if is_included:
                included_columns.append(column)
        return included_columns

    @model_validator(mode="after")
    def add_model_configuration(self):
        keys = [fk.column for fk in self.foreign_keys or []]
        if self.primary_key:
            keys.append(self.primary_key)
        model_columns = [c for c in self.columns if c.name not in keys] if self.columns is not None else None
        if model_columns is None:
            has_tabular_model = True
            has_language_model = True
        elif len(model_columns) == 0:
            has_tabular_model = True
            has_language_model = False
        else:
            enc_types = [c.model_encoding_type or ModelEncodingType.auto for c in model_columns]
            has_tabular_model = any(
                enc_type.startswith(ModelType.tabular) or enc_type == ModelEncodingType.auto for enc_type in enc_types
            )
            has_language_model = any(
                enc_type.startswith(ModelType.language) or enc_type == ModelEncodingType.auto for enc_type in enc_types
            )
        if self.primary_key or (self.foreign_keys and any(fk.is_context for fk in self.foreign_keys)):
            has_tabular_model = True
        if self.tabular_model_configuration and (not has_tabular_model):
            self.tabular_model_configuration = None
        if self.language_model_configuration and (not has_language_model):
            self.language_model_configuration = None
        if has_tabular_model:
            default_model = "MOSTLY_AI/Medium"
            if not self.tabular_model_configuration:
                self.tabular_model_configuration = ModelConfiguration(model=default_model)
            elif not self.tabular_model_configuration.model:
                self.tabular_model_configuration.model = default_model
        if has_language_model:
            default_model = "MOSTLY_AI/LSTMFromScratch-3m"
            if not self.language_model_configuration:
                self.language_model_configuration = ModelConfiguration(model=default_model, max_sequence_window=None)
            elif not self.language_model_configuration.model:
                self.language_model_configuration.model = default_model
            self.language_model_configuration.max_sequence_window = None
        return self

    @model_validator(mode="after")
    def extract_columns_from_data_if_missing(self):
        """extract column names from base64 data if columns are not provided."""
        if self.columns is None and self.data is not None and (len(self.data) >= 512):
            try:
                df = convert_to_df(self.data, format="parquet")
                self.columns = [
                    SourceColumnConfig(name=col_name, model_encoding_type=ModelEncodingType.auto)
                    for col_name in df.columns
                ]
            except Exception:
                pass
        return self

    @field_validator("columns", mode="after")
    @classmethod
    def validate_unique_columns(cls, columns):
        if columns:
            defined_columns = [c.name for c in columns]
            if len(defined_columns) != len(set(defined_columns)):
                raise ValueError("Column names must be unique.")
        return columns

    @model_validator(mode="after")
    def validate_unique_keys(self):
        pk = self.primary_key or []
        fks = [fk.column for fk in self.foreign_keys or []]
        if len(fks) != len(set(fks)):
            raise ValueError("Foreign key column names must be unique.")
        if pk in fks:
            raise ValueError("Primary key column name must not be defined as foreign key.")
        return self

    @field_validator("foreign_keys", mode="after")
    @classmethod
    def validate_at_most_one_context_fk(cls, foreign_keys):
        if foreign_keys:
            context_fks = [fk for fk in foreign_keys if fk.is_context]
            if len(context_fks) > 1:
                raise ValueError("At most one context foreign key is allowed")
        return foreign_keys

    @model_validator(mode="after")
    def validate_keys_exists_in_columns(self):
        if self.columns is not None:
            column_names = {col.name for col in self.columns}
            pk = self.primary_key
            if pk and pk not in column_names:
                raise ValueError(f"Primary key column '{pk}' does not exist in the table's columns.")
            for fk in self.foreign_keys or []:
                if fk.column not in column_names:
                    raise ValueError(f"Foreign key column '{fk.column}' does not exist in the table's columns.")
        return self

    @model_validator(mode="after")
    def validate_pk_and_fks_are_not_overlapping(self):
        primary_key = self.primary_key
        foreign_keys = [fk.column for fk in self.foreign_keys or []]
        if primary_key and primary_key in foreign_keys:
            raise ValueError(f"Column '{primary_key}' is both a primary key and a foreign key.")
        return self


class JobProgress(CustomBaseModel):
    """
    The progress of a job.
    """

    id: str | None = None
    start_date: AwareDatetime | None = Field(
        None,
        alias="startDate",
        description="The UTC date and time when the job has started.\nIf the job has not started yet, then this is None.\n",
        examples=["2024-01-25T12:34:56Z"],
    )
    end_date: AwareDatetime | None = Field(
        None,
        alias="endDate",
        description="The UTC date and time when the job has ended.\nIf the job is still, then this is None.\n",
        examples=["2024-01-25T12:34:56Z"],
    )
    progress: ProgressValue | None = None
    status: ProgressStatus | None = None
    steps: list[ProgressStep] | None = None


class SyntheticTable(CustomBaseModel):
    """
    A synthetic table that will be generated.
    """

    id: str | None = Field(None, description="The unique identifier of a synthetic table.")
    name: str = Field(
        ...,
        description="The name of a source table. It must be unique within a generator.",
        max_length=256,
        min_length=1,
    )
    configuration: SyntheticTableConfiguration | None = None
    tabular_model_metrics: ModelMetrics | None = Field(None, alias="tabularModelMetrics")
    language_model_metrics: ModelMetrics | None = Field(None, alias="languageModelMetrics")
    foreign_keys: list[SourceForeignKey] | None = Field(
        None, alias="foreignKeys", description="The foreign keys of a table."
    )
    total_rows: int | None = Field(
        None,
        alias="totalRows",
        description="The total number of rows for that table in the generated synthetic dataset.\n",
    )
    total_datapoints: int | None = Field(
        None,
        alias="totalDatapoints",
        deprecated=True,
        description="**Deprecated:** This field is no longer valid and will always return `-1`. It will be removed in a future version.\n",
    )
    source_table_total_rows: int | None = Field(
        None,
        alias="sourceTableTotalRows",
        description="The total number of rows in the source table while fetching data for training.\n",
    )

    @model_validator(mode="before")
    @classmethod
    def add_required_fields(cls, values):
        if isinstance(values, dict):
            if "id" not in values:
                values["id"] = str(uuid.uuid4())
        return values


class SyntheticDatasetConfig(CustomBaseModel):
    """
    The configuration for creating a new synthetic dataset.
    """

    generator_id: str | None = Field(None, alias="generatorId", description="The unique identifier of a generator.")
    name: str | None = Field(None, description="The name of a synthetic dataset.")
    description: str | None = Field(None, description="The description of a synthetic dataset.")
    random_state: int | None = Field(
        None,
        alias="randomState",
        description="Seed for the random number generators. If None, the random number generator is initialized randomly, yielding different results for every run.\nSetting it to a specific integer ensures reproducible results across runs.\nUseful when consistent results are desired, e.g. for testing or debugging.\n",
    )
    tables: list[SyntheticTableConfig] | None = None
    delivery: SyntheticDatasetDelivery | None = None
    compute: str | None = Field(
        None, description="The unique identifier of a compute resource. Not applicable for SDK."
    )

    @field_validator("tables", mode="after")
    @classmethod
    def validate_unique_table_names(cls, tables):
        if not tables:
            return tables
        defined_tables = [t.name for t in tables]
        if len(defined_tables) != len(set(defined_tables)):
            raise ValueError("Table names must be unique.")
        return tables

    def validate_against_generator(self, generator: Generator) -> None:
        _SyntheticDataConfigValidation(synthetic_config=self, generator=generator)


class SyntheticProbeConfig(CustomBaseModel):
    """
    The configuration for probing for new synthetic samples.
    """

    generator_id: str | None = Field(None, alias="generatorId", description="The unique identifier of a generator.")
    random_state: int | None = Field(
        None,
        alias="randomState",
        description="Seed for the random number generators. If None, the random number generator is initialized randomly, yielding different results for every run.\nSetting it to a specific integer ensures reproducible results across runs.\nUseful when consistent results are desired, e.g. for testing or debugging.\n",
    )
    tables: list[SyntheticTableConfig] | None = None

    @field_validator("tables", mode="after")
    @classmethod
    def validate_unique_table_names(cls, tables):
        if not tables:
            return tables
        defined_tables = [t.name for t in tables]
        if len(defined_tables) != len(set(defined_tables)):
            raise ValueError("Table names must be unique.")
        return tables

    def validate_against_generator(self, generator: Generator) -> None:
        _SyntheticDataConfigValidation(synthetic_config=self, generator=generator)


class Generator(CustomBaseModel):
    """
    A generator is a set models that can generate synthetic data.

    The generator can be trained on one or more source tables. A quality assurance report is generated for each model.

    """

    id: str = Field(..., description="The unique identifier of a generator.")
    name: str = Field(..., description="The name of a generator.")
    description: str | None = Field(None, description="The description of a generator.")
    training_status: ProgressStatus = Field(..., alias="trainingStatus")
    training_time: AwareDatetime | None = Field(
        None, alias="trainingTime", description="The UTC date and time when the training has finished."
    )
    metadata: Metadata | None = None
    accuracy: float | None = Field(
        None,
        description="The overall accuracy of the trained generator.\nThis is the average of the overall accuracy scores of all trained models.\n",
    )
    tables: list[SourceTable] | None = Field(None, description="The tables of this generator")
    constraints: list[Constraint] | None = Field(None, description="The data constraints to apply.")
    random_state: int | None = Field(
        None,
        alias="randomState",
        description="Seed for the random number generators. If None, the random number generator is initialized randomly, yielding different results for every run.\nSetting it to a specific integer ensures reproducible results across runs.\nUseful when consistent results are desired, e.g. for testing or debugging.\n",
    )
    OPEN_URL_PARTS: ClassVar[list] = ["d", "generators"]
    training: Annotated[Any | None, Field(exclude=True)] = None

    @model_validator(mode="before")
    @classmethod
    def add_required_fields(cls, values):
        if isinstance(values, dict):
            if "id" not in values:
                values["id"] = str(uuid.uuid4())
            if values.get("name") is None:
                values["name"] = "New generator"
            if "training_status" not in values:
                values["training_status"] = ProgressStatus.new
        return values

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training = self.Training(self)

    def update(self, name: str | None = None, description: str | None = None) -> None:
        """
        Update a generator with specific parameters.

        Args:
            name (str | None): The name of the generator.
            description (str | None): The description of the generator.
        """
        patch_config = GeneratorPatchConfig(name=name, description=description)
        self.client._update(generator_id=self.id, config=patch_config)
        self.reload()

    def delete(self) -> None:
        """
        Delete the generator.
        """
        return self.client._delete(generator_id=self.id)

    def config(self) -> GeneratorConfig:
        """
        Retrieve writable generator properties.

        Returns:
            GeneratorConfig: The generator properties as a configuration object.
        """
        return self.client._config(generator_id=self.id)

    def export_to_file(self, file_path: str | Path | None = None) -> Path:
        """
        Export generator and save to file.

        Args:
            file_path (str | Path | None): The file path to save the generator.

        Returns:
            Path: The path to the saved file.
        """
        bytes, filename = self.client._export_to_file(generator_id=self.id)
        file_path = Path(file_path or ".")
        if file_path.is_dir():
            file_path = file_path / filename
        file_path.write_bytes(bytes)
        return file_path

    def clone(self, training_status: Literal["new", "continue"] = "new") -> Generator:
        """
        Clone the generator.

        Args:
            training_status (Literal["new", "continue"]): The training status of the cloned generator. Default is "new".

        Returns:
            Generator: The cloned generator object.
        """
        generator = self.client._clone(generator_id=self.id, training_status=training_status)
        gid = generator.id
        if self.client.local:
            rich.print(f"Created generator [dodger_blue2]{gid}[/]")
        else:
            rich.print(
                f"Created generator [link={self.client.base_url}/d/generators/{gid} dodger_blue2 underline]{gid}[/]"
            )
        return generator

    def reports(self, file_path: str | Path | None = None, display: bool = False) -> Path | None:
        """
        Download or display the quality assurance reports.

        If display is True, the report is rendered inline via IPython display and no file is downloaded.
        Otherwise, the report is downloaded and saved to file_path (or a default location if None).

        Note that reports are not available for generators that were trained with less than 100 samples or had `enable_model_report` set to `False`.

        Args:
            file_path (str | Path | None): The file path to save the zipped reports (ignored if display=True).
            display (bool): If True, render the report inline instead of downloading it.

        Returns:
            Path | None: The path to the saved file if downloading, or None if display=True.
        """
        reports = {}
        for table in self.tables:
            if table.tabular_model_metrics:
                reports[f"{table.name}-tabular.html"] = self.client._report(
                    generator_id=self.id, source_table_id=table.id, model_type="TABULAR"
                )
            if table.language_model_metrics:
                reports[f"{table.name}-language.html"] = self.client._report(
                    generator_id=self.id, source_table_id=table.id, model_type="LANGUAGE"
                )
        if display and rich.console._is_jupyter():
            import html

            from IPython.display import HTML, display

            iframes = ""
            for content in reports.values():
                content = html.escape(content, quote=True)
                iframes += f'<p><iframe srcdoc="{content}" width="100%" height="600"></iframe></p> '
            display(HTML(iframes))
            return None
        else:
            file_path = Path(file_path or ".")
            if file_path.is_dir():
                file_path = file_path / f"generator-{self.id[:8]}-reports.zip"
            if file_path.exists():
                file_path.unlink()
            with zipfile.ZipFile(file_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
                for filename, content in reports.items():
                    zip_file.writestr(filename, content)
            if display:
                rich.print(f"Reports saved to {file_path}")
            return file_path

    class Training:
        def __init__(self, _generator: Generator):
            self.generator = _generator

        def start(self) -> None:
            """
            Start training.
            """
            rich.print("Started generator training")
            self.generator.client._training_start(self.generator.id)

        def cancel(self) -> None:
            """
            Cancel training.
            """
            self.generator.client._training_cancel(self.generator.id)
            self.generator.reload()

        def progress(self) -> JobProgress:
            """
            Retrieve job progress of training.

            Returns:
                JobProgress: The job progress of the training process.
            """
            return self.generator.client._training_progress(self.generator.id)

        def wait(self, progress_bar: bool = True, interval: float = 2) -> None:
            """
            Poll training progress and loop until training has completed.

            Args:
                progress_bar (bool): If true, displays the progress bar. Default is True.
                interval (float): The interval in seconds to poll the job progress. Default is 2 seconds.
            """
            self.generator.client._training_wait(self.generator.id, progress_bar=progress_bar, interval=interval)
            self.generator.reload()
            if self.generator.training_status == ProgressStatus.done:
                rich.print(
                    ":tada: [bold green]Your generator is ready![/] Use it to create synthetic data. Publish it so others can do the same."
                )

        def logs(self, file_path: str | Path | None = None) -> Path:
            """
            Download the training logs and save to file.

            Args:
                file_path (str | Path | None): The file path to save the logs. Default is the current working directory.

            Returns:
                Path: The path to the saved file.
            """
            bytes, filename = self.generator.client._training_logs(generator_id=self.generator.id)
            file_path = Path(file_path or ".")
            if file_path.is_dir():
                file_path = file_path / filename
            file_path.write_bytes(bytes)
            return file_path


class GeneratorConfig(CustomBaseModel):
    """
    The configuration for creating a new generator.
    """

    name: str | None = Field(None, description="The name of a generator.")
    description: str | None = Field(None, description="The description of a generator.")
    random_state: int | None = Field(
        None,
        alias="randomState",
        description="Seed for the random number generators. If None, the random number generator is initialized randomly, yielding different results for every run.\nSetting it to a specific integer ensures reproducible results across runs.\nUseful when consistent results are desired, e.g. for testing or debugging.\n",
    )
    tables: list[SourceTableConfig] | None = Field(None, description="The tables of a generator")
    constraints: list[ConstraintConfig] | None = Field(None, description="The data constraints to apply.")

    @field_validator("tables", mode="after")
    @classmethod
    def validate_unique_table_names(cls, tables):
        defined_tables = [t.name for t in tables or []]
        if len(defined_tables) != len(set(defined_tables)):
            raise ValueError("Table names must be unique.")
        return tables

    @field_validator("tables", mode="after")
    @classmethod
    def validate_each_referenced_table_exist_and_has_primary_key(cls, tables):
        table_map = {table.name: table for table in tables or []}
        for table in tables or []:
            for fk in table.foreign_keys or []:
                ref_table = table_map.get(fk.referenced_table)
                if not ref_table:
                    raise ValueError(
                        f"Foreign key in table '{table.name}' references a non-existent table: '{fk.referenced_table}'."
                    )
                if not ref_table.primary_key:
                    raise ValueError(f"Referenced table '{fk.referenced_table}' does not have a primary key.")
        return tables

    @field_validator("tables", mode="after")
    @classmethod
    def validate_no_circular_context_references(cls, tables):
        if not tables:
            return tables
        table_map = {table.name: table for table in tables}
        visited = set()
        for table in tables:
            if table.name in visited:
                continue
            current_table = table
            seen_tables = set()
            while current_table:
                if current_table.name in seen_tables:
                    raise ValueError(f"Circular reference detected in tables: {', '.join(seen_tables)}")
                seen_tables.add(current_table.name)
                context_fk = next((fk for fk in current_table.foreign_keys or [] if fk.is_context), None)
                if not context_fk or not context_fk.referenced_table:
                    break
                current_table = table_map.get(context_fk.referenced_table)
            visited.update(seen_tables)
        return tables

    @model_validator(mode="after")
    def validate_constraints(self):
        """validate that constraints reference existing tables and columns."""
        if not self.constraints or not self.tables:
            return self
        table_map = {table.name: table for table in self.tables}
        column_usage: dict[str, dict[str, int]] = {}
        for idx, constraint_config in enumerate(self.constraints):
            typed_constraint = convert_constraint_config_to_typed(constraint_config)
            if typed_constraint.table_name not in table_map:
                raise ValueError(f"table '{typed_constraint.table_name}' referenced by constraint not found")
            table = table_map[typed_constraint.table_name]
            table_columns = {col.name: col for col in table.columns or []}
            if not table_columns:
                continue
            typed_constraint.validate(table_columns, column_usage, idx)
        return self

    def _track_column_usage(self, table_name, col_name, constraint_idx, column_usage):
        """track column usage and detect overlaps."""
        if table_name not in column_usage:
            column_usage[table_name] = {}
        if col_name in column_usage[table_name]:
            raise ValueError(f"column '{col_name}' in table '{table_name}' is referenced by multiple constraints")
        column_usage[table_name][col_name] = constraint_idx


class SyntheticDataset(CustomBaseModel):
    """
    A synthetic dataset is created based on a trained generator.

    It consists of synthetic samples, as well as a quality assurance report.

    """

    id: str = Field(..., description="The unique identifier of a synthetic dataset.")
    generator_id: str | None = Field(None, alias="generatorId", description="The unique identifier of a generator.")
    metadata: Metadata | None = None
    name: str = Field(..., description="The name of a synthetic dataset.")
    description: str | None = Field(None, description="The description of a synthetic dataset.")
    generation_status: ProgressStatus = Field(..., alias="generationStatus")
    generation_time: AwareDatetime | None = Field(
        None, alias="generationTime", description="The UTC date and time when the generation has finished."
    )
    tables: list[SyntheticTable] | None = Field(None, description="The tables of this synthetic dataset.")
    delivery: SyntheticDatasetDelivery | None = None
    accuracy: float | None = Field(
        None,
        description="The overall accuracy of the trained generator.\nThis is the average of the overall accuracy scores of all trained models.\n",
    )
    compute: str | None = Field(
        None, description="The unique identifier of a compute resource. Not applicable for SDK."
    )
    random_state: int | None = Field(
        None,
        alias="randomState",
        description="Seed for the random number generators. If None, the random number generator is initialized randomly, yielding different results for every run.\nSetting it to a specific integer ensures reproducible results across runs.\nUseful when consistent results are desired, e.g. for testing or debugging.\n",
    )
    OPEN_URL_PARTS: ClassVar[list] = ["d", "synthetic-datasets"]
    generation: Annotated[Any | None, Field(exclude=True)] = None

    @model_validator(mode="before")
    @classmethod
    def add_required_fields(cls, values):
        if isinstance(values, dict):
            if "id" not in values:
                values["id"] = str(uuid.uuid4())
            if values.get("name") is None:
                values["name"] = "New synthetic dataset"
            if "generation_status" not in values:
                values["generation_status"] = ProgressStatus.new
        return values

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generation = self.Generation(self)

    def update(
        self, name: str | None = None, description: str | None = None, delivery: SyntheticDatasetDelivery | None = None
    ) -> None:
        """
        Update a synthetic dataset with specific parameters.

        Args:
            name (str | None): The name of the synthetic dataset.
            description (str | None): The description of the synthetic dataset.
            delivery (SyntheticDatasetDelivery | None): The delivery configuration for the synthetic dataset.
        """
        patch_config = SyntheticDatasetPatchConfig(name=name, description=description, delivery=delivery)
        self.client._update(synthetic_dataset_id=self.id, config=patch_config)
        self.reload()

    def delete(self) -> None:
        """
        Delete the synthetic dataset.
        """
        return self.client._delete(synthetic_dataset_id=self.id)

    def config(self) -> SyntheticDatasetConfig:
        """
        Retrieve writable synthetic dataset properties.

        Returns:
            SyntheticDatasetConfig: The synthetic dataset properties as a configuration object.
        """
        return self.client._config(synthetic_dataset_id=self.id)

    def download(
        self, file_path: str | Path | None = None, format: Literal["parquet", "csv", "json"] = "parquet"
    ) -> Path:
        """
        Download synthetic dataset and save to file.

        Args:
            file_path (str | Path | None): The file path to save the synthetic dataset.
            format (Literal["parquet", "csv", "json"]): The format of the synthetic dataset. Default is "parquet".

        Returns:
            Path: The path to the saved file.
        """
        bytes, filename = self.client._download(synthetic_dataset_id=self.id, ds_format=format.upper())
        file_path = Path(file_path or ".")
        if file_path.is_dir():
            file_path = file_path / filename
        file_path.write_bytes(bytes)
        return file_path

    def data(self, return_type: Literal["auto", "dict"] = "auto") -> pd.DataFrame | dict[str, pd.DataFrame]:
        """
        Download synthetic dataset and return as dictionary of pandas DataFrames.

        Args:
            return_type (Literal["auto", "dict"]): The type of the return value. "dict" will always provide a dictionary of DataFrames. "auto" will return a single DataFrame for a single-table generator, and a dictionary of DataFrames for a multi-table generator. Default is "auto".

        Returns:
            Union[pd.DataFrame, dict[str, pd.DataFrame]]: The synthetic dataset. See return_type for the format of the return value.
        """
        dfs = self.client._data(synthetic_dataset_id=self.id)
        if return_type == "auto" and len(dfs) == 1:
            return list(dfs.values())[0]
        else:
            return dfs

    def reports(self, file_path: str | Path | None = None, display: bool = False) -> Path | None:
        """
        Download or display the quality assurance reports.

        If display is True, the report is rendered inline via IPython display and no file is downloaded.
        Otherwise, the report is downloaded and saved to file_path (or a default location if None).

        Note that reports are not available for synthetic datasets that generated less than 100 samples or had `enable_data_report` set to `False`.

        Args:
            file_path (str | Path | None): The file path to save the zipped reports (ignored if display=True).
            display (bool): If True, render the report inline instead of downloading it.

        Returns:
            Path | None: The path to the saved file if downloading, or None if display=True.
        """
        reports = {}
        for report_type in [SyntheticDatasetReportType.model, SyntheticDatasetReportType.data]:
            report_infix = "" if report_type == SyntheticDatasetReportType.model else "-data"
            for table in self.tables:
                if table.tabular_model_metrics:
                    reports[f"{table.name}-tabular{report_infix}.html"] = self.client._report(
                        synthetic_dataset_id=self.id,
                        synthetic_table_id=table.id,
                        model_type="TABULAR",
                        report_type=report_type,
                    )
                if table.language_model_metrics:
                    reports[f"{table.name}-language{report_infix}.html"] = self.client._report(
                        synthetic_dataset_id=self.id,
                        synthetic_table_id=table.id,
                        model_type="LANGUAGE",
                        report_type=report_type,
                    )
        if display and rich.console._is_jupyter():
            import html

            from IPython.display import HTML, display

            iframes = ""
            for content in reports.values():
                content = html.escape(content, quote=True)
                iframes += f'<p><iframe srcdoc="{content}" width="100%" height="600"></iframe></p> '
            display(HTML(iframes))
            return None
        else:
            file_path = Path(file_path or ".")
            if file_path.is_dir():
                file_path = file_path / f"synthetic-dataset-{self.id[:8]}-reports.zip"
            if file_path.exists():
                file_path.unlink()
            with zipfile.ZipFile(file_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
                for filename, content in reports.items():
                    zip_file.writestr(filename, content)
            if display:
                rich.print(f"Reports saved to {file_path}")
            return file_path

    class Generation:
        def __init__(self, _synthetic_dataset: SyntheticDataset):
            self.synthetic_dataset = _synthetic_dataset

        def start(self) -> None:
            """
            Start the generation process.
            """
            self.synthetic_dataset.client._generation_start(self.synthetic_dataset.id)
            rich.print("Started synthetic dataset generation")

        def cancel(self) -> None:
            """
            Cancel the generation process.
            """
            self.synthetic_dataset.client._generation_cancel(self.synthetic_dataset.id)
            self.synthetic_dataset.reload()

        def progress(self) -> JobProgress:
            """
            Retrieve the progress of the generation process.

            Returns:
                JobProgress: The progress of the generation process.
            """
            return self.synthetic_dataset.client._generation_progress(self.synthetic_dataset.id)

        def wait(self, progress_bar: bool = True, interval: float = 2) -> None:
            """
            Poll the generation progress and wait until the process is complete.

            Args:
                progress_bar (bool): If true, displays a progress bar. Default is True.
                interval (float): Interval in seconds to poll the job progress. Default is 2 seconds.
            """
            self.synthetic_dataset.client._generation_wait(
                self.synthetic_dataset.id, progress_bar=progress_bar, interval=interval
            )
            self.synthetic_dataset.reload()
            if self.synthetic_dataset.generation_status == ProgressStatus.done:
                rich.print(
                    ":tada: [bold green]Your synthetic dataset is ready![/] Use it to consume the generated data. Publish it so others can do the same."
                )

        def logs(self, file_path: str | Path | None = None) -> Path:
            """
            Download the generation logs and save to file.

            Args:
                file_path (str | Path | None): The file path to save the logs. Default is the current working directory.

            Returns:
                Path: The path to the saved file.
            """
            bytes, filename = self.synthetic_dataset.client._generation_logs(
                synthetic_dataset_id=self.synthetic_dataset.id
            )
            file_path = Path(file_path or ".")
            if file_path.is_dir():
                file_path = file_path / filename
            file_path.write_bytes(bytes)
            return file_path


class _SyntheticTableConfigValidation(CustomBaseModel):
    """
    Validation logic for SyntheticTableConfig against SourceTable
    """

    synthetic_table: SyntheticTableConfig
    source_table: SourceTable

    @model_validator(mode="after")
    def validate_rebalancing_config(self):
        config = self.synthetic_table.configuration
        if config and config.rebalancing:
            rebalancing_column = config.rebalancing.column
            rebalancing_col = next(
                (col for col in self.source_table.columns or [] if col.name == rebalancing_column), None
            )
            if not rebalancing_col:
                raise ValueError(
                    f"Rebalancing column '{rebalancing_column}' not found in table '{self.source_table.name}'"
                )
            if not rebalancing_col.model_encoding_type == ModelEncodingType.tabular_categorical:
                raise ValueError(
                    f"Rebalancing column '{rebalancing_column}' in table '{self.source_table.name}' must be categorical"
                )
            if not rebalancing_col.included:
                raise ValueError(
                    f"Rebalancing column '{rebalancing_column}' in table '{self.source_table.name}' must have `included=True`"
                )
            for value in config.rebalancing.probabilities.keys():
                if value not in rebalancing_col.value_range.values:
                    raise ValueError(
                        f"Rebalancing value '{value}' not found in table '{self.source_table.name}' column '{rebalancing_column}'"
                    )
        return self

    @model_validator(mode="after")
    def validate_imputation_config(self):
        config = self.synthetic_table.configuration
        if config and config.imputation:
            has_tabular_model = self.source_table.tabular_model_configuration is not None
            if not has_tabular_model:
                raise ValueError(f"Table '{self.source_table.name}' specifies imputation but has no tabular model")
            for col in config.imputation.columns:
                if not any(gcol.name == col for gcol in self.source_table.columns or []):
                    raise ValueError(f"Imputation column '{col}' not found in table '{self.source_table.name}'")
        return self

    @model_validator(mode="after")
    def validate_fairness_config(self):
        config = self.synthetic_table.configuration
        if config and config.fairness:
            has_tabular_model = self.source_table.tabular_model_configuration is not None
            if not has_tabular_model:
                raise ValueError(f"Table '{self.source_table.name}' specifies fairness but has no tabular model")
            target_col = config.fairness.target_column
            if not any(col.name == target_col for col in self.source_table.columns or []):
                raise ValueError(f"Fairness target column '{target_col}' not found in table '{self.source_table.name}'")
            for col in config.fairness.sensitive_columns:
                if not any(gcol.name == col for gcol in self.source_table.columns or []):
                    raise ValueError(f"Fairness sensitive column '{col}' not found in table '{self.source_table.name}'")
            if target_col in config.fairness.sensitive_columns:
                raise ValueError(f"Target column '{target_col}' cannot be a sensitive column")
        return self

    @model_validator(mode="after")
    def validate_data_report_disabled_if_both_model_reports_disabled(self):
        configs = [
            cfg
            for cfg in [self.source_table.tabular_model_configuration, self.source_table.language_model_configuration]
            if cfg
        ]
        if all(cfg.enable_model_report is False for cfg in configs):
            if self.synthetic_table.configuration is not None:
                self.synthetic_table.configuration.enable_data_report = False
        return self


class _SyntheticDataConfigValidation(CustomBaseModel):
    """
    Validation logic for SyntheticDatasetConfig and SyntheticProbeConfig against Generator
    """

    synthetic_config: SyntheticDatasetConfig | SyntheticProbeConfig
    generator: Generator

    @model_validator(mode="after")
    def add_missing_tables(self):
        generator_table_map = {t.name: t for t in self.generator.tables}
        if self.synthetic_config.tables is None:
            self.synthetic_config.tables = []
        synthetic_table_map = {t.name: t for t in self.synthetic_config.tables}
        missing_tables = set(generator_table_map.keys()) - set(synthetic_table_map.keys())
        for t in missing_tables:
            self.synthetic_config.tables.append(SyntheticTableConfig(name=t))
        return self

    @model_validator(mode="after")
    def validate_no_extra_tables(self):
        generator_table_map = {t.name: t for t in self.generator.tables}
        synthetic_table_map = {t.name: t for t in self.synthetic_config.tables or []}
        generator_tables = set(generator_table_map.keys())
        extra_tables = set(synthetic_table_map.keys()) - generator_tables
        if extra_tables:
            raise ValueError(
                f"Tables {extra_tables} are not present in the generator. Only {generator_tables} are available."
            )
        return self

    @model_validator(mode="after")
    def validate_tables(self):
        generator_table_map = {t.name: t for t in self.generator.tables}
        synthetic_table_map = {t.name: t for t in self.synthetic_config.tables or []}
        for table_name, synthetic_table in synthetic_table_map.items():
            generator_table = generator_table_map[table_name]
            synthetic_table.validate_against_source_table(
                generator_table, is_probe=isinstance(self.synthetic_config, SyntheticProbeConfig)
            )
        return self
