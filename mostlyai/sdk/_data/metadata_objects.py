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

from typing import Annotated, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.json_schema import SkipJsonSchema

T = TypeVar("T")

Nullable = T | SkipJsonSchema[None]


class SslCertificates(BaseModel):
    # SSL for Postgres
    root_certificate: Nullable[str] = Field(None, alias="rootCertificate", description="Encrypted root certificate.")
    ssl_certificate: Nullable[str] = Field(None, alias="sslCertificate", description="Encrypted client certificate.")
    ssl_certificate_key: Nullable[str] = Field(None, alias="sslCertificateKey", description="Encrypted client key.")
    # SSL for Hive
    keystore: Nullable[str] = Field(None, description="Encrypted keystore.")
    keystore_password: Nullable[str] = Field(None, alias="keystorePassword", description="Encrypted keystore password.")
    ca_certificate: Nullable[str] = Field(None, alias="caCertificate", description="Encrypted CA certificate.")


class AwsS3FileContainerParameters(BaseModel):
    access_key: Nullable[str] = Field(None, alias="accessKey")
    secret_key: Nullable[str] = Field(None, alias="secretKey")
    endpoint_url: Nullable[str] = Field(None, alias="endpointUrl")
    # SSL
    ssl_enabled: Nullable[bool] = Field(False, alias="sslEnabled")
    ca_certificate: Nullable[str] = Field(None, alias="caCertificate", description="Encrypted CA certificate.")


class AzureBlobFileContainerParameters(BaseModel):
    account_name: Nullable[str] = Field(None, alias="accountName")
    account_key: Nullable[str] = Field(None, alias="accountKey")
    client_id: Nullable[str] = Field(None, alias="clientId")
    client_secret: Nullable[str] = Field(None, alias="clientSecret")
    tenant_id: Nullable[str] = Field(None, alias="tenantId")


class GcsContainerParameters(BaseModel):
    key_file: Nullable[str] = Field(None, alias="keyFile")


class MinIOContainerParameters(BaseModel):
    endpoint_url: Nullable[str] = Field(None, alias="endpointUrl")
    access_key: Nullable[str] = Field(None, alias="accessKey")
    secret_key: Nullable[str] = Field(None, alias="secretKey")


class LocalFileContainerParameters(BaseModel):
    pass


class SqlAlchemyContainerParameters(BaseModel):
    model_config = ConfigDict(extra="allow")

    username: Nullable[str] = None
    password: Nullable[str] = None
    host: Nullable[str] = None
    port: Nullable[str | int] = None
    dbname: Nullable[str] = Field(None, alias="database")
    # SSL
    ssl_enabled: Nullable[bool] = Field(False, alias="sslEnabled")
    root_certificate: Nullable[str] = Field(None, alias="rootCertificate", description="Encrypted root certificate.")
    ssl_certificate: Nullable[str] = Field(None, alias="sslCertificate", description="Encrypted client certificate.")
    ssl_certificate_key: Nullable[str] = Field(None, alias="sslCertificateKey", description="Encrypted client key.")
    keystore: Nullable[str] = Field(None, description="Encrypted keystore.")
    keystore_password: Nullable[str] = Field(None, alias="keystorePassword", description="Encrypted keystore password.")
    ca_certificate: Nullable[str] = Field(None, alias="caCertificate", description="Encrypted CA certificate.")
    # Kerberos
    kerberos_enabled: Nullable[bool] = Field(False, alias="kerberosEnabled")
    kerberos_kdc_host: Nullable[str] = Field(None, alias="kerberosKdcHost")
    kerberos_krb5_conf: Nullable[str] = Field(None, alias="kerberosKrb5Conf")
    kerberos_service_principal: Nullable[str] = Field(None, alias="kerberosServicePrincipal")
    kerberos_client_principal: Nullable[str] = Field(None, alias="kerberosClientPrincipal")
    kerberos_keytab: Nullable[str] = Field(
        None,
        alias="kerberosKeytab",
        description="Encrypted content of keytab file of client principal if it is defined. Otherwise, it is the one for service principal.",
    )

    # Uncomment these if we want to enable the SSH connection feature
    # enable_ssh: Nullable[bool] = Field(False, alias="enableSsh")
    # ssh_host: Nullable[str] = Field(None, alias="sshHost")
    # ssh_port: Nullable[int] = Field(None, alias="sshPort")
    # ssh_username: Nullable[str] = Field(None,  alias="sshUsername")
    # ssh_password: Nullable[str] = Field(None, alias="sshPassword")
    # ssh_private_key_path: Nullable[str] = Field(None, alias="sshPrivateKeyPath")

    @field_validator("port")
    def cast_port_to_str(cls, value) -> Nullable[str]:
        return str(value) if value is not None else None


class OracleContainerParameters(SqlAlchemyContainerParameters):
    connection_type: Nullable[str] = Field("SID", alias="connectionType")


class SnowflakeContainerParameters(SqlAlchemyContainerParameters):
    host: Nullable[str] = Field(None, alias="warehouse")
    account: Nullable[str] = None


class BigQueryContainerParameters(SqlAlchemyContainerParameters):
    password: Nullable[str] = Field(None, alias="keyFile")


class DatabricksContainerParameters(SqlAlchemyContainerParameters):
    password: Nullable[str] = Field(None, alias="accessToken")
    dbname: Nullable[str] = Field(None, alias="catalog")
    http_path: Nullable[str] = Field(None, alias="httpPath")
    client_id: Nullable[str] = Field(None, alias="clientId")
    client_secret: Nullable[str] = Field(None, alias="clientSecret")
    tenant_id: Nullable[str] = Field(None, alias="tenantId")


class ConnectionResponse(BaseModel):
    connection_succeeded: bool = Field(
        False,
        alias="connectionSucceeded",
        description="Boolean that shows the connection status",
    )
    message: str = Field("", description="A message describing the result of the test.")


class LocationsResponse(BaseModel):
    locations: list[str] = Field(None, description="The list of locations")


class ColumnSchema(BaseModel):
    name: Nullable[str] = None
    original_data_type: Annotated[Nullable[str], Field(alias="originalDataType")] = None
    default_model_encoding_type: Annotated[Nullable[str], Field(alias="defaultModelEncodingType")] = None


class ConstraintSchema(BaseModel):
    foreign_key: Annotated[
        Nullable[str],
        Field(
            alias="foreignKey",
            description="The name of the foreign key column within this table.",
        ),
    ] = None
    referenced_table: Annotated[
        Nullable[str],
        Field(alias="referencedTable", description="The name of the reference table."),
    ] = None


class TableSchema(BaseModel):
    name: Annotated[Nullable[str], Field(description="The name of the table.")] = None
    totalRows: Annotated[
        Nullable[int],
        Field(
            alias="totalRows",
            description="The total number of rows in the table.",
        ),
    ] = None
    primary_key: Annotated[
        Nullable[str],
        Field(
            alias="primaryKey",
            description="The name of a primary key column. Only applicable for DB connectors.",
        ),
    ] = None
    columns: Annotated[Nullable[list[ColumnSchema]], Field(description="List of table columns.")] = None
    constraints: Annotated[
        Nullable[list[ConstraintSchema]],
        Field(description="List of foreign key relations, whose type is supported. Only applicable for DB connectors."),
    ] = None
    children: Annotated[
        Nullable[list["TableSchema"]],
        Field(description="List of child tables, if includeChildren was set to true."),
    ] = None
    location: Annotated[
        Nullable[str],
        Field(description="The location of the table."),
    ] = None


TableSchema.model_rebuild()
