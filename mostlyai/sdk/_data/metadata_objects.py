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

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SslCertificates(BaseModel):
    # SSL for Postgres
    root_certificate: str | None = Field(None, alias="rootCertificate", description="Encrypted root certificate.")
    ssl_certificate: str | None = Field(None, alias="sslCertificate", description="Encrypted client certificate.")
    ssl_certificate_key: str | None = Field(None, alias="sslCertificateKey", description="Encrypted client key.")
    # SSL for Hive
    keystore: str | None = Field(None, description="Encrypted keystore.")
    keystore_password: str | None = Field(None, alias="keystorePassword", description="Encrypted keystore password.")
    ca_certificate: str | None = Field(None, alias="caCertificate", description="Encrypted CA certificate.")


class AwsS3FileContainerParameters(BaseModel):
    access_key: str | None = Field(None, alias="accessKey")
    secret_key: str | None = Field(None, alias="secretKey")
    role_arn: str | None = Field(None, alias="roleArn")
    external_id: str | None = Field(None, alias="externalId")
    endpoint_url: str | None = Field(None, alias="endpointUrl")
    # SSL
    ssl_enabled: bool | None = Field(False, alias="sslEnabled")
    ca_certificate: str | None = Field(None, alias="caCertificate", description="Encrypted CA certificate.")


class AzureBlobFileContainerParameters(BaseModel):
    account_name: str | None = Field(None, alias="accountName")
    account_key: str | None = Field(None, alias="accountKey")
    client_id: str | None = Field(None, alias="clientId")
    client_secret: str | None = Field(None, alias="clientSecret")
    tenant_id: str | None = Field(None, alias="tenantId")


class GcsContainerParameters(BaseModel):
    key_file: str | None = Field(None, alias="keyFile")


class MinIOContainerParameters(BaseModel):
    endpoint_url: str | None = Field(None, alias="endpointUrl")
    access_key: str | None = Field(None, alias="accessKey")
    secret_key: str | None = Field(None, alias="secretKey")


class LocalFileContainerParameters(BaseModel):
    pass


class SqlAlchemyContainerParameters(BaseModel):
    model_config = ConfigDict(extra="allow")

    username: str | None = None
    password: str | None = None
    host: str | None = None
    port: str | int | None = None
    dbname: str | None = Field(None, alias="database")
    # SSL
    ssl_enabled: bool | None = Field(False, alias="sslEnabled")
    root_certificate: str | None = Field(None, alias="rootCertificate", description="Encrypted root certificate.")
    ssl_certificate: str | None = Field(None, alias="sslCertificate", description="Encrypted client certificate.")
    ssl_certificate_key: str | None = Field(None, alias="sslCertificateKey", description="Encrypted client key.")
    keystore: str | None = Field(None, description="Encrypted keystore.")
    keystore_password: str | None = Field(None, alias="keystorePassword", description="Encrypted keystore password.")
    ca_certificate: str | None = Field(None, alias="caCertificate", description="Encrypted CA certificate.")
    # Kerberos
    kerberos_enabled: bool | None = Field(False, alias="kerberosEnabled")
    kerberos_kdc_host: str | None = Field(None, alias="kerberosKdcHost")
    kerberos_krb5_conf: str | None = Field(None, alias="kerberosKrb5Conf")
    kerberos_service_principal: str | None = Field(None, alias="kerberosServicePrincipal")
    kerberos_client_principal: str | None = Field(None, alias="kerberosClientPrincipal")
    kerberos_keytab: str | None = Field(
        None,
        alias="kerberosKeytab",
        description="Encrypted content of keytab file of client principal if it is defined. Otherwise, it is the one for service principal.",
    )

    # Uncomment these if we want to enable the SSH connection feature
    # enable_ssh: bool | None = Field(False, alias="enableSsh")
    # ssh_host: str | None = Field(None, alias="sshHost")
    # ssh_port: int | None = Field(None, alias="sshPort")
    # ssh_username: str | None = Field(None,  alias="sshUsername")
    # ssh_password: str | None = Field(None, alias="sshPassword")
    # ssh_private_key_path: str | None = Field(None, alias="sshPrivateKeyPath")

    @field_validator("port")
    def cast_port_to_str(cls, value) -> str | None:
        return str(value) if value is not None else None


class OracleContainerParameters(SqlAlchemyContainerParameters):
    connection_type: str | None = Field("SID", alias="connectionType")


class SnowflakeContainerParameters(SqlAlchemyContainerParameters):
    host: str | None = Field(None, alias="warehouse")
    account: str | None = None


class BigQueryContainerParameters(SqlAlchemyContainerParameters):
    password: str | None = Field(None, alias="keyFile")


class DatabricksContainerParameters(SqlAlchemyContainerParameters):
    password: str | None = Field(None, alias="accessToken")
    dbname: str | None = Field(None, alias="catalog")
    http_path: str | None = Field(None, alias="httpPath")
    client_id: str | None = Field(None, alias="clientId")
    client_secret: str | None = Field(None, alias="clientSecret")
    tenant_id: str | None = Field(None, alias="tenantId")


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
    name: str | None = None
    original_data_type: Annotated[str | None, Field(alias="originalDataType")] = None
    default_model_encoding_type: Annotated[str | None, Field(alias="defaultModelEncodingType")] = None


class ConstraintSchema(BaseModel):
    foreign_key: Annotated[
        str | None,
        Field(
            alias="foreignKey",
            description="The name of the foreign key column within this table.",
        ),
    ] = None
    referenced_table: Annotated[
        str | None,
        Field(alias="referencedTable", description="The name of the reference table."),
    ] = None


class TableSchema(BaseModel):
    name: Annotated[str | None, Field(description="The name of the table.")] = None
    totalRows: Annotated[
        int | None,
        Field(
            alias="totalRows",
            description="The total number of rows in the table.",
        ),
    ] = None
    primary_key: Annotated[
        str | None,
        Field(
            alias="primaryKey",
            description="The name of a primary key column. Only applicable for DB connectors.",
        ),
    ] = None
    columns: Annotated[list[ColumnSchema] | None, Field(description="List of table columns.")] = None
    constraints: Annotated[
        list[ConstraintSchema] | None,
        Field(description="List of foreign key relations, whose type is supported. Only applicable for DB connectors."),
    ] = None
    children: Annotated[
        list["TableSchema"] | None,
        Field(description="List of child tables, if includeChildren was set to true."),
    ] = None
    location: Annotated[
        str | None,
        Field(description="The location of the table."),
    ] = None


TableSchema.model_rebuild()
