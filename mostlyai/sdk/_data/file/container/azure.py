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

from typing import Any

import duckdb
from adlfs import AzureBlobFileSystem
from azure.identity import ClientSecretCredential
from azure.storage.blob import BlobServiceClient
from cloudpathlib import AzureBlobClient
from cloudpathlib.azure import AzureBlobPath

from mostlyai.sdk._data.exceptions import MostlyDataException
from mostlyai.sdk._data.file.container.bucket_based import BucketBasedContainer


class AzureBlobFileContainer(BucketBasedContainer):
    SCHEMES = ["http", "https", "az", "azure"]
    DEFAULT_SCHEME = "az"
    DELIMITER_SCHEMA = "azure"
    SECRET_ATTR_NAME = "account_key"

    def __init__(
        self,
        *args,
        account_name,
        account_key=None,
        client_id=None,
        client_secret=None,
        tenant_id=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.account_name = account_name
        self.account_key = account_key
        self.client_id = client_id
        self.client_secret = client_secret
        self.tenant_id = tenant_id

        if account_key:
            # Normal auth
            self.decrypt_secret()
            credential = self.account_key
        elif client_id and client_secret and tenant_id:
            # Service Principal auth
            self.decrypt_secret("client_secret")
            credential = ClientSecretCredential(
                tenant_id=self.tenant_id,
                client_id=self.client_id,
                client_secret=self.client_secret,
            )
        else:
            raise MostlyDataException(
                "Provide the account key or service principal credentials (client_id, client_secret, tenant_id).",
            )

        self.fs = AzureBlobFileSystem(
            account_name=self.account_name,
            account_key=self.account_key,
            credential=credential,
        )
        self._client = AzureBlobClient(
            account_url=f"https://{self.account_name}.blob.core.windows.net",
            credential=credential,
        )
        self._blob_service_client = BlobServiceClient(
            account_url=f"https://{self.account_name}.blob.core.windows.net",
            credential=credential,
            retry_total=0,  # disable retry so that we won't timeout when credentials are incorrect
        )

    @classmethod
    def cloud_path_cls(cls):
        return AzureBlobPath

    @property
    def storage_options(self) -> dict:
        return self.fs.storage_options

    @property
    def transport_params(self) -> dict | None:
        return dict(client=self._blob_service_client)

    @property
    def file_system(self) -> Any:
        return self.fs

    def _check_authenticity(self) -> bool:
        try:
            return self._blob_service_client.get_account_information() is not None
        except Exception as e:
            error_message = str(e).lower()
            if any(
                keyword in error_message
                for keyword in [
                    "cannot connect to host",
                    "account is disabled",
                    "error on post request",
                    "nodename nor servname provided",
                    "failed to resolve",
                ]
            ):
                raise MostlyDataException("Account name is incorrect.")
            elif any(
                keyword in error_message
                for keyword in [
                    "server failed to authenticate the request",
                    "incorrect padding",
                    "invalid base64-encoded string",
                ]
            ):
                raise MostlyDataException("Account key is incorrect.")
            elif "check your tenant" in error_message:
                raise MostlyDataException("Tenant ID is incorrect.")
            elif "application with identifier" in error_message:
                raise MostlyDataException("Client ID is incorrect.")
            elif "invalid client secret" in error_message:
                raise MostlyDataException("Client secret is incorrect.")
            else:
                raise MostlyDataException(f"Authenticity check failed: {str(e)}")

    def _init_duckdb(self, con: duckdb.DuckDBPyConnection) -> None:
        if self.account_key:
            # use connection string authentication
            secret_params = {
                "TYPE": "azure",
                "CONNECTION_STRING": f"DefaultEndpointsProtocol=https;AccountName={self.account_name};AccountKey={self.account_key}",
            }
        else:
            # use service principal authentication
            secret_params = {
                "TYPE": "azure",
                "PROVIDER": "service_principal",
                "ACCOUNT_NAME": self.account_name,
                "CLIENT_ID": self.client_id,
                "CLIENT_SECRET": self.client_secret,
                "TENANT_ID": self.tenant_id,
            }

        self._create_duckdb_secret(con, secret_params)
